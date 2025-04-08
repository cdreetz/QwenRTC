# app/api/endpoints/inference.py
import os
import subprocess
import uuid
import logging
from typing import List, Optional, Union
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form
from pydantic import BaseModel
from app.dependencies import get_model_instance

logger = logging.getLogger(__name__)

# Temporary directory for uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()

class InferenceResponse(BaseModel):
    text: Union[str, List[str]]
    has_audio: Optional[bool] = None
    audio_sample_rate: Optional[int] = None
    audio_waveform: Optional[List[float]] = None
    
class InferenceRequest(BaseModel):
    input_text: str
    speaker: Optional[str] = "Ethan"  # Default speaker
    return_audio: Optional[bool] = True
    max_new_tokens: Optional[int] = 1024
    system_prompt: Optional[str] = None

async def save_uploaded_file(file: UploadFile) -> str:
    """Save an uploaded file to the uploads directory and return the file path"""
    if not file or not file.filename:
        return None

    filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Save the file
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    return file_path

def convert_audio_to_wav(input_path: str) -> str:
    """
    Convert audio file to a standard WAV format that librosa can read
    Uses ffmpeg to ensure proper conversion
    """
    try:
        # Create a new filename with proper extension for the converted file
        output_filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(UPLOAD_DIR, output_filename)
        
        # Use ffmpeg to convert the audio file to standard WAV format with PCM encoding
        # -y: Overwrite output file if it exists
        # -i: Input file
        # -acodec pcm_s16le: Convert to 16-bit PCM WAV
        # -ar 16000: Set sample rate to 16000 Hz (model requirement)
        # -ac 1: Convert to mono (single channel)
        command = [
            "ffmpeg", "-y", 
            "-i", input_path, 
            "-acodec", "pcm_s16le", 
            "-ar", "16000", 
            "-ac", "1", 
            output_path
        ]
        
        # Run the conversion command
        subprocess.run(command, check=True, capture_output=True)
        
        # Check if the output file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Successfully converted audio file to WAV: {output_path}")
            return output_path
        else:
            logger.error(f"Conversion failed: Output file {output_path} is empty or doesn't exist")
            return None
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion error: {e.stderr.decode() if e.stderr else str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}", exc_info=True)
        return None




@router.post("/", response_model=InferenceResponse)
async def inference_endpoint(
    request: InferenceRequest,
    model=Depends(get_model_instance)
):
    """Generate a response from the Qwen 2.5 Omni model"""
    try:
        if not model.is_ready:
            raise HTTPException(status_code=503, detail="Model is not ready for inference")
        
        # Generate response
        response = await model.generate_response(
            input_text=request.input_text,
            speaker=request.speaker,
            return_audio=request.return_audio,
            max_new_tokens=request.max_new_tokens,
            #system_prompt=request.system_prompt
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")



@router.post("/multimodal", response_model=InferenceResponse)
async def multimodal_inference_endpoint(
    background_tasks: BackgroundTasks,
    input_text: str = Form(...),
    speaker: Optional[str] = Form("Ethan"),
    return_audio: Optional[bool] = Form(True),
    max_new_tokens: Optional[int] = Form(1024),
    system_prompt: Optional[str] = Form(None),
    images: List[UploadFile] = File([]),
    videos: List[UploadFile] = File([]),
    audios: List[UploadFile] = File([]),
    model=Depends(get_model_instance)
):
    """Generate a response from the Qwen 2.5 Omni model with multimodal inputs"""
    files_to_cleanup = []


    try:
        if not model.is_ready:
            raise HTTPException(status_code=503, detail="Model is not ready for inference")
        
        # Process uploaded files
        image_paths = []
        video_paths = []
        audio_paths = []
        
        # Save image files
        if images:
            for image in images:
                if image.filename:
                    image_path = await save_uploaded_file(image)
                    if image_path:
                        image_paths.append(image_path)
                        files_to_cleanup.append(image_path)
        
        # Save video files
        if videos:
            for video in videos:
                if video.filename:
                    video_path = await save_uploaded_file(video)
                    if vide_path:
                        video_paths.append(video_path)
                        files_to_cleanup.append(video_path)
        
        # Save audio files
        if audios:
            for audio in audios:
                if audio and audio.filename:
                    try:
                        original_path = await save_uploaded_file(audio)
                        if original_path:
                            files_to_cleanup.append(original_path)

                            converted_path = convert_audio_to_wav(original_path)
                            if converted_path:
                                audio_paths.append(converted_path)
                                files_to_cleanup.append(converted_path)
                                logger.info(f"Successfully processed audio file: {converted_path}")
                            else:
                                logger.warning(f"Could not convert audio file: {original_path}")
                    except Exception as audio_error:
                        logger.error(f"Error processing audio file: {str(audio_error)}", exc_info=True)

        for file_path in files_to_cleanup:
            background_tasks.add_task(os.remove, file_path)


        if system_prompt is None:
            # Check model type to provide appropriate default system prompt
            if hasattr(model, 'model_name') and 'qwen' in model.model_name.lower():
                system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            elif hasattr(model, 'model_name') and 'phi' in model.model_name.lower():
                system_prompt = "You are Phi-4, a helpful AI assistant that can understand images and audio."
            else:
                system_prompt = "You are a helpful AI assistant that can understand multimodal inputs."
        
        logger.info(f"Using system prompt: {system_prompt}")
        
        # Debug logging
        logger.info(f"Input text: {input_text}")
        logger.info(f"Audio paths: {audio_paths}")
        logger.info(f"Image paths: {image_paths}")
        logger.info(f"Video paths: {video_paths}")

        processed_audio_paths = []
        for path in audio_paths:
            try:
                processed_audio_paths.append(path)
            except Exception as audio_error:
                logger.error(f"Error preprocessing audio file {path}: {str(audio_error)}", exc_info=True)
        
        # Generate response
        try:
            response = await model.generate_response(
                input_text=input_text,
                images=image_paths if image_paths else None,
                videos=video_paths if video_paths else None,
                audios=audio_paths if audio_paths else None,
                speaker=speaker,
                return_audio=return_audio,
                max_new_tokens=max_new_tokens,
                system_prompt=system_prompt
            )
        
            return response
        except Exception as model_error:
            logger.error(f"Model generation error: {str(model_error)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Model generation failed: {str(model_error)}")
        
    except Exception as e:
        logger.error(f"Error during multimodal inference: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Multimodal inference failed: {str(e)}")

@router.get("/speakers")
async def get_speakers(model=Depends(get_model_instance)):
    """Get available speakers for audio generation"""
    try:
        speakers = model.get_available_speakers()
        return {"speakers": speakers}
    except Exception as e:
        logger.error(f"Error getting speakers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get speakers: {str(e)}")
