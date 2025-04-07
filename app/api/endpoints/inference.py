# app/api/endpoints/inference.py
import os
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
    filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Save the file
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    return file_path

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
    images: List[UploadFile] = File(None),
    videos: List[UploadFile] = File(None),
    audios: List[UploadFile] = File(None),
    model=Depends(get_model_instance)
):
    """Generate a response from the Qwen 2.5 Omni model with multimodal inputs"""
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
                    image_paths.append(image_path)
                    # Clean up file after response is sent
                    background_tasks.add_task(os.remove, image_path)
        
        # Save video files
        if videos:
            for video in videos:
                if video.filename:
                    video_path = await save_uploaded_file(video)
                    video_paths.append(video_path)
                    # Clean up file after response is sent
                    background_tasks.add_task(os.remove, video_path)
        
        # Save audio files
        if audios:
            for audio in audios:
                if audio.filename:
                    audio_path = await save_uploaded_file(audio)
                    audio_paths.append(audio_path)
                    # Clean up file after response is sent
                    background_tasks.add_task(os.remove, audio_path)
        
        # Generate response
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
        
    except Exception as e:
        logger.error(f"Error during multimodal inference: {str(e)}")
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
