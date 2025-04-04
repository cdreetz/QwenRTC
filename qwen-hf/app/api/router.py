import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
from pydantic import BaseModel
from app.dependencies import model_instance
import base64
import os
import tempfile

router = APIRouter()
logger = logging.getLogger(__name__)

class ModelInfoResponse(BaseModel):
    model_name: str
    device: str
    available_speakers: List[str]
    audio_output_enabled: bool
    
class InferenceRequest(BaseModel):
    text: str
    speaker: Optional[str] = None
    return_audio: Optional[bool] = None
    max_new_tokens: Optional[int] = 1024

class InferenceResponse(BaseModel):
    text: str
    has_audio: bool
    audio_sample_rate: Optional[int] = None
    audio_base64: Optional[str] = None

@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model"""
    if model_instance is None or not model_instance.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    return {
        "model_name": model_instance.model_name,
        "device": model_instance.device,
        "available_speakers": model_instance.get_available_speakers(),
        "audio_output_enabled": model_instance.enable_audio_output
    }

@router.post("/inference/text", response_model=InferenceResponse)
async def text_inference(request: InferenceRequest):
    """Generate a response from text input only"""
    if model_instance is None or not model_instance.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    try:
        response = await model_instance.generate_response(
            input_text=request.text,
            speaker=request.speaker,
            return_audio=request.return_audio,
            max_new_tokens=request.max_new_tokens
        )
        
        # Convert audio waveform to base64 if present
        if response.get("has_audio") and "audio_waveform" in response:
            import numpy as np
            # Convert to int16 PCM format for compatibility
            audio_int16 = (np.array(response["audio_waveform"]) * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            response["audio_base64"] = base64.b64encode(audio_bytes).decode("utf-8")
            del response["audio_waveform"]  # Remove raw waveform to reduce response size
            
        return response
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@router.post("/inference/multimodal", response_model=InferenceResponse)
async def multimodal_inference(
    text: str = Form(...),
    speaker: Optional[str] = Form(None),
    return_audio: Optional[bool] = Form(None),
    max_new_tokens: Optional[int] = Form(1024),
    images: List[UploadFile] = File(None),
    audios: List[UploadFile] = File(None),
    videos: List[UploadFile] = File(None),
):
    """Generate a response from multimodal input (text + optional images, videos, audio)"""
    if model_instance is None or not model_instance.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    # Create temporary files for uploaded media
    temp_files = []
    image_paths = []
    audio_paths = []
    video_paths = []
    
    try:
        # Save uploaded images
        if images:
            for img in images:
                suffix = os.path.splitext(img.filename)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(await img.read())
                    image_paths.append(tmp.name)
                    temp_files.append(tmp.name)
        
        # Save uploaded audio files
        if audios:
            for audio in audios:
                suffix = os.path.splitext(audio.filename)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(await audio.read())
                    audio_paths.append(tmp.name)
                    temp_files.append(tmp.name)
        
        # Save uploaded videos
        if videos:
            for video in videos:
                suffix = os.path.splitext(video.filename)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(await video.read())
                    video_paths.append(tmp.name)
                    temp_files.append(tmp.name)
        
        # Generate response
        response = await model_instance.generate_response(
            input_text=text,
            images=image_paths if image_paths else None,
            audios=audio_paths if audio_paths else None,
            videos=video_paths if video_paths else None,
            speaker=speaker,
            return_audio=return_audio,
            max_new_tokens=max_new_tokens
        )
        
        # Convert audio waveform to base64 if present
        if response.get("has_audio") and "audio_waveform" in response:
            import numpy as np
            # Convert to int16 PCM format for compatibility
            audio_int16 = (np.array(response["audio_waveform"]) * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            response["audio_base64"] = base64.b64encode(audio_bytes).decode("utf-8")
            del response["audio_waveform"]  # Remove raw waveform to reduce response size
            
        return response
        
    except Exception as e:
        logger.error(f"Multimodal inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        # Clean up temporary files
        for file_path in temp_files:
            try:
                os.unlink(file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {file_path}: {str(e)}")
