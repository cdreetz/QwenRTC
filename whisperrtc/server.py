import asyncio
import json
import logging
import os
import time
from pathlib import Path
import base64

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from aiohttp import web

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create data directory
ROOT = Path(__file__).parent
AUDIO_DIR = ROOT / "audio_chunks"
AUDIO_DIR.mkdir(exist_ok=True)

# Global state
last_load_time = time.time()
transcription_pipeline = None

class WhisperServer:
    def __init__(self, model_id="openai/whisper-large-v3-turbo", device=None):
        self.model_id = model_id
        
        # Determine the device
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        
        logging.info(f"Using device: {self.device}, dtype: {self.torch_dtype}")
        self.transcription_pipeline = None
        
        # Initialize the model
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the Whisper model and processor"""
        try:
            logging.info(f"Loading Whisper model: {self.model_id}")
            start_time = time.time()
            
            # Load model
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation="sdpa"  # Use scaled dot product attention for better performance
            )
            model.to(self.device)
            
            # Load processor
            processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Create pipeline
            self.transcription_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=True,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            
            logging.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logging.error(f"Error initializing Whisper model: {e}")
            raise
    
    async def transcribe(self, audio_data, language=None):
        """
        Transcribe audio data with Whisper model
        
        Args:
            audio_data: NumPy array of audio samples (16kHz, mono, float32 in [-1, 1] range)
            language: Optional language code
        
        Returns:
            Dict with transcription results
        """
        try:
            if self.transcription_pipeline is None:
                self.initialize_model()
                
            start_time = time.time()
            
            # Prepare options
            generate_kwargs = {}
            if language:
                generate_kwargs["language"] = language
                # Note: Whisper generally handles auto language detection well,
                # so language is optional but can help accuracy
                
            # Run transcription
            result = await asyncio.to_thread(
                self.transcription_pipeline,
                {"array": audio_data, "sampling_rate": 16000},
                generate_kwargs=generate_kwargs,
                return_timestamps=True
            )
            
            process_time = time.time() - start_time
            logging.info(f"Transcription completed in {process_time:.2f} seconds")
            
            # Add processing metadata
            result["process_time"] = process_time
            
            return result
            
        except Exception as e:
            logging.error(f"Error in transcription: {str(e)}")
            return {"error": str(e)}


async def handle_transcribe(request):
    """Handle incoming transcription requests"""
    try:
        # Parse the request body
        data = await request.json()
        
        # Extract base64-encoded audio
        if "audio_base64" not in data:
            return web.Response(
                status=400,
                text=json.dumps({"error": "Missing audio_base64 field"}),
                content_type="application/json"
            )
            
        audio_base64 = data["audio_base64"]
        
        # Get optional language parameter
        language = data.get("language", None)
        
        try:
            # Decode the base64 audio
            audio_bytes = base64.b64decode(audio_base64)
            
            # Convert to numpy array (assuming 16-bit PCM)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Perform transcription
            result = await app["whisper_server"].transcribe(audio_np, language)
            
            # Return the result
            return web.Response(
                text=json.dumps(result),
                content_type="application/json"
            )
        except Exception as e:
            logging.error(f"Error processing transcription: {e}")
            return web.Response(
                status=500,
                text=json.dumps({"error": str(e)}),
                content_type="application/json"
            )
            
    except json.JSONDecodeError:
        return web.Response(
            status=400,
            text=json.dumps({"error": "Invalid JSON"}),
            content_type="application/json"
        )
        
async def handle_status(request):
    """Return server status information"""
    device_info = {
        "device": app["whisper_server"].device,
        "torch_dtype": str(app["whisper_server"].torch_dtype),
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        device_info.update({
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
            "cuda_memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB",
        })
        
    status = {
        "status": "running",
        "model_id": app["whisper_server"].model_id,
        "device_info": device_info,
        "uptime": time.time() - app["start_time"],
    }
    
    return web.Response(
        text=json.dumps(status),
        content_type="application/json"
    )
    
async def on_startup(app):
    """Initialize components on server startup"""
    app["start_time"] = time.time()
    app["whisper_server"] = WhisperServer(model_id="openai/whisper-large-v3-turbo")
    
async def on_shutdown(app):
    """Clean up resources on server shutdown"""
    logging.info("Shutting down server")
    # Free up GPU memory
    if "whisper_server" in app and app["whisper_server"].transcription_pipeline:
        del app["whisper_server"].transcription_pipeline
        torch.cuda.empty_cache()
        
def main():
    # Create the web application
    app = web.Application()
    
    # Set up startup and shutdown handlers
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    
    # Set up routes
    app.router.add_post("/transcribe", handle_transcribe)
    app.router.add_get("/status", handle_status)
    
    # Run the server
    web.run_app(app, host="0.0.0.0", port=8080)
    
if __name__ == "__main__":
    main()
