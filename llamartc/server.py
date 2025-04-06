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
from aiohttp_cors import setup as setup_cors, ResourceOptions

# Import our LLaMA service
from llama_service import LlamaService

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create data directory
ROOT = Path(__file__).parent
AUDIO_DIR = ROOT / "audio_chunks"
AUDIO_DIR.mkdir(exist_ok=True)

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
                chunk_length_s=30,
                batch_size=16,
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
                
            # Run transcription
            result = await asyncio.to_thread(
                self.transcription_pipeline,
                {"array": audio_data, "sampling_rate": 16000},
                generate_kwargs=generate_kwargs
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
            result = await request.app["whisper_server"].transcribe(audio_np, language)
            
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

async def handle_speech_to_llm(request):
    """Handle incoming speech to LLM requests"""
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
        
        # Get optional parameters
        language = data.get("language", None)
        temperature = data.get("temperature", 0.7)
        max_length = data.get("max_length", 512)
        
        try:
            # Decode the base64 audio
            audio_bytes = base64.b64decode(audio_base64)
            
            # Convert to numpy array (assuming 16-bit PCM)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Step 1: Perform transcription
            transcription_result = await request.app["whisper_server"].transcribe(audio_np, language)
            
            if "error" in transcription_result:
                return web.Response(
                    status=500,
                    text=json.dumps({"error": transcription_result["error"]}),
                    content_type="application/json"
                )
                
            # Extract the transcribed text
            transcribed_text = transcription_result.get("text", "")
            
            if not transcribed_text.strip():
                return web.Response(
                    status=400,
                    text=json.dumps({"error": "No speech detected or transcription failed"}),
                    content_type="application/json"
                )
                
            # Step 2: Generate LLaMA response
            llm_result = await request.app["llama_service"].generate_response(
                transcribed_text,
                max_length=max_length,
                temperature=temperature
            )
            
            # Combine results
            combined_result = {
                "transcription": transcribed_text,
                "response": llm_result.get("response", ""),
                "transcription_time": transcription_result.get("process_time", 0),
                "llm_time": llm_result.get("process_time", 0),
                "total_time": transcription_result.get("process_time", 0) + llm_result.get("process_time", 0)
            }
            
            # Return the combined result
            return web.Response(
                text=json.dumps(combined_result),
                content_type="application/json"
            )
            
        except Exception as e:
            logging.error(f"Error processing speech to LLM: {e}")
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
        "device": request.app["whisper_server"].device,
        "torch_dtype": str(request.app["whisper_server"].torch_dtype),
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
        "whisper_model": request.app["whisper_server"].model_id,
        "llama_model": request.app["llama_service"].model_id,
        "device_info": device_info,
        "uptime": time.time() - request.app["start_time"],
    }
    
    return web.Response(
        text=json.dumps(status),
        content_type="application/json"
    )
    
async def on_startup(app):
    """Initialize components on server startup"""
    app["start_time"] = time.time()
    # Initialize Whisper
    app["whisper_server"] = WhisperServer(model_id="openai/whisper-large-v3-turbo")
    # Initialize LLaMA
    app["llama_service"] = LlamaService(model_id="meta-llama/Llama-3.2-3B-Instruct")
    
async def on_shutdown(app):
    """Clean up resources on server shutdown"""
    logging.info("Shutting down server")
    # Free up GPU memory
    if "whisper_server" in app and app["whisper_server"].transcription_pipeline:
        del app["whisper_server"].transcription_pipeline
    
    if "llama_service" in app and app["llama_service"].llm_pipeline:
        del app["llama_service"].llm_pipeline
        
    torch.cuda.empty_cache()
        
def main():
    # Create the web application
    app = web.Application()
    
    # Set up startup and shutdown handlers
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    cors = setup_cors(app, defaults={
        "*": ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["GET", "POST", "OPTIONS"]
        )
    })
    
    # Set up routes
    route = app.router.add_post("/transcribe", handle_transcribe)
    cors.add(route)
    
    # Add new endpoint for speech-to-LLM
    route = app.router.add_post("/speech_to_llm", handle_speech_to_llm)
    cors.add(route)

    route = app.router.add_get("/status", handle_status)
    cors.add(route)
    
    # Run the server
    web.run_app(app, host="0.0.0.0", port=8080)
    
if __name__ == "__main__":
    main()
