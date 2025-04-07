import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from av import AudioFrame
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
peer_connections: Set[RTCPeerConnection] = set()
SAMPLE_RATE = 24000  # Qwen2.5-Omni uses 24kHz sample rate
CHANNELS = 1

# Load Qwen2.5-Omni model
MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

logger.info(f"Loading model: {MODEL_NAME}")

# Enable flash attention if available
try:
    model = Qwen2_5OmniModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    logger.info("Using flash attention 2")
except Exception as e:
    logger.warning(f"Failed to load model with flash attention: {e}")
    logger.info("Loading model without flash attention")
    model = Qwen2_5OmniModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME)

class AudioTransformTrack(MediaStreamTrack):
    """
    A track that receives audio frames, processes them through Qwen2.5-Omni, and returns audio responses.
    """
    kind = "audio"

    def __init__(self, track, user_id):
        super().__init__()
        self.track = track
        self.user_id = user_id
        self.audio_buffer = np.array([], dtype=np.float32)
        self.is_processing = False
        self.silence_threshold = 0.01
        self.silent_frames = 0
        self.frames_for_silence = int(0.5 * SAMPLE_RATE / 960)  # 0.5 seconds of silence at typical WebRTC frame size
        self.response_queue = asyncio.Queue()
        self.system_message = {
            "role": "system",
            "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        }
        self.conversation_history = [self.system_message]
        self.frame_count = 0
        self.last_active_frame = 0
        self.samples_per_frame = 960  # Typical WebRTC frame size
        
        # Start the background processing task
        asyncio.create_task(self.process_audio_buffer())

    async def process_audio_buffer(self):
        """Process audio buffer when conditions are met."""
        while True:
            await asyncio.sleep(0.1)  # Check periodically
            
            # If we're already processing or the buffer is too small, skip
            if self.is_processing or len(self.audio_buffer) < SAMPLE_RATE * 0.5:  # At least 0.5 seconds
                continue
                
            # Check if we have enough silence after speech
            if self.silent_frames >= self.frames_for_silence:
                self.is_processing = True
                try:
                    # Process the audio
                    audio_data = self.audio_buffer.copy()
                    self.audio_buffer = np.array([], dtype=np.float32)
                    self.silent_frames = 0
                    
                    logger.info(f"Processing audio chunk of length {len(audio_data)}")
                    
                    # Process in background to avoid blocking
                    response_audio = await asyncio.to_thread(self.inference_with_qwen, audio_data)
                    
                    # Add to response queue
                    await self.response_queue.put(response_audio)
                    
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                finally:
                    self.is_processing = False

    def inference_with_qwen(self, audio_data):
        """Run inference with Qwen2.5-Omni model."""
        try:
            # Prepare the conversation
            user_message = {
                "role": "user",
                "content": [{"type": "audio", "audio": audio_data}]
            }
            
            # Add user message to conversation history
            conversation = self.conversation_history + [user_message]
            
            # Process with Qwen2.5-Omni
            text = processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            # Process multimedia info
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            
            # Create model inputs
            inputs = processor(
                text=text, 
                audios=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=False
            )
            
            # Move to appropriate device and dtype
            inputs = inputs.to(model.device).to(model.dtype)
            
            # Generate response
            logger.info("Generating response with Qwen2.5-Omni")
            with torch.no_grad():
                text_ids, audio = model.generate(**inputs, use_audio_in_video=False, spk="Chelsie")
            
            # Decode text
            response_text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            logger.info(f"Generated text: {response_text}")
            
            # Update conversation history with model's response
            model_message = {
                "role": "assistant",
                "content": response_text
            }
            self.conversation_history.append(user_message)
            self.conversation_history.append(model_message)
            
            # Trim conversation history if needed
            if len(self.conversation_history) > 10:  # Keep last 10 messages including system message
                self.conversation_history = [self.system_message] + self.conversation_history[-9:]
            
            # Get audio response
            response_audio = audio.reshape(-1).detach().cpu().numpy()
            
            # Ensure response is float32 in range [-1.0, 1.0]
            if response_audio.dtype != np.float32:
                response_audio = response_audio.astype(np.float32)
            
            # Normalize if needed
            max_val = np.max(np.abs(response_audio))
            if max_val > 1.0:
                response_audio = response_audio / max_val
                
            return response_audio
            
        except Exception as e:
            logger.error(f"Error in Qwen2.5-Omni inference: {e}")
            # Return empty audio in case of error
            return np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second of silence

    async def recv(self):
        """Receive a frame and process it."""
        frame = await self.track.recv()
        self.frame_count += 1
        
        # Extract audio data from the frame
        audio_frame = frame.to_ndarray()
        audio_data = audio_frame.flatten().astype(np.float32) / 32768.0  # Convert to float32 [-1.0, 1.0]
        
        # Check if this frame is silent
        is_silent = np.max(np.abs(audio_data)) < self.silence_threshold
        
        if is_silent:
            self.silent_frames += 1
        else:
            self.silent_frames = 0
            self.last_active_frame = self.frame_count
        
        # Add to our buffer
        self.audio_buffer = np.append(self.audio_buffer, audio_data)
        
        # If we have a response ready, return it
        if not self.response_queue.empty():
            try:
                response_audio = await self.response_queue.get()
                
                # Create audio frames to send back
                chunk_size = self.samples_per_frame
                chunks = [response_audio[i:i+chunk_size] for i in range(0, len(response_audio), chunk_size)]
                
                if chunks:
                    # Get the first chunk
                    chunk = chunks[0]
                    
                    # Put remaining chunks back in the queue in reverse order
                    for remaining_chunk in reversed(chunks[1:]):
                        self.response_queue.put_nowait(remaining_chunk)
                    
                    # Pad the last chunk if needed
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                    
                    # Convert to proper format for AudioFrame
                    chunk = (chunk * 32768).astype(np.int16)
                    
                    # Create new audio frame
                    new_frame = AudioFrame.from_ndarray(
                        chunk.reshape(-1, 1),  # Shape for mono: (samples, channels)
                        format="s16",  # Signed 16-bit
                        layout="mono"
                    )
                    new_frame.sample_rate = SAMPLE_RATE
                    new_frame.pts = frame.pts
                    new_frame.time_base = frame.time_base
                    
                    return new_frame
            except Exception as e:
                logger.error(f"Error preparing response audio: {e}")
        
        # Return the original frame if no response is available
        return frame

@app.on_event("startup")
async def startup():
    logger.info("Starting WebRTC server with Qwen2.5-Omni")

@app.on_event("shutdown")
async def shutdown():
    # Close all peer connections
    logger.info("Shutting down server")
    coros = [pc.close() for pc in peer_connections]
    await asyncio.gather(*coros)
    peer_connections.clear()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pc = RTCPeerConnection()
    peer_connections.add(pc)
    
    # Generate a unique ID for this connection
    user_id = id(websocket)
    logger.info(f"New WebSocket connection: {user_id}")
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state for user {user_id}: {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            if pc in peer_connections:
                peer_connections.remove(pc)

    # Set up media stream handling
    @pc.on("track")
    def on_track(track):
        logger.info(f"Received track from user {user_id}: {track.kind}")
        if track.kind == "audio":
            audio_transform = AudioTransformTrack(track, user_id)
            pc.addTrack(audio_transform)
        
        @track.on("ended")
        async def on_ended():
            logger.info(f"Track {track.kind} ended for user {user_id}")
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data["type"] == "offer":
                offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
                await pc.setRemoteDescription(offer)
                
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                
                await websocket.send_json({
                    "type": answer.type,
                    "sdp": answer.sdp
                })
            
            elif data["type"] == "ice":
                if data["candidate"]:
                    candidate = data["candidate"]
                    await pc.addIceCandidate(candidate)
            
            elif data["type"] == "chat":
                # Handle text chat messages if needed
                logger.info(f"Received chat message from user {user_id}: {data.get('message', '')}")
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
        await pc.close()
        if pc in peer_connections:
            peer_connections.remove(pc)
    
    except Exception as e:
        logger.error(f"Error in WebSocket connection for user {user_id}: {e}")
        await pc.close()
        if pc in peer_connections:
            peer_connections.remove(pc)

@app.get("/")
async def get_index():
    return {
        "message": "Qwen2.5-Omni WebRTC Server",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
