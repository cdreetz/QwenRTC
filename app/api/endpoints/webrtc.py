import asyncio
import logging
import numpy as np
import os
import uuid
import time
import soundfile as sf
import webrtcvad
from typing import Dict, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole
from pydantic import BaseModel
from app.dependencies import get_model_instance

logger = logging.getLogger(__name__)
router = APIRouter()

# Define the data models for WebRTC signaling
class RTCSessionDescriptionDict(BaseModel):
    sdp: str
    type: str

class WebRTCOffer(BaseModel):
    sdp: RTCSessionDescriptionDict

class WebRTCAnswer(BaseModel):
    sdp: RTCSessionDescriptionDict

# Reuse the existing uploads directory for temporary audio files
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Store active connections
peer_connections = {}
websocket_connections = {}

class VADAudioTrack(MediaStreamTrack):
    """
    Media track that processes audio with Voice Activity Detection.
    """
    kind = "audio"
    
    def __init__(self, track, client_id):
        super().__init__()
        self.track = track
        self.client_id = client_id
        
        # Set up VAD with aggressiveness level 3 (most aggressive)
        self.vad = webrtcvad.Vad(3)
        self.sample_rate = 16000  # VAD requires 8000, 16000, 32000, or 48000 Hz
        self.frame_duration = 30  # In milliseconds (10, 20, or 30 ms)
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        
        # Buffer for collecting audio during speech
        self.is_speech = False
        self.speech_buffer = []
        self.silence_frames = 0
        self.max_silence_frames = 30  # ~900ms of silence after speech ends
        
        # Set up processing task
        self._task = asyncio.create_task(self._run_processing())
        self._queue = asyncio.Queue(maxsize=100)  # Limit queue size
        
    async def _run_processing(self):
        """Process audio frames from the queue."""
        try:
            while True:
                frame = await self._queue.get()
                try:
                    await self._process_frame(frame)
                except Exception as e:
                    logger.error(f"Error processing frame: {str(e)}")
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            logger.info(f"Processing task for client {self.client_id} cancelled")
    
    async def _process_frame(self, frame):
        """Process an audio frame with VAD."""
        # Convert the frame to numpy array
        audio_data = frame.to_ndarray()
        
        # Resample if needed to 16kHz (VAD requirement)
        if frame.sample_rate != self.sample_rate:
            resampled = np.interp(
                np.linspace(0, len(audio_data) - 1, int(len(audio_data) * self.sample_rate / frame.sample_rate)),
                np.arange(len(audio_data)),
                audio_data
            )
            audio_data = resampled
        
        # Convert to 16-bit PCM (VAD requirement)
        audio_samples = (audio_data * 32767).astype(np.int16)
        
        # Ensure we have enough samples for a VAD frame
        if len(audio_samples) >= self.frame_size:
            # Check if this frame contains speech
            frame_bytes = audio_samples[:self.frame_size].tobytes()
            is_current_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
            
            # State machine for speech detection
            if is_current_speech and not self.is_speech:
                # Speech start
                self.is_speech = True
                self.speech_buffer = [audio_data]
                self.silence_frames = 0
                logger.debug(f"Speech started for client {self.client_id}")
                
                # Notify client that we're listening
                if self.client_id in websocket_connections:
                    websocket = websocket_connections[self.client_id]
                    if websocket.client_state.CONNECTED:
                        await websocket.send_json({
                            "type": "status",
                            "status": "listening"
                        })
                
            elif is_current_speech and self.is_speech:
                # Continuing speech
                self.speech_buffer.append(audio_data)
                self.silence_frames = 0
                
            elif not is_current_speech and self.is_speech:
                # Potential speech end, count silence frames
                self.silence_frames += 1
                self.speech_buffer.append(audio_data)  # Keep some silence for natural sounding
                
                if self.silence_frames > self.max_silence_frames:
                    # Speech ended, process the buffer
                    await self._process_speech_segment()
                    # Reset state
                    self.is_speech = False
                    self.speech_buffer = []
                    self.silence_frames = 0
    
    async def _process_speech_segment(self):
        """Process a complete speech segment."""
        if not self.speech_buffer:
            return
            
        try:
            # Combine all frames into a single audio segment
            audio_segment = np.concatenate(self.speech_buffer)
            
            # Save to temporary file
            timestamp = int(time.time())
            audio_path = os.path.join(UPLOAD_DIR, f"{self.client_id}_{timestamp}.wav")
            sf.write(audio_path, audio_segment, self.sample_rate)
            
            duration = len(audio_segment) / self.sample_rate
            logger.info(f"Saved speech segment ({duration:.2f}s) to {audio_path}")
            
            # Notify client that we're processing
            if self.client_id in websocket_connections:
                websocket = websocket_connections[self.client_id]
                if websocket.client_state.CONNECTED:
                    await websocket.send_json({
                        "type": "status",
                        "status": "processing"
                    })
            
            # Get model instance
            try:
                # This relies on your dependency injection system
                from app.dependencies import model_instance
                if model_instance and model_instance.is_ready:
                    # Process with your existing model
                    response = await model_instance.generate_response(
                        input_text="",  # Input comes from audio
                        audios=[audio_path],
                        return_audio=False  # Only text for now
                    )
                    
                    # Send response back to client
                    if self.client_id in websocket_connections:
                        websocket = websocket_connections[self.client_id]
                        if websocket.client_state.CONNECTED:
                            await websocket.send_json({
                                "type": "result",
                                "result": response
                            })
                else:
                    logger.error("Model not ready for inference")
                    if self.client_id in websocket_connections:
                        websocket = websocket_connections[self.client_id]
                        if websocket.client_state.CONNECTED:
                            await websocket.send_json({
                                "type": "error",
                                "error": "Model not ready for inference"
                            })
            except Exception as e:
                logger.error(f"Error in model inference: {str(e)}")
                if self.client_id in websocket_connections:
                    websocket = websocket_connections[self.client_id]
                    if websocket.client_state.CONNECTED:
                        await websocket.send_json({
                            "type": "error",
                            "error": f"Error in model inference: {str(e)}"
                        })
            
            # Clean up temporary file
            try:
                os.remove(audio_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error processing speech segment: {str(e)}")
    
    async def recv(self):
        """Receive a frame and queue it for processing."""
        frame = await self.track.recv()
        
        # Queue the frame for processing (non-blocking)
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            # If queue is full, drop the oldest frame
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(frame)
            except:
                pass
            
        # Return the original frame unchanged
        return frame
        
    def stop(self):
        """Stop processing and clean up resources."""
        if self._task and not self._task.cancelled():
            self._task.cancel()

@router.post("/offer")
async def webrtc_offer(offer: WebRTCOffer):
    """Handle a WebRTC offer and return an answer."""
    # Generate a unique client ID
    client_id = str(uuid.uuid4())
    
    # Create a new peer connection
    pc = RTCPeerConnection(
        configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        }
    )
    peer_connections[client_id] = pc
    
    logger.info(f"WebRTC offer received, created peer connection for {client_id}")
    
    # Set up connection state change handler
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state for {client_id}: {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await cleanup_resources(client_id)
    
    # Set up track handler
    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            logger.info(f"Received audio track from {client_id}")
            
            # Create a VAD track
            vad_track = VADAudioTrack(track, client_id)
            
            # Set up media sink to handle the track
            blackhole = MediaBlackhole()
            blackhole.addTrack(vad_track)
            asyncio.ensure_future(blackhole.start())
            
            @track.on("ended")
            async def on_ended():
                logger.info(f"Audio track ended for {client_id}")
                vad_track.stop()
                await cleanup_resources(client_id)
    
    # Set remote description (the offer)
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=offer.sdp.sdp, type=offer.sdp.type)
    )
    
    # Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    # Return the answer with the client ID
    return {
        "sdp": {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        },
        "client_id": client_id
    }

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket for bidirectional communication with the client."""
    # Validate client ID
    if client_id not in peer_connections:
        await websocket.close(code=1008, reason="Invalid client ID")
        return
    
    await websocket.accept()
    websocket_connections[client_id] = websocket
    
    logger.info(f"WebSocket connection established for {client_id}")
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "client_id": client_id
        })
        
        # Wait for messages from client
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({
                "type": "message",
                "content": f"Received: {data}"
            })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {client_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket: {str(e)}")
    finally:
        await cleanup_resources(client_id)

async def cleanup_resources(client_id: str):
    """Clean up resources for a client."""
    # Close peer connection
    if client_id in peer_connections:
        pc = peer_connections[client_id]
        await pc.close()
        del peer_connections[client_id]
        logger.info(f"Closed peer connection for {client_id}")
    
    # Close WebSocket
    if client_id in websocket_connections:
        try:
            await websocket_connections[client_id].close()
        except:
            pass
        del websocket_connections[client_id]
        logger.info(f"Removed WebSocket for {client_id}")