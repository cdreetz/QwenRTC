import asyncio
import json
import logging
import sys
import argparse
from typing import Dict, Optional

import pyaudio
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from aiortc.mediastreams import MediaStreamError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio config
SAMPLE_RATE = 24000  # Match Qwen2.5-Omni's sample rate
CHANNELS = 1
CHUNK = 960  # 40ms chunks at 24kHz
FORMAT = pyaudio.paInt16

# Default WebRTC config
SERVER_URL = "ws://localhost:8000/ws"

class AudioInputTrack(MediaStreamTrack):
    """
    A track that captures audio from the microphone.
    """
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        self.frame_count = 0
        logger.info(f"Audio input initialized with sample rate {SAMPLE_RATE}")

    async def recv(self):
        """Get a frame from the microphone."""
        from av import AudioFrame
        import numpy as np

        data = self.stream.read(CHUNK, exception_on_overflow=False)
        frame = AudioFrame.from_ndarray(
            np.frombuffer(data, np.int16).reshape(-1, 1),
            format="s16",
            layout="mono"
        )
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self.frame_count * CHUNK
        frame.time_base = 1 / SAMPLE_RATE
        self.frame_count += 1
        return frame

    def stop(self):
        """Stop and clean up."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        logger.info("Audio input stopped")

class AudioOutputHandler:
    """
    Handle playing audio output.
    """
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK
        )
        logger.info(f"Audio output initialized with sample rate {SAMPLE_RATE}")

    def play(self, data):
        """Play audio data."""
        try:
            self.stream.write(data)
        except Exception as e:
            logger.error(f"Error playing audio: {e}")

    def stop(self):
        """Stop and clean up."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        logger.info("Audio output stopped")

async def run_client(server_url=SERVER_URL):
    """Main client function."""
    try:
        # Create peer connection
        pc = RTCPeerConnection()
        
        # Set up audio track
        audio_track = AudioInputTrack()
        output_handler = AudioOutputHandler()
        
        # Add the audio track
        pc.addTrack(audio_track)
        
        # Status flags
        is_connected = False
        
        # Set up handlers for received tracks
        @pc.on("track")
        def on_track(track):
            logger.info(f"Received {track.kind} track from server")
            
            async def process_track():
                try:
                    while True:
                        frame = await track.recv()
                        # If it's an audio frame, play it
                        if track.kind == "audio":
                            # Convert the frame to bytes and play
                            import numpy as np
                            audio_data = frame.to_ndarray().flatten().tobytes()
                            output_handler.play(audio_data)
                except MediaStreamError:
                    logger.info("Media stream ended")
            
            asyncio.create_task(process_track())
        
        # Connect to the server
        logger.info(f"Connecting to {server_url}")
        websocket = await websockets.connect(server_url)
        
        # Create and send offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        
        await websocket.send(json.dumps({
            "type": offer.type,
            "sdp": offer.sdp
        }))
        
        # Wait for answer
        response = await websocket.recv()
        data = json.loads(response)
        
        if data["type"] == "answer":
            answer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
            await pc.setRemoteDescription(answer)
            is_connected = True
            logger.info("WebRTC connection established")
        
        # ICE candidate handling
        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate:
                await websocket.send(json.dumps({
                    "type": "ice",
                    "candidate": candidate.to_json()
                }))
        
        # Connection state handling
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state changed to {pc.connectionState}")
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                nonlocal is_connected
                is_connected = False
        
        # Main interaction loop
        print("\n===== Qwen2.5-Omni Voice Interaction =====")
        print("Speak into your microphone. The AI will respond with voice.")
        print("Press Ctrl+C to exit.")
        
        try:
            # Keep running until user terminates
            while is_connected:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\nGracefully shutting down...")
        
        finally:
            # Clean up
            print("Closing connection...")
            audio_track.stop()
            output_handler.stop()
            await pc.close()
            await websocket.close()
            print("Connection closed.")
            
    except Exception as e:
        logger.error(f"Error in client: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni WebRTC Client")
    parser.add_argument("--server", type=str, default=SERVER_URL, help="WebRTC server URL")
    args = parser.parse_args()
    
    try:
        asyncio.run(run_client(args.server))
    except KeyboardInterrupt:
        print("Client terminated by user")
    except Exception as e:
        print(f"Client error: {e}")
        sys.exit(1)
