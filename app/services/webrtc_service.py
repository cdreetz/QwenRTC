import asyncio
import logging
import av
import numpy as np
from typing import Dict, Optional, List, Callable, Awaitable
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay, MediaBlackhole

logger = logging.getLogger(__name__)

class AudioProcessTrack(MediaStreamTrack):
    """
    A MediaStreamTrack that processes audio from another track and runs it through the model.
    """
    kind = "audio"
    
    def __init__(self, track, callback):
        super().__init__()
        self.track = track
        self.callback = callback
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run_audio_processing())
        
    async def _run_audio_processing(self):
        while True:
            try:
                frame = await self._queue.get()
                
                # Convert audio frame to numpy array
                audio_frame = frame.to_ndarray()
                sample_rate = frame.sample_rate
                
                # Run the audio through the callback (model inference)
                await self.callback(audio_frame, sample_rate)
                
            except Exception as e:
                logger.error(f"Error processing audio: {str(e)}")
            finally:
                self._queue.task_done()
    
    async def recv(self):
        frame = await self.track.recv()
        
        # Queue the frame for processing (non-blocking)
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass
            
        # Return the original frame
        return frame
        
    def stop(self):
        if self._task and not self._task.cancelled():
            self._task.cancel()
        super().stop()


class WebRTCService:
    def __init__(self):
        self.relay = MediaRelay()
        self.peer_connections: Dict[str, RTCPeerConnection] = {}
        self.audio_tracks: Dict[str, MediaStreamTrack] = {}
        
    async def create_peer_connection(self, client_id: str, audio_callback: Callable[[np.ndarray, int], Awaitable[None]]) -> RTCPeerConnection:
        """
        Create a new WebRTC peer connection for a client.
        
        Args:
            client_id: Unique identifier for the client
            audio_callback: Async callback function that takes audio data and sample rate
            
        Returns:
            RTCPeerConnection object
        """
        # Create a new peer connection
        pc = RTCPeerConnection()
        self.peer_connections[client_id] = pc
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state for {client_id}: {pc.connectionState}")
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                await self.close_peer_connection(client_id)
                
        @pc.on("track")
        def on_track(track):
            logger.info(f"Track received from {client_id}: {track.kind}")
            
            if track.kind == "audio":
                # Create a processor track that processes the audio
                processor_track = AudioProcessTrack(
                    self.relay.subscribe(track),
                    audio_callback
                )
                self.audio_tracks[client_id] = processor_track
                
                # Add a sink to consume the media
                blackhole = MediaBlackhole()
                blackhole.addTrack(processor_track)
                asyncio.ensure_future(blackhole.start())
                
                @track.on("ended")
                async def on_ended():
                    logger.info(f"Track ended for {client_id}")
                    if client_id in self.audio_tracks:
                        processor_track = self.audio_tracks[client_id]
                        if processor_track:
                            processor_track.stop()
                        del self.audio_tracks[client_id]
                    
        return pc
        
    async def handle_offer(
        self, 
        client_id: str,
        offer: RTCSessionDescription,
        audio_callback: Callable[[np.ndarray, int], Awaitable[None]]
    ) -> RTCSessionDescription:
        """
        Handle a WebRTC offer from a client.
        
        Args:
            client_id: Unique identifier for the client
            offer: SDP offer from client
            audio_callback: Callback function for audio processing
            
        Returns:
            SDP answer
        """
        # Create new peer connection if it doesn't exist
        if client_id not in self.peer_connections:
            pc = await self.create_peer_connection(client_id, audio_callback)
        else:
            pc = self.peer_connections[client_id]
            
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return pc.localDescription
        
    async def close_peer_connection(self, client_id: str):
        """Close peer connection and clean up resources for a client"""
        if client_id in self.peer_connections:
            pc = self.peer_connections[client_id]
            
            # Close tracks
            if client_id in self.audio_tracks:
                track = self.audio_tracks[client_id]
                if track:
                    track.stop()
                del self.audio_tracks[client_id]
                
            # Close the peer connection
            await pc.close()
            del self.peer_connections[client_id]
            
            logger.info(f"Closed peer connection for {client_id}")