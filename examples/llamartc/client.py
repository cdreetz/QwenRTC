import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path
import argparse
import queue
import threading
import wave

import numpy as np
import pyaudio
import requests
from aiohttp import web

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create data directories
ROOT = Path(__file__).parent
AUDIO_DIR = ROOT / "audio_chunks"
AUDIO_DIR.mkdir(exist_ok=True)

# Global variables
audio_queue = queue.Queue()
is_recording = False
llm_mode = True  # Default to LLM mode
transcription_results = []
llm_results = []
server_url = "http://192.168.1.100:8080"  # Change to your PC's IP address

class AudioRecorder:
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paInt16
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.vad_enabled = True
        self.silence_threshold = 0.01
        self.silence_chunks = 15  # Number of chunks of silence before considering a pause
        self.silence_counter = 0
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for the PyAudio stream"""
        if self.is_recording:
            audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
        
    def start(self):
        """Start recording audio"""
        if self.stream is not None and self.is_recording:
            return
            
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        self.is_recording = True
        logging.info("Started recording")
        
    def stop(self):
        """Stop recording audio"""
        if self.stream is None or not self.is_recording:
            return
            
        self.is_recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None
        logging.info("Stopped recording")
        
    def close(self):
        """Close the PyAudio instance"""
        if self.stream is not None:
            self.stop()
        self.audio.terminate()
        
class AudioProcessor:
    def __init__(self, server_url, chunk_duration=3.0, sample_rate=16000):
        self.server_url = server_url
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.is_processing = False
        self.processor_thread = None
        self.llm_mode = True  # Default to LLM mode
        
    def start(self):
        """Start the processor thread"""
        if self.processor_thread is not None:
            return
            
        self.is_processing = True
        self.processor_thread = threading.Thread(target=self._processor_loop)
        self.processor_thread.daemon = True
        self.processor_thread.start()
        
    def stop(self):
        """Stop the processor thread"""
        self.is_processing = False
        if self.processor_thread is not None:
            self.processor_thread.join(timeout=2.0)
            self.processor_thread = None
            
    def _processor_loop(self):
        """Main loop for processing audio chunks"""
        chunk_buffer = []
        samples_needed = int(self.chunk_duration * self.sample_rate)
        samples_collected = 0
        
        while self.is_processing:
            try:
                # Try to get audio data from the queue with a timeout
                audio_data = audio_queue.get(block=True, timeout=0.1)
                chunk_buffer.append(audio_data)
                
                # Calculate how many samples we've collected
                samples_collected += len(audio_data) // 2  # Assuming 16-bit audio (2 bytes per sample)
                
                # If we have enough audio, process it
                if samples_collected >= samples_needed:
                    # Concatenate all chunks
                    audio_bytes = b''.join(chunk_buffer)
                    
                    # Start a new thread to send the audio to the server
                    threading.Thread(
                        target=self._process_audio,
                        args=(audio_bytes,)
                    ).start()
                    
                    # Reset the buffer
                    chunk_buffer = []
                    samples_collected = 0
                    
            except queue.Empty:
                # Queue timeout, continue the loop
                continue
            except Exception as e:
                logging.error(f"Error in processor loop: {e}")
                time.sleep(0.5)  # Avoid tight loop in case of repeated errors
                
    def _process_audio(self, audio_bytes):
        """Process the audio data - either transcribe or send to LLM based on mode"""
        try:
            logging.info(f"Processing audio chunk of size: {len(audio_bytes)} bytes")
            # Convert audio bytes to base64
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Prepare the payload
            payload = {
                "audio_base64": audio_base64,
                # Optionally specify a language
                # "language": "en"
            }
            
            # Choose the endpoint based on mode
            endpoint = "/speech_to_llm" if self.llm_mode else "/transcribe"
            
            # Send to the server
            response = requests.post(
                f"{self.server_url}{endpoint}",
                json=payload,
                timeout=30.0  # Longer timeout for LLM processing
            )
            
            if response.status_code == 200:
                result = response.json()
                logging.info(f"Response received")
                
                if self.llm_mode:
                    # Process LLM result
                    logging.info(f"Transcription: {result.get('transcription', '')}")
                    logging.info(f"LLM Response: {result.get('response', '')}")
                    
                    # Add to results list for the UI
                    llm_results.append({
                        "timestamp": time.time(),
                        "transcription": result.get("transcription", ""),
                        "response": result.get("response", ""),
                        "transcription_time": result.get("transcription_time", 0),
                        "llm_time": result.get("llm_time", 0),
                        "total_time": result.get("total_time", 0)
                    })
                    
                    # Keep only the last 10 results
                    while len(llm_results) > 10:
                        llm_results.pop(0)
                else:
                    # Process transcription result
                    logging.info(f"Transcription received: {result.get('text', '')}")
                    
                    # Add to results list for the UI
                    transcription_results.append({
                        "timestamp": time.time(),
                        "text": result.get("text", ""),
                        "chunks": result.get("chunks", []),
                        "process_time": result.get("process_time", 0)
                    })
                    
                    # Keep only the last 10 results
                    while len(transcription_results) > 10:
                        transcription_results.pop(0)
            else:
                logging.error(f"Error from server: {response.status_code} {response.text}")
                
        except Exception as e:
            logging.error(f"Error processing audio: {e}")
            
    def set_llm_mode(self, mode):
        """Set whether to use LLM or just transcription"""
        self.llm_mode = mode
        logging.info(f"LLM mode set to: {mode}")
            
# Web server for the client UI
async def index(request):
    """Serve the main HTML page"""
    html_path = ROOT / "index.html"

