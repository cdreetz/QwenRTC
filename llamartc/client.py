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
    html_path = ROOT / "client_ui.html"
    if not html_path.exists():
        # Create the HTML file if it doesn't exist
        with open(html_path, "w") as f:
            # Save the HTML code to file
            with open(ROOT / "client_ui.html", "w") as f:
                f.write("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Assistant Client</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        button {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 10px 2px;
            cursor: pointer;
            border-radius: 4px;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        button.active {
            background-color: #45a049;
            box-shadow: 0 0 5px rgba(0,0,0,0.3);
        }
        .toggle-buttons {
            display: flex;
            gap: 10px;
            margin: 10px 0;
        }
        #status {
            margin: 20px 0;
            padding: 10px;
            border-radius: 4px;
            background-color: #f8f9fa;
        }
        #conversationBox {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            min-height: 200px;
            max-height: 500px;
            overflow-y: auto;
            background-color: #f9f9f9;
        }
        .conversationItem {
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }
        .user-message {
            background-color: #e3f2fd;
            padding: 8px 12px;
            border-radius: 8px;
            margin: 5px 0;
        }
        .assistant-message {
            background-color: #f1f8e9;
            padding: 8px 12px;
            border-radius: 8px;
            margin: 5px 0;
        }
        .conversationItem .timestamp {
            font-size: 0.8em;
            color: #666;
        }
        .controls {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        #serverUrl {
            flex-grow: 1;
            padding: 8px;
            font-size: 14px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <h1>Voice Assistant Client</h1>
    
    <div class="controls">
        <input type="text" id="serverUrl" placeholder="Server URL (e.g., http://192.168.1.100:8080)" value="">
        <button id="connectButton">Connect</button>
    </div>
    
    <div class="toggle-buttons">
        <button id="llmModeButton" class="active">LLM Mode</button>
        <button id="transcriptionModeButton">Transcription Only</button>
    </div>
    
    <div>
        <button id="startButton" disabled>Start Recording</button>
        <button id="stopButton" disabled>Stop Recording</button>
    </div>
    
    <div id="status">
        Not connected to server
    </div>
    
    <div id="conversationBox">
        <h3>Conversation:</h3>
        <div id="conversationContent"></div>
    </div>
    
    <script>
        // DOM elements
        const startButton = document.getElementById('startButton');
        const stopButton = document.getElementById('stopButton');
        const llmModeButton = document.getElementById('llmModeButton');
        const transcriptionModeButton = document.getElementById('transcriptionModeButton');
        const statusDiv = document.getElementById('status');
        const conversationContent = document.getElementById('conversationContent');
        const serverUrl = document.getElementById('serverUrl');
        const connectButton = document.getElementById('connectButton');
        
        // Initialize default server URL from localStorage
        serverUrl.value = localStorage.getItem('serverUrl') || 'http://192.168.1.100:8080';
        
        // Set up event listeners
        startButton.addEventListener('click', startRecording);
        stopButton.addEventListener('click', stopRecording);
        connectButton.addEventListener('click', checkServerConnection);
        llmModeButton.addEventListener('click', () => setMode('llm'));
        transcriptionModeButton.addEventListener('click', () => setMode('transcription'));
        
        // Global state
        let isConnected = false;
        let isRecording = false;
        let currentMode = 'llm'; // Default to LLM mode
        let pollingInterval = null;
        
        // Function to set the mode (LLM or transcription only)
        async function setMode(mode) {
            try {
                const response = await fetch('/set_mode', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode: mode })
                });
                
                if (response.ok) {
                    currentMode = mode;
                    
                    // Update UI
                    if (mode === 'llm') {
                        llmModeButton.classList.add('active');
                        transcriptionModeButton.classList.remove('active');
                    } else {
                        llmModeButton.classList.remove('active');
                        transcriptionModeButton.classList.add('active');
                    }
                }
            } catch (e) {
                console.error('Error setting mode:', e);
            }
        }
        
        // Function to check server connection
        async function checkServerConnection() {
            const url = serverUrl.value.trim();
            if (!url) {
                alert('Please enter a valid server URL');
                return;
            }
            
            // Save to localStorage
            localStorage.setItem('serverUrl', url);
            
            try {
                statusDiv.textContent = 'Connecting to server...';
                
                const response = await fetch(`${url}/status`, {
                    method: 'GET',
                    headers: { 'Content-Type': 'application/json' }
                });
                
                if (response.ok) {
                    const data = await response.json();
                    isConnected = true;
                    startButton.disabled = false;
                    statusDiv.textContent = `Connected to server running Whisper (${data.whisper_model}) and LLaMA (${data.llama_model}) on ${data.device_info.device} (${data.device_info.cuda_device_name || 'CPU'})`;
                    
                    // Start polling for conversation updates
                    startPolling();
                } else {
                    isConnected = false;
                    startButton.disabled = true;
                    statusDiv.textContent = `Failed to connect to server: ${response.status} ${response.statusText}`;
                }
            } catch (e) {
                isConnected = false;
                startButton.disabled = true;
                statusDiv.textContent = `Error connecting to server: ${e.message}`;
            }
        }
        
        // Function to start recording
        async function startRecording() {
            if (!isConnected) {
                alert('Not connected to server');
                return;
            }
            
            try {
                const response = await fetch('/start_recording', {
                    method: 'POST'
                });
                
                if (response.ok) {
                    isRecording = true;
                    startButton.disabled = true;
                    stopButton.disabled = false;
                    statusDiv.textContent += ' - Recording';
                } else {
                    statusDiv.textContent = `Failed to start recording: ${response.status} ${response.statusText}`;
                }
            } catch (e) {
                statusDiv.textContent = `Error starting recording: ${e.message}`;
            }
        }
        
        // Function to stop recording
        async function stopRecording() {
            try {
                const response = await fetch('/stop_recording', {
                    method: 'POST'
                });
                
                if (response.ok) {
                    isRecording = false;
                    startButton.disabled = false;
                    stopButton.disabled = true;
                    statusDiv.textContent = statusDiv.textContent.replace(' - Recording', '');
                } else {
                    statusDiv.textContent = `Failed to stop recording: ${response.status} ${response.statusText}`;
                }
            } catch (e) {
                statusDiv.textContent = `Error stopping recording: ${e.message}`;
            }
        }
        
        // Function to poll for conversation updates
        function startPolling() {
            if (pollingInterval) {
                clearInterval(pollingInterval);
            }
            
            pollingInterval = setInterval(async () => {
                try {
                    // Poll the appropriate endpoint based on mode
                    const endpoint = currentMode === 'llm' ? '/llm_results' : '/transcripts';
                    const response = await fetch(endpoint);
                    
                    if (response.ok) {
                        const data = await response.json();
                        updateConversationDisplay(data, currentMode);
                    }
                } catch (e) {
                    console.error('Error polling for updates:', e);
                }
            }, 500);
        }
        
        // Function to update the conversation display
        function updateConversationDisplay(items, mode) {
            conversationContent.innerHTML = '';
            
            if (items.length === 0) {
                conversationContent.innerHTML = '<p>No conversation yet</p>';
                return;
            }
            
            // Sort by timestamp (newest first)
            items.sort((a, b) => b.timestamp - a.timestamp);
            
            // Display each item
            items.forEach(item => {
                const conversationItem = document.createElement('div');
                conversationItem.className = 'conversationItem';
                
                if (mode === 'llm') {
                    // Display user's transcribed message
                    const userMessage = document.createElement('div');
                    userMessage.className = 'user-message';
                    userMessage.textContent = item.transcription || '[No transcription]';
                    
                    // Display assistant's response
                    const assistantMessage = document.createElement('div');
                    assistantMessage.className = 'assistant-message';
                    assistantMessage.textContent = item.response || '[No response]';
                    
                    // Display timestamp and processing time
                    const timestamp = document.createElement('p');
                    timestamp.className = 'timestamp';
                    const date = new Date(item.timestamp * 1000);
                    timestamp.textContent = `${date.toLocaleTimeString()} (transcription: ${item.transcription_time.toFixed(2)}s, LLM: ${item.llm_time.toFixed(2)}s)`;
                    
                    conversationItem.appendChild(userMessage);
                    conversationItem.appendChild(assistantMessage);
                    conversationItem.appendChild(timestamp);
                } else {
                    // Transcription-only mode
                    const text = document.createElement('p');
                    text.textContent = item.text || '[No text]';
                    
                    const timestamp = document.createElement('p');
                    timestamp.className = 'timestamp';
                    const date = new Date(item.timestamp * 1000);
                    timestamp.textContent = `${date.toLocaleTimeString()} (processed in ${item.process_time.toFixed(2)}s)`;
                    
                    conversationItem.appendChild(text);
                    conversationItem.appendChild(timestamp);
                }
                
                conversationContent.appendChild(conversationItem);
            });
        }
        
        // Try to connect to the default server on page load
        window.addEventListener('load', checkServerConnection);
    </script>
</body>
</html>
