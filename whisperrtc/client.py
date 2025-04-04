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
transcription_results = []
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
        
class TranscriptionProcessor:
    def __init__(self, server_url, chunk_duration=3.0, sample_rate=16000):
        self.server_url = server_url
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.is_processing = False
        self.processor_thread = None
        
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
                        target=self._send_audio_for_transcription,
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
                
    def _send_audio_for_transcription(self, audio_bytes):
        """Send audio data to the server for transcription"""
        try:
            # Convert audio bytes to base64
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Prepare the payload
            payload = {
                "audio_base64": audio_base64,
                # Optionally specify a language
                # "language": "en"
            }
            
            # Send to the server
            response = requests.post(
                f"{self.server_url}/transcribe",
                json=payload,
                timeout=10.0
            )
            
            if response.status_code == 200:
                # Process the transcription result
                result = response.json()
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
            logging.error(f"Error sending audio for transcription: {e}")
            
# Web server for the client UI
async def index(request):
    """Serve the main HTML page"""
    html_path = ROOT / "client_ui.html"
    if not html_path.exists():
        # Create the HTML file if it doesn't exist
        with open(html_path, "w") as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Whisper Transcription Client</title>
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
        #status {
            margin: 20px 0;
            padding: 10px;
            border-radius: 4px;
            background-color: #f8f9fa;
        }
        #transcriptBox {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            min-height: 200px;
            max-height: 400px;
            overflow-y: auto;
            background-color: #f9f9f9;
        }
        .transcriptItem {
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }
        .transcriptItem p {
            margin: 5px 0;
        }
        .transcriptItem .timestamp {
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
    <h1>Whisper Transcription Client</h1>
    
    <div class="controls">
        <input type="text" id="serverUrl" placeholder="Server URL (e.g., http://192.168.1.100:8080)" value="">
        <button id="connectButton">Connect</button>
    </div>
    
    <div>
        <button id="startButton" disabled>Start Recording</button>
        <button id="stopButton" disabled>Stop Recording</button>
    </div>
    
    <div id="status">
        Not connected to server
    </div>
    
    <div id="transcriptBox">
        <h3>Transcripts:</h3>
        <div id="transcriptContent"></div>
    </div>
    
    <script>
        // DOM elements
        const startButton = document.getElementById('startButton');
        const stopButton = document.getElementById('stopButton');
        const statusDiv = document.getElementById('status');
        const transcriptContent = document.getElementById('transcriptContent');
        const serverUrl = document.getElementById('serverUrl');
        const connectButton = document.getElementById('connectButton');
        
        // Initialize default server URL from localStorage
        serverUrl.value = localStorage.getItem('serverUrl') || 'http://192.168.1.100:8080';
        
        // Set up event listeners
        startButton.addEventListener('click', startRecording);
        stopButton.addEventListener('click', stopRecording);
        connectButton.addEventListener('click', checkServerConnection);
        
        // Global state
        let isConnected = false;
        let isRecording = false;
        let pollingInterval = null;
        
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
                    statusDiv.textContent = `Connected to server running ${data.model_id} on ${data.device_info.device} (${data.device_info.cuda_device_name || 'CPU'})`;
                    
                    // Start polling for transcripts
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
        
        // Function to poll for transcripts
        function startPolling() {
            if (pollingInterval) {
                clearInterval(pollingInterval);
            }
            
            pollingInterval = setInterval(async () => {
                try {
                    const response = await fetch('/transcripts');
                    if (response.ok) {
                        const data = await response.json();
                        updateTranscriptDisplay(data);
                    }
                } catch (e) {
                    console.error('Error polling for transcripts:', e);
                }
            }, 500);
        }
        
        // Function to update the transcript display
        function updateTranscriptDisplay(transcripts) {
            transcriptContent.innerHTML = '';
            
            if (transcripts.length === 0) {
                transcriptContent.innerHTML = '<p>No transcripts yet</p>';
                return;
            }
            
            // Sort by timestamp (newest first)
            transcripts.sort((a, b) => b.timestamp - a.timestamp);
            
            // Display each transcript
            transcripts.forEach(transcript => {
                const item = document.createElement('div');
                item.className = 'transcriptItem';
                
                const text = document.createElement('p');
                text.textContent = transcript.text || '[No text]';
                
                const timestamp = document.createElement('p');
                timestamp.className = 'timestamp';
                const date = new Date(transcript.timestamp * 1000);
                timestamp.textContent = `${date.toLocaleTimeString()} (processed in ${transcript.process_time.toFixed(2)}s)`;
                
                item.appendChild(text);
                item.appendChild(timestamp);
                transcriptContent.appendChild(item);
            });
        }
        
        // Try to connect to the default server on page load
        window.addEventListener('load', checkServerConnection);
    </script>
</body>
</html>
            """)
            
    return web.FileResponse(html_path)
    
async def get_transcripts(request):
    """Return the list of transcription results"""
    return web.Response(
        text=json.dumps(transcription_results),
        content_type="application/json"
    )
    
async def start_recording(request):
    """Start recording audio"""
    global is_recording
    
    if is_recording:
        return web.Response(
            status=400,
            text=json.dumps({"error": "Already recording"}),
            content_type="application/json"
        )
        
    try:
        app["audio_recorder"].start()
        is_recording = True
        return web.Response(
            text=json.dumps({"status": "started"}),
            content_type="application/json"
        )
    except Exception as e:
        logging.error(f"Error starting recording: {e}")
        return web.Response(
            status=500,
            text=json.dumps({"error": str(e)}),
            content_type="application/json"
        )
        
async def stop_recording(request):
    """Stop recording audio"""
    global is_recording
    
    if not is_recording:
        return web.Response(
            status=400,
            text=json.dumps({"error": "Not recording"}),
            content_type="application/json"
        )
        
    try:
        app["audio_recorder"].stop()
        is_recording = False
        return web.Response(
            text=json.dumps({"status": "stopped"}),
            content_type="application/json"
        )
    except Exception as e:
        logging.error(f"Error stopping recording: {e}")
        return web.Response(
            status=500,
            text=json.dumps({"error": str(e)}),
            content_type="application/json"
        )

async def set_server_url(request):
    """Set the server URL"""
    global server_url
    
    try:
        data = await request.json()
        new_url = data.get("url")
        
        if not new_url:
            return web.Response(
                status=400,
                text=json.dumps({"error": "Missing URL"}),
                content_type="application/json"
            )
            
        server_url = new_url
        app["transcription_processor"].server_url = new_url
        
        return web.Response(
            text=json.dumps({"status": "updated", "url": new_url}),
            content_type="application/json"
        )
    except json.JSONDecodeError:
        return web.Response(
            status=400,
            text=json.dumps({"error": "Invalid JSON"}),
            content_type="application/json"
        )
        
async def on_startup(app):
    """Initialize components on client startup"""
    app["audio_recorder"] = AudioRecorder()
    app["transcription_processor"] = TranscriptionProcessor(server_url)
    app["transcription_processor"].start()
    
async def on_shutdown(app):
    """Clean up resources on client shutdown"""
    if app["audio_recorder"].is_recording:
        app["audio_recorder"].stop()
    app["audio_recorder"].close()
    
    if app["transcription_processor"].is_processing:
        app["transcription_processor"].stop()
    
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Whisper Transcription Client')
    parser.add_argument('--server', type=str, default="http://192.168.1.100:8080",
                        help='URL of the Whisper server (default: http://192.168.1.100:8080)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to run the client web server on (default: 8000)')
    return parser.parse_args()
    
def main():
    global server_url
    
    # Parse command line arguments
    args = parse_arguments()
    server_url = args.server
    
    # Create the web application
    app = web.Application()
    
    # Set up startup and shutdown handlers
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    
    # Set up routes
    app.router.add_get("/", index)
    app.router.add_get("/transcripts", get_transcripts)
    app.router.add_post("/start_recording", start_recording)
    app.router.add_post("/stop_recording", stop_recording)
    app.router.add_post("/set_server", set_server_url)
    
    # Run the server
    web.run_app(app, host="0.0.0.0", port=args.port)
    
if __name__ == "__main__":
    main()
