import os
import json
import time
import argparse
import re
import tempfile
import subprocess
import requests
import ssl
import hashlib
import signal
import sys
import atexit
import logging
import threading
from pathlib import Path
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
import queue
import base64
import wave

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("podcast2video.log")  # Also log to file
    ]
)
logger = logging.getLogger('podcast2video')

# Create a separate logger for HTTP requests with reduced verbosity
http_logger = logging.getLogger('http')
http_logger.setLevel(logging.WARNING)  # Only show warnings for HTTP issues

# Global variables to track state
current_operation = "initializing"
is_cancelling = False
temp_files = []
start_time = time.time()
last_progress_time = time.time()  # For tracking progress updates
progress_thread = None
task_start_time = None  # Track when current task started
task_name = None  # Track current task name
task_stack = []  # Stack to track nested operations
min_task_duration = 0.5  # Increased minimum duration to log (in seconds)

# List of known long-running operations that shouldn't trigger warnings
LONG_RUNNING_OPERATIONS = [
    'rendering',
    'encoding',
    'processing',
    'building video',
    'writing video',
    'concatenating',
    'transcribing',
    'enhancing',
    'generating',
    'creating video',
    'finalizing'
]

# Debug function
def debug_point(message, level=logging.INFO):  # Changed default level to INFO
    """Log debug information with consistent formatting"""
    global current_operation, last_progress_time, task_start_time, task_name, task_stack
    
    # If this is a new task, update task tracking
    if message != current_operation:
        # If we have a previous task, log its completion
        if task_name is not None:
            task_duration = time.time() - task_start_time if task_start_time else 0
            # Only log if duration is significant
            if task_duration >= min_task_duration:
                # Check if this was a long-running operation
                is_long_running = any(op in task_name.lower() for op in LONG_RUNNING_OPERATIONS)
                if is_long_running:
                    logger.info(f"Long-running task completed: {task_name} (duration: {task_duration:.1f}s)")
                else:
                    logger.info(f"Task completed: {task_name} (duration: {task_duration:.1f}s)")
        
        # Start tracking new task
        task_name = message
        task_start_time = time.time()
        last_progress_time = time.time()
        
        # Add to task stack for nested operations
        task_stack.append((message, task_start_time))
        
        # Log start of new task
        logger.info(f"Starting task: {message}")
    
    # Only log if it's a status update or important message
    if level == logging.INFO or "Status" in message or "Task completed" in message:
        logger.log(level, f"Status: {message}")
    current_operation = message
    last_progress_time = time.time()

def complete_task(task_to_complete):
    """Mark a task as complete and remove it from the stack"""
    global task_stack, task_name, task_start_time
    if task_stack and task_stack[-1][0] == task_to_complete:
        task_stack.pop()
        if task_stack:
            # Restore previous task context
            task_name, task_start_time = task_stack[-1]
        else:
            # No more tasks in stack
            task_name = None
            task_start_time = None

# Progress monitoring thread
def start_progress_monitoring(interval=5):
    """Start progress monitoring in a separate thread"""
    def monitor():
        global current_operation, last_progress_time, task_start_time, task_name, task_stack
        
        while True:
            time.sleep(interval)
            current_time = time.time()
            
            # Check if we have an active task
            if current_operation and task_start_time:
                elapsed = current_time - last_progress_time  # Changed from task_start_time to last_progress_time
                
                # Only show warnings for tasks that are actually running
                if elapsed > interval * 2:
                    # Check if this is a long-running operation
                    is_long_running = any(op in current_operation.lower() for op in LONG_RUNNING_OPERATIONS)
                    
                    if not is_long_running:
                        logger.warning(f"No progress updates for {elapsed:.1f}s while: {current_operation}")
                        last_progress_time = current_time  # Reset the timer after warning
            
            # Update last progress time
            last_progress_time = current_time
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    return monitor_thread

# Function with timeout
def with_timeout(func, args=(), kwargs={}, timeout_seconds=60, description="operation"):
    """Run a function with a timeout to prevent hanging"""
    result = [None]
    exception = [None]
    is_finished = [False]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
        finally:
            is_finished[0] = True
    
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        logger.warning(f"{description} is taking longer than {timeout_seconds} seconds")
        logger.warning(f"Still running: {current_operation}")
        return None, TimeoutError(f"{description} exceeded {timeout_seconds}s timeout")
    
    if exception[0]:
        return None, exception[0]
        
    if is_finished[0]:
        # Log completion of long-running operation
        if any(op in description.lower() for op in LONG_RUNNING_OPERATIONS):
            logger.info(f"Long-running operation completed: {description}")
        else:
            logger.info(f"Operation completed: {description}")
    else:
        logger.warning(f"{description} did not complete within {timeout_seconds} seconds")
    return result[0], None

# Import required modules
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    logger.error("PyDub not installed correctly. Run: pip install pydub")
    PYDUB_AVAILABLE = False

import wave

# Correct moviepy imports
try:
    import moviepy
    from moviepy import AudioFileClip, ImageClip, TextClip, CompositeVideoClip
    from moviepy import concatenate_videoclips, VideoFileClip
    MOVIEPY_AVAILABLE = True
    logger.info("Successfully imported MoviePy version: " + moviepy.__version__)
except ImportError as e:
    logger.error(f"MoviePy import error: {e}")
    MOVIEPY_AVAILABLE = False

# Optional API modules
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.error("OpenAI library not installed. Run: pip install openai")
    OPENAI_AVAILABLE = False

try:
    from stability_sdk import client as stability_client
    import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
    STABILITY_AVAILABLE = True
except ImportError:
    logger.error("Stability SDK not installed. Run: pip install stability-sdk")
    STABILITY_AVAILABLE = False

# Add after other global variables
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_key(element_name):
    """Generate a cache key for an element name"""
    # Normalize the element name for consistent caching
    normalized = element_name.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()

def get_cached_research(element_name):
    """Get cached research results if available"""
    cache_key = get_cache_key(element_name)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                logger.info(f"Using cached research for element: {element_name}")
                return cached_data["research"], cached_data["image_prompt"]
        except Exception as e:
            logger.warning(f"Error reading cache for {element_name}: {e}")
    
    return None, None

def cache_research_results(element_name, research, image_prompt):
    """Cache research results for future use"""
    cache_key = get_cache_key(element_name)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                "element_name": element_name,
                "research": research,
                "image_prompt": image_prompt,
                "timestamp": time.time()
            }, f)
        logger.info(f"Cached research results for element: {element_name}")
    except Exception as e:
        logger.warning(f"Error caching results for {element_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert podcast to enhanced video")
    parser.add_argument("--audio", required=True, help="Path to the podcast audio file (MP3 or WAV)")
    parser.add_argument("--output", default="enhanced_podcast.mp4", help="Output video file path")
    parser.add_argument("--temp_dir", default="temp", help="Temporary directory for intermediate files")
    parser.add_argument("--subtitle", choices=["always", "important_parts", "none"], default="always",
                      help="Subtitle display mode: always, important_parts, or none")
    parser.add_argument("--skip_transcription", action="store_true", help="Skip transcription if SRT file exists")
    parser.add_argument("--force_transcription", action="store_true", help="Force transcription even if SRT file exists")
    parser.add_argument("--transcript_dir", default="transcripts", help="Directory to store cached transcripts")
    parser.add_argument("--allow_dalle_fallback", action="store_true", help="Automatically allow DALL-E as a fallback without confirmation")
    parser.add_argument("--non_interactive", action="store_true", help="Run in non-interactive mode (will use defaults for all confirmations)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with additional logging")
    parser.add_argument("--limit_to_one_minute", action="store_true", help="Only process the first minute of audio")
    
    return parser

# Signal handler for clean shutdown
def handle_exit_signal(signum, frame):
    """Handle termination signals like Ctrl+C (SIGINT) or SIGTERM"""
    global is_cancelling
    signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT (Ctrl+C)"
    
    if is_cancelling:
        # If already cancelling and user presses Ctrl+C again, exit immediately
        print(f"\n\nForced exit requested. Exiting immediately without cleanup.")
        sys.exit(1)
    
    is_cancelling = True
    print(f"\n\n{signal_name} received. Gracefully cancelling {current_operation}...")
    print("Please wait for current operation to complete safely...")
    print("(Press Ctrl+C again to force immediate exit without cleanup)")
    
    # The actual cancellation happens in the main loop and long-running functions
    # which check the is_cancelling flag

# Register signal handlers
signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)

# Cleanup function to run on exit
def cleanup():
    """Clean up temporary files and resources on exit"""
    global temp_files, start_time
    
    if is_cancelling:
        elapsed_time = time.time() - start_time
        print(f"\nOperation cancelled after {elapsed_time:.1f} seconds")
        
        # Clean up any temporary files if needed
        cleanup_files = []
        for file in temp_files:
            if os.path.exists(file):
                try:
                    if os.path.isfile(file):
                        os.remove(file)
                        cleanup_files.append(file)
                except Exception as e:
                    print(f"Error removing temporary file {file}: {e}")
        
        if cleanup_files:
            print(f"Cleaned up {len(cleanup_files)} temporary files")
        
        print("Shutdown complete")

# Register cleanup function to run on normal exit
atexit.register(cleanup)

# Check if operation should be cancelled
def check_cancel():
    """Check if the operation has been cancelled and raise exception if needed"""
    if is_cancelling:
        raise KeyboardInterrupt("Operation cancelled by user")

# Test API connectivity before using it
def test_api_connection(client, api_type="openai"):
    """Test API connectivity with a simple request"""
    debug_point(f"Testing {api_type} API connection")
    try:
        if api_type.lower() == "openai":
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            logger.info(f"OpenAI API test successful: {response.choices[0].message.content}")
            return True
        elif api_type.lower() == "deepseek":
            # Try multiple potential DeepSeek model names in case of API changes
            potential_models = ["deepseek-chat", "deepseek-coder"]  # Removed "deepseek-llm" as it doesn't exist
            
            for model in potential_models:
                try:
                    logger.info(f"Trying DeepSeek model: {model}")
                    response = client.chat.completions.create(
                        model=model, 
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=5
                    )
                    logger.info(f"DeepSeek API test successful with model {model}: {response.choices[0].message.content}")
                    # Update the global chat_model variable to use the working model
                    global chat_model
                    chat_model = model
                    return True
                except Exception as model_error:
                    logger.warning(f"DeepSeek model {model} failed: {model_error}")
                    continue
            
            # If we've tried all models and none worked
            logger.error("All DeepSeek models failed the test")
            return False
        else:
            logger.warning(f"Unknown API type: {api_type}")
            return False
    except Exception as e:
        logger.error(f"API test failed for {api_type}: {e}")
        return False

# Initialize clients - add debug information
try:
    debug_point("Initializing API clients")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    if openai_api_key:
        logger.info("OPENAI_API_KEY found in environment")
    else:
        logger.warning("OPENAI_API_KEY not found in environment")
    
    if deepseek_api_key:
        logger.info("DEEPSEEK_API_KEY found in environment")
    else:
        logger.info("DEEPSEEK_API_KEY not found in environment")
    
    # Initialize the AI client based on available keys
    ai_client = None
    api_type = None
    
    # Use DeepSeek if available, otherwise fallback to OpenAI
    if deepseek_api_key:
        debug_point("Initializing DeepSeek client")
        try:
            # Check if OpenAI library is installed
            from openai import OpenAI
            ai_client = OpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com/v1"
            )
            chat_model = "deepseek-chat"  # Changed from "deepseek-llm" to "deepseek-chat" which works
            embedding_model = "deepseek-embedding"
            api_type = "deepseek"
            logger.info("Using DeepSeek API for language models")
            
            # Test DeepSeek API immediately
            if not test_api_connection(ai_client, api_type):
                logger.warning("DeepSeek API test failed. Falling back to OpenAI if available.")
                # Reset client if test failed
                ai_client = None
                api_type = None
                
                # If OpenAI is available, try that instead
                if openai_api_key:
                    logger.info("Falling back to OpenAI API")
                else:
                    logger.error("Both DeepSeek test failed and no OpenAI API key available")
            
        except ImportError:
            logger.error("OpenAI library not installed. Run: pip install openai")
            raise
    
    # If DeepSeek initialization failed or not available, try OpenAI
    if ai_client is None and openai_api_key:
        debug_point("Initializing OpenAI client")
        try:
            # Check if OpenAI library is installed
            from openai import OpenAI
            ai_client = OpenAI(api_key=openai_api_key)
            chat_model = "gpt-4o"
            embedding_model = "text-embedding-3-large"
            api_type = "openai"
            logger.info("Using OpenAI API for language models")
            
            # Test OpenAI API
            if not test_api_connection(ai_client, api_type):
                logger.warning("OpenAI API test failed. API may not be working correctly.")
                # Keep the client but warn the user
        except ImportError:
            logger.error("OpenAI library not installed. Run: pip install openai")
            raise
    
    # If still no working client
    if ai_client is None:
        logger.error("No API clients could be initialized - both OPENAI_API_KEY and DEEPSEEK_API_KEY are missing or invalid")
        
except Exception as e:
    logger.error(f"Error initializing API clients: {e}", exc_info=True)

# Initialize Stability API for images
stability_api = None  # Initialize as None by default

def generate_image_stability(prompt, output_path, width=640, height=480, steps=30, samples=1, allow_dalle_fallback=False, non_interactive=False):
    """Generate an image using Stability API with optional DALL-E fallback"""
    try:
        # Log the request details
        logger.info("Stability AI Image Generation Request:")
        logger.info(f"- Prompt: {prompt}")
        logger.info(f"- Output path: {output_path}")
        logger.info(f"- Dimensions: {width}x{height}")
        logger.info(f"- Allow DALL-E fallback: {allow_dalle_fallback}")
        
        if not stability_api:
            logger.warning("Stability API not initialized")
            if allow_dalle_fallback:
                logger.info("Attempting DALL-E fallback due to missing Stability API")
                return generate_image_dalle(prompt, output_path, width, height)
            else:
                logger.error("Stability API not working and DALL-E fallback not allowed")
                return False
        
        # Generate the image with correct parameters
        response = stability_api.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            json={
                "text_prompts": [{"text": prompt}],
                "width": 1024,  # Use standard 1024x1024 dimensions
                "height": 1024,
                "steps": steps,
                "samples": samples,
                "cfg_scale": 8.0,
                "style_preset": "photographic"  # Use a valid style preset
            }
        )
        
        # Log response details
        logger.info("Received response from Stability API")
        logger.info(f"- Status code: {response.status_code}")
        
        # Check response status
        if response.status_code == 200:
            response_data = response.json()
            
            # Log the full response for debugging
            logger.debug(f"Full API response: {json.dumps(response_data, indent=2)}")
            
            # Check for image data in the response
            if 'artifacts' in response_data and response_data['artifacts']:
                # Find the first image artifact
                for artifact in response_data['artifacts']:
                    if artifact.get('type') == 'image' or artifact.get('type') is None:
                        # Get the base64 image data
                        image_data = artifact.get('base64')
                        if image_data:
                            try:
                                # Decode and save the image
                                with open(output_path, "wb") as f:
                                    f.write(base64.b64decode(image_data))
                                
                                # Verify file was created
                                if os.path.exists(output_path):
                                    file_size = os.path.getsize(output_path)
                                    logger.info(f"Successfully saved image to {output_path} (size: {file_size} bytes)")
                                    print(f"✓ Image generated successfully: {os.path.basename(output_path)}")
                                    return True
                                else:
                                    logger.error(f"Failed to save image to {output_path}")
                                    print(f"✗ Failed to save generated image")
                                    return False
                            except Exception as e:
                                logger.error(f"Error saving image: {str(e)}")
                                print(f"✗ Error saving generated image: {e}")
                                return False
            
            logger.warning("No valid image data found in response")
            print(f"✗ Stability API failed to generate image")
            
            if allow_dalle_fallback:
                logger.info("Attempting DALL-E fallback")
                print("Attempting fallback to DALL-E...")
                return generate_image_dalle(prompt, output_path, width, height)
            else:
                return False
        else:
            logger.warning(f"Invalid response from Stability API: {response.status_code} - {response.text}")
            print(f"✗ Stability API returned error status code: {response.status_code}")
            
            if allow_dalle_fallback:
                logger.info("Attempting DALL-E fallback")
                print("Attempting fallback to DALL-E...")
                return generate_image_dalle(prompt, output_path, width, height)
            else:
                return False
                
    except Exception as e:
        logger.error(f"Error generating image with Stability API: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"✗ Error generating image: {e}")
        
        if allow_dalle_fallback:
            logger.info("Attempting DALL-E fallback after error")
            print("Attempting fallback to DALL-E...")
            return generate_image_dalle(prompt, output_path, width, height)
        else:
            return False

def create_video_segment(segment, output_path, audio_file_path, style="modern"):
    """Create a video segment with visuals and audio"""
    try:
        # Log start of video segment creation
        logger.info(f"Creating video segment: {output_path}")
        
        # Create video clip with visuals
        video_clip = ImageClip(segment["visuals"]["main_image"])
        video_duration = segment["end"] - segment["start"]
        video_clip = video_clip.set_duration(video_duration)
        
        # Extract audio segment
        audio_clip = AudioFileClip(audio_file_path).subclip(segment["start"], segment["end"])
        
        # Combine video and audio
        final_clip = video_clip.set_audio(audio_clip)
        
        # Write video segment with optimized settings
        final_clip.write_videofile(
            output_path,
            fps=24,
            codec='libx264',
            audio_codec='aac',
            preset='ultrafast',  # Faster encoding
            threads=4,  # Use multiple threads
            bitrate='2000k'  # Reduced bitrate for faster processing
        )
        
        # Clean up
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        
        # Log completion
        logger.info(f"Completed video segment creation: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating video segment: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def create_final_video(enhanced_segments, audio_file, output_path, temp_dir):
    """Create the final video with all segments"""
    try:
        # Log start of final video creation
        logger.info("Starting final video creation")
        
        # Create video clips from segments
        video_clips = []
        for i, segment in enumerate(enhanced_segments):
            segment_path = os.path.join(temp_dir, f"segment_{i}.mp4")
            if os.path.exists(segment_path):
                clip = VideoFileClip(segment_path)
                video_clips.append(clip)
        
        # Concatenate all clips
        final_clip = concatenate_videoclips(video_clips)
        
        # Write final video with optimized settings
        final_clip.write_videofile(
            output_path,
            fps=24,
            codec='libx264',
            audio_codec='aac',
            preset='ultrafast',  # Faster encoding
            threads=4,  # Use multiple threads
            bitrate='2000k'  # Reduced bitrate for faster processing
        )
        
        # Clean up
        for clip in video_clips:
            clip.close()
        final_clip.close()
        
        # Log completion
        logger.info(f"Completed final video creation: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating final video: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def validate_audio_file(audio_path):
    """Validate the audio file format and return audio information"""
    try:
        # Check if file exists
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
        
        # Get file extension
        file_ext = os.path.splitext(audio_path)[1].lower()
        
        # Check if it's a WAV file
        if file_ext == '.wav':
            with wave.open(audio_path, 'rb') as wav_file:
                # Get audio properties
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                duration = n_frames / frame_rate
                
                # Log audio properties
                logger.info(f"WAV file properties:")
                logger.info(f"- Channels: {channels}")
                logger.info(f"- Sample width: {sample_width} bytes")
                logger.info(f"- Frame rate: {frame_rate} Hz")
                logger.info(f"- Duration: {duration:.2f} seconds")
                
                return {
                    'format': 'wav',
                    'channels': channels,
                    'sample_width': sample_width,
                    'frame_rate': frame_rate,
                    'duration': duration
                }
        
        # Check if it's an MP3 file
        elif file_ext == '.mp3':
            if not PYDUB_AVAILABLE:
                raise ValueError("PyDub is required for MP3 files. Please install it with: pip install pydub")
            
            audio = AudioSegment.from_mp3(audio_path)
            duration = len(audio) / 1000.0  # Convert milliseconds to seconds
            
            # Log audio properties
            logger.info(f"MP3 file properties:")
            logger.info(f"- Channels: {audio.channels}")
            logger.info(f"- Sample width: {audio.sample_width} bytes")
            logger.info(f"- Frame rate: {audio.frame_rate} Hz")
            logger.info(f"- Duration: {duration:.2f} seconds")
            
            return {
                'format': 'mp3',
                'channels': audio.channels,
                'sample_width': audio.sample_width,
                'frame_rate': audio.frame_rate,
                'duration': duration
            }
        
        else:
            raise ValueError(f"Unsupported audio format: {file_ext}. Please use WAV or MP3 files.")
            
    except wave.Error as e:
        raise ValueError(f"Invalid WAV file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error validating audio file: {str(e)}")

def transcribe_audio(audio_path, force_transcription=False, transcript_dir="transcripts", non_interactive=False, temp_dir="temp"):
    """Transcribe audio file using OpenAI's Whisper model"""
    try:
        # Check if transcription already exists
        transcript_path = os.path.join(transcript_dir, os.path.splitext(os.path.basename(audio_path))[0] + ".srt")
        
        if os.path.exists(transcript_path) and not force_transcription:
            logger.info(f"Using existing transcript: {transcript_path}")
            return transcript_path
        
        # Create transcript directory if it doesn't exist
        os.makedirs(transcript_dir, exist_ok=True)
        
        # Initialize transcription
        logger.info("Starting audio transcription")
        
        # Check for OpenAI API key
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for transcription")
        
        # Initialize OpenAI client specifically for transcription
        transcription_client = OpenAI(api_key=openai_api_key)
        
        # Load audio file with PyDub
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0  # Convert milliseconds to seconds
        
        # Split audio into chunks for processing
        chunk_duration = 300000  # 5 minutes in milliseconds
        chunks = []
        
        for start_time in range(0, len(audio), chunk_duration):
            end_time = min(start_time + chunk_duration, len(audio))
            chunk_path = os.path.join(temp_dir, f"chunk_{start_time}_{end_time}.wav")
            
            # Extract chunk
            chunk = audio[start_time:end_time]
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)
            
            logger.info(f"Created chunk: {chunk_path}")
        
        # Process each chunk
        transcript_segments = []
        for i, chunk_path in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Transcribe chunk using OpenAI's Whisper model
            with open(chunk_path, "rb") as audio_file:
                response = transcription_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
                
                # Add segments to transcript
                for segment in response.segments:
                    transcript_segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    })
            
            # Clean up chunk file
            os.remove(chunk_path)
        
        # Sort segments by start time
        transcript_segments.sort(key=lambda x: x["start"])
        
        # Write SRT file
        with open(transcript_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(transcript_segments, 1):
                # Format timestamps
                start = format_timestamp(segment["start"])
                end = format_timestamp(segment["end"])
                
                # Write segment
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{segment['text']}\n\n")
        
        logger.info(f"Transcription completed: {transcript_path}")
        return transcript_path
        
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def format_timestamp(seconds):
    """Format seconds into SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def extract_segments(transcript_path):
    """Extract segments from SRT transcript file"""
    try:
        segments = []
        current_segment = None
        
        with open(transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    if current_segment:
                        segments.append(current_segment)
                        current_segment = None
                    continue
                
                # If we don't have a current segment, this line should be a number
                if current_segment is None:
                    try:
                        int(line)  # Verify it's a number
                        current_segment = {'index': int(line)}
                    except ValueError:
                        logger.warning(f"Expected segment number, got: {line}")
                        continue
                
                # If we have a current segment but no timestamp, this line should be the timestamp
                elif 'timestamp' not in current_segment:
                    try:
                        start, end = line.split(' --> ')
                        current_segment['timestamp'] = {
                            'start': start,
                            'end': end
                        }
                    except ValueError:
                        logger.warning(f"Invalid timestamp format: {line}")
                        continue
                
                # If we have a current segment and timestamp, this line should be the text
                else:
                    current_segment['text'] = line
        
        # Add the last segment if it exists
        if current_segment:
            segments.append(current_segment)
        
        # Log the number of segments found
        logger.info(f"Extracted {len(segments)} segments from transcript")
        
        # If limit_to_one_minute is True, only keep segments from the first minute
        if args.limit_to_one_minute:
            filtered_segments = []
            for segment in segments:
                start_time = parse_timestamp(segment['timestamp']['start'])
                if start_time <= 60:  # Only keep segments within first minute
                    filtered_segments.append(segment)
            logger.info(f"Limited to {len(filtered_segments)} segments from first minute")
            return filtered_segments
        
        return segments
        
    except Exception as e:
        logger.error(f"Error extracting segments: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []

def parse_timestamp(timestamp):
    """Convert SRT timestamp to seconds"""
    try:
        hours, minutes, seconds = timestamp.split(':')
        seconds, milliseconds = seconds.split(',')
        total_seconds = (
            int(hours) * 3600 +
            int(minutes) * 60 +
            int(seconds) +
            int(milliseconds) / 1000.0
        )
        return total_seconds
    except Exception as e:
        logger.error(f"Error parsing timestamp {timestamp}: {str(e)}")
        return 0

def enhance_segments(segments):
    """Enhance segments with AI descriptions and visual prompts"""
    try:
        enhanced_segments = []
        total_segments = len(segments)
        
        for i, segment in enumerate(segments, 1):
            # Check for cancellation
            if is_cancelling:
                logger.info("Enhancement cancelled by user")
                return None
            
            # Log progress
            logger.info(f"Enhancing segment {i}/{total_segments}")
            
            # Get segment text and timestamps
            text = segment['text']
            start_time = parse_timestamp(segment['timestamp']['start'])
            end_time = parse_timestamp(segment['timestamp']['end'])
            
            # Create prompt for AI enhancement
            prompt = f"""You are a helpful assistant that analyzes podcast segments and provides structured descriptions and visual prompts.
Your task is to analyze this podcast segment and provide a JSON response with the following structure:

{{
    "description": "A clear, concise description of the main topic",
    "visual_prompt": "A detailed visual prompt for generating an image that represents this content",
    "key_points": ["Key point 1", "Key point 2", "Key point 3"]
}}

Podcast segment text: "{text}"

Important: Your response must be valid JSON. Do not include any text before or after the JSON object."""

            # Get AI enhancement
            try:
                response = ai_client.chat.completions.create(
                    model=chat_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that analyzes podcast segments and provides structured descriptions and visual prompts. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                # Get the response content
                content = response.choices[0].message.content.strip()
                
                # Log the raw response for debugging
                logger.debug(f"Raw API response: {content}")
                
                # Try to parse the JSON response
                try:
                    enhancement = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {content}")
                    logger.error(f"JSON error: {str(e)}")
                    # Try to extract JSON from the response if it's wrapped in other text
                    try:
                        # Look for JSON-like structure between curly braces
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            enhancement = json.loads(json_match.group())
                        else:
                            raise ValueError("No JSON structure found in response")
                    except Exception as e:
                        logger.error(f"Failed to extract JSON from response: {str(e)}")
                        # Create a fallback enhancement
                        enhancement = {
                            "description": f"Segment {i}: {text[:100]}...",
                            "visual_prompt": f"An image representing: {text[:100]}...",
                            "key_points": [text[:100]]
                        }
                
                # Validate required fields
                required_fields = ["description", "visual_prompt", "key_points"]
                for field in required_fields:
                    if field not in enhancement:
                        enhancement[field] = f"Missing {field} for segment {i}"
                
                # Create enhanced segment
                enhanced_segment = {
                    'start': start_time,
                    'end': end_time,
                    'text': text,
                    'description': enhancement['description'],
                    'visual_prompt': enhancement['visual_prompt'],
                    'key_points': enhancement['key_points']
                }
                
                enhanced_segments.append(enhanced_segment)
                
                # Log progress
                logger.info(f"Enhanced segment {i}/{total_segments}")
                
            except Exception as e:
                logger.error(f"Error enhancing segment {i}: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Create a fallback enhancement for this segment
                enhanced_segment = {
                    'start': start_time,
                    'end': end_time,
                    'text': text,
                    'description': f"Segment {i}: {text[:100]}...",
                    'visual_prompt': f"An image representing: {text[:100]}...",
                    'key_points': [text[:100]]
                }
                enhanced_segments.append(enhanced_segment)
                continue
        
        logger.info(f"Completed enhancement of {len(enhanced_segments)} segments")
        return enhanced_segments
        
    except Exception as e:
        logger.error(f"Error in enhance_segments: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def generate_visuals(enhanced_segments, temp_dir, allow_dalle_fallback=False, non_interactive=False):
    """Generate visuals for each enhanced segment"""
    try:
        segments_with_visuals = []
        total_segments = len(enhanced_segments)
        
        for i, segment in enumerate(enhanced_segments, 1):
            # Check for cancellation
            if is_cancelling:
                logger.info("Visual generation cancelled by user")
                return None
            
            # Log progress
            logger.info(f"Generating visuals for segment {i}/{total_segments}")
            
            # Create output path for image
            image_path = os.path.join(temp_dir, f"segment_{i}.png")
            
            # Generate image using Stability API with DALL-E fallback
            success = generate_image_stability(
                segment['visual_prompt'],
                image_path,
                width=1024,
                height=1024,
                allow_dalle_fallback=allow_dalle_fallback,
                non_interactive=non_interactive
            )
            
            if success:
                # Add image path to segment
                segment['visuals'] = {
                    'main_image': image_path
                }
                segments_with_visuals.append(segment)
                logger.info(f"Generated visuals for segment {i}/{total_segments}")
            else:
                logger.error(f"Failed to generate visuals for segment {i}")
                # Continue with next segment
                continue
        
        logger.info(f"Completed visual generation for {len(segments_with_visuals)} segments")
        return segments_with_visuals
        
    except Exception as e:
        logger.error(f"Error in generate_visuals: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    parser = main()
    args = parser.parse_args()
    start_time = time.time()
    
    # Start progress monitoring
    progress_thread = start_progress_monitoring(interval=5)
    
    try:
        debug_point("Starting podcast-to-video conversion")
        logger.info(f"Command line arguments: {args}")
        
        # Create temp and transcript directories
        debug_point(f"Creating directories: {args.temp_dir}, {args.transcript_dir}")
        os.makedirs(args.temp_dir, exist_ok=True)
        os.makedirs(args.transcript_dir, exist_ok=True)
        
        # Check for required API keys and warn user
        debug_point("Checking API keys")
        openai_key_available = bool(os.environ.get("OPENAI_API_KEY"))
        deepseek_key_available = bool(os.environ.get("DEEPSEEK_API_KEY"))
        stability_key_available = bool(os.environ.get("STABILITY_API_KEY"))
        
        if not openai_key_available and not deepseek_key_available:
            logger.error("No API keys found - both OPENAI_API_KEY and DEEPSEEK_API_KEY are missing.")
            logger.error("At least one API key is required for transcription and content enhancement.")
            if not args.non_interactive and not get_user_confirmation("Continue without any API keys?", default=False):
                logger.info("Exiting due to missing API keys")
                exit(1)
        elif not openai_key_available and deepseek_key_available:
            # Only log DeepSeek preference if it was successfully tested
            if api_type == "deepseek":
                logger.info("OPENAI_API_KEY not found, but DEEPSEEK_API_KEY is available.")
                logger.info("Will use DeepSeek API for transcription and content enhancement.")
                if not args.non_interactive and not get_user_confirmation("Continue using DeepSeek API?", default=True):
                    logger.info("Exiting as user chose not to use DeepSeek API")
                    exit(1)
            else:
                logger.warning("DeepSeek API key provided but API test failed.")
                if not args.non_interactive and not get_user_confirmation("Continue with potentially non-working APIs?", default=False):
                    logger.info("Exiting due to failed API tests")
                    exit(1)
        elif api_type == "openai":
            logger.info("Using OpenAI API for all operations.")
        
        # Initialize Stability API for images
        try:
            debug_point("Initializing Stability API client")
            stability_api_key = os.environ.get("STABILITY_API_KEY")
            if stability_api_key:
                try:
                    # Initialize the Stability API client with REST API
                    stability_api = requests.Session()
                    stability_api.headers.update({
                        "Authorization": f"Bearer {stability_api_key}",
                        "Content-Type": "application/json"
                    })
                    logger.info("Stability API initialized successfully")
                    
                    # Quick test of Stability API
                    try:
                        debug_point("Testing Stability API connection")
                        # Make a minimal request to test connectivity
                        test_response = stability_api.post(
                            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                            json={
                                "text_prompts": [{"text": "test"}],
                                "width": 1024,  # Use standard 1024x1024 dimensions
                                "height": 1024,
                                "steps": 30,
                                "samples": 1,
                                "cfg_scale": 8.0,
                                "style_preset": "photographic"  # Use a valid style preset
                            }
                        )
                        
                        # Check response status
                        if test_response.status_code == 200:
                            response_data = test_response.json()
                            if 'artifacts' in response_data and response_data['artifacts']:
                                logger.info("Stability API test successful")
                            else:
                                raise ValueError("Stability API test failed: No image artifacts in response")
                        else:
                            raise ValueError(f"Stability API test failed with status code {test_response.status_code}: {test_response.text}")
                            
                    except Exception as e:
                        logger.error(f"Stability API test failed: {e}")
                        logger.error(f"Error type: {type(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        stability_api = None  # Reset the API client if test fails
                        
                        # Exit the program if Stability API fails and DALL-E fallback is not allowed
                        if not args.allow_dalle_fallback:
                            logger.error("Stability API test failed and DALL-E fallback is not allowed.")
                            logger.error("Please check your Stability API key and configuration.")
                            logger.error("Or run with --allow_dalle_fallback to use DALL-E as a fallback.")
                            sys.exit(1)
                        else:
                            logger.warning("Stability API test failed, will use DALL-E as fallback")
                except Exception as e:
                    logger.error(f"Failed to initialize Stability API: {e}", exc_info=True)
                    stability_api = None
            else:
                stability_api = None
                logger.warning("No Stability API key found. Will attempt to use alternatives.")
        except Exception as e:
            logger.error(f"Error in Stability API initialization block: {e}", exc_info=True)
            stability_api = None
        
        # Check for image generation capabilities
        if not stability_key_available and not openai_key_available:
            logger.error("No image generation API keys available (neither STABILITY_API_KEY nor OPENAI_API_KEY)")
            logger.error("Please provide at least one API key for image generation")
            exit(1)
        elif not stability_key_available and not args.allow_dalle_fallback:
            logger.warning("STABILITY_API_KEY not found and DALL-E fallback not allowed.")
            logger.warning("Please provide a Stability API key or run with --allow_dalle_fallback")
            if not args.non_interactive and not get_user_confirmation("Continue using DALL-E for image generation?", default=False):
                logger.info("Exiting due to missing Stability API key")
                exit(1)
        
        # Check for the input audio file
        debug_point(f"Checking input file: {args.audio}")
        if not os.path.exists(args.audio):
            logger.error(f"Input audio file not found: {args.audio}")
            exit(1)
        
        # Validate the audio file format
        try:
            debug_point("Validating audio file format")
            audio_info = validate_audio_file(args.audio)
            logger.info("Audio file validation successful")
        except ValueError as e:
            logger.error(f"Audio validation error: {e}")
            exit(1)
            
        logger.info("Completed initialization and validation")
        
        # Proceed with transcription and conversion
        try:
            # Transcribe audio with timeout protection
            debug_point("Starting audio transcription")
            print("Transcribing audio. This may take several minutes...")
            transcript, error = with_timeout(
                transcribe_audio,
                args=(
                    args.audio, 
                    args.force_transcription,
                    args.transcript_dir,
                    args.non_interactive,
                    args.temp_dir
                ),
                timeout_seconds=600,  # 10 minute timeout for transcription
                description="Audio transcription"
            )
            
            if error:
                logger.error(f"Transcription error or timeout: {error}")
                exit(1)
                
            if transcript is None:
                logger.error("Transcription failed or was cancelled")
                exit(1)
                
            # Extract meaningful segments
            debug_point("Extracting segments from transcript")
            segments = extract_segments(transcript)
            logger.info(f"Extracted {len(segments)} segments")
            
            # Enhance segments with AI (with timeout protection)
            debug_point("Enhancing segments with AI descriptions")
            print("Enhancing segments with AI. This may take several minutes...")
            enhanced_segments, error = with_timeout(
                enhance_segments,
                args=(segments,),
                timeout_seconds=900,  # 15 minute timeout
                description="Segment enhancement"
            )
            
            if error:
                logger.error(f"Segment enhancement error or timeout: {error}")
                exit(1)
                
            if enhanced_segments is None:
                logger.error("Segment enhancement failed or was cancelled")
                exit(1)
            
            # Generate visuals for each segment (with timeout protection)
            debug_point("Generating visuals for each segment")
            print("Generating visuals. This may take several minutes...")
            enhanced_segments_with_visuals, error = with_timeout(
                generate_visuals,
                args=(
                    enhanced_segments, 
                    args.temp_dir,
                    args.allow_dalle_fallback,
                    args.non_interactive
                ),
                timeout_seconds=900,  # 15 minute timeout
                description="Visual generation"
            )
            
            if error:
                logger.error(f"Visual generation error or timeout: {error}")
                exit(1)
                
            if enhanced_segments_with_visuals is None:
                logger.error("Visual generation failed or was cancelled")
                exit(1)
            
            # Create final video (with timeout protection)
            debug_point("Creating final video")
            print("Creating final video. This may take several minutes...")
            output_path, error = with_timeout(
                create_final_video,
                args=(
                    enhanced_segments_with_visuals,
                    args.audio,
                    args.output,
                    args.temp_dir
                ),
                timeout_seconds=900,  # 15 minute timeout
                description="Video creation"
            )
            
            if error:
                logger.error(f"Video creation error or timeout: {error}")
                exit(1)
                
            if output_path is None:
                logger.error("Video creation failed or was cancelled")
                exit(1)
            
            total_time = time.time() - start_time
            logger.info(f"Video creation complete: {args.output} in {total_time:.1f} seconds")
            
            # Add a more prominent final success message
            print("\n")
            print("🎉 " + "=" * 30 + " PROCESS COMPLETED SUCCESSFULLY " + "=" * 30 + " 🎉")
            print("\n")
            print(f"⏱️  \033[1mTotal processing time:\033[0m \033[1;32m{total_time/60:.1f} minutes\033[0m")
            print(f"🔄 \033[1mAll operations completed with no errors\033[0m")
            print("\n")
            print("To play your video:")
            print(f"  \033[1;36mopen \"{os.path.abspath(args.output)}\"\033[0m")
            print("\n" + "=" * 80 + "\n")
            
        except Exception as e:
            logger.error(f"Error during processing: {e}", exc_info=True)
            exit(1)
        
    except KeyboardInterrupt:
        # This catch is for any cancellations that happen outside of the functions
        logger.info("\nOperation cancelled by user.")
        exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        exit(1)
