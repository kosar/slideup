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
import shutil
from datetime import datetime
import numpy as np

# Set up logging first
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("podcast2video.log")  # Also log to file
    ]
)
logger = logging.getLogger('podcast2video')

# Initialize MoviePy
moviepy_editor = None
MOVIEPY_AVAILABLE = False

try:
    import moviepy.editor as moviepy_editor
    MOVIEPY_AVAILABLE = True
    logger.info("Successfully imported MoviePy")
except ImportError as e:
    logger.error(f"MoviePy import error: {e}")
    MOVIEPY_AVAILABLE = False

# Global configuration
TIME_LIMIT_SECONDS = 60  # 6 minutes
TIME_LIMIT_MS = int(TIME_LIMIT_SECONDS * 1000)  # Convert to milliseconds

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

# Import optional modules
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.error("OpenAI library not installed. Run: pip install openai")
    OPENAI_AVAILABLE = False

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

def test_openai_api():
    """Test OpenAI API connectivity."""
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OpenAI API key is not set in environment variables")
        return False
    
    try:
        # Initialize the OpenAI client
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Make a minimal completion request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=5
        )
        
        logger.info("Successfully connected to OpenAI API")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to OpenAI API: {str(e)}")
        return False

def test_stability_api():
    """Test Stability API connectivity."""
    if not os.environ.get("STABILITY_API_KEY"):
        logger.error("Stability API key is not set in environment variables")
        return False
    
    try:
        # Make a simple GET request to check API status
        headers = {
            "Authorization": f"Bearer {os.environ.get('STABILITY_API_KEY')}"
        }
        response = requests.get("https://api.stability.ai/v1/user/balance", headers=headers)
        
        if response.status_code == 200:
            logger.info("Successfully connected to Stability API")
            return True
        else:
            logger.error(f"Failed to connect to Stability API: {response.status_code} {response.text}")
            return False
    except Exception as e:
        logger.error(f"Failed to connect to Stability API: {str(e)}")
        return False

def setup_signal_handling():
    """Set up signal handlers for graceful termination."""
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down gracefully")
        sys.exit(0)
    
    # Register SIGINT (Ctrl+C) handler
    signal.signal(signal.SIGINT, signal_handler)
    # Register SIGTERM handler
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Signal handlers registered")

def cleanup():
    """Clean up resources before exiting."""
    logger.info("Cleaning up resources")
    # Add any cleanup code here (temp files, etc.)

def main():
    parser = argparse.ArgumentParser(description='Convert podcast audio to video with AI-generated visuals')
    parser.add_argument('--audio', type=str, required=True, help='Path to input audio file')
    parser.add_argument('--output', type=str, default='enhanced_podcast.mp4', help='Path to output video file')
    parser.add_argument('--limit_to_one_minute', action='store_true', help='Limit processing to first minute of audio')
    parser.add_argument('--non_interactive', action='store_true', help='Run in non-interactive mode')
    parser.add_argument('--test_openai', action='store_true', help='Test OpenAI API connectivity')
    parser.add_argument('--test_stability', action='store_true', help='Test Stability API connectivity')
    args = parser.parse_args()

    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    try:
        # Check for required API keys
        stability_key_available = bool(os.getenv('STABILITY_API_KEY'))
        if not stability_key_available:
            logger.error("STABILITY_API_KEY not found in environment variables")
            sys.exit(1)

        # Validate input audio file
        if not validate_audio_file(args.audio):
            sys.exit(1)

        # Create necessary directories
        os.makedirs("temp", exist_ok=True)
        os.makedirs("transcripts", exist_ok=True)
        os.makedirs("cache", exist_ok=True)

        # Process the audio file
        process_audio_file(args.audio, args.output, args.limit_to_one_minute, args.non_interactive)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup()

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

def generate_image_stability(prompt, output_path, width=1024, height=1024, steps=20, samples=1, non_interactive=False):
    """Generate an image using Stability API"""
    logger.info(f"Generating image with prompt: {prompt}")
    logger.info(f"- Width: {width}")
    logger.info(f"- Height: {height}")
    logger.info(f"- Steps: {steps}")
    logger.info(f"- Samples: {samples}")

    try:
        # Check if Stability API key is available
        stability_key = os.getenv('STABILITY_API_KEY')
        if not stability_key:
            logger.error("STABILITY_API_KEY not found in environment variables")
            return False

        # Make API request to Stability
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {stability_key}"
            },
            json={
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7.5,
                "height": height,
                "width": width,
                "samples": samples,
                "steps": steps,
                "style_preset": "photographic"
            }
        )

        if response.status_code != 200:
            logger.error(f"Stability API request failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False

        # Process the response and save the image
        data = response.json()
        if "artifacts" in data and len(data["artifacts"]) > 0:
            image_data = base64.b64decode(data["artifacts"][0]["base64"])
            with open(output_path, "wb") as f:
                f.write(image_data)
            logger.info(f"Successfully generated and saved image to {output_path}")
            return True
        else:
            logger.error("No image data in Stability API response")
            return False

    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return False

def create_video_segment(segment, output_path, audio_file_path, style="modern"):
    """Create a video segment with visuals and audio"""
    try:
        # Log segment details
        logger.info(f"Creating video segment: {output_path}")
        logger.info(f"Segment duration: {segment['end'] - segment['start']:.2f} seconds")
        logger.info(f"Segment start: {segment['start']:.2f}, end: {segment['end']:.2f}")
        
        # Create video clip with visuals
        video_clip = moviepy_editor.ImageClip(segment["visuals"]["main_image"])
        segment_duration = segment["end"] - segment["start"]
        
        # Set video duration to match segment duration
        video_clip = video_clip.with_duration(segment_duration)
        logger.info(f"Video clip duration set to: {video_clip.duration:.2f} seconds")
        
        # Extract audio segment with exact timing using ffmpeg directly
        temp_audio = output_path + ".temp.wav"
        audio_cmd = [
            "ffmpeg", "-y",
            "-i", audio_file_path,
            "-ss", str(segment["start"]),
            "-t", str(segment_duration),
            "-vn",  # No video
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            temp_audio
        ]
        logger.info(f"Extracting audio with command: {' '.join(audio_cmd)}")
        result = subprocess.run(audio_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"Error extracting audio: {result.stderr}")
        
        if not os.path.exists(temp_audio):
            raise ValueError(f"Audio extraction failed: {temp_audio} was not created")
        
        # Load the extracted audio
        audio_clip = moviepy_editor.AudioFileClip(temp_audio)
        logger.info(f"Audio clip duration: {audio_clip.duration:.2f} seconds")
        
        # Ensure audio duration matches video duration
        audio_clip = audio_clip.with_duration(segment_duration)
        logger.info(f"Audio clip duration after adjustment: {audio_clip.duration:.2f} seconds")
        
        # Combine video and audio
        final_clip = video_clip.set_audio(audio_clip)
        logger.info(f"Final clip duration: {final_clip.duration:.2f} seconds")
        logger.info(f"Final clip has audio: {final_clip.audio is not None}")
        
        # Write video segment with audio using ffmpeg directly for better control
        temp_output = output_path + ".temp.mp4"
        final_clip.write_videofile(
            temp_output,
            fps=24,
            codec='libx264',
            audio_codec='aac',
            preset='ultrafast',
            threads=4,
            bitrate='2000k',
            audio=True,
            ffmpeg_params=[
                "-strict", "experimental",  # Enable experimental AAC encoder
                "-ac", "2",  # Force stereo audio
                "-ar", "44100",  # Set sample rate
                "-b:a", "192k"  # Set audio bitrate
            ]
        )
        
        # Clean up clips
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        
        # Clean up temporary audio file
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        
        # Verify the temporary output file
        if os.path.exists(temp_output):
            output_clip = moviepy_editor.VideoFileClip(temp_output)
            logger.info(f"Temp output file duration: {output_clip.duration:.2f} seconds")
            logger.info(f"Temp output file has audio: {output_clip.audio is not None}")
            if output_clip.audio is not None:
                logger.info(f"Audio duration: {output_clip.audio.duration:.2f} seconds")
            output_clip.close()
            
            # Validate output video
            if not output_clip.audio:
                raise ValueError(f"Temporary output video {temp_output} has no audio")
            if abs(output_clip.duration - segment_duration) > 0.1:  # Allow 0.1s tolerance
                raise ValueError(f"Temporary output video duration {output_clip.duration:.2f}s does not match expected duration {segment_duration:.2f}s")
            if output_clip.size != (1920, 1080):  # Check resolution
                raise ValueError(f"Temporary output video resolution {output_clip.size} does not match expected 1920x1080")
            if output_clip.fps != 24:  # Check frame rate
                raise ValueError(f"Temporary output video frame rate {output_clip.fps} does not match expected 24 fps")
            
            # If validation passed, move temp file to final output
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_output, output_path)
            logger.info(f"Final output file created: {output_path}")
            
            # Final verification of the output file
            final_clip = moviepy_editor.VideoFileClip(output_path)
            logger.info(f"Final output file duration: {final_clip.duration:.2f} seconds")
            logger.info(f"Final output file has audio: {final_clip.audio is not None}")
            if final_clip.audio is not None:
                logger.info(f"Final audio duration: {final_clip.audio.duration:.2f} seconds")
            final_clip.close()
            
            return output_path
        else:
            raise ValueError(f"Temporary output file {temp_output} was not created")
        
    except Exception as e:
        logger.error(f"Error creating video segment: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Clean up any temporary files
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except:
                pass
        if os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except:
                pass
        return None

def test_video_generation(segment_paths, audio_file, output_path, temp_dir):
    """Test function for video generation with saved segments"""
    try:
        # Always create a test segment for testing
        logger.info("Creating test segment for video generation test")
        
        # Validate input audio file
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return None
        
        logger.info(f"Using audio file: {audio_file}")
        
        # First, create a test video with silence
        silent_video_path = os.path.join(temp_dir, "silent_video.mp4")
        video_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "color=c=blue:s=1920x1080:r=24:d=2",  # 2-second blue video
            "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=stereo:d=2",  # 2 seconds of silence
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-b:v", "2000k",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            silent_video_path
        ]
        logger.info(f"Creating silent video with command: {' '.join(video_cmd)}")
        result = subprocess.run(video_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Error creating silent video: {result.stderr}")
            return None
        
        if not os.path.exists(silent_video_path):
            logger.error(f"Silent video was not created: {silent_video_path}")
            return None
        
        logger.info(f"Silent video created successfully: {silent_video_path}")
        
        # Extract 2 seconds of audio from input file
        audio_extract_path = os.path.join(temp_dir, "extracted_audio.wav")
        audio_cmd = [
            "ffmpeg", "-y",
            "-i", audio_file,
            "-t", "2",
            "-vn",  # No video
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            audio_extract_path
        ]
        logger.info(f"Extracting audio with command: {' '.join(audio_cmd)}")
        result = subprocess.run(audio_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Error extracting audio: {result.stderr}")
            return None
        
        if not os.path.exists(audio_extract_path):
            logger.error(f"Audio was not extracted: {audio_extract_path}")
            return None
        
        logger.info(f"Audio extracted successfully: {audio_extract_path}")
        
        # Create final test segment by replacing audio
        segment_path = os.path.join(temp_dir, "test_segment.mp4")
        combine_cmd = [
            "ffmpeg", "-y",
            "-i", silent_video_path,  # Input 1: video with silence
            "-i", audio_extract_path,  # Input 2: extracted audio
            "-c:v", "copy",  # Copy video stream
            "-c:a", "aac",  # Re-encode audio to AAC
            "-map", "0:v:0",  # Use video from first input
            "-map", "1:a:0",  # Use audio from second input
            "-b:a", "192k",
            "-shortest",
            segment_path
        ]
        logger.info(f"Creating test segment with command: {' '.join(combine_cmd)}")
        result = subprocess.run(combine_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Error creating test segment: {result.stderr}")
            return None
        
        if not os.path.exists(segment_path):
            logger.error(f"Test segment was not created: {segment_path}")
            return None
        
        logger.info(f"Test segment created successfully: {segment_path}")
        
        # Verify the test segment
        logger.info(f"Verifying test segment: {segment_path}")
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "stream=codec_type,codec_name,duration",
            "-of", "json",
            segment_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if probe_result.returncode == 0:
            logger.info(f"Test segment probe result: {probe_result.stdout}")
        else:
            logger.error(f"Error probing test segment: {probe_result.stderr}")
        
        # Create a concat file for ffmpeg
        concat_file = os.path.join(temp_dir, "concat.txt")
        with open(concat_file, "w") as f:
            f.write(f"file '{os.path.abspath(segment_path)}'\n")
        
        logger.info(f"Created concat file: {concat_file}")
        
        # Use ffmpeg concat demuxer to combine segments with explicit stream mapping
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-map", "0:v?",  # Map video stream if present
            "-map", "0:a?",  # Map audio stream if present
            "-c:v", "copy",  # Copy video codec
            "-c:a", "aac",  # Audio codec
            "-b:a", "192k",  # Audio bitrate
            "-ar", "44100",  # Audio sample rate
            "-ac", "2",  # Audio channels (stereo)
            output_path
        ]
        logger.info(f"Concatenating segments with command: {' '.join(concat_cmd)}")
        result = subprocess.run(concat_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Error concatenating segments: {result.stderr}")
            return None
        
        # Verify the output file
        if os.path.exists(output_path):
            probe_cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "stream=codec_type,codec_name,duration",
                "-of", "json",
                output_path
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if probe_result.returncode == 0:
                logger.info(f"Final output FFprobe result: {probe_result.stdout}")
            else:
                logger.error(f"Error probing output file: {probe_result.stderr}")
        
        logger.info(f"Completed test video creation: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error in test video generation: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def create_final_video(enhanced_segments, audio_path, output_path, temp_dir, limit_to_one_minute=False):
    """Create the final video with synchronized audio."""
    try:
        logger.info("Starting final video creation")
        logger.info(f"Number of segments to process: {len(enhanced_segments)}")
        
        # Calculate target duration based on segments and time limit
        if limit_to_one_minute:
            target_duration = min(max(segment["end"] for segment in enhanced_segments), TIME_LIMIT_SECONDS)
            logger.info(f"Using time-limited duration: {target_duration:.2f} seconds")
        else:
            target_duration = max(segment["end"] for segment in enhanced_segments)
            logger.info(f"Using full duration: {target_duration:.2f} seconds")
        
        # First, create video segments from PNG files
        valid_segments = []
        for i, segment in enumerate(enhanced_segments, 1):
            # Skip segments that start after the time limit if limit_to_one_minute is True
            if limit_to_one_minute and segment["start"] >= TIME_LIMIT_SECONDS:
                logger.info(f"Skipping segment {i} as it starts after time limit")
                continue
                
            # Adjust end time if it exceeds the limit and limit_to_one_minute is True
            end_time = min(segment["end"], TIME_LIMIT_SECONDS) if limit_to_one_minute else segment["end"]
            segment_duration = end_time - segment["start"]
            
            # Skip segments with zero duration
            if segment_duration <= 0:
                logger.info(f"Skipping segment {i} as it has zero duration")
                continue
            
            png_path = os.path.join(temp_dir, f"segment_{i}.png")
            segment_path = os.path.join(temp_dir, f"segment_{i}.mp4")
            
            # Create video segment from PNG
            segment_cmd = [
                "ffmpeg",
                "-y",
                "-loop", "1",
                "-i", png_path,
                "-t", str(segment_duration),
                "-vf", "scale=1920:1080",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "stillimage",
                "-pix_fmt", "yuv420p",
                "-r", "24",
                segment_path
            ]
            logger.info(f"Creating video segment {i} with command: {' '.join(segment_cmd)}")
            result = subprocess.run(segment_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise ValueError(f"Error creating video segment {i}: {result.stderr}")
            
            # Verify the segment was created
            if os.path.exists(segment_path):
                valid_segments.append((i, segment_path, segment_duration))
                logger.info(f"Created segment {i} with duration {segment_duration:.2f}s")
            else:
                logger.error(f"Failed to create segment {i}")
        
        if not valid_segments:
            raise ValueError("No valid segments were created")
        
        # Create concat file for ffmpeg
        concat_file = os.path.join(temp_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for i, segment_path, duration in valid_segments:
                # Write segment to concat file with absolute path
                abs_segment_path = os.path.abspath(segment_path)
                f.write(f"file '{abs_segment_path}'\n")
        
        # First, concatenate all segments into a single video
        concat_output = os.path.join(temp_dir, "concat_output.mp4")
        concat_cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            concat_output
        ]
        logger.info(f"Concatenating segments with command: {' '.join(concat_cmd)}")
        result = subprocess.run(concat_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"Error concatenating segments: {result.stderr}")
        
        # Now create the looped video with the target duration
        looped_output = os.path.join(temp_dir, "looped_video.mp4")
        loop_cmd = [
            "ffmpeg",
            "-y",
            "-stream_loop", "-1",
            "-i", concat_output,
            "-t", str(target_duration),
            "-c", "copy",
            looped_output
        ]
        logger.info(f"Creating looped video with command: {' '.join(loop_cmd)}")
        result = subprocess.run(loop_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"Error creating looped video: {result.stderr}")
        
        # Extract the limited audio if needed
        temp_audio = os.path.join(temp_dir, "temp_audio.wav")
        audio_cmd = [
            "ffmpeg",
            "-y",
            "-i", audio_path,
            "-t", str(target_duration),
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            temp_audio
        ]
        logger.info(f"Extracting audio with command: {' '.join(audio_cmd)}")
        result = subprocess.run(audio_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"Error extracting audio: {result.stderr}")
        
        # Finally, combine the looped video with the audio
        temp_output = os.path.join(temp_dir, "final_output.mp4")
        final_cmd = [
            "ffmpeg",
            "-y",
            "-i", looped_output,
            "-i", temp_audio,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "44100",
            "-ac", "2",
            "-t", str(target_duration),
            temp_output
        ]
        logger.info(f"Creating final video with command: {' '.join(final_cmd)}")
        result = subprocess.run(final_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"Error creating final video: {result.stderr}")
        
        # Verify the output file
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "stream=codec_name,codec_type,duration,width,height,r_frame_rate",
            "-of", "json",
            temp_output
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"Error probing output file: {result.stderr}")
        
        probe_data = json.loads(result.stdout)
        streams = probe_data["streams"]
        
        # Verify video stream
        video_stream = next(s for s in streams if s["codec_type"] == "video")
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        if width != 1920 or height != 1080:
            raise ValueError(f"Output video resolution is {width}x{height}, expected 1920x1080")
        
        # Verify frame rate
        fps = eval(video_stream["r_frame_rate"])
        if abs(fps - 24) > 0.1:
            raise ValueError(f"Output video frame rate is {fps}, expected 24")
        
        # Verify durations
        video_duration = float(video_stream["duration"])
        audio_stream = next(s for s in streams if s["codec_type"] == "audio")
        audio_duration = float(audio_stream["duration"])
        
        # Allow for a larger tolerance (0.5 seconds) in duration matching
        duration_tolerance = 0.5
        if abs(video_duration - target_duration) > duration_tolerance:
            raise ValueError(f"Video duration {video_duration:.2f}s does not match target duration {target_duration:.2f}s (tolerance: {duration_tolerance}s)")
        if abs(audio_duration - target_duration) > duration_tolerance:
            logger.warning(f"Audio duration {audio_duration:.2f}s differs from target duration {target_duration:.2f}s by {abs(audio_duration - target_duration):.2f}s")
            # Don't raise an error, just log a warning
        
        # Move the temporary output to the final location
        shutil.move(temp_output, output_path)
        
        # Clean up temporary files
        temp_files = [concat_file, concat_output, looped_output, temp_audio]
        for i, _, _ in valid_segments:
            temp_files.append(os.path.join(temp_dir, f"segment_{i}.mp4"))
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating final video: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Clean up any temporary files
        temp_files = [
            os.path.join(temp_dir, "concat.txt"),
            os.path.join(temp_dir, "concat_output.mp4"),
            os.path.join(temp_dir, "looped_video.mp4"),
            os.path.join(temp_dir, "temp_audio.wav"),
            os.path.join(temp_dir, "final_output.mp4")
        ]
        for i in range(len(enhanced_segments)):
            temp_files.append(os.path.join(temp_dir, f"segment_{i+1}.mp4"))
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.error(f"Error removing temporary file {temp_file}: {e}")
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
            # Use moviepy to get audio properties
            audio = moviepy_editor.AudioFileClip(audio_path)
            duration = audio.duration
            
            # Log audio properties
            logger.info(f"MP3 file properties:")
            logger.info(f"- Duration: {duration:.2f} seconds")
            
            # Close the audio clip
            audio.close()
            
            return {
                'format': 'mp3',
                'duration': duration
            }
        
        else:
            raise ValueError(f"Unsupported audio format: {file_ext}. Please use WAV or MP3 files.")
            
    except wave.Error as e:
        raise ValueError(f"Invalid WAV file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error validating audio file: {str(e)}")

# Add to global configuration
SEGMENT_MIN_DURATION = 12    # Minimum segment duration
SEGMENT_MAX_DURATION = 35    # Maximum segment duration
SEGMENT_TARGET_DURATION = 25 # Ideal segment duration

def normalize_segments(segments, min_duration=12, max_duration=35, target_duration=25):
    """Normalize segment durations by combining or splitting segments."""
    if not segments:
        logger.warning("No segments to normalize")
        return []

    # Log initial state
    logger.info("\n=== SEGMENT NORMALIZATION START ===")
    logger.info(f"Initial segment count: {len(segments)}")
    logger.info("Initial segments:")
    for i, s in enumerate(segments, 1):
        # Calculate duration from start and end times if not present
        if 'duration' not in s and 'start' in s and 'end' in s:
            s['duration'] = s['end'] - s['start']
            logger.info(f"Calculated duration for segment {i}: {s['duration']:.2f}s (from start: {s['start']:.2f}s, end: {s['end']:.2f}s)")
        duration = float(s.get('duration', 0))
        text_preview = s.get('text', '')[:100] + '...' if len(s.get('text', '')) > 100 else s.get('text', '')
        logger.info(f"Segment {i}:")
        logger.info(f"  Duration: {duration:.2f}s")
        logger.info(f"  Text: {text_preview}")
        logger.info(f"  Start: {s.get('start', 0):.2f}s")
        logger.info(f"  End: {s.get('end', 0):.2f}s")
        logger.info("---")

    # Convert durations to float and filter out invalid segments
    valid_segments = []
    for i, s in enumerate(segments, 1):
        try:
            # Calculate duration if not present
            if 'duration' not in s and 'start' in s and 'end' in s:
                s['duration'] = s['end'] - s['start']
                logger.info(f"Calculated duration for segment {i} during validation: {s['duration']:.2f}s")
            duration = float(s.get('duration', 0))
            if duration > 0:
                s['duration'] = duration
                valid_segments.append(s)
                logger.info(f"Segment {i} validated successfully with duration {duration:.2f}s")
            else:
                logger.warning(f"Segment {i} rejected due to invalid duration: {duration:.2f}s")
        except (ValueError, TypeError) as e:
            logger.warning(f"Error processing segment {i}: {e}")
            continue

    if not valid_segments:
        logger.warning("No valid segment durations found")
        logger.warning("Original segments will be returned without normalization")
        return segments  # Return original segments if no valid durations

    # Log segment statistics
    logger.info("\n=== SEGMENT STATISTICS ===")
    durations = [s['duration'] for s in valid_segments]
    logger.info(f"Valid segments: {len(valid_segments)}")
    logger.info(f"Duration range: {min(durations):.2f}s to {max(durations):.2f}s")
    logger.info(f"Average duration: {sum(durations) / len(durations):.2f}s")
    logger.info(f"Total duration: {sum(durations):.2f}s")

    # If total duration is very short, return single combined segment
    total_duration = sum(durations)
    if total_duration <= min_duration:
        logger.info(f"\nTotal duration ({total_duration:.2f}s) is less than minimum ({min_duration}s)")
        logger.info("Returning single combined segment")
        combined_text = " ".join(s.get('text', '') for s in valid_segments)
        combined = valid_segments[0].copy()
        combined.update({
            'text': combined_text,
            'duration': total_duration,
            'start': valid_segments[0].get('start', 0),
            'end': valid_segments[-1].get('end', total_duration)
        })
        logger.info("\n=== FINAL COMBINED SEGMENT ===")
        logger.info(f"Duration: {combined['duration']:.2f}s")
        logger.info(f"Text: {combined['text'][:200]}...")
        logger.info(f"Start: {combined['start']:.2f}s")
        logger.info(f"End: {combined['end']:.2f}s")
        return [combined]

    # First pass: merge very short segments and highly coherent segments
    normalized = []
    current = None

    for i, segment in enumerate(valid_segments, 1):
        if not current:
            current = segment.copy()
            logger.info(f"\nStarting new segment group with segment {i}")
            continue

        current_duration = current['duration']
        next_duration = segment['duration']
        merged_duration = current_duration + next_duration
        coherence_score = calculate_text_coherence(current['text'], segment['text'])
        
        # Define merge conditions
        should_merge = (
            (current_duration < min_duration) or  # Current segment too short
            (next_duration < min_duration) or     # Next segment too short
            (merged_duration <= max_duration and coherence_score > 0.6) or  # High coherence and within max
            (merged_duration <= target_duration and coherence_score > 0.8)  # Very high coherence and within target
        )
        
        if should_merge:
            # Log merge decision
            logger.info(f"\nMerging segments {i-1} and {i}:")
            logger.info(f"Segment 1: {current['text'][:100]}...")
            logger.info(f"Segment 2: {segment['text'][:100]}...")
            logger.info(f"Coherence score: {coherence_score:.2f}")
            logger.info(f"Combined duration: {merged_duration:.2f}s")
            
            # Preserve all fields from both segments when merging
            merged = current.copy()
            merged['duration'] = merged_duration
            merged['text'] = f"{current['text'].rstrip()} {segment['text'].lstrip()}"
            merged['end'] = segment.get('end', current.get('end', 0) + next_duration)
            
            # Merge any other fields that exist in both segments
            for key in segment:
                if key not in ['duration', 'text', 'start', 'end']:
                    if key in current:
                        # If the field exists in both, combine them (assuming they're compatible types)
                        if isinstance(current[key], list) and isinstance(segment[key], list):
                            merged[key] = current[key] + segment[key]
                        elif isinstance(current[key], dict) and isinstance(segment[key], dict):
                            merged[key] = {**current[key], **segment[key]}
                        else:
                            # For other types, keep the most recent value
                            merged[key] = segment[key]
                    else:
                        # If the field only exists in the second segment, copy it
                        merged[key] = segment[key]
            current = merged
            logger.info(f"Successfully merged segments {i-1} and {i}")
        else:
            if current_duration >= min_duration:
                normalized.append(current)
                logger.info(f"Added segment {i-1} to normalized list (duration: {current_duration:.2f}s)")
            current = segment.copy()
            logger.info(f"Starting new segment group with segment {i}")
    
    # Don't forget the last segment
    if current and current['duration'] >= min_duration:
        normalized.append(current)
        logger.info(f"Added final segment to normalized list (duration: {current['duration']:.2f}s)")
    
    # Second pass: split long segments
    final_segments = []
    for i, segment in enumerate(normalized, 1):
        duration = segment['duration']
        
        if duration > max_duration:
            # Calculate number of parts needed
            num_parts = max(2, int(duration / target_duration + 0.5))
            part_duration = duration / num_parts
            text = segment['text']
            start_time = segment.get('start', 0)
            
            logger.info(f"\nSplitting long segment {i}:")
            logger.info(f"Original duration: {duration:.2f}s")
            logger.info(f"Number of parts: {num_parts}")
            logger.info(f"Part duration: {part_duration:.2f}s")
            
            # Split segment into parts
            for j in range(num_parts):
                ratio = (j + 1) / num_parts
                if j == num_parts - 1:
                    # Last part gets the rest
                    part_text = text
                    part_duration = duration - (part_duration * j)
                else:
                    # Find natural boundary for split
                    split_pos = find_sentence_boundary(text, ratio)
                    part_text = text[:split_pos].strip()
                    text = text[split_pos:].strip()
                    part_duration = duration / num_parts
                
                if part_text:  # Only add if there's actual text
                    # Create new segment with all fields from original
                    new_segment = segment.copy()
                    new_segment.update({
                        'duration': part_duration,
                        'text': part_text,
                        'start': start_time + (j * part_duration),
                        'end': start_time + ((j + 1) * part_duration)
                    })
                    final_segments.append(new_segment)
                    logger.info(f"\nPart {j+1}:")
                    logger.info(f"Duration: {part_duration:.2f}s")
                    logger.info(f"Text: {part_text[:100]}...")
                    logger.info(f"Start: {new_segment['start']:.2f}s")
                    logger.info(f"End: {new_segment['end']:.2f}s")
        else:
            final_segments.append(segment)
            logger.info(f"Added segment {i} to final list (duration: {duration:.2f}s)")
    
    # Final pass: clean up and validate
    final_segments.sort(key=lambda x: x.get('start', 0))
    
    # Log final statistics
    logger.info("\n=== FINAL NORMALIZATION RESULTS ===")
    logger.info(f"Final segment count: {len(final_segments)}")
    final_durations = [s['duration'] for s in final_segments]
    logger.info(f"Final duration range: {min(final_durations):.2f}s to {max(final_durations):.2f}s")
    logger.info(f"Final average duration: {sum(final_durations) / len(final_durations):.2f}s")
    logger.info(f"Final total duration: {sum(final_durations):.2f}s")
    
    logger.info("\nFinal segments:")
    for i, s in enumerate(final_segments, 1):
        logger.info(f"\nSegment {i}:")
        logger.info(f"Duration: {s['duration']:.2f}s")
        logger.info(f"Text: {s['text'][:100]}...")
        logger.info(f"Start: {s['start']:.2f}s")
        logger.info(f"End: {s['end']:.2f}s")
    
    logger.info("\n=== SEGMENT NORMALIZATION COMPLETE ===")
    return final_segments

def write_segments_to_srt(segments, output_path):
    """Write segments to SRT file with validation and safety measures."""
    try:
        # Validate segments before writing
        for i, segment in enumerate(segments, 1):
            required_fields = ['start', 'end', 'text']
            missing_fields = [field for field in required_fields if field not in segment]
            if missing_fields:
                raise ValueError(f"Segment {i} is missing required fields: {', '.join(missing_fields)}")
            
            # Validate timing
            if segment['start'] >= segment['end']:
                raise ValueError(f"Segment {i} has invalid timing: start ({segment['start']}) >= end ({segment['end']})")
            
            # Validate text
            if not segment['text'].strip():
                raise ValueError(f"Segment {i} has empty text")
        
        # Create backup of existing file if it exists
        if os.path.exists(output_path):
            backup_path = output_path + '.bak'
            shutil.copy2(output_path, backup_path)
            logger.info(f"Created backup of existing transcript: {backup_path}")
        
        # Write segments to file
        with open(output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, 1):
                # Format timestamps
                start = format_timestamp(segment["start"])
                end = format_timestamp(segment["end"])
                
                # Write segment
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{segment['text']}\n\n")
        
        logger.info(f"Successfully wrote {len(segments)} segments to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error writing segments to SRT file: {str(e)}")
        # Restore backup if it exists
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, output_path)
            logger.info(f"Restored backup file after error: {output_path}")
        return False

def transcribe_audio(audio_path, force_transcription=False, transcript_dir="transcripts", non_interactive=False, temp_dir="temp", limit_to_one_minute=False):
    """Transcribe audio file using OpenAI's Whisper model"""
    try:
        # Check if transcription already exists
        transcript_path = os.path.join(transcript_dir, os.path.splitext(os.path.basename(audio_path))[0] + ".srt")
        
        if os.path.exists(transcript_path) and not force_transcription:
            logger.info(f"Using existing transcript: {transcript_path}")
            # Extract and normalize segments from existing transcript
            segments = extract_segments(transcript_path)
            if segments:
                normalized_segments = normalize_segments(
                    segments,
                    min_duration=SEGMENT_MIN_DURATION,
                    max_duration=SEGMENT_MAX_DURATION,
                    target_duration=SEGMENT_TARGET_DURATION
                )
                # Write normalized segments back to file
                if write_segments_to_srt(normalized_segments, transcript_path):
                    return transcript_path, normalized_segments
                else:
                    logger.error("Failed to write normalized segments back to file")
                    return None, None
            return None, None
        
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
        
        # Initialize MoviePy
        if not init_moviepy():
            raise ImportError("Failed to initialize MoviePy")
        
        # Load audio file with moviepy to get duration
        try:
            audio = moviepy_editor.AudioFileClip(audio_path)
            original_duration = audio.duration
            logger.info(f"Original audio duration: {original_duration:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading audio file with MoviePy: {str(e)}")
            raise
        
        # Determine the duration to process
        if limit_to_one_minute:
            duration_to_process = min(original_duration, TIME_LIMIT_SECONDS)
            logger.info(f"Limiting audio to first {duration_to_process:.2f} seconds")
        else:
            duration_to_process = original_duration
            logger.info(f"Processing full audio duration: {duration_to_process:.2f} seconds")
        
        # Split audio into chunks for processing
        chunk_duration = 60  # 60 seconds per chunk
        chunks = []
        
        for start_time in range(0, int(duration_to_process), chunk_duration):
            end_time = min(start_time + chunk_duration, duration_to_process)
            chunk_path = os.path.join(temp_dir, f"chunk_{start_time}_{end_time}.wav")
            
            try:
                # Use ffmpeg to extract chunk
                cmd = [
                    'ffmpeg', '-y',
                    '-i', audio_path,
                    '-ss', str(start_time),
                    '-t', str(end_time - start_time),
                    '-acodec', 'pcm_s16le',
                    '-ar', '44100',
                    '-ac', '1',
                    chunk_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    chunks.append((chunk_path, start_time))
                    logger.info(f"Created chunk: {chunk_path} (start: {start_time}s, end: {end_time}s)")
                else:
                    logger.error(f"Error creating chunk {start_time}-{end_time}: {result.stderr}")
            except Exception as e:
                logger.error(f"Error processing chunk {start_time}-{end_time}: {str(e)}")
                continue
        
        # Close the main audio clip
        audio.close()
        
        # Process each chunk
        transcript_segments = []
        for i, (chunk_path, chunk_start) in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} (start: {chunk_start}s)")
            
            # Transcribe chunk using OpenAI's Whisper model
            with open(chunk_path, "rb") as audio_file:
                response = transcription_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
                
                # Add segments to transcript with adjusted timestamps
                for segment in response.segments:
                    # Adjust timestamps by adding chunk start time
                    adjusted_start = segment.start + chunk_start
                    adjusted_end = segment.end + chunk_start
                    
                    transcript_segments.append({
                        "start": adjusted_start,
                        "end": adjusted_end,
                        "text": segment.text
                    })
            
            # Clean up chunk file
            os.remove(chunk_path)
        
        # Sort segments by start time
        transcript_segments.sort(key=lambda x: x["start"])
        
        # Normalize segments
        normalized_segments = normalize_segments(
            transcript_segments,
            min_duration=SEGMENT_MIN_DURATION,
            max_duration=SEGMENT_MAX_DURATION,
            target_duration=SEGMENT_TARGET_DURATION
        )
        
        # Write SRT file using normalized segments
        if write_segments_to_srt(normalized_segments, transcript_path):
            logger.info(f"Transcription completed: {transcript_path}")
            logger.info(f"Total normalized segments: {len(normalized_segments)}")
            return transcript_path, normalized_segments
        else:
            logger.error("Failed to write normalized segments to file")
            return None, None
        
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None

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
                    # If text already exists, append to it with a space
                    if 'text' in current_segment:
                        current_segment['text'] += ' ' + line
                    else:
                        current_segment['text'] = line
        
        # Add the last segment if it exists
        if current_segment:
            segments.append(current_segment)
        
        # Log the number of segments found
        logger.info(f"Extracted {len(segments)} segments from transcript")
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

def enhance_segments(segments, limit_to_one_minute=False):
    """Enhance segments with AI descriptions and visual prompts"""
    try:
        enhanced_segments = []
        total_segments = len(segments)
        time_limit = TIME_LIMIT_SECONDS if limit_to_one_minute else float('inf')
        
        for i, segment in enumerate(segments, 1):
            # Check for cancellation
            if is_cancelling:
                logger.info("Enhancement cancelled by user")
                return None
            
            # Validate segment has required fields
            required_fields = ['start', 'end', 'text']
            missing_fields = [field for field in required_fields if field not in segment]
            if missing_fields:
                logger.warning(f"Skipping segment {i}/{total_segments} - missing required fields: {', '.join(missing_fields)}")
                continue
            
            # Get segment text and timestamps
            text = segment['text']
            start_time = segment['start']
            end_time = segment['end']
            
            # Skip segments that start after the time limit
            if start_time >= time_limit:
                logger.info(f"Skipping segment {i}/{total_segments} (start: {start_time:.2f}s) - exceeds time limit")
                continue
                
            # Adjust end time if it exceeds the time limit
            if end_time > time_limit:
                end_time = time_limit
                logger.info(f"Adjusting end time of segment {i}/{total_segments} to {time_limit:.2f}s")
            
            # Log progress
            logger.info(f"Enhancing segment {i}/{total_segments} (start: {start_time:.2f}s, end: {end_time:.2f}s)")
            
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
                    # First try direct JSON parsing
                    enhancement = json.loads(content)
                except json.JSONDecodeError:
                    # If that fails, try to extract JSON from markdown code blocks
                    try:
                        # Look for content between ```json and ``` or just between ``` and ```
                        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                        if json_match:
                            enhancement = json.loads(json_match.group(1))
                        else:
                            # If no code blocks found, try to find JSON between curly braces
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
                continue
        
        # Sort segments by start time to ensure they're sequential
        enhanced_segments.sort(key=lambda x: x['start'])
        
        # Log final segment count
        logger.info(f"Successfully enhanced {len(enhanced_segments)} segments")
        return enhanced_segments
        
    except Exception as e:
        logger.error(f"Error in enhance_segments: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []

def generate_visuals(enhanced_segments, output_dir, non_interactive=False):
    """Generate visuals for each enhanced segment"""
    logger.info("Generating visuals for segments")
    visuals = []
    
    for i, segment in enumerate(enhanced_segments, 1):
        logger.info(f"Generating visuals for segment {i}/{len(enhanced_segments)}")
        try:
            # Generate image for the segment
            image_path = os.path.join(output_dir, f"segment_{i}.png")
            success = generate_image_stability(
                prompt=segment['visual_prompt'],
                output_path=image_path,
                non_interactive=non_interactive
            )
            
            if success:
                visuals.append({
                    'text': segment['text'],
                    'image_path': image_path,
                    'start_time': segment['start'],
                    'end_time': segment['end']
                })
                logger.info(f"Generated visual for segment {i}")
            else:
                logger.error(f"Failed to generate visual for segment {i}")
                
        except Exception as e:
            logger.error(f"Error in generate_visuals: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error("Visual generation failed")
            return None
            
    logger.info(f"Generated {len(visuals)} visuals")
    return visuals

def process_audio_file(audio_path, output_path, limit_to_one_minute=False, non_interactive=False):
    """Process an audio file to create a video with AI-generated visuals."""
    try:
        # Create temp directory
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create transcripts directory
        transcript_dir = "transcripts"
        os.makedirs(transcript_dir, exist_ok=True)
        
        # Log whether we're using time limit
        if limit_to_one_minute:
            logger.info(f"Processing with time limit of {TIME_LIMIT_SECONDS} seconds")
        else:
            logger.info("Processing full audio file without time limit")
        
        # Step 1: Transcribe audio and get normalized segments
        logger.info("Starting audio transcription")
        transcript_path, normalized_segments = transcribe_audio(
            audio_path,
            force_transcription=False,
            transcript_dir=transcript_dir,
            non_interactive=non_interactive,
            temp_dir=temp_dir,
            limit_to_one_minute=limit_to_one_minute
        )
        
        if transcript_path is None or normalized_segments is None:
            logger.error("Transcription failed")
            return None
        
        logger.info(f"Transcription completed: {transcript_path}")
        logger.info(f"Normalized segments: {len(normalized_segments)}")
        
        # Step 2: Enhance segments with AI
        logger.info("Enhancing segments with AI")
        enhanced_segments = enhance_segments(normalized_segments, limit_to_one_minute)
        
        if not enhanced_segments:
            logger.error("Segment enhancement failed")
            return None
        
        logger.info(f"Enhanced {len(enhanced_segments)} segments")
        
        # Step 3: Generate visuals for segments
        logger.info("Generating visuals for segments")
        success = generate_visuals(
            enhanced_segments,
            temp_dir,
            non_interactive=non_interactive
        )
        
        if not success:
            logger.error("Visual generation failed")
            return None
        
        logger.info("Visual generation completed")
        
        # Step 4: Create final video
        logger.info("Creating final video")
        final_video_path = create_final_video(
            enhanced_segments,
            audio_path,
            output_path,
            temp_dir,
            limit_to_one_minute=limit_to_one_minute
        )
        
        if final_video_path is None:
            logger.error("Final video creation failed")
            return None
        
        logger.info(f"Final video created: {final_video_path}")
        return final_video_path
        
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def init_moviepy():
    """Initialize MoviePy and return whether it's available"""
    global moviepy_editor, MOVIEPY_AVAILABLE
    if not MOVIEPY_AVAILABLE:
        try:
            import moviepy.editor as moviepy_editor
            MOVIEPY_AVAILABLE = True
            logger.info("Successfully initialized MoviePy")
        except ImportError as e:
            logger.error(f"Failed to initialize MoviePy: {e}")
            MOVIEPY_AVAILABLE = False
        except Exception as e:
            logger.error(f"Unexpected error initializing MoviePy: {e}")
            MOVIEPY_AVAILABLE = False
    return MOVIEPY_AVAILABLE

if __name__ == "__main__":
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Convert podcast audio to video with AI-generated visuals")
        parser.add_argument("--input", help="Input audio file path")
        parser.add_argument("--output", help="Output video file path")
        parser.add_argument("--temp_dir", default="temp", help="Temporary directory for processing files")
        parser.add_argument("--transcript_dir", default="transcripts", help="Directory for storing transcripts")
        parser.add_argument("--transcribe_only", action="store_true", help="Only transcribe the audio, don't process further")
        parser.add_argument("--force_transcription", action="store_true", help="Force transcription even if transcript exists")
        parser.add_argument("--limit_to_one_minute", action="store_true", help="Limit processing to first minute of audio")
        parser.add_argument("--non_interactive", action="store_true", help="Run in non-interactive mode")
        parser.add_argument("--test_openai", action="store_true", help="Test OpenAI API connectivity")
        parser.add_argument("--test_stability", action="store_true", help="Test Stability API connectivity")
        args = parser.parse_args()
        
        # Setup signal handling for graceful termination
        setup_signal_handling()
        
        # Register cleanup function to run at exit
        atexit.register(cleanup)
        
        # Handle the test commands first
        if args.test_openai:
            success = test_openai_api()
            sys.exit(0 if success else 1)
            
        if args.test_stability:
            success = test_stability_api()
            sys.exit(0 if success else 1)
        
        # Ensure the required arguments are provided
        if not args.input:
            logger.error("Input audio file path is required")
            sys.exit(1)
        
        # Process the audio file
        process_audio_file(args.input, args.output, args.limit_to_one_minute, args.non_interactive)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
