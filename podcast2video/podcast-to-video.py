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
try:
    debug_point("Initializing Stability API client")
    stability_api_key = os.environ.get("STABILITY_API_KEY")
    if stability_api_key:
        try:
            # Check if stability_sdk is installed
            try:
                from stability_sdk import client as stability_client
                from stability_sdk.interfaces.gooseai.generation.generation_pb2 import ARTIFACT_IMAGE
                stability_api = stability_client.StabilityInference(
                    key=stability_api_key,
                    verbose=True
                )
                logger.info("Stability API initialized successfully")
                
                # Quick test of Stability API
                try:
                    debug_point("Testing Stability API connection")
                    # Make a minimal request to test connectivity
                    stability_api.generate(prompt="test", width=64, height=64, steps=1, samples=1)
                    logger.info("Stability API test successful")
                except Exception as e:
                    logger.warning(f"Stability API test failed: {e}")
            except ImportError:
                logger.error("Stability SDK not installed. Run: pip install stability-sdk")
                stability_api = None
        except Exception as e:
            logger.error(f"Failed to initialize Stability API: {e}", exc_info=True)
            stability_api = None
    else:
        stability_api = None
        logger.warning("No Stability API key found. Will attempt to use alternatives.")
except Exception as e:
    logger.error(f"Error in Stability API initialization block: {e}", exc_info=True)
    stability_api = None

# User interaction function
def get_user_confirmation(prompt, default=False, non_interactive=False):
    """Get confirmation from the user for various operations"""
    if non_interactive:
        logger.info(f"{prompt} (Using default: {'Yes' if default else 'No'} due to non-interactive mode)")
        return default
        
    response = input(f"{prompt} (y/n) [{'y' if default else 'n'}]: ").strip().lower()
    if not response:
        return default
    return response.startswith('y')

def generate_cache_path(audio_file_path, transcript_dir):
    """Generate a unique cache path for the transcript based on the audio file"""
    # Create hash of file path + modification time to uniquely identify the audio
    audio_stat = os.stat(audio_file_path)
    file_id = f"{audio_file_path}_{audio_stat.st_size}_{audio_stat.st_mtime}"
    file_hash = hashlib.md5(file_id.encode()).hexdigest()
    
    # Create the cache directory if it doesn't exist
    os.makedirs(transcript_dir, exist_ok=True)
    
    # Generate paths for different transcript formats
    base_path = os.path.join(transcript_dir, file_hash)
    json_path = f"{base_path}.json"
    srt_path = f"{base_path}.srt"
    vtt_path = f"{base_path}.vtt"
    
    return {
        "json": json_path,
        "srt": srt_path,
        "vtt": vtt_path
    }

def validate_audio_file(audio_file_path):
    """Validate the audio file format and return info about it"""
    debug_point(f"Validating audio file: {audio_file_path}")
    
    # Check if file exists
    if not os.path.exists(audio_file_path):
        logger.error(f"Audio file not found: {audio_file_path}")
        raise ValueError(f"Audio file not found: {audio_file_path}")
        
    file_ext = os.path.splitext(audio_file_path)[1].lower()
    
    if file_ext not in ['.mp3', '.wav']:
        logger.error(f"Unsupported audio format: {file_ext}")
        raise ValueError(f"Unsupported audio format: {file_ext}. Only MP3 and WAV formats are supported.")
    
    try:
        # Get audio file information
        audio_info = {}
        
        if file_ext == '.mp3':
            if not PYDUB_AVAILABLE:
                logger.error("PyDub not available for MP3 processing")
                raise ValueError("PyDub library is required to process MP3 files")
                
            debug_point("Processing MP3 file")
            # Use pydub for MP3 info
            audio = AudioSegment.from_mp3(audio_file_path)
            audio_info = {
                'format': 'mp3',
                'channels': audio.channels,
                'sample_width': audio.sample_width,
                'frame_rate': audio.frame_rate,
                'duration_seconds': len(audio) / 1000.0,
            }
        elif file_ext == '.wav':
            debug_point("Processing WAV file")
            # Use wave module for WAV info
            try:
                with wave.open(audio_file_path, 'rb') as wav_file:
                    audio_info = {
                        'format': 'wav',
                        'channels': wav_file.getnchannels(),
                        'sample_width': wav_file.getsampwidth(),
                        'frame_rate': wav_file.getframerate(),
                        'duration_seconds': wav_file.getnframes() / wav_file.getframerate(),
                    }
            except Exception as e:
                logger.error(f"Error reading WAV file with wave module: {e}")
                logger.info("Trying with pydub as fallback")
                
                if not PYDUB_AVAILABLE:
                    logger.error("PyDub not available for WAV fallback processing")
                    raise ValueError("Failed to read WAV file and PyDub fallback not available")
                
                # Fallback to pydub
                audio = AudioSegment.from_wav(audio_file_path)
                audio_info = {
                    'format': 'wav',
                    'channels': audio.channels,
                    'sample_width': audio.sample_width,
                    'frame_rate': audio.frame_rate,
                    'duration_seconds': len(audio) / 1000.0,
                }
        
        logger.info(f"Audio file: {os.path.basename(audio_file_path)}")
        logger.info(f"Format: {audio_info['format']}")
        logger.info(f"Channels: {audio_info['channels']}")
        logger.info(f"Sample rate: {audio_info['frame_rate']} Hz")
        logger.info(f"Duration: {audio_info['duration_seconds']:.2f} seconds")
        
        return audio_info
    
    except Exception as e:
        logger.error(f"Error analyzing audio file: {e}", exc_info=True)
        raise ValueError(f"Error analyzing audio file: {e}")

def normalize_audio_for_transcription(audio_file_path, temp_dir):
    """Normalize audio file to a consistent format for transcription if needed"""
    debug_point(f"Normalizing audio: {audio_file_path}")
    file_ext = os.path.splitext(audio_file_path)[1].lower()
    
    # If already MP3, return the original file path
    if file_ext == '.mp3':
        logger.info("Audio already in MP3 format, no conversion needed")
        return audio_file_path
    
    # For WAV files, convert to MP3 for more consistent transcription results
    if file_ext == '.wav':
        logger.info("Converting WAV to MP3 for optimal transcription...")
        mp3_path = os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(audio_file_path))[0]}.mp3")
        
        # Convert WAV to MP3 using pydub
        try:
            debug_point(f"Reading WAV: {audio_file_path}")
            audio = AudioSegment.from_wav(audio_file_path)
            debug_point(f"Exporting MP3: {mp3_path}")
            # Use lower bitrate to reduce file size (128k instead of 192k)
            audio.export(mp3_path, format="mp3", bitrate="128k")
            logger.info(f"Converted to MP3: {mp3_path}")
            return mp3_path
        except Exception as e:
            logger.error(f"Could not convert WAV to MP3: {e}", exc_info=True)
            logger.warning("Using original WAV file for transcription")
            return audio_file_path
    
    # Fallback for other formats (shouldn't reach here due to validation)
    return audio_file_path

def split_audio_file(audio_file_path, temp_dir, max_size_bytes=25*1024*1024, chunk_duration_ms=600000):
    """Split large audio file into smaller chunks that fit within API size limits
    
    Args:
        audio_file_path: Path to the audio file to split
        temp_dir: Directory to save the chunks
        max_size_bytes: Maximum size in bytes for each chunk (default: 25MB)
        chunk_duration_ms: Maximum duration in ms for each chunk (default: 10 minutes)
        
    Returns:
        List of paths to the audio chunks
    """
    debug_point(f"Splitting large audio file: {audio_file_path}")
    
    # Check file size
    file_size = os.path.getsize(audio_file_path)
    if file_size <= max_size_bytes:
        logger.info(f"Audio file is already under size limit ({file_size} bytes), no splitting needed")
        return [audio_file_path]
    
    logger.info(f"Audio file exceeds size limit ({file_size} bytes), splitting into chunks")
    
    # Load the audio file
    file_ext = os.path.splitext(audio_file_path)[1].lower()
    try:
        if file_ext == '.mp3':
            audio = AudioSegment.from_mp3(audio_file_path)
        elif file_ext == '.wav':
            audio = AudioSegment.from_wav(audio_file_path)
        else:
            logger.error(f"Unsupported format for splitting: {file_ext}")
            return [audio_file_path]
    except Exception as e:
        logger.error(f"Error loading audio for splitting: {e}")
        return [audio_file_path]
    
    # Get length of audio in milliseconds
    audio_length_ms = len(audio)
    
    # Calculate number of chunks
    num_chunks = max(1, int(audio_length_ms / chunk_duration_ms) + 1)
    logger.info(f"Splitting {audio_length_ms/1000:.2f} seconds audio into {num_chunks} chunks")
    
    # Split audio into chunks
    chunk_paths = []
    base_filename = os.path.splitext(os.path.basename(audio_file_path))[0]
    
    for i in range(num_chunks):
        start_ms = i * chunk_duration_ms
        end_ms = min((i + 1) * chunk_duration_ms, audio_length_ms)
        
        if start_ms >= audio_length_ms:
            break
            
        chunk = audio[start_ms:end_ms]
        chunk_path = os.path.join(temp_dir, f"{base_filename}_chunk_{i+1}.mp3")
        
        # Export as MP3 with lower bitrate to ensure small file size
        chunk.export(chunk_path, format="mp3", bitrate="96k")
        chunk_size = os.path.getsize(chunk_path)
        
        logger.info(f"Created chunk {i+1}/{num_chunks}: {chunk_path} ({chunk_size} bytes, {(end_ms-start_ms)/1000:.2f} seconds)")
        chunk_paths.append(chunk_path)
    
    return chunk_paths

def transcribe_audio(audio_file_path, force=False, transcript_dir="transcripts", non_interactive=False, temp_dir="temp"):
    """Transcribe the podcast audio using Whisper model with appropriate API"""
    global current_operation
    current_operation = "audio transcription"
    
    # Track timing statistics
    chunk_times = []
    start_time = time.time()
    
    # Get a normalized version of the audio file for transcription
    normalized_audio_path = normalize_audio_for_transcription(audio_file_path, temp_dir)
    
    # Use the original path for caching and output files
    cache_paths = generate_cache_path(audio_file_path, transcript_dir)
    audio_basename = os.path.basename(audio_file_path)
    
    # Also check for SRT next to the audio file (older convention)
    direct_srt_path = Path(audio_file_path).with_suffix('.srt')
    direct_vtt_path = Path(audio_file_path).with_suffix('.vtt')
    
    if not force:
        # Check cache first
        if os.path.exists(cache_paths["json"]):
            print(f"Found cached transcript for {audio_basename}")
            with open(cache_paths["json"], "r") as f:
                transcript = json.load(f)
            
            # Copy SRT and VTT to audio file location for compatibility
            if os.path.exists(cache_paths["srt"]) and not os.path.exists(direct_srt_path):
                print(f"Copying cached SRT file to {direct_srt_path}")
                with open(cache_paths["srt"], "r") as src, open(direct_srt_path, "w") as dst:
                    dst.write(src.read())
            
            if os.path.exists(cache_paths["vtt"]) and not os.path.exists(direct_vtt_path):
                print(f"Copying cached VTT file to {direct_vtt_path}")
                with open(cache_paths["vtt"], "r") as src, open(direct_vtt_path, "w") as dst:
                    dst.write(src.read())
                    
            return transcript
    
    # If we get here, we need to transcribe
    estimated_cost = estimate_transcription_cost(audio_file_path)
    print(f"Estimated transcription cost: ${estimated_cost:.2f}")
    
    if estimated_cost > 1.0 and not force:
        # Ask for confirmation for expensive transcriptions
        if not get_user_confirmation(
            f"Transcribing this file may cost approximately ${estimated_cost:.2f}. Continue?",
            default=False,
            non_interactive=non_interactive
        ):
            print("Transcription cancelled by user.")
            return None
    
    print(f"Transcribing audio: {audio_basename}...")
    
    # Allow cancellation before starting API call
    check_cancel()
    
    # Check if we need to split the audio file due to size
    audio_size = os.path.getsize(normalized_audio_path)
    max_api_size = 25 * 1024 * 1024  # 25MB to be safe (API limit is ~26MB)
    
    if audio_size > max_api_size:
        logger.info(f"Audio file size ({audio_size} bytes) exceeds API limit, splitting into chunks")
        audio_chunks = split_audio_file(normalized_audio_path, temp_dir)
        logger.info(f"Split audio into {len(audio_chunks)} chunks")
    else:
        audio_chunks = [normalized_audio_path]
        
    # Process each audio chunk
    all_transcripts = []
    combined_transcript = {"segments": []}
    time_offset = 0
    
    for chunk_idx, chunk_path in enumerate(audio_chunks):
        chunk_start = time.time()
        logger.info(f"Processing audio chunk {chunk_idx+1}/{len(audio_chunks)}: {chunk_path}")
        
        # If we have multiple chunks, recalculate duration of this chunk for accurate time offsets
        if len(audio_chunks) > 1:
            try:
                # Get duration of this chunk
                file_ext = os.path.splitext(chunk_path)[1].lower()
                if file_ext == '.mp3':
                    chunk_audio = AudioSegment.from_mp3(chunk_path)
                elif file_ext == '.wav':
                    chunk_audio = AudioSegment.from_wav(chunk_path)
                chunk_duration_sec = len(chunk_audio) / 1000.0
                logger.info(f"Chunk {chunk_idx+1} duration: {chunk_duration_sec:.2f} seconds, time offset: {time_offset:.2f} seconds")
            except Exception as e:
                logger.error(f"Error getting chunk duration: {e}")
                chunk_duration_sec = 0
        
        # Transcribe this chunk
        with open(chunk_path, "rb") as audio_file:
            try:
                if api_type == "deepseek":
                    # Use DeepSeek for audio transcription
                    try:
                        # Confirm with user before using DeepSeek for transcription
                        if not force and not non_interactive and chunk_idx == 0:
                            if not get_user_confirmation(
                                "OpenAI API key not found. Use DeepSeek for transcription?",
                                default=True
                            ):
                                print("Transcription cancelled by user.")
                                return None
                                
                        # Try different potential audio models
                        audio_models = ["whisper-1"]  # Simplified to only use models that are likely to work
                        transcription_success = False
                        
                        for audio_model in audio_models:
                            try:
                                logger.info(f"Attempting transcription with model: {audio_model}")
                                chunk_transcript = ai_client.audio.transcriptions.create(
                                    model=audio_model,
                                    file=audio_file,
                                    response_format="verbose_json"
                                )
                                logger.info(f"DeepSeek transcription completed successfully with model {audio_model}")
                                transcription_success = True
                                break
                            except Exception as audio_error:
                                logger.warning(f"Transcription with model {audio_model} failed: {audio_error}")
                                continue
                        
                        if not transcription_success:
                            logger.error("All DeepSeek audio models failed")
                            # Instead of raising an exception, fall back to OpenAI directly
                            if openai_api_key:
                                logger.info("Falling back to OpenAI for transcription")
                                temp_client = OpenAI(api_key=openai_api_key)
                                chunk_transcript = temp_client.audio.transcriptions.create(
                                    model="whisper-1",
                                    file=audio_file,
                                    response_format="verbose_json"
                                )
                                transcription_success = True
                            else:
                                logger.error("All DeepSeek audio models failed and no OpenAI fallback available")
                                raise Exception("DeepSeek transcription failed with all models")
                    except Exception as e:
                        logger.error(f"DeepSeek transcription failed: {e}")
                        if openai_api_key:
                            # Fallback to OpenAI if DeepSeek fails and OpenAI key is available
                            if not get_user_confirmation(
                                "DeepSeek transcription failed. Fall back to OpenAI Whisper (may incur additional costs)?",
                                default=True,
                                non_interactive=non_interactive
                            ):
                                print("Transcription cancelled by user.")
                                return None
                            
                            logger.info("Falling back to OpenAI for transcription")
                            temp_client = OpenAI(api_key=openai_api_key)
                            chunk_transcript = temp_client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file,
                                response_format="verbose_json"
                            )
                        else:
                            logger.error("DeepSeek transcription failed and no OpenAI fallback available")
                            print("Transcription failed. No API keys available.")
                            return None
                else:
                    # Use OpenAI for transcription (default)
                    chunk_transcript = ai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json"
                    )
                
                all_transcripts.append(chunk_transcript)
                
                # Adjust timestamps for this chunk and add to combined transcript
                if len(audio_chunks) > 1:
                    for segment in chunk_transcript.segments:
                        # Convert segment to a dictionary for JSON serialization
                        segment_dict = {
                            "start": segment.start + time_offset if hasattr(segment, 'start') else 0 + time_offset,
                            "end": segment.end + time_offset if hasattr(segment, 'end') else 0 + time_offset,
                            "text": segment.text if hasattr(segment, 'text') else ""
                        }
                        combined_transcript["segments"].append(segment_dict)
                    
                    # Update time offset for the next chunk
                    time_offset += chunk_duration_sec
                else:
                    # If only one chunk, convert the transcript to a serializable format
                    combined_transcript = {"segments": []}
                    if hasattr(chunk_transcript, 'segments'):
                        for segment in chunk_transcript.segments:
                            segment_dict = {
                                "start": segment.start if hasattr(segment, 'start') else 0,
                                "end": segment.end if hasattr(segment, 'end') else 0,
                                "text": segment.text if hasattr(segment, 'text') else ""
                            }
                            combined_transcript["segments"].append(segment_dict)
                    else:
                        logger.warning("Transcript does not have segments attribute")
                
                # Calculate and display timing statistics for this chunk
                chunk_duration = time.time() - chunk_start
                chunk_times.append(chunk_duration)
                
                # Calculate average time and remaining time
                avg_time = sum(chunk_times) / len(chunk_times)
                remaining_chunks = len(audio_chunks) - (chunk_idx + 1)
                estimated_remaining_time = avg_time * remaining_chunks
                
                # Calculate progress percentage
                progress = (chunk_idx + 1) / len(audio_chunks) * 100
                
                # Display progress and timing information
                print(f"\nTranscription Progress: {progress:.1f}% ({chunk_idx+1}/{len(audio_chunks)} chunks)")
                print(f"Current chunk duration: {chunk_duration:.1f}s")
                print(f"Average chunk duration: {avg_time:.1f}s")
                print(f"Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
                print(f"Total elapsed time: {(time.time() - start_time)/60:.1f} minutes")
                print("-" * 50)
                
                # Check for cancellation after API call
                check_cancel()
                
            except KeyboardInterrupt:
                print("Transcription cancelled during API call")
                return None
            except Exception as e:
                print(f"Error during transcription of chunk {chunk_idx+1}: {e}")
                # If this is the only chunk, return None; otherwise continue with remaining chunks
                if len(audio_chunks) == 1:
                    return None
    
    # Use the combined transcript as our final result
    transcript = combined_transcript
    
    # Generate subtitle files
    srt_path, vtt_path = generate_subtitles(transcript, normalized_audio_path)
    
    # Save to cache
    try:
        check_cancel()
        
        with open(cache_paths["json"], "w") as f:
            # Ensure we're saving a JSON-serializable object
            json.dump(transcript, f)
        
        with open(cache_paths["srt"], "w") as f:
            with open(srt_path, "r") as src:
                f.write(src.read())
        
        with open(cache_paths["vtt"], "w") as f:
            with open(vtt_path, "r") as src:
                f.write(src.read())
        
        print(f"Transcript saved to cache at {cache_paths['json']}")
    except KeyboardInterrupt:
        print("Caching transcript cancelled")
    
    return transcript

def estimate_transcription_cost(audio_file_path):
    """Estimate the cost of transcription based on file size and duration"""
    try:
        # Get the duration of the audio file
        file_ext = os.path.splitext(audio_file_path)[1].lower()
        
        if file_ext == '.mp3':
            audio = AudioSegment.from_mp3(audio_file_path)
            duration_seconds = len(audio) / 1000.0
        elif file_ext == '.wav':
            with wave.open(audio_file_path, 'rb') as wav_file:
                duration_seconds = wav_file.getnframes() / wav_file.getframerate()
        else:
            # Fallback to moviepy if format not directly supported
            audio = AudioFileClip(audio_file_path)
            duration_seconds = audio.duration
            audio.close()
        
        duration_minutes = duration_seconds / 60
        
        # Estimate cost based on OpenAI's pricing
        # Currently around $0.006 per minute for Whisper
        estimated_cost = duration_minutes * 0.006
        
        return estimated_cost
    except Exception as e:
        print(f"Error estimating transcription cost: {e}")
        return 0.0

def generate_subtitles(transcript, audio_file_path):
    """Generate subtitle file in SRT format from transcript"""
    print("Generating subtitle file...")
    
    srt_path = Path(audio_file_path).with_suffix('.srt')
    vtt_path = Path(audio_file_path).with_suffix('.vtt')
    
    # Handle different types of transcript objects
    try:
        # Try to access segments as an attribute (TranscriptionVerbose object)
        if hasattr(transcript, 'segments'):
            transcript_segments = transcript.segments
        # Try to access as a dictionary key
        elif isinstance(transcript, dict) and "segments" in transcript:
            transcript_segments = transcript["segments"]
        else:
            # If neither works, try to convert to dict first
            try:
                transcript_dict = transcript if isinstance(transcript, dict) else transcript.__dict__
                transcript_segments = transcript_dict.get("segments", [])
            except:
                logger.error("Failed to extract segments from transcript in generate_subtitles")
                return "", ""
                
        logger.info(f"Found {len(transcript_segments)} segments in transcript")
    except Exception as e:
        logger.error(f"Error extracting segments from transcript: {e}")
        return "", ""
    
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        for i, segment in enumerate(transcript_segments):
            # Format timestamp for SRT (HH:MM:SS,mmm)
            try:
                # Get start time, trying different ways
                if hasattr(segment, 'start'):
                    start = segment.start
                elif isinstance(segment, dict) and "start" in segment:
                    start = segment["start"]
                else:
                    start = 0
                    
                # Get end time, trying different ways
                if hasattr(segment, 'end'):
                    end = segment.end
                elif isinstance(segment, dict) and "end" in segment:
                    end = segment["end"]
                else:
                    end = 0
                    
                # Get text, trying different ways
                if hasattr(segment, 'text'):
                    text = segment.text
                elif isinstance(segment, dict) and "text" in segment:
                    text = segment["text"]
                else:
                    text = ""
                
                start_time = format_timestamp(start)
                end_time = format_timestamp(end)
                
                # Write SRT entry
                srt_file.write(f"{i+1}\n")
                srt_file.write(f"{start_time} --> {end_time}\n")
                srt_file.write(f"{text.strip()}\n\n")
            except Exception as e:
                logger.error(f"Error processing segment {i}: {e}")
                continue
    
    # Also create WebVTT format
    with open(vtt_path, 'w', encoding='utf-8') as vtt_file:
        vtt_file.write("WEBVTT\n\n")
        for i, segment in enumerate(transcript_segments):
            try:
                # Get start time, trying different ways
                if hasattr(segment, 'start'):
                    start = segment.start
                elif isinstance(segment, dict) and "start" in segment:
                    start = segment["start"]
                else:
                    start = 0
                    
                # Get end time, trying different ways
                if hasattr(segment, 'end'):
                    end = segment.end
                elif isinstance(segment, dict) and "end" in segment:
                    end = segment["end"]
                else:
                    end = 0
                    
                # Get text, trying different ways
                if hasattr(segment, 'text'):
                    text = segment.text
                elif isinstance(segment, dict) and "text" in segment:
                    text = segment["text"]
                else:
                    text = ""
                
                # Format timestamp for VTT (HH:MM:SS.mmm)
                start_time = format_timestamp(start, vtt=True)
                end_time = format_timestamp(end, vtt=True)
                
                # Write VTT entry
                vtt_file.write(f"{start_time} --> {end_time}\n")
                vtt_file.write(f"{text.strip()}\n\n")
            except Exception as e:
                logger.error(f"Error processing segment for VTT {i}: {e}")
                continue
    
    print(f"Subtitle files generated: {srt_path} and {vtt_path}")
    return str(srt_path), str(vtt_path)

def format_timestamp(seconds, vtt=False):
    """Format seconds to SRT/VTT timestamp format"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    
    if vtt:
        # VTT format: HH:MM:SS.mmm
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', '.')
    else:
        # SRT format: HH:MM:SS,mmm
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

def extract_segments(transcript, min_segment_duration=10, max_segment_duration=30):
    """Extract meaningful segments from the transcript with timestamps"""
    print("Extracting meaningful segments...")
    
    segments = []
    current_segment = {"text": "", "start": None, "end": None}
    
    # Handle different types of transcript objects
    try:
        # Try to access segments as an attribute (TranscriptionVerbose object)
        if hasattr(transcript, 'segments'):
            transcript_segments = transcript.segments
        # Try to access as a dictionary key
        elif isinstance(transcript, dict) and "segments" in transcript:
            transcript_segments = transcript["segments"]
        else:
            # If neither works, try to convert to dict first
            try:
                transcript_dict = transcript if isinstance(transcript, dict) else transcript.__dict__
                transcript_segments = transcript_dict.get("segments", [])
            except:
                logger.error("Failed to extract segments from transcript in extract_segments")
                return []
                
        logger.info(f"Processing {len(transcript_segments)} segments for extraction")
    except Exception as e:
        logger.error(f"Error accessing segments in extract_segments: {e}")
        return []
    
    for segment in transcript_segments:
        # Get properties safely depending on the type of segment
        if hasattr(segment, 'start'):
            start_time = segment.start
        elif isinstance(segment, dict) and "start" in segment:
            start_time = segment["start"]
        else:
            logger.warning("Segment missing start time, skipping")
            continue
            
        if hasattr(segment, 'end'):
            end_time = segment.end
        elif isinstance(segment, dict) and "end" in segment:
            end_time = segment["end"]
        else:
            logger.warning("Segment missing end time, skipping")
            continue
            
        if hasattr(segment, 'text'):
            text = segment.text
        elif isinstance(segment, dict) and "text" in segment:
            text = segment["text"]
        else:
            logger.warning("Segment missing text, skipping")
            continue
        
        if current_segment["start"] is None:
            current_segment["start"] = start_time
        
        current_segment["text"] += " " + text
        current_segment["end"] = end_time
        
        segment_duration = end_time - current_segment["start"]
        
        # If segment is long enough and contains a complete thought (ends with punctuation)
        if segment_duration >= min_segment_duration and re.search(r'[.!?]$', current_segment["text"].strip()):
            segments.append(current_segment)
            current_segment = {"text": "", "start": None, "end": None}
        # If segment exceeds max duration, split it anyway
        elif segment_duration >= max_segment_duration:
            segments.append(current_segment)
            current_segment = {"text": "", "start": None, "end": None}
    
    # Add the last segment if it's not empty
    if current_segment["start"] is not None:
        segments.append(current_segment)
    
    print(f"Extracted {len(segments)} meaningful segments")
    return segments

def extract_facts(text):
    """Extract factual elements from the text"""
    try:
        extraction_response = ai_client.chat.completions.create(
            model=chat_model,
            messages=[
                {
                    "role": "system", 
                    "content": """You are a fact extraction specialist for educational content. 
                               Carefully analyze the podcast segment and extract ONLY factual elements
                               that are explicitly mentioned in the transcript: names of people, places, 
                               historical events, inventions, time periods, concepts, etc.
                               
                               Only extract elements that are SPECIFICALLY MENTIONED in the text.
                               DO NOT add any information that is not explicitly stated.
                               Be extremely precise with names, dates, and terminology.
                               
                               Format your response as JSON with the following structure:
                               {
                                   "extracted_elements": [
                                       {
                                           "type": "person/place/event/invention/concept/time_period",
                                           "name": "exact name as mentioned in transcript",
                                           "context": "brief context from the transcript"
                                       }
                                   ]
                               }"""
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        
        extracted_elements = json.loads(extraction_response.choices[0].message.content)
        return extracted_elements.get("extracted_elements", [])
    except Exception as e:
        logger.error(f"Error extracting facts: {e}")
        return []

def extract_key_elements(text):
    """Extract key elements from the text"""
    try:
        extraction_response = ai_client.chat.completions.create(
            model=chat_model,
            messages=[
                {
                    "role": "system", 
                    "content": """You are a key element extraction specialist for educational content.
                               Analyze the podcast segment and identify the most important elements
                               that should be highlighted in the video.
                               
                               Focus on:
                               1. Main topics or themes
                               2. Key concepts or ideas
                               3. Important names or terms
                               4. Significant points or conclusions
                               
                               Format your response as JSON with the following structure:
                               {
                                   "key_elements": [
                                       {
                                           "name": "name or term",
                                           "type": "topic/concept/name/point",
                                           "importance": "high/medium/low",
                                           "context": "brief context from the transcript"
                                       }
                                   ]
                               }"""
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        
        extracted_elements = json.loads(extraction_response.choices[0].message.content)
        return extracted_elements.get("key_elements", [])
    except Exception as e:
        logger.error(f"Error extracting key elements: {e}")
        return []

def research_element(element_name):
    """Research an element and return corrected text"""
    try:
        # Check cache first
        cached_research, cached_image_prompt = get_cached_research(element_name)
        if cached_research:
            logger.info(f"Using cached research for element: {element_name}")
            return cached_research
        
        # Log start of research
        logger.info(f"Starting research for element: {element_name}")
        
        # Create a combined prompt for research and spell checking
        combined_prompt = f"Research: {element_name}"
        
        # Make a single API call for both research and spell checking
        response = ai_client.chat.completions.create(
            model=chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "Research and provide key facts. Fix spelling/grammar. Be concise."
                },
                {"role": "user", "content": combined_prompt}
            ],
            max_tokens=100,  # Reduced from 150
            temperature=0.3  # Added for faster, more focused responses
        )
        
        result = response.choices[0].message.content
        
        # Cache the result
        cache_research_results(element_name, result, cached_image_prompt)
        
        # Log completion
        logger.info(f"Completed research for element: {element_name}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error researching element {element_name}: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Error researching {element_name}: {str(e)}"

def generate_image_prompt(element_name, research):
    """Generate a prompt for image generation"""
    try:
        # Log start of image prompt generation
        logger.info(f"Starting image prompt generation for element: {element_name}")
        
        # Create a concise prompt
        prompt = f"Element: {element_name}\nResearch: {research}"
        
        # Generate image prompt
        response = ai_client.chat.completions.create(
            model=chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "Create a clear, concise image generation prompt."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=100  # Reduced from 150
        )
        
        # Log completion
        logger.info(f"Completed image prompt generation for element: {element_name}")
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating image prompt for {element_name}: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Error generating prompt for {element_name}"

def generate_summary(text, facts, key_elements):
    """Generate a summary of the segment"""
    try:
        # Log start of summary generation
        logger.info("Starting summary generation")
        
        # Create a concise prompt
        prompt = f"Text: {text}\nFacts: {facts}\nElements: {[e['name'] for e in key_elements]}"
        
        # Generate summary
        response = ai_client.chat.completions.create(
            model=chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "Create a concise summary. Fix spelling/grammar."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=150  # Reduced from 200
        )
        
        # Log completion
        logger.info("Completed summary generation")
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Error generating summary: {str(e)}"

def spell_check_content(content):
    """Spell check and correct content"""
    try:
        # Acknowledge progress before spell check
        logger.info(f"Starting spell check for content")
        
        # Update current operation for progress monitoring
        global current_operation
        current_operation = "spell checking content"
        
        # Simplified prompt for faster processing
        spell_check_response = ai_client.chat.completions.create(
            model=chat_model,
            messages=[
                {
                    "role": "system", 
                    "content": "Fix spelling and grammar. Return corrected text only."
                },
                {"role": "user", "content": content}
            ],
            max_tokens=500  # Limit response size
        )
        
        # Acknowledge progress after spell check
        logger.info(f"Completed spell check for content")
        
        return spell_check_response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error during spell check: {e}")
        return content

def process_element(element, segment_index, total_segments):
    """Process a single element (research and image prompt generation)"""
    try:
        # Check cache first
        cached_research, cached_prompt = get_cached_research(element["name"])
        if cached_research and cached_prompt:
            logger.info(f"Using cached results for element {element['name']}")
            return {
                "name": element["name"],
                "research": cached_research,
                "image_prompt": cached_prompt
            }
        
        # Research and spell check combined
        logger.info(f"Starting research for element {element['name']} in segment {segment_index}/{total_segments}")
        research = research_element(element["name"])
        logger.info(f"Completed research for element {element['name']}")
        
        # Generate image prompt
        logger.info(f"Starting image prompt generation for element {element['name']}")
        image_prompt = generate_image_prompt(element["name"], research)
        logger.info(f"Completed image prompt generation for element {element['name']}")
        
        # Cache the results
        cache_research_results(element["name"], research, image_prompt)
        
        return {
            "name": element["name"],
            "research": research,
            "image_prompt": image_prompt
        }
    except Exception as e:
        logger.error(f"Error processing element {element['name']}: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "name": element["name"],
            "research": f"Error researching {element['name']}: {str(e)}",
            "image_prompt": f"Error generating prompt for {element['name']}"
        }

def enhance_segments(segments, non_interactive=False):
    """Enhance segments with AI-generated descriptions and historical context"""
    global current_operation
    current_operation = "enhancing segments"
    
    print("Enhancing segments with AI. This may take several minutes...")
    print("Enhancing segments with well-researched contextual information...")
    
    # Track timing statistics
    segment_times = []
    start_time = time.time()
    
    # Create a thread pool for parallel processing
    max_workers = 3  # Process up to 3 elements concurrently
    executor = ThreadPoolExecutor(max_workers=max_workers)
    
    for i, segment in enumerate(segments):
        segment_start = time.time()
        current_operation = f"enhancing segment {i+1}/{len(segments)}"
        print(f"Enhancing segment {i+1}/{len(segments)}...")
        check_cancel()
        
        try:
            # Extract facts and key elements from the segment
            facts = extract_facts(segment["text"])
            key_elements = extract_key_elements(segment["text"])
            
            # Process elements in parallel
            futures = []
            for element in key_elements:
                future = executor.submit(
                    process_element,
                    element,
                    i + 1,
                    len(segments)
                )
                futures.append(future)
            
            # Wait for all elements to be processed
            for j, future in enumerate(futures):
                try:
                    result = future.result()
                    key_elements[j].update(result)
                except Exception as e:
                    logger.error(f"Error processing element {j+1}: {str(e)}")
                    continue
            
            # Generate summary
            current_operation = f"generating summary for segment {i+1}"
            check_cancel()
            
            # Acknowledge progress before summary generation
            logger.info(f"Starting summary generation for segment {i+1}")
            
            summary = generate_summary(segment["text"], facts, key_elements)
            
            # Acknowledge progress after summary generation
            logger.info(f"Completed summary generation for segment {i+1}")
            
            # Update segment with enhanced information
            segment["facts"] = facts
            segment["key_elements"] = key_elements
            segment["summary"] = summary
            
            # Calculate and display timing statistics
            segment_duration = time.time() - segment_start
            segment_times.append(segment_duration)
            
            # Calculate average time and remaining time
            avg_time = sum(segment_times) / len(segment_times)
            remaining_segments = len(segments) - (i + 1)
            estimated_remaining_time = avg_time * remaining_segments
            
            # Calculate progress percentage
            progress = (i + 1) / len(segments) * 100
            
            # Display progress and timing information
            print(f"\nProgress: {progress:.1f}% ({i+1}/{len(segments)} segments)")
            print(f"Current segment duration: {segment_duration:.1f}s")
            print(f"Average segment duration: {avg_time:.1f}s")
            print(f"Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
            print(f"Total elapsed time: {(time.time() - start_time)/60:.1f} minutes")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print(f"\nEnhancement cancelled during segment {i+1}")
            executor.shutdown(wait=False)
            return segments
        except Exception as e:
            print(f"Error enhancing segment {i+1}: {e}")
            continue
    
    # Clean up the thread pool
    executor.shutdown(wait=True)
    return segments

def generate_visuals(enhanced_segments, output_dir, allow_dalle_fallback=False, non_interactive=False):
    """Generate historically accurate visuals for each enhanced segment"""
    global current_operation
    current_operation = "generating visuals"
    
    print("Generating educational visuals with historical accuracy...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Track timing statistics
    segment_times = []
    start_time = time.time()
    
    # Calculate estimated total cost
    num_images = sum(1 + len(segment.get("key_elements", [])) for segment in enhanced_segments)
    estimated_cost = num_images * 0.04 if stability_api is None else num_images * 0.003
    
    # For large image counts, warn and confirm
    if num_images > 10 and not allow_dalle_fallback and not non_interactive:
        provider = "DALL-E" if stability_api is None else "Stability AI"
        print(f"This will generate {num_images} images using {provider}")
        print(f"Estimated cost: ${estimated_cost:.2f}")
        
        if not get_user_confirmation(
            f"Continue with generating {num_images} images (est. ${estimated_cost:.2f})?",
            default=False,
            non_interactive=non_interactive
        ):
            print("Visual generation limited by user. Using placeholder images.")
            # TODO: Use placeholder images
    
    for i, segment in enumerate(enhanced_segments):
        segment_start = time.time()
        current_operation = f"generating visuals for segment {i+1}/{len(enhanced_segments)}"
        check_cancel()
        
        segment_dir = os.path.join(output_dir, f"segment_{i}")
        if not os.path.exists(segment_dir):
            os.makedirs(segment_dir)
        
        try:
            # Generate main image for the segment
            main_image_path = os.path.join(segment_dir, "main.png")
            
            if not os.path.exists(main_image_path):
                print(f"Generating main image for segment {i+1}/{len(enhanced_segments)}...")
                success = generate_image_stability(
                    segment["summary"], 
                    main_image_path,
                    allow_dalle_fallback=allow_dalle_fallback,
                    non_interactive=non_interactive
                )
                
                if not success and not is_cancelling:
                    print(f"Failed to generate main image for segment {i+1}. Using placeholder.")
                    # TODO: Use a placeholder image
            else:
                print(f"Using existing main image for segment {i+1}/{len(enhanced_segments)}")
            
            segment["main_image"] = main_image_path
            
            # Generate images for key elements
            element_images = []
            for j, element in enumerate(segment.get("key_elements", [])):
                current_operation = f"generating image for element {j+1} in segment {i+1}"
                check_cancel()
                
                element_image_path = os.path.join(segment_dir, f"element_{j}.png")
                
                if not os.path.exists(element_image_path):
                    print(f"Generating image for element {j+1}/{len(segment.get('key_elements', []))} in segment {i+1}/{len(enhanced_segments)}...")
                    success = generate_image_stability(
                        element["image_prompt"], 
                        element_image_path, 
                        width=640, 
                        height=480,
                        allow_dalle_fallback=allow_dalle_fallback,
                        non_interactive=non_interactive
                    )
                    
                    if not success and not is_cancelling:
                        print(f"Failed to generate element image. Using placeholder.")
                        # TODO: Use a placeholder image
                else:
                    print(f"Using existing image for element {j+1}/{len(segment.get('key_elements', []))} in segment {i+1}/{len(enhanced_segments)}")
                
                element["image_path"] = element_image_path
                element_images.append(element_image_path)
                
                # Avoid rate limiting
                time.sleep(0.5)
            
            segment["element_images"] = element_images
            
            # Calculate and display timing statistics
            segment_duration = time.time() - segment_start
            segment_times.append(segment_duration)
            
            # Calculate average time and remaining time
            avg_time = sum(segment_times) / len(segment_times)
            remaining_segments = len(enhanced_segments) - (i + 1)
            estimated_remaining_time = avg_time * remaining_segments
            
            # Calculate progress percentage
            progress = (i + 1) / len(enhanced_segments) * 100
            
            # Display progress and timing information
            print(f"\nVisual Generation Progress: {progress:.1f}% ({i+1}/{len(enhanced_segments)} segments)")
            print(f"Current segment duration: {segment_duration:.1f}s")
            print(f"Average segment duration: {avg_time:.1f}s")
            print(f"Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
            print(f"Total elapsed time: {(time.time() - start_time)/60:.1f} minutes")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print(f"\nVisual generation cancelled during segment {i+1}")
            return enhanced_segments
    
    return enhanced_segments

def generate_image_stability(prompt, output_path, width=768, height=512, allow_dalle_fallback=False, non_interactive=False):
    """Generate an image using Stability API with fallback to DALL-E"""
    global current_operation
    current_operation = "generating image"
    
    try:
        # Log detailed request information
        logger.info("Stability AI Image Generation Request:")
        logger.info(f"- Prompt: {prompt}")
        logger.info(f"- Output path: {output_path}")
        logger.info(f"- Dimensions: {width}x{height}")
        logger.info(f"- Engine ID: {stability_engine_id}")
        
        # Create the image generation request
        request = stability_pb2.GenerationRequest(
            prompt=prompt,
            width=width,
            height=height,
            steps=30,
            cfg_scale=7.0,
            sampler=stability_pb2.SAMPLER_K_DPMPP_2M
        )
        
        # Log request details
        logger.info("Sending request to Stability API...")
        
        # Send the request to Stability API
        response = stability_client.Generate(request)
        
        # Log response details
        logger.info("Received response from Stability API")
        logger.info(f"- Response type: {type(response)}")
        logger.info(f"- Number of artifacts: {len(response.artifacts) if hasattr(response, 'artifacts') else 'No artifacts'}")
        
        # Process the response
        if response.artifacts:
            # Log artifact details
            for i, artifact in enumerate(response.artifacts):
                logger.info(f"Artifact {i+1}:")
                logger.info(f"- Type: {artifact.type}")
                logger.info(f"- Size: {len(artifact.binary) if hasattr(artifact, 'binary') else 'No binary data'}")
            
            # Save the generated image
            with open(output_path, "wb") as f:
                f.write(response.artifacts[0].binary)
            
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
        else:
            logger.warning("No artifacts in Stability API response")
            print(f"✗ Stability API failed to generate image")
            
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
        
        if not os.environ.get("STABILITY_API_KEY") and not args.allow_dalle_fallback and not args.non_interactive:
            logger.warning("STABILITY_API_KEY environment variable not set.")
            logger.warning("Image generation will fall back to more expensive DALL-E API.")
            if not get_user_confirmation("Continue using DALL-E for image generation?", default=False):
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
