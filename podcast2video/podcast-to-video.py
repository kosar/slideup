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

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("podcast2video.log")  # Also log to file
    ]
)
logger = logging.getLogger('podcast2video')

# Global variables to track state
current_operation = "initializing"
is_cancelling = False
temp_files = []
start_time = time.time()
last_progress_time = time.time()  # For tracking progress updates
progress_thread = None

# Debug function
def debug_point(message, level=logging.DEBUG):
    """Log debug information with consistent formatting"""
    logger.log(level, f"DEBUG: {message}")
    # Also track current operation
    global current_operation, last_progress_time
    current_operation = message
    last_progress_time = time.time()

# Progress monitoring thread
def start_progress_monitoring(interval=10):
    """Start a thread to periodically report progress"""
    def monitor_progress():
        global last_progress_time
        last_report = ""
        
        while not is_cancelling:
            time.sleep(interval)
            elapsed = time.time() - start_time
            if last_report != current_operation:
                print(f"Status after {elapsed:.1f}s: {current_operation}")
                last_report = current_operation
                
            # Check for potential hanging
            time_since_update = time.time() - last_progress_time
            if time_since_update > interval * 2:
                print(f"WARNING: No progress updates for {time_since_update:.1f}s while: {current_operation}")
                
    thread = threading.Thread(target=monitor_progress, daemon=True)
    thread.start()
    return thread

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
        print(f"\nWARNING: {description} is taking longer than {timeout_seconds} seconds")
        print(f"Still running: {current_operation}")
        print(f"This might indicate the operation is stuck or just taking a long time.")
        return None, TimeoutError(f"{description} exceeded {timeout_seconds}s timeout")
    
    if exception[0]:
        return None, exception[0]
        
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
        
        # Check for direct SRT file as fallback
        elif os.path.exists(direct_srt_path):
            print(f"Found existing SRT file: {direct_srt_path}")
            # Ask user if they want to use existing SRT or regenerate
            if not get_user_confirmation(
                f"Found existing subtitle file for {audio_basename}. Use it instead of transcribing again?", 
                default=True,
                non_interactive=non_interactive
            ):
                print("User chose to regenerate transcript.")
            else:
                print("Using existing subtitle file")
                # Parse SRT into transcript format
                subtitles = parse_srt(direct_srt_path)
                
                # Create a simplified transcript object from SRT
                transcript = {"segments": []}
                for sub in subtitles:
                    transcript["segments"].append({
                        "start": sub["start"],
                        "end": sub["end"],
                        "text": sub["text"]
                    })
                
                # Still need to generate VTT if it doesn't exist
                if not os.path.exists(direct_vtt_path):
                    with open(direct_vtt_path, 'w', encoding='utf-8') as vtt_file:
                        vtt_file.write("WEBVTT\n\n")
                        for sub in subtitles:
                            start_time = format_timestamp(sub["start"], vtt=True)
                            end_time = format_timestamp(sub["end"], vtt=True)
                            vtt_file.write(f"{start_time} --> {end_time}\n")
                            vtt_file.write(f"{sub['text'].strip()}\n\n")
                
                # Save to cache for future use
                with open(cache_paths["json"], "w") as f:
                    json.dump(transcript, f)
                
                with open(cache_paths["srt"], "w") as f:
                    with open(direct_srt_path, "r") as src:
                        f.write(src.read())
                
                with open(cache_paths["vtt"], "w") as f:
                    with open(direct_vtt_path, "r") as src:
                        f.write(src.read())
                
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

def enhance_segments(segments):
    """Use AI API to enhance each segment with accurate factual content and research"""
    global current_operation
    current_operation = "enhancing segments with AI"
    
    print("Enhancing segments with well-researched contextual information...")
    
    enhanced_segments = []
    
    for i, segment in enumerate(segments):
        print(f"Enhancing segment {i+1}/{len(segments)}...")
        check_cancel()
        
        try:
            # Step 1: Extract key facts, names, concepts from transcript
            current_operation = f"extracting facts from segment {i+1}/{len(segments)}"
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
                    {"role": "user", "content": segment["text"]}
                ],
                response_format={"type": "json_object"}
            )
            
            extracted_elements = json.loads(extraction_response.choices[0].message.content)
            
            # Step 2: Research each extracted element and create accurate content
            researched_elements = []
            
            for j, element in enumerate(extracted_elements.get("extracted_elements", [])):
                current_operation = f"researching element {j+1}/{len(extracted_elements.get('extracted_elements', []))} in segment {i+1}/{len(segments)}"
                check_cancel()
                
                research_response = ai_client.chat.completions.create(
                    model=chat_model,
                    messages=[
                        {
                            "role": "system", 
                            "content": """You are a meticulous educational researcher. 
                                       Research the following element that was mentioned in a podcast.
                                       Provide factually accurate information about this element.
                                       
                                       Focus on verified historical facts, dates, and descriptions.
                                       Double-check any names, dates, and terminology for accuracy.
                                       Be concise but comprehensive.
                                       
                                       Also create a specific, historically accurate image prompt that 
                                       will generate an authentic visual representation.
                                       
                                       Format your response as JSON with the following structure:
                                       {
                                           "type": "element type",
                                           "name": "precise name with correct spelling",
                                           "description": "factual, well-researched description limited to 100 words",
                                           "visual_caption": "short caption for the visual (limited to 40 words)",
                                           "image_prompt": "detailed, historically accurate prompt for image generation"
                                       }"""
                        },
                        {"role": "user", "content": f"Research this element from a podcast: {element['name']}. Context from podcast: {element['context']}"}
                    ],
                    response_format={"type": "json_object"}
                )
                
                researched_element = json.loads(research_response.choices[0].message.content)
                researched_elements.append(researched_element)
                
                # Spell check the content
                current_operation = f"spell checking element {j+1}"
                check_cancel()
                
                spell_check_response = ai_client.chat.completions.create(
                    model=chat_model,
                    messages=[
                        {
                            "role": "system", 
                            "content": """You are a professional copy editor and proofreader.
                                       Carefully check the following content for spelling, grammar, and factual accuracy.
                                       Ensure all names, terminology, and dates are correctly spelled.
                                       
                                       Return the corrected content in the same JSON structure.
                                       If no corrections are needed, return the original content."""
                        },
                        {"role": "user", "content": json.dumps(researched_element)}
                    ],
                    response_format={"type": "json_object"}
                )
                
                # Replace with spell-checked content
                researched_elements[-1] = json.loads(spell_check_response.choices[0].message.content)
                
                # Avoid rate limiting
                time.sleep(0.5)
            
            # Step 3: Create the final enhanced segment with summary and transition
            current_operation = f"finalizing segment {i+1}/{len(segments)}"
            check_cancel()
            
            finalization_response = ai_client.chat.completions.create(
                model=chat_model,
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a video production specialist.
                                   Create a concise summary and visual plan for this podcast segment.
                                   Use the given transcript and researched elements to create an engaging video segment.
                                   
                                   Keep all text concise for on-screen display.
                                   Ensure all text is correctly spelled and grammatically perfect.
                                   Choose appropriate transition styles based on content.
                                   
                                   Format your response as JSON with the following structure:
                                   {
                                       "summary": "brief summary of the segment (50 words max)",
                                       "key_elements": [LIST OF RESEARCHED ELEMENTS],
                                       "caption": "main caption for this segment (30 words max)",
                                       "subtitle_display": "always/important_parts/none",
                                       "transition_style": "cut/fade/slide/zoom"
                                   }"""
                    },
                    {"role": "user", "content": f"Transcript: {segment['text']}\n\nResearched Elements: {json.dumps(researched_elements)}"}
                ],
                response_format={"type": "json_object"}
            )
            
            enhancement = json.loads(finalization_response.choices[0].message.content)
            # Replace key_elements with our researched elements to maintain consistency
            enhancement["key_elements"] = researched_elements
            enhanced_segment = {**segment, **enhancement}
            enhanced_segments.append(enhanced_segment)
            
        except KeyboardInterrupt:
            print(f"\nEnhancement cancelled during segment {i+1}")
            # Return what we have so far, or None if we don't have any segments
            return enhanced_segments if enhanced_segments else None
        except Exception as e:
            print(f"Error enhancing segment {i+1}: {e}")
            # Continue with the next segment instead of failing the entire process
            continue
    
    current_operation = "segment enhancement"
    return enhanced_segments

def generate_visuals(enhanced_segments, output_dir, allow_dalle_fallback=False, non_interactive=False):
    """Generate historically accurate visuals for each enhanced segment"""
    global current_operation
    current_operation = "generating visuals"
    
    print("Generating educational visuals with historical accuracy...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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
        current_operation = f"generating visuals for segment {i+1}/{len(enhanced_segments)}"
        check_cancel()
        
        segment_dir = os.path.join(output_dir, f"segment_{i}")
        if not os.path.exists(segment_dir):
            os.makedirs(segment_dir)
        
        try:
            # Generate main image for the segment
            # Spell check the summary before using it as a prompt
            spell_check_response = ai_client.chat.completions.create(
                model=chat_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional proofreader. Fix any spelling or grammar errors in this text that will be used as an image generation prompt. Maintain the same meaning but ensure perfect spelling."
                    },
                    {"role": "user", "content": segment["summary"]}
                ]
            )
            
            main_image_prompt = spell_check_response.choices[0].message.content
            main_image_path = os.path.join(segment_dir, "main.png")
            
            if not os.path.exists(main_image_path):
                print(f"Generating main image for segment {i+1}/{len(enhanced_segments)}...")
                success = generate_image_stability(
                    main_image_prompt, 
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
                
                # Get the image prompt and spell check it
                image_prompt = element.get("image_prompt", "")
                if not image_prompt and "name" in element:
                    # Create a basic prompt if none exists
                    image_prompt = f"Historically accurate depiction of {element['name']}"
                
                # Spell check the prompt
                spell_check_response = ai_client.chat.completions.create(
                    model=chat_model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a professional proofreader. Fix any spelling or grammar errors in this text that will be used as an image generation prompt. Maintain the same meaning but ensure perfect spelling."
                        },
                        {"role": "user", "content": image_prompt}
                    ]
                )
                
                checked_prompt = spell_check_response.choices[0].message.content
                
                if not os.path.exists(element_image_path):
                    print(f"Generating image for element {j+1}/{len(segment.get('key_elements', []))} in segment {i+1}/{len(enhanced_segments)}...")
                    success = generate_image_stability(
                        checked_prompt, 
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
            
        except KeyboardInterrupt:
            print(f"\nVisual generation cancelled during segment {i+1}")
            # Return what we have so far
            return enhanced_segments
    
    return enhanced_segments

def generate_image_stability(prompt, output_path, width=768, height=512, allow_dalle_fallback=False, non_interactive=False):
    """Generate image using Stability API with historically accurate content focus"""
    global current_operation, temp_files
    current_operation = "generating image"
    
    # Register the output path as a temporary file that might need cleanup
    temp_files.append(output_path)
    
    # Add accuracy and historical details to the prompt
    enhanced_prompt = f"""Create a historically accurate, factually correct image of {prompt}.
    Include authentic period details and accurate visual elements.
    Ensure all text elements are clearly legible and correctly spelled.
    Focus on educational value and historical authenticity."""
    
    if stability_api is None:
        print("Stability API not initialized.")
        if not allow_dalle_fallback and not get_user_confirmation(
            "Stability API not available. Use more expensive DALL-E instead?",
            default=False,
            non_interactive=non_interactive
        ):
            print("Image generation skipped.")
            return False
        use_dalle = True
    else:
        use_dalle = False
    
    if not use_dalle:
        try:
            check_cancel()
            current_operation = "generating image with Stability AI"
            
            answers = stability_api.generate(
                prompt=enhanced_prompt,
                width=width,
                height=height,
                samples=1,
                steps=30,
                cfg_scale=8.0  # Increase CFG scale for better prompt adherence
            )
            
            check_cancel()
            
            for resp in answers:
                for artifact in resp.artifacts:
                    if artifact.type == generation.ARTIFACT_IMAGE:
                        check_cancel()
                        with open(output_path, "wb") as f:
                            f.write(artifact.binary)
                        print(f"Image saved to {output_path}")
                        return True
        except KeyboardInterrupt:
            print("Image generation cancelled")
            return False
        except Exception as e:
            print(f"Error generating image with Stability API: {e}")
            
            if not allow_dalle_fallback and not get_user_confirmation(
                "Stability API failed. Use more expensive DALL-E instead?",
                default=False,
                non_interactive=non_interactive
            ):
                print("Image generation skipped after Stability failure.")
                return False
            use_dalle = True
    
    # If we get here, we need to use DALL-E
    estimated_cost = 0.04  # Approximate cost of a DALL-E 3 standard image
    print(f"Using DALL-E. Estimated cost per image: ${estimated_cost:.2f}")
    
    try:
        check_cancel()
        current_operation = "generating image with DALL-E"
        
        # Check if we should use OpenAI or DeepSeek for fallback
        if deepseek_api_key:
            # Try to use DeepSeek's image generation if available
            try:
                response = ai_client.images.generate(
                    model="deepseek-image",
                    prompt=enhanced_prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )
                image_url = response.data[0].url
            except Exception as deepseek_img_error:
                print(f"DeepSeek image generation failed: {deepseek_img_error}. Falling back to OpenAI.")
                # Fall back to OpenAI
                temp_client = OpenAI(api_key=openai_api_key)
                response = temp_client.images.generate(
                    model="dall-e-3",
                    prompt=enhanced_prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )
                image_url = response.data[0].url
        else:
            # Use OpenAI directly
            response = ai_client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url
        
        # Download the image
        check_cancel()
        image_data = requests.get(image_url).content
        
        check_cancel()
        with open(output_path, "wb") as f:
            f.write(image_data)
        
        print(f"DALL-E image saved to {output_path}")
        return True
        
    except KeyboardInterrupt:
        print("DALL-E image generation cancelled")
        return False
    except Exception as e2:
        print(f"Error in DALL-E image generation: {e2}")
        return False

def create_video_segment(segment, output_path, audio_file_path, style="modern"):
    """Create a video segment from an enhanced segment with subtitles"""
    global current_operation, temp_files
    current_operation = f"creating video segment from {segment['start']:.1f} to {segment['end']:.1f}"
    
    # Register output as a temporary file
    temp_files.append(output_path)
    
    # Extract audio for this segment
    audio_start = segment["start"]
    audio_end = segment["end"]
    segment_duration = audio_end - audio_start
    
    # Create clips for each element
    clips = []
    
    # Add main background image
    main_bg = ImageClip(segment["main_image"]).with_duration(segment_duration)
    main_bg = main_bg.resized(width=1920)  # Resize for 1080p video
    clips.append(main_bg)
    
    # Add caption text
    if "caption" in segment and segment["caption"]:
        caption_txt = TextClip(
            text=segment["caption"], 
            font_size=36, 
            color='white',
            bg_color='black',
            font='Arial',
            size=(1800, None),
            method='caption'
        ).with_position(('center', 100)).with_duration(segment_duration)
        clips.append(caption_txt)
    
    # Add key elements with images and descriptions
    if "key_elements" in segment and segment["element_images"]:
        num_elements = len(segment["key_elements"])
        for i, (element, img_path) in enumerate(zip(segment["key_elements"], segment["element_images"])):
            # Calculate timing for this element
            element_start = segment_duration * i / max(num_elements, 1)
            element_duration = segment_duration / max(num_elements, 1)
            
            # Create element image
            try:
                elem_img = ImageClip(img_path).with_duration(element_duration).with_start(element_start)
                elem_img = elem_img.resized(width=640)  # Smaller size for element images
                
                # Position in the right side
                elem_img = elem_img.with_position(('right', 180))
                clips.append(elem_img)
                
                # Add element name (title)
                name_txt = TextClip(
                    text=f"{element['name']}", 
                    font_size=28, 
                    color='white',
                    bg_color='black',
                    font='Arial',
                    size=(620, None),
                    method='caption'
                ).with_position(('right', 180 + elem_img.size[1] + 10)).with_start(element_start).with_duration(element_duration)
                clips.append(name_txt)
                
                # Add element description below the name
                desc_txt = TextClip(
                    text=f"{element.get('description', element.get('visual_caption', ''))}",
                    font_size=22, 
                    color='white',
                    bg_color='black',
                    font='Arial',
                    size=(620, None),
                    method='caption'
                ).with_position(('right', 180 + elem_img.size[1] + 50)).with_start(element_start).with_duration(element_duration)
                clips.append(desc_txt)
            except Exception as e:
                print(f"Error adding element image: {e}")
    
    # Add subtitles if requested
    subtitle_display = segment.get("subtitle_display", "always")
    
    if subtitle_display != "none":
        # Get the SRT file path
        srt_path = Path(audio_file_path).with_suffix('.srt')
        
        if srt_path.exists():
            try:
                # Parse SRT to get relevant subtitle segments
                subtitles = parse_srt(srt_path)
                relevant_subs = filter_subtitles(subtitles, audio_start, audio_end)
                
                # Add each subtitle as a text clip
                for sub in relevant_subs:
                    # Adjust timestamps relative to segment
                    rel_start = max(0, sub["start"] - audio_start)
                    rel_end = min(segment_duration, sub["end"] - audio_start)
                    
                    # Skip if outside segment bounds
                    if rel_start >= segment_duration or rel_end <= 0:
                        continue
                    
                    # Create subtitle text clip
                    sub_txt = TextClip(
                        text=sub["text"], 
                        font_size=24, 
                        color='white',
                        bg_color='black',
                        font='Arial',
                        size=(1800, None),
                        method='caption'
                    ).with_start(rel_start).with_duration(rel_end - rel_start)
                    
                    # Position at bottom of screen
                    sub_txt = sub_txt.with_position(('center', 'bottom'))
                    clips.append(sub_txt)
            except Exception as e:
                print(f"Error adding subtitles: {e}")
    
    # Composite all clips
    video = CompositeVideoClip(clips, size=(1920, 1080))
    
    # Write to file
    print(f"Rendering video segment...")
    check_cancel()
    video.write_videofile(output_path, fps=24, codec='libx264', audio=None)
    
    return output_path

def parse_srt(srt_path):
    """Parse SRT subtitle file"""
    subtitles = []
    current_sub = None
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Try to parse as a subtitle number
        try:
            int(line)
            # This is a new subtitle entry
            if current_sub:
                subtitles.append(current_sub)
            current_sub = {"index": int(line), "text": ""}
            i += 1
            continue
        except ValueError:
            pass
        
        # Try to parse as a timestamp
        if ' --> ' in line:
            timestamps = line.split(' --> ')
            if len(timestamps) == 2:
                current_sub["start"] = parse_timestamp(timestamps[0])
                current_sub["end"] = parse_timestamp(timestamps[1])
                i += 1
                continue
        
        # Must be subtitle text
        if current_sub:
            if current_sub["text"]:
                current_sub["text"] += " " + line
            else:
                current_sub["text"] = line
        
        i += 1
    
    # Add the last subtitle
    if current_sub:
        subtitles.append(current_sub)
    
    return subtitles

def parse_timestamp(timestamp):
    """Convert SRT timestamp to seconds"""
    # Format: 00:00:00,000
    timestamp = timestamp.replace(',', '.')
    h, m, s = timestamp.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def filter_subtitles(subtitles, start_time, end_time):
    """Filter subtitles to only include those within the given time range"""
    return [sub for sub in subtitles if sub["end"] >= start_time and sub["start"] <= end_time]

def create_final_video(enhanced_segments, audio_file, output_path, temp_dir):
    """Create the final video by concatenating all segments with subtitles"""
    global current_operation, temp_files
    current_operation = "creating final video"
    
    print("Creating final video...")
    
    # Register output as potential temporary file
    temp_files.append(output_path)
    
    # Ensure subtitle files exist
    srt_path = Path(audio_file).with_suffix('.srt')
    if not srt_path.exists():
        print("Warning: No subtitle file found. Generating one...")
        # If we don't have subtitles, transcribe the audio again to get them
        transcribe_audio(audio_file)
    
    # Create video segments
    video_segments = []
    for i, segment in enumerate(enhanced_segments):
        current_operation = f"creating video segment {i+1}/{len(enhanced_segments)}"
        check_cancel()
        
        segment_output = os.path.join(temp_dir, f"segment_{i}.mp4")
        segment_result = create_video_segment(segment, segment_output, audio_file)
        
        if segment_result:
            video_segments.append(segment_output)
        elif is_cancelling:
            print("Final video creation cancelled")
            return None
    
    # Check if we have any segments to work with
    if not video_segments:
        print("No video segments were successfully created. Cannot produce final video.")
        return None
    
    try:
        current_operation = "concatenating video segments"
        check_cancel()
        
        # Load audio file
        audio = AudioFileClip(audio_file)
        
        # Create video clips from segments
        clips = []
        for i, segment in enumerate(enhanced_segments):
            video_file = video_segments[i]
            
            # Load video clip
            video_clip = VideoFileClip(video_file, audio=False)
            
            # Set position in the timeline
            video_clip = video_clip.with_start(segment["start"])
            
            # Add transitions if specified
            transition_style = segment.get("transition_style", "cut")
            if i > 0 and transition_style != "cut":
                # Apply different transitions based on style
                prev_clip = clips[-1]
                if transition_style == "fade":
                    # Add fade transition
                    prev_clip = prev_clip.crossfadeout(0.5)
                    video_clip = video_clip.crossfadein(0.5)
                elif transition_style == "slide":
                    # Slide transition would be more complex, simplified version
                    pass
                elif transition_style == "zoom":
                    # Zoom transition would be more complex, simplified version
                    pass
            
            clips.append(video_clip)
        
        # Concatenate all clips
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Add original audio
        final_video = final_video.with_audio(audio)
        
        # Add any final overlays or effects
        
        # Write final video
        current_operation = "rendering final video"
        print("Writing final video file...")
        check_cancel()
        
        final_video.write_videofile(output_path, fps=24, codec='libx264')
        
        # Print info about created files and artifacts in a more prominent way
        print("\n")
        print("🎬 " + "=" * 30 + " VIDEO CREATION SUCCESSFUL " + "=" * 30 + " 🎬")
        print("\n")
        print("📹 \033[1mFINAL OUTPUT:\033[0m")
        print(f"   📺 Video: \033[1;32m{os.path.abspath(output_path)}\033[0m")
        
        # Print subtitle files if they exist
        srt_path = Path(audio_file).with_suffix('.srt')
        vtt_path = Path(audio_file).with_suffix('.vtt')
        
        print("\n📝 \033[1mSUBTITLES:\033[0m")
        if srt_path.exists():
            print(f"   🔤 SRT: \033[36m{os.path.abspath(srt_path)}\033[0m")
        else:
            print(f"   🔤 SRT: Not generated")
            
        if vtt_path.exists():
            print(f"   🔤 VTT: \033[36m{os.path.abspath(vtt_path)}\033[0m")
        else:
            print(f"   🔤 VTT: Not generated")
        
        # Print segment information
        print("\n🧩 \033[1mSEGMENT ARTIFACTS:\033[0m")
        print(f"   📁 Location: \033[36m{os.path.abspath(temp_dir)}\033[0m")
        for i, segment_path in enumerate(video_segments):
            if os.path.exists(segment_path):
                print(f"   🎞️  Segment {i+1}: {os.path.basename(segment_path)}")
        
        print("\n" + "=" * 80 + "\n")
        
        return output_path
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error creating final video: {e}\n{traceback_str}")
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
