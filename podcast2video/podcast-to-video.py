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
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('podcast2video')

# Global variables to track state
current_operation = "initializing"
is_cancelling = False
temp_files = []
start_time = time.time()

# Debug function
def debug_point(message, level=logging.DEBUG):
    """Log debug information with consistent formatting"""
    logger.log(level, f"DEBUG: {message}")
    # Also track current operation
    global current_operation
    current_operation = message

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
            chat_model = "deepseek-v3"
            embedding_model = "deepseek-v3-embedding"
            logger.info("Using DeepSeek API for language models")
        except ImportError:
            logger.error("OpenAI library not installed. Run: pip install openai")
            raise
    else:
        if not openai_api_key:
            logger.error("No API keys available - both OPENAI_API_KEY and DEEPSEEK_API_KEY are missing")
        else:
            debug_point("Initializing OpenAI client")
            try:
                # Check if OpenAI library is installed
                from openai import OpenAI
                ai_client = OpenAI(api_key=openai_api_key)
                chat_model = "gpt-4o"
                embedding_model = "text-embedding-3-large"
                logger.info("Using OpenAI API for language models")
            except ImportError:
                logger.error("OpenAI library not installed. Run: pip install openai")
                raise
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
            audio.export(mp3_path, format="mp3", bitrate="192k")
            logger.info(f"Converted to MP3: {mp3_path}")
            return mp3_path
        except Exception as e:
            logger.error(f"Could not convert WAV to MP3: {e}", exc_info=True)
            logger.warning("Using original WAV file for transcription")
            return audio_file_path
    
    # Fallback for other formats (shouldn't reach here due to validation)
    return audio_file_path

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
    
    with open(normalized_audio_path, "rb") as audio_file:
        try:
            if deepseek_api_key:
                # Use DeepSeek's equivalent for audio transcription if available
                try:
                    transcript = ai_client.audio.transcriptions.create(
                        model="deepseek-whisper",
                        file=audio_file,
                        response_format="verbose_json"
                    )
                except Exception as e:
                    print(f"DeepSeek transcription failed: {e}. Falling back to OpenAI.")
                    # Fallback to OpenAI if DeepSeek doesn't support the format
                    if not get_user_confirmation(
                        "DeepSeek transcription failed. Fall back to OpenAI Whisper (may incur additional costs)?",
                        default=True,
                        non_interactive=non_interactive
                    ):
                        print("Transcription cancelled by user.")
                        return None
                    
                    temp_client = OpenAI(api_key=openai_api_key)
                    transcript = temp_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json"
                    )
            else:
                transcript = ai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
            
            # Check for cancellation after API call
            check_cancel()
            
        except KeyboardInterrupt:
            print("Transcription cancelled during API call")
            return None
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None
    
    # Generate subtitle files
    srt_path, vtt_path = generate_subtitles(transcript, normalized_audio_path)
    
    # Save to cache
    try:
        check_cancel()
        
        with open(cache_paths["json"], "w") as f:
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
    
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        for i, segment in enumerate(transcript.segments):
            # Format timestamp for SRT (HH:MM:SS,mmm)
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            
            # Write SRT entry
            srt_file.write(f"{i+1}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{segment.text.strip()}\n\n")
    
    # Also create WebVTT format
    with open(vtt_path, 'w', encoding='utf-8') as vtt_file:
        vtt_file.write("WEBVTT\n\n")
        for i, segment in enumerate(transcript.segments):
            # Format timestamp for VTT (HH:MM:SS.mmm)
            start_time = format_timestamp(segment.start, vtt=True)
            end_time = format_timestamp(segment.end, vtt=True)
            
            # Write VTT entry
            vtt_file.write(f"{start_time} --> {end_time}\n")
            vtt_file.write(f"{segment.text.strip()}\n\n")
    
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
    
    for segment in transcript.segments:
        if current_segment["start"] is None:
            current_segment["start"] = segment.start
        
        current_segment["text"] += " " + segment.text
        current_segment["end"] = segment.end
        
        segment_duration = segment.end - current_segment["start"]
        
        # If segment is long enough and contains a complete thought (ends with punctuation)
        if segment_duration >= min_segment_duration and re.search(r'[.!?]$', current_segment["text"].strip()):
            segments.append(current_segment)
            current_segment = {"text": "", "start": None, "end": None}
        # If segment exceeds max duration, split it anyway
        elif segment_duration >= max_segment_duration:
            segments.append(current_segment)
            current_segment = {"text": "", "start": None, "end": None}
    
    # Add the last segment if not empty
    if current_segment["text"] and current_segment["start"] is not None:
        segments.append(current_segment)
    
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
    main_bg = ImageClip(segment["main_image"]).set_duration(segment_duration)
    main_bg = main_bg.resize(width=1920)  # Resize for 1080p video
    clips.append(main_bg)
    
    # Add caption text
    if "caption" in segment and segment["caption"]:
        caption_txt = TextClip(
            segment["caption"], 
            fontsize=36, 
            color='white',
            bg_color='rgba(0,0,0,0.5)',
            font='Arial-Bold',
            size=(1800, None),
            method='caption'
        ).set_position(('center', 100)).set_duration(segment_duration)
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
                elem_img = ImageClip(img_path).set_start(element_start).set_duration(element_duration)
                elem_img = elem_img.resize(width=640)  # Smaller size for element images
                
                # Position in the right side
                elem_img = elem_img.set_position(('right', 180))
                clips.append(elem_img)
                
                # Add element name (title)
                name_txt = TextClip(
                    f"{element['name']}", 
                    fontsize=28, 
                    color='white',
                    bg_color='rgba(0,0,0,0.7)',
                    font='Arial-Bold',
                    size=(620, None),
                    method='caption'
                ).set_position(('right', 180 + elem_img.size[1] + 10)).set_start(element_start).set_duration(element_duration)
                clips.append(name_txt)
                
                # Add element description below the name
                desc_txt = TextClip(
                    f"{element.get('description', element.get('visual_caption', ''))}",
                    fontsize=22, 
                    color='white',
                    bg_color='rgba(0,0,0,0.6)',
                    font='Arial',
                    size=(620, None),
                    method='caption'
                ).set_position(('right', 180 + elem_img.size[1] + 50)).set_start(element_start).set_duration(element_duration)
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
                        sub["text"], 
                        fontsize=24, 
                        color='white',
                        bg_color='rgba(0,0,0,0.7)',
                        font='Arial',
                        size=(1800, None),
                        method='caption'
                    ).set_start(rel_start).set_duration(rel_end - rel_start)
                    
                    # Position at bottom of screen
                    sub_txt = sub_txt.set_position(('center', 'bottom'))
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
            video_clip = video_clip.set_start(segment["start"])
            
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
        final_video = final_video.set_audio(audio)
        
        # Add any final overlays or effects
        
        # Write final video
        current_operation = "rendering final video"
        print("Writing final video file...")
        check_cancel()
        
        final_video.write_videofile(output_path, fps=24, codec='libx264')
        
        # Also create a version with hardcoded subtitles for platforms that don't support SRT
        try:
            # Copy the final video
            hardcoded_output = str(Path(output_path).with_stem(f"{Path(output_path).stem}_with_subs"))
            
            # Use ffmpeg to add hardcoded subtitles
            ffmpeg_cmd = [
                'ffmpeg', 
                '-i', output_path, 
                '-vf', f"subtitles='{srt_path}'", 
                '-c:a', 'copy', 
                hardcoded_output
            ]
            
            subprocess.run(ffmpeg_cmd, check=True)
            print(f"Video with hardcoded subtitles saved to {hardcoded_output}")
        except Exception as e:
            print(f"Warning: Could not create hardcoded subtitle version: {e}")
        
        print(f"Final video saved to {output_path}")
        return output_path
        
    except KeyboardInterrupt:
        print("Final video creation cancelled")
        return None
    except Exception as e:
        print(f"Error creating final video: {e}")
        return None

if __name__ == "__main__":
    parser = main()
    args = parser.parse_args()
    start_time = time.time()
    
    try:
        debug_point("Starting podcast-to-video conversion")
        logger.info(f"Command line arguments: {args}")
        
        # Create temp and transcript directories
        debug_point(f"Creating directories: {args.temp_dir}, {args.transcript_dir}")
        os.makedirs(args.temp_dir, exist_ok=True)
        os.makedirs(args.transcript_dir, exist_ok=True)
        
        # Check for required API keys and warn user
        debug_point("Checking API keys")
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY environment variable not set.")
            logger.warning("This is required for transcription and content enhancement.")
            if not args.non_interactive and not get_user_confirmation("Continue without OpenAI API key?", default=False):
                logger.info("Exiting due to missing OpenAI API key")
                exit(1)
        
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
        logger.info("Stopping at API key check as requested")
        print("\n" + "="*50)
        print("Debug run complete. Stopping before API usage.")
        print("To continue, export your API keys and run the script again.")
        print("="*50 + "\n")
        exit(0)
        
    except KeyboardInterrupt:
        # This catch is for any cancellations that happen outside of the functions
        logger.info("\nOperation cancelled by user.")
        exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        exit(1)
