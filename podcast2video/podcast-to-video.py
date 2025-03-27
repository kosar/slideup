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
import inspect

# Import cost tracker
try:
    from podcast2video.cost_tracker import get_cost_tracker
    COST_TRACKER_AVAILABLE = True
    cost_tracker = get_cost_tracker()
    # Reset cost tracker at startup
    cost_tracker.reset()
except ImportError:
    COST_TRACKER_AVAILABLE = False
    print("Cost tracker module not available. API costs will not be tracked.")

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
TIME_LIMIT_SECONDS = 180  # 3 minutes
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
chat_model = None  # Global variable to store the current chat model

# Import optional modules
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.error("OpenAI library not installed. Run: pip install openai")
    OPENAI_AVAILABLE = False

# Import SpeechRecognition and pocketsphinx for local transcription
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    logger.error("SpeechRecognition library not installed. Run: pip install SpeechRecognition")
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    import pocketsphinx
    POCKETSPHINX_AVAILABLE = True
except ImportError:
    logger.error("pocketsphinx library not installed. Run: pip install pocketsphinx")
    POCKETSPHINX_AVAILABLE = False

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
            
            # Report cost data periodically for the webapp
            if COST_TRACKER_AVAILABLE:
                try:
                    cost_summary = cost_tracker.get_summary()
                    cost_json = json.dumps(cost_summary)
                    print(f"COST_DATA: {cost_json}")
                except Exception as e:
                    logger.error(f"Error reporting cost data: {str(e)}")
            
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
    """Test OpenAI API connection"""
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        return False
    
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Test with a simple chat completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, are you working?"}],
            max_tokens=10
        )
        
        # Track API cost if cost tracker is available
        if COST_TRACKER_AVAILABLE:
            # Extract token usage
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost_tracker.add_openai_chat_cost(
                model="gpt-3.5-turbo",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                operation_name="api_test"
            )
        
        logger.info("OpenAI API test successful!")
        return True
    except Exception as e:
        logger.error(f"OpenAI API test failed: {e}")
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
    """Cleanup operations to perform when exiting"""
    global temp_files
    
    # Generate and output final cost report
    if COST_TRACKER_AVAILABLE:
        try:
            cost_summary = cost_tracker.get_summary()
            logger.info(f"Final cost summary: ${cost_summary['total_cost']:.4f} total")
            logger.info(f"API breakdown: OpenAI chat=${cost_summary['api_breakdown']['openai']['chat']:.4f}, " +
                       f"transcription=${cost_summary['api_breakdown']['openai']['transcription']:.4f}, " +
                       f"Stability image=${cost_summary['api_breakdown']['stability']['image']:.4f}")
            
            # Save detailed report
            report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost_report.json")
            cost_tracker.save_report(report_path)
            logger.info(f"Detailed cost report saved to {report_path}")
            
            # Print cost data in format for webapp to parse
            cost_json = json.dumps(cost_summary)
            print(f"COST_DATA: {cost_json}")
        except Exception as e:
            logger.error(f"Error generating cost report: {str(e)}")
    
    # Cleanup temp files
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.debug(f"Removed temp file: {temp_file}")
        except Exception as e:
            logger.warning(f"Failed to remove temp file {temp_file}: {str(e)}")

def main():
    """Main entry point for the podcast-to-video application"""
    debug_point("Starting podcast-to-video")
    
    # Initialize cost tracking if available
    if COST_TRACKER_AVAILABLE:
        logger.info("API cost tracking enabled")
        # Reset the cost tracker for this run
        cost_tracker.reset()

    parser = argparse.ArgumentParser(description='Convert podcast audio to video with AI-generated visuals')
    parser.add_argument('--audio', type=str, required=True, help='Path to input audio file')
    parser.add_argument('--output', type=str, default='enhanced_podcast.mp4', help='Path to output video file')
    parser.add_argument('--limit_to_one_minute', action='store_true', 
                       help=f'Limit processing to first {TIME_LIMIT_SECONDS} seconds ({TIME_LIMIT_SECONDS/60:.1f} minutes) of audio')
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
    """Test the connection to an API"""
    debug_point(f"Testing {api_type} API connection")
    
    try:
        if api_type == "openai":
            # Simple query to test the API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello, this is a test. Respond with 'OK' if you can see this."}],
                max_tokens=10
            )
            
            # Track API cost if cost tracker is available
            if COST_TRACKER_AVAILABLE:
                # Extract token usage
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                cost_tracker.add_openai_chat_cost(
                    model="gpt-3.5-turbo",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    operation_name="api_connection_test"
                )
            
            result = response.choices[0].message.content.strip()
            logger.info(f"OpenAI API test response: {result}")
            
            if "ok" in result.lower():
                logger.info("OpenAI API connection successful!")
                return True
            else:
                logger.warning(f"OpenAI API returned unexpected response: {result}")
                return True  # Still return true since we got a response
                
        elif api_type == "deepseek":
            # Test DeepSeek API with a simple query
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "Hello, this is a test. Respond with 'OK' if you can see this."}],
                max_tokens=10
            )
            
            # Track API cost if cost tracker is available
            if COST_TRACKER_AVAILABLE:
                # Extract token usage
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                cost_tracker.add_openai_chat_cost(
                    model="deepseek-chat",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    operation_name="api_connection_test"
                )
            
            result = response.choices[0].message.content.strip()
            logger.info(f"DeepSeek API test response: {result}")
            
            if "ok" in result.lower():
                logger.info("DeepSeek API connection successful!")
                return True
            else:
                logger.warning(f"DeepSeek API returned unexpected response: {result}")
                return True  # Still return true since we got a response
                
        elif api_type == "stability":
            # For Stability API, we'll make a test request
            headers = {"Authorization": f"Bearer {os.environ.get('STABILITY_API_KEY')}"}
            response = requests.get(
                "https://api.stability.ai/v1/engines/list", 
                headers=headers
            )
            
            if response.status_code == 200:
                logger.info("Stability API connection successful!")
                return True
            else:
                logger.error(f"Stability API test failed: {response.status_code} - {response.text}")
                return False
        
        return False
    except Exception as e:
        logger.error(f"API test failed: {e}")
        traceback.print_exc()
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
            chat_model = "deepseek-chat"  # DeepSeek's chat model
            embedding_model = "deepseek-embedding"  # DeepSeek's embedding model
            api_type = "deepseek"
            logger.info("Using DeepSeek API for language models")
            
            # Test DeepSeek API immediately
            if not test_api_connection(ai_client, api_type):
                logger.warning("DeepSeek API test failed. Falling back to OpenAI if available.")
                # Reset client if test failed
                ai_client = None
                api_type = None
                chat_model = None
                
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
            chat_model = "gpt-4"  # OpenAI's model
            embedding_model = "text-embedding-3-large"  # OpenAI's embedding model
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

def generate_image_stability(prompt, output_path, width=1024, height=1024, steps=50, samples=1, non_interactive=False, negative_prompt=None):
    """Generate an image using Stability API"""
    debug_point(f"Generating image for: {prompt[:50]}...")
    
    # Basic validation
    if not prompt:
        logger.error("Empty prompt provided for image generation")
        return None
    
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if Stability API key is available
    stability_key = os.getenv('STABILITY_API_KEY')
    if not stability_key:
        logger.error("STABILITY_API_KEY not found in environment variables")
        return None
    
    # Perform a secondary validation using Deepseek if directly calling this function
    # This ensures safety even when generate_image_stability is called directly
    try:
        # Only validate if we haven't already done so in generate_visuals
        is_direct_call = inspect.currentframe().f_back.f_code.co_name != "generate_visuals"
        if is_direct_call:
            logger.info("Direct call to generate_image_stability detected, performing prompt validation")
            is_safe, validated_prompt, validated_negative_prompt, reason = validate_image_prompt_with_deepseek(
                prompt, 
                negative_prompt
            )
            
            if not is_safe:
                logger.warning(f"Direct prompt validation detected potential issues: {reason}")
                prompt = validated_prompt
                negative_prompt = validated_negative_prompt
                logger.info(f"Using corrected prompts from validation")
    except Exception as e:
        # Don't block generation if validation fails
        logger.error(f"Error in direct prompt validation: {str(e)}")
    
    # This is the URL for text-to-image generation
    model_id = "stable-diffusion-xl-1024-v1-0"
    url = f"https://api.stability.ai/v1/generation/{model_id}/text-to-image"
    
    # Prepare headers
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {stability_key}"
    }
    
    # Prepare the text prompts section with positive and negative prompts
    text_prompts = [
        {
            "text": prompt,
            "weight": 1.0
        }
    ]
    
    # Add negative prompt if provided
    if negative_prompt:
        text_prompts.append({
            "text": negative_prompt,
            "weight": -0.8  # Negative weight for things to avoid
        })
        logger.info(f"Added negative prompt to request: {negative_prompt[:100]}...")
        logger.info(f"Negative prompt length: {len(negative_prompt)} characters")
        logger.info(f"Negative prompt weight: -0.8")
        
        # Log each individual negative prompt term for debugging
        negative_items = [item.strip() for item in negative_prompt.split(',') if item.strip()]
        logger.info(f"Negative prompt contains {len(negative_items)} individual terms")
        for i, item in enumerate(negative_items[:10]):  # Log first 10 items to avoid log spam
            logger.info(f"  Negative term {i+1}: '{item}'")
        if len(negative_items) > 10:
            logger.info(f"  ... and {len(negative_items) - 10} more negative terms")
    else:
        logger.warning("No negative prompt provided! This may result in inappropriate images.")
    
    # Prepare body
    body = {
        "text_prompts": text_prompts,
        "height": height,
        "width": width,
        "samples": samples,
        "steps": steps,
        "cfg_scale": 8.0,  # Increased guidance scale for better prompt adherence
        "style_preset": "photographic"  # Can be changed based on style preference
    }
    
    # Log the complete request body for debugging
    logger.info(f"Stability API request configuration: {json.dumps(body, indent=2)}")
    
    # Create more detailed log of exact prompt text for troubleshooting
    detailed_prompt_log = {
        "prompt_full_text": prompt,
        "prompt_length": len(prompt),
        "negative_prompt_full_text": negative_prompt,
        "negative_prompt_length": len(negative_prompt) if negative_prompt else 0,
        "negative_prompt_items_count": len([x for x in negative_prompt.split(',') if x.strip()]) if negative_prompt else 0
    }
    logger.info(f"DETAILED PROMPT LOG: {json.dumps(detailed_prompt_log, indent=2)}")
    
    try:
        response = requests.post(url, headers=headers, json=body)
        
        if response.status_code != 200:
            logger.error(f"Stability API error: {response.status_code} - {response.text}")
            return None
        
        # Parse response
        data = response.json()
        
        # Log successful image generation
        logger.info(f"Stability API request successful - generated {len(data.get('artifacts', []))} images")
        
        # Check for any issues in the response
        if 'artifacts' in data:
            for i, artifact in enumerate(data['artifacts']):
                if 'finish_reason' in artifact:
                    finish_reason = artifact['finish_reason'] 
                    if finish_reason != 'SUCCESS':
                        logger.warning(f"Image {i+1} generation had finish_reason: {finish_reason}")
                        if finish_reason == 'CONTENT_FILTERED':
                            logger.warning("Content was filtered despite negative prompts! Adjust negative prompts if this happens frequently.")
        
        # Track API cost if cost tracker is available
        if COST_TRACKER_AVAILABLE:
            cost_tracker.add_stability_image_cost(
                width=width,
                height=height,
                steps=steps,
                samples=samples,
                model=model_id,
                operation_name="image_generation"
            )
        
        # For each generated image
        for i, image in enumerate(data.get("artifacts", [])):
            # Determine filename
            if samples == 1:
                img_path = output_path
            else:
                # When multiple samples, create numbered files
                base, ext = os.path.splitext(output_path)
                img_path = f"{base}_{i+1}{ext}"
            
            # Save the image
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(image["base64"]))
            
            logger.info(f"Image saved to {img_path}")
        
        return output_path
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        traceback.print_exc()
        return None

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
        
        # Get start/end times with fallback between different naming conventions
        get_start_time = lambda segment: segment.get('start_time', segment.get('start', 0))
        get_end_time = lambda segment: segment.get('end_time', segment.get('end', 0))
        
        # Calculate target duration based on segments and time limit
        if limit_to_one_minute:
            logger.info(f"Applying time limit of {TIME_LIMIT_SECONDS} seconds from global constant")
            full_duration = max(get_end_time(segment) for segment in enhanced_segments)
            target_duration = min(full_duration, TIME_LIMIT_SECONDS)
            logger.info(f"Original full duration: {full_duration:.2f} seconds")
            logger.info(f"Using time-limited duration: {target_duration:.2f} seconds")
        else:
            target_duration = max(get_end_time(segment) for segment in enhanced_segments)
            logger.info(f"Using full duration: {target_duration:.2f} seconds")
        
        # Ensure the images directory exists and is accessible
        image_folder = os.path.join(temp_dir, "images")
        if not os.path.exists(image_folder):
            logger.warning(f"Images folder not found at {image_folder}, creating it")
            os.makedirs(image_folder, exist_ok=True)
        
        # First, create video segments from PNG files
        valid_segments = []
        for i, segment in enumerate(enhanced_segments, 1):
            # Get start/end times with fallback
            start_time = get_start_time(segment)
            end_time = get_end_time(segment)
            
            # Skip segments that start after the time limit if limit_to_one_minute is True
            if limit_to_one_minute and start_time >= TIME_LIMIT_SECONDS:
                logger.info(f"Skipping segment {i} as it starts after time limit of {TIME_LIMIT_SECONDS} seconds")
                continue
                
            # Adjust end time if it exceeds the limit and limit_to_one_minute is True
            original_end_time = end_time
            if limit_to_one_minute:
                end_time = min(end_time, TIME_LIMIT_SECONDS)
                if end_time < original_end_time:
                    logger.info(f"Trimming segment {i} end time from {original_end_time:.2f} to {end_time:.2f} seconds to match time limit")
            
            segment_duration = end_time - start_time
            
            # Skip segments with zero duration
            if segment_duration <= 0:
                logger.info(f"Skipping segment {i} as it has zero duration")
                continue
            
            # Look for the PNG file in the images subfolder
            png_path = os.path.join(image_folder, f"segment_{i}.png")
            
            # Fallback to original path if not found
            if not os.path.exists(png_path):
                fallback_png_path = os.path.join(temp_dir, f"segment_{i}.png")
                if os.path.exists(fallback_png_path):
                    logger.warning(f"Using fallback image path: {fallback_png_path}")
                    png_path = fallback_png_path
                else:
                    logger.error(f"Image not found at either {png_path} or {fallback_png_path}")
                    continue
            
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
        
        # Log detailed information about the created file
        if os.path.exists(output_path):
            file_size_bytes = os.path.getsize(output_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Get final duration with ffprobe
            duration_cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                output_path
            ]
            try:
                duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
                if duration_result.returncode == 0:
                    actual_duration = float(duration_result.stdout.strip())
                    logger.info(f"✅ Final video successfully created: {output_path}")
                    logger.info(f"   - File size: {file_size_mb:.2f} MB ({file_size_bytes} bytes)")
                    logger.info(f"   - Duration: {actual_duration:.2f} seconds ({actual_duration/60:.2f} minutes)")
                else:
                    logger.info(f"✅ Final video created: {output_path}")
                    logger.info(f"   - File size: {file_size_mb:.2f} MB ({file_size_bytes} bytes)")
                    logger.warning(f"   - Could not verify duration: {duration_result.stderr}")
            except Exception as e:
                logger.info(f"✅ Final video created: {output_path}")
                logger.info(f"   - File size: {file_size_mb:.2f} MB ({file_size_bytes} bytes)")
                logger.warning(f"   - Error verifying duration: {str(e)}")
        else:
            logger.error(f"❌ Failed to create final video: {output_path} does not exist after processing")
        
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
SEGMENT_MIN_DURATION = 25    # Minimum segment duration
SEGMENT_MAX_DURATION = 50    # Maximum segment duration
SEGMENT_TARGET_DURATION =35 # Ideal segment duration

def normalize_segments(segments, min_duration=12, max_duration=35, target_duration=25, aggressive_reduction=True):
    """Normalize segment durations by combining or splitting segments with improved topic coherence."""
    if not segments:
        return []

    # Phase 1: Initial pass to combine obviously related segments
    initial_combined = []
    current = None

    for segment in segments:
        if not current:
            current = segment.copy()
            continue

        # Calculate coherence between current and next segment
        coherence_score = calculate_text_coherence(current['text'], segment['text'])
        combined_duration = current['end'] - current['start'] + segment['end'] - segment['start']
        
        # More lenient combination criteria when aggressive_reduction is True
        coherence_threshold = 0.5 if aggressive_reduction else 0.7
        max_combined_duration = max_duration * 1.2 if aggressive_reduction else max_duration
        
        if combined_duration <= max_combined_duration and coherence_score > coherence_threshold:
            # Combine segments
            current['end'] = segment['end']
            current['text'] += ' ' + segment['text']
        else:
            initial_combined.append(current)
            current = segment.copy()

    # Don't forget the last segment
    if current:
        initial_combined.append(current)
    
    # Phase 2: Topic-based clustering to further reduce segments
    if aggressive_reduction and len(initial_combined) > 1:
        topic_clustered = []
        segment_group = [initial_combined[0]]
        current_topic = extract_topic(initial_combined[0]['text'])
        
        for segment in initial_combined[1:]:
            segment_topic = extract_topic(segment['text'])
            segment_duration = segment['end'] - segment['start']
            group_duration = sum(s['end'] - s['start'] for s in segment_group)
            
            # Check if this segment continues the same topic and keeping it won't make the group too long
            if is_same_topic(current_topic, segment_topic) and (group_duration + segment_duration) <= max_duration * 1.5:
                segment_group.append(segment)
            else:
                # Merge the current group and start a new one
                merged_segment = merge_segment_group(segment_group)
                topic_clustered.append(merged_segment)
                segment_group = [segment]
                current_topic = segment_topic
        
        # Add the last group
        if segment_group:
            merged_segment = merge_segment_group(segment_group)
            topic_clustered.append(merged_segment)
        
        normalized = topic_clustered
    else:
        normalized = initial_combined

    # Final pass to adjust segment durations closer to target
    for segment in normalized:
        duration = segment['end'] - segment['start']
        if duration < min_duration:
            # Try to extend short segments
            extension = min(min_duration - duration, 2.0)  # Max 2 second extension
            segment['end'] += extension
        elif duration > max_duration * 1.5:  # More aggressive splitting for very long segments
            # Consider splitting extremely long segments
            midpoint = segment['start'] + duration / 2
            text_parts = split_text_at_sentence_boundary(segment['text'])
            
            if len(text_parts) > 1:
                # Create two segments from this one
                segment1 = segment.copy()
                segment1['end'] = midpoint
                segment1['text'] = text_parts[0]
                
                segment2 = segment.copy()
                segment2['start'] = midpoint
                segment2['text'] = text_parts[1]
                
                # Replace the current segment with these two
                # Note: This would require restructuring the loop to handle this replacement
                # For simplicity, just log that splitting would be beneficial
                logger.info(f"Segment at {segment['start']} is very long ({duration:.2f}s). Consider splitting.")

    # Log reduction statistics
    logger.info(f"Segment reduction: {len(segments)} → {len(normalized)} ({((len(segments) - len(normalized)) / len(segments) * 100):.1f}% reduction)")
    return normalized

def extract_topic(text):
    """Extract the main topic from text using keyword frequency and importance."""
    # Simple implementation: just return the most common substantive words
    # In a full implementation, you might use NLP techniques like TF-IDF or entity extraction
    words = text.lower().split()
    # Filter out common stop words
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
    content_words = [word for word in words if word not in stop_words]
    
    # Count word frequency
    word_counts = {}
    for word in content_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Return the top 3 most frequent words as the "topic"
    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    return [word for word, _ in top_words]

def is_same_topic(topic1, topic2):
    """Determine if two topics are sufficiently similar."""
    # Check for overlap in topic keywords
    common_words = set(topic1).intersection(set(topic2))
    return len(common_words) >= 1  # If they share at least one key word

def merge_segment_group(segment_group):
    """Merge a group of segments into a single segment."""
    if not segment_group:
        return None
    
    merged = segment_group[0].copy()
    merged['text'] = ' '.join(s['text'] for s in segment_group)
    merged['end'] = segment_group[-1]['end']
    return merged

def split_text_at_sentence_boundary(text):
    """Split text into two parts at a sentence boundary near the middle."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:
        # If there's only one sentence, split it in half
        midpoint = len(text) // 2
        return [text[:midpoint], text[midpoint:]]
    
    # Find the middle sentence
    mid_idx = len(sentences) // 2
    part1 = ' '.join(sentences[:mid_idx])
    part2 = ' '.join(sentences[mid_idx:])
    return [part1, part2]

def calculate_text_coherence(text1, text2):
    """Calculate how well two text segments fit together semantically.
    Returns a score between 0 and 1, where 1 indicates high coherence."""
    try:
        # Get the last sentence of text1 and first sentence of text2
        text1_sentences = re.split(r'(?<=[.!?])\s+', text1.strip())
        text2_sentences = re.split(r'(?<=[.!?])\s+', text2.strip())
        
        last_sentence = text1_sentences[-1] if text1_sentences else ""
        first_sentence = text2_sentences[0] if text2_sentences else ""
        
        # Check for incomplete sentence in text1
        if not any(text1.strip().endswith(end) for end in ['.', '!', '?']):
            return 0.9  # High coherence if text1 ends mid-sentence
        
        # Combined text for analysis
        combined = f"{last_sentence} {first_sentence}".lower()
        
        # Check for transitional phrases that indicate connected content
        transition_phrases = ['however', 'therefore', 'furthermore', 'moreover', 'in addition', 
                            'consequently', 'as a result', 'for example', 'for instance',
                            'similarly', 'likewise', 'in contrast', 'on the other hand']
        
        for phrase in transition_phrases:
            if phrase in first_sentence.lower():
                return 0.85  # High coherence for explicit transitions
        
        # Check for pronoun references that likely refer to previous content
        pronoun_start = any(first_sentence.lower().startswith(p) for p in ['he ', 'she ', 'they ', 'it ', 'this ', 'that ', 'these ', 'those '])
        if pronoun_start:
            return 0.8  # High coherence for pronoun reference
            
        # Check for conjunctions at the start of text2
        if any(first_sentence.lower().startswith(c) for c in ['and ', 'but ', 'or ', 'so ', 'because ']):
            return 0.75  # Good coherence for conjunction starts
            
        # Check for thematic continuity
        text1_keywords = set(extract_topic(last_sentence))
        text2_keywords = set(extract_topic(first_sentence))
        keyword_overlap = len(text1_keywords.intersection(text2_keywords))
        
        if keyword_overlap > 0:
            return 0.65 + (0.1 * keyword_overlap)  # 0.65-0.95 based on keyword overlap
            
        # Default moderate coherence
        return 0.4  # Lower default to be more selective
        
    except Exception as e:
        logger.warning(f"Error in coherence calculation: {str(e)}")
        return 0.3  # Lower default on error

def write_segments_to_srt(segments, output_path):
    """Write segments to an SRT file"""
    try:
        # Validate segments
        for i, segment in enumerate(segments, 1):
            # Check for required fields
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

def convert_srt_to_vtt(srt_content):
    """Convert SRT format to VTT format"""
    # Add VTT header
    vtt_content = "WEBVTT\n\n"
    
    # Split by double newline to get segments
    segments = srt_content.strip().split("\n\n")
    
    for segment in segments:
        lines = segment.strip().split("\n")
        
        # Skip empty segments
        if len(lines) < 3:
            continue
        
        # Keep segment number (optional in VTT)
        segment_number = lines[0]
        
        # Convert timestamp format from 00:00:00,000 to 00:00:00.000
        timestamp = lines[1].replace(',', '.')
        
        # Get text content (could be multiple lines)
        text = "\n".join(lines[2:])
        
        # Write the segment to VTT
        vtt_content += segment_number + "\n"
        vtt_content += timestamp + "\n"
        vtt_content += text + "\n\n"
    
    return vtt_content

def transcribe_audio(audio_path, force_transcription=False, transcript_dir="transcripts", non_interactive=False, temp_dir="temp", limit_to_one_minute=False):
    """Transcribe audio using local SpeechRecognition with pocketsphinx"""
    debug_point("Starting audio transcription with local SpeechRecognition")
    
    # Make sure directories exist
    os.makedirs(transcript_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate a unique identifier for this audio file
    audio_hash = hashlib.md5(open(audio_path, 'rb').read()).hexdigest()
    
    # Prepare file paths
    srt_path = os.path.join(transcript_dir, f"{audio_hash}.srt")
    vtt_path = os.path.join(transcript_dir, f"{audio_hash}.vtt")
    
    # If we already have a transcription and force_transcription is False, use it
    if os.path.exists(srt_path) and not force_transcription:
        debug_point(f"Using existing transcription: {srt_path}")
        return srt_path
    
    # Check if SpeechRecognition and pocketsphinx are available
    if not SPEECH_RECOGNITION_AVAILABLE or not POCKETSPHINX_AVAILABLE:
        logger.error("SpeechRecognition or pocketsphinx libraries not available")
        return None
    
    # Preprocess audio if needed
    temp_audio_path = os.path.join(temp_dir, f"temp_{audio_hash}.wav")
    
    if limit_to_one_minute:
        debug_point(f"Limiting audio to {TIME_LIMIT_SECONDS} seconds ({TIME_LIMIT_SECONDS/60:.1f} minutes) for testing")
        # Convert to WAV and limit using ffmpeg
        command = [
            "ffmpeg", "-y", "-i", audio_path, 
            "-t", str(TIME_LIMIT_SECONDS), "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            temp_audio_path
        ]
        logger.info(f"Using time limit of {TIME_LIMIT_SECONDS} seconds from global constant")
    else:
        # Convert to WAV format suitable for SpeechRecognition
        command = [
            "ffmpeg", "-y", "-i", audio_path, 
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            temp_audio_path
        ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        debug_point(f"Preprocessed audio saved to {temp_audio_path}")
        
        # Get audio duration for segmentation
        audio_info = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", 
             "default=noprint_wrappers=1:nokey=1", temp_audio_path],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            universal_newlines=True
        )
        audio_duration = float(audio_info.stdout.strip())
        
        # Initialize SpeechRecognition recognizer
        recognizer = sr.Recognizer()
        
        # Adjust for ambient noise if the audio duration is sufficient
        logger.info(f"Audio duration: {audio_duration} seconds")
        
        # Create segments from the audio file
        # We'll split audio into segments for better recognition
        logger.info("Splitting audio into segments for transcription")
        segment_length = 30  # seconds per segment, pocketsphinx works better with shorter segments
        
        segments = []
        
        # Function to transcribe a segment of audio
        def transcribe_segment(start_time, end_time, segment_idx):
            debug_point(f"Transcribing segment {segment_idx+1}: {start_time:.2f}s - {end_time:.2f}s")
            
            # Create a temporary file for this segment
            segment_path = os.path.join(temp_dir, f"segment_{audio_hash}_{segment_idx}.wav")
            
            # Extract segment using ffmpeg
            segment_cmd = [
                "ffmpeg", "-y", "-i", temp_audio_path,
                "-ss", str(start_time), "-to", str(end_time),
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                segment_path
            ]
            
            try:
                subprocess.run(segment_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Use SpeechRecognition with pocketsphinx
                with sr.AudioFile(segment_path) as source:
                    # Adjust for ambient noise with a short duration
                    recognizer.adjust_for_ambient_noise(source, duration=min(1.0, end_time-start_time))
                    audio_data = recognizer.record(source)
                    
                    try:
                        # Use pocketsphinx for offline recognition
                        text = recognizer.recognize_sphinx(audio_data)
                        logger.info(f"Segment {segment_idx+1} recognized: {text[:30]}...")
                        
                        # Create segment with timestamps
                        return {
                            "index": segment_idx + 1,
                            "start": start_time,
                            "end": end_time,
                            "text": text,
                            "timestamp": {
                                "start": format_timestamp(start_time),
                                "end": format_timestamp(end_time)
                            }
                        }
                    except sr.UnknownValueError:
                        logger.warning(f"Sphinx could not understand audio in segment {segment_idx+1}")
                        return {
                            "index": segment_idx + 1,
                            "start": start_time,
                            "end": end_time,
                            "text": "[inaudible]",
                            "timestamp": {
                                "start": format_timestamp(start_time),
                                "end": format_timestamp(end_time)
                            }
                        }
                    except sr.RequestError as e:
                        logger.error(f"Sphinx error in segment {segment_idx+1}; {e}")
                        return None
                    finally:
                        # Remove temporary segment file
                        if os.path.exists(segment_path):
                            os.remove(segment_path)
            except Exception as e:
                logger.error(f"Error processing segment {segment_idx+1}: {e}")
                return None
        
        # Determine how many segments to create
        num_segments = max(1, int(audio_duration / segment_length))
        logger.info(f"Splitting audio into {num_segments} segments of {segment_length}s each")
        
        # Create a list of start/end times for each segment
        segment_times = []
        for i in range(num_segments):
            start = i * segment_length
            end = min((i + 1) * segment_length, audio_duration)
            segment_times.append((start, end, i))
        
        # Process segments in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
            futures = []
            for start, end, idx in segment_times:
                future = executor.submit(transcribe_segment, start, end, idx)
                futures.append(future)
            
            # Wait for all futures to complete and collect results
            for future in futures:
                segment = future.result()
                if segment:
                    segments.append(segment)
        
        # Sort segments by start time
        segments.sort(key=lambda x: x["start"])
        
        # Convert segments to SRT format
        debug_point("Converting transcription to SRT format")
        srt_content = ""
        
        for i, segment in enumerate(segments):
            srt_content += f"{i+1}\n"
            srt_content += f"{segment['timestamp']['start']} --> {segment['timestamp']['end']}\n"
            srt_content += f"{segment['text']}\n\n"
        
        # Track API cost if cost tracker is available (now using local transcription)
        if COST_TRACKER_AVAILABLE:
            cost_tracker.add_local_transcription_cost(
                duration_seconds=audio_duration,
                engine="pocketsphinx",
                operation_name="local_transcription"
            )
        
        # Save SRT transcription
        with open(srt_path, "w") as f:
            f.write(srt_content)
        
        # Convert to VTT format
        vtt_content = convert_srt_to_vtt(srt_content)
        with open(vtt_path, "w") as f:
            f.write(vtt_content)
        
        debug_point("Transcription completed")
        
        # Remove temporary audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        return srt_path
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        traceback.print_exc()
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
                    if current_segment and 'text' in current_segment and 'timestamp' in current_segment:
                        # Convert timestamp strings to seconds before adding the segment
                        start_str = current_segment['timestamp']['start']
                        end_str = current_segment['timestamp']['end']
                        
                        # Parse timestamps to seconds
                        start_seconds = parse_timestamp(start_str)
                        end_seconds = parse_timestamp(end_str)
                        
                        # Add start and end times in seconds for processing
                        current_segment['start'] = start_seconds
                        current_segment['end'] = end_seconds
                        current_segment['start_time'] = start_seconds  # Add both naming conventions
                        current_segment['end_time'] = end_seconds      # Add both naming conventions
                        
                        segments.append(current_segment)
                        current_segment = None
                    elif current_segment:
                        # Skip incomplete segments
                        logger.warning(f"Skipping incomplete segment: {current_segment}")
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
        if current_segment and 'text' in current_segment and 'timestamp' in current_segment:
            # Convert timestamp strings to seconds
            start_str = current_segment['timestamp']['start']
            end_str = current_segment['timestamp']['end']
            
            # Parse timestamps to seconds
            start_seconds = parse_timestamp(start_str)
            end_seconds = parse_timestamp(end_str)
            
            # Add start and end times in seconds for processing
            current_segment['start'] = start_seconds
            current_segment['end'] = end_seconds
            current_segment['start_time'] = start_seconds  # Add both naming conventions
            current_segment['end_time'] = end_seconds      # Add both naming conventions
            
            segments.append(current_segment)
        
        # Log extracted segments (summary only, not each one)
        logger.info(f"Extracted {len(segments)} segments from transcript")
        
        # Check segments and calculate stats
        total_duration = 0
        for i, segment in enumerate(segments):
            duration = segment['end'] - segment['start']
            total_duration += duration
            
            # Only log unusually long segments as a warning
            if duration > 20:
                logger.info(f"Segment {i+1} is unusually long: {duration:.2f}s")
        
        logger.info(f"Total segments duration: {total_duration:.2f} seconds")
        
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
    debug_point("Starting segment enhancement process")
    
    # Use global variables for API type and model
    global api_type, chat_model
    
    # Get custom enhance prompt from environment if available
    custom_enhance_prompt = os.environ.get("CUSTOM_ENHANCE_PROMPT")
    if custom_enhance_prompt:
        logger.info(f"Using custom enhance prompt from environment variable: {custom_enhance_prompt[:100]}...")
    else:
        logger.info("Using default enhance prompt")
    
    # Initialize segment cache if not already initialized
    if not hasattr(enhance_segments, "segment_cache"):
        enhance_segments.segment_cache = {}
    
    # Check if we have a valid API type and model
    if not api_type or not chat_model:
        logger.error("No valid API type or chat model available")
        return None
    
    # Log which API we're using
    logger.info(f"Using {api_type.upper()} API with model {chat_model} for enhancements")
    
    # Default prompt template for segment enhancement
    default_enhance_prompt = """You are a helpful assistant that analyzes podcast segments and provides structured descriptions and visual prompts.
Your task is to analyze this podcast segment and provide a JSON response with the following structure:

{
    "description": "A clear, concise description of the main topic",
    "visual_prompt": "In [EPOCH_TIME], a detailed scene showing...",
    "key_points": ["Key point 1", "Key point 2", "Key point 3", "Key point N"],
    "epoch_time": "A description of the period in time and approximate location down to the continent level to help set the period for context and will be used by downstream processors."
}

The visual prompt should be detailed and specific, focusing on visual elements that would make a compelling image.
Include details like style, composition, colors, and elements to include in the image. 

IMPORTANT: The visual_prompt MUST start with "In [EPOCH_TIME]," where [EPOCH_TIME] is replaced with the actual time period relevant to the content. For example, "In ancient Greece, 5th century BCE," or "In 1960s America,". This ensures the generated image properly represents the correct historical context.

Be sure to include a time period for which the image should be set, so it is logically aligned with the topic and the period in time that the topic is set in as appropriate and possible to discern from this content. If there are hints you can provide that further sets the tone and mood of the setting that helps the image creation be true to the intention of the discussion so it feels natural to the viewer."""
    
    # Use custom enhance prompt if available, otherwise use default
    enhance_prompt = custom_enhance_prompt if custom_enhance_prompt else default_enhance_prompt
    logger.info(f"Enhance prompt length: {len(enhance_prompt)} characters")
    
    # Client for API - Use the existing global client
    try:
        # Get the global client based on API type
        if api_type.lower() == "openai":
            from openai import OpenAI
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                logger.error("OPENAI_API_KEY environment variable not set")
                return None
            client = OpenAI(api_key=openai_api_key)
        elif api_type.lower() == "deepseek":
            from openai import OpenAI
            deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not deepseek_api_key:
                logger.error("DEEPSEEK_API_KEY environment variable not set")
                return None
            client = OpenAI(base_url="https://api.deepseek.com/v1", api_key=deepseek_api_key)
        else:
            logger.error(f"Unknown API type: {api_type}")
            return None
    except ImportError:
        logger.error("Failed to import OpenAI client")
        return None
    
    # Enhanced segments to return
    enhanced_segments = []
    
    # Get how many segments to process
    if limit_to_one_minute:
        segments_to_process = []
        total_duration = 0
        
        for segment in segments:
            segment_duration = segment["end"] - segment["start"]
            if total_duration + segment_duration <= TIME_LIMIT_SECONDS:
                segments_to_process.append(segment)
                total_duration += segment_duration
            else:
                break
        
        # Use the limited segments
        logger.info(f"Time limit mode: Processing {len(segments_to_process)} of {len(segments)} segments (time limit: {TIME_LIMIT_SECONDS} seconds)")
        segments = segments_to_process
    
    # Process each segment
    for i, segment in enumerate(segments, 1):
        try:
            logger.info(f"Enhancing segment {i}/{len(segments)}")
            logger.info(f"Segment text: {segment['text']}")
            
            # Compute cache key for current segment
            cache_key = get_cache_key(segment['text'])
            
            # Check if segment is in cache
            if cache_key in enhance_segments.segment_cache:
                logger.info("Found segment in cache, using cached enhancement")
                enhanced_segment = enhance_segments.segment_cache[cache_key].copy()
                # Update start and end times to match the current segment
                enhanced_segment['start'] = segment['start']
                enhanced_segment['end'] = segment['end']
                enhanced_segments.append(enhanced_segment)
                continue
            
            # Check if we should cancel
            if check_cancel():
                logger.info("Enhancement cancelled by user")
                return None
            
            # Format prompt for this segment
            prompt_text = f"{enhance_prompt}\n\nHere is the podcast segment to analyze:\n\"{segment['text']}\""
            
            # Get enhancement from the appropriate API
            logger.info(f"Requesting enhancement from {api_type.upper()} for segment {i}")
            completion_result = with_timeout(
                func=client.chat.completions.create,
                kwargs={
                    "model": chat_model,  # Use the global chat model instead of hardcoding
                    "messages": [{"role": "user", "content": prompt_text}],
                    "temperature": 0.2,
                    "timeout": 60  # 60 second timeout
                },
                timeout_seconds=70,  # 70 second timeout (with buffer)
                description=f"{api_type.upper()} enhancement for segment {i}"
            )
            
            if not completion_result:
                logger.error(f"Failed to enhance segment {i}: Timeout")
                return None
            
            # Extract the actual completion object from the result
            # If it's a tuple, the first element should be the API response
            if isinstance(completion_result, tuple):
                completion = completion_result[0]
            else:
                completion = completion_result
            
            # Parse the response
            response_text = completion.choices[0].message.content
            
            # Try to extract JSON
            try:
                # Extract JSON from response
                json_match = re.search(r'({[\s\S]*})', response_text)
                if not json_match:
                    logger.error(f"Failed to extract JSON from response for segment {i}")
                    logger.error(f"Response: {response_text}")
                    return None
                
                json_text = json_match.group(1)
                enhancement = json.loads(json_text)
                
                # Log the extracted enhancement
                logger.info(f"Successfully enhanced segment {i}")
                logger.info(f"Description: {enhancement.get('description', '')[:100]}...")
                logger.info(f"Epoch time: {enhancement.get('epoch_time', '')}")
                
                # Check visual prompt structure
                visual_prompt = enhancement.get('visual_prompt', '')
                logger.info(f"Original visual prompt: {visual_prompt[:100]}...")
                
                # Verify the visual_prompt format (should start with "In [EPOCH_TIME],")
                if not visual_prompt.startswith("In "):
                    logger.warning(f"Visual prompt may not follow required format (should start with 'In [EPOCH_TIME],')")
                    epoch_time = enhancement.get('epoch_time', 'modern times')
                    # Try to fix it if possible
                    if "," in visual_prompt:
                        # Try to preserve text after first comma
                        fixed_prompt = f"In {epoch_time}, {visual_prompt.split(',', 1)[1].strip()}"
                    else:
                        # Just prepend the epoch time
                        fixed_prompt = f"In {epoch_time}, {visual_prompt}"
                    
                    logger.info(f"Fixed visual prompt: {fixed_prompt[:100]}...")
                    enhancement['visual_prompt'] = fixed_prompt
                
                # Create the enhanced segment
                enhanced_segment = {
                    'text': segment['text'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'description': enhancement.get('description', ''),
                    'visual_prompt': enhancement.get('visual_prompt', ''),
                    'key_points': enhancement.get('key_points', []),
                    'epoch_time': enhancement.get('epoch_time', '')
                }
                
                # Add to cache
                enhance_segments.segment_cache[cache_key] = enhanced_segment.copy()
                
                # Add to results
                enhanced_segments.append(enhanced_segment)
                logger.info(f"Enhanced segment {i} with {len(enhanced_segment['key_points'])} key points")
                
                # Track API cost if cost tracker is available
                if COST_TRACKER_AVAILABLE:
                    completion_tokens = completion.usage.completion_tokens
                    prompt_tokens = completion.usage.prompt_tokens
                    
                    # Add the cost
                    cost_tracker.add_openai_chat_cost(
                        model=chat_model,  # Use the global chat model instead of hardcoding
                        input_tokens=prompt_tokens,
                        output_tokens=completion_tokens,
                        operation_name=f"enhance_segment_{i}"
                    )
                
                # Pause briefly to avoid rate limits
                time.sleep(0.5)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from response for segment {i}: {e}")
                logger.error(f"Response: {response_text}")
                return None
                
        except Exception as e:
            logger.error(f"Error enhancing segment {i}: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error("Segment enhancement failed")
            return None
    
    logger.info(f"Successfully enhanced {len(enhanced_segments)} segments")
    return enhanced_segments

def generate_visuals(enhanced_segments, output_dir, non_interactive=False):
    """Generate visuals for each enhanced segment"""
    logger.info("Generating visuals for segments")
    visuals = []
    
    # Get custom visual generation prompt prefix from environment
    custom_visual_prefix = os.environ.get("CUSTOM_VISUAL_PREFIX")
    
    # Get custom negative prompts from environment
    custom_negative_prompts = os.environ.get("CUSTOM_NEGATIVE_PROMPTS")
    if custom_negative_prompts:
        logger.info(f"Found custom negative prompts from environment: {custom_negative_prompts[:100]}...")
        logger.info(f"Custom negative prompts length: {len(custom_negative_prompts)} characters")
        logger.info(f"Number of items in custom negative prompts: {len([x for x in custom_negative_prompts.split(',') if x.strip()])}")
    else:
        logger.warning("No custom negative prompts found in environment variables. Using only auto-generated negative prompts.")
    
    # Default visual generation prompt prefix if no custom prefix is provided
    default_visual_prefix = "A high quality, detailed image with creative and artistic imagery that accurately reflects the specific time period described below, using abstract visual elements where appropriate and avoiding human figures or faces, ensuring the visual elements align naturally with the narrative context of the following scene: "
    
    for i, segment in enumerate(enhanced_segments, 1):
        logger.info(f"Generating visuals for segment {i}/{len(enhanced_segments)}")
        try:
            # Get the epoch time for negative prompts
            epoch_time = segment.get('epoch_time', '')
            
            # Create negative prompts based on epoch time to avoid anachronisms
            negative_prompts = []
            
            # Log the original visual prompt
            logger.info(f"Original visual prompt: {segment['visual_prompt']}")
            
            # Combine the visual generation prefix with the segment's visual prompt
            visual_prefix = custom_visual_prefix if custom_visual_prefix else default_visual_prefix
            combined_prompt = f"{visual_prefix} {segment['visual_prompt']}"
            
            # Debug the combined prompt
            logger.info(f"Combined prompt: {combined_prompt}")
            
            # Generate image for the segment
            image_path = os.path.join(output_dir, f"segment_{i}.png")
            
            # Add anachronism avoidance negative prompts based on epoch time
            if epoch_time:
                # Determine what types of modern elements to avoid based on the epoch time
                modern_items = ["modern skyscrapers", "cars", "smartphones", "computers", "contemporary clothing"]
                
                # Simple logic to check for pre-modern time periods
                is_ancient = any(term in epoch_time.lower() for term in ["ancient", "medieval", "prehistoric", "bronze age", "iron age", "stone age", "bc", "bce", "century"])
                is_pre_20th_century = any(term in epoch_time.lower() for term in ["1800s", "1700s", "1600s", "1500s", "1400s", "1300s", "1200s", "1100s", "1000s", "century", "renaissance", "victorian", "tudor", "colonial"])
                
                # Generate negative prompts based on the time period
                if is_ancient:
                    negative_prompts.extend(modern_items)
                    negative_prompts.extend(["electricity", "modern buildings", "modern technology"])
                    logger.info(f"Added ancient-specific negative prompts for epoch: {epoch_time}")
                
                elif is_pre_20th_century:
                    negative_prompts.extend(["skyscrapers", "cars", "smartphones", "computers", "modern technology"])
                    logger.info(f"Added pre-20th century negative prompts for epoch: {epoch_time}")
            
            # Start with custom negative prompts if available
            negative_prompt_str = ""
            if custom_negative_prompts:
                # Use the custom negative prompts and add any anachronism-specific ones
                negative_prompt_str = custom_negative_prompts
                logger.info(f"Baseline negative prompts (from environment): {negative_prompt_str[:100]}...")
                
                # Add anachronism prompts if they exist
                if negative_prompts:
                    logger.info(f"Adding {len(negative_prompts)} anachronism-specific negative prompts: {', '.join(negative_prompts)}")
                    negative_prompt_str += ", " + ", ".join(negative_prompts)
                
                logger.info(f"FINAL negative prompts (custom + anachronism): {negative_prompt_str[:150]}...")
                logger.info(f"Total negative prompt length: {len(negative_prompt_str)} characters")
                logger.info(f"Total negative prompt items: {len([x for x in negative_prompt_str.split(',') if x.strip()])}")
            else:
                # No custom prompts, make sure we have strong defaults to avoid inappropriate images
                default_negative_prompts = ["humans", "people", "person", "man", "woman", "child", "face", "faces", "portrait", "photorealistic humans", "realistic human faces", "detailed faces", "nude", "naked", "nsfw", "disturbing content"]
                negative_prompts.extend(default_negative_prompts)
                negative_prompt_str = ", ".join(negative_prompts)
                logger.info(f"Using default negative prompts since no custom prompts found: {negative_prompt_str}")
                logger.info(f"Negative prompt items count: {len(negative_prompts)}")
            
            # Validate the prompt before sending to Stability
            logger.info("Validating prompt before sending to Stability API")
            is_safe, validated_prompt, validated_negative_prompt, reason = validate_image_prompt_with_deepseek(
                combined_prompt, 
                negative_prompt_str
            )
            
            if not is_safe:
                logger.warning(f"Prompt validation detected potential issues: {reason}")
                logger.info("Using corrected prompts from validation")
                
                # Log the differences
                if validated_prompt != combined_prompt:
                    logger.info(f"Original prompt: {combined_prompt}")
                    logger.info(f"Corrected prompt: {validated_prompt}")
                
                if negative_prompt_str and validated_negative_prompt != negative_prompt_str:
                    logger.info(f"Original negative prompt: {negative_prompt_str}")
                    logger.info(f"Corrected negative prompt: {validated_negative_prompt}")
            else:
                logger.info(f"Prompt validation passed: {reason}")
            
            # Generate the image with validated prompts
            success = generate_image_stability(
                prompt=validated_prompt,
                output_path=image_path,
                non_interactive=non_interactive,
                negative_prompt=validated_negative_prompt
            )
            
            if success:
                # Create a visual entry with the correct field access
                # Check if start/end or start_time/end_time are available
                start_time = segment.get('start_time', segment.get('start', 0))
                end_time = segment.get('end_time', segment.get('end', 0))
                
                visuals.append({
                    'text': segment['text'],
                    'image_path': image_path,
                    'start_time': start_time,
                    'end_time': end_time
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
    """Process audio file to create a video with AI-generated visuals"""
    debug_point(f"Processing audio file: {audio_path}")
    
    # Show time limit information if enabled
    if limit_to_one_minute:
        logger.info(f"Time limit mode is enabled. Using TIME_LIMIT_SECONDS={TIME_LIMIT_SECONDS} ({TIME_LIMIT_SECONDS/60:.1f} minutes)")
    else:
        logger.info("Time limit mode is disabled. Processing full audio.")
    
    # Check that the audio file exists
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temp directory
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize global state
    debug_point("Initializing processing state")
    global current_operation, is_cancelling, task_start_time, task_name, task_stack
    current_operation = "initialized"
    is_cancelling = False
    task_start_time = time.time()
    task_name = "initialization"
    task_stack = [("initialization", task_start_time)]
    
    # Reset cost tracker at beginning of process if available
    if COST_TRACKER_AVAILABLE:
        cost_tracker.reset()
        logger.info("Cost tracker reset for new processing task")
    
    # Start progress monitoring thread
    debug_point("Starting progress monitoring")
    global progress_thread
    progress_thread = start_progress_monitoring()
    
    # Set up signal handlers for graceful shutdown
    setup_signal_handling()
    
    # Track processing start time
    processing_start = time.time()
    
    try:
        # Step 1: Transcribe the audio
        debug_point("Starting audio transcription")
        transcript_path = transcribe_audio(
            audio_path=audio_path,
            force_transcription=False,
            transcript_dir="transcripts",
            non_interactive=non_interactive,
            temp_dir=temp_dir, 
            limit_to_one_minute=limit_to_one_minute
        )
        
        if not transcript_path:
            logger.error("Transcription failed")
            return False
        
        # Extract segments from transcript
        debug_point("Extracting segments from transcript")
        segments = extract_segments(transcript_path)
        
        if not segments:
            logger.error("No segments extracted from transcript")
            return False
        
        logger.info(f"Extracted {len(segments)} segments from transcript")
        
        # Normalize segments to reduce their number and optimize processing
        debug_point("Normalizing segments")
        original_segment_count = len(segments)
        segments = normalize_segments(segments, min_duration=12, max_duration=35, target_duration=25)
        logger.info(f"Normalized segments: {len(segments)} segments (from original {original_segment_count})")
        
        # Step 2: Enhance segments with AI descriptions and visual prompts
        debug_point("Enhancing segments with AI")
        enhanced_segments = enhance_segments(segments, limit_to_one_minute)
        
        if not enhanced_segments:
            logger.error("No enhanced segments produced")
            return False
        
        logger.info(f"Enhanced {len(enhanced_segments)} segments")
        
        # Step 3: Generate visuals for each segment
        debug_point("Generating visuals")
        image_folder = os.path.join(temp_dir, "images")
        os.makedirs(image_folder, exist_ok=True)
        
        enhanced_segments = generate_visuals(enhanced_segments, image_folder, non_interactive)
        
        if not enhanced_segments:
            logger.error("Failed to generate visuals")
            return False
        
        logger.info(f"Generated visuals for {len(enhanced_segments)} segments")
        
        # Step 4: Create the final video
        debug_point("Creating final video")
        result = create_final_video(
            enhanced_segments=enhanced_segments,
            audio_path=audio_path,
            output_path=output_path,
            temp_dir=temp_dir,
            limit_to_one_minute=limit_to_one_minute
        )
        
        # Calculate processing time
        processing_time = time.time() - processing_start
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        
        # Save cost report if cost tracking is enabled
        if COST_TRACKER_AVAILABLE:
            # Create a report path in the same directory as the output video
            report_dir = os.path.dirname(output_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(report_dir, f"api_costs_{timestamp}.json")
            
            # Save the report
            cost_tracker.save_report(report_path)
            
            # Get the cost summary
            cost_summary = cost_tracker.get_summary()
            
            # Log the total cost
            logger.info(f"Total API cost: ${cost_summary['total_cost']:.4f}")
            logger.info(f"Cost breakdown: OpenAI Chat ${cost_summary['api_breakdown']['openai']['chat']:.4f}, " 
                       f"OpenAI Transcription ${cost_summary['api_breakdown']['openai']['transcription']:.4f}, "
                       f"Stability Image ${cost_summary['api_breakdown']['stability']['image']:.4f}")
            logger.info(f"Cost report saved to {report_path}")
            
            # Print cost data in special format for webapp to parse
            cost_json = json.dumps(cost_summary)
            print(f"COST_DATA: {cost_json}")
        
        complete_task("Creating final video")
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        traceback.print_exc()
        return False
    finally:
        # Output final cost data before finishing
        if COST_TRACKER_AVAILABLE:
            try:
                cost_summary = cost_tracker.get_summary()
                cost_json = json.dumps(cost_summary)
                print(f"COST_DATA: {cost_json}")
            except Exception as e:
                logger.error(f"Error generating final cost report: {str(e)}")
        
        debug_point("Processing completed")

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

def generate_llm_prompt(instructions, llm_client):
    """Generate a prompt using an LLM"""
    debug_point("Generating prompt with LLM")
    
    if not llm_client:
        logger.error("LLM client not provided")
        return None
    
    try:
        # Create the message content
        response = llm_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that generates prompts based on instructions."},
                {"role": "user", "content": instructions}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        # Track API cost if cost tracker is available
        if COST_TRACKER_AVAILABLE:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost_tracker.add_openai_chat_cost(
                model="gpt-4",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                operation_name="generate_prompt"
            )
        
        # Return the generated prompt
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating prompt: {e}")
        return None

def validate_image_prompt_with_deepseek(positive_prompt, negative_prompt=None):
    """
    Use Deepseek to validate that a prompt doesn't inadvertently request human imagery
    and suggest corrections if it does.
    
    Args:
        positive_prompt (str): The positive prompt to check
        negative_prompt (str, optional): The negative prompt to check

    Returns:
        tuple: (is_safe, corrected_positive, corrected_negative, reason)
    """
    logger.info("Validating image prompt with Deepseek")
    
    # Check if we have either a DeepSeek or OpenAI API key
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not deepseek_api_key and not openai_api_key:
        logger.warning("No Deepseek or OpenAI API key found, skipping prompt validation")
        return True, positive_prompt, negative_prompt, "No validation performed"
    
    try:
        # Initialize LLM client - prefer Deepseek if available
        if deepseek_api_key:
            from openai import OpenAI
            llm_client = OpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com/v1"
            )
            model = "deepseek-chat"
            logger.info("Using Deepseek for prompt validation")
        else:
            from openai import OpenAI
            llm_client = OpenAI(api_key=openai_api_key)
            model = "gpt-4"
            logger.info("Using OpenAI for prompt validation (Deepseek not available)")
        
        # Prepare the system prompt for validation
        system_prompt = """You are an expert at analyzing image generation prompts for problematic content.
Your task is to check if the provided prompts might inadvertently lead to generating disturbing human imagery.

Check for:
1. Any phrases requesting realistic humans, faces, or human-like figures
2. Instructions that might result in inappropriate, disturbing or explicit human imagery
3. Contradictions between positive and negative prompts

If you find issues, suggest corrected versions that:
- Remove any requests for human or face depictions
- Keep the core subject matter intact
- Prioritize abstract, conceptual, or scenic imagery instead
- Ensures positive prompts don't contradict negative prompts

Reply in JSON format with the following fields:
{
  "is_safe": boolean,
  "corrected_positive_prompt": "string",
  "corrected_negative_prompt": "string",
  "reason": "string explanation"
}"""

        # Create the user message content
        user_content = f"POSITIVE PROMPT: {positive_prompt}\n\n"
        if negative_prompt:
            user_content += f"NEGATIVE PROMPT: {negative_prompt}\n\n"
        user_content += "Please analyze these prompts and suggest corrections if needed."
        
        # Query the LLM
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # Track API cost if cost tracker is available
        if COST_TRACKER_AVAILABLE:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            if deepseek_api_key:
                # If Deepseek cost tracking is implemented, use that method
                cost_tracker.add_openai_chat_cost(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    operation_name="prompt_validation"
                )
            else:
                cost_tracker.add_openai_chat_cost(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    operation_name="prompt_validation"
                )
        
        # Parse the response
        result_text = response.choices[0].message.content.strip()
        try:
            result = json.loads(result_text)
            
            # Log the validation results
            if result["is_safe"]:
                logger.info("Prompt validation passed: No problematic human imagery detected")
            else:
                logger.warning(f"Prompt validation failed: {result['reason']}")
                logger.info(f"Original positive prompt: {positive_prompt}")
                logger.info(f"Corrected positive prompt: {result['corrected_positive_prompt']}")
                if negative_prompt:
                    logger.info(f"Original negative prompt: {negative_prompt}")
                    logger.info(f"Corrected negative prompt: {result['corrected_negative_prompt']}")
            
            return (
                result["is_safe"],
                result["corrected_positive_prompt"],
                result["corrected_negative_prompt"] if negative_prompt else negative_prompt,
                result["reason"]
            )
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response as JSON: {result_text}")
            return True, positive_prompt, negative_prompt, "Error parsing validation response"
        
    except Exception as e:
        logger.error(f"Error validating prompt: {str(e)}")
        logger.error(traceback.format_exc())
        return True, positive_prompt, negative_prompt, f"Validation error: {str(e)}"

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
        parser.add_argument("--limit_to_one_minute", action="store_true", 
                           help=f'Limit processing to first {TIME_LIMIT_SECONDS} seconds ({TIME_LIMIT_SECONDS/60:.1f} minutes) of audio')
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
