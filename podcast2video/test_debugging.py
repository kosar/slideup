#!/usr/bin/env python3
import os
import sys
import time
import logging
from pathlib import Path
from moviepy.editor import ImageClip, AudioFileClip

# Add progress tracking to see where the app is hanging
def setup_logging():
    """Set up detailed logging to track progress"""
    log_file = "podcast2video_debug.log"
    
    # Configure file logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("podcast2video_debug")
    return logger

def main():
    logger = setup_logging()
    
    # Debug: Print Python path
    logger.info("Python path:")
    for path in sys.path:
        logger.info(f"  {path}")
    
    # Debug: Print MoviePy version and location
    try:
        import moviepy.editor
        logger.info(f"MoviePy version: {moviepy.editor.__version__}")
        logger.info(f"MoviePy location: {moviepy.editor.__file__}")
    except Exception as e:
        logger.error(f"Error importing moviepy: {e}")
        logger.info("MoviePy import error is non-fatal, continuing with tests...")
    
    # Path to the test audio file
    test_audio = "test_resources/test_audio.wav"
    audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_audio)
    
    if not os.path.exists(audio_path):
        logger.error(f"Test audio file not found: {audio_path}")
        sys.exit(1)
    
    logger.info(f"Using test audio file: {audio_path}")
    
    # Import the main module with timeout protection
    try:
        logger.info("Importing podcast-to-video module")
        module_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Module directory: {module_dir}")
        sys.path.append(module_dir)
        
        # Import using the symlink name
        try:
            import podcast_to_video
            logger.info("Successfully imported podcast_to_video")
        except ImportError as e:
            # Try to import directly from the file
            import importlib.util
            logger.info("Trying alternate import method")
            module_path = os.path.join(module_dir, "podcast-to-video.py")
            logger.info(f"Module path: {module_path}")
            spec = importlib.util.spec_from_file_location(
                "podcast_to_video", 
                module_path
            )
            podcast_to_video = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(podcast_to_video)
            logger.info("Successfully imported podcast-to-video.py directly")
        
        # Monkey patch the module to track progress
        original_functions = {}
        
        # Functions to track with their timeouts (in seconds)
        functions_to_track = {
            "transcribe_audio": 120,
            "enhance_segments": 120,
            "generate_visuals": 120,
            "create_final_video": 120,
            "generate_image_stability": 30
        }
        
        # Patch the functions to add logging and timeouts
        for func_name, timeout in functions_to_track.items():
            if hasattr(podcast_to_video, func_name):
                original_func = getattr(podcast_to_video, func_name)
                
                def create_wrapper(f, name, timeout_seconds):
                    def wrapper(*args, **kwargs):
                        logger.info(f"Starting {name}")
                        start_time = time.time()
                        
                        # Update the current operation
                        podcast_to_video.current_operation = f"Running {name}"
                        
                        try:
                            # Set a timer to log if function takes too long
                            result = f(*args, **kwargs)
                            elapsed = time.time() - start_time
                            logger.info(f"Completed {name} in {elapsed:.2f} seconds")
                            return result
                        except Exception as e:
                            logger.error(f"Error in {name}: {e}", exc_info=True)
                            raise
                    return wrapper
                
                # Store the original function
                original_functions[func_name] = original_func
                
                # Replace with wrapped version
                setattr(podcast_to_video, func_name, create_wrapper(original_func, func_name, timeout))
        
        # Run the app with command line arguments
        logger.info("Starting podcast-to-video conversion with test audio")
        
        # Create a parser and parse arguments
        parser = podcast_to_video.main()
        args = parser.parse_args([
            "--audio", audio_path,
            "--output", "test_output.mp4",
            "--temp_dir", "test_temp",
            "--subtitle", "always",
            "--transcript_dir", "test_transcripts",
            "--debug"
        ])
        
        # Process the audio file
        try:
            # Set start time for overall process
            start_time = time.time()
            
            # Create temp and transcript directories
            os.makedirs(args.temp_dir, exist_ok=True)
            os.makedirs(args.transcript_dir, exist_ok=True)
            
            # Validate the audio file
            logger.info("Validating audio file")
            audio_info = podcast_to_video.validate_audio_file(args.audio)
            logger.info(f"Audio validation successful: {audio_info}")
            
            # Transcribe audio
            logger.info("Starting transcription")
            transcript = podcast_to_video.transcribe_audio(
                args.audio, 
                force=True,  # Force transcription for testing
                transcript_dir=args.transcript_dir,
                non_interactive=True,  # Don't ask for confirmation
                temp_dir=args.temp_dir
            )
            
            if transcript is None:
                logger.error("Transcription failed")
                sys.exit(1)
            
            logger.info("Transcription completed successfully")
            
            # Extract meaningful segments
            logger.info("Extracting segments")
            segments = podcast_to_video.extract_segments(transcript)
            logger.info(f"Extracted {len(segments)} segments")
            
            # Enhance segments with AI - this is likely where it might hang
            logger.info("Enhancing segments")
            enhanced_segments = podcast_to_video.enhance_segments(segments)
            logger.info(f"Enhanced {len(enhanced_segments) if enhanced_segments else 0} segments")
            
            # Generate visuals
            logger.info("Generating visuals")
            enhanced_segments = podcast_to_video.generate_visuals(
                enhanced_segments, 
                args.temp_dir,
                allow_dalle_fallback=True,
                non_interactive=True
            )
            
            # Create final video
            logger.info("Creating final video")
            output_path = podcast_to_video.create_final_video(
                enhanced_segments,
                args.audio,
                args.output,
                args.temp_dir
            )
            
            # Report completion
            elapsed_time = time.time() - start_time
            logger.info(f"Process completed in {elapsed_time:.2f} seconds. Output: {output_path}")
            
        except Exception as e:
            logger.error(f"Error during processing: {e}", exc_info=True)
            sys.exit(1)
        
    except ImportError as e:
        logger.error(f"Failed to import podcast-to-video module: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 