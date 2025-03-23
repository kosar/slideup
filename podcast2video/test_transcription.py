#!/usr/bin/env python3
import os
import sys
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transcription_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("transcription_test")

def main():
    # Path to the test audio file
    test_audio = "test_resources/test_audio.wav"
    audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_audio)
    
    if not os.path.exists(audio_path):
        logger.error(f"Test audio file not found: {audio_path}")
        sys.exit(1)
    
    logger.info(f"Using test audio file: {audio_path}")
    
    # Import podcast-to-video module
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # This will import the main module
        from podcast_to_video import transcribe_audio, validate_audio_file
        
        # Create directories for testing
        temp_dir = "test_temp"
        transcript_dir = "test_transcripts"
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(transcript_dir, exist_ok=True)
        
        # Test audio validation
        logger.info("Testing audio validation")
        try:
            audio_info = validate_audio_file(audio_path)
            logger.info(f"Audio validation successful: {audio_info}")
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            sys.exit(1)
        
        # Test transcription with explicit timeout
        logger.info("Testing transcription with timeout")
        start_time = time.time()
        
        # Use a separate thread to monitor and report progress
        def monitor_transcription():
            last_time = time.time()
            while time.time() - start_time < 180:  # 3 minute maximum
                time.sleep(5)
                elapsed = time.time() - start_time
                print(f"Transcription has been running for {elapsed:.1f} seconds")
                
                # Check if it's taking too long
                if elapsed > 120:
                    print("WARNING: Transcription is taking longer than expected (2 minutes)")
                    
                # If 30 seconds passed with no update, it might be hanging
                if time.time() - last_time > 30:
                    print("WARNING: No progress detected for 30 seconds, possible hang")
                    
            print("Transcription monitor timeout reached")
        
        import threading
        monitor_thread = threading.Thread(target=monitor_transcription, daemon=True)
        monitor_thread.start()
        
        try:
            logger.info("Starting transcription test")
            transcript = transcribe_audio(
                audio_path,
                force=True,
                transcript_dir=transcript_dir,
                non_interactive=True,
                temp_dir=temp_dir
            )
            
            elapsed = time.time() - start_time
            if transcript:
                logger.info(f"Transcription completed in {elapsed:.2f} seconds")
                logger.info(f"Transcript has {len(transcript.get('segments', []))} segments")
                
                # Check for output files
                expected_srt = str(Path(audio_path).with_suffix('.srt'))
                if os.path.exists(expected_srt):
                    logger.info(f"SRT file created: {expected_srt}")
                else:
                    logger.warning(f"SRT file not found at expected location: {expected_srt}")
                
                print(f"Transcription successful in {elapsed:.2f} seconds with {len(transcript.get('segments', []))} segments")
            else:
                logger.error("Transcription returned None")
                print("Transcription test failed - no transcript returned")
        except Exception as e:
            logger.error(f"Transcription test failed with error: {e}", exc_info=True)
            print(f"Transcription test failed: {e}")
            
    except ImportError as e:
        logger.error(f"Failed to import needed modules: {e}")
        print(f"Import error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 