# Import the main module with better import name
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import the main module - handle the dash in the filename
    from podcast2video.podcast_to_video import *
except ImportError:
    try:
        # Try alternate import path
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "podcast_to_video", 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "podcast-to-video.py")
        )
        podcast_to_video = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(podcast_to_video)
        
        # Make all attributes available
        globals().update(vars(podcast_to_video))
    except Exception as e:
        print(f"Error importing podcast-to-video module: {e}") 