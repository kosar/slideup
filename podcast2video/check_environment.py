#!/usr/bin/env python3
"""
Environment checker for podcast2video.
This script checks if all required environment variables and dependencies are set up correctly.
"""

import os
import sys
import subprocess
import importlib
import platform

def check_environment():
    """Check if the environment is set up correctly for podcast2video"""
    print("\n====== Podcast2Video Environment Check ======\n")
    
    # System info
    print("System Information:")
    print(f"  Python version: {sys.version}")
    print(f"  Operating system: {platform.system()} {platform.release()}")
    print(f"  Platform: {platform.platform()}")
    print()
    
    # Check for required environment variables
    print("API Keys:")
    check_env_var("OPENAI_API_KEY", "OpenAI API")
    check_env_var("DEEPSEEK_API_KEY", "DeepSeek API")
    check_env_var("STABILITY_API_KEY", "Stability AI API")
    print()
    
    # Check for required Python packages
    print("Required Python Packages:")
    check_package("openai", "OpenAI client library")
    check_package("moviepy", "MoviePy (video editing)")
    check_package("pydub", "PyDub (audio processing)")
    print()
    
    # Check for external dependencies
    print("External Dependencies:")
    check_command("ffmpeg", "FFmpeg (required for audio/video processing)")
    print()
    
    # Suggest fixes
    suggest_fixes()
    
    print("\n====== Environment Check Complete ======\n")
    
def check_env_var(var_name, description):
    """Check if an environment variable is set"""
    value = os.environ.get(var_name)
    if value:
        # Mask the actual API key for security
        masked_value = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else "***"
        print(f"✅ {description} key ({var_name}): {masked_value}")
    else:
        print(f"❌ {description} key ({var_name}): Not found")

def check_package(package_name, description):
    """Check if a Python package is installed"""
    try:
        importlib.import_module(package_name)
        try:
            version = importlib.import_module(package_name).__version__
            print(f"✅ {description} ({package_name}): Installed (version {version})")
        except AttributeError:
            print(f"✅ {description} ({package_name}): Installed (version unknown)")
    except ImportError:
        print(f"❌ {description} ({package_name}): Not installed")

def check_command(command, description):
    """Check if an external command is available"""
    try:
        result = subprocess.run(["which", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            # Also check version
            try:
                version_result = subprocess.run([command, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                version_info = version_result.stdout.strip().split('\n')[0] if version_result.stdout else "Version unknown"
                print(f"✅ {description}: Installed at {result.stdout.strip()} ({version_info})")
            except Exception:
                print(f"✅ {description}: Installed at {result.stdout.strip()}")
        else:
            print(f"❌ {description}: Not found")
    except Exception as e:
        print(f"❌ {description}: Error checking ({str(e)})")

def suggest_fixes():
    """Suggest fixes for common issues"""
    missing_items = []
    
    # Check for missing API keys
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("DEEPSEEK_API_KEY"):
        missing_items.append("No API keys found. At least one of OPENAI_API_KEY or DEEPSEEK_API_KEY is required.")
    
    # Check for missing packages
    packages_to_check = ["openai", "moviepy", "pydub"]
    missing_packages = []
    
    for package in packages_to_check:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        packages_str = ", ".join(missing_packages)
        missing_items.append(f"Missing required packages: {packages_str}. Install with: pip install {packages_str}")
    
    # Check for ffmpeg
    result = subprocess.run(["which", "ffmpeg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        missing_items.append("FFmpeg not found. Install ffmpeg using your package manager.")
        
    # Print suggestions
    if missing_items:
        print("\nSuggested fixes:")
        for i, item in enumerate(missing_items, 1):
            print(f"  {i}. {item}")
    else:
        print("\nAll required components appear to be installed!")
        print("If you're still experiencing issues, check the logs for more detailed error messages.")

if __name__ == "__main__":
    check_environment() 