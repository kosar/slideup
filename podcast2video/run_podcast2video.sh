#!/bin/bash
set -e

# Script to set up the environment and run podcast2video

# Navigate to the project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

# Source API keys from the environment file
if [ -f "$PROJECT_ROOT/.keys_donotcheckin.env" ]; then
    echo "Sourcing API keys from .keys_donotcheckin.env..."
    set -a
    source "$PROJECT_ROOT/.keys_donotcheckin.env"
    set +a
else
    echo "Error: API keys file not found at $PROJECT_ROOT/.keys_donotcheckin.env"
    echo "Please ensure your API keys are properly set up before running."
    exit 1
fi

# Verify required API keys are set
# Removed OPENAI_API_KEY from required keys as we now use local transcription
required_keys=("STABILITY_API_KEY")
missing_keys=()

for key in "${required_keys[@]}"; do
    if [ -z "${!key}" ]; then
        missing_keys+=("$key")
    fi
done

if [ ${#missing_keys[@]} -ne 0 ]; then
    echo "Error: The following required API keys are not set:"
    printf '%s\n' "${missing_keys[@]}"
    echo "Please ensure all required API keys are set in .keys_donotcheckin.env"
    exit 1
fi

# Activate virtual environment
VENV_PATH="$PROJECT_ROOT/podcast2video/podcast_env"
if [ -f "$VENV_PATH/bin/activate" ]; then
    echo "Activating virtual environment at $VENV_PATH..."
    source "$VENV_PATH/bin/activate"
    echo "Virtual environment activated successfully."
    python --version
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please create and set up your virtual environment before running."
    exit 1
fi

# Navigate to the podcast2video directory
cd "$PROJECT_ROOT/podcast2video"
echo "Changed directory to $(pwd)"

# Run environment check
echo "Running environment check..."
python check_environment.py

# Check if input audio file exists
if [ ! -f "input.wav" ]; then
    echo "Error: input.wav not found in the current directory"
    echo "Please ensure you have an input audio file named 'input.wav'"
    exit 1
fi

# Run the podcast-to-video script
echo -e "\nRunning podcast-to-video conversion..."
python podcast-to-video.py \
    --audio input.wav \
    --output enhanced_podcast.mp4 \
    --temp_dir temp \
    --transcript_dir transcripts \
    --subtitle always \
    # --limit_to_one_minute

echo -e "\nConversion completed. Check the logs for details."
echo "Remember to deactivate your virtual environment when done:"
echo "  deactivate" 