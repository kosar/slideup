#!/bin/bash
set -e

# Script to set up the environment and run podcast2video tests

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
    echo "Please ensure your API keys are properly set up before running tests."
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
    echo "Please create and set up your virtual environment before running tests."
    exit 1
fi

# Navigate to the podcast2video directory
cd "$PROJECT_ROOT/podcast2video"
echo "Changed directory to $(pwd)"

# Run environment check
echo "Running environment check..."
python check_environment.py

# Execute the test script
echo -e "\nRunning tests..."
./run_tests.sh -y

echo -e "\nAll tests completed. Check the logs for details."
echo "Remember to deactivate your virtual environment when done:"
echo "  deactivate" 