#!/bin/bash

# Set strict error handling
set -e

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored section headers
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}! $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "Required command '$1' not found. Please install it first."
        return 1
    fi
    return 0
}

# Function to check Homebrew package status
check_brew_package() {
    local package=$1
    if ! brew list $package &>/dev/null; then
        print_warning "$package is not installed"
        echo "To install: brew install $package"
        return 1
    else
        local outdated=$(brew outdated | grep "^$package\$" || true)
        if [ -n "$outdated" ]; then
            print_warning "$package is outdated"
            echo "To update: brew upgrade $package"
            return 2
        else
            print_success "$package is up to date"
            return 0
        fi
    fi
}

print_header "Podcast2Video Core Functionality Test"
echo "This script will test the core functionality of podcast-to-video.py"
echo "It will create a test environment and run basic conversion tests."

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    print_header "Checking Homebrew Setup"
    
    # Check if Homebrew is installed
    if ! command -v brew &>/dev/null; then
        print_error "Homebrew is not installed"
        echo "Install Homebrew first: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    else
        print_success "Homebrew is installed"
        
        # Update Homebrew itself
        echo -e "\nChecking Homebrew updates..."
        if brew update > /tmp/brew_update.log 2>&1; then
            if grep -q "Already up to date" /tmp/brew_update.log; then
                print_success "Homebrew is up to date"
            else
                print_success "Homebrew updated successfully"
            fi
        else
            print_warning "Homebrew update failed - check /tmp/brew_update.log for details"
        fi
        rm -f /tmp/brew_update.log
        
        # Check required Homebrew packages
        print_header "Checking Homebrew Packages"
        check_brew_package "ffmpeg"
        check_brew_package "python@3.11"  # or whatever version you're using
        
        # Optional but recommended packages
        echo -e "\nChecking optional packages:"
        check_brew_package "git" || true
    fi
else
    print_warning "Not running on macOS - skipping Homebrew checks"
fi

# Check for required commands
print_header "Checking Required Commands"
required_commands=("python3" "ffmpeg" "pip")
missing_commands=0
for cmd in "${required_commands[@]}"; do
    if check_command $cmd; then
        print_success "Found $cmd"
    else
        missing_commands=$((missing_commands + 1))
    fi
done

if [ $missing_commands -gt 0 ]; then
    print_error "Please install missing commands and try again."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "You can install missing packages using Homebrew:"
        echo "  brew install ffmpeg python@3.11"
    fi
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Source API keys if available
print_header "Checking API Keys"
if [ -f "$PARENT_DIR/.keys_donotcheckin.env" ]; then
    print_success "Found API keys file"
    source "$PARENT_DIR/.keys_donotcheckin.env"
    if [ -n "$OPENAI_API_KEY" ] && [ -n "$STABILITY_API_KEY" ]; then
        print_success "API keys loaded successfully"
    else
        print_warning "API keys file found but some keys are missing"
        echo "Expected keys: OPENAI_API_KEY, STABILITY_API_KEY"
    fi
else
    print_warning "No API keys file found at $PARENT_DIR/.keys_donotcheckin.env"
    echo "The test will continue but may fail if API keys are required"
fi

# Check/Create virtual environment
print_header "Setting up Python Virtual Environment"
VENV_DIR="$SCRIPT_DIR/venv"

if [ -d "$VENV_DIR" ]; then
    print_success "Found existing virtual environment"
    source "$VENV_DIR/bin/activate"
else
    print_warning "Creating new virtual environment"
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    print_header "Installing Required Packages"
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r requirements.txt
fi

# Create test audio file
print_header "Creating Test Audio File"
TEST_AUDIO="$SCRIPT_DIR/test_resources/test_audio.wav"
mkdir -p "$SCRIPT_DIR/test_resources"

if [ ! -f "$TEST_AUDIO" ] || [ "$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$TEST_AUDIO")" != "5.000000" ]; then
    print_warning "Generating 5-second test audio file with speech"
    # Generate a text file with test speech
    echo "This is a test audio file for podcast to video conversion. We are testing the duration matching functionality." > "$SCRIPT_DIR/test_resources/test_speech.txt"
    
    # Use macOS say command to generate speech
    if [[ "$OSTYPE" == "darwin"* ]]; then
        say -v Samantha -r 175 -f "$SCRIPT_DIR/test_resources/test_speech.txt" -o "$TEST_AUDIO.aiff"
        # Convert to WAV format
        ffmpeg -y -i "$TEST_AUDIO.aiff" -acodec pcm_s16le -ar 44100 -ac 2 "$TEST_AUDIO"
        rm -f "$TEST_AUDIO.aiff"
    else
        # Fallback to ffmpeg sine wave if not on macOS
        ffmpeg -y -f lavfi -i "sine=frequency=440:duration=5" -ar 44100 "$TEST_AUDIO"
    fi
    
    # Clean up
    rm -f "$SCRIPT_DIR/test_resources/test_speech.txt"
else
    print_success "Using existing test audio file"
fi

# Force regenerate the transcript
rm -f "$SCRIPT_DIR/transcripts/test_audio.srt"

# Run the test
print_header "Running Core Functionality Test"
echo "Testing with a 5-second audio file..."

TEST_OUTPUT="$SCRIPT_DIR/test_resources/test_output.mp4"
TEST_LOG="$SCRIPT_DIR/test_resources/test.log"

# Run podcast-to-video.py with test file
python3 "$SCRIPT_DIR/podcast-to-video.py" \
    --input "$TEST_AUDIO" \
    --output "$TEST_OUTPUT" \
    --force_transcription \
    --non_interactive 2>&1 | tee "$TEST_LOG"

# Check results
print_header "Checking Test Results"

# Check if output file exists and is not empty
if [ -f "$TEST_OUTPUT" ] && [ -s "$TEST_OUTPUT" ]; then
    print_success "Output video file created successfully"
    
    # Get video duration using ffprobe
    VIDEO_DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$TEST_OUTPUT")
    AUDIO_DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$TEST_AUDIO")
    
    if (( $(echo "$VIDEO_DURATION > 0" | bc -l) )); then
        print_success "Video duration: ${VIDEO_DURATION}s"
        print_success "Audio duration: ${AUDIO_DURATION}s"
        
        # Allow for a small difference (0.1 seconds) to account for encoding variations
        DURATION_DIFF=$(echo "scale=3; ($VIDEO_DURATION - $AUDIO_DURATION)" | bc)
        if (( $(echo "sqrt($DURATION_DIFF * $DURATION_DIFF) <= 0.1" | bc -l) )); then
            print_success "Duration check passed: Video matches audio length"
        else
            print_warning "Duration mismatch: Video (${VIDEO_DURATION}s) differs from audio (${AUDIO_DURATION}s)"
        fi
    else
        print_error "Video duration check failed: ${VIDEO_DURATION}s"
    fi
    
    # Check if transcription was created
    if [ -f "$SCRIPT_DIR/transcripts/test_audio.srt" ]; then
        print_success "Transcription file created"
    else
        print_warning "No transcription file found"
    fi
else
    print_error "Failed to create output video file"
    echo "Check the log file at $TEST_LOG for details"
fi

# Check for common error patterns in the log
if grep -q "Error:" "$TEST_LOG"; then
    print_warning "Found errors in the log file:"
    grep "Error:" "$TEST_LOG"
fi

print_header "Test Summary"
if [ -f "$TEST_OUTPUT" ] && [ -s "$TEST_OUTPUT" ]; then
    print_success "Core functionality test completed successfully"
    echo "- Output video: $TEST_OUTPUT"
    echo "- Log file: $TEST_LOG"
    echo "- Test audio: $TEST_AUDIO"
else
    print_error "Core functionality test failed"
    echo "Please check the log file at $TEST_LOG for details"
fi

# Cleanup
deactivate

echo -e "\nTest completed. Review the output above for any warnings or errors." 