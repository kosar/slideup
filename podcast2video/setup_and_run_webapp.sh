#!/bin/bash

# Set strict error handling
set -e

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default port for the webapp
PORT=5050

# Generated directories to track for cleanup
GENERATED_DIRS=(
    "webapp/venv"
    "webapp/uploads"
    "webapp/outputs"
    "webapp/temp"
    "webapp/transcripts"
    "webapp/cache"
    "webapp/static/uploads"
)

# Generated files to track for cleanup
GENERATED_FILES=(
    "webapp/webapp.log"
    "webapp/podcast2video.log"
    "test_resources/test_audio.wav"
    "test_resources/test_output.mp4"
)

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

# Function to clean up generated files and directories
cleanup() {
    print_header "Cleaning up generated files and directories"
    
    # Remove generated directories
    for dir in "${GENERATED_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            print_warning "Removing directory: $dir"
            rm -rf "$dir"
        fi
    done
    
    # Remove generated files
    for file in "${GENERATED_FILES[@]}"; do
        if [ -f "$file" ]; then
            print_warning "Removing file: $file"
            rm -f "$file"
        fi
    done
    
    print_success "Cleanup completed"
}

# Function to wait for server to be ready
wait_for_server() {
    local url="$1"
    local max_attempts=30
    local attempt=1
    
    print_header "Waiting for server to be ready"
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null; then
            print_success "Server is ready!"
            return 0
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done
    
    print_error "Server failed to start after $max_attempts seconds"
    return 1
}

# Function to test the API endpoints
test_api_endpoints() {
    local base_url="$1"
    print_header "Testing API endpoints"
    
    # Test the test_apis endpoint
    local response=$(curl -s "${base_url}/test_apis")
    if [[ $response == *"SUCCESS"* ]]; then
        print_success "API test endpoint check passed"
        return 0
    else
        print_error "API test endpoint check failed"
        return 1
    fi
}

# Function to kill running Flask server
kill_server() {
    local pid=$1
    if [ -n "$pid" ]; then
        print_warning "Killing Flask server (PID: $pid)"
        kill $pid 2>/dev/null || true
        sleep 2
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cleanup)
            cleanup
            exit 0
            ;;
        --port)
            PORT="$2"
            shift
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS"
    exit 1
fi

# Check Homebrew and required packages
print_header "Checking Homebrew Setup"
if ! command -v brew &>/dev/null; then
    print_error "Homebrew is not installed"
    echo "Install Homebrew first: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

print_success "Homebrew is installed"
brew update &>/dev/null || true

# Check required Homebrew packages
print_header "Checking Required Packages"
required_packages=("ffmpeg" "python@3.11" "git")
for package in "${required_packages[@]}"; do
    check_brew_package "$package"
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Source API keys
print_header "Loading API Keys"
if [ -f "$PARENT_DIR/.keys_donotcheckin.env" ]; then
    print_success "Found API keys file"
    source "$PARENT_DIR/.keys_donotcheckin.env"
    if [ -n "$OPENAI_API_KEY" ] && [ -n "$STABILITY_API_KEY" ]; then
        print_success "API keys loaded successfully"
        export OPENAI_API_KEY
        export STABILITY_API_KEY
    else
        print_error "API keys are missing from .keys_donotcheckin.env"
        exit 1
    fi
else
    print_error "No API keys file found at $PARENT_DIR/.keys_donotcheckin.env"
    exit 1
fi

# Set up Python virtual environment
print_header "Setting up Python Virtual Environment"
cd "$SCRIPT_DIR/webapp"
VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    print_warning "Removing existing virtual environment"
    rm -rf "$VENV_DIR"
fi

print_warning "Creating new virtual environment with Python 3.11"
/usr/local/opt/python@3.11/bin/python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install requirements
print_header "Installing Requirements"
pip install --upgrade pip

# Install Pillow first with specific version that works with Python 3.11
print_warning "Installing Pillow separately"
pip install "Pillow>=10.0.0"

# Now install the rest of the requirements
print_warning "Installing other requirements"
pip install Flask==2.3.3 Werkzeug==2.3.7 python-dotenv==1.0.0 gunicorn==21.2.0 \
    requests==2.31.0 jsonschema==4.19.0 moviepy==1.0.3 numpy>=1.22.0 \
    openai>=1.0.0 ffmpeg-python>=0.2.0 pydub>=0.25.1

# Create necessary directories
print_header "Creating Required Directories"
for dir in "${GENERATED_DIRS[@]}"; do
    if [[ "$dir" == webapp/* ]]; then
        dir_name="${dir#webapp/}"
        mkdir -p "$dir_name"
        print_success "Created directory: $dir_name"
    fi
done

# Test core functionality
print_header "Testing Core Functionality"
cd "$SCRIPT_DIR"
./test_core_functionality.sh

# Start the Flask server
print_header "Starting Flask Server"
cd "$SCRIPT_DIR/webapp"

# Kill any existing Flask server on the same port
lsof -i:$PORT -t | xargs kill -9 2>/dev/null || true

# Start Flask server in background
FLASK_ENV=development FLASK_APP=app.py python3 -m flask run --port $PORT &
SERVER_PID=$!

# Wait for server to be ready
if ! wait_for_server "http://localhost:$PORT"; then
    kill_server $SERVER_PID
    print_error "Server failed to start"
    exit 1
fi

# Test API endpoints
if ! test_api_endpoints "http://localhost:$PORT"; then
    kill_server $SERVER_PID
    print_error "API endpoint tests failed"
    exit 1
fi

print_success "Server is running successfully at http://localhost:$PORT"
print_success "Generated artifacts:"
echo "  Directories:"
for dir in "${GENERATED_DIRS[@]}"; do
    echo "    - $dir"
done
echo "  Files:"
for file in "${GENERATED_FILES[@]}"; do
    echo "    - $file"
done
echo
print_warning "To clean up generated files and directories, run: $0 --cleanup"
echo "To stop the server, press Ctrl+C"

# Wait for the server process
wait $SERVER_PID 