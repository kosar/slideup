#!/bin/bash

# Exit on error, but allow for cleanup
set -e
trap cleanup EXIT

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${2:-$NC}$1${NC}"
}

success() {
    log "✓ $1" "$GREEN"
}

warning() {
    log "! $1" "$YELLOW"
}

error() {
    log "✗ $1" "$RED"
}

# Cleanup function
cleanup() {
    if [ "$1" == "--cleanup" ]; then
        log "\n=== Cleaning up generated files ==="
        rm -rf webapp/venv webapp/uploads webapp/outputs webapp/temp webapp/transcripts webapp/cache webapp/static/uploads
        rm -f webapp/webapp.log webapp/podcast2video.log test_resources/test_audio.wav test_resources/test_output.mp4
        success "Cleanup completed"
        exit 0
    fi
}

# Check if cleanup was requested
if [ "$1" == "--cleanup" ]; then
    cleanup --cleanup
fi

# Function to check and create directories
create_directories() {
    local dirs=("venv" "uploads" "outputs" "temp" "transcripts" "cache" "static/uploads")
    
    log "\n=== Creating Required Directories ==="
    for dir in "${dirs[@]}"; do
        if [ ! -d "webapp/$dir" ]; then
            mkdir -p "webapp/$dir"
            success "Created directory: $dir"
        else
            success "Directory exists: $dir"
        fi
    done
}

# Function to check and install Python packages
install_requirements() {
    log "\n=== Installing Requirements ==="
    
    # Upgrade pip first
    warning "Upgrading pip..."
    ./venv/bin/pip install --upgrade pip || {
        error "Failed to upgrade pip"
        return 1
    }
    
    # Install Pillow separately (it sometimes needs special handling)
    warning "Installing Pillow separately"
    ./venv/bin/pip install "Pillow>=10.0.0" || {
        error "Failed to install Pillow"
        return 1
    }
    
    # Install other requirements
    warning "Installing other requirements"
    ./venv/bin/pip install -r requirements.txt || {
        error "Failed to install requirements"
        return 1
    }
    
    success "All requirements installed successfully"
}

# Function to check and cleanup port
cleanup_port() {
    local port=$1
    log "\n=== Checking Port $port ==="
    
    if lsof -i :$port > /dev/null 2>&1; then
        warning "Port $port is in use. Attempting to clean up..."
        # Try to find and kill the process using the port
        local pid=$(lsof -ti :$port)
        if [ ! -z "$pid" ]; then
            warning "Killing process $pid using port $port"
            kill -9 $pid
            sleep 2  # Give the system time to free up the port
            success "Port $port is now available"
        else
            error "Could not find process using port $port"
            exit 1
        fi
    else
        success "Port $port is available"
    fi
}

# Function to cleanup on exit
cleanup() {
    # Kill the server if it's running
    if [ ! -z "$SERVER_PID" ]; then
        warning "Shutting down server (PID: $SERVER_PID)..."
        kill -9 $SERVER_PID 2>/dev/null
    fi
}

# Get the absolute path of the keys file
get_keys_file_path() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local slideup_dir="$(cd "$script_dir/.." && pwd)"
    echo "$slideup_dir/.keys_donotcheckin.env"
}

# Function to load API keys
load_api_keys() {
    local keys_file=$(get_keys_file_path)
    log "\n=== Loading API Keys ==="
    if [ -f "$keys_file" ]; then
        set -a  # automatically export all variables
        source "$keys_file"
        set +a  # stop automatically exporting
        success "Found API keys file at: $keys_file"
        success "API keys loaded successfully"
    else
        warning "API keys file not found at: $keys_file"
        warning "Some features may not work."
    fi
}

# Function to test the server
test_server() {
    log "\n=== Testing Server ==="
    
    # Clean up the port if needed
    cleanup_port 5050
    
    # Start the server in the background with environment from parent shell
    PYTHONPATH="$PYTHONPATH:$(pwd)" ./venv/bin/python webapp/app.py &
    SERVER_PID=$!
    
    # Wait for server to start
    log "\n=== Waiting for server to be ready ==="
    for i in {1..30}; do
        if curl -s http://localhost:5050 > /dev/null; then
            success "Server is ready!"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            error "Server process died unexpectedly"
            exit 1
        fi
        echo -n "."
        sleep 1
    done
    
    # Test API endpoints
    log "\n=== Testing API endpoints ==="
    if curl -s http://localhost:5050/test_apis > /dev/null; then
        success "API test endpoint check passed"
    else
        error "API test endpoint check failed"
    fi
    
    success "Server is running successfully at http://localhost:5050"
    
    # Keep the script running and show server logs
    log "\nServer is running. Press Ctrl+C to stop."
    wait $SERVER_PID
}

# Main setup process
main() {
    # Handle cleanup flag
    if [ "$1" = "--cleanup" ]; then
        cleanup_port 5050
        exit 0
    fi
    
    # Check Homebrew and required packages
    log "\n=== Checking Homebrew Setup ==="
    if ! command -v brew &> /dev/null; then
        error "Homebrew not found. Please install Homebrew first."
        exit 1
    fi
    success "Homebrew is installed"
    
    log "\n=== Checking Required Packages ==="
    local packages=("ffmpeg" "python@3.11" "git" "imagemagick")
    for package in "${packages[@]}"; do
        if ! brew list "$package" &> /dev/null; then
            warning "Installing $package..."
            brew install "$package"
        else
            success "$package is up to date"
        fi
    done
    
    # Load API keys once at the start
    load_api_keys
    
    # Set up Python virtual environment
    log "\n=== Setting up Python Virtual Environment ==="
    if [ ! -d "venv" ]; then
        warning "Creating new virtual environment with Python 3.11"
        python3.11 -m venv venv
    else
        success "Found existing virtual environment"
    fi
    
    # Create directories
    create_directories
    
    # Install requirements
    install_requirements
        
    # Test core functionality
    log "\n=== Testing Core Functionality ==="
    ./venv/bin/python podcast-to-video.py --test_openai || {
        warning "Core functionality test failed, but continuing with setup..."
    }
    
    # Start the server and test it
    test_server
    
    # List generated artifacts
    log "\n=== Generated Artifacts ==="
    success "Generated artifacts:"
    echo "  Directories:"
    for dir in webapp/venv webapp/uploads webapp/outputs webapp/temp webapp/transcripts webapp/cache webapp/static/uploads; do
        echo "    - $dir"
    done
    echo "  Files:"
    for file in webapp/webapp.log webapp/podcast2video.log test_resources/test_audio.wav test_resources/test_output.mp4; do
        echo "    - $file"
    done
    
    warning "To clean up generated files and directories, run: ./setup_and_run_webapp.sh --cleanup"
    log "To stop the server, press Ctrl+C"
}

# Run main setup
main "$@" 