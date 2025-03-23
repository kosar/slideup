#!/bin/bash
# Test script for podcast2video app

set -e  # Exit on error

# Check if running in automatic mode
AUTO_YES=false
if [ "$1" == "-y" ] || [ "$1" == "--yes" ]; then
  AUTO_YES=true
  echo "Running in non-interactive mode (auto-yes to all prompts)"
fi

# Ensure we're in the right directory
cd "$(dirname "$0")"

# First, check the environment
echo "==================================="
echo "Checking environment..."
echo "==================================="
python3 check_environment.py

# Ask if the user wants to continue after environment check
if [ "$AUTO_YES" = true ]; then
  response="y"
  echo "Continue with tests? (y/n)"
  echo "Auto-answer: $response"
else
  echo "Continue with tests? (y/n)"
  read -r response
fi

if [[ ! "$response" =~ ^[Yy]$ ]]; then
  echo "Tests cancelled by user."
  exit 0
fi

# Make sure our test audio exists
if [ ! -f "test_resources/test_audio.wav" ]; then
  echo "Test audio file not found. Creating..."
  mkdir -p test_resources
  ffmpeg -f lavfi -i anullsrc=r=44100:cl=stereo -t 20 -c:a pcm_s16le -ar 44100 test_resources/test_audio.wav
fi

# Create test directories
mkdir -p test_temp
mkdir -p test_transcripts

# Test 1: Just test transcription
echo "==================================="
echo "Running transcription test..."
echo "==================================="
python3 test_transcription.py

# Ask if the user wants to continue with the full test
if [ "$AUTO_YES" = true ]; then
  response="y"
  echo "Continue with full debug test? (y/n)"
  echo "Auto-answer: $response"
else
  echo "Continue with full debug test? (y/n)"
  read -r response
fi

if [[ ! "$response" =~ ^[Yy]$ ]]; then
  echo "Full test skipped by user."
else
  # Test 2: Run the full debug test
  echo "==================================="
  echo "Running full debug test..."
  echo "==================================="
  python3 test_debugging.py
fi

# Cleanup
if [ "$AUTO_YES" = true ]; then
  response="y"
  echo "Tests completed. Would you like to clean up temporary files? (y/n)"
  echo "Auto-answer: $response"
else
  echo "Tests completed. Would you like to clean up temporary files? (y/n)"
  read -r response
fi

if [[ "$response" =~ ^[Yy]$ ]]; then
  echo "Cleaning up..."
  rm -rf test_temp
  rm -rf test_transcripts
  echo "Cleanup completed."
else
  echo "Skipping cleanup. Temporary files remain in test_temp/ and test_transcripts/"
fi

echo "Test suite completed!" 