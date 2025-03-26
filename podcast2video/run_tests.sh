#!/bin/bash
# Test runner script for podcast2video

# Set colors for terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Store the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set up Python path
export PYTHONPATH=$SCRIPT_DIR:$PYTHONPATH

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Check Python environment
echo -e "${YELLOW}Checking Python environment...${NC}"
python -m check_environment

# Function to run a test and check its exit code
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -e "\n${YELLOW}Running $test_name...${NC}"
    eval $test_command
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}$test_name PASSED${NC}"
        return 0
    else
        echo -e "${RED}$test_name FAILED${NC}"
        return 1
    fi
}

# Create a temp directory for test output
TEST_TEMP_DIR="$SCRIPT_DIR/test_temp"
mkdir -p "$TEST_TEMP_DIR"

# Track test results
PASSED=0
FAILED=0
TOTAL=0

# Run unit tests for cost tracker module
run_test "Cost Tracker Unit Tests" "python -m unittest discover -s . -p 'test_cost_tracker.py'"
if [ $? -eq 0 ]; then
    PASSED=$((PASSED+1))
else
    FAILED=$((FAILED+1))
fi
TOTAL=$((TOTAL+1))

# Run end-to-end tests for cost tracker integration
# Only run if API keys are available
if [ -n "$OPENAI_API_KEY" ]; then
    run_test "Cost Tracker E2E Tests" "python -m unittest discover -s . -p 'test_cost_tracker_e2e.py'"
    if [ $? -eq 0 ]; then
        PASSED=$((PASSED+1))
    else
        FAILED=$((FAILED+1))
    fi
    TOTAL=$((TOTAL+1))
else
    echo -e "${YELLOW}Skipping Cost Tracker E2E Tests - OPENAI_API_KEY not set${NC}"
fi

# Run transcription test
if [ -f "$SCRIPT_DIR/test_resources/test_audio.mp3" ] && [ -n "$OPENAI_API_KEY" ]; then
    run_test "Transcription Test" "python -m test_transcription"
    if [ $? -eq 0 ]; then
        PASSED=$((PASSED+1))
    else
        FAILED=$((FAILED+1))
    fi
    TOTAL=$((TOTAL+1))
else
    echo -e "${YELLOW}Skipping Transcription Test - Missing test_audio.mp3 or OPENAI_API_KEY${NC}"
fi

# Run debugging tests
run_test "Debugging Features Test" "python -m test_debugging"
if [ $? -eq 0 ]; then
    PASSED=$((PASSED+1))
else
    FAILED=$((FAILED+1))
fi
TOTAL=$((TOTAL+1))

# Run core functionality test if all required API keys are available
if [ -n "$OPENAI_API_KEY" ] && [ -n "$STABILITY_API_KEY" ]; then
    run_test "Core Functionality Test" "bash test_core_functionality.sh"
    if [ $? -eq 0 ]; then
        PASSED=$((PASSED+1))
    else
        FAILED=$((FAILED+1))
    fi
    TOTAL=$((TOTAL+1))
else
    echo -e "${YELLOW}Skipping Core Functionality Test - Missing API keys${NC}"
    echo -e "${YELLOW}Expected keys: OPENAI_API_KEY, STABILITY_API_KEY${NC}"
fi

# Display summary
echo -e "\n${YELLOW}Test Summary${NC}"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo -e "Total: $TOTAL"

# Clean up
echo -e "\n${YELLOW}Cleaning up temporary files...${NC}"
if [ -d "$TEST_TEMP_DIR" ]; then
    rm -rf "$TEST_TEMP_DIR"
fi

# Return appropriate exit code
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi 