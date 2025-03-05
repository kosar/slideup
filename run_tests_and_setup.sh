#!/bin/bash

# Exit on error
set -e

# Print commands before executing them
set -x

echo "Starting test and setup process for slideup..."

# Create directories if they don't exist
mkdir -p ./test_results
mkdir -p ./logs

# Detect project type and run appropriate setup
if [ -f "./requirements.txt" ]; then
  echo "Python project detected. Setting up virtual environment..."
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  
  # Run Python tests if pytest is available
  if command -v pytest &> /dev/null; then
    echo "Running Python tests..."
    pytest
  fi

elif [ -f "./Makefile" ]; then
  echo "Makefile detected. Running make..."
  make
  
  # Run tests if available
  if grep -q "test" Makefile; then
    echo "Running tests via make..."
    make test
  fi
  
elif [ -f "./gradlew" ]; then
  echo "Gradle project detected. Building..."
  ./gradlew build
  
elif [ -f "./mvnw" ] || [ -f "pom.xml" ]; then
  echo "Maven project detected. Building..."
  ./mvnw package || mvn package

elif [ -f "./go.mod" ]; then
  echo "Go project detected. Building..."
  go build ./...
  go test ./...
  
else
  echo "No specific project structure detected."
  
  # Generic file collection for tests (modify patterns as needed)
  echo "Looking for test files..."
  find . -name "*_test.*" -o -name "test_*.*" | sort
fi

# Copy example configuration if it exists
for config_example in $(find . -name "*.example.*"); do
  config_file="${config_example/.example/}"
  if [ ! -f "$config_file" ]; then
    echo "Creating config file from $config_example"
    cp "$config_example" "$config_file"
  fi
done

echo "Setup and tests completed successfully!"
