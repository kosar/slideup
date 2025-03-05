#!/bin/bash

# Script to install all required dependencies for the project

echo "Installing required Python packages..."

# Install from requirements.txt
pip install -r requirements.txt

# Make sure anthropic is installed (sometimes this needs special handling)
pip install anthropic>=0.4.0

# Install optional but recommended packages
pip install packaging serpapi langchain_openai langchain_anthropic

echo "All dependencies installed!"
