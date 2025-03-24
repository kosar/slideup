# Podcast2Video Web Application Development Plan

This document outlines the step-by-step plan for developing a Flask web application that serves as a user-friendly interface for the podcast2video Python script. The plan follows the requirements specified in `webapp_requirements.txt`.

## 1. Project Setup and Environment Configuration

### 1.1 Initial Setup
- Create the webapp directory structure:
  ```
  webapp/
  ├── app.py
  ├── templates/
  │   ├── index.html
  │   └── config.html
  ├── static/
  │   ├── css/
  │   ├── js/
  │   └── img/
  ├── uploads/
  └── outputs/
  ```
- Create a virtual environment for the project
- Set up requirements.txt with necessary dependencies:
  - Flask
  - Werkzeug (for secure file handling)
  - python-dotenv (for .env file handling)
  - Any dependencies required by podcast2video.py

### 1.2 Environment Configuration
- Implement loading of the `.keys_donotcheckin.env` file (located in the parent directory)
- Set up Flask configuration for:
  - File upload size limits
  - Secure file upload handling
  - Session management for prompt customization

### Troubleshooting Plan:
- Check file permissions on .env file if environment variables aren't loading
- Verify path references using absolute paths if relative paths cause issues
- Use detailed logging to trace environment setup issues

## 2. Podcast2Video Script Integration

### 2.1 Code Analysis
- Thoroughly analyze the podcast2video.py script located one directory up to understand:
  - Command-line arguments and parameters
  - Core processing functions
  - Hard-coded AI prompts
  - Error handling mechanisms
  - Dependencies and environment variables

### 2.2 Integration Setup
- Create proper import mechanisms to access the podcast2video module:
  ```python
  import sys
  import os
  # Add the parent directory to sys.path
  sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  # Import functions from podcast2video
  from podcast2video import transcribe_audio, enhance_segments, generate_visuals, create_final_video
  ```
- Create wrapper functions that adapt the CLI-oriented functions for web use
- Extract hard-coded prompts to make them configurable

### Troubleshooting Plan:
- Use try/except blocks around imports with detailed error messages
- Create test functions to verify each podcast2video function works when called from the web app
- Implement proper logging of function calls and responses

## 3. Flask Application Backend

### 3.1 Core Flask Setup
- Set up the main Flask application in app.py
- Configure session management for storing user preferences
- Implement secure file upload handling
- Create routing for all necessary endpoints:
  - Home/upload page
  - Configuration page
  - Processing endpoints
  - Progress tracking endpoint
  - Download endpoint

### 3.2 Processing Implementation
- Implement file upload validation (file type, size)
- Create background processing mechanism for long-running tasks
- Implement progress tracking and status updates
- Set up proper error handling and user feedback

### Troubleshooting Plan:
- Implement detailed logging for all processing steps
- Create timeout handling for long-running processes
- Add file validation checks before processing begins
- Implement graceful error handling with user-friendly messages

## 4. Frontend Development

### 4.1 Main Interface (index.html)
- Create a clean, responsive layout using Bootstrap
- Implement file upload interface with drag-and-drop support
- Add configuration options matching podcast2video parameters
- Create expandable/collapsible sections for advanced options
- Implement progress visualization
- Add result display and download section

### 4.2 Configuration Interface (config.html)
- Create simple form for API key configuration
- Implement prompt customization interface with:
  - Text areas for each customizable prompt
  - Default/reset options
  - Validation feedback
  - Example templates and guidance

### 4.3 JavaScript Functionality
- Implement AJAX for asynchronous file uploads
- Add progress polling and updates
- Implement form validation
- Add UI interactivity and feedback

### Troubleshooting Plan:
- Test on multiple browsers to ensure compatibility
- Implement graceful degradation for older browsers
- Add detailed client-side validation with helpful error messages
- Use console logging for JavaScript debugging

## 5. Prompt Customization Implementation

### 5.1 Prompt Extraction
- Identify and extract all hard-coded prompts from podcast2video.py, including:
  - AI enhancement prompts (from enhance_segments function)
  - Visual generation prompts (for Stability API)
  - Any other text used for AI interactions

### 5.2 Prompt Management
- Create a system for storing and retrieving customized prompts
- Implement default values and reset functionality
- Add validation for prompt content
- Create a simple interface for prompt customization

### Troubleshooting Plan:
- Create sample prompts that are known to work well
- Implement length and content validation for prompts
- Add a testing mechanism to verify prompt effectiveness
- Create detailed logs of prompt usage and outcomes

## 6. Integration and Testing

### 6.1 End-to-End Testing
- Test the complete workflow from upload to video generation
- Verify all options and configurations work correctly
- Test with various audio file types and lengths
- Verify error handling works as expected

### 6.2 Performance Optimization
- Optimize file handling for larger uploads
- Implement caching where appropriate
- Add resource cleanup for temporary files

### Troubleshooting Plan:
- Create test cases for all major functionality
- Implement detailed logging throughout the application
- Add monitoring for long-running processes
- Create recovery mechanisms for failed processes

## 7. Documentation and Deployment

### 7.1 User Documentation
- Create simple user instructions within the interface
- Add tooltips and help text for complex options
- Document the prompt customization process

### 7.2 Deployment Preparation
- Finalize requirements.txt with all dependencies
- Create setup instructions for deployment
- Document environment variable requirements

### Troubleshooting Plan:
- Create a deployment checklist
- Test in a clean environment to verify all dependencies are captured
- Document common deployment issues and solutions

## Implementation Notes

1. Focus on simplicity and reliability - create a minimal, intuitive UI that makes podcast2video.py accessible to users without technical knowledge
2. Make prompt customization optional - users should be able to use default prompts
3. Ensure proper error handling throughout the application
4. Maintain consistency with the original script's functionality
5. Implement responsive design for different device sizes
6. Use standard Flask patterns for maintainability

This plan serves as a comprehensive guide for developing the podcast2video web application. By following these steps systematically, we can create a functional, user-friendly interface that leverages the existing podcast2video.py script while adding web accessibility and customization options. 