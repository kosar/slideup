import os
import sys
import json
import time
import uuid
import logging
import subprocess
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import threading

# Add the parent directory to sys.path - needed for finding the podcast-to-video.py script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Note: We're not directly importing functions from podcast-to-video.py
# to avoid MoviePy dependency issues. Instead, we use subprocess to call
# the script directly. See process_audio_file_background function for details.

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("webapp.log")
    ]
)
logger = logging.getLogger('webapp')

# Load environment variables
try:
    from dotenv import load_dotenv
    
    # Look two directories up for the keys file (.keys_donotcheckin.env)
    keys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.keys_donotcheckin.env'))
    
    if os.path.exists(keys_path):
        # Load environment variables directly from the keys file
        load_dotenv(keys_path)
        logger.info(f"Loaded environment variables from {keys_path}")
    else:
        logger.warning(f"Keys file not found at {keys_path}")
    
    # Log API key status
    if not os.environ.get('OPENAI_API_KEY'):
        logger.warning("OPENAI_API_KEY not found in environment")
    if not os.environ.get('STABILITY_API_KEY'):
        logger.warning("STABILITY_API_KEY not found in environment")
    
except Exception as e:
    logger.error(f"Error loading environment variables: {str(e)}")

# Flask app configuration
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav'}

# Default prompt templates - extracted from the original script
DEFAULT_PROMPTS = {
    'enhance_segments': """You are a helpful assistant that analyzes podcast segments and provides structured descriptions and visual prompts.
Your task is to analyze this podcast segment and provide a JSON response with the following structure:

{
    "description": "A clear, concise description of the main topic",
    "visual_prompt": "A detailed visual prompt for generating an image that represents this content",
    "key_points": ["Key point 1", "Key point 2", "Key point 3"]
}

The visual prompt should be detailed and specific, focusing on visual elements that would make a compelling image. 
Include details like style, composition, colors, and elements to include in the image.""",
    'visual_generation': "A high quality, detailed image of the following scene: "
}

# Background processes tracking
processing_tasks = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_task_status(task_id):
    """Get the status of a specific task"""
    if task_id in processing_tasks:
        return processing_tasks[task_id]
    return {'status': 'unknown', 'message': 'Task not found'}

def process_audio_file_background(task_id, input_file, output_file, limit_to_one_minute=False, non_interactive=True, 
                                enhance_prompt=None, visual_prompt=None):
    """Background processing of the audio file"""
    try:
        processing_tasks[task_id] = {'status': 'starting', 'progress': 0, 'message': 'Initializing processing'}
        
        # Set up directory paths
        webapp_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(webapp_dir, 'uploads', 'temp_' + task_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Update status
        processing_tasks[task_id] = {'status': 'processing', 'progress': 10, 'message': 'Processing audio...'}
        
        # Call the podcast-to-video.py script directly instead of using the imported functions
        # This avoids the MoviePy import errors
        logger.info(f"Starting processing task {task_id} using subprocess")
        
        # Build command with all necessary arguments
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "podcast-to-video.py"),
            "--input", input_file,
            "--output", output_file,
            "--temp_dir", temp_dir,
            "--non_interactive"
        ]
        
        if limit_to_one_minute:
            cmd.append("--limit_to_one_minute")
            
        # Note about custom prompts:
        # Custom prompts are stored in the session but cannot be directly passed to the command line
        # as the script doesn't support this. In a real implementation, you would need to modify
        # the original script to accept these parameters or use environment variables.
        if enhance_prompt or visual_prompt:
            logger.warning("Custom prompts are stored but cannot be passed to the script - " +
                          "the script doesn't support command line arguments for prompts")
            
        # Run the command
        logger.info(f"Running processing command: {' '.join(cmd)}")
        
        result = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            bufsize=1, 
            universal_newlines=True,
            env=os.environ  # Pass current environment including API keys
        )
        
        # Start a separate thread to monitor stderr
        def monitor_stderr():
            for line in iter(result.stderr.readline, ''):
                logger.error(line.strip())
        
        stderr_thread = threading.Thread(target=monitor_stderr)
        stderr_thread.daemon = True
        stderr_thread.start()
        
        # Monitor progress based on log output
        for line in iter(result.stdout.readline, ''):
            logger.info(line.strip())
            
            # Update progress based on log lines
            if "Starting transcription" in line:
                processing_tasks[task_id] = {'status': 'transcribing', 'progress': 20, 'message': 'Transcribing audio...'}
            elif "Enhancing segment" in line:
                processing_tasks[task_id] = {'status': 'enhancing', 'progress': 40, 'message': 'Enhancing segments with AI...'}
            elif "Generating visuals" in line:
                processing_tasks[task_id] = {'status': 'generating_visuals', 'progress': 60, 'message': 'Generating visuals...'}
            elif "Starting final video creation" in line:
                processing_tasks[task_id] = {'status': 'creating_video', 'progress': 80, 'message': 'Creating final video...'}
            elif "Video creation completed" in line or "Processing complete" in line:
                processing_tasks[task_id] = {
                    'status': 'completed', 
                    'progress': 100, 
                    'message': 'Processing complete!',
                    'output_file': os.path.basename(output_file)
                }
        
        # Check if there was an error
        result.wait()
        if result.returncode != 0:
            # Read any remaining stderr
            error_output = result.stderr.read()
            logger.error(f"Processing failed with return code {result.returncode}: {error_output}")
            processing_tasks[task_id] = {'status': 'failed', 'progress': 0, 'message': f'Error: Processing failed with code {result.returncode}'}
            return
        
        # All done!
        if 'status' not in processing_tasks[task_id] or processing_tasks[task_id]['status'] != 'completed':
            processing_tasks[task_id] = {
                'status': 'completed', 
                'progress': 100, 
                'message': 'Processing complete!',
                'output_file': os.path.basename(output_file)
            }
            logger.info(f"Processing completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"Error in process_audio_file_background: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {sys.exc_info()}")
        processing_tasks[task_id] = {
            'status': 'failed', 
            'progress': 0, 
            'message': f'Error: {str(e)}'
        }

@app.route('/')
def index():
    """Main page with file upload form"""
    return render_template('index.html')

@app.route('/config')
def config_page():
    """Configuration page for API keys and prompts"""
    # Get current prompts from session or use defaults
    prompts = session.get('prompts', DEFAULT_PROMPTS)
    api_keys = {
        'openai_api_key': os.environ.get('OPENAI_API_KEY', ''),
        'stability_api_key': os.environ.get('STABILITY_API_KEY', '')
    }
    return render_template('config.html', prompts=prompts, api_keys=api_keys)

@app.route('/save_config', methods=['POST'])
def save_config():
    """Save configuration settings"""
    try:
        # Get current prompts
        prompts = session.get('prompts', DEFAULT_PROMPTS.copy())
        
        # Update prompts from form
        if 'enhance_segments' in request.form:
            prompts['enhance_segments'] = request.form['enhance_segments']
        if 'visual_generation' in request.form:
            prompts['visual_generation'] = request.form['visual_generation']
        
        # Save prompts to session
        session['prompts'] = prompts
        
        # We don't update API keys here as they should be set in the environment
        # Flash a notice about this
        flash('Prompts saved successfully. Note: API keys must be set in the environment file.', 'success')
        
        return redirect(url_for('config_page'))
    except Exception as e:
        flash(f'Error saving configuration: {str(e)}', 'error')
        return redirect(url_for('config_page'))

@app.route('/reset_prompts', methods=['POST'])
def reset_prompts():
    """Reset prompts to defaults"""
    session['prompts'] = DEFAULT_PROMPTS.copy()
    flash('Prompts reset to defaults.', 'success')
    return redirect(url_for('config_page'))

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing"""
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
        
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
        
    if file and allowed_file(file.filename):
        # Create unique task ID
        task_id = str(uuid.uuid4())
        
        # Save the file
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(file_path)
        
        # Get processing options
        limit_to_one_minute = 'limit_to_one_minute' in request.form
        non_interactive = True  # Always non-interactive for web app
        
        # Get custom prompts if provided
        prompts = session.get('prompts', DEFAULT_PROMPTS)
        enhance_prompt = prompts.get('enhance_segments', DEFAULT_PROMPTS['enhance_segments'])
        visual_prompt = prompts.get('visual_generation', DEFAULT_PROMPTS['visual_generation'])
        
        # Set up output path
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
        output_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{task_id}_output.mp4")
        
        # Start processing in background
        processing_thread = threading.Thread(
            target=process_audio_file_background,
            args=(task_id, file_path, output_file, limit_to_one_minute, non_interactive, enhance_prompt, visual_prompt)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        # Redirect to status page
        return redirect(url_for('status', task_id=task_id))
    
    flash('Invalid file type. Please upload an MP3 or WAV file.', 'error')
    return redirect(url_for('index'))

@app.route('/status/<task_id>')
def status(task_id):
    """Status page for a specific task"""
    task_status = get_task_status(task_id)
    return render_template('status.html', task_id=task_id, status=task_status)

@app.route('/api/status/<task_id>')
def api_status(task_id):
    """API endpoint for getting task status"""
    return jsonify(get_task_status(task_id))

@app.route('/download/<task_id>')
def download(task_id):
    """Download the generated video file"""
    task_status = get_task_status(task_id)
    if task_status.get('status') == 'completed' and 'output_file' in task_status:
        return send_from_directory(app.config['OUTPUT_FOLDER'], task_status['output_file'], as_attachment=True)
    flash('File not ready for download.', 'error')
    return redirect(url_for('status', task_id=task_id))

@app.route('/test_apis')
def test_apis_endpoint():
    """Test endpoint to verify API connectivity"""
    results = test_apis()
    
    # If the request wants JSON (likely an API call), return JSON
    if request.headers.get('Accept') == 'application/json' or request.args.get('format') == 'json':
        return jsonify(results)
    
    # Otherwise render the HTML template (for browser viewing)
    return render_template('test_apis.html', test_results=results)

def test_apis():
    """Test API connectivity with OpenAI and Stability AI"""
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'tests': []
    }
    
    # Test OpenAI API
    openai_result = {
        'api': 'OpenAI',
        'status': 'failed',
        'message': ''
    }
    
    try:
        # Check if OpenAI API key is set
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            openai_result['message'] = 'API key not found in environment'
        else:
            # Make a minimal API request to OpenAI using subprocess
            test_cmd = [
                sys.executable,
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "podcast-to-video.py"),
                "--test_openai"
            ]
            
            logger.info(f"Testing OpenAI API connection")
            proc = subprocess.run(test_cmd, capture_output=True, text=True, env=os.environ)
            
            # Check if the command was successful and look for success message
            if proc.returncode == 0 or "Successfully connected to OpenAI API" in proc.stdout:
                openai_result['status'] = 'success'
                openai_result['message'] = 'Connection successful'
            else:
                # Filter out MoviePy errors from the message
                error_msg = proc.stderr.strip() or proc.stdout.strip() or 'Unknown error'
                if "MoviePy" not in error_msg:
                    openai_result['message'] = error_msg
                else:
                    # If it's just a MoviePy error but the API test succeeded
                    if "Successfully connected to OpenAI API" in proc.stdout:
                        openai_result['status'] = 'success'
                        openai_result['message'] = 'Connection successful (MoviePy warnings can be ignored)'
                    else:
                        openai_result['message'] = 'API test failed'
    except Exception as e:
        openai_result['message'] = f'Error: {str(e)}'
    
    results['tests'].append(openai_result)
    
    # Test Stability API
    stability_result = {
        'api': 'Stability AI',
        'status': 'failed',
        'message': ''
    }
    
    try:
        # Check if Stability API key is set
        stability_api_key = os.environ.get('STABILITY_API_KEY')
        if not stability_api_key:
            stability_result['message'] = 'API key not found in environment'
        else:
            # Make a minimal API request to Stability AI using subprocess
            test_cmd = [
                sys.executable,
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "podcast-to-video.py"),
                "--test_stability"
            ]
            
            logger.info(f"Testing Stability API connection")
            proc = subprocess.run(test_cmd, capture_output=True, text=True, env=os.environ)
            
            # Check if the command was successful and look for success message
            if proc.returncode == 0 or "Successfully connected to Stability API" in proc.stdout:
                stability_result['status'] = 'success'
                stability_result['message'] = 'Connection successful'
            else:
                # Filter out MoviePy errors from the message
                error_msg = proc.stderr.strip() or proc.stdout.strip() or 'Unknown error'
                if "MoviePy" not in error_msg:
                    stability_result['message'] = error_msg
                else:
                    # If it's just a MoviePy error but the API test succeeded
                    if "Successfully connected to Stability API" in proc.stdout:
                        stability_result['status'] = 'success'
                        stability_result['message'] = 'Connection successful (MoviePy warnings can be ignored)'
                    else:
                        stability_result['message'] = 'API test failed'
    except Exception as e:
        stability_result['message'] = f'Error: {str(e)}'
    
    results['tests'].append(stability_result)
    
    # Overall status
    all_success = all(test['status'] == 'success' for test in results['tests'])
    results['overall_status'] = 'success' if all_success else 'failed'
    
    return results

def run_test_harness():
    """Run API tests from the command line"""
    logger.info("Running API test harness...")
    results = test_apis()
    
    # Print results in a formatted way
    print(f"\n=== API Test Results ({results['timestamp']}) ===")
    print(f"Overall Status: {results['overall_status'].upper()}")
    print("\nDetailed Results:")
    
    for test in results['tests']:
        print(f"- {test['api']}: {test['status'].upper()} - {test['message']}")
    
    # Return success if all tests passed
    return results['overall_status'] == 'success'

# Note: We're using subprocess to call the original podcast-to-video.py script
# rather than importing its components directly to avoid dependency issues with MoviePy

if __name__ == '__main__':
    # Check for test command
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        success = run_test_harness()
        sys.exit(0 if success else 1)
    
    # Create required directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    # Start the app on port 5050 to avoid conflicts
    app.run(debug=True, host='0.0.0.0', port=5050)
