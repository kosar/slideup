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
from datetime import datetime
import traceback

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
    "visual_prompt": "A detailed visual prompt for generating an image that represents this content, with an eye towards interesting facts and specifics from the content with key points identified (as many as you see fit).",
    "key_points": ["Key point 1", "Key point 2", "Key point 3", "Key point N"],
    "epoch_time": "A description of the period in time and approximate location down to the continent level to help set the period for context and will be used by downstream processors."
}

The visual prompt should be detailed and specific, focusing on visual elements that would make a compelling image. 
Include details like style, composition, colors, and elements to include in the image. Be sure to include a time period for which the image should be set, so it is logically aligned with the topic and the period in time that the topic is set in as appropriate and possible to discern from this content. If there are hints you can provide that further sets the tone and mood of the setting that helps the image creation be true to the intention of the discussion so it feels natural to the viewer.""",
    'visual_generation': "A high quality, detailed image that uses creative imagery that is highly realistic especially when it comes to humans, being sure the human depictions are true to the natural human form and being very careful to depict imagery set in a specific time period that is aligned with the description below so it fits together naturally between the narrative and the visual imagery, of the following scene: "
}

# Background processes tracking
processing_tasks = {}
processing_processes = {}  # Track actual subprocess handles

# Default cost tracking structure
DEFAULT_COST_DATA = {
    'total_cost': 0.0,
    'api_breakdown': {
        'openai': {
            'chat': 0.0,
            'transcription': 0.0
        },
        'stability': {
            'image': 0.0
        }
    }
}

# Add datetime filter for templates
@app.template_filter('datetime')
def format_datetime(value):
    if value is None:
        return ""
    try:
        if isinstance(value, (int, float)):
            # Handle timestamp
            return datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(value, str):
            # Try to parse ISO format string
            return datetime.fromisoformat(value).strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(value, datetime):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        # Try to convert to float if it's a string representation of a number
        if isinstance(value, str) and value.replace('.', '').isdigit():
            return datetime.fromtimestamp(float(value)).strftime('%Y-%m-%d %H:%M:%S')
        return str(value)
    except (ValueError, TypeError) as e:
        logger.warning(f"Error formatting datetime value '{value}': {str(e)}")
        return str(value)

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
        # Initialize task status
        processing_tasks[task_id] = {
            'status': 'starting',
            'progress': 0,
            'message': 'Initializing processing',
            'logs': [],
            'start_time': time.time(),
            'output_file': None,
            'cost_data': DEFAULT_COST_DATA.copy()  # Initialize with default cost data
        }
        
        # Set up directory paths
        webapp_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(webapp_dir, 'uploads', 'temp_' + task_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Update status
        processing_tasks[task_id].update({
            'status': 'processing',
            'progress': 10,
            'message': 'Processing audio...'
        })
        
        # Call the podcast-to-video.py script directly
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
        
        # Start the subprocess with pipe for output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Store process for potential cancellation
        processing_processes[task_id] = process
        
        # Read output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output = output.strip()
                logger.info(f"Process output: {output}")
                
                # Check for cost tracking information in output
                if "COST_DATA:" in output:
                    try:
                        # Extract cost data JSON
                        cost_json = output.split("COST_DATA:", 1)[1].strip()
                        cost_data = json.loads(cost_json)
                        processing_tasks[task_id]['cost_data'] = cost_data
                        logger.info(f"Updated cost data: ${cost_data['total_cost']:.4f} total")
                    except Exception as e:
                        logger.error(f"Error parsing cost data: {str(e)}")
                
                # Update status based on output
                if "Starting audio transcription" in output:
                    processing_tasks[task_id].update({
                        'progress': 20,
                        'message': 'Transcribing audio...'
                    })
                elif "Transcription completed" in output:
                    processing_tasks[task_id].update({
                        'progress': 40,
                        'message': 'Enhancing segments...'
                    })
                elif "Enhanced" in output and "segments" in output:
                    processing_tasks[task_id].update({
                        'progress': 60,
                        'message': 'Generating visuals...'
                    })
                elif "Generated" in output and "visuals" in output:
                    processing_tasks[task_id].update({
                        'progress': 80,
                        'message': 'Creating final video...'
                    })
                elif "Received signal" in output and "shutting down" in output:
                    # Process is shutting down due to signal (likely from cancellation)
                    logger.info(f"Process for task {task_id} is shutting down gracefully")
                
                # Add to logs
                processing_tasks[task_id]['logs'].append(output)
        
        # Get the return code
        return_code = process.poll()
        
        # Check if the task was cancelled first
        if task_id in processing_tasks and processing_tasks[task_id].get('status') in ['cancelled', 'cancelling']:
            logger.info(f"Task {task_id} was cancelled - not checking for output file")
            return
        
        # Get error output in case we need it
        error_output = process.stderr.read()
        
        # Check for specific error messages in logs that indicate failure
        error_message = None
        for log in processing_tasks[task_id]['logs']:
            if "ERROR - Error during transcription:" in log:
                error_parts = log.split("ERROR - Error during transcription:", 1)
                if len(error_parts) > 1:
                    error_message = f"Transcription error: {error_parts[1].strip()}"
                    break
            elif "ERROR -" in log:
                error_parts = log.split("ERROR -", 1)
                if len(error_parts) > 1:
                    error_message = error_parts[1].strip()
                    break
                
        # Only check for successful completion if not cancelled and no error found
        if return_code == 0 and not error_message:
            # Check if output file exists
            if os.path.exists(output_file):
                processing_tasks[task_id].update({
                    'status': 'completed',
                    'progress': 100,
                    'message': 'Processing completed successfully',
                    'output_file': os.path.basename(output_file),
                    'end_time': time.time()
                })
            else:
                # Output file doesn't exist despite return code 0
                # This could happen if the process exits normally but fails to produce output
                error_message = f"Output file not found: {os.path.basename(output_file)}"
        
        # If we have an error, either from logs or missing output file
        if error_message or return_code != 0:
            processing_tasks[task_id].update({
                'status': 'failed',
                'progress': 0,
                'message': error_message or f"Process failed with return code {return_code}: {error_output}",
                'end_time': time.time()
            })
            logger.error(f"Processing failed: {processing_tasks[task_id]['message']}")
            # Note: We don't raise an exception here anymore, to keep the cost data
            
    except FileNotFoundError as e:
        # Special handling for file not found - check if cancelled first
        if task_id in processing_tasks and processing_tasks[task_id].get('status') in ['cancelled', 'cancelling']:
            logger.info(f"Ignoring FileNotFoundError for cancelled task {task_id}")
            return
        
        # If not cancelled and not already failed, log the error and update status
        logger.error(f"Error in background processing: {str(e)}")
        if task_id in processing_tasks and processing_tasks[task_id].get('status') not in ['cancelled', 'failed']:
            processing_tasks[task_id].update({
                'status': 'failed',
                'progress': 0,
                'message': f'Processing failed: {str(e)}',
                'end_time': time.time()
            })
            
    except Exception as e:
        logger.error(f"Error in background processing: {str(e)}")
        # Check if the task was cancelled or already failed before setting failed status
        if task_id in processing_tasks and processing_tasks[task_id].get('status') not in ['cancelled', 'failed']:
            processing_tasks[task_id].update({
                'status': 'failed',
                'progress': 0,
                'message': f'Processing failed: {str(e)}',
                'end_time': time.time()
            })
    finally:
        # Remove process from tracking dict when done
        if task_id in processing_processes:
            processing_processes.pop(task_id, None)

def cancel_task(task_id):
    """Cancel a running processing task."""
    logger.info(f"Cancellation requested for task {task_id}")
    
    if task_id not in processing_tasks:
        logger.warning(f"Cannot cancel: Task {task_id} not found")
        return False, "Task not found"
    
    if processing_tasks[task_id]['status'] not in ['starting', 'processing']:
        logger.warning(f"Cannot cancel: Task {task_id} is already {processing_tasks[task_id]['status']}")
        return False, f"Task is {processing_tasks[task_id]['status']}"
    
    # Update status to cancelling
    processing_tasks[task_id].update({
        'status': 'cancelling',
        'message': 'Cancelling processing...'
    })
    
    # Try to terminate the process
    success = False
    try:
        # Check if we have a process handle for this task
        if task_id in processing_processes:
            process = processing_processes[task_id]
            logger.info(f"Terminating process for task {task_id}")
            
            # Terminate process
            process.terminate()
            
            # Wait a short time for graceful termination
            for _ in range(5):
                if process.poll() is not None:
                    break
                time.sleep(0.5)
                
            # Force kill if still running
            if process.poll() is None:
                logger.warning(f"Process for task {task_id} did not terminate gracefully, forcing kill")
                process.kill()
            
            # Clean up process reference - use pop with default to avoid KeyError
            # The key might be removed by another thread between our check and deletion
            processing_processes.pop(task_id, None)
        else:
            logger.warning(f"No process found for task {task_id}, updating status only")
        
        # Update status - do this whether or not we found a process
        # (the process might have completed or failed already)
        processing_tasks[task_id].update({
            'status': 'cancelled',
            'progress': 0,
            'message': 'Processing cancelled by user',
            'end_time': time.time()
        })
        
        # Add log entry
        processing_tasks[task_id]['logs'].append("Processing cancelled by user.")
        
        success = True
            
    except Exception as e:
        error_message = f"Error during cancellation: {str(e)}"
        logger.error(f"Error cancelling task {task_id}: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Even if there's an error, mark the task as cancelled for the UI
        # This ensures the user gets a reasonable response even if cleanup fails
        processing_tasks[task_id].update({
            'status': 'cancelled',
            'message': 'Processing cancelled (with cleanup errors)',
            'end_time': time.time()
        })
        
        # Add log entry about the error
        processing_tasks[task_id]['logs'].append(f"Error during cancellation cleanup: {str(e)}")
        
        return True, "Task cancelled (with cleanup errors)"
    
    return success, "Task cancelled successfully"

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
    logger.info("Upload request received")
    logger.info(f"Request files: {request.files}")
    logger.info(f"Request form data: {request.form}")
    
    if 'file' not in request.files:
        logger.error("No file part in the request")
        flash('No file part', 'error')
        return redirect(url_for('index'))
        
    file = request.files['file']
    logger.info(f"File received: {file.filename}")
    
    if file.filename == '':
        logger.error("No file selected")
        flash('No file selected', 'error')
        return redirect(url_for('index'))
        
    if file and allowed_file(file.filename):
        # Create unique task ID
        task_id = str(uuid.uuid4())
        logger.info(f"Created task ID: {task_id}")
        
        try:
            # Save the file
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
            logger.info(f"Saving file to: {file_path}")
            file.save(file_path)
            logger.info(f"File saved successfully. Size: {os.path.getsize(file_path)} bytes")
            
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
            logger.info("Starting background processing")
            processing_thread = threading.Thread(
                target=process_audio_file_background,
                args=(task_id, file_path, output_file, limit_to_one_minute, non_interactive, enhance_prompt, visual_prompt)
            )
            processing_thread.daemon = True
            processing_thread.start()
            logger.info("Background processing thread started")
            
            # Redirect to status page
            return redirect(url_for('status', task_id=task_id))
            
        except Exception as e:
            logger.error(f"Error during file upload processing: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            flash(f'Error processing upload: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    logger.error(f"Invalid file type: {file.filename}")
    flash('Invalid file type. Please upload an MP3 or WAV file.', 'error')
    return redirect(url_for('index'))

@app.route('/status/<task_id>')
def status(task_id):
    """Status page for a specific task"""
    task_status = get_task_status(task_id)
    if task_status is None:
        return render_template('status.html', task_id=task_id, error="Task not found")
    
    # Create a copy of task_status to avoid modifying the original
    formatted_status = task_status.copy()
    
    # Only convert timestamps if they are not already datetime objects
    if 'start_time' in formatted_status and not isinstance(formatted_status['start_time'], datetime):
        try:
            formatted_status['start_time'] = float(formatted_status['start_time'])
        except (ValueError, TypeError):
            formatted_status['start_time'] = None
            
    if 'end_time' in formatted_status and not isinstance(formatted_status['end_time'], datetime):
        try:
            formatted_status['end_time'] = float(formatted_status['end_time'])
        except (ValueError, TypeError):
            formatted_status['end_time'] = None
    
    return render_template('status.html', task_id=task_id, status=formatted_status)

@app.route('/api/status/<task_id>')
def api_status(task_id):
    """API endpoint for getting task status"""
    task_status = get_task_status(task_id)
    if task_status['status'] == 'unknown':
        return jsonify({'error': 'Task not found'}), 404
    
    # Ensure all necessary fields are present
    response_data = {
        'status': task_status.get('status', 'unknown'),
        'progress': task_status.get('progress', 0),
        'message': task_status.get('message', ''),
        'logs': task_status.get('logs', []),
        'start_time': task_status.get('start_time'),
        'output_file': task_status.get('output_file'),
        'cost_data': task_status.get('cost_data', DEFAULT_COST_DATA.copy())
    }
    
    return jsonify(response_data)

@app.route('/download/<task_id>')
def download(task_id):
    """Download the generated video file"""
    task_status = get_task_status(task_id)
    if task_status.get('status') == 'completed' and 'output_file' in task_status:
        output_file = task_status['output_file']
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], output_file)
        
        if os.path.exists(file_path):
            return send_from_directory(
                app.config['OUTPUT_FOLDER'],
                output_file,
                as_attachment=True,
                download_name=output_file
            )
        else:
            flash('Output file not found.', 'error')
    else:
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

@app.route('/api/cancel/<task_id>', methods=['POST'])
def api_cancel_task(task_id):
    """API endpoint for cancelling a task"""
    try:
        # Check if task exists
        if task_id not in processing_tasks:
            return jsonify({'status': 'error', 'message': 'Task not found'}), 404
            
        # Task exists but might be in a state that can't be cancelled
        if processing_tasks[task_id]['status'] not in ['starting', 'processing']:
            current_status = processing_tasks[task_id]['status']
            return jsonify({
                'status': current_status,
                'message': f'Task is already {current_status}'
            }), 200
        
        # Log the cancellation request
        logger.info(f"API request to cancel task {task_id}")
        
        # Attempt to cancel the task
        success, message = cancel_task(task_id)
        
        # Always check the actual status after attempting cancellation
        actual_status = processing_tasks[task_id]['status']
        actual_message = processing_tasks[task_id]['message']
        
        if actual_status == 'cancelled':
            # If the task is marked as cancelled in the data structure, report success
            # even if there were errors during the cleanup process
            return jsonify({'status': 'cancelled', 'message': actual_message}), 200
        elif success:
            # This case should be covered by the above check, but just in case
            return jsonify({'status': 'cancelled', 'message': message}), 200
        else:
            # If cancellation failed but didn't raise an exception
            return jsonify({'status': 'error', 'message': message}), 400
    except Exception as e:
        # Log the unexpected error
        logger.error(f"Unexpected error in cancel API: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Try to update the task status if possible
        try:
            if task_id in processing_tasks:
                # Even if there's an API error, try to mark the task as cancelled
                # to give a better user experience
                processing_tasks[task_id].update({
                    'status': 'cancelled',
                    'message': 'Processing cancelled (with errors)',
                    'end_time': time.time()
                })
                return jsonify({'status': 'cancelled', 'message': 'Task cancelled but encountered errors'}), 200
        except:
            # If even that fails, just return the original error
            pass
            
        return jsonify({'status': 'error', 'message': 'Server error while cancelling task'}), 500

@app.route('/api/time_limit')
def get_time_limit():
    """Get the time limit in minutes from the podcast-to-video.py file"""
    try:
        # Path to the podcast-to-video.py script
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "podcast-to-video.py")
        
        # Use grep to extract the TIME_LIMIT_SECONDS constant
        result = subprocess.run(
            ['grep', '-A', '1', 'TIME_LIMIT_SECONDS', script_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Error reading time limit from podcast-to-video.py: {result.stderr}")
            return jsonify({"minutes": 1, "seconds": 60})
        
        # Extract the time limit value using regex
        import re
        match = re.search(r'TIME_LIMIT_SECONDS\s*=\s*(\d+)', result.stdout)
        if match:
            seconds = int(match.group(1))
            minutes = seconds / 60
            return jsonify({"minutes": minutes, "seconds": seconds})
        else:
            logger.error(f"Could not find TIME_LIMIT_SECONDS in: {result.stdout}")
            return jsonify({"minutes": 1, "seconds": 60})
    
    except Exception as e:
        logger.error(f"Error getting time limit: {str(e)}")
        return jsonify({"minutes": 1, "seconds": 60})

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
