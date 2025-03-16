import os
import time
import uuid
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import shutil  # Import shutil for directory operations
import signal    # Import signal for handling termination
import sys       # Import sys to exit the program properly
import threading  # Import threading for lock management
import errno  # Import errno for error handling
import md2pptx  # Our existing module

from werkzeug.utils import secure_filename  # Add import for secure filename handling
from env_utils import check_env_keys, store_original_env, restore_env

app = Flask(__name__, template_folder='templates')

# Global dictionary to track job statuses
job_status = {}

# Global flags and resources
is_shutting_down = False
cleanup_lock = threading.Lock()
active_threads = set()

# Define the directory to store all generated and temporary files
GENERATED_FILES_DIR = os.path.join(os.getcwd(), 'generated_files')

def init_app():
    """Initialize the application and create necessary directories."""
    if not os.path.exists(GENERATED_FILES_DIR):
        os.makedirs(GENERATED_FILES_DIR)

def cleanup_resources():
    """Clean up resources during shutdown."""
    global is_shutting_down
    
    with cleanup_lock:
        if is_shutting_down:
            return
        is_shutting_down = True
    
    try:
        # Clean up the generated files directory if it exists
        if os.path.exists(GENERATED_FILES_DIR):
            try:
                shutil.rmtree(GENERATED_FILES_DIR, ignore_errors=True)
                print("[DEBUG] Cleaned up generated files directory")
            except Exception as e:
                print(f"[WARNING] Error cleaning up directory: {e}")
    except Exception as e:
        print(f"[ERROR] Cleanup error: {e}")

def upload_to_drive(file_path, file_name):
    """Upload PPTX to Google Drive folder."""
    print("[DEBUG] Uploading file to Google Drive:", file_path)
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if not folder_id:
        print("[DEBUG] GOOGLE_DRIVE_FOLDER_ID not set. Skipping Drive upload.")
        return None
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")
    if not os.path.exists(creds_path):
        print("[DEBUG] Google credentials not found. Skipping Drive upload.")
        return None

    service = build('drive', 'v3')  # Adjust if using different auth flow
    media = MediaFileUpload(file_path, mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation')
    file_metadata = {'name': file_name, 'parents': [folder_id]}
    uploaded_file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = uploaded_file.get('id')
    print(f"[DEBUG] File uploaded with file_id={file_id}")
    return file_id

def is_text(content):
    """Helper function to check if the content is text."""
    try:
        content.encode('utf-8')
        return True
    except UnicodeEncodeError:
        return False

def generate_pptx(uploaded_filename, markdown_text, session_id, add_notes, add_images_stability, stability_prompt, speaker_notes_prompt, deepseek_prompt=None):
    """Run the generation and store the result for user to download."""
    try:
        # Mark job as running
        job_status[session_id]['status'] = 'running'
        job_status[session_id]['logs'] = []

        def log(message):
            print(message)
            job_status[session_id]['logs'].append(message)

        log("Starting PPTX generation process.")

        # Record the start time
        job_status[session_id]['start_time'] = time.time()

        # Create unique filename within GENERATED_FILES_DIR
        now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        if uploaded_filename:
            output_filename = f"{uploaded_filename.split('.')[0]}_{now_str}_{session_id}.pptx"
        else:
            output_filename = f"slides_{now_str}_{session_id}.pptx"
        output_filepath = os.path.join(GENERATED_FILES_DIR, output_filename)

        # Write user markdown to a temp file within GENERATED_FILES_DIR
        tmp_md_file = os.path.join(GENERATED_FILES_DIR, f"temp_{session_id}.md")
        with open(tmp_md_file, "w") as f:
            f.write(markdown_text)

        log(f"Temporary markdown file created at {tmp_md_file}.")

        # Call our existing md2pptx logic with the output path
        md2pptx.main([
            "md2pptx.py",
            tmp_md_file,
            "--add-notes" if add_notes else "",
            "--add-images-stability" if add_images_stability else "",
            "--stability-prompt", stability_prompt if stability_prompt else "",
            "--speaker-notes-prompt", speaker_notes_prompt if speaker_notes_prompt else "",
            "--deepseek-prompt", deepseek_prompt if deepseek_prompt else ""
        ], alt_output=output_filepath)

        log(f"PPTX file generated at {output_filepath}.")

        # Upload to Google Drive
        drive_file_id = upload_to_drive(output_filepath, output_filename)
        if drive_file_id:
            job_status[session_id]['drive_file_id'] = drive_file_id
            log(f"Uploaded PPTX to Google Drive with file ID: {drive_file_id}.")

        # Record the end time and calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - job_status[session_id]['start_time']
        job_status[session_id]['end_time'] = end_time
        job_status[session_id]['elapsed_time'] = elapsed_time

        # Mark job as complete
        job_status[session_id]['status'] = 'complete'
        job_status[session_id]['pptx_file'] = output_filename
        job_status[session_id]['output_filename'] = output_filename
        log(f"Session {session_id} finished with output {output_filename}.")

    except Exception as e:
        log(f"Error in generate_pptx: {e}")
        job_status[session_id]['status'] = 'failed'
        job_status[session_id]['error'] = str(e)
    finally:
        # Calculate elapsed time
        if job_status[session_id]['start_time'] is not None:
            job_status[session_id]['end_time'] = time.time()
            job_status[session_id]['elapsed_time'] = job_status[session_id]['end_time'] - job_status[session_id]['start_time']

def get_status(session_id):
    """Get the status of a job."""
    if session_id not in job_status:
        return jsonify({"status": "unknown"}), 404
    # Include elapsed_time if job is complete
    response = job_status[session_id].copy()
    if response.get('status') == 'complete' and 'elapsed_time' in response:
        # Format elapsed_time into a more readable format with fractional seconds
        elapsed = timedelta(seconds=round(response['elapsed_time'], 2))
        # Format as HH:MM:SS.FF
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        fractional = seconds - int(seconds)
        response['elapsed_time_formatted'] = f"{int(hours)}:{int(minutes):02}:{int(seconds):02}.{int(fractional * 100):02}"
    return jsonify(response)

@app.route('/')
def index():
    """
    Show the main markdown to PPTX conversion page
    """
    return render_template('md2pptx.html')

@app.route('/index')
def old_index():
    """
    Redirect /index to root for backward compatibility
    """
    return redirect(url_for('index'))

@app.route('/env_keys_check', methods=['GET'])
def env_keys_check():
    """Return a JSON object indicating which API keys are set in environment variables."""
    return check_env_keys()

@app.route('/start_job', methods=['POST'])
def start_job():
    session_id = str(uuid.uuid4())
    user_uploaded_filename = ""
    
    # Initialize job status
    job_status[session_id] = {
        "status": "queued",
        "created": time.time(),
        "pptx_file": None,
        "drive_file_id": None,
        "logs": []
    }
    
    try:
        # Store original environment
        original_env = store_original_env()
        job_status[session_id]['original_env'] = original_env
        
        # Get API keys and update environment
        for key, env_var in {
            'openaiApiKey': 'OPENAI_API_KEY',
            'stabilityApiKey': 'STABILITY_API_KEY',
            'deepseekApiKey': 'DEEPSEEK_API_KEY'
        }.items():
            if value := request.form.get(key, ''):
                os.environ[env_var] = value
        
        # Handle file upload or text input
        if 'markdown_file' in request.files and request.files['markdown_file'].filename:
            file = request.files['markdown_file']
            filename = secure_filename(file.filename)
            user_uploaded_filename = filename
            file_path = os.path.join(GENERATED_FILES_DIR, filename)
            
            try:
                file.save(file_path)
                with open(file_path, 'r', encoding='utf-8') as f:
                    markdown_input = f.read()
                if not is_text(markdown_input):
                    raise ValueError("Uploaded file is not a text file")
                os.remove(file_path)
            except Exception as e:
                raise ValueError(f"Error processing uploaded file: {e}")
        else:
            markdown_input = request.form.get('markdown_input', '').strip()
            if not markdown_input:
                raise ValueError("No Markdown text provided")
        
        # Get other parameters
        add_notes = request.form.get('add_notes') == 'true'
        add_images = request.form.get('add_images_stability') == 'true'
        stability_prompt = request.form.get('stabilityPrompt', '')
        deepseek_prompt = request.form.get('deepseekPrompt', '')
        speaker_notes_prompt = request.form.get('speakerNotesPrompt', '')
        
        # Update job status
        job_status[session_id].update({
            'markdown_input': markdown_input,
            'add_notes': add_notes,
            'add_images_stability': add_images,
            'stability_prompt': stability_prompt,
            'deepseek_prompt': deepseek_prompt,
            'speaker_notes_prompt': speaker_notes_prompt
        })
        
        # Start processing thread
        t = threading.Thread(
            target=generate_pptx,
            args=(user_uploaded_filename, markdown_input, session_id, add_notes,
                  add_images, stability_prompt, speaker_notes_prompt, deepseek_prompt)
        )
        t.daemon = True
        active_threads.add(t)
        t.start()
        
        return jsonify({
            "session_id": session_id,
            "markdown_input": markdown_input
        })
        
    except Exception as e:
        if session_id in job_status:
            job_status[session_id]['status'] = 'failed'
            job_status[session_id]['error'] = str(e)
        return jsonify({"error": str(e)}), 400

@app.route('/status/<session_id>', methods=['GET'])
def status_route(session_id):
    return get_status(session_id)

@app.route('/logs/<session_id>', methods=['GET'])
def get_logs(session_id):
    if session_id not in job_status:
        return jsonify({"logs": [], "status": "unknown"}), 404
    return jsonify({"logs": job_status[session_id].get('logs', []), "status": job_status[session_id].get('status', 'unknown')})

@app.route('/download/<session_id>', methods=['GET'])
def download_file(session_id):
    if session_id not in job_status or not job_status[session_id].get('pptx_file'):
        return jsonify({"error": "File not found"}), 404
    pptx_file = job_status[session_id]['pptx_file']
    return jsonify({"download_url": f"/download_file/{session_id}", "filename": pptx_file})

@app.route('/download_file/<session_id>', methods=['GET'])
def download_file_actual(session_id):
    if session_id not in job_status or not job_status[session_id].get('pptx_file'):
        return jsonify({"error": "File not found"}), 404
    pptx_file = job_status[session_id]['pptx_file']
    return send_from_directory(
        directory=GENERATED_FILES_DIR,
        path=pptx_file,
        as_attachment=True,
        download_name=pptx_file
    )

@app.route('/help', methods=['GET'])
def help_page():
    return render_template('help.html')

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    global is_shutting_down
    if not is_shutting_down:  # Only handle signal once
        print("\n[DEBUG] Signal received. Cleaning up and exiting...")
        cleanup_resources()
        # Force exit all threads immediately
        print("[DEBUG] Force exiting...")
        os._exit(0)

if __name__ == '__main__':
    # Register the signal handler for SIGINT (Ctrl+C) and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("[INFO] Starting Markdown to PPTX webapp")
        print("[INFO] Main index page will be available at: http://localhost:5002/")
        print("[INFO] Help page will be available at: http://localhost:5002/help")
        
        # Initialize the app
        with app.app_context():
            init_app()
        
        # Run without debug mode and reloader to ensure clean shutdown
        app.run(
            host='127.0.0.1',
            port=5002,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down gracefully...")
        cleanup_resources()
        os._exit(0)
    except Exception as e:
        print(f"[ERROR] Server error: {e}")
        cleanup_resources()
        os._exit(1)