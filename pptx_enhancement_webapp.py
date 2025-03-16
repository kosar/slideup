import os
import time
import uuid
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import signal
import atexit
import shutil
import openai
if hasattr(openai, "errors") and not hasattr(openai, "error"):
    openai.error = openai.errors
from env_utils import check_env_keys, store_original_env, restore_env

app = Flask(__name__, template_folder='templates')

# A simple dictionary to track jobs
jobs = {}

# Global flags and resources
is_shutting_down = False
cleanup_lock = threading.Lock()
active_threads = set()

# Directory for generated files
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

@app.route('/env_keys_check', methods=['GET'])
def env_keys_check():
    """Return a JSON object indicating which API keys are set in environment variables."""
    return check_env_keys()

@app.route('/')
def index():
    """
    Redirect root URL to the enhance page
    """
    return redirect(url_for('enhance_page'))

@app.route('/enhance', methods=['GET'])
def enhance_page():
    """
    Show the HTML page for enhancing PPTX files
    """
    return render_template('enhance.html')

@app.route('/enhance/upload', methods=['POST'])
def upload_pptx():
    """Handle PPTX file upload and start enhancement process."""
    session_id = str(uuid.uuid4())
    
    # Initialize job tracking
    jobs[session_id] = {
        "status": "queued",
        "logs": [],
        "start_time": None,
        "end_time": None,
        "elapsed_time": 0
    }
    
    try:
        # Store original environment
        original_env = store_original_env()
        jobs[session_id]['original_env'] = original_env
        
        # Get API keys and update environment
        for key, env_var in {
            'OPENAI_API_KEY': 'OPENAI_API_KEY',
            'STABILITY_API_KEY': 'STABILITY_API_KEY',
            'DEEPSEEK_API_KEY': 'DEEPSEEK_API_KEY'
        }.items():
            if value := request.form.get(key, ''):
                os.environ[env_var] = value
        
        # Get custom prompts
        stability_prompt = request.form.get('stability_prompt', '')
        deepseek_prompt = request.form.get('deepseek_prompt', '')
        jobs[session_id].update({
            'stability_prompt': stability_prompt,
            'deepseek_prompt': deepseek_prompt
        })
        
        # Handle file upload
        if 'pptx_file' not in request.files or not request.files['pptx_file'].filename:
            raise ValueError("No PPTX file provided")
        
        pptx_file = request.files['pptx_file']
        filename = secure_filename(pptx_file.filename)
        saved_path = os.path.join(GENERATED_FILES_DIR, f"{session_id}_{filename}")
        
        try:
            pptx_file.save(saved_path)
            if not os.path.exists(saved_path):
                raise FileNotFoundError(f"Failed to save file at {saved_path}")
        except Exception as e:
            raise ValueError(f"Failed to save uploaded file: {e}")
        
        # Update job tracking
        jobs[session_id].update({
            'file_path': saved_path,
            'orig_filename': filename
        })
        
        # Start enhancement thread
        t = threading.Thread(target=enhance_pptx_thread, args=(session_id,))
        t.daemon = True
        active_threads.add(t)
        t.start()
        
        return jsonify({"session_id": session_id})
        
    except Exception as e:
        if session_id in jobs:
            jobs[session_id]['status'] = "failed"
            jobs[session_id]['error'] = str(e)
        return jsonify({"error": str(e)}), 400

@app.route('/enhance/status/<session_id>', methods=['GET'])
def status(session_id):
    """
    Return JSON status and logs for the given session_id.
    """
    try:
        if session_id not in jobs:
            print(f"[DEBUG] Session not found: {session_id}")
            return jsonify({"error": "Session not found"}), 404

        job = jobs[session_id].copy()
        
        # Remove any non-JSON serializable objects
        if 'original_env' in job:
            del job['original_env']
            
        if job['status'] == 'complete':
            # Format elapsed time
            elapsed = timedelta(seconds=round(job['elapsed_time'], 2))
            hours, remainder = divmod(elapsed.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            fractional = seconds - int(seconds)
            job['elapsed_time_formatted'] = f"{int(hours)}:{int(minutes):02}:{int(seconds):02}.{int(fractional*100):02}"
        
        print(f"[DEBUG] Status for {session_id}: {job['status']}")
        return jsonify(job)
        
    except Exception as e:
        print(f"[ERROR] Error in status route: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/enhance/download/<session_id>', methods=['GET'])
def download_file(session_id):
    """
    Download the enhanced PPTX file from the session.
    """
    if session_id not in jobs:
        return jsonify({"error": "Session not found"}), 404
    if jobs[session_id]['status'] != 'complete':
        return jsonify({"error": "Job not complete"}), 400

    output_file = jobs[session_id]['output_file']
    if not output_file or not os.path.exists(output_file):
        return jsonify({"error": "Enhanced file not found"}), 404

    filename = os.path.basename(output_file)
    from flask import send_file
    return send_file(
        output_file,
        as_attachment=True,
        download_name=filename,
        mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation'
    )

@app.route('/help', methods=['GET'])
def help_page():
    """
    Show the help page documentation
    """
    return render_template('help.html')

def enhance_pptx_thread(session_id):
    """Thread function that handles the actual enhancement process."""
    try:
        jobs[session_id]['status'] = "running"
        jobs[session_id]['start_time'] = time.time()

        saved_path = jobs[session_id]['file_path']
        orig_filename = jobs[session_id]['orig_filename']
        filename_base, _ = os.path.splitext(orig_filename)
        output_file = os.path.join(GENERATED_FILES_DIR, f"{filename_base}_enhanced.pptx")
        jobs[session_id]['output_file'] = output_file
        stability_prompt = jobs[session_id].get('stability_prompt', None)

        # Import here to avoid circular imports
        from pptx2enhancedpptx import enhance_pptx
        
        if not os.path.exists(saved_path):
            raise FileNotFoundError(f"Input file not found: {saved_path}")
            
        # Pass the custom prompt to the enhance_pptx function
        enhance_pptx(saved_path, output_file, stability_prompt=stability_prompt)
        
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file was not created: {output_file}")
            
        jobs[session_id]['status'] = "complete"
        
    except Exception as e:
        print(f"[ERROR] Enhancement error: {str(e)}")
        jobs[session_id]['status'] = "failed"
        jobs[session_id]['error'] = str(e)
    finally:
        # Calculate elapsed time
        if jobs[session_id]['start_time'] is not None:
            jobs[session_id]['end_time'] = time.time()
            jobs[session_id]['elapsed_time'] = jobs[session_id]['end_time'] - jobs[session_id]['start_time']
        
        # Remove thread from registry
        current = threading.current_thread()
        if current in active_threads:
            active_threads.remove(current)
        
        # Restore environment variables
        if 'original_env' in jobs[session_id]:
            restore_env(jobs[session_id]['original_env'])
            del jobs[session_id]['original_env']

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
        print("[INFO] Starting PPTX enhancement webapp")
        print("[INFO] Main index page will be available at: http://localhost:5003/enhance")
        print("[INFO] Help page will be available at: http://localhost:5003/help")
        
        # Initialize the app
        with app.app_context():
            init_app()
        
        # Run without debug mode and reloader to ensure clean shutdown
        app.run(
            host='127.0.0.1',
            port=5003,
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

@app.teardown_appcontext
def teardown_app_context(exception):
    """Clean up resources when the application context ends."""
    if not is_shutting_down:
        cleanup_resources()