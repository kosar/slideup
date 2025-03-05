import os
import time
import uuid
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory
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

# Directory for generated files
GENERATED_FILES_DIR = os.path.join(os.getcwd(), 'generated_files')
if not os.path.exists(GENERATED_FILES_DIR):
    os.makedirs(GENERATED_FILES_DIR)

# Cleanup function to remove temporary files
def cleanup_files():
    print("Cleaning up temporary files in", GENERATED_FILES_DIR)
    if os.path.exists(GENERATED_FILES_DIR):
        try:
            shutil.rmtree(GENERATED_FILES_DIR)
            os.makedirs(GENERATED_FILES_DIR)
        except Exception as e:
            print("Error cleaning up temporary files:", e)

# Register cleanup function for normal exit
atexit.register(cleanup_files)

# Robust signal handler for SIGINT and SIGTERM
def shutdown_handler(signal_received, frame):
    cleanup_files()
    exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

@app.route('/env_keys_check', methods=['GET'])
def env_keys_check():
    """Return a JSON object indicating which API keys are set in environment variables."""
    return check_env_keys()

@app.route('/enhance', methods=['GET'])
def enhance_page():
    """
    Show the HTML page for enhancing PPTX files
    """
    return render_template('enhance.html')

@app.route('/enhance/upload', methods=['POST'])
def upload_pptx():
    """
    Receive the PPTX file, store optional API keys, and start the enhancement process.
    Return a JSON response with a session_id for tracking.
    """
    # Store original environment variables
    original_env = store_original_env()

    try:
        # Optional: override environment variables if form fields are set
        openai_key = request.form.get('OPENAI_API_KEY', '')
        stability_key = request.form.get('STABILITY_API_KEY', '')
        deepseek_key = request.form.get('DEEPSEEK_API_KEY', '')
        stability_prompt = request.form.get('stability_prompt', '')  # Get the custom stability prompt
        deepseek_prompt = request.form.get('deepseek_prompt', '')  # Get the custom deepseek prompt
        
        if openai_key:
            os.environ['OPENAI_API_KEY'] = openai_key
        if stability_key:
            os.environ['STABILITY_API_KEY'] = stability_key
        if deepseek_key:
            os.environ['DEEPSEEK_API_KEY'] = deepseek_key

        # Debug log
        print(f"[DEBUG] Form data keys: {list(request.form.keys())}")
        print(f"[DEBUG] Files keys: {list(request.files.keys())}")

        if 'pptx_file' not in request.files or request.files['pptx_file'].filename == '':
            # Restore original environment variables before returning
            restore_env(original_env)
            print("[ERROR] No PPTX file provided")
            return jsonify({"error": "No PPTX file provided"}), 400

        pptx_file = request.files['pptx_file']
        filename = secure_filename(pptx_file.filename)
        
        # Check if GENERATED_FILES_DIR exists, create if not
        if not os.path.exists(GENERATED_FILES_DIR):
            os.makedirs(GENERATED_FILES_DIR)
            print(f"[DEBUG] Created directory: {GENERATED_FILES_DIR}")
        
        session_id = str(uuid.uuid4())
        jobs[session_id] = {
            "status": "queued",
            "logs": [],
            "start_time": None,
            "end_time": None,
            "file_path": "",
            "output_file": "",
            "elapsed_time": 0,
            "stability_prompt": stability_prompt,
            "deepseek_prompt": deepseek_prompt,
            "original_env": original_env
        }
        jobs[session_id]['orig_filename'] = filename
        jobs[session_id]['logs'].append(f"Received file: {filename}")

        # Save uploaded PPTX
        saved_path = os.path.join(GENERATED_FILES_DIR, f"{session_id}_{filename}")
        try:
            pptx_file.save(saved_path)
            jobs[session_id]['file_path'] = saved_path
            jobs[session_id]['logs'].append(f"File saved successfully to {saved_path}")
            print(f"[DEBUG] File saved to {saved_path}")
            
            # Check if file exists after saving
            if not os.path.exists(saved_path):
                raise FileNotFoundError(f"Failed to save file at {saved_path}")
                
        except Exception as e:
            print(f"[ERROR] File save error: {str(e)}")
            jobs[session_id]['logs'].append(f"Error saving file: {str(e)}")
            jobs[session_id]['status'] = "failed"
            return jsonify({"error": f"Failed to save uploaded file: {str(e)}"}), 500

        # Spin off a thread to enhance
        try:
            t = threading.Thread(target=enhance_pptx_thread, args=(session_id,))
            t.start()
            print(f"[DEBUG] Started enhancement thread for session {session_id}")
            return jsonify({"session_id": session_id})
        except Exception as e:
            print(f"[ERROR] Thread start error: {str(e)}")
            jobs[session_id]['logs'].append(f"Error starting enhancement process: {str(e)}")
            jobs[session_id]['status'] = "failed"
            return jsonify({"error": f"Failed to start enhancement process: {str(e)}"}), 500
            
    except Exception as e:
        # Restore environment variables if there was an error
        restore_env(original_env)
        print(f"[ERROR] Unhandled exception in upload_pptx: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def enhance_pptx_thread(session_id):
    """
    Thread function that handles the actual enhancement process.
    """
    try:
        print(f"[DEBUG] Starting enhancement for session {session_id}")
        jobs[session_id]['status'] = "running"
        jobs[session_id]['start_time'] = time.time()

        saved_path = jobs[session_id]['file_path']
        orig_filename = jobs[session_id]['orig_filename']
        filename_base, _ = os.path.splitext(orig_filename)
        output_file = os.path.join(GENERATED_FILES_DIR, f"{filename_base}_enhanced.pptx")
        jobs[session_id]['output_file'] = output_file
        stability_prompt = jobs[session_id].get('stability_prompt', None)

        logs = jobs[session_id]['logs']
        logs.append("=== Enhancement Process Starting ===")
        logs.append("Enhancement in progress. Please wait...")
        
        # Import here to avoid circular imports
        try:
            from pptx2enhancedpptx import enhance_pptx
            
            # Verify that the input file exists before processing
            if not os.path.exists(saved_path):
                raise FileNotFoundError(f"Input file not found: {saved_path}")
                
            # Pass the custom prompt to the enhance_pptx function
            enhance_pptx(saved_path, output_file, stability_prompt=stability_prompt)
            
            # Verify the output file was created
            if not os.path.exists(output_file):
                raise FileNotFoundError(f"Output file was not created: {output_file}")
                
            time.sleep(2)
            total_time = time.time() - jobs[session_id]['start_time']
            logs.append("=== Enhancement Process Completed ===")
            logs.append(f"Total Execution Time: {total_time:.2f} seconds")
            jobs[session_id]['status'] = "complete"
            
        except ImportError as e:
            logs.append(f"Error: Module pptx2enhancedpptx not found. {str(e)}")
            jobs[session_id]['status'] = "failed"
            print(f"[ERROR] Import error: {str(e)}")
            
        except FileNotFoundError as e:
            logs.append(f"Error: {str(e)}")
            jobs[session_id]['status'] = "failed"
            print(f"[ERROR] File error: {str(e)}")
            
        except Exception as e:
            import traceback
            debug_info = traceback.format_exc()
            logs.append("Error during enhancement: " + str(e))
            logs.append("Debug info: " + debug_info)
            jobs[session_id]['status'] = "failed"
            print(f"[ERROR] Enhancement error: {str(e)}")
            print(debug_info)
            
    except Exception as e:
        print(f"[ERROR] Unhandled exception in enhance_pptx_thread: {str(e)}")
        if session_id in jobs:
            jobs[session_id]['status'] = "failed"
            jobs[session_id]['logs'].append(f"Unhandled error: {str(e)}")
    finally:
        # Restore original environment variables
        if session_id in jobs and 'original_env' in jobs[session_id]:
            restore_env(jobs[session_id]['original_env'])
            # Remove sensitive data
            del jobs[session_id]['original_env']

        if session_id in jobs:
            jobs[session_id]['end_time'] = time.time()
            if jobs[session_id]['start_time'] is not None:  # Check to avoid errors
                jobs[session_id]['elapsed_time'] = jobs[session_id]['end_time'] - jobs[session_id]['start_time']

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

if __name__ == '__main__':
    print("[INFO] Starting PPTX enhancement webapp")
    app.run(debug=True, port=5003)