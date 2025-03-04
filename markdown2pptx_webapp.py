import os
import time
import uuid
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import shutil  # Import shutil for directory operations
import signal    # Import signal for handling termination
import sys       # Import sys to exit the program properly
import threading  # Import threading for lock management
import errno  # Import errno for error handling

# ...existing code...
import md2pptx  # Our existing module
# ...existing code...

from werkzeug.utils import secure_filename  # Add import for secure filename handling

app = Flask(__name__, template_folder='templates')

# Global dictionary to track job statuses
job_status = {}

# Periodic cleanup
SESSION_TIMEOUT = 3600  # 1 hour

def cleanup_sessions():
    now = time.time()
    to_remove = []
    for sid, data in job_status.items():
        if now - data['created'] > SESSION_TIMEOUT:
            to_remove.append(sid)
    for sid in to_remove:
        del job_status[sid]
    threading.Timer(300, cleanup_sessions).start()

cleanup_sessions()

# Define the directory to store all generated and temporary files
GENERATED_FILES_DIR = os.path.join(os.getcwd(), 'generated_files')

# Ensure the directory exists
if not os.path.exists(GENERATED_FILES_DIR):
    os.makedirs(GENERATED_FILES_DIR)

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

def generate_pptx(uploaded_filename, markdown_text, session_id, add_notes, add_images_stability, stability_prompt, speaker_notes_prompt):
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
            output_filename=f"{uploaded_filename.split(".")[:1][0]}_{now_str}_{session_id}.pptx"
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
            "--speaker-notes-prompt", speaker_notes_prompt if speaker_notes_prompt else ""
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
        job_status[session_id]['output_filename'] = output_filename  # Added line
        log(f"Session {session_id} finished with output {output_filename}.")

    except Exception as e:
        log(f"Error in generate_pptx: {e}")
        job_status[session_id]['status'] = 'failed'
        job_status[session_id]['error'] = str(e)

@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/start_job', methods=['POST'])
def start_job():
    session_id = str(uuid.uuid4())  # Move session_id generation to the beginning
    user_uploaded_filename=""

    # Initialize the job_status entry before any operations
    job_status[session_id] = {
        "status": "queued",
        "created": time.time(),
        "pptx_file": None,
        "drive_file_id": None,
        "logs": [],
        "markdown_input": "",  # Will be updated later
        "add_notes": False,    # Will be updated later
        "add_images_stability": False  # Will be updated later
    }

    if 'markdown_file' in request.files and request.files['markdown_file'].filename != '':
        # Handle file upload
        file = request.files['markdown_file']
        filename = secure_filename(file.filename)
        user_uploaded_filename=filename
        file_path = os.path.join(GENERATED_FILES_DIR, filename)
        try:
            job_status[session_id]['logs'].append(f"Uploading file: {filename}.")
            file.save(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                markdown_input = f.read()
            # Basic check to see if the file is text
            if not is_text(markdown_input):
                job_status[session_id]['logs'].append("Uploaded file is not a text file.")
                job_status[session_id]['status'] = 'failed'
                job_status[session_id]['error'] = 'Uploaded file is not a text file.'
                return jsonify({"error": "Uploaded file is not a text file."}), 400
            job_status[session_id]['logs'].append("File uploaded and read successfully.")
            os.remove(file_path)
        except Exception as e:
            job_status[session_id]['logs'].append(f"Error processing uploaded file: {e}")
            job_status[session_id]['status'] = 'failed'
            job_status[session_id]['error'] = f"Error processing uploaded file: {e}"
            return jsonify({"error": f"Error processing uploaded file: {e}"}), 400
    else:
        # Fallback to textarea input
        markdown_input = request.form.get('markdown_input', '')

    add_notes = request.form.get('add_notes') == 'true'
    add_images = request.form.get('add_images_stability') == 'true'

    # Update the initialized job_status entry with actual data
    job_status[session_id]['markdown_input'] = markdown_input
    job_status[session_id]['add_notes'] = add_notes
    job_status[session_id]['add_images_stability'] = add_images

    if not markdown_input.strip():
        return jsonify({"error": "No Markdown text provided"}), 400

    job_status[session_id]['logs'].append("Job queued.")

    stability_prompt = request.form.get('stabilityPrompt', '')
    speaker_notes_prompt = request.form.get('speakerNotesPrompt', '')
    t = threading.Thread(target=generate_pptx, args=(user_uploaded_filename, markdown_input, session_id, add_notes, add_images, stability_prompt, speaker_notes_prompt))
    t.start()

    return jsonify({
        "session_id": session_id,
        "markdown_input": markdown_input
    })

def get_status(session_id):
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
    return jsonify({"download_url": f"/download_file/{session_id}", "filename": pptx_file})  # Modified line

@app.route('/download_file/<session_id>', methods=['GET'])
def download_file_actual(session_id):
    if session_id not in job_status or not job_status[session_id].get('pptx_file'):
        return jsonify({"error": "File not found"}), 404
    pptx_file = job_status[session_id]['pptx_file']
    return send_from_directory(
        directory=GENERATED_FILES_DIR,
        path=pptx_file,
        as_attachment=True,
        download_name=pptx_file  # Updated parameter
    )

# Initialize a lock and a flag to prevent multiple cleanups
cleanup_lock = threading.Lock()
is_cleaning_up = False

# Define the onexc handler for shutil.rmtree
def ignore_errors(func, path, exc_info):
    if exc_info[1].errno == errno.ENOENT:
        # File or directory already deleted, ignore
        pass
    else:
        # Re-raise the exception for other errors
        raise

# Modify the cleanup_generated_files function
def cleanup_generated_files():
    global is_cleaning_up
    with cleanup_lock:
        if is_cleaning_up:
            return  # Cleanup already in progress or completed
        is_cleaning_up = True
    if os.path.exists(GENERATED_FILES_DIR):
        try:
            shutil.rmtree(GENERATED_FILES_DIR, onexc=ignore_errors)  # Changed onerror to onexc
            print(f"[DEBUG] Cleaned up generated files directory: {GENERATED_FILES_DIR}")
        except Exception as e:
            print(f"[ERROR] Failed to clean up directory {GENERATED_FILES_DIR}: {e}")

# Update the signal_handler function to use the enhanced cleanup
def signal_handler(sig, frame):
    print("\n[DEBUG] Signal received. Cleaning up and exiting...")
    cleanup_generated_files()
    sys.exit(0)

# Register the signal handler for SIGINT (Ctrl+C) and SIGTERM
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5002)
    except:
        cleanup_generated_files()
        raise