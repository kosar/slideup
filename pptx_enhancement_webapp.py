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
# ...existing code imports if needed...

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

@app.route('/index3', methods=['GET'])
def index3():
    """
    Show a simple HTML upload page where the user can:
    1) Enter API keys or rely on environment variables
    2) Upload a PPTX file
    3) See logs and status updates
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PPTX Enhancer</title>
        <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto:400,500">
        <style>
            body {
                font-family: 'Roboto', sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #74ABE2, #5563DE);
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                color: #333;
            }
            .card {
                background: #fff;
                padding: 2em;
                border-radius: 8px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                max-width: 600px;  // updated from 400px to 600px for more horizontal space
                width: 100%;
            }
            h1 {
                text-align: center;
                color: #444;
            }
            label {
                display: block;
                margin-top: 1em;
                font-weight: 500;
            }
            input[type="text"], input[type="file"] {
                margin-top: 0.5em;
                padding: 0.7em;
                width: 100%;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            button {
                margin-top: 1em;
                width: 100%;
                padding: 0.8em;
                background: #5563DE;
                color: #fff;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                transition: background 0.3s ease;
            }
            button:hover:not(.disabled) {
                background: #3F53B5;
            }
            .logs {
                margin-top: 1em;
                padding: 1em;
                background: #fafafa;
                border: 1px solid #ddd;
                max-height: 200px;
                overflow-y: auto;
                white-space: pre-line;
                border-radius: 4px;
            }
            .status {
                font-weight: bold;
            }
            .download-button {
                margin-top: 1em;
                width: 100%;
                padding: 0.8em;
                background: #388E3C;
                color: #fff;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                transition: background 0.3s ease;
            }
            .download-button:hover {
                background: #2E7D32;
            }
            .disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Enhance PPTX File (Index 3)</h1>
            <form id="uploadForm" enctype="multipart/form-data">
                <label>OpenAI API Key (optional):</label>
                <input type="text" name="OPENAI_API_KEY" id="OPENAI_API_KEY">
                <label>Stability API Key (optional):</label>
                <input type="text" name="STABILITY_API_KEY" id="STABILITY_API_KEY">
                <label>Deepseek API Key (optional):</label>
                <input type="text" name="DEEPSEEK_API_KEY" id="DEEPSEEK_API_KEY" placeholder="temporarily offline">
                <label>Select PPTX File:</label>
                <input type="file" name="pptx_file">
                <button id="submitButton" type="submit">Upload & Enhance</button>
            </form>
            <p>Job Status: <span id="jobStatus" class="status"></span></p>
            <div class="logs" id="logsContainer"></div>
            <button class="download-button" id="downloadButton" style="display: none;">Download Enhanced PPTX</button>
        </div>
        <script>
        // On page load, restore API keys if available
        window.onload = function() {
            const openaiField = document.getElementById('OPENAI_API_KEY');
            const stabilityField = document.getElementById('STABILITY_API_KEY');
            const deepseekField = document.getElementById('DEEPSEEK_API_KEY');
            if (sessionStorage.getItem('OPENAI_API_KEY')) {
                openaiField.value = sessionStorage.getItem('OPENAI_API_KEY');
            }
            if (sessionStorage.getItem('STABILITY_API_KEY')) {
                stabilityField.value = sessionStorage.getItem('STABILITY_API_KEY');
            }
            if (sessionStorage.getItem('DEEPSEEK_API_KEY')) {
                deepseekField.value = sessionStorage.getItem('DEEPSEEK_API_KEY');
            }
        };

        // Save API keys whenever their value changes
        document.getElementById('OPENAI_API_KEY').addEventListener('input', function() {
            sessionStorage.setItem('OPENAI_API_KEY', this.value);
        });
        document.getElementById('STABILITY_API_KEY').addEventListener('input', function() {
            sessionStorage.setItem('STABILITY_API_KEY', this.value);
        });
        document.getElementById('DEEPSEEK_API_KEY').addEventListener('input', function() {
            sessionStorage.setItem('DEEPSEEK_API_KEY', this.value);
        });

        const form = document.getElementById('uploadForm');
        const jobStatus = document.getElementById('jobStatus');
        const logsContainer = document.getElementById('logsContainer');
        const downloadButton = document.getElementById('downloadButton');
        const submitButton = document.getElementById('submitButton');
        let pollingInterval;

        form.addEventListener('submit', async (evt) => {
            evt.preventDefault();
            const pptxFile = form.querySelector('input[type="file"]').files[0];
            if (!pptxFile || !pptxFile.name.toLowerCase().endsWith('.pptx')) {
                jobStatus.textContent = 'Please select a .pptx file.';
                return;
            }
            submitButton.classList.add('disabled');
            submitButton.disabled = true;
            jobStatus.textContent = 'Uploading...';
            logsContainer.textContent = '';

            const formData = new FormData(form);
            try {
                const response = await fetch('/upload_pptx3', { method: 'POST', body: formData });
                const data = await response.json();
                if (data.session_id) {
                    jobStatus.textContent = 'Processing...';
                    startPolling(data.session_id);
                } else {
                    jobStatus.textContent = data.error || 'Error';
                    submitButton.classList.remove('disabled');
                    submitButton.disabled = false;
                }
            } catch(e) {
                jobStatus.textContent = 'Error: ' + e;
                submitButton.classList.remove('disabled');
                submitButton.disabled = false;
            }
        });

        function startPolling(sessionId) {
            if (pollingInterval) clearInterval(pollingInterval);
            pollingInterval = setInterval(async () => {
                try {
                    const res = await fetch('/status3/' + sessionId);
                    const job = await res.json();
                    if (job.error) {
                        jobStatus.textContent = job.error;
                        clearInterval(pollingInterval);
                        submitButton.disabled = false;
                        submitButton.classList.remove('disabled');
                        return;
                    }
                    jobStatus.textContent = job.status;
                    logsContainer.innerHTML = job.logs ? job.logs.join('<br>') : '';
                    if (job.status === 'complete' || job.status === 'failed') {
                        clearInterval(pollingInterval);
                        submitButton.disabled = false;
                        submitButton.classList.remove('disabled');
                        if (job.status === 'complete') {
                            downloadButton.style.display = 'inline-block';
                            downloadButton.onclick = () => downloadFile(sessionId);
                        }
                    }
                } catch(e) {
                    jobStatus.textContent = 'Error: ' + e;
                    clearInterval(pollingInterval);
                    submitButton.disabled = false;
                    submitButton.classList.remove('disabled');
                }
            }, 2000);
        }

        async function downloadFile(sessionId) {
            // Trigger download by changing the window location so that the browser handles the filename.
            window.location.href = '/download3/' + sessionId;
        }
        </script>
    </body>
    </html>
    """

@app.route('/upload_pptx3', methods=['POST'])
def upload_pptx3():
    """
    Receive the PPTX file, store optional API keys, and start the enhancement process.
    Return a JSON response with a session_id for tracking.
    """
    # Optional: override environment variables if form fields are set
    openai_key = request.form.get('OPENAI_API_KEY', '')
    stability_key = request.form.get('STABILITY_API_KEY', '')
    deepseek_key = request.form.get('DEEPSEEK_API_KEY', '')
    if openai_key:
        os.environ['OPENAI_API_KEY'] = openai_key
    if stability_key:
        os.environ['STABILITY_API_KEY'] = stability_key
    if deepseek_key:
        os.environ['DEEPSEEK_API_KEY'] = deepseek_key

    if 'pptx_file' not in request.files or request.files['pptx_file'].filename == '':
        return jsonify({"error": "No PPTX file provided"}), 400

    pptx_file = request.files['pptx_file']
    filename = secure_filename(pptx_file.filename)
    session_id = str(uuid.uuid4())
    jobs[session_id] = {
        "status": "queued",
        "logs": [],
        "start_time": None,
        "end_time": None,
        "file_path": "",
        "output_file": "",
        "elapsed_time": 0
    }
    jobs[session_id]['orig_filename'] = filename  # store original filename

    # Save uploaded PPTX
    saved_path = os.path.join(GENERATED_FILES_DIR, f"{session_id}_{filename}")
    pptx_file.save(saved_path)
    jobs[session_id]['file_path'] = saved_path

    # Spin off a thread to enhance
    t = threading.Thread(target=enhance_pptx_thread, args=(session_id,))
    t.start()

    return jsonify({"session_id": session_id})

def enhance_pptx_thread(session_id):
    jobs[session_id]['status'] = "running"
    jobs[session_id]['start_time'] = time.time()

    saved_path = jobs[session_id]['file_path']
    orig_filename = jobs[session_id]['orig_filename']
    filename_base, _ = os.path.splitext(orig_filename)
    output_file = os.path.join(GENERATED_FILES_DIR, f"{filename_base}_enhanced.pptx")
    jobs[session_id]['output_file'] = output_file

    try:
        logs = jobs[session_id]['logs']
        logs.append("=== Enhancement Process Starting ===")
        logs.append("Enhancement in progress. Please wait...")
        
        from pptx2enhancedpptx import enhance_pptx
        enhance_pptx(saved_path, output_file)
        
        time.sleep(2)
        total_time = time.time() - jobs[session_id]['start_time']
        logs.append("=== Enhancement Process Completed ===")
        logs.append(f"Total Execution Time: {total_time:.2f} seconds")
        jobs[session_id]['status'] = "complete"
    except Exception as e:
        import traceback
        debug_info = traceback.format_exc()
        logs = jobs[session_id]['logs']
        logs.append("Error during enhancement: " + str(e))
        logs.append("Debug info: " + debug_info)
        jobs[session_id]['status'] = "failed"

    jobs[session_id]['end_time'] = time.time()
    jobs[session_id]['elapsed_time'] = jobs[session_id]['end_time'] - jobs[session_id]['start_time']

@app.route('/status3/<session_id>', methods=['GET'])
def status3(session_id):
    """
    Return JSON status and logs for the given session_id.
    """
    if session_id not in jobs:
        return jsonify({"error": "Session not found"}), 404

    job = jobs[session_id].copy()
    if job['status'] == 'complete':
        # Format elapsed time
        elapsed = timedelta(seconds=round(job['elapsed_time'], 2))
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        fractional = seconds - int(seconds)
        job['elapsed_time_formatted'] = f"{int(hours)}:{int(minutes):02}:{int(seconds):02}.{int(fractional*100):02}"
    return jsonify(job)

@app.route('/download3/<session_id>', methods=['GET'])
def download3(session_id):
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
    # ...existing code...
    app.run(debug=True, port=5003)