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
            textarea {
                width: 100%;
                margin-top: 0.5em;
                padding: 0.7em;
                border: 1px solid #ccc;
                border-radius: 4px;
                min-height: 80px;
                font-family: 'Roboto', sans-serif;
                font-size: 14px;
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
            /* Advanced settings styling */
            .advanced-settings {
                margin-top: 1em;
                border-top: 1px solid #eee;
                padding-top: 1em;
            }
            .advanced-toggle {
                color: #5563DE;
                cursor: pointer;
                font-weight: 500;
                display: flex;
                align-items: center;
            }
            .advanced-toggle::after {
                content: "▼";
                margin-left: 5px;
                transition: transform 0.3s;
            }
            .advanced-toggle.collapsed::after {
                transform: rotate(-90deg);
            }
            .advanced-content {
                padding: 1em 0;
                display: none;
            }
            .help-text {
                font-size: 12px;
                color: #666;
                margin-top: 0.5em;
            }
            .env-key-status {
                font-size: 11px;
                color: #666;
                margin-left: 5px;
            }
            .api-keys-section {
                background-color: #f8f8f8;
                border: 1px solid #e0e0e0;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            
            .api-keys-section h3 {
                margin-top: 0;
                margin-bottom: 10px;
                color: #5563DE;
            }
            
            .help-text {
                font-size: 12px;
                color: #666;
                margin-top: 8px;
                margin-bottom: 15px;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Enhance PPTX File (Index 3)</h1>
            <form id="uploadForm" enctype="multipart/form-data">
                <label>Select PPTX File:</label>
                <input type="file" name="pptx_file">
                
                <!-- Advanced settings section with API keys and prompts -->
                <div class="advanced-settings">
                    <div id="advancedToggle" class="advanced-toggle collapsed">Advanced Settings</div>
                    <div id="advancedContent" class="advanced-content">
                        <!-- API Keys Section -->
                        <div class="api-keys-section">
                            <h3>API Keys</h3>
                            <div class="api-key-fields">
                                <label for="OPENAI_API_KEY">OpenAI API Key: 
                                    <span id="openaiEnvKeyStatus" class="env-key-status"></span>
                                </label>
                                <input type="password" name="OPENAI_API_KEY" id="OPENAI_API_KEY" placeholder="sk-..." style="width: 100%;">
                                
                                <label for="STABILITY_API_KEY">Stability API Key:
                                    <span id="stabilityEnvKeyStatus" class="env-key-status"></span>
                                </label>
                                <input type="password" name="STABILITY_API_KEY" id="STABILITY_API_KEY" placeholder="sk-..." style="width: 100%;">
                                
                                <label for="DEEPSEEK_API_KEY">DeepSeek API Key:
                                    <span id="deepseekEnvKeyStatus" class="env-key-status"></span>
                                </label>
                                <input type="password" name="DEEPSEEK_API_KEY" id="DEEPSEEK_API_KEY" placeholder="sk-..." style="width: 100%;">
                                
                                <p class="help-text">API keys are saved in your browser. If provided, these override server keys.</p>
                            </div>
                        </div>
                        
                        <label for="stability_prompt">Image Generation Prompt:</label>
                        <textarea name="stability_prompt" id="stability_prompt" rows="6">Create a conceptual visualization of an abstract concept representing: {concept} STYLE: Abstract, symbolic imagery. Use dynamic shapes, flowing lines, and vibrant colors to convey the essence of the content. Explore color symbolism and metaphorical representations. Avoid realistic objects or scenes. Focus on conveying the underlying ideas. DO NOT include text, numbers, or literal interpretations. Only abstract visual elements. BLUR TEXT in your images if generated. COMPOSITION: Create a balanced, harmonious composition.</textarea>
                        <p class="help-text">This prompt template is used for generating images for your slides. You can customize it to match your preferred visual style.</p>
                        
                        <label for="deepseek_prompt">DeepSeek Prompt Engineer Instructions:</label>
                        <textarea name="deepseek_prompt" id="deepseek_prompt" rows="6">You are a creative prompt engineer for AI image generation. Create VISUAL, ABSTRACT prompts for business concepts based on the content provided as context, with these rules:
1. Focus on symbolic representations, not literal
2. Use vivid color combinations and abstract shapes
3. Incorporate dynamic compositions
4. Reference artistic styles that resonate with a business audience to complement words with visuals
5. Keep under 500 words
6. STYLE: Abstract, symbolic imagery. Use dynamic shapes, flowing lines, and vibrant colors to convey the essence of the content.
7. NO WORDS should be in the image unless they are well understood and spelled correctly. Emphasizes abstract visual elements. BLUR TEXT in your images if generated. When conveying scientific concepts, use metaphorical representations.
8. COMPOSITION: Create a balanced, harmonious composition. Use symmetry, contrast, and focal points to guide the viewer's eye.</textarea>
                        <p class="help-text">These instructions guide the AI in creating prompts for image generation. Advanced users can modify this to change the style of generated images.</p>
                    </div>
                </div>
                
                <button id="submitButton" type="submit">Upload & Enhance</button>
            </form>
            <p>Job Status: <span id="jobStatus" class="status"></span></p>
            <div class="logs" id="logsContainer"></div>
            <button class="download-button" id="downloadButton" style="display: none;">Download Enhanced PPTX</button>
        </div>
        <script>
        // On page load, restore API keys if available and check server keys
        window.onload = function() {
            const openaiField = document.getElementById('OPENAI_API_KEY');
            const stabilityField = document.getElementById('STABILITY_API_KEY');
            const deepseekField = document.getElementById('DEEPSEEK_API_KEY');
            
            // Check server environment variables
            checkServerKeys();
            
            // Restore from localStorage
            if (localStorage.getItem('openaiApiKey')) {
                openaiField.value = localStorage.getItem('openaiApiKey');
            }
            if (localStorage.getItem('stabilityApiKey')) {
                stabilityField.value = localStorage.getItem('stabilityApiKey');
            }
            if (localStorage.getItem('deepseekApiKey')) {
                deepseekField.value = localStorage.getItem('deepseekApiKey');
            }
            
            // Set up the advanced settings toggle
            document.getElementById('advancedToggle').addEventListener('click', function() {
                const content = document.getElementById('advancedContent');
                const isVisible = content.style.display === 'block';
                content.style.display = isVisible ? 'none' : 'block';
                this.classList.toggle('collapsed', !isVisible);
            });
        };
        
        // Check if server has environment variables set
        async function checkServerKeys() {
            try {
                const response = await fetch('/env_keys_check');
                const data = await response.json();
                
                document.getElementById('openaiEnvKeyStatus').innerHTML = 
                    data.openai_key_set ? '<small>(Server key available)</small>' : '';
                document.getElementById('stabilityEnvKeyStatus').innerHTML = 
                    data.stability_key_set ? '<small>(Server key available)</small>' : '';
                document.getElementById('deepseekEnvKeyStatus').innerHTML = 
                    data.deepseek_key_set ? '<small>(Server key available)</small>' : '';
            } catch (e) {
                console.error("Failed to check server API keys:", e);
            }
        }

        // Save API keys whenever their value changes
        document.getElementById('OPENAI_API_KEY').addEventListener('input', function() {
            localStorage.setItem('openaiApiKey', this.value);
        });
        document.getElementById('STABILITY_API_KEY').addEventListener('input', function() {
            localStorage.setItem('stabilityApiKey', this.value);
        });
        document.getElementById('DEEPSEEK_API_KEY').addEventListener('input', function() {
            localStorage.setItem('deepseekApiKey', this.value);
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
    # Store original environment variables
    original_env = store_original_env()

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

    if 'pptx_file' not in request.files or request.files['pptx_file'].filename == '':
        # Restore original environment variables before returning
        restore_env(original_env)
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
        "elapsed_time": 0,
        "stability_prompt": stability_prompt,  # Store the custom stability prompt
        "deepseek_prompt": deepseek_prompt,    # Store the custom deepseek prompt
        "original_env": original_env          # Store original environment variables
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
    try:
        jobs[session_id]['status'] = "running"
        jobs[session_id]['start_time'] = time.time()

        saved_path = jobs[session_id]['file_path']
        orig_filename = jobs[session_id]['orig_filename']
        filename_base, _ = os.path.splitext(orig_filename)
        output_file = os.path.join(GENERATED_FILES_DIR, f"{filename_base}_enhanced.pptx")
        jobs[session_id]['output_file'] = output_file
        stability_prompt = jobs[session_id].get('stability_prompt', None)  # Get the custom prompt

        logs = jobs[session_id]['logs']
        logs.append("=== Enhancement Process Starting ===")
        logs.append("Enhancement in progress. Please wait...")
        
        from pptx2enhancedpptx import enhance_pptx
        # Pass the custom prompt to the enhance_pptx function
        enhance_pptx(saved_path, output_file, stability_prompt=stability_prompt)
        
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
    finally:
        # Restore original environment variables
        if 'original_env' in jobs[session_id]:
            restore_env(jobs[session_id]['original_env'])
            # Remove sensitive data
            del jobs[session_id]['original_env']

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