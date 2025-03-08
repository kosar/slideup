# SlideUp: Create Awesome Presentations Easily! ✨

Welcome to SlideUp! This guide will help you turn your text outlines into beautiful PowerPoint presentations, even if you're not a tech whiz. Let's get started!

## What is SlideUp? 🤔

SlideUp is like a magic wand for your presentation outlines. You give it a simple text file, and it creates a PowerPoint presentation for you. It's perfect for:

*   Turning meeting notes into shareable slides.
*   Creating presentations from research outlines.
*   Making your ideas shine without spending hours on design.

## Getting Started 🚀

### 1. Install SlideUp (Don't worry, it's easier than it sounds!) 💻

Think of this like installing an app on your phone. Here's how:

1.  **Open your Terminal:**
    *   **On Mac:** Press `Command + Spacebar`, type "Terminal", and press `Enter`.
    *   **On Windows:** Search for "Command Prompt" or "PowerShell" in the Start Menu.

2.  **Copy and Paste the following commands, one at a time, into the Terminal and press `Enter` after each:**

    ```bash
    # Clone the repository (if you haven't already)
    git clone https://github.com/yourusername/slideup.git
    cd slideup
    
    # Set up the environment
    python3 -m venv venv
    source venv/bin/activate   # For Mac
    # venv\Scripts\activate  # For Windows (remove the '#' to uncomment)
    pip install -r requirements.txt
    ```

    *   **What are these commands doing?**
        *   `python3 -m venv venv`: This creates a special "sandbox" for SlideUp to live in, so it doesn't mess with other things on your computer.
        *   `source venv/bin/activate` (or `venv\Scripts\activate` on Windows): This "turns on" the sandbox.
        *   `pip install -r requirements.txt`: This installs all the little helper programs that SlideUp needs to work.

3.  **Wait for it to finish:** The Terminal will show you what's happening. It might take a few minutes.

### 2. Get Your API Keys (Optional, but Recommended!) 🔑

SlideUp can do even *more* if you give it special keys called "API keys". These keys unlock extra features like:

*   **Speaker Notes:** SlideUp can automatically write notes for each slide, so you know what to say!
*   **Amazing Images:** SlideUp can add beautiful, relevant images to your slides.

**How to get API keys:**

1.  **OpenAI (for Speaker Notes):**
    *   Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys) and create an account (if you don't have one).
    *   Click "+ Create new secret key".
    *   Copy the key! (Keep it safe, like a password).

2.  **Stability AI (for Images):**
    *   Go to [https://stability.ai/](https://stability.ai/) and create an account.
    *   Find your API key in your account settings.
    *   Copy the key!

3.  **DeepSeek (for better Images):**
    *   Go to [https://platform.deepseek.com/](https://platform.deepseek.com/) and create an account.
    *   Find your API key in your account settings.
    *   Copy the key!

**Setting the API Keys:**

There are two ways to set the API keys:

*   **Option 1: Using the Web Interface (Easiest):** 
    * When you use SlideUp in your web browser, you can paste the keys into the fields in the "Advanced Settings" section.
    * Click on "Advanced Settings" to expand this section.
    * Your keys will be saved in your browser's storage, so you only need to enter them once.
    * You'll see a small indicator next to each field if a key is already set on the server.

*   **Option 2: Using the Terminal (More Permanent):**
    1.  Copy and paste these commands into your Terminal, replacing `"YOUR_API_KEY"` with your actual keys:

        ```bash
        export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
        export STABILITY_API_KEY="YOUR_STABILITY_API_KEY"
        export DEEPSEEK_API_KEY="YOUR_DEEPSEEK_API_KEY"
        ```

        *   **Important:** This only sets the keys for your current Terminal session. To make them permanent, you'll need to add these lines to your `.bashrc` or `.zshrc` file (ask a techy friend for help if you're not sure how!).

### 3. Create Your Presentation Outline (The Magic Ingredient!) 📝

SlideUp needs a text file to work its magic. Here's how to create one:

1.  **Open a text editor:** (Like Notepad on Windows or TextEdit on Mac).
2.  **Write your outline:** Use the following format:

    ```markdown
    ### Slide 1: Title Slide
    * **Title:** My Awesome Presentation
    * **Subtitle:** A Guide to Greatness
    * **Presenter:** Your Name
    * **Date:** Today's Date

    ### Slide 2: My First Slide
    * Bullet point 1
    * Bullet point 2
        * Sub-bullet point

    ### Slide 3: Another Great Slide
    * More bullet points!
    ```

    *   **Important Notes:**
        *   Each slide starts with `### Slide N:`.
        *   For title slides, include `Title Slide` in the slide title.
        *   Use `*` for bullet points.
        *   Save the file with a `.md` extension (e.g., `my_presentation.md`).

### 4. Run SlideUp! 🎬

Now you have two options to run SlideUp:

#### Option A: Flask Web Applications (Simple & Straightforward)

1.  **Open your Terminal** (if it's not already open).
2.  **Make sure you're in the SlideUp folder and your virtual environment is active.**
3.  **Start the web app** you want to use:

    ```bash
    # For creating presentations from markdown
    python flask_apps/markdown2pptx_webapp.py
    ```
    
    Or if you want to enhance existing presentations:
    
    ```bash
    # For enhancing existing PPTX files
    python flask_apps/pptx_enhancement_webapp.py
    ```

4.  **Open your web browser:** 
    * For Markdown to PowerPoint: Navigate to `http://127.0.0.1:5002/`
    * For PowerPoint Enhancement: Navigate to `http://127.0.0.1:5003/`

#### Option B: Streamlit Application (More Features)

The Streamlit application provides an enhanced interface with more features:

1.  **Open your Terminal** (if it's not already open).
2.  **Make sure you're in the SlideUp folder and your virtual environment is active.**
3.  **Start the Streamlit app:**

    ```bash
    streamlit run agentic/app.py
    ```

4.  **The app will automatically open** in your default web browser at `http://localhost:8501`

### 5. Make Your Presentation! ✨

Using the **Flask Web Applications**:

1.  **You'll see the SlideUp webpage.**
2.  **Choose your input file:** 
    * For Markdown: Enter your presentation outline in the text box or click "Upload Markdown File"
    * For PPTX Enhancement: Click "Select PPTX File" to choose an existing PowerPoint file
3.  **Check the boxes:** If you want speaker notes and images, check the "Add Speaker Notes" and "Add Images" boxes (Markdown version only).
4.  **Advanced customization (optional):**
    * Click the "Advanced Settings" dropdown to reveal more options
    * Enter your API keys if needed (they'll be saved for future use)
    * Customize the prompts used for image generation and speaker notes if desired
5.  **Click "Generate PPTX" or "Upload & Enhance":** Let SlideUp do its thing!
6.  **Watch the progress:** The page will show you what's happening.
7.  **Download your presentation:** When it's done, a "Download" link will appear. Click it to download your beautiful new PowerPoint presentation!

Using the **Streamlit Application**:

1. **Choose the task** you want to perform (Create from Markdown or Enhance PowerPoint)
2. **Upload your file** or paste your Markdown content
3. **Configure options** using the sidebar controls
4. **Click "Generate"** and wait for the processing to complete
5. **Download** your finished presentation

## Customizing Your Slides 🎨

SlideUp offers advanced customization options:

* **Image Generation Prompt:** Control how AI generates images for your slides
* **DeepSeek Prompt Engineer Instructions:** Fine-tune how concepts are translated into image prompts
* **Speaker Notes Generation Prompt:** Customize how speaker notes are written

To access these options:

1. Click the "Advanced Settings" dropdown in the Flask apps, or use the sidebar in the Streamlit app
2. Edit the text in each box to match your preferred style
3. Generate your presentation as usual

These customization options interact with the AI services defined in the [`lib/ai_services.py`](https://github.com/yourusername/slideup/blob/main/lib/ai_services.py) module.

## Troubleshooting 🛠️

*   **"I get an error when I run the web app!"**
    *   Make sure you've followed all the steps in "Installing SlideUp" correctly.
    *   Check that you're running the command from the root `slideup` directory.
    *   Verify that your virtual environment is activated (you should see `(venv)` at the beginning of your command prompt).
    *   Try checking the logs for more specific error information.

*   **"My presentation doesn't have speaker notes or images!"**
    *   Make sure you've entered your API keys correctly in Advanced Settings.
    *   Check the "Add Speaker Notes" and "Add Images" boxes before generating.
    *   Look at the logs area on the page for any error messages.
    *   Verify that your API keys are valid and have sufficient credits.

*   **"The images look weird or don't match my content!"**
    *   Try customizing the image generation prompt in Advanced Settings.
    *   Provide more detailed bullet points on your slides to guide the AI.
    *   Try switching between Stability AI and DeepSeek to see which works better for your content.

*   **"SlideUp is taking a long time!"**
    *   Generating speaker notes and images can take a few minutes, especially if the servers are busy. Be patient!
    *   Watch the Workflow Progress area for updates on what's happening.
    *   For complex presentations with many slides, consider breaking them into smaller files.

## Advanced Tips & Tricks 🤓

*   **Reuse API keys:** Once entered, your API keys will be saved in your browser for future sessions.
*   **Experiment with different prompts:** Try different wording in the Advanced Settings to get different styles of images and speaker notes.
*   **Server keys:** If you see "(Server key available)" next to an API key field, it means that key is already set on the server and you don't need to enter it again.
*   **Customize post-generation:** SlideUp creates a basic presentation, but you can always open it in PowerPoint and add your own personal touches.
*   **Try the agentic implementation:** For more sophisticated presentations, use the Streamlit app which leverages multiple specialized AI agents working together.
*   **Explore the code:** If you're technically inclined, check out how SlideUp works by exploring the [codebase on GitHub](https://github.com/yourusername/slideup).

## Have Fun! 🎉

SlideUp is designed to make your life easier. Experiment, get creative, and enjoy making awesome presentations!
