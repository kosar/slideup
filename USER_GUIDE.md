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

*   **Option 1: Using the Web Interface (Easiest):** When you use SlideUp in your web browser, you can just paste the keys into the boxes on the page. SlideUp will remember them temporarily.
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

Okay, time to make some slides!

1.  **Open your Terminal** (if it's not already open).
2.  **Navigate to the SlideUp folder:** Use the `cd` command to go to the folder where you saved the SlideUp files. For example:

    ```bash
    cd /Users/yourname/Documents/slideup  # Replace with your actual folder path
    ```

3.  **Start SlideUp:** Copy and paste this command into the Terminal and press `Enter`:

    ```bash
    python webapp.py
    ```

    *   **What's happening?** This command starts the SlideUp program.

4.  **Open your web browser:** SlideUp will tell you the address to go to (usually `http://127.0.0.1:5002/` or `http://127.0.0.1:5003/`). Type that address into your browser's address bar and press `Enter`.

### 5. Make Your Presentation! ✨

1.  **You'll see the SlideUp webpage.**
2.  **Enter your API keys** (if you have them) in the boxes.
3.  **Choose your Markdown file:** Click the "Choose File" button and select your `.md` file.
4.  **Check the boxes:** If you want speaker notes and images, check the "Add Speaker Notes" and "Add Images" boxes.
5.  **Click "Generate PPTX":** Let SlideUp do its thing!
6.  **Watch the progress:** The page will show you what's happening.
7.  **Download your presentation:** When it's done, a "Download" link will appear. Click it to download your beautiful new PowerPoint presentation!

## Troubleshooting 🛠️

*   **"I get an error when I run `python webapp.py`!"**
    *   Make sure you've followed all the steps in "Installing SlideUp" correctly.
    *   Double-check that you're in the right folder in the Terminal (use the `cd` command).
    *   If you still have problems, try searching the error message online or ask a techy friend for help.
*   **"My presentation doesn't have speaker notes or images!"**
    *   Make sure you've entered your API keys correctly.
    *   Check the "Add Speaker Notes" and "Add Images" boxes before generating the presentation.
    *   If you're using the Terminal to set your API keys, make sure you've added them to your `.bashrc` or `.zshrc` file (or that you're setting them in each new Terminal session).
*   **"The images look weird!"**
    *   SlideUp uses AI to generate images, and sometimes it doesn't get it quite right. Try tweaking your slide titles or bullet points to give the AI better hints.
*   **"SlideUp is taking a long time!"**
    *   Generating speaker notes and images can take a few minutes, especially if the servers are busy. Be patient!

## Advanced Tips & Tricks 🤓

*   **Customize your slides:** SlideUp creates a basic presentation, but you can always open it in PowerPoint and add your own personal touches.
*   **Experiment with your outlines:** The better your outline, the better your presentation will be. Try different ways of organizing your ideas.
*   **Contribute to SlideUp:** If you're a techy person, you can help make SlideUp even better! Check out the code and contribute your ideas.

## Have Fun! 🎉

SlideUp is designed to make your life easier. Experiment, get creative, and enjoy making awesome presentations!
