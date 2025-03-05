# SlideUp: Markdown to PPTX Converter

SlideUp is an AI-powered tool that simplifies the process of creating professional presentations from markdown text. It combines the ease of markdown with the power of large language models to create visually appealing PowerPoint presentations with minimal effort.

## Features

- Convert markdown files directly to PowerPoint presentations
- Enhance existing PowerPoint files with AI-generated images
- Automatically generate speaker notes based on slide content
- Create abstract, business-appropriate illustrations for each slide
- Simple web interface for both creation and enhancement workflows
- Customizable prompts for image generation and speaker notes

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/slideup.git
cd slideup
```

2. Set Up the Environment

Before you begin, ensure you have Python 3.6 or higher installed. It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate  # On Windows
```

3. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

4. Configure API Keys (Optional)

SlideUp can function in a basic mode without any API keys. However, for enhanced functionality, you'll need API keys for:

-   **OpenAI**: For generating speaker notes.
-   **Stability AI**: For generating images.
-   **DeepSeek**: For generating image prompts and cleaning text.

Set these API keys as environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export STABILITY_API_KEY="your_stability_api_key"
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

Alternatively, you can enter the API keys directly into the web interface, which will store them temporarily in your browser's session storage.

## Usage

1. Prepare Your Markdown File

Create a markdown file (`.md`) with the content for your presentation. Use the following structure:

```markdown
### Slide 1: Title Slide
* **Title:** My Presentation Title
* **Subtitle:** An informative subtitle
* **Presenter:** Your Name
* **Date:** October 26, 2024

### Slide 2: Content Slide
My First Slide Title

* Bullet point 1
* Bullet point 2
    * Sub-bullet point
* **References:**
    * [Reference Text](http://example.com)

### Slide 3: Another Content Slide
My Second Slide Title

* Another bullet point
```

-   Slides are separated by lines starting with `### Slide N:`.
-   Title slides should be indicated by including the words `Title Slide` in the slide header.
-   Title slide content should be formatted as key-value pairs.
-   Content slides start with a title, followed by bullet points.
-   References should be listed at the end of the slide, indicated by `* **References:**`.

2. Run the Web Application

Start the Flask web application:

```bash
python webapp.py
```

This will start the server on `http://127.0.0.1:5003/`.

3. Use the Web Interface

1.  Open `http://127.0.0.1:5003/index3` in your web browser.
2.  Enter your API keys (optional, if not set as environment variables).
3.  Upload your markdown file.
4.  Click "Upload & Enhance".
5.  Monitor the job status and logs in the interface.
6.  Once the job is complete, a download button will appear. Click it to download your enhanced PPTX file.

4. Command Line Usage (Alternative)

You can also run the script directly from the command line:

```bash
python md2pptx.py <markdown_file> [--add-notes] [--add-images-stability]
```

-   `<markdown_file>`: Path to your markdown file.
-   `--add-notes`: (Optional) Add speaker notes to the presentation (requires OpenAI API key).
-   `--add-images-stability`: (Optional) Add images to the presentation (requires Stability AI and DeepSeek API keys).

Example:

```bash
python md2pptx.py my_presentation.md --add-notes --add-images-stability
```

The generated PPTX file will be saved in the `generated_files` directory.

5. Clean Up

To remove temporary files, you can manually delete the `generated_files` directory or rely on the automatic cleanup that occurs on normal exit (e.g., when pressing Ctrl+C).

## Troubleshooting

-   **API Keys**: Ensure your API keys are valid and have sufficient credits.
-   **Dependencies**: Make sure all dependencies are installed correctly.
-   **File Format**: Verify your markdown file is correctly formatted.
-   **Network**: Check your network connection if you encounter issues with API calls.
