# slideup
SlideUp is a web application that generates PowerPoint presentations from text outlines. Simply provide a text outline, and SlideUp will create a visually appealing and informative presentation. It can function in a basic mode without any API keys, or in an enhanced mode if API keys are provided.

In basic mode, SlideUp will generate a presentation using only the provided text outline. This includes creating slides with titles and bullet points based on the structure of the markdown file. However, in enhanced mode, SlideUp leverages the power of three APIs to add more advanced features:

- **OpenAI**: Generates speaker notes for each slide, providing additional context and information that can be used during the presentation.
- **Stability AI**: Generates images that can be included in the slides, making the presentation more visually appealing.
- **DeepSeek**: Generates image prompts and cleans the text to ensure that the content is clear and professional.

To use these enhanced features, you need to provide the respective API keys. If the API keys are not provided, SlideUp will still function, but without the advanced features mentioned above.

## Usage

### 1. Set Up the Environment

Before you begin, ensure you have Python 3.6 or higher installed. It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys (Optional)

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

### 4. Prepare Your Markdown File

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

### 5. Run the Web Application

Start the Flask web application:

```bash
python webapp.py
```

This will start the server on `http://127.0.0.1:5003/`.

### 6. Use the Web Interface

1.  Open `http://127.0.0.1:5003/index3` in your web browser.
2.  Enter your API keys (optional, if not set as environment variables).
3.  Upload your markdown file.
4.  Click "Upload & Enhance".
5.  Monitor the job status and logs in the interface.
6.  Once the job is complete, a download button will appear. Click it to download your enhanced PPTX file.

### 7. Command Line Usage (Alternative)

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

### 8. Clean Up

To remove temporary files, you can manually delete the `generated_files` directory or rely on the automatic cleanup that occurs on normal exit (e.g., when pressing Ctrl+C).

### Troubleshooting

-   **API Keys**: Ensure your API keys are valid and have sufficient credits.
-   **Dependencies**: Make sure all dependencies are installed correctly.
-   **File Format**: Verify your markdown file is correctly formatted.
-   **Network**: Check your network connection if you encounter issues with API calls.
