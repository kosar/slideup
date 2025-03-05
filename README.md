# SlideUp: Presentation Creation and Enhancement Tools

SlideUp is an AI-powered toolkit that offers two main functionalities:
1. **Markdown to PPTX Converter**: Transform markdown text into professional PowerPoint presentations
2. **PPTX Enhancement Tool**: Enhance existing PowerPoint files with AI-generated images and speaker notes

Both tools leverage large language models to create visually appealing presentations with minimal effort.

## Features

### Markdown to PPTX Converter
- Convert markdown files directly to PowerPoint presentations
- Automatically generate slide layouts based on content
- Create structured presentations with proper formatting

### PPTX Enhancement Tool
- Add AI-generated images to existing PowerPoint slides
- Generate speaker notes based on slide content
- Create abstract, business-appropriate illustrations for each slide

### Common Features
- Simple web interface for both creation and enhancement workflows
- Customizable prompts for image generation and speaker notes
- API integration with multiple AI providers

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

### Markdown to PPTX Converter

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

3. Use the Markdown to PPTX Interface

1.  Open `http://127.0.0.1:5003/index3` in your web browser.
2.  Enter your API keys (optional, if not set as environment variables).
3.  Upload your markdown file.
4.  Click "Upload & Enhance".
5.  Monitor the job status and logs in the interface.
6.  Once the job is complete, a download button will appear. Click it to download your enhanced PPTX file.

### PPTX Enhancement Tool

1. Prepare Your PowerPoint File

For best results with the enhancement tool, follow these guidelines:

- Keep slide content concise and clear
- Use descriptive titles that convey the main point of the slide
- Organize content with bullet points and clear hierarchy
- Avoid overly complex layouts or custom designs
- Include sufficient text content for the AI to understand the context

2. Use the PPTX Enhancement Interface

1.  Open `http://127.0.0.1:5003/` in your web browser.
2.  Enter your API keys (optional, if not set as environment variables).
3.  Upload your existing PowerPoint file.
4.  Select enhancement options (add images, add speaker notes).
5.  Click "Upload & Enhance".
6.  Monitor the job status and logs in the interface.
7.  Once complete, download your enhanced presentation.

### Tips for Effective PPTX Enhancement

- **Text Content**: Ensure slides have sufficient text for AI to understand the topic
- **Slide Organization**: Consistent layouts work best
- **Image Generation**: Simple, abstract concepts generate better illustrations
- **Speaker Notes**: More detailed slide content leads to better speaker notes
- **Titles**: Clear, descriptive titles help generate relevant enhancements
- **Processing Time**: Complex presentations take longer to enhance
- **Results Variation**: AI-generated content may vary in quality between slides

### Command Line Usage (Alternative)

You can also run the scripts directly from the command line:

For markdown to PPTX conversion:
```bash
python md2pptx.py <markdown_file> [--add-notes] [--add-images-stability]
```

For enhancing existing PPTX:
```bash
python enhance_pptx.py <pptx_file> [--add-notes] [--add-images-stability]
```

Example:

```bash
python md2pptx.py my_presentation.md --add-notes --add-images-stability
```

The generated or enhanced PPTX files will be saved in the `generated_files` directory.

## Troubleshooting

-   **API Keys**: Ensure your API keys are valid and have sufficient credits.
-   **Dependencies**: Make sure all dependencies are installed correctly.
-   **File Format**: Verify your files are correctly formatted.
-   **Network**: Check your network connection if you encounter issues with API calls.
-   **Image Generation Failures**: If image generation fails, try with simpler slide content or different AI providers.
-   **Speaker Notes Inconsistency**: Speaker notes quality depends on the clarity and context of slide content.
