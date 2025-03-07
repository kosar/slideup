# SlideUp AI

A presentation assistant powered by AI that helps you create and improve PowerPoint presentations. This project offers two separate implementations with the same functionality:

1. **Agentic Implementation**: Using CrewAI framework with a Streamlit interface
2. **Flask Implementation**: A traditional web application using Flask

## Features

- **Markdown to PowerPoint Conversion**: Quickly turn your Markdown documents into professional presentations
- **PowerPoint Enhancement**: Automatically improve the design, content, structure, and formatting of your presentations

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/slideup.git
cd slideup
```

2. Create a virtual environment and activate it
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key
```bash
export OPENAI_API_KEY="your-api-key"  # On Windows: set OPENAI_API_KEY="your-api-key"
```

## Usage

### Option 1: Agentic Implementation (CrewAI + Streamlit)

This implementation uses CrewAI for the backend logic and Streamlit for the user interface.

#### Start the Streamlit App

```bash
# 
streamlit run agentic/app.py
```

The application will open in your default web browser.

#### Markdown to PowerPoint

1. Select the "Markdown to PowerPoint" tab
2. Upload a Markdown file or paste your Markdown content in the text area
3. Configure presentation settings (theme, color scheme, etc.)
4. Click "Generate PowerPoint" to create your presentation
5. Download the generated PowerPoint file

#### PowerPoint Enhancement

1. Select the "PowerPoint Enhancement" tab
2. Upload a PowerPoint (.pptx) file
3. Select enhancement options (visuals, content, structure, formatting)
4. Adjust the enhancement level and add any specific instructions
5. Click "Enhance PowerPoint" to improve your presentation
6. Download the enhanced PowerPoint file

### Option 2: Flask Web Application

This is a traditional web application implementation using Flask.

#### Start the Flask Server

```bash
cd src/slideup
python flask_app/app.py
```

The server will start on http://localhost:5000 (or the port specified in the app).

#### Using the Web Interface

1. Open your browser and navigate to http://localhost:5000
2. Choose from the available tools:
   - "Convert Markdown to PowerPoint": Upload or paste Markdown and set options
   - "Enhance PowerPoint": Upload a PowerPoint file and select enhancement options
3. Submit your request and download the resulting file when processing is complete

#### API Endpoints

The Flask implementation also provides API endpoints:

- `POST /api/markdown-to-pptx`: Convert Markdown to PowerPoint
- `POST /api/enhance-pptx`: Enhance an existing PowerPoint file

See API documentation within the app for request and response formats.

## Development

### Project Structure

- `agentic/`: CrewAI-based implementation
  - `app.py`: Streamlit application
  - `crews/`: AI crew implementations
    - `markdown2pptx_crew.py`: Markdown to PowerPoint conversion crew
    - `pptx_enhancement_crew.py`: PowerPoint enhancement crew
  - `utils/`: Utility functions
  - `tests/`: Test files

- `flask_app/`: Flask-based implementation
  - `app.py`: Flask application server
  - `services/`: Service modules for processing
  - `templates/`: HTML templates
  - `static/`: CSS, JavaScript, and static assets

### Running Tests

```bash
cd src/slideup
python -m unittest discover agentic/tests
```

## Requirements

- Python 3.8+
- Streamlit (for agentic implementation)
- Flask (for web app implementation)
- python-pptx
- CrewAI
- LangChain
- OpenAI API key

## License

MIT
