# SlideUp AI: Demos Deep Dive

### Slide 1: Title Slide
- **title:** SlideUp AI: Demos Deep Dive
- **subtitle:** Exploring the Codebase
- **presenter:** GitHub Copilot
- **date:** October 26, 2023

### Slide 2: SlideUp AI Overview
- **text:**
    *   SlideUp AI helps create and enhance PowerPoint presentations.
    *   Two main implementations: Flask and Agentic.
    *   This presentation focuses on the three demos:
        *   Markdown to PPTX (Flask)
        *   PPTX Enhancement (Flask)
        *   Agentic App (Streamlit)

### Slide 3: Core Architecture
- **text:**
    *   Key modules:
        *   `env_utils.py` (in root): API key handling for Flask apps
        *   `md2pptx.py` (in root): Core slide generation logic (used by Flask)
        *   `pptx_enhancement_webapp.py`: (in `flask_apps/`) Enhances existing PPTX files
        *   Agentic equivalents in `agentic/` directory
    *   These modules ensure separation of concerns and code reuse.

### Slide 4: Flask Demo 1 - Markdown to PPTX
- **text:**
    *   `markdown2pptx_webapp.py` converts Markdown to PPTX.
    *   Uses `md2pptx.py` to parse Markdown and create slides.
    *   Optionally uses `ai_services.py` for speaker notes and images.
    *   Access via `http://127.0.0.1:5002/`.

### Slide 5: Flask Demo 2 - PPTX Enhancement
- **text:**
    *   `pptx_enhancement_webapp.py` enhances existing PPTX files.
    *   Uses `pptx_enhancement_webapp.py` to manipulate PowerPoint files.
    *   Optionally uses AI for image generation.
    *   Access via `http://127.0.0.1:5003/`.

### Slide 6: Flask Demos Workflow
- **text:**
    1.  User provides input via web interface.
    2.  Flask app processes request.
    3.  Core modules handle core logic and AI services.
    4.  PowerPoint file is generated and returned.

### Slide 7: Agentic Demo - Streamlit App
- **text:**
    *   `agentic/app.py` provides a Streamlit interface for both Markdown to PPTX and PPTX enhancement.
    *   Uses CrewAI to coordinate specialized agents.
    *   Run via `streamlit run agentic/app.py`.
    *   Access via `http://localhost:8501`.

### Slide 8: Agentic Implementation - Agents and Crews
- **text:**
    *   Specialized agents:
        *   Content Agent: Enhances text content.
        *   Designer Agent: Handles visual design and image prompts.
        *   Formatter Agent: Ensures consistent formatting.
    *   Crews coordinate agents:
        *   `markdown2pptx_crew.py`
        *   `pptx_enhancement_crew.py`

### Slide 9: Agentic Demos Workflow
- **text:**
    1.  User interacts with Streamlit interface.
    2.  Crew is initialized based on task.
    3.  Crew coordinates specialized agents.
    4.  Agents collaborate to generate the presentation.

### Slide 10: Conclusion
- **text:**
    *   SlideUp AI offers two powerful implementations for presentation creation and enhancement.
    *   Flask apps provide simple, focused interfaces.
    *   Agentic app leverages AI agents for sophisticated results.
    *   Explore the code, contribute, and make awesome presentations!
