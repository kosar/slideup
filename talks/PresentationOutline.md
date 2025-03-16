# SlideUp AI: Codebase Overview (20-Minute Presentation)

## I. Introduction (2 minutes)
    *   Briefly introduce SlideUp AI and its purpose.
    *   Mention the two implementations: Flask and Agentic.
    *   Highlight the three demos: Markdown to PPTX (Flask), PPTX Enhancement (Flask), and Agentic App.

## II. Core Architecture (5 minutes)
    *   Explain the core modules: `env_utils.py` (API key handling), `md2pptx.py` (Markdown conversion), `pptx_enhancement_webapp.py` (PPTX enhancement), and the agentic equivalents.
    *   Describe how these modules are used in the Flask and Agentic implementations.
    *   Briefly touch on the separation of concerns.

## III. Flask Implementation (5 minutes)
    *   Explain the structure of `flask_apps/`.
    *   Describe `markdown2pptx_webapp.py` and `pptx_enhancement_webapp.py`.
    *   Show the workflow: User input -> Flask -> core modules -> Output.
    *   Mention the HTML templates in `templates/`.

## IV. Agentic Implementation (5 minutes)
    *   Explain the structure of `agentic/`.
    *   Describe the roles of agents: Content, Designer, Formatter.
    *   Explain how crews (`markdown2pptx_crew.py`, `pptx_enhancement_crew.py`) coordinate agents.
    *   Show the workflow: User input -> Streamlit -> Crews -> Agents -> Output.

## V. Demos and Usage (3 minutes)
    *   Briefly demonstrate the three demos.
    *   Mention how to run each demo (Flask apps on ports 5002/5003, Streamlit app via `streamlit run`).
    *   Point to the #USER_GUIDE.md for detailed instructions.

## VI. Conclusion (1 minute)
    *   Summarize the key takeaways.
    *   Encourage contributions and further exploration.
