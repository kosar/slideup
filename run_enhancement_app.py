import os
import sys
import streamlit.web.bootstrap as bootstrap
from pathlib import Path

# Set the absolute path to the app
app_path = os.path.join(os.path.dirname(__file__), "agentic", "web", "pptx_enhancement_app.py")

# Run the app
if __name__ == "__main__":
    sys.argv = ["streamlit", "run", app_path]
    bootstrap.run(app_path, "", [], {})
