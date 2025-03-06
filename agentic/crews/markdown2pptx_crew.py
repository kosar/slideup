import os
import sys
from typing import Optional
import json

from crewai import Crew, Task, Agent

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Use relative imports
from ..agents.markdown_analyzer import MarkdownAnalyzerAgent
from ..agents.pptx_generator import PPTXGeneratorAgent
from ..utils.logger import setup_logger

logger = setup_logger("MarkdownToPPTXCrew")

class MarkdownToPPTXCrew:
    def __init__(self, use_direct_calls=False):
        self.use_direct_calls = use_direct_calls

    def run(self, markdown_text, output_path):
        # Placeholder for the actual conversion logic
        with open(output_path, "w") as f:
            f.write("This is a placeholder for the converted PowerPoint file.")
        return output_path

def main():
    """
    A simple demonstration of using the MarkdownToPPTXCrew independently of a web interface.
    """
    # Sample Markdown content
    sample_markdown = """
    # Title Slide
    ## Presenter: John Doe
    ## Date: 2024-01-01

    # First Content Slide
    - Introduction
    - Main point
    - Conclusion

    ```python
    def hello_world():
        print("Hello, world!")
    ```
    """

    # Define the output file path
    output_file_path = "output_presentation.pptx"

    # Create an instance of the MarkdownToPPTXCrew
    markdown_to_pptx = MarkdownToPPTXCrew(use_direct_calls=True)

    # Run the crew to convert the Markdown content to a PowerPoint presentation
    result = markdown_to_pptx.run(sample_markdown, output_file_path)

    if result:
        print(f"PowerPoint presentation created successfully at: {result}")
    else:
        print("Failed to create PowerPoint presentation.")

if __name__ == "__main__":
    main()
