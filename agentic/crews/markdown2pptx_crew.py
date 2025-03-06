import os
import sys
from typing import Optional
import json

from crewai import Crew, Task, Agent

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Project root: {project_root}")
    print(f"sys.path: {sys.path}")

# Use relative imports
from ..agents.markdown_analyzer import MarkdownAnalyzerAgent
from ..agents.pptx_generator import PPTXGeneratorAgent
from ..utils.logger import setup_logger

logger = setup_logger("MarkdownToPPTXCrew")

class MarkdownToPPTXCrew:
    """
    A CrewAI crew that orchestrates Markdown analysis and PowerPoint generation.
    This implementation provides both a CrewAI workflow and direct function calls
    for more efficient processing.
    """

    def __init__(self, use_direct_calls: bool = True):
        """
        Initialize the crew.
        
        Args:
            use_direct_calls: If True, bypass the LLM and use direct function calls for efficiency
        """
        self.use_direct_calls = use_direct_calls
        self.markdown_analyzer = MarkdownAnalyzerAgent()
        self.pptx_generator = PPTXGeneratorAgent()
        self.analyzer_agent = self.markdown_analyzer.agent
        self.generator_agent = self.pptx_generator.agent

    def create_crew(self, markdown_content: str):
        """
        Creates the CrewAI crew with the necessary agents and tasks.
        
        Args:
            markdown_content: The Markdown content to analyze.
        """
        analyze_task = Task(
            description=f"Analyze the following Markdown content and create a structured slide plan:\n\n{markdown_content}",
            agent=self.analyzer_agent,
            expected_output="A JSON-like structure representing the slide plan."
        )

        generate_task = Task(
            description="Generate a PowerPoint presentation from the structured slide plan. The slide plan was created by analyzing Markdown content.",
            agent=self.generator_agent,
            expected_output="The file path of the generated PowerPoint file."
        )

        crew = Crew(
            agents=[self.analyzer_agent, self.generator_agent],
            tasks=[analyze_task, generate_task],
            verbose=True  
        )

        return crew

    def run(self, markdown_content: str, output_file_path: str) -> Optional[str]:
        """
        Runs the workflow to convert Markdown content to a PowerPoint presentation.

        Args:
            markdown_content: The Markdown content to convert.
            output_file_path: The desired output file path for the PowerPoint presentation.

        Returns:
            The file path of the generated PowerPoint file, or None if an error occurred.
        """
        try:
            if self.use_direct_calls:
                # Use direct function calls for efficiency
                logger.info("Using direct function calls for Markdown to PowerPoint conversion")
                
                # Step 1: Analyze markdown and create slide plan
                slide_plan = self.markdown_analyzer.analyze_markdown(markdown_content)
                logger.info(f"Generated slide plan with {len(slide_plan.get('slides', []))} slides")
                
                # Step 2: Generate PowerPoint file
                result = self.pptx_generator.generate_pptx_from_analysis(slide_plan, output_file_path)
                
                # Check if file was created successfully
                if os.path.exists(output_file_path):
                    logger.info(f"Successfully created PowerPoint file at {output_file_path}")
                    return output_file_path
                else:
                    logger.error(f"Failed to create PowerPoint file: {result}")
                    return None
            else:
                # Use CrewAI for the workflow
                logger.info("Using CrewAI for Markdown to PowerPoint conversion")
                crew = self.create_crew(markdown_content)
                result = crew.kickoff()
                
                # Extract the string result from the CrewOutput object
                result_str = str(result)
                logger.info(f"Crew result: {result_str}")
                
                # Try to find the generated presentation file
                possible_files = [
                    result_str.strip(),  # The result as-is
                    "presentation.pptx",  # File name from the agent's output
                    "output.pptx",       # Default output name
                    "output_presentation.pptx"  # Alternative name
                ]
                
                # Try each possible file path
                for file_path in possible_files:
                    if file_path and os.path.exists(file_path):
                        logger.info(f"Found generated file: {file_path}")
                        if file_path != output_file_path:
                            os.rename(file_path, output_file_path)
                        return output_file_path
                
                logger.error(f"Could not find any generated PowerPoint file. Agent returned: {result}")
                return None

        except Exception as e:
            logger.error(f"Error running the conversion: {str(e)}", exc_info=True)
            return None


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
    markdown_to_pptx = MarkdownToPPTXCrew(use_direct_calls=True)  # Set to True for direct function calls

    # Run the crew to convert the Markdown content to a PowerPoint presentation
    result = markdown_to_pptx.run(sample_markdown, output_file_path)

    if result:
        print(f"PowerPoint presentation created successfully at: {result}")
    else:
        print("Failed to create PowerPoint presentation.")


if __name__ == "__main__":
    main()
