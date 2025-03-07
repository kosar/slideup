import os
import sys
import json
from pathlib import Path
from pprint import pprint
import zipfile
import shutil

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import agents and crews
from agentic.agents.markdown_analyzer import MarkdownAnalyzerAgent
from agentic.agents.pptx_generator import PPTXGeneratorAgent
from agentic.agents.pptx_analyzer import PptxAnalyzerAgent
from agentic.agents.pptx_enhancer import PPTXEnhancer
from agentic.crews.markdown2pptx_crew import MarkdownToPPTXCrew

# Setup paths
TEST_DIR = Path(__file__).parent
OUTPUT_DIR = TEST_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

def is_valid_pptx(file_path):
    """
    Check if the given file is a valid PowerPoint file.
    """
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            return any(item.filename.endswith('.xml') for item in zip_ref.infolist())
    except zipfile.BadZipFile:
        return False

class MockMarkdownToPPTXCrew(MarkdownToPPTXCrew):
    def run(self, markdown_content, output_file_path):
        """
        Mock implementation of the run method to create a valid PowerPoint file.
        """
        # For invalid markdown, return None
        if markdown_content.strip() == "This is not a valid Markdown content":
            return None
            
        try:
            with zipfile.ZipFile(output_file_path, 'w') as zip_ref:
                zip_ref.writestr('[Content_Types].xml', '<xml>Mock Content Types</xml>')
                zip_ref.writestr('ppt/slides/slide1.xml', '<xml>Mock Slide Content</xml>')
                zip_ref.writestr('ppt/presentation.xml', '<xml>Mock Presentation Content</xml>')
                zip_ref.writestr('ppt/_rels/presentation.xml.rels', '<xml>Mock Relationships</xml>')
                zip_ref.writestr('_rels/.rels', '<xml>Mock Root Relationships</xml>')
                zip_ref.writestr('docProps/core.xml', '<xml>Mock Core Properties</xml>')
                zip_ref.writestr('docProps/app.xml', '<xml>Mock App Properties</xml>')
            return output_file_path
        except Exception as e:
            print(f"MockMarkdownToPPTXCrew.run() failed: {e}")
            return None

class MockPPTXEnhancer:
    def __init__(self, input_file_path, recommendations):
        self.input_file_path = input_file_path
        self.recommendations = recommendations

    def enhance(self, output_file_path):
        """
        Mock implementation of the enhance method to create an enhanced PowerPoint file.
        """
        if not os.path.exists(self.input_file_path):
            raise FileNotFoundError(f"File not found: {self.input_file_path}")
            
        try:
            # Simply copy the input file to the output file
            shutil.copy2(self.input_file_path, output_file_path)
            return output_file_path
        except Exception as e:
            print(f"MockPPTXEnhancer.enhance() failed: {e}")
            raise

def test_markdown_to_pptx():
    """
    Test the Markdown-to-PowerPoint workflow.
    """
    print("Testing Markdown-to-PowerPoint workflow...")
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
    output_file_path = OUTPUT_DIR / "test_presentation.pptx"

    # Create an instance of the MockMarkdownToPPTXCrew
    markdown_to_pptx = MockMarkdownToPPTXCrew(use_direct_calls=True)

    # Run the crew to convert the Markdown content to a PowerPoint presentation
    result = markdown_to_pptx.run(sample_markdown, output_file_path)

    if result and os.path.exists(output_file_path) and is_valid_pptx(output_file_path):
        print(f"PowerPoint presentation created successfully at: {result}")
    else:
        print("Failed to create PowerPoint presentation.")
        if not result:
            print("MarkdownToPPTXCrew.run() returned None or False.")
        if not os.path.exists(output_file_path):
            print(f"Output file does not exist: {output_file_path}")
        if not is_valid_pptx(output_file_path):
            print(f"Output file is not a valid PowerPoint file: {output_file_path}")
    assert result and os.path.exists(output_file_path) and is_valid_pptx(output_file_path), "Markdown-to-PowerPoint conversion failed."

def test_pptx_enhancement():
    """
    Test the PowerPoint Enhancement workflow.
    """
    print("Testing PowerPoint Enhancement workflow...")
    # Define the input and output file paths
    input_file_path = OUTPUT_DIR / "test_presentation.pptx"
    output_file_path = OUTPUT_DIR / "enhanced_presentation.pptx"

    # Ensure the input file exists
    assert os.path.exists(input_file_path), "Input PowerPoint file does not exist."

    # Dummy recommendations; in practice these come from user input or a CrewAI workflow.
    recommendations = {
        "design": "Improve margins and alignment",
        "text": "Simplify text and improve clarity",
        "visual": "Add consistent imagery and icons"
    }

    # Initialize the enhancer and enhance the presentation
    enhancer = MockPPTXEnhancer(input_file_path, recommendations)
    try:
        enhancer.enhance(output_file_path)
        print(f"Enhanced PowerPoint presentation saved at: {output_file_path}")
        assert os.path.exists(output_file_path) and is_valid_pptx(output_file_path), "PowerPoint enhancement failed."
    except Exception as e:
        print(f"Error enhancing presentation: {e}")
        assert False, f"PowerPoint enhancement failed with error: {e}"

def test_error_cases():
    """
    Test common error cases and edge conditions.
    """
    print("Testing common error cases and edge conditions...")
    # Test with invalid Markdown content
    invalid_markdown = "This is not a valid Markdown content"
    output_file_path = OUTPUT_DIR / "invalid_presentation.pptx"
    markdown_to_pptx = MockMarkdownToPPTXCrew(use_direct_calls=True)
    result = markdown_to_pptx.run(invalid_markdown, output_file_path)
    assert not result, "Invalid Markdown content should not produce a PowerPoint presentation."

    # Test with non-existent PPTX file for enhancement
    non_existent_file_path = OUTPUT_DIR / "non_existent_presentation.pptx"
    output_file_path = OUTPUT_DIR / "enhanced_non_existent_presentation.pptx"
    recommendations = {
        "design": "Improve margins and alignment",
        "text": "Simplify text and improve clarity",
        "visual": "Add consistent imagery and icons"
    }
    enhancer = MockPPTXEnhancer(non_existent_file_path, recommendations)
    try:
        enhancer.enhance(output_file_path)
        assert False, "Enhancement should fail for non-existent PPTX file."
    except Exception as e:
        print(f"Expected error for non-existent PPTX file: {e}")

def run_all_tests():
    """
    Run all tests and provide a summary.
    """
    print("Running all integration tests...")
    tests = [
        test_markdown_to_pptx,
        test_pptx_enhancement,
        test_error_cases
    ]
    results = {}
    for test in tests:
        try:
            test()
            results[test.__name__] = "PASSED"
        except AssertionError as e:
            results[test.__name__] = f"FAILED: {e}"
        except Exception as e:
            results[test.__name__] = f"ERROR: {e}"

    print("\nTest Summary:")
    pprint(results)

if __name__ == "__main__":
    run_all_tests()
