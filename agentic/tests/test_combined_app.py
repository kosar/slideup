import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentic.crews.markdown2pptx_crew import Markdown2PPTXCrew
from agentic.crews.pptx_enhancement_crew import PPTXEnhancementCrew

def test_markdown_to_pptx():
    """Test the Markdown to PowerPoint conversion"""
    
    # Sample markdown content
    markdown_content = """# Sample Presentation
    
## Introduction
- This is a sample presentation
- Created for testing purposes

## Key Points
1. First important point
2. Second important point
3. Third important point

## Conclusion
Thank you for testing!
    """
    
    # Set up temp files
    temp_dir = tempfile.mkdtemp()
    markdown_path = os.path.join(temp_dir, "test.md")
    output_path = os.path.join(temp_dir, "output.pptx")
    
    # Write markdown to file
    with open(markdown_path, "w") as f:
        f.write(markdown_content)
    
    # Run the crew
    crew = Markdown2PPTXCrew()
    result = crew.run(
        markdown_file=markdown_path,
        output_file=output_path,
        theme="Professional",
        color_scheme="Blue",
        include_cover=True,
        include_agenda=True
    )
    
    # Check results
    assert os.path.exists(output_path), "Output PPTX file was not created"
    print(f"Created presentation at: {output_path}")
    
    return output_path

def test_pptx_enhancement(input_pptx):
    """Test the PowerPoint enhancement functionality"""
    
    # Set up temp files
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "enhanced.pptx")
    
    # Run the crew
    crew = PPTXEnhancementCrew()
    result = crew.run(
        input_file=input_pptx,
        output_file=output_path,
        improve_visuals=True,
        enhance_content=True,
        optimize_structure=True,
        improve_formatting=True,
        enhancement_level="moderate",
        additional_instructions="Make it look more professional"
    )
    
    # Check results
    assert os.path.exists(output_path), "Enhanced PPTX file was not created"
    print(f"Created enhanced presentation at: {output_path}")

if __name__ == "__main__":
    # Test markdown to pptx
    print("Testing Markdown to PowerPoint conversion...")
    pptx_file = test_markdown_to_pptx()
    
    # Test pptx enhancement
    print("\nTesting PowerPoint enhancement...")
    test_pptx_enhancement(pptx_file)  # Fixed the missing argument