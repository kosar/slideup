import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the crew directly
from agentic.crews.pptx_enhancement_crew import PPTXEnhancementCrew

def main():
    """Demo script showing how to use the PPTX Enhancement Crew."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize the crew
    crew = PPTXEnhancementCrew(api_key=api_key)
    
    # Define input and output files
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # This is /Users/kosar/src/slideup
    input_file = project_root / "data/sample_presentation.pptx"
    output_file = project_root / "output/enhanced_presentation.pptx"
    
    # Make sure output directory exists
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Define enhancement options
    enhancement_options = {
        "improve_design": True,
        "improve_content": True,
        "improve_structure": True,
        "verbosity": "medium"  # Options: low, medium, high
    }
    
    print(f"Enhancing presentation: {input_file}")
    enhanced_file = crew.enhance_presentation(
        input_file=str(input_file),
        output_file=str(output_file),
        enhancement_options=enhancement_options
    )
    
    print(f"Enhanced presentation saved to: {enhanced_file}")

if __name__ == "__main__":
    main()
