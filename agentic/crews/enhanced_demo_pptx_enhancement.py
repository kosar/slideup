import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the crew directly
from agentic.crews.pptx_enhancement_crew import PPTXEnhancementCrew

def main():
    """Enhanced demo script showing how to use the PPTX Enhancement Crew with more options."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Enhance a PowerPoint presentation.")
    parser.add_argument("--input", "-i", help="Input PPTX file path")
    parser.add_argument("--output", "-o", help="Output PPTX file path")
    parser.add_argument("--design", "-d", action="store_true", help="Improve design", default=True)
    parser.add_argument("--content", "-c", action="store_true", help="Improve content", default=True)
    parser.add_argument("--structure", "-s", action="store_true", help="Improve structure", default=True)
    parser.add_argument("--level", "-l", choices=["light", "moderate", "comprehensive"], 
                        default="moderate", help="Enhancement level")
    parser.add_argument("--audience", "-a", default="General audience", 
                        help="Target audience (e.g., 'Technical professionals', 'Executive leadership')")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output", default=True)
    
    args = parser.parse_args()
    
    # Load environment variables for API keys
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize the crew
    crew = PPTXEnhancementCrew(api_key=api_key, verbose=args.verbose)
    
    # Define input and output files
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # This is /Users/kosar/src/slideup
    
    input_file = args.input if args.input else project_root / "data/sample_presentation.pptx"
    output_file = args.output if args.output else project_root / "output/enhanced_presentation.pptx"
    
    # Convert to Path objects if they're strings
    if isinstance(input_file, str):
        input_file = Path(input_file)
    if isinstance(output_file, str):
        output_file = Path(output_file)
    
    # Make sure output directory exists
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Define enhancement options
    enhancement_options = {
        "improve_design": args.design,
        "improve_content": args.content,
        "improve_structure": args.structure,
        "enhancement_level": args.level,
        "target_audience": args.audience,
        "style_guide": {
            "preferred_colors": ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"],
            "preferred_fonts": ["Arial", "Calibri", "Georgia"]
        }
    }
    
    print(f"\n===== PPTX Enhancement Demo =====")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Enhancement options: {enhancement_options}")
    print(f"=================================\n")
    
    try:
        enhanced_file = crew.enhance_presentation(
            input_file=str(input_file),
            output_file=str(output_file),
            enhancement_options=enhancement_options
        )
        
        print(f"\n✅ Enhancement complete!")
        print(f"Enhanced presentation saved to: {enhanced_file}")
        
    except Exception as e:
        print(f"\n❌ Enhancement failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
