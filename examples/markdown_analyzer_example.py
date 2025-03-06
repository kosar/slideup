"""
Example demonstrating the Markdown Analyzer Agent.
"""

import os
import json
from pathlib import Path

from agentic.agents.markdown_analyzer import MarkdownAnalyzerAgent


def analyze_markdown_file(file_path: str):
    """
    Analyze a markdown file and print the results.
    
    Args:
        file_path: Path to the markdown file
    """
    # Read the markdown file
    with open(file_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    print(f"Analyzing markdown file: {file_path}")
    print(f"Content length: {len(markdown_content)} characters")
    
    # Create the analyzer
    analyzer = MarkdownAnalyzerAgent()
    
    # Analyze the content
    print("\nRunning analysis...")
    result = analyzer.analyze_markdown(markdown_content)
    
    # Save the results
    output_dir = Path("example_output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "markdown_analysis_result.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nAnalysis completed. Results saved to {output_file}")
    
    # Print summary
    print("\nSummary:")
    if "plan" in result and "slides" in result["plan"]:
        print(f"Slides planned: {len(result['plan']['slides'])}")
    elif "total_slides" in result:
        print(f"Slides planned: {result['total_slides']}")
    
    # Extract any enhancement recommendations
    enhanced_slides = sum(1 for slide in result.get("slides", []) 
                         if slide.get("enhanced", False))
    
    print(f"Enhanced slides: {enhanced_slides}")
    print(f"See {output_file} for complete analysis results")


def run_example():
    """Run the Markdown Analyzer example."""
    # Check for the example markdown file
    example_md = Path("examples/example_markdown.md")
    
    if example_md.exists():
        analyze_markdown_file(str(example_md))
    else:
        # Create a simple example markdown file
        example_content = """# Example Presentation

A demonstration of the Markdown Analyzer

## Introduction

- This is a sample presentation
- Created to test the Markdown Analyzer
- Shows various Markdown elements

## Code Example
```
print("Hello, World!")
```

## Conclusion

- The Markdown Analyzer is a useful tool
- It helps in analyzing and enhancing markdown content
"""

        with open(example_md, 'w', encoding='utf-8') as f:
            f.write(example_content)
        
        print(f"Created example markdown file: {example_md}")
        analyze_markdown_file(str(example_md))


if __name__ == "__main__":
    run_example()

