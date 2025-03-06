"""
A CrewAI agent that analyzes Markdown content and converts it to a structured format
suitable for PowerPoint presentation generation.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional

from pydantic import Field
from crewai import Agent
from crewai.tools import BaseTool

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from agentic.utils.markdown_parser import MarkdownParser
from agentic.utils.logger import setup_logger


logger = setup_logger("MarkdownAnalyzer")


class AnalyzeMarkdownStructureTool(BaseTool):
    """Tool that analyzes Markdown structure using the MarkdownParser utility."""
    name: str = "Analyze Markdown Structure"
    description: str = "Analyzes the structure of a Markdown document to identify headers, lists, code blocks, tables, and images"
    markdown_parser: Optional[Any] = Field(default=None, exclude=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        self.markdown_parser = MarkdownParser()
    
    def _run(self, markdown_content: str) -> str:
        """
        Analyze the structure of a Markdown document.
        
        Args:
            markdown_content: The Markdown content to analyze
            
        Returns:
            JSON string representing the analyzed structure
        """
        try:
            # Extract structural elements
            headers = self.markdown_parser.extract_headers(markdown_content)
            lists = self.markdown_parser.extract_lists(markdown_content)
            code_blocks = self.markdown_parser.extract_code_blocks(markdown_content)
            tables = self.markdown_parser.extract_tables(markdown_content)
            images = self.markdown_parser.extract_images(markdown_content)
            
            # Count header types
            h1_count = sum(1 for h in headers if h['level'] == 1)
            h2_count = sum(1 for h in headers if h['level'] == 2)
            h3_plus_count = sum(1 for h in headers if h['level'] >= 3)
            
            # Create the structured analysis
            analysis = {
                "summary": {
                    "headers": {
                        "h1": h1_count,
                        "h2": h2_count,
                        "h3_plus": h3_plus_count,
                        "total": len(headers)
                    },
                    "lists": len(lists),
                    "code_blocks": len(code_blocks),
                    "tables": len(tables),
                    "images": len(images)
                },
                "headers": headers,
                "lists": lists,
                "code_blocks": code_blocks,
                "tables": tables,
                "images": images
            }
            
            return json.dumps(analysis, indent=2)
        
        except Exception as e:
            logger.error(f"Error analyzing Markdown structure: {e}")
            return json.dumps({"error": str(e)})


class PlanSlideConversionTool(BaseTool):
    """Tool that plans how to convert Markdown to slides."""
    name: str = "Plan Slide Conversion"
    description: str = "Plans how to convert Markdown content into PowerPoint slides by determining slide boundaries and layouts"
    markdown_parser: Optional[Any] = Field(default=None, exclude=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        self.markdown_parser = MarkdownParser()
    
    def _run(self, markdown_content: str) -> str:
        """
        Plan the conversion of Markdown content to slides.
        
        Args:
            markdown_content: The Markdown content to convert
            
        Returns:
            JSON string with the slide plan
        """
        try:
            # Use the MarkdownParser to extract slide content
            slides_data = self.markdown_parser.md_to_slides_content(markdown_content)
            
            # Determine if there's a title slide
            has_title_slide = len(slides_data) > 0 and any(
                item.get('type') == 'subheading' for item in slides_data[0].get('content', [])
            )
            
            # Create the slide plan
            slide_plan = {
                "total_slides": len(slides_data),
                "has_title_slide": has_title_slide,
                "slides": []
            }
            
            # Process each slide
            for slide_data in slides_data:
                slide = {
                    "title": slide_data.get('title', 'Untitled Slide'),
                    "content": slide_data.get('content', []),
                    "layout_recommendation": "title_slide" if slide_data.get('title') == slides_data[0].get('title') and has_title_slide else "content_slide"
                }
                
                # Set more specific layout recommendations based on content
                if any(item.get('type') == 'code' for item in slide.get('content', [])):
                    slide["layout_recommendation"] = "code_slide"
                elif any(item.get('type') == 'table' for item in slide.get('content', [])):
                    slide["layout_recommendation"] = "table_slide"
                elif any(item.get('type') == 'image' for item in slide.get('content', [])):
                    slide["layout_recommendation"] = "image_slide"
                
                slide_plan["slides"].append(slide)
            
            return json.dumps(slide_plan, indent=2)
        
        except Exception as e:
            logger.error(f"Error planning slide conversion: {e}")
            return json.dumps({"error": str(e)})


class EnhanceSlideContentTool(BaseTool):
    """Tool that enhances slide content with additional details."""
    name: str = "Enhance Slide Content"
    description: str = "Enhances slide content by generating additional bullet points, suggestions for visuals, or explanatory text"
    
    def _run(self, slide_json: str) -> str:
        """
        Enhance the content of a slide.
        
        Args:
            slide_json: JSON string representing a slide
            
        Returns:
            Enhanced slide JSON string
        """
        try:
            slide = json.loads(slide_json)
            
            # For a title slide, add some suggested bullet points
            if slide.get("layout_recommendation") == "title_slide" and not any(
                item.get('type') == 'list' for item in slide.get('content', [])
            ):
                slide["content"].append({
                    "type": "list",
                    "list_type": "unordered",
                    "items": [
                        "First key point about the presentation topic",
                        "Second important consideration to keep in mind",
                        "Third insight that adds value to the audience",
                        "Final takeaway that summarizes the message"
                    ]
                })
                slide["enhanced"] = True
            
            return json.dumps(slide, indent=2)
            
        except Exception as e:
            logger.error(f"Error enhancing slide content: {e}")
            return slide_json  # Return original content if enhancement fails


class MarkdownAnalyzerAgent:
    """
    Agent that analyzes Markdown content and prepares it for conversion to PowerPoint.
    """
    
    def __init__(self):
        self.analyze_tool = AnalyzeMarkdownStructureTool()
        self.plan_tool = PlanSlideConversionTool()
        self.enhance_tool = EnhanceSlideContentTool()
        self.agent = self._setup_agent()
    
    def _setup_agent(self) -> Agent:
        """
        Set up the CrewAI agent with the necessary tools.
        """
        return Agent(
            role="Markdown Presentation Analyst",
            goal="Convert Markdown documents into well-structured slide presentations",
            backstory="Experienced in analyzing documents and creating great presentations from them.",
            tools=[self.analyze_tool, self.plan_tool, self.enhance_tool],
            verbose=False,
            allow_delegation=False
        )
    
    def analyze_markdown(self, markdown_content: str) -> Dict[str, Any]:
        """
        Direct interface to analyze Markdown without going through the LLM.
        
        Args:
            markdown_content: The Markdown content to analyze
            
        Returns:
            The slide plan as a Python dictionary
        """
        # Analyze the structure
        analysis_json = self.analyze_tool._run(markdown_content)
        
        # Plan the slides
        plan_json = self.plan_tool._run(markdown_content)
        plan = json.loads(plan_json)
        
        # Enhance each slide
        for i, slide in enumerate(plan.get("slides", [])):
            enhanced_slide_json = self.enhance_tool._run(json.dumps(slide))
            plan["slides"][i] = json.loads(enhanced_slide_json)
        
        return plan


# Test function to demonstrate usage
def test_markdown_analyzer():
    """Test the MarkdownAnalyzerAgent with a sample Markdown string."""
    sample_md = """
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
    
    analyzer = MarkdownAnalyzerAgent()
    result = analyzer.analyze_markdown(sample_md)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    test_markdown_analyzer()
