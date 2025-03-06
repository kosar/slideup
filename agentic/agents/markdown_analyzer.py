"""
Markdown Analyzer Agent for converting Markdown to PowerPoint.

This agent analyzes Markdown structure and plans its conversion to slides.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import re
from pydantic import PrivateAttr

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool

from ..utils.markdown_parser import MarkdownParser
from ..utils.llm_interface import LLMInterface
from ..utils.logger import setup_logger, ErrorHandler


class AnalyzeMarkdownStructureTool(BaseTool):
    """Tool to analyze the structure of a Markdown document."""
    
    name: str = "Analyze Markdown Structure"
    description: str = "Analyzes the structure of a Markdown document to identify headers, lists, code blocks, tables, and images"
    
    # Use Pydantic's PrivateAttr for attributes not in the schema
    _md_parser: MarkdownParser = PrivateAttr(default_factory=MarkdownParser)
    
    def _run(self, markdown_content: str) -> str:
        """
        Analyze the structure of a Markdown document.
        
        Args:
            markdown_content: The Markdown content to analyze
            
        Returns:
            JSON string containing the analysis results
        """
        # Extract structural elements
        headers = self._md_parser.extract_headers(markdown_content)
        lists = self._md_parser.extract_lists(markdown_content)
        code_blocks = self._md_parser.extract_code_blocks(markdown_content)
        tables = self._md_parser.extract_tables(markdown_content)
        images = self._md_parser.extract_images(markdown_content)
        
        # Count different elements for summary
        h1_count = sum(1 for h in headers if h['level'] == 1)
        h2_count = sum(1 for h in headers if h['level'] == 2)
        h3_plus_count = sum(1 for h in headers if h['level'] > 2)
        
        # Create analysis results
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


class PlanSlideConversionTool(BaseTool):
    """Tool to plan how to convert Markdown elements into PowerPoint slides."""
    
    name: str = "Plan Slide Conversion"
    description: str = "Plans how to convert Markdown content into PowerPoint slides by determining slide boundaries and layouts"
    
    # Use Pydantic's PrivateAttr for attributes not in the schema
    _md_parser: MarkdownParser = PrivateAttr(default_factory=MarkdownParser)
    
    def _run(self, markdown_content: str) -> str:
        """
        Plan the conversion of Markdown content into PowerPoint slides.
        
        Args:
            markdown_content: The Markdown content to convert
            
        Returns:
            JSON string containing the slide plan
        """
        # Convert markdown to slides content using our existing utility
        slides_data = self._md_parser.md_to_slides_content(markdown_content)
        
        # Add layout recommendations based on content
        for slide in slides_data:
            slide['layout_recommendation'] = self._recommend_layout(slide)
        
        slide_plan = {
            "total_slides": len(slides_data),
            "has_title_slide": bool(slides_data and "title" in slides_data[0]),
            "slides": slides_data
        }
        
        return json.dumps(slide_plan, indent=2)
    
    def _recommend_layout(self, slide_data: Dict[str, Any]) -> str:
        """
        Recommend a slide layout based on content.
        
        Args:
            slide_data: Data for a single slide
            
        Returns:
            Recommended layout type
        """
        content = slide_data.get('content', [])
        
        # Check for content types
        has_images = any(item['type'] == 'image' for item in content)
        has_code = any(item['type'] == 'code' for item in content)
        has_table = any(item['type'] == 'table' for item in content)
        has_list = any(item['type'] == 'list' for item in content)
        
        # Make recommendations based on content types
        if slide_data.get('title', '').lower().startswith(('title', 'introduction')):
            return "title_slide"
        elif has_images and has_code:
            return "blank_with_title"  # Complex slide needs custom layout
        elif has_images and not (has_code or has_table):
            return "picture_with_caption"
        elif has_code and not has_images:
            return "code_slide"
        elif has_table and not has_images:
            return "table_slide"
        elif has_list and not (has_images or has_code or has_table):
            return "bullet_slide"
        else:
            return "content_slide"  # Default layout


class EnhanceSlideContentTool(BaseTool):
    """Tool to enhance slide content using LLM."""
    
    name: str = "Enhance Slide Content"
    description: str = "Enhances slide content by generating additional bullet points, suggestions for visuals, or explanatory text"
    
    # Use Pydantic's PrivateAttr for attributes not in the schema  
    _llm: LLMInterface = PrivateAttr(default_factory=LLMInterface)
    
    def _run(self, slide_json: str) -> str:
        """
        Enhance slide content with LLM-generated suggestions.
        
        Args:
            slide_json: JSON string containing slide data
            
        Returns:
            Enhanced slide content as JSON string
        """
        try:
            slide = json.loads(slide_json)
            title = slide.get('title', '')
            
            # Generate enhancements based on slide content
            if not any(item['type'] == 'list' for item in slide.get('content', [])):
                # Generate bullet points for slides without lists
                prompt = f"Create 3-5 key bullet points for a presentation slide titled '{title}'. Make them concise and informative."
                response = self._llm.completion(prompt=prompt)
                
                # Parse bullet points from response
                bullet_points = [
                    line.strip().lstrip('•-*').strip() 
                    for line in response.split('\n') 
                    if line.strip() and any(line.strip().startswith(c) for c in '•-*')
                ]
                
                if not bullet_points:
                    # If no bullet markers were found, try splitting by lines
                    bullet_points = [line.strip() for line in response.split('\n') if line.strip()]
                
                if bullet_points:
                    slide['content'].append({
                        'type': 'list',
                        'list_type': 'unordered',
                        'items': bullet_points
                    })
                    slide['enhanced'] = True
            
            return json.dumps(slide, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to enhance slide content: {str(e)}",
                "original_slide": slide_json
            })


class MarkdownAnalyzerAgent:
    """Agent that analyzes Markdown and plans its conversion to PowerPoint."""
    
    def __init__(
        self,
        llm_interface: Optional[LLMInterface] = None,
        markdown_parser: Optional[MarkdownParser] = None
    ):
        """
        Initialize the Markdown Analyzer Agent.
        
        Args:
            llm_interface: LLM interface for text generation
            markdown_parser: Markdown parser utility
        """
        self.logger = setup_logger("markdown_analyzer")
        self.error_handler = ErrorHandler(self.logger)
        
        # Set up tools
        self.analyze_tool = AnalyzeMarkdownStructureTool()
        self.plan_tool = PlanSlideConversionTool()
        self.enhance_tool = EnhanceSlideContentTool()
        
        # Create CrewAI agent
        self.agent = self._setup_agent()
    
    def _setup_agent(self) -> Agent:
        """Set up the CrewAI agent with appropriate tools and configuration."""
        return Agent(
            role="Markdown Presentation Analyst",
            goal="Analyze Markdown content and create optimal PowerPoint conversion plans",
            backstory="""
            You are an expert in document analysis and presentation design. 
            Your specialty is taking Markdown content and transforming it into 
            well-structured PowerPoint presentations. You understand both the 
            semantic structure of documents and the visual principles of effective slides.
            """,
            tools=[
                self.analyze_tool, 
                self.plan_tool,
                self.enhance_tool
            ],
            verbose=True,
            allow_delegation=False
        )
    
    def analyze_markdown(self, markdown_content: str) -> Dict[str, Any]:
        """
        Analyze Markdown content and plan its conversion to slides.
        
        Args:
            markdown_content: Markdown content as a string
            
        Returns:
            Dictionary containing analysis and slide plan
        """
        try:
            # Create a task for the agent
            task = Task(
                description=f"""
                Analyze the following Markdown content:
                
                ---
                {markdown_content[:1000]}...
                ---
                
                1. Analyze the markdown structure to identify all key components 
                2. Create a slide conversion plan with appropriate slide layouts
                3. Enhance content for slides that need more information
                4. Provide a summary of your analysis and recommendations
                
                Return a detailed conversion plan that can be used to create a PowerPoint presentation.
                """,
                agent=self.agent,
                expected_output="A detailed analysis and conversion plan in JSON format"
            )
            
            # Create a simple crew with just this agent
            crew = Crew(
                agents=[self.agent],
                tasks=[task],
                verbose=True
            )
            
            # Execute the task
            result = crew.kickoff()
            
            # Parse the result and return structured data
            # Note: We'll need to extract the JSON from the response
            conversion_plan = self._extract_json_from_response(result)
            return conversion_plan
            
        except Exception as e:
            error_msg = f"Failed to analyze Markdown content"
            self.error_handler.handle_exception(e, error_msg)
            return {
                "error": error_msg,
                "details": str(e),
                "analysis": None,
                "plan": None
            }
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract and parse JSON from the agent's response."""
        try:
            # Look for JSON-like structures in the response
            json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            matches = re.findall(json_pattern, response)
            
            if matches:
                # Try each match until we find valid JSON
                for match in matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON blocks with markers, try to parse the whole thing
            return json.loads(response)
            
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse JSON from response")
            return {
                "raw_response": response,
                "error": "Failed to parse JSON from response"
            }


def test_markdown_analyzer():
    """Test function for the Markdown Analyzer Agent."""
    # Sample Markdown content
    markdown_content = """
    # Test Presentation
    
    This is an introduction paragraph.
    
    ## First Section
    
    - Point 1
    - Point 2
    - Point 3
    
    ## Code Example
    
    ```python
    def hello_world():
        print("Hello, World!")
    ```
    
    ## Table Section
    
    | Header 1 | Header 2 |
    |----------|----------|
    | Cell 1   | Cell 2   |
    | Cell 3   | Cell 4   |
    
    ![Example Image](image.png)
    """
    
    # Create and run the analyzer
    analyzer = MarkdownAnalyzerAgent()
    result = analyzer.analyze_markdown(markdown_content)
    
    # Print the results
    print(json.dumps(result, indent=2))
    
    return result


if __name__ == "__main__":
    test_markdown_analyzer()
