"""
Tests for the Markdown Analyzer Agent.
"""

import unittest
import json
import textwrap
from agentic.agents.markdown_analyzer import (
    MarkdownAnalyzerAgent, 
    AnalyzeMarkdownStructureTool,
    PlanSlideConversionTool
)


class TestMarkdownAnalyzerTools(unittest.TestCase):
    """Test case for Markdown Analyzer tools."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyze_tool = AnalyzeMarkdownStructureTool()
        self.plan_tool = PlanSlideConversionTool()
        
        # Sample markdown content for testing - using textwrap.dedent to remove indentation
        self.sample_md = textwrap.dedent("""
        # Test Presentation
        
        Introduction text
        
        ## First Section
        
        - Bullet 1
        - Bullet 2
        
        ## Code Example
        
        ```python
        print("Hello")
        ```
        
        ## Table
        
        | Col1 | Col2 |
        |------|------|
        | A    | B    |
        """)
    
    def test_analyze_tool(self):
        """Test the AnalyzeMarkdownStructureTool."""
        result = self.analyze_tool._run(self.sample_md)
        
        # Parse the result
        analysis = json.loads(result)
        
        # Check that the analysis contains expected fields
        self.assertIn("summary", analysis)
        self.assertIn("headers", analysis)
        self.assertIn("lists", analysis)
        self.assertIn("code_blocks", analysis)
        
        # Check header counts
        self.assertEqual(analysis["summary"]["headers"]["h1"], 1)
        self.assertEqual(analysis["summary"]["headers"]["h2"], 3)
        
        # Check list detection
        self.assertGreaterEqual(len(analysis["lists"]), 1)
        
        # Check code block detection
        self.assertEqual(len(analysis["code_blocks"]), 1)
        self.assertEqual(analysis["code_blocks"][0]["language"], "python")
    
    def test_plan_tool(self):
        """Test the PlanSlideConversionTool."""
        result = self.plan_tool._run(self.sample_md)
        
        # Parse the result
        plan = json.loads(result)
        
        # Check that the plan contains expected fields
        self.assertIn("total_slides", plan)
        self.assertIn("slides", plan)
        
        # Check slide count - we expect 4 slides (title + 3 sections)
        self.assertGreaterEqual(plan["total_slides"], 4)
        
        # Check that each slide has a layout recommendation
        for slide in plan["slides"]:
            self.assertIn("layout_recommendation", slide)


class TestMarkdownAnalyzerAgent(unittest.TestCase):
    """Test case for the MarkdownAnalyzerAgent."""
    
    @unittest.skip("Skip full agent test which requires LLM API")
    def test_analyze_markdown(self):
        """Test the analyze_markdown method of the agent."""
        # Create the agent
        agent = MarkdownAnalyzerAgent()
        
        # Sample markdown
        sample_md = """
        # Simple Presentation
        
        ## Slide One
        
        - Point A
        - Point B
        """
        
        # Run analysis
        result = agent.analyze_markdown(sample_md)
        
        # Check that the result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Basic validation of result structure
        # Note: exact structure depends on the agent's implementation
        self.assertFalse("error" in result)


if __name__ == "__main__":
    unittest.main()
