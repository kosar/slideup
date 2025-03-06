from crewai import Agent
from crewai.tools import BaseTool
from agentic.agents.pptx_enhancer import PPTXEnhancer

class EnhancePresentation(BaseTool):
    """Tool for enhancing PowerPoint presentations."""
    
    name: str = "enhance_presentation"
    description: str = "Enhance a PowerPoint presentation to improve its design, content, and structure"
    
    def __init__(self, enhancer_func):
        super().__init__()
        self._enhancer_func = enhancer_func
    
    def _run(self, input_file: str, output_file: str, enhancement_options: dict = None) -> str:
        return self._enhancer_func(input_file, output_file, enhancement_options)

class PPTXEnhancerAgent:
    """Agent wrapper for the PPTXEnhancer class."""
    
    def __init__(self, api_key=None):
        """Initialize the PPTX Enhancer Agent.
        
        Args:
            api_key: API key for LLM services
        """
        self.api_key = api_key
        
        # Create the tool with the enhancer function
        enhance_tool = EnhancePresentation(enhancer_func=self.enhance_presentation)
        
        self.agent = Agent(
            role="PowerPoint Enhancement Specialist",
            goal="Enhance PowerPoint presentations to be more visually appealing and effective",
            backstory=(
                "You are an expert in enhancing PowerPoint presentations with years of "
                "experience in design, content structure, and visual communication. "
                "You specialize in turning ordinary presentations into professional, "
                "impactful decks."
            ),
            verbose=True,
            allow_delegation=False,
            tools=[enhance_tool]
        )
        
    def enhance_presentation(self, input_file, output_file, enhancement_options):
        """
        Enhance a PowerPoint presentation.
        
        Args:
            input_file: Path to the input PPTX file
            output_file: Path where the enhanced PPTX will be saved
            enhancement_options: Dictionary of enhancement options
        
        Returns:
            String with the path to the enhanced presentation and a summary of changes
        """
        # Map the enhancement options to recommendations format expected by PPTXEnhancer
        recommendations = {
            "design": "Improve margins and alignment",
            "text": "Simplify text and improve clarity",
            "visual": "Add consistent imagery and icons"
        }
        
        if enhancement_options.get("improve_design", True):
            recommendations["design"] = "Improve margins, alignment, and overall layout design"
        
        if enhancement_options.get("improve_content", True):
            recommendations["text"] = "Simplify text, improve clarity and messaging effectiveness"
        
        if enhancement_options.get("improve_structure", True):
            recommendations["visual"] = "Add consistent imagery, icons and improve slide structure"
        
        # Initialize the enhancer and enhance the presentation
        enhancer = PPTXEnhancer(input_file, recommendations)
        enhancer.enhance(output_file)
        
        return f"Enhanced presentation saved to {output_file} with the following improvements: {recommendations}"
