import os
import sys
import shutil
from crewai import Crew, Task
from pathlib import Path

# Add parent directory to path so we can import from agentic.agents
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import our agents with correct class names
from agentic.agents.pptx_analyzer_agent import PptxAnalyzerAgentWithAPIKey
from agentic.agents.pptx_enhancer_agent import PPTXEnhancerAgent

class PPTXEnhancementCrew:
    """Crew for enhancing PowerPoint presentations by analyzing and improving them."""
    
    def __init__(self, api_key=None, verbose=True):
        """
        Initialize the enhancement crew.
        
        Args:
            api_key: API key for LLM services
            verbose: Whether to show detailed output
        """
        self.api_key = api_key
        self.verbose = verbose
        self.analyzer_agent = None
        self.enhancer_agent = None
        
    def setup_agents(self):
        """Set up the analyzer and enhancer agents."""
        self.analyzer_agent = PptxAnalyzerAgentWithAPIKey(api_key=self.api_key)
        self.enhancer_agent = PPTXEnhancerAgent(api_key=self.api_key)
        
    def enhance_presentation(self, input_file, output_file, enhancement_options=None):
        """
        Enhance a PowerPoint presentation.
        
        Args:
            input_file: Path to the input PPTX file
            output_file: Path where the enhanced PPTX will be saved
            enhancement_options: Dictionary of enhancement options:
                - improve_design: Improve visual design (bool)
                - improve_content: Improve content quality (bool)
                - improve_structure: Improve presentation structure (bool)
                - style_guide: Dictionary with style preferences
                - target_audience: Target audience description
                - enhancement_level: How aggressive to be (light, moderate, comprehensive)
        
        Returns:
            Path to the enhanced presentation
        """
        # Default options
        default_options = {
            "improve_design": True,
            "improve_content": True,
            "improve_structure": True,
            "enhancement_level": "moderate",
            "style_guide": {"preferred_colors": [], "preferred_fonts": []},
            "target_audience": "General audience"
        }
        
        # Apply defaults for missing options
        if not enhancement_options:
            enhancement_options = default_options
        else:
            for key, value in default_options.items():
                if key not in enhancement_options:
                    enhancement_options[key] = value
            
        # Ensure input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Create backup
        backup_file = f"{input_file}.backup"
        try:
            shutil.copy2(input_file, backup_file)
            if self.verbose:
                print(f"Created backup at {backup_file}")
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")
            
        # Setup agents if not already done
        if not self.analyzer_agent or not self.enhancer_agent:
            self.setup_agents()
            
        # Create analysis task with properly formatted context items
        analysis_task = Task(
            description=self._create_analysis_prompt(input_file, enhancement_options),
            expected_output="Detailed analysis of the presentation with specific recommendations for improvement",
            agent=self.analyzer_agent.agent,
            tools=[self.analyzer_agent.agent.tools[0]],  # Pass the analyze tool explicitly
            context=[
                {
                    "description": "Presentation file path to analyze",
                    "expected_output": "Analysis results",
                    "file_path": input_file,
                    "enhancement_options": enhancement_options
                }
            ]
        )
        
        # Create enhancement task with properly formatted context items
        enhancement_task = Task(
            description=self._create_enhancement_prompt(input_file, output_file, enhancement_options),
            expected_output="Enhanced PowerPoint presentation with design, content, and structure improvements",
            agent=self.enhancer_agent.agent,
            tools=[self.enhancer_agent.agent.tools[0]],  # Pass the enhance tool explicitly
            context=[
                {
                    "description": "Presentation enhancement parameters",
                    "expected_output": "Enhanced presentation confirmation",
                    "input_file": input_file,
                    "output_file": output_file,
                    "enhancement_options": enhancement_options
                }
            ],
            dependencies=[analysis_task]
        )
        
        # Create and run the crew
        crew = Crew(
            agents=[self.analyzer_agent.agent, self.enhancer_agent.agent],
            tasks=[analysis_task, enhancement_task],
            verbose=self.verbose
        )
        
        try:
            result = crew.kickoff()
            
            # Verify that output file was created
            if not os.path.exists(output_file):
                print(f"Warning: Output file not found at {output_file}")
                print("Using original file as a fallback...")
                shutil.copy2(input_file, output_file)
                
            return output_file
            
        except Exception as e:
            print(f"Error occurred during enhancement: {e}")
            # Try to restore from backup
            if os.path.exists(backup_file):
                print("Restoring from backup...")
                shutil.copy2(backup_file, output_file)
            
            raise
        finally:
            # Clean up backup
            if os.path.exists(backup_file):
                os.remove(backup_file)
    
    def _create_analysis_prompt(self, input_file, options):
        """Create a detailed prompt for the analysis task."""
        prompt = f"Analyze the presentation {input_file} thoroughly and identify areas for improvement.\n\n"
        
        if options["improve_design"]:
            prompt += "- Evaluate the visual design including colors, fonts, layout, and overall consistency.\n"
        
        if options["improve_content"]:
            prompt += "- Assess the content quality, clarity, and messaging effectiveness.\n"
        
        if options["improve_structure"]:
            prompt += "- Review the presentation structure, flow, and logical organization.\n"
        
        prompt += f"\nTarget audience: {options['target_audience']}\n"
        prompt += f"Enhancement level: {options['enhancement_level']}\n"
        
        return prompt
    
    def _create_enhancement_prompt(self, input_file, output_file, options):
        """Create a detailed prompt for the enhancement task."""
        prompt = f"Enhance the presentation from {input_file} and save to {output_file} based on the analysis provided.\n\n"
        
        prompt += "Apply the following improvements:\n"
        if options["improve_design"]:
            prompt += "- Improve the visual design for better aesthetics and clarity.\n"
        
        if options["improve_content"]:
            prompt += "- Enhance the content quality and messaging.\n"
        
        if options["improve_structure"]:
            prompt += "- Improve the presentation structure and flow.\n"
        
        if options["style_guide"]["preferred_colors"]:
            colors = ", ".join(options["style_guide"]["preferred_colors"])
            prompt += f"- Use the following preferred colors: {colors}.\n"
            
        if options["style_guide"]["preferred_fonts"]:
            fonts = ", ".join(options["style_guide"]["preferred_fonts"])
            prompt += f"- Use the following preferred fonts: {fonts}.\n"
        
        prompt += f"\nTarget audience: {options['target_audience']}\n"
        prompt += f"Enhancement level: {options['enhancement_level']}\n"
        
        return prompt
