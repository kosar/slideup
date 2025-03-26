"""
API Cost Tracking Module

This module provides functionality to track API costs for various services used in the podcast2video application.
It calculates costs for OpenAI API (GPT models, audio transcription) and Stability AI image generation.

Core functionality:
- Track API calls and their costs in real-time
- Calculate accumulated costs for different operations
- Provide cost summaries and detailed breakdowns
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

# Set up logging
logger = logging.getLogger('cost_tracker')

# Define API cost constants
class OpenAICostConfig:
    """OpenAI API cost constants"""
    # GPT Models - costs per 1K input/output tokens
    GPT_4_INPUT = 0.03
    GPT_4_OUTPUT = 0.06
    GPT_35_TURBO_INPUT = 0.0015
    GPT_35_TURBO_OUTPUT = 0.002
    # Audio transcription - cost per minute
    WHISPER_API = 0.006  # $0.006 per minute

class StabilityCostConfig:
    """Stability API cost constants"""
    # Image generation costs per image in dollars (1 credit = $0.01)
    # Based on Stability AI's pricing: https://platform.stability.ai/pricing
    
    # Models and their costs
    STABLE_IMAGE_ULTRA = 0.08       # 8 credits ($0.08)
    STABLE_DIFFUSION_3_LARGE = 0.065  # 6.5 credits ($0.065)
    STABLE_DIFFUSION_3_TURBO = 0.04   # 4 credits ($0.04)
    STABLE_DIFFUSION_3_MEDIUM = 0.035 # 3.5 credits ($0.035)
    STABLE_IMAGE_CORE = 0.03        # 3 credits ($0.03)
    SDXL_1_0 = 0.009               # 0.9 credits ($0.009)
    SD_1_6 = 0.009                 # 0.9 credits ($0.009)
    
    # Default model cost - SDXL is used in the app (stable-diffusion-xl-1024-v1-0)
    DEFAULT_MODEL_COST = SDXL_1_0
    
    # Legacy size-based costs (kept for backward compatibility)
    IMAGE_512x512 = DEFAULT_MODEL_COST
    IMAGE_768x768 = DEFAULT_MODEL_COST
    IMAGE_1024x1024 = DEFAULT_MODEL_COST
    
    # Adjustments for steps
    STEPS_MULTIPLIER = {
        30: 1.0,  # Base cost for 30 steps
        50: 1.0,  # Same cost for 50 steps
        100: 1.0  # Same cost for 100 steps
    }

class CostEntry:
    """Represents a single cost entry for an API call"""
    
    def __init__(self, 
                 api_type: str, 
                 operation: str,
                 cost: float,
                 details: Dict[str, Any],
                 timestamp: Optional[datetime] = None):
        self.api_type = api_type  # 'openai' or 'stability'
        self.operation = operation  # 'chat', 'transcription', 'image'
        self.cost = cost  # in USD
        self.details = details  # Additional details like tokens, model, etc.
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert the entry to a dictionary"""
        return {
            'api_type': self.api_type,
            'operation': self.operation,
            'cost': self.cost,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CostEntry':
        """Create a CostEntry from a dictionary"""
        return cls(
            api_type=data['api_type'],
            operation=data['operation'],
            cost=data['cost'],
            details=data['details'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

class CostTracker:
    """Main class for tracking API costs"""
    
    def __init__(self):
        self.entries: List[CostEntry] = []
        self.total_cost: float = 0.0
        self.start_time = datetime.now()
        self.api_totals = {
            'openai': {
                'chat': 0.0,
                'transcription': 0.0
            },
            'stability': {
                'image': 0.0
            }
        }
    
    def add_openai_chat_cost(self, 
                             model: str, 
                             input_tokens: int, 
                             output_tokens: int, 
                             operation_name: str = "chat") -> float:
        """
        Calculate and track the cost of an OpenAI chat completion call
        
        Args:
            model (str): The model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
            input_tokens (int): Number of input tokens
            output_tokens (int): Number of output tokens
            operation_name (str): A descriptive name for this operation
            
        Returns:
            float: The cost of this operation
        """
        # Calculate cost based on model
        if 'gpt-4' in model:
            input_cost = (input_tokens / 1000) * OpenAICostConfig.GPT_4_INPUT
            output_cost = (output_tokens / 1000) * OpenAICostConfig.GPT_4_OUTPUT
        else:  # Default to gpt-3.5-turbo pricing
            input_cost = (input_tokens / 1000) * OpenAICostConfig.GPT_35_TURBO_INPUT
            output_cost = (output_tokens / 1000) * OpenAICostConfig.GPT_35_TURBO_OUTPUT
        
        total_cost = input_cost + output_cost
        
        # Create details for this entry
        details = {
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost
        }
        
        # Add the entry
        entry = CostEntry(
            api_type='openai',
            operation=operation_name,
            cost=total_cost,
            details=details
        )
        self.entries.append(entry)
        
        # Update totals
        self.total_cost += total_cost
        self.api_totals['openai']['chat'] += total_cost
        
        logger.debug(f"Added OpenAI chat cost: ${total_cost:.6f} for {operation_name} using {model}")
        return total_cost
    
    def add_openai_transcription_cost(self, 
                                     duration_seconds: float,
                                     model: str = "whisper-1",
                                     operation_name: str = "transcription") -> float:
        """
        Calculate and track the cost of an OpenAI audio transcription
        
        Args:
            duration_seconds (float): Audio duration in seconds
            model (str): The model name (currently only whisper-1 supported)
            operation_name (str): A descriptive name for this operation
            
        Returns:
            float: The cost of this operation
        """
        # Calculate cost (Whisper API charges per minute)
        duration_minutes = duration_seconds / 60
        total_cost = duration_minutes * OpenAICostConfig.WHISPER_API
        
        # Create details for this entry
        details = {
            'model': model,
            'duration_seconds': duration_seconds,
            'duration_minutes': duration_minutes
        }
        
        # Add the entry
        entry = CostEntry(
            api_type='openai',
            operation=operation_name,
            cost=total_cost,
            details=details
        )
        self.entries.append(entry)
        
        # Update totals
        self.total_cost += total_cost
        self.api_totals['openai']['transcription'] += total_cost
        
        logger.debug(f"Added OpenAI transcription cost: ${total_cost:.6f} for {duration_seconds}s audio")
        return total_cost
    
    def add_stability_image_cost(self,
                                 width: int,
                                 height: int,
                                 steps: int = 30,
                                 samples: int = 1,
                                 model: str = "stable-diffusion-xl-1024-v1-0",
                                 operation_name: str = "image") -> float:
        """
        Calculate and track the cost of a Stability API image generation
        
        Args:
            width (int): Image width
            height (int): Image height
            steps (int): Number of diffusion steps
            samples (int): Number of images generated
            model (str): The Stability AI model used
            operation_name (str): A descriptive name for this operation
            
        Returns:
            float: The cost of this operation
        """
        # Determine base cost based on model
        if model == "stable-diffusion-xl-1024-v1-0" or model.startswith("sdxl"):
            base_cost = StabilityCostConfig.SDXL_1_0
        elif model == "stable-diffusion-v1-6" or model.startswith("sd-1"):
            base_cost = StabilityCostConfig.SD_1_6
        elif "stable-image-ultra" in model:
            base_cost = StabilityCostConfig.STABLE_IMAGE_ULTRA
        elif "stable-diffusion-3-large" in model and "turbo" in model:
            base_cost = StabilityCostConfig.STABLE_DIFFUSION_3_TURBO
        elif "stable-diffusion-3-large" in model:
            base_cost = StabilityCostConfig.STABLE_DIFFUSION_3_LARGE
        elif "stable-diffusion-3-medium" in model:
            base_cost = StabilityCostConfig.STABLE_DIFFUSION_3_MEDIUM  
        elif "stable-image-core" in model:
            base_cost = StabilityCostConfig.STABLE_IMAGE_CORE
        else:
            # Fall back to legacy sizing logic for unrecognized models
            if width <= 512 and height <= 512:
                base_cost = StabilityCostConfig.IMAGE_512x512
            elif width <= 768 and height <= 768:
                base_cost = StabilityCostConfig.IMAGE_768x768
            else:
                base_cost = StabilityCostConfig.IMAGE_1024x1024
        
        # Adjust for steps - find closest step count in our config
        steps_key = min(StabilityCostConfig.STEPS_MULTIPLIER.keys(), 
                      key=lambda k: abs(k - steps))
        steps_multiplier = StabilityCostConfig.STEPS_MULTIPLIER[steps_key]
        
        # Calculate total
        total_cost = base_cost * steps_multiplier * samples
        
        # Create details for this entry
        details = {
            'model': model,
            'width': width,
            'height': height,
            'steps': steps,
            'samples': samples,
            'base_cost': base_cost,
            'steps_multiplier': steps_multiplier
        }
        
        # Add the entry
        entry = CostEntry(
            api_type='stability',
            operation=operation_name,
            cost=total_cost,
            details=details
        )
        self.entries.append(entry)
        
        # Update totals
        self.total_cost += total_cost
        self.api_totals['stability']['image'] += total_cost
        
        logger.debug(f"Added Stability image cost: ${total_cost:.6f} for {width}x{height} image, model: {model}")
        return total_cost
    
    def get_current_cost(self) -> float:
        """Get the current total cost"""
        return self.total_cost
    
    def get_summary(self) -> Dict:
        """Get a summary of all costs"""
        return {
            'total_cost': self.total_cost,
            'start_time': self.start_time.isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'api_breakdown': self.api_totals,
            'entry_count': len(self.entries)
        }
    
    def get_detailed_report(self) -> Dict:
        """Get a detailed report including all entries"""
        summary = self.get_summary()
        summary['entries'] = [entry.to_dict() for entry in self.entries]
        return summary
    
    def save_report(self, file_path: str) -> None:
        """Save the detailed cost report to a JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.get_detailed_report(), f, indent=2)
        logger.info(f"Cost report saved to {file_path}")

    def reset(self) -> None:
        """Reset the cost tracker"""
        self.entries = []
        self.total_cost = 0.0
        self.start_time = datetime.now()
        self.api_totals = {
            'openai': {
                'chat': 0.0,
                'transcription': 0.0
            },
            'stability': {
                'image': 0.0
            }
        }
        logger.info("Cost tracker reset")


# Create a singleton instance to use throughout the application
_cost_tracker = CostTracker()

def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance"""
    return _cost_tracker 