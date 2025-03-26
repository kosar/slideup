"""
podcast2video module
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger('podcast2video')

# Add the parent directory to the Python path if necessary
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import the cost tracking module
try:
    from podcast2video.cost_tracker import (
        CostTracker, 
        CostEntry, 
        OpenAICostConfig, 
        StabilityCostConfig, 
        get_cost_tracker
    )
except ImportError as e:
    logger.error(f"Error importing cost tracker: {e}")

# Expose key functionality
__all__ = [
    'CostTracker',
    'CostEntry',
    'OpenAICostConfig',
    'StabilityCostConfig',
    'get_cost_tracker'
] 