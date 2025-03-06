"""
Logging utility for the SlideUp application.
"""

import logging
import sys
import traceback
from typing import Optional

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up and configure a logger.
    
    Args:
        name: The name of the logger
        level: The logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler if not already present
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class ErrorHandler:
    """
    Utility class to handle errors consistently.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize with a logger.
        
        Args:
            logger: The logger to use for error reporting
        """
        self.logger = logger
    
    def handle_exception(self, exception: Exception, message: Optional[str] = None) -> None:
        """
        Handle an exception with proper logging.
        
        Args:
            exception: The exception that occurred
            message: Optional contextual message
        """
        if message:
            self.logger.error(f"{message}: {str(exception)}")
        else:
            self.logger.error(str(exception))
            
        self.logger.debug(traceback.format_exc())
