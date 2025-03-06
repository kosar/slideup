"""
Logging and error handling utilities.
"""

import logging
import sys
import traceback
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union, List, Callable


def setup_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    console_output: bool = True
) -> logging.Logger:
    """
    Set up and configure a logger.
    
    Args:
        name: Name of the logger
        log_file: Path to the log file (optional)
        level: Logging level
        format_str: Log message format string
        console_output: Whether to output logs to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(format_str)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Add file handler if log_file is provided
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


class ErrorHandler:
    """Error handling utility class."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the error handler.
        
        Args:
            logger: Logger instance to use for error logging
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def handle_exception(self, exc: Exception, error_msg: str = "An error occurred", 
                        raise_error: bool = False) -> str:
        """
        Handle an exception with proper logging.
        
        Args:
            exc: The exception to handle
            error_msg: Custom error message
            raise_error: Whether to re-raise the exception after logging
            
        Returns:
            Error message including exception details
        """
        error_detail = f"{error_msg}: {str(exc)}"
        self.logger.error(error_detail)
        self.logger.error("Traceback: %s", traceback.format_exc())
        
        if raise_error:
            raise exc
            
        return error_detail
    
    def log_warning(self, message: str, *args, **kwargs) -> None:
        """Log a warning message."""
        self.logger.warning(message, *args, **kwargs)
    
    def log_error(self, message: str, *args, **kwargs) -> None:
        """Log an error message."""
        self.logger.error(message, *args, **kwargs)
    
    def log_critical(self, message: str, *args, **kwargs) -> None:
        """Log a critical message."""
        self.logger.critical(message, *args, **kwargs)
    
    @staticmethod
    def safe_exec(func: Callable, *args, 
                 default_return: Any = None, 
                 logger: Optional[logging.Logger] = None, 
                 error_msg: str = "Function execution failed",
                 **kwargs) -> Any:
        """
        Safely execute a function with exception handling.
        
        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            default_return: Default return value if execution fails
            logger: Logger to use for error logging
            error_msg: Error message to log on failure
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Function result or default_return on error
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if logger:
                logger.error(f"{error_msg}: {str(e)}")
                logger.error("Traceback: %s", traceback.format_exc())
            return default_return
