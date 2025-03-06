import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Model configurations
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4-turbo")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))

# Application settings
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

def validate_config():
    """Validates that all required configuration is present."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    
    # Add other validation as needed
    return True

"""
Configuration manager for SlideUp.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy

from .utils.file_handler import FileHandler


class Config:
    """Configuration manager for SlideUp."""
    
    DEFAULT_CONFIG = {
        "general": {
            "output_dir": "./output",
            "template_path": "",
            "image_dir": "./images",
            "default_theme": "default"
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "presentation": {
            "default_width": 10,
            "default_height": 7.5,
            "default_font": "Arial",
            "default_font_size": 18
        },
        "logging": {
            "level": "INFO",
            "file": "./logs/slideup.log",
            "console": True
        },
        "themes": {
            "default": {
                "title_font": "Arial",
                "title_size": 32,
                "subtitle_size": 24,
                "body_font": "Arial",
                "body_size": 18,
                "colors": {
                    "background": "#FFFFFF",
                    "title": "#000000",
                    "text": "#333333",
                    "accent": "#4472C4"
                }
            }
        }
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to a JSON or YAML configuration file
        """
        self.file_handler = FileHandler()
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        
        # Load configuration from a file if provided
        if config_path:
            self.load_config(config_path)
        
        # Check for configuration in standard locations if not provided
        elif not config_path:
            self._load_from_default_locations()
    
    def _load_from_default_locations(self) -> None:
        """Load configuration from standard locations."""
        # Check for config in the current directory
        if os.path.exists("slideup.json"):
            self.load_config("slideup.json")
            return
        
        if os.path.exists("slideup.yaml") or os.path.exists("slideup.yml"):
            config_path = "slideup.yaml" if os.path.exists("slideup.yaml") else "slideup.yml"
            self.load_config(config_path)
            return
        
        # Check for config in user's home directory
        home_dir = os.path.expanduser("~")
        
        if os.path.exists(os.path.join(home_dir, ".slideup", "config.json")):
            self.load_config(os.path.join(home_dir, ".slideup", "config.json"))
            return
            
        if os.path.exists(os.path.join(home_dir, ".slideup", "config.yaml")):
            self.load_config(os.path.join(home_dir, ".slideup", "config.yaml"))
            return
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to a JSON or YAML configuration file
        """
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                print(f"Config file not found: {config_path}")
                return
                
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                new_config = self.file_handler.read_yaml(config_path)
            elif config_path.suffix.lower() == '.json':
                new_config = self.file_handler.read_json(config_path)
            else:
                print(f"Unsupported config file format: {config_path}")
                return
                
            # Deep merge the loaded config with defaults
            self.config = self._deep_merge(self.config, new_config)
            print(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            print(f"Error loading config from {config_path}: {str(e)}")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Dictionary with values to override
            
        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
                
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key path.
        
        Args:
            key: Dot-separated key path (e.g., 'general.output_dir')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        parts = key.split('.')
        current = self.config
        
        for part in parts:
            if part not in current:
                return default
            current = current[part]
            
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Dot-separated key path (e.g., 'general.output_dir')
            value: Value to set
        """
        parts = key.split('.')
        current = self.config
        
        # Navigate to the deepest dict
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Set the value
        current[parts[-1]] = value
    
    def save(self, config_path: Union[str, Path]) -> None:
        """
        Save the current configuration to a file.
        
        Args:
            config_path: Path where to save the configuration
        """
        config_path = Path(config_path)
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            self.file_handler.write_yaml(config_path, self.config)
        else:
            # Default to JSON
            self.file_handler.write_json(config_path, self.config)
