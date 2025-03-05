"""
Configuration settings for agents.
"""

from typing import Dict, Any, Optional
import os
import json
from pathlib import Path


class AgentConfig:
    """
    Class to manage agent configurations.
    """
    
    CONFIG_DIR = Path("/Users/kosar/src/slideup/configs")
    DEFAULT_CONFIG_FILE = CONFIG_DIR / "default_agent_config.json"
    
    @classmethod
    def create_default_config(cls) -> None:
        """Create default configuration file if it doesn't exist."""
        if not cls.CONFIG_DIR.exists():
            cls.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        if not cls.DEFAULT_CONFIG_FILE.exists():
            default_config = {
                "default_provider": "openai",
                "openai": {
                    "model": "gpt-4",
                    "temperature": 0.7
                },
                "anthropic": {
                    "model": "claude-2",
                    "temperature": 0.7
                },
                "tools": {
                    "search_enabled": True,
                    "memory_enabled": True
                },
                "logging": {
                    "level": "INFO",
                    "file": "/Users/kosar/src/slideup/logs/agent.log"
                }
            }
            
            with open(cls.DEFAULT_CONFIG_FILE, 'w') as f:
                json.dump(default_config, f, indent=2)
    
    @classmethod
    def load_config(cls, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load agent configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file. If None, uses the default.
        
        Returns:
            The configuration as a dictionary.
        """
        if config_path is None:
            config_path = cls.DEFAULT_CONFIG_FILE
        
        # Create default config if it doesn't exist
        if not config_path.exists():
            cls.create_default_config()
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @classmethod
    def get_provider_config(cls, provider: str) -> Dict[str, Any]:
        """
        Get configuration for a specific provider.
        
        Args:
            provider: The provider name (e.g., 'openai', 'anthropic').
        
        Returns:
            Provider-specific configuration.
        """
        config = cls.load_config()
        return config.get(provider, {})
