"""
Utility to check API availability and handle missing API keys gracefully.
"""

import os
from functools import wraps
from typing import Optional, Callable, Any


class APIKeyMissingError(Exception):
    """Exception raised when a required API key is missing."""
    pass


def get_api_key(key_name: str) -> Optional[str]:
    """
    Get an API key from environment variables.
    
    Args:
        key_name: The name of the environment variable containing the API key.
    
    Returns:
        The API key if found, None otherwise.
    """
    return os.environ.get(key_name)


def requires_api_key(key_name: str):
    """
    Decorator to check if a required API key is available.
    
    Args:
        key_name: The name of the environment variable containing the API key.
    
    Returns:
        The decorated function that checks for the API key before execution.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            api_key = get_api_key(key_name)
            if not api_key:
                raise APIKeyMissingError(
                    f"The {key_name} is required but was not found in the environment variables. "
                    "Please add it to your .env file or set it as an environment variable."
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def check_openai_availability() -> bool:
    """
    Check if OpenAI API is available.
    
    Returns:
        True if the API key is set, False otherwise.
    """
    return bool(get_api_key("OPENAI_API_KEY"))


def check_anthropic_availability() -> bool:
    """
    Check if Anthropic API is available.
    
    Returns:
        True if the API key is set, False otherwise.
    """
    return bool(get_api_key("ANTHROPIC_API_KEY"))


def check_serpapi_availability() -> bool:
    """
    Check if SerpAPI is available.
    
    Returns:
        True if the API key is set, False otherwise.
    """
    return bool(get_api_key("SERPAPI_API_KEY"))


def get_available_llm_provider() -> Optional[str]:
    """
    Get the first available LLM provider.
    
    Returns:
        The name of the available provider (openai or anthropic) or None if none are available.
    """
    if check_openai_availability():
        return "openai"
    if check_anthropic_availability():
        return "anthropic"
    return None
