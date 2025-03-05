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
