"""
Base agent implementation with graceful handling of missing API keys.
"""

from crewai import Agent
from typing import Optional, Dict, Any
import logging

from ..utils import (
    get_available_llm_provider,
    check_openai_availability,
    check_anthropic_availability,
    APIKeyMissingError
)

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base class for creating agents with graceful API key handling.
    """
    
    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        verbose: bool = False,
        allow_delegation: bool = False,
        provider: Optional[str] = None,
        **kwargs
    ):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.verbose = verbose
        self.allow_delegation = allow_delegation
        self.provider = provider
        self.extra_kwargs = kwargs
        self._agent = None
        
    def initialize(self) -> Optional[Agent]:
        """
        Initialize the agent with proper error handling for missing API keys.
        
        Returns:
            An initialized Agent object or None if initialization fails.
        """
        try:
            # If no provider specified, try to find an available one
            if not self.provider:
                self.provider = get_available_llm_provider()
                
                if not self.provider:
                    logger.warning(
                        "No LLM provider available. Please set either OPENAI_API_KEY "
                        "or ANTHROPIC_API_KEY in your environment variables."
                    )
                    print(f"⚠️ Agent '{self.role}' cannot be initialized: No LLM provider available.")
                    return None
            
            # Check if the specified provider is available
            if self.provider == "openai" and not check_openai_availability():
                logger.warning("OpenAI API key missing but required for this agent.")
                print(f"⚠️ Agent '{self.role}' cannot be initialized: OpenAI API key missing.")
                return None
                
            if self.provider == "anthropic" and not check_anthropic_availability():
                logger.warning("Anthropic API key missing but required for this agent.")
                print(f"⚠️ Agent '{self.role}' cannot be initialized: Anthropic API key missing.")
                return None
            
            # Initialize the agent
            self._agent = Agent(
                role=self.role,
                goal=self.goal,
                backstory=self.backstory,
                verbose=self.verbose,
                allow_delegation=self.allow_delegation,
                provider=self.provider,
                **self.extra_kwargs
            )
            
            return self._agent
            
        except APIKeyMissingError as e:
            logger.error(f"API key error: {str(e)}")
            print(f"⚠️ Agent '{self.role}' failed to initialize: {str(e)}")
            return None
            
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            print(f"⚠️ Agent '{self.role}' failed to initialize due to an unexpected error: {str(e)}")
            return None
    
    @property
    def agent(self) -> Optional[Agent]:
        """
        Get the initialized agent.
        
        Returns:
            The CrewAI Agent object or None if not initialized.
        """
        if self._agent is None:
            self._agent = self.initialize()
        return self._agent
