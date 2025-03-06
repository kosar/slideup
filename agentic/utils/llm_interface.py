"""
Simple interface for accessing LLM services.
"""

import os
import json
import requests
from typing import Dict, Any, List, Optional, Union, Callable
import time
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    CUSTOM = "custom"


class LLMInterface:
    """Interface for interacting with LLM services."""
    
    def __init__(
        self, 
        provider: Union[LLMProvider, str] = LLMProvider.OPENAI,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize the LLM interface.
        
        Args:
            provider: LLM provider to use
            api_key: API key for the provider (defaults to environment variables)
            model: Model name to use
            base_url: Custom base URL for API calls
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.provider = provider if isinstance(provider, LLMProvider) else LLMProvider(provider)
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set API key from arguments or environment variables
        if api_key:
            self.api_key = api_key
        else:
            if self.provider == LLMProvider.OPENAI:
                self.api_key = os.environ.get("OPENAI_API_KEY", "")
            elif self.provider == LLMProvider.ANTHROPIC:
                self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            elif self.provider == LLMProvider.AZURE:
                self.api_key = os.environ.get("AZURE_OPENAI_KEY", "")
            else:
                self.api_key = os.environ.get("LLM_API_KEY", "")
        
        # Set base URL
        self.base_url = base_url
        if not self.base_url:
            if self.provider == LLMProvider.OPENAI:
                self.base_url = "https://api.openai.com/v1"
            elif self.provider == LLMProvider.ANTHROPIC:
                self.base_url = "https://api.anthropic.com/v1"
            elif self.provider == LLMProvider.AZURE:
                # Azure requires endpoint specification
                self.base_url = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    
    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare request headers based on the provider."""
        headers = {"Content-Type": "application/json"}
        
        if self.provider == LLMProvider.OPENAI:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.provider == LLMProvider.ANTHROPIC:
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        elif self.provider == LLMProvider.AZURE:
            headers["api-key"] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        return headers
    
    def _prepare_payload(
        self, 
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare request payload based on the provider."""
        if self.provider == LLMProvider.OPENAI:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
            
        elif self.provider == LLMProvider.ANTHROPIC:
            # Convert messages format to Anthropic format
            system_prompt = ""
            conversation = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt += msg["content"] + "\n"
                else:
                    conversation.append({"role": msg["role"], "content": msg["content"]})
            
            payload = {
                "model": self.model,
                "system": system_prompt.strip(),
                "messages": conversation,
                "temperature": temperature
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
                
        elif self.provider == LLMProvider.AZURE:
            payload = {
                "messages": messages,
                "temperature": temperature
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
                
        else:
            # Generic format for custom providers
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
        
        # Add any additional parameters
        for key, value in kwargs.items():
            payload[key] = value
            
        return payload
    
    def _extract_response(self, response_data: Dict[str, Any]) -> str:
        """Extract the model's response text from the API response."""
        if self.provider == LLMProvider.OPENAI:
            return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
        elif self.provider == LLMProvider.ANTHROPIC:
            return response_data.get("content", [{}])[0].get("text", "")
            
        elif self.provider == LLMProvider.AZURE:
            return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
        else:
            # Generic extraction (best effort)
            if "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0].get("message", {}).get("content", "")
            return str(response_data)
    
    def chat(
        self, 
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> str:
        """
        Send a chat request to the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature parameter for response randomness
            max_tokens: Maximum tokens in the response
            callback: Optional callback function for streaming responses
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Model's response as a string
        """
        headers = self._prepare_headers()
        payload = self._prepare_payload(messages, temperature, max_tokens, **kwargs)
        
        # Determine the endpoint based on provider
        if self.provider == LLMProvider.OPENAI:
            endpoint = f"{self.base_url}/chat/completions"
        elif self.provider == LLMProvider.ANTHROPIC:
            endpoint = f"{self.base_url}/messages"
        elif self.provider == LLMProvider.AZURE:
            # Azure requires deployment name in the URL
            deployment_name = self.model
            endpoint = f"{self.base_url}/openai/deployments/{deployment_name}/chat/completions?api-version=2023-05-15"
        else:
            # Default endpoint structure
            endpoint = f"{self.base_url}/chat/completions"
        
        # Make the request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return self._extract_response(response.json())
                
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt == self.max_retries - 1:  # Last attempt
                    raise Exception(f"LLM API request failed after {self.max_retries} attempts: {str(e)}")
                time.sleep(self.retry_delay)
    
    def completion(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Simplified method to get a completion from the LLM.
        
        Args:
            prompt: The user's prompt
            system_prompt: Optional system instructions
            temperature: Temperature parameter
            max_tokens: Maximum tokens in the response
            **kwargs: Additional parameters
            
        Returns:
            Model's response as a string
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages, temperature, max_tokens, **kwargs)