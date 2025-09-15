"""
LLM Provider configurations for Tax Compliance RAG system.
"""

import os
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

# Import providers
try:
    from langchain_ollama import OllamaLLM
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import google.generativeai as genai
    from langchain.llms.base import LLM
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Create a placeholder class to avoid NameError
    class ChatOpenAI:
        pass

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    def get_llm(self) -> Any:
        """Return the LLM instance."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass

class GeminiLLM:
    """Simple Gemini LLM wrapper."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self._llm_type = "gemini"
    
    def invoke(self, prompt: str) -> str:
        """Generate response from prompt."""
        try:
            # Configure generation with timeout and limits
            generation_config = {
                'max_output_tokens': 1024,
                'temperature': 0.1,
            }
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if response.text:
                return response.text
            else:
                return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            # Return a more helpful error message
            if "timeout" in str(e).lower():
                return "The request timed out. Please try asking a simpler question or try again."
            elif "quota" in str(e).lower() or "limit" in str(e).lower():
                return "API quota exceeded. Please try again in a few minutes."
            else:
                return f"I encountered an error processing your question. Please try rephrasing it or ask a simpler question."

class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or "AIzaSyBl1OdwUvfZLokVfNQtdCFuNAyE6oJ2N40"
        self.model_name = model_name
    
    def get_llm(self) -> GeminiLLM:
        if not self.is_available():
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable.")
        return GeminiLLM(self.api_key, self.model_name)
    
    def is_available(self) -> bool:
        return GEMINI_AVAILABLE and self.api_key is not None

class OllamaProvider(BaseLLMProvider):
    """Ollama local provider."""
    
    def __init__(self, model_name: str = "mistral:7b"):
        self.model_name = model_name
    
    def get_llm(self):
        if not self.is_available():
            raise ValueError("Ollama not available. Please install and start Ollama.")
        
        return OllamaLLM(
            model=self.model_name,
            temperature=0.1,
            num_ctx=4096,
            num_predict=512,
        )
    
    def is_available(self) -> bool:
        if not OLLAMA_AVAILABLE:
            return False
        
        try:
            import requests
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            return response.status_code == 200
        except:
            return False

class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
    
    def get_llm(self):
        if not self.is_available():
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        return ChatOpenAI(
            model=self.model_name,
            temperature=0.1,
            api_key=self.api_key
        )
    
    def is_available(self) -> bool:
        return OPENAI_AVAILABLE and self.api_key is not None

class LLMProviderManager:
    """Manager for different LLM providers."""
    
    def __init__(self):
        self.providers = {
            "gemini": GeminiProvider(),
            "ollama": OllamaProvider(),
            "openai": OpenAIProvider()
        }
    
    def get_available_providers(self) -> Dict[str, bool]:
        """Get status of all providers."""
        return {
            name: provider.is_available() 
            for name, provider in self.providers.items()
        }
    
    def get_provider(self, provider_name: str, **kwargs) -> BaseLLMProvider:
        """Get a specific provider with custom configuration."""
        if provider_name == "gemini":
            return GeminiProvider(
                api_key=kwargs.get("api_key"),
                model_name=kwargs.get("model_name", "gemini-1.5-flash")
            )
        elif provider_name == "ollama":
            return OllamaProvider(
                model_name=kwargs.get("model_name", "mistral:7b")
            )
        elif provider_name == "openai":
            return OpenAIProvider(
                api_key=kwargs.get("api_key"),
                model_name=kwargs.get("model_name", "gpt-4o-mini")
            )
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    
    def get_best_available_llm(self, preferred_provider: Optional[str] = None):
        """Get the best available LLM, optionally with preference."""
        
        # If a preferred provider is specified and available, use it
        if preferred_provider and preferred_provider in self.providers:
            provider = self.providers[preferred_provider]
            if provider.is_available():
                logger.info(f"Using preferred provider: {preferred_provider}")
                return provider.get_llm()
        
        # Fallback priority: Gemini > Ollama > OpenAI
        priority_order = ["gemini", "ollama", "openai"]
        
        for provider_name in priority_order:
            provider = self.providers[provider_name]
            if provider.is_available():
                logger.info(f"Using available provider: {provider_name}")
                return provider.get_llm()
        
        raise ValueError("No LLM providers are available. Please configure at least one provider.")

