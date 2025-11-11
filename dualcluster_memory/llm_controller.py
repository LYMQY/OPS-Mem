from typing import Dict, Optional, Literal, Any
import os
import json
from abc import ABC, abstractmethod

class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str, response_format: Optional[dict] = None, temperature: float = 0.7) -> str:
        """Get completion from LLM"""
        pass

class GeminiController(BaseLLMController):
    def __init__(self, model: str = "gemini-2.5", api_key: Optional[str] = None):
        try:
            from google import genai
            self.model = model
            if api_key is None:
                api_key = os.getenv('GOOGLE_API_KEY')
            if api_key is None:
                raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
            self.client = genai.Client(api_key=api_key)
        except ImportError:
            raise ImportError("Google Generative AI package not found. Install it with: pip install google-generative-ai")
    
    def get_completion(self, prompt: str, response_format: Optional[dict] = None, temperature: float = 0.7) -> str:
        kwargs = {}
        if response_format:
            kwargs["response_format"] = response_format
        response = self.client.models.generate_content(
            model=self.model, 
            contents=[prompt]
        )
        return response.choices[0].text

class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
    
    def get_completion(self, prompt: str, response_format: Optional[dict] = None, temperature: float = 0.7) -> str:
        kwargs = {}
        if response_format:
            kwargs["response_format"] = response_format
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000,
            **kwargs
        )
        return response.choices[0].message.content

class DeepseekController(BaseLLMController):
    def __init__(self, model: str = "deepseek-ai/DeepSeek-V3", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = os.getenv('DEEPSEEK_API_KEY')
            if api_key is None:
                raise ValueError("Deepseek API key not found. Set DEEPSEEK_API_KEY environment variable.")
            self.client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")

    def get_completion(self, prompt: str, response_format: Optional[dict] = None, temperature: float = 0.7) -> str:
        kwargs = {}
        if response_format:
            kwargs["response_format"] = response_format
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000,
            **kwargs
        )
        return response.choices[0].message.content

class LLMController:
    """LLM-based controller for memory metadata generation"""
    def __init__(self, 
                 backend: Literal["openai", "gemini", "deepseek"] = "openai",
                 model: str = "gpt-4", 
                 api_key: Optional[str] = None):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key)
        elif backend == "gemini":
            self.llm = GeminiController(model, api_key)
        elif backend == "deepseek":
            self.llm = DeepseekController(model, api_key)
        else:
            raise ValueError("Backend must be one of: 'openai', 'gemini', 'deepseek'")
            
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        return self.llm.get_completion(prompt, response_format, temperature)
