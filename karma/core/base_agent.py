"""
Base agent class for the KARMA framework.

This module provides the abstract base class that all KARMA agents inherit from,
ensuring consistent interfaces and shared functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
import time
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all KARMA agents.

    This class provides common functionality including LLM client management,
    token tracking, error handling, and performance monitoring.

    Attributes:
        client: OpenAI client instance
        model_name: LLM model identifier
        system_prompt: System prompt for the agent
        metrics: Performance tracking metrics
    """

    def __init__(self, client: OpenAI, model_name: str, system_prompt: str = ""):
        """
        Initialize the base agent.

        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
            system_prompt: System prompt for the agent
        """
        self.client = client
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.metrics = {
            'total_calls': 0,
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_time': 0.0,
            'error_count': 0
        }

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Main processing method for the agent.

        This method must be implemented by all concrete agent classes.
        """
        pass

    def _make_llm_call(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, int, int, float]:
        """
        Make a call to the LLM with error handling and metrics tracking.

        Args:
            prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Override system prompt for this call

        Returns:
            Tuple of (response_content, prompt_tokens, completion_tokens, processing_time)

        Raises:
            Exception: If the LLM call fails
        """
        start_time = time.time()

        try:
            messages = [
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": prompt}
            ]

            call_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature
            }

            if max_tokens:
                call_kwargs["max_tokens"] = max_tokens

            response = self.client.chat.completions.create(**call_kwargs)

            content = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            processing_time = time.time() - start_time

            # Update metrics
            self.metrics['total_calls'] += 1
            self.metrics['total_prompt_tokens'] += prompt_tokens
            self.metrics['total_completion_tokens'] += completion_tokens
            self.metrics['total_time'] += processing_time

            return content, prompt_tokens, completion_tokens, processing_time

        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics['error_count'] += 1
            self.metrics['total_time'] += processing_time

            logger.error(f"{self.__class__.__name__} LLM call failed: {str(e)}")
            raise

    def _extract_float_from_text(self, text: str, default: float = 0.5) -> float:
        """
        Extract a float value from text response.

        Args:
            text: Text containing float value
            default: Default value if no float found

        Returns:
            Extracted float value, clamped to [0.0, 1.0]
        """
        import re

        # Look for float numbers in the text
        float_matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if float_matches:
            try:
                score = float(float_matches[0])
                # Clamp to [0.0, 1.0]
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        return default

    def _parse_json_response(self, response: str) -> List[Dict]:
        """
        Parse JSON response from LLM with error handling.

        Args:
            response: LLM response containing JSON

        Returns:
            Parsed JSON data as list of dictionaries
        """
        import json

        try:
            # Try to find JSON array in response
            if "[" in response and "]" in response:
                json_str = response[response.find("["):response.rfind("]")+1]
                return json.loads(json_str)

            # Try to parse the entire response as JSON
            return json.loads(response)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return []

    def get_metrics(self) -> Dict:
        """
        Get performance metrics for this agent.

        Returns:
            Dictionary of performance metrics
        """
        return self.metrics.copy()

    def reset_metrics(self):
        """Reset all performance metrics."""
        self.metrics = {
            'total_calls': 0,
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_time': 0.0,
            'error_count': 0
        }

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(model={self.model_name})"

    def __repr__(self) -> str:
        """Detailed representation of the agent."""
        return f"{self.__class__.__name__}(model={self.model_name}, calls={self.metrics['total_calls']})"