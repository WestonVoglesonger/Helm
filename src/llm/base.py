"""Abstract base class for language model adapters.

The `BaseLLMAdapter` defines the interface expected by the orchestrator.  It
exposes two methods: `call`, which sends a prompt to the underlying model and
returns a response, and `embed`, which returns an embedding vector for a piece
of text.  Subclasses should override these methods with concrete
implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseLLMAdapter(ABC):
    """Interface for language model adapters."""

    @abstractmethod
    async def call(
        self, system_prompt: str, user_msg: str, tools: Optional[list] = None
    ) -> str:
        """Send a prompt to the underlying model and return its response.

        Args:
            system_prompt: The compiled system prompt assembled by the
                prompt compiler.
            user_msg: The raw user message.
            tools: Optional list of tools/functions available to the model.

        Returns:
            The model's response as a string.
        """
        raise NotImplementedError

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Return an embedding vector for the given text."""
        raise NotImplementedError