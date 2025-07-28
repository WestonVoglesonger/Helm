"""OpenAI adapter implementation.

This module implements the `BaseLLMAdapter` interface using OpenAI's API.  It
supports both generation and embedding endpoints.  The adapter is designed to
be easily swapped out by extending `BaseLLMAdapter` with another backend.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from openai import AsyncOpenAI

from .base import BaseLLMAdapter
from ..config import get_settings


logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for the OpenAI API."""

    def __init__(self) -> None:
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.embedding_model = settings.embedding_model

    async def call(
        self, system_prompt: str, user_msg: str, tools: Optional[list] = None
    ) -> str:
        """Call the OpenAI chat completion API.

        This method is asynchronous; it wraps the OpenAI async API under the
        hood.  If no API key is configured, it raises a RuntimeError.
        """
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please configure your API key."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )
        except Exception as exc:
            logger.exception("OpenAI chat completion failed: %s", exc)
            raise
        # Extract the text from the first choice
        return response.choices[0].message.content or ""

    async def embed(self, text: str) -> List[float]:
        """Call the OpenAI embeddings endpoint."""
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please configure your API key."
            )
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model, input=[text]
            )
        except Exception as exc:
            logger.exception("OpenAI embedding failed: %s", exc)
            raise
        # `data` is a list of dicts, take first
        return response.data[0].embedding