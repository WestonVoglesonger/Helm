"""Embedding utility functions.

This module wraps the embedding model used for vectorisation.  It currently
delegates to the `OpenAIAdapter` but can be swapped out without affecting
other components.
"""

from __future__ import annotations

from typing import List

from ..llm.openai import OpenAIAdapter


async def embed_text(text: str) -> List[float]:
    """Return an embedding for the given text."""
    adapter = OpenAIAdapter()
    return await adapter.embed(text)