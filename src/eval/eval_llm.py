"""Stub for an evaluation LLM.

This module provides a placeholder interface to an evaluation model that can
score responses for quality.  For now, it returns a constant value.  You can
extend this module to call a separate OpenAI model (or another provider) to
compute a reward.
"""

from __future__ import annotations

from typing import Tuple


async def eval_llm(system_prompt: str, user_msg: str, model_response: str) -> float:
    """Return a reward from an evaluation LLM.

    The default implementation returns 0.5.  Replace with a real call as
    needed.
    """
    return 0.5