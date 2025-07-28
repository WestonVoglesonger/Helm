"""Heuristic evaluation functions.

This module defines simple evaluation heuristics that assign a reward to the
model's response.  It also extracts metrics for logging (e.g., token count,
latency).  A more sophisticated evaluator can be plugged in later.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Dict, Tuple

from ..prompt.linter import count_tokens


logger = logging.getLogger(__name__)


def evaluate_response(system_prompt: str, user_msg: str, model_response: str) -> Tuple[float, Dict[str, float]]:
    """Compute a reward and metrics for a given response.

    Args:
        system_prompt: The compiled system prompt.
        user_msg: The raw user message.
        model_response: The LLM's output.

    Returns:
        A tuple `(reward, metrics)` where reward is a float and metrics is a
        dictionary containing evaluation signals (see schema).
    """
    start = perf_counter()
    # Trivial heuristic: reward is inverse of response length (shorter is better)
    response_tokens = count_tokens(model_response)
    reward = max(0.0, 1.0 - response_tokens / 1000.0)
    # Collect metrics
    latency_ms = (perf_counter() - start) * 1000.0
    metrics = {
        "len_tokens": count_tokens(system_prompt) + count_tokens(user_msg) + response_tokens,
        "pref_violation_count": 0.0,
        "latency_ms": latency_ms,
    }
    return reward, metrics