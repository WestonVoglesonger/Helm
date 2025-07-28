"""Prompt linter for enforcing token budgets and structure.

This module contains helper functions that validate compiled prompts before
sending them to the LLM.  The linter checks that the prompt does not exceed
configured token limits and that only allowed sections/snippets are included.

Token counting is approximate: it uses whitespace splitting, which is
sufficient for coarse limits.  For production, integrate with tiktoken.
"""

from __future__ import annotations

import logging
import re
from typing import Tuple

from ..config import get_settings


logger = logging.getLogger(__name__)


def count_tokens(text: str) -> int:
    """Approximate token count via whitespace splitting."""
    # A very rough estimate: split on whitespace
    return len(text.split())


def lint_prompt(prompt: str) -> Tuple[bool, str]:
    """Check that the prompt adheres to budget constraints.

    Returns a tuple `(ok, reason)` where `ok` is True if the prompt is valid
    and `reason` contains an error message when invalid.
    """
    settings = get_settings()
    total_tokens = count_tokens(prompt)
    budget = (
        settings.prompt_budget_system
        + settings.prompt_budget_user_ctx
        + settings.prompt_budget_snippets
    )
    if total_tokens > budget:
        return False, f"Prompt exceeds total budget ({total_tokens} > {budget})"
    # Example structural check: ensure snippet markers follow a pattern
    # e.g. `[[snippet:xyz]]` (optional demonstration).  This is a placeholder.
    pattern = re.compile(r"\[\[snippet:.+?\]\]")
    if not pattern.search(prompt):
        logger.debug("No snippet markers found; this may be acceptable.")
    return True, ""