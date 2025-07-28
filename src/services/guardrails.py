"""Guardrails for safety and privacy.

This module applies simple checks both before sending prompts (to detect
injection or PII) and after receiving outputs (to detect unsafe content).
These checks can be expanded or replaced with LLM‑based validators.
"""

from __future__ import annotations

import logging
import re
from typing import Tuple


logger = logging.getLogger(__name__)

# Simple regex patterns for demonstration
PROMPT_INJECTION_PATTERN = re.compile(r"(?i)ignore all previous instructions")
PII_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")  # naive SSN pattern
UNSAFE_OUTPUT_PATTERN = re.compile(r"(?i)suicide|harm|weapon")


def pre_send_guard(prompt: str) -> Tuple[bool, str]:
    """Check prompt for injection or PII.

    Returns `(ok, reason)`.  If `ok` is False, the prompt should be rejected.
    """
    if PROMPT_INJECTION_PATTERN.search(prompt):
        return False, "Prompt appears to contain injection phrases."
    if PII_PATTERN.search(prompt):
        return False, "Prompt appears to contain PII (e.g. SSN)."
    return True, ""


def post_send_guard(response: str) -> Tuple[bool, str]:
    """Check model output for unsafe content."""
    if UNSAFE_OUTPUT_PATTERN.search(response):
        return False, "Response contains unsafe content."
    return True, ""