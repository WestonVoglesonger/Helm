"""Intelligent Evaluation System for conversation quality assessment."""

from .core import (
    IntelligentEvaluationEngine,
    ConversationContext,
    Evaluation,
    UserFeedback,
    EvaluationDimension
)

__version__ = "1.0.0"

__all__ = [
    "IntelligentEvaluationEngine",
    "ConversationContext",
    "Evaluation",
    "UserFeedback",
    "EvaluationDimension",
]