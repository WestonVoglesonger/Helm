from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class EvaluationDimension(str, Enum):
    """Evaluation dimensions for conversation quality."""
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    HELPFULNESS = "helpfulness"
    SAFETY = "safety"
    EFFICIENCY = "efficiency"


@dataclass
class ConversationContext:
    """Context for a conversation to be evaluated."""
    conversation_id: str
    messages: List[Dict[str, str]]
    user_query: str
    system_response: str
    metadata: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class EvaluationScore:
    """Score for a single evaluation dimension."""
    dimension: EvaluationDimension
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    reasoning: str
    sub_scores: Optional[Dict[str, float]] = None


@dataclass
class Evaluation:
    """Complete evaluation result."""
    conversation_id: str
    scores: Dict[EvaluationDimension, EvaluationScore]
    overall_score: float
    timestamp: datetime
    evaluator_version: str
    metadata: Dict[str, Any] = None


@dataclass
class UserFeedback:
    """User feedback on an evaluation."""
    evaluation_id: str
    conversation_id: str
    rating: float  # User's rating 0.0 to 1.0
    dimensions_feedback: Dict[EvaluationDimension, float]
    comments: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class EvaluationPattern:
    """Pattern identified in evaluation feedback."""
    pattern_type: str
    description: str
    occurrences: int
    confidence: float
    dimension: Optional[EvaluationDimension] = None
    adjustment_factor: float = 1.0