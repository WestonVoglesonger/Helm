from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import asyncio
from ..core.models import ConversationContext, EvaluationScore, EvaluationDimension


class BaseEvaluator(ABC):
    """Base class for all dimension evaluators."""
    
    def __init__(self, dimension: EvaluationDimension):
        self.dimension = dimension
        self.weights = self._initialize_weights()
        
    @abstractmethod
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize evaluation weights for this dimension."""
        pass
    
    @abstractmethod
    async def evaluate(self, context: ConversationContext) -> EvaluationScore:
        """Evaluate the conversation for this dimension."""
        pass
    
    @abstractmethod
    async def extract_features(self, context: ConversationContext) -> Dict[str, float]:
        """Extract relevant features from the conversation."""
        pass
    
    def normalize_score(self, raw_score: float, min_val: float = 0, max_val: float = 1) -> float:
        """Normalize score to 0-1 range."""
        if max_val == min_val:
            return 0.5
        return max(0, min(1, (raw_score - min_val) / (max_val - min_val)))
    
    def calculate_weighted_score(self, features: Dict[str, float]) -> float:
        """Calculate weighted score from features."""
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            return 0.5
            
        weighted_sum = sum(
            features.get(feature, 0) * weight 
            for feature, weight in self.weights.items()
        )
        return weighted_sum / total_weight
    
    def update_weights(self, adjustments: Dict[str, float]):
        """Update evaluation weights based on feedback."""
        for feature, adjustment in adjustments.items():
            if feature in self.weights:
                self.weights[feature] *= adjustment
                # Keep weights in reasonable bounds
                self.weights[feature] = max(0.1, min(10.0, self.weights[feature]))