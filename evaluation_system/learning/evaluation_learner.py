import asyncio
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta

from ..core.models import (
    Evaluation,
    UserFeedback,
    EvaluationDimension,
    EvaluationPattern
)
from .pattern_recognizer import PatternRecognizer


class EvaluationLearner:
    """Learning module that improves evaluation criteria based on user feedback."""
    
    def __init__(self, history_size: int = 1000):
        self.feedback_history = deque(maxlen=history_size)
        self.pattern_recognizer = PatternRecognizer()
        
        # Dimension adjustment history
        self.dimension_adjustments = defaultdict(list)
        
        # Weight adjustments for dimensions
        self.weight_adjustments = {
            dimension: 1.0 for dimension in EvaluationDimension
        }
        
        # Feature weight adjustments per dimension
        self.feature_adjustments = defaultdict(lambda: defaultdict(float))
        
        # Learning rate
        self.learning_rate = 0.1
        
        # Pattern cache
        self.pattern_cache = {}
        self.cache_timestamp = datetime.now()
        self.cache_ttl = timedelta(hours=1)
    
    async def learn_from_feedback(self, evaluation: Evaluation, feedback: UserFeedback):
        """Learn from a single feedback instance."""
        # Store feedback
        self.feedback_history.append((evaluation, feedback))
        
        # Calculate discrepancy between evaluation and user feedback
        discrepancy = self._calculate_discrepancy(evaluation, feedback)
        
        # Update dimension weights based on discrepancy
        await self._update_dimension_weights(discrepancy)
        
        # Identify patterns if enough history
        if len(self.feedback_history) >= 10:
            patterns = await self.pattern_recognizer.analyze(
                list(self.feedback_history)
            )
            
            # Apply pattern-based adjustments
            await self._apply_pattern_adjustments(patterns)
    
    async def batch_learn(self, feedback_list: List[UserFeedback]):
        """Learn from multiple feedback instances."""
        for feedback in feedback_list:
            # Find corresponding evaluation
            evaluation = self._find_evaluation_for_feedback(feedback)
            if evaluation:
                await self.learn_from_feedback(evaluation, feedback)
    
    def _calculate_discrepancy(self, evaluation: Evaluation, 
                              feedback: UserFeedback) -> Dict[EvaluationDimension, float]:
        """Calculate discrepancy between evaluation scores and user feedback."""
        discrepancies = {}
        
        # Overall discrepancy
        overall_discrepancy = feedback.rating - evaluation.overall_score
        
        # Dimension-specific discrepancies
        for dimension, score in evaluation.scores.items():
            if dimension in feedback.dimensions_feedback:
                user_score = feedback.dimensions_feedback[dimension]
                discrepancies[dimension] = user_score - score.score
            else:
                # Infer from overall discrepancy
                discrepancies[dimension] = overall_discrepancy * 0.5
        
        return discrepancies
    
    async def _update_dimension_weights(self, discrepancy: Dict[EvaluationDimension, float]):
        """Update dimension weights based on discrepancy."""
        for dimension, disc_value in discrepancy.items():
            # If user rates higher than system, increase weight
            # If user rates lower than system, decrease weight
            adjustment = 1.0 + (disc_value * self.learning_rate)
            
            # Apply adjustment
            self.weight_adjustments[dimension] *= adjustment
            
            # Keep in bounds
            self.weight_adjustments[dimension] = max(0.5, min(2.0, 
                                                            self.weight_adjustments[dimension]))
            
            # Track adjustment
            self.dimension_adjustments[dimension].append({
                "timestamp": datetime.now(),
                "adjustment": adjustment,
                "discrepancy": disc_value
            })
    
    async def _apply_pattern_adjustments(self, patterns: List[EvaluationPattern]):
        """Apply adjustments based on identified patterns."""
        for pattern in patterns:
            if pattern.confidence > 0.7:  # Only apply high-confidence patterns
                if pattern.dimension:
                    # Dimension-specific pattern
                    self.weight_adjustments[pattern.dimension] *= pattern.adjustment_factor
                else:
                    # General pattern - apply to all dimensions
                    for dimension in EvaluationDimension:
                        self.weight_adjustments[dimension] *= (pattern.adjustment_factor * 0.5)
    
    async def get_weight_adjustments(self) -> Dict[EvaluationDimension, float]:
        """Get current weight adjustments for dimensions."""
        # Check if cache is valid
        if (datetime.now() - self.cache_timestamp) < self.cache_ttl:
            if "weight_adjustments" in self.pattern_cache:
                return self.pattern_cache["weight_adjustments"]
        
        # Recalculate if needed
        adjustments = self.weight_adjustments.copy()
        
        # Apply smoothing to prevent drastic changes
        for dimension in adjustments:
            adjustments[dimension] = self._smooth_adjustment(adjustments[dimension])
        
        # Cache results
        self.pattern_cache["weight_adjustments"] = adjustments
        self.cache_timestamp = datetime.now()
        
        return adjustments
    
    async def get_dimension_patterns(self) -> Dict[EvaluationDimension, Dict[str, float]]:
        """Get feature weight adjustments for each dimension based on patterns."""
        dimension_patterns = defaultdict(dict)
        
        # Analyze recent feedback for each dimension
        recent_feedback = list(self.feedback_history)[-100:]  # Last 100 feedback items
        
        for dimension in EvaluationDimension:
            # Calculate feature importance based on feedback correlation
            feature_importance = await self._calculate_feature_importance(
                dimension, recent_feedback
            )
            
            dimension_patterns[dimension] = feature_importance
        
        return dimension_patterns
    
    async def _calculate_feature_importance(self, dimension: EvaluationDimension,
                                          feedback_data: List[Tuple[Evaluation, UserFeedback]]) -> Dict[str, float]:
        """Calculate feature importance for a dimension based on feedback."""
        feature_scores = defaultdict(list)
        user_ratings = []
        
        for evaluation, feedback in feedback_data:
            if dimension in evaluation.scores and dimension in feedback.dimensions_feedback:
                eval_score = evaluation.scores[dimension]
                user_score = feedback.dimensions_feedback[dimension]
                
                # Collect sub-scores and user ratings
                if eval_score.sub_scores:
                    for feature, score in eval_score.sub_scores.items():
                        feature_scores[feature].append(score)
                    user_ratings.append(user_score)
        
        # Calculate correlation between features and user ratings
        feature_importance = {}
        
        if user_ratings:
            for feature, scores in feature_scores.items():
                if len(scores) == len(user_ratings):
                    # Calculate correlation
                    correlation = np.corrcoef(scores, user_ratings)[0, 1]
                    
                    # Convert correlation to importance weight
                    # Higher correlation = more important
                    importance = 1.0 + (correlation * 0.5)
                    feature_importance[feature] = max(0.5, min(2.0, importance))
        
        return feature_importance
    
    def _smooth_adjustment(self, value: float, alpha: float = 0.7) -> float:
        """Apply exponential smoothing to adjustment value."""
        # Smooth towards 1.0 (no adjustment)
        return alpha * value + (1 - alpha) * 1.0
    
    def _find_evaluation_for_feedback(self, feedback: UserFeedback) -> Optional[Evaluation]:
        """Find evaluation corresponding to feedback."""
        for evaluation, _ in reversed(self.feedback_history):
            if evaluation.conversation_id == feedback.conversation_id:
                return evaluation
        return None
    
    def get_learning_statistics(self) -> Dict[str, any]:
        """Get statistics about the learning process."""
        stats = {
            "total_feedback": len(self.feedback_history),
            "dimension_adjustments": {},
            "recent_trends": {},
            "confidence_metrics": {}
        }
        
        # Calculate adjustment statistics per dimension
        for dimension in EvaluationDimension:
            if dimension in self.dimension_adjustments:
                adjustments = self.dimension_adjustments[dimension]
                if adjustments:
                    recent_adjustments = adjustments[-10:]  # Last 10
                    stats["dimension_adjustments"][dimension.value] = {
                        "current_weight": self.weight_adjustments[dimension],
                        "total_adjustments": len(adjustments),
                        "average_adjustment": np.mean([a["adjustment"] for a in adjustments]),
                        "trend": np.mean([a["adjustment"] for a in recent_adjustments]) - 1.0
                    }
        
        # Calculate recent trends
        if len(self.feedback_history) >= 20:
            recent_feedback = list(self.feedback_history)[-20:]
            
            for dimension in EvaluationDimension:
                discrepancies = []
                for eval, feedback in recent_feedback:
                    if dimension in eval.scores and dimension in feedback.dimensions_feedback:
                        disc = feedback.dimensions_feedback[dimension] - eval.scores[dimension].score
                        discrepancies.append(disc)
                
                if discrepancies:
                    stats["recent_trends"][dimension.value] = {
                        "average_discrepancy": np.mean(discrepancies),
                        "improving": np.mean(discrepancies[-5:]) < np.mean(discrepancies[:5])
                    }
        
        # Calculate confidence in current weights
        for dimension in EvaluationDimension:
            if dimension in self.dimension_adjustments:
                adjustments = [a["adjustment"] for a in self.dimension_adjustments[dimension][-20:]]
                if adjustments:
                    # Low variance in adjustments = high confidence
                    variance = np.var(adjustments)
                    confidence = 1.0 - min(1.0, variance * 2)
                    stats["confidence_metrics"][dimension.value] = confidence
        
        return stats
    
    async def reset_learning(self, dimension: Optional[EvaluationDimension] = None):
        """Reset learning for a specific dimension or all dimensions."""
        if dimension:
            self.weight_adjustments[dimension] = 1.0
            self.dimension_adjustments[dimension].clear()
            if dimension in self.feature_adjustments:
                self.feature_adjustments[dimension].clear()
        else:
            # Reset all
            for dim in EvaluationDimension:
                self.weight_adjustments[dim] = 1.0
            self.dimension_adjustments.clear()
            self.feature_adjustments.clear()
        
        # Clear cache
        self.pattern_cache.clear()
        self.cache_timestamp = datetime.now()