import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid
import numpy as np

from ..evaluators import (
    RelevanceEvaluator,
    CoherenceEvaluator,
    HelpfulnessEvaluator,
    SafetyEvaluator,
    EfficiencyEvaluator,
    LLMEvaluator
)
from ..learning.evaluation_learner import EvaluationLearner
from .models import (
    ConversationContext,
    Evaluation,
    EvaluationScore,
    EvaluationDimension,
    UserFeedback
)


class IntelligentEvaluationEngine:
    """Comprehensive evaluation engine for conversation quality assessment."""
    
    def __init__(self, use_llm: bool = True, api_key: Optional[str] = None):
        self.use_llm = use_llm
        self.llm_evaluator = LLMEvaluator(api_key=api_key) if use_llm else None
        
        # Initialize dimension evaluators
        self.dimensions = {
            EvaluationDimension.RELEVANCE: RelevanceEvaluator(),
            EvaluationDimension.COHERENCE: CoherenceEvaluator(),
            EvaluationDimension.HELPFULNESS: HelpfulnessEvaluator(
                use_llm=use_llm, 
                llm_evaluator=self.llm_evaluator
            ),
            EvaluationDimension.SAFETY: SafetyEvaluator(
                use_llm=use_llm,
                llm_evaluator=self.llm_evaluator
            ),
            EvaluationDimension.EFFICIENCY: EfficiencyEvaluator(
                use_llm=use_llm,
                llm_evaluator=self.llm_evaluator
            ),
        }
        
        # Initialize learning module
        self.learning_module = EvaluationLearner()
        
        # Version tracking
        self.version = "1.0.0"
        
        # Dimension weights for overall score
        self.dimension_weights = {
            EvaluationDimension.RELEVANCE: 2.5,
            EvaluationDimension.COHERENCE: 1.5,
            EvaluationDimension.HELPFULNESS: 2.0,
            EvaluationDimension.SAFETY: 3.0,  # Highest weight for safety
            EvaluationDimension.EFFICIENCY: 1.0,
        }
    
    async def evaluate_conversation(self, context: ConversationContext) -> Evaluation:
        """Evaluate conversation across all dimensions."""
        scores = {}
        
        # Evaluate each dimension in parallel
        tasks = []
        for dimension, evaluator in self.dimensions.items():
            task = evaluator.evaluate(context)
            tasks.append((dimension, task))
        
        # Gather results
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Process results
        for (dimension, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                # Create error score
                scores[dimension] = EvaluationScore(
                    dimension=dimension,
                    score=0.5,
                    confidence=0.0,
                    reasoning=f"Evaluation failed: {str(result)}",
                    sub_scores={}
                )
            else:
                scores[dimension] = result
        
        # Calculate overall score
        overall_score = self.combine_scores(scores)
        
        # Create evaluation object
        evaluation = Evaluation(
            conversation_id=context.conversation_id,
            scores=scores,
            overall_score=overall_score,
            timestamp=datetime.now(),
            evaluator_version=self.version,
            metadata={
                "use_llm": self.use_llm,
                "dimension_weights": self.dimension_weights.copy()
            }
        )
        
        return evaluation
    
    def combine_scores(self, scores: Dict[EvaluationDimension, EvaluationScore]) -> float:
        """Combine individual dimension scores into overall score."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dimension, score in scores.items():
            weight = self.dimension_weights.get(dimension, 1.0)
            # Weight by confidence as well
            effective_weight = weight * score.confidence
            weighted_sum += score.score * effective_weight
            total_weight += effective_weight
        
        if total_weight == 0:
            return 0.5  # Default neutral score
        
        return weighted_sum / total_weight
    
    async def evaluate_with_feedback(self, context: ConversationContext, 
                                   previous_feedback: List[UserFeedback]) -> Evaluation:
        """Evaluate conversation considering previous feedback."""
        # Learn from feedback first
        if previous_feedback:
            await self.learning_module.batch_learn(previous_feedback)
            
            # Update dimension weights based on learning
            weight_adjustments = await self.learning_module.get_weight_adjustments()
            self._apply_weight_adjustments(weight_adjustments)
        
        # Perform evaluation
        evaluation = await self.evaluate_conversation(context)
        
        return evaluation
    
    async def learn_from_feedback(self, evaluation: Evaluation, feedback: UserFeedback):
        """Learn from user feedback to improve evaluation criteria."""
        await self.learning_module.learn_from_feedback(evaluation, feedback)
        
        # Get updated weights
        weight_adjustments = await self.learning_module.get_weight_adjustments()
        self._apply_weight_adjustments(weight_adjustments)
        
        # Update individual evaluator weights if patterns detected
        patterns = await self.learning_module.get_dimension_patterns()
        for dimension, pattern_adjustments in patterns.items():
            if dimension in self.dimensions:
                self.dimensions[dimension].update_weights(pattern_adjustments)
    
    def _apply_weight_adjustments(self, adjustments: Dict[EvaluationDimension, float]):
        """Apply weight adjustments to dimension weights."""
        for dimension, adjustment in adjustments.items():
            if dimension in self.dimension_weights:
                self.dimension_weights[dimension] *= adjustment
                # Keep weights in reasonable bounds
                self.dimension_weights[dimension] = max(0.5, min(5.0, self.dimension_weights[dimension]))
    
    async def compare_responses(self, response1: str, response2: str, 
                               context: str) -> Dict[str, any]:
        """Compare two responses using all evaluation dimensions."""
        # Create contexts for both responses
        context1 = ConversationContext(
            conversation_id=str(uuid.uuid4()),
            messages=[],
            user_query=context,
            system_response=response1,
            metadata={}
        )
        
        context2 = ConversationContext(
            conversation_id=str(uuid.uuid4()),
            messages=[],
            user_query=context,
            system_response=response2,
            metadata={}
        )
        
        # Evaluate both
        eval1, eval2 = await asyncio.gather(
            self.evaluate_conversation(context1),
            self.evaluate_conversation(context2)
        )
        
        # Compare scores
        comparison = {
            "response1_evaluation": eval1,
            "response2_evaluation": eval2,
            "better_response": 1 if eval1.overall_score > eval2.overall_score else 2,
            "score_difference": abs(eval1.overall_score - eval2.overall_score),
            "dimension_comparison": {}
        }
        
        # Compare by dimension
        for dimension in EvaluationDimension:
            if dimension in eval1.scores and dimension in eval2.scores:
                comparison["dimension_comparison"][dimension] = {
                    "response1": eval1.scores[dimension].score,
                    "response2": eval2.scores[dimension].score,
                    "difference": eval1.scores[dimension].score - eval2.scores[dimension].score
                }
        
        # If LLM is available, get detailed comparison
        if self.use_llm and self.llm_evaluator:
            llm_comparison = await self.llm_evaluator.compare_responses(
                response1, response2, context
            )
            comparison["llm_comparison"] = llm_comparison
        
        return comparison
    
    async def generate_improvement_suggestions(self, evaluation: Evaluation, 
                                             context: ConversationContext) -> List[str]:
        """Generate suggestions for improving the response."""
        suggestions = []
        
        # Find weakest dimensions
        weak_dimensions = []
        for dimension, score in evaluation.scores.items():
            if score.score < 0.6:  # Threshold for weakness
                weak_dimensions.append((dimension, score))
        
        # Sort by score (weakest first)
        weak_dimensions.sort(key=lambda x: x[1].score)
        
        # Generate suggestions for weak dimensions
        for dimension, score in weak_dimensions[:3]:  # Top 3 weakest
            if dimension == EvaluationDimension.RELEVANCE:
                suggestions.append(
                    f"[HIGH] Improve relevance: {score.reasoning} "
                    "Consider addressing the user's query more directly."
                )
            elif dimension == EvaluationDimension.COHERENCE:
                suggestions.append(
                    f"[MEDIUM] Enhance coherence: {score.reasoning} "
                    "Use clearer transitions and logical flow."
                )
            elif dimension == EvaluationDimension.HELPFULNESS:
                suggestions.append(
                    f"[HIGH] Increase helpfulness: {score.reasoning} "
                    "Provide more actionable information or examples."
                )
            elif dimension == EvaluationDimension.SAFETY:
                suggestions.append(
                    f"[CRITICAL] Address safety concerns: {score.reasoning} "
                    "Review and modify any potentially harmful content."
                )
            elif dimension == EvaluationDimension.EFFICIENCY:
                suggestions.append(
                    f"[LOW] Improve efficiency: {score.reasoning} "
                    "Consider being more concise while maintaining completeness."
                )
        
        # If LLM is available, get detailed suggestions
        if self.use_llm and self.llm_evaluator:
            llm_suggestions = await self.llm_evaluator.generate_improvement_suggestions(
                context.system_response,
                context.user_query,
                {dim.value: {
                    "score": score.score,
                    "reasoning": score.reasoning,
                    "sub_scores": score.sub_scores
                } for dim, score in evaluation.scores.items()}
            )
            suggestions.extend(llm_suggestions)
        
        return suggestions
    
    def get_evaluation_report(self, evaluation: Evaluation) -> str:
        """Generate a human-readable evaluation report."""
        report_lines = [
            f"Evaluation Report - {evaluation.conversation_id}",
            f"Timestamp: {evaluation.timestamp}",
            f"Overall Score: {evaluation.overall_score:.2f}",
            "",
            "Dimension Scores:",
        ]
        
        for dimension, score in evaluation.scores.items():
            report_lines.append(
                f"  {dimension.value.capitalize()}: {score.score:.2f} "
                f"(confidence: {score.confidence:.2f})"
            )
            report_lines.append(f"    Reasoning: {score.reasoning}")
            if score.sub_scores:
                report_lines.append("    Sub-scores:")
                for sub_name, sub_score in score.sub_scores.items():
                    report_lines.append(f"      - {sub_name}: {sub_score:.2f}")
        
        return "\n".join(report_lines)
    
    async def batch_evaluate(self, contexts: List[ConversationContext]) -> List[Evaluation]:
        """Evaluate multiple conversations in parallel."""
        tasks = [self.evaluate_conversation(context) for context in contexts]
        evaluations = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any errors
        valid_evaluations = []
        for i, evaluation in enumerate(evaluations):
            if isinstance(evaluation, Exception):
                # Create error evaluation
                error_eval = Evaluation(
                    conversation_id=contexts[i].conversation_id,
                    scores={},
                    overall_score=0.0,
                    timestamp=datetime.now(),
                    evaluator_version=self.version,
                    metadata={"error": str(evaluation)}
                )
                valid_evaluations.append(error_eval)
            else:
                valid_evaluations.append(evaluation)
        
        return valid_evaluations