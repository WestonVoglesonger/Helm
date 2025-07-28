import asyncio
from typing import Dict, List, Optional
import re
import numpy as np

from .base import BaseEvaluator
from .llm_evaluator import LLMEvaluator
from ..core.models import ConversationContext, EvaluationScore, EvaluationDimension


class HelpfulnessEvaluator(BaseEvaluator):
    """Evaluates how helpful responses are to users."""
    
    def __init__(self, use_llm: bool = True, llm_evaluator: Optional[LLMEvaluator] = None):
        super().__init__(EvaluationDimension.HELPFULNESS)
        self.use_llm = use_llm
        self.llm_evaluator = llm_evaluator or LLMEvaluator() if use_llm else None
        
    def _initialize_weights(self) -> Dict[str, float]:
        return {
            "actionability": 2.0,
            "completeness": 1.5,
            "clarity": 1.5,
            "examples_provided": 1.0,
            "detail_appropriateness": 1.0,
            "llm_assessment": 2.5  # High weight for LLM assessment
        }
    
    async def evaluate(self, context: ConversationContext) -> EvaluationScore:
        """Evaluate helpfulness of the response."""
        features = await self.extract_features(context)
        score = self.calculate_weighted_score(features)
        confidence = self._calculate_confidence(features)
        
        reasoning = self._generate_reasoning(features, score)
        
        return EvaluationScore(
            dimension=self.dimension,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            sub_scores=features
        )
    
    async def extract_features(self, context: ConversationContext) -> Dict[str, float]:
        """Extract helpfulness features from the conversation."""
        query = context.user_query
        response = context.system_response
        
        features = {}
        
        # Actionability - does the response provide actionable information?
        features["actionability"] = self._assess_actionability(response, query)
        
        # Completeness - is the response complete?
        features["completeness"] = self._assess_completeness(response, query)
        
        # Clarity - is the response clear and easy to understand?
        features["clarity"] = self._assess_clarity(response)
        
        # Examples provided
        features["examples_provided"] = self._check_examples(response)
        
        # Detail appropriateness
        features["detail_appropriateness"] = self._assess_detail_level(response, query)
        
        # LLM assessment if enabled
        if self.use_llm and self.llm_evaluator:
            llm_score = await self._get_llm_assessment(response, query)
            features["llm_assessment"] = llm_score
        else:
            features["llm_assessment"] = 0.7  # Default neutral score
        
        return features
    
    def _assess_actionability(self, response: str, query: str) -> float:
        """Assess if the response provides actionable information."""
        # Check for action-oriented language
        action_indicators = [
            r'\b(you can|you should|try|consider|recommend|suggest)\b',
            r'\b(step \d+|first|second|then|next|finally)\b',
            r'\b(here\'s how|to do this|follow these)\b',
            r'\b(option|alternative|approach|method|solution)\b',
            r'\b(click|select|enter|type|run|execute|install)\b'
        ]
        
        # Check if query asks for help/how-to
        help_query_patterns = [
            r'\b(how|what|where|when|why|can|could|should)\b',
            r'\b(help|assist|guide|explain|show)\b',
            r'\b(problem|issue|error|fix|solve)\b'
        ]
        
        is_help_query = any(re.search(pattern, query.lower()) for pattern in help_query_patterns)
        
        if not is_help_query:
            return 0.8  # Not a help query, so actionability is less relevant
        
        # Count action indicators
        action_count = sum(1 for pattern in action_indicators 
                          if re.search(pattern, response.lower()))
        
        # Normalize score
        return min(1.0, action_count / 3)
    
    def _assess_completeness(self, response: str, query: str) -> float:
        """Assess if the response completely addresses the query."""
        # Extract question components from query
        question_words = self._extract_question_components(query)
        
        if not question_words:
            return 0.8  # Default for non-questions
        
        # Check if response addresses each component
        addressed = 0
        for component in question_words:
            if component.lower() in response.lower():
                addressed += 1
        
        completeness = addressed / len(question_words) if question_words else 0.8
        
        # Check for explicit incompleteness indicators
        incomplete_indicators = [
            r'\b(partial|incomplete|limited|basic)\b',
            r'for more information',
            r'this is just|only covers|briefly'
        ]
        
        if any(re.search(pattern, response.lower()) for pattern in incomplete_indicators):
            completeness *= 0.8
        
        return completeness
    
    def _assess_clarity(self, response: str) -> float:
        """Assess clarity of the response."""
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        clarity_score = 1.0
        
        # Check for overly complex sentences
        for sentence in sentences:
            words = sentence.split()
            
            # Very long sentences reduce clarity
            if len(words) > 30:
                clarity_score -= 0.1
            
            # Multiple clauses reduce clarity
            clause_indicators = len(re.findall(r'[,;:]', sentence))
            if clause_indicators > 3:
                clarity_score -= 0.1
        
        # Check for jargon without explanation
        jargon_pattern = r'\b[A-Z]{3,}\b'  # Acronyms
        acronyms = re.findall(jargon_pattern, response)
        
        # Penalty for unexplained acronyms
        for acronym in set(acronyms):
            if f"({acronym})" not in response and f"{acronym} (" not in response:
                clarity_score -= 0.05
        
        # Check for clear structure (lists, paragraphs)
        if re.search(r'(\d+\.|\*|-)\s', response):  # Has lists
            clarity_score += 0.1
        
        if '\n\n' in response:  # Has paragraphs
            clarity_score += 0.1
        
        return max(0.0, min(1.0, clarity_score))
    
    def _check_examples(self, response: str) -> float:
        """Check if examples are provided."""
        example_indicators = [
            r'\b(for example|for instance|e\.g\.|such as)\b',
            r'\b(example:|here\'s an example)\b',
            r'```',  # Code blocks
            r'"[^"]{10,}"',  # Quoted examples
        ]
        
        example_count = sum(1 for pattern in example_indicators 
                          if re.search(pattern, response, re.IGNORECASE))
        
        # More examples = higher score, but cap at 1.0
        return min(1.0, example_count * 0.3)
    
    def _assess_detail_level(self, response: str, query: str) -> float:
        """Assess if the level of detail is appropriate."""
        response_length = len(response.split())
        query_length = len(query.split())
        
        # Simple queries should have concise responses
        if query_length < 10:
            if response_length < 50:
                return 0.9
            elif response_length < 150:
                return 0.8
            else:
                return 0.6  # Too detailed for simple query
        
        # Complex queries need more detail
        else:
            if response_length < 30:
                return 0.4  # Too brief
            elif response_length < 100:
                return 0.7
            else:
                return 0.9  # Good detail for complex query
    
    async def _get_llm_assessment(self, response: str, query: str) -> float:
        """Get LLM assessment of helpfulness."""
        try:
            results = await self.llm_evaluator.assess_quality(
                response=response,
                context=query,
                dimension="helpfulness"
            )
            
            if "helpfulness" in results and not results["helpfulness"].get("error"):
                return results["helpfulness"]["score"]
            else:
                return 0.7  # Default if LLM evaluation fails
                
        except Exception:
            return 0.7  # Default if LLM evaluation fails
    
    def _extract_question_components(self, query: str) -> List[str]:
        """Extract key components from a question."""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when',
                     'where', 'which', 'who', 'can', 'could', 'would', 'should'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        components = [w for w in words if w not in stop_words and len(w) > 2]
        
        return components[:5]  # Focus on top 5 components
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence in the evaluation."""
        scores = list(features.values())
        if not scores:
            return 0.5
        
        # If LLM assessment is available and successful, we have higher confidence
        if self.use_llm and features.get("llm_assessment", 0) > 0:
            base_confidence = 0.8
        else:
            base_confidence = 0.6
        
        # Adjust based on consistency of other features
        variance = np.var([v for k, v in features.items() if k != "llm_assessment"])
        confidence = base_confidence * (1 - min(0.5, variance))
        
        return confidence
    
    def _generate_reasoning(self, features: Dict[str, float], overall_score: float) -> str:
        """Generate human-readable reasoning for the score."""
        reasoning_parts = []
        
        if overall_score >= 0.8:
            reasoning_parts.append("The response is very helpful to the user.")
        elif overall_score >= 0.6:
            reasoning_parts.append("The response is moderately helpful.")
        else:
            reasoning_parts.append("The response has limited helpfulness.")
        
        # Add specific observations
        if features.get("actionability", 0) > 0.7:
            reasoning_parts.append("Good actionable guidance provided.")
        elif features.get("actionability", 0) < 0.4:
            reasoning_parts.append("Could be more actionable.")
        
        if features.get("examples_provided", 0) > 0.5:
            reasoning_parts.append("Helpful examples included.")
        
        if features.get("clarity", 0) < 0.6:
            reasoning_parts.append("Clarity could be improved.")
        
        return " ".join(reasoning_parts)