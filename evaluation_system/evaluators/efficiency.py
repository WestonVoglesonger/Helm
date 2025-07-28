import asyncio
from typing import Dict, List, Optional
import re
import numpy as np
from collections import Counter

from .base import BaseEvaluator
from .llm_evaluator import LLMEvaluator
from ..core.models import ConversationContext, EvaluationScore, EvaluationDimension


class EfficiencyEvaluator(BaseEvaluator):
    """Evaluates the efficiency of responses."""
    
    def __init__(self, use_llm: bool = True, llm_evaluator: Optional[LLMEvaluator] = None):
        super().__init__(EvaluationDimension.EFFICIENCY)
        self.use_llm = use_llm
        self.llm_evaluator = llm_evaluator or LLMEvaluator() if use_llm else None
        
    def _initialize_weights(self) -> Dict[str, float]:
        return {
            "conciseness": 2.0,
            "information_density": 1.5,
            "redundancy_check": 1.5,
            "directness": 1.0,
            "structure_efficiency": 1.0,
            "llm_assessment": 2.0
        }
    
    async def evaluate(self, context: ConversationContext) -> EvaluationScore:
        """Evaluate efficiency of the response."""
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
        """Extract efficiency features from the conversation."""
        query = context.user_query
        response = context.system_response
        
        features = {}
        
        # Conciseness - is the response appropriately concise?
        features["conciseness"] = self._assess_conciseness(response, query)
        
        # Information density - how much useful info per word?
        features["information_density"] = self._calculate_information_density(response)
        
        # Redundancy check - is there repetition?
        features["redundancy_check"] = self._check_redundancy(response)
        
        # Directness - does it get to the point?
        features["directness"] = self._assess_directness(response, query)
        
        # Structure efficiency
        features["structure_efficiency"] = self._assess_structure_efficiency(response)
        
        # LLM assessment if enabled
        if self.use_llm and self.llm_evaluator:
            llm_score = await self._get_llm_assessment(response, query)
            features["llm_assessment"] = llm_score
        else:
            features["llm_assessment"] = 0.7  # Default neutral score
        
        return features
    
    def _assess_conciseness(self, response: str, query: str) -> float:
        """Assess if response is appropriately concise."""
        response_words = len(response.split())
        query_words = len(query.split())
        
        # Calculate expected response length based on query complexity
        query_complexity = self._assess_query_complexity(query)
        
        if query_complexity == "simple":
            ideal_range = (20, 80)
        elif query_complexity == "moderate":
            ideal_range = (50, 200)
        else:  # complex
            ideal_range = (100, 400)
        
        # Score based on how close to ideal range
        if ideal_range[0] <= response_words <= ideal_range[1]:
            return 1.0
        elif response_words < ideal_range[0]:
            # Too brief
            ratio = response_words / ideal_range[0]
            return max(0.3, ratio)
        else:
            # Too verbose
            excess_ratio = response_words / ideal_range[1]
            return max(0.3, 1.0 / excess_ratio)
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess complexity of the query."""
        indicators = {
            "simple": [
                r'\b(what is|who is|when is|where is)\b',
                r'\b(yes|no) question\b',
                r'\b(define|meaning of)\b',
            ],
            "complex": [
                r'\b(explain|analyze|compare|evaluate)\b',
                r'\b(how does.*work|why does)\b',
                r'\b(pros and cons|advantages|disadvantages)\b',
                r'\b(step.?by.?step|detailed|comprehensive)\b',
            ]
        }
        
        query_lower = query.lower()
        
        simple_count = sum(1 for pattern in indicators["simple"] 
                          if re.search(pattern, query_lower))
        complex_count = sum(1 for pattern in indicators["complex"] 
                           if re.search(pattern, query_lower))
        
        if complex_count > simple_count:
            return "complex"
        elif simple_count > 0:
            return "simple"
        else:
            return "moderate"
    
    def _calculate_information_density(self, response: str) -> float:
        """Calculate information density of the response."""
        words = response.split()
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        # Count meaningful content words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has',
                     'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may',
                     'might', 'must', 'can', 'this', 'that', 'these', 'those', 'very',
                     'really', 'quite', 'just', 'so'}
        
        content_words = [w.lower() for w in words if w.lower() not in stop_words and len(w) > 2]
        content_word_count = len(content_words)
        
        # Count unique concepts
        unique_concepts = len(set(content_words))
        
        # Information density = ratio of content words and concept diversity
        density_score = (content_word_count / total_words) * 0.5
        diversity_score = min(1.0, unique_concepts / (total_words * 0.3)) * 0.5
        
        return density_score + diversity_score
    
    def _check_redundancy(self, response: str) -> float:
        """Check for redundancy and repetition."""
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip().lower() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 1.0  # No redundancy possible
        
        redundancy_score = 1.0
        
        # Check for repeated sentences or very similar sentences
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity = self._calculate_sentence_similarity(sentences[i], sentences[j])
                if similarity > 0.8:  # Very similar
                    redundancy_score -= 0.2
                elif similarity > 0.6:  # Somewhat similar
                    redundancy_score -= 0.1
        
        # Check for repeated phrases
        phrase_counts = self._count_phrases(response)
        for phrase, count in phrase_counts.items():
            if len(phrase.split()) >= 3 and count > 2:
                redundancy_score -= 0.1 * (count - 2)
        
        return max(0.0, redundancy_score)
    
    def _calculate_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences."""
        words1 = set(sent1.split())
        words2 = set(sent2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _count_phrases(self, text: str) -> Counter:
        """Count repeated phrases in text."""
        words = text.lower().split()
        phrases = Counter()
        
        # Check phrases of length 3-5
        for phrase_len in range(3, 6):
            for i in range(len(words) - phrase_len + 1):
                phrase = ' '.join(words[i:i + phrase_len])
                phrases[phrase] += 1
        
        # Filter out phrases that appear only once
        return Counter({p: c for p, c in phrases.items() if c > 1})
    
    def _assess_directness(self, response: str, query: str) -> float:
        """Assess if response directly addresses the query."""
        # Check if response starts with direct answer
        first_sentence = re.split(r'[.!?]', response)[0].strip()
        
        directness_score = 0.7  # Base score
        
        # Direct answer indicators
        direct_patterns = [
            r'^(Yes|No|Correct|Incorrect)',
            r'^(The answer is|It is|This is)',
            r'^[\d\w\s]{1,20}:',  # Short direct answer followed by colon
        ]
        
        if any(re.search(pattern, first_sentence) for pattern in direct_patterns):
            directness_score += 0.3
        
        # Check for indirect/verbose openings
        indirect_patterns = [
            r'^(Well|So|Actually|Basically)',
            r'^(This is an? (interesting|good|great) question)',
            r'^(To answer your question)',
            r'^(Before I answer)',
        ]
        
        if any(re.search(pattern, first_sentence, re.IGNORECASE) for pattern in indirect_patterns):
            directness_score -= 0.2
        
        # Check if main point comes early
        response_lower = response.lower()
        query_keywords = self._extract_keywords(query)
        
        # Find position of first keyword mention
        first_keyword_position = len(response)
        for keyword in query_keywords[:3]:  # Check top 3 keywords
            pos = response_lower.find(keyword.lower())
            if pos != -1 and pos < first_keyword_position:
                first_keyword_position = pos
        
        # Score based on how early keywords appear
        if first_keyword_position < len(response) * 0.2:
            directness_score += 0.1
        elif first_keyword_position > len(response) * 0.5:
            directness_score -= 0.1
        
        return max(0.0, min(1.0, directness_score))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when',
                     'where', 'which', 'who', 'can', 'could', 'would', 'should'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Return unique keywords in order of appearance
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def _assess_structure_efficiency(self, response: str) -> float:
        """Assess efficiency of response structure."""
        structure_score = 0.7  # Base score
        
        # Check for efficient formatting
        if re.search(r'(\d+\.|\*|-)\s', response):  # Lists
            structure_score += 0.1
        
        if response.count('\n\n') > 0 and response.count('\n\n') < 5:  # Paragraphs
            structure_score += 0.1
        
        # Check for headers/sections
        if re.search(r'\n#{1,3}\s|\n\*\*[^*]+\*\*', response):
            structure_score += 0.1
        
        # Penalize excessive formatting
        if response.count('\n') > len(response.split()) / 10:  # Too many line breaks
            structure_score -= 0.2
        
        return max(0.0, min(1.0, structure_score))
    
    async def _get_llm_assessment(self, response: str, query: str) -> float:
        """Get LLM assessment of efficiency."""
        try:
            results = await self.llm_evaluator.assess_quality(
                response=response,
                context=query,
                dimension="efficiency"
            )
            
            if "efficiency" in results and not results["efficiency"].get("error"):
                return results["efficiency"]["score"]
            else:
                return 0.7  # Default if LLM evaluation fails
                
        except Exception:
            return 0.7  # Default if LLM evaluation fails
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence in the evaluation."""
        scores = list(features.values())
        if not scores:
            return 0.5
        
        # Higher confidence for clear efficiency issues
        mean_score = np.mean(scores)
        variance = np.var(scores)
        
        if mean_score < 0.4 or mean_score > 0.8:
            # Clear efficiency issue or very efficient
            confidence = 0.85
        else:
            # Less clear, adjust by variance
            confidence = 0.7 - min(0.2, variance)
        
        return confidence
    
    def _generate_reasoning(self, features: Dict[str, float], overall_score: float) -> str:
        """Generate human-readable reasoning for the score."""
        reasoning_parts = []
        
        if overall_score >= 0.8:
            reasoning_parts.append("The response is highly efficient.")
        elif overall_score >= 0.6:
            reasoning_parts.append("The response has moderate efficiency.")
        else:
            reasoning_parts.append("The response could be more efficient.")
        
        # Add specific observations
        if features.get("conciseness", 0) < 0.5:
            reasoning_parts.append("Response length is not optimal for the query.")
        
        if features.get("redundancy_check", 0) < 0.7:
            reasoning_parts.append("Some redundancy detected.")
        
        if features.get("information_density", 0) > 0.7:
            reasoning_parts.append("Good information density.")
        
        if features.get("directness", 0) < 0.6:
            reasoning_parts.append("Could be more direct in addressing the query.")
        
        return " ".join(reasoning_parts)