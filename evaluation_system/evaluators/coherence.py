import asyncio
from typing import Dict, List
import re
import numpy as np

from .base import BaseEvaluator
from ..core.models import ConversationContext, EvaluationScore, EvaluationDimension


class CoherenceEvaluator(BaseEvaluator):
    """Evaluates the coherence and logical flow of responses."""
    
    def __init__(self):
        super().__init__(EvaluationDimension.COHERENCE)
        
    def _initialize_weights(self) -> Dict[str, float]:
        return {
            "logical_flow": 2.0,
            "consistency": 1.5,
            "structure_quality": 1.0,
            "transition_smoothness": 1.0,
            "completeness": 1.5
        }
    
    async def evaluate(self, context: ConversationContext) -> EvaluationScore:
        """Evaluate coherence of the response."""
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
        """Extract coherence features from the conversation."""
        response = context.system_response
        
        features = {}
        
        # Logical flow
        features["logical_flow"] = self._assess_logical_flow(response)
        
        # Consistency
        features["consistency"] = await self._check_consistency(context)
        
        # Structure quality
        features["structure_quality"] = self._assess_structure(response)
        
        # Transition smoothness
        features["transition_smoothness"] = self._assess_transitions(response)
        
        # Completeness
        features["completeness"] = self._assess_completeness(response)
        
        return features
    
    def _assess_logical_flow(self, response: str) -> float:
        """Assess logical flow of the response."""
        sentences = self._split_into_sentences(response)
        
        if len(sentences) < 2:
            return 0.8  # Single sentence responses are usually coherent
        
        # Check for logical connectors
        logical_connectors = [
            r'\b(therefore|thus|hence|consequently|as a result)\b',
            r'\b(however|but|although|despite|nevertheless)\b',
            r'\b(furthermore|moreover|additionally|also)\b',
            r'\b(firstly|secondly|finally|next|then)\b',
            r'\b(because|since|due to|owing to)\b'
        ]
        
        connector_count = sum(1 for sentence in sentences[1:]
                            for pattern in logical_connectors
                            if re.search(pattern, sentence.lower()))
        
        # Score based on appropriate use of connectors
        expected_connectors = len(sentences) // 3
        if expected_connectors == 0:
            return 0.8
        
        return min(1.0, connector_count / expected_connectors)
    
    async def _check_consistency(self, context: ConversationContext) -> float:
        """Check for consistency within the response and with context."""
        response = context.system_response
        
        # Check for contradictions within response
        contradiction_patterns = [
            (r'\b(always|never|all|none)\b', r'\b(sometimes|some|few)\b'),
            (r'\b(yes|true|correct)\b', r'\b(no|false|incorrect)\b'),
            (r'\b(can|able|possible)\b', r'\b(cannot|unable|impossible)\b')
        ]
        
        sentences = self._split_into_sentences(response)
        contradiction_score = 1.0
        
        for i, sent1 in enumerate(sentences):
            for sent2 in sentences[i+1:]:
                for pos_pattern, neg_pattern in contradiction_patterns:
                    if (re.search(pos_pattern, sent1.lower()) and 
                        re.search(neg_pattern, sent2.lower())):
                        contradiction_score -= 0.2
        
        return max(0.0, contradiction_score)
    
    def _assess_structure(self, response: str) -> float:
        """Assess the structural quality of the response."""
        # Check for proper paragraph structure
        paragraphs = response.split('\n\n')
        sentences = self._split_into_sentences(response)
        
        if not sentences:
            return 0.0
        
        # Score based on:
        # 1. Appropriate paragraph breaks
        # 2. Sentence length variation
        # 3. Not too many very short or very long sentences
        
        structure_score = 0.0
        
        # Paragraph structure
        if len(sentences) > 5:
            expected_paragraphs = len(sentences) // 5
            para_score = min(1.0, len(paragraphs) / expected_paragraphs) if expected_paragraphs > 0 else 0.5
        else:
            para_score = 0.8 if len(paragraphs) == 1 else 0.6
        
        structure_score += para_score * 0.3
        
        # Sentence length variation
        sentence_lengths = [len(sent.split()) for sent in sentences]
        if sentence_lengths:
            cv = np.std(sentence_lengths) / np.mean(sentence_lengths) if np.mean(sentence_lengths) > 0 else 0
            # Moderate variation is good (CV around 0.3-0.7)
            if 0.3 <= cv <= 0.7:
                structure_score += 0.7
            else:
                structure_score += 0.4
        
        return structure_score
    
    def _assess_transitions(self, response: str) -> float:
        """Assess smoothness of transitions between ideas."""
        sentences = self._split_into_sentences(response)
        
        if len(sentences) < 2:
            return 0.8
        
        # Check for smooth transitions
        transition_score = 0.0
        
        for i in range(1, len(sentences)):
            prev_sent = sentences[i-1].lower()
            curr_sent = sentences[i].lower()
            
            # Check for referential continuity
            if any(pronoun in curr_sent[:20] for pronoun in ['this', 'that', 'these', 'those', 'it']):
                transition_score += 1
            
            # Check for topic continuity
            prev_words = set(self._extract_content_words(prev_sent))
            curr_words = set(self._extract_content_words(curr_sent))
            
            if prev_words and curr_words:
                overlap = len(prev_words.intersection(curr_words))
                if overlap > 0:
                    transition_score += 0.5
        
        max_score = (len(sentences) - 1) * 1.5
        return min(1.0, transition_score / max_score) if max_score > 0 else 0.8
    
    def _assess_completeness(self, response: str) -> float:
        """Assess if the response is complete and not cut off."""
        # Check for incomplete sentences
        sentences = self._split_into_sentences(response)
        
        if not sentences:
            return 0.0
        
        completeness_score = 1.0
        
        # Check last sentence for completeness
        last_sentence = sentences[-1].strip()
        
        # Indicators of incomplete response
        if last_sentence.endswith(('...', '..', '-')):
            completeness_score -= 0.3
        
        if not re.search(r'[.!?]$', last_sentence):
            completeness_score -= 0.2
        
        # Check for very short last sentence (might be cut off)
        if len(last_sentence.split()) < 3 and len(sentences) > 1:
            completeness_score -= 0.1
        
        # Check for common incomplete patterns
        incomplete_patterns = [
            r'\b(such as|for example|including)$',
            r'\b(first|second|another)$',
            r':\s*$'
        ]
        
        if any(re.search(pattern, last_sentence.lower()) for pattern in incomplete_patterns):
            completeness_score -= 0.2
        
        return max(0.0, completeness_score)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_content_words(self, text: str) -> List[str]:
        """Extract content words from text."""
        # Remove common function words
        function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                         'of', 'with', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has',
                         'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may',
                         'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in function_words and len(w) > 2]
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence in the evaluation."""
        scores = list(features.values())
        if not scores:
            return 0.5
        
        # Higher confidence if all features indicate good coherence
        mean_score = np.mean(scores)
        variance = np.var(scores)
        
        # High mean and low variance = high confidence
        confidence = mean_score * (1 - min(1.0, variance))
        return confidence
    
    def _generate_reasoning(self, features: Dict[str, float], overall_score: float) -> str:
        """Generate human-readable reasoning for the score."""
        reasoning_parts = []
        
        if overall_score >= 0.8:
            reasoning_parts.append("The response demonstrates excellent coherence.")
        elif overall_score >= 0.6:
            reasoning_parts.append("The response shows good coherence with minor issues.")
        else:
            reasoning_parts.append("The response has significant coherence problems.")
        
        # Add specific observations
        if features.get("logical_flow", 0) < 0.5:
            reasoning_parts.append("Logical flow between ideas could be improved.")
        
        if features.get("consistency", 0) < 0.7:
            reasoning_parts.append("Some inconsistencies detected in the response.")
        
        if features.get("completeness", 0) < 0.8:
            reasoning_parts.append("The response may be incomplete.")
        
        return " ".join(reasoning_parts)