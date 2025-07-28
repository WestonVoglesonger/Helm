import asyncio
from typing import Dict, List
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .base import BaseEvaluator
from ..core.models import ConversationContext, EvaluationScore, EvaluationDimension


class RelevanceEvaluator(BaseEvaluator):
    """Evaluates the relevance of responses to user queries."""
    
    def __init__(self):
        super().__init__(EvaluationDimension.RELEVANCE)
        self.vectorizer = TfidfVectorizer(max_features=100)
        
    def _initialize_weights(self) -> Dict[str, float]:
        return {
            "semantic_similarity": 2.0,
            "keyword_overlap": 1.5,
            "topic_coherence": 1.0,
            "question_answered": 2.5,
            "context_awareness": 1.0
        }
    
    async def evaluate(self, context: ConversationContext) -> EvaluationScore:
        """Evaluate relevance of the response."""
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
        """Extract relevance features from the conversation."""
        query = context.user_query
        response = context.system_response
        
        features = {}
        
        # Semantic similarity using TF-IDF
        features["semantic_similarity"] = await self._calculate_semantic_similarity(query, response)
        
        # Keyword overlap
        features["keyword_overlap"] = self._calculate_keyword_overlap(query, response)
        
        # Topic coherence
        features["topic_coherence"] = await self._assess_topic_coherence(query, response)
        
        # Check if question was answered
        features["question_answered"] = self._check_question_answered(query, response)
        
        # Context awareness
        features["context_awareness"] = self._assess_context_awareness(context)
        
        return features
    
    async def _calculate_semantic_similarity(self, query: str, response: str) -> float:
        """Calculate semantic similarity between query and response."""
        try:
            # Combine query and response for fitting
            texts = [query, response]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception:
            return 0.5
    
    def _calculate_keyword_overlap(self, query: str, response: str) -> float:
        """Calculate keyword overlap between query and response."""
        # Extract important words (excluding stop words)
        query_words = set(self._extract_keywords(query.lower()))
        response_words = set(self._extract_keywords(response.lower()))
        
        if not query_words:
            return 0.5
            
        overlap = len(query_words.intersection(response_words))
        return min(1.0, overlap / len(query_words))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction - remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has',
                     'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may',
                     'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
                     'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where',
                     'why', 'how', 'than', 'then'}
        
        words = re.findall(r'\b\w+\b', text)
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    async def _assess_topic_coherence(self, query: str, response: str) -> float:
        """Assess if response stays on topic."""
        # Simple heuristic: check if response mentions key entities from query
        query_entities = self._extract_entities(query)
        response_entities = self._extract_entities(response)
        
        if not query_entities:
            return 0.7  # Default score if no entities found
            
        mentioned = sum(1 for entity in query_entities if entity in response.lower())
        return min(1.0, mentioned / len(query_entities))
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entities from text."""
        # Simple entity extraction - capitalized words and phrases
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # Also extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted)
        return [e.lower() for e in entities]
    
    def _check_question_answered(self, query: str, response: str) -> float:
        """Check if a question was answered."""
        # Check if query is a question
        question_patterns = [
            r'\?$',
            r'^(what|where|when|why|who|how|which|can|could|would|should|is|are|do|does)',
        ]
        
        is_question = any(re.search(pattern, query.lower()) for pattern in question_patterns)
        
        if not is_question:
            return 0.8  # Not a question, so relevance is good by default
        
        # Check for answer indicators in response
        answer_indicators = [
            r'\b(is|are|was|were|will|would|should|can|could)\b',
            r'\b(yes|no|maybe|perhaps|definitely|certainly)\b',
            r'\b(because|since|due to|as a result)\b',
            r'\d+',  # Numbers often indicate answers
        ]
        
        indicators_found = sum(1 for pattern in answer_indicators 
                             if re.search(pattern, response.lower()))
        
        return min(1.0, indicators_found / 3)
    
    def _assess_context_awareness(self, context: ConversationContext) -> float:
        """Assess if response considers conversation context."""
        if len(context.messages) <= 2:
            return 0.7  # Default for single-turn conversations
        
        # Check if response references previous messages
        prev_messages = " ".join(msg.get("content", "") for msg in context.messages[:-2])
        response = context.system_response
        
        # Simple check: does response reference previous context?
        prev_keywords = self._extract_keywords(prev_messages.lower())
        response_words = set(self._extract_keywords(response.lower()))
        
        if not prev_keywords:
            return 0.7
            
        context_refs = len(set(prev_keywords).intersection(response_words))
        return min(1.0, context_refs / 5)  # Normalize to max of 5 references
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence in the evaluation."""
        # Higher confidence if features are consistent
        scores = list(features.values())
        if not scores:
            return 0.5
            
        variance = np.var(scores)
        # Low variance means high confidence
        confidence = 1.0 - min(1.0, variance * 2)
        return confidence
    
    def _generate_reasoning(self, features: Dict[str, float], overall_score: float) -> str:
        """Generate human-readable reasoning for the score."""
        reasoning_parts = []
        
        if overall_score >= 0.8:
            reasoning_parts.append("The response is highly relevant to the query.")
        elif overall_score >= 0.6:
            reasoning_parts.append("The response is moderately relevant to the query.")
        else:
            reasoning_parts.append("The response has low relevance to the query.")
        
        # Add specific observations
        if features.get("semantic_similarity", 0) > 0.7:
            reasoning_parts.append("Strong semantic similarity detected.")
        
        if features.get("question_answered", 0) < 0.5:
            reasoning_parts.append("The question may not have been directly answered.")
        
        if features.get("keyword_overlap", 0) > 0.6:
            reasoning_parts.append("Good keyword overlap between query and response.")
        
        return " ".join(reasoning_parts)