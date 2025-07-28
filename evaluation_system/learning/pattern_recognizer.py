import asyncio
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from ..core.models import (
    Evaluation,
    UserFeedback,
    EvaluationPattern,
    EvaluationDimension
)


class PatternRecognizer:
    """Recognizes patterns in evaluation feedback to improve future evaluations."""
    
    def __init__(self):
        self.pattern_threshold = 0.1  # Minimum pattern frequency
        self.confidence_threshold = 0.6  # Minimum confidence for pattern
        
    async def analyze(self, feedback_history: List[Tuple[Evaluation, UserFeedback]]) -> List[EvaluationPattern]:
        """Analyze feedback history to identify patterns."""
        patterns = []
        
        # Analyze dimension-specific patterns
        dimension_patterns = await self._analyze_dimension_patterns(feedback_history)
        patterns.extend(dimension_patterns)
        
        # Analyze cross-dimension patterns
        cross_patterns = await self._analyze_cross_dimension_patterns(feedback_history)
        patterns.extend(cross_patterns)
        
        # Analyze temporal patterns
        temporal_patterns = await self._analyze_temporal_patterns(feedback_history)
        patterns.extend(temporal_patterns)
        
        # Analyze outlier patterns
        outlier_patterns = await self._analyze_outlier_patterns(feedback_history)
        patterns.extend(outlier_patterns)
        
        # Filter and rank patterns
        filtered_patterns = self._filter_patterns(patterns)
        
        return filtered_patterns
    
    async def _analyze_dimension_patterns(self, 
                                        feedback_history: List[Tuple[Evaluation, UserFeedback]]) -> List[EvaluationPattern]:
        """Analyze patterns specific to each dimension."""
        patterns = []
        
        for dimension in EvaluationDimension:
            # Collect discrepancies for this dimension
            discrepancies = []
            
            for evaluation, feedback in feedback_history:
                if dimension in evaluation.scores and dimension in feedback.dimensions_feedback:
                    eval_score = evaluation.scores[dimension].score
                    user_score = feedback.dimensions_feedback[dimension]
                    discrepancy = user_score - eval_score
                    
                    discrepancies.append({
                        "discrepancy": discrepancy,
                        "eval_score": eval_score,
                        "user_score": user_score,
                        "sub_scores": evaluation.scores[dimension].sub_scores
                    })
            
            if len(discrepancies) >= 10:
                # Analyze consistent over/under estimation
                avg_discrepancy = np.mean([d["discrepancy"] for d in discrepancies])
                
                if abs(avg_discrepancy) > 0.1:  # Significant consistent bias
                    pattern = EvaluationPattern(
                        pattern_type="dimension_bias",
                        description=f"{dimension.value} consistently {'over' if avg_discrepancy < 0 else 'under'}-estimated",
                        occurrences=len(discrepancies),
                        confidence=min(0.9, abs(avg_discrepancy) * 3),
                        dimension=dimension,
                        adjustment_factor=1.0 + (avg_discrepancy * 0.5)
                    )
                    patterns.append(pattern)
                
                # Analyze sub-score patterns
                sub_patterns = self._analyze_sub_score_patterns(dimension, discrepancies)
                patterns.extend(sub_patterns)
        
        return patterns
    
    def _analyze_sub_score_patterns(self, dimension: EvaluationDimension,
                                  discrepancies: List[Dict]) -> List[EvaluationPattern]:
        """Analyze patterns in sub-scores for a dimension."""
        patterns = []
        sub_score_correlations = defaultdict(list)
        
        for disc_data in discrepancies:
            discrepancy = disc_data["discrepancy"]
            sub_scores = disc_data.get("sub_scores", {})
            
            for sub_name, sub_score in sub_scores.items():
                sub_score_correlations[sub_name].append((sub_score, discrepancy))
        
        # Find sub-scores that correlate with discrepancies
        for sub_name, correlations in sub_score_correlations.items():
            if len(correlations) >= 10:
                scores, discrepancies = zip(*correlations)
                
                # Calculate correlation
                if np.std(scores) > 0 and np.std(discrepancies) > 0:
                    correlation = np.corrcoef(scores, discrepancies)[0, 1]
                    
                    if abs(correlation) > 0.5:  # Strong correlation
                        pattern = EvaluationPattern(
                            pattern_type="sub_score_correlation",
                            description=f"{sub_name} in {dimension.value} correlates with user disagreement",
                            occurrences=len(correlations),
                            confidence=abs(correlation),
                            dimension=dimension,
                            adjustment_factor=1.0 - (correlation * 0.3)
                        )
                        patterns.append(pattern)
        
        return patterns
    
    async def _analyze_cross_dimension_patterns(self,
                                              feedback_history: List[Tuple[Evaluation, UserFeedback]]) -> List[EvaluationPattern]:
        """Analyze patterns across multiple dimensions."""
        patterns = []
        
        # Collect data for clustering
        feature_vectors = []
        discrepancies = []
        
        for evaluation, feedback in feedback_history:
            feature_vector = []
            overall_discrepancy = feedback.rating - evaluation.overall_score
            
            # Create feature vector from all dimension scores
            for dimension in EvaluationDimension:
                if dimension in evaluation.scores:
                    feature_vector.append(evaluation.scores[dimension].score)
                else:
                    feature_vector.append(0.5)  # Default
            
            if len(feature_vector) == len(EvaluationDimension):
                feature_vectors.append(feature_vector)
                discrepancies.append(overall_discrepancy)
        
        if len(feature_vectors) >= 20:
            # Cluster similar evaluations
            patterns_found = self._cluster_based_patterns(feature_vectors, discrepancies)
            patterns.extend(patterns_found)
        
        return patterns
    
    def _cluster_based_patterns(self, feature_vectors: List[List[float]], 
                              discrepancies: List[float]) -> List[EvaluationPattern]:
        """Find patterns using clustering."""
        patterns = []
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_vectors)
        
        # Perform clustering
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(scaled_features)
        
        # Analyze each cluster
        cluster_labels = clustering.labels_
        unique_clusters = set(cluster_labels) - {-1}  # Exclude noise
        
        for cluster_id in unique_clusters:
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_discrepancies = [discrepancies[i] for i in cluster_indices]
            
            avg_discrepancy = np.mean(cluster_discrepancies)
            
            if abs(avg_discrepancy) > 0.15:  # Significant pattern
                # Identify cluster characteristics
                cluster_features = [feature_vectors[i] for i in cluster_indices]
                avg_features = np.mean(cluster_features, axis=0)
                
                # Find dominant dimensions
                dominant_dims = []
                for i, dim in enumerate(EvaluationDimension):
                    if avg_features[i] > 0.7 or avg_features[i] < 0.3:
                        dominant_dims.append(dim.value)
                
                pattern = EvaluationPattern(
                    pattern_type="cluster_bias",
                    description=f"Evaluation bias when {', '.join(dominant_dims)} scores are {'high' if avg_discrepancy > 0 else 'low'}",
                    occurrences=len(cluster_indices),
                    confidence=min(0.8, abs(avg_discrepancy) * 2),
                    adjustment_factor=1.0 + (avg_discrepancy * 0.4)
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _analyze_temporal_patterns(self,
                                       feedback_history: List[Tuple[Evaluation, UserFeedback]]) -> List[EvaluationPattern]:
        """Analyze patterns over time."""
        patterns = []
        
        # Sort by timestamp
        sorted_history = sorted(feedback_history, 
                              key=lambda x: x[0].timestamp)
        
        if len(sorted_history) >= 20:
            # Analyze trend in discrepancies
            recent_discrepancies = []
            old_discrepancies = []
            
            midpoint = len(sorted_history) // 2
            
            for i, (evaluation, feedback) in enumerate(sorted_history):
                discrepancy = feedback.rating - evaluation.overall_score
                
                if i < midpoint:
                    old_discrepancies.append(discrepancy)
                else:
                    recent_discrepancies.append(discrepancy)
            
            # Check if accuracy is improving or degrading
            old_avg = np.mean(np.abs(old_discrepancies))
            recent_avg = np.mean(np.abs(recent_discrepancies))
            
            change = recent_avg - old_avg
            
            if abs(change) > 0.1:
                pattern = EvaluationPattern(
                    pattern_type="temporal_trend",
                    description=f"Evaluation accuracy {'degrading' if change > 0 else 'improving'} over time",
                    occurrences=len(sorted_history),
                    confidence=min(0.8, abs(change) * 3),
                    adjustment_factor=1.0 if change < 0 else 0.9  # Reduce confidence if degrading
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _analyze_outlier_patterns(self,
                                      feedback_history: List[Tuple[Evaluation, UserFeedback]]) -> List[EvaluationPattern]:
        """Analyze patterns in outlier cases."""
        patterns = []
        
        # Identify outliers (large discrepancies)
        outliers = []
        
        for evaluation, feedback in feedback_history:
            discrepancy = feedback.rating - evaluation.overall_score
            
            if abs(discrepancy) > 0.3:  # Significant disagreement
                outliers.append({
                    "evaluation": evaluation,
                    "feedback": feedback,
                    "discrepancy": discrepancy
                })
        
        if len(outliers) >= 5:
            # Analyze common characteristics of outliers
            common_features = self._find_common_features(outliers)
            
            for feature_type, feature_value, count in common_features:
                if count / len(outliers) > 0.5:  # Common in majority of outliers
                    pattern = EvaluationPattern(
                        pattern_type="outlier_characteristic",
                        description=f"Large discrepancies often occur when {feature_type} is {feature_value}",
                        occurrences=count,
                        confidence=count / len(outliers),
                        adjustment_factor=0.8  # Reduce confidence for edge cases
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _find_common_features(self, outliers: List[Dict]) -> List[Tuple[str, str, int]]:
        """Find common features among outlier cases."""
        feature_counts = defaultdict(Counter)
        
        for outlier_data in outliers:
            evaluation = outlier_data["evaluation"]
            
            # Check dimension scores
            for dimension, score in evaluation.scores.items():
                if score.score > 0.8:
                    feature_counts["high_score"][dimension.value] += 1
                elif score.score < 0.3:
                    feature_counts["low_score"][dimension.value] += 1
                
                # Check confidence levels
                if score.confidence < 0.5:
                    feature_counts["low_confidence"][dimension.value] += 1
        
        # Extract common features
        common_features = []
        for feature_type, counts in feature_counts.items():
            for feature_value, count in counts.most_common(3):
                if count >= 3:  # Appears in at least 3 outliers
                    common_features.append((feature_type, feature_value, count))
        
        return common_features
    
    def _filter_patterns(self, patterns: List[EvaluationPattern]) -> List[EvaluationPattern]:
        """Filter and rank patterns by importance."""
        # Filter by confidence
        filtered = [p for p in patterns if p.confidence >= self.confidence_threshold]
        
        # Remove duplicate or conflicting patterns
        unique_patterns = []
        seen_descriptions = set()
        
        for pattern in sorted(filtered, key=lambda x: x.confidence, reverse=True):
            # Simple deduplication by description similarity
            if pattern.description not in seen_descriptions:
                unique_patterns.append(pattern)
                seen_descriptions.add(pattern.description)
        
        # Sort by importance (confidence * occurrences)
        unique_patterns.sort(
            key=lambda p: p.confidence * np.log1p(p.occurrences),
            reverse=True
        )
        
        return unique_patterns[:10]  # Return top 10 patterns