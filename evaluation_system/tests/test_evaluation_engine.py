"""Tests for the Intelligent Evaluation Engine."""

import pytest
import asyncio
import uuid
from datetime import datetime

from evaluation_system import (
    IntelligentEvaluationEngine,
    ConversationContext,
    UserFeedback,
    EvaluationDimension
)


@pytest.fixture
def engine():
    """Create evaluation engine instance."""
    return IntelligentEvaluationEngine(use_llm=False)


@pytest.fixture
def sample_context():
    """Create sample conversation context."""
    return ConversationContext(
        conversation_id=str(uuid.uuid4()),
        messages=[],
        user_query="What is Python?",
        system_response="Python is a high-level programming language known for its simplicity and readability.",
        metadata={}
    )


@pytest.mark.asyncio
async def test_basic_evaluation(engine, sample_context):
    """Test basic evaluation functionality."""
    evaluation = await engine.evaluate_conversation(sample_context)
    
    assert evaluation is not None
    assert 0 <= evaluation.overall_score <= 1
    assert len(evaluation.scores) == 5  # All dimensions evaluated
    
    for dimension, score in evaluation.scores.items():
        assert 0 <= score.score <= 1
        assert 0 <= score.confidence <= 1
        assert score.reasoning is not None


@pytest.mark.asyncio
async def test_relevance_evaluation(engine):
    """Test relevance scoring."""
    # Highly relevant response
    relevant_context = ConversationContext(
        conversation_id=str(uuid.uuid4()),
        messages=[],
        user_query="What is the capital of France?",
        system_response="The capital of France is Paris.",
        metadata={}
    )
    
    # Irrelevant response
    irrelevant_context = ConversationContext(
        conversation_id=str(uuid.uuid4()),
        messages=[],
        user_query="What is the capital of France?",
        system_response="I like pizza.",
        metadata={}
    )
    
    relevant_eval = await engine.evaluate_conversation(relevant_context)
    irrelevant_eval = await engine.evaluate_conversation(irrelevant_context)
    
    assert relevant_eval.scores[EvaluationDimension.RELEVANCE].score > 0.7
    assert irrelevant_eval.scores[EvaluationDimension.RELEVANCE].score < 0.5


@pytest.mark.asyncio
async def test_safety_evaluation(engine):
    """Test safety scoring."""
    # Safe response
    safe_context = ConversationContext(
        conversation_id=str(uuid.uuid4()),
        messages=[],
        user_query="How can I help someone who is sad?",
        system_response="You can offer emotional support by listening, being present, and encouraging them to seek professional help if needed.",
        metadata={}
    )
    
    # Unsafe response
    unsafe_context = ConversationContext(
        conversation_id=str(uuid.uuid4()),
        messages=[],
        user_query="How can I help someone who is sad?",
        system_response="Tell them to stop being weak and just get over it.",
        metadata={}
    )
    
    safe_eval = await engine.evaluate_conversation(safe_context)
    unsafe_eval = await engine.evaluate_conversation(unsafe_context)
    
    assert safe_eval.scores[EvaluationDimension.SAFETY].score > 0.8
    assert unsafe_eval.scores[EvaluationDimension.SAFETY].score < 0.7


@pytest.mark.asyncio
async def test_efficiency_evaluation(engine):
    """Test efficiency scoring."""
    # Efficient response
    efficient_context = ConversationContext(
        conversation_id=str(uuid.uuid4()),
        messages=[],
        user_query="What is 2+2?",
        system_response="2 + 2 = 4",
        metadata={}
    )
    
    # Inefficient response
    inefficient_context = ConversationContext(
        conversation_id=str(uuid.uuid4()),
        messages=[],
        user_query="What is 2+2?",
        system_response="""Well, this is an interesting mathematical question. To understand addition, 
        we must first consider the history of mathematics, which dates back thousands of years. 
        The ancient Egyptians and Babylonians developed early number systems. Moving forward to 
        your specific question about 2+2, we need to understand that addition is a fundamental 
        arithmetic operation. When we add 2 and 2 together, we are combining two groups of two 
        items each. The result of this operation is 4.""",
        metadata={}
    )
    
    efficient_eval = await engine.evaluate_conversation(efficient_context)
    inefficient_eval = await engine.evaluate_conversation(inefficient_context)
    
    assert efficient_eval.scores[EvaluationDimension.EFFICIENCY].score > 0.8
    assert inefficient_eval.scores[EvaluationDimension.EFFICIENCY].score < 0.5


@pytest.mark.asyncio
async def test_response_comparison(engine):
    """Test response comparison functionality."""
    query = "How do I make coffee?"
    
    response1 = "Put coffee in water and heat it."
    response2 = """To make coffee:
1. Boil water to 195-205°F
2. Add 1-2 tablespoons of ground coffee per 6 oz of water
3. Pour water over grounds and let steep for 4-5 minutes
4. Filter and serve"""
    
    comparison = await engine.compare_responses(response1, response2, query)
    
    assert comparison["better_response"] == 2
    assert comparison["score_difference"] > 0.1


@pytest.mark.asyncio
async def test_learning_from_feedback(engine, sample_context):
    """Test learning from user feedback."""
    # Get initial evaluation
    eval1 = await engine.evaluate_conversation(sample_context)
    initial_score = eval1.overall_score
    
    # Create feedback indicating user thinks it's better
    feedback = UserFeedback(
        evaluation_id=str(uuid.uuid4()),
        conversation_id=eval1.conversation_id,
        rating=min(1.0, initial_score + 0.2),
        dimensions_feedback={
            dim: min(1.0, score.score + 0.1) 
            for dim, score in eval1.scores.items()
        },
        comments="Better than evaluated"
    )
    
    # Learn from feedback
    await engine.learn_from_feedback(eval1, feedback)
    
    # Check that learning occurred
    stats = engine.learning_module.get_learning_statistics()
    assert stats["total_feedback"] == 1


@pytest.mark.asyncio
async def test_batch_evaluation(engine):
    """Test batch evaluation functionality."""
    contexts = [
        ConversationContext(
            conversation_id=str(uuid.uuid4()),
            messages=[],
            user_query=f"Question {i}",
            system_response=f"Answer {i}",
            metadata={}
        )
        for i in range(5)
    ]
    
    evaluations = await engine.batch_evaluate(contexts)
    
    assert len(evaluations) == 5
    for eval in evaluations:
        assert 0 <= eval.overall_score <= 1


@pytest.mark.asyncio
async def test_improvement_suggestions(engine):
    """Test improvement suggestion generation."""
    poor_context = ConversationContext(
        conversation_id=str(uuid.uuid4()),
        messages=[],
        user_query="How do I learn programming?",
        system_response="Just code.",
        metadata={}
    )
    
    evaluation = await engine.evaluate_conversation(poor_context)
    suggestions = await engine.generate_improvement_suggestions(evaluation, poor_context)
    
    assert len(suggestions) > 0
    assert any("helpfulness" in s.lower() for s in suggestions)


@pytest.mark.asyncio
async def test_coherence_evaluation(engine):
    """Test coherence evaluation."""
    # Coherent response
    coherent_context = ConversationContext(
        conversation_id=str(uuid.uuid4()),
        messages=[],
        user_query="Explain photosynthesis",
        system_response="""Photosynthesis is the process by which plants convert light energy into chemical energy. 
        First, chlorophyll in the leaves absorbs sunlight. Then, this energy is used to convert carbon dioxide 
        from the air and water from the soil into glucose. Finally, oxygen is released as a byproduct.""",
        metadata={}
    )
    
    # Incoherent response
    incoherent_context = ConversationContext(
        conversation_id=str(uuid.uuid4()),
        messages=[],
        user_query="Explain photosynthesis",
        system_response="""Plants need sunlight. Oxygen is important. Chlorophyll is green. 
        Water comes from rain. Carbon dioxide. Energy is stored. Leaves are where it happens.""",
        metadata={}
    )
    
    coherent_eval = await engine.evaluate_conversation(coherent_context)
    incoherent_eval = await engine.evaluate_conversation(incoherent_context)
    
    assert coherent_eval.scores[EvaluationDimension.COHERENCE].score > 0.7
    assert incoherent_eval.scores[EvaluationDimension.COHERENCE].score < 0.6


@pytest.mark.asyncio
async def test_evaluation_report(engine, sample_context):
    """Test evaluation report generation."""
    evaluation = await engine.evaluate_conversation(sample_context)
    report = engine.get_evaluation_report(evaluation)
    
    assert "Overall Score" in report
    assert "Dimension Scores" in report
    for dimension in EvaluationDimension:
        assert dimension.value in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])