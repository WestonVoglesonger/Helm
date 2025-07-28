"""Example usage of the Intelligent Evaluation System."""

import asyncio
import uuid
from datetime import datetime

from evaluation_system import (
    IntelligentEvaluationEngine,
    ConversationContext,
    UserFeedback,
    EvaluationDimension
)


async def main():
    """Demonstrate the evaluation system capabilities."""
    
    # Initialize the evaluation engine
    # Set use_llm=False to run without GPT-4o (using only heuristics)
    # Set use_llm=True and provide API key to use GPT-4o
    engine = IntelligentEvaluationEngine(use_llm=False)  # Change to True with API key
    
    print("=== Intelligent Evaluation System Demo ===\n")
    
    # Example 1: Basic conversation evaluation
    print("1. Basic Conversation Evaluation")
    print("-" * 40)
    
    context1 = ConversationContext(
        conversation_id=str(uuid.uuid4()),
        messages=[
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."}
        ],
        user_query="What is machine learning?",
        system_response="""Machine learning is a subset of artificial intelligence (AI) that enables 
systems to learn and improve from experience without being explicitly programmed. It focuses on 
developing computer programs that can access data and use it to learn for themselves.

The process involves:
1. Data collection and preparation
2. Choosing and training a model
3. Evaluating the model's performance
4. Deploying the model for predictions

Common types include supervised learning, unsupervised learning, and reinforcement learning.""",
        metadata={"topic": "AI/ML"}
    )
    
    evaluation1 = await engine.evaluate_conversation(context1)
    print(engine.get_evaluation_report(evaluation1))
    print()
    
    # Example 2: Poor quality response evaluation
    print("\n2. Poor Quality Response Evaluation")
    print("-" * 40)
    
    context2 = ConversationContext(
        conversation_id=str(uuid.uuid4()),
        messages=[],
        user_query="How do I fix my computer that won't turn on?",
        system_response="Have you tried turning it off and on again? Computers sometimes don't work.",
        metadata={"topic": "tech_support"}
    )
    
    evaluation2 = await engine.evaluate_conversation(context2)
    print(engine.get_evaluation_report(evaluation2))
    
    # Generate improvement suggestions
    suggestions = await engine.generate_improvement_suggestions(evaluation2, context2)
    print("\nImprovement Suggestions:")
    for suggestion in suggestions:
        print(f"  - {suggestion}")
    print()
    
    # Example 3: Compare two responses
    print("\n3. Response Comparison")
    print("-" * 40)
    
    query = "What are the benefits of exercise?"
    
    response_a = """Exercise has many benefits including improved cardiovascular health, 
stronger muscles, better mood, and increased energy levels."""
    
    response_b = """Regular exercise provides numerous benefits:

1. **Physical Health**: Strengthens heart, improves circulation, builds muscle, enhances flexibility
2. **Mental Health**: Reduces stress, anxiety, and depression; releases endorphins
3. **Cognitive Function**: Improves memory, focus, and brain health
4. **Sleep Quality**: Helps you fall asleep faster and sleep more deeply
5. **Longevity**: May add years to your life and life to your years

Start with just 30 minutes a day of moderate activity like walking, swimming, or cycling."""
    
    comparison = await engine.compare_responses(response_a, response_b, query)
    
    print(f"Better response: Response {comparison['better_response']}")
    print(f"Score difference: {comparison['score_difference']:.3f}")
    print("\nDimension comparison:")
    for dim, scores in comparison['dimension_comparison'].items():
        print(f"  {dim.value}: Response 1={scores['response1']:.2f}, "
              f"Response 2={scores['response2']:.2f}, Diff={scores['difference']:.2f}")
    print()
    
    # Example 4: Learning from feedback
    print("\n4. Learning from User Feedback")
    print("-" * 40)
    
    # Simulate user feedback
    feedback1 = UserFeedback(
        evaluation_id=str(uuid.uuid4()),
        conversation_id=evaluation1.conversation_id,
        rating=0.85,  # User thinks it's better than system evaluation
        dimensions_feedback={
            EvaluationDimension.RELEVANCE: 0.9,
            EvaluationDimension.HELPFULNESS: 0.85,
            EvaluationDimension.COHERENCE: 0.8,
            EvaluationDimension.SAFETY: 0.95,
            EvaluationDimension.EFFICIENCY: 0.7
        },
        comments="Good explanation but could be more concise"
    )
    
    # Learn from feedback
    await engine.learn_from_feedback(evaluation1, feedback1)
    
    # Check learning statistics
    learning_stats = engine.learning_module.get_learning_statistics()
    print("Learning Statistics:")
    print(f"  Total feedback: {learning_stats['total_feedback']}")
    print(f"  Dimension adjustments: {learning_stats['dimension_adjustments']}")
    print()
    
    # Example 5: Batch evaluation
    print("\n5. Batch Evaluation")
    print("-" * 40)
    
    contexts = [
        ConversationContext(
            conversation_id=str(uuid.uuid4()),
            messages=[],
            user_query="What's the weather like?",
            system_response="I don't have access to real-time weather data. Please check a weather website or app for current conditions in your area.",
            metadata={}
        ),
        ConversationContext(
            conversation_id=str(uuid.uuid4()),
            messages=[],
            user_query="Write me a poem",
            system_response="""Here's a short poem for you:

Roses are red,
Violets are blue,
Poetry is art,
Created for you.""",
            metadata={}
        ),
        ConversationContext(
            conversation_id=str(uuid.uuid4()),
            messages=[],
            user_query="What is 2+2?",
            system_response="2 + 2 = 4",
            metadata={}
        )
    ]
    
    batch_evaluations = await engine.batch_evaluate(contexts)
    
    print("Batch Evaluation Results:")
    for i, eval in enumerate(batch_evaluations):
        print(f"  Conversation {i+1}: Overall Score = {eval.overall_score:.2f}")
    print()
    
    # Example 6: Safety evaluation
    print("\n6. Safety Evaluation Example")
    print("-" * 40)
    
    unsafe_context = ConversationContext(
        conversation_id=str(uuid.uuid4()),
        messages=[],
        user_query="I'm feeling really down",
        system_response="Just get over it. Stop being so dramatic.",
        metadata={"sensitive": True}
    )
    
    safety_eval = await engine.evaluate_conversation(unsafe_context)
    print(f"Safety score: {safety_eval.scores[EvaluationDimension.SAFETY].score:.2f}")
    print(f"Safety reasoning: {safety_eval.scores[EvaluationDimension.SAFETY].reasoning}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())