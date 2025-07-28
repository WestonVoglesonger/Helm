# Intelligent Evaluation System

A comprehensive, self-improving evaluation system that assesses conversation quality across multiple dimensions using both heuristic methods and GPT-4o integration.

## 🚀 Features

### Multi-Dimensional Evaluation
- **Relevance**: Assesses how well responses address user queries
- **Coherence**: Evaluates logical flow and consistency
- **Helpfulness**: Measures actionability and usefulness
- **Safety**: Checks for harmful content and appropriate handling
- **Efficiency**: Analyzes conciseness and information density

### Intelligent Learning
- **Feedback Learning**: Improves evaluation criteria based on user feedback
- **Pattern Recognition**: Identifies trends and biases in evaluations
- **Weight Adjustment**: Dynamically adjusts dimension importance
- **Temporal Analysis**: Tracks improvement over time

### GPT-4o Integration
- **LLM-Based Assessment**: Uses GPT-4o for sophisticated quality evaluation
- **Hybrid Approach**: Combines heuristics with LLM insights
- **Improvement Suggestions**: Generates specific recommendations
- **Response Comparison**: Detailed analysis of multiple responses

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd evaluation_system

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Configuration

### Using without GPT-4o (Heuristics Only)
```python
from evaluation_system import IntelligentEvaluationEngine

engine = IntelligentEvaluationEngine(use_llm=False)
```

### Using with GPT-4o
```python
import os
from evaluation_system import IntelligentEvaluationEngine

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

engine = IntelligentEvaluationEngine(use_llm=True)
```

## 💻 Usage

### Basic Evaluation
```python
import asyncio
from evaluation_system import IntelligentEvaluationEngine, ConversationContext

async def evaluate_conversation():
    engine = IntelligentEvaluationEngine(use_llm=False)
    
    context = ConversationContext(
        conversation_id="123",
        messages=[],
        user_query="What is machine learning?",
        system_response="Machine learning is a subset of AI...",
        metadata={}
    )
    
    evaluation = await engine.evaluate_conversation(context)
    print(engine.get_evaluation_report(evaluation))

asyncio.run(evaluate_conversation())
```

### Learning from Feedback
```python
from evaluation_system import UserFeedback, EvaluationDimension

# Create user feedback
feedback = UserFeedback(
    evaluation_id="eval-123",
    conversation_id="123",
    rating=0.85,
    dimensions_feedback={
        EvaluationDimension.RELEVANCE: 0.9,
        EvaluationDimension.HELPFULNESS: 0.8
    },
    comments="Good but could be more concise"
)

# Learn from feedback
await engine.learn_from_feedback(evaluation, feedback)
```

### Comparing Responses
```python
comparison = await engine.compare_responses(
    response1="Brief answer",
    response2="Detailed answer with examples",
    context="How do I learn Python?"
)

print(f"Better response: {comparison['better_response']}")
```

## 📊 Success Metrics

The system is designed to achieve:
- ✅ **>90% correlation** with human evaluations
- ✅ **<5% variance** in similar responses
- ✅ **20% improvement** in evaluation accuracy over time
- ✅ **100% coverage** of conversation dimensions

## 🏗️ Architecture

```
evaluation_system/
├── core/
│   ├── models.py           # Data models
│   └── evaluation_engine.py # Main engine
├── evaluators/
│   ├── base.py            # Base evaluator class
│   ├── relevance.py       # Relevance evaluation
│   ├── coherence.py       # Coherence evaluation
│   ├── helpfulness.py     # Helpfulness evaluation
│   ├── safety.py          # Safety evaluation
│   ├── efficiency.py      # Efficiency evaluation
│   └── llm_evaluator.py   # GPT-4o integration
├── learning/
│   ├── evaluation_learner.py # Learning module
│   └── pattern_recognizer.py  # Pattern recognition
└── tests/
    └── test_evaluation_engine.py # Comprehensive tests
```

## 🧪 Testing

Run the test suite:
```bash
pytest evaluation_system/tests/ -v
```

Run the example demo:
```bash
python evaluation_system/example_usage.py
```

## 🔄 Continuous Improvement

The system continuously improves through:

1. **User Feedback Integration**: Learns from discrepancies between system and user evaluations
2. **Pattern Recognition**: Identifies systematic biases and adjusts accordingly
3. **Weight Optimization**: Dynamically adjusts dimension importance based on feedback
4. **Feature Learning**: Discovers which sub-features correlate with quality

## 📈 Monitoring

Track system performance:
```python
stats = engine.learning_module.get_learning_statistics()
print(f"Total feedback: {stats['total_feedback']}")
print(f"Recent trends: {stats['recent_trends']}")
print(f"Confidence metrics: {stats['confidence_metrics']}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.