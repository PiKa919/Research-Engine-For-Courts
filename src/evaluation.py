# src/evaluation.py

import os
from typing import List, Dict, Any
import logging

# Optional deepeval imports with fallbacks
try:
    from deepeval import evaluate
    from deepeval.metrics import (
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        AnswerRelevancyMetric,
        FaithfulnessMetric
    )
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    # Fallback classes for when deepeval is not available
    class MockMetric:
        def __init__(self, *args, **kwargs):
            pass
        
        def measure(self, test_case):
            return {"score": 0.5, "reason": "DeepEval not available - mock evaluation"}
    
    class MockTestCase:
        def __init__(self, input, actual_output, expected_output=None, retrieval_context=None):
            self.input = input
            self.actual_output = actual_output
            self.expected_output = expected_output
            self.retrieval_context = retrieval_context
    
    # Mock classes
    ContextualPrecisionMetric = MockMetric
    ContextualRecallMetric = MockMetric
    ContextualRelevancyMetric = MockMetric
    AnswerRelevancyMetric = MockMetric
    FaithfulnessMetric = MockMetric
    LLMTestCase = MockTestCase
    
    def evaluate(*args, **kwargs):
        return {"average_score": 0.5, "message": "DeepEval not available - using mock evaluation"}
    
    DEEPEVAL_AVAILABLE = False
    
from .retrieval import create_rag_chain

logger = logging.getLogger(__name__)

def create_test_cases(questions_and_answers: List[Dict[str, str]]) -> List[LLMTestCase]:
    """
    Create DeepEval test cases from questions and expected answers.
    
    Args:
        questions_and_answers: List of dicts with 'question' and 'expected_answer' keys
        
    Returns:
        List of LLMTestCase objects
    """
    if not DEEPEVAL_AVAILABLE:
        logger.warning("DeepEval not available - returning mock test cases")
        return [LLMTestCase(qa['question'], "Mock response", qa['expected_answer']) for qa in questions_and_answers]
    
    async_rag_chain, sync_rag_chain = create_rag_chain()
    test_cases = []
    
    for qa in questions_and_answers:
        question = qa['question']
        expected_answer = qa['expected_answer']
        
        # Get actual response from RAG chain
        try:
            response = sync_rag_chain(question)  # Use sync chain for evaluation
            actual_answer = response if isinstance(response, str) else response.get('answer', '')

            # Get context (retrieved documents)
            context = response.get('context', []) if isinstance(response, dict) else []
            test_case = LLMTestCase(
                input=question,
                actual_output=actual_answer,
                expected_output=expected_answer,
                retrieval_context=context
            )
            test_cases.append(test_case)
            
        except Exception as e:
            logger.error(f"Error creating test case for question '{question}': {e}")
            continue
    
    return test_cases

def evaluate_rag_system(test_cases: List[LLMTestCase]) -> Dict[str, Any]:
    """
    Evaluate the RAG system using DeepEval metrics.
    
    Args:
        test_cases: List of LLMTestCase objects
        
    Returns:
        Dictionary with evaluation results
    """
    if not test_cases:
        return {"error": "No test cases provided"}
    
    if not DEEPEVAL_AVAILABLE:
        logger.warning("DeepEval not available - returning mock evaluation results")
        return {
            "total_test_cases": len(test_cases),
            "metrics_results": {
                "ContextualPrecisionMetric": {"score": 0.5, "reason": "DeepEval not available"},
                "ContextualRecallMetric": {"score": 0.5, "reason": "DeepEval not available"},
                "ContextualRelevancyMetric": {"score": 0.5, "reason": "DeepEval not available"},
                "AnswerRelevancyMetric": {"score": 0.5, "reason": "DeepEval not available"},
                "FaithfulnessMetric": {"score": 0.5, "reason": "DeepEval not available"}
            },
            "message": "Mock evaluation - install deepeval for real metrics"
        }
    
    try:
        # Define metrics
        metrics = [
            ContextualPrecisionMetric(),
            ContextualRecallMetric(),
            ContextualRelevancyMetric(),
            AnswerRelevancyMetric(),
            FaithfulnessMetric()
        ]
        
        # Run evaluation
        evaluation_results = evaluate(
            test_cases=test_cases,
            metrics=metrics
        )
        
        # Process results
        results_summary = {
            "total_test_cases": len(test_cases),
            "metrics_results": {}
        }
        
        for metric in metrics:
            results_summary["metrics_results"][metric.__class__.__name__] = {
                "score": metric.score if hasattr(metric, 'score') else None,
                "reason": getattr(metric, 'reason', None)
            }
        
        return results_summary
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return {"error": str(e)}

def run_sample_evaluation():
    """
    Run a sample evaluation with predefined test cases.
    """
    # Sample test cases - replace with your actual test data
    sample_qa = [
        {
            "question": "What is the main purpose of the Commercial Courts Act, 2015?",
            "expected_answer": "The Commercial Courts Act, 2015 establishes commercial courts for speedy disposal of commercial disputes."
        },
        {
            "question": "What are the key provisions of the Code of Civil Procedure, 1908?",
            "expected_answer": "The Code of Civil Procedure, 1908 contains provisions for civil procedure in Indian courts."
        }
    ]
    
    test_cases = create_test_cases(sample_qa)
    results = evaluate_rag_system(test_cases)
    
    return results

if __name__ == "__main__":
    results = run_sample_evaluation()
    print("Evaluation Results:")
    print(results)