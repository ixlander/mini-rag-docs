from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationItem:
    """Single evaluation item with question, ground truth answer and relevant chunks."""
    question: str
    ground_truth_answer: str
    relevant_chunk_ids: List[str]
    workspace_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics."""
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    num_samples: int


@dataclass
class AnswerMetrics:
    """Answer quality evaluation metrics."""
    faithfulness: float
    answer_relevance: float
    num_samples: int


@dataclass
class EvaluationResults:
    """Complete evaluation results."""
    retrieval: RetrievalMetrics
    answer: AnswerMetrics
    detailed_results: List[Dict[str, Any]]


def load_evaluation_dataset(filepath: str) -> List[EvaluationItem]:
    """
    Load evaluation dataset from JSON file.
    
    Expected format:
    [
        {
            "question": "What is RAG?",
            "ground_truth_answer": "RAG stands for Retrieval-Augmented Generation...",
            "relevant_chunk_ids": ["doc1::chunk0001", "doc1::chunk0002"],
            "workspace_id": "optional_workspace_id",
            "metadata": {"category": "definition"}
        },
        ...
    ]
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {filepath}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = []
    for item in data:
        items.append(EvaluationItem(
            question=item['question'],
            ground_truth_answer=item['ground_truth_answer'],
            relevant_chunk_ids=item.get('relevant_chunk_ids', []),
            workspace_id=item.get('workspace_id'),
            metadata=item.get('metadata', {})
        ))
    
    return items


def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Calculate Precision@K for retrieval."""
    if not retrieved_ids or k == 0:
        return 0.0
    
    retrieved_at_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for chunk_id in retrieved_at_k if chunk_id in relevant_ids)
    return relevant_retrieved / min(k, len(retrieved_at_k))


def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Calculate Recall@K for retrieval."""
    if not relevant_ids:
        return 0.0
    
    retrieved_at_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for chunk_id in retrieved_at_k if chunk_id in relevant_ids)
    return relevant_retrieved / len(relevant_ids)


def calculate_mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Calculate Mean Reciprocal Rank for a single query."""
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K."""
    if not relevant_ids or not retrieved_ids:
        return 0.0
    
    retrieved_at_k = retrieved_ids[:k]
    
    # DCG: sum of rel_i / log2(i+1)
    dcg = sum(
        (1.0 if chunk_id in relevant_ids else 0.0) / np.log2(idx + 2)
        for idx, chunk_id in enumerate(retrieved_at_k)
    )
    
    # IDCG: DCG for perfect ranking
    ideal_length = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_length))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_faithfulness(answer: str, retrieved_contexts: List[str]) -> float:
    """
    Calculate faithfulness score (simple keyword-based approach).
    Measures if answer is grounded in retrieved contexts.
    
    Returns value between 0 and 1.
    """
    if not answer or not retrieved_contexts:
        return 0.0
    
    # Simple token overlap approach
    answer_tokens = set(answer.lower().split())
    if not answer_tokens:
        return 0.0
    
    context_text = " ".join(retrieved_contexts).lower()
    context_tokens = set(context_text.split())
    
    if not context_tokens:
        return 0.0
    
    overlap = answer_tokens.intersection(context_tokens)
    return len(overlap) / len(answer_tokens)


def calculate_answer_relevance(answer: str, question: str) -> float:
    """
    Calculate answer relevance to question (simple keyword-based approach).
    
    Returns value between 0 and 1.
    """
    if not answer or not question:
        return 0.0
    
    # Simple token overlap approach
    answer_tokens = set(answer.lower().split())
    question_tokens = set(question.lower().split())
    
    if not answer_tokens or not question_tokens:
        return 0.0
    
    overlap = answer_tokens.intersection(question_tokens)
    # Use harmonic mean of overlap ratios
    precision = len(overlap) / len(answer_tokens) if answer_tokens else 0
    recall = len(overlap) / len(question_tokens) if question_tokens else 0
    
    if precision + recall == 0:
        return 0.0
    
    # F1-like score
    return 2 * (precision * recall) / (precision + recall)


def evaluate_retrieval(
    retrieved_chunk_ids: List[str],
    relevant_chunk_ids: List[str],
    k: int = 5
) -> Dict[str, float]:
    """
    Evaluate retrieval performance for a single query.
    
    Args:
        retrieved_chunk_ids: List of retrieved chunk IDs (in ranked order)
        relevant_chunk_ids: List of ground truth relevant chunk IDs
        k: Number of top results to consider
    
    Returns:
        Dictionary with retrieval metrics
    """
    relevant_set = set(relevant_chunk_ids)
    
    return {
        'precision_at_k': calculate_precision_at_k(retrieved_chunk_ids, relevant_set, k),
        'recall_at_k': calculate_recall_at_k(retrieved_chunk_ids, relevant_set, k),
        'mrr': calculate_mrr(retrieved_chunk_ids, relevant_set),
        'ndcg_at_k': calculate_ndcg_at_k(retrieved_chunk_ids, relevant_set, k),
    }


def evaluate_answer(
    answer: str,
    question: str,
    retrieved_contexts: List[str]
) -> Dict[str, float]:
    """
    Evaluate answer quality.
    
    Args:
        answer: Generated answer
        question: Original question
        retrieved_contexts: List of retrieved context texts
    
    Returns:
        Dictionary with answer quality metrics
    """
    return {
        'faithfulness': calculate_faithfulness(answer, retrieved_contexts),
        'answer_relevance': calculate_answer_relevance(answer, question),
    }


def evaluate_rag_system(
    evaluation_items: List[EvaluationItem],
    rag_function: callable,
    k: int = 5,
    verbose: bool = False
) -> EvaluationResults:
    """
    Evaluate RAG system end-to-end.
    
    Args:
        evaluation_items: List of evaluation items with questions and ground truth
        rag_function: Function that takes (workspace_id, question) and returns
                      {'answer': str, 'citations': List[str], 'retrieved_chunks': List[Dict]}
        k: Number of top results to consider for retrieval metrics
        verbose: Whether to print detailed progress
    
    Returns:
        EvaluationResults object with aggregated metrics
    """
    retrieval_metrics_list = []
    answer_metrics_list = []
    detailed_results = []
    
    for idx, item in enumerate(evaluation_items):
        if verbose:
            logger.info(f"Evaluating item {idx + 1}/{len(evaluation_items)}: {item.question[:50]}...")
        
        try:
            # Get RAG response
            response = rag_function(item.workspace_id, item.question)
            
            answer = response.get('answer', '')
            citations = response.get('citations', [])
            retrieved_chunks = response.get('retrieved_chunks', [])
            
            # Extract chunk IDs and texts
            retrieved_chunk_ids = [c.get('chunk_id') for c in retrieved_chunks if 'chunk_id' in c]
            retrieved_texts = [c.get('text', '') for c in retrieved_chunks]
            
            # Evaluate retrieval
            retrieval_result = evaluate_retrieval(
                retrieved_chunk_ids,
                item.relevant_chunk_ids,
                k=k
            )
            retrieval_metrics_list.append(retrieval_result)
            
            # Evaluate answer
            answer_result = evaluate_answer(
                answer,
                item.question,
                retrieved_texts
            )
            answer_metrics_list.append(answer_result)
            
            # Store detailed result
            detailed_results.append({
                'question': item.question,
                'ground_truth_answer': item.ground_truth_answer,
                'generated_answer': answer,
                'relevant_chunk_ids': item.relevant_chunk_ids,
                'retrieved_chunk_ids': retrieved_chunk_ids,
                'citations': citations,
                'retrieval_metrics': retrieval_result,
                'answer_metrics': answer_result,
                'metadata': item.metadata
            })
            
        except Exception as e:
            logger.error(f"Error evaluating item {idx + 1}: {e}")
            detailed_results.append({
                'question': item.question,
                'error': str(e)
            })
            continue
    
    # Aggregate retrieval metrics
    if retrieval_metrics_list:
        avg_retrieval = RetrievalMetrics(
            precision_at_k=np.mean([m['precision_at_k'] for m in retrieval_metrics_list]),
            recall_at_k=np.mean([m['recall_at_k'] for m in retrieval_metrics_list]),
            mrr=np.mean([m['mrr'] for m in retrieval_metrics_list]),
            ndcg_at_k=np.mean([m['ndcg_at_k'] for m in retrieval_metrics_list]),
            num_samples=len(retrieval_metrics_list)
        )
    else:
        avg_retrieval = RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0)
    
    # Aggregate answer metrics
    if answer_metrics_list:
        avg_answer = AnswerMetrics(
            faithfulness=np.mean([m['faithfulness'] for m in answer_metrics_list]),
            answer_relevance=np.mean([m['answer_relevance'] for m in answer_metrics_list]),
            num_samples=len(answer_metrics_list)
        )
    else:
        avg_answer = AnswerMetrics(0.0, 0.0, 0)
    
    return EvaluationResults(
        retrieval=avg_retrieval,
        answer=avg_answer,
        detailed_results=detailed_results
    )


def save_evaluation_results(results: EvaluationResults, output_path: str) -> None:
    """Save evaluation results to JSON file."""
    output = {
        'retrieval_metrics': {
            'precision_at_k': float(results.retrieval.precision_at_k),
            'recall_at_k': float(results.retrieval.recall_at_k),
            'mrr': float(results.retrieval.mrr),
            'ndcg_at_k': float(results.retrieval.ndcg_at_k),
            'num_samples': results.retrieval.num_samples
        },
        'answer_metrics': {
            'faithfulness': float(results.answer.faithfulness),
            'answer_relevance': float(results.answer.answer_relevance),
            'num_samples': results.answer.num_samples
        },
        'detailed_results': results.detailed_results
    }
    
    Path(output_path).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    logger.info(f"Evaluation results saved to {output_path}")
