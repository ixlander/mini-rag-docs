from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        model_name = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Evaluation: loading embedder %s on %s", model_name, device)
        _embedder = SentenceTransformer(model_name, device=device)
    return _embedder


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return dot / norm


@dataclass
class EvaluationItem:
    question: str
    ground_truth_answer: str
    relevant_chunk_ids: List[str]
    workspace_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    num_samples: int


@dataclass
class AnswerMetrics:
    faithfulness: float
    answer_relevance: float
    num_samples: int


@dataclass
class EvaluationResults:
    retrieval: RetrievalMetrics
    answer: AnswerMetrics
    detailed_results: List[Dict[str, Any]]


def load_evaluation_dataset(filepath: str) -> List[EvaluationItem]:
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
    if not retrieved_ids or k == 0:
        return 0.0
    
    retrieved_at_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for chunk_id in retrieved_at_k if chunk_id in relevant_ids)
    return relevant_retrieved / min(k, len(retrieved_at_k))


def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    
    retrieved_at_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for chunk_id in retrieved_at_k if chunk_id in relevant_ids)
    return relevant_retrieved / len(relevant_ids)


def calculate_mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids or not retrieved_ids:
        return 0.0
    
    retrieved_at_k = retrieved_ids[:k]
    
    dcg = sum(
        (1.0 if chunk_id in relevant_ids else 0.0) / np.log2(idx + 2)
        for idx, chunk_id in enumerate(retrieved_at_k)
    )
    
    ideal_length = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_length))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_faithfulness(answer: str, retrieved_contexts: List[str]) -> float:
    if not answer or not retrieved_contexts:
        return 0.0

    context_text = " ".join(c.strip() for c in retrieved_contexts if c.strip())
    if not context_text:
        return 0.0

    embedder = _get_embedder()
    embs = embedder.encode([answer, context_text], convert_to_numpy=True)
    return float(max(0.0, _cosine_sim(embs[0], embs[1])))


def calculate_answer_relevance(answer: str, question: str, ground_truth: str = "") -> float:
    if not answer or not question:
        return 0.0

    texts = [answer, question]
    if ground_truth:
        texts.append(ground_truth)

    embedder = _get_embedder()
    embs = embedder.encode(texts, convert_to_numpy=True)

    sim_q = max(0.0, _cosine_sim(embs[0], embs[1]))

    if ground_truth:
        sim_gt = max(0.0, _cosine_sim(embs[0], embs[2]))
        return float(0.4 * sim_q + 0.6 * sim_gt)

    return float(sim_q)


def evaluate_retrieval(
    retrieved_chunk_ids: List[str],
    relevant_chunk_ids: List[str],
    k: int = 5
) -> Dict[str, float]:
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
    retrieved_contexts: List[str],
    ground_truth: str = ""
) -> Dict[str, float]:
    return {
        'faithfulness': calculate_faithfulness(answer, retrieved_contexts),
        'answer_relevance': calculate_answer_relevance(answer, question, ground_truth),
    }


def evaluate_rag_system(
    evaluation_items: List[EvaluationItem],
    rag_function: callable,
    k: int = 5,
    verbose: bool = False
) -> EvaluationResults:
    retrieval_metrics_list = []
    answer_metrics_list = []
    detailed_results = []
    
    for idx, item in enumerate(evaluation_items):
        if verbose:
            logger.info(f"Evaluating item {idx + 1}/{len(evaluation_items)}: {item.question[:50]}...")
        
        try:
            response = rag_function(item.workspace_id, item.question)
            
            answer = response.get('answer', '')
            citations = response.get('citations', [])
            retrieved_chunks = response.get('retrieved_chunks', [])
            
            retrieved_chunk_ids = [c.get('chunk_id') for c in retrieved_chunks if 'chunk_id' in c]
            retrieved_texts = [c.get('text', '') for c in retrieved_chunks]
            
            retrieval_result = evaluate_retrieval(
                retrieved_chunk_ids,
                item.relevant_chunk_ids,
                k=k
            )
            retrieval_metrics_list.append(retrieval_result)
            
            answer_result = evaluate_answer(
                answer,
                item.question,
                retrieved_texts,
                ground_truth=item.ground_truth_answer
            )
            answer_metrics_list.append(answer_result)
            
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
