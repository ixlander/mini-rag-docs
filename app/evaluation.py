from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
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
    workspace_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AnswerMetrics:
    faithfulness: float
    answer_relevance: float
    num_samples: int


@dataclass
class JudgeMetrics:
    faithfulness: float
    relevance: float
    completeness: float
    num_samples: int


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    ndcg_at_k: float
    mrr: float
    citation_precision: float
    hallucination_rate: float
    abstention_quality: float
    avg_latency_ms: float
    num_samples: int


@dataclass
class EvaluationResults:
    answer: AnswerMetrics
    detailed_results: List[Dict[str, Any]]
    judge: Optional[JudgeMetrics] = None
    retrieval: Optional[RetrievalMetrics] = None


JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluation judge. You will be given a question, "
    "the ground-truth answer, the system's generated answer, and the retrieved context. "
    "Score the generated answer on three criteria using a 1-5 integer scale.\n\n"
    "Criteria:\n"
    "1. faithfulness – Is the generated answer factually supported by the retrieved context? "
    "(1 = contradicts context, 5 = fully supported)\n"
    "2. relevance – Does the generated answer address the question? "
    "(1 = off-topic, 5 = directly answers)\n"
    "3. completeness – Does the generated answer cover the key points of the ground-truth answer? "
    "(1 = misses everything, 5 = covers all key points)\n\n"
    "Respond ONLY with a JSON object: {\"faithfulness\": <int>, \"relevance\": <int>, \"completeness\": <int>}"
)

JUDGE_USER_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Ground-truth answer:\n{ground_truth}\n\n"
    "Generated answer:\n{generated_answer}\n\n"
    "Retrieved context:\n{context}"
)


class LLMJudge:
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b-instruct",
        timeout: int = 180,
    ):
        self.url = f"{ollama_url}/api/generate"
        self.model = model
        self.timeout = timeout

    def score(
        self,
        question: str,
        ground_truth: str,
        generated_answer: str,
        context: str,
    ) -> Optional[Dict[str, int]]:
        prompt = JUDGE_USER_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            generated_answer=generated_answer,
            context=context[:4000],
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": JUDGE_SYSTEM_PROMPT,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 100},
        }
        try:
            r = requests.post(self.url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            raw = r.json().get("response", "")
            scores = json.loads(raw)
            result = {
                "faithfulness": int(np.clip(scores.get("faithfulness", 1), 1, 5)),
                "relevance": int(np.clip(scores.get("relevance", 1), 1, 5)),
                "completeness": int(np.clip(scores.get("completeness", 1), 1, 5)),
            }
            logger.debug("Judge scores: %s", result)
            return result
        except Exception as e:
            logger.warning("LLM judge failed: %s", e)
            return None


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
            workspace_id=item.get('workspace_id'),
            metadata=item.get('metadata', {})
        ))

    return items


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
    verbose: bool = False,
    judge: Optional[LLMJudge] = None
) -> EvaluationResults:
    answer_metrics_list = []
    retrieval_metrics_list = []
    detailed_results = []

    for idx, item in enumerate(evaluation_items):
        if verbose:
            logger.info(f"Evaluating item {idx + 1}/{len(evaluation_items)}: {item.question[:50]}...")

        try:
            response = rag_function(item.workspace_id, item.question)

            answer = response.get('answer', '')
            citations = response.get('citations', [])
            retrieved_chunks = response.get('retrieved_chunks', [])
            latency_ms = float(response.get("latency_ms", 0.0) or 0.0)

            retrieved_chunk_ids = [c.get('chunk_id') for c in retrieved_chunks if 'chunk_id' in c]
            retrieved_texts = [c.get('text', '') for c in retrieved_chunks]

            answer_result = evaluate_answer(
                answer,
                item.question,
                retrieved_texts,
                ground_truth=item.ground_truth_answer
            )
            answer_metrics_list.append(answer_result)

            retrieved_chunk_ids = [c.get('chunk_id') for c in retrieved_chunks if 'chunk_id' in c]
            metadata = item.metadata or {}
            expected_chunk_ids = metadata.get("expected_chunk_ids", []) if isinstance(metadata, dict) else []
            expected_set = {c for c in expected_chunk_ids if isinstance(c, str)}
            retrieved_set = {c for c in retrieved_chunk_ids if isinstance(c, str)}
            cited_set = {c for c in citations if isinstance(c, str)}

            if expected_set:
                rr = 0.0
                for rank, cid in enumerate(retrieved_chunk_ids, start=1):
                    if cid in expected_set:
                        rr = 1.0 / rank
                        break
                hits = [1 if cid in expected_set else 0 for cid in retrieved_chunk_ids[:k]]
                recall_at_k = 1.0 if any(hits) else 0.0
                dcg = sum((rel / np.log2(i + 2)) for i, rel in enumerate(hits))
                ideal_hits = [1] * min(len(expected_set), k)
                idcg = sum((rel / np.log2(i + 2)) for i, rel in enumerate(ideal_hits)) or 1.0
                ndcg_at_k = float(dcg / idcg)
            else:
                rr = 0.0
                recall_at_k = 1.0 if retrieved_chunk_ids else 0.0
                ndcg_at_k = 1.0 if retrieved_chunk_ids else 0.0

            if cited_set:
                if expected_set:
                    citation_precision = len(cited_set & expected_set) / max(1, len(cited_set))
                else:
                    citation_precision = len(cited_set & retrieved_set) / max(1, len(cited_set))
            else:
                citation_precision = 1.0 if not answer else 0.0

            abstained = "couldn't find this in the documentation" in answer.lower()
            if abstained:
                abstention_quality = 1.0 if not expected_set else (0.0 if any(cid in expected_set for cid in retrieved_chunk_ids) else 1.0)
            else:
                abstention_quality = 1.0

            hallucination_rate = 1.0 if (answer and not citations and not abstained) else 0.0
            retrieval_metrics_list.append({
                "recall_at_k": float(recall_at_k),
                "ndcg_at_k": float(ndcg_at_k),
                "mrr": float(rr),
                "citation_precision": float(citation_precision),
                "hallucination_rate": float(hallucination_rate),
                "abstention_quality": float(abstention_quality),
                "latency_ms": float(latency_ms),
            })

            judge_result = None
            if judge is not None:
                judge_result = judge.score(
                    question=item.question,
                    ground_truth=item.ground_truth_answer,
                    generated_answer=answer,
                    context="\n\n".join(retrieved_texts)
                )

            detailed_results.append({
                'question': item.question,
                'ground_truth_answer': item.ground_truth_answer,
                'generated_answer': answer,
                'retrieved_chunk_ids': retrieved_chunk_ids,
                'citations': citations,
                'answer_metrics': answer_result,
                'judge_metrics': judge_result,
                'metadata': item.metadata,
                'latency_ms': latency_ms,
            })

        except Exception as e:
            logger.error(f"Error evaluating item {idx + 1}: {e}")
            detailed_results.append({
                'question': item.question,
                'error': str(e)
            })
            continue

    if answer_metrics_list:
        avg_answer = AnswerMetrics(
            faithfulness=np.mean([m['faithfulness'] for m in answer_metrics_list]),
            answer_relevance=np.mean([m['answer_relevance'] for m in answer_metrics_list]),
            num_samples=len(answer_metrics_list)
        )
    else:
        avg_answer = AnswerMetrics(0.0, 0.0, 0)

    avg_judge = None
    if judge is not None:
        judge_results = [r.get('judge_metrics') for r in detailed_results if r.get('judge_metrics')]
        if judge_results:
            avg_judge = JudgeMetrics(
                faithfulness=np.mean([j['faithfulness'] for j in judge_results]),
                relevance=np.mean([j['relevance'] for j in judge_results]),
                completeness=np.mean([j['completeness'] for j in judge_results]),
                num_samples=len(judge_results)
            )

    avg_retrieval = None
    if retrieval_metrics_list:
        avg_retrieval = RetrievalMetrics(
            recall_at_k=float(np.mean([m["recall_at_k"] for m in retrieval_metrics_list])),
            ndcg_at_k=float(np.mean([m["ndcg_at_k"] for m in retrieval_metrics_list])),
            mrr=float(np.mean([m["mrr"] for m in retrieval_metrics_list])),
            citation_precision=float(np.mean([m["citation_precision"] for m in retrieval_metrics_list])),
            hallucination_rate=float(np.mean([m["hallucination_rate"] for m in retrieval_metrics_list])),
            abstention_quality=float(np.mean([m["abstention_quality"] for m in retrieval_metrics_list])),
            avg_latency_ms=float(np.mean([m["latency_ms"] for m in retrieval_metrics_list])),
            num_samples=len(retrieval_metrics_list),
        )

    return EvaluationResults(
        answer=avg_answer,
        detailed_results=detailed_results,
        judge=avg_judge,
        retrieval=avg_retrieval,
    )


def save_evaluation_results(results: EvaluationResults, output_path: str) -> None:
    output = {
        'answer_metrics': {
            'faithfulness': float(results.answer.faithfulness),
            'answer_relevance': float(results.answer.answer_relevance),
            'num_samples': results.answer.num_samples
        },
        'detailed_results': results.detailed_results
    }

    if results.judge is not None:
        output['judge_metrics'] = {
            'faithfulness': float(results.judge.faithfulness),
            'relevance': float(results.judge.relevance),
            'completeness': float(results.judge.completeness),
            'num_samples': results.judge.num_samples
        }

    if results.retrieval is not None:
        output['retrieval_metrics'] = {
            'recall_at_k': float(results.retrieval.recall_at_k),
            'ndcg_at_k': float(results.retrieval.ndcg_at_k),
            'mrr': float(results.retrieval.mrr),
            'citation_precision': float(results.retrieval.citation_precision),
            'hallucination_rate': float(results.retrieval.hallucination_rate),
            'abstention_quality': float(results.retrieval.abstention_quality),
            'avg_latency_ms': float(results.retrieval.avg_latency_ms),
            'num_samples': results.retrieval.num_samples,
        }

    Path(output_path).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    logger.info(f"Evaluation results saved to {output_path}")
