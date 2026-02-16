from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.evaluation import (
    EvaluationItem,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_faithfulness,
    calculate_answer_relevance,
    evaluate_retrieval,
    evaluate_answer,
    load_evaluation_dataset,
    save_evaluation_results,
    EvaluationResults,
    RetrievalMetrics,
    AnswerMetrics,
)


class TestPrecisionAtK:
    def test_perfect_precision(self):
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk1", "chunk2", "chunk3"}
        assert calculate_precision_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_precision(self):
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk1", "chunk3"}
        assert calculate_precision_at_k(retrieved, relevant, 3) == pytest.approx(2/3)

    def test_zero_precision(self):
        retrieved = ["chunk1", "chunk2"]
        relevant = {"chunk3", "chunk4"}
        assert calculate_precision_at_k(retrieved, relevant, 2) == 0.0

    def test_empty_retrieved(self):
        assert calculate_precision_at_k([], {"chunk1"}, 3) == 0.0

    def test_k_larger_than_retrieved(self):
        retrieved = ["chunk1", "chunk2"]
        relevant = {"chunk1"}
        assert calculate_precision_at_k(retrieved, relevant, 5) == 0.5


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk1", "chunk2"}
        assert calculate_recall_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_recall(self):
        retrieved = ["chunk1", "chunk2"]
        relevant = {"chunk1", "chunk2", "chunk3"}
        assert calculate_recall_at_k(retrieved, relevant, 2) == pytest.approx(2/3)

    def test_zero_recall(self):
        retrieved = ["chunk1", "chunk2"]
        relevant = {"chunk3", "chunk4"}
        assert calculate_recall_at_k(retrieved, relevant, 2) == 0.0

    def test_empty_relevant(self):
        assert calculate_recall_at_k(["chunk1"], set(), 1) == 0.0


class TestMRR:
    def test_first_position(self):
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk1"}
        assert calculate_mrr(retrieved, relevant) == 1.0

    def test_second_position(self):
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk2"}
        assert calculate_mrr(retrieved, relevant) == 0.5

    def test_third_position(self):
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk3"}
        assert calculate_mrr(retrieved, relevant) == pytest.approx(1/3)

    def test_no_relevant(self):
        retrieved = ["chunk1", "chunk2"]
        relevant = {"chunk3"}
        assert calculate_mrr(retrieved, relevant) == 0.0


class TestNDCGAtK:
    def test_perfect_ranking(self):
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = {"chunk1", "chunk2", "chunk3"}
        assert calculate_ndcg_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_ranking(self):
        retrieved = ["chunk1", "chunk3", "chunk2"]
        relevant = {"chunk1", "chunk2"}
        score = calculate_ndcg_at_k(retrieved, relevant, 3)
        assert 0.0 < score < 1.0

    def test_zero_ndcg(self):
        retrieved = ["chunk1", "chunk2"]
        relevant = {"chunk3"}
        assert calculate_ndcg_at_k(retrieved, relevant, 2) == 0.0

    def test_empty_retrieved(self):
        assert calculate_ndcg_at_k([], {"chunk1"}, 3) == 0.0


class TestFaithfulness:
    def test_perfect_faithfulness(self):
        answer = "The quick brown fox"
        contexts = ["The quick brown fox jumps over the lazy dog"]
        assert calculate_faithfulness(answer, contexts) == 1.0

    def test_partial_faithfulness(self):
        answer = "The quick brown fox"
        contexts = ["The quick brown"]
        score = calculate_faithfulness(answer, contexts)
        assert 0.0 < score < 1.0

    def test_zero_faithfulness(self):
        answer = "hello world"
        contexts = ["different text entirely"]
        score = calculate_faithfulness(answer, contexts)
        assert score >= 0.0

    def test_empty_answer(self):
        assert calculate_faithfulness("", ["some context"]) == 0.0

    def test_empty_contexts(self):
        assert calculate_faithfulness("answer", []) == 0.0


class TestAnswerRelevance:
    def test_perfect_relevance(self):
        answer = "The answer is yes"
        question = "Is the answer yes"
        score = calculate_answer_relevance(answer, question)
        assert score > 0.5

    def test_partial_relevance(self):
        answer = "The sky is blue"
        question = "What color is the sky"
        score = calculate_answer_relevance(answer, question)
        assert score > 0.0

    def test_zero_relevance(self):
        answer = "hello world"
        question = "something completely different"
        score = calculate_answer_relevance(answer, question)
        assert score >= 0.0

    def test_empty_answer(self):
        assert calculate_answer_relevance("", "question") == 0.0

    def test_empty_question(self):
        assert calculate_answer_relevance("answer", "") == 0.0


class TestEvaluateRetrieval:
    def test_complete_evaluation(self):
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = ["chunk1", "chunk3"]
        
        result = evaluate_retrieval(retrieved, relevant, k=3)
        
        assert 'precision_at_k' in result
        assert 'recall_at_k' in result
        assert 'mrr' in result
        assert 'ndcg_at_k' in result
        assert result['precision_at_k'] == pytest.approx(2/3)
        assert result['recall_at_k'] == 1.0


class TestEvaluateAnswer:
    def test_complete_evaluation(self):
        answer = "The answer is in the documentation"
        question = "Where is the answer"
        contexts = ["The answer is clearly stated in the documentation"]
        
        result = evaluate_answer(answer, question, contexts)
        
        assert 'faithfulness' in result
        assert 'answer_relevance' in result
        assert result['faithfulness'] > 0.5
        assert result['answer_relevance'] > 0.0


class TestLoadEvaluationDataset:
    def test_load_valid_dataset(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = [
                {
                    "question": "What is RAG?",
                    "ground_truth_answer": "RAG is Retrieval-Augmented Generation",
                    "relevant_chunk_ids": ["doc1::chunk0001"],
                    "workspace_id": "test_workspace",
                    "metadata": {"category": "definition"}
                }
            ]
            json.dump(data, f)
            filepath = f.name
        
        try:
            items = load_evaluation_dataset(filepath)
            assert len(items) == 1
            assert items[0].question == "What is RAG?"
            assert items[0].workspace_id == "test_workspace"
            assert len(items[0].relevant_chunk_ids) == 1
        finally:
            Path(filepath).unlink()

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_evaluation_dataset("/nonexistent/file.json")

    def test_load_minimal_dataset(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = [
                {
                    "question": "Test question?",
                    "ground_truth_answer": "Test answer"
                }
            ]
            json.dump(data, f)
            filepath = f.name
        
        try:
            items = load_evaluation_dataset(filepath)
            assert len(items) == 1
            assert items[0].question == "Test question?"
            assert items[0].relevant_chunk_ids == []
            assert items[0].workspace_id is None
        finally:
            Path(filepath).unlink()


class TestSaveEvaluationResults:
    def test_save_results(self):
        results = EvaluationResults(
            retrieval=RetrievalMetrics(
                precision_at_k=0.8,
                recall_at_k=0.9,
                mrr=0.85,
                ndcg_at_k=0.88,
                num_samples=10
            ),
            answer=AnswerMetrics(
                faithfulness=0.75,
                answer_relevance=0.80,
                num_samples=10
            ),
            detailed_results=[
                {
                    'question': 'Test?',
                    'generated_answer': 'Yes'
                }
            ]
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            save_evaluation_results(results, filepath)
            
            with open(filepath, 'r') as f:
                loaded = json.load(f)
            
            assert loaded['retrieval_metrics']['precision_at_k'] == 0.8
            assert loaded['answer_metrics']['faithfulness'] == 0.75
            assert len(loaded['detailed_results']) == 1
        finally:
            Path(filepath).unlink()


class TestEvaluationItem:
    def test_create_item(self):
        item = EvaluationItem(
            question="Test question",
            ground_truth_answer="Test answer",
            relevant_chunk_ids=["chunk1", "chunk2"]
        )
        assert item.question == "Test question"
        assert len(item.relevant_chunk_ids) == 2
        assert item.workspace_id is None
