from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.evaluation import (
    EvaluationItem,
    JudgeMetrics,
    LLMJudge,
    calculate_faithfulness,
    calculate_answer_relevance,
    evaluate_answer,
    load_evaluation_dataset,
    save_evaluation_results,
    EvaluationResults,
    AnswerMetrics,
)


class TestFaithfulness:
    def test_high_faithfulness(self):
        answer = "The quick brown fox jumps over the lazy dog"
        contexts = ["The quick brown fox jumps over the lazy dog"]
        score = calculate_faithfulness(answer, contexts)
        assert score > 0.8

    def test_partial_faithfulness(self):
        answer = "The economy grew by 5% in 2024"
        contexts = ["GDP growth reached 5% during the year 2024"]
        score = calculate_faithfulness(answer, contexts)
        assert 0.3 < score < 1.0

    def test_low_faithfulness(self):
        answer = "The weather is sunny today"
        contexts = ["Bank CenterCredit reported net profit of 202 billion tenge"]
        score = calculate_faithfulness(answer, contexts)
        assert score < 0.8

    def test_empty_answer(self):
        assert calculate_faithfulness("", ["some context"]) == 0.0

    def test_empty_contexts(self):
        assert calculate_faithfulness("answer", []) == 0.0


class TestAnswerRelevance:
    def test_high_relevance(self):
        answer = "The net profit was 202 billion tenge"
        question = "What was the net profit?"
        ground_truth = "Net profit exceeded 202 billion tenge"
        score = calculate_answer_relevance(answer, question, ground_truth)
        assert score > 0.6

    def test_partial_relevance(self):
        answer = "The bank has 4000 employees"
        question = "How many staff does the bank have?"
        score = calculate_answer_relevance(answer, question)
        assert score > 0.0

    def test_low_relevance(self):
        answer = "The weather is sunny today"
        question = "What is the bank's net profit?"
        score = calculate_answer_relevance(answer, question)
        assert score < 0.8

    def test_empty_answer(self):
        assert calculate_answer_relevance("", "question") == 0.0

    def test_empty_question(self):
        assert calculate_answer_relevance("answer", "") == 0.0


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

    def test_with_ground_truth(self):
        answer = "Net profit was 202 billion tenge"
        question = "What was the net profit?"
        contexts = ["The net profit exceeded 202 billion tenge in 2024"]
        ground_truth = "Net profit exceeded 202 billion tenge"

        result = evaluate_answer(answer, question, contexts, ground_truth=ground_truth)

        assert result['faithfulness'] > 0.5
        assert result['answer_relevance'] > 0.5


class TestLoadEvaluationDataset:
    def test_load_valid_dataset(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = [
                {
                    "question": "What is RAG?",
                    "ground_truth_answer": "RAG is Retrieval-Augmented Generation",
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
            assert items[0].workspace_id is None
        finally:
            Path(filepath).unlink()


class TestSaveEvaluationResults:
    def test_save_results(self):
        results = EvaluationResults(
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

            assert loaded['answer_metrics']['faithfulness'] == 0.75
            assert len(loaded['detailed_results']) == 1
        finally:
            Path(filepath).unlink()


class TestEvaluationItem:
    def test_create_item(self):
        item = EvaluationItem(
            question="Test question",
            ground_truth_answer="Test answer",
        )
        assert item.question == "Test question"
        assert item.workspace_id is None


class TestJudgeMetrics:
    def test_create_judge_metrics(self):
        jm = JudgeMetrics(faithfulness=4.0, relevance=5.0, completeness=3.5, num_samples=10)
        assert jm.faithfulness == 4.0
        assert jm.relevance == 5.0
        assert jm.completeness == 3.5
        assert jm.num_samples == 10

    def test_evaluation_results_without_judge(self):
        results = EvaluationResults(
            answer=AnswerMetrics(0.75, 0.80, 10),
            detailed_results=[]
        )
        assert results.judge is None

    def test_evaluation_results_with_judge(self):
        results = EvaluationResults(
            answer=AnswerMetrics(0.75, 0.80, 10),
            detailed_results=[],
            judge=JudgeMetrics(4.2, 4.5, 3.8, 10)
        )
        assert results.judge is not None
        assert results.judge.faithfulness == 4.2

    def test_save_results_with_judge(self):
        results = EvaluationResults(
            answer=AnswerMetrics(0.75, 0.80, 10),
            detailed_results=[],
            judge=JudgeMetrics(4.0, 4.5, 3.5, 10)
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        try:
            save_evaluation_results(results, filepath)
            with open(filepath, 'r') as f:
                loaded = json.load(f)
            assert 'judge_metrics' in loaded
            assert loaded['judge_metrics']['faithfulness'] == 4.0
            assert loaded['judge_metrics']['relevance'] == 4.5
            assert loaded['judge_metrics']['completeness'] == 3.5
        finally:
            Path(filepath).unlink()

    def test_llm_judge_init(self):
        judge = LLMJudge(ollama_url="http://example.com", model="test-model", timeout=60)
        assert judge.model == "test-model"
        assert judge.timeout == 60
        assert "example.com" in judge.url
