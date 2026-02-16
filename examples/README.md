# Evaluation Examples

This directory contains examples for evaluating the RAG system.

## Files

- **run_evaluation.py** — Main evaluation script
- **evaluation_dataset_example.json** — Sample evaluation dataset
- **evaluation_dataset_bcc.json** — Real evaluation dataset (24 questions, BCC bank documents)

## Quick Start

### 1. Prepare Your Workspace

Before running evaluation, ensure you have:
1. Created a workspace using the API
2. Uploaded documents to the workspace
3. Built the FAISS index for the workspace

```bash
# Example workflow
curl -X POST http://127.0.0.1:8000/workspaces
# Returns: {"workspace_id":"YOUR_WORKSPACE_ID"}

curl -X POST http://127.0.0.1:8000/upload/YOUR_WORKSPACE_ID -F "files=@document.pdf"
curl -X POST http://127.0.0.1:8000/build_index/YOUR_WORKSPACE_ID
```

### 2. Create Evaluation Dataset

Create a JSON file with evaluation questions and ground truth:

```json
[
  {
    "question": "What is RAG?",
    "ground_truth_answer": "RAG is Retrieval-Augmented Generation...",
    "workspace_id": "YOUR_WORKSPACE_ID",
    "metadata": {"category": "definition"}
  }
]
```

### 3. Run Evaluation

```bash
python examples/run_evaluation.py \
  --dataset your_dataset.json \
  --output results.json \
  --k 5 \
  --verbose
```

With LLM-as-judge scoring (requires Ollama):

```bash
python examples/run_evaluation.py \
  --dataset your_dataset.json \
  --output results.json \
  --k 5 \
  --judge \
  --judge-model qwen2.5:3b-instruct \
  --verbose
```

### 4. View Results

The script will print a summary and save detailed results to JSON:

```
EVALUATION RESULTS
============================================================

Answer Quality Metrics (Embedding):
  Faithfulness: 0.8340
  Answer Relevance: 0.9000
  Samples: 10

LLM-as-Judge Metrics (1-5 scale):
  Faithfulness: 4.20
  Relevance: 4.50
  Completeness: 3.80
  Samples: 10
============================================================
```

## Metrics Explained

### Answer Quality Metrics (Embedding-Based)

- **Faithfulness**: Cosine similarity between answer and retrieved context embeddings
  - Higher is better (0.0 to 1.0)
  - Measures if the answer is grounded in what was retrieved

- **Answer Relevance**: Weighted cosine similarity between answer, question, and ground truth
  - Higher is better (0.0 to 1.0)
  - 40% question similarity + 60% ground truth similarity

### LLM-as-Judge Metrics (optional, `--judge` flag)

A second LLM call scores each answer on a 1-5 integer scale:

- **Faithfulness**: Is the answer supported by the retrieved context?
- **Relevance**: Does the answer address the question?
- **Completeness**: Does the answer cover the key points of the ground truth?

## Notes

- The evaluation script requires that the workspace index is already built
- The LLM judge requires a running Ollama instance
- For production evaluation, create a comprehensive dataset covering various question types and difficulty levels
