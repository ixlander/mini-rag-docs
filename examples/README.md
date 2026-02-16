# Evaluation Examples

This directory contains examples for evaluating the RAG system.

## Files

- **run_evaluation.py** — Main evaluation script
- **evaluation_dataset_example.json** — Sample evaluation dataset

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
    "relevant_chunk_ids": ["doc1::chunk0001", "doc2::chunk0003"],
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

### 4. View Results

The script will print a summary and save detailed results to JSON:

```
EVALUATION RESULTS
============================================================

Retrieval Metrics:
  Precision@5: 0.7500
  Recall@5: 0.8500
  MRR: 0.9200
  NDCG@5: 0.8100
  Samples: 10

Answer Quality Metrics:
  Faithfulness: 0.7200
  Answer Relevance: 0.6800
  Samples: 10
============================================================
```

## Metrics Explained

### Retrieval Metrics

- **Precision@K**: Of the top K retrieved chunks, what fraction are relevant?
  - Higher is better (0.0 to 1.0)
  - Measures retrieval accuracy

- **Recall@K**: Of all relevant chunks, what fraction appear in the top K?
  - Higher is better (0.0 to 1.0)
  - Measures retrieval completeness

- **MRR (Mean Reciprocal Rank)**: How high does the first relevant chunk appear?
  - Higher is better (0.0 to 1.0)
  - 1.0 means the first result is always relevant

- **NDCG@K**: Considers both relevance and ranking quality
  - Higher is better (0.0 to 1.0)
  - Penalizes relevant chunks that appear lower in results

### Answer Quality Metrics

- **Faithfulness**: Is the answer grounded in the retrieved context?
  - Higher is better (0.0 to 1.0)
  - Measures if the answer uses information from retrieved documents

- **Answer Relevance**: Is the answer relevant to the question?
  - Higher is better (0.0 to 1.0)
  - Measures if the answer addresses what was asked

## Notes

- The evaluation script requires that the workspace index is already built
- Chunk IDs in the evaluation dataset should match those in your FAISS index
- The example dataset is for demonstration purposes only
- For production evaluation, create a comprehensive dataset covering various question types and difficulty levels
