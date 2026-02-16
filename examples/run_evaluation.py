#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluation import (
    load_evaluation_dataset,
    evaluate_rag_system,
    save_evaluation_results,
)
from app.rag_workspace import WorkspaceRAG, WorkspaceRAGConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def create_rag_function(rag_instance: WorkspaceRAG):
    def rag_function(workspace_id: str, question: str) -> dict:
        if not workspace_id:
            raise ValueError("workspace_id is required")
        
        artifacts_dir = f"artifacts/workspaces/{workspace_id}"
        
        artifacts_path = Path(artifacts_dir)
        if not artifacts_path.exists() or not (artifacts_path / "faiss.index").exists():
            raise FileNotFoundError(
                f"Index not found for workspace {workspace_id}. "
                f"Please build the index first using /build_index/{workspace_id}"
            )
        
        response = rag_instance.answer(
            artifacts_dir=artifacts_dir,
            question=question,
            debug=True
        )
        
        retrieved_chunks = []
        if 'debug' in response and 'context_preview' in response['debug']:
            for chunk_preview in response['debug']['context_preview']:
                retrieved_chunks.append({
                    'chunk_id': chunk_preview['chunk_id'],
                    'title': chunk_preview.get('title', ''),
                    'section': chunk_preview.get('section', ''),
                    'text': chunk_preview.get('text_preview', '')
                })
        
        return {
            'answer': response.get('answer', ''),
            'citations': response.get('citations', []),
            'retrieved_chunks': retrieved_chunks
        }
    
    return rag_function


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system performance")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to evaluation dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results (default: evaluation_results.json)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top results to consider for retrieval metrics (default: 5)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress during evaluation"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Loading evaluation dataset from {args.dataset}")
    try:
        evaluation_items = load_evaluation_dataset(args.dataset)
        logger.info(f"Loaded {len(evaluation_items)} evaluation items")
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return 1
    
    logger.info("Initializing RAG system...")
    try:
        rag_config = WorkspaceRAGConfig()
        rag = WorkspaceRAG(rag_config)
        rag_function = create_rag_function(rag)
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        return 1
    
    logger.info(f"Running evaluation on {len(evaluation_items)} items...")
    try:
        results = evaluate_rag_system(
            evaluation_items=evaluation_items,
            rag_function=rag_function,
            k=args.k,
            verbose=args.verbose
        )
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return 1
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    logger.info("\nRetrieval Metrics:")
    logger.info(f"  Precision@{args.k}: {results.retrieval.precision_at_k:.4f}")
    logger.info(f"  Recall@{args.k}: {results.retrieval.recall_at_k:.4f}")
    logger.info(f"  MRR: {results.retrieval.mrr:.4f}")
    logger.info(f"  NDCG@{args.k}: {results.retrieval.ndcg_at_k:.4f}")
    logger.info(f"  Samples: {results.retrieval.num_samples}")
    
    logger.info("\nAnswer Quality Metrics:")
    logger.info(f"  Faithfulness: {results.answer.faithfulness:.4f}")
    logger.info(f"  Answer Relevance: {results.answer.answer_relevance:.4f}")
    logger.info(f"  Samples: {results.answer.num_samples}")
    logger.info("="*60 + "\n")
    
    logger.info(f"Saving detailed results to {args.output}")
    try:
        save_evaluation_results(results, args.output)
        logger.info("Evaluation complete!")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
