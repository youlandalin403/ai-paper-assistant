"""Version 5.1: Evaluate evidence retrieval at k=1, 3, 5.

Run from the ai-paper-assistant project directory:
    python evaluate_retrieval.py

Requires: output.txt, chunk_text.py, eval_questions.py,
          numpy, sentence-transformers, scikit-learn.
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from chunk_text import chunk_text
from eval_questions import EVAL_QUESTIONS


def normalize(text: str) -> str:
    """Ignore case and repeated whitespace when matching an evidence phrase."""
    return " ".join(text.lower().split())


def main() -> None:
    full_text = Path("output.txt").read_text(encoding="utf-8")
    chunks = chunk_text(full_text, chunk_size=1000, overlap=200)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = model.encode(chunks)
    questions = [item["question"] for item in EVAL_QUESTIONS]
    query_embeddings = model.encode(questions)

    # Each row is one question; each column is one paper chunk.
    scores = cosine_similarity(query_embeddings, chunk_embeddings)
    print("chunk_embeddings:", chunk_embeddings.shape)
    print("query_embeddings:", query_embeddings.shape)
    print("similarity matrix:", scores.shape)

    ks = (1, 3, 5)
    hit_counts = {k: 0 for k in ks}

    for question_index, item in enumerate(EVAL_QUESTIONS):
        # Rank all chunks for this question by decreasing cosine similarity.
        ranked_indices = np.argsort(scores[question_index])[::-1]
        evidence = normalize(item["evidence_text"])

        # Automatically labeled by matching a short phrase in the source text.
        gold_indices = [
            index
            for index, chunk in enumerate(chunks)
            if evidence in normalize(chunk)
        ]
        first_rank = next(
            (rank for rank, index in enumerate(ranked_indices, 1) if index in gold_indices),
            None,
        )

        print(f"\n{item['id']}: {item['question']}")
        print("  Top-5 chunk indices:", ranked_indices[:5].tolist())
        print("  Evidence chunk indices:", gold_indices)
        print("  First evidence rank:", first_rank if first_rank else "NOT FOUND")

        for k in ks:
            hit = any(index in gold_indices for index in ranked_indices[:k])
            hit_counts[k] += int(hit)
            print(f"  Hit@{k}: {'YES' if hit else 'NO'}")

        if not gold_indices:
            print("  WARNING: No chunk contains the exact evidence phrase after whitespace normalization.")
            print("  Check PDF parsing, chunk boundaries, and the evidence_text label.")

    n = len(EVAL_QUESTIONS)
    print("\n===== SUMMARY =====")
    for k in ks:
        print(f"Hit@{k}: {hit_counts[k]}/{n} = {hit_counts[k] / n:.1%}")

    print("\nNOTE: This checks evidence-phrase retrieval, not whether the LLM's answer is correct.")


if __name__ == "__main__":
    main()
