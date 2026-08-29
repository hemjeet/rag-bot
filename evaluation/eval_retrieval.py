"""Lightweight retrieval evaluation utilities (recall@k, MRR).

These helpers are intentionally DB/embedding agnostic: pass in the retrieved
document IDs and the relevant document IDs for each query.
"""

from typing import Dict, Iterable, List, Set


def recall_at_k(retrieved: Iterable[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant documents found within the top-k retrieved docs."""
    rel = set(relevant)
    if not rel:
        return 1.0
    top_k = list(retrieved)[:k]
    return len(rel.intersection(top_k)) / len(rel)


def reciprocal_rank(retrieved: Iterable[str], relevant: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant document, or 0.0 if none found."""
    rel = set(relevant)
    for idx, doc_id in enumerate(retrieved):
        if doc_id in rel:
            return 1.0 / (idx + 1)
    return 0.0


def mean_reciprocal_rank(
    results: Dict[str, List[str]], qrels: Dict[str, List[str]]
) -> float:
    """Mean reciprocal rank across all queries."""
    if not results:
        return 0.0
    total = sum(
        reciprocal_rank(results[q], qrels.get(q, [])) for q in results
    )
    return total / len(results)


def evaluate_retrieval(
    results: Dict[str, List[str]],
    qrels: Dict[str, List[str]],
    k: int = 5,
) -> Dict[str, float]:
    """Return recall@{k} and MRR for a set of query results."""
    recalls = [
        recall_at_k(results[q], qrels.get(q, []), k) for q in results
    ]
    return {
        f"recall@{k}": sum(recalls) / len(recalls) if recalls else 0.0,
        "mrr": mean_reciprocal_rank(results, qrels),
    }


if __name__ == "__main__":
    example_results = {
        "q1": ["d3", "d1", "d2"],
        "q2": ["d2", "d4", "d1"],
    }
    example_qrels = {
        "q1": ["d1", "d3"],
        "q2": ["d2"],
    }
    print(evaluate_retrieval(example_results, example_qrels, k=5))
