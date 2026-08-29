import logging
from typing import List, Dict, Any, Optional
from src.config import settings

logger = logging.getLogger(__name__)


class ReciprocalRankFusion:
    def __init__(
        self,
        k: int = settings.rrf_k,
        top_n: int = settings.hybrid_top_n,
    ):
        self.k = k
        self.top_n = top_n

    def fuse(
        self,
        dense_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[str]:
        n = top_n if top_n is not None else self.top_n
        rrf_scores = {}
        doc_map = {}

        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            score = 1.0 / (self.k + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score
            doc_map[doc_id] = doc["text"]

        for rank, doc in enumerate(keyword_results):
            doc_id = doc["id"]
            score = 1.0 / (self.k + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score
            doc_map[doc_id] = doc["text"]

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        fused = [doc_map[doc_id] for doc_id in sorted_ids[:n]]
        logger.debug(
            "Fused %d dense + %d keyword results into %d contexts.",
            len(dense_results), len(keyword_results), len(fused),
        )
        return fused


_default_fusion = ReciprocalRankFusion()


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    top_n: int = settings.hybrid_top_n,
) -> List[str]:
    return _default_fusion.fuse(dense_results, keyword_results, top_n=top_n)