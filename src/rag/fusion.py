import logging
from typing import List, Dict, Any, Optional
from src.config import settings

logger = logging.getLogger(__name__)


def _preview(text: str, length: int = 120) -> str:
    """Return a single-line preview of a chunk."""
    cleaned = " ".join(text.split())
    return cleaned[:length] + "..." if len(cleaned) > length else cleaned


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

        # Log summary
        logger.info(
            "[FUSION] dense=%d keyword=%d unique=%d -> top_n=%d output=%d rrf_k=%d",
            len(dense_results), len(keyword_results), len(rrf_scores), n,
            len(fused), self.k,
        )

        # Log each retrieved chunk with its ID, RRF score, and text preview
        for rank, doc_id in enumerate(sorted_ids[:n]):
            logger.info(
                "[CHUNK] rank=%d id='%s' rrf_score=%.6f preview=%r",
                rank + 1, doc_id, rrf_scores[doc_id],
                _preview(doc_map[doc_id]),
            )

        return fused


_default_fusion = ReciprocalRankFusion()


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    top_n: int = settings.hybrid_top_n,
) -> List[str]:
    return _default_fusion.fuse(dense_results, keyword_results, top_n=top_n)
