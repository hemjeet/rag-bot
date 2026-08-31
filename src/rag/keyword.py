import logging
import time
from typing import List, Dict, Any, Optional
from src.db.session import get_pool
from src.config import settings

logger = logging.getLogger(__name__)


def _truncate(text: str, length: int = 60) -> str:
    return text[:length] + "..." if len(text) > length else text


class KeywordSearcher:
    def __init__(self, top_k: int = settings.sparse_top_k):
        self.top_k = top_k

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        collection_name: str = "legal_documents",
    ) -> List[Dict[str, Any]]:
        k = top_k if top_k is not None else self.top_k
        start = time.perf_counter()
        pool = get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT c.id, c.document, ts_rank(c.content_tsv, plainto_tsquery('english', %s)) AS rank
                    FROM chunks c
                    JOIN collections col ON c.collection_id = col.uuid
                    WHERE col.name = %s
                      AND c.content_tsv @@ plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    (query, collection_name, query, k),
                )
                rows = await cur.fetchall()

        results = [{"id": r[0], "text": r[1], "score": float(r[2])} for r in rows]
        top_scores = [f"{r['score']:.4f}" for r in results[:3]]

        logger.info(
            "[KEYWORD] query=%r collection='%s' top_k=%d results=%d "
            "top_scores=[%s] time=%.2fs",
            _truncate(query), collection_name, k, len(results),
            ", ".join(top_scores) if top_scores else "none",
            time.perf_counter() - start,
        )
        return results


_default_keyword_searcher = KeywordSearcher()


async def keyword_search(
    query: str,
    top_k: int = settings.sparse_top_k,
    collection_name: str = "legal_documents",
) -> List[Dict[str, Any]]:
    return await _default_keyword_searcher.search(
        query=query, top_k=top_k, collection_name=collection_name
    )
