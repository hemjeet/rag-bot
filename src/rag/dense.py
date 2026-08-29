import logging
import time
from typing import List, Dict, Any, Optional
from src.db.session import get_pool
from src.rag.embeddings import Embeddings
from src.config import settings

logger = logging.getLogger(__name__)


class DenseSearcher:
    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        top_k: int = settings.dense_top_k,
    ):
        self.embeddings = embeddings or Embeddings()
        self.top_k = top_k

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        collection_name: str = "legal_documents",
    ) -> List[Dict[str, Any]]:
        k = top_k if top_k is not None else self.top_k
        start = time.perf_counter()
        query_vec = await self.embeddings.embed_query(query)
        pool = get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT c.id, c.document, (c.embedding <=> %s::vector) AS distance
                    FROM chunks c
                    JOIN collections col ON c.collection_id = col.uuid
                    WHERE col.name = %s
                    ORDER BY distance ASC
                    LIMIT %s
                    """,
                    (str(query_vec), collection_name, k),
                )
                rows = await cur.fetchall()
        logger.debug(
            "Dense search for collection '%s' returned %d results in %.2fs.",
            collection_name, len(rows), time.perf_counter() - start,
        )
        return [{"id": r[0], "text": r[1], "score": float(r[2])} for r in rows]


_default_dense_searcher = DenseSearcher()


async def dense_search(
    query: str,
    top_k: int = settings.dense_top_k,
    collection_name: str = "legal_documents",
) -> List[Dict[str, Any]]:
    return await _default_dense_searcher.search(
        query=query, top_k=top_k, collection_name=collection_name
    )



    