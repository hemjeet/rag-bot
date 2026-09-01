import logging
import time
from typing import List, Optional, Tuple
from src.db.session import get_pool
from src.rag.embeddings import embed_query

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    A semantic cache that stores and retrieves query-answer pairs based on
    cosine similarity of query embeddings.

    Attributes:
        threshold (float): Minimum similarity score to consider a cache hit.
        ttl_hours (float | None): Time-to-live for cached entries in hours.
                                   None means entries never expire.
    """

    def __init__(
        self,
        threshold: float = 0.95,
        ttl_hours: Optional[float] = None,
    ):
        self.threshold = threshold
        self.ttl_hours = ttl_hours

    async def get(
        self, query: str, collection_name: Optional[str] = None
    ) -> Tuple[Optional[Tuple[str, float]], List[float]]:
        """
        Retrieve a cached answer if a semantically similar query exists.

        Returns a tuple of (cache_result, query_embedding):
        - cache_result is (answer, similarity) if similarity >= threshold, else None.
        - query_embedding is always returned so callers can reuse it for store(),
          avoiding a redundant embedding API call on cache misses.
        """
        start = time.perf_counter()
        query_emb = await embed_query(query)
        emb_str = str(query_emb)

        pool = get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                # Build query with optional TTL and collection filters
                conditions: List[str] = []
                params: list = [emb_str]

                if self.ttl_hours is not None:
                    conditions.append(
                        "created_at >= now() - %s * interval '1 hour'"
                    )
                    params.append(self.ttl_hours)

                if collection_name is not None:
                    conditions.append("collection_name = %s")
                    params.append(collection_name)

                where_clause = ""
                if conditions:
                    where_clause = "WHERE " + " AND ".join(conditions)

                params.append(emb_str)  # for ORDER BY clause

                await cur.execute(
                    f"""
                    SELECT id, answer, hit_count,
                           1 - (query_embedding <=> %s::vector) AS similarity
                    FROM semantic_cache
                    {where_clause}
                    ORDER BY query_embedding <=> %s::vector
                    LIMIT 1
                    """,
                    tuple(params),
                )
                row = await cur.fetchone()
                if row and row[3] >= self.threshold:
                    cache_id, answer, prev_hits, similarity = row
                    await cur.execute(
                        "UPDATE semantic_cache SET hit_count = hit_count + 1 WHERE id = %s",
                        (cache_id,),
                    )
                    logger.info(
                        "[CACHE-HIT] similarity=%.4f hit_count=%d -> %d time=%.2fs",
                        similarity, prev_hits, prev_hits + 1, time.perf_counter() - start,
                    )
                    return (answer, similarity), query_emb

        logger.info("[CACHE-MISS] time=%.2fs", time.perf_counter() - start)
        return None, query_emb

    async def store(
        self,
        query: str,
        answer: str,
        collection_name: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> None:
        """
        Store a query-answer pair in the semantic cache.
        Skips if a semantically similar query already exists for this collection.

        If query_embedding is provided, it is reused instead of re-calling the
        embedding API, saving one round-trip per uncached query.
        """
        start = time.perf_counter()
        query_emb = query_embedding if query_embedding is not None else await embed_query(query)
        emb_str = str(query_emb)

        pool = get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                # Check for existing similar entry before inserting
                dup_conditions = ["query_embedding <=> %s::vector < %s"]
                dup_params: list = [emb_str, 1 - self.threshold]

                if collection_name is not None:
                    dup_conditions.append("collection_name = %s")
                    dup_params.append(collection_name)

                await cur.execute(
                    f"""
                    SELECT id FROM semantic_cache
                    WHERE {" AND ".join(dup_conditions)}
                    LIMIT 1
                    """,
                    tuple(dup_params),
                )
                if await cur.fetchone():
                    logger.info(
                        "[CACHE-SKIP] similar query already cached (%.2fs)",
                        time.perf_counter() - start,
                    )
                    return

                await cur.execute(
                    """
                    INSERT INTO semantic_cache
                        (query_text, query_embedding, answer, collection_name)
                    VALUES (%s, %s::vector, %s, %s)
                    """,
                    (query, emb_str, answer, collection_name),
                )
                logger.info(
                    "[CACHE-STORE] query=%r collection=%s answer_len=%d (%.2fs)",
                    query[:60], collection_name, len(answer),
                    time.perf_counter() - start,
                )

    async def flush(self, collection_name: str) -> int:
        """
        Remove all cached entries for a specific collection.
        Call this after re-ingesting or updating documents in a collection
        to prevent stale answers from being served.

        Returns the number of deleted rows.
        """
        pool = get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM semantic_cache WHERE collection_name = %s",
                    (collection_name,),
                )
                deleted = cur.rowcount
                logger.info(
                    "[CACHE-FLUSH] collection='%s' deleted=%d",
                    collection_name, deleted,
                )
                return deleted

    async def cleanup(self) -> int:
        """
        Remove expired entries based on the configured TTL.
        No-op if ttl_hours is None.

        Returns the number of deleted rows.
        """
        if self.ttl_hours is None:
            logger.debug("[CACHE-CLEANUP] no TTL configured, skipping")
            return 0
        pool = get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM semantic_cache WHERE created_at < now() - %s * interval '1 hour'",
                    (self.ttl_hours,),
                )
                deleted = cur.rowcount
                logger.info(
                    "[CACHE-CLEANUP] ttl_hours=%.1f deleted=%d",
                    self.ttl_hours, deleted,
                )
                return deleted
