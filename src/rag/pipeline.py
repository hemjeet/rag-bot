import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from src.rag.dense import DenseSearcher
from src.rag.keyword import KeywordSearcher
from src.rag.fusion import ReciprocalRankFusion
from src.rag.generator import Generator
from src.rag.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


def _truncate(text: str, length: int = 80) -> str:
    return text[:length] + "..." if len(text) > length else text


class HybridPipeline:
    def __init__(
        self,
        dense_searcher: Optional[DenseSearcher] = None,
        keyword_searcher: Optional[KeywordSearcher] = None,
        fusion: Optional[ReciprocalRankFusion] = None,
        generator: Optional[Generator] = None,
        cache: Optional[SemanticCache] = None,
    ):
        self.dense_searcher = dense_searcher or DenseSearcher()
        self.keyword_searcher = keyword_searcher or KeywordSearcher()
        self.fusion = fusion or ReciprocalRankFusion()
        self.generator = generator or Generator()
        self.cache = cache or SemanticCache()
        # self.cache = cache or SemanticCache(ttl_hours=24)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _should_use_bm25(self, query: str) -> bool:
        """Decide if sparse/keyword search is needed for this query.

        Kept as a fallback; the pipeline now prefers the combined route_query().
        """
        return await self.generator.should_use_bm25(query)

    async def _retrieve_single(
        self,
        query: str,
        collection_name: str,
        requires_bm25: bool,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run dense (and optionally keyword) retrieval for a single query."""
        start = time.perf_counter()
        if requires_bm25:
            dense_results, keyword_results = await asyncio.gather(
                self.dense_searcher.search(query, collection_name=collection_name),
                self.keyword_searcher.search(query, collection_name=collection_name),
            )
        else:
            dense_results = await self.dense_searcher.search(
                query, collection_name=collection_name
            )
            keyword_results = []
        logger.debug(
            "[RETRIEVE-SINGLE] query=%r dense=%d keyword=%d (%.2fs)",
            _truncate(query, 60), len(dense_results), len(keyword_results),
            time.perf_counter() - start,
        )
        return dense_results, keyword_results

    async def _retrieve_multi_hop(
        self,
        sub_queries: List[str],
        collection_name: str,
        requires_bm25: bool,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run retrieval for each sub-query concurrently and merge results.

        Results are deduplicated by document ID so the same chunk is never
        counted twice across different sub-queries.
        """
        start = time.perf_counter()
        tasks = [
            self._retrieve_single(sq, collection_name, requires_bm25)
            for sq in sub_queries
        ]
        results = await asyncio.gather(*tasks)

        # Merge & deduplicate by doc ID (preserve first-seen order)
        seen_dense: set = set()
        seen_keyword: set = set()
        all_dense: List[Dict[str, Any]] = []
        all_keyword: List[Dict[str, Any]] = []

        total_dense_raw = 0
        total_keyword_raw = 0

        for dense_batch, keyword_batch in results:
            total_dense_raw += len(dense_batch)
            total_keyword_raw += len(keyword_batch)
            for doc in dense_batch:
                if doc["id"] not in seen_dense:
                    seen_dense.add(doc["id"])
                    all_dense.append(doc)
            for doc in keyword_batch:
                if doc["id"] not in seen_keyword:
                    seen_keyword.add(doc["id"])
                    all_keyword.append(doc)

        logger.info(
            "[RETRIEVE-MULTI-HOP] %d sub-queries → raw_dense=%d raw_keyword=%d "
            "→ deduped_dense=%d deduped_keyword=%d (%.2fs)",
            len(sub_queries), total_dense_raw, total_keyword_raw,
            len(all_dense), len(all_keyword), time.perf_counter() - start,
        )
        return all_dense, all_keyword

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        query: str,
        collection_name: str = "legal_documents",
        use_bm25: Optional[bool] = None,
        session_id: Optional[str] = None,
    ) -> str:
        start = time.perf_counter()
        logger.info(
            "[PIPELINE] query=%r collection='%s' session_id='%s' use_bm25=%s",
            _truncate(query), collection_name, session_id, use_bm25,
        )

        # Step 0: Semantic cache check
        cached, query_emb = await self.cache.get(query, collection_name=collection_name)
        if cached is not None:
            answer, similarity = cached
            logger.info(
                "[PIPELINE] cache hit similarity=%.4f answer_len=%d (%.2fs)",
                similarity, len(answer), time.perf_counter() - start,
            )
            return answer

        # Step 1: Combined routing (BM25 + multi-hop in one LLM call)
        t0 = time.perf_counter()
        if use_bm25 is not None:
            routing = {"use_bm25": use_bm25, "is_multi_hop": False}
            # Still need multi-hop decision when bm25 is forced
            routing = await self.generator.route_query(query)
            routing["use_bm25"] = use_bm25
        else:
            routing = await self.generator.route_query(query)
        requires_bm25 = routing["use_bm25"]
        logger.info(
            "[ROUTER] bm25=%s multi_hop=%s (decided in %.2fs)",
            "YES" if requires_bm25 else "NO",
            "YES" if routing["is_multi_hop"] else "NO",
            time.perf_counter() - t0,
        )

        # Step 2: Decomposition (only calls LLM if multi-hop)
        t_decomp = time.perf_counter()
        sub_queries = await self.generator._decompose_query(query, is_multi_hop=routing["is_multi_hop"])
        is_multi_hop = len(sub_queries) > 1
        logger.info(
            "[DECOMPOSE] multi_hop=%s sub_queries=%d %r (in %.2fs)",
            is_multi_hop, len(sub_queries),
            [_truncate(sq, 60) for sq in sub_queries],
            time.perf_counter() - t_decomp,
        )

        # Step 3: Retrieval (single or multi-hop)
        t1 = time.perf_counter()
        if is_multi_hop:
            dense_results, keyword_results = await self._retrieve_multi_hop(
                sub_queries, collection_name, requires_bm25,
            )
        else:
            dense_results, keyword_results = await self._retrieve_single(
                query, collection_name, requires_bm25,
            )
        logger.info(
            "[RETRIEVAL] dense=%d keyword=%d (in %.2fs)",
            len(dense_results), len(keyword_results), time.perf_counter() - t1,
        )

        # Step 4: Fusion
        t2 = time.perf_counter()
        contexts = self.fusion.fuse(dense_results, keyword_results)
        logger.info(
            "[FUSION] %d+%d -> %d contexts (in %.2fs)",
            len(dense_results), len(keyword_results), len(contexts),
            time.perf_counter() - t2,
        )

        # Step 5: Generation (always uses the original query for coherent answer)
        t3 = time.perf_counter()
        answer = await self.generator.generate_answer(
            query=query,
            contexts=contexts,
            session_id=session_id,
        )
        logger.info(
            "[GENERATION] answer_len=%d (in %.2fs)", len(answer), time.perf_counter() - t3,
        )

        # Step 6: Store in semantic cache (reuse embedding from Step 0)
        await self.cache.store(query, answer, collection_name=collection_name, query_embedding=query_emb)

        total = time.perf_counter() - start
        logger.info(
            "[PIPELINE] completed in %.2fs (multi_hop=%s, sub_queries=%d, "
            "dense=%d, keyword=%d, contexts=%d, answer=%d chars)",
            total, is_multi_hop, len(sub_queries),
            len(dense_results), len(keyword_results), len(contexts), len(answer),
        )
        return answer

    async def run_stream(
        self,
        query: str,
        collection_name: str = "legal_documents",
        use_bm25: Optional[bool] = None,
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        start = time.perf_counter()
        logger.info(
            "[PIPELINE-STREAM] query=%r collection='%s' session_id='%s' use_bm25=%s",
            _truncate(query), collection_name, session_id, use_bm25,
        )

        # Step 0: Semantic cache check
        cached, query_emb = await self.cache.get(query, collection_name=collection_name)
        if cached is not None:
            answer, similarity = cached
            logger.info(
                "[PIPELINE-STREAM] cache hit similarity=%.4f answer_len=%d (%.2fs)",
                similarity, len(answer), time.perf_counter() - start,
            )
            # Simulate streaming for consistent UX (~10ms per word)
            words = answer.split(" ")
            for i, word in enumerate(words):
                separator = " " if i < len(words) - 1 else ""
                yield word + separator
                await asyncio.sleep(0.01)
            return

        # Step 1: Combined routing (BM25 + multi-hop in one LLM call)
        t0 = time.perf_counter()
        if use_bm25 is not None:
            routing = await self.generator.route_query(query)
            routing["use_bm25"] = use_bm25
        else:
            routing = await self.generator.route_query(query)
        requires_bm25 = routing["use_bm25"]
        logger.info(
            "[ROUTER] bm25=%s multi_hop=%s (decided in %.2fs)",
            "YES" if requires_bm25 else "NO",
            "YES" if routing["is_multi_hop"] else "NO",
            time.perf_counter() - t0,
        )

        # Step 2: Decomposition (only calls LLM if multi-hop)
        t_decomp = time.perf_counter()
        sub_queries = await self.generator._decompose_query(query, is_multi_hop=routing["is_multi_hop"])
        is_multi_hop = len(sub_queries) > 1
        logger.info(
            "[DECOMPOSE] multi_hop=%s sub_queries=%d %r (in %.2fs)",
            is_multi_hop, len(sub_queries),
            [_truncate(sq, 60) for sq in sub_queries],
            time.perf_counter() - t_decomp,
        )

        # Step 3: Retrieval (single or multi-hop)
        t1 = time.perf_counter()
        if is_multi_hop:
            dense_results, keyword_results = await self._retrieve_multi_hop(
                sub_queries, collection_name, requires_bm25,
            )
        else:
            dense_results, keyword_results = await self._retrieve_single(
                query, collection_name, requires_bm25,
            )
        logger.info(
            "[RETRIEVAL] dense=%d keyword=%d (in %.2fs)",
            len(dense_results), len(keyword_results), time.perf_counter() - t1,
        )

        # Step 4: Fusion
        t2 = time.perf_counter()
        contexts = self.fusion.fuse(dense_results, keyword_results)
        logger.info(
            "[FUSION] %d+%d -> %d contexts (in %.2fs)",
            len(dense_results), len(keyword_results), len(contexts),
            time.perf_counter() - t2,
        )

        # Step 5: Streaming generation
        t3 = time.perf_counter()
        chunk_count = 0
        full_chunks: List[str] = []
        async for chunk in self.generator.generate_answer_stream(
            query=query,
            contexts=contexts,
            session_id=session_id,
        ):
            chunk_count += 1
            full_chunks.append(chunk)
            yield chunk
        logger.info(
            "[GENERATION-STREAM] %d chunks (in %.2fs)", chunk_count, time.perf_counter() - t3,
        )

        # Store the complete answer in semantic cache (reuse embedding from Step 0)
        complete_answer = "".join(full_chunks)
        await self.cache.store(query, complete_answer, collection_name=collection_name, query_embedding=query_emb)

        total = time.perf_counter() - start
        logger.info(
            "[PIPELINE-STREAM] completed in %.2fs (multi_hop=%s, sub_queries=%d, "
            "dense=%d, keyword=%d, contexts=%d, chunks=%d)",
            total, is_multi_hop, len(sub_queries),
            len(dense_results), len(keyword_results), len(contexts), chunk_count,
        )
