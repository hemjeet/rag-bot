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

    async def warmup(self, collection_name: str = "legal_documents") -> None:
        """Prime every HTTP + DB connection with a synthetic query.

        Runs the same code paths a real user query would hit (embedding API,
        dense search, keyword search, semantic cache, router LLM, main LLM)
        in parallel so the first real user does not pay a cold-start penalty.

        Does NOT store anything in the semantic cache or memory manager.
        """
        _QUERY = "what is the contract about"
        t0 = time.perf_counter()
        logger.info("[WARMUP] starting connection warmup with %r ...", _QUERY)

        # 1. Embedding API — TLS + OpenAI connection
        try:
            await self.dense_searcher.embeddings.embed_query(_QUERY)
        except Exception:
            logger.warning("[WARMUP] embedding warmup failed (non-fatal)")

        # 2-5. Dense search, keyword search, cache GET, router LLM — all independent
        async def _dense():
            try:
                await self.dense_searcher.search(
                    _QUERY, collection_name=collection_name
                )
            except Exception:
                logger.warning("[WARMUP] dense search warmup failed (non-fatal)")

        async def _keyword():
            try:
                await self.keyword_searcher.search(
                    _QUERY, collection_name=collection_name
                )
            except Exception:
                logger.warning("[WARMUP] keyword search warmup failed (non-fatal)")

        async def _cache_get():
            try:
                await self.cache.get(_QUERY, collection_name=collection_name)
            except Exception:
                logger.warning("[WARMUP] semantic cache get warmup failed (non-fatal)")

        async def _router():
            try:
                await self.generator.route_query(_QUERY)
            except Exception:
                logger.warning("[WARMUP] router LLM warmup failed (non-fatal)")

        await asyncio.gather(
            _dense(),
            _keyword(),
            _cache_get(),
            _router(),
            return_exceptions=True,
        )

        # 6. Main LLM — minimal call to open TLS connection (1 token, no contexts)
        try:
            await self.generator.generate_answer(
                query=_QUERY,
                contexts=[],
                session_id=None,
            )
        except Exception:
            logger.warning("[WARMUP] main LLM warmup failed (non-fatal)")

        logger.info(
            "[WARMUP] all connections primed in %.2fs", time.perf_counter() - t0
        )

    async def _retrieve_single(
        self,
        query: str,
        collection_name: str,
        requires_bm25: bool,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        start = time.perf_counter()
        try:
            if requires_bm25:
                dense_results, keyword_results = await asyncio.gather(
                    self.dense_searcher.search(query, collection_name=collection_name),
                    self.keyword_searcher.search(
                        query, collection_name=collection_name
                    ),
                )
            else:
                dense_results = await self.dense_searcher.search(
                    query, collection_name=collection_name
                )
                keyword_results = []
        except Exception:
            logger.exception(
                "[RETRIEVE] search failed for query=%r", _truncate(query, 40)
            )
            dense_results, keyword_results = [], []
        logger.debug(
            "[RETRIEVE-SINGLE] query=%r dense=%d keyword=%d (%.2fs)",
            _truncate(query, 60),
            len(dense_results),
            len(keyword_results),
            time.perf_counter() - start,
        )
        return dense_results, keyword_results

    async def _retrieve_multi_hop(
        self,
        sub_queries: List[str],
        collection_name: str,
        requires_bm25: bool,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        start = time.perf_counter()
        tasks = [
            self._retrieve_single(sq, collection_name, requires_bm25)
            for sq in sub_queries
        ]
        results = await asyncio.gather(*tasks)

        seen_dense: set = set()
        seen_keyword: set = set()
        all_dense: List[Dict[str, Any]] = []
        all_keyword: List[Dict[str, Any]] = []

        for dense_batch, keyword_batch in results:
            for doc in dense_batch:
                if doc["id"] not in seen_dense:
                    seen_dense.add(doc["id"])
                    all_dense.append(doc)
            for doc in keyword_batch:
                if doc["id"] not in seen_keyword:
                    seen_keyword.add(doc["id"])
                    all_keyword.append(doc)

        logger.info(
            "[RETRIEVE-MULTI-HOP] %d sub-queries → dense=%d keyword=%d (%.2fs)",
            len(sub_queries),
            len(all_dense),
            len(all_keyword),
            time.perf_counter() - start,
        )
        return all_dense, all_keyword

    async def run(
        self,
        query: str,
        collection_name: str = "legal_documents",
        use_bm25: Optional[bool] = None,
        session_id: Optional[str] = None,
    ) -> str:
        start = time.perf_counter()
        logger.info(
            "[PIPELINE] query=%r collection='%s' session_id=%s bm25=%s",
            _truncate(query),
            collection_name,
            session_id,
            use_bm25,
        )

        # Step 0: Semantic cache check
        cached, query_emb = await self.cache.get(query, collection_name=collection_name)
        if cached is not None:
            answer, similarity = cached
            logger.info(
                "[PIPELINE] cache hit similarity=%.4f answer_len=%d (%.2fs)",
                similarity,
                len(answer),
                time.perf_counter() - start,
            )
            return answer

        # Step 1: Routing — skip LLM call when use_bm25 is explicitly provided
        t0 = time.perf_counter()
        if use_bm25 is not None:
            # Caller forced BM25; still need multi-hop decision
            try:
                routing = await self.generator.route_query(query)
            except Exception:
                logger.exception(
                    "[ROUTER] route_query failed, defaulting to single-hop"
                )
                routing = {"use_bm25": use_bm25, "is_multi_hop": False}
            routing["use_bm25"] = use_bm25
        else:
            try:
                routing = await self.generator.route_query(query)
            except Exception:
                logger.exception(
                    "[ROUTER] route_query failed, defaulting to single-hop, no bm25"
                )
                routing = {"use_bm25": False, "is_multi_hop": False}
        requires_bm25 = routing["use_bm25"]
        logger.info(
            "[ROUTER] bm25=%s multi_hop=%s (%.2fs)",
            "YES" if requires_bm25 else "NO",
            "YES" if routing["is_multi_hop"] else "NO",
            time.perf_counter() - t0,
        )

        # Step 2: Decomposition
        t_decomp = time.perf_counter()
        try:
            sub_queries = await self.generator.decompose_query(
                query, is_multi_hop=routing["is_multi_hop"]
            )
        except Exception:
            logger.exception("[DECOMPOSE] failed, using original query")
            sub_queries = [query]
        is_multi_hop = len(sub_queries) > 1
        logger.info(
            "[DECOMPOSE] multi_hop=%s sub_queries=%d (%.2fs)",
            is_multi_hop,
            len(sub_queries),
            time.perf_counter() - t_decomp,
        )

        # Step 3: Retrieval
        t1 = time.perf_counter()
        if is_multi_hop:
            dense_results, keyword_results = await self._retrieve_multi_hop(
                sub_queries,
                collection_name,
                requires_bm25,
            )
        else:
            dense_results, keyword_results = await self._retrieve_single(
                query,
                collection_name,
                requires_bm25,
            )
        logger.info(
            "[RETRIEVAL] dense=%d keyword=%d (%.2fs)",
            len(dense_results),
            len(keyword_results),
            time.perf_counter() - t1,
        )

        # Step 4: Fusion
        t2 = time.perf_counter()
        contexts = self.fusion.fuse(dense_results, keyword_results)
        logger.info(
            "[FUSION] %d+%d -> %d contexts (%.2fs)",
            len(dense_results),
            len(keyword_results),
            len(contexts),
            time.perf_counter() - t2,
        )

        # Step 5: Generation
        t3 = time.perf_counter()
        try:
            answer = await self.generator.generate_answer(
                query=query,
                contexts=contexts,
                session_id=session_id,
            )
        except Exception:
            logger.exception("[GENERATION] failed")
            answer = "I'm sorry, I encountered an error while generating an answer. Please try again."
        logger.info(
            "[GENERATION] answer_len=%d (%.2fs)",
            len(answer),
            time.perf_counter() - t3,
        )

        # Step 6: Cache store
        if answer and not answer.startswith("I'm sorry"):
            await self.cache.store(
                query,
                answer,
                collection_name=collection_name,
                query_embedding=query_emb,
            )

        total = time.perf_counter() - start
        logger.info(
            "[PIPELINE] completed in %.2fs (multi_hop=%s, dense=%d, keyword=%d, contexts=%d)",
            total,
            is_multi_hop,
            len(dense_results),
            len(keyword_results),
            len(contexts),
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
            "[PIPELINE-STREAM] query=%r collection='%s' session_id=%s bm25=%s",
            _truncate(query),
            collection_name,
            session_id,
            use_bm25,
        )

        # Step 0: Cache check
        cached, query_emb = await self.cache.get(query, collection_name=collection_name)
        if cached is not None:
            answer, similarity = cached
            logger.info(
                "[PIPELINE-STREAM] cache hit similarity=%.4f (%.2fs)",
                similarity,
                time.perf_counter() - start,
            )
            words = answer.split(" ")
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                await asyncio.sleep(0.01)
            return

        # Step 1: Routing
        t0 = time.perf_counter()
        if use_bm25 is not None:
            try:
                routing = await self.generator.route_query(query)
            except Exception:
                routing = {"use_bm25": use_bm25, "is_multi_hop": False}
            routing["use_bm25"] = use_bm25
        else:
            try:
                routing = await self.generator.route_query(query)
            except Exception:
                routing = {"use_bm25": False, "is_multi_hop": False}
        requires_bm25 = routing["use_bm25"]
        logger.info(
            "[ROUTER] bm25=%s multi_hop=%s (%.2fs)",
            "YES" if requires_bm25 else "NO",
            "YES" if routing["is_multi_hop"] else "NO",
            time.perf_counter() - t0,
        )

        # Step 2: Decomposition
        try:
            sub_queries = await self.generator.decompose_query(
                query, is_multi_hop=routing["is_multi_hop"]
            )
        except Exception:
            sub_queries = [query]
        is_multi_hop = len(sub_queries) > 1

        # Step 3: Retrieval
        t1 = time.perf_counter()
        if is_multi_hop:
            dense_results, keyword_results = await self._retrieve_multi_hop(
                sub_queries,
                collection_name,
                requires_bm25,
            )
        else:
            dense_results, keyword_results = await self._retrieve_single(
                query,
                collection_name,
                requires_bm25,
            )
        logger.info(
            "[RETRIEVAL] dense=%d keyword=%d (%.2fs)",
            len(dense_results),
            len(keyword_results),
            time.perf_counter() - t1,
        )

        # Step 4: Fusion
        t2 = time.perf_counter()
        contexts = self.fusion.fuse(dense_results, keyword_results)
        logger.info(
            "[FUSION] %d+%d -> %d contexts (%.2fs)",
            len(dense_results),
            len(keyword_results),
            len(contexts),
            time.perf_counter() - t2,
        )

        # Step 5: Streaming generation
        t3 = time.perf_counter()
        chunk_count = 0
        full_chunks: List[str] = []
        try:
            async for chunk in self.generator.generate_answer_stream(
                query=query,
                contexts=contexts,
                session_id=session_id,
            ):
                chunk_count += 1
                full_chunks.append(chunk)
                yield chunk
        except Exception:
            logger.exception("[GENERATION-STREAM] failed")
            yield "\n[Error: internal server error]"
        logger.info(
            "[GENERATION-STREAM] %d chunks (%.2fs)",
            chunk_count,
            time.perf_counter() - t3,
        )

        # Cache store
        complete_answer = "".join(full_chunks)
        if complete_answer:
            await self.cache.store(
                query,
                complete_answer,
                collection_name=collection_name,
                query_embedding=query_emb,
            )

        total = time.perf_counter() - start
        logger.info(
            "[PIPELINE-STREAM] completed in %.2fs (multi_hop=%s, dense=%d, keyword=%d, chunks=%d)",
            total,
            is_multi_hop,
            len(dense_results),
            len(keyword_results),
            chunk_count,
        )
