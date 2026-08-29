import asyncio
import logging
import time
from typing import Optional, AsyncGenerator
from src.rag.dense import DenseSearcher
from src.rag.keyword import KeywordSearcher
from src.rag.fusion import ReciprocalRankFusion
from src.rag.generator import Generator

logger = logging.getLogger(__name__)


class HybridPipeline:
    def __init__(
        self,
        dense_searcher: Optional[DenseSearcher] = None,
        keyword_searcher: Optional[KeywordSearcher] = None,
        fusion: Optional[ReciprocalRankFusion] = None,
        generator: Optional[Generator] = None,
    ):
        self.dense_searcher = dense_searcher or DenseSearcher()
        self.keyword_searcher = keyword_searcher or KeywordSearcher()
        self.fusion = fusion or ReciprocalRankFusion()
        self.generator = generator or Generator()

    async def _should_use_bm25(self, query: str) -> bool:
        """Decide if sparse/keyword search is needed for this query."""
        return await self.generator.should_use_bm25(query)

    async def run(
        self,
        query: str,
        collection_name: str = "legal_documents",
        use_bm25: Optional[bool] = None,
        session_id: Optional[str] = None,
    ) -> str:
        start = time.perf_counter()
        logger.info(
            "Running pipeline for query on collection '%s' (session_id=%s, use_bm25=%s).",
            collection_name, session_id, use_bm25,
        )

        if use_bm25 is None:
            requires_bm25 = await self._should_use_bm25(query)
        else:
            requires_bm25 = use_bm25

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

        contexts = self.fusion.fuse(dense_results, keyword_results)
        answer = await self.generator.generate_answer(
            query=query,
            contexts=contexts,
            session_id=session_id,
        )
        logger.info(
            "Pipeline run completed in %.2fs (dense=%d, keyword=%d, contexts=%d).",
            time.perf_counter() - start, len(dense_results), len(keyword_results), len(contexts),
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
            "Running streaming pipeline for query on collection '%s' (session_id=%s, use_bm25=%s).",
            collection_name, session_id, use_bm25,
        )

        if use_bm25 is None:
            requires_bm25 = await self._should_use_bm25(query)
        else:
            requires_bm25 = use_bm25

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

        contexts = self.fusion.fuse(dense_results, keyword_results)
        async for chunk in self.generator.generate_answer_stream(
            query=query,
            contexts=contexts,
            session_id=session_id,
        ):
            yield chunk

        logger.info(
            "Streaming pipeline completed in %.2fs (dense=%d, keyword=%d, contexts=%d).",
            time.perf_counter() - start, len(dense_results), len(keyword_results), len(contexts),
        )


