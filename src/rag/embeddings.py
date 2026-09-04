import logging
import time
from datetime import timedelta
from typing import List, Optional
from openai import AsyncOpenAI
from aiobreaker import CircuitBreaker
from tenacity import retry, stop_after_attempt, wait_exponential
from src.config import settings

logger = logging.getLogger(__name__)

# Embedding API circuit breaker (separate from DB and LLM breakers)
embed_breaker = CircuitBreaker(
    fail_max=settings.embed_cb_failures,
    timeout_duration=timedelta(seconds=settings.embed_cb_timeout),
)

# OpenAI embedding API limit: max 2048 texts per request
_BATCH_SIZE = 2048


class Embeddings:
    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
        model: Optional[str] = None,
    ):
        self.client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.embedding_model

    @embed_breaker
    @retry(
        stop=stop_after_attempt(settings.embed_max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using OpenAI, with automatic batching."""
        if not texts:
            return []

        start = time.perf_counter()
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    timeout=60.0,
                )
                all_embeddings.extend([item.embedding for item in response.data])
            except Exception:
                logger.exception(
                    "[EMBED] failed to embed batch %d-%d", i, i + len(batch)
                )
                raise

        logger.info(
            "[EMBED] embedded %d texts with '%s' in %.2fs",
            len(all_embeddings),
            self.model,
            time.perf_counter() - start,
        )
        return all_embeddings

    @embed_breaker
    @retry(
        stop=stop_after_attempt(settings.embed_max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        start = time.perf_counter()
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                timeout=30.0,
            )
            logger.debug(
                "[EMBED] query embedded with '%s' in %.2fs",
                self.model,
                time.perf_counter() - start,
            )
            return response.data[0].embedding
        except Exception:
            logger.exception(
                "[EMBED] query embedding failed (%.2fs)", time.perf_counter() - start
            )
            raise


_default_embeddings: Optional[Embeddings] = None


def _get_embeddings() -> Embeddings:
    global _default_embeddings
    if _default_embeddings is None:
        _default_embeddings = Embeddings()
    return _default_embeddings


async def embed_texts(texts: List[str]) -> List[List[float]]:
    return await _get_embeddings().embed_texts(texts)


async def embed_query(text: str) -> List[float]:
    return await _get_embeddings().embed_query(text)
