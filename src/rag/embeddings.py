import logging
import time
from typing import List, Optional
from openai import AsyncOpenAI
from src.config import settings

logger = logging.getLogger(__name__)


class Embeddings:
    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
        model: Optional[str] = None,
    ):
        self.client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.embedding_model

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using OpenAI."""
        start = time.perf_counter()
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        logger.debug(
            "Embedded %d texts with model '%s' in %.2fs.",
            len(texts), self.model, time.perf_counter() - start,
        )
        return [item.embedding for item in response.data]

    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        start = time.perf_counter()
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        logger.debug(
            "Embedded query with model '%s' in %.2fs.",
            self.model, time.perf_counter() - start,
        )
        return response.data[0].embedding


_default_embeddings = Embeddings()


async def embed_texts(texts: List[str]) -> List[List[float]]:
    return await _default_embeddings.embed_texts(texts)


async def embed_query(text: str) -> List[float]:
    return await _default_embeddings.embed_query(text)