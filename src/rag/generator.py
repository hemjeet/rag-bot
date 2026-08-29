import logging
import time
from typing import List, Optional, Dict, AsyncGenerator
from openai import AsyncOpenAI
from src.config import settings
from src.rag.memory import MemoryManager
from src.rag.prompts import build_bm25_router_prompt, build_system_prompt

logger = logging.getLogger(__name__)


class Generator:
    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        router_model: Optional[str] = None,
        max_history_turns: int = 5,
    ):
        api_key = settings.deepseek_api_key or settings.openai_api_key
        use_deepseek = bool(settings.deepseek_api_key)
        base_url = settings.deepseek_base_url if use_deepseek else None
        self.client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        # Default model names must match the provider actually being used.
        default_model = "deepseek-chat" if use_deepseek else "gpt-4o-mini"
        self.model = model or settings.llm_model or default_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.router_model = router_model or settings.router_model or default_model
        self.memory = MemoryManager(max_history=max_history_turns)

    async def should_use_bm25(self, query: str) -> bool:
        """Determine whether BM25 (exact keyword search) significantly improves retrieval for this query."""
        start = time.perf_counter()
        prompt = build_bm25_router_prompt(query)
        response = await self.client.chat.completions.create(
            model=self.router_model,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or ""
        decision = content.strip().upper() == "YES"
        logger.debug(
            "BM25 routing decision for query: %s (%.2fs).",
            "YES" if decision else "NO", time.perf_counter() - start,
        )
        return decision

    async def generate_answer(
        self,
        query: str,
        contexts: List[str],
        session_id: Optional[str] = None,
    ) -> str:
        """Generate an answer using retrieved context passages and conversation history."""
        system_prompt = build_system_prompt(contexts)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        # Append recent history if session_id is provided
        if session_id:
            recent_history = await self.memory.get_history(session_id)
            messages.extend(recent_history)

        # Current user query
        messages.append({"role": "user", "content": query})

        start = time.perf_counter()
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
        )
        answer = response.choices[0].message.content or ""
        logger.info(
            "Generated answer with model '%s' from %d contexts in %.2fs.",
            self.model, len(contexts), time.perf_counter() - start,
        )

        # Update session memory
        if session_id:
            await self.memory.add_message(session_id, "user", query)
            await self.memory.add_message(session_id, "assistant", answer)

        return answer

    async def generate_answer_stream(
        self,
        query: str,
        contexts: List[str],
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream the generated answer chunk-by-chunk and save to session memory upon completion."""
        system_prompt = build_system_prompt(contexts)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        if session_id:
            recent_history = await self.memory.get_history(session_id)
            messages.extend(recent_history)

        messages.append({"role": "user", "content": query})

        start = time.perf_counter()
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            stream=True,
        )

        full_chunks: List[str] = []
        async for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_chunks.append(delta)
                    yield delta

        logger.info(
            "Streamed answer with model '%s' from %d contexts in %.2fs.",
            self.model, len(contexts), time.perf_counter() - start,
        )

        # Record conversation in session memory once stream completes
        complete_answer = "".join(full_chunks)
        if session_id:
            await self.memory.add_message(session_id, "user", query)
            await self.memory.add_message(session_id, "assistant", complete_answer)


_default_generator = Generator()


async def should_use_bm25(query: str) -> bool:
    return await _default_generator.should_use_bm25(query)


async def generate_answer(
    query: str,
    contexts: List[str],
    session_id: Optional[str] = None,
) -> str:
    return await _default_generator.generate_answer(
        query=query,
        contexts=contexts,
        session_id=session_id,
    )


async def generate_answer_stream(
    query: str,
    contexts: List[str],
    session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    async for chunk in _default_generator.generate_answer_stream(
        query=query,
        contexts=contexts,
        session_id=session_id,
    ):
        yield chunk



