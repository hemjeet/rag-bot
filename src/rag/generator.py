import json
import logging
import time
from typing import List, Optional, Dict, AsyncGenerator
from openai import AsyncOpenAI
from src.config import settings
from src.rag.memory import MemoryManager
from src.rag.prompts import build_bm25_router_prompt, build_system_prompt, build_multi_hop_prompt, build_decomposition_prompt, build_combined_router_prompt

logger = logging.getLogger(__name__)


class Generator:
    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        router_client: Optional[AsyncOpenAI] = None,
        router_model: Optional[str] = None,
        max_history_turns: int = 5,
    ):
        # --- Generation client (DeepSeek or OpenAI) ---
        api_key = settings.deepseek_api_key or settings.openai_api_key
        use_deepseek = bool(settings.deepseek_api_key)
        base_url = settings.deepseek_base_url if use_deepseek else None
        self.client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        default_model = "deepseek-v4-flash" if use_deepseek else "gpt-4o-mini"
        self.model = model or settings.llm_model or default_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature

        # --- Router client (always OpenAI for cheap/fast classification) ---
        self.router_client = router_client or AsyncOpenAI(
            api_key=settings.openai_api_key,
        )
        self.router_model = router_model or settings.router_model or "gpt-4o-mini"
        self.memory = MemoryManager(max_history=max_history_turns)

    
    async def route_query(self, query: str) -> Dict[str, bool]:
        """Make BM25 and multi-hop decisions in a single LLM call.

        Returns a dict with 'use_bm25' and 'is_multi_hop' boolean fields.
        """
        start = time.perf_counter()
        prompt = build_combined_router_prompt(query)
        response = await self.router_client.chat.completions.create(
            model=self.router_model,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        result = {
            "use_bm25": bool(parsed.get("use_bm25", False)),
            "is_multi_hop": bool(parsed.get("is_multi_hop", False)),
        }
        logger.info(
            "[ROUTER] model='%s' raw_response='%s' bm25=%s multi_hop=%s (%.2fs)",
            self.router_model, content.strip(),
            "YES" if result["use_bm25"] else "NO",
            "YES" if result["is_multi_hop"] else "NO",
            time.perf_counter() - start,
        )
        return result

    async def _decompose_query(self, query: str, is_multi_hop: bool = False) -> List[str]:
        """Decompose a complex query into simpler sub-queries if multi-hop.

        When is_multi_hop is provided by the caller (from route_query), the
        separate _is_multi_hop check is skipped, saving one LLM round-trip.
        """
        start = time.perf_counter()
        if is_multi_hop:
            prompt = build_decomposition_prompt(query)
            response = await self.router_client.chat.completions.create(
                model=self.router_model,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            sub_queries = parsed.get("new_query", [query])
            logger.info(
                "[DECOMPOSE] query decomposed into %d sub-queries (%.2fs): %r",
                len(sub_queries), time.perf_counter() - start, sub_queries,
            )
            return sub_queries
        else:
            logger.info(
                "[DECOMPOSE] query is atomic, no decomposition needed (%.2fs).",
                time.perf_counter() - start,
            )
            return [query]


    async def should_use_bm25(self, query: str) -> bool:
        """Determine whether BM25 (exact keyword search) significantly improves retrieval for this query."""
        start = time.perf_counter()
        prompt = build_bm25_router_prompt(query)
        response = await self.router_client.chat.completions.create(
            model=self.router_model,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or ""
        decision = content.strip().upper() == "YES"
        logger.info(
            "[BM25-ROUTER] model='%s' raw_response='%s' decision=%s (%.2fs)",
            self.router_model, content.strip(), "YES" if decision else "NO",
            time.perf_counter() - start,
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
        usage = response.usage
        logger.info(
            "[GENERATE] model='%s' contexts=%d history=%d answer_len=%d "
            "prompt_tokens=%s completion_tokens=%s total_tokens=%s (%.2fs)",
            self.model, len(contexts), len(messages) - 2, len(answer),
            usage.prompt_tokens if usage else "?",
            usage.completion_tokens if usage else "?",
            usage.total_tokens if usage else "?",
            time.perf_counter() - start,
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

        answer_len = len("".join(full_chunks))
        logger.info(
            "[GENERATE-STREAM] model='%s' contexts=%d history=%d answer_len=%d (%.2fs)",
            self.model, len(contexts), len(messages) - 2, answer_len,
            time.perf_counter() - start,
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



