import json
import logging
import time
from typing import List, Optional, Dict, AsyncGenerator
from openai import AsyncOpenAI
from src.config import settings
from src.rag.memory import MemoryManager
from src.rag.prompts import (
    build_bm25_router_prompt,
    build_system_prompt,
    build_decomposition_prompt,
    build_combined_router_prompt,
)
from datetime import timedelta
from aiobreaker import CircuitBreaker
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# LLM Circuit Breaker
llm_breaker = CircuitBreaker(
    fail_max=settings.llm_cb_failures,
    timeout_duration=timedelta(seconds=settings.llm_cb_timeout),
)

# Max tokens for context sent to LLM (prevents token limit errors)
_MAX_CONTEXT_CHARS = 80000


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
        api_key = settings.deepseek_api_key or settings.openai_api_key
        use_deepseek = bool(settings.deepseek_api_key)
        base_url = settings.deepseek_base_url if use_deepseek else None
        self.client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)
        default_model = "deepseek-v4-flash" if use_deepseek else "gpt-4o-mini"
        self.model = model or settings.llm_model or default_model
        self.temperature = (
            temperature if temperature is not None else settings.llm_temperature
        )

        self.router_client = router_client or AsyncOpenAI(
            api_key=settings.openai_api_key
        )
        self.router_model = router_model or settings.router_model or "gpt-4o-mini"
        self.memory = MemoryManager(max_history=max_history_turns)

    @llm_breaker
    @retry(
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def route_query(self, query: str) -> Dict[str, bool]:
        """BM25 + multi-hop routing in a single LLM call."""
        start = time.perf_counter()
        prompt = build_combined_router_prompt(query)
        try:
            response = await self.router_client.chat.completions.create(
                model=self.router_model,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=30.0,
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            result = {
                "use_bm25": bool(parsed.get("use_bm25", False)),
                "is_multi_hop": bool(parsed.get("is_multi_hop", False)),
            }
            logger.info(
                "[ROUTER] model='%s' bm25=%s multi_hop=%s (%.2fs)",
                self.router_model,
                "YES" if result["use_bm25"] else "NO",
                "YES" if result["is_multi_hop"] else "NO",
                time.perf_counter() - start,
            )
            return result
        except json.JSONDecodeError:
            logger.warning(
                "[ROUTER] invalid JSON from LLM, defaulting to single-hop, no bm25 (%.2fs)",
                time.perf_counter() - start,
            )
            return {"use_bm25": False, "is_multi_hop": False}
        except Exception:
            logger.exception(
                "[ROUTER] LLM call failed (%.2fs)", time.perf_counter() - start
            )
            raise

    @llm_breaker
    @retry(
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def decompose_query(
        self, query: str, is_multi_hop: bool = False
    ) -> List[str]:
        """Decompose a complex query into sub-queries if multi-hop."""
        start = time.perf_counter()
        if not is_multi_hop:
            logger.info(
                "[DECOMPOSE] atomic query, no decomposition (%.2fs)",
                time.perf_counter() - start,
            )
            return [query]

        prompt = build_decomposition_prompt(query)
        try:
            response = await self.router_client.chat.completions.create(
                model=self.router_model,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=30.0,
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            raw = parsed.get("new_query", [query])
            # Validate: ensure it's a list of strings
            if isinstance(raw, str):
                raw = [raw]
            sub_queries = [str(q) for q in raw if q] or [query]
            logger.info(
                "[DECOMPOSE] %d sub-queries (%.2fs): %r",
                len(sub_queries),
                time.perf_counter() - start,
                sub_queries,
            )
            return sub_queries
        except json.JSONDecodeError:
            logger.warning(
                "[DECOMPOSE] invalid JSON, using original query (%.2fs)",
                time.perf_counter() - start,
            )
            return [query]
        except Exception:
            logger.exception(
                "[DECOMPOSE] LLM call failed (%.2fs)", time.perf_counter() - start
            )
            raise

    async def should_use_bm25(self, query: str) -> bool:
        """Legacy single-decision BM25 router."""
        start = time.perf_counter()
        prompt = build_bm25_router_prompt(query)
        try:
            response = await self.router_client.chat.completions.create(
                model=self.router_model,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                timeout=30.0,
            )
            content = response.choices[0].message.content or ""
            decision = content.strip().upper() == "YES"
            logger.info(
                "[BM25-ROUTER] decision=%s (%.2fs)",
                "YES" if decision else "NO",
                time.perf_counter() - start,
            )
            return decision
        except Exception:
            logger.exception(
                "[BM25-ROUTER] failed (%.2fs)", time.perf_counter() - start
            )
            return False

    @llm_breaker
    @retry(
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate_answer(
        self,
        query: str,
        contexts: List[str],
        session_id: Optional[str] = None,
    ) -> str:
        """Generate an answer using retrieved context and conversation history."""
        system_prompt = build_system_prompt(contexts)
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        if session_id:
            recent_history = await self.memory.get_history(session_id)
            messages.extend(recent_history)

        messages.append({"role": "user", "content": query})

        start = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
                timeout=60.0,
            )
            answer = response.choices[0].message.content or ""
            usage = response.usage
            logger.info(
                "[GENERATE] model='%s' contexts=%d answer_len=%d "
                "tokens=%s/%s/%s (%.2fs)",
                self.model,
                len(contexts),
                len(answer),
                usage.prompt_tokens if usage else "?",
                usage.completion_tokens if usage else "?",
                usage.total_tokens if usage else "?",
                time.perf_counter() - start,
            )
        except Exception:
            logger.exception(
                "[GENERATE] LLM call failed (%.2fs)", time.perf_counter() - start
            )
            raise

        if session_id:
            await self.memory.add_message(session_id, "user", query)
            await self.memory.add_message(session_id, "assistant", answer)

        return answer

    @llm_breaker
    @retry(
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate_answer_stream(
        self,
        query: str,
        contexts: List[str],
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream the generated answer chunk-by-chunk."""
        system_prompt = build_system_prompt(contexts)
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        if session_id:
            recent_history = await self.memory.get_history(session_id)
            messages.extend(recent_history)

        messages.append({"role": "user", "content": query})

        start = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
                stream=True,
                timeout=60.0,
            )
            full_chunks: List[str] = []
            async for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        full_chunks.append(delta)
                        yield delta

            logger.info(
                "[GENERATE-STREAM] model='%s' answer_len=%d (%.2fs)",
                self.model,
                len("".join(full_chunks)),
                time.perf_counter() - start,
            )

            complete_answer = "".join(full_chunks)
            if session_id and complete_answer:
                await self.memory.add_message(session_id, "user", query)
                await self.memory.add_message(session_id, "assistant", complete_answer)
        except Exception:
            logger.exception(
                "[GENERATE-STREAM] failed (%.2fs)", time.perf_counter() - start
            )
            raise
