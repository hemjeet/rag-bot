import os

os.environ.setdefault("DATABASE_URL", "postgresql://u:p@localhost/db")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest
from src.rag.fusion import ReciprocalRankFusion
from src.rag.memory import MemoryManager
from src.rag.prompts import (
    build_system_prompt,
    build_combined_router_prompt,
    build_decomposition_prompt,
)


# --- RRF Tests ---


def test_rrf_fuses_and_orders_results():
    fusion = ReciprocalRankFusion(k=60, top_n=3)
    dense = [
        {"id": "a", "text": "Passage A"},
        {"id": "b", "text": "Passage B"},
    ]
    keyword = [
        {"id": "b", "text": "Passage B"},
        {"id": "c", "text": "Passage C"},
    ]
    texts = fusion.fuse(dense, keyword)
    assert texts == ["Passage B", "Passage A", "Passage C"]


def test_rrf_respects_top_n():
    fusion = ReciprocalRankFusion(k=60, top_n=1)
    dense = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]
    assert fusion.fuse(dense, []) == ["A"]


def test_rrf_empty_results():
    fusion = ReciprocalRankFusion(k=60, top_n=5)
    assert fusion.fuse([], []) == []


def test_rrf_single_result():
    fusion = ReciprocalRankFusion(k=60, top_n=5)
    dense = [{"id": "x", "text": "X"}]
    assert fusion.fuse(dense, []) == ["X"]


def test_rrf_deduplicates_across_lists():
    fusion = ReciprocalRankFusion(k=60, top_n=10)
    dense = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]
    keyword = [{"id": "a", "text": "A"}, {"id": "c", "text": "C"}]
    result = fusion.fuse(dense, keyword)
    assert result.count("A") == 1
    assert "B" in result
    assert "C" in result


# --- Memory Tests ---


@pytest.mark.asyncio
async def test_memory_add_and_get():
    mem = MemoryManager(max_history=3)
    await mem.add_message("s1", "user", "hello")
    await mem.add_message("s1", "assistant", "hi there")
    history = await mem.get_history("s1")
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "hello"}
    assert history[1] == {"role": "assistant", "content": "hi there"}


@pytest.mark.asyncio
async def test_memory_max_history_trim():
    mem = MemoryManager(max_history=2)
    for i in range(6):
        await mem.add_message("s1", "user", f"msg{i}")
    history = await mem.get_history("s1")
    assert len(history) == 4  # max_history=2 turns = 4 messages
    assert history[0]["content"] == "msg2"
    assert history[-1]["content"] == "msg5"


@pytest.mark.asyncio
async def test_memory_separate_sessions():
    mem = MemoryManager(max_history=5)
    await mem.add_message("s1", "user", "hello s1")
    await mem.add_message("s2", "user", "hello s2")
    assert len(await mem.get_history("s1")) == 1
    assert len(await mem.get_history("s2")) == 1
    assert (await mem.get_history("s1"))[0]["content"] == "hello s1"


@pytest.mark.asyncio
async def test_memory_clear_session():
    mem = MemoryManager(max_history=5)
    await mem.add_message("s1", "user", "hello")
    await mem.clear_session("s1")
    assert await mem.get_history("s1") == []


@pytest.mark.asyncio
async def test_memory_eviction_at_capacity():
    mem = MemoryManager(max_history=5, max_sessions=2)
    await mem.add_message("s1", "user", "a")
    await mem.add_message("s2", "user", "b")
    await mem.add_message("s3", "user", "c")  # evicts s1
    assert await mem.get_history("s1") == []
    assert len(await mem.get_history("s2")) == 1
    assert len(await mem.get_history("s3")) == 1


@pytest.mark.asyncio
async def test_memory_session_count():
    mem = MemoryManager(max_history=5)
    await mem.add_message("s1", "user", "a")
    await mem.add_message("s2", "user", "b")
    assert await mem.session_count() == 2


# --- Prompt Tests ---


def test_system_prompt_with_context():
    prompt = build_system_prompt(["Context A", "Context B"])
    assert "Context A" in prompt
    assert "Context B" in prompt
    assert "Legal RAG Assistant" in prompt


def test_system_prompt_empty_context():
    prompt = build_system_prompt([])
    assert "No relevant document context" in prompt
    assert "Legal RAG Assistant" in prompt
    assert "Context:" not in prompt


def test_combined_router_prompt():
    prompt = build_combined_router_prompt("What is the liability cap?")
    assert "BM25" in prompt
    assert "multi-hop" in prompt.lower() or "multi_hop" in prompt.lower()
    assert "What is the liability cap?" in prompt


def test_decomposition_prompt():
    prompt = build_decomposition_prompt("What are the termination clauses?")
    assert "termination" in prompt.lower()
    assert "new_query" in prompt
