"""Chainlit frontend for the Hybrid RAG pipeline.

This module is a **pure UI layer** — it owns zero database connections.
It is mounted into the FastAPI application defined in ``app.py`` via
``chainlit.utils.mount_chainlit`` and reuses the already-initialized
``app.state.pipeline`` (which uses the single psycopg connection pool
created in ``src.db.session``).

Chainlit's built-in data-persistence layer is explicitly disabled so that
it never opens its own ``asyncpg`` pool, avoiding Supabase connection
exhaustion errors.
"""

import asyncio
import logging
import sys

# psycopg async does not work with the default Windows ProactorEventLoop.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import chainlit as cl
import chainlit.data as cl_data

# ---------------------------------------------------------------------------
# Disable Chainlit's database persistence completely.
# Chainlit auto-detects DATABASE_URL and opens its own asyncpg pool for
# persisting chat steps/threads.  We don't need that — session memory is
# handled by our own MemoryManager, and the only DB pool should be the one
# in src.db.session (managed by app.py's lifespan).
#
# Setting _data_layer = None is NOT enough — get_data_layer() checks the
# _data_layer_initialized flag and will re-create the pool if it's False.
# We must set BOTH to prevent Chainlit from opening an asyncpg pool.
# ---------------------------------------------------------------------------
cl_data._data_layer = None
cl_data._data_layer_initialized = True

from src.logging_config import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "legal_documents"


def _get_pipeline():
    """Return the pipeline from app.py's ``app.state`` (deferred import)."""
    from app import app as fastapi_app

    pipeline = getattr(fastapi_app.state, "pipeline", None)
    if pipeline is None:
        raise RuntimeError("RAG pipeline is not initialized yet.")
    return pipeline


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("collection_name", DEFAULT_COLLECTION)

    await cl.Message(
        content=(
            f"Welcome to the Legal RAG assistant. "
            f"I answer questions using the `{DEFAULT_COLLECTION}` collection.\n\n"
            "Ask me anything about your legal documents."
        )
    ).send()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Summarize key legal points",
            message="Summarize the key legal points from the documents.",
        ),
        cl.Starter(
            label="Find a specific clause",
            message="Find clauses related to termination and liability.",
        ),
        cl.Starter(
            label="Explain an obligation",
            message="What obligations do the parties have under the agreement?",
        ),
    ]


@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()
    if not query:
        return

    try:
        pipeline = _get_pipeline()
    except RuntimeError as exc:
        await cl.Message(content=f"[Error: {exc}]").send()
        return

    collection_name = cl.user_session.get("collection_name") or DEFAULT_COLLECTION
    session_id = cl.user_session.get("id")

    msg = cl.Message(content="")
    await msg.send()

    try:
        async for token in pipeline.run_stream(
            query=query,
            collection_name=collection_name,
            use_bm25=None,
            session_id=session_id,
        ):
            await msg.stream_token(token)
    except Exception as exc:
        logger.exception("Error while answering chat message.")
        await msg.stream_token(f"\n\n[Error: {exc}]")

    await msg.update()
