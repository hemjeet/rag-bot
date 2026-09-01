import asyncio
import sys
import logging
import time

# psycopg async does not work with the default Windows ProactorEventLoop.
# Set the selector policy before uvicorn can create its event loop.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.db.session import init_pool, close_pool
from src.db import session as db_session
from src.rag.embeddings import Embeddings
from src.rag.dense import DenseSearcher
from src.rag.keyword import KeywordSearcher
from src.rag.fusion import ReciprocalRankFusion
from src.rag.generator import Generator
from src.rag.pipeline import HybridPipeline
from src.api.deps import get_pipeline
from src.api.routes import router as api_router
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize async PostgreSQL connection pool (registers pgvector adapters)
    await init_pool()

    # 2. Pre-load RAG singletons once during startup
    embeddings = Embeddings()
    dense_searcher = DenseSearcher(embeddings=embeddings)
    keyword_searcher = KeywordSearcher()
    fusion = ReciprocalRankFusion()
    generator = Generator()

    pipeline = HybridPipeline(
        dense_searcher=dense_searcher,
        keyword_searcher=keyword_searcher,
        fusion=fusion,
        generator=generator,
    )

    # Attach all pre-warmed singletons to app.state
    app.state.embeddings = embeddings
    app.state.dense_searcher = dense_searcher
    app.state.keyword_searcher = keyword_searcher
    app.state.fusion = fusion
    app.state.generator = generator
    app.state.pipeline = pipeline

    # 3. Warm up HTTP connections to avoid cold-start latency on first query.
    #    This pre-establishes TLS connections to OpenAI (embeddings + router)
    #    and DeepSeek (generation) so the first user request is fast.
    try:
        warmup_start = time.perf_counter()
        await asyncio.gather(
            embeddings.embed_query("warmup"),           # OpenAI embeddings connection
            generator.router_client.chat.completions.create(  # OpenAI router connection
                model=generator.router_model,
                temperature=0.0,
                messages=[{"role": "user", "content": "Reply with OK"}],
                max_tokens=2,
            ),
        )
        logger.info("HTTP connections warmed up in %.2fs.", time.perf_counter() - warmup_start)
    except Exception as exc:
        logger.warning("Connection warmup failed (non-fatal): %s", exc)

    logger.info("Database connection pool and RAG pipeline components initialized successfully.")
    try:
        yield
    finally:
        # Shutdown: Immediately terminate pool to release Supabase connections
        # before uvicorn spawns a new worker on reload.
        await close_pool()
        logger.info("Application shutdown complete.")


app = FastAPI(
    title="Hybrid RAG API with pgvector",
    description="Production-ready Hybrid RAG combining pgvector HNSW dense search, PostgreSQL TSVector keyword search, Reciprocal Rank Fusion (RRF), and LLM generation.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware. allow_credentials must be False when using a wildcard origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    # Skip noisy Chainlit WebSocket polling logs
    if not request.url.path.startswith("/chainlit/ws/"):
        logger.info(
            "%s %s -> %d (%.1fms)",
            request.method, request.url.path, response.status_code, duration_ms,
        )
    return response


# Include API routes
app.include_router(api_router, prefix="/api", tags=["RAG"])


# Mount the Chainlit UI into this same FastAPI app so a single server serves
# both the API and the chat frontend (available at /chainlit).
try:
    from chainlit.utils import mount_chainlit

    mount_chainlit(app, target="chainlit_app.py", path="/chainlit")
    logger.info("Chainlit UI mounted at /chainlit.")
except Exception as exc:  # pragma: no cover - optional frontend
    logger.warning("Chainlit UI could not be mounted: %s", exc)


@app.get("/health", tags=["Health"])
async def health_check():
    if db_session.pool is None or getattr(app.state, "pipeline", None) is None:
        raise HTTPException(status_code=503, detail="Service is not ready")
    return {"status": "healthy", "service": "hybrid-rag-pgvector"}


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to Hybrid RAG with pgvector API",
        "docs_url": "/docs",
        "health_check": "/health",
    }


class ChatRequest(BaseModel):
    message: str = Field(..., description="Query or prompt message")
    session_id: Optional[str] = Field("default", description="Session ID for conversation memory")
    collection_name: str = Field("legal_documents", description="Collection name")
    use_bm25: Optional[bool] = Field(None, description="Optional override for BM25 keyword retrieval")


class ChatResponse(BaseModel):
    answer: str
    session_id: Optional[str] = None
    collection_name: str
    use_bm25: Optional[bool] = None


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    pipeline: HybridPipeline = Depends(get_pipeline),
):
    """Chat with the Hybrid RAG system."""
    logger.info(
        "Chat endpoint called (collection='%s', session_id='%s', use_bm25=%s).",
        request.collection_name, request.session_id, request.use_bm25,
    )
    try:
        answer = await pipeline.run(
            query=request.message,
            collection_name=request.collection_name,
            use_bm25=request.use_bm25,
            session_id=request.session_id,
        )
        return ChatResponse(
            answer=answer,
            session_id=request.session_id,
            collection_name=request.collection_name,
            use_bm25=request.use_bm25,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in /chat endpoint.")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(
    request: ChatRequest,
    pipeline: HybridPipeline = Depends(get_pipeline),
):
    """Stream chat response token-by-token using Chunked/Event stream."""
    logger.info(
        "Chat stream endpoint called (collection='%s', session_id='%s', use_bm25=%s).",
        request.collection_name, request.session_id, request.use_bm25,
    )

    async def token_generator():
        try:
            async for chunk in pipeline.run_stream(
                query=request.message,
                collection_name=request.collection_name,
                use_bm25=request.use_bm25,
                session_id=request.session_id,
            ):
                yield chunk
        except Exception as e:
            logger.exception("Error during chat stream generation.")
            yield f"\n[Error during generation: {str(e)}]"

    return StreamingResponse(token_generator(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
