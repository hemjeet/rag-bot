import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from aiobreaker import CircuitBreakerError
from pydantic import BaseModel, Field

from src.db.session import init_pool, close_pool, check_db_health, db_breaker
from src.db import session as db_session
from src.rag.pipeline import HybridPipeline
from src.api.deps import get_pipeline
from src.rag.generator import llm_breaker
from src.rag.embeddings import embed_breaker

logger = logging.getLogger(__name__)

# --- Pydantic models (consolidated from routes.py) ---


class QueryRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=10000, description="User query or question"
    )
    collection_name: str = Field(
        "legal_documents", description="Target collection name"
    )
    use_bm25: Optional[bool] = Field(
        None, description="Force BM25 enable/disable, or leave None for auto-routing"
    )
    session_id: Optional[str] = Field(
        None, description="Optional conversation session ID for multi-turn memory"
    )


class QueryResponse(BaseModel):
    query: str
    answer: str
    collection_name: str
    session_id: Optional[str] = None


# --- App setup ---

app = FastAPI(
    title="Hybrid RAG API with pgvector",
    description="Production-ready Hybrid RAG combining pgvector HNSW dense search, PostgreSQL TSVector keyword search, Reciprocal Rank Fusion (RRF), and LLM generation.",
    version="0.1.0",
)

# CORS: restrict to environment-configured origins
_cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled exception in middleware for %s %s",
            request.method,
            request.url.path,
        )
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    if not request.url.path.startswith("/chainlit/ws/"):
        logger.info(
            "%s %s -> %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
    return response


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()

    from src.rag.embeddings import Embeddings
    from src.rag.dense import DenseSearcher
    from src.rag.keyword import KeywordSearcher
    from src.rag.fusion import ReciprocalRankFusion
    from src.rag.generator import Generator

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

    app.state.pipeline = pipeline

    # Prime all connections (embedding API, pgvector, TSVector, cache, LLMs)
    await pipeline.warmup()

    logger.info("Pipeline initialized.")
    try:
        yield
    finally:
        await close_pool()
        logger.info("Shutdown complete.")


app.router.lifespan_context = lifespan


# --- Routes ---


@app.get("/health", tags=["Health"])
async def health_check():
    if db_session.pool is None or getattr(app.state, "pipeline", None) is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    db_ok = await check_db_health()
    if not db_ok:
        raise HTTPException(status_code=503, detail="Database unreachable")
    return {"status": "healthy"}


@app.get("/health/breakers", tags=["Health"])
async def breaker_status():
    """Return the state of all circuit breakers for monitoring."""
    def _state(breaker):
        return {
            "state": breaker.state.name,
            "fail_count": breaker.fail_counter.count,
            "success_count": breaker.success_counter.count,
            "last_failure": str(breaker.last_failure) if breaker.last_failure else None,
        }

    return {
        "db": _state(db_breaker),
        "llm": _state(llm_breaker),
        "embed": _state(embed_breaker),
    }


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/chainlit")


@app.exception_handler(CircuitBreakerError)
async def circuit_breaker_exception_handler(request: Request, exc: CircuitBreakerError):
    logger.warning("Circuit breaker tripped: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "The service is temporarily overloaded. Please try again in 30 seconds."},
    )


@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(
    request: QueryRequest,
    pipeline: HybridPipeline = Depends(get_pipeline),
):
    logger.info(
        "[API] query=%r collection='%s' session_id=%s bm25=%s",
        request.query[:80],
        request.collection_name,
        request.session_id,
        request.use_bm25,
    )
    try:
        answer = await pipeline.run(
            query=request.query,
            collection_name=request.collection_name,
            use_bm25=request.use_bm25,
            session_id=request.session_id,
        )
        return QueryResponse(
            query=request.query,
            answer=answer,
            collection_name=request.collection_name,
            session_id=request.session_id,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in /api/query")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/query/stream", tags=["RAG"])
async def query_rag_stream(
    request: QueryRequest,
    pipeline: HybridPipeline = Depends(get_pipeline),
):
    async def token_generator():
        try:
            async for chunk in pipeline.run_stream(
                query=request.query,
                collection_name=request.collection_name,
                use_bm25=request.use_bm25,
                session_id=request.session_id,
            ):
                yield chunk
        except CircuitBreakerError:
            logger.warning("Circuit breaker tripped during streaming.")
            yield "\n[System is currently overloaded. Please try again later.]"
        except Exception:
            logger.exception("Error during streaming query")
            yield "\n[Error: internal server error]"

    return StreamingResponse(token_generator(), media_type="text/plain")


# Mount Chainlit UI
try:
    from chainlit.utils import mount_chainlit

    mount_chainlit(app, target="chainlit_app.py", path="/chainlit")
    logger.info("Chainlit UI mounted at /chainlit.")
except Exception as exc:
    logger.warning("Chainlit UI could not be mounted: %s", exc)


if __name__ == "__main__":
    import uvicorn

    workers = int(os.environ.get("WORKERS", 0))
    if workers > 0:
        uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=workers)
    else:
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
