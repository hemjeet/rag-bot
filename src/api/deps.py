import logging

from fastapi import HTTPException, Request

from src.rag.pipeline import HybridPipeline

logger = logging.getLogger(__name__)


def get_pipeline(request: Request) -> HybridPipeline:
    """Retrieve the pre-initialized HybridPipeline from the FastAPI app state."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        logger.error("Pipeline not initialized when handling request to %s.", request.url.path)
        raise HTTPException(
            status_code=503,
            detail="Hybrid RAG pipeline is not initialized yet. Please try again shortly.",
        )
    return pipeline
