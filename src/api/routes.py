import logging

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from src.api.deps import get_pipeline
from src.api.schemas import (
    QueryRequest,
    QueryResponse,
)
from src.rag.pipeline import HybridPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse, summary="Query the Hybrid RAG pipeline")
async def query_rag(
    request: QueryRequest,
    pipeline: HybridPipeline = Depends(get_pipeline),
):
    logger.info(
        "Query endpoint called (collection='%s', session_id='%s', use_bm25=%s).",
        request.collection_name, request.session_id, request.use_bm25,
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
    except Exception as e:
        logger.exception("Unhandled error in /query endpoint.")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream", summary="Stream query answer token-by-token")
async def query_rag_stream(
    request: QueryRequest,
    pipeline: HybridPipeline = Depends(get_pipeline),
):
    logger.info(
        "Streaming query endpoint called (collection='%s', session_id='%s', use_bm25=%s).",
        request.collection_name, request.session_id, request.use_bm25,
    )

    async def token_generator():
        try:
            async for chunk in pipeline.run_stream(
                query=request.query,
                collection_name=request.collection_name,
                use_bm25=request.use_bm25,
                session_id=request.session_id,
            ):
                yield chunk
        except Exception as e:
            logger.exception("Error during streaming query generation.")
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(token_generator(), media_type="text/plain")
