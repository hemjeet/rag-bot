from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query or question")
    collection_name: str = Field("legal_documents", description="Target collection name")
    use_bm25: Optional[bool] = Field(None, description="Force BM25 enable/disable, or leave None for auto-routing")
    session_id: Optional[str] = Field("default", description="Optional conversation session ID for multi-turn memory")


class QueryResponse(BaseModel):
    query: str
    answer: str
    collection_name: str
    session_id: Optional[str] = None

