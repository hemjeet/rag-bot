# Hybrid RAG with pgvector

A production-ready **Hybrid Retrieval-Augmented Generation** system combining **pgvector HNSW dense search**, **PostgreSQL TSVector keyword search**, **Reciprocal Rank Fusion (RRF)**, and **LLM generation** — backed by **Supabase** and served via **FastAPI** with a **Chainlit** chat UI.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Hybrid Retrieval** | Combines dense (semantic) and sparse (BM25 keyword) search for superior recall |
| **Adaptive Query Routing** | LLM-based router automatically decides when BM25 keyword search is beneficial |
| **Reciprocal Rank Fusion** | Merges dense + sparse result lists using RRF scoring for optimal context selection |
| **Streaming Responses** | Real-time token-by-token streaming via `StreamingResponse` |
| **Multi-Session Memory** | Async-safe conversation history with `MemoryManager` (per-session, configurable depth) |
| **Chainlit Chat UI** | Beautiful, production-ready chat frontend mounted at `/chainlit` |
| **Single Connection Pool** | One `psycopg` async pool shared across all components — no connection exhaustion |
| **Lifespan Preloading** | All RAG components initialized once at startup via FastAPI lifespan |
| **Structured Logging** | Request timing, pipeline stages, and error tracking with rotating file + console logs |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     FastAPI Application                       │
│                                                              │
│  Lifespan Startup                                            │
│  ├─ init_pool()         → Single psycopg AsyncConnectionPool │
│  ├─ Embeddings()        → OpenAI text-embedding-3-small      │
│  ├─ DenseSearcher()     → pgvector HNSW cosine search        │
│  ├─ KeywordSearcher()   → PostgreSQL TSVector + ts_rank       │
│  ├─ ReciprocalRankFusion()                                   │
│  ├─ Generator()         → DeepSeek / OpenAI LLM              │
│  └─ HybridPipeline()    → Orchestrates the full RAG flow     │
│                                                              │
│  Endpoints                                                   │
│  ├─ POST /chat          → JSON response                      │
│  ├─ POST /chat/stream   → Streaming response                 │
│  ├─ POST /api/query     → JSON response (API router)         │
│  ├─ POST /api/query/stream → Streaming response (API router) │
│  ├─ GET  /health        → Readiness check                    │
│  └─ /chainlit           → Chainlit chat UI                   │
│                                                              │
└────────────────────┬─────────────────────────────────────────┘
                     │  Single psycopg pool (max 5 connections)
                     ▼
            Supabase PostgreSQL + pgvector
            ├─ collections table
            ├─ chunks table (VECTOR + TSVECTOR)
            ├─ HNSW index (vector_cosine_ops)
            └─ GIN index (content_tsv)
```

### Query Pipeline Flow

```
User Query
    │
    ▼
┌─────────────────────┐
│  BM25 Router (LLM)  │──→ Should keyword search be used?
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│ Dense  │ │ Keyword  │   (run concurrently if BM25 = YES)
│ Search │ │ Search   │
└───┬────┘ └────┬─────┘
    │           │
    ▼           ▼
┌─────────────────────┐
│  Reciprocal Rank    │──→ Merge & re-rank results
│  Fusion (RRF)       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  LLM Generator      │──→ Generate answer with context + session memory
│  (stream or batch)   │
└─────────────────────┘
```

---

## 📁 Project Structure

```
hybrid-rag-pgvector/
├── app.py                  # FastAPI application (lifespan, middleware, endpoints)
├── main.py                 # Entry point (Windows event loop policy + uvicorn)
├── chainlit_app.py         # Chainlit chat UI (pure frontend, no DB connections)
├── sql/
│   └── init_db.sql         # PostgreSQL schema (collections, chunks, indexes, triggers)
├── src/
│   ├── config.py           # Pydantic settings (env vars, model config, pool sizing)
│   ├── logging_config.py   # Structured logging setup (console + rotating file)
│   ├── db/
│   │   └── session.py      # Async connection pool (psycopg + pgvector registration)
│   ├── api/
│   │   ├── deps.py         # FastAPI dependencies (get_pipeline)
│   │   ├── schemas.py      # Pydantic request/response models
│   │   └── routes.py       # API router (/api/query, /api/query/stream)
│   └── rag/
│       ├── embeddings.py   # OpenAI embedding wrapper
│       ├── dense.py        # pgvector HNSW cosine similarity search
│       ├── keyword.py      # PostgreSQL TSVector + ts_rank BM25 search
│       ├── fusion.py       # Reciprocal Rank Fusion (RRF) scoring
│       ├── memory.py       # Async-safe multi-session conversation memory
│       ├── prompts.py      # System & router prompt templates
│       ├── generator.py    # LLM answer generation (batch + streaming)
│       └── pipeline.py     # HybridPipeline orchestration (run + run_stream)
├── tests/
│   ├── test_retrieval.py
│   └── test_pipeline.py
├── .env.example            # Environment variable template
├── .chainlit/
│   └── config.toml         # Chainlit UI configuration
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

- **Python** ≥ 3.11
- **Supabase** project (or any PostgreSQL instance with pgvector enabled)
- **OpenAI API key** (for embeddings)
- **DeepSeek API key** (optional — for generation; falls back to OpenAI)

### 1. Clone & Install

```bash
git clone https://github.com/your-username/hybrid-rag-pgvector.git
cd hybrid-rag-pgvector
pip install -e .
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```ini
OPENAI_API_KEY='sk-...'
DEEPSEEK_API_KEY='sk-...'                          # Optional
DEEPSEEK_BASE_URL='https://api.deepseek.com'       # Optional
DATABASE_URL='postgresql://user:pass@host:5432/db'
```

### 3. Initialize Database Schema

Run the SQL in your Supabase SQL Editor (or `psql`):

```bash
psql $DATABASE_URL -f sql/init_db.sql
```

This creates:
- `collections` table with UUID primary key
- `chunks` table with `VECTOR(1536)` and `TSVECTOR` columns
- HNSW index for fast cosine similarity search
- GIN index for full-text keyword search
- Auto-update trigger that builds weighted TSVector from metadata fields

### 4. Start the Server

```bash
python main.py
```

The server starts at `http://localhost:8000` with:
- **Swagger docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Chainlit UI**: [http://localhost:8000/chainlit](http://localhost:8000/chainlit)
- **Health check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 📡 API Reference

### Chat Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a query, receive a full JSON response |
| `POST` | `/chat/stream` | Send a query, receive streaming tokens |
| `POST` | `/api/query` | Same as `/chat` (API router) |
| `POST` | `/api/query/stream` | Same as `/chat/stream` (API router) |

#### Request Body

```json
{
  "message": "What is the penalty for breaching Clause 8.2?",
  "session_id": "session_alpha",
  "collection_name": "legal_documents",
  "use_bm25": null
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | `string` | *required* | The user's question |
| `session_id` | `string` | `"default"` | Session ID for conversation memory |
| `collection_name` | `string` | `"legal_documents"` | Target document collection |
| `use_bm25` | `bool \| null` | `null` | Override BM25 routing (`null` = auto-detect) |

#### Response (JSON)

```json
{
  "answer": "Under Clause 8.2, a liquidated penalty of $50,000 applies immediately...",
  "session_id": "session_alpha",
  "collection_name": "legal_documents",
  "use_bm25": null
}
```

### Utility Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Returns 200 if pool + pipeline are ready, 503 otherwise |
| `GET` | `/` | Welcome message with links to docs |

---

## 💬 Example Usage

### cURL — Standard Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the termination provisions?", "session_id": "s1"}'
```

### cURL — Streaming Chat

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize the confidentiality clause", "session_id": "s1"}'
```

### Python — Streaming Client

```python
import httpx, asyncio

async def stream_chat():
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", "http://localhost:8000/chat/stream",
            json={"message": "What liabilities exist?", "session_id": "s1"}
        ) as resp:
            async for chunk in resp.aiter_text():
                print(chunk, end="", flush=True)

asyncio.run(stream_chat())
```

---

## ⚙️ Configuration

All settings are loaded from environment variables (or `.env` file) via Pydantic Settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | *required* | PostgreSQL connection string |
| `OPENAI_API_KEY` | *required* | OpenAI API key (embeddings) |
| `DEEPSEEK_API_KEY` | `None` | DeepSeek API key (generation) |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek base URL |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EMBEDDING_DIMENSION` | `1536` | Embedding vector dimension |
| `DENSE_TOP_K` | `10` | Dense search result count |
| `SPARSE_TOP_K` | `10` | Keyword search result count |
| `RRF_K` | `60` | RRF smoothing constant |
| `HYBRID_TOP_N` | `5` | Final context count after fusion |
| `LLM_MODEL` | auto-detected | Generation model name |
| `ROUTER_MODEL` | auto-detected | BM25 routing model name |
| `LLM_TEMPERATURE` | `0.0` | Generation temperature |
| `DB_POOL_MIN_SIZE` | `1` | Minimum pool connections |
| `DB_POOL_MAX_SIZE` | `5` | Maximum pool connections |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FILE` | `agent.log` | Log file path |

---

## 🐳 Docker

```bash
docker build -t hybrid-rag-pgvector .
docker run -p 8000:8000 --env-file .env hybrid-rag-pgvector
```

---

## 🧪 Testing

```bash
pytest -v
```

---

## 📄 License

MIT
