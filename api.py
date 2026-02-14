"""FastAPI REST API for YouTube RAG Chatbot."""

from contextlib import asynccontextmanager
from time import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from rag_pipeline import RAGPipeline

pipeline: RAGPipeline = None

# Simple in-memory cache for repeated queries.
# key: (normalized_question, top_k), value: {"expires_at": ts, "result": dict}
QUERY_CACHE: dict[tuple[str, int], dict] = {}
CACHE_TTL_SECONDS = config.CACHE_TTL_SECONDS


def _cache_key(question: str, top_k: int) -> tuple[str, int]:
    normalized_question = " ".join(question.strip().lower().split())
    return normalized_question, top_k


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG pipeline on startup."""
    global pipeline

    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline(rebuild_index=False)
    print("RAG pipeline ready!")
    yield
    print("Shutting down...")


app = FastAPI(
    title="YouTube RAG API",
    description="Query YouTube transcripts using RAG with hybrid search",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask")
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of sources to retrieve"
    )


class Source(BaseModel):
    title: str
    url: str
    start_time: float
    score: float
    text_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    query: str


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and index status."""
    return HealthResponse(
        status="healthy",
        index_loaded=pipeline is not None and pipeline.index is not None,
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG pipeline with a question."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Fast path: return cached response for repeated queries.
        key = _cache_key(request.question, request.top_k)
        cached = QUERY_CACHE.get(key)
        now_ts = time()
        if cached and cached["expires_at"] > now_ts:
            return QueryResponse(**cached["result"])

        if request.top_k != config.TOP_K:
            pipeline.hybrid_retriever.top_k = request.top_k

        result = pipeline.query(request.question)
        response_payload = {
            "answer": result["answer"],
            "sources": [Source(**s) for s in result["sources"]],
            "query": request.question,
        }

        QUERY_CACHE[key] = {
            "expires_at": now_ts + CACHE_TTL_SECONDS,
            "result": response_payload,
        }

        return QueryResponse(**response_payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild")
async def rebuild_index():
    """Rebuild the vector index from source data."""
    global pipeline

    try:
        print("Rebuilding index...")
        pipeline = RAGPipeline(rebuild_index=True)
        QUERY_CACHE.clear()
        return {"status": "success", "message": "Index rebuilt successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
