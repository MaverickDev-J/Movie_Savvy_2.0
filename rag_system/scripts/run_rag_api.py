import sys
from pathlib import Path
import yaml
import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent))

# Import shared RAG logic from run_rag.py (no duplication)
from rag_system.scripts.run_rag import (
    run_rag,
    get_generator,
    get_function_manager,
    config,
    logger,
)

# Load environment variables
load_dotenv(dotenv_path=BASE_DIR / ".env")

# ── FastAPI App ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Movie Savvy RAG API",
    description="Multi-source RAG pipeline for entertainment queries",
    version="2.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ───────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k_sampling: Optional[int] = None
    similarity_threshold: Optional[float] = None


class QueryResponse(BaseModel):
    query: str
    response: str
    content_sources_used: dict
    query_analysis: dict
    parameters: dict


# ── Startup ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Initialize generator and function manager on startup."""
    logger.info("Starting Movie Savvy RAG API...")
    try:
        get_generator()
        get_function_manager()
        logger.info("All components initialized successfully ✔")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")


# ── Endpoints ───────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query using the RAG pipeline (POST)."""
    try:
        raw_result = await run_rag(
            query=request.query,
            top_k=request.top_k,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k_sampling=request.top_k_sampling,
            similarity_threshold=request.similarity_threshold,
        )

        # run_rag returns a string; wrap it into a structured response
        if isinstance(raw_result, dict):
            return raw_result
        else:
            return QueryResponse(
                query=request.query,
                response=str(raw_result),
                content_sources_used={},
                query_analysis={},
                parameters={},
            )

    except Exception as e:
        logger.exception(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query")
async def process_query_get(
    query: str,
    top_k: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k_sampling: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
):
    """Process a query using the RAG pipeline (GET)."""
    request = QueryRequest(
        query=query,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k_sampling=top_k_sampling,
        similarity_threshold=similarity_threshold,
    )
    return await process_query(request)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Movie Savvy RAG API is running"}


@app.get("/status")
async def get_status():
    """System status and configuration."""
    from rag_system.scripts.run_rag import generator, function_manager

    return {
        "status": "running",
        "components": {
            "generator": generator is not None,
            "function_manager": function_manager is not None,
            "tavily_api_key": os.getenv("TAVILY_API_KEY") is not None,
        },
        "config": {
            "retrieval_top_k": config.get("retrieval", {}).get("top_k", "N/A"),
            "similarity_threshold": config.get("retrieval", {}).get("similarity_threshold", "N/A"),
            "generation_params": config.get("generation", {}).get("parameters", {}),
        },
    }


# ── Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
