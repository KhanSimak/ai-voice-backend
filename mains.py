import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from rag import create_vectorstore, get_llm, ask_question

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize vectorstore and LLM once at startup."""
    logger.info("Initializing vectorstore and LLM...")
    try:
        app.state.vectorstore = create_vectorstore()
        app.state.llm = get_llm()
        logger.info("Vectorstore and LLM ready.")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GreenValley Clinic RAG API",
    description="Ask questions about GreenValley Clinic using a PDF knowledge base.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ──────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "GreenValley Clinic RAG API",
        "usage": "POST /ask with JSON body: {\"question\": \"your question here\"}",
    }


@app.get("/health")
async def health():
    ready = (
        hasattr(app.state, "vectorstore")
        and app.state.vectorstore is not None
        and hasattr(app.state, "llm")
        and app.state.llm is not None
    )
    return {"status": "ready" if ready else "initializing"}


@app.post("/ask", response_model=AnswerResponse)
async def ask(body: QuestionRequest):
    if not body.question or not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    if not hasattr(app.state, "vectorstore") or app.state.vectorstore is None:
        raise HTTPException(status_code=503, detail="Vectorstore not yet initialized. Please try again shortly.")

    if not hasattr(app.state, "llm") or app.state.llm is None:
        raise HTTPException(status_code=503, detail="LLM not yet initialized. Please try again shortly.")

    try:
        answer = ask_question(app.state.vectorstore, app.state.llm, body.question)
        return AnswerResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error answering question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("mains:app", host="0.0.0.0", port=port, reload=False)