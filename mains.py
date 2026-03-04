import os
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import Request

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from rag import create_vectorstore, get_llm, ask_question

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initializing vectorstore and LLM...")
    try:
        app.state.vectorstore = create_vectorstore()
        app.state.llm = get_llm()
        logger.info("Startup complete.")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG + Retell AI API",
    description="PDF knowledge base QA and Retell AI voice webhook.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas — existing /ask endpoint ─────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str


# ── Schemas — Retell webhook ──────────────────────────────────────────────────

class RetellResponse(BaseModel):
    response: str


# ── Voice-optimised RAG answer ────────────────────────────────────────────────

def ask_question_for_voice(vectorstore, llm, question: str, history: list) -> str:
    """
    Same FAISS + LLM pipeline as /ask, tuned for voice:
    - Shorter answers (2-3 sentences)
    - No markdown or bullet points
    - Conversation history included for multi-turn context
    """
    question = question.strip()
    if not question:
        return "I didn't catch that. Could you please repeat your question?"

    docs = vectorstore.similarity_search(question, k=3)

    if docs:
        context = "\n\n".join(
            f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content.strip()}"
            for doc in docs
        )
        system_content = (
            "You are a helpful, friendly voice assistant answering questions about our services. "
            "You are speaking aloud on a phone call — keep answers SHORT (2-3 sentences max). "
            "Do NOT use bullet points, numbered lists, markdown, or special characters. "
            "Speak naturally and conversationally. "
            "Answer using ONLY the context below. "
            "If the context does not contain the answer, say: "
            "'I don't have that information right now, but our team can help you with that.' "
            f"\n\nCONTEXT FROM KNOWLEDGE BASE:\n{context}"
        )
    else:
        system_content = (
            "You are a helpful, friendly voice assistant. "
            "You are speaking aloud on a phone call — keep answers SHORT (2-3 sentences max). "
            "Do NOT use bullet points, numbered lists, or markdown. "
            "If you don't know something, offer to connect the caller with the team."
        )

    messages = [SystemMessage(content=system_content)]

    # Include last 6 turns of conversation history for multi-turn awareness
    for msg in history[-6:]:
     role = msg.get("role", "")
    content = msg.get("content", "")
    if role == "user":
        messages.append(HumanMessage(content=content))
    elif role == "assistant":
        messages.append(AIMessage(content=content))


    response = llm.invoke(messages)
    if hasattr(response, "content"):
        return response.content.strip()
    return str(response).strip()


# ── Existing endpoint: /ask ───────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "ok",
        "endpoints": {
            "POST /ask": "Text Q&A over PDF knowledge base",
            "POST /retell-webhook": "Retell AI voice call webhook",
            "GET /health": "Health check",
        },
    }

@app.get("/health")
async def health():
    ready = (
        hasattr(app.state, "vectorstore") and app.state.vectorstore is not None
        and hasattr(app.state, "llm") and app.state.llm is not None
    )
    return {"status": "ready" if ready else "initializing"}

@app.post("/ask", response_model=AnswerResponse)
async def ask(body: QuestionRequest):
    if not body.question or not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if not hasattr(app.state, "vectorstore") or app.state.vectorstore is None:
        raise HTTPException(status_code=503, detail="Service not ready yet. Please retry shortly.")
    try:
        answer = ask_question(app.state.vectorstore, app.state.llm, body.question)
        return AnswerResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error in /ask: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")


# ── New endpoint: /retell-webhook ─────────────────────────────────────────────

from fastapi import Request

@app.post("/retell-webhook")
async def retell_webhook(request: Request):
    body = await request.json()
    call = body.get("call", {})
    transcript = call.get("transcript_object", [])

    # Extract latest user message
    latest_user_message = ""
    for msg in reversed(transcript):
         if msg.get("role") == "user" and msg.get("content", "").strip():
            latest_user_message = msg["content"].strip()
         break

    if not latest_user_message:
        return {"response": "I didn't catch that. Could you repeat your question?"}

    # History = everything except last user message
    history = transcript[:-1] if len(transcript) > 1 else []

    answer = ask_question_for_voice(
        vectorstore=app.state.vectorstore,
        llm=app.state.llm,
        question=latest_user_message,
        history=history,
    )

    return {"response": answer}
# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("mains:app", host="0.0.0.0", port=port, reload=False)