import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

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

class RetellMessage(BaseModel):
    role: str        # "user" or "assistant"
    content: str

class RetellRequest(BaseModel):
    call_id: str
    interaction_type: str   # "response_required" | "reminder_required" | "call_details" | "ping_pong"
    transcript: list[RetellMessage] = []
    response_id: Optional[int] = None

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
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))

    messages.append(HumanMessage(content=question))

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

@app.post("/retell-webhook", response_model=RetellResponse)
async def retell_webhook(body: RetellRequest):
    """
    Retell POSTs here during every live call turn.
    Uses the same FAISS RAG pipeline as /ask but voice-optimised.
    """
    logger.info(
        f"Retell [{body.interaction_type}] | call_id={body.call_id} "
        f"| turns={len(body.transcript)}"
    )

    # Return empty string for non-response interaction types
    if body.interaction_type not in ("response_required", "reminder_required"):
        return RetellResponse(response="")

    # Graceful fallback if vectorstore not ready
    if not hasattr(app.state, "vectorstore") or app.state.vectorstore is None:
        return RetellResponse(
            response="I'm just getting started. Please give me a moment and try again."
        )

    # Opening greeting when call starts with no transcript yet
    if not body.transcript:
        return RetellResponse(response="Hello! How can I help you today?")

    # Find the most recent user message
    latest_user_message = ""
    for msg in reversed(body.transcript):
        if msg.role == "user" and msg.content.strip():
            latest_user_message = msg.content.strip()
            break

    if not latest_user_message:
        return RetellResponse(response="I didn't catch that. Could you say that again?")

    # History = everything except the last message (which we pass as the question)
    history = body.transcript[:-1] if len(body.transcript) > 1 else []

    try:
        answer = ask_question_for_voice(
            vectorstore=app.state.vectorstore,
            llm=app.state.llm,
            question=latest_user_message,
            history=history,
        )
        logger.info(f"Response for call_id={body.call_id}: {answer[:100]}...")
        return RetellResponse(response=answer)

    except Exception as e:
        logger.error(f"Error in /retell-webhook call_id={body.call_id}: {e}", exc_info=True)
        return RetellResponse(
            response="I'm sorry, I ran into a technical issue. Please try again or speak with our team."
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("mains:app", host="0.0.0.0", port=port, reload=False)