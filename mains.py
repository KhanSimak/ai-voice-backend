import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Retell AI Voice Agent")


# ---------------------------------------------------------------------------
# Replace with your real RAG
# ---------------------------------------------------------------------------
def rag_query(query: str) -> str:
    raise NotImplementedError("Implement rag_query() with your RAG pipeline.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOCTOR_TRIGGERS = {"doctor", "doctors", "physician", "specialist"}
PARTIAL_TRIGGERS = {"name", "data", "list", "show", "give"}

FALLBACK_EMPTY = "Sorry, I didn't catch that. Can you repeat?"
FALLBACK_NO_DATA = "No doctors found right now."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_message(body: dict) -> str | None:
    # 1. Direct message (if Retell sends it)
    msg = body.get("message") or body.get("text") or body.get("query")
    if msg:
        return msg.strip()

    # 2. Handle Retell call object
    call = body.get("call", {})
    transcript = call.get("transcript_object", [])

    user_messages = [
        item.get("content", "").strip()
        for item in transcript
        if item.get("role") == "user"
    ]

    if user_messages:
        return user_messages[-1]

    return None

def _sanitize_rag_response(rag_response: str) -> str:
    lines = [l.strip() for l in rag_response.splitlines() if l.strip()]
    if not lines:
        return FALLBACK_NO_DATA
    return ", ".join(lines[:2])


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post("/chat")
async def chat(request: Request):

    try:
        body = await request.json()
    except Exception:
      try:
        raw = await request.body()
        body = {"message": raw.decode("utf-8")} if raw else {}
      except Exception:
        body = {}

    logger.info(f"Request: {body}")

    user_message = _extract_message(body)

    if not user_message:
        return {"message": FALLBACK_EMPTY}

    rag_prompt = _build_rag_prompt(user_message)
    logger.info(f"RAG query: {rag_prompt}")

    try:
        rag_response = rag_query(rag_prompt)
    except NotImplementedError:
        return {"message": "RAG not configured"}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"message": "Error fetching data"}

    final_response = _sanitize_rag_response(rag_response)

    logger.info(f"Final: {final_response}")
    return {"message": final_response}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}