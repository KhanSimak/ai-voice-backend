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
    # Normal case
    if "message" in body:
        return body["message"]

    # Retell case
    if "call" in body and "transcript" in body["call"]:
        transcript = body["call"]["transcript"]

        # get last user line
        lines = transcript.split("\n")
        for line in reversed(lines):
            if line.lower().startswith("user:"):
                return line.replace("User:", "").strip()

    return None

def _sanitize_rag_response(rag_response: str) -> str:
    lines = [l.strip() for l in rag_response.splitlines() if l.strip()]
    if not lines:
        return FALLBACK_NO_DATA
    return ", ".join(lines[:2])


def _build_rag_prompt(user_text: str) -> str:
    lower = user_text.lower()

    if user_text.lower() in {"name", "data", "list", "show", "give"}:
        return "List all available doctors."

    if "doctor" in lower:
        return f"List doctors: {user_text}"

    if "only" in lower or "just" in lower:
        return f"Filter doctors: {user_text}"

    return user_text

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