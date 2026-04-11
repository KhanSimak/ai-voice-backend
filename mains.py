import logging
import json
from fastapi import FastAPI, Request

from rag import load_vectorstore, get_llm, ask_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Retell AI Voice Agent")


# -----------------------------
# SAFE GLOBAL INIT
# -----------------------------
try:
    vectorstore = load_vectorstore()
    llm = get_llm()
    logger.info("✅ RAG system loaded successfully")
except Exception as e:
    logger.error(f"❌ Startup failed: {e}")
    vectorstore = None
    llm = None


# -----------------------------
# SAFE JSON PARSER
# -----------------------------
async def safe_get_body(request: Request):
    try:
        return await request.json()
    except Exception:
        try:
            raw = await request.body()
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}


# -----------------------------
# CLEAN RETELL TRANSCRIPT
# -----------------------------
def extract_user_text(body: dict) -> str:
    text = ""

    if "message" in body:
        text = body["message"]

    elif "call" in body:
        text = body["call"].get("transcript", "")

    if not text:
        return ""

    # remove noise
    text = text.replace("User:", "")
    text = text.replace("Agent:", "")
    text = text.strip()

    if len(text) < 2:
        return ""

    return text


# -----------------------------
# CHAT ENDPOINT
# -----------------------------
@app.post("/chat")
async def chat(request: Request):
    try:
        if not vectorstore or not llm:
            return {"message": "System not ready"}

        body = await safe_get_body(request)

        logger.info(f"📩 RAW BODY: {body}")

        user_message = extract_user_text(body)

        if not user_message:
            return {"message": "No valid input received"}

        logger.info(f"👤 CLEAN USER: {user_message}")

        answer = ask_question(vectorstore, llm, user_message)

        logger.info(f"🤖 AI: {answer}")

        return {"message": answer}

    except Exception as e:
        logger.error(f"❌ CHAT ERROR: {str(e)}")
        return {"message": "Internal server error"}


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rag_ready": vectorstore is not None,
        "llm_ready": llm is not None
    }