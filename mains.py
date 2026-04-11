import logging
import json
from fastapi import FastAPI, Request

from rag import load_vectorstore, get_llm, ask_question, should_use_rag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Voice AI RAG System")


# ----------------------------
# SAFE INIT
# ----------------------------
try:
    vectorstore = load_vectorstore()
    llm = get_llm()
    logger.info("✅ RAG Loaded")
except Exception as e:
    logger.error(f"❌ INIT FAILED: {e}")
    vectorstore = None
    llm = None


# ----------------------------
# SAFE BODY PARSER
# ----------------------------
async def safe_body(request: Request):
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


# ----------------------------
# EXTRACT RETELL TEXT
# ----------------------------
def extract_text(body: dict) -> str:
    text = ""

    if "message" in body:
        text = body["message"]

    elif "call" in body:
        text = body["call"].get("transcript", "")

    text = text.replace("User:", "").replace("Agent:", "").strip()

    return text


# ----------------------------
# CHAT ENDPOINT
# ----------------------------
@app.post("/chat")
async def chat(request: Request):
    try:
        if not vectorstore or not llm:
            return {"message": "System not ready"}

        body = await safe_body(request)

        logger.info(f"📩 BODY: {body}")

        user_text = extract_text(body)

        if not user_text:
            return {"message": "No input"}

        logger.info(f"👤 USER: {user_text}")

        # 🔥 IMPORTANT: SKIP RAG FOR SMALL TALK
        if not should_use_rag(user_text):
            logger.info("🚫 Skipping RAG")
            return {"message": "Hi! I can help you with doctors or appointments."}

        answer = ask_question(vectorstore, llm, user_text)

        logger.info(f"🤖 AI: {answer}")

        return {"message": answer}

    except Exception as e:
        logger.error(f"❌ CHAT ERROR: {e}")
        return {"message": "Internal error"}


# ----------------------------
# HEALTH
# ----------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rag": vectorstore is not None,
        "llm": llm is not None
    }