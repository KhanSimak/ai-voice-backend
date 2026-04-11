import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag import load_vectorstore, get_llm, ask_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Retell AI Voice Agent")


# -----------------------------
# GLOBAL INIT (ONLY ONCE)
# -----------------------------
vectorstore = load_vectorstore()
llm = get_llm()



async def safe_get_body(request: Request):
    try:
        return await request.json()
    except Exception:
        raw = await request.body()
        if not raw:
            return {}

        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

# -----------------------------
# CHAT ENDPOINT
# -----------------------------
@app.post("/chat")
async def chat(request: Request):
    try:
        body = await safe_get_body(request)

        logger.info(f"📩 RAW BODY: {body}")

        user_message = body.get("message", "")

        if not user_message:
            user_message = body.get("call", {}).get("transcript", "")

        if not user_message:
            return {"message": "No input received"}

        logger.info(f"👤 USER: {user_message}")

        answer = ask_question(vectorstore, llm, user_message)

        logger.info(f"🤖 AI: {answer}")

        return {"message": answer}

    except Exception as e:
        logger.error(f"❌ CHAT ERROR: {str(e)}")
        return {"message": "Internal error"}
# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}