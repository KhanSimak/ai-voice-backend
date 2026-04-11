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


# -----------------------------
# CHAT ENDPOINT
# -----------------------------
@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    logger.info(f"📩 RAW BODY: {body}")

    # 🔥 SAFE EXTRACTION
    user_message = ""

    if isinstance(body, dict):
        user_message = body.get("message", "")

        # fallback for retell transcript
        if not user_message and "call" in body:
            try:
                user_message = body["call"].get("transcript", "")
            except:
                user_message = ""

    user_message = user_message.strip()

    logger.info(f"👤 USER: {user_message}")

    if not user_message:
        return JSONResponse(
            content={"message": "Empty request received"},
            status_code=200
        )

    answer = ask_question(vectorstore, llm, user_message)

    logger.info(f"🤖 AI: {answer}")

    return {"message": answer}
# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}