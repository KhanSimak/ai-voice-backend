import redis
import json
import os

REDIS_URL = os.getenv("REDIS_URL")

# Create Redis client safely
if REDIS_URL:
    r = redis.from_url(REDIS_URL, decode_responses=True)
    try:
        r.ping()
        print("✅ Redis connected")
    except Exception as e:
        print("❌ Redis connection failed:", e)
        r = None
else:
    print("⚠️ REDIS_URL not set")
    r = None


# ---------------- GET HISTORY ---------------- #
def get_history(call_id):
    if not r:
        return []

    try:
        data = r.get(call_id)
        if data:
            return json.loads(data)
    except Exception as e:
        print("Redis GET error:", e)

    return []


# ---------------- APPEND MESSAGE ---------------- #
def append_message(call_id, role, content):
    if not r:
        return

    try:
        history = get_history(call_id)
        history.append({"role": role, "content": content})
        r.set(call_id, json.dumps(history))
    except Exception as e:
        print("Redis SET error:", e)