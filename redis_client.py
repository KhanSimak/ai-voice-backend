import redis
import json
import os

REDIS_URL = "redis://default:gQAAAAAAAV4hAAIncDFkYjlhOGE0NmRlNDA0YjA3YWM0NzJiZTM5N2E1NzJjMHAxODk2MzM@charming-worm-89633.upstash.io:6379"

r = redis.from_url(REDIS_URL, decode_responses=True)

def get_history(call_id):
    data = r.get(call_id)
    if data:
        return json.loads(data)
    return []

def append_message(call_id, role, content):
    history = get_history(call_id)
    history.append({"role": role, "content": content})
    r.set(call_id, json.dumps(history))


