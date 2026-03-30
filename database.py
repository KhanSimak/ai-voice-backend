import os

print("🔥 ALL ENV VARS:", dict(os.environ))   # shows everything

DATABASE_URL = os.environ.get("DATABASE_URL")

print("🔥 DATABASE_URL VALUE:", DATABASE_URL)

raise Exception("STOP HERE")  # stop app intentionally