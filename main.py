import logging
from logging.handlers import RotatingFileHandler
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
import faiss
import numpy as np
import tiktoken
import uuid
import time
import threading
from typing import List, Optional
import requests
import os
import pickle   # ⭐ ADDED

# -----------------------------------
# FLASK + LOGGING CONFIGURATION
# -----------------------------------
app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

handler = RotatingFileHandler("server.log", maxBytes=1_000_000, backupCount=5)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
app.logger.addHandler(handler)

app.logger.info("🚀 Flask service starting...")

# -----------------------------------
# EMBEDDING CACHE (SERVER-SIDE)
# -----------------------------------
CACHE_FILE = "embedding_cache.pkl"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        embedding_cache = pickle.load(f)
else:
    embedding_cache = {}

def save_cache():
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(embedding_cache, f)

app.logger.info(f"📚 Cache loaded with {len(embedding_cache)} items")

# -----------------------------------
# NOMIC CLOUD EMBEDDINGS (WITH CACHE)
# -----------------------------------
NOMIC_URL = "https://api-atlas.nomic.ai/v1/embedding/text"

def get_cloud_embedding(texts: List[str], api_key: str) -> np.ndarray:
    """
    Uses server-side cache to prevent repeated API calls.
    Only fetches embeddings not already in cache.
    """
    global embedding_cache

    cached = []
    missing = []
    missing_idx = []

    # 1. CHECK CACHE
    for i, text in enumerate(texts):
        if text in embedding_cache:
            cached.append((i, embedding_cache[text]))
        else:
            missing.append(text)
            missing_idx.append(i)

    fetched = []

    # 2. FETCH ONLY MISSING EMBEDDINGS
    if missing:
        app.logger.info(f"📡 Fetching {len(missing)} embeddings from Nomic")

        response = requests.post(
            NOMIC_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={"texts": missing},
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(response.text)

        data = response.json()
        fetched = data["embeddings"]

        # SAVE NEW ITEMS TO CACHE
        for idx, emb in zip(missing_idx, fetched):
            embedding_cache[texts[idx]] = np.array(emb, dtype=np.float32)

        save_cache()  # persist to disk

    # 3. REBUILD ORDERED ARRAY
    final = [None] * len(texts)

    for idx, emb in cached:
        final[idx] = np.array(emb, dtype=np.float32)

    for idx, emb in zip(missing_idx, fetched):
        final[idx] = np.array(emb, dtype=np.float32)

    return np.vstack(final)

# -----------------------------------
# USER SESSION MODELS
# -----------------------------------
SESSION_TIMEOUT = 1800
user_sessions = {}

class TextInput(BaseModel):
    text: str
    user_id: Optional[str] = None
    api_key: str

class QueryInput(BaseModel):
    query: str
    top_k: int = 5
    user_id: str
    api_key: str

class DeleteSessionInput(BaseModel):
    user_id: str

# -----------------------------------
# TEXT CHUNKING
# -----------------------------------
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []

    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + chunk_size]
        chunks.append(encoding.decode(chunk))
        i += chunk_size - overlap

    return chunks

# -----------------------------------
# INITIALIZE VECTOR STORE
# -----------------------------------
@app.route('/initialize', methods=['POST'])
def build_vector_store():
    payload = TextInput(**request.json)
    user_id = payload.user_id or str(uuid.uuid4())

    text_chunks = chunk_text(payload.text)
    embeddings = get_cloud_embedding(text_chunks, payload.api_key)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    user_sessions[user_id] = {
        "index": index,
        "chunks": text_chunks,
        "last_accessed": time.time(),
    }

    return jsonify({"status": "success", "user_id": user_id})

# -----------------------------------
# QUERY VECTOR STORE
# -----------------------------------
@app.route('/query_vector_store', methods=['POST'])
def query_vector_store():
    payload = QueryInput(**request.json)

    if payload.user_id not in user_sessions:
        return jsonify({"error": "User session not found"}), 404

    session = user_sessions[payload.user_id]

    query_emb = get_cloud_embedding([payload.query], payload.api_key)[0]

    D, I = session["index"].search(
        np.array([query_emb], dtype=np.float32),
        payload.top_k
    )

    results = [session["chunks"][idx] for idx in I[0]]

    session["last_accessed"] = time.time()

    return jsonify({"matches": results})

# -----------------------------------
# DELETE SESSION
# -----------------------------------
@app.route('/delete_session', methods=['POST'])
def delete_session():
    payload = DeleteSessionInput(**request.json)
    user_sessions.pop(payload.user_id, None)
    return jsonify({"status": "success"})

# -----------------------------------
# CLEANUP THREAD
# -----------------------------------
def cleanup_sessions():
    while True:
        now = time.time()
        expired = [
            uid for uid, s in user_sessions.items()
            if now - s["last_accessed"] > SESSION_TIMEOUT
        ]
        for uid in expired:
            del user_sessions[uid]
        time.sleep(600)

threading.Thread(target=cleanup_sessions, daemon=True).start()

# -----------------------------------
# START FLASK SERVER
# -----------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
