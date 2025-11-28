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

# -----------------------------------
# FLASK + LOGGING CONFIGURATION
# -----------------------------------
app = Flask(__name__)
CORS(app)

# Logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# rotating file log (works on Render)
handler = RotatingFileHandler("server.log", maxBytes=1_000_000, backupCount=5)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
app.logger.addHandler(handler)

app.logger.info("🚀 Flask service starting...")


# -----------------------------------
# NOMIC CLOUD EMBEDDINGS (CORRECT AUTH)
# -----------------------------------
NOMIC_URL = "https://api-atlas.nomic.ai/v1/embedding/text"

def get_cloud_embedding(texts: List[str], api_key: str) -> np.ndarray:
    """
    Calls Nomic Embedding API to generate embeddings.
    API key must be sent in Authorization header (Bearer).
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    body = {"texts": texts}

    app.logger.info(f"📡 Sending embedding request to Nomic ({len(texts)} chunks)")

    try:
        response = requests.post(
            NOMIC_URL,
            headers=headers,
            json=body,
            timeout=30
        )
    except Exception as e:
        app.logger.error("❌ Request error calling Nomic API:")
        app.logger.error(str(e))
        raise

    app.logger.info(f"📥 Nomic Response Status: {response.status_code}")

    if response.status_code != 200:
        app.logger.error(f"❌ Nomic API Error Body: {response.text}")
        raise Exception(f"Nomic API Error: {response.status_code} {response.text}")

    data = response.json()

    if "embeddings" not in data:
        app.logger.error(f"❌ Missing embeddings in Nomic response: {data}")
        raise Exception("Malformed Nomic API response: no embeddings")

    embeds = np.array(data["embeddings"], dtype=np.float32)
    if embeds.ndim == 1:
        embeds = np.expand_dims(embeds, axis=0)

    app.logger.info("✅ Embeddings received successfully.")
    return embeds


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
    app.logger.info("✂️ Chunking text...")
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []

    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + chunk_size]
        chunks.append(encoding.decode(chunk))
        i += chunk_size - overlap

    app.logger.info(f"📦 Total chunks created: {len(chunks)}")
    return chunks


# -----------------------------------
# INITIALIZE VECTOR STORE
# -----------------------------------
@app.route('/initialize', methods=['POST'])
def build_vector_store():
    try:
        app.logger.info("📥 /initialize request received")
        app.logger.info(f"Request JSON: {request.json}")

        payload = TextInput(**request.json)
    except ValidationError as e:
        app.logger.error("❌ Validation error in /initialize")
        return jsonify({'error': e.errors()}), 400
    except Exception:
        app.logger.error("❌ Unexpected error in /initialize")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': 'Invalid request'}), 400

    user_id = payload.user_id or str(uuid.uuid4())
    app.logger.info(f"🆔 Creating new session: {user_id}")

    try:
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

        app.logger.info(f"✅ Session {user_id} initialized successfully.")
        return jsonify({
            "status": "success",
            "user_id": user_id,
            "chunks_stored": len(text_chunks)
        })

    except Exception as e:
        app.logger.error("❌ Error in /initialize:")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# -----------------------------------
# QUERY VECTOR STORE
# -----------------------------------
@app.route('/query_vector_store', methods=['POST'])
def query_vector_store():
    try:
        app.logger.info("📥 /query_vector_store request received")
        app.logger.info(f"Request JSON: {request.json}")

        payload = QueryInput(**request.json)
    except ValidationError as e:
        app.logger.error("❌ Validation error in query")
        return jsonify({'error': e.errors()}), 400
    except Exception:
        app.logger.error("❌ Unexpected error parsing request")
        return jsonify({'error': 'Invalid request'}), 400

    if payload.user_id not in user_sessions:
        app.logger.warning(f"⚠️ Missing session: {payload.user_id}")
        return jsonify({"error": "User session not found"}), 404

    session = user_sessions[payload.user_id]

    try:
        query_embedding = get_cloud_embedding([payload.query], payload.api_key)[0]

        index = session["index"]
        chunks = session["chunks"]

        D, I = index.search(
            np.array([query_embedding], dtype=np.float32),
            payload.top_k
        )

        results = [chunks[idx] for idx in I[0] if idx < len(chunks)]

        session["last_accessed"] = time.time()

        app.logger.info(f"🔍 Query success for user {payload.user_id}")
        return jsonify({"matches": results})

    except Exception as e:
        app.logger.error("❌ Error in /query_vector_store")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# -----------------------------------
# DELETE SESSION
# -----------------------------------
@app.route('/delete_session', methods=['POST'])
def delete_session():
    try:
        payload = DeleteSessionInput(**request.json)
    except:
        return jsonify({"error": "Invalid request"}), 400

    if payload.user_id in user_sessions:
        del user_sessions[payload.user_id]
        app.logger.info(f"🗑️ Deleted session {payload.user_id}")
        return jsonify({"status": "success"})

    return jsonify({"error": "User session not found"}), 404


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
            app.logger.info(f"🧹 Auto-deleted expired session {uid}")
        time.sleep(600)

threading.Thread(target=cleanup_sessions, daemon=True).start()


# -----------------------------------
# START FLASK USING RENDER PORT
# -----------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.logger.info(f"🚀 Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port)
