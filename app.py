from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tiktoken
import uuid
import time
import threading
from typing import List, Optional
import os

app = Flask(__name__)
CORS(app)  # Allow all origins

model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

SESSION_TIMEOUT = 1800  # 30 minutes
user_sessions = {}

class TextInput(BaseModel):
    text: str
    user_id: Optional[str] = None

class QueryInput(BaseModel):
    query: str
    top_k: int = 5
    user_id: str

class DeleteSessionInput(BaseModel):
    user_id: str

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        i += chunk_size - overlap
    return chunks

def embed_chunks(chunks: List[str]) -> np.ndarray:
    return model.encode(chunks, show_progress_bar=False)

def embed_query(query: str) -> np.ndarray:
    return model.encode([query])[0]

def update_last_access(user_id: str):
    if user_id in user_sessions:
        user_sessions[user_id]["last_accessed"] = time.time()

@app.route('/initialize', methods=['POST'])
def build_vector_store():
    try:
        payload = TextInput(**request.json)
    except ValidationError as e:
        return jsonify({'error': e.errors()}), 400

    user_id = payload.user_id or str(uuid.uuid4())
    text_chunks = chunk_text(payload.text)

    if not text_chunks:
        return jsonify({"error": "No text chunks found after chunking."}), 400

    embeddings = embed_chunks(text_chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype(np.float32))

    user_sessions[user_id] = {
        "index": index,
        "chunks": text_chunks,
        "last_accessed": time.time(),
    }

    return jsonify({
        "status": "success",
        "user_id": user_id,
        "chunks_stored": len(text_chunks)
    })

@app.route('/query_vector_store', methods=['POST'])
def query_vector_store():
    try:
        payload = QueryInput(**request.json)
    except ValidationError as e:
        return jsonify({'error': e.errors()}), 400

    user_id = payload.user_id
    if user_id not in user_sessions:
        return jsonify({"error": f"User session '{user_id}' not found."}), 404

    update_last_access(user_id)
    session = user_sessions[user_id]
    vector_index = session["index"]
    text_chunks = session["chunks"]

    query_embedding = embed_query(payload.query).astype(np.float32)
    D, I = vector_index.search(np.array([query_embedding]), payload.top_k)

    results = []
    for idx in I[0]:
        if idx < len(text_chunks):
            results.append(text_chunks[idx])

    return jsonify({"matches": results})

@app.route('/delete_session', methods=['POST'])
def delete_session():
    try:
        payload = DeleteSessionInput(**request.json)
    except ValidationError as e:
        return jsonify({'error': e.errors()}), 400

    user_id = payload.user_id
    if user_id in user_sessions:
        del user_sessions[user_id]
        return jsonify({
            "status": "success",
            "message": f"Session '{user_id}' deleted."
        })
    else:
        return jsonify({"error": f"User session '{user_id}' not found."}), 404

def cleanup_sessions():
    while True:
        now = time.time()
        expired_users = []
        for user_id, session in list(user_sessions.items()):
            if now - session["last_accessed"] > SESSION_TIMEOUT:
                expired_users.append(user_id)
        for user_id in expired_users:
            del user_sessions[user_id]
            print(f"Session '{user_id}' expired and deleted.")
        time.sleep(600)  # Check every 10 minutes

# Start background session cleanup thread
cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
