#!/usr/bin/env python3
"""
Persistent Semantic Search Server

Pre-loads embeddings into memory on startup. Responds to search queries
via HTTP in ~500ms instead of ~3s (eliminates Python subprocess overhead).

Usage:
    python search_server.py                  # Start on port 8082
    python search_server.py --port 8083      # Custom port
    python search_server.py --daemon         # Run as background process

Endpoints:
    POST /search     {"query": "...", "limit": 5, "threshold": 0.65}
    POST /recall     {"ids": ["id1", "id2"]}  — register recalls
    GET  /health     — check server status
    POST /reload     — reload embeddings from disk

Hook integration:
    Hooks check localhost:8082/health first. If up, use HTTP.
    If down, fall back to subprocess (existing behavior).
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Lock

sys.stdout.reconfigure(encoding="utf-8")

MEMORY_DIR = Path(__file__).parent
EMBEDDINGS_FILE = MEMORY_DIR / "embeddings.json"
EMBEDDING_API = "http://localhost:8080/v1/embeddings"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_PORT = 8082

# --- Global state (loaded once, reloaded on /reload) ---
_lock = Lock()
_memory_ids: list[str] = []
_matrix: np.ndarray | None = None
_norms: np.ndarray | None = None
_previews: dict[str, str] = {}
_load_time: float = 0
_memory_count: int = 0


def load_embeddings():
    """Load embeddings from disk into memory."""
    global _memory_ids, _matrix, _norms, _previews, _load_time, _memory_count

    t0 = time.time()
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    memories = data.get("memories", {})
    _memory_ids = list(memories.keys())
    _previews = {mid: memories[mid].get("preview", "") for mid in _memory_ids}

    # Build matrix and pre-compute norms
    _matrix = np.array(
        [memories[mid]["embedding"] for mid in _memory_ids], dtype=np.float32
    )
    _norms = np.linalg.norm(_matrix, axis=1)
    _norms[_norms < 1e-10] = 1e-10  # avoid division by zero

    _load_time = time.time() - t0
    _memory_count = len(_memory_ids)


def get_query_embedding(query: str) -> np.ndarray | None:
    """Get embedding vector for a query string via the local API."""
    import requests

    try:
        resp = requests.post(
            EMBEDDING_API,
            json={"input": [query], "model": EMBEDDING_MODEL},
            timeout=5,
        )
        if resp.status_code == 200:
            vec = resp.json()["data"][0]["embedding"]
            return np.array(vec, dtype=np.float32)
    except Exception:
        pass
    return None


def search(query: str, limit: int = 5, threshold: float = 0.0) -> list[dict]:
    """Search memories by cosine similarity."""
    query_vec = get_query_embedding(query)
    if query_vec is None:
        return []

    with _lock:
        q_norm = np.linalg.norm(query_vec)
        if q_norm < 1e-10:
            return []
        sims = (_matrix @ query_vec) / (_norms * q_norm)
        top_idx = np.argsort(sims)[::-1][:limit]

        results = []
        for idx in top_idx:
            score = float(sims[idx])
            if score >= threshold:
                mid = _memory_ids[idx]
                results.append(
                    {"id": mid, "score": score, "preview": _previews.get(mid, "")}
                )
    return results


def register_recalls(ids: list[str]):
    """Register memory IDs as recalled in session state."""
    import subprocess

    try:
        subprocess.run(
            [sys.executable, str(MEMORY_DIR / "memory_manager.py"), "register-recall"]
            + ids,
            capture_output=True,
            text=True,
            timeout=3,
            cwd=str(MEMORY_DIR),
        )
    except Exception:
        pass


class SearchHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = json.dumps(
                {
                    "status": "ok",
                    "memories": _memory_count,
                    "load_time": round(_load_time, 3),
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length > 0 else b"{}"

        if self.path == "/search":
            try:
                body = json.loads(raw)
                query = body.get("query", "")
                limit = body.get("limit", 5)
                threshold = body.get("threshold", 0.0)

                results = search(query, limit=limit, threshold=threshold)

                resp = json.dumps({"results": results}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            except Exception as e:
                resp = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        elif self.path == "/recall":
            try:
                body = json.loads(raw)
                ids = body.get("ids", [])
                if ids:
                    register_recalls(ids)
                resp = json.dumps({"registered": len(ids)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            except Exception as e:
                resp = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        elif self.path == "/reload":
            try:
                with _lock:
                    load_embeddings()
                resp = json.dumps(
                    {"reloaded": _memory_count, "load_time": round(_load_time, 3)}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            except Exception as e:
                resp = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default request logging."""
        pass


def main():
    parser = argparse.ArgumentParser(description="Persistent semantic search server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--daemon", action="store_true", help="Print PID and detach (Windows: just run)"
    )
    args = parser.parse_args()

    print(f"Loading embeddings from {EMBEDDINGS_FILE}...")
    load_embeddings()
    print(f"Loaded {_memory_count} memories in {_load_time:.2f}s")

    server = HTTPServer(("127.0.0.1", args.port), SearchHandler)
    print(f"Search server listening on http://127.0.0.1:{args.port}")
    print(f"  POST /search  — query memories")
    print(f"  POST /recall  — register recalls")
    print(f"  GET  /health  — server status")
    print(f"  POST /reload  — refresh embeddings")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
