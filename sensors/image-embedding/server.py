"""
Image Embedding Service — jina-clip-v2 on CPU.

Provides image and text embeddings in the same vector space (1024 dims).
Enables cross-modal search: text query finds similar images, image finds similar images.

Endpoints:
    POST /embed-image   — Embed an image (base64 or file path)
    POST /embed-text    — Embed text into the same space as images
    POST /embed-batch   — Embed multiple images at once
    GET  /info          — Model info and status
    GET  /health        — Health check
"""

import base64
import io
import logging
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image-embedding")

MODEL_ID = "jinaai/jina-clip-v2"
EMBEDDING_DIM = 1024
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info(f"Loading {MODEL_ID} (CPU)...")
    start = time.time()
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
    model.to("cpu")
    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s — {EMBEDDING_DIM}d embeddings")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Drift Image Embedding Service", lifespan=lifespan)


class ImageRequest(BaseModel):
    image: str  # base64-encoded image data
    normalize: bool = True


class TextRequest(BaseModel):
    text: str
    normalize: bool = True


class BatchImageRequest(BaseModel):
    images: list[str]  # list of base64-encoded images
    normalize: bool = True


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    dimensions: int
    model: str
    elapsed_ms: float


class BatchEmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    dimensions: int
    count: int
    model: str
    elapsed_ms: float


def decode_image(b64_data: str) -> Image.Image:
    """Decode base64 image data to PIL Image."""
    try:
        img_bytes = base64.b64decode(b64_data)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


@app.post("/embed-image", response_model=EmbeddingResponse)
async def embed_image(req: ImageRequest):
    """Embed a single image. Send base64-encoded image data."""
    start = time.time()
    img = decode_image(req.image)
    with torch.no_grad():
        emb = model.encode(img, normalize_embeddings=req.normalize)
    elapsed = (time.time() - start) * 1000
    return EmbeddingResponse(
        embedding=emb.tolist(),
        dimensions=len(emb),
        model=MODEL_ID,
        elapsed_ms=round(elapsed, 1),
    )


@app.post("/embed-text", response_model=EmbeddingResponse)
async def embed_text(req: TextRequest):
    """Embed text into the same vector space as images."""
    start = time.time()
    with torch.no_grad():
        emb = model.encode(req.text, normalize_embeddings=req.normalize)
    elapsed = (time.time() - start) * 1000
    return EmbeddingResponse(
        embedding=emb.tolist(),
        dimensions=len(emb),
        model=MODEL_ID,
        elapsed_ms=round(elapsed, 1),
    )


@app.post("/embed-batch", response_model=BatchEmbeddingResponse)
async def embed_batch(req: BatchImageRequest):
    """Embed multiple images at once."""
    start = time.time()
    images = [decode_image(b64) for b64 in req.images]
    with torch.no_grad():
        embs = model.encode(images, normalize_embeddings=req.normalize, batch_size=4)
    elapsed = (time.time() - start) * 1000
    return BatchEmbeddingResponse(
        embeddings=[e.tolist() for e in embs],
        dimensions=EMBEDDING_DIM,
        count=len(images),
        model=MODEL_ID,
        elapsed_ms=round(elapsed, 1),
    )


@app.get("/info")
async def info():
    return {
        "model_id": MODEL_ID,
        "dimensions": EMBEDDING_DIM,
        "device": "cpu",
        "capabilities": ["image", "text"],
        "cross_modal": True,
        "status": "ready" if model else "loading",
    }


@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}
