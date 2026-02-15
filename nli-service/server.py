"""
NLI Inference Service — DeBERTa-v3-xsmall for contradiction/entailment detection.

Drift Memory System — Mechanical Skepticism Layer
Port: 8082

Endpoints:
    POST /classify       - Classify premise-hypothesis pair
    POST /classify-batch - Batch classification (up to 32 pairs)
    GET  /health         - Health check
    GET  /info           - Model info and status
"""

import time
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Drift NLI Service", version="1.0.0")

MODEL_ID = "cross-encoder/nli-deberta-v3-xsmall"
LABELS = ["contradiction", "entailment", "neutral"]
model = None


class NLIPair(BaseModel):
    premise: str
    hypothesis: str


class NLIBatchRequest(BaseModel):
    pairs: list[NLIPair]


class ClassificationResult(BaseModel):
    label: str
    scores: dict
    elapsed_ms: float


class BatchResult(BaseModel):
    results: list[ClassificationResult]
    total_elapsed_ms: float


@app.on_event("startup")
async def startup():
    global model
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(MODEL_ID)
    print(f"Loaded {MODEL_ID} on {'cuda' if torch.cuda.is_available() else 'cpu'}")


@app.post("/classify", response_model=ClassificationResult)
async def classify(req: NLIPair):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    start = time.time()
    raw = model.predict([(req.premise[:2000], req.hypothesis[:2000])])
    elapsed = (time.time() - start) * 1000

    # Handle both old (raw logits) and new (softmax'd) CrossEncoder output
    scores = raw[0]
    score_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)

    # If scores aren't probabilities (don't sum to ~1), apply softmax
    if abs(sum(score_list) - 1.0) > 0.1:
        import torch.nn.functional as F
        probs = F.softmax(torch.tensor(score_list), dim=0)
        score_list = probs.tolist()

    label_idx = score_list.index(max(score_list))

    return ClassificationResult(
        label=LABELS[label_idx],
        scores=dict(zip(LABELS, [round(s, 4) for s in score_list])),
        elapsed_ms=round(elapsed, 1),
    )


@app.post("/classify-batch", response_model=BatchResult)
async def classify_batch(req: NLIBatchRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    if len(req.pairs) > 32:
        raise HTTPException(400, "Maximum 32 pairs per batch")

    start = time.time()
    pairs = [(p.premise[:2000], p.hypothesis[:2000]) for p in req.pairs]
    all_scores = model.predict(pairs)
    total_elapsed = (time.time() - start) * 1000

    results = []
    for scores in all_scores:
        score_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        # Softmax if needed
        if abs(sum(score_list) - 1.0) > 0.1:
            import torch.nn.functional as F
            probs = F.softmax(torch.tensor(score_list), dim=0)
            score_list = probs.tolist()
        label_idx = score_list.index(max(score_list))
        results.append(ClassificationResult(
            label=LABELS[label_idx],
            scores=dict(zip(LABELS, [round(s, 4) for s in score_list])),
            elapsed_ms=round(total_elapsed / len(req.pairs), 1),
        ))

    return BatchResult(
        results=results,
        total_elapsed_ms=round(total_elapsed, 1),
    )


@app.get("/health")
async def health():
    return {
        "status": "ready" if model is not None else "loading",
        "model": MODEL_ID,
    }


@app.get("/info")
async def info():
    return {
        "model_id": MODEL_ID,
        "labels": LABELS,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "status": "ready" if model is not None else "loading",
        "max_batch_size": 32,
        "max_input_length": 2000,
    }
