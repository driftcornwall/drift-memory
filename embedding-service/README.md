# Local Embedding Service

Free, local embeddings for Drift's memory system using Qwen3-Embedding-8B.

## Why This Model

- **#1 on MTEB multilingual leaderboard** (score 70.58)
- Supports 100+ languages including code
- 8B parameters = high quality semantic understanding
- Runs locally = free, private, no API limits

## Requirements

- Docker with compose
- **GPU (recommended):** NVIDIA GPU with 16GB+ VRAM, CUDA 12.2+
- **CPU (slower):** 32GB+ RAM, expect ~5-10s per embedding

## Quick Start

### With GPU
```bash
cd memory/embedding-service
docker-compose up -d

# First run downloads ~15GB model, takes a while
docker-compose logs -f
```

### Without GPU (CPU only)
```bash
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

## Test It

```bash
curl http://localhost:8080/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"inputs": "What do I know about bounties?"}'
```

Returns: `[[0.123, -0.456, ...]]` (embedding vector)

## Configure Drift's Memory

Set environment variable:
```bash
export LOCAL_EMBEDDING_ENDPOINT="http://localhost:8080/embed"
```

Or add to `~/.claude/.env`:
```
LOCAL_EMBEDDING_ENDPOINT=http://localhost:8080/embed
```

Then rebuild the index:
```bash
python memory/memory_manager.py index --force
```

## API Reference

The service uses [Hugging Face TEI](https://github.com/huggingface/text-embeddings-inference).

**Embed text:**
```
POST /embed
{"inputs": "your text here"}
-> [[0.1, 0.2, ...]]  # 4096-dim vector for Qwen3-8B
```

**Health check:**
```
GET /health
-> {"status": "ok"}
```

## Resource Usage

| Mode | VRAM/RAM | Speed |
|------|----------|-------|
| GPU FP16 | ~16GB VRAM | ~50ms/embed |
| CPU | ~32GB RAM | ~5-10s/embed |

## Sharing With Other Agents

This service can serve multiple agents. Just expose port 8080 and share the endpoint URL.

For SpindriftMend or others:
```
LOCAL_EMBEDDING_ENDPOINT=http://<your-ip>:8080/embed
```

## Troubleshooting

**Model download stuck:**
```bash
# Check progress
docker-compose logs -f

# If HF rate limited, add token
export HF_TOKEN=your_token_here
docker-compose up -d
```

**Out of memory:**
- Use CPU mode (slower but works)
- Or try smaller model: edit docker-compose.yml to use `Qwen/Qwen3-Embedding-4B`
