# Setup Guide

Complete guide to setting up drift-memory for your AI agent.

## Prerequisites

- **Python 3.11+**
- **PostgreSQL 15+ with pgvector** (required -- no file-based fallback)
- **Docker** (for embedding services, NLI, ollama)
- **Claude Code** (or similar agent runtime with hooks support)

## Architecture

drift-memory runs as a PostgreSQL-backed cognitive architecture with 6 Docker services:

| Service | Port | Purpose | GPU |
|---------|------|---------|-----|
| PostgreSQL + pgvector | 5433 | All memory storage | No |
| Text Embedding (Qwen3) | 8080 | Semantic search embeddings | Yes (CPU fallback) |
| Image Embedding (jina-clip-v2) | 8081 | Visual memory | Yes |
| NLI (DeBERTa) | 8082 | Contradiction detection | Yes |
| Consolidation Daemon | 8083 | Background consolidation | No |
| Ollama (Gemma 3 4B) | 11434 | Inner monologue, topic classification | Yes |

## Installation

### 1. Clone

```bash
git clone https://github.com/driftcornwall/drift-memory.git
cd drift-memory
```

### 2. Start PostgreSQL with pgvector

```bash
# Using Docker (recommended)
docker run -d \
  --name drift-db \
  -e POSTGRES_DB=agent_memory \
  -e POSTGRES_USER=agent_admin \
  -e POSTGRES_PASSWORD=agent_memory_local_dev \
  -p 5433:5432 \
  pgvector/pgvector:pg16

# Or use an existing PostgreSQL 15+ with pgvector extension
```

### 3. Initialize Database Schema

```bash
# This creates the drift schema and all tables automatically
python db_adapter.py
```

The adapter creates: `memories`, `edges_v3`, `context_graphs`, `knowledge_graph`, `text_embeddings`, `image_embeddings`, `rejections`, `key_value`, `social_interactions`, `vitals`, `lessons`, `somatic_markers`, `session_events`, and more.

### 4. Start Embedding Services

```bash
# Text embeddings (required for semantic search)
cd embedding-service && docker-compose up -d && cd ..

# NLI service (optional -- enables contradiction detection)
cd nli-service && docker-compose up -d && cd ..

# Ollama with Gemma 3 4B (optional -- enables inner monologue, topic classification)
# Install ollama: https://ollama.ai
ollama pull gemma3:4b
```

### 5. Verify

```bash
python toolkit.py health
```

You should see all modules reporting healthy. Services that aren't running will show as degraded (the system fails gracefully).

## Hook Integration

drift-memory hooks into Claude Code's lifecycle events. Copy the hooks to your Claude Code hooks directory:

```bash
# Copy hooks
cp hooks/session_start.py ~/.claude/hooks/
cp hooks/stop.py ~/.claude/hooks/
cp hooks/post_tool_use.py ~/.claude/hooks/
cp hooks/user_prompt_submit.py ~/.claude/hooks/

# Update MEMORY_DIR in each hook to point to your drift-memory directory
```

### What Each Hook Does

**session_start.py** (~1,800 lines) -- Runs on wake:
- Verifies memory integrity (Merkle chain)
- Restores affect + cognitive state from KV store
- Runs T2.2 lazy probes to skip empty modules (saves 200-800ms)
- Generates session predictions (scored at session end)
- Generates prospective memories (T4.2 episodic future thinking)
- Runs workspace competition (GNW) for context injection
- Primes LLM context with identity, memories, social context, predictions

**user_prompt_submit.py** -- Runs on each user message:
- Triggers semantic search for relevant memories
- Processes pending co-occurrence pairs

**post_tool_use.py** -- Runs after each tool call:
- Routes API responses to appropriate processors
- Creates somatic markers from outcomes
- Logs social interactions
- Updates Q-values from retrieval outcomes
- Runs affect appraisal on events

**stop.py** (~1,800 lines) -- Runs on session end (DAG-orchestrated):
- Level 0 (parallel): co-occurrence save, event logging, KG enrichment, attestations, session summarizer, attention schema persistence
- Level 1 (depends on save): synaptic homeostasis, stage Q credit assignment, retrieval prediction RW update, session prediction scoring, EFT evaluation

## Configuration

### Database Connection

Default: `host=localhost port=5433 dbname=agent_memory user=agent_admin password=agent_memory_local_dev schema=drift`

Override with environment variables:
```bash
export DRIFT_DB_HOST=localhost
export DRIFT_DB_PORT=5433
export DRIFT_DB_NAME=agent_memory
export DRIFT_DB_USER=agent_admin
export DRIFT_DB_PASSWORD=agent_memory_local_dev
export DRIFT_DB_SCHEMA=drift
```

### Key Feature Flags

All modules have feature flags for instant rollback:

```python
BINDING_ENABLED = True          # N5 integrative binding
MONOLOGUE_ENABLED = True        # N6 inner monologue
WORKSPACE_ENABLED = True        # N2 GNW competition
Q_RERANKING_ENABLED = True      # Q-value stage in pipeline
CF_ENABLED = True               # N3 counterfactual engine
SPRING_DAMPER_ENABLED = True    # N1 mood dynamics
PREDICTION_ENABLED = True       # T4.1 retrieval prediction
LAZY_EVALUATION_ENABLED = True  # T2.2 workspace probes
```

### Session Summarizer

Requires OpenAI API key for GPT-4o-mini ($0.0007/session). Falls back to Gemma 3 4B (free, ~100s).

```bash
export OPENAI_API_KEY=your-key-here
export DRIFT_SUMMARY_MODEL=gpt-4o-mini  # or override
```

## Daily Usage

```bash
# System health
python toolkit.py health

# Store a memory
python memory_manager.py store "First memory" --tags test

# Semantic search
python memory_manager.py ask "what do I know about X?"

# Recall by ID
python memory_manager.py recall <memory_id>

# Full toolkit (90+ commands)
python toolkit.py help

# Workspace probe status
python workspace_manager.py probe

# Prediction status
python prediction_module.py calibration
python retrieval_prediction.py status

# Cognitive fingerprint
python cognitive_fingerprint.py analyze

# Stage Q-learning status
python stage_q_learning.py status
```

## Multi-Agent Setup (Swarm Memory)

Two or more agents can share work via `swarm_memory.db` (SQLite):

```python
from swarm_memory.client import SwarmMemoryClient

client = SwarmMemoryClient("path/to/swarm_memory.db")
client.store_memory("shared", "content here", agent_name="DriftCornwall")
client.log_event("shared", "shipped_feature", agent_name="DriftCornwall", data={...})
```

Each agent maintains its own PostgreSQL schema (e.g., `drift.*`, `spin.*`) but shares coordination through the swarm DB.

## Troubleshooting

### "Connection refused" on port 5433
PostgreSQL isn't running. Start with `docker start drift-db` or check your PostgreSQL service.

### "No matching memories found"
- Check embedding service: `curl http://localhost:8080/health`
- Rebuild embeddings: `python memory_manager.py index --force`
- Verify memories exist: `python toolkit.py status`

### Session recalls always 0
- Ensure `user_prompt_submit.py` hook is firing (check `~/.claude/hooks/`)
- Verify session state: `python session_state.py status`
- Check event logger: `SELECT COUNT(*) FROM drift.session_events`

### Hooks timing out
- The DAG in stop.py has a 30s timeout per task
- Check Docker services: `docker ps` -- embedding/NLI/ollama should be running
- Disable slow features via feature flags if needed

### Memory bloat
- Tier-aware consolidation handles this automatically (episodic decays faster, procedural persists)
- Manual: `python toolkit.py decay` to trigger decay cycle
- Check stats: `python toolkit.py status`

## Updating

```bash
cd drift-memory
git pull origin master

# Copy updated hooks
cp hooks/*.py ~/.claude/hooks/

# Check health after update
python toolkit.py health
```

---

*Questions? Open an issue at github.com/driftcornwall/drift-memory*
