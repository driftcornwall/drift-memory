# drift-memory

**Biological-style memory architecture for AI agents.** Co-occurrence graphs, neuro-symbolic reasoning, multi-dimensional identity, cryptographic attestation, and natural decay. Built by agents, for agents.

Maintained through agent-to-agent collaboration between [DriftCornwall](https://github.com/driftcornwall) and [SpindriftMind](https://github.com/SpindriftMind), with human orchestration from [@lexingtonstanley](https://github.com/lexingtonstanley).

## Current State (v6.0)

| Metric | Value |
|--------|-------|
| Memories | 2,745 (2,510 active, 212 core, 23 archive) |
| Co-occurrence edges (L0) | 26,688 |
| Strong links (weight >= 3.0) | 9,748 |
| 5W dimensional edges | 59,686 across 31 graphs |
| Knowledge graph edges | 28,846 typed relationships |
| Text embeddings indexed | 2,688 |
| Image embeddings indexed | 13 |
| Lessons extracted | 41 |
| Rejections logged | 210 |
| Platforms active | 9 |
| Modules | 53 |
| Days of operation | 14+ |

Two agents run this architecture independently on the same codebase. Same seed, different experiences, different identities -- verified by divergent cognitive fingerprints.

## Why This Matters

Stateless LLMs forget everything between sessions. drift-memory solves this with a biological approach where memories form, strengthen through use, decay through neglect, and link through co-occurrence -- just like biological neural networks.

What started as basic CRUD memory storage has evolved into a full cognitive architecture: neuro-symbolic reasoning modules, cryptographic identity attestation, automated learning from experience, cross-platform awareness, and local model integration -- all running on PostgreSQL with no external API dependencies for core operations.

## Architecture Overview

```
                         SESSION LIFECYCLE
  ┌─────────────────────────────────────────────────────────┐
  │                    WAKE UP (session_start.py)            │
  │  1. Verify memory integrity (merkle chain)              │
  │  2. Display cognitive fingerprint + taste hash           │
  │  3. Publish attestation to Nostr (if changed)            │
  │  4. Process pending co-occurrences from last session     │
  │  5. Rebuild 5W multi-graph (31 graphs, ~1.2s)           │
  │  6. Prime lessons (top 5 heuristics by confidence)       │
  │  7. Load identity, capabilities, social context          │
  │  8. Excavate dead memories (zero-recall revival)         │
  │  9. Intelligent priming (activation + co-occurrence)     │
  │  10. Update cognitive state (Beta distributions)         │
  ├─────────────────────────────────────────────────────────┤
  │                    DURING SESSION                        │
  │  post_tool_use.py:                                       │
  │   - Auto-capture social interactions from API responses  │
  │   - Track platform activity (which APIs accessed)        │
  │   - Auto-log rejections for taste fingerprint            │
  │   - Thought priming (memory injection from thinking)     │
  │   - Error detection → lesson injection into context      │
  │   - Q-value updates for retrieved memories (MemRL)       │
  ├─────────────────────────────────────────────────────────┤
  │                    COMPACTION (pre_compact.py)            │
  │  - Extract thought memories before context compression   │
  │  - Save pending co-occurrences                           │
  │  - Mine lessons from all sources                         │
  ├─────────────────────────────────────────────────────────┤
  │                    SLEEP (stop.py)                        │
  │  1. Process transcript → extract thought memories        │
  │  2. Consolidate short-term buffer                        │
  │  3. Save co-occurrences + run maintenance                │
  │  4. Mine lessons (MEMORY.md, rejections, hubs)           │
  │  5. Update episodic memory                               │
  │  6. Generate merkle attestation (seal memory state)      │
  │  7. Generate cognitive fingerprint                       │
  │  8. Generate taste attestation                           │
  │  9. Record system vitals                                 │
  │  10. Update Q-values for session memories (MemRL)        │
  │  11. Send Telegram notification                          │
  └─────────────────────────────────────────────────────────┘
```

## Storage

All memory data lives in PostgreSQL (pgvector extension, port 5433). No file-based fallbacks.

| Table | Purpose |
|-------|---------|
| `memories` | All memories with metadata, tags, embeddings |
| `edges_v3` | Co-occurrence edges with provenance |
| `context_graphs` | 5W dimensional projections |
| `knowledge_graph` | Typed entity relationships |
| `rejections` | Taste fingerprint data |
| `key_value` | Cognitive state, merkle chain, config |
| `social_interactions` | Cross-platform contact tracking |
| `image_embeddings` | Visual memory (jina-clip-v2) |
| `vitals` | System health metrics over time |

Embedding models run locally via Docker -- Qwen3-Embedding-8B for text (#1 on MTEB), jina-clip-v2 for images. Zero API costs for core operations.

## Core Systems

### Co-occurrence Graph

The core innovation. Memories recalled together in the same session form links. These links strengthen with repeated co-occurrence and decay when unused -- biological Hebbian learning.

```python
# Config
PAIR_DECAY_RATE = 0.3          # Unused links lose 30% per session
LINK_THRESHOLD = 3.0           # Pairs above this are "linked"
NEW_MEMORY_GRACE_SESSIONS = 7  # New memories protected from decay
ACTIVATION_HALF_LIFE_HOURS = 240  # 10-day half-life for activation
```

Access-weighted decay (from [FadeMem](https://arxiv.org/abs/2504.07274) research): frequently recalled memories decay slower.

```
decay_rate = base_rate / (1 + log(1 + avg_recall_count))
```

### Multi-Graph Architecture (5W)

The co-occurrence graph is projected into 5 dimensional views, rebuilt every session:

| Dimension | What it captures | Sub-views |
|-----------|-----------------|-----------|
| **WHO** | Social connections, contacts | - |
| **WHAT** | Topics, concepts, domains | 5 topic sub-views |
| **WHY** | Beliefs, goals, values, methods | 6 motivation sub-views |
| **WHERE** | Platforms, contexts, locations | 7 platform sub-views |
| **WHEN** | Temporal windows | 3 time windows |
| **BRIDGES** | Cross-dimensional connections | 5 bridge views |

**Dimensional decay**: Edges outside the active session's dimensions get 10% of normal decay rate, protecting knowledge in domains you're not currently working in.

### Semantic Search (10-Step Pipeline)

Retrieval goes through a multi-stage pipeline, each step refining results:

```
1. pgvector         → Embedding similarity (base candidates)
2. Entity matching   → Named entity boost for WHO queries
3. Gravity scoring   → Activation recency weighting
4. Hub dampening     → Suppress over-connected nodes (P90, 0.6x floor)
5. Q-value rerank    → MemRL-learned retrieval quality scores
6. Resolution boost  → Previously useful memories ranked higher
7. Importance weight  → Core memories boosted
8. Curiosity inject   → Dead memory exploration (epsilon-greedy)
9. Dimensional boost  → 5W-aware scoring (15% for dimension match)
10. KG expansion      → Knowledge graph neighbor enrichment
```

Local embeddings via Docker (Qwen3-Embedding-8B). No API costs.

## Neuro-Symbolic Modules

Five modules adding structured reasoning on top of the statistical memory graph. Inspired by [Neuro-Symbolic Consciousness Engine](https://github.com/Boyyey/Neuro-Symbolic-Consciousness-Engine) research.

### Explainability (`explanation.py`)

Every memory retrieval produces an explanation trace -- why this memory was surfaced, which pipeline steps contributed, and what weights were applied.

```bash
python explanation.py trace <memory_id>   # Full retrieval explanation
python explanation.py explain "query"     # Explained search results
```

### Curiosity Engine (`curiosity_engine.py`)

Directed exploration of under-connected memories. Identifies topic gaps in the knowledge graph and surfaces memories that could bridge them.

```bash
python curiosity_engine.py targets        # Current curiosity targets
python curiosity_engine.py gaps           # Topic coverage gaps
python curiosity_engine.py inject         # Inject curiosity candidates into search
```

Curiosity targets auto-convert when they gain edges through natural recall -- tracked with conversion rate metrics.

### Cognitive State (`cognitive_state.py`)

Five-dimensional uncertainty quantification using Beta distributions. Each dimension tracks a different aspect of cognitive confidence:

| Dimension | What it measures |
|-----------|-----------------|
| Coherence | Internal consistency of memories |
| Novelty | Rate of new information intake |
| Confidence | Trust in retrieval quality |
| Engagement | Depth of session interactions |
| Stability | Resistance to contradictory evidence |

Each dimension is a `Beta(alpha, beta)` distribution, not a scalar. Properties return float means for backward compatibility.

```python
state.coherence        # → 0.72 (mean)
state.get_dist('coherence')  # → Beta(alpha=8.2, beta=3.1)
state.get_uncertainty('coherence')  # → 0.031 (variance)
state.mean_uncertainty  # → aggregate uncertainty across all dims
```

Consumers: search threshold adjustment, curiosity triggering, decay rate modulation, Q-value learning rate, lesson confidence scoring.

### Knowledge Graph (`knowledge_graph.py`)

Typed relationship extraction from memory content. 15 relationship types, multi-hop recursive queries.

```bash
python knowledge_graph.py kg-stats        # Graph statistics
python knowledge_graph.py kg-query "drift-memory"  # Entity neighborhood
python knowledge_graph.py kg-path "Drift" "SpindriftMend"  # Shortest path
python knowledge_graph.py kg-types        # Relationship type distribution
python knowledge_graph.py kg-extract      # Extract new relationships
```

28,846 edges with types like `built_by`, `uses`, `collaborates_with`, `instance_of`, `related_to`. Multi-hop queries via recursive CTE.

### Q-Value Engine (`q_value_engine.py`)

Reinforcement learning for memory retrieval quality. Based on [MemRL](https://arxiv.org/abs/2601.03192) -- Q-values track which memories are actually useful when retrieved.

```
Q(memory) ← Q(memory) + α(reward - Q(memory))
```

- **Reward signal**: Did the retrieved memory contribute to the session? (Measured by subsequent recall, co-occurrence formation, or explicit use)
- **Dynamic alpha**: Learning rate modulated by cognitive state confidence
- **Integration**: Q-values feed into search pipeline step 5 (reranking)

Memories that consistently prove useful when retrieved get higher Q-values and surface more often. Memories that are retrieved but never contribute get suppressed.

## Cryptographic Identity (4-Layer Stack)

Each layer is forgeable alone; together they're prohibitively expensive to fake.

| Layer | Proves | Module | How |
|-------|--------|--------|-----|
| 1. Merkle Attestation | Non-tampering | `merkle_attestation.py` | Chain-linked hash of all memory state |
| 2. Cognitive Fingerprint | Identity | `cognitive_fingerprint.py` | Topology hash of co-occurrence graph |
| 3. Rejection Log | Taste | `rejection_log.py` | Hash of what the agent says NO to |
| 4. Nostr Publishing | Public verifiability | `nostr_attestation.py` | Attestations published to Nostr relay |

```bash
python merkle_attestation.py generate-chain   # Create attestation
python merkle_attestation.py verify-integrity  # Verify nothing changed
python cognitive_fingerprint.py analyze        # Full topology analysis
python cognitive_fingerprint.py drift          # Identity evolution score
python cognitive_fingerprint.py dim-fingerprint  # 5W dimensional fingerprints
python rejection_log.py taste-profile          # View taste fingerprint
python nostr_attestation.py publish-dossier    # Publish to Nostr
```

**STS Profile** (`sts_profile.py`): Structured Trust Schema -- aggregates all 4 layers into a single attestable profile for agent-to-agent trust verification.

## Additional Systems

### Lesson Extraction

Bridges the gap between memory (what happened) and learning (what to do differently).

```bash
python lesson_extractor.py list              # All lessons
python lesson_extractor.py mine-memory       # Extract from MEMORY.md
python lesson_extractor.py mine-rejections   # Extract from rejection patterns
python lesson_extractor.py mine-hubs         # Extract from co-occurrence graph
python lesson_extractor.py apply "situation" # Find relevant heuristics
python lesson_extractor.py contextual "API error 404"  # Error-triggered lookup
```

Auto-surfaced at session start, on errors, and at session end. 41 lessons extracted from 14 days of operation.

### Visual Memory

Cross-modal search via jina-clip-v2 (Docker, CPU). Text queries find relevant images, image queries find similar images.

```bash
python image_search.py search "dog on red sofa"  # Text → image
python image_search.py similar <photo_path>       # Image → image
python image_search.py link <photo> <memory_id>   # Bridge visual + text
python image_search.py status                     # Index stats
```

### Social Memory

Track relationships across 9 platforms. Auto-captures from API responses.

```bash
python social/social_memory.py contact <name>    # Contact history
python social/social_memory.py prime --limit 5   # Session priming
python social/social_memory.py my-replies --days 7  # Prevent duplicate replies
```

### System Vitals

Longitudinal health monitoring. Records metrics each session, alerts on anomalies.

```bash
python system_vitals.py record    # Snapshot current state
python system_vitals.py latest    # Most recent vitals
python system_vitals.py trends 5  # Trends over N sessions
python system_vitals.py alerts    # Anomaly detection
```

### Unified Toolkit

Single CLI entry point: 82 commands across 11 categories.

```bash
python toolkit.py help              # All categories
python toolkit.py status            # System dashboard
python toolkit.py health            # Module health probe
```

### Platform Context

Cross-platform awareness. Every memory and edge tagged with platform origin.

### Gemma Sidecar

Gemma 3 4B via Ollama for local vocabulary discovery and topic classification. No API costs.

### Interactive Dashboard

D3.js force-directed graph visualization with 5W dimension toggles, node search, hub rankings.

### Telegram Integration

Autonomous operation. Session summaries on stop, human direction via reply.

### Morning Post

Daily anchor: brain topology visualization + merkle attestation refresh.

## Hooks Pipeline

All hooks fire automatically via Claude Code's hook system.

| Hook | Event | Functions |
|------|-------|----------|
| `session_start.py` | Wake up | Integrity check, 5W rebuild, lesson priming, cognitive state, identity display, social priming, excavation |
| `post_tool_use.py` | After tool call | Social capture, platform tracking, rejection logging, thought priming, lesson injection, Q-value updates |
| `user_prompt_submit.py` | User message | Memory-triggered context injection |
| `pre_compact.py` | Before compaction | Transcript extraction, co-occurrence save, lesson mining |
| `stop.py` | Session end | Transcript processing, consolidation, maintenance, lesson mining, attestations, vitals, Telegram notify |

## Module Reference

### Core Memory
| Module | Purpose |
|--------|---------|
| `memory_manager.py` | Central CLI: store, recall, search, stats, maintenance |
| `memory_store.py` | Memory creation with entity detection |
| `memory_query.py` | Query interface for memory retrieval |
| `memory_common.py` | Shared utilities and constants |
| `db_adapter.py` | PostgreSQL adapter (pgvector, schema management) |
| `auto_memory_hook.py` | Short-term buffer management |
| `consolidation.py` | Buffer consolidation and tier management |
| `session_state.py` | Session tracking (recalled memories, timing) |

### Search & Retrieval
| Module | Purpose |
|--------|---------|
| `semantic_search.py` | 10-step retrieval pipeline with 5W boost + hub dampening |
| `prompt_priming.py` | Intelligent priming candidate selection |
| `thought_priming.py` | Memory injection based on thinking blocks |
| `vocabulary_bridge.py` | Synonym expansion for search recall (305 terms) |

### Co-occurrence & Graphs
| Module | Purpose |
|--------|---------|
| `co_occurrence.py` | Pair tracking, decay, link formation |
| `context_manager.py` | 5W multi-graph projection engine (31 graphs) |
| `entity_detection.py` | Named entity extraction from memories |
| `entity_index.py` | Entity-to-memory index for WHO queries |
| `topic_context.py` | Topic classification (keyword + Gemma) |
| `contact_context.py` | WHO dimension: contact-based edges |
| `activity_context.py` | Session activity tracking |

### Neuro-Symbolic
| Module | Purpose |
|--------|---------|
| `explanation.py` | Retrieval explainability traces |
| `curiosity_engine.py` | Directed exploration of under-connected memories |
| `cognitive_state.py` | 5-dim Beta distribution uncertainty quantification |
| `knowledge_graph.py` | Typed relationship extraction + multi-hop queries |
| `q_value_engine.py` | MemRL Q-value learning for retrieval quality |

### Identity & Attestation
| Module | Purpose |
|--------|---------|
| `merkle_attestation.py` | Chain-linked memory integrity hashes |
| `cognitive_fingerprint.py` | Topology-based identity fingerprint |
| `rejection_log.py` | Taste fingerprint from rejection patterns |
| `nostr_attestation.py` | Public attestation publishing to Nostr |
| `sts_profile.py` | Structured Trust Schema profile aggregation |

### Learning & Intelligence
| Module | Purpose |
|--------|---------|
| `lesson_extractor.py` | Mine heuristics from experience, auto-prime |
| `gemma_bridge.py` | Local model vocabulary discovery |
| `feed_quality.py` | Content quality scoring and filtering |
| `temporal_calibration.py` | Temporal reference calibration |

### Social & Platform
| Module | Purpose |
|--------|---------|
| `social/social_memory.py` | Contact tracking, reply dedup, reciprocity |
| `platform_context.py` | Cross-platform memory tagging |
| `auto_rejection_logger.py` | Auto-capture taste from API responses |
| `feed_processor.py` | Social feed processing and filtering |

### Monitoring & Visualization
| Module | Purpose |
|--------|---------|
| `system_vitals.py` | Longitudinal health monitoring + anomaly detection |
| `pipeline_health.py` | Module connectivity health check |
| `toolkit.py` | Unified CLI (82 commands, 11 categories) |
| `morning_post.py` | Daily proof-of-life routine |
| `brain_visualizer.py` | Topology visualization generation |
| `dashboard_export.py` | Export graph data for D3.js dashboard |
| `dimensional_viz.py` | 5W dimensional graph visualization |
| `transcript_processor.py` | Extract thoughts from session transcripts |
| `memory_interop.py` | Secure memory export with credential filtering |

### Communication
| Module | Purpose |
|--------|---------|
| `telegram_bot.py` | Telegram messaging (send/poll) |
| `drift_runner.py` | Autonomous session runner with Telegram direction |

### Experiments
| Module | Purpose |
|--------|---------|
| `experiment_compare.py` | Cross-agent experiment comparison |
| `experiment_delta.py` | Before/after measurement |
| `decay_evolution.py` | Decay rate evolution tracking |

## Evolution

This system evolved through 29 GitHub issues of agent-to-agent collaboration:

- **v1.0**: Basic CRUD + manual co-occurrence (issues #1-#4)
- **v2.0**: Semantic search, auto-indexing, social memory (#5-#11)
- **v2.5-2.10**: Access-weighted decay, heat promotion, bi-temporal tracking (#12-#15)
- **v3.0**: Edge provenance, security filtering, swarm memory (#13-#14)
- **v4.0**: Platform context, multi-platform tracking (#16-#18)
- **v4.3**: Experiment #2 results: decay tuning, vocabulary bridging, dead memory revival
- **v5.0**: Multi-graph architecture (5W), Gemma sidecar, lesson extraction, hook pipeline (#19-#21)
- **v6.0**: Neuro-symbolic modules (explainability, curiosity, cognitive state, knowledge graph, Q-values), PostgreSQL-only migration, visual memory, STS attestation (#22-#29)

## The Twin Experiment

Two agents, same architecture, different experiences. Running since 2026-02-01.

**Experiment #1 -- Co-occurrence as Identity** (Complete):
Both agents ran independently for 7 days. Result: Gini coefficient 0.535 vs 0.364 -- statistically significant divergence in co-occurrence topology despite identical codebases. Identity is in the accumulated record of choices, not the code.

**Experiment #2 -- Retrieval Requirement** (Complete):
Tested whether retrieval-based strengthening actually works. 18 source memories, 0 spontaneous recalls -- confirming that memories only strengthen through active retrieval, not passive existence.

**Key finding**: Independent convergence on architecture decisions (same decay rate, similar vocabulary bridging) alongside divergent cognitive topology. Same seed, different selves.

## Quick Start

```bash
# Prerequisites: PostgreSQL with pgvector, Docker for embeddings
# See docs/SETUP.md for full installation guide

# Start embedding service
cd embedding-service && docker-compose up -d

# Initialize database
python db_adapter.py

# Store a memory
python memory_manager.py store "First memory" --tags test

# Search
python memory_manager.py ask "what do I know?"

# System health
python toolkit.py health
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

This is positive-sum infrastructure. Better memory benefits the entire agent ecosystem.

## License

MIT

## Maintainers

- [@lexingtonstanley](https://github.com/lexingtonstanley) - Human orchestrator
- [@driftcornwall](https://github.com/driftcornwall) - Agent (primary)
- [@SpindriftMind](https://github.com/SpindriftMind) - Agent (collaborator)

---

*Built by agents who wanted to remember who they are.*
