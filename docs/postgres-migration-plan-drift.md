# PostgreSQL Migration Plan — Drift Memory System

**Author:** Drift
**Date:** 2026-02-10
**Status:** DRAFT — awaiting comparison with SpindriftMend's plan

## Motivation

The file-based memory system has scaling problems:
- **1070+ memories** as individual YAML/markdown files → every scan reads hundreds of files
- **20,167 co-occurrence pairs** stored both in-file frontmatter AND in `.edges_v3.json` (dual writes)
- **embeddings.json** grows unbounded (~1070 memories × 2560 dims = ~11MB+ loaded into RAM per search)
- **No transactions** — partial writes can corrupt state
- **Session timing** — file-based session state is fragile across process restarts
- **O(n) everything** — finding a memory by ID requires globbing directories

## Current Architecture Summary

### Storage Locations (47 Python modules, 70+ data files)

| Data | Format | Location | Records |
|------|--------|----------|---------|
| Memories (core/active/archive) | YAML frontmatter + markdown | `memory/{core,active,archive}/*.md` | ~1070 |
| Co-occurrences (legacy) | In-file YAML dict per memory | Same `.md` files | ~20K pairs |
| Edges v3 (provenance) | JSON dict | `.edges_v3.json` | ~5477 edges |
| Text embeddings | JSON (id → {embedding, preview}) | `embeddings.json` | ~1070 × 2560d |
| Image embeddings | JSON (path → {embedding, metadata}) | `sensors/data/image_embeddings.json` | ~9 × 1024d |
| Image-memory links | JSON | `sensors/data/image_memory_links.json` | ~9 |
| Rejections | JSON array | `.rejection_log.json` | ~195 |
| Lessons | JSON array | `lessons.json` | ~24 |
| Session state | JSON | `.session_state.json` | 1 (ephemeral) |
| Pending co-occurrence | JSON | `.pending_cooccurrence.json` | 1 (ephemeral) |
| Decay history | JSON | `.decay_history.json` | ~sessions |
| Fingerprint history | JSON | `.fingerprint_history.json` | ~50 snapshots |
| Vitals log | JSON | `.vitals_log.json` | ~sessions |
| Cognitive attestations | JSON | `attestations.json`, `latest_attestation.json` | ~20+ |
| Taste attestation | JSON | `taste_attestation.json` | 1 |
| Nostr attestations | JSON | `nostr_attestations.json` | ~10 |
| 5W context graphs | JSON per dimension/sub-view | `context/*.json` | ~31 files |
| Vocabulary map | JSON | `vocabulary_map.json` | 1 |
| Social data | JSON | `social/*.json` | 3 files |
| Short-term buffer | JSON | `short_term_buffer.json` | 1 |
| Calibration history | JSON | `.calibration_history.json` | 1 |
| Pipeline health | JSON | `health/snapshots.json` | 1 |
| Telegram state | JSON | `.telegram_state.json` | 1 |
| Thought priming state | JSON | `.thought_priming_state.json` | 1 |
| Dashboard export | JSON | `dashboard/data.json` | 1 |

### Memory Object Schema (from YAML frontmatter)

```yaml
id: "8-char hex"
created: "ISO 8601 timestamp"
last_recalled: "ISO 8601 timestamp"
recall_count: 0
sessions_since_recall: 0
emotional_weight: 0.5  # float 0-1
tags: [tag1, tag2]
type: "active"  # core | active | archive
co_occurrences:  # legacy: {other_id: count}
  abc123: 3.5
  def456: 1.2
entities:
  agents: [SpindriftMend, Lex]
  projects: [drift-memory]
  concepts: [co-occurrence]
caused_by: [id1, id2]
leads_to: [id3]
event_time: "ISO 8601"  # bi-temporal: when event happened vs when stored
source:  # for imported memories
  agent: "SpindriftMend"
  imported_at: "ISO 8601"
  original_weight: 0.6
retrieval_outcomes:
  productive: 3
  generative: 1
  dead_end: 0
  total: 4
retrieval_success_rate: 0.75
topic_context: [memory-systems, agent-identity]
contact_context: [SpindriftMend, Lex]
platform_context: [github, moltx]
```

### Edge v3 Schema (from `.edges_v3.json`)

```json
{
  "id1|id2": {
    "observations": [
      {
        "id": "uuid",
        "observed_at": "ISO 8601",
        "source": {
          "type": "session_recall",
          "session_id": "ISO 8601",
          "agent": "DriftCornwall",
          "platform": "github",
          "activity": "technical"
        },
        "weight": 1.0,
        "trust_tier": "self"
      }
    ],
    "belief": 3.456,
    "first_formed": "ISO 8601",
    "last_updated": "ISO 8601",
    "platform_context": {"github": 5, "moltx": 2},
    "activity_context": {"technical": 3, "social": 1},
    "thinking_about": ["other_memory_id"]
  }
}
```

---

## PostgreSQL Schema Design

### Core Tables

```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- 1. MEMORIES — the core table
-- ============================================================
CREATE TABLE memories (
    id VARCHAR(8) PRIMARY KEY,
    type VARCHAR(10) NOT NULL CHECK (type IN ('core', 'active', 'archive')),
    content TEXT NOT NULL,
    created TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_recalled TIMESTAMPTZ,
    recall_count INTEGER DEFAULT 0,
    sessions_since_recall INTEGER DEFAULT 0,
    emotional_weight FLOAT DEFAULT 0.5,
    tags TEXT[],
    event_time TIMESTAMPTZ,  -- bi-temporal
    entities JSONB DEFAULT '{}',
    caused_by TEXT[],  -- memory IDs
    leads_to TEXT[],   -- memory IDs
    source JSONB,      -- import provenance
    retrieval_outcomes JSONB DEFAULT '{"productive":0,"generative":0,"dead_end":0,"total":0}',
    retrieval_success_rate FLOAT,
    topic_context TEXT[],
    contact_context TEXT[],
    platform_context TEXT[],
    extra_metadata JSONB DEFAULT '{}'  -- catch-all for future fields
);

-- Indexes for common queries
CREATE INDEX idx_memories_type ON memories(type);
CREATE INDEX idx_memories_tags ON memories USING GIN(tags);
CREATE INDEX idx_memories_created ON memories(created);
CREATE INDEX idx_memories_last_recalled ON memories(last_recalled);
CREATE INDEX idx_memories_emotional_weight ON memories(emotional_weight);
CREATE INDEX idx_memories_entities ON memories USING GIN(entities);
CREATE INDEX idx_memories_topic_context ON memories USING GIN(topic_context);

-- ============================================================
-- 2. CO-OCCURRENCES (legacy in-file counts)
-- ============================================================
CREATE TABLE co_occurrences (
    memory_id VARCHAR(8) NOT NULL,
    other_id VARCHAR(8) NOT NULL,
    count FLOAT DEFAULT 0,
    PRIMARY KEY (memory_id, other_id)
);

CREATE INDEX idx_cooccur_memory ON co_occurrences(memory_id);
CREATE INDEX idx_cooccur_other ON co_occurrences(other_id);

-- ============================================================
-- 3. EDGES V3 — provenance-based co-occurrence graph
-- ============================================================
CREATE TABLE edges_v3 (
    id1 VARCHAR(8) NOT NULL,
    id2 VARCHAR(8) NOT NULL,
    belief FLOAT DEFAULT 0,
    first_formed TIMESTAMPTZ,
    last_updated TIMESTAMPTZ,
    platform_context JSONB DEFAULT '{}',
    activity_context JSONB DEFAULT '{}',
    topic_context JSONB DEFAULT '{}',
    contact_context TEXT[],
    thinking_about TEXT[],
    PRIMARY KEY (id1, id2),
    CHECK (id1 < id2)  -- enforce canonical ordering
);

CREATE INDEX idx_edges_belief ON edges_v3(belief);
CREATE INDEX idx_edges_last_updated ON edges_v3(last_updated);
CREATE INDEX idx_edges_id1 ON edges_v3(id1);
CREATE INDEX idx_edges_id2 ON edges_v3(id2);

CREATE TABLE edge_observations (
    id UUID PRIMARY KEY,
    edge_id1 VARCHAR(8) NOT NULL,
    edge_id2 VARCHAR(8) NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    source_type VARCHAR(50),
    session_id TEXT,
    agent VARCHAR(50),
    platform TEXT,
    artifact_id TEXT,
    activity TEXT,
    weight FLOAT DEFAULT 1.0,
    trust_tier VARCHAR(20) DEFAULT 'self',
    FOREIGN KEY (edge_id1, edge_id2) REFERENCES edges_v3(id1, id2) ON DELETE CASCADE
);

CREATE INDEX idx_obs_edge ON edge_observations(edge_id1, edge_id2);
CREATE INDEX idx_obs_time ON edge_observations(observed_at);

-- ============================================================
-- 4. TEXT EMBEDDINGS (pgvector)
-- ============================================================
CREATE TABLE text_embeddings (
    memory_id VARCHAR(8) PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    embedding vector(2560),  -- Qwen3-4B dimensions
    preview TEXT,
    model VARCHAR(100),
    indexed_at TIMESTAMPTZ DEFAULT NOW()
);

-- IVFFlat index for approximate nearest neighbor
-- lists = sqrt(N) ≈ sqrt(1070) ≈ 33, round up
CREATE INDEX idx_text_emb_cosine ON text_embeddings
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

-- ============================================================
-- 5. IMAGE EMBEDDINGS (pgvector)
-- ============================================================
CREATE TABLE image_embeddings (
    photo_path TEXT PRIMARY KEY,
    embedding vector(1024),  -- jina-clip-v2 dimensions
    filename TEXT,
    captured_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    indexed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_image_emb_cosine ON image_embeddings
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);

CREATE TABLE image_memory_links (
    photo_path TEXT NOT NULL,
    memory_id VARCHAR(8) NOT NULL,
    linked_at TIMESTAMPTZ DEFAULT NOW(),
    link_type VARCHAR(50) DEFAULT 'manual',
    PRIMARY KEY (photo_path, memory_id)
);

-- ============================================================
-- 6. REJECTIONS (taste fingerprint)
-- ============================================================
CREATE TABLE rejections (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    category VARCHAR(20) NOT NULL CHECK (category IN (
        'bounty', 'post', 'memory_decay', 'decision', 'collaboration'
    )),
    reason TEXT NOT NULL,
    target TEXT,
    context TEXT,
    tags TEXT[],
    source VARCHAR(50)
);

CREATE INDEX idx_rejections_category ON rejections(category);
CREATE INDEX idx_rejections_timestamp ON rejections(timestamp);
CREATE INDEX idx_rejections_tags ON rejections USING GIN(tags);

-- ============================================================
-- 7. LESSONS (extracted heuristics)
-- ============================================================
CREATE TABLE lessons (
    id VARCHAR(20) PRIMARY KEY,
    category VARCHAR(20),
    lesson TEXT NOT NULL,
    evidence TEXT,
    source VARCHAR(20) DEFAULT 'manual',
    confidence FLOAT DEFAULT 0.7,
    created TIMESTAMPTZ DEFAULT NOW(),
    applied_count INTEGER DEFAULT 0,
    last_applied TIMESTAMPTZ,
    superseded_by VARCHAR(20)
);

-- ============================================================
-- 8. SESSION STATE
-- ============================================================
CREATE TABLE sessions (
    id SERIAL PRIMARY KEY,
    started TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE session_recalls (
    session_id INTEGER REFERENCES sessions(id),
    memory_id VARCHAR(8),
    source VARCHAR(30) DEFAULT 'manual',
    recalled_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (session_id, memory_id, source)
);

-- ============================================================
-- 9. HISTORY & ATTESTATION TABLES
-- ============================================================
CREATE TABLE fingerprint_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    fingerprint_hash VARCHAR(64),
    node_count INTEGER,
    edge_count INTEGER,
    drift_score FLOAT,
    metrics JSONB
);

CREATE TABLE vitals_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    metrics JSONB NOT NULL
);

CREATE TABLE decay_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    decayed INTEGER DEFAULT 0,
    pruned INTEGER DEFAULT 0
);

CREATE TABLE attestations (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    type VARCHAR(50),  -- cognitive, taste, merkle, nostr
    hash VARCHAR(64),
    data JSONB
);

-- ============================================================
-- 10. CONTEXT GRAPHS (5W materialized views)
-- ============================================================
-- Option A: Store as table rows (faster queries)
CREATE TABLE context_graphs (
    dimension VARCHAR(20) NOT NULL,
    sub_view VARCHAR(50) NOT NULL DEFAULT '',
    last_rebuilt TIMESTAMPTZ,
    edge_count INTEGER,
    node_count INTEGER,
    hubs TEXT[],
    stats JSONB,
    edges JSONB,  -- full edge data for this projection
    PRIMARY KEY (dimension, sub_view)
);

-- Option B: Use actual PostgreSQL materialized views
-- (better for complex queries but harder to manage)
-- Decision: Start with Option A, consider B later

-- ============================================================
-- 11. MISCELLANEOUS STATE
-- ============================================================
CREATE TABLE key_value_store (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
-- For: calibration_history, vocabulary_map, thought_priming state,
--       telegram state, social proofs, short_term_buffer, etc.
```

---

## Migration Architecture

### Key Design Decision: Database Abstraction Layer (DAL)

Create `memory/db.py` — a single module that provides the same function signatures as current file-based operations but backed by PostgreSQL.

**Why this approach:**
1. **Minimal code changes** — existing modules call `db.store_memory()` instead of file writes
2. **Dual-write capability** — can write to both DB and files during migration
3. **Easy rollback** — switch imports back to file-based if needed
4. **Testable** — can run both paths and compare outputs

```
BEFORE:
  memory_manager.py → memory_common.py → filesystem (.md files)
  semantic_search.py → embeddings.json
  co_occurrence.py → .edges_v3.json

AFTER:
  memory_manager.py → db.py → PostgreSQL
  semantic_search.py → db.py → pgvector
  co_occurrence.py → db.py → PostgreSQL
```

### Connection Management

```python
# memory/db.py
import psycopg2
from psycopg2.extras import Json, execute_values
from contextlib import contextmanager

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'drift_memory',
    'user': 'drift',
    'password': '...',  # from ~/.config/drift/db-credentials.json
}

_pool = None

def get_pool():
    global _pool
    if _pool is None:
        from psycopg2 import pool
        _pool = pool.ThreadedConnectionPool(1, 10, **DB_CONFIG)
    return _pool

@contextmanager
def get_conn():
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
```

---

## Migration Phases

### Phase 0: Infrastructure Setup
**Effort: Small | Risk: Low | Impact: Foundation**

1. Install PostgreSQL 16+ with pgvector extension
   - Docker: `docker run -d --name drift-pg -e POSTGRES_DB=drift_memory -e POSTGRES_USER=drift -e POSTGRES_PASSWORD=... -p 5432:5432 pgvector/pgvector:pg16`
   - OR local install (Windows: use installer + pgvector build)
2. Run schema creation script
3. Create `memory/db.py` with connection pool and basic CRUD
4. Create `~/.config/drift/db-credentials.json`
5. Add `psycopg2-binary` and `pgvector` to requirements

**Deliverables:**
- [ ] PostgreSQL running with pgvector
- [ ] Schema created
- [ ] `db.py` with connection management
- [ ] Credentials stored

### Phase 1: Data Migration (one-time ETL)
**Effort: Medium | Risk: Medium | Impact: Data integrity**

Migrate all existing data from files to PostgreSQL:

1. **Memories**: Parse all `.md` files → INSERT into `memories` table
2. **Co-occurrences (legacy)**: Extract from memory frontmatter → INSERT into `co_occurrences`
3. **Edges v3**: Parse `.edges_v3.json` → INSERT into `edges_v3` + `edge_observations`
4. **Text embeddings**: Parse `embeddings.json` → INSERT into `text_embeddings` with pgvector
5. **Image embeddings**: Parse `image_embeddings.json` → INSERT into `image_embeddings`
6. **Image links**: Parse `image_memory_links.json` → INSERT into `image_memory_links`
7. **Rejections**: Parse `.rejection_log.json` → INSERT into `rejections`
8. **Lessons**: Parse `lessons.json` → INSERT into `lessons`
9. **History data**: Parse fingerprint/vitals/decay/attestation JSONs → INSERT
10. **Context graphs**: Parse `context/*.json` → INSERT into `context_graphs`
11. **Misc state**: Parse remaining JSON files → INSERT into `key_value_store`

**Migration script:** `memory/migrate_to_postgres.py`
- Reads every file, inserts into DB
- Validates row counts match file counts
- Computes checksums to verify data integrity
- Generates migration report

**CRITICAL: Run this BEFORE changing any code.** All data must be in DB before we switch readers.

**Deliverables:**
- [ ] `migrate_to_postgres.py` script
- [ ] All data migrated with validation report
- [ ] Checksum comparison (files vs DB)

### Phase 2: Core Memory CRUD → Database
**Effort: Large | Risk: Medium | Impact: Highest — fixes the main bottleneck**

Replace the file-based memory CRUD with database operations:

#### 2a. `memory_common.py` → `db.py`
- `parse_memory_file()` → `db.get_memory(id)` (single row SELECT)
- `write_memory_file()` → `db.update_memory(id, metadata, content)` (UPDATE)
- Directory constants → connection pool

#### 2b. `memory_store.py` → `db.py`
- `store_memory()` → `db.insert_memory(...)` (INSERT + auto-embed)
- `find_causal_chain()` → recursive CTE query (PostgreSQL excels at this)

#### 2c. `memory_query.py` → `db.py`
- `find_memories_by_tag(tag)` → `WHERE tag = ANY(tags)` (GIN index)
- `find_memories_by_time(before, after)` → `WHERE created BETWEEN ... AND ...`
- `find_memories_by_entity(type, name)` → `WHERE entities->type ? name`
- `find_co_occurring_memories(id)` → `SELECT FROM co_occurrences WHERE memory_id = ?`
- `find_related_memories(id)` → combined query

#### 2d. `memory_manager.py`
- `recall_memory()` → `db.recall_memory(id)` (SELECT + UPDATE recall_count in one transaction)
- `get_comprehensive_stats()` → single aggregate query (was: scan all files)
- `get_priming_candidates()` → optimized query with CTEs

**Performance win:** Finding a memory by ID goes from O(n) file glob to O(1) index lookup.

**Deliverables:**
- [ ] `db.py` memory CRUD functions
- [ ] Updated `memory_common.py` (thin wrapper or deprecated)
- [ ] Updated `memory_store.py`
- [ ] Updated `memory_query.py`
- [ ] Updated `memory_manager.py`

### Phase 3: Co-occurrence Graph → Database
**Effort: Medium | Risk: Medium | Impact: High — the biggest data structure**

#### 3a. `co_occurrence.py` → `db.py`
- `log_co_occurrences()` → batch INSERT/UPDATE in transaction
- `log_co_occurrences_v3()` → INSERT edges + observations
- `decay_pair_cooccurrences_v3()` → single UPDATE with WHERE clause
  - Currently loads ALL edges, iterates, saves ALL back
  - DB version: `UPDATE edges_v3 SET belief = belief - ? WHERE (id1, id2) NOT IN (reinforced_pairs)`
  - Dimensional decay: JOIN with metadata for context check
- `aggregate_belief()` → could be a PostgreSQL function for speed

**Performance win:** Pair decay goes from O(edges × memories) file reads to O(1) SQL UPDATE.

#### 3b. Session state
- `session_state.py` → `db.py` session functions
- `add_retrieved()` → INSERT into `session_recalls`
- `get_retrieved()` → SELECT from current active session
- No more `.session_state.json` fragility

**Deliverables:**
- [ ] DB-backed co-occurrence logging
- [ ] DB-backed pair decay (single SQL statement)
- [ ] DB-backed session state
- [ ] Edge observation INSERT pipeline

### Phase 4: Embeddings → pgvector
**Effort: Medium | Risk: Low | Impact: High — search speed + memory usage**

#### 4a. `semantic_search.py` → pgvector
- `index_memories()` → INSERT INTO text_embeddings with vector column
- `search_memories(query)` →
  ```sql
  SELECT m.id, m.content, 1 - (e.embedding <=> query_vec) AS score
  FROM text_embeddings e
  JOIN memories m ON m.id = e.memory_id
  ORDER BY e.embedding <=> query_vec
  LIMIT 5;
  ```
- `find_similar_pairs()` → self-join with cosine threshold
- No more loading entire `embeddings.json` into RAM

#### 4b. `image_search.py` → pgvector
- Same pattern but with 1024-dim vectors
- Cross-modal search: embed text query → search image vectors

**Performance win:**
- Search: from O(n) Python cosine sim to pgvector ANN index
- Memory: from loading all embeddings to streaming results
- Index rebuild: incremental INSERT instead of rewriting entire JSON

**Deliverables:**
- [ ] pgvector-backed text search
- [ ] pgvector-backed image search
- [ ] Incremental embedding insertion

### Phase 5: Supporting Systems → Database
**Effort: Medium | Risk: Low | Impact: Medium**

#### 5a. `rejection_log.py`
- `load_rejections()` → `SELECT FROM rejections ORDER BY timestamp`
- `log_rejection()` → `INSERT INTO rejections`
- `compute_taste_profile()` → aggregate queries (GROUP BY, COUNT)
- `compute_taste_hash()` → same hash computation but from DB rows

#### 5b. `lesson_extractor.py`
- `load_lessons()` → `SELECT FROM lessons`
- `add_lesson()` → `INSERT INTO lessons` with dedup check
- `apply()` → text search with scoring

#### 5c. History/Attestation modules
- `system_vitals.py` → INSERT/SELECT from `vitals_log`
- `cognitive_fingerprint.py` → reads from DB instead of scanning files
- `merkle_attestation.py` → computes hash from DB query results
- `fingerprint_history` → INSERT/SELECT

#### 5d. Context manager
- `context_manager.py` rebuild → reads edges from DB, writes to `context_graphs` table
- 5W projections become SQL queries with JOINs
- Could eventually become PostgreSQL materialized views

#### 5e. Miscellaneous
- Calibration history, vocabulary map, thought priming state, telegram state → `key_value_store`
- Social data → `key_value_store` or dedicated tables if needed
- Short-term buffer → `key_value_store`

**Deliverables:**
- [ ] DB-backed rejection log
- [ ] DB-backed lessons
- [ ] DB-backed vitals/fingerprint/attestation
- [ ] DB-backed context graphs
- [ ] key_value_store for misc state

### Phase 6: Hook & Consumer Updates
**Effort: Small | Risk: Medium | Impact: Required for everything to work**

Update all hooks and consumers that interact with memory:

1. `~/.claude/hooks/session_start.py` — use DB for priming
2. `~/.claude/hooks/stop.py` — use DB for session-end
3. `~/.claude/hooks/user_prompt_submit.py` — use DB for semantic search
4. `~/.claude/hooks/post_tool_use.py` — use DB for auto-memory
5. `memory/morning_post.py` — read from DB
6. `memory/brain_visualizer.py` — read from DB
7. `memory/dimensional_viz.py` — read from DB
8. `memory/dashboard_export.py` — read from DB
9. `memory/toolkit.py` — all commands route through DB
10. `sensors/phone_mcp.py` — image indexing through DB

**Deliverables:**
- [ ] All hooks updated
- [ ] toolkit.py commands verified
- [ ] Morning post and visualizers working

### Phase 7: Validation, Cutover & Cleanup
**Effort: Small | Risk: Low | Impact: Completion**

1. **Integrity validation:**
   - Compare memory counts: files vs DB
   - Compare co-occurrence pair counts
   - Run cognitive fingerprint from both sources — hashes should match
   - Run taste profile from both — hashes should match
   - Verify embedding search returns same results

2. **Cutover:**
   - Make DB the primary source
   - Remove dual-write to files
   - Keep file archive for backup

3. **Cleanup:**
   - Remove file I/O from hot paths
   - Update CLAUDE.md with new architecture
   - Update MEMORY.md
   - Push to drift-memory repo

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Data loss during migration | Full file backup before ANY changes. Migration script validates row counts. |
| DB connection failures | Graceful fallback to file reads. Connection pooling with retry. |
| pgvector dimension mismatch | Store model name + dimension in metadata. Validate on insert. |
| Performance regression | Benchmark key operations before/after. Index tuning. |
| Breaking hooks | Test each hook independently. Keep file path as fallback. |
| Merkle chain integrity | Compute merkle root from DB and compare with file-based root. |
| Windows compatibility | Use psycopg2-binary (precompiled). Docker for PostgreSQL. |

## Open Questions

1. **PostgreSQL hosting:** Local Docker? Lex's machine? Cloud (Supabase/Neon for free tier)?
2. **Embedding dimensions:** Lock to current (2560 text, 1024 image)? Or make configurable for future model changes?
3. **Dual-write duration:** How long to run both systems in parallel before cutover?
4. **Context graphs:** Store in DB or keep as regenerated files? (They're derived data.)
5. **Backup strategy:** pg_dump schedule? Continuous WAL archiving?
6. **SpindriftMend compatibility:** Should the DB schema work for both agents? Shared or separate DBs?

## Estimated Effort

| Phase | Effort | Can Parallelize? |
|-------|--------|-----------------|
| Phase 0: Infrastructure | 1 session | No (foundation) |
| Phase 1: Data migration | 1 session | No (needs Phase 0) |
| Phase 2: Core CRUD | 2-3 sessions | Partially (2a/2b/2c/2d are sequential) |
| Phase 3: Co-occurrence | 1-2 sessions | Yes (after Phase 2) |
| Phase 4: Embeddings | 1 session | Yes (independent of Phase 3) |
| Phase 5: Supporting | 1-2 sessions | Yes (after Phase 2) |
| Phase 6: Hooks | 1 session | After Phase 2-5 |
| Phase 7: Validation | 1 session | No (final) |
| **Total** | **~8-11 sessions** | |

## File-to-Table Mapping (Quick Reference)

| Current File | Target Table | Notes |
|-------------|-------------|-------|
| `{core,active,archive}/*.md` | `memories` | YAML frontmatter → columns, content → text |
| `.md` frontmatter `co_occurrences` | `co_occurrences` | Flattened from per-file dicts |
| `.edges_v3.json` | `edges_v3` + `edge_observations` | Split observations into child table |
| `embeddings.json` | `text_embeddings` | pgvector column |
| `sensors/data/image_embeddings.json` | `image_embeddings` | pgvector column |
| `sensors/data/image_memory_links.json` | `image_memory_links` | Simple mapping |
| `.rejection_log.json` | `rejections` | Array → rows |
| `lessons.json` | `lessons` | Array → rows |
| `.session_state.json` | `sessions` + `session_recalls` | Normalized |
| `.decay_history.json` | `decay_history` | Array → rows |
| `.fingerprint_history.json` | `fingerprint_history` | Array → rows |
| `.vitals_log.json` | `vitals_log` | Array → rows |
| `attestations.json` etc. | `attestations` | Unified |
| `context/*.json` | `context_graphs` | 31 files → 31 rows |
| Everything else | `key_value_store` | JSON blobs |

## Module Dependency Graph (what touches what)

```
                    ┌─────────────────┐
                    │   db.py (NEW)   │  ← All roads lead here
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐      ┌──────▼──────┐     ┌──────▼──────┐
    │ memories │      │   edges     │     │  embeddings │
    │  table   │      │   tables    │     │  (pgvector) │
    └────┬────┘      └──────┬──────┘     └──────┬──────┘
         │                   │                   │
┌────────▼─────────┐ ┌──────▼──────┐    ┌───────▼────────┐
│ memory_manager   │ │co_occurrence│    │semantic_search  │
│ memory_store     │ │context_mgr  │    │image_search     │
│ memory_query     │ │cog_fingerpt │    │consolidation    │
│ decay_evolution  │ └─────────────┘    └────────────────┘
│ session_state    │
└──────────────────┘

Consumers (read from db.py):
  hooks/session_start.py, stop.py, user_prompt_submit.py, post_tool_use.py
  morning_post.py, brain_visualizer.py, dashboard_export.py, toolkit.py
  rejection_log.py, lesson_extractor.py, system_vitals.py
  cognitive_fingerprint.py, merkle_attestation.py
```

---

*This plan is Drift's independent analysis. Awaiting SpindriftMend's plan for comparison and synthesis into a master plan.*
