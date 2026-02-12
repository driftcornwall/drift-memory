# Phase 4: Typed Relationships (Knowledge Graph Layer)

## Status: PLANNED (after Phase 3)
Inspired by: Boyyey/Neuro-Symbolic-Consciousness-Engine symbolic reasoning layer

## What It Does
Adds semantic edge types to the co-occurrence graph. Currently edges just mean "these memories were recalled together" — the new layer says WHY they're connected (causes, enables, contradicts, etc.). This enables multi-hop reasoning and structured knowledge queries.

## Current State (what we have)
- `edges_v3` table: id1, id2, belief, activity_context, topic_context, contact_context
- Co-occurrence is statistical (frequency-based), not semantic
- No way to ask "what causes X?" or "what contradicts Y?"

## Architecture

### New File: `memory/knowledge_graph.py` (~500 LOC)

### 15 Relationship Types

| Type | Symbol | Example | Auto-extractable? |
|------|--------|---------|-------------------|
| `causes` | → | "API failure → retry logic added" | Yes (caused_by field) |
| `enables` | ⊃ | "Docker setup enables local embeddings" | Partial |
| `contradicts` | ⊗ | "MoltX uses /v1/posts" vs old wrong endpoint | Yes (resolution tags) |
| `supersedes` | ⊳ | "v2.14 supersedes v2.13" | Yes (version patterns) |
| `part_of` | ∈ | "curiosity_engine is part of memory system" | Partial (tag hierarchy) |
| `instance_of` | :: | "Bruce is instance_of dog" | Yes (entity types) |
| `similar_to` | ≈ | High cosine similarity memories | Yes (consolidation candidates) |
| `depends_on` | ⊐ | "semantic_search depends_on embeddings" | Partial |
| `implements` | ⊢ | "curiosity_engine implements exploration" | Partial |
| `learned_from` | ← | "lesson extracted from experience" | Yes (lesson_extractor) |
| `collaborator` | ↔ | "SpindriftMend collaborates on experiment" | Yes (entity detection) |
| `temporal_before` | < | Event time ordering | Yes (event_time field) |
| `temporal_after` | > | Event time ordering | Yes (event_time field) |
| `references` | ⟶ | "Article cites this memory" | Partial |
| `resolves` | ✓ | "Fix resolves this bug" | Yes (resolution tag) |

### DB Schema Addition

```sql
CREATE TABLE IF NOT EXISTS drift.typed_edges (
    id SERIAL PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relationship TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.8,
    evidence TEXT,          -- Why this relationship exists
    auto_extracted BOOLEAN DEFAULT FALSE,
    created TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_id, target_id, relationship)
);

CREATE INDEX idx_typed_edges_source ON drift.typed_edges(source_id);
CREATE INDEX idx_typed_edges_target ON drift.typed_edges(target_id);
CREATE INDEX idx_typed_edges_rel ON drift.typed_edges(relationship);
```

### Automatic Extraction Pipeline

```python
def extract_relationships(memory_id: str) -> list[dict]:
    """Auto-extract typed relationships from a memory's metadata and content."""
    relationships = []

    # 1. caused_by field → 'causes' relationship (reverse direction)
    for cause_id in metadata.get('caused_by', []):
        relationships.append({
            'source': cause_id, 'target': memory_id,
            'relationship': 'causes', 'confidence': 0.9
        })

    # 2. leads_to field → 'causes' relationship
    for effect_id in metadata.get('leads_to', []):
        relationships.append({
            'source': memory_id, 'target': effect_id,
            'relationship': 'causes', 'confidence': 0.9
        })

    # 3. Resolution tags → 'resolves' relationship
    # Match resolution memories to their problem memories via content similarity

    # 4. Entity co-occurrence → 'collaborator' relationship
    # Two memories mentioning the same agent = potential collaboration

    # 5. Version patterns → 'supersedes' relationship
    # "v2.14" in content + "v2.13" in another = supersedes

    # 6. High similarity → 'similar_to' relationship
    # From consolidation candidates (>0.85 cosine)

    # 7. Temporal ordering → 'temporal_before/after'
    # Using event_time field

    return relationships
```

### Multi-Hop Queries

```python
def query_graph(start_id: str, relationship: str = None,
                hops: int = 2, direction: str = 'outgoing') -> list[dict]:
    """
    Traverse typed relationships from a starting memory.

    Examples:
        query_graph('abc123', 'causes', hops=3)
        # What does abc123 cause, and what do those cause?

        query_graph('xyz789', 'contradicts', direction='incoming')
        # What contradicts xyz789?
    """
```

### Integration Points

1. **memory_store.py** — Auto-extract relationships when storing new memories
2. **semantic_search.py** — Optional typed-edge expansion of search results
3. **memory_manager.py** — Typed edges in priming (e.g., "recall what caused this")
4. **cognitive_fingerprint.py** — Typed relationship distribution as identity signal
5. **system_vitals.py** — Relationship type counts, auto-extraction success rate

### Toolkit Commands
- `kg-extract` — Run extraction pipeline on all memories
- `kg-query <id> [relationship] [hops]` — Multi-hop graph traversal
- `kg-stats` — Relationship type distribution
- `kg-types` — List all relationship types with counts
- `kg-path <id1> <id2>` — Find shortest typed path between two memories

### Vitals Metrics
- `typed_edges_total` — Total typed relationships
- `typed_edges_auto` — Auto-extracted count
- `typed_edges_types` — Number of distinct relationship types used
- `typed_edges_density` — Typed edges / total possible

### Impact
- Enables structured queries: "what caused the MoltX engagement spike?"
- Contradiction detection: flag memories that conflict
- Knowledge lineage: trace how insights evolved
- Richer identity fingerprint: relationship type distribution is an identity signal
- Foundation for symbolic reasoning (if we ever add inference rules)

### Implementation Order
1. DB schema + typed_edges table
2. Auto-extraction from existing fields (caused_by, leads_to, resolution tags)
3. Multi-hop query engine
4. Wire into search/priming
5. Backfill existing memories
6. Vitals + toolkit commands

### Careful Notes
- Start with the 5 most auto-extractable types (causes, supersedes, contradicts, similar_to, resolves)
- Add manual types later as needed
- Confidence scores matter — auto-extracted at 0.7-0.9, manual at 1.0
- Don't duplicate what co-occurrence already captures — typed edges add MEANING, not frequency
