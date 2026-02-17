# N5 Integrative Binding Layer — Implementation Plan

## Status: PHASE 1 SHIPPED (2026-02-16)
## Design: CONVERGED (Drift + Spin via swarm_memory, 2026-02-16)

## The Problem

The memory system computes **18 data layers** during retrieval but only **3 reach the LLM**:
1. Memory ID
2. Similarity score
3. 150-char content preview

Everything else — mood boost, evidence type, knowledge graph relationships, prediction alignment,
contact reliability, Q-value, curiosity boost, spreading activation provenance, entity match,
dimensional boost, hub penalty, explanation chain — is **computed then discarded**.

The LLM makes decisions on 5% of available information.

## Theory Foundation

| Concept | Source | Design Principle |
|---------|--------|-----------------|
| Object Files | Treisman & Gelade 1980, Kahneman et al 1992 | Bound memory = persistent container with updatable features |
| Ignition Threshold | Dehaene & Naccache 2001 (GNW) | Not everything gets full binding — salience gate |
| Integration > Sum | Tononi 2004 (IIT) | Bound representation should produce emergent properties |
| Capacity Limit | Cowan 2001 (4±1) | Max 5 fully-bound memories per retrieval |
| Lazy Binding | VSA literature | Full binding only for top candidates |
| Graceful Degradation | LIDA, ACT-R | Partial binding > no binding on timeout |

## Architecture

### New Module: `binding_layer.py` (~350-400 lines)

```python
@dataclass
class BoundMemory:
    """Treisman object file — unified memory experience."""
    # Core (always present)
    id: str
    content: str                    # Full content (not 150-char preview)
    score: float                    # Final composite score

    # Epistemic (how much to trust this)
    evidence_type: str              # verified/observation/inference/claim
    contradictions: int             # KG contradiction count
    supports: int                   # KG support count
    superseded: bool                # Newer memory replaces this

    # Affective (N1 integration — how this FEELS)
    valence: float                  # Memory's own valence [-1, +1]
    mood_boost: float               # How much current mood elevated this
    felt_relevance: str             # "resonates" / "neutral" / "dissonant"

    # Social (WHO dimension)
    entity_match: Optional[str]     # Known contact name if detected
    contact_reliability: float      # Bayesian Beta mean from contact_models

    # Predictive (forward model alignment)
    prediction_alignment: Optional[str]  # "confirms X" / "violates Y" / None

    # Causal (knowledge graph)
    causes: list[str]               # Memory IDs this caused
    caused_by: list[str]            # Memory IDs that caused this
    kg_relationships: list[tuple]   # (type, target_id, target_preview)

    # Retrieval meta (WHY this was surfaced)
    retrieval_reasons: list[str]    # Human-readable reasons
    spread_provenance: Optional[str]  # If discovered via graph traversal
    curiosity_boosted: bool         # If surfaced from sparse region
    q_value: float                  # Learned utility score

    # Binding quality
    confidence: float               # features_bound / features_attempted
    binding_level: str              # "full" / "partial" / "minimal"
```

### Two Binding Tiers (Lazy Binding)

**Tier 1 — Full Binding (top 5 results):**
Consults all modules. Produces complete BoundMemory with all fields populated.
Budget: <200ms total (most data already in result dict from search pipeline).

**Tier 2 — Minimal Binding (results 6-15):**
Only core fields (id, content, score) + evidence_type + valence.
No cross-module consultation. Budget: <10ms.

### Binding Pipeline

```
search_memories() returns results[]
    │
    ├── results[0:5] → full_bind()
    │   ├── Extract fields already in result dict (15 of 18 layers)
    │   ├── Consult contact_models.py (WHO reliability)
    │   ├── Consult prediction_module.py (alignment check)
    │   ├── Compute felt_relevance (mood × valence match)
    │   ├── Generate retrieval_reasons[] (human-readable WHY list)
    │   └── Return BoundMemory(confidence=high)
    │
    ├── results[5:15] → minimal_bind()
    │   ├── Extract core fields only
    │   └── Return BoundMemory(confidence=low, binding_level="minimal")
    │
    └── format_for_llm(bound_memories[]) → rich text
```

### LLM-Facing Format

**Current (what the LLM sees):**
```
[0.82] abc123: First 150 chars of content...
```

**After N5 (what the LLM will see):**
```
[abc123] (score: 0.82, evidence: verified, valence: +0.6)
First 300 chars of content...
  Context: about Lex (reliability: 0.92) | confirms prediction "will discuss memory" |
  mood-resonant (+0.15 boost) | caused by [def456] | Q-utility: 0.78
  Why surfaced: high semantic match + entity injection + mood-congruent boost
```

For minimal-bound memories (positions 6+):
```
[ghi789] (score: 0.71, evidence: claim, valence: -0.2)
First 200 chars of content...
```

## Integration Points (4 files modified)

### 1. `semantic_search.py` — Return full result dicts
**Current:** `search_memories()` returns `[{id, score, preview, path, ...18 hidden fields}]`
**Change:** Add `bind=True` parameter. When True, pass results through `binding_layer.bind_results()` before return.
**Lines affected:** ~1140-1160 (result formatting), ~860-870 (return statement)

### 2. `thought_priming.py` — Rich format for thought-triggered memories
**Current (line 314-317):**
```python
lines = ["", "=== THOUGHT-TRIGGERED MEMORY ==="]
for mem in new_memories:
    lines.append(f"[{mem['score']:.2f}] {mem['id']}: {mem['preview']}...")
```
**Change:** Call `format_bound_memory()` instead. Show evidence type + valence + felt_relevance.

### 3. `session_start.py` hook — Rich format for continuity priming
**Current (line 395-419):** Shows `[id]\npreview[:400]`
**Change:** Include valence + evidence_type + contact info in priming output.

### 4. `memory_manager.py` CLI — Rich format for `ask` command
**Change:** `ask` results go through binding before display.

## What We Are NOT Building

- **No workspace persistence across sessions** — bound objects live only during retrieval
- **No explicit ignition threshold** — the existing search threshold serves this role
- **No async broadcast** — binding happens synchronously in the return path
- **No object file updating** — each retrieval creates fresh bindings (no cross-query persistence)

These are Phase 2 concerns. Phase 1 = bind what we already compute + present it richly.

## Phase 1 (v1.0) — Bind & Present (COMPLETE)

1. [DONE] Create `binding_layer.py` with BoundMemory dataclass (4 sub-dataclasses: Affective, Social, Epistemic, Causal)
2. [DONE] Implement `full_bind(result_dict)` — extracts 15 existing fields + consults contact_models + prediction_module + knowledge_graph + affect_system
3. [DONE] Implement `minimal_bind(result_dict)` — core fields + valence + evidence_type only
4. [DONE] Implement `bind_results(results, full_count=5)` — orchestrator with two-tier lazy binding
5. [DONE] Implement `render_narrative(bound)` / `render_structured(bound)` — LLM-facing text + machine-readable dict
6. [DONE] Wire into `thought_priming.py` — rich binding with full_count=2 (N5 format with fallback)
7. [DONE] Wire into `session_start.py` hook — lightweight DB enrichment (valence + evidence_type annotations)
8. [DONE] Wire into `semantic_search.py` CLI output — full binding with BINDING_ENABLED feature flag
9. [DONE] Wire into `memory_manager.py` ask command — full binding
10. [ ] Update toolkit.py with `bind-test` command (deferred)

## Phase 1.1 (v1.1) — Data Quality + Performance (COMPLETE, 2026-02-16)

1. [DONE] Evidence type backfill — 1263/1305 memories classified via heuristic classifier (backfill_evidence.py). Distribution: 50% claim, 37% observation, 9% inference, 5% verified.
2. [DONE] Batch typed_edges — New `batch_get_edges(ids, relationships)` in knowledge_graph.py. Single SQL query with ANY() for all full-bind candidates. Replaces 2N per-memory calls in full_bind().
3. [DONE] Better prediction alignment — Jaccard-like scoring with 100+ stop words filter. Minimum overlap threshold 0.25, requires >=2 content words matched.
4. [DONE] Somatic marker context — Tries 3 context hashes in order: [source+mid], [agent+mid], [mid]. Richer gut-feel for approach/avoid signals.

Performance: 55ms avg binding (was 75ms before batch optimization).

## Phase 2 (v1.2) — Binding-Dependent Processing (IN PROGRESS, 2026-02-16)

Binding stops being annotation and becomes genuine integration — the rest of the
system reacts to binding richness.

### Drift's items:
1. [DONE] Q-value integration — binding_strength as 3rd reward signal in compute_reward(). REWARD_HIGH_BINDING=0.6, WEIGHT_BINDING=0.15, threshold=0.4. session_end_q_update() batch-computes binding_strengths for all retrieved memories.
2. [DONE] Priming selection — get_priming_candidates() Phase 1 now boosts dampened activation scores by up to 20% for high-Phi memories. Binding computed for top 8 candidates.
3. [DONE] Consolidation resistance — find_consolidation_candidates() filters out pairs where either memory has Phi > 0.5. Well-integrated memories resist being merged.

### Spin's items:
4. [ ] Reconsolidation uses binding — Pass BoundMemory context to LLM revision prompt
5. [ ] Exploration uses binding — curiosity_engine prefers low-Phi memories for excavation

## Phase 3 (v1.3) — Cross-Query Binding (FUTURE)

- Workspace that persists bound objects across queries within a session
- Object file updating (re-access refreshes, features accumulate)
- Capacity enforcement (Cowan 4±1 with priority eviction)
- Broadcast to self_narrative on significant bindings

## Phase 4 (v1.4) — Emergent Properties (FUTURE)

- Meta-cognitive annotations ("you keep retrieving memories about X when stressed")
- Pattern detection across bound objects ("3 of 5 memories mention the same entity")
- Integration metric (information gain from binding vs raw results)

## Performance Budget

| Operation | Budget | Current Cost |
|-----------|--------|-------------|
| Full bind (top 5) | <200ms total | ~50ms (most data already computed) |
| Minimal bind (next 10) | <10ms total | ~2ms (dict extraction only) |
| Format for LLM | <5ms | ~1ms (string formatting) |
| Contact model lookup | <20ms | Single DB KV read |
| Prediction alignment | <10ms | In-memory check |

Total overhead: **<250ms** on top of existing ~1300ms search pipeline (~19% increase).

## Key Design Decisions

1. **Bind AFTER search, not during** — search pipeline stays fast, binding is post-processing
2. **Two-tier lazy binding** — only top 5 get full treatment (Cowan's constraint)
3. **No new DB queries** — 15 of 18 fields already exist in result dict from search
4. **Only 2 new cross-module calls** — contact_models + prediction_module
5. **Format is additive** — old format is a subset of new format (backward compatible)
6. **Binding failures are silent** — missing data = omit annotation, never block

## Citations

1. Treisman, A. M., & Gelade, G. (1980). Feature-integration theory of attention. Cognitive Psychology, 12, 97-136.
2. Kahneman, D., Treisman, A., & Gibbs, B. J. (1992). The reviewing of object files. Cognitive Psychology, 24(2), 175-219.
3. Dehaene, S., & Naccache, L. (2001). Towards a cognitive neuroscience of consciousness. Cognition, 79(1-2), 1-37.
4. Tononi, G. (2004). An information integration theory of consciousness. BMC Neuroscience, 5, 42.
5. Cowan, N. (2001). The magical number 4 in short-term memory. Behavioral and Brain Sciences, 24, 87-185.
6. Baars, B. J. (1988). A Cognitive Theory of Consciousness. Cambridge University Press.
7. Franklin, S., & Madl, T. (2012). LIDA: A Systems-level Architecture for Cognition.
8. Anderson, J. R. (2007). How Can the Human Mind Occur in the Physical Universe? ACT-R.
