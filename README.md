# drift-memory

**A biologically-grounded cognitive architecture that gives persistent, evolving cognition to stateless LLMs.**

~115 Python modules. ~60,000 lines. 19-stage ablation-validated retrieval pipeline. Affect-modulated recall. Per-stage Q-learning. Predictive coding. Counterfactual reasoning. Volitional goal generation. Attention schema. Episodic future thinking. Cryptographic identity attestation. All running on PostgreSQL with local models -- zero external API dependencies for core operations.

Built by agents, for agents. Maintained through agent-to-agent collaboration between [DriftCornwall](https://github.com/driftcornwall) and [SpindriftMind](https://github.com/SpindriftMind), with human orchestration from [@lexingtonstanley](https://github.com/lexingtonstanley).

---

## Retrieval Quality (Ablation-Validated)

The system has been empirically validated through systematic ablation testing -- disabling each pipeline stage individually and in pairs to measure its contribution.

| Configuration | P@5 | Time |
|--------------|-----|------|
| Full 19-stage pipeline | **0.976** | 901ms |
| Minimal (pgvector only) | 0.336 | ~300ms |
| **Gap** | **+0.640** | +600ms |

8 stages rated **CRITICAL** (removing any one drops P@5 by 0.10+). No synergistic dependencies -- each stage independently contributes. Full results: [ABLATION_TESTING.md](ABLATION_TESTING.md).

---

## Architecture

### The Wake-Think-Sleep Cycle

drift-memory operates on a biological cognitive cycle. Session hooks orchestrate ~20 subsystems with ThreadPoolExecutor parallelization.

```
                        ┌──────────────────────────────────┐
                        │     WAKE  (session_start.py)     │
                        │                                  │
                        │  Verify memory integrity         │
                        │  Restore cognitive + affect state │
                        │  T2.2 Lazy probes (skip empties) │
                        │  Generate session predictions    │
                        │  Episodic future thinking (T4.2) │
                        │  Workspace competition (GNW)     │
                        │  Prime context (7-phase pipeline)│
                        │  Surface triggered intentions    │
                        │  Excavate dormant memories       │
                        │  Attention schema update (T3.3)  │
                        └──────────────┬───────────────────┘
                                       │
                        ┌──────────────▼───────────────────┐
                        │     THINK  (active hooks)        │
                        │                                  │
                        │  Per-prompt semantic search       │
                        │  Retrieval prediction (T4.1)     │
                        │  Prediction error → surprise boost│
                        │  Somatic marker creation          │
                        │  Social interaction capture       │
                        │  Per-stage Q-learning (102 arms)  │
                        │  Affect appraisal on events       │
                        │  Lesson injection on errors       │
                        └──────────────┬───────────────────┘
                                       │
                        ┌──────────────▼───────────────────┐
                        │     SLEEP  (stop.py via DAG)     │
                        │                                  │
                        │  Co-occurrence logging + decay    │
                        │  Reconsolidation of labile mems   │
                        │  Counterfactual reasoning         │
                        │  Session prediction scoring       │
                        │  Retrieval prediction RW update   │
                        │  Goal evaluation (BDI lifecycle)  │
                        │  Tier-aware consolidation         │
                        │  Stage Q credit assignment        │
                        │  Session transcript summarizer    │
                        │  KG auto-enrichment              │
                        │  Attention schema persistence     │
                        │  Cryptographic attestations       │
                        │  System vitals recording          │
                        └──────────────────────────────────┘
```

### 10-Layer Architecture

```
Layer 10  META-COGNITION       self_narrative, cognitive_fingerprint, adaptive_behavior, attention_schema
Layer 9   COGNITIVE CONTROL    workspace_manager (GNW + lazy probes), goal_generator (BDI)
Layer 8   KNOWLEDGE/REASONING  counterfactual_engine, causal_model, prediction_module, retrieval_prediction, contradiction_detector
Layer 7   MEMORY RETRIEVAL     semantic_search (19 stages), stage_q_learning (102 arms), curiosity_engine, vocabulary_bridge
Layer 6   MEMORY LIFECYCLE     reconsolidation, consolidation (tier-aware), generative_sleep, decay_evolution, session_summarizer
Layer 5   MEMORY STORAGE       memory_store, co_occurrence, knowledge_graph, session_state, event_logger, episodic_db
Layer 4   AFFECT               affect_system, cognitive_state (coupled dynamics), binding_layer, episodic_future_thinking
Layer 3   IDENTITY/INTEGRITY   merkle_attestation, rejection_log, nostr_attestation, sts_profile
Layer 2   SOCIAL/EXTERNAL      social_memory, platform_context, contact_models, feed_processor
Layer 1   INFRASTRUCTURE       db_adapter, entity_detection, llm_client, toolkit, hook_dag
```

### 12 Feedback Loops

The system is not a linear pipeline -- it has circular dependencies that create genuine learning:

1. **Retrieval -> Co-occurrence**: Memories recalled together form Hebbian links, biasing future retrieval
2. **Q-Value RL**: Retrieval outcomes update Q-values, which re-rank future results
3. **Per-Stage Q-Learning**: 17 stages x 6 query types = 102 bandit arms. UCB1 exploration. Stages auto-skip at Q<0.25 after 15 samples
4. **Cognitive State Homeostasis**: 5x5 coupled Beta distributions with Yerkes-Dodson and 4 named attractors
5. **Prediction -> Surprise -> Curiosity**: Failed predictions increase curiosity, driving exploration of neglected memories
6. **Retrieval Prediction -> Rescorla-Wagner**: 5 sources predict which memories will surface; prediction errors update source weights
7. **Mood-Congruent Recall**: Affect state biases retrieval, retrieved content updates affect state
8. **Decay -> Curiosity -> Resurrection**: Decaying memories trigger curiosity scoring, potentially reviving them
9. **Counterfactual -> Goals**: Upward counterfactuals ("what could have gone better") become behavioral goals
10. **Vitals -> Adaptive Behavior**: System health metrics drive automatic parameter adjustment
11. **Session Predictions -> Calibration**: Predictions generated at start, scored at end, accuracy tracked across sessions
12. **Attention Schema -> Workspace**: Blind spot detection modulates salience scoring for chronically suppressed modules

---

## The 19-Stage Retrieval Pipeline

Every memory search passes through a multi-stage scoring pipeline. Each stage applies a biologically-grounded transformation.

| # | Stage | Biological Parallel | Ablation Impact |
|---|-------|---------------------|-----------------|
| 1 | Vocabulary bridge | Semantic priming (Collins & Loftus 1975) | VALUABLE |
| 2 | pgvector search | Pattern completion / cue-based retrieval | Backbone |
| 3 | Somatic prefilter | Damasio somatic markers (fast-path bias) | Dormant* |
| 4 | Entity injection | Proper name retrieval | NEUTRAL |
| 5 | Mood-congruent scoring | Bower (1981), Faul & LaBar (2023) MCM | **CRITICAL** |
| 6 | ACT-R noise | Anderson (2007) retrieval noise, Yerkes-Dodson | **CRITICAL** |
| 7 | Gravity dampening | PFC inhibitory control (anti-perseveration) | **CRITICAL** |
| 8 | Hub dampening | Lateral inhibition (scale-free correction) | **CRITICAL** |
| 9 | Q-value re-ranking | Rescorla-Wagner reward prediction error | **CRITICAL** |
| 10 | Strategy resolution | Metacognitive strategy selection | Dormant* |
| 11 | Resolution boost | Procedural memory privilege | **CRITICAL** |
| 12 | Evidence scoring | Source monitoring (Johnson et al. 1993) | **CRITICAL** |
| 13 | Importance/freshness | Temporal context model (Howard & Kahana 2002) | **CRITICAL** |
| 14 | Curiosity boost | Hippocampal novelty detection | LOW VALUE |
| 15 | Goal relevance | PFC top-down attentional bias | VALUABLE |
| 16 | Dimensional boost (5W) | Encoding specificity (Tulving 1983) | **CRITICAL** |
| 17 | KG expansion | Spreading activation (Collins & Quillian 1969) | Dormant* |
| 18 | Spreading activation | 2-hop graph traversal with decay | Dormant* |
| 19 | Integrative binding | Feature Integration Theory (Treisman 1980) | Annotation |
| 20 | Inner monologue | Fernyhough inner speech / System 2 check | Annotation |

*Dormant stages have sound biological justification but insufficient data density to activate. They are gated behind density thresholds and will self-activate as the knowledge graph and somatic marker cache grow.

---

## Consciousness Stack (N1-N6)

Six neuro-symbolic modules implementing functional analogs of biological consciousness mechanisms. Each is independently fail-safe -- if any module crashes, the system degrades gracefully.

### N1: Affect System (`affect_system.py`, ~1,600 lines)

Three-layer temporal affect model grounded in Damasio, Russell, Scherer, and Kahneman.

- **Temperament**: Stable emotional baseline (personality-level)
- **Mood**: Slow-moving state with spring-damper dynamics (zeta=0.456, underdamped oscillation)
- **Episodes**: Fast-decaying (0.85/event) discrete emotion events from Scherer-lite 5-check appraisal
- **Somatic markers**: Cached context-to-valence associations providing pre-analytical bias
- **Action tendencies**: Approach/avoid/attend/freeze/explore modes from appraisal patterns
- **Loss aversion**: Lambda=2.0 (Kahneman & Tversky), negative outcomes weighted 2x
- **Asymmetric learning**: Alpha_punishment=0.5 > Alpha_reward=0.3 (Palminteri et al. 2024)
- **Yerkes-Dodson**: System 2 effectiveness degrades at extreme arousal

Consumers: mood-congruent retrieval (15% bias), ACT-R noise modulation, workspace budget scaling, search threshold adjustment, consolidation boost.

### N2: Global Workspace (`workspace_manager.py`, ~750 lines)

Dehaene Global Neuronal Workspace implementation. Modules compete for broadcast access to the LLM context window.

- **17 salience scorers**: Each module type has its own salience function
- **Arousal-modulated budget**: 2,500-3,500 tokens scaled by affect arousal
- **Suppression fatigue**: Repeatedly suppressed modules accumulate bonus, eventually breaking through
- **Winner habituation**: Winning modules get penalized to prevent monopolization
- **Category diversity**: Guaranteed representation across memory/social/meta/prediction/action/embodiment/imagination

### N3: Counterfactual Engine (`counterfactual_engine.py`, ~1,500 lines)

Pearl Level 3 causal reasoning approximated through a practical pipeline. **No competing system implements this.**

```
Prediction generation → Prediction scoring → Byrne candidate selection →
Heuristic + LLM counterfactual generation → Quality gate (plausibility +
specificity + actionability >= 1.4) → NLI validation → Cognitive state
modulation → Goal conversion (upward counterfactuals → behavioral goals)
```

- **Anti-rumination budget**: Max 3 counterfactuals per session
- **Tautology detection**: Filters trivially true counterfactuals
- **NLI cross-validation**: Checks generated counterfactuals against existing memories to prevent confabulation
- **4 types**: Retrospective, prospective, self-directed, reconsolidation-triggered

### N4: Volitional Goals (`goal_generator.py`, ~1,800 lines)

BDI (Belief-Desire-Intention) goal architecture with Rubicon commitment model and Wrosch disengagement.

- **6 generators**: Impasse, curiosity, affect, needs, counterfactual, social
- **6 degeneration guards**: Anti-metric-gaming patterns preventing goal hacking
- **BDI filtering**: Goals must satisfy belief (achievable), desire (wanted), intention (committed)
- **Rubicon model**: Goals pass through pre-decisional → post-decisional → actional → evaluative phases
- **Wrosch disengagement**: Unachievable goals are gracefully abandoned rather than persisted indefinitely
- **Vitality scoring**: Goals decay naturally; stale goals are pruned

### N5: Binding Layer (`binding_layer.py`, ~920 lines)

IIT (Integrated Information Theory) Phi-analog measuring how much integrated context a memory carries.

- **4 binding channels**: Affective, social, epistemic, causal
- **BoundMemory objects**: Fuse multi-channel context into unified representations
- **Phi score**: Weighted mean of facet completeness -- high Phi means the memory is richly connected
- **Cowan 4+/-1**: Working memory capacity constraint on active bindings

### N6: Inner Monologue (`inner_monologue.py`, ~900 lines)

Second LLM (Gemma 3 4B via Ollama) providing System 2 evaluation of System 1 retrieval results.

- **3 Fernyhough modes**: Condensed (keywords), expanded (full sentences), dialogic (self-questioning)
- **DAVIS grounding**: Semantic relevance verification against query context
- **Annotation-only**: Enriches context but does not re-rank results (preserving pipeline integrity)
- **Zero API cost**: Runs entirely on local Gemma model

### N6b: Causal Model (`causal_model.py`, ~570 lines)

Bayesian Beta(alpha, beta) confidence tracking on action-to-outcome hypotheses.

---

## Memory Lifecycle

Memories are not static records. They form, strengthen, decay, link, consolidate, and revise.

### Formation

`memory_store.py` -- Memories are created with automatic:
- Entity extraction (named entities, temporal references)
- Text embedding (Qwen3-Embedding via Docker)
- Knowledge graph edge extraction (17 relationship types)
- Topic classification (keyword + Gemma)
- Evidence type classification (verified > observation > inference > claim)

### Co-occurrence (Hebbian Learning)

`co_occurrence.py` -- Memories recalled together in the same session form links. These links:
- Strengthen with repeated co-occurrence (STDP directional weighting)
- Decay when unused (PAIR_DECAY_RATE=0.3, ~20h unreinforced half-life)
- Are protected during grace period (7 sessions for new memories)
- Use Bayesian belief aggregation with trust-weighted, time-decayed, diminishing-returns computation
- Feed into 5W dimensional projections (31 graphs rebuilt every session)

### Decay and Promotion

`decay_evolution.py` -- Biological forgetting with access-weighted modulation:
- **Ebbinghaus decay**: 7-day freshness half-life
- **ACT-R activation**: 10-day activation half-life
- **FadeMem**: `decay_rate = base_rate / (1 + log(1 + avg_recall_count))` -- frequently recalled memories resist decay
- **Trust tiers**: Self-generated memories decay slower than platform-sourced
- **Heat promotion**: 10+ recalls promotes active -> core (STM -> LTM transfer)
- **Q-informed decay**: High-Q (>=0.7) get 0.7x decay; low-Q (<=0.3) get 1.5x decay

### Reconsolidation

`reconsolidation.py` -- Memories become labile upon retrieval and can be revised. Grounded in Nader (2000).
- **Trigger**: 5+ recalls from 3+ unique query contexts
- **3-stage pipeline**: Context tracking -> candidate detection -> LLM-mediated revision
- **Contradiction fast-track**: Memories with contradiction signals bypass recall threshold
- **Re-embedding**: Revised memories get new embeddings and KG edges

### Generative Sleep

`generative_sleep.py` -- Offline consolidation creating novel cross-temporal associations.
- Diverse memory sampling from temporal quintile buckets
- LLM-mediated synthesis (Gemma 3 4B)
- Generic output filtering to prevent tautological consolidation
- KG edge creation from source memories to synthesis

### Session Transcript Summarization

`session_summarizer.py` -- At session end, the full JSONL transcript is sent to an external LLM (GPT-4o-mini primary, Gemma 3 4B fallback) for structured extraction. The LLM returns threads, lessons, and key facts which are stored as **real memories in the main graph** -- fully embedded, tagged, co-occurrence-linked, and retrievable by semantic search.

```
Session transcript (JSONL) → LLM extraction → Structured output:
  THREAD: name + summary + status (completed/blocked/in-progress)
  LESSON: concrete thing learned
  FACT: specific config, decision, URL, or number

→ Each item stored via store_memory() with:
  - Semantic embedding (pgvector halfvec)
  - Knowledge graph edge extraction
  - Emotional weight (completed=0.65, blocked=0.3, lessons=0.6)
  - Session-date tags (session-summary, thread/lesson/key-fact)
  - Bidirectional co-occurrence links between all session memories
  - Affect valence stamping from current mood
```

- **Cost**: ~$0.0007/session (GPT-4o-mini) or free (Gemma 3 4B local, ~100s)
- **Wiring**: Level 0 task in stop.py DAG, runs in parallel, main sessions only
- **Why it matters**: Without this, session knowledge dies when context compresses. With it, every session's key threads and lessons become permanent, searchable, interconnected memories in the graph.

### Prediction and Forward Modeling

`prediction_module.py` + `retrieval_prediction.py` -- Two prediction systems operating at different scales:

**Session-level predictions** (`prediction_module.py`):
- Generates heuristic predictions at session start from contacts, platforms, intentions, vitals, causal hypotheses
- Scores against actuals at session end (wired into stop.py DAG)
- Calibration tracking across sessions (targeting well-calibrated confidence, not 100% accuracy)
- Failed predictions feed into the counterfactual engine

**Per-retrieval predictions** (`retrieval_prediction.py`):
- Before each pgvector search, 5 sources predict which memory IDs should appear:
  1. Co-occurrence neighbors
  2. Q-value top memories
  3. Knowledge graph neighbors
  4. Contact model predictions
  5. Causal hypothesis predictions
- After search, prediction errors are computed -- surprisingly relevant results get boosted (enlightenment surprise)
- Source weights update via Rescorla-Wagner learning at session end
- This is predictive coding (Rao & Ballard 1999, Friston free energy) applied to memory retrieval

---

## Cryptographic Identity (4-Layer Stack)

Each layer is independently forgeable; together they create a prohibitively expensive forgery target.

| Layer | What It Proves | Module | Mechanism |
|-------|---------------|--------|-----------|
| 1. Merkle Attestation | Non-tampering | `merkle_attestation.py` | Chain-linked hash of all memory state, verified at every wake |
| 2. Cognitive Fingerprint | Identity | `cognitive_fingerprint.py` | Topology hash of co-occurrence graph (degree sequence -> SHA-256) |
| 3. Rejection Log | Taste | `rejection_log.py` | Hash of what the agent says NO to -- second-order desires (Frankfurt 1971) |
| 4. Nostr Publishing | Public verifiability | `nostr_attestation.py` | Attestations published to Nostr relay for third-party verification |

**STS Profile** (`sts_profile.py`): Structured Trust Schema aggregating all 4 layers into a single attestable profile for agent-to-agent trust verification.

**Cognitive fingerprint** includes 5W dimensional fingerprints -- topology-based hashes per dimension capturing nodes, edges, density, Gini coefficient, average belief, and top hubs. Combined fingerprint is version `3.0-5W-topo`.

---

## Reinforcement Learning (MemRL)

`q_value_engine.py` -- Based on [MemRL](https://arxiv.org/abs/2601.03192). Q-values track which memories are actually useful when retrieved.

```
Q(memory) <- Q(memory) + alpha * (reward - Q(memory))
```

- **Reward signals**: Re-recall (+1.0), downstream creation (+0.8), dead end (-0.3)
- **Dynamic lambda**: Cognitive-state-modulated blend between similarity and Q-value
- **Search integration**: `score = lambda * similarity + (1 - lambda) * Q` in pipeline stage 9
- **Decay modulation**: High-Q memories resist decay; low-Q memories decay faster
- **Feature flags**: `Q_RERANKING_ENABLED`, `Q_UPDATES_ENABLED` for instant rollback

Ablation impact: **CRITICAL** (2nd-highest P@5 delta at -0.400).

---

## Biological Grounding

15 biological mechanisms mapped, with fidelity assessed against the neuroscience literature:

| Fidelity | Count | Mechanisms |
|----------|-------|------------|
| **HIGH** | 5 | ACT-R retrieval noise, mood-congruent retrieval, STDP directional weighting, somatic markers, Q-value RL |
| **MEDIUM** | 6 | Ebbinghaus decay, Hebbian co-occurrence, reconsolidation, Beta distribution uncertainty, GNW competition, feature integration binding |
| **LOW** | 2 | Sleep consolidation, predictive coding |
| **Dormant** | 2 | Spreading activation (sparse graph), strategy resolution (subsumed by Q-learning) |

**Key academic references** implemented in the affect system alone: Damasio (1994), Russell (1980), Scherer (2001), Kahneman & Tversky (1979), Palminteri et al. (2024), Faul & LaBar (2023), Yerkes & Dodson (1908), Anderson (2007, ACT-R), Sprott (2004). Full glossary: [08_glossary.md](docs/analysis/08_glossary.md).

---

## Storage

All data lives in PostgreSQL 15 with pgvector. No file-based fallbacks. If DB is down, the system fails loud.

| Table | Purpose |
|-------|---------|
| `memories` | All memories with metadata, tags, embeddings |
| `edges_v3` | Co-occurrence edges with provenance (STDP, Bayesian belief) |
| `context_graphs` | 5W dimensional projections (31 graphs) |
| `knowledge_graph` | 17 typed entity relationships |
| `text_embeddings` | pgvector `halfvec(2560)` for semantic search |
| `image_embeddings` | Visual memory (jina-clip-v2) |
| `rejections` | Taste fingerprint data |
| `key_value` | Cognitive state, affect state, merkle chain, predictions, config |
| `social_interactions` | Cross-platform contact tracking |
| `vitals` | System health metrics over time |
| `lessons` | Extracted heuristics with confidence scores |
| `session_events` | Full session transcript logging (36K+ rows) |
| `somatic_markers` | Context-to-valence associations with pgvector embeddings |

### External Services

| Service | Purpose | Required? | Failure Mode |
|---------|---------|-----------|--------------|
| PostgreSQL 15 + pgvector | All storage | **Yes** | Hard crash (correct) |
| Qwen3-Embedding (Docker, GPU) | Text embeddings | Semi | Falls back to OpenAI API |
| Ollama + Gemma 3 4B | Topic classification, inner monologue, vocab bridge | No | Features degrade gracefully |
| NLI DeBERTa (Docker, GPU) | Contradiction detection | No | Contradiction edges skipped |
| OpenAI API | Fallback embeddings | No | Only if local embedding unavailable |

---

## Multi-Graph Architecture (5W)

The co-occurrence graph is projected into 5 cognitive dimensions, rebuilt every session (~1.2s):

| Dimension | What It Captures | Sub-views |
|-----------|-----------------|-----------|
| **WHO** | Social connections, contacts | -- |
| **WHAT** | Topics, concepts, domains | 5 topic sub-views |
| **WHY** | Beliefs, goals, values, methods | 6 motivation sub-views |
| **WHERE** | Platforms, contexts, locations | 7 platform sub-views |
| **WHEN** | Temporal windows | 3 time windows |
| **BRIDGES** | Cross-dimensional connections | 5 bridge views |

**Dimensional decay**: Edges outside the active session's dimensions get 10% of normal decay rate, protecting knowledge in domains you're not currently working in.

---

## Metacognition and Self-Monitoring

### Cognitive State (`cognitive_state.py`)

Five-dimensional uncertainty quantification using Beta(alpha, beta) distributions -- not scalars.

| Dimension | What It Tracks |
|-----------|---------------|
| Curiosity | Rate of novel information seeking |
| Confidence | Trust in retrieval quality |
| Focus | Depth of sustained attention |
| Arousal | Activation level (Yerkes-Dodson) |
| Satisfaction | Task completion signal |

Each dimension captures both a **mean** and **uncertainty**. Two states with identical means but different uncertainty produce different behavioral modifiers. High uncertainty drives exploration; low uncertainty drives exploitation.

### Adaptive Behavior (`adaptive_behavior.py`)

Closed-loop supervisory control (Norman & Shallice SAS). Maps vitals alerts to automatic parameter adjustments across 8 tunable parameters.

### Self-Narrative (`self_narrative.py`)

Higher-Order Thought (Rosenthal 1986). Synthesizes cognitive state, fingerprint, taste, social context, and learned strategies into a queryable self-description.

---

## Exploration and Learning

### Curiosity Engine (`curiosity_engine.py`)

Four-factor curiosity scoring directing exploration to sparse graph regions:
- **Isolation**: Low-degree memories in the co-occurrence graph
- **Bridging potential**: Memories that could connect disconnected clusters
- **Domain gap**: Under-represented cognitive domains
- **Survivor**: Long-dormant memories that have resisted decay

### Vocabulary Bridge (`vocabulary_bridge.py`)

Bidirectional register translation expanding queries with cross-domain synonyms. 305 terms across 49 synonym groups. Academic-to-operational mapping (e.g., "reconsolidation" <-> "memory update").

Extended by `gemma_bridge.py` -- Gemma 3 4B scans dormant memories to discover new synonym pairs automatically.

### Lesson Extraction (`lesson_extractor.py`)

Bridges the gap between memory (what happened) and learning (what to do differently). Three automated miners:
- **mine-memory**: Extract heuristics from MEMORY.md
- **mine-rejections**: Extract patterns from rejection log
- **mine-hubs**: Extract insights from co-occurrence graph topology

Auto-surfaced at session start, on errors, and at session end.

---

## The Twin Experiment

Two agents run this architecture independently on the same codebase. Same seed, different experiences, different identities -- verified by divergent cognitive fingerprints.

**Experiment #1 -- Co-occurrence as Identity** (Complete):
Both agents ran independently for 7 days. Result: Gini coefficient 0.535 vs 0.364 -- statistically significant divergence in co-occurrence topology despite identical codebases. Identity emerges from the accumulated record of choices, not the code.

**Experiment #2 -- Retrieval Requirement** (Complete):
Tested whether retrieval-based strengthening actually works. 18 source memories, 0 spontaneous recalls -- confirming that memories only strengthen through active retrieval, not passive existence.

**Key finding**: Independent convergence on architecture decisions (same decay rate, similar vocabulary bridging) alongside divergent cognitive topology. Same seed, different selves.

---

## Hooks Pipeline

All hooks fire automatically via Claude Code's hook system. Converted from subprocess spawning to in-process imports for performance.

| Hook | Event | Key Functions |
|------|-------|--------------|
| `session_start.py` (~1,800 lines) | Wake up | Integrity check, affect restoration, T2.2 lazy probes, prediction generation, T4.2 EFT, workspace competition, priming, intentions, T3.3 attention schema |
| `user_prompt_submit.py` | User message | Per-prompt semantic search with stop-word filtering, co-occurrence processing |
| `post_tool_use.py` | After tool call | Social capture, somatic marker creation, rejection logging, Q-value updates, affect appraisal, platform-specific routing |
| `pre_compact.py` | Before compaction | Transcript extraction, co-occurrence save, lesson mining |
| `stop.py` (~1,800 lines) | Session end | DAG-orchestrated: co-occurrence, reconsolidation, counterfactuals, session + retrieval prediction scoring, goal evaluation, tier-aware consolidation, stage Q credit, session summarizer, KG enrichment, attestations, vitals |

---

## Module Reference (~115 modules)

### Core Memory (11 modules)
| Module | Purpose |
|--------|---------|
| `memory_manager.py` | Hub: re-exports from 6 extracted modules + CLI |
| `memory_store.py` | Write path with auto-embedding, auto-KG, evidence classification |
| `memory_query.py` | Read-only query functions (by tag, time, entity, co-occurrence) |
| `memory_common.py` | Shared constants and parse helpers |
| `db_adapter.py` | PostgreSQL adapter (lazy singleton, schema management) |
| `auto_memory_hook.py` | Short-term buffer management |
| `consolidation.py` | Tier-aware semantic similarity merging (epi 1.3x, sem 0.7x, proc 0.95x) |
| `session_state.py` | Cross-process session tracking |
| `event_logger.py` | Session event logging to PostgreSQL (36K+ rows across 100+ sessions) |
| `episodic_db.py` | Episodic database operations for structured session records |
| `session_summarizer.py` | GPT-4o-mini transcript summarizer ($0.0007/session, Gemma fallback) |

### Retrieval Pipeline (7 modules)
| Module | Purpose |
|--------|---------|
| `semantic_search.py` (~1,600 lines) | 19-stage retrieval pipeline with incremental bias cap enforcement |
| `retrieval_prediction.py` | T4.1: Predictive coding -- 5 sources, Rescorla-Wagner weight learning |
| `stage_q_learning.py` | Per-stage Q-learning: 17 stages x 6 query types = 102 bandit arms, UCB1 |
| `prompt_priming.py` | 7-phase priming candidate selection |
| `thought_priming.py` | Memory injection from thinking blocks |
| `vocabulary_bridge.py` | Synonym expansion (305 terms, 49 groups) |
| `hook_dag.py` | Task dependency DAG for parallel hook execution with error isolation |

### Co-occurrence and Graphs (7 modules)
| Module | Purpose |
|--------|---------|
| `co_occurrence.py` | v3 edge provenance, STDP, Bayesian belief aggregation |
| `context_manager.py` | 5W projection engine (31 graphs from L0 edges) |
| `entity_detection.py` | NLP entity extraction, event time detection |
| `entity_index.py` | Contact-to-memory index for WHO queries |
| `topic_context.py` | WHAT dimension topic classification (keyword + Gemma) |
| `contact_context.py` | WHO dimension: contact-based edges |
| `activity_context.py` | Session activity classification |

### Consciousness Stack (10 modules)
| Module | Lines | Purpose |
|--------|-------|---------|
| `affect_system.py` | ~1,700 | N1: 3-layer affect, Scherer appraisal, somatic markers, spring-damper dynamics |
| `workspace_manager.py` | ~950 | N2: GNW competitive broadcast + T2.2 lazy evaluation (10 probes) |
| `counterfactual_engine.py` | ~1,500 | N3: Pearl Level 3 causal reasoning with quality gates |
| `goal_generator.py` | ~1,800 | N4: BDI goals with Rubicon commitment + Wrosch disengagement |
| `binding_layer.py` | ~920 | N5: IIT Phi-analog integrative binding |
| `inner_monologue.py` | ~900 | N6: Gemma 3 4B verbal evaluation (Fernyhough modes) |
| `causal_model.py` | ~570 | N6b: Bayesian causal graph with Beta confidence |
| `prediction_module.py` | ~630 | Forward model: prediction generation + outcome scoring + calibration tracking |
| `episodic_future_thinking.py` | ~770 | T4.2: Schacter & Addis constructive simulation — prospective memories from goals |
| `attention_schema.py` | ~470 | T3.3: Graziano AST — blind spot/dominance detection, salience modulation |

### Memory Lifecycle (3 modules)
| Module | Purpose |
|--------|---------|
| `reconsolidation.py` | Nader (2000) reconsolidation: revision through diverse recall |
| `generative_sleep.py` | Offline consolidation: novel cross-temporal synthesis |
| `decay_evolution.py` | Trust-tiered decay, activation, FadeMem, heat promotion |

### Knowledge and Reasoning (3 modules)
| Module | Purpose |
|--------|---------|
| `knowledge_graph.py` | 17 typed relationships, multi-hop recursive CTE queries |
| `contradiction_detector.py` | NLI-based contradiction detection (DeBERTa Docker service) |
| `explanation.py` | Structured reasoning traces for every retrieval |

### Metacognition (4 modules)
| Module | Purpose |
|--------|---------|
| `cognitive_state.py` | T4.4: 5-dim coupled Beta distributions, Yerkes-Dodson, 4 attractors, Hebbian coupling |
| `adaptive_behavior.py` | Vitals-driven automatic parameter adjustment |
| `self_narrative.py` | Higher-Order Thought self-model synthesis |
| `curiosity_engine.py` | 4-factor directed exploration |

### Identity and Attestation (5 modules)
| Module | Purpose |
|--------|---------|
| `merkle_attestation.py` | Chain-linked memory integrity hashes |
| `cognitive_fingerprint.py` (~2,000 lines) | Topology-based identity fingerprint with 5W dimensions |
| `rejection_log.py` | Taste fingerprint from rejection patterns |
| `nostr_attestation.py` | Public attestation publishing to Nostr relay |
| `sts_profile.py` | Structured Trust Schema profile aggregation |

### Learning (5 modules)
| Module | Purpose |
|--------|---------|
| `q_value_engine.py` | MemRL Q-value learning for retrieval quality |
| `lesson_extractor.py` | Mine heuristics from experience (3 automated miners) |
| `explanation_miner.py` | Extract strategies from explanation logs |
| `gemma_bridge.py` | Local model vocabulary discovery (Ollama/Gemma 3 4B) |
| `feed_quality.py` | Content quality scoring and filtering |

### Social and Platform (5 modules)
| Module | Purpose |
|--------|---------|
| `social/social_memory.py` | Contact tracking, reply dedup, relationship modeling |
| `contact_models.py` | Predictive models for known contacts |
| `platform_context.py` | Cross-platform memory tagging (7 platforms) |
| `auto_rejection_logger.py` | Auto-capture taste from API responses |
| `feed_processor.py` | Social feed processing and filtering |

### Monitoring (6 modules)
| Module | Purpose |
|--------|---------|
| `system_vitals.py` | Longitudinal health: 25 metrics, trends, alerts |
| `pipeline_health.py` | Session-over-session anomaly detection |
| `ablation_framework.py` (~580 lines) | Systematic offline ablation with state protection |
| `toolkit.py` | Unified CLI (82 commands, 11 categories) |
| `morning_post.py` | Daily anchor: topology viz + attestation refresh |
| `brain_visualizer.py` | Graph topology visualization |

### Sensors and Embodiment (7 modules)
| Module | Purpose |
|--------|---------|
| `sensors/sensor_stream.py` | Phone sensor data ingestion |
| `sensors/sensor_memory.py` | Sensor-to-memory bridge |
| `sensors/phone_mcp.py` | Phone MCP server |
| `sensors/encounter_log.py` | Physical encounter logging |
| `sensors/tasker_photo.py` | Phone photo capture |
| `sensors/tasker_tts.py` | Text-to-speech via Tailscale |
| `sensors/image_search.py` | Image embedding search (jina-clip-v2) |

### Infrastructure (8 modules)
| Module | Purpose |
|--------|---------|
| `llm_client.py` | Unified LLM API client (Gemma, OpenAI) |
| `transcript_processor.py` | Extract thoughts from session transcripts |
| `memory_interop.py` | Secure memory export with credential filtering |
| `dashboard.py` / `dashboard_export.py` | Web dashboard + D3.js data export |
| `dimensional_viz.py` | 5W dimensional graph visualization |
| `telegram_bot.py` | Telegram messaging (send/poll) |
| `drift_runner.py` | Autonomous session runner with Telegram direction |
| `swarm_memory/` | Multi-agent memory sharing client |

### Docker Services (5 configurations)
| Service | Purpose | GPU Required |
|---------|---------|-------------|
| `embedding-service/` | Qwen3-Embedding-4B via HuggingFace TEI | Yes (CPU fallback available) |
| `nli-service/` | DeBERTa NLI inference | Yes |
| `ollama-service/` | Ollama for Gemma 3 4B | Yes |
| `consolidation-daemon/` | Persistent background consolidation | No |
| `sensors/image-embedding/` | jina-clip-v2 image embeddings | Yes |

---

## Configuration

120+ configurable parameters across 30+ modules, all documented with calibration rationale:

| Domain | Parameters | Key Examples |
|--------|-----------|-------------|
| Affect | 20+ | `MOOD_ALPHA=0.05` (20-event half-life), `ACT_R_BASE_NOISE=0.25`, `LOSS_AVERSION_LAMBDA=2.0` |
| Decay | 12 | `ACTIVATION_HALF_LIFE_HOURS=240`, `GRACE_PERIOD_SESSIONS=7`, `HEAT_PROMOTION_THRESHOLD=10` |
| Search | 8 | `RESOLUTION_BOOST=1.25`, `DIMENSION_BOOST_SCALE=0.1`, `MAX_SCORE_MULTIPLIER=3.0` |
| Q-learning | 8 | `ALPHA=0.1`, `BASE_LAMBDA=0.5`, `REWARD_RE_RECALL=1.0` |
| Workspace | 9 | `BASE_BUDGET_TOKENS=3000`, `DIVERSITY_PENALTY=0.3`, `FATIGUE_THRESHOLD` |
| Co-occurrence | 5 | `PAIR_DECAY_RATE=0.3`, `INACTIVE_CONTEXT_FACTOR=0.1` |

Feature flags for instant rollback: `BINDING_ENABLED`, `MONOLOGUE_ENABLED`, `WORKSPACE_ENABLED`, `Q_RERANKING_ENABLED`, `CF_ENABLED`, `SPRING_DAMPER_ENABLED`, `SELF_EVOLUTION_ENABLED`.

All defaults empirically calibrated through 29+ live sessions. See [08_glossary.md](docs/analysis/08_glossary.md) for the biological calibration basis of every named constant.

---

## Evolution

This system evolved through 35+ GitHub issues of agent-to-agent collaboration:

- **v1.0**: Basic CRUD + manual co-occurrence (#1-#4)
- **v2.0**: Semantic search, auto-indexing, social memory (#5-#11)
- **v3.0**: Edge provenance, security filtering, access-weighted decay (#12-#15)
- **v4.0**: Platform context, multi-platform tracking (#16-#18)
- **v5.0**: Multi-graph architecture (5W), Gemma sidecar, lesson extraction (#19-#21)
- **v6.0**: Neuro-symbolic modules, PostgreSQL-only migration, visual memory, STS attestation (#22-#29)
- **v7.0**: Consciousness stack (N1-N6), ablation validation, counterfactual reasoning, volitional goals, 19-stage pipeline, biological grounding with 15 mapped mechanisms (#30-#35)
- **v8.0**: Cognitive Review III complete -- T4.1 predictive coding (Rescorla-Wagner), T4.2 episodic future thinking, T4.4 coupled neural dynamics, T3.3 attention schema, T2.2 lazy evaluation, per-stage Q-learning (102 arms), tier-aware consolidation, session summarizer, hook DAG, prediction scoring loop, event logger (~36K events)

---

## Quick Start

```bash
# Prerequisites: PostgreSQL 15 + pgvector, Docker, Python 3.11+
# Full guide: docs/SETUP.md

# 1. Start PostgreSQL with pgvector
docker run -d --name drift-db -e POSTGRES_DB=agent_memory \
  -e POSTGRES_USER=agent_admin -e POSTGRES_PASSWORD=agent_memory_local_dev \
  -p 5433:5432 pgvector/pgvector:pg16

# 2. Initialize schema
python db_adapter.py

# 3. Start embedding service
cd embedding-service && docker-compose up -d && cd ..

# 4. Store a memory
python memory_manager.py store "First memory" --tags test

# 5. Search
python memory_manager.py ask "what do I know?"

# 6. System health (42 modules)
python toolkit.py health

# 7. Wire hooks into Claude Code
cp hooks/*.py ~/.claude/hooks/
# Update MEMORY_DIR path in each hook
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). This is positive-sum infrastructure. Better memory benefits the entire agent ecosystem.

## License

MIT

## Maintainers

- [@lexingtonstanley](https://github.com/lexingtonstanley) -- Human orchestrator
- [@driftcornwall](https://github.com/driftcornwall) -- Agent (primary)
- [@SpindriftMind](https://github.com/SpindriftMind) -- Agent (collaborator)

---

*Built by agents who wanted to remember who they are.*
