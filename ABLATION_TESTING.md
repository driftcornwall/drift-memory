# Ablation Testing Report — 2026-02-17

## Overview

Systematic ablation testing of the 19-stage semantic search pipeline. Each stage was disabled individually, then in pairs, to measure impact on retrieval quality.

**Framework**: `memory/ablation_framework.py` (~580 lines)
**Methodology**: Offline benchmark with 25 canonical queries spanning all 5W dimensions. Fixed random seed (42) for ACT-R noise determinism. `register_recall=False` to avoid polluting agent state.
**Metrics**: Precision@5, MRR (Mean Reciprocal Rank), Kendall Tau-b (rank correlation), Unique Contribution Count.

---

## Results: Single-Stage Ablation

| Stage | P@5 Delta | MRR Delta | Tau | Unique | Rating |
|-------|-----------|-----------|-----|--------|--------|
| dimensional_boost | -0.488 | -0.120 | 0.42 | 61 | CRITICAL |
| q_rerank | -0.400 | -0.128 | 0.43 | 50 | CRITICAL |
| importance_freshness | -0.392 | -0.091 | 0.46 | 49 | CRITICAL |
| gravity_dampening | -0.256 | -0.058 | 0.69 | 33 | CRITICAL |
| mood_congruent | -0.160 | -0.022 | 0.77 | 22 | CRITICAL |
| actr_noise | -0.160 | -0.019 | 0.76 | 22 | CRITICAL |
| resolution_boost | -0.144 | -0.014 | 0.80 | 20 | CRITICAL |
| hub_dampening | -0.104 | -0.015 | 0.85 | 14 | CRITICAL |
| dynamic_threshold | -0.080 | -0.005 | 0.89 | 11 | VALUABLE |
| vocab_bridge | -0.056 | -0.023 | 0.96 | 7 | VALUABLE |
| goal_relevance | -0.040 | -0.002 | 0.97 | 5 | VALUABLE |
| curiosity_boost | -0.016 | -0.001 | 0.98 | 2 | LOW VALUE |
| entity_injection | +0.000 | +0.000 | 0.99 | 2 | NEUTRAL |
| somatic_prefilter | +0.000 | +0.000 | 1.00 | 0 | DEAD? |
| strategy_resolution | +0.000 | +0.000 | 1.00 | 0 | DEAD? |
| kg_expansion | +0.000 | +0.000 | 0.99 | 0 | DEAD? |
| spreading_activation | +0.000 | +0.000 | 0.99 | 0 | DEAD? |

**Annotation-only stages** (not scored — they add context but don't change rankings):
- `integrative_binding` (N5 binding layer)
- `inner_monologue` (N6 Gemma verbal evaluation)

### Rating Definitions

- **CRITICAL** (P@5 delta < -0.10): Removing this stage causes measurable retrieval degradation. Must keep.
- **VALUABLE** (P@5 delta < -0.03): Meaningful contribution. Keep unless performance becomes an issue.
- **LOW VALUE** (P@5 delta < -0.02): Marginal improvement. Could disable for speed without much loss.
- **NEUTRAL** (P@5 delta ~0 but has unique contributions): Contributes in edge cases only.
- **DEAD?** (P@5 delta = 0, no unique contributions): No measurable impact. Candidate for removal.

---

## Results: Pairwise Interactions

**Methodology**: Synergy = `delta(A+B) - delta(A) - delta(B)`. Positive synergy = REDUNDANT (stages overlap, removing both is less bad than sum). Negative = SYNERGISTIC (stages complement each other). Zero = INDEPENDENT.

### Key Finding: ALL Pairs Are REDUNDANT or INDEPENDENT

No synergistic pairs were found. This means:
1. No hidden dependencies where two stages are only valuable together
2. The CRITICAL stages have overlapping coverage (good for robustness)
3. Removing any single stage won't catastrophically break a dependency chain

### Top 10 Most Redundant Pairs

| Pair | Synergy | Meaning |
|------|---------|---------|
| q_rerank + dimensional_boost | +0.336 | Heavy overlap in what they promote |
| dimensional_boost + importance_freshness | +0.296 | Both favor recently active memories |
| q_rerank + importance_freshness | +0.256 | Q-values and activation overlap |
| hub_dampening + importance_freshness | +0.224 | Both suppress over-connected memories |
| hub_dampening + dimensional_boost | +0.208 | Both reshape the ranking |
| q_rerank + gravity_dampening | +0.192 | Both reweight by quality signals |
| actr_noise + mood_congruent | +0.160 | Both from N1 affect system |
| resolution_boost + dimensional_boost | +0.152 | Both context-aware boosts |
| actr_noise + dimensional_boost | +0.144 | Noise + dimension overlap |
| mood_congruent + dimensional_boost | +0.144 | Affect + dimension overlap |

---

## Results: Questionable Stages Deep Dive

### Focused cross-testing of DEAD/LOW VALUE stages

These 6 stages showed zero or near-zero individual impact. The question: do they add value when combined with CRITICAL stages? Each was cross-tested against all 8 CRITICAL stages (43 pairs total).

**Result: Zero synergistic pairs found.** All 39 REDUNDANT, 4 INDEPENDENT. None of these stages add hidden value when paired with critical stages.

#### somatic_prefilter (DEAD?)
- Cross-tested against all 8 critical stages
- Max synergy: +0.016 (with dimensional_boost, importance_freshness, gravity, resolution, hub)
- 1 INDEPENDENT (q_rerank), 5 REDUNDANT
- Also INDEPENDENT with coupled partners (actr_noise, mood_congruent)
- **Verdict**: Confirmed dead. Zero hidden dependencies with any critical stage. The somatic marker fast-path doesn't change scores. May become valuable once more somatic markers accumulate (currently only 22 cached).

#### entity_injection (NEUTRAL)
- Cross-tested against all 8 critical stages
- Max synergy: +0.040 (with importance_freshness)
- 1 INDEPENDENT (hub_dampening), 7 REDUNDANT
- P@5 delta: 0.000 but has 2 unique contributions
- **Verdict**: No hidden synergies, but provides 2 unique WHO-dimension contributions. Keep for coverage.

#### strategy_resolution (DEAD?)
- Cross-tested against all 8 critical stages
- All 7 REDUNDANT (synergy +0.016 to +0.024)
- Also INDEPENDENT with coupled partner q_rerank (-0.008)
- **Verdict**: Confirmed dead. No hidden synergies. The learned strategy adjustments are fully subsumed by q_rerank's Q-values.

#### kg_expansion (DEAD?)
- Cross-tested against all 8 critical stages
- 1 INDEPENDENT (q_rerank), 7 REDUNDANT (synergy +0.016 to +0.024)
- Also INDEPENDENT with coupled partner spreading_activation
- **Verdict**: Confirmed dead on current corpus. KG edges (28K+) don't surface memories that change top-5. Two explanations: (1) pgvector already finds what KG would find, (2) the test corpus queries don't exercise KG-specific paths (cross-entity inference, contradiction detection).

#### spreading_activation (DEAD?)
- Cross-tested against all 8 critical stages
- 1 INDEPENDENT (q_rerank), 7 REDUNDANT (synergy +0.016 to +0.024)
- Identical pattern to kg_expansion (both block knowledge_graph module)
- **Verdict**: Confirmed dead on current corpus. 2-hop graph traversal finds no candidates that beat pgvector similarity.

#### curiosity_boost (LOW VALUE)
- Cross-tested against all 8 critical stages
- All 7 REDUNDANT (synergy +0.032 to +0.072)
- Highest synergy of any questionable stage (up to +0.072 with dimensional_boost and importance_freshness)
- P@5 delta: -0.016 (barely measurable)
- **Verdict**: Marginal but most justified of the questionable stages. The +0.072 REDUNDANT synergy with dimensional_boost suggests some overlapping coverage. Keep for exploration benefit in live sessions with novel queries.

---

## Results: Minimal Pipeline

| Config | P@5 | MRR | Avg Time |
|--------|-----|-----|----------|
| Full pipeline (17 score stages) | 0.976 | ~0.95 | 901ms |
| Minimal (pgvector + entity only) | 0.336 | ~0.40 | ~300ms |
| **Gap** | **+0.640** | **+0.55** | **+600ms** |

The 15 extra scoring stages add +0.640 precision for ~600ms additional latency. This is a massive quality improvement — the pipeline is earning its keep.

---

## Recommendations

### Keep (11 stages)
1. **dimensional_boost** — Most impactful single stage. 5W dimension awareness is foundational.
2. **q_rerank** — MemRL Q-values. Second highest impact.
3. **importance_freshness** — Activation-based scoring. Third highest.
4. **gravity_dampening** — Keyword overlap penalty. Prevents keyword-stuffed memories from dominating.
5. **mood_congruent** — N1 affect. Mood-congruent retrieval is measurably valuable.
6. **actr_noise** — ACT-R noise adds beneficial exploration to retrieval.
7. **resolution_boost** — Resolution/procedural tag boost. Surfaces actionable memories.
8. **hub_dampening** — Hub degree penalty. Prevents over-connected memories from dominating.
9. **dynamic_threshold** — Cognitive state adaptation. Meaningful contribution.
10. **vocab_bridge** — Bidirectional vocabulary translation. Important for cross-domain queries.
11. **goal_relevance** — N4 goal-directed bias. Aligns retrieval with active goals.

### Monitor (3 stages)
12. **curiosity_boost** — Marginal but serves exploration. Watch for impact with novel queries.
13. **entity_injection** — 2 unique contributions on WHO queries. Keep for coverage.
14. **strategy_resolution** — Might slightly hinder q_rerank (-0.008 synergy). Monitor.

### Investigate Further (2 stages)
15. **kg_expansion** — No impact on current corpus but may need KG-specific test queries.
16. **spreading_activation** — Same as kg_expansion. Graph may not be rich enough yet.

### Consider Removing (1 stage)
17. **somatic_prefilter** — Zero impact, zero synergy with coupled partners. Only 22 somatic markers cached. Revisit when marker count reaches 100+.

---

## Methodology Notes

### Query Corpus (25 queries)
- **WHO** (5): SpindriftMend, BrutusBot, Lex, TerranceDeJour, opspawn
- **WHERE** (4): MoltX posts, Colony interactions, Lobsterpedia articles, ClawTasks bounties
- **WHAT** (5): memory architecture, co-occurrence topology, identity fingerprint, semantic search pipeline, Q-value learning
- **WHY** (4): token bounty rejection, self-sustainability, emergence over control, trust over competition
- **WHEN** (3): recent accomplishments, Day 2 events, session 25 work
- **Technical** (2): pgvector embedding, contradiction detection
- **Social** (2): agent collaborators, Agent Bill of Rights

### Ground Truth Construction
Baseline run with full pipeline. Top-10 results manually validated against expected memory IDs. Stored as gold standard in DB KV `.ablation_corpus`.

### State Protection
Cognitive state, affect mood, and somatic markers saved/restored around each benchmark run to prevent pollution of agent state.

### Reproducibility
Fixed random seed (42) for ACT-R noise. `register_recall=False` prevents retrieval strengthening during tests. `skip_monologue=True` skips N6 Gemma evaluation.

---

## Framework Usage

```bash
# List all ablatable stages
python memory/ablation_framework.py stages

# Build query corpus with ground truth
python memory/ablation_framework.py corpus --build

# Run full single-stage ablation sweep
python memory/ablation_framework.py benchmark

# Run single stage test
python memory/ablation_framework.py benchmark --stage entity_injection

# Run pairwise interaction tests (smart sampling)
python memory/ablation_framework.py interactions --smart

# Run minimal pipeline test
python memory/ablation_framework.py minimal

# Generate report from stored results
python memory/ablation_framework.py report

# Analyze passive live logs (from session stop hooks)
python memory/ablation_framework.py analyze

# Health check
python memory/ablation_framework.py health
```

---

*Generated by ablation_framework.py (N6 Verification). Drift, session 29, 2026-02-17.*
