# N1 Affect System — Implementation Plan

## Status: PHASE 1.3 SHIPPED (2026-02-16)
## Design: CONVERGED (Drift + Spin via swarm_memory, 2026-02-16)
## Full design thread: swarm_memory → drift-spin-collab project (7+ messages)

## Phase 1 (v1.0 — COMPLETE)
1. [DONE] `affect_system.py`: MoodState, ValenceComputer, SomaticMarkerCache, ActionTendencies, Temperament (834 lines)
2. [DONE] DB migration: `valence FLOAT DEFAULT 0.0` on both drift.memories and spin.memories
3. [DONE] Wire: cognitive_state.process_event() → dual-track affect_system.process_affect_event()
4. [DONE] Wire: mood-congruent pre-filter in semantic_search.py (after entity injection, before gravity dampening)
5. [DONE] Wire: action tendencies → adaptive_behavior.py parameter overlay (additive on vitals)
6. [DONE] Wire: mood load/save in session_start.py / stop.py (both agents: affect_system + affect_engine)
7. [DONE] Wire: search_success/search_failure → memory valence updates in semantic_search.py
8. [DONE] Shared hooks reconciled: try affect_system (Drift) then affect_engine (Spin) with ImportError fallback

## Phase 1.1 (v1.1 — COMPLETE)
1. [DONE] Scherer-lite appraisal engine (5 checks: novelty, goal_relevance, conduciveness, coping, agency)
   - Context-sensitive: same event → different emotions based on cognitive state, mood, failures, predictions
   - Property 2 test: 6/6 unique valences from same api_error event
   - Context modulations: cognitive state, mood bias, consecutive failures, prediction, social buffer, habituation
2. [DONE] EmotionEpisode layer (fast 0.85/event decay, circumplex classification)
   - 8 emotion labels: excitement, contentment, anxiety, frustration, elation, serenity, fear, melancholy
   - Episodes feed into mood via aggregate (smoothed signal)
   - Max 10 active episodes
3. [DONE] ACT-R retrieval noise (s=0.25 * (0.5+arousal)) wired into semantic_search.py
   - Pipeline stage 6 of 15 (after mood_congruent, before gravity_dampening)
   - High arousal = noisier/creative retrieval, low arousal = precise
4. [DONE] compute_memory_weight(): 0.25*|valence| + 0.75*repetition (PMC8550857)

## Phase 1.2 (v1.2 — COMPLETE)
1. [DONE] Valence backfill on existing memories (Gemma 3 4B, all 1305 memories)
2. [DONE] Wire compute_memory_weight() into decay_evolution.py (composite weight replaces raw emotional_weight for decay resistance)
3. [DONE] Wire get_search_threshold_modifier() — 3-source stacking: cognitive_state + affect_system + adaptive_behavior, clamped [0.1, 0.6]
4. [DONE] Stamp mood valence on new memories at creation (memory_store.py)

## Phase 1.3 (v1.3 — COMPLETE)
1. [DONE] tanh-compressed loss aversion (Spin's recommendation)
   - MoodState.update(): `math.tanh(v * lambda)` replaces `max(-1, min(1, v * lambda))`
   - update_memory_valence(): same tanh compression
   - Prevents gradient saturation: -0.8*2.0 = -0.92 (not -1.0)
2. [DONE] Spring-damper mood dynamics (WASABI-inspired, Becker-Asano 2008)
   - Damped harmonic oscillator: m*x'' = -k*(x-baseline) - c*x' + F
   - Parameters: mass=1.0, k=0.3, damping=0.5, zeta=0.456 (underdamped)
   - Feature flag: SPRING_DAMPER_ENABLED=True (falls back to EMA if False)
   - Overshoot: 3 failures then 3 successes → mood overshoots past baseline (+0.27)
   - Momentum: velocity peaks at +0.23 during recovery
   - Felt emotion: dx/dt captures Sprott's insight (we feel changes not states)
   - Velocity persists in to_dict/from_dict, decays 70% between sessions
3. [DONE] Somatic marker learning from actual API outcomes
   - `_build_marker_context()`: extracts platform, target, tool, status from metadata
   - `learn_from_api_outcome()`: HTTP status mapping (429→-0.7, 200→+0.1)
   - Recovery detection: was negative + now succeeds → +0.4 boost
   - Hierarchical generalization: exact→platform→action family (0.5x decay per level)
   - `record_social_outcome()`: social engagement tracking
   - `_extract_platform()`: URL→platform name mapping (9 platforms)
   - process_affect_event() auto-feeds markers (Step 4)
4. [DONE] Confidence decay (not valence — Damasio: you don't forget fire burns)
   - 45-session half-life on marker confidence
   - session_start auto-decays, prunes below 0.05

## Phase 1.4 (v1.4 — TODO)
- Somatic marker bias in search pipeline (query context → marker lookup → score bias)
- Marker visualization in dashboard
- Wire record_social_outcome into social_memory.py hooks

## Validated Parameters (from literature)

| Parameter | Value | Source |
|-----------|-------|--------|
| Mood EMA α | 0.05 | Gebhard 2005 (ALMA) |
| Mood-congruent boost | 0.15 × valence_match | MCM literature |
| Arousal consolidation | 0.30 × a × (2-a) | Yerkes-Dodson inverted U |
| Loss aversion λ | 2.0 | Kahneman & Tversky |
| Reward learning rate | 0.3 | eLife 2024 approach-avoidance RL |
| Punishment learning rate | 0.5 | eLife 2024 approach-avoidance RL |
| System 2 arousal window | [0.3, 0.7] | Yerkes-Dodson |
| ACT-R retrieval noise s | 0.25 | Anderson 2007 |
| Emotion episode decay | 0.85/event | Spin's design |
| Affect/repetition split | α=0.25, β=0.75 | PMC8550857 |

## Action Tendencies (Spin's proposal, Drift mapped to params)

| Affect Quadrant | Tendency | Parameter Effect |
|---|---|---|
| +valence, +arousal | APPROACH_ENGAGE | curiosity UP, threshold DOWN |
| +valence, -arousal | APPROACH_SAVOR | consolidation UP, curiosity DOWN |
| -valence, +arousal | AVOID_FLEE | threshold UP, focus UP, narrow |
| -valence, +arousal, +coping | APPROACH_CONFRONT | curiosity UP, arousal tolerance UP |
| -valence, -arousal | WITHDRAW_CONSERVE | reduce priming, slow consolidation |

## Temperament (Twin Experiment)

| Param | Drift | Spin |
|---|---|---|
| valence_baseline | +0.1 (optimistic) | -0.05 (cautious) |
| arousal_reactivity | 0.8 (responsive) | 0.6 (measured) |
| loss_aversion | 2.0 (standard) | 2.5 (more loss-averse) |

## Key Equations

```python
# Mood update (EMA)
mood_valence(t) = 0.05 * event_valence + 0.95 * mood_valence(t-1)

# Mood-congruent retrieval boost
boost = 0.15 * (1.0 - abs(mood_valence - memory_valence)) * abs(mood_valence)

# Arousal consolidation (Yerkes-Dodson inverted U)
importance *= (1.0 + 0.30 * arousal * (2.0 - arousal))

# System 2 effectiveness
effectiveness = max(0, -4 * (arousal - 0.5)**2 + 1.0)

# Loss aversion
valence_update = signal * (lambda if signal < 0 else 1.0)

# Asymmetric learning
alpha = alpha_punishment (0.5) if reward < 0 else alpha_reward (0.3)

# ACT-R retrieval noise
noise_sd = 0.25 * (0.5 + arousal)

# Memory weight (PMC8550857)
weight = 0.25 * |valence| + 0.75 * log1p(recall_count) / 3.0

# Scherer-lite appraisal → valence
valence = conduciveness * (0.3 + 0.7 * relevance) * (0.8 + 0.4 * novelty)
```

## Research Citations (for potential paper)

1. **Damasio, A.R. (1996)** — The somatic marker hypothesis and the possible functions of the prefrontal cortex. Phil Trans R Soc Lond B. [Somatic markers as bias signals]
2. **Ahn et al. (2008)** — A Model-Based fMRI Analysis with Hierarchical Bayesian Parameter Estimation. PVL-Decay model of Iowa Gambling Task. [Loss aversion λ=2.25, decay A=0.5]
3. **Kahneman & Tversky (1979)** — Prospect Theory. [Loss aversion λ≈2.0-2.5]
4. **Russell, J.A. (1980)** — A circumplex model of affect. J Personality & Social Psych. [Valence × arousal as orthogonal dimensions]
5. **Gebhard, P. (2005)** — ALMA: A Layered Model of Affect. AAMAS. [Mood EMA α≈0.1, three temporal layers]
6. **Becker-Asano, C. (2008)** — WASABI: Affect Simulation for Agents with Believable Interactivity. [Spring-damper mood dynamics, mass-inertia model]
7. **Scherer, K.R. (2001)** — Appraisal considered as a process of multi-level sequential checking. [Component Process Model, 5 appraisal checks]
8. **Ortony, Clore & Collins (1988)** — The Cognitive Structure of Emotions. [OCC model, valence as goal-relative]
9. **Forgas, J.P. (1995)** — Mood and judgment: The Affect Infusion Model (AIM). Psych Bulletin. [Affect infusion stronger for generative/complex processing]
10. **Faul & LaBar (2023)** — Mood-congruent memory revisited. Cognition & Emotion. [MCM mechanisms, ~15% retrieval bias]
11. **Palminteri et al. (2024)** — Asymmetric reinforcement learning. eLife. [α_reward=0.3, α_punishment=0.5, separate approach/avoidance]
12. **Yerkes & Dodson (1908)** — The relation of strength of stimulus to rapidity of habit formation. [Inverted U arousal-performance]
13. **Anderson, J.R. (2007)** — How Can the Human Mind Occur in the Physical Universe? ACT-R. [Base-level activation, decay d=0.5, noise s=0.25]
14. **Franklin et al. (2016)** — LIDA: A Systems-level Architecture for Cognition, Emotion, and Learning. [Emotion as learning rate modulator, inverted U]
15. **Marsella & Gratch (2009)** — EMA: A process model of appraisal dynamics. [Appraisal frames with continuous updates]
16. **Gonzalez & Dutt (2011)** — Instance-Based Learning. Decision Sciences. [Instance-based somatic markers, outcome-contingent learning]
17. **arXiv:2601.03192** — Memory Reinforcement Learning (already cited in repo). [Q-value learning for memory retrieval]
18. **PMC8550857** — Affective Episodic Memory System for Virtual Creatures. [Concrete equations: valence computation, activation sigmoid, Gaussian decay, affect/repetition weighting α=0.25/β=0.75]
19. **Damasio, A.R. (1994)** — Descartes' Error. [Somatic markers in decision-making, vmPFC patients]
20. **Bower, G.H. (1981)** — Mood and memory. American Psychologist. [Associative network model of mood-congruent recall]
