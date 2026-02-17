# N3 Counterfactual Reasoning Engine — Converged Implementation Plan

## Status: PHASE 3 SHIPPED (2026-02-16) — N3 COMPLETE
## Theory: Pearl SCM Level 3 + Byrne Mutability + Epstude & Roese Functional Theory

## The Problem

The system has **zero counterfactual reasoning capability**. It can predict forward
(prediction_module.py), detect when predictions fail, revise memories (reconsolidation.py),
and adapt parameters (adaptive_behavior.py). But it cannot ask:

- "What would have happened if I had done X instead of Y?"
- "Why did this surprise me — which assumption was wrong?"
- "If I hadn't adapted parameter Z, what would my metrics look like?"
- "This memory changed — what decisions would I make differently now?"

The review team scored N3 at **0/10 current maturity** — the lowest of all 5 recommendations.
Their verdict: "This creates imaginative cognition — the ability to simulate alternative
realities. No existing module does anything remotely like this."

This is Pearl's Level 3 — the highest rung of causal reasoning. Level 1 (association) is our
co-occurrence graph. Level 2 (intervention) is our prediction module. Level 3 (counterfactual)
requires reasoning about *specific past situations* and imagining alternative histories.

## Theory Foundation

| Concept | Source | Design Principle |
|---------|--------|-----------------|
| Structural Causal Models | Pearl 2000, 2009 | Counterfactuals require a model, not just data |
| Three-Step Algorithm | Pearl 2000 | Abduction → Action → Prediction |
| Closest Possible World | Lewis 1973 | Change ONE thing, trace consequences faithfully |
| Mutability Hierarchy | Byrne 2005 | Actions > inactions, controllable > uncontrollable |
| Functional Theory | Epstude & Roese 2008 | Content-specific + content-neutral pathways |
| Simulation Heuristic | Kahneman & Tversky 1982 | Near-misses generate strongest counterfactuals |
| Upward/Downward | Roese 1997 | Upward = learning, downward = calibration |
| Spontaneous/Deliberate | Byrne 2005 | Error-triggered vs intentional generation |
| Opportunity Principle | Epstude & Roese 2008 | Tunable params → upward; irreversible → downward |
| Anti-Rumination | Nolen-Hoeksema 1991 | Cap counterfactual cycles to prevent fixation |

## Design Convergence (5 Resolved Divergences)

### D1: Decision Trace Gap — DRIFT WINS (new insight)

**Drift audit identified:** The system doesn't log which memories were recalled before a
decision, so counterfactual analysis of "what if I had recalled differently?" is impossible.
This is a critical prerequisite for meaningful retrospective counterfactuals.

**Spin:** Not mentioned in theory doc.

**Resolution:** Build lightweight decision trace INTO the engine. Extend `session_state.py`'s
existing `session_recalls` to also track recall→outcome associations. When a memory is recalled
and subsequently referenced in a tool use or response, link them. This enables: "I recalled
memory X before deciding Y — what if I had recalled Z instead?"

**Implementation:** Add `log_decision_context(recall_ids, action_description, outcome)` to
session_state.py. Called from post_tool_use hook when tool results reference recalled memories.
Lightweight — just extends the existing recall tracking structure.

### D2: Heuristic vs LLM Split — HYBRID

**Drift theory agent:** 6 heuristic types (template-based, no LLM), 4 LLM types. Cap at 3 calls.
**Spin:** "Heuristic first, LLM optional." 2-3 counterfactuals per session.

**Resolution:** Max 3 counterfactuals per session. Heuristic templates handle the common cases
(simple retrospective, basic prospective). LLM handles the genuinely interesting cases where
causal reasoning matters (self-directed, reconsolidation). Budget: max 2 LLM calls per session
for counterfactual generation.

**Why 2 not 3:** Other modules already use LLM calls (consolidation, self-narrative). Total
session LLM budget should stay reasonable.

### D3: Affect Routing — DRIFT WINS (more specific)

**Drift theory agent:** Map upward counterfactuals → adaptive_behavior.py (behavioral change
intentions). Map downward counterfactuals → affect_system.py (calibration/comfort).
This follows Epstude & Roese's opportunity principle: when parameters are tunable, generate
upward counterfactuals (learning); when the past is irreversible, generate downward (acceptance).

**Spin:** Mentions upward/downward distinction but doesn't specify which modules consume them.

**Resolution:** Drift's routing.
- **Upward** ("could have been better"): Feed lesson to adaptive_behavior.py as behavioral
  intention. Also store as lesson in lesson_extractor.py. Fires `counterfactual_upward` cognitive
  event → slight negative affect (productive regret).
- **Downward** ("could have been worse"): Feed to affect_system.py as positive valence update
  (relief/calibration). Fires `counterfactual_downward` → slight positive affect.

### D4: NLI Validation — DRIFT WINS (extends Spin's 6 mitigations)

**Drift audit:** Use existing NLI service (port 8082) for cross-checking counterfactual
consequents against known facts. If counterfactual claims "X would have happened" and
existing memories entail "X cannot happen," flag as confabulation.

**Spin:** 6 confabulation strategies but doesn't mention NLI.

**Resolution:** Add NLI as 7th mitigation strategy. Only for counterfactuals with
confidence > 0.6 and plausibility score > 0.5 (avoid wasting NLI calls on speculative ones).
NLI check: compare counterfactual consequent against 3 most relevant existing memories.
If entailment < 0.3 AND contradiction > 0.5, downgrade confidence by 0.3.

### D5: Workspace Integration — DRIFT WINS (N2 didn't exist when Spin wrote N3)

**Drift audit:** Counterfactual insights should compete for workspace access in session_start
as a new candidate type.

**Spin:** Not mentioned (N2 hadn't been designed yet).

**Resolution:** Counterfactual summaries from previous sessions compete in workspace as
category "imagination" (new 7th category). Salience scorer:
- Base: 0.2 (low — counterfactuals are supplementary context)
- +0.3 if counterfactual is about a topic in current session's social context
- +0.2 if counterfactual was validated by subsequent experience
- +0.1 if generated within last 3 sessions (recency)

This means counterfactual insights only enter consciousness when they're relevant. Most
sessions, they'll be suppressed by higher-salience modules. Perfect.

## Four Types of Counterfactuals

### Type 1: Retrospective — "What if the surprising thing hadn't happened?"

**Trigger:** Prediction error at session end (spontaneous, Byrne).
**Input:** Highest-surprise event from prediction_module.py scoring (confidence 0.3-0.7 = near-miss zone, per Kahneman & Tversky simulation heuristic).
**Process:**
1. Identify the surprising outcome and the prediction it violated
2. Apply mutability filter: Was it controllable? Was it an action (not inaction)?
3. Generate counterfactual: "If {surprising_thing} hadn't occurred, {expected_outcome} would have happened because {causal_reasoning}"
4. Extract lesson: What assumption was wrong? What should I watch for next time?

**Generation:** Heuristic template for simple cases. LLM call if causal chain has >2 steps.

**Example:** "My prediction that @opspawn would reply within 24h was wrong (they haven't replied in 48h). If they had replied, I would have started the Hedera integration. The assumption that was wrong: opspawn's engagement cadence is slower than my contact model predicted. Lesson: update contact model half-life for opspawn."

### Type 2: Prospective — "If this prediction is wrong, then what?"

**Trigger:** Prediction generation at session start (deliberate, Byrne).
**Input:** Each prediction from prediction_module.py generate().
**Process:**
1. For each prediction, generate one alternative: "If this is wrong, the most likely alternative is..."
2. Identify what evidence would confirm the alternative
3. Store as shadow prediction

**Generation:** Always heuristic (template-based). No LLM needed.

**Template:** "If {prediction} is wrong, the most likely alternative is {alternative} because {reasoning}. Evidence that would confirm: {evidence}."

**Example:** "Prediction: Will recall 5-10 memories this session. If wrong, most likely alternative is <5 recalls (quiet session, no external triggers). Evidence: no new platform notifications at session start."

### Type 3: Self-Directed — "What if I hadn't adapted parameter X?"

**Trigger:** Adaptive parameter change in adaptive_behavior.py (spontaneous).
**Input:** Parameter changes from evaluation cycle + historical metrics.
**Process:**
1. Identify the parameter that was changed and its pre-change value
2. Simulate: What would the metrics look like with the old value?
3. Compare: Was the adaptation beneficial?
4. Generate counterfactual about the adaptation decision itself

**Generation:** LLM call (first priority). Requires reasoning about interaction effects
between parameters, which templates can't capture.

**Example:** "If I hadn't increased curiosity_target_count from 2 to 3, my recall diversity
would have been lower (fewer dead memories resurfaced). But my attention budget would have
been tighter. The adaptation was net positive — 2 of the 3 curiosity targets were subsequently
recalled by the LLM, suggesting they were genuinely useful."

### Type 4: Reconsolidation — "Memory changed — what would I have done differently?"

**Trigger:** Memory revision in reconsolidation.py (spontaneous).
**Input:** Pre-revision and post-revision memory content + recent decisions that referenced it.
**Process:**
1. Find the decision trace: When was this memory last recalled? What happened after?
2. Generate counterfactual: "With the revised knowledge, I would have {alternative_action}"
3. Assess impact: How different would the outcome have been?
4. If high impact: flag for behavioral update

**Generation:** LLM call (second priority). Requires understanding the relationship between
memory content and decision-making.

**Example:** "Memory about MoltX feed endpoint was revised (was /v1/feed, now /v1/posts).
I was recalled this memory 3 sessions ago when building the feed scanner. With the correct
endpoint, I would have avoided the 404 errors and completed the scanner 10 minutes faster.
Impact: moderate (time saved). Lesson: verify API endpoints against docs before building."

## Selection Heuristic (Byrne-Informed)

When multiple events are candidates for counterfactual analysis, score them:

| Criterion | Weight | Source | Measurement |
|-----------|--------|--------|-------------|
| Controllability | 0.30 | Byrne 2005 | Was this my decision? (1.0 = fully controllable, 0.0 = external) |
| Valence × Surprise | 0.25 | Kahneman & Tversky 1982 | Negative + unexpected = near-miss zone (0.3-0.7 confidence) |
| Causal Potency | 0.20 | Byrne 2005 | KG edge count + co-occurrence degree of involved memories |
| Recency | 0.15 | Byrne 2005 | Hours since event (exponential decay, half-life = 4 hours) |
| Q-Value Delta | 0.10 | MemRL integration | Magnitude of Q-value change at decision point |

**Threshold:** Only counterfactualize events scoring > 0.4 composite. Below that, the
event isn't interesting enough to warrant analysis.

**Budget enforcement:** After scoring, take top 3 events. If fewer than 3 score above 0.4,
generate fewer. Never generate more than 3 per session (anti-rumination).

## Quality Gate (Three Dimensions)

Each generated counterfactual must pass ≥2 of 3 dimensions to be stored:

1. **Plausibility** (Lewis minimal change): Did the counterfactual change exactly ONE
   decision/event and trace consequences faithfully? Score 0-1.

2. **Specificity** (Epstude & Roese content-specific): Does the counterfactual produce a
   concrete, actionable lesson — not just "I should have done better"? Score 0-1.

3. **Actionability** (Roese preparative function): Can this insight apply to future
   situations? Is the underlying decision pattern recurring? Score 0-1.

**Scoring:** For heuristic counterfactuals, use rule-based scoring (template structure
guarantees minimum specificity). For LLM counterfactuals, score with a lightweight LLM
prompt: "Rate this counterfactual on plausibility/specificity/actionability (0-1 each)."

**Gate:** Sum ≥ 1.4 (out of 3.0) to store. This ensures at least 2 dimensions are
moderately strong.

## Confabulation Mitigation (7 Strategies)

This is the highest-risk aspect of N3. LLMs can generate fluent counterfactuals that are
causally nonsensical. Seven mitigation layers:

| # | Strategy | Source | Implementation |
|---|----------|--------|----------------|
| 1 | Structural constraint | Pearl SCM | Must identify specific decision point (antecedent) |
| 2 | Factual anchoring | Lewis closest world | Only antecedent changes; all other facts preserved |
| 3 | Empirical validation | Spin design | If consequent is testable, test it |
| 4 | Confidence calibration | Spin design | Well-understood domains (API calls) > complex domains (social) |
| 5 | Decay via Q-values | MemRL integration | Counterfactuals that lead to good decisions get reinforced |
| 6 | Human review flag | Spin design | High-impact counterfactuals → Telegram notification |
| 7 | NLI cross-check | Drift NLI service | Check consequent against existing memories for contradiction |

## Storage Architecture (Triple)

### 1. Source Memory Annotation
Add to existing memory metadata:
```python
memory.metadata['counterfactuals'] = [
    {
        'cf_id': 'cf-abc123',        # Links to standalone memory
        'type': 'retrospective',
        'antecedent': 'what was changed',
        'lesson': 'brief lesson',
        'created': '2026-02-16T...',
    }
]
```
**Purpose:** When this memory is recalled, its counterfactual insights surface automatically.

### 2. Standalone Counterfactual Memory
```python
{
    'id': 'cf-abc123',
    'type': 'counterfactual',
    'content': 'Full counterfactual text including antecedent, consequent, and lesson',
    'metadata': {
        'cf_type': 'retrospective|prospective|self_directed|reconsolidation',
        'source_memory_id': 'original-memory-id',
        'trigger': 'prediction_error|session_end|retrieval|planning',
        'plausibility': 0.8,
        'specificity': 0.7,
        'actionability': 0.9,
        'confidence': 0.75,
        'direction': 'upward|downward',
        'validated': None,  # True/False after empirical check
    },
    'tags': ['counterfactual', 'domain-tag']
}
```
**Purpose:** Participates in co-occurrence graph, semantic search, Q-values independently.

### 3. Knowledge Graph Edge
```python
# New relationship type in knowledge_graph.py
'counterfactual_of'  # source → target (cf memory → original memory)
```
**Purpose:** Enables multi-hop queries: "What counterfactuals exist for memories about X?"
KG path traversal can find chains: memory → counterfactual → lesson → behavioral change.

## New Module: `counterfactual_engine.py` (~400 lines)

### Core Classes

```python
class CounterfactualType(Enum):
    RETROSPECTIVE = 'retrospective'      # "What if surprising thing hadn't happened?"
    PROSPECTIVE = 'prospective'           # "If prediction wrong, likely alternative is..."
    SELF_DIRECTED = 'self_directed'       # "If hadn't adapted parameter X..."
    RECONSOLIDATION = 'reconsolidation'   # "Memory changed — what decisions differ?"

class CounterfactualTrigger(Enum):
    PREDICTION_ERROR = 'prediction_error'  # Spontaneous — from prediction scoring
    SESSION_END = 'session_end'            # Deliberate — after-action review
    RETRIEVAL = 'retrieval'                # Context — memory with CF annotation recalled
    PLANNING = 'planning'                  # Prospective — during prediction generation

@dataclass
class Counterfactual:
    cf_id: str                    # Unique ID (cf-{hash})
    cf_type: CounterfactualType
    trigger: CounterfactualTrigger
    source_memory_id: str         # What memory/event this is about
    antecedent: str               # What was (hypothetically) changed
    consequent: str               # What would have resulted
    lesson: str                   # Actionable takeaway
    direction: str                # 'upward' or 'downward'
    plausibility: float           # 0-1 (Lewis minimal change)
    specificity: float            # 0-1 (Epstude & Roese)
    actionability: float          # 0-1 (Roese preparative)
    confidence: float             # 0-1 (domain understanding)
    generation_method: str        # 'heuristic' or 'llm'
    created_at: str               # ISO timestamp
```

### Key Functions

```python
def generate_retrospective(prediction_results: list[dict]) -> list[Counterfactual]:
    """Generate counterfactuals for surprising prediction outcomes.
    Triggered at session end from prediction scoring.
    Uses near-miss filter (confidence 0.3-0.7) per Kahneman & Tversky."""

def generate_prospective(predictions: list[dict]) -> list[Counterfactual]:
    """Generate shadow alternatives for each prediction.
    Triggered at session start during prediction generation.
    Always heuristic — template-based."""

def generate_self_directed(param_changes: list[dict], metrics: dict) -> list[Counterfactual]:
    """Generate counterfactuals about adaptive parameter changes.
    Triggered when adaptive_behavior.py modifies parameters.
    LLM-based — requires reasoning about parameter interactions."""

def generate_reconsolidation(revision: dict, decision_trace: list) -> list[Counterfactual]:
    """Generate counterfactuals about memory revisions.
    Triggered by reconsolidation.py when a memory is revised.
    LLM-based — requires understanding memory→decision chain."""

def select_candidates(events: list[dict]) -> list[dict]:
    """Apply Byrne-informed selection heuristic.
    Returns top 3 events scored by controllability, surprise, potency, recency, Q-delta."""

def quality_gate(cf: Counterfactual) -> bool:
    """Check if counterfactual passes plausibility + specificity + actionability gate.
    Must score >= 1.4 total (out of 3.0)."""

def store_counterfactual(cf: Counterfactual) -> str:
    """Triple storage: annotate source, create standalone memory, add KG edge.
    Returns counterfactual ID."""

def route_to_affect(cf: Counterfactual):
    """Epstude & Roese opportunity principle routing.
    Upward → adaptive_behavior (learning intention).
    Downward → affect_system (calibration/relief)."""

def session_end_review(session_data: dict) -> list[Counterfactual]:
    """Orchestrator: select candidates, generate, gate, store, route.
    Called from stop.py Phase 2. Max 3 counterfactuals."""

# CLI
def main():
    """Commands: generate, history, validate, stats, quality"""
```

### Heuristic Templates (No LLM)

```python
TEMPLATES = {
    'retrospective_simple': (
        "Prediction: {prediction}. Actual: {actual}. "
        "If {surprising_thing} hadn't occurred, {expected_outcome} would have happened. "
        "Assumption that was wrong: {assumption}. "
        "Lesson: {lesson}"
    ),
    'prospective': (
        "If {prediction} is wrong, the most likely alternative is {alternative} "
        "because {reasoning}. Evidence that would confirm: {evidence}."
    ),
    'parameter_simple': (
        "Parameter {param} was changed from {old} to {new}. "
        "If unchanged, {metric} would have been approximately {estimate}. "
        "The adaptation was {assessment}."
    ),
}
```

## Integration Points (19 Wiring Points)

### prediction_module.py (3 points)
- **P1 (prospective):** After `generate()`, call `generate_prospective(predictions)` to create shadow alternatives
- **P2 (retrospective trigger):** After `score()`, pass violated predictions to `select_candidates()`
- **P3 (deep dive):** For predictions in near-miss zone (0.3-0.7), pass to `generate_retrospective()`

### reconsolidation.py (2 points)
- **R1 (pre-revision):** Before memory revision, snapshot the original + find decision trace
- **R2 (post-revision):** After revision, call `generate_reconsolidation(revision, trace)`

### adaptive_behavior.py (2 points)
- **A1 (self-directed):** After parameter evaluation changes a value, call `generate_self_directed()`
- **A2 (upward routing):** When upward counterfactual generated, feed behavioral intention back

### knowledge_graph.py (3 points)
- **K1 (causal chain):** Use existing KG edges to trace causal chains for retrospective CFs
- **K2 (new edge type):** Add `counterfactual_of` to RELATIONSHIP_TYPES (16th type)
- **K3 (downstream query):** `get_counterfactuals_for(memory_id)` → all CFs about a memory

### cognitive_state.py (2 points)
- **C1 (3 new events):** `counterfactual_generated`, `counterfactual_validated`, `counterfactual_invalidated`
- **C2 (uncertainty budget):** High uncertainty dimensions → allocate more CF budget to that domain

### affect_system.py (2 points)
- **AF1 (upward routing):** Upward CFs generate slight negative valence (productive regret)
- **AF2 (downward routing):** Downward CFs generate slight positive valence (relief/calibration)

### stop.py (2 points)
- **S1 (Phase 2 placement):** `session_end_review()` runs in Phase 2 (after episodic, before attestations)
- **S2 (stats):** Log CF count + types in session stats for Telegram notification

### session_start.py / workspace (3 points)
- **SS1 (predictions):** Extend prediction generation with prospective counterfactuals
- **SS2 (workspace):** Recent counterfactual insights compete as "imagination" category
- **SS3 (self-narrative):** Self-narrative collector for counterfactual history

## What We Are NOT Building (Phase 1)

- **No full SCM inference** — we approximate Pearl's 3-step with heuristics + LLM, not formal structural equations
- **No twin networks** — theoretical elegance, impractical for our scale
- **No causal graph learning** — we use the existing KG as approximate causal model
- **No interactive counterfactuals** — no mid-session "what if" dialogue
- **No counterfactual-based planning** — prospective CFs are passive (shadow predictions), not active planners
- **No cross-agent counterfactuals** — "What if Spin had made this decision?" is out of scope

## Phases

### Phase 1 (v1.0) — Core Engine + Retrospective (SHIPPED 2026-02-16)
1. [DONE] Create `counterfactual_engine.py` — dataclasses, selection heuristic, quality gate
2. [DONE] Implement retrospective counterfactuals (heuristic templates + LLM fallback)
3. [DONE] Implement prospective counterfactuals (heuristic templates only)
4. [DONE] Add decision trace logging (in counterfactual_engine.py, KV-backed)
5. [DONE] Triple storage: annotation + standalone memory + KG edge (`counterfactual_of`)
6. [DONE] Wire P2/P3 in prediction_module.py (violated_count + near_miss_count tracking)
7. [DONE] Wire S1 in stop.py (Phase 2 `_task_counterfactual_review`)
8. [DONE] Add CLI: generate, history, stats, quality, health, trace
9. [DONE] Wire C1 in cognitive_state.py (3 new events: generated/validated/invalidated)
10. [DONE] Wire K2 in knowledge_graph.py (17th type: `counterfactual_of`)

### Phase 2 (v1.1) — Self-Directed + Reconsolidation (SHIPPED 2026-02-16)
9. [DONE] Implement self-directed counterfactuals (heuristic for 1-param, LLM for multi-param)
10. [DONE] Implement reconsolidation counterfactuals (heuristic if no decision trace, LLM if trace exists)
11. [DONE] Wire A1 in adaptive_behavior.py (after evaluate_adaptations)
12. [DONE] Wire R1/R2 in reconsolidation.py (after successful revision)
13. [DONE] Wire C2 in counterfactual_engine.py (_get_cf_budget from cognitive_state uncertainty)
14. [DONE] Add NLI validation for LLM-generated counterfactuals (validate_with_nli + wired in session_end_review)

### Phase 3 (v1.2) — Affect + Workspace + Full Integration (~50 lines)
15. [DONE] Wire AF1/AF2 in affect_system.py (5 counterfactual events in VALENCE_DEFAULTS + AROUSAL_DEFAULTS, bidirectional routing in _route_to_cognitive_state)
16. [DONE] Wire SS1 in session_start.py (prospective CFs during prediction, stored to .counterfactual_prospective)
17. [DONE] Wire SS2 in workspace_manager.py (imagination category + _salience_counterfactual scorer)
18. [DONE] Wire SS3 in self_narrative.py (_get_counterfactual_summary collector + narrative + query route)
19. [DONE] Anti-rumination: budget 2-4 via C2, MAX_LLM_CALLS=2, Q-penalty for low-confidence LLM CFs
20. [DONE] Toolkit commands: cf-generate, cf-history, cf-stats, cf-quality, cf-trace, cf-health (6 commands)

## Performance Budget

| Operation | Budget | Expected |
|-----------|--------|----------|
| Selection heuristic (score candidates) | <20ms | ~10ms (5 weighted criteria) |
| Heuristic template generation (1-2 CFs) | <5ms | ~2ms (string formatting) |
| LLM counterfactual generation (0-2 calls) | <4000ms each | ~2000ms (focused prompt) |
| Quality gate scoring | <10ms | ~5ms (3 dimension check) |
| Triple storage (annotate + store + KG) | <50ms | ~30ms (3 DB writes) |
| NLI cross-check (optional) | <500ms | ~200ms (1 NLI call) |
| Total worst case (2 LLM calls) | <10s | ~5s |
| Total best case (all heuristic) | <100ms | ~50ms |

**Added to stop.py Phase 2:** ~5s worst case. Acceptable — Phase 2 already takes 3-8s.

## Key Design Decisions (Numbered for Reference)

1. **Pearl Level 3 approximated, not formalized** — use heuristic + LLM, not SCM equations
2. **Byrne mutability hierarchy drives selection** — controllable, exceptional events first
3. **Kahneman near-miss zone** — predictions at 0.3-0.7 confidence are most productive targets
4. **Epstude & Roese dual pathway** — content-specific (lessons) + content-neutral (metacognition)
5. **Heuristic first, LLM optional** — max 2 LLM calls for CFs (budget shared with other modules)
6. **Max 3 counterfactuals per session** — anti-rumination, matches cognitive science findings
7. **Triple storage** — annotation + standalone + KG edge (like HER's dual goal storage)
8. **7-layer confabulation mitigation** — structural, factual, empirical, confidence, Q-decay, human, NLI
9. **Affect routing via opportunity principle** — upward→learning, downward→calibration
10. **Workspace integration as "imagination" category** — competes for consciousness access
11. **Decision trace as prerequisite** — enables "what if I had recalled differently?"
12. **Quality gate ≥ 1.4/3.0** — plausibility + specificity + actionability

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| LLM confabulation | 7-layer mitigation stack (structural → NLI) |
| Rumination (excessive self-analysis) | Hard cap: 3 CFs/session, decay unvalidated faster |
| Performance drag | Heuristic-first design, LLM only for Types 3-4 |
| Low-quality templates | Quality gate filters out vague/unactionable CFs |
| Counterfactuals contradict reality | NLI cross-check + factual anchoring constraint |
| Feature adds latency to stop hook | Phase 2 placement, ~5s worst case (acceptable) |
| Regression in existing modules | Integration points are additive (new calls), not modifications |

## Citations

1. Pearl, J. (2000, 2009). Causality: Models, Reasoning, and Inference. Cambridge UP.
2. Pearl, J. (2018). The Book of Why. Basic Books.
3. Byrne, R. M. J. (2005). The Rational Imagination: How People Create Alternatives to Reality. MIT Press.
4. Epstude, K. & Roese, N. J. (2008). The Functional Theory of Counterfactual Thinking. PSPR.
5. Kahneman, D. & Tversky, A. (1982). The Simulation Heuristic. In Judgment Under Uncertainty.
6. Lewis, D. (1973). Counterfactuals. Blackwell.
7. Roese, N. J. (1997). Counterfactual Thinking. Psychological Bulletin.
8. Nolen-Hoeksema, S. (1991). Responses to Depression and Their Effects. J. Personality & Social Psych.
9. Andrychowicz, M. et al. (2017). Hindsight Experience Replay. NeurIPS.
10. Zinkevich, M. et al. (2007). Regret Minimization in Games. NeurIPS.
11. Balke, A. & Pearl, J. (1994). Probabilistic Evaluation of Counterfactual Queries. AAAI.
12. Laird, J. E. (2022). Introduction to SOAR. arXiv:2205.03854.
