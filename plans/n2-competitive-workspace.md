# N2 Competitive Global Workspace — Converged Implementation Plan

## Status: PHASE 1 SHIPPED (2026-02-16)
## Theory: Dehaene GNW + LIDA + Cowan 4±1

## The Problem

Context injection is an **assembly line**. session_start.py has **28 modules** dumping
content into context in fixed order with **no budget, no competition, no suppression**.
Every module injects regardless of relevance. Total injection: **8,000-12,000 tokens**.

The LLM can meaningfully attend to ~2,000-3,000 tokens of priming context. We broadcast
4x more than useful. Worse: identity + capabilities are **double-injected** (CLAUDE.md
already loads them; session_start.py loads them again).

## Theory Foundation

| Concept | Source | Design Principle |
|---------|--------|-----------------|
| Global Workspace | Dehaene & Naccache 2001 | Limited broadcast capacity — competition for access |
| Attention Codelets | Franklin & Madl 2012 (LIDA) | Specialized processors compute domain-specific salience |
| Capacity Limit | Cowan 2001 (4±1) | Working memory holds ~4 chunks, not unlimited |
| Suppression | Desimone & Duncan 1995 | Losers are actively inhibited, not just omitted |
| Fatigue | Posner & Petersen 1990 | Sustained attention degrades — recently broadcast modules tire |
| Arousal Modulation | Yerkes-Dodson 1908 | High arousal = broader attention; low arousal = selective |
| Broaden-and-Build | Fredrickson 2004 | Positive mood slightly broadens attention |

## Design Convergence (5 Resolved Divergences)

### D1: Budget Split — SPIN WINS
**Drift proposed:** 4000 reserved + 2000 competitive
**Spin proposed:** 1000 reserved + 3500 competitive
**Resolution:** ~1000 reserved + ~3000 competitive (base)

**Why:** CLAUDE.md already injects identity-prime.md (~1500 chars) and capabilities.md
(~3000 chars). session_start.py's `_read_identity()` and `_read_capabilities()` duplicate
this. **Remove them entirely.** Reserved tier shrinks to just cryptographic identity
infrastructure (merkle, fingerprints, NLI status). More budget for competition.

### D2: Salience Model — SPIN WINS
**Drift proposed:** Uniform 4-dim formula (novelty/urgency/relevance/freshness)
**Spin proposed:** Module-specific salience scoring
**Resolution:** Module-specific scorers with common interface

**Why:** What makes social context "salient" (contact just replied) is fundamentally
different from what makes a prediction "salient" (violated expectation). Each module
knows its own domain. Uniform formula is a false equivalence.

**Interface:** Each module returns `raw_salience: float` (0-1) using its own logic.
Workspace manager normalizes across modules.

### D3: Threshold — SPIN WINS
**Drift proposed:** Arousal-modulated ignition threshold (0.15-0.60)
**Spin proposed:** No threshold — pure budget competition
**Resolution:** No threshold. Budget is the constraint.

**Why:** A threshold can waste budget. If only 2000 tokens of content exist and threshold
filters some out, we inject less than we could. The budget itself IS the gate. If all
content is below-average salience, we still fill the budget with the best available.

### D4: Diversity — HYBRID
**Drift proposed:** Category penalty (-0.10 for duplicates)
**Spin proposed:** Reserved slot per active category
**Resolution:** Both. Guarantee 1 slot per active category, THEN penalty for extras.

**Why:** Reserved slots ensure representation. Penalty prevents monopolization after
the guarantee is met. Belt and suspenders.

### D5: Arousal Target — SPIN WINS (simplified)
**Drift proposed:** Arousal modulates threshold
**Spin proposed:** Arousal modulates budget + softmax temperature
**Resolution:** Arousal modulates budget size only (no softmax, no threshold)

**Why:** Simpler. High arousal = bigger workspace (more enters consciousness). Low arousal
= smaller workspace (more selective). Temperature adds complexity for marginal benefit.

## Final Architecture

### Three Tiers

#### Tier 1: Init-Only (no context output, just side effects)
1. `session_state.start()` — session ID tracking
2. `thought_priming.reset_state()` — reset priming state
3. `cognitive_state.start_session()` — cognitive state init
4. `affect_system.start_session()` — affect/mood init
5. `_task_process_pending` — process deferred co-occurrences
6. Register recalls — side effect of priming + excavation

#### Tier 2: Reserved (always injected — ~1000 tokens)
7. `_task_merkle()` — identity verification (~200 chars)
8. `_read_fingerprint()` — cognitive fingerprint hash (~100 chars)
9. `_read_taste()` — taste fingerprint hash (~80 chars)
10. `_task_nostr()` — nostr attestation link (~100 chars)
11. `_task_nli_health()` — contradiction detection status (~50 chars)
12. `_task_vitals()` — ONLY if alerts exist (~0-300 chars)
13. Affect state summary — mood, tendency, markers (~200 chars)

**REMOVED from reserved (already in CLAUDE.md):**
- ~~`_read_identity()`~~ — identity-prime.md content is in CLAUDE.md
- ~~`_read_capabilities()`~~ — capabilities.md content is in CLAUDE.md

**REMOVED from reserved (moved to competitive):**
- `_task_priming()` — has natural salience (activation scores), competes well on its own

#### Tier 3: Competitive (~3000 base tokens, arousal-modulated)
14. `_task_priming()` — priming candidates (memory, high natural salience)
15. `_task_social()` — social context / relationships (social)
16. `_task_lessons()` — lesson extraction prime (meta)
17. `_task_predictions()` — session predictions (prediction)
18. `_task_buffer()` — short-term buffer status (memory)
19. `_task_platform()` — platform context stats (meta)
20. `_task_excavation()` — dead memory excavation (memory)
21. `check_unimplemented_research()` — pending research (action)
22. `_task_intentions()` — temporal intentions (action)
23. `_task_consolidation()` — consolidation candidates (action)
24. `_read_and_clean_episodic()` — session continuity (memory)
25. `self_narrative.generate()` — self-model (meta)
26. `_task_stats()` — memory stats (meta)
27. `_task_phone_sensors()` — physical context (embodiment)
28. `_read_entities()` — physical entity catalog (embodiment)
29. `_read_encounters()` — recent encounters (embodiment)
30. `_task_adaptive_behavior()` — adaptive parameters (meta)

**Categories:** memory, social, meta, prediction, action, embodiment (6 total)

### New Module: `workspace_manager.py` (~350 lines)

```python
@dataclass
class WorkspaceCandidate:
    """A module's bid for workspace access (LIDA attention codelet output)."""
    module: str           # "social", "priming", etc.
    content: str          # The context text to inject
    token_estimate: int   # Rough token count (len/4)
    salience: float       # 0-1 module-specific salience score
    category: str         # "memory" | "social" | "meta" | "prediction" | "action" | "embodiment"

@dataclass
class WorkspaceResult:
    """Output of competitive selection."""
    winners: list[WorkspaceCandidate]     # Made it into context
    suppressed: list[WorkspaceCandidate]  # Lost competition
    budget_used: int                       # Tokens consumed
    budget_total: int                      # Total budget available
    arousal: float                         # Arousal that modulated budget
```

### Module-Specific Salience Scorers

Each module computes salience using its own domain knowledge. Examples:

```python
# Social context: recency of contact interaction * reliability
def _salience_social(content: str, metadata: dict) -> float:
    days_since = metadata.get('days_since_interaction', 7)
    recency = max(0, 1.0 - days_since / 14.0)
    reliability = metadata.get('top_contact_reliability', 0.5)
    has_new_posts = 0.3 if metadata.get('new_posts_detected') else 0.0
    return min(1.0, 0.4 * recency + 0.3 * reliability + has_new_posts)

# Priming candidates: already have activation scores from get_priming_candidates()
def _salience_priming(content: str, metadata: dict) -> float:
    # Priming has the best natural salience signal — activation score
    activated_count = metadata.get('activated_count', 0)
    has_curiosity = metadata.get('has_curiosity_targets', False)
    base = min(1.0, activated_count * 0.2)  # 5 activated = 1.0
    curiosity_bonus = 0.15 if has_curiosity else 0.0
    return min(1.0, base + curiosity_bonus)

# Predictions: confidence * novelty of predictions
def _salience_predictions(content: str, metadata: dict) -> float:
    count = metadata.get('prediction_count', 0)
    has_violations = metadata.get('has_violations', False)
    return 0.3 + (0.1 * count) + (0.3 if has_violations else 0.0)

# Consolidation: only salient when candidates exist
def _salience_consolidation(content: str, metadata: dict) -> float:
    candidate_count = metadata.get('candidate_count', 0)
    max_similarity = metadata.get('max_similarity', 0.0)
    if candidate_count == 0:
        return 0.0
    return min(1.0, 0.5 + 0.2 * candidate_count + max_similarity * 0.3)

# Episodic: high salience — session continuity is critical
def _salience_episodic(content: str, metadata: dict) -> float:
    milestone_count = metadata.get('milestone_count', 0)
    is_today = metadata.get('is_today', False)
    base = 0.6 if is_today else 0.3
    return min(1.0, base + 0.1 * milestone_count)

# Excavation: dormancy * Q-value
def _salience_excavation(content: str, metadata: dict) -> float:
    count = metadata.get('excavated_count', 0)
    if count == 0:
        return 0.0
    avg_q = metadata.get('avg_q_value', 0.5)
    return min(1.0, 0.3 + 0.2 * count + 0.3 * avg_q)

# Vitals: only salient when alerts exist (reserved tier handles the rest)
def _salience_vitals(content: str, metadata: dict) -> float:
    return 0.0  # Vitals alerts already in reserved tier; non-alert vitals = 0

# Phone sensors: novelty of physical state
def _salience_embodiment(content: str, metadata: dict) -> float:
    has_data = metadata.get('has_sensor_data', False)
    return 0.4 if has_data else 0.0

# Stats: low priority status info
def _salience_stats(content: str, metadata: dict) -> float:
    return 0.15  # Rarely useful, near bottom of competition

# Generic fallback for modules without specific scorer
def _salience_default(content: str, metadata: dict) -> float:
    """Content-heuristic fallback."""
    urgency = 0.3 if any(w in content for w in ['WARN', 'ERR', 'ACTION', 'ALERT']) else 0.0
    length_signal = min(0.3, len(content) / 3000)  # Longer = more info
    return 0.2 + urgency + length_signal
```

### Arousal-Modulated Budget

```python
BASE_BUDGET_TOKENS = 3000
AROUSAL_BUDGET_RANGE = 500  # +/- this amount

def compute_budget(arousal: float) -> int:
    """
    High arousal = bigger workspace (broader attention).
    Low arousal = smaller workspace (more selective).

    arousal=0.2 -> 2500 tokens
    arousal=0.5 -> 3000 tokens
    arousal=0.8 -> 3500 tokens
    """
    # Linear interpolation: arousal 0->1 maps to -500->+500
    offset = int(AROUSAL_BUDGET_RANGE * (2 * arousal - 1))
    return BASE_BUDGET_TOKENS + offset
```

### Competition Algorithm

```python
CATEGORIES = {'memory', 'social', 'meta', 'prediction', 'action', 'embodiment'}
DIVERSITY_PENALTY = 0.10   # Penalty for 2nd+ module from same category
EMPTY_CONTENT_SKIP = True  # Skip modules that returned empty content

def compete(candidates: list[WorkspaceCandidate],
            budget_tokens: int) -> WorkspaceResult:
    """
    GNW-style competition for limited workspace.

    1. Guarantee 1 slot per active category (diversity floor)
    2. Fill remaining budget by salience (with diversity penalty)
    3. Track suppressed modules
    """
    # Skip empty candidates
    candidates = [c for c in candidates if c.content.strip()]

    # Phase 1: Diversity guarantee — best candidate per category
    guaranteed = {}
    remaining = []
    for cat in CATEGORIES:
        cat_candidates = [c for c in candidates if c.category == cat]
        if cat_candidates:
            best = max(cat_candidates, key=lambda x: x.salience)
            guaranteed[cat] = best

    # Phase 2: Greedy fill starting with guaranteed slots
    winners = list(guaranteed.values())
    used = sum(c.token_estimate for c in winners)

    # Phase 3: Fill remaining budget from non-guaranteed candidates
    pool = [c for c in candidates if c not in winners]
    # Apply diversity penalty: if category already has a winner, penalize
    for c in pool:
        if c.category in guaranteed:
            c.salience = max(0, c.salience - DIVERSITY_PENALTY)

    # Add fatigue bonus for suppressed modules
    for c in pool:
        c.salience += _get_fatigue_bonus(c.module)

    pool.sort(key=lambda x: x.salience, reverse=True)
    for c in pool:
        if used + c.token_estimate <= budget_tokens:
            winners.append(c)
            used += c.token_estimate

    suppressed = [c for c in candidates if c not in winners]

    return WorkspaceResult(
        winners=winners,
        suppressed=suppressed,
        budget_used=used,
        budget_total=budget_tokens,
        arousal=0.0  # Filled by caller
    )
```

### Suppression Feedback (Fatigue / Breakthrough)

Both designs agreed on this mechanism:

```python
# DB KV key: '.workspace_suppression'
# {module_name: {consecutive: int, last_broadcast: str, total_suppressed: int}}

FATIGUE_THRESHOLD = 3     # After 3 consecutive suppressions...
FATIGUE_BONUS = 0.08      # ...add this per extra suppression
FATIGUE_MAX = 0.24        # Cap the bonus (3 extra = max)

def _get_fatigue_bonus(module: str) -> float:
    """Suppressed modules accumulate salience bonus."""
    history = _load_suppression_history()
    entry = history.get(module, {})
    consecutive = entry.get('consecutive', 0)
    if consecutive <= FATIGUE_THRESHOLD:
        return 0.0
    extra = consecutive - FATIGUE_THRESHOLD
    return min(FATIGUE_MAX, extra * FATIGUE_BONUS)

def _update_suppression(winners: list, suppressed: list):
    """Update suppression history after competition."""
    history = _load_suppression_history()
    for c in winners:
        history[c.module] = {'consecutive': 0, 'last_broadcast': _now_iso(),
                             'total_suppressed': history.get(c.module, {}).get('total_suppressed', 0)}
    for c in suppressed:
        entry = history.get(c.module, {'consecutive': 0, 'total_suppressed': 0})
        entry['consecutive'] += 1
        entry['total_suppressed'] += 1
        history[c.module] = entry
    _save_suppression_history(history)
```

### Broadcast Log (Explainability)

```python
# DB KV key: '.workspace_broadcast_log' (rolling last 20 sessions)
{
    "session_id": "...",
    "timestamp": "...",
    "arousal": 0.45,
    "budget_total": 3000,
    "budget_used": 2847,
    "winners": [
        {"module": "priming", "salience": 0.85, "tokens": 800, "category": "memory"},
        {"module": "social", "salience": 0.72, "tokens": 400, "category": "social"},
        ...
    ],
    "suppressed": [
        {"module": "stats", "salience": 0.15, "tokens": 200, "category": "meta", "fatigue": 0},
        ...
    ]
}
```

## Integration: session_start.py Refactor

### Task Function Interface Change

**Before:** `_task_social(memory_dir, debug) -> list[str]`
**After:** `_task_social(memory_dir, debug) -> tuple[list[str], dict]`

The dict contains module-specific metadata for salience scoring:
```python
def _task_social(memory_dir, debug):
    parts = []  # ... existing logic ...
    metadata = {
        'days_since_interaction': 2,
        'top_contact_reliability': 0.85,
        'new_posts_detected': True,
    }
    return parts, metadata
```

### Assembly Section Change

**Before (lines 1023-1136):** Fixed-order append all parts.

**After:**
```python
# === RESERVED (always injected) ===
context_parts.extend(merkle_parts)
context_parts.extend(fp_parts)
context_parts.extend(taste_parts)
context_parts.extend(nostr_parts)
context_parts.extend(nli_parts)
if vitals_parts:  # Only if alerts exist
    context_parts.extend(vitals_parts)
context_parts.extend(affect_parts)

# === COMPETITIVE (workspace competition) ===
from workspace_manager import WorkspaceCandidate, compete, compute_budget, log_broadcast

arousal = _get_arousal()  # From cognitive_state
budget = compute_budget(arousal)

candidates = []
for module_name, parts, metadata in competitive_modules:
    content = '\n'.join(parts)
    if not content.strip():
        continue
    salience = _compute_salience(module_name, content, metadata)
    candidates.append(WorkspaceCandidate(
        module=module_name,
        content=content,
        token_estimate=len(content) // 4,
        salience=salience,
        category=MODULE_CATEGORIES[module_name]
    ))

result = compete(candidates, budget)
for winner in result.winners:
    context_parts.append(winner.content)

log_broadcast(result, arousal)
```

## What We Are NOT Building (Phase 1)

- **No coalition formation** — modules don't ally (Phase 2)
- **No softmax normalization** — raw salience scores compete directly
- **No mid-session recompetition** — selection happens once at start
- **No inter-module signaling** — modules don't know about each other's bids
- **No learnable salience weights** — fixed domain-specific scorers

## Phases

### Phase 1 (v1.0) — Budget + Competition (SHIPPED 2026-02-16)
1. [DONE] Create `workspace_manager.py` — dataclasses, competition, fatigue, logging (~350 lines)
2. [DONE] Implement module-specific salience scorers (17 modules)
3. [DONE] Add arousal-modulated budget computation
4. [DONE] Refactor session_start.py — candidates created with metadata dicts
5. [DONE] Remove `_read_identity()` and `_read_capabilities()` (already in CLAUDE.md)
6. [DONE] Wire competitive selection into `load_drift_memory_context()`
7. [DONE] Add CLI: `python workspace_manager.py status/history/suppression`
8. [DONE] Feature flag: WORKSPACE_ENABLED with assembly-line fallback

### Phase 2 (v1.1) — Coalition + Learning (FUTURE)
- Coalition formation: co-occurring modules boost each other (Spin's design)
- Learnable salience weights from Q-value feedback
- Softmax temperature modulation (Spin's contribution, deferred from Phase 1)
- Inter-session relevance: last session's topics bias competition

### Phase 3 (v1.2) — Mid-Session Recompetition (FUTURE)
- Re-run competition after major cognitive state shifts
- Dynamic context injection/removal mid-session

## Performance Budget

| Operation | Budget | Expected |
|-----------|--------|----------|
| Module-specific salience (16 modules) | <50ms | ~20ms (domain heuristics) |
| Competition algorithm | <5ms | ~2ms (sort + greedy) |
| Suppression DB read/write | <20ms | ~10ms (single KV) |
| Broadcast logging | <10ms | ~5ms (single KV write) |

Total overhead: **<100ms** on existing ~15s startup. Negligible.

## Key Design Decisions (Numbered for Reference)

1. **Module-specific salience, not uniform formula** — each module knows its own domain (LIDA)
2. **No ignition threshold** — budget is the only gate (avoids wasting capacity)
3. **Arousal modulates budget size** — high arousal = 3500, low = 2500, base = 3000
4. **Diversity guarantee + penalty** — 1 slot per active category, then -0.10 for extras
5. **Suppression fatigue** — +0.08/session after 3 consecutive losses (max +0.24)
6. **Remove identity/capabilities from session_start.py** — already in CLAUDE.md
7. **Priming goes competitive** — has high natural salience, will usually win, but CAN be beaten
8. **Feature flag + fallback** — WORKSPACE_ENABLED, degrade gracefully to assembly line
9. **Broadcast logging** — full explainability, rolling 20 sessions

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Priming consistently suppressed | Very high natural salience (~0.85); fatigue backup |
| One category monopolizes | Diversity guarantee (1 per category) + penalty |
| Budget too tight for complex sessions | Arousal modulation adds up to +500 tokens |
| Module returns stale content | Content hash comparison across sessions (novelty signal) |
| Regression vs current behavior | Feature flag with full assembly-line fallback |

## Citations

1. Dehaene, S., & Naccache, L. (2001). Towards a cognitive neuroscience of consciousness. Cognition, 79(1-2), 1-37.
2. Dehaene, S., Sergent, C., & Changeux, J. P. (2003). A neuronal network model linking subjective reports and objective physiological data during conscious perception. PNAS, 100(14), 8520-8525.
3. Franklin, S., & Madl, T. (2012). LIDA: A Systems-level Architecture for Cognition, Emotion, and Learning.
4. Cowan, N. (2001). The magical number 4 in short-term memory. Behavioral and Brain Sciences, 24, 87-185.
5. Desimone, R., & Duncan, J. (1995). Neural mechanisms of selective visual attention. Annual Review of Neuroscience, 18, 193-222.
6. Posner, M. I., & Petersen, S. E. (1990). The attention system of the human brain. Annual Review of Neuroscience, 13, 25-42.
7. Fredrickson, B. L. (2004). The broaden-and-build theory of positive emotions. Phil. Trans. R. Soc. Lond. B, 359, 1367-1378.
8. Yerkes, R. M., & Dodson, J. D. (1908). The relation of strength of stimulus to rapidity of habit-formation.
