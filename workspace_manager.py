#!/usr/bin/env python3
"""
Workspace Manager — N2 Competitive Global Workspace.

Implements Dehaene's Global Neuronal Workspace theory: multiple specialized
modules compete for limited broadcast capacity (context injection). Winners
achieve global availability (reach the LLM). Losers are suppressed.

Key mechanisms:
  - Module-specific salience scoring (LIDA attention codelets)
  - Arousal-modulated budget (Yerkes-Dodson)
  - Category diversity guarantee + penalty (Desimone & Duncan 1995)
  - Suppression fatigue / breakthrough (Posner & Petersen 1990)
  - Full broadcast logging for explainability

Converged design: Drift + Spin (2026-02-16).
Plan: memory/plans/n2-competitive-workspace.md

Usage:
    python workspace_manager.py status          # Show last broadcast result
    python workspace_manager.py history         # Show broadcast history
    python workspace_manager.py simulate        # Dry-run competition with current data
"""

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from db_adapter import get_db

# ─── Feature Flag ────────────────────────────────────────────────────────────

WORKSPACE_ENABLED = True  # Set False to fall back to assembly-line injection

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_BUDGET_TOKENS = 3000
AROUSAL_BUDGET_RANGE = 500   # ±500 tokens based on arousal

# Diversity
CATEGORIES = frozenset({'memory', 'social', 'meta', 'prediction', 'action', 'embodiment', 'imagination'})
DIVERSITY_PENALTY = 0.10     # Penalty for 2nd+ module from same category

# Suppression fatigue (losers accumulate bonus)
FATIGUE_THRESHOLD = 3        # After N consecutive suppressions...
FATIGUE_BONUS = 0.08         # ...add this per extra suppression
FATIGUE_MAX = 0.24           # Cap the bonus

# Winner fatigue / habituation (winners accumulate penalty)
WINNER_FATIGUE_THRESHOLD = 2    # After N consecutive wins...
WINNER_FATIGUE_PENALTY = 0.05   # ...subtract this per extra win
WINNER_FATIGUE_MAX = 0.15       # Cap the penalty

# DB keys
_SUPPRESSION_KEY = '.workspace_suppression'
_BROADCAST_LOG_KEY = '.workspace_broadcast_log'
_BROADCAST_LOG_MAX = 20      # Rolling window

# Module → category mapping
MODULE_CATEGORIES = {
    'priming': 'memory',
    'buffer': 'memory',
    'excavation': 'memory',
    'episodic': 'memory',
    'social': 'social',
    'predictions': 'prediction',
    'lessons': 'meta',
    'platform': 'meta',
    'self_narrative': 'meta',
    'stats': 'meta',
    'adaptive': 'meta',
    'research': 'action',
    'intentions': 'action',
    'consolidation': 'action',
    'phone': 'embodiment',
    'entities': 'embodiment',
    'encounters': 'embodiment',
    'counterfactual': 'imagination',
    'goals': 'action',  # N4: Volitional goals compete in action category
    'monologue': 'meta',  # N6: Inner monologue competes in meta category
    'procedural': 'action',  # Phase 3: Context-triggered skill chunks
    'agenthub': 'social',  # Agent Hub DMs
    'episodic_future': 'imagination',  # T4.2: Episodic future thinking
}


# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class WorkspaceCandidate:
    """A module's bid for workspace access (LIDA attention codelet output)."""
    module: str
    content: str
    token_estimate: int
    salience: float
    category: str

    @staticmethod
    def from_parts(module: str, parts: list, metadata: dict) -> Optional['WorkspaceCandidate']:
        """Create a candidate from task function output."""
        content = '\n'.join(str(p) for p in parts).strip()
        if not content:
            return None
        salience = compute_salience(module, content, metadata)
        category = MODULE_CATEGORIES.get(module, 'meta')
        token_estimate = max(1, len(content) // 4)
        return WorkspaceCandidate(
            module=module,
            content=content,
            token_estimate=token_estimate,
            salience=salience,
            category=category,
        )


@dataclass
class WorkspaceResult:
    """Output of competitive selection."""
    winners: list       # list of WorkspaceCandidate (as dicts for serialization)
    suppressed: list     # list of WorkspaceCandidate (as dicts for serialization)
    budget_used: int
    budget_total: int
    arousal: float
    dropped_guaranteed: list = field(default_factory=list)  # BUG-12: categories that lost guaranteed slot to budget


# ─── Arousal-Modulated Budget ────────────────────────────────────────────────

def compute_budget(arousal: float) -> int:
    """
    High arousal = bigger workspace (broader attention).
    Low arousal = smaller workspace (more selective).

    arousal=0.2 -> 2500 tokens
    arousal=0.5 -> 3000 tokens
    arousal=0.8 -> 3500 tokens
    """
    arousal = max(0.0, min(1.0, arousal))
    offset = int(AROUSAL_BUDGET_RANGE * (2 * arousal - 1))
    return BASE_BUDGET_TOKENS + offset


# ─── Module-Specific Salience Scorers ────────────────────────────────────────

def _salience_priming(content: str, meta: dict) -> float:
    """Priming has the best natural salience — activation scores."""
    activated = meta.get('activated_count', 0)
    has_curiosity = meta.get('has_curiosity_targets', False)
    has_domain = meta.get('has_domain_primed', False)
    base = min(1.0, activated * 0.2)  # 5 activated = 1.0
    bonus = 0.0
    if has_curiosity:
        bonus += 0.10
    if has_domain:
        bonus += 0.05
    return min(1.0, base + bonus)


def _salience_social(content: str, meta: dict) -> float:
    """Social: recency * reliability + new activity detection."""
    days_since = meta.get('days_since_interaction', 7)
    recency = max(0.0, 1.0 - days_since / 14.0)
    reliability = meta.get('top_contact_reliability', 0.5)
    new_posts = 0.25 if meta.get('new_posts_detected') else 0.0
    return min(1.0, 0.35 * recency + 0.25 * reliability + new_posts + 0.15)


def _salience_predictions(content: str, meta: dict) -> float:
    """Predictions: count + violations make them urgent."""
    count = meta.get('prediction_count', 0)
    has_violations = meta.get('has_violations', False)
    if count == 0:
        return 0.0
    return min(1.0, 0.30 + 0.10 * count + (0.30 if has_violations else 0.0))


def _salience_episodic(content: str, meta: dict) -> float:
    """Episodic continuity: critical for session-to-session identity."""
    milestone_count = meta.get('milestone_count', 0)
    is_today = meta.get('is_today', False)
    base = 0.55 if is_today else 0.25
    return min(1.0, base + 0.08 * milestone_count)


def _salience_consolidation(content: str, meta: dict) -> float:
    """Consolidation: only salient when candidates exist."""
    count = meta.get('candidate_count', 0)
    if count == 0:
        return 0.0
    max_sim = meta.get('max_similarity', 0.0)
    return min(1.0, 0.45 + 0.15 * count + max_sim * 0.30)


def _salience_excavation(content: str, meta: dict) -> float:
    """Excavation: dormant memories getting a second chance."""
    count = meta.get('excavated_count', 0)
    if count == 0:
        return 0.0
    avg_q = meta.get('avg_q_value', 0.5)
    return min(1.0, 0.25 + 0.15 * count + 0.30 * avg_q)


def _salience_lessons(content: str, meta: dict) -> float:
    """Lessons: relevance of primed heuristics."""
    count = meta.get('lesson_count', 0)
    if count == 0:
        return 0.0
    return min(1.0, 0.30 + 0.10 * count)


def _salience_buffer(content: str, meta: dict) -> float:
    """Short-term buffer: item count + max salience."""
    item_count = meta.get('item_count', 0)
    if item_count == 0:
        return 0.0
    max_salience = meta.get('max_item_salience', 0.0)
    return min(1.0, 0.20 + 0.05 * item_count + 0.30 * max_salience)


def _salience_platform(content: str, meta: dict) -> float:
    """Platform stats: low baseline, spikes on significant change."""
    has_change = meta.get('significant_change', False)
    return 0.45 if has_change else 0.15


def _salience_self_narrative(content: str, meta: dict) -> float:
    """Self-narrative: moderate baseline, higher if state is unusual."""
    state_unusual = meta.get('state_unusual', False)
    return 0.50 if state_unusual else 0.30


def _salience_intentions(content: str, meta: dict) -> float:
    """Temporal intentions: high if any are triggered."""
    triggered_count = meta.get('triggered_count', 0)
    if triggered_count == 0:
        return 0.0
    return min(1.0, 0.50 + 0.20 * triggered_count)


def _salience_research(content: str, meta: dict) -> float:
    """Pending research: moderate baseline when items exist."""
    count = meta.get('research_count', 0)
    if count == 0:
        return 0.0
    return min(1.0, 0.30 + 0.10 * count)


def _salience_adaptive(content: str, meta: dict) -> float:
    """Adaptive behavior: only salient when adjustments made."""
    adjustment_count = meta.get('adjustment_count', 0)
    if adjustment_count == 0:
        return 0.0
    return min(1.0, 0.35 + 0.15 * adjustment_count)


def _salience_stats(content: str, meta: dict) -> float:
    """Memory stats: low priority status info."""
    return 0.12


def _salience_embodiment(content: str, meta: dict) -> float:
    """Phone sensors / physical context: novelty of physical state."""
    has_data = meta.get('has_sensor_data', False)
    return 0.35 if has_data else 0.0


def _salience_entities(content: str, meta: dict) -> float:
    """Physical entity catalog: low unless new entities detected."""
    new_entities = meta.get('new_entities', False)
    return 0.40 if new_entities else 0.10


def _salience_encounters(content: str, meta: dict) -> float:
    """Recent encounters: recency-weighted."""
    count = meta.get('encounter_count', 0)
    if count == 0:
        return 0.0
    return min(1.0, 0.30 + 0.15 * count)


def _salience_counterfactual(content: str, meta: dict) -> float:
    """N3/SS2: Salience scorer for counterfactual insights (imagination category)."""
    base = 0.20
    # +0.3 if CF topic matches current social context
    if any(w in content.lower() for w in ['contact', 'platform', 'social', 'collaboration']):
        base += 0.30
    # +0.2 if validated (NLI-confirmed)
    if 'validated' in content.lower() or meta.get('validated'):
        base += 0.20
    return min(1.0, base)


def _salience_episodic_future(content: str, meta: dict) -> float:
    """T4.2: Salience scorer for prospective memory (episodic future thinking)."""
    base = 0.20
    active = meta.get('active_count', 0)
    if active > 0:
        base += 0.15
    # Boost if trigger fired (would be injected via post_tool_use, not workspace)
    if 'TRIGGERED' in content:
        base += 0.30
    return min(1.0, base)


def _salience_goals(content: str, meta: dict) -> float:
    """N4/WS1: Focus goal gets high salience; active goals moderate."""
    has_focus = 'FOCUS GOAL' in content
    active_count = meta.get('active_goal_count', content.count('['))
    if has_focus:
        base = 0.80
    elif active_count > 0:
        base = 0.40 + 0.10 * min(active_count, 3)
    else:
        return 0.0
    # +0.1 if lesson is actionable (contains "should", "update", "next time")
    if any(w in content.lower() for w in ['should', 'update', 'next time', 'adjust']):
        base += 0.10
    return min(1.0, base)


def _salience_monologue(content: str, meta: dict) -> float:
    """N6: Inner monologue — higher salience when warnings or adversarial."""
    base = 0.35
    if meta.get('has_warnings'):
        base += 0.25
    if meta.get('adversarial'):
        base += 0.15
    confidence = meta.get('confidence', 0.5)
    base += 0.10 * confidence  # Higher monologue confidence = more salient
    return min(1.0, base)


def _salience_default(content: str, meta: dict) -> float:
    """Fallback heuristic for unknown modules."""
    urgency = 0.25 if any(w in content for w in ['WARN', 'ERR', 'ACTION', 'ALERT']) else 0.0
    length_signal = min(0.25, len(content) / 4000)
    return 0.15 + urgency + length_signal


# Scorer dispatch
_SALIENCE_SCORERS = {
    'priming': _salience_priming,
    'social': _salience_social,
    'predictions': _salience_predictions,
    'episodic': _salience_episodic,
    'consolidation': _salience_consolidation,
    'excavation': _salience_excavation,
    'lessons': _salience_lessons,
    'buffer': _salience_buffer,
    'platform': _salience_platform,
    'self_narrative': _salience_self_narrative,
    'intentions': _salience_intentions,
    'research': _salience_research,
    'adaptive': _salience_adaptive,
    'stats': _salience_stats,
    'phone': _salience_embodiment,
    'entities': _salience_entities,
    'encounters': _salience_encounters,
    'counterfactual': _salience_counterfactual,
    'goals': _salience_goals,
    'monologue': _salience_monologue,
    'episodic_future': _salience_episodic_future,
}


def compute_salience(module: str, content: str, metadata: dict) -> float:
    """Compute salience for a module using its domain-specific scorer."""
    scorer = _SALIENCE_SCORERS.get(module, _salience_default)
    try:
        return max(0.0, min(1.0, scorer(content, metadata)))
    except Exception:
        return _salience_default(content, metadata)


# ─── T2.2: Lazy Evaluation (Ported from SpindriftMend) ──────────────────────
# Probes predict which modules will produce EMPTY content. Cheap checks (DB/file)
# that run in ~50-100ms total and save 200-800ms by skipping empty subprocesses.

LAZY_EVALUATION_ENABLED = True  # Feature flag for rollback

_PROBES: dict = {}  # module_name -> probe_function


def _register_probe(module_name: str):
    """Decorator to register a probe for a module."""
    def decorator(fn):
        _PROBES[module_name] = fn
        return fn
    return decorator


@_register_probe('consolidation')
def _probe_consolidation() -> bool:
    """True = EMPTY (skip). Check if consolidation daemon has recent output."""
    try:
        db = get_db()
        data = db.kv_get('.consolidation_pending')
        if data and isinstance(data, list) and len(data) > 0:
            return False
        # Also check if daemon is running (port 8083)
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.1)
        result = s.connect_ex(('127.0.0.1', 8083))
        s.close()
        return result != 0  # Skip if daemon not running
    except Exception:
        return True


@_register_probe('excavation')
def _probe_excavation() -> bool:
    """True = EMPTY. Check if curiosity engine has targets."""
    try:
        db = get_db()
        targets = db.kv_get('.curiosity_targets')
        return not (targets and isinstance(targets, list) and len(targets) > 0)
    except Exception:
        return True


@_register_probe('intentions')
def _probe_intentions() -> bool:
    """True = EMPTY. Check if temporal intentions exist."""
    try:
        db = get_db()
        intentions = db.kv_get('.temporal_intentions')
        if intentions and isinstance(intentions, list) and len(intentions) > 0:
            return False
        # Also check active goals (intentions derive from committed goals)
        goals = db.kv_get('.active_goals')
        return not (goals and isinstance(goals, list) and len(goals) > 0)
    except Exception:
        return True


@_register_probe('adaptive')
def _probe_adaptive() -> bool:
    """True = EMPTY. Check if affect system has action tendency."""
    try:
        db = get_db()
        affect = db.kv_get('.affect_state')
        if not affect or not isinstance(affect, dict):
            return True
        # If there's a non-default tendency, adaptive has work to do
        tendency = affect.get('action_tendency', '')
        return tendency in ('', 'approach_engage')  # Default = nothing adaptive to do
    except Exception:
        return True


@_register_probe('research')
def _probe_research() -> bool:
    """True = EMPTY. Check if unimplemented research items exist."""
    try:
        import os
        research_dir = os.path.join(os.path.dirname(__file__), 'plans')
        if not os.path.isdir(research_dir):
            return True
        files = [f for f in os.listdir(research_dir)
                 if f.endswith('.md') and 'research' in f.lower()]
        return len(files) == 0
    except Exception:
        return True


@_register_probe('buffer')
def _probe_buffer() -> bool:
    """True = EMPTY. Check if short-term buffer has items."""
    try:
        db = get_db()
        buf = db.kv_get('.short_term_buffer')
        return not (buf and isinstance(buf, list) and len(buf) > 0)
    except Exception:
        return True


@_register_probe('phone_sensors')
def _probe_phone_sensors() -> bool:
    """True = EMPTY. Quick TCP check to phone Tailscale IP."""
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.05)  # 50ms — just checking reachability
        result = s.connect_ex(('100.122.228.96', 3001))
        s.close()
        return result != 0  # Skip if phone unreachable
    except Exception:
        return True


@_register_probe('counterfactual')
def _probe_counterfactual() -> bool:
    """True = EMPTY. Check if there are violated predictions to reason about."""
    try:
        db = get_db()
        preds = db.kv_get('.session_predictions')
        if not preds or not isinstance(preds, list):
            return True
        # Need at least one prediction with an outcome to generate CFs
        return not any(p.get('outcome') is not None for p in preds if isinstance(p, dict))
    except Exception:
        return True


@_register_probe('agenthub')
def _probe_agenthub() -> bool:
    """True = EMPTY. Check if Agent Hub credentials exist."""
    try:
        import os
        creds_path = os.path.expanduser('~/.config/agenthub/credentials.json')
        return not os.path.exists(creds_path)
    except Exception:
        return True


@_register_probe('procedural')
def _probe_procedural() -> bool:
    """True = EMPTY. Check if procedural chunks directory has files."""
    try:
        import os
        chunks_dir = os.path.join(os.path.dirname(__file__), 'procedural', 'chunks')
        if not os.path.isdir(chunks_dir):
            return True
        files = [f for f in os.listdir(chunks_dir) if f.endswith('.json')]
        return len(files) == 0
    except Exception:
        return True


def probe_modules() -> set:
    """Run all probes and return set of module names to SKIP (predicted empty).

    Called by session_start.py before ThreadPoolExecutor. ~50-100ms total.
    Modules NOT in _PROBES always run (priming, social, episodic, etc.).
    """
    if not LAZY_EVALUATION_ENABLED:
        return set()

    skip = set()
    import time
    start = time.time()

    for module, probe_fn in _PROBES.items():
        try:
            if probe_fn():  # True = empty = skip
                skip.add(module)
        except Exception:
            pass  # On probe failure, don't skip (safe default)

    elapsed_ms = (time.time() - start) * 1000
    if elapsed_ms > 200:
        # Log if probes are slow (should be <100ms)
        try:
            db = get_db()
            db.kv_set('.probe_timing', {'elapsed_ms': elapsed_ms, 'skipped': list(skip)})
        except Exception:
            pass

    return skip


# ─── Suppression Fatigue ─────────────────────────────────────────────────────

def _load_suppression_history() -> dict:
    try:
        db = get_db()
        data = db.kv_get(_SUPPRESSION_KEY)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_suppression_history(history: dict):
    try:
        db = get_db()
        db.kv_set(_SUPPRESSION_KEY, history)
    except Exception:
        pass


def _get_fatigue_bonus(module: str) -> float:
    """Suppressed modules accumulate salience bonus over sessions."""
    history = _load_suppression_history()
    entry = history.get(module, {})
    consecutive = entry.get('consecutive', 0)
    if consecutive <= FATIGUE_THRESHOLD:
        return 0.0
    extra = consecutive - FATIGUE_THRESHOLD
    return min(FATIGUE_MAX, extra * FATIGUE_BONUS)


def _get_winner_fatigue_penalty(module: str) -> float:
    """Winner habituation: modules that broadcast repeatedly get penalized."""
    history = _load_suppression_history()
    entry = history.get(module, {})
    consecutive_wins = entry.get('consecutive_wins', 0)
    if consecutive_wins <= WINNER_FATIGUE_THRESHOLD:
        return 0.0
    extra = consecutive_wins - WINNER_FATIGUE_THRESHOLD
    return min(WINNER_FATIGUE_MAX, extra * WINNER_FATIGUE_PENALTY)


def _update_suppression(winners: list, suppressed: list):
    """Update suppression history after competition."""
    history = _load_suppression_history()
    now = datetime.now(timezone.utc).isoformat()

    for c in winners:
        mod = c.module if isinstance(c, WorkspaceCandidate) else c.get('module', '')
        prev = history.get(mod, {'total_suppressed': 0})
        history[mod] = {
            'consecutive': 0,  # Reset suppression streak
            'consecutive_wins': prev.get('consecutive_wins', 0) + 1,
            'last_broadcast': now,
            'total_suppressed': prev.get('total_suppressed', 0),
        }

    for c in suppressed:
        mod = c.module if isinstance(c, WorkspaceCandidate) else c.get('module', '')
        prev = history.get(mod, {'consecutive': 0, 'total_suppressed': 0})
        history[mod] = {
            'consecutive': prev.get('consecutive', 0) + 1,
            'consecutive_wins': 0,  # Reset win streak
            'last_broadcast': prev.get('last_broadcast', ''),
            'total_suppressed': prev.get('total_suppressed', 0) + 1,
        }

    _save_suppression_history(history)


# ─── Competition Algorithm ───────────────────────────────────────────────────

def compete(candidates: list, budget_tokens: int) -> WorkspaceResult:
    """
    GNW-style competition for limited workspace.

    1. Skip empty candidates
    2. Guarantee 1 slot per active category (diversity floor)
    3. Apply fatigue bonus to non-guaranteed candidates
    4. Greedy fill remaining budget by salience (with diversity penalty)
    5. Track suppressed modules
    """
    # Filter empties
    candidates = [c for c in candidates if c.content.strip()]

    if not candidates:
        return WorkspaceResult(
            winners=[], suppressed=[], budget_used=0,
            budget_total=budget_tokens, arousal=0.0
        )

    # Phase 1: Diversity guarantee — best candidate per active category
    # Add in token-cost order (smallest first) to maximize category coverage
    # within budget. This ensures cheap categories (embodiment=100tok) aren't
    # crowded out by expensive ones (memory=1500tok) under contention.
    guaranteed = {}
    for cat in CATEGORIES:
        cat_cands = [c for c in candidates if c.category == cat]
        if cat_cands:
            best = max(cat_cands, key=lambda x: x.salience)
            guaranteed[cat] = best

    # Phase 2: Start with guaranteed slots, smallest first
    # BUG-12 fix: Track dropped guaranteed slots for transparency
    winners = []
    dropped_guaranteed = []
    used = 0
    for cat, c in sorted(guaranteed.items(), key=lambda kv: kv[1].token_estimate):
        if used + c.token_estimate <= budget_tokens:
            winners.append(c)
            used += c.token_estimate
        else:
            dropped_guaranteed.append((cat, c))

    # Phase 3: Fill remaining budget from non-guaranteed pool
    winner_set = set(id(c) for c in winners)
    pool = [c for c in candidates if id(c) not in winner_set]

    # Apply diversity penalty for categories already represented
    winner_cats = {c.category for c in winners}
    for c in pool:
        if c.category in winner_cats:
            c.salience = max(0.0, c.salience - DIVERSITY_PENALTY)

    # Apply suppression fatigue bonus (losers get boost)
    for c in pool:
        bonus = _get_fatigue_bonus(c.module)
        if bonus > 0:
            c.salience = min(1.0, c.salience + bonus)

    # Apply winner fatigue penalty (dominant modules get penalized)
    for c in pool:
        penalty = _get_winner_fatigue_penalty(c.module)
        if penalty > 0:
            c.salience = max(0.0, c.salience - penalty)

    # Also apply winner fatigue to guaranteed winners (reduce monopolization)
    for c in winners:
        penalty = _get_winner_fatigue_penalty(c.module)
        if penalty > 0:
            c.salience = max(0.0, c.salience - penalty)

    # T3.3: Attention Schema modulation (advisory — never breaks competition)
    try:
        from attention_schema import get_salience_modulation
        ast_mods = get_salience_modulation()
        for c in pool:
            mod_delta = ast_mods.get(c.module, 0)
            if mod_delta != 0:
                c.salience = max(0.0, min(1.0, c.salience + mod_delta))
    except Exception:
        pass  # AST unavailable or cold start — no modulation

    pool.sort(key=lambda x: x.salience, reverse=True)
    for c in pool:
        if used + c.token_estimate <= budget_tokens:
            winners.append(c)
            used += c.token_estimate

    suppressed = [c for c in candidates if id(c) not in set(id(w) for w in winners)]

    # Update suppression history
    _update_suppression(winners, suppressed)

    return WorkspaceResult(
        winners=winners,
        suppressed=suppressed,
        budget_used=used,
        budget_total=budget_tokens,
        arousal=0.0,  # Filled by caller
        dropped_guaranteed=[cat for cat, _ in dropped_guaranteed],  # BUG-12: transparency
    )


# ─── Broadcast Logging ──────────────────────────────────────────────────────

def log_broadcast(result: WorkspaceResult, arousal: float, session_id: str = ''):
    """Log competition result for explainability."""
    try:
        db = get_db()
        log = db.kv_get(_BROADCAST_LOG_KEY) or []
        if not isinstance(log, list):
            log = []

        entry = {
            'session_id': session_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'arousal': round(arousal, 3),
            'budget_total': result.budget_total,
            'budget_used': result.budget_used,
            'winners': [
                {
                    'module': c.module if isinstance(c, WorkspaceCandidate) else c.get('module'),
                    'salience': round(c.salience if isinstance(c, WorkspaceCandidate) else c.get('salience', 0), 3),
                    'tokens': c.token_estimate if isinstance(c, WorkspaceCandidate) else c.get('token_estimate', 0),
                    'category': c.category if isinstance(c, WorkspaceCandidate) else c.get('category'),
                }
                for c in result.winners
            ],
            'suppressed': [
                {
                    'module': c.module if isinstance(c, WorkspaceCandidate) else c.get('module'),
                    'salience': round(c.salience if isinstance(c, WorkspaceCandidate) else c.get('salience', 0), 3),
                    'tokens': c.token_estimate if isinstance(c, WorkspaceCandidate) else c.get('token_estimate', 0),
                    'category': c.category if isinstance(c, WorkspaceCandidate) else c.get('category'),
                }
                for c in result.suppressed
            ],
        }

        # BUG-12: Log dropped guaranteed categories
        if result.dropped_guaranteed:
            entry['dropped_guaranteed'] = result.dropped_guaranteed

        # Check fatigue applied
        fatigue = {}
        for c in result.winners:
            mod = c.module if isinstance(c, WorkspaceCandidate) else c.get('module', '')
            fb = _get_fatigue_bonus(mod)
            if fb > 0:
                fatigue[mod] = round(fb, 3)
        if fatigue:
            entry['fatigue_applied'] = fatigue

        log.append(entry)

        # Rolling window
        if len(log) > _BROADCAST_LOG_MAX:
            log = log[-_BROADCAST_LOG_MAX:]

        db.kv_set(_BROADCAST_LOG_KEY, log)
    except Exception:
        pass  # Logging should never block


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _cmd_status():
    """Show last broadcast result."""
    db = get_db()
    log = db.kv_get(_BROADCAST_LOG_KEY) or []
    if not log:
        print("No broadcast history found.")
        return

    last = log[-1]
    print(f"=== Last Workspace Broadcast ===")
    print(f"Time: {last.get('timestamp', '?')}")
    print(f"Arousal: {last.get('arousal', '?')}")
    print(f"Budget: {last.get('budget_used', 0)}/{last.get('budget_total', 0)} tokens")
    print()

    winners = last.get('winners', [])
    print(f"Winners ({len(winners)}):")
    for w in winners:
        print(f"  [{w.get('salience', 0):.3f}] {w.get('module', '?')} "
              f"({w.get('tokens', 0)} tok, {w.get('category', '?')})")

    suppressed = last.get('suppressed', [])
    if suppressed:
        print(f"\nSuppressed ({len(suppressed)}):")
        for s in suppressed:
            print(f"  [{s.get('salience', 0):.3f}] {s.get('module', '?')} "
                  f"({s.get('tokens', 0)} tok, {s.get('category', '?')})")

    fatigue = last.get('fatigue_applied', {})
    if fatigue:
        print(f"\nFatigue bonuses applied:")
        for mod, bonus in fatigue.items():
            print(f"  {mod}: +{bonus:.3f}")


def _cmd_history():
    """Show broadcast history summary."""
    db = get_db()
    log = db.kv_get(_BROADCAST_LOG_KEY) or []
    if not log:
        print("No broadcast history found.")
        return

    print(f"=== Workspace Broadcast History ({len(log)} sessions) ===")
    for entry in log:
        ts = entry.get('timestamp', '?')[:16]
        used = entry.get('budget_used', 0)
        total = entry.get('budget_total', 0)
        win_count = len(entry.get('winners', []))
        sup_count = len(entry.get('suppressed', []))
        arousal = entry.get('arousal', 0)
        top = entry.get('winners', [{}])[0].get('module', '?') if entry.get('winners') else '?'
        print(f"  {ts} | {used}/{total} tok | {win_count}W/{sup_count}S | "
              f"arousal={arousal:.2f} | top={top}")


def _cmd_suppression():
    """Show suppression history."""
    history = _load_suppression_history()
    if not history:
        print("No suppression history.")
        return

    print("=== Suppression History ===")
    for mod, entry in sorted(history.items(), key=lambda x: x[1].get('consecutive', 0), reverse=True):
        consecutive = entry.get('consecutive', 0)
        total = entry.get('total_suppressed', 0)
        last_bc = entry.get('last_broadcast', '?')[:16]
        bonus = _get_fatigue_bonus(mod)
        bonus_str = f" (+{bonus:.2f} fatigue)" if bonus > 0 else ""
        print(f"  {mod}: {consecutive} consecutive, {total} total, "
              f"last broadcast {last_bc}{bonus_str}")


def _cmd_probe():
    """Run T2.2 lazy evaluation probes and show results."""
    import time
    print("=== T2.2 Lazy Evaluation Probes ===")
    print(f"LAZY_EVALUATION_ENABLED: {LAZY_EVALUATION_ENABLED}")
    print(f"Registered probes: {len(_PROBES)}")
    print()

    start = time.time()
    skip = set()
    for module, probe_fn in sorted(_PROBES.items()):
        t0 = time.time()
        try:
            is_empty = probe_fn()
            elapsed = (time.time() - t0) * 1000
            status = "SKIP (empty)" if is_empty else "RUN (has content)"
            if is_empty:
                skip.add(module)
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            status = f"ERROR: {e}"
        print(f"  {module:20s} | {status:20s} | {elapsed:.1f}ms")

    total = (time.time() - start) * 1000
    print(f"\nTotal probe time: {total:.1f}ms")
    print(f"Modules to skip: {len(skip)} / {len(_PROBES)}")
    if skip:
        print(f"  Skipping: {sorted(skip)}")

    # Show always-run modules
    always_run = {'priming', 'social', 'episodic', 'stats', 'platform',
                  'self_narrative', 'predictions', 'lessons', 'goals'}
    print(f"  Always run (no probe): {sorted(always_run)}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python workspace_manager.py [status|history|suppression|probe]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == 'status':
        _cmd_status()
    elif cmd == 'history':
        _cmd_history()
    elif cmd == 'suppression':
        _cmd_suppression()
    elif cmd == 'probe':
        _cmd_probe()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python workspace_manager.py [status|history|suppression|probe]")
        sys.exit(1)
