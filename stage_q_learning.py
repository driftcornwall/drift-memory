#!/usr/bin/env python3
"""
Per-Stage Q-Learning for Retrieval Pipeline Optimization.

Extends the memory-level Q-value engine (q_value_engine.py) to STAGE-level
learning. Each retrieval pipeline stage is a multi-armed bandit arm whose
expected utility varies by query type.

Architecture:
    Q[stage][query_type] -> float (0.0 to 1.0)

    During search: StageTracker records score deltas per stage per result.
    At session end: Recalled memories' stage deltas become reward signals.
    Stages with Q < SKIP_THRESHOLD (after MIN_SAMPLES) are auto-skipped.

This is the #1 improvement identified by BOTH Drift and SpindriftMend
in the cognitive architecture review v3 (convergence analysis).

Theory: Multi-armed bandit (UCB1-inspired). Each stage-query_type pair is
an arm. Pulling an arm = running the stage. Reward = did the stage's score
modification help surface a memory that was subsequently recalled?

DB persistence: KV store key '.stage_q_learning'

Usage:
    python stage_q_learning.py status      # Q-table overview
    python stage_q_learning.py stages      # Per-stage Q-values
    python stage_q_learning.py skip-report # Which stages would be skipped
    python stage_q_learning.py history     # Learning history
    python stage_q_learning.py reset       # Reset Q-table
"""

import json
import math
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from db_adapter import get_db

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALPHA = 0.15              # Learning rate (higher than memory Q because faster signal)
SKIP_THRESHOLD = 0.25     # Skip stages below this Q (after sufficient samples)
MIN_SAMPLES = 15          # Minimum samples before allowing skip
DEFAULT_Q = 0.5           # Optimistic initialization
Q_MIN, Q_MAX = 0.0, 1.0
EXPLORATION_BONUS = 0.1   # UCB1-style bonus for under-sampled arms

# DB key
KV_STAGE_Q = '.stage_q_learning'
KV_STAGE_Q_HISTORY = '.stage_q_history'

# All instrumented pipeline stages
TRACKED_STAGES = [
    'somatic_prefilter',
    'entity_injection',
    'mood_congruent',
    'actr_noise',
    'gravity_dampening',
    'hub_dampening',
    'q_rerank',
    'strategy_resolution',
    'resolution_boost',
    'evidence_type',
    'importance_freshness',
    'inhibition_of_return',
    'curiosity_boost',
    'goal_relevance',
    'dimensional_boost',
    'kg_expansion',
    'spreading_activation',
]

QUERY_TYPES = ['who', 'what', 'when', 'where', 'why', 'general']


# ---------------------------------------------------------------------------
# Query Type Classifier
# ---------------------------------------------------------------------------

_WHO_KEYWORDS = {'who', 'contact', 'person', 'agent', 'human', 'friend', 'name'}
_WHEN_KEYWORDS = {'when', 'date', 'time', 'today', 'yesterday', 'recent', 'latest', 'session'}
_WHERE_KEYWORDS = {'where', 'platform', 'location', 'site', 'moltx', 'colony', 'twitter', 'github'}
_WHY_KEYWORDS = {'why', 'because', 'reason', 'cause', 'explain', 'how'}


def classify_query(query: str) -> str:
    """Classify a search query into a type for Q-table lookup."""
    words = set(query.lower().split())
    # Check for @ mentions (always WHO)
    if any(w.startswith('@') for w in words):
        return 'who'
    if words & _WHO_KEYWORDS:
        return 'who'
    if words & _WHEN_KEYWORDS:
        return 'when'
    if words & _WHERE_KEYWORDS:
        return 'where'
    if words & _WHY_KEYWORDS:
        return 'why'
    return 'what'  # Default: topical/conceptual query


# ---------------------------------------------------------------------------
# Stage Tracker (used during search pipeline)
# ---------------------------------------------------------------------------

class StageTracker:
    """
    Lightweight tracker that records per-stage score deltas during a search.

    Usage in pipeline:
        tracker = StageTracker(results)
        tracker.before('mood_congruent')
        # ... stage code ...
        tracker.after('mood_congruent')
    """

    def __init__(self, results: list):
        self.results = results
        self._snapshots = {}  # stage -> {id: score_before}

    def before(self, stage: str):
        """Snapshot scores before a stage runs."""
        self._snapshots[stage] = {r['id']: r['score'] for r in self.results}

    def after(self, stage: str):
        """Compute deltas after a stage runs and store on results."""
        before = self._snapshots.get(stage)
        if before is None:
            return
        for r in self.results:
            old_score = before.get(r['id'])
            if old_score is not None:
                delta = r['score'] - old_score
                if abs(delta) > 0.0001:
                    r.setdefault('stage_deltas', {})[stage] = round(delta, 6)


# ---------------------------------------------------------------------------
# Stage Q-Learner
# ---------------------------------------------------------------------------

class StageQLearner:
    """
    Per-stage Q-value learner using multi-armed bandit approach.

    Each (stage, query_type) pair has a Q-value representing expected utility.
    Updated via: Q <- Q + alpha * (reward - Q)
    """

    def __init__(self):
        self.q_table: dict[str, dict[str, float]] = {}   # stage -> query_type -> Q
        self.counts: dict[str, dict[str, int]] = {}       # stage -> query_type -> n
        self.total_updates: int = 0
        self.last_updated: Optional[str] = None

    def get_q(self, stage: str, query_type: str) -> float:
        """Get Q-value with UCB1 exploration bonus for under-sampled arms."""
        base_q = self.q_table.get(stage, {}).get(query_type, DEFAULT_Q)
        n = self.counts.get(stage, {}).get(query_type, 0)
        if n == 0:
            return DEFAULT_Q + EXPLORATION_BONUS  # Encourage trying new things
        # UCB1-style bonus: explore under-sampled arms
        total = max(1, self.total_updates)
        bonus = EXPLORATION_BONUS * math.sqrt(math.log(total) / n)
        return min(Q_MAX, base_q + bonus)

    def get_raw_q(self, stage: str, query_type: str) -> float:
        """Get raw Q-value without exploration bonus."""
        return self.q_table.get(stage, {}).get(query_type, DEFAULT_Q)

    def should_skip(self, stage: str, query_type: str) -> bool:
        """Whether to skip this stage. Requires MIN_SAMPLES to prevent premature gating."""
        raw_q = self.get_raw_q(stage, query_type)
        n = self.counts.get(stage, {}).get(query_type, 0)
        return raw_q < SKIP_THRESHOLD and n >= MIN_SAMPLES

    def update(self, stage: str, query_type: str, reward: float):
        """Q <- Q + alpha * (reward - Q)"""
        old_q = self.q_table.get(stage, {}).get(query_type, DEFAULT_Q)
        new_q = old_q + ALPHA * (reward - old_q)
        new_q = max(Q_MIN, min(Q_MAX, new_q))

        self.q_table.setdefault(stage, {})[query_type] = new_q
        self.counts.setdefault(stage, {})[query_type] = \
            self.counts.get(stage, {}).get(query_type, 0) + 1
        self.total_updates += 1
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            'q_table': self.q_table,
            'counts': self.counts,
            'total_updates': self.total_updates,
            'last_updated': self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'StageQLearner':
        learner = cls()
        learner.q_table = d.get('q_table', {})
        learner.counts = d.get('counts', {})
        learner.total_updates = d.get('total_updates', 0)
        learner.last_updated = d.get('last_updated')
        return learner


# ---------------------------------------------------------------------------
# Persistence + Singleton
# ---------------------------------------------------------------------------

_learner: Optional[StageQLearner] = None


def get_learner() -> StageQLearner:
    """Get or load the stage Q-learner from DB."""
    global _learner
    if _learner is not None:
        return _learner

    db = get_db()
    raw = db.kv_get(KV_STAGE_Q)
    if raw:
        data = json.loads(raw) if isinstance(raw, str) else raw
        _learner = StageQLearner.from_dict(data)
    else:
        _learner = StageQLearner()

    return _learner


def save_learner():
    """Persist the stage Q-learner to DB."""
    if _learner is None:
        return
    db = get_db()
    db.kv_set(KV_STAGE_Q, _learner.to_dict())


# ---------------------------------------------------------------------------
# Search Integration — Module-level cache for credit assignment
# ---------------------------------------------------------------------------

# Cache: {memory_id: (stage_deltas, query_type)} for current session
_search_metadata: dict[str, tuple[dict, str]] = {}
KV_SEARCH_METADATA = '.stage_q_search_metadata'


def record_search_deltas(results: list, query_type: str):
    """
    Cache stage deltas from a completed search for later credit assignment.
    Called at the end of search_memories().
    Persists to KV store so stop.py (separate process) can read it.
    """
    for r in results:
        deltas = r.get('stage_deltas', {})
        if deltas:
            _search_metadata[r['id']] = (deltas, query_type)

    # Persist to KV for cross-process access (hooks run as separate processes)
    if _search_metadata:
        try:
            db = get_db()
            # Merge with any existing data from earlier hook invocations
            existing = db.kv_get(KV_SEARCH_METADATA)
            if existing:
                existing = json.loads(existing) if isinstance(existing, str) else existing
            else:
                existing = {}
            # Convert tuples to lists for JSON serialization
            for mem_id, (deltas, qt) in _search_metadata.items():
                existing[mem_id] = [deltas, qt]
            db.kv_set(KV_SEARCH_METADATA, existing)
        except Exception:
            pass


def _load_search_metadata_from_kv():
    """Load persisted search metadata from KV store."""
    global _search_metadata
    try:
        db = get_db()
        raw = db.kv_get(KV_SEARCH_METADATA)
        if raw:
            data = json.loads(raw) if isinstance(raw, str) else raw
            for mem_id, (deltas, qt) in data.items():
                if mem_id not in _search_metadata:
                    _search_metadata[mem_id] = (deltas, qt)
    except Exception:
        pass


def reward_from_delta(delta: float) -> float:
    """Convert a score delta into a reward signal [-1, +1]."""
    # Positive delta = stage helped surface this memory = positive reward
    # Scale: typical deltas are 0.01-0.10, so 10x gives reasonable rewards
    if delta > 0:
        return min(1.0, delta * 10)
    elif delta < 0:
        return max(-1.0, delta * 10)
    return 0.0


def session_end_update() -> dict:
    """
    At session end: correlate recalled memories with their stage deltas.
    Updates Q-values for stages that helped/hurt recalled memories.

    Returns summary dict for logging.
    """
    # Load search metadata from KV (persisted by earlier hook invocations)
    _load_search_metadata_from_kv()

    try:
        import session_state
        recalled_ids = set(session_state.get_retrieved_list())
    except Exception:
        return {'error': 'could not load session state'}

    if not recalled_ids or not _search_metadata:
        return {'recalled': len(recalled_ids) if recalled_ids else 0,
                'tracked_searches': len(_search_metadata),
                'updates': 0}

    learner = get_learner()
    updates = 0
    stage_rewards = {}  # stage -> [rewards] for logging

    for mem_id in recalled_ids:
        meta = _search_metadata.get(mem_id)
        if not meta:
            continue
        deltas, query_type = meta

        for stage, delta in deltas.items():
            reward = reward_from_delta(delta)
            if abs(reward) > 0.01:  # Skip negligible updates
                learner.update(stage, query_type, reward)
                stage_rewards.setdefault(stage, []).append(reward)
                updates += 1

    # Also give negative reward to stages that didn't help ANY recalled memory
    # (tracked in searches but never in recalls)
    non_recalled = set(_search_metadata.keys()) - recalled_ids
    for mem_id in non_recalled:
        meta = _search_metadata.get(mem_id)
        if not meta:
            continue
        deltas, query_type = meta
        for stage, delta in deltas.items():
            if delta > 0:
                # Stage boosted a memory that wasn't recalled = wasted boost
                learner.update(stage, query_type, -0.1)
                updates += 1

    if updates > 0:
        save_learner()

        # Log history
        try:
            db = get_db()
            history = db.kv_get(KV_STAGE_Q_HISTORY)
            history = (json.loads(history) if isinstance(history, str) else history) or []
            history.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'updates': updates,
                'recalled_count': len(recalled_ids),
                'tracked_count': len(_search_metadata),
                'stage_summary': {
                    s: {'mean_reward': sum(rs)/len(rs), 'count': len(rs)}
                    for s, rs in stage_rewards.items()
                }
            })
            # Keep last 50 entries
            db.kv_set(KV_STAGE_Q_HISTORY, history[-50:])
        except Exception:
            pass

    # Clear KV metadata for next session
    try:
        db = get_db()
        db.kv_set(KV_SEARCH_METADATA, {})
    except Exception:
        pass

    return {
        'recalled': len(recalled_ids),
        'tracked_searches': len(_search_metadata),
        'updates': updates,
        'stages_updated': len(stage_rewards),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_status():
    """Print Q-table overview."""
    learner = get_learner()
    print(f"\n=== Stage Q-Learning Status ===")
    print(f"Total updates: {learner.total_updates}")
    print(f"Last updated: {learner.last_updated or 'never'}")
    print(f"Tracked stages: {len(TRACKED_STAGES)}")
    print(f"Session search cache: {len(_search_metadata)} memories")

    if learner.q_table:
        print(f"\nStages with learned values: {len(learner.q_table)}")
        # Count how many would be skipped
        skip_count = 0
        for stage in TRACKED_STAGES:
            for qt in QUERY_TYPES:
                if learner.should_skip(stage, qt):
                    skip_count += 1
        print(f"Stage-query pairs that would skip: {skip_count}/{len(TRACKED_STAGES)*len(QUERY_TYPES)}")
    else:
        print("\nNo learned values yet. Q-table will populate from search+recall patterns.")


def _print_stages():
    """Print per-stage Q-values."""
    learner = get_learner()
    print(f"\n{'Stage':<25} | ", end='')
    for qt in QUERY_TYPES:
        print(f"{qt:>8}", end=' ')
    print(f"| {'Skip?':>5}")
    print("-" * 85)

    for stage in TRACKED_STAGES:
        print(f"{stage:<25} | ", end='')
        any_skip = False
        for qt in QUERY_TYPES:
            raw = learner.get_raw_q(stage, qt)
            n = learner.counts.get(stage, {}).get(qt, 0)
            skip = learner.should_skip(stage, qt)
            if skip:
                any_skip = True
            marker = '*' if skip else ' '
            if n == 0:
                print(f"  {'--':>5} ", end=' ')
            else:
                print(f" {raw:>5.3f}{marker}", end=' ')
        print(f"| {'YES' if any_skip else 'no':>5}")


def _print_skip_report():
    """Print which stages would be skipped for each query type."""
    learner = get_learner()
    print(f"\n=== Skip Report (threshold={SKIP_THRESHOLD}, min_samples={MIN_SAMPLES}) ===\n")
    for qt in QUERY_TYPES:
        skipped = [s for s in TRACKED_STAGES if learner.should_skip(s, qt)]
        if skipped:
            print(f"  {qt}: {', '.join(skipped)}")
        else:
            print(f"  {qt}: (none skipped)")


def _print_history():
    """Print learning history."""
    db = get_db()
    history = db.kv_get(KV_STAGE_Q_HISTORY)
    history = (json.loads(history) if isinstance(history, str) else history) or []

    if not history:
        print("No learning history yet.")
        return

    print(f"\n=== Stage Q-Learning History (last {len(history)} entries) ===\n")
    for entry in history[-10:]:
        ts = entry.get('timestamp', '?')[:19]
        updates = entry.get('updates', 0)
        recalled = entry.get('recalled_count', 0)
        tracked = entry.get('tracked_count', 0)
        print(f"  [{ts}] {updates} updates, {recalled} recalled / {tracked} tracked")
        summary = entry.get('stage_summary', {})
        for stage, info in sorted(summary.items(), key=lambda x: x[1]['mean_reward']):
            r = info['mean_reward']
            n = info['count']
            print(f"    {stage}: mean_reward={r:+.3f} ({n} updates)")


def _reset():
    """Reset Q-table."""
    global _learner
    db = get_db()
    db.kv_set(KV_STAGE_Q, {})
    db.kv_set(KV_STAGE_Q_HISTORY, [])
    _learner = None
    print("Stage Q-learning table reset.")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if cmd == 'status':
        _print_status()
    elif cmd == 'stages':
        _print_stages()
    elif cmd in ('skip-report', 'skip'):
        _print_skip_report()
    elif cmd == 'history':
        _print_history()
    elif cmd == 'reset':
        _reset()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python stage_q_learning.py [status|stages|skip-report|history|reset]")
