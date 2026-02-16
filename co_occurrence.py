#!/usr/bin/env python3
"""
Co-occurrence System — Edge provenance, pair decay, and co-occurrence logging.

Extracted from memory_manager.py (Phase 5).
Manages the co-occurrence graph: logging pairs from session recalls,
v3 edge provenance with observations/beliefs, and pair decay.

PostgreSQL is the ONLY data store. No file fallbacks.
"""

import json
import math
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

from memory_common import (
    ALL_DIRS,
    parse_memory_file,
    SELF_EVOLUTION_ENABLED, get_agent_name,
)
from db_adapter import get_db, db_to_file_metadata
from decay_evolution import calculate_evolution_decay_multiplier, log_decay_event
import session_state

# Rejection log integration — auto-capture pruned pairs as taste signal
try:
    from rejection_log import log_rejection as _log_taste_rejection
except ImportError:
    def _log_taste_rejection(**kwargs):
        pass

# --- Configuration ---

CO_OCCURRENCE_BOOST = 0.1
SESSION_TIMEOUT_HOURS = 4
PAIR_DECAY_RATE = 0.3  # Was 0.5. Pairs survive ~20h unreinforced (was ~12h).
ACCESS_WEIGHTED_DECAY = True


def _get_adaptive_decay_rate() -> float:
    """Get decay rate from adaptive behavior (R8 wiring). Falls back to PAIR_DECAY_RATE."""
    try:
        from adaptive_behavior import get_adaptation
        rate = get_adaptation('cooccurrence_decay_rate')
        return rate if rate is not None else PAIR_DECAY_RATE
    except Exception:
        return PAIR_DECAY_RATE

# v5.0: Dimensional decay — only full-decay edges in active W-dimensions
# Edges outside the session's context get dramatically reduced decay
# Credit: Multi-graph RFC (Issue #19), joint design with SpindriftMind
INACTIVE_CONTEXT_FACTOR = 0.1  # Non-overlapping edges decay at 0.03 instead of 0.3

# v2.13: Self-evolution flag (now imported from memory_common — no peer-module coupling)

# v3.0: Edge Provenance
OBSERVATION_MAX_AGE_DAYS = 30


def _get_recall_timestamps() -> dict[str, datetime]:
    """
    Query session_recalls for the current session to get per-recall timestamps.
    Returns {memory_id: recalled_at} for computing temporal direction.
    """
    try:
        db = get_db()
        session_id = session_state.get_db_session_id()
        if not session_id:
            return {}
        import psycopg2.extras
        with db._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT memory_id, recalled_at
                    FROM {db._table('session_recalls')}
                    WHERE session_id = %s
                    ORDER BY recalled_at ASC
                """, (session_id,))
                return {row['memory_id']: row['recalled_at'] for row in cur.fetchall()}
    except Exception:
        return {}


def _compute_direction_weight(id1: str, id2: str, recall_times: dict) -> float:
    """
    Compute direction_weight for a pair based on recall order.
    +1.0 = id1 recalled before id2 (id1 -> id2 temporal flow)
    -1.0 = id2 recalled before id1
     0.0 = simultaneous or unknown
    """
    t1 = recall_times.get(id1)
    t2 = recall_times.get(id2)
    if t1 is None or t2 is None:
        return 0.0
    if t1 < t2:
        return 1.0
    elif t2 < t1:
        return -1.0
    return 0.0
TRUST_TIERS = {
    'self': 1.0,
    'verified_agent': 0.8,
    'platform': 0.6,
    'unknown': 0.3
}


# --- Co-occurrence logging ---


def end_session_cooccurrence() -> list:
    """
    Bridge function for hook compatibility (subagent_stop, teammate_idle, pre_compact).

    Saves pending co-occurrence data to DB KV for deferred processing at next
    session_start. This APPENDS to existing pending data, so multiple hooks
    can fire before the next session_start without data loss.

    Returns list of saved memory IDs (for len() compatibility with hook callers).
    """
    retrieved = session_state.get_retrieved_list()
    if not retrieved:
        return []

    count = save_pending_cooccurrence()
    return retrieved if count > 0 else []


def log_co_occurrences() -> int:
    """
    Log co-occurrences between all memories retrieved this session.
    Call at session end to strengthen links between co-retrieved memories.
    Returns number of pairs updated.
    """
    retrieved = session_state.get_retrieved_list()
    if len(retrieved) < 2:
        return 0

    pairs_updated = 0
    db = get_db()
    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor() as cur:
            for i, id1 in enumerate(retrieved):
                for id2 in retrieved[i + 1:]:
                    for memory_id, other_id in [(id1, id2), (id2, id1)]:
                        cur.execute(f"""
                            INSERT INTO {db._table('co_occurrences')} (memory_id, other_id, count)
                            VALUES (%s, %s, 1)
                            ON CONFLICT (memory_id, other_id)
                            DO UPDATE SET count = {db._table('co_occurrences')}.count + 1
                        """, (memory_id, other_id))
                    pairs_updated += 1

    # Fire cognitive state events for new edges
    if pairs_updated > 0:
        try:
            from cognitive_state import process_event
            process_event('cooccurrence_formed')
        except Exception:
            pass

    print(f"Logged {pairs_updated} co-occurrence pairs from {len(retrieved)} memories")
    return pairs_updated


def log_co_occurrences_v3() -> tuple[int, str]:
    """
    Log co-occurrences to edges_v3 DB table with activity context.
    Layer 2.1 integration - tags edges with WHY they formed.

    Returns: (pairs_updated, session_activity)
    """
    retrieved = session_state.get_retrieved_list()
    if len(retrieved) < 2:
        return 0, None

    session_activity = None
    try:
        from activity_context import get_session_activity
        activity_data = get_session_activity()
        session_activity = activity_data.get('dominant') if activity_data else None
    except Exception:
        pass

    session_platforms = []
    try:
        from platform_context import get_session_platforms
        session_platforms = get_session_platforms()
    except Exception:
        pass

    db = get_db()
    session_id = datetime.now(timezone.utc).isoformat()
    pairs_updated = 0

    # R10: Get recall timestamps for temporal direction
    recall_times = _get_recall_timestamps()

    for i, id1 in enumerate(retrieved):
        for id2 in retrieved[i + 1:]:
            pair = tuple(sorted([id1, id2]))

            # Fetch existing edge once per pair
            existing_edge = db.get_edge(pair[0], pair[1])

            # Build context dicts for the edge, merging with existing
            platform_context = existing_edge.get('platform_context', {}) if existing_edge else {}
            for plat in session_platforms:
                platform_context[plat] = platform_context.get(plat, 0) + 1

            activity_context = existing_edge.get('activity_context', {}) if existing_edge else {}
            if session_activity:
                activity_context[session_activity] = activity_context.get(session_activity, 0) + 1

            # Compute belief: for new edges start at 1.0; for existing, increment
            belief = (existing_edge.get('belief', 0.0) if existing_edge else 0.0) + 1.0

            db.upsert_edge(
                pair[0], pair[1],
                belief=belief,
                platform_context=platform_context,
                activity_context=activity_context,
                topic_context=existing_edge.get('topic_context', {}) if existing_edge else {},
            )

            # R10: Compute direction_weight from recall order
            dw = _compute_direction_weight(id1, id2, recall_times)

            db.add_observation(
                pair[0], pair[1],
                source_type='session_recall',
                session_id=session_id,
                agent=get_agent_name(),
                platform=','.join(session_platforms) if session_platforms else None,
                activity=session_activity,
                direction_weight=dw,
            )

            pairs_updated += 1

    return pairs_updated, session_activity


# --- Deferred co-occurrence (v2.16) ---

def save_pending_cooccurrence() -> int:
    """
    v2.16: Fast session end - save retrieved IDs to DB KV for deferred processing.
    The expensive co-occurrence calculation happens at NEXT session start.

    v2.17: APPENDS to existing pending data instead of overwriting.
    Multiple stop hooks can fire before a single start hook processes them.
    v3.0: DB KV store instead of file.
    """
    retrieved = session_state.get_retrieved_list()
    if not retrieved:
        print("No memories to save to pending.")
        return 0

    # R10: Capture recall timestamps for deferred direction_weight computation
    recall_times_raw = _get_recall_timestamps()
    recall_times_iso = {mid: t.isoformat() if hasattr(t, 'isoformat') else str(t)
                        for mid, t in recall_times_raw.items()}

    new_entry = {
        'retrieved': retrieved,
        'session_id': datetime.now(timezone.utc).isoformat(),
        'agent': get_agent_name(),
        'saved_at': datetime.now(timezone.utc).isoformat(),
        'recall_times': recall_times_iso,
    }

    # Read existing pending data from DB KV and append
    db = get_db()
    existing_sessions = []
    existing_raw = db.kv_get('.pending_cooccurrence')
    if existing_raw:
        existing = json.loads(existing_raw) if isinstance(existing_raw, str) else existing_raw
        if 'sessions' in existing:
            existing_sessions = existing['sessions']
        elif 'retrieved' in existing:
            existing_sessions = [existing]

    existing_sessions.append(new_entry)
    db.kv_set('.pending_cooccurrence', {'sessions': existing_sessions})

    print(f"Saved {len(retrieved)} memories to pending co-occurrence ({len(existing_sessions)} session(s) queued).")
    return len(retrieved)


def process_pending_cooccurrence() -> int:
    """
    v2.16: Process pending co-occurrence data from previous session(s).
    Call this at session START (when there's time).

    v2.17: Handles multiple queued sessions.
    v3.0: Reads from DB KV store instead of file.
    """
    db = get_db()

    raw = db.kv_get('.pending_cooccurrence')
    if not raw:
        return 0

    pending = json.loads(raw) if isinstance(raw, str) else raw

    # Handle both formats
    if 'sessions' in pending:
        sessions = pending['sessions']
    elif 'retrieved' in pending:
        sessions = [pending]
    else:
        db.kv_set('.pending_cooccurrence', None)
        return 0

    total_pairs = 0
    total_memories = 0

    for session_data in sessions:
        retrieved = session_data.get('retrieved', [])
        if len(retrieved) < 2:
            continue

        total_memories += len(retrieved)
        session_id = session_data.get('session_id', '')
        agent = session_data.get('agent', get_agent_name())
        print(f"Processing {len(retrieved)} pending memories from session {session_id[:19]}...")

        # R10: Reconstruct recall_times dict from saved ISO strings
        recall_times_iso = session_data.get('recall_times', {})
        recall_times = {}
        for mid, ts in recall_times_iso.items():
            try:
                recall_times[mid] = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                pass

        pairs_updated = 0
        for i, id1 in enumerate(retrieved):
            for id2 in retrieved[i + 1:]:
                existing = db.get_edge(id1, id2)
                belief = (existing.get('belief', 0.0) if existing else 0.0) + 1.0
                db.upsert_edge(
                    id1, id2,
                    belief=belief,
                    platform_context=existing.get('platform_context', {}) if existing else {},
                    activity_context=existing.get('activity_context', {}) if existing else {},
                    topic_context=existing.get('topic_context', {}) if existing else {},
                )
                # R10: Compute direction_weight from saved timestamps
                dw = _compute_direction_weight(id1, id2, recall_times)
                db.add_observation(
                    id1, id2,
                    source_type='deferred_recall',
                    session_id=session_id,
                    agent=agent,
                    direction_weight=dw,
                )
                pairs_updated += 1

        total_pairs += pairs_updated

    # Clear pending data
    db.kv_set('.pending_cooccurrence', None)

    # Auto-detect curiosity conversions: check if any processed memory went from 0 edges to >0
    try:
        from curiosity_engine import log_curiosity_conversion, _load_curiosity_log
        curiosity_log = _load_curiosity_log(db)
        recent_targets = set()
        for entry in curiosity_log.get('surfaced', [])[-5:]:
            for t in entry.get('targets', []):
                recent_targets.add(t.get('id', '') if isinstance(t, dict) else t)
        all_processed = set()
        for session_data in sessions:
            all_processed.update(session_data.get('retrieved', []))
        converted = recent_targets & all_processed
        for mem_id in converted:
            neighbors = db.get_neighbors(mem_id)
            if neighbors:
                log_curiosity_conversion(mem_id, len(neighbors))
                print(f"  Curiosity conversion: {mem_id} -> {len(neighbors)} edges")
    except Exception:
        pass  # Non-critical, don't break the main pipeline

    print(f"Processed co-occurrences: {len(sessions)} session(s), {total_memories} memories, {total_pairs} pairs updated")
    return total_pairs


# --- V3.0 Edge Provenance System (credit: SpindriftMend PR #5) ---

def _create_observation(
    source_type: str,
    weight: float = 1.0,
    trust_tier: str = 'self',
    session_id: Optional[str] = None,
    agent: str = None,
    platform: Optional[str] = None,
    artifact_id: Optional[str] = None,
    activity: Optional[str] = None
) -> dict:
    """Create a new observation record."""
    if agent is None:
        agent = get_agent_name()
    obs = {
        'id': str(uuid.uuid4()),
        'observed_at': datetime.now(timezone.utc).isoformat(),
        'source': {
            'type': source_type,
            'session_id': session_id,
            'agent': agent,
            'platform': platform,
            'artifact_id': artifact_id
        },
        'weight': weight,
        'trust_tier': trust_tier
    }
    if activity:
        obs['source']['activity'] = activity
    return obs


def aggregate_belief(observations: list[dict], decay_rate: float = 0.1) -> float:
    """
    Compute belief score from observations.
    Applies trust tier weighting, time decay, and diminishing returns.
    """
    if not observations:
        return 0.0

    now = datetime.now(timezone.utc)
    total = 0.0
    source_counts = Counter()

    for obs in observations:
        try:
            obs_time = datetime.fromisoformat(obs['observed_at'].replace('Z', '+00:00'))
            if obs_time.tzinfo is None:
                obs_time = obs_time.replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            obs_time = now

        age_days = (now - obs_time).total_seconds() / 86400

        trust_tier = obs.get('trust_tier', 'unknown')
        trust_mult = TRUST_TIERS.get(trust_tier, 0.3)

        time_mult = max(0.1, 1.0 - (age_days / OBSERVATION_MAX_AGE_DAYS) * decay_rate)

        source_key = (
            obs.get('source', {}).get('type', 'unknown'),
            obs.get('source', {}).get('agent', 'unknown')
        )
        source_counts[source_key] += 1
        source_mult = 1.0 if source_counts[source_key] <= 3 else 1.0 / math.sqrt(source_counts[source_key] - 2)

        base_weight = obs.get('weight', 1.0)

        contribution = base_weight * trust_mult * time_mult * source_mult
        total += contribution

    return round(total, 3)


def add_observation(
    id1: str,
    id2: str,
    source_type: str = 'session_recall',
    weight: float = 1.0,
    trust_tier: str = 'self',
    **source_kwargs
) -> dict:
    """
    Add an observation to an edge and recompute belief.
    Creates edge if it doesn't exist.
    """
    db = get_db()
    pair = tuple(sorted([id1, id2]))

    # Get existing edge or create new
    existing = db.get_edge(pair[0], pair[1])
    new_belief = (existing.get('belief', 0.0) if existing else 0.0) + weight

    db.upsert_edge(
        pair[0], pair[1],
        belief=new_belief,
        platform_context=existing.get('platform_context', {}) if existing else {},
        activity_context=existing.get('activity_context', {}) if existing else {},
        topic_context=existing.get('topic_context', {}) if existing else {},
    )
    db.add_observation(
        pair[0], pair[1],
        source_type=source_type,
        weight=weight,
        trust_tier=trust_tier,
        **source_kwargs,
    )

    # Recompute belief from all observations (trust-weighted, time-decayed)
    # This was the original design — lost during DB migration
    try:
        import psycopg2.extras
        with db._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT observed_at, source_type, weight, trust_tier, platform, agent
                    FROM {db._table('edge_observations')}
                    WHERE edge_id1 = %s AND edge_id2 = %s
                    ORDER BY observed_at DESC
                """, (pair[0], pair[1]))
                rows = cur.fetchall()
        if rows:
            obs_list = [
                {
                    'observed_at': r['observed_at'].isoformat() if r.get('observed_at') else datetime.now(timezone.utc).isoformat(),
                    'weight': r.get('weight', 1.0),
                    'trust_tier': r.get('trust_tier', 'unknown'),
                    'source': {'type': r.get('source_type', 'unknown'), 'agent': r.get('agent', 'unknown')},
                }
                for r in rows
            ]
            aggregated = aggregate_belief(obs_list)
            db.upsert_edge(
                pair[0], pair[1],
                belief=aggregated,
                platform_context=existing.get('platform_context', {}) if existing else {},
                activity_context=existing.get('activity_context', {}) if existing else {},
                topic_context=existing.get('topic_context', {}) if existing else {},
            )
    except Exception:
        pass  # Fall back to simple counter if aggregation fails

    # Return edge data in the format callers expect
    updated = db.get_edge(pair[0], pair[1])
    return updated if updated else {'belief': new_belief, 'last_updated': datetime.now(timezone.utc).isoformat()}


def migrate_to_v3():
    """
    Migrate legacy in-file co-occurrences to v3.0 edges format (DB).
    One-time migration utility. Reads from files, writes to DB.
    """
    db = get_db()

    # Check if edges already exist in DB
    stats = db.edge_stats()
    if stats.get('total_edges', 0) > 0:
        print(f"DB already has {stats['total_edges']} edges. Skipping migration.")
        return

    migration_time = datetime.now(timezone.utc).isoformat()
    migrated_pairs = set()
    edges_migrated = 0

    for directory in ALL_DIRS:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, _ = parse_memory_file(filepath)
            mem_id = metadata.get('id')
            if not mem_id:
                continue

            co_occurrences = metadata.get('co_occurrences', {})
            for other_id, count in co_occurrences.items():
                pair = tuple(sorted([mem_id, other_id]))
                if pair in migrated_pairs:
                    continue

                db.upsert_edge(
                    pair[0], pair[1],
                    belief=float(count),
                )
                db.add_observation(
                    pair[0], pair[1],
                    source_type='legacy_migration',
                    agent=get_agent_name(),
                    weight=float(count),
                    trust_tier='self',
                )
                migrated_pairs.add(pair)
                edges_migrated += 1

    if edges_migrated:
        print(f"Migrated {edges_migrated} edges to DB.")
    else:
        print("No co-occurrences found to migrate.")


# --- Pair decay ---

def _get_recall_count(memory_id: str) -> int:
    """Get the recall_count for a memory by ID. Returns 0 if not found."""
    row = get_db().get_memory(memory_id)
    return row.get('recall_count', 0) if row else 0


def _get_memory_metadata(memory_id: str) -> Optional[dict]:
    """Get metadata for a memory by ID."""
    row = get_db().get_memory(memory_id)
    if row:
        meta, _ = db_to_file_metadata(row)
        return meta
    return None


def _calculate_effective_decay(memory_id: str, other_id: str) -> float:
    """
    Calculate effective decay rate based on access counts and retrieval success.
    Credit: FadeMem paper (access frequency), MemRL/MemEvolve (self-evolution)
    """
    adaptive_rate = _get_adaptive_decay_rate()
    base_decay = adaptive_rate

    if ACCESS_WEIGHTED_DECAY:
        recall_1 = _get_recall_count(memory_id)
        recall_2 = _get_recall_count(other_id)
        avg_recall = (recall_1 + recall_2) / 2
        base_decay = adaptive_rate / (1 + math.log(1 + avg_recall))

    if SELF_EVOLUTION_ENABLED:
        meta1 = _get_memory_metadata(memory_id)
        meta2 = _get_memory_metadata(other_id)

        mult1 = calculate_evolution_decay_multiplier(meta1) if meta1 else 1.0
        mult2 = calculate_evolution_decay_multiplier(meta2) if meta2 else 1.0
        evolution_mult = (mult1 + mult2) / 2

        base_decay *= evolution_mult

    return base_decay


def decay_pair_cooccurrences() -> tuple[int, int]:
    """
    Apply soft decay to co-occurrence pairs that weren't reinforced this session.
    Credit: SpindriftMend (PR #2), FadeMem (v2.8 access weighting)

    Uses the co_occurrences DB table (v2 legacy counter pairs).
    """
    retrieved = session_state.get_retrieved_list()
    reinforced_pairs = set()
    for i, id1 in enumerate(retrieved):
        for id2 in retrieved[i + 1:]:
            reinforced_pairs.add(tuple(sorted([id1, id2])))

    db = get_db()
    import psycopg2.extras

    pairs_decayed = 0
    pairs_pruned = 0

    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Load all co-occurrence pairs
            cur.execute(f"SELECT memory_id, other_id, count FROM {db._table('co_occurrences')}")
            all_pairs = cur.fetchall()

        with conn.cursor() as cur:
            for row in all_pairs:
                memory_id = row['memory_id']
                other_id = row['other_id']
                count = row['count']
                pair = tuple(sorted([memory_id, other_id]))

                if pair not in reinforced_pairs:
                    effective_decay = _calculate_effective_decay(memory_id, other_id)
                    new_count = count - effective_decay

                    if new_count <= 0:
                        cur.execute(f"""
                            DELETE FROM {db._table('co_occurrences')}
                            WHERE memory_id = %s AND other_id = %s
                        """, (memory_id, other_id))
                        pairs_pruned += 1
                    else:
                        cur.execute(f"""
                            UPDATE {db._table('co_occurrences')}
                            SET count = %s
                            WHERE memory_id = %s AND other_id = %s
                        """, (new_count, memory_id, other_id))
                        pairs_decayed += 1

    # Each pair has two rows (bidirectional), so divide by 2
    decayed = pairs_decayed // 2
    pruned = pairs_pruned // 2

    log_decay_event(decayed, pruned)

    if pruned > 0:
        try:
            _log_taste_rejection(
                category='memory_decay',
                reason=f'{pruned} co-occurrence pairs pruned — associations faded from disuse',
                target=f'session decay: {decayed} weakened, {pruned} forgotten',
                tags=['auto-decay', 'co-occurrence-prune'],
                source='internal',
            )
        except Exception:
            pass

    print(f"Pair decay: {decayed} decayed, {pruned} pruned")
    return decayed, pruned


def _build_metadata_cache() -> dict[str, dict]:
    """Pre-load all memory metadata for fast lookup."""
    db = get_db()
    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, type, recall_count, emotional_weight, sessions_since_recall,
                       tags, entities, extra_metadata, retrieval_outcomes, retrieval_success_rate,
                       topic_context, contact_context, platform_context, source
                FROM {db._table('memories')}
            """)
            cache = {}
            for row in cur.fetchall():
                meta, _ = db_to_file_metadata(dict(row))
                cache[row['id']] = meta
            return cache


def _calculate_effective_decay_cached(
    memory_id: str,
    other_id: str,
    metadata_cache: dict[str, dict]
) -> float:
    """Calculate effective decay using pre-cached metadata. O(1) lookup."""
    base_decay = PAIR_DECAY_RATE

    if ACCESS_WEIGHTED_DECAY:
        meta1 = metadata_cache.get(memory_id, {})
        meta2 = metadata_cache.get(other_id, {})
        recall_1 = meta1.get('recall_count', 0)
        recall_2 = meta2.get('recall_count', 0)
        avg_recall = (recall_1 + recall_2) / 2
        base_decay = PAIR_DECAY_RATE / (1 + math.log(1 + avg_recall))

    if SELF_EVOLUTION_ENABLED:
        meta1 = metadata_cache.get(memory_id)
        meta2 = metadata_cache.get(other_id)
        mult1 = calculate_evolution_decay_multiplier(meta1) if meta1 else 1.0
        mult2 = calculate_evolution_decay_multiplier(meta2) if meta2 else 1.0
        evolution_mult = (mult1 + mult2) / 2
        base_decay *= evolution_mult

    return base_decay


def _get_edge_dimensions(
    pair_key: tuple[str, str],
    edge_data: dict,
    metadata_cache: dict[str, dict],
) -> dict[str, set]:
    """
    Extract W-dimensions from an edge for dimensional decay.
    Returns {dimension: set_of_values} for WHERE/WHY/WHAT/WHO.
    """
    dims = {}

    # WHERE: platform_context on the edge
    pc = edge_data.get('platform_context', {})
    platforms = {p for p in pc if not p.startswith('_')}
    if platforms:
        dims['where'] = platforms

    # WHY: activity_context on the edge
    ac = edge_data.get('activity_context', {})
    if ac:
        dims['why'] = set(ac.keys())

    # WHAT: topic_context on edge + memory-level topic_context
    tc = edge_data.get('topic_context', {})
    topics = set()
    if isinstance(tc, dict):
        topics.update(tc.get('union', []))
        topics.update(tc.get('shared', []))
    for mid in pair_key:
        topics.update(metadata_cache.get(mid, {}).get('topic_context', []))
    if topics:
        dims['what'] = topics

    # WHO: contact_context on edge + memory-level contact_context
    contacts = set(edge_data.get('contact_context', []))
    for mid in pair_key:
        contacts.update(metadata_cache.get(mid, {}).get('contact_context', []))
    if contacts:
        dims['who'] = contacts

    return dims


def _has_dimension_overlap(
    edge_dims: dict[str, set],
    session_dims: dict[str, list],
) -> bool:
    """Check if any edge dimension overlaps with active session dimensions."""
    for dim, active_values in session_dims.items():
        edge_values = edge_dims.get(dim, set())
        if edge_values & set(active_values):
            return True
    return False


def decay_pair_cooccurrences_v3() -> tuple[int, int]:
    """
    Apply soft decay to edges in edges_v3 DB table.
    v5.0: Dimensional decay — edges outside the session's active context
    get dramatically reduced decay (INACTIVE_CONTEXT_FACTOR).

    Credit: Multi-graph RFC (Issue #19), SpindriftMind joint design
    """
    retrieved = session_state.get_retrieved_list()
    reinforced_pairs = set()
    for i, id1 in enumerate(retrieved):
        for id2 in retrieved[i + 1:]:
            reinforced_pairs.add(tuple(sorted([id1, id2])))

    db = get_db()

    # Load all edges from DB
    raw_edges = db.get_all_edges()
    if not raw_edges:
        print("Pair decay (v3): 0 decayed, 0 pruned (no edges)")
        return 0, 0

    # Convert to tuple-keyed dict for processing
    edges = {}
    for key_str, edge_data in raw_edges.items():
        parts = key_str.split('|')
        if len(parts) == 2:
            edges[tuple(parts)] = edge_data

    metadata_cache = _build_metadata_cache()

    # v5.0: Get active session dimensions for dimensional decay
    session_dims = {}
    try:
        from context_manager import get_session_dimensions
        session_dims = get_session_dimensions()
    except ImportError:
        pass

    edges_decayed = 0
    edges_pruned = 0
    edges_protected = 0  # v5.0: edges that got reduced decay
    protected_pairs = set()  # v5.0: pairs that get reduced decay

    for pair_key, edge_data in edges.items():
        normalized = tuple(sorted(pair_key))

        if normalized not in reinforced_pairs:
            # v5.0: Dimensional decay — check if edge is outside session context
            if session_dims:
                edge_dims = _get_edge_dimensions(pair_key, edge_data, metadata_cache)
                if edge_dims and not _has_dimension_overlap(edge_dims, session_dims):
                    protected_pairs.add(normalized)
                    edges_protected += 1

            old_belief = edge_data.get('belief', 1.0)
            effective_rate = PAIR_DECAY_RATE * INACTIVE_CONTEXT_FACTOR if normalized in protected_pairs else PAIR_DECAY_RATE
            new_belief = old_belief * (1 - effective_rate)

            if new_belief <= 0.01:
                edges_pruned += 1
            else:
                edges_decayed += 1

    # Pass 1: Full-rate batch decay, excluding reinforced AND protected edges
    adaptive_rate = _get_adaptive_decay_rate()
    all_excluded = [list(p) for p in reinforced_pairs | protected_pairs]
    db.batch_decay_edges(adaptive_rate, exclude_pairs=all_excluded if all_excluded else None)

    # Pass 2: Reduced-rate decay for dimensionally protected edges
    if protected_pairs:
        reduced_rate = adaptive_rate * INACTIVE_CONTEXT_FACTOR
        for pair in protected_pairs:
            key_str = f"{pair[0]}|{pair[1]}"
            edge_data = raw_edges.get(key_str, {})
            old_belief = edge_data.get('belief', 1.0)
            new_belief = old_belief * (1 - reduced_rate)
            if new_belief > 0.01:
                db.upsert_edge(
                    pair[0], pair[1], belief=new_belief,
                    platform_context=edge_data.get('platform_context'),
                    activity_context=edge_data.get('activity_context'),
                    topic_context=edge_data.get('topic_context'),
                )

    db.prune_weak_edges(threshold=0.01)

    log_decay_event(edges_decayed, edges_pruned)

    if edges_pruned > 0:
        try:
            _log_taste_rejection(
                category='memory_decay',
                reason=f'{edges_pruned} co-occurrence edges pruned — associations faded from disuse',
                target=f'session decay: {edges_decayed} weakened, {edges_pruned} forgotten',
                tags=['auto-decay', 'co-occurrence-prune', 'v3'],
                source='internal',
            )
        except Exception:
            pass

    protected_msg = f", {edges_protected} protected" if edges_protected else ""
    print(f"Pair decay (v3): {edges_decayed} decayed, {edges_pruned} pruned{protected_msg}")
    return edges_decayed, edges_pruned
