#!/usr/bin/env python3
"""
Decay Evolution — Memory lifecycle operations.

Extracted from memory_manager.py (Phase 4).
Handles trust tiers, activation scoring, session maintenance,
compression, promotion, retrieval outcomes, and adaptive decay.

DB-ONLY: All memory reads/writes go through PostgreSQL via db_adapter.
No file-based fallbacks. If DB is down, we crash loud.
"""

import json
import math
from datetime import datetime, timezone
from typing import Optional

from memory_common import MEMORY_ROOT
from db_adapter import get_db, db_to_file_metadata
from entity_detection import detect_entities
import session_state

# --- Configuration constants (decay/evolution-specific) ---

DECAY_THRESHOLD_SESSIONS = 7
EMOTIONAL_WEIGHT_THRESHOLD = 0.6
RECALL_COUNT_THRESHOLD = 5
HEAT_PROMOTION_THRESHOLD = 10
HEAT_PROMOTION_ENABLED = True
IMPORTED_PRUNE_SESSIONS = 14
GRACE_PERIOD_SESSIONS = 7  # New memories immune from decay for 7 sessions (~2.3 days)

# Trust-based decay for imported memories (v2.11, credit: SpindriftMend)
DECAY_MULTIPLIERS = {
    'self': 1.0,
    'verified_agent': 1.5,
    'platform': 2.0,
    'unknown': 3.0,
}

# Self-evolution — canonical value lives in memory_common.py (shared config)
from memory_common import SELF_EVOLUTION_ENABLED
SUCCESS_DECAY_BONUS = 0.7
FAILURE_DECAY_PENALTY = 1.5
SUCCESS_THRESHOLD = 0.6
FAILURE_THRESHOLD = 0.3
MIN_RETRIEVALS_FOR_EVOLUTION = 3

# Activation decay — Hebbian time-based (v2.14, credit: SpindriftMend v3.1)
ACTIVATION_DECAY_ENABLED = True
ACTIVATION_HALF_LIFE_HOURS = 24 * 10  # 10 days (was 7). Memories stay primed longer.
ACTIVATION_MIN_FLOOR = 0.01


# --- Trust tier functions ---

def get_memory_trust_tier(metadata: dict) -> str:
    """
    Extract trust tier from memory metadata.

    Priority:
    1. Explicit source.trust_tier field (from import)
    2. Presence of 'imported:AgentName' tag -> verified_agent
    3. Default to 'self' (my own memories)
    """
    source = metadata.get('source', {})
    if isinstance(source, dict):
        tier = source.get('trust_tier')
        if tier and tier in DECAY_MULTIPLIERS:
            return tier

    for tag in metadata.get('tags', []):
        if isinstance(tag, str) and tag.startswith('imported:'):
            return 'verified_agent'

    return 'self'


def get_decay_multiplier(metadata: dict) -> float:
    """Get decay rate multiplier based on trust tier."""
    tier = get_memory_trust_tier(metadata)
    return DECAY_MULTIPLIERS.get(tier, 1.0)


def is_imported_memory(metadata: dict) -> bool:
    """Check if memory was imported from another agent."""
    source = metadata.get('source', {})
    if isinstance(source, dict) and source.get('agent'):
        return True

    for tag in metadata.get('tags', []):
        if isinstance(tag, str) and tag.startswith('imported:'):
            return True

    return False


def list_imported_memories() -> list:
    """List all imported memories with their trust tiers."""
    imported = []
    db = get_db()

    for type_ in ('active', 'archive'):
        rows = db.list_memories(type_=type_, limit=5000)
        for row in rows:
            metadata, content = db_to_file_metadata(row)
            if is_imported_memory(metadata):
                imported.append({
                    'id': metadata.get('id', ''),
                    'trust_tier': get_memory_trust_tier(metadata),
                    'decay_multiplier': get_decay_multiplier(metadata),
                    'recall_count': metadata.get('recall_count', 0),
                    'sessions_since_recall': metadata.get('sessions_since_recall', 0),
                    'source_agent': metadata.get('source', {}).get('agent', 'unknown') if isinstance(metadata.get('source'), dict) else 'unknown',
                    'emotional_weight': metadata.get('emotional_weight', 0.5),
                })
    return imported


# --- Session maintenance ---

def session_maintenance():
    """
    Run at session end/start to:
    1. Increment sessions_since_recall for all active memories
    2. Identify decay candidates (with trust-based decay multipliers for imports)
    3. Prune never-recalled imports after IMPORTED_PRUNE_SESSIONS
    4. Report status

    v2.11: Trust-based decay for imported memories
    v2.12: DB-accelerated path (bulk SQL UPDATE, ~0.2s vs ~13s)
    v3.0: DB-only — no file fallback
    """
    import psycopg2.extras

    db = get_db()

    # Step 1: Bulk increment sessions_since_recall for ALL active memories
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {db._table('memories')}
                SET sessions_since_recall = COALESCE(sessions_since_recall, 0) + 1
                WHERE type = 'active'
            """)
            updated = cur.rowcount

    # Step 2: Fetch all active memories to classify
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, type, tags, recall_count, emotional_weight,
                       sessions_since_recall, source, extra_metadata
                FROM {db._table('memories')}
                WHERE type = 'active'
            """)
            rows = cur.fetchall()

    # Also count core/archive for reporting
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT type, COUNT(*) FROM {db._table('memories')}
                GROUP BY type
            """)
            type_counts = dict(cur.fetchall())

    decay_candidates = []
    reinforced = []
    prune_candidates = []

    for row in rows:
        meta = {
            'id': row['id'],
            'tags': row.get('tags') or [],
            'recall_count': row.get('recall_count', 0),
            'emotional_weight': row.get('emotional_weight', 0.5),
            'sessions_since_recall': row.get('sessions_since_recall', 0),
            'source': row.get('source') or row.get('extra_metadata', {}).get('source', {}),
        }

        sessions = meta['sessions_since_recall']
        decay_multiplier = get_decay_multiplier(meta)
        effective_sessions = sessions * decay_multiplier
        emotional_weight = meta.get('emotional_weight') or 0.5
        recall_count = meta.get('recall_count', 0)

        should_resist_decay = (
            emotional_weight >= EMOTIONAL_WEIGHT_THRESHOLD or
            recall_count >= RECALL_COUNT_THRESHOLD
        )
        in_grace_period = sessions < GRACE_PERIOD_SESSIONS
        is_import = is_imported_memory(meta)

        if is_import and recall_count == 0 and sessions >= IMPORTED_PRUNE_SESSIONS:
            prune_candidates.append(meta)
        elif not in_grace_period and effective_sessions >= DECAY_THRESHOLD_SESSIONS and not should_resist_decay:
            meta['_decay_multiplier'] = decay_multiplier
            decay_candidates.append(meta)
        elif should_resist_decay:
            reinforced.append(meta)

    # Step 3: Report (compact)
    print(f"\n=== Memory Session Maintenance (DB) ===\n")
    print(f"Active memories: {type_counts.get('active', 0)} (incremented {updated})")
    print(f"Core memories: {type_counts.get('core', 0)}")
    print(f"Archived memories: {type_counts.get('archive', 0)}")

    if decay_candidates:
        print(f"\nDecay candidates: {len(decay_candidates)}")
        for meta in decay_candidates[:5]:
            m = meta.get('_decay_multiplier', 1.0)
            eff = meta['sessions_since_recall'] * m
            print(f"  - {meta['id']}: {meta['sessions_since_recall']} sessions (eff: {eff:.1f}), weight={meta.get('emotional_weight') or 0.5:.2f}")
        if len(decay_candidates) > 5:
            print(f"  ... and {len(decay_candidates) - 5} more")

    if prune_candidates:
        print(f"\nPrune candidates: {len(prune_candidates)}")

    if reinforced:
        print(f"\nReinforced: {len(reinforced)}")
        for meta in reinforced[:3]:
            print(f"  - {meta['id']}: recalls={meta.get('recall_count')}, weight={meta.get('emotional_weight') or 0.5:.2f}")

    return decay_candidates, prune_candidates


# --- Compression and promotion ---

def compress_memory(memory_id: str, compressed_content: str):
    """
    Compress a memory - move to archive with reduced content.
    The original content is lost but can be referenced.
    """
    db = get_db()
    row = db.get_memory(memory_id)
    if not row:
        print(f"Memory {memory_id} not found in DB")
        return None

    metadata, original_content = db_to_file_metadata(row)

    db.update_memory(
        memory_id,
        type='archive',
        content=compressed_content,
        extra_metadata={
            **(row.get('extra_metadata') or {}),
            'compressed_at': datetime.now(timezone.utc).isoformat(),
            'original_length': len(original_content),
        }
    )

    print(f"Compressed: {memory_id} -> archive")
    return memory_id


def promote_hot_memories() -> list[str]:
    """
    Promote frequently-accessed memories from active to core.
    Called at session-end to elevate important memories.

    Credit: memU heat-based promotion pattern (v2.9)
    """
    if not HEAT_PROMOTION_ENABLED:
        return []

    db = get_db()
    rows = db.list_memories(type_='active', limit=5000)

    promoted = []
    for row in rows:
        metadata, content = db_to_file_metadata(row)
        recall_count = metadata.get('recall_count', 0)

        if recall_count >= HEAT_PROMOTION_THRESHOLD:
            memory_id = metadata.get('id', '')

            db.update_memory(
                memory_id,
                type='core',
                extra_metadata={
                    **(row.get('extra_metadata') or {}),
                    'promoted_at': datetime.now(timezone.utc).isoformat(),
                    'promoted_reason': f'recall_count={recall_count} >= {HEAT_PROMOTION_THRESHOLD}',
                }
            )

            promoted.append(memory_id)
            print(f"Promoted to core: {memory_id} (recall_count={recall_count})")

    if promoted:
        print(f"Heat promotion: {len(promoted)} memories promoted to core")
    return promoted


# --- Decay event logging ---

def log_decay_event(decayed: int, pruned: int):
    """Log a decay event for stats tracking. DB-only via KV store."""
    db = get_db()

    # Load existing history from DB KV
    history = {"sessions": []}
    existing = db.kv_get('.decay_history')
    if existing:
        history = json.loads(existing) if isinstance(existing, str) else existing

    history["sessions"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decayed": decayed,
        "pruned": pruned
    })

    history["sessions"] = history["sessions"][-20:]
    db.kv_set('.decay_history', history)


# --- Entity backfill (maintenance) ---

def backfill_entities(dry_run: bool = True) -> dict:
    """
    Backfill entities field for existing memories that don't have it.

    Args:
        dry_run: If True, just report what would be updated

    Returns:
        Stats dict with counts
    """
    stats = {'updated': 0, 'skipped': 0, 'already_has': 0}

    db = get_db()

    for type_ in ('core', 'active'):
        rows = db.list_memories(type_=type_, limit=5000)
        for row in rows:
            metadata, content = db_to_file_metadata(row)
            memory_id = metadata.get('id', '')

            if metadata.get('entities'):
                stats['already_has'] += 1
                continue

            detected = detect_entities(content, metadata.get('tags', []))

            if not detected:
                stats['skipped'] += 1
                continue

            if dry_run:
                print(f"Would update {memory_id}: {detected}")
                stats['updated'] += 1
            else:
                db.update_memory(memory_id, entities=detected)
                stats['updated'] += 1

    return stats


# --- Activation scoring (v2.14) ---

def calculate_activation(metadata: dict) -> float:
    """
    Calculate memory activation score using exponential time decay.

    Inspired by SpindriftMend's Hebbian learning implementation.
    Formula: A(t) = A_0 * e^(-lambda*t)

    Returns:
        Activation score (0.0 to 1.0+, can exceed 1.0 for highly reinforced memories)
    """
    if not ACTIVATION_DECAY_ENABLED:
        return metadata.get('emotional_weight', 0.5)

    emotional_weight = metadata.get('emotional_weight', 0.5)

    recall_count = metadata.get('recall_count', 1)
    recall_bonus = math.log(recall_count + 1) / 5

    base_activation = emotional_weight + recall_bonus

    # Grace period: new memories immune from decay for first N sessions
    sessions_since = metadata.get('sessions_since_recall', 0)
    if sessions_since < GRACE_PERIOD_SESSIONS:
        return max(base_activation, 0.5)  # Keep at least 0.5 during grace period

    last_recalled_str = metadata.get('last_recalled')
    if last_recalled_str:
        try:
            if hasattr(last_recalled_str, 'isoformat'):
                last_recalled_str = last_recalled_str.isoformat()
            last_recalled = datetime.fromisoformat(str(last_recalled_str).replace('Z', '+00:00'))
            if last_recalled.tzinfo is None:
                last_recalled = last_recalled.replace(tzinfo=timezone.utc)
            hours_since_recall = (datetime.now(timezone.utc) - last_recalled).total_seconds() / 3600
        except (ValueError, TypeError):
            hours_since_recall = ACTIVATION_HALF_LIFE_HOURS
    else:
        hours_since_recall = ACTIVATION_HALF_LIFE_HOURS

    lambda_rate = math.log(2) / ACTIVATION_HALF_LIFE_HOURS
    decay_factor = math.exp(-lambda_rate * hours_since_recall)

    activation = base_activation * decay_factor

    min_floor = 0.1 if emotional_weight >= EMOTIONAL_WEIGHT_THRESHOLD else ACTIVATION_MIN_FLOOR

    return max(min_floor, round(activation, 4))


def get_most_activated_memories(limit: int = 10) -> list[tuple[str, float, dict]]:
    """
    Get the most activated memories (highest time-weighted activation).

    Returns:
        List of (memory_id, activation_score, metadata, preview) tuples
    """
    results = []
    db = get_db()

    for type_ in ('core', 'active'):
        rows = db.list_memories(type_=type_, limit=5000)
        for row in rows:
            metadata, content = db_to_file_metadata(row)
            memory_id = metadata.get('id', '')
            activation = calculate_activation(metadata)
            results.append((memory_id, activation, metadata, content[:100]))

    results.sort(key=lambda x: x[1], reverse=True)
    return [(r[0], r[1], r[2], r[3]) for r in results[:limit]]


# --- Retrieval outcomes (v2.13 self-evolution) ---

def log_retrieval_outcome(memory_id: str, outcome: str) -> bool:
    """
    Log the outcome of a memory retrieval for self-evolution.

    Outcomes:
    - "productive": Memory led to another recall or useful work
    - "generative": Memory led to creation of new memory (highest value)
    - "dead_end": Memory was recalled but nothing followed
    """
    valid_outcomes = {"productive", "generative", "dead_end"}
    if outcome not in valid_outcomes:
        print(f"Invalid outcome: {outcome}. Must be one of {valid_outcomes}")
        return False

    db = get_db()
    row = db.get_memory(memory_id)
    if not row:
        return False

    metadata, content = db_to_file_metadata(row)

    outcomes = metadata.get('retrieval_outcomes', {
        'productive': 0,
        'generative': 0,
        'dead_end': 0,
        'total': 0
    })

    outcomes[outcome] = outcomes.get(outcome, 0) + 1
    outcomes['total'] = outcomes.get('total', 0) + 1

    total = outcomes['total']
    retrieval_success_rate = None
    if total > 0:
        successes = outcomes.get('productive', 0) + (outcomes.get('generative', 0) * 2)
        retrieval_success_rate = round(successes / (total + outcomes.get('generative', 0)), 3)

    db.update_memory(
        memory_id,
        retrieval_outcomes=outcomes,
        retrieval_success_rate=retrieval_success_rate,
    )

    return True


def get_retrieval_success_rate(memory_id: str) -> Optional[float]:
    """Get the retrieval success rate for a memory."""
    db = get_db()
    row = db.get_memory(memory_id)
    if not row:
        return None
    metadata, _ = db_to_file_metadata(row)
    return metadata.get('retrieval_success_rate')


def calculate_evolution_decay_multiplier(metadata: dict) -> float:
    """
    Calculate decay multiplier based on retrieval success (self-evolution).

    High success rate -> slower decay (memory is valuable)
    Low success rate -> faster decay (memory is noise)
    Insufficient data -> no adjustment (multiplier = 1.0)
    """
    if not SELF_EVOLUTION_ENABLED:
        return 1.0

    outcomes = metadata.get('retrieval_outcomes', {})
    total = outcomes.get('total', 0)

    if total < MIN_RETRIEVALS_FOR_EVOLUTION:
        return 1.0

    success_rate = metadata.get('retrieval_success_rate', 0.5)

    if success_rate >= SUCCESS_THRESHOLD:
        return SUCCESS_DECAY_BONUS
    elif success_rate <= FAILURE_THRESHOLD:
        return FAILURE_DECAY_PENALTY
    else:
        return 1.0


def auto_log_retrieval_outcomes() -> dict:
    """
    Automatically infer retrieval outcomes from session patterns.
    Call at session-end to update success rates.
    """
    retrieved = session_state.get_retrieved_list()

    if not retrieved:
        return {"productive": 0, "generative": 0, "dead_end": 0}

    results = {"productive": 0, "generative": 0, "dead_end": 0}

    for i, mem_id in enumerate(retrieved[:-1]):
        log_retrieval_outcome(mem_id, "productive")
        results["productive"] += 1

    # Check for generative outcomes: memories created today that were caused_by retrieved memories
    today = datetime.now(timezone.utc).date().isoformat()
    db = get_db()
    rows = db.list_memories(type_='active', limit=5000)
    for row in rows:
        metadata, _ = db_to_file_metadata(row)
        created = metadata.get('created', '')
        if hasattr(created, 'isoformat'):
            created = created.isoformat()
        created = str(created)[:10]
        if created != today:
            continue

        caused_by = metadata.get('caused_by', [])
        for mem_id in retrieved:
            if mem_id in caused_by:
                log_retrieval_outcome(mem_id, "generative")
                results["generative"] += 1

    # Check if last retrieved memory was a dead end
    if retrieved:
        last_mem = retrieved[-1]
        last_row = db.get_memory(last_mem)
        if last_row:
            last_metadata, _ = db_to_file_metadata(last_row)
            outcomes = last_metadata.get('retrieval_outcomes', {})
            if outcomes.get('generative', 0) == 0:
                log_retrieval_outcome(last_mem, "dead_end")
                results["dead_end"] += 1

    return results
