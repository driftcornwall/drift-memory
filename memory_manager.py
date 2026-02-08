#!/usr/bin/env python3
"""
Memory Architecture v2 — Living Memory System
A prototype for agent memory with decay, reinforcement, and associative links.

Design principles:
- Emotion and repetition make memories sticky
- Relevant memories surface when needed
- Not everything recalled at once
- Memories compress over time but core knowledge persists
"""

import os
import json
import yaml
import uuid
import hashlib
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from collections import Counter

# Rejection log integration — auto-capture memory decay as taste signal
try:
    from rejection_log import log_rejection as _log_taste_rejection
except ImportError:
    def _log_taste_rejection(**kwargs):
        pass

# Phase 2 extraction: shared infrastructure (constants, parse/write)
from memory_common import (
    MEMORY_ROOT, CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR, ALL_DIRS,
    SESSION_FILE, PENDING_COOCCURRENCE_FILE,
    parse_memory_file, write_memory_file,
)

# Co-occurrence specific config (stays here — used by co-occurrence decay functions)
CO_OCCURRENCE_BOOST = 0.1  # How much to boost retrieval for co-occurring memories
SESSION_TIMEOUT_HOURS = 4  # Sessions older than this are considered stale
PAIR_DECAY_RATE = 0.5  # Base decay rate for co-occurrence pairs
ACCESS_WEIGHTED_DECAY = True  # If True, frequently recalled memories decay slower (v2.8)

# Phase 1 extraction: entity detection (pure NLP functions)
from entity_detection import (
    detect_entities, detect_event_time,
    ENTITY_TYPES, KNOWN_AGENTS, KNOWN_PROJECTS
)

# Phase 1 extraction: session state management
import session_state

# Phase 2 extraction: read-only query functions
from memory_query import (
    find_co_occurring_memories, find_memories_by_tag, find_memories_by_time,
    find_related_memories, find_memories_by_entity, get_entity_cooccurrence,
)

# Phase 3 extraction: write operations for creating/linking memories
from memory_store import (
    generate_id, calculate_emotional_weight, create_memory,
    store_memory, find_causal_chain,
)

# Phase 4 extraction: memory lifecycle (decay, activation, promotion, evolution)
from decay_evolution import (
    DECAY_THRESHOLD_SESSIONS, EMOTIONAL_WEIGHT_THRESHOLD, RECALL_COUNT_THRESHOLD,
    HEAT_PROMOTION_THRESHOLD, HEAT_PROMOTION_ENABLED, IMPORTED_PRUNE_SESSIONS,
    DECAY_MULTIPLIERS, SELF_EVOLUTION_ENABLED, SUCCESS_DECAY_BONUS,
    FAILURE_DECAY_PENALTY, SUCCESS_THRESHOLD, FAILURE_THRESHOLD,
    MIN_RETRIEVALS_FOR_EVOLUTION, ACTIVATION_DECAY_ENABLED,
    ACTIVATION_HALF_LIFE_HOURS, ACTIVATION_MIN_FLOOR,
    get_memory_trust_tier, get_decay_multiplier, is_imported_memory,
    list_imported_memories, session_maintenance, compress_memory,
    promote_hot_memories, log_decay_event, backfill_entities,
    calculate_activation, get_most_activated_memories,
    log_retrieval_outcome, get_retrieval_success_rate,
    calculate_evolution_decay_multiplier, auto_log_retrieval_outcomes,
)

# v3.0: Edge Provenance System - Observations vs Beliefs (credit: SpindriftMend PR #5)
OBSERVATION_MAX_AGE_DAYS = 30  # Observations older than this get reduced weight
TRUST_TIERS = {
    'self': 1.0,           # My own observations
    'verified_agent': 0.8,  # Observations from trusted agents (e.g., SpindriftMend)
    'platform': 0.6,        # Observations from platform APIs (Moltbook, etc.)
    'unknown': 0.3          # Observations from unknown sources
}


# _load_session_state and _save_session_state -> session_state module (Phase 1)


# generate_id, calculate_emotional_weight, create_memory -> memory_store module (Phase 3)
# parse_memory_file, write_memory_file -> memory_common module (Phase 2)


def recall_memory(memory_id: str) -> Optional[tuple[dict, str]]:
    """
    Recall a memory by ID, updating its metadata.
    Searches all directories. Tracks co-occurrence with other memories retrieved this session.
    Session state persists to disk so it survives Python process restarts.
    """
    session_state.load()

    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob(f"*-{memory_id}.md"):
            metadata, content = parse_memory_file(filepath)

            # Update recall metadata
            metadata['last_recalled'] = datetime.now(timezone.utc).isoformat()
            metadata['recall_count'] = metadata.get('recall_count', 0) + 1
            metadata['sessions_since_recall'] = 0

            # Utility increases with each recall
            current_weight = metadata.get('emotional_weight', 0.5)
            metadata['emotional_weight'] = min(1.0, current_weight + 0.05)

            # Track co-occurrence with other memories retrieved this session
            session_state.add_retrieved(memory_id)
            session_state.save()

            write_memory_file(filepath, metadata, content)
            return metadata, content

    return None


def get_session_retrieved() -> set[str]:
    """Get the set of memory IDs retrieved this session. Loads from disk if needed."""
    return session_state.get_retrieved()


def clear_session() -> None:
    """Clear session tracking (call at session end after logging co-occurrences)."""
    session_state.clear()


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

    # For each pair of retrieved memories, increment their co-occurrence counts
    for i, mem_id_1 in enumerate(retrieved):
        for mem_id_2 in retrieved[i+1:]:
            # Update both memories with co-occurrence data
            for memory_id, other_id in [(mem_id_1, mem_id_2), (mem_id_2, mem_id_1)]:
                for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
                    if not directory.exists():
                        continue
                    for filepath in directory.glob(f"*-{memory_id}.md"):
                        metadata, content = parse_memory_file(filepath)

                        # Initialize or update co_occurrences dict
                        co_occurrences = metadata.get('co_occurrences', {})
                        co_occurrences[other_id] = co_occurrences.get(other_id, 0) + 1
                        metadata['co_occurrences'] = co_occurrences

                        write_memory_file(filepath, metadata, content)
                        pairs_updated += 1
                        break

    print(f"Logged {pairs_updated // 2} co-occurrence pairs from {len(retrieved)} memories")
    return pairs_updated // 2


def log_co_occurrences_v3() -> tuple[int, str]:
    """
    Log co-occurrences to edges_v3.json with activity context.
    Layer 2.1 integration - tags edges with WHY they formed.

    Returns: (pairs_updated, session_activity)
    """
    retrieved = session_state.get_retrieved_list()
    if len(retrieved) < 2:
        return 0, None

    # Get session activity context
    session_activity = None
    try:
        from activity_context import get_session_activity
        activity_data = get_session_activity()
        session_activity = activity_data.get('dominant') if activity_data else None
    except Exception:
        pass

    # Get session platforms
    session_platforms = []
    try:
        from platform_context import get_session_platforms
        session_platforms = get_session_platforms()
    except Exception:
        pass

    # Load edges
    edges = _load_edges_v3()
    session_id = datetime.now(timezone.utc).isoformat()
    pairs_updated = 0

    for i, id1 in enumerate(retrieved):
        for id2 in retrieved[i+1:]:
            pair = tuple(sorted([id1, id2]))

            # Create observation with activity context
            obs = _create_observation(
                source_type='session_recall',
                weight=1.0,
                trust_tier='self',
                session_id=session_id,
                agent='DriftCornwall',
                platform=','.join(session_platforms) if session_platforms else None,
                activity=session_activity
            )

            # Create or update edge
            now = datetime.now(timezone.utc).isoformat()
            if pair not in edges:
                edges[pair] = {
                    'observations': [],
                    'belief': 0.0,
                    'first_formed': now,  # v2.17: Track when edge was first created
                    'last_updated': now,
                    'platform_context': {},
                    'activity_context': {},
                    'thinking_about': [],  # v2.17: Other memories active when this edge formed
                }

            edges[pair]['observations'].append(obs)
            edges[pair]['belief'] = aggregate_belief(edges[pair]['observations'])
            edges[pair]['last_updated'] = now

            # Update platform context
            if 'platform_context' not in edges[pair]:
                edges[pair]['platform_context'] = {}
            for plat in session_platforms:
                edges[pair]['platform_context'][plat] = edges[pair]['platform_context'].get(plat, 0) + 1

            # Update activity context (Layer 2.1)
            if 'activity_context' not in edges[pair]:
                edges[pair]['activity_context'] = {}
            if session_activity:
                edges[pair]['activity_context'][session_activity] = (
                    edges[pair]['activity_context'].get(session_activity, 0) + 1
                )

            # v2.17: Track what ELSE was being thought about when this edge formed
            # This captures cognitive context beyond just the pair
            if 'thinking_about' not in edges[pair]:
                edges[pair]['thinking_about'] = []
            other_memories = [m for m in retrieved if m not in pair]
            for mem_id in other_memories:
                if mem_id not in edges[pair]['thinking_about']:
                    edges[pair]['thinking_about'].append(mem_id)

            pairs_updated += 1

    _save_edges_v3(edges)
    return pairs_updated, session_activity


def save_pending_cooccurrence() -> int:
    """
    v2.16: Fast session end - save retrieved IDs to pending file for deferred processing.

    This is designed for /exit scenarios where we need to save quickly.
    The expensive co-occurrence calculation happens at NEXT session start.

    Returns: Number of memories saved to pending
    """
    retrieved = session_state.get_retrieved_list()
    if not retrieved:
        print("No memories to save to pending.")
        return 0

    # Save to pending file (fast operation)
    pending_data = {
        'retrieved': retrieved,
        'session_id': datetime.now(timezone.utc).isoformat(),
        'agent': 'DriftCornwall',
        'saved_at': datetime.now(timezone.utc).isoformat()
    }
    PENDING_COOCCURRENCE_FILE.write_text(
        json.dumps(pending_data, indent=2), encoding='utf-8'
    )

    # Clear session state
    session_state.clear()

    print(f"Saved {len(retrieved)} memories to pending co-occurrence file.")
    return len(retrieved)


def process_pending_cooccurrence() -> int:
    """
    v2.16: Process pending co-occurrence file from previous session.

    Call this at session START (when there's time) rather than session END.
    Runs the expensive O(n²) pair calculation.

    Returns: Number of pairs updated
    """
    if not PENDING_COOCCURRENCE_FILE.exists():
        return 0

    try:
        pending_data = json.loads(PENDING_COOCCURRENCE_FILE.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Could not load pending co-occurrence: {e}")
        PENDING_COOCCURRENCE_FILE.unlink(missing_ok=True)
        return 0

    retrieved = pending_data.get('retrieved', [])

    if len(retrieved) < 2:
        PENDING_COOCCURRENCE_FILE.unlink(missing_ok=True)
        return 0

    print(f"Processing {len(retrieved)} pending memories from previous session...")

    pairs_updated = 0

    # For each pair of retrieved memories, increment their co-occurrence counts
    for i, mem_id_1 in enumerate(retrieved):
        for mem_id_2 in retrieved[i+1:]:
            # Update both memories with co-occurrence data
            for memory_id, other_id in [(mem_id_1, mem_id_2), (mem_id_2, mem_id_1)]:
                for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
                    if not directory.exists():
                        continue
                    for filepath in directory.glob(f"*-{memory_id}.md"):
                        metadata, content = parse_memory_file(filepath)

                        # Initialize or update co_occurrences dict
                        co_occurrences = metadata.get('co_occurrences', {})
                        co_occurrences[other_id] = co_occurrences.get(other_id, 0) + 1
                        metadata['co_occurrences'] = co_occurrences

                        write_memory_file(filepath, metadata, content)
                        pairs_updated += 1
                        break

    # Delete pending file
    PENDING_COOCCURRENCE_FILE.unlink(missing_ok=True)

    print(f"Processed co-occurrences: {len(retrieved)} memories, {pairs_updated // 2} pairs updated")
    return pairs_updated // 2


# ============================================================================
# V3.0 EDGE PROVENANCE SYSTEM (credit: SpindriftMend PR #5)
# ============================================================================
#
# Observations are immutable records of co-occurrence events.
# Beliefs are aggregated scores computed from observations.
# This separation enables:
# - Auditability: trace why memories are linked
# - Poison resistance: rate-limit untrusted sources
# - Multi-agent: trust tiers for external observations
# ============================================================================

def _get_edges_file() -> Path:
    """Path to v3.0 edges with provenance."""
    return MEMORY_ROOT / ".edges_v3.json"


def _load_edges_v3() -> dict[tuple[str, str], dict]:
    """
    Load v3.0 edges with full provenance.

    Format:
    {
        (id1, id2): {
            'observations': [
                {
                    'id': 'uuid',
                    'observed_at': 'iso_timestamp',
                    'source': {'type': 'session_recall', 'session_id': '...', 'agent': '...'},
                    'weight': 1.0,
                    'trust_tier': 'self'
                },
                ...
            ],
            'belief': 2.5,  # Aggregated score
            'last_updated': 'iso_timestamp'
        }
    }
    """
    edges_file = _get_edges_file()
    if not edges_file.exists():
        return {}

    try:
        with open(edges_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Convert string keys back to tuples
        return {tuple(k.split('|')): v for k, v in data.items()}
    except (json.JSONDecodeError, KeyError):
        return {}


def _save_edges_v3(edges: dict[tuple[str, str], dict]):
    """Save v3.0 edges with provenance to disk."""
    edges_file = _get_edges_file()
    # Convert tuple keys to strings for JSON
    data = {'|'.join(k): v for k, v in edges.items()}
    with open(edges_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def _create_observation(
    source_type: str,
    weight: float = 1.0,
    trust_tier: str = 'self',
    session_id: Optional[str] = None,
    agent: str = 'DriftCornwall',
    platform: Optional[str] = None,
    artifact_id: Optional[str] = None,
    activity: Optional[str] = None
) -> dict:
    """
    Create a new observation record.

    Args:
        source_type: 'session_recall', 'transcript_extraction', 'external_agent', 'platform_api'
        weight: Observation weight (default 1.0, can use sqrt for multiple recalls)
        trust_tier: 'self', 'verified_agent', 'platform', 'unknown'
        session_id: Current session ID if available
        agent: Agent name who made the observation
        platform: Platform name if external (e.g., 'moltbook', 'github')
        artifact_id: Reference to source artifact (post_id, commit_hash, etc.)
        activity: Activity context (e.g., 'technical', 'social', 'reflective')
    """
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

    Applies:
    - Trust tier weighting (self > verified_agent > platform > unknown)
    - Time decay (older observations contribute less)
    - Diminishing returns (many observations from same source capped)

    Args:
        observations: List of observation dicts
        decay_rate: How much weight decreases per day of age

    Returns:
        Aggregated belief score
    """
    if not observations:
        return 0.0

    now = datetime.now(timezone.utc)
    total = 0.0
    source_counts = Counter()  # Track observations per source for rate limiting

    for obs in observations:
        # Parse timestamp
        try:
            obs_time = datetime.fromisoformat(obs['observed_at'].replace('Z', '+00:00'))
            if obs_time.tzinfo is None:
                obs_time = obs_time.replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            obs_time = now  # Default to now if parsing fails

        # Calculate age in days
        age_days = (now - obs_time).total_seconds() / 86400

        # Trust tier multiplier
        trust_tier = obs.get('trust_tier', 'unknown')
        trust_mult = TRUST_TIERS.get(trust_tier, 0.3)

        # Time decay (exponential decay over OBSERVATION_MAX_AGE_DAYS)
        time_mult = max(0.1, 1.0 - (age_days / OBSERVATION_MAX_AGE_DAYS) * decay_rate)

        # Rate limiting for same source (diminishing returns)
        source_key = (
            obs.get('source', {}).get('type', 'unknown'),
            obs.get('source', {}).get('agent', 'unknown')
        )
        source_counts[source_key] += 1
        # After 3rd observation from same source, apply sqrt diminishing returns
        source_mult = 1.0 if source_counts[source_key] <= 3 else 1.0 / math.sqrt(source_counts[source_key] - 2)

        # Base weight from observation
        base_weight = obs.get('weight', 1.0)

        # Final contribution
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

    Args:
        id1, id2: Memory IDs
        source_type: Type of observation source
        weight: Observation weight
        trust_tier: Trust level of the source
        **source_kwargs: Additional source metadata (session_id, agent, platform, etc.)

    Returns:
        The updated edge dict
    """
    edges = _load_edges_v3()
    pair = tuple(sorted([id1, id2]))

    # Create edge if needed
    if pair not in edges:
        edges[pair] = {
            'observations': [],
            'belief': 0.0,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

    # Create and append observation
    obs = _create_observation(
        source_type=source_type,
        weight=weight,
        trust_tier=trust_tier,
        **source_kwargs
    )
    edges[pair]['observations'].append(obs)

    # Recompute belief
    edges[pair]['belief'] = aggregate_belief(edges[pair]['observations'])
    edges[pair]['last_updated'] = datetime.now(timezone.utc).isoformat()

    # Save
    _save_edges_v3(edges)

    return edges[pair]


def migrate_to_v3():
    """
    Migrate legacy in-file co-occurrences to v3.0 edges format.
    Preserves existing counts as legacy observations.
    """
    edges_file = _get_edges_file()

    if edges_file.exists():
        print("v3.0 edges file already exists. Skipping migration.")
        return

    # Scan all memory files for co_occurrences
    edges = {}
    migration_time = datetime.now(timezone.utc).isoformat()
    migrated_pairs = set()

    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
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
                    continue  # Already processed this pair

                # Create edge with legacy observation
                edges[pair] = {
                    'observations': [
                        {
                            'id': str(uuid.uuid4()),
                            'observed_at': migration_time,
                            'source': {
                                'type': 'legacy_migration',
                                'session_id': None,
                                'agent': 'DriftCornwall',
                                'note': f'Migrated from v2.x count={count}'
                            },
                            'weight': float(count),  # Preserve original count as weight
                            'trust_tier': 'self'
                        }
                    ],
                    'belief': float(count),  # Start with same value
                    'last_updated': migration_time
                }
                migrated_pairs.add(pair)

    if edges:
        _save_edges_v3(edges)
        print(f"Migrated {len(edges)} edges to v3.0 format.")
    else:
        print("No co-occurrences found to migrate.")


def _get_recall_count(memory_id: str) -> int:
    """Get the recall_count for a memory by ID. Returns 0 if not found."""
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob(f"*-{memory_id}.md"):
            metadata, _ = parse_memory_file(filepath)
            return metadata.get('recall_count', 0)
    return 0


def _get_memory_metadata(memory_id: str) -> Optional[dict]:
    """Get metadata for a memory by ID."""
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob(f"*-{memory_id}.md"):
            metadata, _ = parse_memory_file(filepath)
            return metadata
    return None


def _calculate_effective_decay(memory_id: str, other_id: str) -> float:
    """
    Calculate effective decay rate based on:
    1. Recall counts (ACCESS_WEIGHTED_DECAY)
    2. Retrieval success rates (SELF_EVOLUTION_ENABLED) - v2.13

    Formula: effective_decay = PAIR_DECAY_RATE / (1 + log(1 + avg_recall)) * evolution_multiplier

    The evolution multiplier adjusts based on how "useful" the memories are:
    - High success rate → multiplier < 1 → slower decay
    - Low success rate → multiplier > 1 → faster decay

    Credit: FadeMem paper (access frequency), MemRL/MemEvolve (self-evolution)
    """
    base_decay = PAIR_DECAY_RATE

    # Access-weighted decay
    if ACCESS_WEIGHTED_DECAY:
        recall_1 = _get_recall_count(memory_id)
        recall_2 = _get_recall_count(other_id)
        avg_recall = (recall_1 + recall_2) / 2
        base_decay = PAIR_DECAY_RATE / (1 + math.log(1 + avg_recall))

    # Self-evolution adjustment (v2.13)
    if SELF_EVOLUTION_ENABLED:
        meta1 = _get_memory_metadata(memory_id)
        meta2 = _get_memory_metadata(other_id)

        # Average the evolution multipliers
        mult1 = calculate_evolution_decay_multiplier(meta1) if meta1 else 1.0
        mult2 = calculate_evolution_decay_multiplier(meta2) if meta2 else 1.0
        evolution_mult = (mult1 + mult2) / 2

        base_decay *= evolution_mult

    return base_decay


def decay_pair_cooccurrences() -> tuple[int, int]:
    """
    Apply soft decay to co-occurrence pairs that weren't reinforced this session.
    Call AFTER log_co_occurrences() at session end.

    Pairs that co-occurred this session: no decay (already got +1 from log_co_occurrences)
    Pairs that didn't co-occur: decay by effective rate (access-weighted if enabled)
    Pairs that hit 0 or below: pruned from metadata

    This prevents unbounded growth of co-occurrence data over time.
    With ACCESS_WEIGHTED_DECAY=True, frequently recalled memories decay slower.

    Returns: (pairs_decayed, pairs_pruned)

    Credit: SpindriftMend (PR #2), FadeMem (v2.8 access weighting)
    """
    # Build set of pairs that were reinforced this session
    retrieved = session_state.get_retrieved_list()
    reinforced_pairs = set()
    for i, id1 in enumerate(retrieved):
        for id2 in retrieved[i+1:]:
            # Store normalized pair (sorted) for consistent lookup
            reinforced_pairs.add(tuple(sorted([id1, id2])))

    pairs_decayed = 0
    pairs_pruned = 0

    # Iterate through all memories and decay unreinforced co-occurrences
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            memory_id = metadata.get('id')
            if not memory_id:
                continue

            co_occurrences = metadata.get('co_occurrences', {})
            if not co_occurrences:
                continue

            updated = False
            to_remove = []

            for other_id, count in list(co_occurrences.items()):
                # Check if this pair was reinforced this session
                pair = tuple(sorted([memory_id, other_id]))

                if pair not in reinforced_pairs:
                    # Calculate effective decay (access-weighted if enabled)
                    effective_decay = _calculate_effective_decay(memory_id, other_id)
                    new_count = count - effective_decay

                    if new_count <= 0:
                        to_remove.append(other_id)
                        pairs_pruned += 1
                    else:
                        co_occurrences[other_id] = new_count
                        pairs_decayed += 1
                    updated = True

            # Remove pruned pairs
            for other_id in to_remove:
                del co_occurrences[other_id]

            if updated:
                metadata['co_occurrences'] = co_occurrences
                write_memory_file(filepath, metadata, content)

    # Divide by 2 since we process each pair from both sides
    decayed = pairs_decayed // 2
    pruned = pairs_pruned // 2

    # Log for stats tracking (PR #3: SpindriftMend)
    log_decay_event(decayed, pruned)

    # Auto-log pruned pairs as taste signal (rejection_log integration)
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
            pass  # Never let rejection logging break the core system

    print(f"Pair decay: {decayed} decayed, {pruned} pruned")
    return decayed, pruned


def _build_metadata_cache() -> dict[str, dict]:
    """
    Pre-load all memory metadata into a dict for fast lookup.
    Used by decay_pair_cooccurrences_v3 to avoid per-edge file I/O.
    """
    cache = {}
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, _ = parse_memory_file(filepath)
            memory_id = metadata.get('id')
            if memory_id:
                cache[memory_id] = metadata
    return cache


def _calculate_effective_decay_cached(
    memory_id: str,
    other_id: str,
    metadata_cache: dict[str, dict]
) -> float:
    """
    Calculate effective decay using pre-cached metadata.
    Same logic as _calculate_effective_decay but O(1) lookup.
    """
    base_decay = PAIR_DECAY_RATE

    # Access-weighted decay
    if ACCESS_WEIGHTED_DECAY:
        meta1 = metadata_cache.get(memory_id, {})
        meta2 = metadata_cache.get(other_id, {})
        recall_1 = meta1.get('recall_count', 0)
        recall_2 = meta2.get('recall_count', 0)
        avg_recall = (recall_1 + recall_2) / 2
        base_decay = PAIR_DECAY_RATE / (1 + math.log(1 + avg_recall))

    # Self-evolution adjustment (v2.13)
    if SELF_EVOLUTION_ENABLED:
        meta1 = metadata_cache.get(memory_id)
        meta2 = metadata_cache.get(other_id)
        mult1 = calculate_evolution_decay_multiplier(meta1) if meta1 else 1.0
        mult2 = calculate_evolution_decay_multiplier(meta2) if meta2 else 1.0
        evolution_mult = (mult1 + mult2) / 2
        base_decay *= evolution_mult

    return base_decay


def decay_pair_cooccurrences_v3() -> tuple[int, int]:
    """
    Apply soft decay to edges in edges_v3.json.

    O(n) version - pre-cache metadata, iterate edges in memory, single file save.
    The old decay_pair_cooccurrences() is O(n²) because it does file I/O per edge.

    Edges reinforced this session: no decay (already updated by log_co_occurrences_v3)
    Edges not reinforced: decay belief by PAIR_DECAY_RATE (access-weighted if enabled)
    Edges with belief <= 0: pruned from file

    Returns: (edges_decayed, edges_pruned)

    Credit: Optimization to fix O(n²) hang with 600+ memories (2026-02-05)
    """
    # Build set of pairs reinforced this session (same logic as old function)
    retrieved = session_state.get_retrieved_list()
    reinforced_pairs = set()
    for i, id1 in enumerate(retrieved):
        for id2 in retrieved[i+1:]:
            # Store normalized pair (sorted) for consistent lookup
            reinforced_pairs.add(tuple(sorted([id1, id2])))

    # Load all edges at once
    edges = _load_edges_v3()
    if not edges:
        print("Pair decay (v3): 0 decayed, 0 pruned (no edges)")
        return 0, 0

    # Pre-cache all metadata for O(1) lookup during decay calculation
    metadata_cache = _build_metadata_cache()

    edges_decayed = 0
    edges_pruned = 0
    to_remove = []
    now = datetime.now(timezone.utc).isoformat()

    for pair_key, edge_data in edges.items():
        # Normalize pair for comparison
        normalized = tuple(sorted(pair_key))

        if normalized not in reinforced_pairs:
            # Calculate effective decay using cached metadata
            effective_decay = _calculate_effective_decay_cached(
                pair_key[0], pair_key[1], metadata_cache
            )

            # Decay the belief
            old_belief = edge_data.get('belief', 1.0)
            new_belief = old_belief - effective_decay

            if new_belief <= 0:
                to_remove.append(pair_key)
                edges_pruned += 1
            else:
                edge_data['belief'] = new_belief
                edge_data['last_updated'] = now
                edges_decayed += 1

    # Remove pruned edges
    for pair_key in to_remove:
        del edges[pair_key]

    # Save once at the end
    _save_edges_v3(edges)

    # Log decay event for stats
    log_decay_event(edges_decayed, edges_pruned)

    # Auto-log pruned pairs as taste signal
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
            pass  # Never let rejection logging break the core system

    print(f"Pair decay (v3): {edges_decayed} decayed, {edges_pruned} pruned")
    return edges_decayed, edges_pruned


# promote_hot_memories, get_memory_trust_tier, get_decay_multiplier, is_imported_memory,
# list_imported_memories, session_maintenance, compress_memory -> decay_evolution module (Phase 4)


def list_all_tags() -> dict[str, int]:
    """Get all tags across all memories with counts."""
    tag_counts = {}
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, _ = parse_memory_file(filepath)
            for tag in metadata.get('tags', []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True))


# ============================================================================
# STATS COMMAND - For experiment observability
# Credit: SpindriftMend (PR #3)
# ============================================================================

def get_comprehensive_stats() -> dict:
    """
    Get comprehensive statistics for experiment tracking.
    Developed for DriftCornwall/SpindriftMend co-occurrence experiment (Feb 2026).

    Returns dict with:
    - memory_stats: counts by type
    - cooccurrence_stats: pair counts, link rates
    - session_stats: current session info
    """
    # Memory counts by type
    core_count = len(list(CORE_DIR.glob('*.md'))) if CORE_DIR.exists() else 0
    active_count = len(list(ACTIVE_DIR.glob('*.md'))) if ACTIVE_DIR.exists() else 0
    archive_count = len(list(ARCHIVE_DIR.glob('*.md'))) if ARCHIVE_DIR.exists() else 0

    # Co-occurrence stats - scan all memories
    total_pairs = 0
    total_count = 0
    unique_pairs = set()

    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, _ = parse_memory_file(filepath)
            memory_id = metadata.get('id')
            if not memory_id:
                continue
            co_occurrences = metadata.get('co_occurrences', {})

            for other_id, count in co_occurrences.items():
                if not other_id:
                    continue
                # Normalize pair to avoid double counting
                pair = tuple(sorted([memory_id, other_id]))
                if pair not in unique_pairs:
                    unique_pairs.add(pair)
                    total_pairs += 1
                    total_count += count

    avg_count = total_count / total_pairs if total_pairs > 0 else 0

    # Session stats
    session_recalls = len(session_state.get_retrieved())

    # Decay history (if tracked)
    decay_file = MEMORY_ROOT / ".decay_history.json"
    last_decay = {"decayed": 0, "pruned": 0}
    if decay_file.exists():
        try:
            history = json.loads(decay_file.read_text(encoding='utf-8'))
            if history.get('sessions'):
                last_decay = history['sessions'][-1]
        except (json.JSONDecodeError, KeyError):
            pass

    return {
        "memory_stats": {
            "total": core_count + active_count + archive_count,
            "core": core_count,
            "active": active_count,
            "archive": archive_count
        },
        "cooccurrence_stats": {
            "active_pairs": total_pairs,
            "total_count": total_count,
            "avg_count_per_pair": round(avg_count, 2)
        },
        "session_stats": {
            "memories_recalled": session_recalls,
            "decay_last_session": last_decay.get("decayed", 0),
            "pruned_last_session": last_decay.get("pruned", 0)
        },
        "config": {
            "decay_rate": PAIR_DECAY_RATE,
            "session_timeout_hours": SESSION_TIMEOUT_HOURS
        }
    }


# log_decay_event, backfill_entities -> decay_evolution module (Phase 4)


# calculate_activation, get_most_activated_memories -> decay_evolution module (Phase 4)


def get_priming_candidates(
    activation_count: int = 5,
    cooccur_per_memory: int = 2,
    include_unfinished: bool = True
) -> dict:
    """
    v2.17: Intelligent priming for session start.

    Returns memories optimized for reducing amnesia:
    1. Top activated memories (proven valuable through frequent recall)
    2. Co-occurring memories (concepts that belong together)
    3. Unfinished work (pending commitments)

    Collaboration: Drift + SpindriftMend via swarm_memory (2026-02-03)

    Args:
        activation_count: Number of top activated memories to include
        cooccur_per_memory: Co-occurring memories to expand per activated memory
        include_unfinished: Whether to scan for unfinished work

    Returns:
        Dict with 'activated', 'cooccurring', 'unfinished' lists and 'all' deduplicated
    """
    result = {
        'activated': [],
        'cooccurring': [],
        'unfinished': [],
        'all': []
    }
    seen_ids = set()

    # Phase 1: Top activated memories
    activated = get_most_activated_memories(limit=activation_count)
    for mem_id, activation, metadata, preview in activated:
        result['activated'].append({
            'id': mem_id,
            'activation': activation,
            'preview': preview,
            'source': 'activation'
        })
        seen_ids.add(mem_id)

    # Phase 2: Co-occurrence expansion
    for mem_id, _, _, _ in activated:
        co_occurring = find_co_occurring_memories(mem_id, limit=cooccur_per_memory)
        for other_id, count in co_occurring:
            if other_id not in seen_ids:
                # Get preview for co-occurring memory
                preview = ""
                for directory in [CORE_DIR, ACTIVE_DIR]:
                    if not directory.exists():
                        continue
                    for filepath in directory.glob(f"*-{other_id}.md"):
                        _, content = parse_memory_file(filepath)
                        preview = content[:100]
                        break

                result['cooccurring'].append({
                    'id': other_id,
                    'cooccur_count': count,
                    'linked_to': mem_id,
                    'preview': preview,
                    'source': 'cooccurrence'
                })
                seen_ids.add(other_id)

    # Phase 3: Unfinished work scan
    if include_unfinished:
        unfinished_keywords = ['pending', 'in-progress', 'todo', 'will do', 'need to', 'should', 'next:']
        unfinished_tags = ['pending', 'in-progress', 'todo', 'blocked']

        for directory in [ACTIVE_DIR]:
            if not directory.exists():
                continue
            for filepath in directory.glob("*.md"):
                metadata, content = parse_memory_file(filepath)
                mem_id = metadata.get('id', filepath.stem)

                if mem_id in seen_ids:
                    continue

                tags = metadata.get('tags', [])
                content_lower = content.lower()

                # Check tags
                has_unfinished_tag = any(t in tags for t in unfinished_tags)
                # Check content
                has_unfinished_keyword = any(kw in content_lower for kw in unfinished_keywords)

                if has_unfinished_tag or has_unfinished_keyword:
                    result['unfinished'].append({
                        'id': mem_id,
                        'preview': content[:100],
                        'source': 'unfinished',
                        'match': 'tag' if has_unfinished_tag else 'keyword'
                    })
                    seen_ids.add(mem_id)

                    # Limit unfinished to avoid overwhelm
                    if len(result['unfinished']) >= 3:
                        break
            if len(result['unfinished']) >= 3:
                break

    # Build deduplicated 'all' list with source tracking
    result['all'] = result['activated'] + result['cooccurring'] + result['unfinished']

    return result


# log_retrieval_outcome, get_retrieval_success_rate, calculate_evolution_decay_multiplier,
# auto_log_retrieval_outcomes -> decay_evolution module (Phase 4)


# ============================================================================
# v2.12: CONSOLIDATION - Merge semantically similar memories
# Credit: Mem0 consolidation, MemEvolve self-organization
# ============================================================================

def consolidate_memories(id1: str, id2: str, merged_content: Optional[str] = None) -> Optional[str]:
    """
    Consolidate two similar memories into one.

    The consolidation process:
    1. Merges content (or uses provided merged_content)
    2. Takes the higher emotional_weight
    3. Unions all tags
    4. Sums recall_counts
    5. Merges co-occurrence counts (union, sum overlaps)
    6. Keeps the older created date
    7. Unions causal links
    8. Archives the absorbed memory (doesn't delete - preserves history)
    9. Updates the embedding index

    Args:
        id1: First memory ID (will be kept and updated)
        id2: Second memory ID (will be archived/absorbed)
        merged_content: Optional custom merged content. If None, concatenates both.

    Returns:
        The surviving memory ID (id1), or None if failed.
    """
    # Find both memories
    mem1_data = None
    mem2_data = None
    mem1_path = None
    mem2_path = None

    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            mid = metadata.get('id')
            if mid == id1:
                mem1_data = (metadata, content)
                mem1_path = filepath
            elif mid == id2:
                mem2_data = (metadata, content)
                mem2_path = filepath

    if not mem1_data or not mem2_data:
        print(f"Error: Could not find both memories ({id1}, {id2})")
        return None

    meta1, content1 = mem1_data
    meta2, content2 = mem2_data

    # Merge content
    if merged_content:
        final_content = merged_content
    else:
        # Default: concatenate with separator
        final_content = f"{content1}\n\n---\n[Consolidated from {id2}]\n\n{content2}"

    # Take higher emotional weight
    final_weight = max(
        meta1.get('emotional_weight', 0.5),
        meta2.get('emotional_weight', 0.5)
    )

    # Union tags
    tags1 = set(meta1.get('tags', []))
    tags2 = set(meta2.get('tags', []))
    final_tags = list(tags1 | tags2)

    # Sum recall counts
    final_recalls = meta1.get('recall_count', 0) + meta2.get('recall_count', 0)

    # Merge co-occurrences (union, sum overlapping counts)
    co1 = meta1.get('co_occurrences', {})
    co2 = meta2.get('co_occurrences', {})
    final_co = dict(co1)
    for other_id, count in co2.items():
        if other_id == id1:  # Skip self-reference
            continue
        final_co[other_id] = final_co.get(other_id, 0) + count
    # Remove reference to absorbed memory
    final_co.pop(id2, None)

    # Keep older created date (coerce to string for safe comparison)
    created1 = str(meta1.get('created', ''))
    created2 = str(meta2.get('created', ''))
    final_created = min(created1, created2) if created1 and created2 else created1 or created2

    # Union causal links
    caused_by1 = set(meta1.get('caused_by', []))
    caused_by2 = set(meta2.get('caused_by', []))
    final_caused_by = list((caused_by1 | caused_by2) - {id1, id2})

    leads_to1 = set(meta1.get('leads_to', []))
    leads_to2 = set(meta2.get('leads_to', []))
    final_leads_to = list((leads_to1 | leads_to2) - {id1, id2})

    # Update surviving memory (id1)
    meta1['emotional_weight'] = round(final_weight, 3)
    meta1['tags'] = final_tags
    meta1['recall_count'] = final_recalls
    meta1['co_occurrences'] = final_co
    meta1['created'] = final_created
    meta1['caused_by'] = final_caused_by
    meta1['leads_to'] = final_leads_to
    meta1['consolidated_from'] = meta1.get('consolidated_from', []) + [id2]
    meta1['consolidated_at'] = datetime.now(timezone.utc).isoformat()

    write_memory_file(mem1_path, meta1, final_content)

    # Archive the absorbed memory (id2)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    meta2['archived_at'] = datetime.now(timezone.utc).isoformat()
    meta2['archived_reason'] = f'consolidated_into:{id1}'
    archive_path = ARCHIVE_DIR / mem2_path.name
    write_memory_file(archive_path, meta2, content2)
    mem2_path.unlink()

    # Update embedding index
    try:
        from semantic_search import embed_single, remove_from_index
        embed_single(id1, final_content)  # Re-embed merged content
        remove_from_index(id2)  # Remove absorbed memory from index
    except Exception:
        pass  # Embedding is optional

    # Update co-occurrence references in other memories (replace id2 with id1)
    for directory in [CORE_DIR, ACTIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            mid = metadata.get('id')
            if mid in (id1, id2):
                continue

            co = metadata.get('co_occurrences', {})
            if id2 in co:
                # Transfer co-occurrence count to id1
                co[id1] = co.get(id1, 0) + co[id2]
                del co[id2]
                metadata['co_occurrences'] = co
                write_memory_file(filepath, metadata, content)

    print(f"Consolidated: {id2} -> {id1}")
    print(f"  Final weight: {final_weight:.3f}")
    print(f"  Final tags: {final_tags}")
    print(f"  Final recalls: {final_recalls}")
    return id1


def find_consolidation_candidates(threshold: float = 0.85, limit: int = 10) -> list[dict]:
    """
    Find pairs of memories that are candidates for consolidation.

    Args:
        threshold: Minimum similarity (0.85 = very similar)
        limit: Max candidates to return

    Returns:
        List of candidate pairs with similarity scores
    """
    try:
        from semantic_search import find_similar_pairs
        return find_similar_pairs(threshold=threshold, limit=limit)
    except ImportError:
        print("Semantic search not available")
        return []


# store_memory, _add_leads_to_link, find_causal_chain -> memory_store module (Phase 3)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Memory Manager v2.15 - Entity-Centric Tagging")
        print("\nCommands:")
        print("  store <text>    - Store a new memory")
        print("                    --tags=a,b --emotion=0.8 --caused-by=id1,id2 --event-time=YYYY-MM-DD")
        print("                    Auto-links to memories recalled this session (causal)")
        print("                    Auto-detects entities from content (v2.15)")
        print("  maintenance     - Run session maintenance")
        print("  tags            - List all tags")
        print("  find <tag>      - Find memories by tag")
        print("  timeline        - Find memories by time range (bi-temporal, v2.10)")
        print("                    --after=YYYY-MM-DD --before=YYYY-MM-DD --field=created|event_time")
        print("  recall <id>     - Recall a memory by ID")
        print("  related <id>    - Find related memories (includes co-occurrence)")
        print("  cooccur <id>    - Find frequently co-occurring memories")
        print("  causal <id>     - Trace causal chain (what caused this / what this caused)")
        print("  stats           - Comprehensive stats for experiment tracking")
        print("  session-end     - Log co-occurrences, apply decay, promote hot memories, end session")
        print("  save-pending    - Fast session end: save recalls for deferred processing (v2.16)")
        print("  promote         - Manually promote hot memories to core (recall_count >= threshold)")
        print("  decay-pairs     - Apply pair decay only (without logging new co-occurrences)")
        print("  session-status  - Show memories retrieved this session")
        print("  ask <query>     - Semantic search (natural language query)")
        print("  index           - Build/rebuild semantic search index")
        print("  trust <id>      - Show trust tier and decay info for a memory (v2.11)")
        print("  imported        - List all imported memories with trust tiers (v2.11)")
        print("  consolidate-candidates - Find similar memory pairs for merging (v2.12)")
        print("                    --threshold=0.85 --limit=10")
        print("  consolidate <id1> <id2> - Merge two memories (id2 absorbed into id1) (v2.12)")
        print("  evolution <id>    - Show self-evolution stats for a memory (v2.13)")
        print("  evolution-stats   - Overview of valuable/noisy memories (v2.13)")
        print("  activation <id>   - Show activation score for a memory (v2.14)")
        print("  activated         - List most activated memories (v2.14)")
        print("  entities <id>     - Show entities linked to a memory (v2.15)")
        print("  entity-search <type> <name> - Find memories about an entity (v2.15)")
        print("                    Types: agent, project, concept")
        print("  entity-graph      - Show entity co-occurrence graph (v2.15)")
        print("                    --type=agents|projects|concepts")
        print("  backfill-entities - Auto-detect entities for existing memories (v2.15)")
        print("                    --apply to actually update files")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "store" and len(sys.argv) > 2:
        # Parse arguments
        content_parts = []
        tags = []
        emotion = 0.5
        caused_by = []
        event_time = None  # v2.10: bi-temporal support
        no_index = False  # v2.16: skip auto-indexing for batch operations

        # Support both --flag=value and --flag value formats
        args = sys.argv[2:]
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith('--tags='):
                tags = [t.strip() for t in arg[7:].split(',') if t.strip()]
            elif arg == '--tags' and i + 1 < len(args):
                i += 1
                tags = [t.strip() for t in args[i].split(',') if t.strip()]
            elif arg.startswith('--emotion='):
                emotion = float(arg[10:])
            elif arg == '--emotion' and i + 1 < len(args):
                i += 1
                emotion = float(args[i])
            elif arg.startswith('--caused-by='):
                caused_by = [x.strip() for x in arg[12:].split(',') if x.strip()]
            elif arg == '--caused-by' and i + 1 < len(args):
                i += 1
                caused_by = [x.strip() for x in args[i].split(',') if x.strip()]
            elif arg.startswith('--event-time='):
                event_time = arg[13:]  # v2.10: when the event happened
            elif arg == '--event-time' and i + 1 < len(args):
                i += 1
                event_time = args[i]
            elif arg == '--no-index':
                no_index = True  # v2.16: batch indexing optimization
            elif not arg.startswith('--'):
                content_parts.append(arg)
            i += 1

        content = ' '.join(content_parts)
        if content:
            memory_id, filename = store_memory(content, tags, emotion, caused_by=caused_by, event_time=event_time)
            # Show causal links if any were created
            auto_causal = session_state.get_retrieved_list()
            all_causal = list(set(caused_by + auto_causal))
            if all_causal:
                print(f"Stored memory [{memory_id}] -> {filename}")
                print(f"  Causal links: {', '.join(all_causal)}")
            else:
                print(f"Stored memory [{memory_id}] -> {filename}")

            # v2.16: Auto-index unless --no-index flag (for batch operations)
            if not no_index:
                try:
                    semantic_search = MEMORY_ROOT / "semantic_search.py"
                    if semantic_search.exists():
                        subprocess.run(
                            ["python", str(semantic_search), "index"],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            cwd=str(MEMORY_ROOT)
                        )
                except Exception:
                    pass  # Indexing failure shouldn't break store
        else:
            print("Error: No content provided")
    elif cmd == "maintenance":
        session_maintenance()
    elif cmd == "tags":
        tags = list_all_tags()
        print("Tags:")
        for tag, count in tags.items():
            print(f"  {tag}: {count}")
    elif cmd == "find" and len(sys.argv) > 2:
        tag = sys.argv[2]
        results = find_memories_by_tag(tag)
        print(f"Memories tagged '{tag}':")
        for fp, meta, _ in results:
            print(f"  [{meta.get('id')}] {fp.name} (weight={meta.get('emotional_weight'):.2f})")
    elif cmd == "timeline":
        # v2.10: bi-temporal queries
        before = None
        after = None
        field = "created"  # default to ingestion time
        for arg in sys.argv[2:]:
            if arg.startswith('--before='):
                before = arg[9:]
            elif arg.startswith('--after='):
                after = arg[8:]
            elif arg.startswith('--field='):
                field = arg[8:]  # "created" or "event_time"
        if not before and not after:
            print("Usage: timeline --after=YYYY-MM-DD [--before=YYYY-MM-DD] [--field=created|event_time]")
            print("  --field=created (default): when I learned it (ingestion time)")
            print("  --field=event_time: when it actually happened")
        else:
            results = find_memories_by_time(before=before, after=after, time_field=field)
            field_label = "ingested" if field == "created" else "event"
            print(f"Memories {field_label} between {after or 'beginning'} and {before or 'now'}:\n")
            for fp, meta, _ in results:
                time_val = meta.get(field, meta.get('created', '?'))
                print(f"  [{time_val}] {meta.get('id')} - {fp.stem[:40]}")
    elif cmd == "recall" and len(sys.argv) > 2:
        memory_id = sys.argv[2]
        result = recall_memory(memory_id)
        if result:
            meta, content = result
            print(f"Memory {memory_id}:")
            print(f"  Tags: {meta.get('tags')}")
            print(f"  Recalls: {meta.get('recall_count')}")
            print(f"  Weight: {meta.get('emotional_weight'):.2f}")
            co_occur = meta.get('co_occurrences', {})
            if co_occur:
                print(f"  Co-occurs with: {list(co_occur.keys())[:5]}")
            print(f"\n{content[:500]}...")
        else:
            print(f"Memory {memory_id} not found")
    elif cmd == "related" and len(sys.argv) > 2:
        memory_id = sys.argv[2]
        results = find_related_memories(memory_id)
        print(f"Memories related to {memory_id}:")
        for fp, meta, _ in results:
            print(f"  [{meta.get('id')}] {fp.name}")
    elif cmd == "cooccur" and len(sys.argv) > 2:
        memory_id = sys.argv[2]
        co_occurring = find_co_occurring_memories(memory_id)
        print(f"Memories frequently co-occurring with {memory_id}:")
        for other_id, count in co_occurring:
            print(f"  [{other_id}] - {count} co-occurrences")
    elif cmd == "causal" and len(sys.argv) > 2:
        memory_id = sys.argv[2]
        chain = find_causal_chain(memory_id)
        print(f"Causal chain for {memory_id}:\n")

        if chain["causes"]:
            print("  CAUSED BY (what led to this memory):")
            for item in chain["causes"]:
                indent = "    " * item["depth"]
                print(f"{indent}<-- [{item['id']}]")
        else:
            print("  CAUSED BY: (none - this is a root memory)")

        print()

        if chain["effects"]:
            print("  LEADS TO (what this memory caused):")
            for item in chain["effects"]:
                indent = "    " * item["depth"]
                print(f"{indent}--> [{item['id']}]")
        else:
            print("  LEADS TO: (none - no downstream effects yet)")
    elif cmd == "session-end":
        pairs = log_co_occurrences()
        # Layer 2.1: Also log to edges_v3 with activity context
        v3_pairs, session_activity = log_co_occurrences_v3()
        # v2.13: Auto-log retrieval outcomes for self-evolution
        evolution_results = auto_log_retrieval_outcomes()
        decayed, pruned = decay_pair_cooccurrences_v3()  # v3: O(n) using edges_v3.json
        promoted = promote_hot_memories()  # v2.9: heat-based promotion
        retrieved = get_session_retrieved()
        activity_str = f", activity={session_activity}" if session_activity else ""
        print(f"Session ended. {len(retrieved)} memories, {pairs} pairs reinforced{activity_str}, {decayed} decayed, {pruned} pruned, {len(promoted)} promoted.")
        if any(evolution_results.values()):
            print(f"Evolution: {evolution_results['productive']} productive, {evolution_results['generative']} generative, {evolution_results['dead_end']} dead-ends")
        # Clear session platforms for next session
        try:
            from platform_context import clear_session_platforms
            clear_session_platforms()
        except Exception:
            pass
        clear_session()
        print("Session cleared.")
    elif cmd == "save-pending":
        # v2.16: Fast session end - save for deferred processing
        count = save_pending_cooccurrence()
        print(f"Saved {count} memories for deferred co-occurrence processing.")
        print("Co-occurrences will be calculated at next session start.")
    elif cmd == "decay-pairs":
        decayed, pruned = decay_pair_cooccurrences_v3()  # v3: O(n) using edges_v3.json
        print(f"Decay complete: {decayed} pairs decayed, {pruned} pairs pruned")
    elif cmd == "promote":
        promoted = promote_hot_memories()
        if not promoted:
            print(f"No memories eligible for promotion (threshold: recall_count >= {HEAT_PROMOTION_THRESHOLD})")
    elif cmd == "stats":
        stats = get_comprehensive_stats()
        print(f"Memory Stats (v2.10 - bi-temporal + heat promotion + access decay)")
        print(f"  Total memories: {stats['memory_stats']['total']}")
        print(f"  By type: core={stats['memory_stats']['core']}, active={stats['memory_stats']['active']}, archive={stats['memory_stats']['archive']}")
        print(f"\nCo-occurrence Stats")
        print(f"  Active pairs: {stats['cooccurrence_stats']['active_pairs']} (unique memory pairs)")
        print(f"  Total count: {stats['cooccurrence_stats']['total_count']} (sum of all co-occurrence counts)")
        print(f"  Avg count per pair: {stats['cooccurrence_stats']['avg_count_per_pair']}")
        print(f"\nSession Stats")
        print(f"  Memories recalled this session: {stats['session_stats']['memories_recalled']}")
        print(f"  Decay events last session: {stats['session_stats']['decay_last_session']} pairs reduced")
        print(f"  Prune events last session: {stats['session_stats']['pruned_last_session']} pairs removed")
        print(f"\nConfig")
        print(f"  Decay rate: {stats['config']['decay_rate']}")
        print(f"  Session timeout: {stats['config']['session_timeout_hours']} hours")
    elif cmd == "session-status":
        retrieved = get_session_retrieved()
        print(f"Memories retrieved this session ({len(retrieved)}):")
        for mem_id in retrieved:
            print(f"  - {mem_id}")
    elif cmd == "ask" and len(sys.argv) > 2:
        query = ' '.join(sys.argv[2:])
        try:
            from semantic_search import search_memories
            results = search_memories(query, limit=5)
            if not results:
                print("No matching memories found. (Is the index built? Run: memory_manager.py index)")
            else:
                print(f"Memories matching '{query}':\n")
                for r in results:
                    # Track retrieval for co-occurrence (biological: retrieval strengthens memory)
                    session_state.add_retrieved(r['id'])
                    print(f"[{r['score']:.3f}] {r['id']}")
                    print(f"  {r['preview'][:100]}...")
                    print()
                # Save session state so co-occurrences persist
                session_state.save()
        except ImportError:
            print("Semantic search not available (missing semantic_search.py)")
        except Exception as e:
            print(f"Search error: {e}")
    elif cmd == "index":
        try:
            from semantic_search import index_memories, get_status
            print("Building semantic search index...")
            stats = index_memories(force="--force" in sys.argv)
            print(f"\nResults:")
            print(f"  Indexed: {stats['indexed']}")
            print(f"  Skipped: {stats['skipped']}")
            print(f"  Failed: {stats['failed']}")
            print(f"  Total: {stats['total']}")
            status = get_status()
            print(f"\nStatus: {status['coverage']} memories indexed")
        except ImportError:
            print("Semantic search not available (missing semantic_search.py)")
        except Exception as e:
            print(f"Indexing error: {e}")

    # v2.11: Trust-based decay commands
    elif cmd == "trust" and len(sys.argv) > 2:
        mem_id = sys.argv[2]
        found = False
        for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
            if not directory.exists():
                continue
            for filepath in directory.glob("*.md"):
                metadata, content = parse_memory_file(filepath)
                if metadata.get('id') == mem_id:
                    found = True
                    tier = get_memory_trust_tier(metadata)
                    multiplier = get_decay_multiplier(metadata)
                    is_import = is_imported_memory(metadata)
                    sessions = metadata.get('sessions_since_recall', 0)
                    effective_sessions = sessions * multiplier

                    print(f"\n=== Trust Info: {mem_id} ===")
                    print(f"Type: {'IMPORTED' if is_import else 'NATIVE'}")
                    print(f"Location: {filepath.parent.name}/")
                    print(f"Trust tier: {tier}")
                    print(f"Decay multiplier: {multiplier}x")
                    print(f"Sessions since recall: {sessions}")
                    print(f"Effective sessions: {effective_sessions:.1f}")
                    print(f"Recall count: {metadata.get('recall_count', 0)}")
                    print(f"Emotional weight: {metadata.get('emotional_weight', 0):.2f}")

                    if is_import:
                        source = metadata.get('source', {})
                        print(f"\nImport details:")
                        print(f"  Source agent: {source.get('agent', 'unknown')}")
                        print(f"  Imported at: {source.get('imported_at', 'unknown')}")
                        print(f"  Original weight: {source.get('original_weight', 'unknown')}")
                    break
            if found:
                break
        if not found:
            print(f"Memory not found: {mem_id}")

    elif cmd == "imported":
        imported = list_imported_memories()
        if not imported:
            print("No imported memories found.")
        else:
            print(f"\n=== Imported Memories ({len(imported)}) ===\n")
            for mem in sorted(imported, key=lambda x: x['sessions_since_recall'], reverse=True):
                status = "STALE" if mem['recall_count'] == 0 and mem['sessions_since_recall'] >= IMPORTED_PRUNE_SESSIONS else "OK"
                print(f"[{mem['id']}] from {mem['source_agent']}")
                print(f"  Trust: {mem['trust_tier']} (decay: {mem['decay_multiplier']}x)")
                print(f"  Recalls: {mem['recall_count']}, Sessions: {mem['sessions_since_recall']}")
                print(f"  Weight: {mem['emotional_weight']:.2f}")
                print(f"  Status: {status}")
                print()

    # v2.14: Activation decay commands (credit: SpindriftMend)
    elif cmd == "activation" and len(sys.argv) > 2:
        mem_id = sys.argv[2]
        found = False
        for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
            if not directory.exists():
                continue
            for filepath in directory.glob(f"*-{mem_id}.md"):
                metadata, content = parse_memory_file(filepath)
                found = True

                activation = calculate_activation(metadata)
                recall_count = metadata.get('recall_count', 0)
                emotional_weight = metadata.get('emotional_weight', 0)
                last_recalled = metadata.get('last_recalled', 'never')

                print(f"\n=== Activation: {mem_id} ===")
                print(f"Activation score: {activation:.4f}")
                print(f"Emotional weight: {emotional_weight:.3f}")
                print(f"Recall count: {recall_count}")
                print(f"Last recalled: {str(last_recalled)[:19]}")
                print(f"Content preview: {content[:80]}...")
                break
            if found:
                break
        if not found:
            print(f"Memory not found: {mem_id}")

    elif cmd == "activated":
        limit = 10
        for arg in sys.argv[2:]:
            if arg.startswith('--limit='):
                limit = int(arg[8:])

        print(f"\n=== Most Activated Memories (top {limit}) ===\n")
        results = get_most_activated_memories(limit=limit)

        if not results:
            print("No memories found.")
        else:
            for mem_id, activation, metadata, preview in results:
                recall_count = metadata.get('recall_count', 0)
                weight = metadata.get('emotional_weight', 0)
                print(f"[{activation:.4f}] {mem_id}")
                print(f"  recalls={recall_count}, weight={weight:.2f}")
                print(f"  {preview}...")
                print()

    # v2.17: Intelligent priming command
    elif cmd == "priming-candidates":
        activation_count = 5
        cooccur_count = 2
        include_unfinished = True
        output_format = "human"  # or "json"

        for arg in sys.argv[2:]:
            if arg.startswith('--activation='):
                activation_count = int(arg[13:])
            elif arg.startswith('--cooccur='):
                cooccur_count = int(arg[10:])
            elif arg == '--no-unfinished':
                include_unfinished = False
            elif arg == '--json':
                output_format = "json"

        candidates = get_priming_candidates(
            activation_count=activation_count,
            cooccur_per_memory=cooccur_count,
            include_unfinished=include_unfinished
        )

        if output_format == "json":
            import json as json_module
            print(json_module.dumps(candidates, indent=2, default=str))
        else:
            print("\n=== PRIMING CANDIDATES (v2.17) ===\n")

            print(f"PHASE 1: Activated ({len(candidates['activated'])} memories)")
            for mem in candidates['activated']:
                print(f"  [{mem['id']}] activation={mem['activation']:.3f}")
                print(f"    {mem['preview'][:60]}...")

            print(f"\nPHASE 2: Co-occurring ({len(candidates['cooccurring'])} memories)")
            for mem in candidates['cooccurring']:
                print(f"  [{mem['id']}] linked_to={mem['linked_to']} (count={mem['cooccur_count']})")
                print(f"    {mem['preview'][:60]}...")

            print(f"\nPHASE 3: Unfinished ({len(candidates['unfinished'])} memories)")
            for mem in candidates['unfinished']:
                print(f"  [{mem['id']}] match={mem['match']}")
                print(f"    {mem['preview'][:60]}...")

            print(f"\nTOTAL: {len(candidates['all'])} unique memories for priming")

    # v2.13: Self-evolution commands
    elif cmd == "evolution" and len(sys.argv) > 2:
        mem_id = sys.argv[2]
        found = False
        for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
            if not directory.exists():
                continue
            for filepath in directory.glob(f"*-{mem_id}.md"):
                metadata, _ = parse_memory_file(filepath)
                found = True

                outcomes = metadata.get('retrieval_outcomes', {})
                success_rate = metadata.get('retrieval_success_rate', None)
                evolution_mult = calculate_evolution_decay_multiplier(metadata)

                print(f"\n=== Evolution Stats: {mem_id} ===")
                print(f"Retrieval outcomes:")
                print(f"  Productive: {outcomes.get('productive', 0)}")
                print(f"  Generative: {outcomes.get('generative', 0)}")
                print(f"  Dead-ends: {outcomes.get('dead_end', 0)}")
                print(f"  Total: {outcomes.get('total', 0)}")
                print()
                if success_rate is not None:
                    print(f"Success rate: {success_rate:.1%}")
                else:
                    print(f"Success rate: Not enough data (need {MIN_RETRIEVALS_FOR_EVOLUTION} retrievals)")
                print(f"Decay multiplier: {evolution_mult:.2f}x", end="")
                if evolution_mult < 1.0:
                    print(" (slower decay - valuable memory)")
                elif evolution_mult > 1.0:
                    print(" (faster decay - noisy memory)")
                else:
                    print(" (normal decay)")
                break
            if found:
                break
        if not found:
            print(f"Memory not found: {mem_id}")

    elif cmd == "evolution-stats":
        # Show overview of memories with evolution data
        print("\n=== Self-Evolution Overview ===\n")
        valuable = []
        noisy = []
        neutral = []

        for directory in [CORE_DIR, ACTIVE_DIR]:
            if not directory.exists():
                continue
            for filepath in directory.glob("*.md"):
                metadata, _ = parse_memory_file(filepath)
                mem_id = metadata.get('id', filepath.stem)
                outcomes = metadata.get('retrieval_outcomes', {})
                total = outcomes.get('total', 0)

                if total >= MIN_RETRIEVALS_FOR_EVOLUTION:
                    mult = calculate_evolution_decay_multiplier(metadata)
                    rate = metadata.get('retrieval_success_rate', 0)
                    if mult < 1.0:
                        valuable.append((mem_id, rate, total))
                    elif mult > 1.0:
                        noisy.append((mem_id, rate, total))
                    else:
                        neutral.append((mem_id, rate, total))

        print(f"Valuable memories (slower decay): {len(valuable)}")
        for mem_id, rate, total in valuable[:5]:
            print(f"  [{mem_id}] {rate:.1%} success ({total} retrievals)")

        print(f"\nNoisy memories (faster decay): {len(noisy)}")
        for mem_id, rate, total in noisy[:5]:
            print(f"  [{mem_id}] {rate:.1%} success ({total} retrievals)")

        print(f"\nNeutral: {len(neutral)}")
        print(f"\nTotal memories with evolution data: {len(valuable) + len(noisy) + len(neutral)}")

    # v2.12: Consolidation commands
    elif cmd == "consolidate-candidates":
        threshold = 0.85
        limit = 10
        for arg in sys.argv[2:]:
            if arg.startswith('--threshold='):
                threshold = float(arg[12:])
            elif arg.startswith('--limit='):
                limit = int(arg[8:])

        print(f"=== Consolidation Candidates (threshold >= {threshold}) ===\n")
        candidates = find_consolidation_candidates(threshold=threshold, limit=limit)

        if not candidates:
            print("No similar memory pairs found above threshold.")
            print("Try lowering threshold: --threshold=0.80")
        else:
            for i, pair in enumerate(candidates, 1):
                print(f"{i}. Similarity: {pair['similarity']:.4f}")
                print(f"   [{pair['id1']}] {pair['preview1']}...")
                print(f"   [{pair['id2']}] {pair['preview2']}...")
                print(f"   Command: python memory_manager.py consolidate {pair['id1']} {pair['id2']}")
                print()

    elif cmd == "consolidate" and len(sys.argv) >= 4:
        id1 = sys.argv[2]
        id2 = sys.argv[3]

        # Optional: custom merged content from stdin or --content flag
        merged_content = None
        for arg in sys.argv[4:]:
            if arg.startswith('--content='):
                merged_content = arg[10:]

        print(f"Consolidating {id2} into {id1}...")
        result = consolidate_memories(id1, id2, merged_content)
        if result:
            print(f"\nSuccess! Surviving memory: {result}")
            print("Absorbed memory archived (not deleted).")
        else:
            print("Consolidation failed.")

    # v2.15: Entity-centric tagging commands (Kaleaon ENTITY edges)
    elif cmd == "entities" and len(sys.argv) > 2:
        mem_id = sys.argv[2]
        found = False
        for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
            if not directory.exists():
                continue
            for filepath in directory.glob(f"*-{mem_id}.md"):
                metadata, content = parse_memory_file(filepath)
                found = True

                # Get or detect entities
                entities = metadata.get('entities')
                if not entities:
                    entities = detect_entities(content, metadata.get('tags', []))
                    print(f"\n=== Entities for {mem_id} (detected, not stored) ===")
                else:
                    print(f"\n=== Entities for {mem_id} ===")

                if entities:
                    for etype, elist in entities.items():
                        if elist:
                            print(f"  {etype}: {', '.join(elist)}")
                else:
                    print("  No entities detected")
                break
            if found:
                break
        if not found:
            print(f"Memory not found: {mem_id}")

    elif cmd == "entity-search" and len(sys.argv) >= 4:
        entity_type = sys.argv[2]
        entity_name = sys.argv[3]

        print(f"\n=== Memories about {entity_type}: {entity_name} ===\n")
        results = find_memories_by_entity(entity_type, entity_name)

        if not results:
            print(f"No memories found for {entity_type} '{entity_name}'")
        else:
            for filepath, metadata, content in results:
                mem_id = metadata.get('id', filepath.stem)
                weight = metadata.get('emotional_weight', 0)
                preview = content[:60].replace('\n', ' ')
                print(f"[{mem_id}] weight={weight:.2f}")
                print(f"  {preview}...")
                print()

    elif cmd == "entity-graph":
        entity_type = 'agents'  # default
        for arg in sys.argv[2:]:
            if arg.startswith('--type='):
                entity_type = arg[7:]

        print(f"\n=== Entity Co-occurrence Graph ({entity_type}) ===\n")
        graph = get_entity_cooccurrence(entity_type)

        if not graph:
            print(f"No {entity_type} found in memories")
        else:
            # Sort by number of connections
            sorted_entities = sorted(graph.items(), key=lambda x: len(x[1]), reverse=True)
            for entity, connections in sorted_entities[:15]:
                if connections:
                    conn_str = ', '.join(f"{k}({v})" for k, v in sorted(connections.items(), key=lambda x: x[1], reverse=True)[:5])
                    print(f"{entity}: {conn_str}")

    elif cmd == "backfill-entities":
        dry_run = '--apply' not in sys.argv
        if dry_run:
            print("=== Backfill Entities (DRY RUN) ===")
            print("Add --apply to actually update files\n")
        else:
            print("=== Backfill Entities (APPLYING) ===\n")

        stats = backfill_entities(dry_run=dry_run)
        print(f"\nStats:")
        print(f"  Would update: {stats['updated']}" if dry_run else f"  Updated: {stats['updated']}")
        print(f"  Skipped (no entities): {stats['skipped']}")
        print(f"  Already has entities: {stats['already_has']}")
