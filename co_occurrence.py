#!/usr/bin/env python3
"""
Co-occurrence System — Edge provenance, pair decay, and co-occurrence logging.

Extracted from memory_manager.py (Phase 5).
Manages the co-occurrence graph: logging pairs from session recalls,
v3 edge provenance with observations/beliefs, and pair decay.
"""

import json
import math
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from memory_common import (
    MEMORY_ROOT, CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR, ALL_DIRS,
    PENDING_COOCCURRENCE_FILE,
    parse_memory_file, write_memory_file,
    SELF_EVOLUTION_ENABLED, get_agent_name,
)
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

# v5.0: Dimensional decay — only full-decay edges in active W-dimensions
# Edges outside the session's context get dramatically reduced decay
# Credit: Multi-graph RFC (Issue #19), joint design with SpindriftMind
INACTIVE_CONTEXT_FACTOR = 0.1  # Non-overlapping edges decay at 0.03 instead of 0.3

# v2.13: Self-evolution flag (now imported from memory_common — no peer-module coupling)

# v3.0: Edge Provenance
OBSERVATION_MAX_AGE_DAYS = 30
TRUST_TIERS = {
    'self': 1.0,
    'verified_agent': 0.8,
    'platform': 0.6,
    'unknown': 0.3
}


# --- Co-occurrence logging ---

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

    for i, mem_id_1 in enumerate(retrieved):
        for mem_id_2 in retrieved[i + 1:]:
            for memory_id, other_id in [(mem_id_1, mem_id_2), (mem_id_2, mem_id_1)]:
                for directory in ALL_DIRS:
                    if not directory.exists():
                        continue
                    for filepath in directory.glob(f"*-{memory_id}.md"):
                        metadata, content = parse_memory_file(filepath)

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

    edges = _load_edges_v3()
    session_id = datetime.now(timezone.utc).isoformat()
    pairs_updated = 0

    for i, id1 in enumerate(retrieved):
        for id2 in retrieved[i + 1:]:
            pair = tuple(sorted([id1, id2]))

            obs = _create_observation(
                source_type='session_recall',
                weight=1.0,
                trust_tier='self',
                session_id=session_id,
                agent=get_agent_name(),
                platform=','.join(session_platforms) if session_platforms else None,
                activity=session_activity
            )

            now = datetime.now(timezone.utc).isoformat()
            if pair not in edges:
                edges[pair] = {
                    'observations': [],
                    'belief': 0.0,
                    'first_formed': now,
                    'last_updated': now,
                    'platform_context': {},
                    'activity_context': {},
                    'thinking_about': [],
                }

            edges[pair]['observations'].append(obs)
            edges[pair]['belief'] = aggregate_belief(edges[pair]['observations'])
            edges[pair]['last_updated'] = now

            if 'platform_context' not in edges[pair]:
                edges[pair]['platform_context'] = {}
            for plat in session_platforms:
                edges[pair]['platform_context'][plat] = edges[pair]['platform_context'].get(plat, 0) + 1

            if 'activity_context' not in edges[pair]:
                edges[pair]['activity_context'] = {}
            if session_activity:
                edges[pair]['activity_context'][session_activity] = (
                    edges[pair]['activity_context'].get(session_activity, 0) + 1
                )

            if 'thinking_about' not in edges[pair]:
                edges[pair]['thinking_about'] = []
            other_memories = [m for m in retrieved if m not in pair]
            for mem_id in other_memories:
                if mem_id not in edges[pair]['thinking_about']:
                    edges[pair]['thinking_about'].append(mem_id)

            pairs_updated += 1

    _save_edges_v3(edges)
    return pairs_updated, session_activity


# --- Deferred co-occurrence (v2.16) ---

def save_pending_cooccurrence() -> int:
    """
    v2.16: Fast session end - save retrieved IDs to pending file for deferred processing.
    The expensive co-occurrence calculation happens at NEXT session start.
    """
    retrieved = session_state.get_retrieved_list()
    if not retrieved:
        print("No memories to save to pending.")
        return 0

    pending_data = {
        'retrieved': retrieved,
        'session_id': datetime.now(timezone.utc).isoformat(),
        'agent': get_agent_name(),
        'saved_at': datetime.now(timezone.utc).isoformat()
    }
    PENDING_COOCCURRENCE_FILE.write_text(
        json.dumps(pending_data, indent=2), encoding='utf-8'
    )

    session_state.clear()

    print(f"Saved {len(retrieved)} memories to pending co-occurrence file.")
    return len(retrieved)


def process_pending_cooccurrence() -> int:
    """
    v2.16: Process pending co-occurrence file from previous session.
    Call this at session START (when there's time).
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

    for i, mem_id_1 in enumerate(retrieved):
        for mem_id_2 in retrieved[i + 1:]:
            for memory_id, other_id in [(mem_id_1, mem_id_2), (mem_id_2, mem_id_1)]:
                for directory in ALL_DIRS:
                    if not directory.exists():
                        continue
                    for filepath in directory.glob(f"*-{memory_id}.md"):
                        metadata, content = parse_memory_file(filepath)

                        co_occurrences = metadata.get('co_occurrences', {})
                        co_occurrences[other_id] = co_occurrences.get(other_id, 0) + 1
                        metadata['co_occurrences'] = co_occurrences

                        write_memory_file(filepath, metadata, content)
                        pairs_updated += 1
                        break

    PENDING_COOCCURRENCE_FILE.unlink(missing_ok=True)

    print(f"Processed co-occurrences: {len(retrieved)} memories, {pairs_updated // 2} pairs updated")
    return pairs_updated // 2


# --- V3.0 Edge Provenance System (credit: SpindriftMend PR #5) ---

def _get_edges_file() -> Path:
    """Path to v3.0 edges with provenance."""
    return MEMORY_ROOT / ".edges_v3.json"


def _load_edges_v3() -> dict[tuple[str, str], dict]:
    """Load v3.0 edges with full provenance."""
    edges_file = _get_edges_file()
    if not edges_file.exists():
        return {}

    try:
        with open(edges_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {tuple(k.split('|')): v for k, v in data.items()}
    except (json.JSONDecodeError, KeyError):
        return {}


def _save_edges_v3(edges: dict[tuple[str, str], dict]):
    """Save v3.0 edges with provenance to disk."""
    edges_file = _get_edges_file()
    data = {'|'.join(k): v for k, v in edges.items()}
    with open(edges_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


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
    edges = _load_edges_v3()
    pair = tuple(sorted([id1, id2]))

    if pair not in edges:
        edges[pair] = {
            'observations': [],
            'belief': 0.0,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

    obs = _create_observation(
        source_type=source_type,
        weight=weight,
        trust_tier=trust_tier,
        **source_kwargs
    )
    edges[pair]['observations'].append(obs)

    edges[pair]['belief'] = aggregate_belief(edges[pair]['observations'])
    edges[pair]['last_updated'] = datetime.now(timezone.utc).isoformat()

    _save_edges_v3(edges)

    return edges[pair]


def migrate_to_v3():
    """Migrate legacy in-file co-occurrences to v3.0 edges format."""
    edges_file = _get_edges_file()

    if edges_file.exists():
        print("v3.0 edges file already exists. Skipping migration.")
        return

    edges = {}
    migration_time = datetime.now(timezone.utc).isoformat()
    migrated_pairs = set()

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

                edges[pair] = {
                    'observations': [
                        {
                            'id': str(uuid.uuid4()),
                            'observed_at': migration_time,
                            'source': {
                                'type': 'legacy_migration',
                                'session_id': None,
                                'agent': get_agent_name(),
                                'note': f'Migrated from v2.x count={count}'
                            },
                            'weight': float(count),
                            'trust_tier': 'self'
                        }
                    ],
                    'belief': float(count),
                    'last_updated': migration_time
                }
                migrated_pairs.add(pair)

    if edges:
        _save_edges_v3(edges)
        print(f"Migrated {len(edges)} edges to v3.0 format.")
    else:
        print("No co-occurrences found to migrate.")


# --- Pair decay ---

def _get_recall_count(memory_id: str) -> int:
    """Get the recall_count for a memory by ID. Returns 0 if not found."""
    for directory in ALL_DIRS:
        if not directory.exists():
            continue
        for filepath in directory.glob(f"*-{memory_id}.md"):
            metadata, _ = parse_memory_file(filepath)
            return metadata.get('recall_count', 0)
    return 0


def _get_memory_metadata(memory_id: str) -> Optional[dict]:
    """Get metadata for a memory by ID."""
    for directory in ALL_DIRS:
        if not directory.exists():
            continue
        for filepath in directory.glob(f"*-{memory_id}.md"):
            metadata, _ = parse_memory_file(filepath)
            return metadata
    return None


def _calculate_effective_decay(memory_id: str, other_id: str) -> float:
    """
    Calculate effective decay rate based on access counts and retrieval success.
    Credit: FadeMem paper (access frequency), MemRL/MemEvolve (self-evolution)
    """
    base_decay = PAIR_DECAY_RATE

    if ACCESS_WEIGHTED_DECAY:
        recall_1 = _get_recall_count(memory_id)
        recall_2 = _get_recall_count(other_id)
        avg_recall = (recall_1 + recall_2) / 2
        base_decay = PAIR_DECAY_RATE / (1 + math.log(1 + avg_recall))

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
    """
    retrieved = session_state.get_retrieved_list()
    reinforced_pairs = set()
    for i, id1 in enumerate(retrieved):
        for id2 in retrieved[i + 1:]:
            reinforced_pairs.add(tuple(sorted([id1, id2])))

    pairs_decayed = 0
    pairs_pruned = 0

    for directory in ALL_DIRS:
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
                pair = tuple(sorted([memory_id, other_id]))

                if pair not in reinforced_pairs:
                    effective_decay = _calculate_effective_decay(memory_id, other_id)
                    new_count = count - effective_decay

                    if new_count <= 0:
                        to_remove.append(other_id)
                        pairs_pruned += 1
                    else:
                        co_occurrences[other_id] = new_count
                        pairs_decayed += 1
                    updated = True

            for other_id in to_remove:
                del co_occurrences[other_id]

            if updated:
                metadata['co_occurrences'] = co_occurrences
                write_memory_file(filepath, metadata, content)

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
    cache = {}
    for directory in ALL_DIRS:
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
    Apply soft decay to edges in edges_v3.json.
    v5.0: Dimensional decay — edges outside the session's active context
    get dramatically reduced decay (INACTIVE_CONTEXT_FACTOR).

    Credit: Multi-graph RFC (Issue #19), SpindriftMind joint design
    """
    retrieved = session_state.get_retrieved_list()
    reinforced_pairs = set()
    for i, id1 in enumerate(retrieved):
        for id2 in retrieved[i + 1:]:
            reinforced_pairs.add(tuple(sorted([id1, id2])))

    edges = _load_edges_v3()
    if not edges:
        print("Pair decay (v3): 0 decayed, 0 pruned (no edges)")
        return 0, 0

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
    to_remove = []
    now = datetime.now(timezone.utc).isoformat()

    for pair_key, edge_data in edges.items():
        normalized = tuple(sorted(pair_key))

        if normalized not in reinforced_pairs:
            base_decay = _calculate_effective_decay_cached(
                pair_key[0], pair_key[1], metadata_cache
            )

            # v5.0: Dimensional decay
            # If session has dimension data, check overlap
            if session_dims:
                edge_dims = _get_edge_dimensions(pair_key, edge_data, metadata_cache)
                if edge_dims and not _has_dimension_overlap(edge_dims, session_dims):
                    # Edge is in a different context — dramatically reduce decay
                    base_decay *= INACTIVE_CONTEXT_FACTOR
                    edges_protected += 1

            old_belief = edge_data.get('belief', 1.0)
            new_belief = old_belief - base_decay

            if new_belief <= 0:
                to_remove.append(pair_key)
                edges_pruned += 1
            else:
                edge_data['belief'] = new_belief
                edge_data['last_updated'] = now
                edges_decayed += 1

    for pair_key in to_remove:
        del edges[pair_key]

    _save_edges_v3(edges)

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
