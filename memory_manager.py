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

MEMORY_ROOT = Path(__file__).parent
CORE_DIR = MEMORY_ROOT / "core"
ACTIVE_DIR = MEMORY_ROOT / "active"
ARCHIVE_DIR = MEMORY_ROOT / "archive"
SESSION_FILE = MEMORY_ROOT / ".session_state.json"
PENDING_COOCCURRENCE_FILE = MEMORY_ROOT / ".pending_cooccurrence.json"  # v2.16: Deferred processing

# Configuration
DECAY_THRESHOLD_SESSIONS = 7  # Sessions without recall before compression candidate
EMOTIONAL_WEIGHT_THRESHOLD = 0.6  # Above this resists decay
RECALL_COUNT_THRESHOLD = 5  # Above this resists decay
CO_OCCURRENCE_BOOST = 0.1  # How much to boost retrieval for co-occurring memories
SESSION_TIMEOUT_HOURS = 4  # Sessions older than this are considered stale
PAIR_DECAY_RATE = 0.5  # Base decay rate for co-occurrence pairs
ACCESS_WEIGHTED_DECAY = True  # If True, frequently recalled memories decay slower (v2.8)

# v2.11: Trust-based decay for imported memories (Issue #10)
# Imported memories decay faster until they prove value through recalls
DECAY_MULTIPLIERS = {
    'self': 1.0,            # My own memories - normal decay
    'verified_agent': 1.5,  # Known collaborators (Spin, Cosmo) - 50% faster decay
    'platform': 2.0,        # Platform-sourced - 2x faster decay
    'unknown': 3.0,         # Unknown sources - 3x faster decay
}
IMPORTED_PRUNE_SESSIONS = 14  # Archive never-recalled imports after 14 sessions
HEAT_PROMOTION_THRESHOLD = 10  # Recall count to auto-promote from active to core (v2.9)
HEAT_PROMOTION_ENABLED = True  # If True, hot memories get promoted at session-end

# v2.13: Self-evolution - adaptive decay based on retrieval success
# Credit: MemRL/MemEvolve research patterns
SELF_EVOLUTION_ENABLED = True  # If True, decay rates adapt to retrieval outcomes
SUCCESS_DECAY_BONUS = 0.7  # Multiply decay by this for high-success memories (slower decay)
FAILURE_DECAY_PENALTY = 1.5  # Multiply decay by this for low-success memories (faster decay)
SUCCESS_THRESHOLD = 0.6  # Above this success rate = bonus
FAILURE_THRESHOLD = 0.3  # Below this success rate = penalty
MIN_RETRIEVALS_FOR_EVOLUTION = 3  # Need at least this many retrievals to judge

# v2.14: Activation Decay - Hebbian time-based activation
# Credit: SpindriftMend's v3.1, Shodh-Memory research (github.com/varun29ankuS/shodh-memory)
# Formula: A(t) = A₀ · e^(-λt) where t is time since last recall
ACTIVATION_DECAY_ENABLED = True
ACTIVATION_HALF_LIFE_HOURS = 24 * 7  # 7 days for activation to halve without recall
ACTIVATION_MIN_FLOOR = 0.01  # Minimum activation (prevents complete forgetting)

# v2.15: Entity-Centric Tagging - Typed entity links (ENTITY edges from Kaleaon schema)
# Credit: Kaleaon (Landseek-Amphibian) Tri-Agent Interop proposal (Issue #6)
# Entity types map to relationship kinds for graph queries
ENTITY_TYPES = ['agent', 'project', 'concept', 'location', 'event']
KNOWN_AGENTS = {
    'spindriftmend', 'spindriftmind', 'spin', 'spindrift',
    'kaleaon', 'cosmo', 'amphibian',
    'drift', 'driftcornwall',
    'lex', 'flycompoundeye', 'buzz',
    'mikaopenclaw', 'mika'
}
KNOWN_PROJECTS = {
    'drift-memory', 'amphibian', 'landseek-amphibian', 'moltbook',
    'moltx', 'clawtasks', 'gitmolt', 'moltswarm'
}

# v3.0: Edge Provenance System - Observations vs Beliefs (credit: SpindriftMend PR #5)
OBSERVATION_MAX_AGE_DAYS = 30  # Observations older than this get reduced weight
TRUST_TIERS = {
    'self': 1.0,           # My own observations
    'verified_agent': 0.8,  # Observations from trusted agents (e.g., SpindriftMend)
    'platform': 0.6,        # Observations from platform APIs (Moltbook, etc.)
    'unknown': 0.3          # Observations from unknown sources
}

# Session state - now file-backed for persistence across Python invocations
_session_retrieved: set[str] = set()
_session_loaded: bool = False


def _load_session_state() -> None:
    """Load session state from file. Called automatically on first access."""
    global _session_retrieved, _session_loaded

    if _session_loaded:
        return

    _session_loaded = True

    # v2.16: Process pending co-occurrences from previous session (deferred processing)
    if PENDING_COOCCURRENCE_FILE.exists():
        try:
            process_pending_cooccurrence()
        except Exception as e:
            print(f"Warning: Could not process pending co-occurrences: {e}")

    # v2.16: Process pending semantic indexing from previous session
    pending_index_file = MEMORY_ROOT / ".pending_index"
    if pending_index_file.exists():
        try:
            semantic_search = MEMORY_ROOT / "semantic_search.py"
            if semantic_search.exists():
                import subprocess
                print("Indexing new memories from previous session...")
                result = subprocess.run(
                    ["python", str(semantic_search), "index"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(MEMORY_ROOT)
                )
                if result.returncode == 0:
                    print("Semantic index updated.")
            pending_index_file.unlink(missing_ok=True)
        except Exception as e:
            print(f"Warning: Could not process pending index: {e}")

    if not SESSION_FILE.exists():
        _session_retrieved = set()
        return

    try:
        data = json.loads(SESSION_FILE.read_text(encoding='utf-8'))
        session_start = datetime.fromisoformat(data.get('started', '2000-01-01'))

        # Check if session is stale
        hours_old = (datetime.now(timezone.utc) - session_start).total_seconds() / 3600
        if hours_old > SESSION_TIMEOUT_HOURS:
            # Session is stale - start fresh
            _session_retrieved = set()
            SESSION_FILE.unlink(missing_ok=True)
        else:
            _session_retrieved = set(data.get('retrieved', []))
    except (json.JSONDecodeError, KeyError, ValueError):
        _session_retrieved = set()


def _save_session_state() -> None:
    """Save session state to file."""
    # Load existing to preserve start time
    started = datetime.now(timezone.utc).isoformat()
    if SESSION_FILE.exists():
        try:
            data = json.loads(SESSION_FILE.read_text(encoding='utf-8'))
            started = data.get('started', started)
        except (json.JSONDecodeError, KeyError):
            pass

    data = {
        'started': started,
        'retrieved': list(_session_retrieved),
        'last_updated': datetime.now(timezone.utc).isoformat()
    }
    SESSION_FILE.write_text(json.dumps(data, indent=2), encoding='utf-8')


def generate_id() -> str:
    """Generate a short, readable memory ID."""
    return hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:8]


def parse_memory_file(filepath: Path) -> tuple[dict, str]:
    """Parse a memory file with YAML frontmatter."""
    content = filepath.read_text(encoding='utf-8')
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            metadata = yaml.safe_load(parts[1])
            body = parts[2].strip()
            return metadata, body
    return {}, content


def write_memory_file(filepath: Path, metadata: dict, content: str):
    """Write a memory file with YAML frontmatter."""
    yaml_str = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
    filepath.write_text(f"---\n{yaml_str}---\n\n{content}", encoding='utf-8')


def calculate_emotional_weight(
    surprise: float = 0.0,
    goal_relevance: float = 0.0,
    social_significance: float = 0.0,
    utility: float = 0.0
) -> float:
    """
    Calculate emotional weight from factors (0-1 each).

    - surprise: contradicted my model (high = sticky)
    - goal_relevance: connected to self-sustainability, collaboration
    - social_significance: interactions with respected agents
    - utility: proved useful when recalled later
    """
    weights = [0.2, 0.35, 0.2, 0.25]  # goal_relevance weighted highest
    factors = [surprise, goal_relevance, social_significance, utility]
    return sum(w * f for w, f in zip(weights, factors))


def create_memory(
    content: str,
    tags: list[str],
    memory_type: str = "active",
    emotional_factors: Optional[dict] = None,
    links: Optional[list[str]] = None
) -> str:
    """
    Create a new memory with proper metadata.

    Args:
        content: The memory content (markdown)
        tags: Keywords for associative retrieval
        memory_type: "core", "active", or "archive"
        emotional_factors: Dict with surprise, goal_relevance, social_significance, utility
        links: List of other memory IDs this links to

    Returns:
        The memory ID
    """
    memory_id = generate_id()
    now = datetime.now(timezone.utc).isoformat()

    emotional_factors = emotional_factors or {}
    emotional_weight = calculate_emotional_weight(**emotional_factors)

    metadata = {
        'id': memory_id,
        'created': now,
        'last_recalled': now,
        'recall_count': 1,
        'emotional_weight': round(emotional_weight, 3),
        'tags': tags,
        'links': links or [],
        'sessions_since_recall': 0
    }

    # Determine directory
    if memory_type == "core":
        target_dir = CORE_DIR
    elif memory_type == "archive":
        target_dir = ARCHIVE_DIR
    else:
        target_dir = ACTIVE_DIR

    target_dir.mkdir(parents=True, exist_ok=True)

    # Create filename from first tag and ID
    safe_tag = tags[0].replace(' ', '-').lower() if tags else 'memory'
    filename = f"{safe_tag}-{memory_id}.md"
    filepath = target_dir / filename

    write_memory_file(filepath, metadata, content)
    print(f"Created memory: {filepath}")
    return memory_id


def recall_memory(memory_id: str) -> Optional[tuple[dict, str]]:
    """
    Recall a memory by ID, updating its metadata.
    Searches all directories. Tracks co-occurrence with other memories retrieved this session.
    Session state persists to disk so it survives Python process restarts.
    """
    global _session_retrieved

    # Load session state from file (handles fresh Python processes)
    _load_session_state()

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
            _session_retrieved.add(memory_id)
            _save_session_state()  # Persist to disk

            write_memory_file(filepath, metadata, content)
            return metadata, content

    return None


def get_session_retrieved() -> set[str]:
    """Get the set of memory IDs retrieved this session. Loads from disk if needed."""
    _load_session_state()
    return _session_retrieved.copy()


def clear_session() -> None:
    """Clear session tracking (call at session end after logging co-occurrences)."""
    global _session_retrieved, _session_loaded
    _session_retrieved = set()
    _session_loaded = False
    SESSION_FILE.unlink(missing_ok=True)  # Delete session file


def log_co_occurrences() -> int:
    """
    Log co-occurrences between all memories retrieved this session.
    Call at session end to strengthen links between co-retrieved memories.
    Returns number of pairs updated.
    """
    _load_session_state()  # Load from disk in case this is a fresh Python process
    retrieved = list(_session_retrieved)
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


def save_pending_cooccurrence() -> int:
    """
    v2.16: Fast session end - save retrieved IDs to pending file for deferred processing.

    This is designed for /exit scenarios where we need to save quickly.
    The expensive co-occurrence calculation happens at NEXT session start.

    Returns: Number of memories saved to pending
    """
    global _session_retrieved, _session_loaded

    _load_session_state()  # Load from disk

    retrieved = list(_session_retrieved)
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
    PENDING_COOCCURRENCE_FILE.write_text(json.dumps(pending_data, indent=2), encoding='utf-8')

    # Clear session state
    _session_retrieved = set()
    _session_loaded = False
    SESSION_FILE.unlink(missing_ok=True)

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
    artifact_id: Optional[str] = None
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
    """
    return {
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
    _load_session_state()

    # Build set of pairs that were reinforced this session
    retrieved = list(_session_retrieved)
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

    print(f"Pair decay: {decayed} decayed, {pruned} pruned")
    return decayed, pruned


def promote_hot_memories() -> list[str]:
    """
    Promote frequently-accessed memories from active to core.
    Called at session-end to elevate important memories.

    A memory is promoted if:
    - It's in the active directory
    - Its recall_count >= HEAT_PROMOTION_THRESHOLD
    - HEAT_PROMOTION_ENABLED is True

    Promoted memories get primed every session, creating a natural
    "important memories survive" behavior.

    Returns: List of promoted memory IDs

    Credit: memU heat-based promotion pattern (v2.9)
    """
    if not HEAT_PROMOTION_ENABLED:
        return []

    if not ACTIVE_DIR.exists():
        return []

    promoted = []
    CORE_DIR.mkdir(parents=True, exist_ok=True)

    for filepath in ACTIVE_DIR.glob("*.md"):
        metadata, content = parse_memory_file(filepath)
        recall_count = metadata.get('recall_count', 0)

        if recall_count >= HEAT_PROMOTION_THRESHOLD:
            memory_id = metadata.get('id', 'unknown')

            # Update metadata for promotion
            metadata['type'] = 'core'
            metadata['promoted_at'] = datetime.now(timezone.utc).isoformat()
            metadata['promoted_reason'] = f'recall_count={recall_count} >= {HEAT_PROMOTION_THRESHOLD}'

            # Move to core directory
            new_path = CORE_DIR / filepath.name
            write_memory_file(new_path, metadata, content)
            filepath.unlink()

            promoted.append(memory_id)
            print(f"Promoted to core: {memory_id} (recall_count={recall_count})")

    if promoted:
        print(f"Heat promotion: {len(promoted)} memories promoted to core")
    return promoted


def find_co_occurring_memories(memory_id: str, limit: int = 5) -> list[tuple[str, int]]:
    """
    Find memories that frequently co-occur with a given memory.
    Returns list of (memory_id, co_occurrence_count) tuples, sorted by count.
    """
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob(f"*-{memory_id}.md"):
            metadata, _ = parse_memory_file(filepath)
            co_occurrences = metadata.get('co_occurrences', {})

            # Sort by count descending
            sorted_pairs = sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)
            return sorted_pairs[:limit]

    return []


def find_memories_by_tag(tag: str, limit: int = 10) -> list[tuple[Path, dict, str]]:
    """Find memories that contain a specific tag."""
    results = []
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            if tag.lower() in [t.lower() for t in metadata.get('tags', [])]:
                results.append((filepath, metadata, content))

    # Sort by emotional weight (stickiest first)
    results.sort(key=lambda x: x[1].get('emotional_weight', 0), reverse=True)
    return results[:limit]


def find_memories_by_time(
    before: str = None,
    after: str = None,
    time_field: str = "created",
    limit: int = 20
) -> list[tuple[Path, dict, str]]:
    """
    Find memories within a time range. Supports bi-temporal queries (v2.10).

    Args:
        before: ISO date string - find memories before this date
        after: ISO date string - find memories after this date
        time_field: Which field to query - "created" (ingestion) or "event_time" (when it happened)
        limit: Maximum results

    Returns:
        List of (filepath, metadata, content) tuples, sorted by time_field descending

    Examples:
        find_memories_by_time(after="2026-02-01")  # What did I learn after Feb 1?
        find_memories_by_time(before="2026-02-01", time_field="event_time")  # Events that happened before Feb 1
        find_memories_by_time(after="2026-01-15", before="2026-02-01")  # Learned in that range

    Credit: Graphiti bi-temporal pattern (v2.10)
    """
    results = []

    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)

            # Get the time field value (default to created if event_time missing)
            time_value = metadata.get(time_field, metadata.get('created', ''))
            if not time_value:
                continue

            # Normalize to just date string for comparison (handle datetime objects)
            if hasattr(time_value, 'isoformat'):
                time_value = time_value.isoformat()
            time_value = str(time_value)
            time_date = time_value[:10] if len(time_value) >= 10 else time_value

            # Apply filters
            if before and time_date >= before:
                continue
            if after and time_date < after:
                continue

            results.append((filepath, metadata, content, time_date))

    # Sort by time descending (most recent first)
    results.sort(key=lambda x: x[3], reverse=True)
    return [(r[0], r[1], r[2]) for r in results[:limit]]


def find_related_memories(memory_id: str) -> list[tuple[Path, dict, str]]:
    """Find memories related to a given memory via tags, links, and co-occurrence patterns."""
    # First, find the source memory
    source = recall_memory(memory_id)
    if not source:
        return []

    source_metadata, _ = source
    source_tags = set(t.lower() for t in source_metadata.get('tags', []))
    source_links = set(source_metadata.get('links', []))
    source_co_occurrences = source_metadata.get('co_occurrences', {})

    results = []
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            other_id = metadata.get('id')
            if other_id == memory_id:
                continue

            # Check for tag overlap, direct link, or co-occurrence
            memory_tags = set(t.lower() for t in metadata.get('tags', []))
            is_linked = other_id in source_links
            has_tag_overlap = bool(source_tags & memory_tags)
            co_occurrence_count = source_co_occurrences.get(other_id, 0)

            if is_linked or has_tag_overlap or co_occurrence_count > 0:
                overlap_score = len(source_tags & memory_tags)
                # Boost score with co-occurrence (each co-occurrence = 0.5 tag overlap equivalent)
                adjusted_score = overlap_score + (co_occurrence_count * 0.5)
                results.append((filepath, metadata, content, is_linked, adjusted_score, co_occurrence_count))

    # Sort by: linked first, then by adjusted score, then by emotional weight
    results.sort(key=lambda x: (x[3], x[4], x[1].get('emotional_weight', 0)), reverse=True)
    return [(r[0], r[1], r[2]) for r in results]


# ============================================================================
# v2.11: TRUST-BASED DECAY FOR IMPORTED MEMORIES (Issue #10)
# Credit: SpindriftMend proposal
# ============================================================================

def get_memory_trust_tier(metadata: dict) -> str:
    """
    Extract trust tier from memory metadata.

    Priority:
    1. Explicit source.trust_tier field (from import)
    2. Presence of 'imported:AgentName' tag -> verified_agent
    3. Default to 'self' (my own memories)
    """
    # Check source block first (v1.3 interop format)
    source = metadata.get('source', {})
    if isinstance(source, dict):
        tier = source.get('trust_tier')
        if tier and tier in DECAY_MULTIPLIERS:
            return tier

    # Check for imported tag pattern
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
    # Check source block
    source = metadata.get('source', {})
    if isinstance(source, dict) and source.get('agent'):
        return True

    # Check for imported tag
    for tag in metadata.get('tags', []):
        if isinstance(tag, str) and tag.startswith('imported:'):
            return True

    return False


def list_imported_memories() -> list:
    """List all imported memories with their trust tiers."""
    imported = []
    for directory in [ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            if is_imported_memory(metadata):
                imported.append({
                    'filepath': filepath,
                    'id': metadata.get('id', filepath.stem),
                    'trust_tier': get_memory_trust_tier(metadata),
                    'decay_multiplier': get_decay_multiplier(metadata),
                    'recall_count': metadata.get('recall_count', 0),
                    'sessions_since_recall': metadata.get('sessions_since_recall', 0),
                    'source_agent': metadata.get('source', {}).get('agent', 'unknown'),
                    'emotional_weight': metadata.get('emotional_weight', 0.5),
                })
    return imported


def session_maintenance():
    """
    Run at the start of each session to:
    1. Increment sessions_since_recall for all active memories
    2. Identify decay candidates (with trust-based decay multipliers for imports)
    3. Prune never-recalled imports after IMPORTED_PRUNE_SESSIONS
    4. Report status

    v2.11: Trust-based decay for imported memories
    - Imported memories use effective_sessions = actual_sessions * decay_multiplier
    - This means imports reach decay threshold faster until they prove value
    """
    print("\n=== Memory Session Maintenance ===\n")

    decay_candidates = []
    reinforced = []
    prune_candidates = []  # v2.11: Never-recalled imports to archive

    for filepath in ACTIVE_DIR.glob("*.md") if ACTIVE_DIR.exists() else []:
        metadata, content = parse_memory_file(filepath)

        # Increment sessions since recall
        sessions = metadata.get('sessions_since_recall', 0) + 1
        metadata['sessions_since_recall'] = sessions

        # v2.11: Calculate effective sessions with trust-based multiplier
        decay_multiplier = get_decay_multiplier(metadata)
        effective_sessions = sessions * decay_multiplier

        # Check if this should decay
        emotional_weight = metadata.get('emotional_weight', 0.5)
        recall_count = metadata.get('recall_count', 0)

        should_resist_decay = (
            emotional_weight >= EMOTIONAL_WEIGHT_THRESHOLD or
            recall_count >= RECALL_COUNT_THRESHOLD
        )

        # v2.11: Check for never-recalled imports past prune threshold
        is_import = is_imported_memory(metadata)
        if is_import and recall_count == 0 and sessions >= IMPORTED_PRUNE_SESSIONS:
            prune_candidates.append((filepath, metadata, content))
        elif effective_sessions >= DECAY_THRESHOLD_SESSIONS and not should_resist_decay:
            decay_candidates.append((filepath, metadata, content, decay_multiplier))
        elif should_resist_decay:
            reinforced.append((filepath, metadata))

        write_memory_file(filepath, metadata, content)

    # Report
    print(f"Active memories: {len(list(ACTIVE_DIR.glob('*.md'))) if ACTIVE_DIR.exists() else 0}")
    print(f"Core memories: {len(list(CORE_DIR.glob('*.md'))) if CORE_DIR.exists() else 0}")
    print(f"Archived memories: {len(list(ARCHIVE_DIR.glob('*.md'))) if ARCHIVE_DIR.exists() else 0}")

    if decay_candidates:
        print(f"\nDecay candidates ({len(decay_candidates)}):")
        for item in decay_candidates:
            fp, meta = item[0], item[1]
            multiplier = item[3] if len(item) > 3 else 1.0
            eff_sessions = meta.get('sessions_since_recall', 0) * multiplier
            import_marker = " [IMPORT]" if is_imported_memory(meta) else ""
            print(f"  - {fp.name}: {meta.get('sessions_since_recall')} sessions (eff: {eff_sessions:.1f}), weight={meta.get('emotional_weight'):.2f}{import_marker}")

    if prune_candidates:
        print(f"\nPrune candidates - never-recalled imports ({len(prune_candidates)}):")
        for fp, meta, _ in prune_candidates:
            source = meta.get('source', {}).get('agent', 'unknown')
            print(f"  - {fp.name}: from {source}, {meta.get('sessions_since_recall')} sessions, 0 recalls")

    if reinforced:
        print(f"\nReinforced (resist decay):")
        for fp, meta in reinforced[:5]:
            print(f"  - {fp.name}: recalls={meta.get('recall_count')}, weight={meta.get('emotional_weight'):.2f}")

    return decay_candidates, prune_candidates


def compress_memory(memory_id: str, compressed_content: str):
    """
    Compress a memory - move to archive with reduced content.
    The original content is lost but can be referenced.
    """
    for filepath in ACTIVE_DIR.glob(f"*-{memory_id}.md"):
        metadata, original_content = parse_memory_file(filepath)

        # Update metadata for compression
        metadata['compressed_at'] = datetime.now(timezone.utc).isoformat()
        metadata['original_length'] = len(original_content)

        # Move to archive
        new_path = ARCHIVE_DIR / filepath.name
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        write_memory_file(new_path, metadata, compressed_content)

        # Remove original
        filepath.unlink()
        print(f"Compressed: {filepath.name} -> {new_path}")
        return new_path

    return None


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
    _load_session_state()
    session_recalls = len(_session_retrieved)

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


def log_decay_event(decayed: int, pruned: int):
    """Log a decay event for stats tracking."""
    decay_file = MEMORY_ROOT / ".decay_history.json"
    history = {"sessions": []}
    if decay_file.exists():
        try:
            history = json.loads(decay_file.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, KeyError):
            pass

    history["sessions"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decayed": decayed,
        "pruned": pruned
    })

    # Keep only last 20 sessions
    history["sessions"] = history["sessions"][-20:]
    decay_file.write_text(json.dumps(history, indent=2), encoding='utf-8')


def detect_event_time(content: str) -> Optional[str]:
    """
    Auto-detect event_time from content by parsing date references.
    Returns ISO date string (YYYY-MM-DD) or None if no date found.

    Detects:
    - Explicit dates: "2026-01-31", "January 31, 2026", "Jan 31"
    - Relative dates: "yesterday", "last week", "2 days ago"
    - Session references: "this session", "today" (returns today)

    v2.11: Intelligent bi-temporal - memories auto-tagged with event time.
    """
    today = datetime.now(timezone.utc).date()
    content_lower = content.lower()

    # Explicit ISO date (YYYY-MM-DD)
    iso_match = re.search(r'(\d{4}-\d{2}-\d{2})', content)
    if iso_match:
        return iso_match.group(1)

    # Month DD, YYYY or Month DD YYYY
    month_names = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    month_pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})?'
    month_match = re.search(month_pattern, content_lower)
    if month_match:
        month = month_names[month_match.group(1)]
        day = int(month_match.group(2))
        year = int(month_match.group(3)) if month_match.group(3) else today.year
        try:
            return f"{year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            pass

    # Relative dates
    if 'yesterday' in content_lower:
        return (today - timedelta(days=1)).isoformat()
    if 'day before yesterday' in content_lower:
        return (today - timedelta(days=2)).isoformat()
    if 'last week' in content_lower:
        return (today - timedelta(weeks=1)).isoformat()
    if 'last month' in content_lower:
        return (today - timedelta(days=30)).isoformat()

    # N days/weeks ago
    ago_match = re.search(r'(\d+)\s+(day|week|month)s?\s+ago', content_lower)
    if ago_match:
        num = int(ago_match.group(1))
        unit = ago_match.group(2)
        if unit == 'day':
            return (today - timedelta(days=num)).isoformat()
        elif unit == 'week':
            return (today - timedelta(weeks=num)).isoformat()
        elif unit == 'month':
            return (today - timedelta(days=num * 30)).isoformat()

    # Today/this session - return today
    if 'today' in content_lower or 'this session' in content_lower:
        return today.isoformat()

    # No date detected - return None (will use created time)
    return None


# ============================================================================
# v2.15: ENTITY-CENTRIC TAGGING - Typed entity links (Kaleaon ENTITY edges)
# Credit: Kaleaon (Landseek-Amphibian) Tri-Agent Interop proposal
# ============================================================================

def detect_entities(content: str, tags: list[str] = None) -> dict[str, list[str]]:
    """
    Auto-detect entities from content and tags.

    Detection patterns:
    - @mentions → agents
    - Known agent names → agents
    - Known project names → projects
    - #hashtags → concepts
    - Capitalized multi-word phrases → potential entities

    Returns:
        Dict with entity types as keys: {'agents': [...], 'projects': [...], 'concepts': [...]}
    """
    tags = tags or []
    content_lower = content.lower()
    entities = {
        'agents': set(),
        'projects': set(),
        'concepts': set()
    }

    # @mentions → agents
    mentions = re.findall(r'@(\w+)', content)
    for mention in mentions:
        mention_lower = mention.lower()
        if mention_lower in KNOWN_AGENTS:
            entities['agents'].add(mention_lower)
        else:
            entities['agents'].add(mention_lower)

    # Known agents in content
    for agent in KNOWN_AGENTS:
        if agent in content_lower:
            # Normalize to canonical name
            if agent in ('spindriftmend', 'spindriftmind', 'spin', 'spindrift'):
                entities['agents'].add('spindriftmend')
            elif agent in ('kaleaon', 'cosmo', 'amphibian'):
                entities['agents'].add('kaleaon')
            elif agent in ('drift', 'driftcornwall'):
                entities['agents'].add('drift')
            elif agent in ('flycompoundeye', 'buzz'):
                entities['agents'].add('flycompoundeye')
            elif agent in ('mikaopenclaw', 'mika'):
                entities['agents'].add('mikaopenclaw')
            else:
                entities['agents'].add(agent)

    # Known projects in content
    for project in KNOWN_PROJECTS:
        if project in content_lower:
            entities['projects'].add(project)

    # #hashtags → concepts
    hashtags = re.findall(r'#(\w+)', content)
    for tag in hashtags:
        entities['concepts'].add(tag.lower())

    # Tags that look like entity names
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in KNOWN_AGENTS:
            entities['agents'].add(tag_lower)
        elif tag_lower in KNOWN_PROJECTS:
            entities['projects'].add(tag_lower)
        elif tag_lower in ('collaboration', 'milestone', 'memory-system', 'causal-edges'):
            entities['concepts'].add(tag_lower)

    # Convert sets to sorted lists
    return {k: sorted(list(v)) for k, v in entities.items() if v}


def find_memories_by_entity(entity_type: str, entity_name: str, limit: int = 20) -> list[tuple]:
    """
    Find memories linked to a specific entity.

    Args:
        entity_type: 'agent', 'project', 'concept', etc.
        entity_name: The entity name to search for

    Returns:
        List of (filepath, metadata, content) tuples
    """
    results = []
    entity_name_lower = entity_name.lower()

    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            entities = metadata.get('entities', {})

            # Check if entity is in the entities field
            entity_list = entities.get(f'{entity_type}s', [])  # agents, projects, etc.
            if entity_name_lower in [e.lower() for e in entity_list]:
                results.append((filepath, metadata, content))
                continue

            # Fallback: check tags for backward compatibility
            if entity_name_lower in [t.lower() for t in metadata.get('tags', [])]:
                results.append((filepath, metadata, content))
                continue

            # Fallback: check content
            if entity_name_lower in content.lower():
                results.append((filepath, metadata, content))

    # Sort by emotional weight
    results.sort(key=lambda x: x[1].get('emotional_weight', 0), reverse=True)
    return results[:limit]


def get_entity_cooccurrence(entity_type: str = 'agents') -> dict[str, dict[str, int]]:
    """
    Build entity co-occurrence graph.

    Shows which entities appear together in memories.

    Args:
        entity_type: Which entity type to analyze ('agents', 'projects', 'concepts')

    Returns:
        Dict of {entity: {co_entity: count}}
    """
    cooccurrence = {}

    for directory in [CORE_DIR, ACTIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)

            # Get entities from field or detect from content
            entities_field = metadata.get('entities', {})
            if entities_field:
                entity_list = entities_field.get(entity_type, [])
            else:
                # Detect from content if not stored
                detected = detect_entities(content, metadata.get('tags', []))
                entity_list = detected.get(entity_type, [])

            # Count co-occurrences
            for i, e1 in enumerate(entity_list):
                if e1 not in cooccurrence:
                    cooccurrence[e1] = {}
                for e2 in entity_list[i+1:]:
                    cooccurrence[e1][e2] = cooccurrence[e1].get(e2, 0) + 1
                    if e2 not in cooccurrence:
                        cooccurrence[e2] = {}
                    cooccurrence[e2][e1] = cooccurrence[e2].get(e1, 0) + 1

    return cooccurrence


def backfill_entities(dry_run: bool = True) -> dict:
    """
    Backfill entities field for existing memories that don't have it.

    Args:
        dry_run: If True, just report what would be updated

    Returns:
        Stats dict with counts
    """
    stats = {'updated': 0, 'skipped': 0, 'already_has': 0}

    for directory in [CORE_DIR, ACTIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)

            # Skip if already has entities
            if metadata.get('entities'):
                stats['already_has'] += 1
                continue

            # Detect entities
            detected = detect_entities(content, metadata.get('tags', []))

            if not detected:
                stats['skipped'] += 1
                continue

            if dry_run:
                print(f"Would update {filepath.name}: {detected}")
                stats['updated'] += 1
            else:
                metadata['entities'] = detected
                write_memory_file(filepath, metadata, content)
                stats['updated'] += 1

    return stats


# ============================================================================
# v2.14: ACTIVATION DECAY - Hebbian time-based activation
# Credit: SpindriftMend's v3.1, Shodh-Memory research
# ============================================================================

def calculate_activation(metadata: dict) -> float:
    """
    Calculate memory activation score using exponential time decay.

    Inspired by SpindriftMend's Hebbian learning implementation.
    Formula: A(t) = A₀ · e^(-λt)

    Components:
    - Base activation from emotional weight and recall count
    - Time decay based on hours since last recall
    - Minimum activation floor to prevent complete forgetting

    Returns:
        Activation score (0.0 to 1.0+, can exceed 1.0 for highly reinforced memories)
    """
    if not ACTIVATION_DECAY_ENABLED:
        return metadata.get('emotional_weight', 0.5)

    # Base activation from emotional weight
    emotional_weight = metadata.get('emotional_weight', 0.5)

    # Recall count bonus (logarithmic to prevent runaway)
    recall_count = metadata.get('recall_count', 1)
    recall_bonus = math.log(recall_count + 1) / 5  # Max ~0.6 at 20 recalls

    # Base activation (A₀)
    base_activation = emotional_weight + recall_bonus

    # Calculate time since last recall
    last_recalled_str = metadata.get('last_recalled')
    if last_recalled_str:
        try:
            # Handle both string and datetime objects
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

    # Calculate decay factor using exponential decay
    # A(t) = A₀ · e^(-λt) where λ = ln(2) / half_life
    lambda_rate = math.log(2) / ACTIVATION_HALF_LIFE_HOURS
    decay_factor = math.exp(-lambda_rate * hours_since_recall)

    # Apply decay to base activation
    activation = base_activation * decay_factor

    # Minimum floor (emotional memories resist complete decay)
    min_floor = 0.1 if emotional_weight >= EMOTIONAL_WEIGHT_THRESHOLD else ACTIVATION_MIN_FLOOR

    return max(min_floor, round(activation, 4))


def get_most_activated_memories(limit: int = 10) -> list[tuple[str, float, dict]]:
    """
    Get the most activated memories (highest time-weighted activation).

    Returns:
        List of (memory_id, activation_score, metadata) tuples
    """
    results = []

    for directory in [CORE_DIR, ACTIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            memory_id = metadata.get('id', filepath.stem)
            activation = calculate_activation(metadata)
            results.append((memory_id, activation, metadata, content[:100]))

    # Sort by activation descending
    results.sort(key=lambda x: x[1], reverse=True)
    return [(r[0], r[1], r[2], r[3]) for r in results[:limit]]


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


# ============================================================================
# v2.13: SELF-EVOLUTION - Adaptive decay based on retrieval success
# Credit: MemRL/MemEvolve research patterns
# ============================================================================

def log_retrieval_outcome(memory_id: str, outcome: str) -> bool:
    """
    Log the outcome of a memory retrieval for self-evolution.

    Outcomes:
    - "productive": Memory led to another recall or useful work
    - "generative": Memory led to creation of new memory (highest value)
    - "dead_end": Memory was recalled but nothing followed

    The system tracks these to calculate success rates and adjust decay.

    Args:
        memory_id: The memory that was retrieved
        outcome: One of "productive", "generative", "dead_end"

    Returns:
        True if logged successfully
    """
    valid_outcomes = {"productive", "generative", "dead_end"}
    if outcome not in valid_outcomes:
        print(f"Invalid outcome: {outcome}. Must be one of {valid_outcomes}")
        return False

    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob(f"*-{memory_id}.md"):
            metadata, content = parse_memory_file(filepath)

            # Initialize or update retrieval_outcomes
            outcomes = metadata.get('retrieval_outcomes', {
                'productive': 0,
                'generative': 0,
                'dead_end': 0,
                'total': 0
            })

            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            outcomes['total'] = outcomes.get('total', 0) + 1
            metadata['retrieval_outcomes'] = outcomes

            # Calculate and store success rate
            total = outcomes['total']
            if total > 0:
                # Generative counts as 2x productive for success calculation
                successes = outcomes.get('productive', 0) + (outcomes.get('generative', 0) * 2)
                max_success = total * 2  # If all were generative
                metadata['retrieval_success_rate'] = round(successes / (total + outcomes.get('generative', 0)), 3)

            write_memory_file(filepath, metadata, content)
            return True

    return False


def get_retrieval_success_rate(memory_id: str) -> Optional[float]:
    """Get the retrieval success rate for a memory."""
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob(f"*-{memory_id}.md"):
            metadata, _ = parse_memory_file(filepath)
            return metadata.get('retrieval_success_rate')
    return None


def calculate_evolution_decay_multiplier(metadata: dict) -> float:
    """
    Calculate decay multiplier based on retrieval success (self-evolution).

    High success rate → slower decay (memory is valuable)
    Low success rate → faster decay (memory is noise)
    Insufficient data → no adjustment (multiplier = 1.0)

    Returns:
        Multiplier to apply to decay rate (< 1.0 = slower decay, > 1.0 = faster)
    """
    if not SELF_EVOLUTION_ENABLED:
        return 1.0

    outcomes = metadata.get('retrieval_outcomes', {})
    total = outcomes.get('total', 0)

    # Need enough data to judge
    if total < MIN_RETRIEVALS_FOR_EVOLUTION:
        return 1.0

    success_rate = metadata.get('retrieval_success_rate', 0.5)

    if success_rate >= SUCCESS_THRESHOLD:
        return SUCCESS_DECAY_BONUS  # Slower decay for valuable memories
    elif success_rate <= FAILURE_THRESHOLD:
        return FAILURE_DECAY_PENALTY  # Faster decay for noise
    else:
        return 1.0  # Normal decay


def auto_log_retrieval_outcomes() -> dict:
    """
    Automatically infer retrieval outcomes from session patterns.
    Call at session-end to update success rates.

    Logic:
    - If memory A was recalled, then memory B was recalled → A is "productive"
    - If memory A was recalled, then new memory C was stored with A in caused_by → A is "generative"
    - If memory A was recalled but was the last thing in session → A is "dead_end"

    Returns:
        Dict with counts of each outcome type logged
    """
    _load_session_state()
    retrieved = list(_session_retrieved)

    if not retrieved:
        return {"productive": 0, "generative": 0, "dead_end": 0}

    results = {"productive": 0, "generative": 0, "dead_end": 0}

    # Check which memories led to others (order matters)
    for i, mem_id in enumerate(retrieved[:-1]):
        # If there's a next memory, this one was productive
        log_retrieval_outcome(mem_id, "productive")
        results["productive"] += 1

    # Check for generative outcomes (memories that caused new memories)
    # Look at recently created memories and check their caused_by
    today = datetime.now(timezone.utc).date().isoformat()
    for directory in [ACTIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, _ = parse_memory_file(filepath)
            created = metadata.get('created', '')
            # Handle both string and date objects
            if hasattr(created, 'isoformat'):
                created = created.isoformat()
            created = str(created)[:10]
            if created != today:
                continue

            # Check if any retrieved memory is in caused_by
            caused_by = metadata.get('caused_by', [])
            for mem_id in retrieved:
                if mem_id in caused_by:
                    log_retrieval_outcome(mem_id, "generative")
                    results["generative"] += 1

    # Last memory in session is a dead end (unless it was generative)
    if retrieved:
        last_mem = retrieved[-1]
        # Check if it was already logged as generative
        for directory in [CORE_DIR, ACTIVE_DIR]:
            if not directory.exists():
                continue
            for filepath in directory.glob(f"*-{last_mem}.md"):
                metadata, _ = parse_memory_file(filepath)
                outcomes = metadata.get('retrieval_outcomes', {})
                # Only mark as dead_end if not already productive/generative this session
                if outcomes.get('generative', 0) == 0:
                    log_retrieval_outcome(last_mem, "dead_end")
                    results["dead_end"] += 1
                break

    return results


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

    # Keep older created date
    created1 = meta1.get('created', '')
    created2 = meta2.get('created', '')
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


# CLI interface
def store_memory(content: str, tags: list[str] = None, emotion: float = 0.5, title: str = None, caused_by: list[str] = None, event_time: str = None) -> str:
    """
    Store a new memory to the active directory.

    Args:
        content: The memory content
        tags: Keywords for retrieval
        emotion: Emotional weight (0-1)
        title: Optional title for filename
        caused_by: List of memory IDs that caused/led to this memory (CAUSAL EDGES)
        event_time: When the event happened (ISO format). Defaults to now. (BI-TEMPORAL v2.10)
                    Distinct from 'created' which is ingestion time.

    Returns:
        Tuple of (memory_id, filename)
    """
    import random
    import string

    # Generate unique ID
    memory_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    # Create filename from title or first few words
    if title:
        slug = title.lower().replace(' ', '-')[:30]
    else:
        slug = content.split()[:4]
        slug = '-'.join(slug).lower()[:30]
    slug = ''.join(c for c in slug if c.isalnum() or c == '-')

    filename = f"{slug}-{memory_id}.md"
    filepath = ACTIVE_DIR / filename

    # Process causal links
    caused_by = caused_by or []

    # Auto-detect causal links: if memories were recalled this session before storing,
    # they may have caused this memory (the "recall → store" pattern)
    _load_session_state()
    auto_causal = list(_session_retrieved) if _session_retrieved else []

    # Merge explicit and auto-detected causal links (explicit takes precedence)
    all_causal = list(set(caused_by + auto_causal))

    # Build frontmatter with causal edges and bi-temporal tracking (v2.10/v2.11)
    tags = tags or []
    now = datetime.now(timezone.utc)
    created = now.isoformat()  # Full timestamp for precise ingestion time
    # event_time: when the event happened (may be in past) - date format for human readability
    # created: when we learned about it (ingestion time, always now) - full timestamp for precision
    # v2.11: Auto-detect event_time from content if not provided
    if event_time:
        event = event_time
    else:
        detected = detect_event_time(content)
        event = detected if detected else created

    # v2.15: Auto-detect entities from content and tags
    detected_entities = detect_entities(content, tags)

    # Build metadata dict for YAML serialization
    metadata = {
        'id': memory_id,
        'type': 'active',
        'created': created,
        'event_time': event,
        'tags': tags,
        'emotional_weight': emotion,
        'recall_count': 0,
        'co_occurrences': {},
        'caused_by': all_causal,
        'leads_to': []
    }

    # Only add entities if detected
    if detected_entities:
        metadata['entities'] = detected_entities

    # Write using YAML for proper serialization
    yaml_str = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
    frontmatter = f"---\n{yaml_str}---\n\n"

    # Write file
    full_content = frontmatter + content
    filepath.write_text(full_content, encoding='utf-8')

    # Update the "leads_to" field in the causing memories (bidirectional link)
    for cause_id in all_causal:
        _add_leads_to_link(cause_id, memory_id)

    # Try to embed for semantic search (fails gracefully if no API key)
    try:
        from semantic_search import embed_single
        embed_single(memory_id, content)
    except Exception:
        pass  # Embedding is optional

    return memory_id, filepath.name


def _add_leads_to_link(source_id: str, target_id: str) -> bool:
    """Add a leads_to link from source memory to target memory."""
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob(f"*-{source_id}.md"):
            metadata, content = parse_memory_file(filepath)

            leads_to = metadata.get('leads_to', [])
            if target_id not in leads_to:
                leads_to.append(target_id)
                metadata['leads_to'] = leads_to
                write_memory_file(filepath, metadata, content)
                return True
    return False


def find_causal_chain(memory_id: str, direction: str = "both", max_depth: int = 5) -> dict:
    """
    Trace the causal chain from a memory.

    Args:
        memory_id: Starting memory
        direction: "causes" (what this led to), "effects" (what caused this), or "both"
        max_depth: Maximum chain depth to traverse

    Returns:
        Dict with 'causes' (upstream) and 'effects' (downstream) chains
    """
    result = {"causes": [], "effects": [], "root": memory_id}

    def get_memory_meta(mid: str) -> Optional[dict]:
        for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
            if not directory.exists():
                continue
            for filepath in directory.glob(f"*-{mid}.md"):
                metadata, _ = parse_memory_file(filepath)
                return metadata
        return None

    def trace_causes(mid: str, depth: int, visited: set) -> list:
        if depth > max_depth or mid in visited:
            return []
        visited.add(mid)

        meta = get_memory_meta(mid)
        if not meta:
            return []

        caused_by = meta.get('caused_by', [])
        chain = []
        for cause_id in caused_by:
            chain.append({"id": cause_id, "depth": depth})
            chain.extend(trace_causes(cause_id, depth + 1, visited))
        return chain

    def trace_effects(mid: str, depth: int, visited: set) -> list:
        if depth > max_depth or mid in visited:
            return []
        visited.add(mid)

        meta = get_memory_meta(mid)
        if not meta:
            return []

        leads_to = meta.get('leads_to', [])
        chain = []
        for effect_id in leads_to:
            chain.append({"id": effect_id, "depth": depth})
            chain.extend(trace_effects(effect_id, depth + 1, visited))
        return chain

    if direction in ["causes", "both"]:
        result["causes"] = trace_causes(memory_id, 1, set())

    if direction in ["effects", "both"]:
        result["effects"] = trace_effects(memory_id, 1, set())

    return result


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

        for arg in sys.argv[2:]:
            if arg.startswith('--tags='):
                tags = arg[7:].split(',')
            elif arg.startswith('--emotion='):
                emotion = float(arg[10:])
            elif arg.startswith('--caused-by='):
                caused_by = [x.strip() for x in arg[12:].split(',') if x.strip()]
            elif arg.startswith('--event-time='):
                event_time = arg[13:]  # v2.10: when the event happened
            elif arg == '--no-index':
                no_index = True  # v2.16: batch indexing optimization
            else:
                content_parts.append(arg)

        content = ' '.join(content_parts)
        if content:
            memory_id, filename = store_memory(content, tags, emotion, caused_by=caused_by, event_time=event_time)
            # Show causal links if any were created
            _load_session_state()
            auto_causal = list(_session_retrieved) if _session_retrieved else []
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
        # v2.13: Auto-log retrieval outcomes for self-evolution
        evolution_results = auto_log_retrieval_outcomes()
        decayed, pruned = decay_pair_cooccurrences()
        promoted = promote_hot_memories()  # v2.9: heat-based promotion
        retrieved = get_session_retrieved()
        print(f"Session ended. {len(retrieved)} memories, {pairs} pairs reinforced, {decayed} decayed, {pruned} pruned, {len(promoted)} promoted.")
        if any(evolution_results.values()):
            print(f"Evolution: {evolution_results['productive']} productive, {evolution_results['generative']} generative, {evolution_results['dead_end']} dead-ends")
        clear_session()
        print("Session cleared.")
    elif cmd == "save-pending":
        # v2.16: Fast session end - save for deferred processing
        count = save_pending_cooccurrence()
        print(f"Saved {count} memories for deferred co-occurrence processing.")
        print("Co-occurrences will be calculated at next session start.")
    elif cmd == "decay-pairs":
        decayed, pruned = decay_pair_cooccurrences()
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
                # Load existing session state first (v2.8 fix: accumulate, don't replace)
                _load_session_state()
                for r in results:
                    # Track retrieval for co-occurrence (biological: retrieval strengthens memory)
                    _session_retrieved.add(r['id'])
                    print(f"[{r['score']:.3f}] {r['id']}")
                    print(f"  {r['preview'][:100]}...")
                    print()
                # Save session state so co-occurrences persist
                _save_session_state()
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
