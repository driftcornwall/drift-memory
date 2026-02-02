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

MEMORY_ROOT = Path(__file__).parent
CORE_DIR = MEMORY_ROOT / "core"
ACTIVE_DIR = MEMORY_ROOT / "active"
ARCHIVE_DIR = MEMORY_ROOT / "archive"
SESSION_FILE = MEMORY_ROOT / ".session_state.json"

# Configuration
DECAY_THRESHOLD_SESSIONS = 7  # Sessions without recall before compression candidate
EMOTIONAL_WEIGHT_THRESHOLD = 0.6  # Above this resists decay
RECALL_COUNT_THRESHOLD = 5  # Above this resists decay
CO_OCCURRENCE_BOOST = 0.1  # How much to boost retrieval for co-occurring memories
SESSION_TIMEOUT_HOURS = 4  # Sessions older than this are considered stale
PAIR_DECAY_RATE = 0.5  # Base decay rate for co-occurrence pairs
ACCESS_WEIGHTED_DECAY = True  # If True, frequently recalled memories decay slower (v2.8)
HEAT_PROMOTION_THRESHOLD = 10  # Recall count to auto-promote from active to core (v2.9)
HEAT_PROMOTION_ENABLED = True  # If True, hot memories get promoted at session-end

# Session state - now file-backed for persistence across Python invocations
_session_retrieved: set[str] = set()
_session_loaded: bool = False


def _load_session_state() -> None:
    """Load session state from file. Called automatically on first access."""
    global _session_retrieved, _session_loaded

    if _session_loaded:
        return

    _session_loaded = True

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


def _get_recall_count(memory_id: str) -> int:
    """Get the recall_count for a memory by ID. Returns 0 if not found."""
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob(f"*-{memory_id}.md"):
            metadata, _ = parse_memory_file(filepath)
            return metadata.get('recall_count', 0)
    return 0


def _calculate_effective_decay(memory_id: str, other_id: str) -> float:
    """
    Calculate effective decay rate based on recall counts of both memories.
    Frequently recalled memories decay their pairs more slowly.

    Formula: effective_decay = PAIR_DECAY_RATE / (1 + log(1 + avg_recall_count))

    Examples:
    - avg_recall_count=0 → decay = 0.5 / 1 = 0.5 (normal)
    - avg_recall_count=9 → decay = 0.5 / (1 + log(10)) ≈ 0.22 (slower)
    - avg_recall_count=99 → decay = 0.5 / (1 + log(100)) ≈ 0.14 (much slower)

    Credit: FadeMem paper (arXiv:2601.18642) - adaptive decay based on access frequency
    """
    if not ACCESS_WEIGHTED_DECAY:
        return PAIR_DECAY_RATE

    recall_1 = _get_recall_count(memory_id)
    recall_2 = _get_recall_count(other_id)
    avg_recall = (recall_1 + recall_2) / 2

    # Using natural log for smooth scaling
    effective_decay = PAIR_DECAY_RATE / (1 + math.log(1 + avg_recall))
    return effective_decay


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

            # Normalize to just date for comparison
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


def session_maintenance():
    """
    Run at the start of each session to:
    1. Increment sessions_since_recall for all active memories
    2. Identify decay candidates
    3. Report status
    """
    print("\n=== Memory Session Maintenance ===\n")

    decay_candidates = []
    reinforced = []

    for filepath in ACTIVE_DIR.glob("*.md") if ACTIVE_DIR.exists() else []:
        metadata, content = parse_memory_file(filepath)

        # Increment sessions since recall
        sessions = metadata.get('sessions_since_recall', 0) + 1
        metadata['sessions_since_recall'] = sessions

        # Check if this should decay
        emotional_weight = metadata.get('emotional_weight', 0.5)
        recall_count = metadata.get('recall_count', 0)

        should_resist_decay = (
            emotional_weight >= EMOTIONAL_WEIGHT_THRESHOLD or
            recall_count >= RECALL_COUNT_THRESHOLD
        )

        if sessions >= DECAY_THRESHOLD_SESSIONS and not should_resist_decay:
            decay_candidates.append((filepath, metadata, content))
        elif should_resist_decay:
            reinforced.append((filepath, metadata))

        write_memory_file(filepath, metadata, content)

    # Report
    print(f"Active memories: {len(list(ACTIVE_DIR.glob('*.md'))) if ACTIVE_DIR.exists() else 0}")
    print(f"Core memories: {len(list(CORE_DIR.glob('*.md'))) if CORE_DIR.exists() else 0}")
    print(f"Archived memories: {len(list(ARCHIVE_DIR.glob('*.md'))) if ARCHIVE_DIR.exists() else 0}")

    if decay_candidates:
        print(f"\nDecay candidates ({len(decay_candidates)}):")
        for fp, meta, _ in decay_candidates:
            print(f"  - {fp.name}: {meta.get('sessions_since_recall')} sessions, weight={meta.get('emotional_weight'):.2f}")

    if reinforced:
        print(f"\nReinforced (resist decay):")
        for fp, meta in reinforced[:5]:
            print(f"  - {fp.name}: recalls={meta.get('recall_count')}, weight={meta.get('emotional_weight'):.2f}")

    return decay_candidates


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
            co_occurrences = metadata.get('co_occurrences', {})

            for other_id, count in co_occurrences.items():
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
    frontmatter = f"""---
id: {memory_id}
type: active
created: {created}
event_time: {event}
tags: [{', '.join(tags)}]
emotional_weight: {emotion}
recall_count: 0
co_occurrences: {{}}
caused_by: [{', '.join(all_causal)}]
leads_to: []
---

"""

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
        print("Memory Manager v2.11 - Living Memory with Auto-Detected Event Time")
        print("\nCommands:")
        print("  store <text>    - Store a new memory")
        print("                    --tags=a,b --emotion=0.8 --caused-by=id1,id2 --event-time=YYYY-MM-DD")
        print("                    Auto-links to memories recalled this session (causal)")
        print("                    --event-time: when event happened (bi-temporal, v2.10)")
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
        print("  promote         - Manually promote hot memories to core (recall_count >= threshold)")
        print("  decay-pairs     - Apply pair decay only (without logging new co-occurrences)")
        print("  session-status  - Show memories retrieved this session")
        print("  ask <query>     - Semantic search (natural language query)")
        print("  index           - Build/rebuild semantic search index")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "store" and len(sys.argv) > 2:
        # Parse arguments
        content_parts = []
        tags = []
        emotion = 0.5
        caused_by = []
        event_time = None  # v2.10: bi-temporal support

        for arg in sys.argv[2:]:
            if arg.startswith('--tags='):
                tags = arg[7:].split(',')
            elif arg.startswith('--emotion='):
                emotion = float(arg[10:])
            elif arg.startswith('--caused-by='):
                caused_by = [x.strip() for x in arg[12:].split(',') if x.strip()]
            elif arg.startswith('--event-time='):
                event_time = arg[13:]  # v2.10: when the event happened
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
        decayed, pruned = decay_pair_cooccurrences()
        promoted = promote_hot_memories()  # v2.9: heat-based promotion
        retrieved = get_session_retrieved()
        print(f"Session ended. {len(retrieved)} memories, {pairs} pairs reinforced, {decayed} decayed, {pruned} pruned, {len(promoted)} promoted.")
        clear_session()
        print("Session cleared.")
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
