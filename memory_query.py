#!/usr/bin/env python3
"""
Memory Query — Read-only search and retrieval functions.

Extracted from memory_manager.py (Phase 2).
All functions here are pure reads — they scan memory files and return results
without modifying any state.
"""

from pathlib import Path

from memory_common import (
    CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR, ALL_DIRS,
    parse_memory_file,
)
from entity_detection import detect_entities


def _get_memory_by_id(memory_id: str) -> tuple[dict, str] | None:
    """
    Find and parse a memory by ID without side effects.

    Unlike recall_memory(), this does NOT update recall counts,
    emotional weight, or session tracking. Pure read.
    """
    for directory in ALL_DIRS:
        if not directory.exists():
            continue
        for filepath in directory.glob(f"*-{memory_id}.md"):
            return parse_memory_file(filepath)
    return None


def find_co_occurring_memories(memory_id: str, limit: int = 5) -> list[tuple[str, int]]:
    """
    Find memories that frequently co-occur with a given memory.
    Returns list of (memory_id, co_occurrence_count) tuples, sorted by count.
    """
    for directory in ALL_DIRS:
        if not directory.exists():
            continue
        for filepath in directory.glob(f"*-{memory_id}.md"):
            metadata, _ = parse_memory_file(filepath)
            co_occurrences = metadata.get('co_occurrences', {})
            sorted_pairs = sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)
            return sorted_pairs[:limit]
    return []


def find_memories_by_tag(tag: str, limit: int = 10) -> list[tuple[Path, dict, str]]:
    """Find memories that contain a specific tag."""
    results = []
    for directory in ALL_DIRS:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            if tag.lower() in [t.lower() for t in metadata.get('tags', [])]:
                results.append((filepath, metadata, content))

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
        find_memories_by_time(before="2026-02-01", time_field="event_time")  # Events before Feb 1
        find_memories_by_time(after="2026-01-15", before="2026-02-01")  # Learned in that range

    Credit: Graphiti bi-temporal pattern (v2.10)
    """
    results = []

    for directory in ALL_DIRS:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)

            time_value = metadata.get(time_field, metadata.get('created', ''))
            if not time_value:
                continue

            if hasattr(time_value, 'isoformat'):
                time_value = time_value.isoformat()
            time_value = str(time_value)
            time_date = time_value[:10] if len(time_value) >= 10 else time_value

            if before and time_date >= before:
                continue
            if after and time_date < after:
                continue

            results.append((filepath, metadata, content, time_date))

    results.sort(key=lambda x: x[3], reverse=True)
    return [(r[0], r[1], r[2]) for r in results[:limit]]


def find_related_memories(memory_id: str) -> list[tuple[Path, dict, str]]:
    """Find memories related to a given memory via tags, links, and co-occurrence patterns."""
    source = _get_memory_by_id(memory_id)
    if not source:
        return []

    source_metadata, _ = source
    source_tags = set(t.lower() for t in source_metadata.get('tags', []))
    source_links = set(source_metadata.get('links', []))
    source_co_occurrences = source_metadata.get('co_occurrences', {})

    results = []
    for directory in ALL_DIRS:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            other_id = metadata.get('id')
            if other_id == memory_id:
                continue

            memory_tags = set(t.lower() for t in metadata.get('tags', []))
            is_linked = other_id in source_links
            has_tag_overlap = bool(source_tags & memory_tags)
            co_occurrence_count = source_co_occurrences.get(other_id, 0)

            if is_linked or has_tag_overlap or co_occurrence_count > 0:
                overlap_score = len(source_tags & memory_tags)
                adjusted_score = overlap_score + (co_occurrence_count * 0.5)
                results.append((filepath, metadata, content, is_linked, adjusted_score, co_occurrence_count))

    results.sort(key=lambda x: (x[3], x[4], x[1].get('emotional_weight', 0)), reverse=True)
    return [(r[0], r[1], r[2]) for r in results]


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

    for directory in ALL_DIRS:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            entities = metadata.get('entities', {})

            entity_list = entities.get(f'{entity_type}s', [])
            if entity_name_lower in [e.lower() for e in entity_list]:
                results.append((filepath, metadata, content))
                continue

            if entity_name_lower in [t.lower() for t in metadata.get('tags', [])]:
                results.append((filepath, metadata, content))
                continue

            if entity_name_lower in content.lower():
                results.append((filepath, metadata, content))

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

            entities_field = metadata.get('entities', {})
            if entities_field:
                entity_list = entities_field.get(entity_type, [])
            else:
                detected = detect_entities(content, metadata.get('tags', []))
                entity_list = detected.get(entity_type, [])

            for i, e1 in enumerate(entity_list):
                if e1 not in cooccurrence:
                    cooccurrence[e1] = {}
                for e2 in entity_list[i + 1:]:
                    cooccurrence[e1][e2] = cooccurrence[e1].get(e2, 0) + 1
                    if e2 not in cooccurrence:
                        cooccurrence[e2] = {}
                    cooccurrence[e2][e1] = cooccurrence[e2].get(e1, 0) + 1

    return cooccurrence
