#!/usr/bin/env python3
"""
Memory Query — Read-only search and retrieval functions.

Extracted from memory_manager.py (Phase 2).
All reads go directly to PostgreSQL. No file system. No fallbacks.
"""

from pathlib import Path
from db_adapter import get_db, db_to_file_metadata
from entity_detection import detect_entities


def _get_memory_by_id(memory_id: str) -> tuple[dict, str] | None:
    """
    Find and parse a memory by ID without side effects.

    Unlike recall_memory(), this does NOT update recall counts,
    emotional weight, or session tracking. Pure read.
    """
    db = get_db()
    row = db.get_memory(memory_id)
    if row:
        return db_to_file_metadata(row)
    return None


def find_co_occurring_memories(memory_id: str, limit: int = 5) -> list[tuple[str, int]]:
    """
    Find memories that frequently co-occur with a given memory.
    Returns list of (memory_id, co_occurrence_count) tuples, sorted by count.
    """
    db = get_db()
    neighbors = db.get_neighbors(memory_id, limit=limit)
    result = []
    for edge in neighbors:
        other_id = edge['id2'] if edge['id1'] == memory_id else edge['id1']
        result.append((other_id, edge['belief']))
    return result


def find_memories_by_tag(tag: str, limit: int = 10) -> list[tuple[Path, dict, str]]:
    """Find memories that contain a specific tag."""
    db = get_db()
    rows = db.list_memories(tags=[tag], limit=limit)
    results = []
    for row in rows:
        metadata, content = db_to_file_metadata(row)
        fake_path = Path(f"db://{row['type']}/{row['id']}.md")
        results.append((fake_path, metadata, content))
    results.sort(key=lambda x: x[1].get('emotional_weight', 0), reverse=True)
    return results


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
    """
    db = get_db()
    col = time_field if time_field in ('created', 'event_time') else 'created'
    conditions = []
    values = []
    if after:
        conditions.append(f"{col} >= %s")
        values.append(after)
    if before:
        conditions.append(f"{col} < %s")
        values.append(before)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    values.append(limit)

    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT * FROM {db._table('memories')} {where} ORDER BY {col} DESC LIMIT %s",
                values
            )
            rows = [dict(r) for r in cur.fetchall()]

    results = []
    for row in rows:
        metadata, content = db_to_file_metadata(row)
        fake_path = Path(f"db://{row['type']}/{row['id']}.md")
        results.append((fake_path, metadata, content))
    return results


def find_related_memories(memory_id: str) -> list[tuple[Path, dict, str]]:
    """Find memories related to a given memory via tags, links, and co-occurrence patterns."""
    db = get_db()
    source_row = db.get_memory(memory_id)
    if not source_row:
        return []
    source_meta, _ = db_to_file_metadata(source_row)
    source_tags = set(t.lower() for t in source_meta.get('tags', []))

    # Get co-occurring neighbors from edges_v3
    neighbors = db.get_neighbors(memory_id, limit=50)
    neighbor_ids = set()
    neighbor_belief = {}
    for edge in neighbors:
        other_id = edge['id2'] if edge['id1'] == memory_id else edge['id1']
        neighbor_ids.add(other_id)
        neighbor_belief[other_id] = edge['belief']

    # Get tag-overlapping memories
    tag_matches = set()
    for tag in source_meta.get('tags', []):
        for row in db.list_memories(tags=[tag], limit=20):
            if row['id'] != memory_id:
                tag_matches.add(row['id'])

    all_ids = neighbor_ids | tag_matches
    results = []
    for mid in all_ids:
        row = db.get_memory(mid)
        if not row:
            continue
        metadata, content = db_to_file_metadata(row)
        memory_tags = set(t.lower() for t in metadata.get('tags', []))
        co_count = neighbor_belief.get(mid, 0)
        has_tag_overlap = bool(source_tags & memory_tags)
        overlap_score = len(source_tags & memory_tags)
        adjusted_score = overlap_score + (co_count * 0.5)
        is_linked = mid in neighbor_ids
        fake_path = Path(f"db://{row['type']}/{row['id']}.md")
        results.append((fake_path, metadata, content, is_linked, adjusted_score, co_count))

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
    db = get_db()
    # DB stores entities as JSON — use find_by_entity plus fulltext fallback
    rows = db.find_by_entity(f'{entity_type}s', entity_name, limit=limit)
    if not rows:
        rows = db.search_fulltext(entity_name, limit=limit)
    results = []
    for row in rows:
        metadata, content = db_to_file_metadata(row)
        fake_path = Path(f"db://{row['type']}/{row['id']}.md")
        results.append((fake_path, metadata, content))
    results.sort(key=lambda x: x[1].get('emotional_weight', 0), reverse=True)
    return results[:limit]


def get_entity_cooccurrence(entity_type: str = 'agents') -> dict[str, dict[str, int]]:
    """
    Build entity co-occurrence graph from DB.

    Shows which entities appear together in memories.

    Args:
        entity_type: Which entity type to analyze ('agents', 'projects', 'concepts')

    Returns:
        Dict of {entity: {co_entity: count}}
    """
    db = get_db()
    cooccurrence = {}

    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT id, content, tags, entities FROM {db._table('memories')} WHERE type IN ('core', 'active')"
            )
            rows = cur.fetchall()

    for row in rows:
        entities_field = row.get('entities', {}) or {}
        if entities_field:
            entity_list = entities_field.get(entity_type, [])
        else:
            detected = detect_entities(row.get('content', ''), row.get('tags', []))
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
