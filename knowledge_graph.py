#!/usr/bin/env python3
"""
Knowledge Graph — Typed semantic relationships between memories.

Adds meaning to co-occurrence edges. Where co-occurrence says "these were recalled
together", typed edges say WHY: causes, enables, contradicts, supersedes, etc.

Supports auto-extraction from existing memory metadata and multi-hop traversal.

DB-ONLY: All data stored in drift.typed_edges.

Usage:
    python knowledge_graph.py extract [--limit N]    # Auto-extract from all memories
    python knowledge_graph.py query <id> [rel] [hops] # Multi-hop traversal
    python knowledge_graph.py stats                   # Relationship type distribution
    python knowledge_graph.py types                   # List all types with counts
    python knowledge_graph.py path <id1> <id2>        # Find shortest typed path
"""

import json
import re
import sys
from datetime import datetime, timezone
from typing import Optional

import psycopg2.extras
from db_adapter import get_db, db_to_file_metadata

# --- Relationship Types ---

RELATIONSHIP_TYPES = {
    'causes':          {'symbol': '->', 'description': 'A led to B happening'},
    'enables':         {'symbol': '=>', 'description': 'A makes B possible'},
    'contradicts':     {'symbol': '!=', 'description': 'A conflicts with B'},
    'supersedes':      {'symbol': '>>', 'description': 'A replaces/updates B'},
    'part_of':         {'symbol': 'in', 'description': 'A is a component of B'},
    'instance_of':     {'symbol': '::', 'description': 'A is an example of B'},
    'similar_to':      {'symbol': '~~', 'description': 'A and B are semantically close'},
    'depends_on':      {'symbol': '<-', 'description': 'A requires B to work'},
    'implements':      {'symbol': '|-', 'description': 'A implements the concept in B'},
    'learned_from':    {'symbol': '<~', 'description': 'A was extracted/learned from B'},
    'collaborator':    {'symbol': '<>', 'description': 'A and B involve same agent/entity'},
    'temporal_before': {'symbol': '< ', 'description': 'A happened before B'},
    'temporal_after':  {'symbol': '> ', 'description': 'A happened after B'},
    'references':      {'symbol': '->', 'description': 'A mentions/cites B'},
    'resolves':        {'symbol': 'ok', 'description': 'A fixes the problem in B'},
    'supports':        {'symbol': '++', 'description': 'A provides evidence for B'},
    'counterfactual_of': {'symbol': '?!', 'description': 'A is a counterfactual analysis of B'},
}


# --- DB Operations ---

def add_edge(source_id: str, target_id: str, relationship: str,
             confidence: float = 0.8, evidence: str = None,
             auto_extracted: bool = False) -> Optional[dict]:
    """Add a typed relationship. Upserts on (source, target, rel)."""
    if relationship not in RELATIONSHIP_TYPES:
        return None

    db = get_db()
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                INSERT INTO {db._table('typed_edges')}
                (source_id, target_id, relationship, confidence, evidence, auto_extracted, created)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (source_id, target_id, relationship)
                DO UPDATE SET confidence = GREATEST(
                    {db._table('typed_edges')}.confidence, EXCLUDED.confidence
                ),
                evidence = COALESCE(EXCLUDED.evidence, {db._table('typed_edges')}.evidence),
                auto_extracted = EXCLUDED.auto_extracted
                RETURNING *
            """, (source_id, target_id, relationship, confidence, evidence, auto_extracted))
            row = cur.fetchone()
            return dict(row) if row else None


def get_edges_from(source_id: str, relationship: str = None) -> list[dict]:
    """Get outgoing typed edges from a source memory."""
    db = get_db()
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if relationship:
                cur.execute(f"""
                    SELECT * FROM {db._table('typed_edges')}
                    WHERE source_id = %s AND relationship = %s
                    ORDER BY confidence DESC
                """, (source_id, relationship))
            else:
                cur.execute(f"""
                    SELECT * FROM {db._table('typed_edges')}
                    WHERE source_id = %s
                    ORDER BY relationship, confidence DESC
                """, (source_id,))
            return [dict(r) for r in cur.fetchall()]


def get_edges_to(target_id: str, relationship: str = None) -> list[dict]:
    """Get incoming typed edges to a target memory."""
    db = get_db()
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if relationship:
                cur.execute(f"""
                    SELECT * FROM {db._table('typed_edges')}
                    WHERE target_id = %s AND relationship = %s
                    ORDER BY confidence DESC
                """, (target_id, relationship))
            else:
                cur.execute(f"""
                    SELECT * FROM {db._table('typed_edges')}
                    WHERE target_id = %s
                    ORDER BY relationship, confidence DESC
                """, (target_id,))
            return [dict(r) for r in cur.fetchall()]


def get_all_edges(memory_id: str) -> list[dict]:
    """Get all typed edges involving a memory (both directions)."""
    db = get_db()
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT *, 'outgoing' as direction FROM {db._table('typed_edges')}
                WHERE source_id = %s
                UNION ALL
                SELECT *, 'incoming' as direction FROM {db._table('typed_edges')}
                WHERE target_id = %s
                ORDER BY relationship, confidence DESC
            """, (memory_id, memory_id))
            return [dict(r) for r in cur.fetchall()]


def batch_get_edges(memory_ids: list[str], relationships: list[str] = None) -> dict[str, list[dict]]:
    """
    Batch-fetch typed edges for multiple memories in a single query.

    Returns dict mapping memory_id -> list of edges (both directions).
    Much faster than N individual get_edges_from/get_edges_to calls.
    """
    if not memory_ids:
        return {}

    db = get_db()
    result = {mid: [] for mid in memory_ids}

    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            rel_filter = ""
            params = [tuple(memory_ids)]
            if relationships:
                rel_filter = "AND relationship = ANY(%s)"
                params.append(relationships)

            cur.execute(f"""
                SELECT *, 'outgoing' as direction FROM {db._table('typed_edges')}
                WHERE source_id = ANY(%s) {rel_filter}
                UNION ALL
                SELECT *, 'incoming' as direction FROM {db._table('typed_edges')}
                WHERE target_id = ANY(%s) {rel_filter}
                ORDER BY confidence DESC
            """, params + params)  # params duplicated for both halves of UNION

            for row in cur.fetchall():
                row = dict(row)
                # Assign to the memory that owns this edge
                sid = row.get('source_id', '')
                tid = row.get('target_id', '')
                if sid in result:
                    result[sid].append(row)
                if tid in result and tid != sid:
                    result[tid].append(row)

    return result


def delete_edge(source_id: str, target_id: str, relationship: str) -> bool:
    """Delete a specific typed relationship."""
    db = get_db()
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                DELETE FROM {db._table('typed_edges')}
                WHERE source_id = %s AND target_id = %s AND relationship = %s
            """, (source_id, target_id, relationship))
            return cur.rowcount > 0


# --- Multi-Hop Traversal ---

def traverse(start_id: str, relationship: str = None,
             hops: int = 2, direction: str = 'outgoing',
             min_confidence: float = 0.3) -> list[dict]:
    """
    Multi-hop graph traversal using recursive CTE.

    Args:
        start_id: Starting memory ID
        relationship: Filter to specific relationship type (or None for all)
        hops: Maximum traversal depth
        direction: 'outgoing' (follow source->target), 'incoming' (target->source), 'both'
        min_confidence: Minimum edge confidence to follow

    Returns:
        List of reachable edges with depth information
    """
    db = get_db()
    table = db._table('typed_edges')

    # Build direction-specific SQL
    if direction == 'outgoing':
        base_where = "source_id = %s"
        join_on = "te.source_id = gt.target_id"
    elif direction == 'incoming':
        base_where = "target_id = %s"
        join_on = "te.target_id = gt.source_id"
    else:  # both
        base_where = "(source_id = %s OR target_id = %s)"
        join_on = "(te.source_id = gt.target_id OR te.target_id = gt.source_id)"

    # Build base and recursive relationship clauses with table aliases
    base_rel_clause = "AND relationship = %s" if relationship else ""
    rec_rel_clause = "AND te.relationship = %s" if relationship else ""

    params = []
    if direction == 'both':
        params.extend([start_id, start_id])
    else:
        params.append(start_id)
    if relationship:
        params.append(relationship)
    params.append(min_confidence)

    # Recursive params
    params.append(hops)
    if relationship:
        params.append(relationship)
    params.append(min_confidence)

    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                WITH RECURSIVE graph_traverse AS (
                    SELECT source_id, target_id, relationship, confidence,
                           evidence, 1 as depth,
                           ARRAY[source_id, target_id] as path
                    FROM {table}
                    WHERE {base_where} {base_rel_clause}
                    AND confidence >= %s

                    UNION ALL

                    SELECT te.source_id, te.target_id, te.relationship, te.confidence,
                           te.evidence, gt.depth + 1,
                           gt.path || te.target_id
                    FROM {table} te
                    JOIN graph_traverse gt ON {join_on}
                    WHERE gt.depth < %s
                    {rec_rel_clause}
                    AND te.confidence >= %s
                    AND NOT te.target_id = ANY(gt.path)
                )
                SELECT DISTINCT source_id, target_id, relationship, confidence,
                       evidence, depth
                FROM graph_traverse
                ORDER BY depth, confidence DESC
            """, params)
            return [dict(r) for r in cur.fetchall()]


def find_path(id1: str, id2: str, max_hops: int = 5) -> Optional[list[dict]]:
    """
    Find shortest typed path between two memories (BFS).

    Returns:
        List of edges forming the path, or None if no path exists
    """
    db = get_db()
    table = db._table('typed_edges')

    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                WITH RECURSIVE path_search AS (
                    SELECT source_id, target_id, relationship, confidence,
                           1 as depth,
                           ARRAY[source_id] as visited,
                           ARRAY[ROW(source_id, target_id, relationship, confidence)::text] as edges
                    FROM {table}
                    WHERE source_id = %s

                    UNION ALL

                    SELECT te.source_id, te.target_id, te.relationship, te.confidence,
                           ps.depth + 1,
                           ps.visited || te.source_id,
                           ps.edges || ROW(te.source_id, te.target_id, te.relationship, te.confidence)::text
                    FROM {table} te
                    JOIN path_search ps ON te.source_id = ps.target_id
                    WHERE ps.depth < %s
                    AND NOT te.source_id = ANY(ps.visited)
                )
                SELECT edges, depth
                FROM path_search
                WHERE target_id = %s
                ORDER BY depth
                LIMIT 1
            """, (id1, max_hops, id2))
            row = cur.fetchone()
            if row:
                return {'depth': row['depth'], 'edges': row['edges']}
            return None


# --- Statistics ---

def get_stats() -> dict:
    """Get typed edge statistics."""
    db = get_db()
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE auto_extracted) as auto_extracted,
                    COUNT(*) FILTER (WHERE NOT auto_extracted) as manual,
                    COUNT(DISTINCT relationship) as types_used,
                    COUNT(DISTINCT source_id) as unique_sources,
                    COUNT(DISTINCT target_id) as unique_targets,
                    AVG(confidence) as avg_confidence
                FROM {db._table('typed_edges')}
            """)
            row = dict(cur.fetchone())
            row['avg_confidence'] = round(float(row['avg_confidence'] or 0), 3)

            # Type distribution
            cur.execute(f"""
                SELECT relationship, COUNT(*) as count,
                       AVG(confidence) as avg_conf
                FROM {db._table('typed_edges')}
                GROUP BY relationship
                ORDER BY count DESC
            """)
            row['by_type'] = {
                r['relationship']: {
                    'count': r['count'],
                    'avg_confidence': round(float(r['avg_conf'] or 0), 3)
                }
                for r in cur.fetchall()
            }

            # Density: typed edges / total possible (memories^2)
            cur.execute(f"""
                SELECT COUNT(*) as cnt FROM {db._table('memories')}
                WHERE type IN ('core', 'active')
            """)
            mem_count = cur.fetchone()['cnt']
            row['memory_count'] = mem_count
            max_possible = mem_count * (mem_count - 1) if mem_count > 1 else 1
            row['density'] = round(row['total'] / max_possible, 6) if max_possible > 0 else 0

            return row


# --- Auto-Extraction Pipeline ---

def extract_from_memory(memory_id: str) -> list[dict]:
    """
    Auto-extract typed relationships from a single memory's metadata and content.

    Extracts from:
    1. caused_by field -> 'causes' (reverse: cause -> this memory)
    2. leads_to field -> 'causes' (forward: this memory -> effect)
    3. Entity overlap -> 'collaborator'
    4. Resolution tag -> 'resolves'

    Returns list of edges created.
    """
    db = get_db()
    row = db.get_memory(memory_id)
    if not row:
        return []

    meta, _ = db_to_file_metadata(row)
    meta = meta or {}
    content = row.get('content', '')
    tags = row.get('tags', []) or []
    created = []

    # 1. Causal links from caused_by
    for cause_id in (meta.get('caused_by') or []):
        # Verify cause exists
        if db.get_memory(cause_id):
            edge = add_edge(cause_id, memory_id, 'causes',
                            confidence=0.9,
                            evidence='caused_by metadata field',
                            auto_extracted=True)
            if edge:
                created.append(edge)

    # 2. Causal links from leads_to
    for effect_id in (meta.get('leads_to') or []):
        if db.get_memory(effect_id):
            edge = add_edge(memory_id, effect_id, 'causes',
                            confidence=0.9,
                            evidence='leads_to metadata field',
                            auto_extracted=True)
            if edge:
                created.append(edge)

    # 3. Resolution tag -> resolves relationship
    if 'resolution' in tags or 'fix' in tags or 'resolved' in tags:
        # Look for problem memories in caused_by chain
        for cause_id in (meta.get('caused_by') or []):
            cause_row = db.get_memory(cause_id)
            if cause_row:
                cause_tags = cause_row.get('tags', []) or []
                if any(t in cause_tags for t in ['bug', 'error', 'problem', 'issue']):
                    edge = add_edge(memory_id, cause_id, 'resolves',
                                    confidence=0.85,
                                    evidence='resolution tag + bug cause',
                                    auto_extracted=True)
                    if edge:
                        created.append(edge)

    # 4. Entity-based collaborator edges
    entities = row.get('entities', {}) or {}
    if entities:
        entity_names = set()
        for etype, elist in entities.items():
            if isinstance(elist, list):
                entity_names.update(e.lower() for e in elist)
            elif isinstance(elist, dict):
                entity_names.update(e.lower() for e in elist.keys())

        if entity_names:
            _link_by_entities(memory_id, entity_names, db, created)

    # 5. Lesson links (learned_from)
    if 'lesson' in tags:
        for cause_id in (meta.get('caused_by') or []):
            if db.get_memory(cause_id):
                edge = add_edge(memory_id, cause_id, 'learned_from',
                                confidence=0.85,
                                evidence='lesson tag + caused_by',
                                auto_extracted=True)
                if edge:
                    created.append(edge)

    # 6. depends_on (reverse of causes — if A caused this, this depends_on A)
    for cause_id in (meta.get('caused_by') or []):
        if db.get_memory(cause_id):
            edge = add_edge(memory_id, cause_id, 'depends_on',
                            confidence=0.75,
                            evidence='caused_by implies dependency',
                            auto_extracted=True)
            if edge:
                created.append(edge)

    # 7. supersedes (newer memory contradicting an older one)
    if 'contradicts' in (meta.get('_relations') or {}):
        for contra_id in meta['_relations']['contradicts']:
            contra_row = db.get_memory(contra_id)
            if contra_row:
                # Newer memory supersedes older
                created_at = row.get('created', '')
                contra_created = contra_row.get('created', '')
                if str(created_at) > str(contra_created):
                    edge = add_edge(memory_id, contra_id, 'supersedes',
                                    confidence=0.80,
                                    evidence='newer contradicting memory',
                                    auto_extracted=True)
                    if edge:
                        created.append(edge)

    # 8. part_of (tag-based hierarchy)
    SYSTEM_TAGS = {
        'affect': 'architecture', 'cognition': 'architecture',
        'memory': 'architecture', 'identity': 'architecture',
        'social': 'platform', 'moltx': 'platform', 'colony': 'platform',
        'github': 'platform', 'twitter': 'platform',
    }
    for tag in tags:
        parent_tag = SYSTEM_TAGS.get(tag)
        if parent_tag:
            # Find memories tagged with the parent
            _link_by_tag_hierarchy(memory_id, tag, parent_tag, db, created)

    # 9. references (content mentions other memory IDs)
    _extract_references(memory_id, content, db, created)

    # 10. supports (memories with same conclusion from different evidence)
    if any(t in tags for t in ['confirmed', 'verified', 'evidence', 'supports']):
        for cause_id in (meta.get('caused_by') or []):
            if db.get_memory(cause_id):
                edge = add_edge(memory_id, cause_id, 'supports',
                                confidence=0.80,
                                evidence='confirmation/evidence tag + causal link',
                                auto_extracted=True)
                if edge:
                    created.append(edge)

    return created


def _link_by_tag_hierarchy(memory_id: str, child_tag: str, parent_tag: str,
                           db, created: list, limit: int = 5):
    """Create part_of edges between memories in tag hierarchies."""
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id FROM {db._table('memories')}
                WHERE id != %s
                AND type IN ('core', 'active')
                AND %s = ANY(tags)
                AND NOT (%s = ANY(tags))
                LIMIT %s
            """, (memory_id, parent_tag, child_tag, limit))
            for row in cur.fetchall():
                edge = add_edge(memory_id, row['id'], 'part_of',
                                confidence=0.65,
                                evidence=f'tag hierarchy: {child_tag} part_of {parent_tag}',
                                auto_extracted=True)
                if edge:
                    created.append(edge)


def _extract_references(memory_id: str, content: str, db, created: list):
    """Find memory IDs mentioned in content and create reference edges."""
    # Match memory ID patterns (8+ char alphanumeric with optional hyphens)
    id_pattern = re.compile(r'\b([a-z0-9]{8,12})\b')
    potential_ids = set(id_pattern.findall(content))

    for pid in list(potential_ids)[:10]:  # Cap at 10
        if pid == memory_id:
            continue
        if db.get_memory(pid):
            edge = add_edge(memory_id, pid, 'references',
                            confidence=0.70,
                            evidence='memory ID found in content',
                            auto_extracted=True)
            if edge:
                created.append(edge)


def _link_by_entities(memory_id: str, entity_names: set, db, created: list,
                      limit: int = 10):
    """Find other memories sharing entities and create collaborator edges."""
    # Find memories with overlapping entities (limited scan)
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Search entities JSONB for matching names
            for name in list(entity_names)[:5]:  # Cap at 5 entities
                cur.execute(f"""
                    SELECT id FROM {db._table('memories')}
                    WHERE id != %s
                    AND type IN ('core', 'active')
                    AND entities::text ILIKE %s
                    LIMIT %s
                """, (memory_id, f'%{name}%', limit))
                for row in cur.fetchall():
                    edge = add_edge(memory_id, row['id'], 'collaborator',
                                    confidence=0.6,
                                    evidence=f'shared entity: {name}',
                                    auto_extracted=True)
                    if edge:
                        created.append(edge)


def extract_similarity_edges(limit: int = 200, threshold: float = 0.85,
                             verbose: bool = False) -> dict:
    """
    Extract 'similar_to' edges using embedding cosine similarity.

    Finds pairs of memories with high semantic similarity that don't
    already have a similar_to edge. Uses pgvector's cosine distance.

    Args:
        limit: Max new edges to create
        threshold: Cosine similarity threshold (0.85 = very similar)
        verbose: Print progress

    Returns:
        Dict with edges_created count
    """
    db = get_db()
    created = 0

    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Find high-similarity pairs that don't already have similar_to edges
            # Uses pgvector's <=> operator (cosine distance, so 1-distance = similarity)
            # Embeddings are in text_embeddings table, not memories
            emb_table = db._table('text_embeddings')
            cur.execute(f"""
                SELECT e1.memory_id as id1, e2.memory_id as id2,
                       1 - (e1.embedding <=> e2.embedding) as similarity
                FROM {emb_table} e1
                JOIN {emb_table} e2
                    ON e1.memory_id < e2.memory_id
                JOIN {db._table('memories')} m1 ON m1.id = e1.memory_id
                JOIN {db._table('memories')} m2 ON m2.id = e2.memory_id
                WHERE m1.type IN ('core', 'active')
                AND m2.type IN ('core', 'active')
                AND 1 - (e1.embedding <=> e2.embedding) >= %s
                AND NOT EXISTS (
                    SELECT 1 FROM {db._table('typed_edges')} te
                    WHERE te.source_id = e1.memory_id AND te.target_id = e2.memory_id
                    AND te.relationship = 'similar_to'
                )
                ORDER BY similarity DESC
                LIMIT %s
            """, (threshold, limit))

            pairs = cur.fetchall()
            if verbose:
                print(f"  Found {len(pairs)} high-similarity pairs above {threshold}")

            for pair in pairs:
                edge = add_edge(pair['id1'], pair['id2'], 'similar_to',
                                confidence=round(float(pair['similarity']), 3),
                                evidence=f'cosine similarity {pair["similarity"]:.3f}',
                                auto_extracted=True)
                if edge:
                    created += 1
                    if verbose and created % 50 == 0:
                        print(f"    ... {created} similar_to edges created")

    return {'edges_created': created, 'pairs_found': len(pairs)}


def extract_temporal_edges(limit: int = 500, verbose: bool = False) -> dict:
    """
    Extract temporal_before/temporal_after edges for causally linked memories.

    Uses existing causal chains (caused_by/leads_to) to establish temporal ordering.
    """
    db = get_db()
    created = 0

    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Find causal pairs and add temporal ordering
            cur.execute(f"""
                SELECT te.source_id, te.target_id, m1.created as source_time, m2.created as target_time
                FROM {db._table('typed_edges')} te
                JOIN {db._table('memories')} m1 ON m1.id = te.source_id
                JOIN {db._table('memories')} m2 ON m2.id = te.target_id
                WHERE te.relationship = 'causes'
                AND NOT EXISTS (
                    SELECT 1 FROM {db._table('typed_edges')} te2
                    WHERE te2.source_id = te.source_id AND te2.target_id = te.target_id
                    AND te2.relationship = 'temporal_before'
                )
                LIMIT %s
            """, (limit,))

            pairs = cur.fetchall()
            for pair in pairs:
                # Cause happened before effect
                edge = add_edge(pair['source_id'], pair['target_id'], 'temporal_before',
                                confidence=0.90,
                                evidence='causal ordering implies temporal ordering',
                                auto_extracted=True)
                if edge:
                    created += 1

    return {'edges_created': created}


def extract_all(limit: int = 500, verbose: bool = False) -> dict:
    """
    Run auto-extraction pipeline across all memories.

    Returns:
        Dict with total_processed, edges_created, by_type counts
    """
    db = get_db()

    # Get memories that haven't been processed yet
    # (check if they have any auto-extracted outgoing edges)
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT m.id
                FROM {db._table('memories')} m
                LEFT JOIN {db._table('typed_edges')} te
                    ON te.source_id = m.id AND te.auto_extracted = TRUE
                WHERE m.type IN ('core', 'active')
                AND te.id IS NULL
                ORDER BY m.created DESC
                LIMIT %s
            """, (limit,))
            ids_to_process = [r['id'] for r in cur.fetchall()]

    total_edges = 0
    by_type = {}
    for i, mem_id in enumerate(ids_to_process):
        edges = extract_from_memory(mem_id)
        for e in edges:
            rel = e.get('relationship', '?')
            by_type[rel] = by_type.get(rel, 0) + 1
        total_edges += len(edges)
        if verbose and edges:
            print(f"  [{i+1}/{len(ids_to_process)}] {mem_id}: {len(edges)} edges")

    # Phase 2: Run diversity extractors (similar_to, temporal)
    if verbose:
        print("\n  Phase 2: Similarity edges...")
    sim_result = extract_similarity_edges(limit=200, verbose=verbose)
    total_edges += sim_result['edges_created']
    if sim_result['edges_created'] > 0:
        by_type['similar_to'] = sim_result['edges_created']

    if verbose:
        print("  Phase 3: Temporal edges...")
    temp_result = extract_temporal_edges(limit=500, verbose=verbose)
    total_edges += temp_result['edges_created']
    if temp_result['edges_created'] > 0:
        by_type['temporal_before'] = temp_result['edges_created']

    return {
        'total_processed': len(ids_to_process),
        'edges_created': total_edges,
        'by_type': by_type,
    }


# --- CLI ---

def _preview_memory(memory_id: str, max_len: int = 60) -> str:
    """Get a short preview of a memory's content."""
    db = get_db()
    row = db.get_memory(memory_id)
    if not row:
        return f"[{memory_id}] (not found)"
    content = (row.get('content', '') or '')[:max_len]
    content = content.replace('\n', ' ')
    if len(row.get('content', '')) > max_len:
        content += '...'
    return f"[{memory_id}] {content}"


def main():
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    args = sys.argv[1:]
    if not args or args[0] in ('--help', '-h'):
        print(__doc__.strip())
        return

    cmd = args[0]

    if cmd == 'extract':
        limit = 500
        verbose = True
        for a in args[1:]:
            if a.startswith('--limit'):
                limit = int(a.split('=')[1]) if '=' in a else int(args[args.index(a) + 1])
            if a == '-q':
                verbose = False

        print(f"\nExtracting typed relationships (limit={limit})...\n")
        result = extract_all(limit=limit, verbose=verbose)
        print(f"\nProcessed: {result['total_processed']} memories")
        print(f"Edges created: {result['edges_created']}")
        if result['by_type']:
            print("By type:")
            for rel, count in sorted(result['by_type'].items(), key=lambda x: -x[1]):
                sym = RELATIONSHIP_TYPES.get(rel, {}).get('symbol', '?')
                print(f"  {sym} {rel}: {count}")

    elif cmd == 'query':
        if len(args) < 2:
            print("Usage: knowledge_graph.py query <memory_id> [relationship] [hops]")
            return
        mem_id = args[1]
        rel = args[2] if len(args) > 2 and args[2] in RELATIONSHIP_TYPES else None
        hops = int(args[3]) if len(args) > 3 else (int(args[2]) if len(args) > 2 and args[2].isdigit() else 2)

        print(f"\nTraversing from {mem_id} (hops={hops}, rel={rel or 'all'}):\n")
        results = traverse(mem_id, relationship=rel, hops=hops)
        if not results:
            print("  No typed relationships found.")
            return

        for edge in results:
            sym = RELATIONSHIP_TYPES.get(edge['relationship'], {}).get('symbol', '?')
            src_preview = _preview_memory(edge['source_id'], 40)
            tgt_preview = _preview_memory(edge['target_id'], 40)
            print(f"  [depth={edge['depth']}] {src_preview}")
            print(f"    {sym} {edge['relationship']} (conf={edge['confidence']:.2f})")
            print(f"    {tgt_preview}")
            print()

    elif cmd == 'stats':
        stats = get_stats()
        print(f"\n=== Knowledge Graph Stats ===\n")
        print(f"  Total edges: {stats['total']}")
        print(f"  Auto-extracted: {stats['auto_extracted']}")
        print(f"  Manual: {stats['manual']}")
        print(f"  Types used: {stats['types_used']}")
        print(f"  Unique sources: {stats['unique_sources']}")
        print(f"  Unique targets: {stats['unique_targets']}")
        print(f"  Avg confidence: {stats['avg_confidence']}")
        print(f"  Density: {stats['density']}")
        print()
        if stats['by_type']:
            print("  By type:")
            for rel, info in stats['by_type'].items():
                sym = RELATIONSHIP_TYPES.get(rel, {}).get('symbol', '?')
                desc = RELATIONSHIP_TYPES.get(rel, {}).get('description', '')
                print(f"    {sym} {rel:20s} {info['count']:>5d}  (conf={info['avg_confidence']:.2f})  {desc}")

    elif cmd == 'types':
        print(f"\n=== Relationship Types ({len(RELATIONSHIP_TYPES)}) ===\n")
        stats = get_stats()
        for rel, info in RELATIONSHIP_TYPES.items():
            count = stats.get('by_type', {}).get(rel, {}).get('count', 0)
            print(f"  {info['symbol']:>2s} {rel:20s} {count:>5d}  {info['description']}")

    elif cmd == 'path':
        if len(args) < 3:
            print("Usage: knowledge_graph.py path <id1> <id2>")
            return
        result = find_path(args[1], args[2])
        if result:
            print(f"\nPath found (depth={result['depth']}):")
            for edge_str in result['edges']:
                print(f"  {edge_str}")
        else:
            print(f"\nNo typed path found between {args[1]} and {args[2]}.")

    else:
        print(f"Unknown command: {cmd}")
        print("Available: extract, query, stats, types, path")
        sys.exit(1)


if __name__ == '__main__':
    main()
