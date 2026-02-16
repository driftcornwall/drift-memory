#!/usr/bin/env python3
"""
Reconsolidation — Memory Revision Through Retrieval

Phase 3 of the Voss Review implementation plan.
"Every act of recall is an act of reconstruction."

Memories that are frequently recalled in diverse contexts, or that have
contradiction signals, are candidates for revision. This module identifies
candidates, queues them, and (in Stage 3) processes revisions via LLM.

Storage: DB key_value_store, key pattern '.reconsolidation.*'

Usage:
    python reconsolidation.py candidates           # Show revision candidates
    python reconsolidation.py queue                # Queue candidates for revision
    python reconsolidation.py process [--dry-run]  # Process queued revisions (Stage 3)
    python reconsolidation.py history <id>         # Show revision history for a memory
    python reconsolidation.py stats                # Reconsolidation statistics
"""

import json
import sys
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path(__file__).parent
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

# Thresholds
MIN_RECALLS_FOR_REVISION = 5       # Minimum recalls since last revision
MIN_UNIQUE_QUERIES = 3             # Minimum distinct query contexts
CONTRADICTION_FAST_TRACK = True    # Memories with contradictions skip recall threshold
MAX_REVISIONS_PER_SESSION = 5      # Safety cap
MAX_CANDIDATES = 20                # Limit candidate scan


def _get_adaptive_candidate_limit() -> int:
    """Scale candidate limit by adaptive reconsolidation_frequency (R8 wiring)."""
    try:
        from adaptive_behavior import get_adaptation, DEFAULTS
        freq = get_adaptation('reconsolidation_frequency')
        default_freq = DEFAULTS.get('reconsolidation_frequency', 1.0)
        # Higher frequency = more candidates (scale linearly)
        return max(5, int(MAX_CANDIDATES * (freq / default_freq)))
    except Exception:
        return MAX_CANDIDATES


def _get_db():
    from db_adapter import get_db
    return get_db()


def _now_iso():
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


# ============================================================
# Stage 2: Revision Candidate Detection
# ============================================================

def find_candidates(limit: int = None) -> list[dict]:
    """
    Find memories that are candidates for reconsolidation.
    Limit defaults to adaptive candidate count (R8 wiring).

    Two paths to candidacy:
    1. Standard: recall_count_since_revision >= MIN_RECALLS and diverse queries
    2. Accelerated: Memory has active contradiction signals

    Returns list of candidate dicts with id, reason, recall_count, context_diversity, etc.
    """
    if limit is None:
        limit = _get_adaptive_candidate_limit()
    db = _get_db()
    candidates = []

    with db._conn() as conn:
        import psycopg2.extras
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Find memories with recall tracking data
            cur.execute(f"""
                SELECT id, content, recall_count, extra_metadata
                FROM {db._table('memories')}
                WHERE extra_metadata IS NOT NULL
                  AND extra_metadata ? 'recall_count_since_revision'
                ORDER BY (extra_metadata->>'recall_count_since_revision')::int DESC
                LIMIT %s
            """, (limit * 2,))  # Overfetch since we'll filter

            for row in cur.fetchall():
                extra = row['extra_metadata'] or {}
                recalls_since = extra.get('recall_count_since_revision', 0)
                contexts = extra.get('recall_contexts', [])
                contradictions = extra.get('contradiction_signals', 0)

                # Calculate context diversity (unique queries)
                unique_queries = set()
                for ctx in contexts:
                    q = ctx.get('query', '')[:50].lower().strip()
                    if q:
                        unique_queries.add(q)

                # Standard path
                standard_ready = (recalls_since >= MIN_RECALLS_FOR_REVISION
                                  and len(unique_queries) >= MIN_UNIQUE_QUERIES)

                # Accelerated path (contradictions)
                contradiction_ready = (CONTRADICTION_FAST_TRACK
                                       and contradictions > 0
                                       and recalls_since >= 2)

                if standard_ready or contradiction_ready:
                    reason = []
                    if standard_ready:
                        reason.append(f'{recalls_since} recalls, {len(unique_queries)} unique queries')
                    if contradiction_ready:
                        reason.append(f'{contradictions} contradiction(s)')

                    candidates.append({
                        'id': row['id'],
                        'content_preview': (row['content'] or '')[:120],
                        'recall_count': row['recall_count'],
                        'recalls_since_revision': recalls_since,
                        'context_diversity': len(unique_queries),
                        'contradiction_signals': contradictions,
                        'reason': ' + '.join(reason),
                        'path': 'accelerated' if contradiction_ready and not standard_ready else 'standard',
                        'recent_queries': list(unique_queries)[:5],
                    })

    # Sort: contradictions first, then by recall diversity
    candidates.sort(key=lambda c: (
        -c['contradiction_signals'],
        -c['context_diversity'],
        -c['recalls_since_revision']
    ))

    return candidates[:limit]


def queue_candidates(candidates: list[dict] = None) -> list[dict]:
    """
    Queue revision candidates in KV store for batch processing.
    Returns list of queued candidates.
    """
    if candidates is None:
        candidates = find_candidates()

    db = _get_db()
    queued = []

    # Check existing queue
    existing = db.kv_get('.reconsolidation.queue') or {}
    queue = existing.get('items', [])
    queued_ids = {item['id'] for item in queue}

    for c in candidates:
        if c['id'] in queued_ids:
            continue
        queue.append({
            'id': c['id'],
            'reason': c['reason'],
            'path': c['path'],
            'queued_at': _now_iso(),
        })
        queued.append(c)

    if queued:
        db.kv_set('.reconsolidation.queue', {
            'items': queue,
            'last_updated': _now_iso(),
        })

    return queued


def get_queue() -> list[dict]:
    """Get current revision queue."""
    db = _get_db()
    data = db.kv_get('.reconsolidation.queue') or {}
    return data.get('items', [])


def clear_queue():
    """Clear the revision queue."""
    db = _get_db()
    db.kv_set('.reconsolidation.queue', {'items': [], 'last_updated': _now_iso()})


# ============================================================
# Stage 3: LLM-Based Revision
# ============================================================

def process_revisions(dry_run: bool = True, max_revisions: int = MAX_REVISIONS_PER_SESSION) -> list[dict]:
    """
    Process queued revision candidates via LLM.

    For each candidate:
    1. Load memory + recall contexts + contradiction info
    2. Call LLM to produce revised version
    3. Store revision: update content, append old version to history, re-embed
    4. Reset counters
    """
    db = _get_db()
    queue = get_queue()
    results = []

    for item in queue[:max_revisions]:
        memory_id = item['id']
        row = db.get_memory(memory_id)
        if not row:
            continue

        extra = row.get('extra_metadata') or {}
        contexts = extra.get('recall_contexts', [])
        original_content = row.get('content') or ''

        result = {
            'id': memory_id,
            'content_preview': original_content[:120],
            'reason': item['reason'],
            'path': item['path'],
            'recall_contexts_count': len(contexts),
            'unique_queries': list(set(
                ctx.get('query', '')[:50].lower() for ctx in contexts if ctx.get('query')
            ))[:5],
        }

        if dry_run:
            result['action'] = 'dry_run'
            results.append(result)
            continue

        # Build contradiction info string
        contradiction_info = ''
        contra_count = extra.get('contradiction_signals', 0)
        if contra_count > 0:
            try:
                from knowledge_graph import get_edges_from, get_edges_to
                contra_edges = get_edges_from(memory_id, 'contradicts') + get_edges_to(memory_id, 'contradicts')
                if contra_edges:
                    parts = []
                    for e in contra_edges[:3]:
                        other_id = e['target_id'] if e['source_id'] == memory_id else e['source_id']
                        parts.append(f"Contradicts {other_id} (confidence: {e.get('confidence', '?')})")
                    contradiction_info = '; '.join(parts)
            except Exception:
                contradiction_info = f'{contra_count} contradiction(s) detected'

        # Call LLM for revision
        try:
            from llm_client import revise_memory
            llm_result = revise_memory(original_content, contexts, contradiction_info)
            revised = llm_result.get('revised_content', '')
            backend = llm_result.get('backend', 'none')
        except Exception as e:
            result['action'] = 'llm_error'
            result['error'] = str(e)
            results.append(result)
            continue

        if not revised or len(revised) < 20 or revised.strip() == original_content.strip():
            # LLM didn't produce a meaningful revision
            extra['recall_count_since_revision'] = 0
            extra['last_revision_check'] = _now_iso()
            versions = extra.get('versions', [])
            versions.append({
                'checked_at': _now_iso(),
                'action': 'reviewed_no_change',
                'backend': backend,
                'recall_contexts_at_review': len(contexts),
            })
            extra['versions'] = versions[-10:]
            db.update_memory(memory_id, extra_metadata=extra)
            result['action'] = 'no_change'
            results.append(result)
            continue

        # Store revision: update content, save old version
        versions = extra.get('versions', [])
        versions.append({
            'revised_at': _now_iso(),
            'action': 'revised',
            'backend': backend,
            'previous_content': original_content[:500],
            'recall_contexts_at_review': len(contexts),
            'contradiction_signals': contra_count,
        })
        extra['versions'] = versions[-10:]
        extra['recall_count_since_revision'] = 0
        extra['last_revision_check'] = _now_iso()
        extra['recall_contexts'] = []  # Clear contexts after revision

        db.update_memory(memory_id, content=revised, extra_metadata=extra)

        # Re-embed the revised content
        try:
            from semantic_search import _embed_text
            embedding = _embed_text(revised)
            if embedding:
                import psycopg2.extras
                with db._conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"UPDATE {db._table('memories')} SET embedding = %s WHERE id = %s",
                            (embedding, memory_id)
                        )
        except Exception:
            pass  # Re-embedding is best-effort

        # Re-extract KG edges from revised content
        try:
            from knowledge_graph import extract_and_store
            extract_and_store(memory_id, revised)
        except Exception:
            pass

        result['action'] = 'revised'
        result['backend'] = backend
        result['revised_preview'] = revised[:120]
        results.append(result)

    if not dry_run and results:
        processed_ids = {r['id'] for r in results if r.get('action') != 'dry_run'}
        remaining = [item for item in queue if item['id'] not in processed_ids]
        db.kv_set('.reconsolidation.queue', {
            'items': remaining,
            'last_updated': _now_iso(),
        })

    return results


# ============================================================
# History & Stats
# ============================================================

def get_revision_history(memory_id: str) -> dict:
    """Get revision history for a specific memory."""
    db = _get_db()
    row = db.get_memory(memory_id)
    if not row:
        return {'error': f'Memory {memory_id} not found'}

    extra = row.get('extra_metadata') or {}
    return {
        'id': memory_id,
        'content_preview': (row.get('content') or '')[:200],
        'recall_count': row.get('recall_count', 0),
        'recalls_since_revision': extra.get('recall_count_since_revision', 0),
        'contradiction_signals': extra.get('contradiction_signals', 0),
        'recall_contexts': extra.get('recall_contexts', []),
        'versions': extra.get('versions', []),
        'last_revision_check': extra.get('last_revision_check'),
    }


def get_stats() -> dict:
    """Get reconsolidation statistics."""
    db = _get_db()
    stats = {
        'memories_with_tracking': 0,
        'total_recall_contexts': 0,
        'memories_with_contradictions': 0,
        'candidates_ready': 0,
        'queue_length': len(get_queue()),
        'total_revisions': 0,
    }

    with db._conn() as conn:
        import psycopg2.extras
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Count memories with tracking
            cur.execute(f"""
                SELECT
                    COUNT(*) FILTER (WHERE extra_metadata ? 'recall_count_since_revision') as tracked,
                    COUNT(*) FILTER (WHERE (extra_metadata->>'contradiction_signals')::int > 0) as contradicted
                FROM {db._table('memories')}
                WHERE extra_metadata IS NOT NULL
            """)
            row = cur.fetchone()
            stats['memories_with_tracking'] = row['tracked'] or 0
            stats['memories_with_contradictions'] = row['contradicted'] or 0

    candidates = find_candidates()
    stats['candidates_ready'] = len(candidates)

    return stats


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Reconsolidation — Memory Revision')
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('candidates', help='Show revision candidates')

    sub.add_parser('queue', help='Queue candidates for revision')

    p_process = sub.add_parser('process', help='Process queued revisions')
    p_process.add_argument('--dry-run', action='store_true', default=True,
                           help='Show what would be revised (default)')
    p_process.add_argument('--execute', action='store_true',
                           help='Actually process revisions')

    p_history = sub.add_parser('history', help='Revision history for a memory')
    p_history.add_argument('id', help='Memory ID')

    sub.add_parser('stats', help='Reconsolidation statistics')

    args = parser.parse_args()

    if args.command == 'candidates':
        candidates = find_candidates()
        if not candidates:
            print('No revision candidates found yet.')
            print('(Memories need 5+ recalls from 3+ unique queries to qualify)')
        else:
            print(f'Found {len(candidates)} revision candidate(s):\n')
            for c in candidates:
                path_tag = f'[{c["path"].upper()}]'
                print(f'  {path_tag} {c["id"]}')
                print(f'    {c["content_preview"][:80]}...')
                print(f'    Reason: {c["reason"]}')
                print(f'    Queries: {", ".join(c["recent_queries"][:3])}')
                print()

    elif args.command == 'queue':
        queued = queue_candidates()
        existing = get_queue()
        print(f'Queued {len(queued)} new candidate(s). Total in queue: {len(existing)}')
        for q in queued:
            print(f'  + {q["id"]}: {q["reason"]}')

    elif args.command == 'process':
        dry_run = not args.execute
        results = process_revisions(dry_run=dry_run)
        mode = 'DRY RUN' if dry_run else 'EXECUTING'
        print(f'[{mode}] Processing {len(results)} revision(s):\n')
        for r in results:
            print(f'  {r["id"]}: {r["action"]}')
            print(f'    {r["content_preview"][:80]}...')
            print(f'    Reason: {r["reason"]}')
            print(f'    Unique queries: {", ".join(r["unique_queries"][:3])}')
            print()

    elif args.command == 'history':
        h = get_revision_history(args.id)
        if 'error' in h:
            print(h['error'])
        else:
            print(f'Revision history for {h["id"]}:')
            print(f'  Content: {h["content_preview"][:80]}...')
            print(f'  Total recalls: {h["recall_count"]}')
            print(f'  Recalls since revision: {h["recalls_since_revision"]}')
            print(f'  Contradiction signals: {h["contradiction_signals"]}')
            print(f'  Recall contexts: {len(h["recall_contexts"])}')
            print(f'  Versions: {len(h["versions"])}')
            if h['recall_contexts']:
                print(f'\n  Recent recall contexts:')
                for ctx in h['recall_contexts'][-5:]:
                    print(f'    [{ctx.get("ts", "?")}] q="{ctx.get("query", "?")[:50]}"')

    elif args.command == 'stats':
        s = get_stats()
        print('Reconsolidation Statistics:')
        print(f'  Memories with tracking: {s["memories_with_tracking"]}')
        print(f'  Memories with contradictions: {s["memories_with_contradictions"]}')
        print(f'  Candidates ready for revision: {s["candidates_ready"]}')
        print(f'  Queue length: {s["queue_length"]}')
        print(f'  Total recall contexts logged: {s["total_recall_contexts"]}')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
