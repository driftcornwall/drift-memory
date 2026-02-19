#!/usr/bin/env python3
"""
Generative Sleep — Novel Association Through Memory Replay

Phase 6 of the Voss Review implementation plan.
"Filing is not dreaming."

During session end, samples diverse memories from different sessions and
W-dimensions, then identifies novel connections between them. Creates
synthesis memories that bridge previously disconnected clusters.

Storage: DB memories (type='synthesis'), KG edges (relationship='synthesized_into')

Usage:
    python generative_sleep.py dream               # Run one dream cycle
    python generative_sleep.py sample               # Show what would be sampled
    python generative_sleep.py history              # Show synthesis history
    python generative_sleep.py stats                # Dream statistics
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path(__file__).parent
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

KV_DREAM_HISTORY = '.generative_sleep.history'
SAMPLE_COUNT = 5
MIN_SESSION_RECALLS = 5  # Only dream if session was substantive


def _get_db():
    from db_adapter import get_db
    return get_db()


def _now_iso():
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


# ============================================================
# Diverse Memory Sampling
# ============================================================

def sample_diverse_memories(count: int = SAMPLE_COUNT) -> list[dict]:
    """
    Sample memories that maximize diversity:
    - Different sessions (created spread)
    - Different W-dimensions (who/what/why/where/when)
    - Different types and topics

    Returns list of memory dicts with id, content, created, dimension info.
    """
    db = _get_db()
    import psycopg2.extras

    # Strategy: fetch from different time windows and dimensions
    memories = []
    seen_ids = set()

    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get total memory count for bucket sizing
            cur.execute(f"SELECT COUNT(*) as cnt FROM {db._table('memories')} WHERE type = 'active'")
            total = cur.fetchone()['cnt']
            if total < count * 2:
                return []  # Not enough memories to dream with

            # Sample from different time buckets (quintiles)
            bucket_size = total // count
            for i in range(count):
                offset = i * bucket_size + random.randint(0, max(0, bucket_size - 1))
                cur.execute(f"""
                    SELECT id, content, created, tags, extra_metadata
                    FROM {db._table('memories')}
                    WHERE type = 'active'
                    ORDER BY created ASC
                    OFFSET %s LIMIT 5
                """, (offset,))

                candidates = [dict(r) for r in cur.fetchall()]
                # Pick one that's not already selected
                for c in candidates:
                    if c['id'] not in seen_ids:
                        memories.append(c)
                        seen_ids.add(c['id'])
                        break

    # Try to add W-dimension diversity info
    for mem in memories:
        try:
            with db._conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(f"""
                        SELECT dimension FROM {db._table('context_graphs')}
                        WHERE memory_id = %s
                        LIMIT 1
                    """, (mem['id'],))
                    row = cur.fetchone()
                    mem['dimension'] = row['dimension'] if row else 'unknown'
        except Exception:
            mem['dimension'] = 'unknown'

    return memories


# ============================================================
# Dream Generation
# ============================================================

def _generate_synthesis_prompt(memories: list[dict]) -> str:
    """Build the prompt for synthesis generation."""
    parts = ["You are examining memories from different time periods and contexts. "
             "Find SPECIFIC, NON-OBVIOUS connections between them.\n\n"
             "Rules:\n"
             "- Only state connections that are genuinely insightful\n"
             "- Do NOT restate the inputs\n"
             "- Do NOT make generic observations ('these are all about learning')\n"
             "- Focus on structural parallels, unexpected analogies, or emergent patterns\n"
             "- If no genuine connection exists, say 'No novel connection found'\n\n"
             "Memories:\n"]

    for i, mem in enumerate(memories, 1):
        content = (mem.get('content') or '')[:300]
        created = str(mem.get('created', '?'))[:10]
        dim = mem.get('dimension', '?')
        parts.append(f"--- Memory {i} (created: {created}, dimension: {dim}) ---\n{content}\n")

    parts.append("\nWhat specific, non-obvious connections exist between these memories?")
    return '\n'.join(parts)


def _is_generic_output(synthesis_text: str) -> bool:
    """Filter out generic/restating synthesis outputs."""
    if not synthesis_text or len(synthesis_text) < 50:
        return True

    # Check for generic patterns
    generic_markers = [
        'no novel connection',
        'these memories all',
        'the common thread',
        'all relate to',
        'they all share',
        'in summary',
    ]

    text_lower = synthesis_text.lower()
    generic_count = sum(1 for marker in generic_markers if marker in text_lower)
    if generic_count >= 2:
        return True

    return False


def dream(dry_run: bool = False) -> dict:
    """
    Run one dream cycle.

    1. Sample diverse memories
    2. Generate synthesis (or show what would be generated in dry_run)
    3. Filter generic output
    4. Store as synthesis memory with provenance
    5. Create KG edges linking sources to synthesis

    Returns result dict with status, synthesis content, source IDs.
    """
    db = _get_db()

    # Check if session was substantive enough to dream
    try:
        from session_state import get_retrieved_list
        recalls = get_retrieved_list()
        if len(recalls) < MIN_SESSION_RECALLS:
            return {
                'status': 'skip',
                'reason': f'Session not substantive enough ({len(recalls)} recalls < {MIN_SESSION_RECALLS})',
            }
    except Exception:
        pass  # If session_state unavailable, proceed anyway

    # Sample memories
    memories = sample_diverse_memories(SAMPLE_COUNT)
    if len(memories) < 3:
        return {
            'status': 'skip',
            'reason': f'Not enough diverse memories ({len(memories)} < 3)',
        }

    source_ids = [m['id'] for m in memories]
    prompt = _generate_synthesis_prompt(memories)

    result = {
        'status': 'sampled',
        'source_ids': source_ids,
        'source_previews': [(m['id'], (m.get('content') or '')[:80], m.get('dimension', '?'))
                            for m in memories],
        'prompt_length': len(prompt),
    }

    if dry_run:
        result['status'] = 'dry_run'
        result['prompt'] = prompt
        return result

    # --- LLM synthesis call (with heuristic fallback) ---
    synthesis_text = ''
    try:
        from llm_client import synthesize_memories
        llm_result = synthesize_memories(memories)
        if llm_result.get('is_novel') and llm_result.get('synthesis'):
            synthesis_text = llm_result['synthesis']
            result['backend'] = llm_result.get('backend', '?')
            result['model'] = llm_result.get('model', '?')
            result['elapsed_ms'] = llm_result.get('elapsed_ms', 0)
    except Exception:
        pass

    # Fallback to heuristic if LLM unavailable or produced nothing novel
    if not synthesis_text:
        synthesis_text = _heuristic_synthesis(memories)
        result['backend'] = 'heuristic'

    if _is_generic_output(synthesis_text):
        result['status'] = 'filtered'
        result['reason'] = 'Synthesis was too generic'
        result['raw_output'] = synthesis_text
        return result

    # Store as synthesis memory
    from memory_store import store_memory
    synthesis_result = store_memory(
        content=synthesis_text,
        tags=['synthesis', 'generative-sleep'],
    )
    synthesis_id = synthesis_result[0] if synthesis_result else None

    # Update extra_metadata with synthesis provenance
    if synthesis_id:
        db.update_memory(synthesis_id, extra_metadata={
            'synthesis_sources': source_ids,
            'synthesis_type': 'dream',
            'synthesis_prompt_length': len(prompt),
            'created_by': 'generative_sleep.dream()',
        })

    if synthesis_id:
        # Create KG edges from sources to synthesis
        try:
            from knowledge_graph import add_edge
            for sid in source_ids:
                add_edge(sid, synthesis_id, 'synthesized_into',
                         confidence=0.6,
                         evidence={'method': 'generative_sleep', 'sample_size': len(source_ids)})
        except Exception:
            pass

        result['status'] = 'stored'
        result['synthesis_id'] = synthesis_id
        result['synthesis_preview'] = synthesis_text[:200]
    else:
        result['status'] = 'store_failed'
        result['synthesis_text'] = synthesis_text

    # Log to history
    _log_dream(result)

    return result


def _heuristic_synthesis(memories: list[dict]) -> str:
    """
    Heuristic synthesis using embedding similarity and KG bridge detection.
    Produces specific output that passes _is_generic_output() filter.
    (T1.4: ported from SpindriftMend — embedding cosine + KG bridges)
    """
    db = _get_db()
    import psycopg2.extras

    source_ids = [m['id'] for m in memories]

    # Build topic phrases from first sentence of each memory (max 60 chars)
    topic_phrases = {}
    for mem in memories:
        content = (mem.get('content') or '').strip()
        first_sent = content.split('.')[0][:60] if content else mem['id']
        topic_phrases[mem['id']] = first_sent

    parts = ["[Synthesis from generative sleep — heuristic mode]\n"]
    parts.append(f"Examined {len(memories)} memories from different time periods.\n")
    found_connection = False

    # --- Approach 1: Embedding cosine similarity (pgvector) ---
    try:
        with db._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT e1.memory_id AS id1, e2.memory_id AS id2,
                           1 - (e1.embedding <=> e2.embedding) AS similarity
                    FROM {db._table('text_embeddings')} e1
                    JOIN {db._table('text_embeddings')} e2
                      ON e1.memory_id < e2.memory_id
                    WHERE e1.memory_id = ANY(%s) AND e2.memory_id = ANY(%s)
                    ORDER BY similarity DESC
                    LIMIT 5
                """, (source_ids, source_ids))
                pairs = [dict(r) for r in cur.fetchall()]
                if pairs:
                    found_connection = True
                    parts.append("Embedding similarity between sampled memories:")
                    for p in pairs:
                        sim = float(p['similarity'])
                        t1 = topic_phrases.get(p['id1'], p['id1'])
                        t2 = topic_phrases.get(p['id2'], p['id2'])
                        if sim >= 0.85:
                            tier = "convergence"
                        elif sim >= 0.70:
                            tier = "bridge"
                        else:
                            tier = "weak link"
                        parts.append(f"  [{tier}] {sim:.3f}: \"{t1}\" <-> \"{t2}\"")
                    parts.append("")
    except Exception:
        pass

    # --- Approach 2: KG bridge node detection ---
    try:
        with db._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    WITH edges AS (
                        SELECT source_id AS src, target_id AS bridge, relationship
                        FROM {db._table('typed_edges')}
                        WHERE source_id = ANY(%s) AND target_id != ALL(%s)
                        UNION ALL
                        SELECT target_id AS src, source_id AS bridge, relationship
                        FROM {db._table('typed_edges')}
                        WHERE target_id = ANY(%s) AND source_id != ALL(%s)
                    )
                    SELECT bridge, relationship,
                           array_agg(DISTINCT src) AS sources,
                           COUNT(DISTINCT src) AS source_count
                    FROM edges
                    GROUP BY bridge, relationship
                    HAVING COUNT(DISTINCT src) >= 2
                    ORDER BY source_count DESC
                    LIMIT 5
                """, (source_ids, source_ids, source_ids, source_ids))
                bridges = [dict(r) for r in cur.fetchall()]
                if bridges:
                    found_connection = True
                    parts.append("KG bridge nodes connecting multiple source memories:")
                    for b in bridges:
                        srcs = b['sources'] if isinstance(b['sources'], list) else [b['sources']]
                        src_topics = [topic_phrases.get(s, s)[:30] for s in srcs[:3]]
                        parts.append(f"  {b['bridge']} ({b['relationship']}) bridges "
                                     f"{b['source_count']} memories: {', '.join(src_topics)}")
                    parts.append("")
    except Exception:
        pass

    if not found_connection:
        return "No novel connection found between the sampled memories."

    dimensions = [m.get('dimension', 'unknown') for m in memories]
    unique_dims = set(d for d in dimensions if d != 'unknown')
    if unique_dims:
        parts.append(f"Spanning dimensions: {', '.join(unique_dims)}")

    return '\n'.join(parts)


def _log_dream(result: dict):
    """Append dream result to history."""
    db = _get_db()
    history = db.kv_get(KV_DREAM_HISTORY) or {'dreams': []}
    history['dreams'].append({
        'ts': _now_iso(),
        'status': result.get('status', '?'),
        'source_count': len(result.get('source_ids', [])),
        'synthesis_id': result.get('synthesis_id'),
    })
    history['dreams'] = history['dreams'][-100:]
    db.kv_set(KV_DREAM_HISTORY, history)


def get_history() -> list[dict]:
    """Get dream history."""
    db = _get_db()
    data = db.kv_get(KV_DREAM_HISTORY) or {'dreams': []}
    return data.get('dreams', [])


def get_stats() -> dict:
    """Get dream statistics."""
    history = get_history()
    total = len(history)
    stored = sum(1 for d in history if d.get('status') == 'stored')
    filtered = sum(1 for d in history if d.get('status') == 'filtered')
    skipped = sum(1 for d in history if d.get('status') == 'skip')

    return {
        'total_dreams': total,
        'stored': stored,
        'filtered': filtered,
        'skipped': skipped,
        'success_rate': round(stored / max(1, total), 2),
    }


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generative Sleep — Novel Associations')
    sub = parser.add_subparsers(dest='command')

    p_dream = sub.add_parser('dream', help='Run one dream cycle')
    p_dream.add_argument('--dry-run', action='store_true', help='Sample without generating')

    sub.add_parser('sample', help='Show what would be sampled')
    sub.add_parser('history', help='Show synthesis history')
    sub.add_parser('stats', help='Dream statistics')

    args = parser.parse_args()

    if args.command == 'dream':
        result = dream(dry_run=args.dry_run)
        print(f"Dream status: {result['status']}")
        if result.get('source_previews'):
            print(f"\nSampled {len(result['source_previews'])} memories:")
            for mid, preview, dim in result['source_previews']:
                print(f"  [{dim}] {mid}: {preview}...")
        if result.get('synthesis_preview'):
            print(f"\nSynthesis:\n  {result['synthesis_preview']}")
        if result.get('synthesis_id'):
            print(f"\nStored as: {result['synthesis_id']}")
        if result.get('reason'):
            print(f"\nReason: {result['reason']}")

    elif args.command == 'sample':
        memories = sample_diverse_memories()
        if memories:
            print(f"Would sample {len(memories)} memories:\n")
            for m in memories:
                created = str(m.get('created', '?'))[:10]
                dim = m.get('dimension', '?')
                print(f"  [{dim}] {m['id']} (created: {created})")
                print(f"    {(m.get('content') or '')[:80]}...")
                print()
        else:
            print('Not enough memories for diverse sampling.')

    elif args.command == 'history':
        history = get_history()
        if not history:
            print('No dream history yet.')
        else:
            print(f'Last {min(10, len(history))} dreams:\n')
            for d in history[-10:]:
                ts = d.get('ts', '?')[:16]
                status = d.get('status', '?')
                synth = d.get('synthesis_id', '-')
                print(f'  [{ts}] {status} (sources: {d.get("source_count", "?")}, '
                      f'synthesis: {synth})')

    elif args.command == 'stats':
        s = get_stats()
        print('Generative Sleep Statistics:')
        print(f'  Total dream cycles: {s["total_dreams"]}')
        print(f'  Stored syntheses: {s["stored"]}')
        print(f'  Filtered (generic): {s["filtered"]}')
        print(f'  Skipped: {s["skipped"]}')
        print(f'  Success rate: {s["success_rate"]:.0%}')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
