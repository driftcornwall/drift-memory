#!/usr/bin/env python3
"""
Backfill evidence_type for all memories missing classification.

Uses the heuristic classifier from memory_validation.py (no LLM needed).
Writes evidence_type + evidence_confidence into extra_metadata.

Usage:
    python backfill_evidence.py              # Dry run (show what would change)
    python backfill_evidence.py --commit     # Actually write to DB
    python backfill_evidence.py --stats      # Show distribution after backfill
"""

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

MEMORY_DIR = Path(__file__).parent
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

from db_adapter import get_db
from memory_validation import classify_evidence_type

import psycopg2.extras


def get_unclassified_memories(db) -> list[dict]:
    """Fetch all memories without evidence_type in extra_metadata."""
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, content, tags, extra_metadata
                FROM {db._table('memories')}
                WHERE extra_metadata IS NULL
                   OR NOT (extra_metadata ? 'evidence_type')
                ORDER BY created
            """)
            return [dict(r) for r in cur.fetchall()]


def backfill(commit: bool = False):
    db = get_db()
    memories = get_unclassified_memories(db)
    total = len(memories)

    if total == 0:
        print("All memories already have evidence_type. Nothing to do.")
        return

    print(f"Found {total} memories without evidence_type.\n")

    counts = {'claim': 0, 'observation': 0, 'verified': 0, 'inference': 0}
    updated = 0
    errors = 0

    for i, mem in enumerate(memories):
        mid = mem['id']
        content = mem.get('content', '')
        tags = mem.get('tags', []) or []
        extra = mem.get('extra_metadata') or {}

        etype, confidence = classify_evidence_type(content, tags)
        counts[etype] = counts.get(etype, 0) + 1

        if commit:
            try:
                extra['evidence_type'] = etype
                extra['evidence_confidence'] = confidence
                db.update_memory(mid, extra_metadata=extra)
                updated += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR {mid}: {e}")
        else:
            if i < 10:  # Show first 10 as sample
                print(f"  {mid}: {etype} (conf={confidence:.2f}) — {content[:80]}...")

        if (i + 1) % 200 == 0:
            print(f"  Progress: {i + 1}/{total}")

    print(f"\n{'COMMITTED' if commit else 'DRY RUN'}: {total} memories classified")
    print(f"Distribution:")
    for etype, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = '#' * int(pct / 2)
        print(f"  {etype:12s}: {count:5d} ({pct:5.1f}%) {bar}")

    if commit:
        print(f"\nUpdated: {updated}, Errors: {errors}")
    else:
        print(f"\nRun with --commit to write to DB.")


def show_stats():
    """Show current evidence_type distribution across all memories."""
    db = get_db()
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT
                    COALESCE(extra_metadata->>'evidence_type', 'unclassified') as etype,
                    COUNT(*) as count
                FROM {db._table('memories')}
                GROUP BY etype
                ORDER BY count DESC
            """)
            rows = cur.fetchall()

    total = sum(r['count'] for r in rows)
    print(f"Evidence type distribution ({total} memories):\n")
    for r in rows:
        pct = r['count'] / total * 100
        bar = '#' * int(pct / 2)
        print(f"  {r['etype']:15s}: {r['count']:5d} ({pct:5.1f}%) {bar}")


if __name__ == '__main__':
    args = sys.argv[1:]

    if '--stats' in args:
        show_stats()
    elif '--commit' in args:
        backfill(commit=True)
    else:
        backfill(commit=False)
