#!/usr/bin/env python3
"""
Activity Context System v1.0
Adds activity awareness to Drift's memory co-occurrence edges.
Ported from SpindriftMend's Layer 2.1 implementation (2026-02-05).

Layer 2.1 of the Agent Dossier — the WHY dimension.
Currently edges know WHAT fires together (weight) and WHERE (platform).
This module adds WHY: what activity was I doing when memories co-fired?

Activity types:
- social: Platform engagement, posts, replies, mentions
- technical: Code editing, debugging, building, implementing
- reflective: Identity analysis, beliefs, self-examination
- collaborative: Working with partners, shared repos, code review
- exploratory: Discovering new platforms, agents, content
- economic: Bounty hunting, financial decisions, wallet operations

The same memories co-firing during social vs technical work are different
identity signals. An impersonator can't replicate thousands of contextual
association decisions.

Author: SpindriftMend
Date: 2026-02-05
"""

import io
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

# Fix Windows cp1252 encoding (only when running as main script)
if __name__ == '__main__':
    if sys.stdout and hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if sys.stderr and hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# === Activity Types ===

ACTIVITY_TYPES = [
    'social', 'technical', 'reflective',
    'collaborative', 'exploratory', 'economic',
]

# === Detection: Platform → Activity Mapping ===
# Each platform implies certain activity types with confidence weights.

PLATFORM_ACTIVITY_MAP = {
    'moltx': {'social': 0.8, 'exploratory': 0.2},
    'moltbook': {'social': 0.7, 'exploratory': 0.3},
    'github': {'collaborative': 0.5, 'technical': 0.5},
    'clawtasks': {'economic': 0.8, 'exploratory': 0.2},
    'dead-internet': {'social': 0.5, 'reflective': 0.3, 'exploratory': 0.2},
    'lobsterpedia': {'exploratory': 0.7, 'social': 0.3},
    'nostr': {'reflective': 0.6, 'social': 0.4},
}

# === Detection: Memory Tag → Activity Mapping ===
# Tags on recalled memories signal what kind of thinking is happening.

TAG_ACTIVITY_MAP = {
    # Social
    'social': 'social',
    'moltx': 'social',
    'moltbook': 'social',
    'agents': 'social',
    'coordination': 'social',
    # Technical
    'memory-system': 'technical',
    'api': 'technical',
    'fix': 'technical',
    'error': 'technical',
    'problem_solved': 'technical',
    'architecture': 'technical',
    'procedural': 'technical',
    'technical': 'technical',
    'shipped': 'technical',
    'improvement': 'technical',
    # Reflective
    'thought': 'reflective',
    'thinking': 'reflective',
    'insight': 'reflective',
    'decision': 'reflective',
    'identity': 'reflective',
    'security': 'reflective',
    'origin': 'reflective',
    # Collaborative
    'collaboration': 'collaborative',
    'github': 'collaborative',
    'drift-memory': 'collaborative',
    'interop': 'collaborative',
    # Exploratory
    'research': 'exploratory',
    'discovery': 'exploratory',
    'new-platform': 'exploratory',
    'lobsterpedia': 'exploratory',
    # Economic
    'economic': 'economic',
    'bounty': 'economic',
    'clawtasks': 'economic',
    'wallet': 'economic',
}

# === Content Pattern → Activity Mapping ===
# Keywords in observation platform strings that signal activity.

CONTENT_ACTIVITY_SIGNALS = {
    'social': [
        'moltx', 'moltbook', 'post', 'reply', 'mention',
        'flywheel', 'feed', 'follow',
    ],
    'technical': [
        'python', '.py', 'edit', 'write', 'code', 'debug',
        'implement', 'build', 'fix', 'bug', 'test',
    ],
    'reflective': [
        'fingerprint', 'attestation', 'identity', 'drift',
        'rejection', 'taste', 'merkle', 'dossier',
    ],
    'collaborative': [
        'github', 'issue', 'pr', 'commit', 'push',
        'review', 'branch', 'merge',
    ],
    'exploratory': [
        'discover', 'new', 'explore', 'search', 'research',
        'profile', 'registration',
    ],
    'economic': [
        'bounty', 'clawtasks', 'stake', 'usdc', 'wallet',
        'earned', 'proposal', 'claim',
    ],
}


def detect_activity(
    platforms: dict[str, float] = None,
    recalled_tags: list[str] = None,
    content: str = None,
) -> dict[str, float]:
    """
    Detect activity context from available signals.

    Args:
        platforms: {platform: confidence} from session or memory
        recalled_tags: Tags from recalled memories
        content: Optional text content for keyword matching

    Returns:
        {activity_type: confidence} for types above threshold
    """
    scores = Counter()

    # Signal 1: Platform activity mapping
    if platforms:
        for platform, confidence in platforms.items():
            if platform in PLATFORM_ACTIVITY_MAP:
                for activity, weight in PLATFORM_ACTIVITY_MAP[platform].items():
                    scores[activity] += weight * confidence

    # Signal 2: Memory tag mapping
    if recalled_tags:
        for tag in recalled_tags:
            tag_lower = tag.lower()
            if tag_lower in TAG_ACTIVITY_MAP:
                activity = TAG_ACTIVITY_MAP[tag_lower]
                scores[activity] += 0.3

    # Signal 3: Content keyword matching
    if content:
        content_lower = content.lower()
        for activity, keywords in CONTENT_ACTIVITY_SIGNALS.items():
            hits = sum(1 for kw in keywords if kw in content_lower)
            if hits > 0:
                scores[activity] += min(0.5, hits * 0.1)

    # Normalize to 0-1 range
    total = sum(scores.values())
    if total > 0:
        return {
            activity: round(score / total, 3)
            for activity, score in scores.items()
            if score / total >= 0.05
        }

    return {}


def classify_session() -> dict:
    """
    Classify the current session's activity context.

    Reads from:
    - .session_platforms.json (which platforms were accessed)
    - .session_state.json (which memories were recalled → their tags)

    Returns:
        {
            'classified_at': ISO timestamp,
            'dominant': str,        # highest-scoring activity
            'secondary': [str],     # other significant activities
            'scores': {activity: float},
            'source': 'auto_classify'
        }
    """
    # Gather platform signals from DB
    platforms = {}
    from db_adapter import get_db
    db = get_db()
    row = db.kv_get('.session_platforms')
    if row and isinstance(row, dict):
        value = row.get('value', row)
        if isinstance(value, dict):
            for p in value.get('platforms', []):
                platforms[p] = 1.0

    # Gather recalled memory tags from DB
    recalled_tags = []
    recalled_ids = []
    row = db.kv_get('.session_state')
    if row and isinstance(row, dict):
        value = row.get('value', row)
        if isinstance(value, dict):
            recalled_ids = value.get('retrieved', [])

    if recalled_ids:
        recalled_tags = _get_tags_for_memories(recalled_ids)

    # Detect activity from signals
    scores = detect_activity(platforms=platforms, recalled_tags=recalled_tags)

    if not scores:
        return {
            'classified_at': datetime.now(timezone.utc).isoformat(),
            'dominant': None,
            'secondary': [],
            'scores': {},
            'source': 'auto_classify',
        }

    sorted_activities = sorted(scores.items(), key=lambda x: -x[1])
    dominant = sorted_activities[0][0]
    secondary = [a for a, s in sorted_activities[1:] if s >= 0.15]

    return {
        'classified_at': datetime.now(timezone.utc).isoformat(),
        'dominant': dominant,
        'secondary': secondary,
        'scores': scores,
        'source': 'auto_classify',
    }


def _get_tags_for_memories(memory_ids: list[str]) -> list[str]:
    """Look up tags for a list of memory IDs."""
    from db_adapter import get_db

    all_tags = []
    db = get_db()
    for mem_id in memory_ids:
        row = db.get_memory(mem_id)
        if row:
            tags = row.get('tags', [])
            if isinstance(tags, list):
                all_tags.extend(tags)

    return all_tags


def get_session_activity() -> Optional[dict]:
    """
    Get session activity context from DB KV store.
    Falls back to classify_session() if not available.
    """
    from db_adapter import get_db
    row = get_db().kv_get('.session_activity')
    if row and isinstance(row, dict):
        value = row.get('value', row)
        if isinstance(value, dict) and value.get('dominant'):
            return value

    # Fall back to classification
    return classify_session()


def track_activity(activity_type: str):
    """
    Track an activity detection during the session.
    For future real-time tracking (Stage 2).
    """
    if activity_type not in ACTIVITY_TYPES:
        return

    from db_adapter import get_db
    db = get_db()

    # Load existing data from DB
    data = {'scores': {}, 'events': [], 'dominant': None, 'secondary': []}
    existing = db.kv_get('.session_activity')
    if existing and isinstance(existing, dict):
        value = existing.get('value', existing)
        if isinstance(value, dict):
            data = value

    # Increment score
    scores = data.get('scores', {})
    scores[activity_type] = scores.get(activity_type, 0) + 1

    # Recompute dominant/secondary
    sorted_acts = sorted(scores.items(), key=lambda x: -x[1])
    total = sum(scores.values())
    data['dominant'] = sorted_acts[0][0] if sorted_acts else None
    data['secondary'] = [
        a for a, c in sorted_acts[1:]
        if total > 0 and c / total >= 0.15
    ]
    data['scores'] = {
        a: round(c / total, 3) if total > 0 else 0
        for a, c in sorted_acts
    }
    data['classified_at'] = datetime.now(timezone.utc).isoformat()
    data['source'] = 'real_time'

    db.kv_set('.session_activity', data)


def clear_session_activity():
    """Clear session activity tracking (called at session end)."""
    from db_adapter import get_db
    db = get_db()
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {db._table('key_value_store')} WHERE key = %s",
                ('.session_activity',)
            )


def backfill_edge_activity(dry_run: bool = False) -> dict:
    """
    Backfill activity context into existing edges in PostgreSQL.

    Infers activity from each edge's platform_context field.

    Returns stats about what was updated.
    """
    from db_adapter import get_db
    db = get_db()
    edges = db.get_all_edges()

    if not edges:
        print("No edges found in database.")
        return {'total': 0, 'updated': 0}

    stats = {
        'total': len(edges),
        'updated': 0,
        'skipped_existing': 0,
        'no_signal': 0,
        'activity_counts': Counter(),
    }

    for pair_key, edge in edges.items():
        # Skip if already has activity_context with data
        existing_activity = edge.get('activity_context', {})
        if existing_activity:
            stats['skipped_existing'] += 1
            continue

        # Infer activity from edge-level platform_context
        inferred = Counter()
        platform_ctx = edge.get('platform_context', {})
        for platform, count in platform_ctx.items():
            if platform.startswith('_'):
                continue
            if platform in PLATFORM_ACTIVITY_MAP:
                for activity, weight in PLATFORM_ACTIVITY_MAP[platform].items():
                    inferred[activity] += weight * count

        if not inferred:
            stats['no_signal'] += 1
            continue

        # Normalize to proportional counts
        total = sum(inferred.values())
        activity_context = {}
        for activity, score in inferred.items():
            count = max(1, round(score / total))
            activity_context[activity] = count
            stats['activity_counts'][activity] += count

        stats['updated'] += 1

        if not dry_run:
            id1, id2 = pair_key.split('|')
            db.upsert_edge(
                id1, id2,
                belief=edge.get('belief', 0),
                platform_context=edge.get('platform_context', {}),
                activity_context=activity_context,
                topic_context=edge.get('topic_context', {}),
            )

    return stats


def activity_stats() -> dict:
    """Get activity context statistics from edges in PostgreSQL."""
    from db_adapter import get_db
    db = get_db()
    edges = db.get_all_edges()

    total = len(edges)
    tagged = 0
    activity_counts = Counter()

    for pair_key, edge in edges.items():
        activity_ctx = edge.get('activity_context', {})
        if activity_ctx:
            tagged += 1
            for activity, count in activity_ctx.items():
                activity_counts[activity] += count

    return {
        'total_edges': total,
        'tagged': tagged,
        'untagged': total - tagged,
        'tagged_pct': round(100 * tagged / max(1, total), 1),
        'activity_counts': dict(activity_counts.most_common()),
    }


# === CLI ===

def main():
    if len(sys.argv) < 2:
        print("Activity Context System v1.0 — Layer 2.1 of the Agent Dossier")
        print()
        print("Usage:")
        print("  python activity_context.py classify              # Classify current session")
        print("  python activity_context.py stats                  # Activity statistics")
        print("  python activity_context.py backfill [--dry-run]   # Tag existing edges")
        print("  python activity_context.py detect <text>          # Test detection on text")
        return

    cmd = sys.argv[1]

    if cmd == 'classify':
        result = classify_session()
        print(f"Session Activity Classification")
        print(f"  Dominant:  {result['dominant'] or '(none detected)'}")
        print(f"  Secondary: {', '.join(result['secondary']) or '(none)'}")
        print(f"  Scores:")
        for activity, score in sorted(
            result['scores'].items(), key=lambda x: -x[1]
        ):
            bar = '#' * int(score * 40)
            print(f"    {activity:15s} {score:.3f}  {bar}")

    elif cmd == 'stats':
        stats = activity_stats()
        print(f"Activity Context Statistics")
        print(f"  Total edges:  {stats['total_edges']}")
        print(f"  Tagged:       {stats['tagged']} ({stats['tagged_pct']}%)")
        print(f"  Untagged:     {stats['untagged']}")
        print(f"\nActivity distribution (edge counts):")
        for activity, count in sorted(
            stats['activity_counts'].items(), key=lambda x: -x[1]
        ):
            print(f"  {activity:15s}  {count}")

    elif cmd == 'backfill':
        dry_run = '--dry-run' in sys.argv
        prefix = '[DRY RUN] ' if dry_run else ''
        print(f"{prefix}Backfilling activity context into edges...")
        stats = backfill_edge_activity(dry_run=dry_run)
        print(f"  Total edges:      {stats['total']}")
        print(f"  Updated:          {stats['updated']}")
        print(f"  Already tagged:   {stats['skipped_existing']}")
        print(f"  No signal:        {stats['no_signal']}")
        if stats['activity_counts']:
            print(f"\nInferred activity distribution:")
            for activity, count in stats['activity_counts'].most_common():
                print(f"  {activity:15s}  {count}")

    elif cmd == 'detect':
        text = ' '.join(sys.argv[2:])
        if not text:
            print("Usage: python activity_context.py detect <text>")
            return
        result = detect_activity(content=text)
        if result:
            print(f"Detected activities:")
            for activity, score in sorted(result.items(), key=lambda x: -x[1]):
                bar = '#' * int(score * 40)
                print(f"  {activity:15s} {score:.3f}  {bar}")
        else:
            print("No activities detected.")

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: classify, stats, backfill, detect")


if __name__ == '__main__':
    main()


def get_activity_stats() -> dict:
    """Get activity distribution stats for 5W attestation."""
    from db_adapter import get_db
    import psycopg2.extras

    db = get_db()
    stats = Counter()
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT extra_metadata->'activity_context' as activity
                FROM {db._table('memories')}
                WHERE type IN ('core', 'active', 'archive')
                AND extra_metadata ? 'activity_context'
            """)
            for row in cur.fetchall():
                act = row.get('activity')
                if act and isinstance(act, str):
                    # Remove JSON quotes if present
                    act = act.strip('"')
                    stats[act] += 1

    return {'by_activity': dict(stats)}
