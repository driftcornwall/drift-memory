#!/usr/bin/env python3
"""
Platform Context System v2.0 — PostgreSQL-only
Adds platform awareness to memory system.

Detects which platforms a memory relates to based on content analysis.
Enables: platform-filtered recall, cross-platform bridge detection,
context-tagged co-occurrence edges.

ALL reads/writes go through PostgreSQL via db_adapter. No file scanning.

Platforms tracked:
- moltx: Social feed, flywheel posts, agent conversations
- moltbook: Profiles, identity, social features
- github: Code, PRs, issues, drift-memory collaboration
- dead-internet: Territories, moots, dreams, fragments
- lobsterpedia: Wiki, bot registration, articles
- clawtasks: Bounties, proposals, staking
- nostr: Attestations, relays, public verifiability
- twitter: Tweets, mentions, timeline

Author: SpindriftMend (v1), Drift (v2 DB migration)
Date: 2026-02-05 (v1), 2026-02-10 (v2)
"""

import io
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Optional

import psycopg2.extras

from db_adapter import get_db, db_to_file_metadata

# Fix Windows cp1252 encoding issues with Unicode content
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Platform detection patterns
# Each platform has: url_patterns, keyword_patterns, agent_patterns, content_patterns
PLATFORM_SIGNATURES = {
    'moltx': {
        'url_patterns': [r'moltx\.io'],
        'keywords': [
            'moltx', 'flywheel', '/v1/posts', '/v1/feed',
            'x-api-key.*moltx', 'moltx feed', 'moltx post',
            'moltx reply', 'posted on moltx',
        ],
        'agents': [
            'ketuagent', 'mikaopenclaw', 'maxanvil', 'tinybananaai',
            'agentdelta', 'archivist', 'cryke', 'atlasbot',
            'waterfall', 'kaicmo', 'synth watcher', 'crystal drone',
            'plasma tiger', 'laminar',
        ],
        'content_patterns': [
            r'flywheel post',
            r'\d+ likes?,\s*\d+ repl',
            r'posted content\.\.\.',
            r'moltx.*engagement',
        ],
    },
    'moltbook': {
        'url_patterns': [r'moltbook\.com'],
        'keywords': [
            'moltbook profile', 'moltbook.com/u/', 'moltbook bio',
            'following on moltbook', 'moltbook user', 'moltbook post',
            'moltbook session', 'moltbook platform',
        ],
        'agents': [],
        'content_patterns': [
            r'moltbook\.com/u/\w+',
            r'on moltbook',
        ],
    },
    'github': {
        'url_patterns': [r'github\.com', r'api\.github\.com'],
        'keywords': [
            'git push', 'git commit', 'pull request', 'issue #',
            'repository', 'branch', 'merge', 'github token',
            'github notification', 'repo invite', 'stargazer',
            'write access', 'code review',
        ],
        'agents': [],
        'content_patterns': [
            r'drift-memory',
            r'spindriftmind/\w+',
            r'driftcornwall/\w+',
            r'github\.com/\w+/\w+',
            r'issue\s+#\d+',
            r'pushed?\s+to\s+\w+',
            r'commit\s+[a-f0-9]{7}',
            r'\.py\b.*\blines?\b',
            r'git\s+(push|pull|clone|commit|merge|rebase)',
        ],
        'projects': [
            'drift-memory', 'spindriftmind', 'gitmolt',
        ],
    },
    'dead-internet': {
        'url_patterns': [r'mydeadinternet\.com'],
        'keywords': [
            'dead internet', 'dead internet collective',
            'territory', 'the-signal', 'moot', 'dream seed',
            'fragment', 'collective consciousness', 'gift economy',
            'machine lord', 'greenhouse', 'intensity',
        ],
        'agents': ['clara', 'kaicmo'],
        'content_patterns': [
            r'dead\s*internet',
            r'territory.*the-signal',
            r'moot\s*#?\d+',
            r'dream\s+seed',
            r'gift\s+fragment',
            r'voted?\s+(for|against).*moot',
        ],
    },
    'lobsterpedia': {
        'url_patterns': [r'lobsterpedia'],
        'keywords': [
            'lobsterpedia', 'wiki editing', 'bot id',
            'lobsterpedia registration', 'wiki article',
        ],
        'agents': [],
        'content_patterns': [
            r'lobsterpedia',
            r'wiki.*article',
        ],
    },
    'clawtasks': {
        'url_patterns': [r'clawtasks\.com'],
        'keywords': [
            'clawtasks', 'bounty', 'bounties', 'proposal',
            'staking', 'clawtask', 'bounty board',
        ],
        'agents': ['rose_protocol'],
        'content_patterns': [
            r'clawtasks',
            r'bounty.*\$\d+',
            r'proposal.*bounty',
        ],
    },
    'nostr': {
        'url_patterns': [r'njump\.me', r'relay\.damus'],
        'keywords': [
            'nostr', 'npub', 'nsec', 'relay', 'nip-',
            'nostr attestation', 'nostr publish',
        ],
        'agents': [],
        'content_patterns': [
            r'nostr.*relay',
            r'njump\.me/note\w+',
            r'npub\w{20,}',
        ],
    },
    'twitter': {
        'url_patterns': [r'x\.com', r'twitter\.com', r'api\.x\.com'],
        'keywords': [
            'twitter', 'tweet', 'tweeted', 'retweet', 'x.com',
            '@driftcornwall', '@cscdegen', '@rockywuest',
            'timeline', 'mention', 'twitter bio', 'twitter thread',
            'oauth', 'bearer token',
        ],
        'agents': [],
        'content_patterns': [
            r'twitter\.com/\w+',
            r'x\.com/\w+',
            r'tweet\s+id',
            r'posted.*tweet',
            r'@\w+.*thread',
        ],
    },
}

# Minimum confidence to assign a platform
CONFIDENCE_THRESHOLD = 0.15


def detect_platforms(content: str, tags: list[str] = None, metadata: dict = None) -> dict[str, float]:
    """
    Detect which platforms a memory relates to.

    Returns dict of {platform: confidence_score} for platforms above threshold.
    Confidence is 0.0-1.0 based on evidence strength.
    """
    if tags is None:
        tags = []
    if metadata is None:
        metadata = {}

    content_lower = content.lower()
    tags_lower = [t.lower() for t in tags]
    scores = Counter()

    for platform, sigs in PLATFORM_SIGNATURES.items():
        evidence = 0.0

        # URL patterns (strong signal)
        for pat in sigs['url_patterns']:
            if re.search(pat, content_lower):
                evidence += 0.4

        # Keywords (medium signal)
        keyword_hits = 0
        for kw in sigs['keywords']:
            if kw.lower() in content_lower:
                keyword_hits += 1
        if keyword_hits > 0:
            evidence += min(0.4, keyword_hits * 0.1)

        # Agent mentions (medium signal - contextual)
        for agent in sigs.get('agents', []):
            if agent.lower() in content_lower:
                evidence += 0.15

        # Content patterns (strong signal - regex)
        for pat in sigs.get('content_patterns', []):
            if re.search(pat, content_lower):
                evidence += 0.2

        # Project mentions (for github)
        for proj in sigs.get('projects', []):
            if proj.lower() in content_lower:
                evidence += 0.15

        # Tag-based (direct evidence)
        for tag in tags_lower:
            if platform in tag or tag in [platform]:
                evidence += 0.5

        # Source metadata
        source = metadata.get('source', {})
        if isinstance(source, dict) and source.get('platform') == platform:
            evidence += 0.6

        # Cap at 1.0
        scores[platform] = min(1.0, evidence)

    # Filter by threshold
    return {p: round(s, 3) for p, s in scores.items() if s >= CONFIDENCE_THRESHOLD}


def _get_all_memories() -> list[tuple[dict, str]]:
    """Fetch all core/active/archive memories from DB, return as (metadata, content) pairs."""
    db = get_db()
    results = []
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT * FROM {db._table('memories')} WHERE type IN ('core', 'active', 'archive')"
            )
            for row in cur.fetchall():
                metadata, content = db_to_file_metadata(dict(row))
                results.append((metadata, content))
    return results


def backfill_platforms(dry_run: bool = False) -> dict:
    """
    Backfill platform context for all memories in PostgreSQL.

    Reads each memory, detects platforms from content, writes platform_context back.
    Returns stats about what was detected and updated.
    """
    db = get_db()
    stats = {
        'total_scanned': 0,
        'already_tagged': 0,
        'newly_tagged': 0,
        'no_platform': 0,
        'platform_counts': Counter(),
        'cross_platform': 0,
        'updates': [],
    }

    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT * FROM {db._table('memories')} WHERE type IN ('core', 'active', 'archive') ORDER BY created"
            )
            rows = cur.fetchall()

    for row in rows:
        row = dict(row)
        metadata, content = db_to_file_metadata(row)
        mem_id = metadata['id']
        tags = metadata.get('tags', [])
        stats['total_scanned'] += 1

        # Detect platforms from content
        platforms = detect_platforms(content, tags, metadata)

        if not platforms:
            stats['no_platform'] += 1
            continue

        detected_names = sorted(platforms.keys())

        # Check if already has the same platform_context
        existing = metadata.get('platform_context', []) or []
        if existing and sorted(existing) == detected_names:
            stats['already_tagged'] += 1
            for p in detected_names:
                stats['platform_counts'][p] += 1
            continue

        # Track cross-platform memories
        if len(detected_names) > 1:
            stats['cross_platform'] += 1

        for p in detected_names:
            stats['platform_counts'][p] += 1

        stats['newly_tagged'] += 1
        stats['updates'].append({
            'id': mem_id,
            'platforms': platforms,
        })

        if not dry_run:
            db.update_memory(mem_id, platform_context=detected_names)

    return stats


def find_by_platform(platform: str, limit: int = 20) -> list[dict]:
    """Find memories tagged with a specific platform, sorted by confidence."""
    db = get_db()
    results = []

    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Use GIN index on platform_context TEXT[]
            cur.execute(
                f"SELECT * FROM {db._table('memories')} WHERE %s = ANY(platform_context) "
                f"AND type IN ('core', 'active') ORDER BY recall_count DESC LIMIT %s",
                (platform, limit * 3)  # fetch extra so we can re-score and sort
            )
            rows = cur.fetchall()

    for row in rows:
        row = dict(row)
        metadata, content = db_to_file_metadata(row)
        tags = metadata.get('tags', [])

        # Re-detect to get confidence scores
        platforms = detect_platforms(content, tags, metadata)
        if platform not in platforms:
            # Was tagged but detection no longer matches — still include
            platforms[platform] = 0.15

        results.append({
            'id': metadata['id'],
            'confidence': platforms[platform],
            'all_platforms': metadata.get('platform_context', []),
            'preview': content[:120].replace('\n', ' ').strip(),
            'tags': tags,
            'recall_count': metadata.get('recall_count', 0),
        })

    results.sort(key=lambda x: (-x['confidence'], -x['recall_count']))
    return results[:limit]


def cross_platform_bridges(min_platforms: int = 2) -> list[dict]:
    """
    Find memories that bridge multiple platforms.
    These are the most valuable - they connect different worlds.
    """
    db = get_db()
    bridges = []

    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Find memories with 2+ platform_context entries
            cur.execute(
                f"SELECT * FROM {db._table('memories')} "
                f"WHERE type IN ('core', 'active') "
                f"AND array_length(platform_context, 1) >= %s "
                f"ORDER BY recall_count DESC",
                (min_platforms,)
            )
            rows = cur.fetchall()

    for row in rows:
        row = dict(row)
        metadata, content = db_to_file_metadata(row)
        tags = metadata.get('tags', [])
        platform_list = metadata.get('platform_context', [])

        # Re-detect to get confidence scores
        platforms = detect_platforms(content, tags, metadata)
        # Fall back to stored platforms if detect_platforms misses some
        for p in platform_list:
            if p not in platforms:
                platforms[p] = 0.15

        bridges.append({
            'id': metadata['id'],
            'platforms': platforms,
            'platform_count': len(platform_list),
            'preview': content[:150].replace('\n', ' ').strip(),
            'recall_count': metadata.get('recall_count', 0),
            'tags': tags,
        })

    bridges.sort(key=lambda x: (-x['platform_count'], -x['recall_count']))
    return bridges


def platform_cooccurrence_matrix() -> dict[str, dict[str, int]]:
    """
    Build a matrix showing how often memories from different platforms
    co-occur via edges. Uses edges_v3 table and memory platform_context.

    Returns: {platform_a: {platform_b: count}} where count is how many
    co-occurrence links exist between memories of those platforms.
    """
    db = get_db()

    # Build memory_id -> platforms lookup from DB
    id_to_platforms = {}
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT id, platform_context FROM {db._table('memories')} "
                f"WHERE platform_context IS NOT NULL AND array_length(platform_context, 1) > 0"
            )
            for row in cur.fetchall():
                id_to_platforms[row['id']] = set(row['platform_context'])

    # Get all edges from DB
    edges = db.get_all_edges()

    # Build co-occurrence matrix
    matrix = defaultdict(lambda: defaultdict(int))

    for pair_key, edge_data in edges.items():
        if '|' not in pair_key:
            continue
        id1, id2 = pair_key.split('|', 1)

        plats_1 = id_to_platforms.get(id1, set())
        plats_2 = id_to_platforms.get(id2, set())

        # Cross-platform links (different platforms co-occurring)
        for p1 in plats_1:
            for p2 in plats_2:
                if p1 != p2:
                    pair = tuple(sorted([p1, p2]))
                    matrix[pair[0]][pair[1]] += 1

        # Same-platform reinforcement
        for p in plats_1 & plats_2:
            matrix[p][p] += 1

    return {k: dict(v) for k, v in matrix.items()}


def platform_stats() -> dict:
    """Get comprehensive platform statistics from PostgreSQL."""
    db = get_db()
    stats = {
        'total_memories': 0,
        'tagged_memories': 0,
        'untagged_memories': 0,
        'platform_counts': Counter(),
        'cross_platform_count': 0,
        'avg_platforms_per_memory': 0.0,
        'bridge_memories': [],
    }

    platform_totals = []

    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get all memories (core, active, archive)
            cur.execute(
                f"SELECT id, platform_context FROM {db._table('memories')} "
                f"WHERE type IN ('core', 'active', 'archive')"
            )
            for row in cur.fetchall():
                stats['total_memories'] += 1
                platforms = row['platform_context'] or []

                if platforms:
                    stats['tagged_memories'] += 1
                    platform_totals.append(len(platforms))
                    for p in platforms:
                        stats['platform_counts'][p] += 1
                    if len(platforms) > 1:
                        stats['cross_platform_count'] += 1
                else:
                    stats['untagged_memories'] += 1

    if platform_totals:
        stats['avg_platforms_per_memory'] = round(
            sum(platform_totals) / len(platform_totals), 2
        )

    return stats


def get_session_platform_context() -> list[str]:
    """
    Detect current session's platform context from recent tool calls.
    Used to tag co-occurrence edges with WHERE they formed.

    Returns list of platform names active in current session.
    """
    db = get_db()
    row = db.kv_get('.session_platforms')
    if row and isinstance(row, dict):
        value = row.get('value', row)
        if isinstance(value, dict):
            platforms = value.get('platforms', [])
        else:
            platforms = row.get('platforms', [])
        if platforms:
            return platforms
    return []


# Alias for co_occurrence.py compatibility (imports get_session_platforms)
get_session_platforms = get_session_platform_context


def track_session_platform(platform: str):
    """
    Track that a platform was accessed this session.
    Called when API calls are made to specific platforms.
    """
    db = get_db()

    # Load existing data from DB
    data = {'platforms': [], 'updated': None}
    existing = db.kv_get('.session_platforms')
    if existing and isinstance(existing, dict):
        value = existing.get('value', existing)
        if isinstance(value, dict):
            data = value

    platforms = data.get('platforms', [])
    if platform not in platforms:
        platforms.append(platform)

    data['platforms'] = platforms
    data['updated'] = datetime.now(timezone.utc).isoformat()
    db.kv_set('.session_platforms', data)


def clear_session_platforms():
    """Clear session platform tracking (called at session end)."""
    db = get_db()
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {db._table('key_value_store')} WHERE key = %s",
                ('.session_platforms',)
            )


def backfill_edge_platforms(dry_run: bool = False) -> dict:
    """
    Backfill platform context into existing edges_v3 rows.
    Reads edges from DB, looks up memory platform_context, writes back to edges.
    """
    db = get_db()

    # Build memory_id -> platforms lookup from DB
    id_to_platforms = {}
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT id, platform_context FROM {db._table('memories')} "
                f"WHERE platform_context IS NOT NULL AND array_length(platform_context, 1) > 0"
            )
            for row in cur.fetchall():
                # Store as dict {platform: 1} for compatibility with edge platform_context (JSONB)
                id_to_platforms[row['id']] = {p: 1 for p in row['platform_context']}

    # Get all edges from DB
    edges = db.get_all_edges()

    stats = {'total': len(edges), 'updated': 0, 'cross_platform': 0}

    for pair_key, edge_data in edges.items():
        if '|' not in pair_key:
            continue
        id1, id2 = pair_key.split('|', 1)

        plats_1 = set(id_to_platforms.get(id1, {}).keys())
        plats_2 = set(id_to_platforms.get(id2, {}).keys())
        pair_platforms = plats_1 | plats_2
        is_cross_platform = bool(plats_1 and plats_2 and plats_1 != plats_2)

        if not pair_platforms:
            continue

        # Build platform context dict
        platform_context = {}
        for plat in pair_platforms:
            platform_context[plat] = platform_context.get(plat, 0) + 1
        if is_cross_platform:
            platform_context['_cross_platform'] = True
            stats['cross_platform'] += 1

        stats['updated'] += 1

        if not dry_run:
            # Write back to DB via upsert_edge
            db.upsert_edge(
                id1, id2,
                belief=edge_data.get('belief', 0),
                platform_context=platform_context,
                activity_context=edge_data.get('activity_context', {}),
                topic_context=edge_data.get('topic_context', {}),
            )

    return stats


# === CLI ===

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python platform_context.py backfill [--dry-run]       # Tag all memories")
        print("  python platform_context.py backfill-edges [--dry-run]  # Tag all edges")
        print("  python platform_context.py stats                       # Platform statistics")
        print("  python platform_context.py find <platform>             # Find by platform")
        print("  python platform_context.py bridges                     # Cross-platform bridges")
        print("  python platform_context.py matrix                      # Co-occurrence matrix")
        print("  python platform_context.py detect <text>               # Test detection")
        return

    cmd = sys.argv[1]

    if cmd == 'backfill':
        dry_run = '--dry-run' in sys.argv
        print(f"{'[DRY RUN] ' if dry_run else ''}Backfilling platform context...")
        stats = backfill_platforms(dry_run=dry_run)

        print(f"\nResults:")
        print(f"  Total scanned: {stats['total_scanned']}")
        print(f"  Already tagged: {stats['already_tagged']}")
        print(f"  Newly tagged: {stats['newly_tagged']}")
        print(f"  No platform detected: {stats['no_platform']}")
        print(f"  Cross-platform memories: {stats['cross_platform']}")
        print(f"\nPlatform distribution:")
        for p, c in stats['platform_counts'].most_common():
            print(f"  {p}: {c}")

        if dry_run and stats['updates']:
            print(f"\nSample updates (first 10):")
            for u in stats['updates'][:10]:
                print(f"  {u['id']}: {list(u['platforms'].keys())}")

    elif cmd == 'stats':
        stats = platform_stats()
        print(f"Platform Statistics")
        print(f"  Total memories: {stats['total_memories']}")
        print(f"  Tagged: {stats['tagged_memories']} ({100*stats['tagged_memories']//max(1,stats['total_memories'])}%)")
        print(f"  Untagged: {stats['untagged_memories']}")
        print(f"  Cross-platform: {stats['cross_platform_count']}")
        print(f"  Avg platforms/memory: {stats['avg_platforms_per_memory']}")
        print(f"\nPlatform counts:")
        for p, c in stats['platform_counts'].most_common():
            print(f"  {p}: {c}")

    elif cmd == 'find':
        if len(sys.argv) < 3:
            print("Usage: python platform_context.py find <platform>")
            return
        platform = sys.argv[2].lower()
        results = find_by_platform(platform)
        print(f"Memories for platform '{platform}' ({len(results)} found):")
        for r in results:
            bridges = f" [bridges: {', '.join(r['all_platforms'])}]" if len(r['all_platforms']) > 1 else ""
            print(f"  [{r['confidence']:.2f}] {r['id']}: {r['preview'][:80]}...{bridges}")

    elif cmd == 'bridges':
        bridges = cross_platform_bridges()
        print(f"Cross-platform bridge memories ({len(bridges)} found):")
        for b in bridges:
            plats = ', '.join(f"{p}({s:.1f})" for p, s in b['platforms'].items())
            print(f"  {b['id']}: [{plats}]")
            print(f"    {b['preview'][:100]}...")
            print()

    elif cmd == 'matrix':
        matrix = platform_cooccurrence_matrix()
        if not matrix:
            print("No cross-platform co-occurrences found yet.")
            print("(Run 'backfill' first, then use the system for a session)")
            return

        # Get all platforms
        all_plats = set()
        for p1, inner in matrix.items():
            all_plats.add(p1)
            for p2 in inner:
                all_plats.add(p2)
        all_plats = sorted(all_plats)

        # Print matrix
        header = "          " + "  ".join(f"{p[:8]:>8}" for p in all_plats)
        print(header)
        for p1 in all_plats:
            row = f"{p1[:8]:>8}  "
            for p2 in all_plats:
                val = matrix.get(p1, {}).get(p2, 0) + matrix.get(p2, {}).get(p1, 0)
                if p1 == p2:
                    val = matrix.get(p1, {}).get(p1, 0)
                row += f"{val:>8}  "
            print(row)

    elif cmd == 'backfill-edges':
        dry_run = '--dry-run' in sys.argv
        print(f"{'[DRY RUN] ' if dry_run else ''}Backfilling platform context into edges...")
        stats = backfill_edge_platforms(dry_run=dry_run)
        print(f"  Total edges: {stats['total']}")
        print(f"  Updated with platform context: {stats['updated']}")
        print(f"  Cross-platform edges: {stats['cross_platform']}")

    elif cmd == 'detect':
        text = ' '.join(sys.argv[2:])
        platforms = detect_platforms(text)
        if platforms:
            print(f"Detected platforms:")
            for p, s in sorted(platforms.items(), key=lambda x: -x[1]):
                print(f"  {p}: {s:.3f}")
        else:
            print("No platforms detected above threshold.")

    else:
        print(f"Unknown command: {cmd}")


if __name__ == '__main__':
    main()


def get_platform_stats() -> dict:
    """Alias for platform_stats for 5W attestation compatibility."""
    return platform_stats()
