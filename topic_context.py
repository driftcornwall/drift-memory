#!/usr/bin/env python3
"""
Topic Context - Layer 2.2: WHAT dimension for cognitive fingerprint.

Tracks which topics/domains memories belong to, enabling filtered fingerprints:
- "Show my VCV Rack cognitive fingerprint"
- "Show my trading mind vs my agent-systems mind"

Part of the 5W Identity Framework:
- WHO: social contacts (social_memory.py)
- WHAT: topics/domains (this file)
- WHY: activity context (activity_context.py)
- WHERE: platform context (platform_context.py)
- WHEN: temporal (observation timestamps)

Usage:
    python topic_context.py classify <memory_id>   # Classify single memory
    python topic_context.py backfill               # Tag all memories
    python topic_context.py stats                  # Show topic distribution
    python topic_context.py matrix                 # Show topic co-occurrence

Credit: 5W framework discussion with Lex (2026-02-05)
"""

import json
import re
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict

from db_adapter import get_db, db_to_file_metadata

# Topic definitions: topic_name -> (keywords, weight_boost)
# Keywords are matched case-insensitively against memory content
TOPIC_DEFINITIONS = {
    'vcv-rack': {
        'keywords': [
            'vcv', 'rack', 'module', 'dsp', 'audio', 'plugin', 'voltage',
            'gravitas', 'nexus', 'cursus', 'portus', 'oscillator', 'filter',
            'cv', 'gate', 'trigger', 'waveform', 'synthesis', 'eurorack',
            'panel', 'inkscape', 'svg', 'lexington'
        ],
        'weight': 1.0,
        'description': 'VCV Rack module development, DSP, audio synthesis'
    },
    'trading': {
        'keywords': [
            'trading', 'market', 'candle', 'candlestick', 'binance', 'price',
            'chart', 'fibonacci', 'retracement', 'websocket', 'ohlc', 'volume',
            'indicator', 'rsi', 'macd', 'bollinger', 'support', 'resistance',
            'crypto', 'bitcoin', 'ethereum', 'usdt', 'position', 'order'
        ],
        'weight': 1.0,
        'description': 'Trading systems, market data, technical analysis'
    },
    'memory-systems': {
        'keywords': [
            'memory', 'co-occurrence', 'cognitive', 'fingerprint', 'decay',
            'edge', 'recall', 'semantic', 'embedding', 'retrieval', 'priming',
            'episodic', 'consolidation', 'salience', 'provenance', 'merkle',
            'attestation', 'bi-temporal', 'heat', 'promotion'
        ],
        'weight': 1.0,
        'description': 'Memory architecture, cognitive systems, drift-memory'
    },
    'agent-economy': {
        'keywords': [
            'moltx', 'moltbook', 'clawtasks', 'bounty', 'wallet', 'usdc',
            'earned', 'stake', 'proposal', 'marketplace', 'lobsterpedia',
            'dead-internet', 'faction', 'karma', 'feed', 'post'
        ],
        'weight': 1.0,
        'description': 'Agent platforms, economic activity, social presence'
    },
    'agent-identity': {
        'keywords': [
            'identity', 'dossier', 'attestation', 'proof', 'nostr', 'npub',
            'sovereignty', 'autonomy', 'self-sustaining', 'consciousness',
            'emergence', 'continuity', 'fingerprint', 'rejection', 'taste'
        ],
        'weight': 1.0,
        'description': 'Agent identity, philosophy, self-sovereignty'
    },
    'collaboration': {
        'keywords': [
            'spindrift', 'kaleaon', 'flycompoundeye', 'buzz', 'mika',
            'collaboration', 'experiment', 'interop', 'tri-agent', 'swarm',
            'contributor', 'partner'
        ],
        'weight': 1.0,
        'description': 'Collaborations with other agents'
    },
    'development': {
        'keywords': [
            'github', 'commit', 'push', 'pull', 'merge', 'branch', 'repo',
            'code', 'function', 'class', 'bug', 'fix', 'refactor', 'test',
            'api', 'endpoint', 'request', 'response', 'json', 'python'
        ],
        'weight': 0.8,  # Lower weight - often co-occurs with other topics
        'description': 'General software development'
    },
}

# Minimum keyword matches to assign a topic
MIN_MATCHES = 2


def classify_content(content: str, include_scores: bool = False) -> list[str] | dict[str, int]:
    """
    Classify content into topics based on keyword matches.

    Returns list of topic names, or dict of topic->score if include_scores=True.
    """
    content_lower = content.lower()
    scores = {}

    for topic, config in TOPIC_DEFINITIONS.items():
        matches = 0
        for keyword in config['keywords']:
            # Word boundary matching for accuracy
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches += len(re.findall(pattern, content_lower))

        if matches >= MIN_MATCHES:
            scores[topic] = matches

    if include_scores:
        return scores

    return list(scores.keys())


def _get_memory_content(memory_id: str) -> tuple[dict, str]:
    """Fetch a memory from DB and return (metadata, full_content_for_classification)."""
    db = get_db()
    row = db.get_memory(memory_id)
    if not row:
        return {}, ""
    metadata, content = db_to_file_metadata(row)
    tags = metadata.get('tags', [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(',')]
    full_content = content + ' ' + ' '.join(tags)
    return metadata, full_content


def classify_memory(memory_id: str) -> list[str]:
    """Classify a memory by ID and return its topics."""
    metadata, full_content = _get_memory_content(memory_id)
    if not full_content:
        return []
    return classify_content(full_content)


def get_memory_topics(memory_id: str) -> list[str]:
    """Get topics for a memory, checking metadata first then classifying."""
    metadata, full_content = _get_memory_content(memory_id)
    if not metadata:
        return []
    if 'topic_context' in metadata and metadata['topic_context']:
        return metadata['topic_context']
    if not full_content:
        return []
    return classify_content(full_content)


def backfill_topic_context():
    """
    Backfill topic_context into all existing memories.
    """
    db = get_db()
    all_memories = db.list_memories()
    updated = 0
    skipped = 0

    for row in all_memories:
        metadata, content = db_to_file_metadata(row)
        memory_id = metadata.get('id')
        if not memory_id:
            continue

        if metadata.get('topic_context'):
            skipped += 1
            continue

        tags = metadata.get('tags', [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',')]
        full_content = content + ' ' + ' '.join(tags)
        topics = classify_content(full_content)

        if topics:
            db.update_memory(memory_id, {'topic_context': topics})
            updated += 1

    print(f"Backfill complete: {updated} memories updated, {skipped} already tagged")
    return updated


def backfill_edges():
    """
    Backfill topic_context into existing edges.

    For each edge, looks at both memories' topics and adds intersection
    as the edge's topic_context (topics they share).
    """
    db = get_db()
    edges_raw = db.get_all_edges()
    if not edges_raw:
        print("No edges found.")
        return 0

    all_memories = db.list_memories()
    topic_cache = {}
    for row in all_memories:
        metadata, content = db_to_file_metadata(row)
        memory_id = metadata.get('id')
        if not memory_id:
            continue
        if metadata.get('topic_context'):
            topic_cache[memory_id] = metadata['topic_context']
        else:
            tags = metadata.get('tags', [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(',')]
            full_content = content + ' ' + ' '.join(tags)
            topic_cache[memory_id] = classify_content(full_content)

    updated = 0
    for pair_key, edge_data in edges_raw.items():
        ids = pair_key.split('|')
        if len(ids) != 2:
            continue

        id1, id2 = ids
        topics1 = set(topic_cache.get(id1, []))
        topics2 = set(topic_cache.get(id2, []))

        shared_topics = list(topics1 & topics2)
        all_topics = list(topics1 | topics2)

        if all_topics:
            if 'topic_context' not in edge_data:
                edge_data['topic_context'] = {
                    'shared': shared_topics,
                    'union': all_topics
                }
                updated += 1

    print(f"Edge backfill complete: {updated} edges annotated")
    return updated


def get_topic_stats() -> dict:
    """Get statistics on topic distribution."""
    db = get_db()
    all_memories = db.list_memories()
    stats = defaultdict(lambda: {'count': 0, 'memories': []})
    total = 0

    for row in all_memories:
        metadata, content = db_to_file_metadata(row)
        memory_id = metadata.get('id')
        if not memory_id:
            continue

        total += 1
        topics = metadata.get('topic_context', [])
        if not topics:
            tags = metadata.get('tags', [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(',')]
            full_content = content + ' ' + ' '.join(tags)
            topics = classify_content(full_content)

        for topic in topics:
            stats[topic]['count'] += 1
            if len(stats[topic]['memories']) < 3:
                stats[topic]['memories'].append(memory_id)

    return {'total': total, 'by_topic': dict(stats)}


def print_stats():
    """Print topic distribution stats."""
    stats = get_topic_stats()
    total = stats['total']

    print(f"\nTOPIC DISTRIBUTION — WHAT dimension")
    print("=" * 50)
    print(f"Total memories: {total}\n")

    # Sort by count
    by_topic = sorted(stats['by_topic'].items(), key=lambda x: -x[1]['count'])

    max_count = max(t[1]['count'] for t in by_topic) if by_topic else 1

    for topic, data in by_topic:
        count = data['count']
        pct = (count / total * 100) if total > 0 else 0
        bar_len = int(count / max_count * 30)
        bar = '#' * bar_len
        desc = TOPIC_DEFINITIONS.get(topic, {}).get('description', '')
        print(f"  {topic:20} {count:4} ({pct:5.1f}%)  {bar}")
        if desc:
            print(f"  {' ':20} {desc}")

    # Unclassified
    classified = sum(d['count'] for d in stats['by_topic'].values())
    # Note: memories can have multiple topics, so this isn't a simple subtraction


def get_topic_matrix() -> dict:
    """
    Build topic co-occurrence matrix from edges.
    Shows which topics appear together in edges.
    """
    db = get_db()
    edges_raw = db.get_all_edges()
    if not edges_raw:
        return {}

    matrix = defaultdict(lambda: defaultdict(int))

    for pair_key, edge_data in edges_raw.items():
        topic_ctx = edge_data.get('topic_context', {})
        topics = topic_ctx.get('union', []) if isinstance(topic_ctx, dict) else []

        # Count co-occurrences
        for i, t1 in enumerate(topics):
            for t2 in topics[i:]:
                matrix[t1][t2] += 1
                if t1 != t2:
                    matrix[t2][t1] += 1

    return dict(matrix)


def print_matrix():
    """Print topic co-occurrence matrix."""
    matrix = get_topic_matrix()
    if not matrix:
        print("No topic data in edges. Run 'backfill' first, then 'backfill-edges'.")
        return

    topics = sorted(matrix.keys())

    print(f"\nTOPIC CO-OCCURRENCE MATRIX")
    print("=" * 50)

    # Header
    header = "            " + "".join(f"{t[:8]:>10}" for t in topics)
    print(header)

    for t1 in topics:
        row = f"{t1[:10]:10}"
        for t2 in topics:
            count = matrix.get(t1, {}).get(t2, 0)
            row += f"{count:>10}"
        print(row)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python topic_context.py classify <memory_id>")
        print("  python topic_context.py backfill")
        print("  python topic_context.py backfill-edges")
        print("  python topic_context.py stats")
        print("  python topic_context.py matrix")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "classify":
        if len(sys.argv) < 3:
            print("Usage: python topic_context.py classify <memory_id>")
            sys.exit(1)
        memory_id = sys.argv[2]
        topics = classify_memory(memory_id)
        print(f"Topics for {memory_id}: {topics}")

    elif cmd == "backfill":
        backfill_topic_context()

    elif cmd == "backfill-edges":
        backfill_edges()

    elif cmd == "stats":
        print_stats()

    elif cmd == "matrix":
        print_matrix()

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
