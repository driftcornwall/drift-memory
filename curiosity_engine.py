#!/usr/bin/env python3
"""
Curiosity Engine — Directed exploration of sparse graph regions.

Instead of random dead memory excavation, this module identifies WHERE
the co-occurrence graph is thin and directs attention there. The insight:
random exploration wastes compute, but curiosity-driven exploration grows
the graph where it needs growth most.

Four scoring components:
1. Isolation score    — memories with few co-occurrence edges (lonely nodes)
2. Bridging score     — memories that could connect disparate clusters
3. Domain gap score   — cognitive domains with few memories relative to others
4. Survivor score     — old memories that survived decay but are never recalled

The composite curiosity score identifies the best targets for directed
exploration — memories that, if recalled together, would maximally grow
the co-occurrence graph.

DB-ONLY: All reads/writes go through PostgreSQL via db_adapter.

Usage:
    python curiosity_engine.py scan              # Top curiosity targets
    python curiosity_engine.py map               # Sparse region analysis
    python curiosity_engine.py stats             # Exploration history
    python curiosity_engine.py targets [N]       # Get N targets for priming
    python curiosity_engine.py log-conversion    # Log successful exploration
"""

import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Optional

from db_adapter import get_db, db_to_file_metadata
from memory_common import MEMORY_ROOT

# --- Configuration ---

# Weight of each component in the composite curiosity score
ISOLATION_WEIGHT = 0.35      # Lonely nodes need connections most
BRIDGING_WEIGHT = 0.25       # Bridge candidates create structural value
DOMAIN_GAP_WEIGHT = 0.25     # Domain balance prevents cognitive blind spots
SURVIVOR_WEIGHT = 0.15       # Old survivors might be forgotten treasures

# Thresholds
MIN_MEMORIES_FOR_CURIOSITY = 50     # Don't run on tiny graphs
LOW_DEGREE_THRESHOLD = 3            # Memories with <= this many edges are "isolated"
HIGH_DEGREE_THRESHOLD = 20          # Memories with >= this many edges are "hubs"
SURVIVOR_MIN_SESSIONS = 10          # Must have survived this many sessions
SURVIVOR_MAX_RECALLS = 2            # But been recalled at most this many times
CURIOSITY_TOP_N = 10                # Default number of targets to surface

# Cognitive domains (shared with memory_manager.py priming)
COGNITIVE_DOMAINS = {
    'reflection': ['thought', 'thinking', 'output', 'source:self'],
    'social': ['social', 'collaboration', 'spindrift', 'spindriftmend',
               'kaleaon', 'moltx', 'moltbook'],
    'technical': ['insight', 'problem_solved', 'error', 'bug', 'fix',
                  'resolution', 'memory-system', 'architecture', 'api'],
    'economic': ['economic', 'bounty', 'clawtasks', 'wallet', 'earned'],
    'identity': ['identity', 'values', 'milestone', 'shipped', 'dossier',
                 'attestation', 'critical'],
}

# KV keys for persistence
KV_CURIOSITY_LOG = '.curiosity_log'
KV_CURIOSITY_TARGETS = '.curiosity_targets'


# --- Graph Analysis ---

def _build_degree_map() -> dict[str, int]:
    """
    Build a map of memory_id -> co-occurrence degree from the DB.
    Uses edges_v3 (provenance edges) for the count.
    """
    db = get_db()
    import psycopg2.extras

    degree = defaultdict(int)
    with db._conn() as conn:
        with conn.cursor() as cur:
            # Count edges per memory from edges_v3 (columns: id1, id2)
            cur.execute(f"""
                SELECT node_id, COUNT(*) as deg
                FROM (
                    SELECT id1 AS node_id FROM {db._table('edges_v3')}
                    UNION ALL
                    SELECT id2 AS node_id FROM {db._table('edges_v3')}
                ) sub
                GROUP BY node_id
            """)
            for row in cur.fetchall():
                degree[row[0]] = row[1]

    return dict(degree)


def _build_adjacency() -> dict[str, set]:
    """Build adjacency list from edges_v3."""
    db = get_db()
    adj = defaultdict(set)
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT id1, id2 FROM {db._table('edges_v3')}
            """)
            for row in cur.fetchall():
                adj[row[0]].add(row[1])
                adj[row[1]].add(row[0])
    return dict(adj)


def _get_all_active_memories() -> list[dict]:
    """Get all active+core memories with metadata."""
    db = get_db()
    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, type, tags, recall_count, sessions_since_recall,
                       emotional_weight, importance, freshness, content,
                       created, last_recalled
                FROM {db._table('memories')}
                WHERE type IN ('core', 'active')
            """)
            return [dict(r) for r in cur.fetchall()]


# --- Scoring Components ---

def score_isolation(memories: list[dict], degree_map: dict) -> dict[str, float]:
    """
    Isolation score: how disconnected is this memory from the graph?
    High score = few edges = needs connections.

    Score = 1.0 - (degree / max_degree), clamped.
    Memories with 0 edges get score 1.0.
    """
    if not degree_map:
        return {m['id']: 1.0 for m in memories}

    max_degree = max(degree_map.values()) if degree_map else 1
    scores = {}

    for mem in memories:
        mid = mem['id']
        degree = degree_map.get(mid, 0)

        if degree == 0:
            scores[mid] = 1.0
        elif degree <= LOW_DEGREE_THRESHOLD:
            # Linear scale from 0.5 to 1.0 for low-degree nodes
            scores[mid] = 0.5 + 0.5 * (1.0 - degree / max(LOW_DEGREE_THRESHOLD, 1))
        elif degree >= HIGH_DEGREE_THRESHOLD:
            # Hubs have no isolation need
            scores[mid] = 0.0
        else:
            # Smooth decay between thresholds
            scores[mid] = max(0.0, 1.0 - (degree / max_degree))

    return scores


def score_bridging(memories: list[dict], adjacency: dict) -> dict[str, float]:
    """
    Bridging score: could this memory connect otherwise-disconnected regions?

    A memory has high bridging potential if its neighbors don't know each other.
    This is related to betweenness centrality but much cheaper to compute.

    For each memory, look at its neighbor set. If neighbors have low overlap
    with each other's neighbors, this memory is a potential bridge.
    """
    scores = {}

    for mem in memories:
        mid = mem['id']
        neighbors = adjacency.get(mid, set())

        if len(neighbors) < 2:
            # Can't bridge with fewer than 2 connections
            # But isolated memories get bridging = 0 (isolation handles them)
            scores[mid] = 0.0
            continue

        # Count how many of my neighbors are NOT connected to each other
        neighbor_list = list(neighbors)
        disconnected_pairs = 0
        total_pairs = 0

        for i, n1 in enumerate(neighbor_list):
            n1_neighbors = adjacency.get(n1, set())
            for n2 in neighbor_list[i + 1:]:
                total_pairs += 1
                if n2 not in n1_neighbors:
                    disconnected_pairs += 1

        if total_pairs == 0:
            scores[mid] = 0.0
        else:
            # Ratio of disconnected neighbor pairs = bridging potential
            scores[mid] = disconnected_pairs / total_pairs

    return scores


def score_domain_gap(memories: list[dict]) -> dict[str, float]:
    """
    Domain gap score: memories in underrepresented domains get higher scores.

    Counts memories per cognitive domain, identifies the least-covered domains,
    and boosts memories that belong to those domains.
    """
    # Count memories per domain
    domain_counts = Counter()
    memory_domains = {}  # memory_id -> set of domains

    for mem in memories:
        tags = set(mem.get('tags') or [])
        mem_doms = set()
        for domain, domain_tags in COGNITIVE_DOMAINS.items():
            if tags & set(domain_tags):
                domain_counts[domain] += 1
                mem_doms.add(domain)
        memory_domains[mem['id']] = mem_doms

    if not domain_counts:
        return {m['id']: 0.5 for m in memories}

    # Find under-represented domains
    max_count = max(domain_counts.values()) if domain_counts else 1
    domain_gaps = {}
    for domain in COGNITIVE_DOMAINS:
        count = domain_counts.get(domain, 0)
        # Gap = how much smaller this domain is vs the largest
        domain_gaps[domain] = 1.0 - (count / max_count) if max_count > 0 else 1.0

    # Score each memory based on its domains' gaps
    scores = {}
    for mem in memories:
        mid = mem['id']
        doms = memory_domains.get(mid, set())
        if not doms:
            # Untagged memories — moderate gap score (they're uncategorized)
            scores[mid] = 0.3
        else:
            # Average gap of this memory's domains
            scores[mid] = sum(domain_gaps.get(d, 0.5) for d in doms) / len(doms)

    return scores


def score_survivor(memories: list[dict]) -> dict[str, float]:
    """
    Survivor score: old memories that survived decay but are rarely recalled.

    These are potential forgotten treasures — they've persisted through many
    sessions but something keeps them from being retrieved. They deserve
    a second look.
    """
    scores = {}

    for mem in memories:
        mid = mem['id']
        sessions_since = mem.get('sessions_since_recall', 0) or 0
        recall_count = mem.get('recall_count', 0) or 0
        importance = mem.get('importance', 0.5) or 0.5

        if sessions_since >= SURVIVOR_MIN_SESSIONS and recall_count <= SURVIVOR_MAX_RECALLS:
            # Old and rarely recalled — high survivor score
            # Scale by age (older = more curious) and importance (important survivors matter more)
            age_factor = min(1.0, sessions_since / (SURVIVOR_MIN_SESSIONS * 3))
            scores[mid] = age_factor * (0.5 + 0.5 * importance)
        elif sessions_since >= SURVIVOR_MIN_SESSIONS // 2 and recall_count <= 1:
            # Moderate survivor — somewhat old and never recalled
            scores[mid] = 0.3
        else:
            scores[mid] = 0.0

    return scores


# --- Composite Scoring ---

def compute_curiosity_scores(limit: int = CURIOSITY_TOP_N) -> list[dict]:
    """
    Compute composite curiosity scores for all active memories.
    Returns top N most curious targets.

    Each target includes:
    - id: memory ID
    - curiosity_score: composite score (0-1)
    - components: breakdown of each score
    - preview: content preview
    - reason: human-readable explanation of why this is curious
    """
    memories = _get_all_active_memories()

    if len(memories) < MIN_MEMORIES_FOR_CURIOSITY:
        return []

    # Build graph structures
    degree_map = _build_degree_map()
    adjacency = _build_adjacency()

    # Compute all component scores
    isolation = score_isolation(memories, degree_map)
    bridging = score_bridging(memories, adjacency)
    domain_gap = score_domain_gap(memories)
    survivor = score_survivor(memories)

    # Composite score
    results = []
    for mem in memories:
        mid = mem['id']
        iso = isolation.get(mid, 0)
        bri = bridging.get(mid, 0)
        dom = domain_gap.get(mid, 0)
        srv = survivor.get(mid, 0)

        composite = (
            ISOLATION_WEIGHT * iso +
            BRIDGING_WEIGHT * bri +
            DOMAIN_GAP_WEIGHT * dom +
            SURVIVOR_WEIGHT * srv
        )

        if composite < 0.1:
            continue  # Not curious enough

        # Determine primary reason
        scores = {'isolation': iso, 'bridging': bri, 'domain_gap': dom, 'survivor': srv}
        primary = max(scores, key=scores.get)
        reasons = {
            'isolation': f'Only {degree_map.get(mid, 0)} edges — disconnected from graph',
            'bridging': f'Could bridge {int(bri * 100)}% disconnected neighbor pairs',
            'domain_gap': f'In underrepresented domain (gap={dom:.2f})',
            'survivor': f'Survived {mem.get("sessions_since_recall", 0)} sessions with {mem.get("recall_count", 0)} recalls',
        }

        results.append({
            'id': mid,
            'curiosity_score': round(composite, 4),
            'components': {
                'isolation': round(iso, 3),
                'bridging': round(bri, 3),
                'domain_gap': round(dom, 3),
                'survivor': round(srv, 3),
            },
            'degree': degree_map.get(mid, 0),
            'preview': (mem.get('content') or '')[:120],
            'reason': reasons[primary],
            'primary_factor': primary,
        })

    results.sort(key=lambda x: x['curiosity_score'], reverse=True)
    return results[:limit]


def get_curiosity_targets(count: int = 5) -> list[dict]:
    """
    Get curiosity targets for priming. Designed to be called by
    get_priming_candidates() as a replacement for random excavation.

    Returns diverse targets — picks from different primary factors
    to ensure broad exploration.
    """
    all_targets = compute_curiosity_scores(limit=count * 4)
    if not all_targets:
        return []

    # Diversify by primary factor
    by_factor = defaultdict(list)
    for t in all_targets:
        by_factor[t['primary_factor']].append(t)

    selected = []
    seen = set()
    factors = list(by_factor.keys())

    # Round-robin through factors
    idx = 0
    while len(selected) < count and idx < len(all_targets):
        factor = factors[idx % len(factors)]
        candidates = by_factor.get(factor, [])
        for c in candidates:
            if c['id'] not in seen:
                selected.append(c)
                seen.add(c['id'])
                break
        idx += 1

    # Fill remaining from top-scored if round-robin didn't fill
    for t in all_targets:
        if len(selected) >= count:
            break
        if t['id'] not in seen:
            selected.append(t)
            seen.add(t['id'])

    return selected[:count]


# --- Graph Sparsity Analysis ---

def analyze_sparsity() -> dict:
    """
    Analyze overall graph sparsity and identify sparse regions.

    Returns:
        - graph_density: edges / possible edges
        - degree_distribution: histogram of node degrees
        - isolated_count: nodes with 0 edges
        - sparse_domains: domains with fewest internal connections
        - sparsity_score: 0-1 composite (higher = sparser = more curiosity needed)
    """
    memories = _get_all_active_memories()
    degree_map = _build_degree_map()

    n = len(memories)
    if n < 2:
        return {'sparsity_score': 1.0, 'note': 'Too few memories for analysis'}

    # Degree distribution
    degrees = [degree_map.get(m['id'], 0) for m in memories]
    isolated = sum(1 for d in degrees if d == 0)
    low_degree = sum(1 for d in degrees if 0 < d <= LOW_DEGREE_THRESHOLD)
    medium_degree = sum(1 for d in degrees if LOW_DEGREE_THRESHOLD < d < HIGH_DEGREE_THRESHOLD)
    high_degree = sum(1 for d in degrees if d >= HIGH_DEGREE_THRESHOLD)

    # Graph density = actual edges / possible edges
    total_edges = sum(degrees) // 2  # Each edge counted twice
    possible_edges = n * (n - 1) // 2
    density = total_edges / possible_edges if possible_edges > 0 else 0

    # Domain internal connectivity
    domain_connectivity = {}
    for domain, domain_tags in COGNITIVE_DOMAINS.items():
        domain_mems = [m['id'] for m in memories if set(m.get('tags') or []) & set(domain_tags)]
        if len(domain_mems) < 2:
            domain_connectivity[domain] = 0
            continue
        internal_edges = 0
        for mid in domain_mems:
            for other in degree_map:
                if other in domain_mems and other != mid:
                    internal_edges += 1
        # Normalize by possible internal edges
        possible = len(domain_mems) * (len(domain_mems) - 1)
        domain_connectivity[domain] = internal_edges / possible if possible > 0 else 0

    # Composite sparsity score
    # Higher = sparser = more curiosity needed
    isolation_ratio = isolated / n if n > 0 else 0
    low_ratio = (isolated + low_degree) / n if n > 0 else 0
    sparsity_score = round(
        0.4 * isolation_ratio +
        0.3 * (1.0 - density) +
        0.3 * low_ratio,
        4
    )

    return {
        'total_memories': n,
        'total_edges': total_edges,
        'graph_density': round(density, 6),
        'degree_distribution': {
            'isolated': isolated,
            'low (1-3)': low_degree,
            'medium (4-19)': medium_degree,
            'high (20+)': high_degree,
        },
        'avg_degree': round(sum(degrees) / n, 2) if n > 0 else 0,
        'median_degree': sorted(degrees)[n // 2] if n > 0 else 0,
        'domain_connectivity': {k: round(v, 4) for k, v in domain_connectivity.items()},
        'sparsity_score': sparsity_score,
    }


# --- Exploration Tracking ---

def log_curiosity_surfaced(target_ids: list[str]):
    """Log which curiosity targets were surfaced for priming."""
    db = get_db()
    log = _load_curiosity_log(db)
    log['surfaced'].append({
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'targets': target_ids,
    })
    # Keep last 50 entries
    log['surfaced'] = log['surfaced'][-50:]
    db.kv_set(KV_CURIOSITY_LOG, log)


def log_curiosity_conversion(target_id: str, new_edges: int):
    """
    Log when a curiosity target successfully created new co-occurrence edges.
    This is the key metric — did directed exploration actually grow the graph?
    """
    db = get_db()
    log = _load_curiosity_log(db)
    log['conversions'].append({
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'target_id': target_id,
        'new_edges': new_edges,
    })
    log['conversions'] = log['conversions'][-100:]
    log['total_conversions'] = log.get('total_conversions', 0) + 1
    log['total_edges_from_curiosity'] = log.get('total_edges_from_curiosity', 0) + new_edges
    db.kv_set(KV_CURIOSITY_LOG, log)


def get_curiosity_stats() -> dict:
    """Get exploration statistics."""
    db = get_db()
    log = _load_curiosity_log(db)

    total_surfaced = sum(len(s.get('targets', [])) for s in log.get('surfaced', []))
    total_conversions = log.get('total_conversions', 0)
    total_edges = log.get('total_edges_from_curiosity', 0)

    conversion_rate = total_conversions / total_surfaced if total_surfaced > 0 else 0

    # Sparsity trend: compare recent sparsity to earlier
    sparsity = analyze_sparsity()

    return {
        'total_surfaced': total_surfaced,
        'total_conversions': total_conversions,
        'conversion_rate': round(conversion_rate, 3),
        'total_new_edges_from_curiosity': total_edges,
        'sessions_with_curiosity': len(log.get('surfaced', [])),
        'sparsity_score': sparsity.get('sparsity_score', 0),
        'graph_density': sparsity.get('graph_density', 0),
        'isolated_memories': sparsity.get('degree_distribution', {}).get('isolated', 0),
    }


def _load_curiosity_log(db=None) -> dict:
    """Load curiosity log from DB KV store."""
    if db is None:
        db = get_db()
    raw = db.kv_get(KV_CURIOSITY_LOG)
    if raw:
        return json.loads(raw) if isinstance(raw, str) else raw
    return {
        'surfaced': [],
        'conversions': [],
        'total_conversions': 0,
        'total_edges_from_curiosity': 0,
    }


# --- CLI ---

def main():
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    args = sys.argv[1:]
    if not args or args[0] in ('--help', '-h'):
        print(__doc__.strip())
        return

    cmd = args[0]

    if cmd == 'scan':
        limit = int(args[1]) if len(args) > 1 else CURIOSITY_TOP_N
        print(f"\n=== Curiosity Engine — Top {limit} Targets ===\n")
        targets = compute_curiosity_scores(limit=limit)
        if not targets:
            print("Not enough memories for curiosity analysis.")
            return
        for i, t in enumerate(targets, 1):
            c = t['components']
            print(f"{i}. [{t['curiosity_score']:.3f}] {t['id']} (degree={t['degree']})")
            print(f"   {t['reason']}")
            print(f"   iso={c['isolation']:.2f} bri={c['bridging']:.2f} dom={c['domain_gap']:.2f} srv={c['survivor']:.2f}")
            print(f"   {t['preview'][:80]}...")
            print()

    elif cmd == 'map':
        print("\n=== Graph Sparsity Map ===\n")
        sparsity = analyze_sparsity()
        print(f"Total memories: {sparsity['total_memories']}")
        print(f"Total edges: {sparsity['total_edges']}")
        print(f"Graph density: {sparsity['graph_density']:.6f}")
        print(f"Avg degree: {sparsity['avg_degree']}")
        print(f"Median degree: {sparsity['median_degree']}")
        print()
        print("Degree distribution:")
        for bucket, count in sparsity['degree_distribution'].items():
            bar = '#' * min(count, 50)
            print(f"  {bucket:>15s}: {count:4d} {bar}")
        print()
        print("Domain connectivity:")
        for domain, conn in sorted(sparsity['domain_connectivity'].items(), key=lambda x: x[1]):
            bar = '#' * int(conn * 40)
            print(f"  {domain:>12s}: {conn:.4f} {bar}")
        print()
        print(f"Sparsity score: {sparsity['sparsity_score']:.4f} (higher = needs more exploration)")

    elif cmd == 'stats':
        print("\n=== Curiosity Engine Statistics ===\n")
        stats = get_curiosity_stats()
        print(f"Targets surfaced: {stats['total_surfaced']}")
        print(f"Conversions: {stats['total_conversions']}")
        print(f"Conversion rate: {stats['conversion_rate']:.1%}")
        print(f"New edges from curiosity: {stats['total_new_edges_from_curiosity']}")
        print(f"Sessions with curiosity: {stats['sessions_with_curiosity']}")
        print()
        print(f"Current graph:")
        print(f"  Sparsity score: {stats['sparsity_score']:.4f}")
        print(f"  Graph density: {stats['graph_density']:.6f}")
        print(f"  Isolated memories: {stats['isolated_memories']}")

    elif cmd == 'targets':
        count = int(args[1]) if len(args) > 1 else 5
        targets = get_curiosity_targets(count=count)
        if not targets:
            print("No curiosity targets available.")
            return
        print(f"\n=== {len(targets)} Curiosity Targets (diversified) ===\n")
        for t in targets:
            print(f"[{t['curiosity_score']:.3f}] {t['id']} — {t['primary_factor']}")
            print(f"  {t['reason']}")
            print(f"  {t['preview'][:80]}...")
            print()

    elif cmd == 'log-conversion':
        if len(args) < 3:
            print("Usage: curiosity_engine.py log-conversion <memory_id> <new_edges>")
            return
        target_id = args[1]
        new_edges = int(args[2])
        log_curiosity_conversion(target_id, new_edges)
        print(f"Logged conversion: {target_id} -> {new_edges} new edges")

    else:
        print(f"Unknown command: {cmd}")
        print("Available: scan, map, stats, targets, log-conversion")
        sys.exit(1)


if __name__ == '__main__':
    main()
