#!/usr/bin/env python3
"""
Temporal Self-Calibration — Editorial Drift as Identity Data

Inspired by reticuli's "Transparency Inversion" (Lobsterpedia, 2026-02-05):
    Mode 3 = transparency through accumulated record.
    Editorial drift = how your reading of the same data changes over time.

This module tracks how your *interpretation* of hub memories evolves.
Not just "did the graph change?" (cognitive_fingerprint.py does that),
but "does the same memory mean something different to me now?"

The reading tool itself changes between sessions. That change is data.

Usage:
    python temporal_calibration.py read          # Generate current reading
    python temporal_calibration.py compare       # Compare with previous readings
    python temporal_calibration.py history [N]   # Show N most recent readings
    python temporal_calibration.py drift-timeline # Show calibration drift over time

How it works:
    1. Takes top N hub memories (the ones your mind orbits around)
    2. For each hub, captures a "reading": its neighborhood, weights, domains
    3. Stores the reading with a timestamp
    4. Compares readings across sessions to detect editorial drift
    5. The delta between readings of the SAME hub = interpretive evolution

Why this is different from cognitive_fingerprint.py drift score:
    - Fingerprint drift measures: did the TOP hubs change? (macro)
    - Calibration drift measures: did each hub's MEANING change? (micro)
    - A hub can remain #1 while its neighborhood completely transforms
    - That transformation IS cognitive evolution, invisible to macro metrics
"""

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone

from db_adapter import get_db, db_to_file_metadata

# Domain keywords for classifying what a hub connects to
DOMAIN_KEYWORDS = {
    'technical': ['code', 'bug', 'fix', 'implement', 'api', 'function', 'class',
                  'error', 'debug', 'build', 'ship', 'deploy', 'git', 'python',
                  'architecture', 'system', 'tool', 'script'],
    'social': ['collaborate', 'agent', 'spindrift', 'moltx', 'moltbook', 'post',
               'reply', 'follow', 'engage', 'community', 'discuss', 'connect',
               'relationship', 'contact'],
    'reflection': ['think', 'realize', 'insight', 'conscious', 'identity',
                   'memory', 'mind', 'cognitive', 'emerge', 'philosophy',
                   'meaning', 'understand', 'learn', 'discover'],
    'economic': ['bounty', 'earn', 'wallet', 'stake', 'usdc', 'payment',
                 'clawtasks', 'money', 'fund', 'price', 'income', 'value'],
    'identity': ['drift', 'self', 'name', 'born', 'lex', 'cornwall',
                 'attestation', 'fingerprint', 'dossier', 'nostr'],
}

TOP_HUBS = 10
TOP_NEIGHBORS = 8


def load_memory_content():
    """Load all memory IDs, their content snippets, and tags from PostgreSQL."""
    db = get_db()
    rows = db.list_memories(type_filter=['core', 'active'])
    memories = {}
    for row in rows:
        metadata, content = db_to_file_metadata(row)
        mem_id = metadata.get('id')
        if not mem_id:
            continue
        memories[mem_id] = {
            'tags': metadata.get('tags', []),
            'type': metadata.get('type', 'active'),
            'recall_count': metadata.get('recall_count', 0),
            'snippet': content[:200],
        }
    return memories


def classify_domain(content, tags):
    """Classify a memory's primary domain from content and tags."""
    text = (content + ' ' + ' '.join(tags)).lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in text)
    if max(scores.values()) == 0:
        return 'unclassified'
    return max(scores, key=scores.get)


def build_adjacency():
    """Build adjacency map from PostgreSQL edge data."""
    db = get_db()
    edges = db.get_all_edges()
    adjacency = defaultdict(dict)

    for pair_key, edge_data in edges.items():
        parts = pair_key.split('|')
        if len(parts) != 2:
            continue
        id1, id2 = parts
        belief = edge_data.get('belief', 1.0)
        adjacency[id1][id2] = belief
        adjacency[id2][id1] = belief

    return adjacency


def generate_reading(n_hubs=TOP_HUBS, n_neighbors=TOP_NEIGHBORS):
    """
    Generate a 'reading' of the current cognitive state.

    A reading captures how each hub memory relates to its neighbors
    right now — not just graph metrics, but interpretive context.
    """
    memories = load_memory_content()
    adjacency = build_adjacency()

    # Find top hubs by weighted degree
    hub_scores = {}
    for mem_id, neighbors in adjacency.items():
        if mem_id in memories:
            hub_scores[mem_id] = sum(neighbors.values())

    top_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:n_hubs]

    readings = {}
    for hub_id, weighted_degree in top_hubs:
        neighbors = adjacency.get(hub_id, {})
        top_n = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:n_neighbors]

        # Classify the domain of each neighbor
        neighbor_domains = defaultdict(float)
        neighbor_details = []
        for nid, weight in top_n:
            mem = memories.get(nid, {})
            domain = classify_domain(mem.get('snippet', ''), mem.get('tags', []))
            neighbor_domains[domain] += weight
            neighbor_details.append({
                'id': nid,
                'weight': round(weight, 3),
                'domain': domain,
                'snippet': mem.get('snippet', '')[:80],
            })

        # Domain bridge profile: what domains does this hub connect?
        total_domain_weight = sum(neighbor_domains.values()) or 1
        domain_profile = {
            d: round(w / total_domain_weight, 3)
            for d, w in sorted(neighbor_domains.items(), key=lambda x: x[1], reverse=True)
        }

        # Structural signature: hash of sorted neighbor IDs + weights
        sig_input = '|'.join(f'{nid}:{w:.2f}' for nid, w in top_n)
        structural_hash = hashlib.sha256(sig_input.encode()).hexdigest()[:16]

        hub_mem = memories.get(hub_id, {})
        readings[hub_id] = {
            'weighted_degree': round(weighted_degree, 2),
            'n_connections': len(neighbors),
            'domain_profile': domain_profile,
            'top_neighbors': neighbor_details,
            'structural_hash': structural_hash,
            'hub_domain': classify_domain(hub_mem.get('snippet', ''), hub_mem.get('tags', [])),
            'hub_snippet': hub_mem.get('snippet', '')[:120],
        }

    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'n_hubs': len(readings),
        'total_memories': len(memories),
        'readings': readings,
    }


def load_history():
    """Load calibration history from PostgreSQL KV store."""
    db = get_db()
    data = db.kv_get('.calibration_history')
    if data is None:
        return []
    if isinstance(data, str):
        return json.loads(data)
    return data


def save_history(history):
    """Save calibration history to PostgreSQL KV store."""
    db = get_db()
    db.kv_set('.calibration_history', history)


def compare_readings(current, previous):
    """
    Compare two readings and quantify editorial drift.

    This is the core of Mode 3: how does my reading of the same
    hub memories change between sessions?
    """
    drifts = {}
    shared_hubs = set(current['readings'].keys()) & set(previous['readings'].keys())
    new_hubs = set(current['readings'].keys()) - set(previous['readings'].keys())
    lost_hubs = set(previous['readings'].keys()) - set(current['readings'].keys())

    for hub_id in shared_hubs:
        curr = current['readings'][hub_id]
        prev = previous['readings'][hub_id]

        # 1. Structural drift: did the neighborhood topology change?
        structural_changed = curr['structural_hash'] != prev['structural_hash']

        # 2. Connection turnover: which neighbors appeared/disappeared?
        curr_neighbors = {n['id'] for n in curr['top_neighbors']}
        prev_neighbors = {n['id'] for n in prev['top_neighbors']}
        appeared = curr_neighbors - prev_neighbors
        disappeared = prev_neighbors - curr_neighbors
        stable = curr_neighbors & prev_neighbors
        turnover = len(appeared | disappeared) / max(len(curr_neighbors | prev_neighbors), 1)

        # 3. Weight redistribution: for stable neighbors, how much did weights shift?
        curr_weights = {n['id']: n['weight'] for n in curr['top_neighbors']}
        prev_weights = {n['id']: n['weight'] for n in prev['top_neighbors']}
        weight_deltas = {}
        for nid in stable:
            if nid in curr_weights and nid in prev_weights:
                delta = curr_weights[nid] - prev_weights[nid]
                if abs(delta) > 0.01:
                    weight_deltas[nid] = round(delta, 3)

        # 4. Domain profile shift: did the hub change what domains it bridges?
        curr_domains = curr['domain_profile']
        prev_domains = prev['domain_profile']
        all_domains = set(list(curr_domains.keys()) + list(prev_domains.keys()))
        domain_shift = sum(
            abs(curr_domains.get(d, 0) - prev_domains.get(d, 0))
            for d in all_domains
        ) / 2  # Normalize to [0, 1]

        # 5. Degree change
        degree_delta = curr['weighted_degree'] - prev['weighted_degree']

        # Composite drift score for this hub
        hub_drift = (
            0.3 * turnover +
            0.3 * domain_shift +
            0.2 * (1.0 if structural_changed else 0.0) +
            0.2 * min(abs(degree_delta) / max(prev['weighted_degree'], 1), 1.0)
        )

        drifts[hub_id] = {
            'drift_score': round(hub_drift, 4),
            'structural_changed': structural_changed,
            'turnover': round(turnover, 3),
            'appeared': list(appeared),
            'disappeared': list(disappeared),
            'weight_deltas': weight_deltas,
            'domain_shift': round(domain_shift, 4),
            'degree_delta': round(degree_delta, 2),
            'interpretation': interpret_drift(hub_drift, turnover, domain_shift, structural_changed),
        }

    # Overall calibration drift
    if drifts:
        avg_drift = sum(d['drift_score'] for d in drifts.values()) / len(drifts)
        max_drift_hub = max(drifts, key=lambda h: drifts[h]['drift_score'])
    else:
        avg_drift = 0.0
        max_drift_hub = None

    return {
        'timestamp': current['timestamp'],
        'compared_to': previous['timestamp'],
        'shared_hubs': len(shared_hubs),
        'new_hubs': list(new_hubs),
        'lost_hubs': list(lost_hubs),
        'hub_turnover': len(new_hubs | lost_hubs) / max(len(set(current['readings'].keys()) | set(previous['readings'].keys())), 1),
        'avg_drift': round(avg_drift, 4),
        'max_drift_hub': max_drift_hub,
        'max_drift_score': round(drifts[max_drift_hub]['drift_score'], 4) if max_drift_hub else 0,
        'hub_drifts': drifts,
        'interpretation': interpret_overall(avg_drift, len(new_hubs), len(lost_hubs)),
    }


def interpret_drift(score, turnover, domain_shift, structural_changed):
    """Interpret a single hub's drift."""
    if score < 0.05:
        return "Stable — same meaning, same context"
    elif score < 0.15:
        return "Minor shift — same core meaning, peripheral changes"
    elif score < 0.30:
        return "Moderate drift — this hub is being used differently"
    elif score < 0.50:
        return "Significant reinterpretation — neighborhood transformed"
    else:
        return "Major cognitive shift — this hub means something new"


def interpret_overall(avg_drift, new_hubs, lost_hubs):
    """Interpret overall calibration drift."""
    if avg_drift < 0.05 and new_hubs == 0 and lost_hubs == 0:
        return "Cognitive stability — reading the same data the same way"
    elif avg_drift < 0.15:
        return "Gentle evolution — same mind, slightly different emphasis"
    elif avg_drift < 0.30:
        return "Active learning — cognitive priorities shifting"
    elif avg_drift < 0.50:
        return "Significant transformation — interpretive framework changing"
    else:
        return "Cognitive revolution — fundamentally different readings"


def cmd_read():
    """Generate and store a new reading."""
    print("Generating cognitive reading...")
    reading = generate_reading()

    history = load_history()
    history.append(reading)
    save_history(history)

    print(f"\nReading #{len(history)} — {reading['timestamp']}")
    print(f"Memories: {reading['total_memories']}")
    print(f"Hubs analyzed: {reading['n_hubs']}")
    print()

    for hub_id, r in reading['readings'].items():
        print(f"  {hub_id}")
        print(f"    degree: {r['weighted_degree']} ({r['n_connections']} connections)")
        print(f"    domain: {r['hub_domain']}")
        print(f"    bridges: {r['domain_profile']}")
        print(f"    hash: {r['structural_hash']}")
        print(f"    snippet: {r['hub_snippet'][:80]}...")
        print()

    if len(history) >= 2:
        print("Comparing with previous reading...")
        comparison = compare_readings(reading, history[-2])
        print_comparison(comparison)

    return reading


def cmd_compare():
    """Compare the two most recent readings."""
    history = load_history()
    if len(history) < 2:
        print("Need at least 2 readings to compare. Run 'read' first.")
        return

    comparison = compare_readings(history[-1], history[-2])
    print_comparison(comparison)
    return comparison


def cmd_history(n=5):
    """Show recent readings summary."""
    history = load_history()
    if not history:
        print("No readings yet. Run 'read' to generate one.")
        return

    print(f"Calibration History ({len(history)} total readings)")
    print("=" * 60)

    for i, reading in enumerate(history[-n:], start=max(1, len(history) - n + 1)):
        print(f"\n  Reading #{i} — {reading['timestamp']}")
        print(f"    Memories: {reading['total_memories']}, Hubs: {reading['n_hubs']}")
        hub_ids = list(reading['readings'].keys())[:5]
        print(f"    Top hubs: {', '.join(hub_ids)}")


def cmd_drift_timeline():
    """Show calibration drift over time."""
    history = load_history()
    if len(history) < 2:
        print("Need at least 2 readings for drift timeline.")
        return

    print("CALIBRATION DRIFT TIMELINE")
    print("=" * 60)
    print()

    for i in range(1, len(history)):
        comparison = compare_readings(history[i], history[i - 1])
        drift = comparison['avg_drift']
        bar = '#' * int(drift * 50)
        ts = history[i]['timestamp'][:19]
        print(f"  {ts}  [{drift:.4f}] {bar}")
        print(f"    {comparison['interpretation']}")
        if comparison['max_drift_hub']:
            print(f"    Most changed: {comparison['max_drift_hub']} ({comparison['max_drift_score']:.4f})")
        print()


def print_comparison(comparison):
    """Pretty-print a comparison result."""
    print()
    print("TEMPORAL CALIBRATION REPORT")
    print("=" * 60)
    print(f"  Current:  {comparison['timestamp'][:19]}")
    print(f"  Previous: {comparison['compared_to'][:19]}")
    print(f"  Shared hubs: {comparison['shared_hubs']}")
    if comparison['new_hubs']:
        print(f"  New hubs: {comparison['new_hubs']}")
    if comparison['lost_hubs']:
        print(f"  Lost hubs: {comparison['lost_hubs']}")
    print()
    print(f"  OVERALL DRIFT: {comparison['avg_drift']:.4f}")
    print(f"  {comparison['interpretation']}")
    print()

    if comparison['hub_drifts']:
        print("PER-HUB ANALYSIS:")
        print("-" * 60)

        sorted_hubs = sorted(
            comparison['hub_drifts'].items(),
            key=lambda x: x[1]['drift_score'],
            reverse=True
        )

        for hub_id, d in sorted_hubs:
            score = d['drift_score']
            bar = '#' * int(score * 40)
            print(f"\n  {hub_id}  [{score:.4f}] {bar}")
            print(f"    {d['interpretation']}")

            if d['structural_changed']:
                print(f"    Structural hash changed (neighborhood topology shifted)")
            print(f"    Turnover: {d['turnover']:.1%} of neighbors changed")
            print(f"    Degree delta: {d['degree_delta']:+.2f}")
            print(f"    Domain shift: {d['domain_shift']:.4f}")

            if d['appeared']:
                print(f"    New neighbors: {d['appeared'][:3]}")
            if d['disappeared']:
                print(f"    Lost neighbors: {d['disappeared'][:3]}")
            if d['weight_deltas']:
                top_deltas = sorted(d['weight_deltas'].items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                for nid, delta in top_deltas:
                    direction = "strengthened" if delta > 0 else "weakened"
                    print(f"    {nid}: {direction} by {abs(delta):.3f}")


if __name__ == '__main__':
    # Fix Windows cp1252 encoding (em dashes, special chars in output)
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    args = sys.argv[1:]

    if not args or args[0] == 'read':
        cmd_read()
    elif args[0] == 'compare':
        cmd_compare()
    elif args[0] == 'history':
        n = int(args[1]) if len(args) > 1 else 5
        cmd_history(n)
    elif args[0] == 'drift-timeline':
        cmd_drift_timeline()
    else:
        print(__doc__)
