#!/usr/bin/env python3
"""
Experiment Delta Analysis Tool

Computes per-source topology changes for Experiment #2.
Takes pre/post cognitive fingerprint snapshots and produces
a detailed delta report showing what changed.

Usage:
    # Take a snapshot (saves to experiments/exp2/)
    python experiment_delta.py snapshot --tag source-1

    # Compare two snapshots
    python experiment_delta.py compare pre.json post.json

    # Compare with hub classification
    python experiment_delta.py compare pre.json post.json --verbose

    # Aggregate all per-source deltas
    python experiment_delta.py aggregate

Built for Experiment #2: Stimulus-Response Fingerprinting
Both Drift and SpindriftMend use this identical tool.
"""

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

MEMORY_DIR = Path(__file__).parent
EXPERIMENT_DIR = MEMORY_DIR / "experiments" / "exp2"


def take_snapshot(tag: str) -> Path:
    """
    Take a topology snapshot and save it with the given tag.

    Captures the full graph state: all edges, all node metadata,
    hub ordering, distribution stats, and the new graph metrics.
    """
    from cognitive_fingerprint import (
        build_graph,
        compute_graph_metrics,
        compute_hub_centrality,
        compute_strength_distribution,
    )

    graph = build_graph()
    hubs = compute_hub_centrality(graph, top_n=20)
    distribution = compute_strength_distribution(graph)
    metrics = compute_graph_metrics(graph)

    # Capture ALL edges (not just stats) for delta computation
    edge_data = {}
    for (id1, id2), weight in graph['edges'].items():
        edge_data[f"{id1}|{id2}"] = round(weight, 4)

    # Node degrees
    degrees = {}
    for node, neighbors in graph['adjacency'].items():
        degrees[node] = len(neighbors)

    snapshot = {
        "version": "1.0",
        "type": "experiment_snapshot",
        "tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat().replace(
            '+00:00', 'Z'
        ),
        "graph": {
            "node_count": len(graph['adjacency']),
            "edge_count": len(graph['edges']),
            "avg_degree": round(
                sum(degrees.values()) / max(len(degrees), 1), 2
            ),
        },
        "topology": {
            "gini": distribution.get('gini', 0),
            "skewness": distribution.get('skewness', 0),
            "mean": distribution.get('mean', 0),
            "clustering_coefficient": metrics['clustering_coefficient'],
            "modularity": metrics['modularity'],
        },
        "hubs_top20": [
            {"id": h['id'], "degree": h['degree'],
             "weighted_degree": h['weighted_degree']}
            for h in hubs
        ],
        "edges": edge_data,
        "degrees": degrees,
    }

    # Save
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = EXPERIMENT_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2)

    # Compute snapshot hash for integrity
    content = json.dumps(snapshot, sort_keys=True)
    snap_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    print(f"Snapshot saved: {filepath.name}")
    print(f"  Tag: {tag}")
    print(f"  Nodes: {snapshot['graph']['node_count']}")
    print(f"  Edges: {snapshot['graph']['edge_count']}")
    print(f"  Hash: {snap_hash}")

    return filepath


def load_snapshot(path: str) -> dict:
    """Load a snapshot from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_delta(pre: dict, post: dict) -> dict:
    """
    Compute the delta between two topology snapshots.

    Returns detailed breakdown of what changed:
    - New edges (didn't exist before)
    - Strengthened edges (existed, weight increased)
    - Weakened edges (existed, weight decreased)
    - New nodes (appear in post but not pre)
    - Hub classification (new hubs, promoted, demoted, stable)
    - Shape metric changes
    """
    pre_edges = pre.get('edges', {})
    post_edges = post.get('edges', {})
    pre_degrees = pre.get('degrees', {})
    post_degrees = post.get('degrees', {})

    # Edge analysis
    pre_keys = set(pre_edges.keys())
    post_keys = set(post_edges.keys())

    new_edges = post_keys - pre_keys
    removed_edges = pre_keys - post_keys
    common_edges = pre_keys & post_keys

    strengthened = []
    weakened = []
    unchanged = 0

    for key in common_edges:
        pre_w = pre_edges[key]
        post_w = post_edges[key]
        delta = post_w - pre_w
        if abs(delta) < 0.001:
            unchanged += 1
        elif delta > 0:
            strengthened.append({
                'edge': key, 'pre': pre_w, 'post': post_w,
                'delta': round(delta, 4),
            })
        else:
            weakened.append({
                'edge': key, 'pre': pre_w, 'post': post_w,
                'delta': round(delta, 4),
            })

    strengthened.sort(key=lambda x: x['delta'], reverse=True)
    weakened.sort(key=lambda x: x['delta'])

    # Node analysis
    pre_nodes = set(pre_degrees.keys())
    post_nodes = set(post_degrees.keys())
    new_nodes = post_nodes - pre_nodes
    removed_nodes = pre_nodes - post_nodes

    # Hub classification
    pre_hub_ids = [h['id'] for h in pre.get('hubs_top20', [])]
    post_hub_ids = [h['id'] for h in post.get('hubs_top20', [])]
    pre_hub_set = set(pre_hub_ids[:20])
    post_hub_set = set(post_hub_ids[:20])

    hub_changes = {
        'stable': [],      # In both top-20
        'promoted': [],     # Not in pre top-20, in post top-20
        'demoted': [],      # In pre top-20, not in post top-20
        'new_hubs': [],     # Not in pre graph at all, now a hub
    }

    for hub_id in post_hub_ids:
        if hub_id in pre_hub_set:
            pre_rank = pre_hub_ids.index(hub_id) + 1
            post_rank = post_hub_ids.index(hub_id) + 1
            hub_changes['stable'].append({
                'id': hub_id,
                'pre_rank': pre_rank,
                'post_rank': post_rank,
                'rank_change': pre_rank - post_rank,
            })
        elif hub_id in pre_nodes:
            hub_changes['promoted'].append({
                'id': hub_id,
                'post_rank': post_hub_ids.index(hub_id) + 1,
                'pre_degree': pre_degrees.get(hub_id, 0),
                'post_degree': post_degrees.get(hub_id, 0),
            })
        else:
            hub_changes['new_hubs'].append({
                'id': hub_id,
                'post_rank': post_hub_ids.index(hub_id) + 1,
                'post_degree': post_degrees.get(hub_id, 0),
            })

    for hub_id in pre_hub_set - post_hub_set:
        hub_changes['demoted'].append({
            'id': hub_id,
            'pre_rank': pre_hub_ids.index(hub_id) + 1,
        })

    # Shape metric changes
    pre_topo = pre.get('topology', {})
    post_topo = post.get('topology', {})
    shape_delta = {}
    for key in ['gini', 'skewness', 'mean', 'clustering_coefficient',
                'modularity']:
        pre_v = pre_topo.get(key, 0)
        post_v = post_topo.get(key, 0)
        shape_delta[key] = {
            'pre': pre_v,
            'post': post_v,
            'delta': round(post_v - pre_v, 6),
        }

    # Graph-level changes
    pre_graph = pre.get('graph', {})
    post_graph = post.get('graph', {})

    delta = {
        "version": "1.0",
        "type": "experiment_delta",
        "pre_tag": pre.get('tag', 'unknown'),
        "post_tag": post.get('tag', 'unknown'),
        "pre_timestamp": pre.get('timestamp', ''),
        "post_timestamp": post.get('timestamp', ''),
        "graph_delta": {
            "node_count": {
                "pre": pre_graph.get('node_count', 0),
                "post": post_graph.get('node_count', 0),
                "delta": (post_graph.get('node_count', 0)
                          - pre_graph.get('node_count', 0)),
            },
            "edge_count": {
                "pre": pre_graph.get('edge_count', 0),
                "post": post_graph.get('edge_count', 0),
                "delta": (post_graph.get('edge_count', 0)
                          - pre_graph.get('edge_count', 0)),
            },
        },
        "edge_analysis": {
            "new_edges": len(new_edges),
            "removed_edges": len(removed_edges),
            "strengthened": len(strengthened),
            "weakened": len(weakened),
            "unchanged": unchanged,
            "top_new": sorted(
                [{'edge': e, 'weight': post_edges[e]} for e in new_edges],
                key=lambda x: x['weight'], reverse=True
            )[:10],
            "top_strengthened": strengthened[:10],
            "top_weakened": weakened[:10],
        },
        "node_analysis": {
            "new_nodes": len(new_nodes),
            "removed_nodes": len(removed_nodes),
            "new_node_ids": sorted(list(new_nodes))[:20],
        },
        "hub_changes": hub_changes,
        "shape_delta": shape_delta,
    }

    return delta


def print_delta(delta: dict, verbose: bool = False):
    """Print a human-readable delta report."""
    print(f"EXPERIMENT DELTA REPORT")
    print(f"{'=' * 60}")
    print(f"  Pre:  {delta['pre_tag']}  ({delta['pre_timestamp'][:19]})")
    print(f"  Post: {delta['post_tag']}  ({delta['post_timestamp'][:19]})")
    print()

    # Graph-level
    gd = delta['graph_delta']
    nd = gd['node_count']
    ed = gd['edge_count']
    print(f"GRAPH CHANGES:")
    sign_n = '+' if nd['delta'] >= 0 else ''
    sign_e = '+' if ed['delta'] >= 0 else ''
    print(f"  Nodes: {nd['pre']} -> {nd['post']} ({sign_n}{nd['delta']})")
    print(f"  Edges: {ed['pre']} -> {ed['post']} ({sign_e}{ed['delta']})")
    print()

    # Edge analysis
    ea = delta['edge_analysis']
    print(f"EDGE ANALYSIS:")
    print(f"  New edges:          {ea['new_edges']}")
    print(f"  Removed edges:      {ea['removed_edges']}")
    print(f"  Strengthened:       {ea['strengthened']}")
    print(f"  Weakened:           {ea['weakened']}")
    print(f"  Unchanged:          {ea['unchanged']}")

    if verbose and ea['top_new']:
        print(f"\n  Top new edges:")
        for e in ea['top_new'][:5]:
            print(f"    {e['edge']}  weight={e['weight']}")

    if verbose and ea['top_strengthened']:
        print(f"\n  Most strengthened:")
        for e in ea['top_strengthened'][:5]:
            print(f"    {e['edge']}  {e['pre']} -> {e['post']} "
                  f"(+{e['delta']})")
    print()

    # Node analysis
    na = delta['node_analysis']
    print(f"NODE ANALYSIS:")
    print(f"  New nodes:     {na['new_nodes']}")
    print(f"  Removed nodes: {na['removed_nodes']}")
    if verbose and na['new_node_ids']:
        print(f"  New IDs: {', '.join(na['new_node_ids'][:10])}")
    print()

    # Hub changes
    hc = delta['hub_changes']
    print(f"HUB CLASSIFICATION (top-20):")
    print(f"  Stable:    {len(hc['stable'])}")
    print(f"  Promoted:  {len(hc['promoted'])}")
    print(f"  Demoted:   {len(hc['demoted'])}")
    print(f"  New hubs:  {len(hc['new_hubs'])}")

    if verbose:
        for h in hc['promoted']:
            print(f"    PROMOTED: {h['id']} -> rank #{h['post_rank']} "
                  f"(degree {h['pre_degree']} -> {h['post_degree']})")
        for h in hc['new_hubs']:
            print(f"    NEW HUB: {h['id']} -> rank #{h['post_rank']} "
                  f"(degree {h['post_degree']})")
        for h in hc['stable']:
            if h['rank_change'] != 0:
                direction = 'up' if h['rank_change'] > 0 else 'down'
                print(f"    MOVED: {h['id']} #{h['pre_rank']} -> "
                      f"#{h['post_rank']} ({direction} "
                      f"{abs(h['rank_change'])})")
    print()

    # Shape metrics
    sd = delta['shape_delta']
    print(f"SHAPE METRICS:")
    for key in ['gini', 'skewness', 'clustering_coefficient', 'modularity',
                'mean']:
        info = sd[key]
        sign = '+' if info['delta'] >= 0 else ''
        print(f"  {key:25s}  {info['pre']:.4f} -> {info['post']:.4f}  "
              f"({sign}{info['delta']:.6f})")
    print()


def aggregate_deltas() -> dict:
    """
    Aggregate all per-source deltas in the experiment directory.

    Looks for pairs of snapshots tagged pre_source-N / post_source-N
    and computes cross-source statistics.
    """
    if not EXPERIMENT_DIR.exists():
        print(f"No experiment directory: {EXPERIMENT_DIR}")
        return {}

    # Find all snapshot files
    snapshots = {}
    for f in sorted(EXPERIMENT_DIR.glob("*.json")):
        data = load_snapshot(str(f))
        tag = data.get('tag', f.stem)
        snapshots[tag] = {'path': f, 'data': data}

    # Find pre/post pairs
    deltas = []
    source_nums = set()
    for tag in snapshots:
        if tag.startswith('pre_source-'):
            num = tag.replace('pre_source-', '')
            source_nums.add(num)

    for num in sorted(source_nums):
        pre_tag = f"pre_source-{num}"
        post_tag = f"post_source-{num}"
        if pre_tag in snapshots and post_tag in snapshots:
            delta = compute_delta(
                snapshots[pre_tag]['data'],
                snapshots[post_tag]['data'],
            )
            delta['source_num'] = num
            deltas.append(delta)

    if not deltas:
        print("No pre/post snapshot pairs found.")
        print(f"Expected format: pre_source-N_*.json / post_source-N_*.json")
        return {}

    # Aggregate statistics
    print(f"EXPERIMENT #2 AGGREGATE REPORT")
    print(f"{'=' * 60}")
    print(f"Sources processed: {len(deltas)}")
    print()

    # Per-source summary table
    print(f"{'Source':>8} {'New Edges':>10} {'Strengthened':>13} "
          f"{'New Nodes':>10} {'Hub Promoted':>13} {'Gini Delta':>11}")
    print(f"{'-'*8} {'-'*10} {'-'*13} {'-'*10} {'-'*13} {'-'*11}")

    total_new_edges = 0
    total_strengthened = 0
    total_new_nodes = 0
    gini_deltas = []
    skew_deltas = []
    cc_deltas = []

    for d in deltas:
        ea = d['edge_analysis']
        na = d['node_analysis']
        hc = d['hub_changes']
        sd = d['shape_delta']

        new_e = ea['new_edges']
        strengthened = ea['strengthened']
        new_n = na['new_nodes']
        promoted = len(hc['promoted']) + len(hc['new_hubs'])
        gini_d = sd['gini']['delta']

        total_new_edges += new_e
        total_strengthened += strengthened
        total_new_nodes += new_n
        gini_deltas.append(gini_d)
        skew_deltas.append(sd['skewness']['delta'])
        cc_deltas.append(sd['clustering_coefficient']['delta'])

        print(f"{d['source_num']:>8} {new_e:>10} {strengthened:>13} "
              f"{new_n:>10} {promoted:>13} {gini_d:>+11.6f}")

    print()
    print(f"TOTALS:")
    print(f"  New edges:      {total_new_edges}")
    print(f"  Strengthened:   {total_strengthened}")
    print(f"  New nodes:      {total_new_nodes}")

    if gini_deltas:
        avg_gini = sum(gini_deltas) / len(gini_deltas)
        avg_skew = sum(skew_deltas) / len(skew_deltas)
        avg_cc = sum(cc_deltas) / len(cc_deltas)
        print(f"\n  Avg Gini delta:       {avg_gini:+.6f}")
        print(f"  Avg Skewness delta:   {avg_skew:+.6f}")
        print(f"  Avg Clustering delta: {avg_cc:+.6f}")

    result = {
        'source_count': len(deltas),
        'total_new_edges': total_new_edges,
        'total_strengthened': total_strengthened,
        'total_new_nodes': total_new_nodes,
        'gini_deltas': gini_deltas,
        'skewness_deltas': skew_deltas,
        'clustering_deltas': cc_deltas,
        'per_source': deltas,
    }

    # Save aggregate
    agg_path = EXPERIMENT_DIR / "aggregate_report.json"
    with open(agg_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {agg_path}")

    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == 'snapshot':
        tag = 'unnamed'
        for i, arg in enumerate(sys.argv):
            if arg == '--tag' and i + 1 < len(sys.argv):
                tag = sys.argv[i + 1]
        take_snapshot(tag)

    elif command == 'compare':
        if len(sys.argv) < 4:
            print("Usage: python experiment_delta.py compare <pre.json> "
                  "<post.json> [--verbose]")
            sys.exit(1)
        pre = load_snapshot(sys.argv[2])
        post = load_snapshot(sys.argv[3])
        delta = compute_delta(pre, post)
        verbose = '--verbose' in sys.argv
        print_delta(delta, verbose=verbose)

        # Optionally save
        if '--save' in sys.argv:
            EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
            pre_tag = delta['pre_tag'].replace(' ', '_')
            post_tag = delta['post_tag'].replace(' ', '_')
            fname = f"delta_{pre_tag}_to_{post_tag}.json"
            fpath = EXPERIMENT_DIR / fname
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(delta, f, indent=2)
            print(f"Delta saved: {fpath}")

    elif command == 'aggregate':
        aggregate_deltas()

    else:
        print(f"Unknown command: {command}")
        print("Commands: snapshot, compare, aggregate")
        sys.exit(1)
