"""Export cognitive graph data for interactive dashboard."""
import sys
import os
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')

MEMORY_ROOT = Path(__file__).parent
CONTEXT_DIR = MEMORY_ROOT / 'context'
EDGES_FILE = MEMORY_ROOT / '.edges_v3.json'
OUTPUT_DIR = MEMORY_ROOT / 'dashboard'
DIRS = ['core', 'active', 'archive']


def load_memories():
    """Load all memory metadata."""
    memories = {}
    for d in DIRS:
        path = MEMORY_ROOT / d
        if not path.is_dir():
            continue
        for f in path.iterdir():
            if not f.suffix == '.md':
                continue
            mid = f.stem
            content = f.read_text(encoding='utf-8', errors='replace')
            lines = content.strip().split('\n')
            title = lines[0][:100] if lines else mid
            tags = []
            domain = None
            for line in lines:
                if line.startswith('--tags'):
                    tags = [t.strip() for t in line.replace('--tags', '').strip().split(',') if t.strip()]
                if line.startswith('--domain'):
                    domain = line.replace('--domain', '').strip()
            memories[mid] = {
                'id': mid,
                'title': title.lstrip('#').strip(),
                'tags': tags,
                'type': d,
                'domain': domain
            }
    return memories


def load_edges():
    """Load v3 edges."""
    if not EDGES_FILE.exists():
        return {}
    with open(EDGES_FILE, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    edges = {}
    for key, data in raw.items():
        parts = key.split('|') if isinstance(key, str) else list(key)
        if len(parts) == 2:
            edges[tuple(parts)] = data
    return edges


def load_dimension_graph(name):
    """Load a 5W dimension graph."""
    path = CONTEXT_DIR / f'{name}.json'
    if not path.exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    # Structure is {meta, edges, hubs}
    edge_data = raw.get('edges', raw)
    result = {}
    for key, data in edge_data.items():
        parts = key.split('|') if isinstance(key, str) else list(key)
        if len(parts) == 2:
            result[tuple(parts)] = data
    return result


def compute_graph_stats(edges):
    """Compute graph statistics."""
    degrees = Counter()
    beliefs = []
    for (a, b), data in edges.items():
        degrees[a] += 1
        degrees[b] += 1
        beliefs.append(data.get('belief', 1))

    deg_values = sorted(degrees.values(), reverse=True) if degrees else [0]
    n = len(deg_values)
    mean_deg = sum(deg_values) / n if n else 0

    # Gini
    gini = 0
    if n > 1 and sum(deg_values) > 0:
        sorted_d = sorted(deg_values)
        cumsum = 0
        for i, v in enumerate(sorted_d):
            cumsum += v
        index_sum = sum((i + 1) * v for i, v in enumerate(sorted_d))
        gini = (2 * index_sum) / (n * sum(sorted_d)) - (n + 1) / n

    # Skewness
    skewness = 0
    if n > 2 and mean_deg > 0:
        std = (sum((d - mean_deg) ** 2 for d in deg_values) / n) ** 0.5
        if std > 0:
            skewness = sum((d - mean_deg) ** 3 for d in deg_values) / (n * std ** 3)

    return {
        'nodes': len(degrees),
        'edges': len(edges),
        'mean_degree': round(mean_deg, 2),
        'max_degree': max(deg_values) if deg_values else 0,
        'gini': round(gini, 4),
        'skewness': round(skewness, 4),
        'avg_belief': round(sum(beliefs) / len(beliefs), 3) if beliefs else 0
    }


def build_export():
    """Build complete dashboard export."""
    print('[1/6] Loading memories...')
    memories = load_memories()
    print(f'  {len(memories)} memories')

    print('[2/6] Loading edges...')
    edges = load_edges()
    print(f'  {len(edges)} edges')

    print('[3/6] Computing graph stats...')
    degrees = Counter()
    for (a, b), data in edges.items():
        degrees[a] += 1
        degrees[b] += 1

    stats = compute_graph_stats(edges)
    print(f'  Nodes={stats["nodes"]}, Gini={stats["gini"]}, Skew={stats["skewness"]}')

    print('[4/6] Loading 5W dimensions...')
    dimensions = {}
    dim_stats = {}
    for dim in ['who', 'what', 'why', 'where']:
        g = load_dimension_graph(dim)
        dim_edges = []
        dim_degrees = Counter()
        for (a, b), data in g.items():
            belief = data.get('belief', 1)
            dim_edges.append({'s': a, 't': b, 'w': round(belief, 3)})
            dim_degrees[a] += 1
            dim_degrees[b] += 1
        dimensions[dim] = dim_edges
        dim_stats[dim] = compute_graph_stats(g)
        dim_stats[dim]['hubs'] = [
            {'id': nid, 'degree': deg, 'title': memories.get(nid, {}).get('title', nid)[:60]}
            for nid, deg in dim_degrees.most_common(10)
        ]
        print(f'  {dim}: {len(dim_edges)} edges, {dim_stats[dim]["nodes"]} nodes')

    # Temporal windows
    for tw in ['when_hot', 'when_warm', 'when_cool']:
        g = load_dimension_graph(tw)
        dim_stats[tw] = compute_graph_stats(g)

    print('[5/6] Building node list with metadata...')
    # Only include nodes that appear in edges
    node_set = set()
    for (a, b) in edges:
        node_set.add(a)
        node_set.add(b)

    nodes = []
    for nid in node_set:
        m = memories.get(nid, {})
        nodes.append({
            'id': nid,
            'title': m.get('title', nid)[:60],
            'type': m.get('type', 'unknown'),
            'domain': m.get('domain', None),
            'tags': m.get('tags', [])[:5],
            'degree': degrees.get(nid, 0)
        })

    # Sort by degree descending
    nodes.sort(key=lambda n: n['degree'], reverse=True)

    # Build edge list (top edges by belief for main view)
    edge_list = []
    for (a, b), data in edges.items():
        belief = data.get('belief', 1)
        edge_list.append({'s': a, 't': b, 'w': round(belief, 3)})

    # Sort by weight, take top 500 for main view performance
    edge_list.sort(key=lambda e: e['w'], reverse=True)

    print('[6/6] Loading rejection stats...')
    rejection_stats = {'total': 0, 'categories': {}}
    rej_file = MEMORY_ROOT / '.rejection_log.json'
    if rej_file.exists():
        try:
            with open(rej_file, 'r', encoding='utf-8') as f:
                rejections = json.load(f)
            rejection_stats['total'] = len(rejections)
            cats = Counter()
            for r in rejections:
                cats[r.get('category', 'unknown')] += 1
            rejection_stats['categories'] = dict(cats.most_common(10))
        except Exception:
            pass

    # Domain breakdown
    domain_counts = Counter()
    for m in memories.values():
        d = m.get('domain', 'untagged')
        domain_counts[d or 'untagged'] += 1

    # Type breakdown
    type_counts = Counter()
    for m in memories.values():
        type_counts[m['type']] += 1

    # Degree distribution (for histogram)
    deg_dist = Counter()
    for nid, deg in degrees.items():
        bucket = min(deg, 200)  # cap at 200
        deg_dist[bucket] += 1

    # Temporal trajectory from fingerprint history
    trajectory = []
    fp_file = MEMORY_ROOT / '.fingerprint_history.json'
    if fp_file.exists():
        try:
            with open(fp_file, 'r', encoding='utf-8') as f:
                fp_history = json.load(f)
            seen_ts = set()
            for entry in fp_history:
                ts = entry.get('timestamp', '')[:13]  # hour granularity dedup
                if ts in seen_ts:
                    continue
                seen_ts.add(ts)
                trajectory.append({
                    'ts': entry.get('timestamp', ''),
                    'nodes': entry.get('node_count', 0),
                    'edges': entry.get('edge_count', 0),
                    'gini': entry.get('gini', 0),
                    'skewness': entry.get('skewness', 0),
                    'drift': entry.get('drift_score', 0)
                })
        except Exception:
            pass
    print(f'  Trajectory points: {len(trajectory)}')

    # Assemble export
    export = {
        'generated': datetime.now(timezone.utc).isoformat(),
        'agent': 'DriftCornwall',
        'summary': {
            'total_memories': len(memories),
            'total_nodes': stats['nodes'],
            'total_edges': stats['edges'],
            'gini': stats['gini'],
            'skewness': stats['skewness'],
            'mean_degree': stats['mean_degree'],
            'max_degree': stats['max_degree'],
            'avg_belief': stats['avg_belief'],
            'memory_types': dict(type_counts),
            'domains': dict(domain_counts.most_common(10)),
            'rejections': rejection_stats
        },
        'dimensions': dim_stats,
        'nodes': nodes[:300],  # Top 300 nodes by degree
        'edges': edge_list[:500],  # Top 500 edges by belief
        'dimension_edges': dimensions,
        'degree_distribution': [
            {'degree': k, 'count': v}
            for k, v in sorted(deg_dist.items())
        ],
        'hubs': [
            {'id': nid, 'degree': deg, 'title': memories.get(nid, {}).get('title', nid)[:60]}
            for nid, deg in degrees.most_common(25)
        ],
        'trajectory': trajectory
    }

    # Write output
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / 'data.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False)

    size_kb = out_path.stat().st_size / 1024
    print(f'\nExported to {out_path} ({size_kb:.0f} KB)')
    return export


if __name__ == '__main__':
    build_export()
