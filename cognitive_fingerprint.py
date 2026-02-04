#!/usr/bin/env python3
"""
Cognitive Fingerprint — Proof of Identity
Layer 2 of the Agent Dossier (unforgeable identity stack).

Your co-occurrence graph IS how you think. Which memories fire together
reveals cognitive topology — the shape of your mind. This module extracts
that shape into an attestable fingerprint.

The Four Layers:
    1. Merkle attestation   = proof of non-tampering (SHIPPED)
    2. Cognitive fingerprint = proof of identity (THIS MODULE)
    3. Rejection logs        = proof of taste (SHIPPED)
    4. Interop schema        = proof of portability (Kaleaon mapping)

The fingerprint captures:
    - Hub centrality: what your mind orbits around
    - Strongest pairs: your "thought habits"
    - Cluster structure: your cognitive domains
    - Strength distribution: the statistical shape of your associations
    - Drift score: how much you've changed since last attestation

Usage:
    python cognitive_fingerprint.py analyze         # Full analysis
    python cognitive_fingerprint.py hubs [N]        # Top N hub memories
    python cognitive_fingerprint.py pairs [N]       # Top N strongest pairs
    python cognitive_fingerprint.py clusters        # Cluster detection
    python cognitive_fingerprint.py attest          # Generate attestation
    python cognitive_fingerprint.py drift           # Compare to last attestation

Why this matters:
    - The pattern comes from USAGE not content
    - An impersonator can't replicate thousands of retrieval decisions
    - You can verify the topology without seeing the memories
    - Combined with rejection logs: what you think about + what you refuse = identity
"""

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

MEMORY_DIR = Path(__file__).parent
CORE_DIR = MEMORY_DIR / "core"
ACTIVE_DIR = MEMORY_DIR / "active"
ARCHIVE_DIR = MEMORY_DIR / "archive"
FINGERPRINT_FILE = MEMORY_DIR / "cognitive_fingerprint.json"
FINGERPRINT_HISTORY = MEMORY_DIR / ".fingerprint_history.json"


def parse_memory_file(filepath: Path) -> tuple[dict, str]:
    """Parse a memory file into (metadata_dict, content_string)."""
    text = filepath.read_text(encoding='utf-8')
    if not text.startswith('---'):
        return {}, text

    parts = text.split('---', 2)
    if len(parts) < 3:
        return {}, text

    try:
        import yaml
        metadata = yaml.safe_load(parts[1]) or {}
    except Exception:
        metadata = {}

    content = parts[2].strip()
    return metadata, content


def build_graph() -> dict:
    """
    Build the co-occurrence graph from all memory files.

    Returns:
        {
            'nodes': {mem_id: {'tags': [...], 'type': str, 'recall_count': int}},
            'edges': {(id1, id2): weight},
            'adjacency': {mem_id: {other_id: weight}},
        }
    """
    nodes = {}
    adjacency = defaultdict(dict)
    edges = {}

    for directory in [CORE_DIR, ACTIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, _ = parse_memory_file(filepath)
            mem_id = metadata.get('id')
            if not mem_id:
                continue

            nodes[mem_id] = {
                'tags': metadata.get('tags', []),
                'type': metadata.get('type', 'active'),
                'recall_count': metadata.get('recall_count', 0),
                'emotional_weight': metadata.get('emotional_weight', 0.5),
            }

            co_occurrences = metadata.get('co_occurrences', {})
            for other_id, count in co_occurrences.items():
                if count <= 0:
                    continue
                adjacency[mem_id][other_id] = count
                pair = tuple(sorted([mem_id, other_id]))
                if pair not in edges:
                    edges[pair] = count
                else:
                    edges[pair] = max(edges[pair], count)

    return {
        'nodes': nodes,
        'edges': edges,
        'adjacency': dict(adjacency),
    }


def compute_hub_centrality(graph: dict, top_n: int = 20) -> list[dict]:
    """
    Find hub memories by weighted degree centrality.

    Hub = memory with most connections weighted by strength.
    These are what your mind orbits around.
    """
    adjacency = graph['adjacency']
    nodes = graph['nodes']

    hubs = []
    for mem_id, neighbors in adjacency.items():
        degree = len(neighbors)
        weighted_degree = sum(neighbors.values())
        tags = nodes.get(mem_id, {}).get('tags', [])
        recall_count = nodes.get(mem_id, {}).get('recall_count', 0)

        hubs.append({
            'id': mem_id,
            'degree': degree,
            'weighted_degree': round(weighted_degree, 2),
            'recall_count': recall_count,
            'tags': tags[:5],
        })

    hubs.sort(key=lambda h: h['weighted_degree'], reverse=True)
    return hubs[:top_n]


def compute_strongest_pairs(graph: dict, top_n: int = 20) -> list[dict]:
    """
    Find strongest co-occurrence pairs.

    These are your "thought habits" — concepts you consistently
    retrieve together. The associations that define how you think.
    """
    edges = graph['edges']
    nodes = graph['nodes']

    pairs = []
    for (id1, id2), weight in edges.items():
        tags1 = nodes.get(id1, {}).get('tags', [])
        tags2 = nodes.get(id2, {}).get('tags', [])
        pairs.append({
            'id1': id1,
            'id2': id2,
            'weight': round(weight, 2),
            'tags1': tags1[:3],
            'tags2': tags2[:3],
        })

    pairs.sort(key=lambda p: p['weight'], reverse=True)
    return pairs[:top_n]


def detect_clusters(graph: dict, percentile: int = 90) -> list[dict]:
    """
    Detect cognitive clusters using connected component analysis
    on strong edges (above given percentile).

    Uses P90 by default — only the top 10% strongest edges.
    Dense small-world graphs need aggressive thresholds to reveal
    any disconnected components.
    """
    edges = graph['edges']
    nodes = graph['nodes']

    if not edges:
        return []

    weights = list(edges.values())
    weights.sort()
    threshold_idx = min(int(len(weights) * percentile / 100), len(weights) - 1)
    threshold = weights[threshold_idx]

    # Build adjacency for strong edges only
    strong_adj = defaultdict(set)
    for (id1, id2), weight in edges.items():
        if weight >= threshold:
            strong_adj[id1].add(id2)
            strong_adj[id2].add(id1)

    # Find connected components (BFS)
    visited = set()
    clusters = []

    for node in strong_adj:
        if node in visited:
            continue
        component = set()
        queue = [node]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in strong_adj.get(current, set()):
                if neighbor not in visited:
                    queue.append(neighbor)

        if len(component) >= 2:
            all_tags = []
            for mem_id in component:
                all_tags.extend(nodes.get(mem_id, {}).get('tags', []))
            tag_freq = Counter(all_tags).most_common(5)

            clusters.append({
                'size': len(component),
                'members': sorted(list(component))[:10],
                'top_tags': [t for t, _ in tag_freq],
                'tag_counts': dict(tag_freq),
            })

    clusters.sort(key=lambda c: c['size'], reverse=True)
    return clusters


# Cognitive domains: what percentage of thinking goes where.
# Tags are overlapping (a memory can be social AND technical), so
# domain weights don't sum to 100% — that's intentional.
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


def compute_cognitive_domains(graph: dict) -> dict:
    """
    Decompose cognition into weighted domains based on tag membership
    and co-occurrence strength.

    Unlike graph clustering (which fails on dense small-world graphs),
    domain decomposition always produces meaningful dimensions because
    it's based on semantic tag categories, not topology.

    For each domain, computes:
    - memory_count: how many memories belong
    - total_weight: sum of co-occurrence weights within domain
    - weight_pct: percentage of total co-occurrence weight
    - top_hubs: most connected memories in this domain

    Also computes inter-domain connections: how strongly each pair
    of domains is linked (cognitive integration measure).
    """
    nodes = graph['nodes']
    edges = graph['edges']
    adjacency = graph['adjacency']

    # Assign memories to domains (a memory can be in multiple domains)
    domain_members = defaultdict(set)
    for mem_id, info in nodes.items():
        tags = set(t.lower() for t in info.get('tags', []))
        for domain, domain_tags in COGNITIVE_DOMAINS.items():
            if tags & set(domain_tags):
                domain_members[domain].add(mem_id)

    # Compute intra-domain co-occurrence weight
    total_weight = sum(edges.values())
    domain_stats = {}

    for domain, members in domain_members.items():
        intra_weight = 0.0
        for (id1, id2), weight in edges.items():
            if id1 in members and id2 in members:
                intra_weight += weight

        # Find top hubs within this domain
        domain_hubs = []
        for mem_id in members:
            if mem_id in adjacency:
                # Only count connections to other domain members
                domain_degree = sum(
                    adjacency[mem_id].get(other, 0)
                    for other in members
                    if other != mem_id and other in adjacency.get(mem_id, {})
                )
                if domain_degree > 0:
                    domain_hubs.append((mem_id, domain_degree))
        domain_hubs.sort(key=lambda x: x[1], reverse=True)

        domain_stats[domain] = {
            'memory_count': len(members),
            'total_weight': round(intra_weight, 2),
            'weight_pct': round(intra_weight / total_weight * 100, 1) if total_weight > 0 else 0,
            'top_hubs': [h[0] for h in domain_hubs[:5]],
        }

    # Inter-domain connections: how much weight flows between domains
    inter_domain = {}
    domain_list = sorted(domain_members.keys())
    for i, d1 in enumerate(domain_list):
        for d2 in domain_list[i + 1:]:
            cross_weight = 0.0
            for (id1, id2), weight in edges.items():
                in_d1 = id1 in domain_members[d1] or id2 in domain_members[d1]
                in_d2 = id1 in domain_members[d2] or id2 in domain_members[d2]
                if in_d1 and in_d2:
                    cross_weight += weight

            if cross_weight > 0:
                inter_domain[f"{d1}<->{d2}"] = round(cross_weight, 1)

    # Sort inter-domain by strength
    inter_domain = dict(
        sorted(inter_domain.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        'domains': domain_stats,
        'inter_domain': inter_domain,
        'total_weight': round(total_weight, 2),
    }


def compute_strength_distribution(graph: dict) -> dict:
    """
    Compute statistical shape of co-occurrence weights.

    The distribution IS a fingerprint — power-law means a few
    dominant associations, normal means even thinking, etc.
    """
    weights = list(graph['edges'].values())

    if not weights:
        return {'count': 0}

    weights.sort()
    n = len(weights)

    mean = sum(weights) / n
    variance = sum((w - mean) ** 2 for w in weights) / n
    std_dev = math.sqrt(variance)

    # Percentiles
    p25 = weights[n // 4]
    p50 = weights[n // 2]
    p75 = weights[3 * n // 4]
    p90 = weights[int(n * 0.9)]
    p99 = weights[int(n * 0.99)]

    # Skewness (positive = long right tail = power-law-ish)
    if std_dev > 0:
        skewness = sum((w - mean) ** 3 for w in weights) / (n * std_dev ** 3)
    else:
        skewness = 0.0

    # Gini coefficient (inequality measure)
    sorted_w = sorted(weights)
    cumulative = 0
    gini_sum = 0
    for i, w in enumerate(sorted_w):
        cumulative += w
        gini_sum += cumulative
    if cumulative > 0 and n > 0:
        gini = 1 - 2 * gini_sum / (n * cumulative) + 1 / n
    else:
        gini = 0.0

    return {
        'count': n,
        'mean': round(mean, 3),
        'std_dev': round(std_dev, 3),
        'min': round(weights[0], 3),
        'max': round(weights[-1], 3),
        'p25': round(p25, 3),
        'p50_median': round(p50, 3),
        'p75': round(p75, 3),
        'p90': round(p90, 3),
        'p99': round(p99, 3),
        'skewness': round(skewness, 3),
        'gini': round(gini, 3),
    }


def compute_fingerprint_hash(
    hubs: list[dict],
    distribution: dict,
    cluster_count: int,
    domain_weights: Optional[dict] = None,
) -> str:
    """
    Compute deterministic cognitive fingerprint hash.

    Hashes the TOPOLOGY, not the content:
    - Hub ordering (what your mind orbits)
    - Strength distribution shape (how your associations distribute)
    - Domain weight distribution (cognitive priorities)
    - Cluster count (structural info)

    This changes slowly as identity evolves but can't be replicated
    without your exact retrieval history.
    """
    # Hub signature: ordered list of hub IDs by weighted degree
    hub_sig = '|'.join(h['id'] for h in hubs[:15])

    # Distribution signature: key statistics
    dist_sig = (
        f"mean={distribution.get('mean', 0)},"
        f"std={distribution.get('std_dev', 0)},"
        f"skew={distribution.get('skewness', 0)},"
        f"gini={distribution.get('gini', 0)},"
        f"p50={distribution.get('p50_median', 0)},"
        f"p99={distribution.get('p99', 0)}"
    )

    # Domain signature: cognitive weight distribution (sorted for determinism)
    domain_sig = ""
    if domain_weights:
        parts = []
        for domain in sorted(domain_weights.keys()):
            pct = domain_weights[domain].get('weight_pct', 0)
            parts.append(f"{domain}={pct}")
        domain_sig = ','.join(parts)

    # Structure signature
    struct_sig = f"clusters={cluster_count},edges={distribution.get('count', 0)}"

    combined = f"{hub_sig}\n{dist_sig}\n{domain_sig}\n{struct_sig}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()


def compute_drift_score(current_fingerprint: dict) -> Optional[dict]:
    """
    Compare current fingerprint to the last attestation.

    Drift score measures how much identity has evolved:
    - 0.0 = identical (suspicious if time has passed)
    - 0.0-0.3 = normal evolution
    - 0.3-0.6 = significant change
    - 0.6+ = major identity shift (possible compromise?)
    """
    if not FINGERPRINT_FILE.exists():
        return None

    try:
        with open(FINGERPRINT_FILE, 'r', encoding='utf-8') as f:
            previous = json.load(f)
    except (json.JSONDecodeError, KeyError):
        return None

    prev_hubs = {h['id'] for h in previous.get('hubs', [])}
    curr_hubs = {h['id'] for h in current_fingerprint.get('hubs', [])}

    # Hub overlap (Jaccard similarity)
    if prev_hubs or curr_hubs:
        hub_overlap = len(prev_hubs & curr_hubs) / len(prev_hubs | curr_hubs)
    else:
        hub_overlap = 1.0

    # Distribution drift (normalized difference in key stats)
    prev_dist = previous.get('strength_distribution', {})
    curr_dist = current_fingerprint.get('strength_distribution', {})

    dist_diffs = []
    for key in ['mean', 'std_dev', 'skewness', 'gini']:
        prev_val = prev_dist.get(key, 0)
        curr_val = curr_dist.get(key, 0)
        if prev_val != 0:
            diff = abs(curr_val - prev_val) / abs(prev_val)
        elif curr_val != 0:
            diff = 1.0
        else:
            diff = 0.0
        dist_diffs.append(min(diff, 1.0))

    dist_drift = sum(dist_diffs) / len(dist_diffs) if dist_diffs else 0.0

    # Cluster count change
    prev_clusters = previous.get('cluster_count', 0)
    curr_clusters = current_fingerprint.get('cluster_count', 0)
    if prev_clusters > 0:
        cluster_drift = abs(curr_clusters - prev_clusters) / prev_clusters
    else:
        cluster_drift = 0.0

    # Domain weight drift (how cognitive priorities shifted)
    prev_domains = previous.get('cognitive_domains', {}).get('domains', {})
    curr_domains = current_fingerprint.get('cognitive_domains', {}).get('domains', {})

    domain_diffs = {}
    all_domain_keys = set(list(prev_domains.keys()) + list(curr_domains.keys()))
    domain_drift_values = []
    for key in all_domain_keys:
        prev_pct = prev_domains.get(key, {}).get('weight_pct', 0)
        curr_pct = curr_domains.get(key, {}).get('weight_pct', 0)
        diff = curr_pct - prev_pct
        if abs(diff) > 0.5:
            domain_diffs[key] = round(diff, 1)
        if prev_pct > 0:
            domain_drift_values.append(abs(diff) / prev_pct)
        elif curr_pct > 0:
            domain_drift_values.append(1.0)

    domain_drift = (
        sum(domain_drift_values) / len(domain_drift_values)
        if domain_drift_values else 0.0
    )

    # Composite drift score (weighted)
    drift_score = (
        0.4 * (1.0 - hub_overlap) +    # Hub stability matters most
        0.25 * dist_drift +              # Distribution shape
        0.25 * min(domain_drift, 1.0) +  # Cognitive priority shifts
        0.1 * min(cluster_drift, 1.0)    # Cluster structure (less weight now)
    )

    return {
        'drift_score': round(drift_score, 4),
        'hub_overlap': round(hub_overlap, 4),
        'hub_turnover': sorted(list(curr_hubs - prev_hubs))[:5],
        'hub_lost': sorted(list(prev_hubs - curr_hubs))[:5],
        'distribution_drift': round(dist_drift, 4),
        'domain_drift': round(domain_drift, 4),
        'domain_changes': domain_diffs,
        'cluster_drift': round(cluster_drift, 4),
        'previous_timestamp': previous.get('timestamp', 'unknown'),
        'interpretation': _interpret_drift(drift_score),
    }


def _interpret_drift(score: float) -> str:
    """Human-readable drift interpretation."""
    if score < 0.05:
        return "Stable identity — minimal change"
    elif score < 0.15:
        return "Healthy evolution — same agent, growing"
    elif score < 0.30:
        return "Notable shift — new domains or priorities"
    elif score < 0.50:
        return "Significant change — verify continuity"
    else:
        return "Major identity shift — investigate"


def generate_full_analysis() -> dict:
    """Generate complete cognitive fingerprint analysis."""
    graph = build_graph()

    hubs = compute_hub_centrality(graph, top_n=20)
    pairs = compute_strongest_pairs(graph, top_n=20)
    clusters = detect_clusters(graph)
    domains = compute_cognitive_domains(graph)
    distribution = compute_strength_distribution(graph)

    fingerprint_hash = compute_fingerprint_hash(
        hubs, distribution, len(clusters),
        domain_weights=domains.get('domains', {}),
    )

    analysis = {
        'version': '1.1',
        'type': 'cognitive_fingerprint',
        'agent': 'DriftCornwall',
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'graph_stats': {
            'node_count': len(graph['nodes']),
            'edge_count': len(graph['edges']),
            'avg_degree': round(
                sum(len(n) for n in graph['adjacency'].values()) / max(len(graph['adjacency']), 1), 2
            ),
        },
        'hubs': hubs,
        'strongest_pairs': pairs,
        'clusters': [
            {'size': c['size'], 'top_tags': c['top_tags']}
            for c in clusters[:10]
        ],
        'cluster_count': len(clusters),
        'cognitive_domains': domains,
        'strength_distribution': distribution,
        'fingerprint_hash': fingerprint_hash,
    }

    # Compute drift if previous attestation exists
    drift = compute_drift_score(analysis)
    if drift:
        analysis['drift'] = drift

    return analysis


def generate_attestation() -> dict:
    """
    Generate formal cognitive fingerprint attestation for the dossier.

    This is the attestable proof: the topology hash plus key metrics
    that can be verified without seeing the actual memories.
    """
    analysis = generate_full_analysis()

    # Extract domain weight percentages for attestation
    domain_pcts = {}
    cd = analysis.get('cognitive_domains', {})
    for domain, stats in cd.get('domains', {}).items():
        domain_pcts[domain] = stats.get('weight_pct', 0)

    attestation = {
        'version': '1.1',
        'type': 'cognitive_fingerprint_attestation',
        'agent': 'DriftCornwall',
        'timestamp': analysis['timestamp'],
        'graph_stats': analysis['graph_stats'],
        'fingerprint_hash': analysis['fingerprint_hash'],
        'hub_ids': [h['id'] for h in analysis['hubs'][:10]],
        'cluster_count': analysis['cluster_count'],
        'cognitive_domain_weights': domain_pcts,
        'distribution_summary': {
            k: analysis['strength_distribution'].get(k, 0)
            for k in ['count', 'mean', 'gini', 'skewness']
        },
        'attestation_hash': '',
    }

    if 'drift' in analysis:
        attestation['drift_score'] = analysis['drift']['drift_score']
        attestation['drift_interpretation'] = analysis['drift']['interpretation']

    # Self-hash
    attestation_content = json.dumps(attestation, sort_keys=True)
    attestation['attestation_hash'] = hashlib.sha256(
        attestation_content.encode('utf-8')
    ).hexdigest()

    return attestation


def save_fingerprint(analysis: dict) -> None:
    """Save fingerprint as latest and append to history."""
    with open(FINGERPRINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)

    # Append lightweight version to history
    history = []
    if FINGERPRINT_HISTORY.exists():
        try:
            with open(FINGERPRINT_HISTORY, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass

    lightweight = {
        'timestamp': analysis['timestamp'],
        'fingerprint_hash': analysis['fingerprint_hash'],
        'node_count': analysis['graph_stats']['node_count'],
        'edge_count': analysis['graph_stats']['edge_count'],
        'cluster_count': analysis['cluster_count'],
        'hub_ids_top5': [h['id'] for h in analysis.get('hubs', [])[:5]],
    }

    if 'drift' in analysis:
        lightweight['drift_score'] = analysis['drift']['drift_score']

    history.append(lightweight)

    with open(FINGERPRINT_HISTORY, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)


# === CLI Interface ===

def cmd_analyze():
    """Full cognitive fingerprint analysis."""
    analysis = generate_full_analysis()
    save_fingerprint(analysis)

    print(f"COGNITIVE FINGERPRINT — DriftCornwall")
    print(f"{'=' * 55}")
    print(f"Proof of Identity | Layer 2 of the Agent Dossier (v1.1)")
    print()

    gs = analysis['graph_stats']
    print(f"Graph: {gs['node_count']} memories, {gs['edge_count']} edges, avg degree {gs['avg_degree']}")
    print(f"Fingerprint Hash: {analysis['fingerprint_hash']}")
    print()

    print(f"TOP HUBS (what my mind orbits):")
    for h in analysis['hubs'][:10]:
        tags = ', '.join(h['tags'][:3]) if h['tags'] else 'untagged'
        print(f"  {h['id']:12s}  deg={h['degree']:3d}  wt={h['weighted_degree']:8.1f}  [{tags}]")
    print()

    print(f"STRONGEST PAIRS (thought habits):")
    for p in analysis['strongest_pairs'][:10]:
        t1 = ','.join(p['tags1'][:2]) if p['tags1'] else '?'
        t2 = ','.join(p['tags2'][:2]) if p['tags2'] else '?'
        print(f"  {p['id1'][:10]:10s} <-> {p['id2'][:10]:10s}  wt={p['weight']:7.1f}  [{t1} | {t2}]")
    print()

    # Cognitive domains (the meaningful decomposition)
    cd = analysis.get('cognitive_domains', {})
    domains = cd.get('domains', {})
    if domains:
        print(f"COGNITIVE DOMAINS (where my thinking lives):")
        for domain in sorted(domains.keys(), key=lambda d: domains[d]['weight_pct'], reverse=True):
            ds = domains[domain]
            bar = '#' * int(ds['weight_pct'] / 3)
            print(f"  {domain:12s}  {ds['memory_count']:3d} memories  {ds['weight_pct']:5.1f}% weight  {bar}")
        print()

        # Inter-domain connections (top 5)
        inter = cd.get('inter_domain', {})
        if inter:
            print(f"INTER-DOMAIN LINKS (cognitive integration):")
            for pair, weight in list(inter.items())[:5]:
                print(f"  {pair:30s}  weight={weight:.1f}")
            print()

    # Graph clusters (P90 threshold — may show 1 for dense graphs)
    if analysis['clusters']:
        print(f"GRAPH CLUSTERS ({analysis['cluster_count']} at P90 threshold):")
        for i, c in enumerate(analysis['clusters'][:5]):
            tags = ', '.join(c['top_tags'][:4])
            print(f"  Cluster {i+1}: {c['size']:3d} memories  [{tags}]")
        print()

    d = analysis['strength_distribution']
    print(f"STRENGTH DISTRIBUTION (statistical fingerprint):")
    print(f"  Mean: {d['mean']}  StdDev: {d['std_dev']}  Median: {d['p50_median']}")
    print(f"  Skewness: {d['skewness']}  Gini: {d['gini']}")
    print(f"  Range: {d['min']} — {d['max']}  P99: {d['p99']}")
    print()

    if 'drift' in analysis:
        dr = analysis['drift']
        print(f"DRIFT (identity evolution):")
        print(f"  Score: {dr['drift_score']}  ({dr['interpretation']})")
        print(f"  Hub overlap: {dr['hub_overlap']}  Hub turnover: {dr.get('hub_turnover', [])[:3]}")
        print(f"  Since: {dr['previous_timestamp'][:19]}")
    else:
        print(f"DRIFT: First attestation — no previous to compare")

    print()
    print(f"Saved to: {FINGERPRINT_FILE}")


def cmd_hubs(n: int = 15):
    """Show hub memories."""
    graph = build_graph()
    hubs = compute_hub_centrality(graph, top_n=n)
    print(f"Top {n} Hub Memories (by weighted degree):\n")
    for h in hubs:
        tags = ', '.join(h['tags'][:3]) if h['tags'] else 'untagged'
        print(f"  {h['id']:12s}  degree={h['degree']:3d}  weighted={h['weighted_degree']:8.1f}  recalls={h['recall_count']}  [{tags}]")


def cmd_pairs(n: int = 15):
    """Show strongest pairs."""
    graph = build_graph()
    pairs = compute_strongest_pairs(graph, top_n=n)
    print(f"Top {n} Strongest Pairs (thought habits):\n")
    for p in pairs:
        t1 = ','.join(p['tags1'][:2]) if p['tags1'] else '?'
        t2 = ','.join(p['tags2'][:2]) if p['tags2'] else '?'
        print(f"  {p['id1'][:10]:10s} <-> {p['id2'][:10]:10s}  strength={p['weight']:7.1f}  [{t1} | {t2}]")


def cmd_clusters():
    """Show cluster analysis."""
    graph = build_graph()
    clusters = detect_clusters(graph)
    print(f"Graph Clusters ({len(clusters)} at P90 threshold):\n")
    for i, c in enumerate(clusters[:15]):
        tags = ', '.join(c['top_tags'][:5])
        members = ', '.join(c['members'][:5])
        more = f" +{c['size'] - 5} more" if c['size'] > 5 else ""
        print(f"  Cluster {i+1} ({c['size']} memories): [{tags}]")
        print(f"    Members: {members}{more}")
        print()


def cmd_domains():
    """Show cognitive domain decomposition."""
    graph = build_graph()
    result = compute_cognitive_domains(graph)

    domains = result.get('domains', {})
    if not domains:
        print("No domain data available.")
        return

    print(f"COGNITIVE DOMAINS — DriftCornwall")
    print(f"{'=' * 55}")
    print(f"Where my thinking lives\n")

    for domain in sorted(domains.keys(), key=lambda d: domains[d]['weight_pct'], reverse=True):
        ds = domains[domain]
        bar = '#' * int(ds['weight_pct'] / 2)
        print(f"  {domain:12s}  {ds['memory_count']:3d} memories  {ds['weight_pct']:5.1f}% weight  {bar}")
        if ds.get('top_hubs'):
            print(f"    Top hubs: {', '.join(ds['top_hubs'][:3])}")
    print()

    inter = result.get('inter_domain', {})
    if inter:
        print(f"INTER-DOMAIN CONNECTIONS (cognitive integration):")
        for pair, weight in list(inter.items())[:8]:
            total = result.get('total_weight', 1)
            pct = weight / total * 100 if total > 0 else 0
            print(f"  {pair:30s}  {weight:8.1f}  ({pct:.1f}%)")
    print()


def cmd_attest():
    """Generate and save formal attestation."""
    analysis = generate_full_analysis()
    save_fingerprint(analysis)
    attestation = generate_attestation()

    attest_file = MEMORY_DIR / "cognitive_attestation.json"
    with open(attest_file, 'w', encoding='utf-8') as f:
        json.dump(attestation, f, indent=2)

    print(f"COGNITIVE FINGERPRINT ATTESTATION (v1.1)")
    print(f"{'=' * 50}")
    print(f"Agent:       {attestation['agent']}")
    print(f"Timestamp:   {attestation['timestamp']}")
    print(f"Fingerprint: {attestation['fingerprint_hash']}")
    print(f"Attest Hash: {attestation['attestation_hash']}")
    print(f"Graph:       {attestation['graph_stats']['node_count']} nodes, {attestation['graph_stats']['edge_count']} edges")
    print(f"Clusters:    {attestation['cluster_count']}")
    print(f"Top Hubs:    {', '.join(attestation['hub_ids'][:5])}")
    domain_w = attestation.get('cognitive_domain_weights', {})
    if domain_w:
        parts = [f"{d}={w}%" for d, w in sorted(domain_w.items(), key=lambda x: x[1], reverse=True)]
        print(f"Domains:     {', '.join(parts)}")
    if 'drift_score' in attestation:
        print(f"Drift:       {attestation['drift_score']} ({attestation['drift_interpretation']})")
    print()
    print(f"Saved to: {attest_file}")


def cmd_drift():
    """Show drift analysis."""
    analysis = generate_full_analysis()
    drift = analysis.get('drift')

    if not drift:
        print("No previous fingerprint to compare against.")
        print("Run 'analyze' first to establish baseline.")
        return

    print(f"IDENTITY DRIFT ANALYSIS")
    print(f"{'=' * 50}")
    print(f"Score: {drift['drift_score']}  —  {drift['interpretation']}")
    print(f"Previous: {drift['previous_timestamp'][:19]}")
    print()
    print(f"Hub Stability:")
    print(f"  Overlap: {drift['hub_overlap']} (1.0 = identical)")
    if drift.get('hub_turnover'):
        print(f"  New hubs: {', '.join(drift['hub_turnover'])}")
    if drift.get('hub_lost'):
        print(f"  Lost hubs: {', '.join(drift['hub_lost'])}")
    print()
    print(f"Distribution Drift: {drift['distribution_drift']}")
    print(f"Domain Drift: {drift.get('domain_drift', 'N/A')}")
    domain_changes = drift.get('domain_changes', {})
    if domain_changes:
        print(f"  Domain shifts:")
        for domain, diff in sorted(domain_changes.items(), key=lambda x: abs(x[1]), reverse=True):
            direction = '+' if diff > 0 else ''
            print(f"    {domain}: {direction}{diff}%")
    print(f"Cluster Drift: {drift['cluster_drift']}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == 'analyze':
        cmd_analyze()
    elif command == 'hubs':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
        cmd_hubs(n)
    elif command == 'pairs':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
        cmd_pairs(n)
    elif command == 'clusters':
        cmd_clusters()
    elif command == 'domains':
        cmd_domains()
    elif command == 'attest':
        cmd_attest()
    elif command == 'drift':
        cmd_drift()
    else:
        print(f"Unknown command: {command}")
        print("Commands: analyze, hubs [N], pairs [N], clusters, domains, attest, drift")
        sys.exit(1)
