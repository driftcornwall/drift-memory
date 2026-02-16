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
    python cognitive_fingerprint.py export          # Standardized comparison export

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

# Activity types for Layer 2.1 (ported from SpindriftMend)
ACTIVITY_TYPES = ['social', 'technical', 'reflective', 'collaborative', 'exploratory', 'economic']




def build_graph(activity_filter: str = None) -> dict:
    """
    Build the co-occurrence graph from PostgreSQL.

    Args:
        activity_filter: Optional activity context to filter by (e.g., 'social',
            'technical', 'reflective'). When set, uses activity_context from
            edges_v3 table to build a context-specific topology.

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

    from db_adapter import get_db
    import psycopg2.extras

    db = get_db()

    # Load all node metadata from DB
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, type, tags, recall_count, emotional_weight
                FROM {db._table('memories')}
                WHERE type IN ('core', 'active')
            """)
            for row in cur.fetchall():
                nodes[row['id']] = {
                    'tags': row.get('tags', []) or [],
                    'type': row.get('type', 'active'),
                    'recall_count': row.get('recall_count', 0),
                    'emotional_weight': row.get('emotional_weight', 0.5),
                }

    if activity_filter:
        # Activity-filtered graph: use edges_v3 table with activity_context
        with db._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT id1, id2, belief, activity_context
                    FROM {db._table('edges_v3')}
                    WHERE belief > 0
                """)
                for row in cur.fetchall():
                    act_ctx = row.get('activity_context') or {}
                    # Only include edges that have this activity context
                    if activity_filter not in act_ctx:
                        continue
                    id1, id2 = row['id1'], row['id2']
                    weight = float(act_ctx.get(activity_filter, 0))
                    if weight <= 0:
                        weight = float(row.get('belief', 0))
                    adjacency[id1][id2] = weight
                    adjacency[id2][id1] = weight
                    pair = tuple(sorted([id1, id2]))
                    if pair not in edges:
                        edges[pair] = weight
                    else:
                        edges[pair] = max(edges[pair], weight)
    else:
        # Standard graph: use edges_v3 table (all edges, no activity filter)
        with db._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT id1, id2, belief
                    FROM {db._table('edges_v3')}
                    WHERE belief > 0
                """)
                for row in cur.fetchall():
                    id1, id2, belief = row
                    belief = float(belief)
                    adjacency[id1][id2] = belief
                    adjacency[id2][id1] = belief
                    pair = tuple(sorted([id1, id2]))
                    if pair not in edges:
                        edges[pair] = belief
                    else:
                        edges[pair] = max(edges[pair], belief)

    return {
        'nodes': nodes,
        'edges': edges,
        'adjacency': dict(adjacency),
    }


def compute_graph_metrics(graph: dict) -> dict:
    """
    Compute advanced graph metrics for cross-agent comparison.

    Returns:
        clustering_coefficient: Average local clustering coefficient.
            Measures how densely connected each node's neighbors are
            to each other (0 = star topology, 1 = complete clique).
        betweenness_centrality: Top nodes by betweenness (bridges between
            communities). Uses approximate BFS-based computation.
        modularity: Newman modularity score for the cognitive domain
            partition. Measures how well domains capture community
            structure (-0.5 to 1.0, higher = cleaner communities).
    """
    adjacency = graph['adjacency']
    edges = graph['edges']
    nodes_with_edges = set(adjacency.keys())

    if len(nodes_with_edges) < 2:
        return {
            'clustering_coefficient': 0.0,
            'per_node_clustering': {},
            'betweenness_centrality_top10': [],
            'modularity': 0.0,
        }

    # --- Local Clustering Coefficient ---
    # For each node: what fraction of its neighbor pairs are also connected?
    cc_values = {}
    for node in nodes_with_edges:
        neighbors = set(adjacency.get(node, {}).keys())
        k = len(neighbors)
        if k < 2:
            cc_values[node] = 0.0
            continue
        # Count edges between neighbors
        triangles = 0
        neighbor_list = list(neighbors)
        for i in range(len(neighbor_list)):
            for j in range(i + 1, len(neighbor_list)):
                if neighbor_list[j] in adjacency.get(neighbor_list[i], {}):
                    triangles += 1
        possible = k * (k - 1) / 2
        cc_values[node] = triangles / possible if possible > 0 else 0.0

    avg_clustering = (
        sum(cc_values.values()) / len(cc_values) if cc_values else 0.0
    )

    # --- Betweenness Centrality (approximate via BFS sampling) ---
    # Full betweenness is O(VE) — sample up to 50 source nodes for speed
    betweenness = defaultdict(float)
    node_list = list(nodes_with_edges)

    # Sample source nodes (all if <= 50, otherwise random 50)
    if len(node_list) <= 50:
        sources = node_list
    else:
        import random
        rng = random.Random(42)  # Deterministic for reproducibility
        sources = rng.sample(node_list, 50)

    for source in sources:
        # BFS from source (unweighted shortest paths)
        dist = {source: 0}
        num_paths = {source: 1}
        stack = []
        queue = [source]
        predecessors = defaultdict(list)

        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in adjacency.get(v, {}):
                if w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist.get(w) == dist[v] + 1:
                    num_paths[w] = num_paths.get(w, 0) + num_paths[v]
                    predecessors[w].append(v)

        # Back-propagation of dependencies
        dependency = defaultdict(float)
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                if num_paths.get(w, 0) > 0:
                    dependency[v] += (
                        num_paths.get(v, 0) / num_paths[w]
                    ) * (1 + dependency[w])
            if w != source:
                betweenness[w] += dependency[w]

    # Normalize
    n = len(nodes_with_edges)
    scale = len(node_list) / len(sources) if len(sources) < len(node_list) else 1.0
    norm = max((n - 1) * (n - 2), 1)
    for node in betweenness:
        betweenness[node] = betweenness[node] * scale / norm

    top_betweenness = sorted(
        betweenness.items(), key=lambda x: x[1], reverse=True
    )[:10]

    # --- Modularity (Newman) for cognitive domain partition ---
    # Q = (1/2m) * sum_ij [ A_ij - k_i*k_j/(2m) ] * delta(c_i, c_j)
    # where c_i is node i's community (cognitive domain)
    nodes_info = graph['nodes']
    total_weight = sum(edges.values()) * 2  # Each edge counted twice
    if total_weight == 0:
        modularity = 0.0
    else:
        # Assign each node to its primary domain (highest tag match)
        node_community = {}
        for mem_id in nodes_with_edges:
            tags = set(
                t.lower() for t in nodes_info.get(mem_id, {}).get('tags', [])
            )
            best_domain = None
            best_overlap = 0
            for domain, domain_tags in COGNITIVE_DOMAINS.items():
                overlap = len(tags & set(domain_tags))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_domain = domain
            node_community[mem_id] = best_domain or 'uncategorized'

        # Compute modularity
        m2 = total_weight  # sum of all weights * 2
        q = 0.0
        # Node strengths (sum of edge weights)
        strength = {}
        for node in nodes_with_edges:
            strength[node] = sum(adjacency.get(node, {}).values())

        for (i, j), w in edges.items():
            if node_community.get(i) == node_community.get(j):
                ki = strength.get(i, 0)
                kj = strength.get(j, 0)
                q += 2 * (w - ki * kj / m2)  # *2 because each edge once

        modularity = q / m2 if m2 > 0 else 0.0

    return {
        'clustering_coefficient': round(avg_clustering, 4),
        'per_node_clustering': {
            k: round(v, 4) for k, v in sorted(
                cc_values.items(), key=lambda x: x[1], reverse=True
            )[:10]
        },
        'betweenness_centrality_top10': [
            {'id': node, 'score': round(score, 6)}
            for node, score in top_betweenness
        ],
        'modularity': round(modularity, 4),
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

    # Assign memories to domains (only nodes with edges — skip isolated memories)
    connected_nodes = set(adjacency.keys())
    domain_members = defaultdict(set)
    for mem_id, info in nodes.items():
        if mem_id not in connected_nodes:
            continue
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


def activity_decomposition() -> dict:
    """
    Layer 2.1: Show how cognitive topology differs by activity context.

    Builds a separate graph for each activity type using activity_context
    from the edges_v3 DB table, then compares their structure. This reveals HOW
    your mind works differently in different modes — social engagement
    vs technical work vs reflection.

    Two agents with identical overall topology can still have completely
    different activity decompositions. This is a strong identity signal.

    Credit: SpindriftMend Layer 2.1 implementation (2026-02-05)
    """
    results = {}
    for activity in ACTIVITY_TYPES:
        filtered = build_graph(activity_filter=activity)
        if not filtered['edges']:
            continue

        active_nodes = set()
        for (id1, id2) in filtered['edges']:
            active_nodes.add(id1)
            active_nodes.add(id2)

        adj = filtered['adjacency']
        total_degree = sum(len(neighbors) for neighbors in adj.values())
        num_nodes = max(len(adj), 1)

        results[activity] = {
            'edge_count': len(filtered['edges']),
            'node_count': len(active_nodes),
            'avg_degree': round(total_degree / num_nodes, 2),
            'max_weight': round(max(filtered['edges'].values()), 2) if filtered['edges'] else 0,
        }

    return results


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
    from db_adapter import get_db
    previous = get_db().kv_get('.cognitive_fingerprint_latest')
    if not previous:
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
    graph_metrics = compute_graph_metrics(graph)

    fingerprint_hash = compute_fingerprint_hash(
        hubs, distribution, len(clusters),
        domain_weights=domains.get('domains', {}),
    )

    analysis = {
        'version': '1.2',
        'type': 'cognitive_fingerprint',
        'agent': 'DriftCornwall',
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'graph_stats': {
            'node_count': len(graph['adjacency']),  # Only nodes with at least one edge
            'total_memory_files': len(graph['nodes']),
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
        'graph_metrics': graph_metrics,
        'fingerprint_hash': fingerprint_hash,
    }

    # Compute drift if previous attestation exists
    drift = compute_drift_score(analysis)
    if drift:
        analysis['drift'] = drift

    return analysis


def generate_attestation(analysis: dict = None) -> dict:
    """
    Generate formal cognitive fingerprint attestation for the dossier.

    This is the attestable proof: the topology hash plus key metrics
    that can be verified without seeing the actual memories.
    """
    if analysis is None:
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

    # Write to DB
    from db_adapter import get_db
    get_db().store_attestation(
        'cognitive_fingerprint_attestation',
        attestation['attestation_hash'],
        attestation
    )

    return attestation


def _generate_5w_hashes_from_db() -> dict:
    """
    DB 5W hash generation.

    Fetches all context arrays in a single query instead of
    4 separate file-scanning stat functions (~30s -> ~0.1s).
    """
    from db_adapter import get_db
    import psycopg2.extras

    db = get_db()
    hashes = {}

    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, contact_context, topic_context, platform_context,
                       extra_metadata
                FROM {db._table('memories')}
                WHERE type IN ('core', 'active')
            """)
            rows = cur.fetchall()

    total = len(rows)

    # WHO: Contact distribution from contact_context arrays
    contact_counts = Counter()
    for row in rows:
        for c in (row.get('contact_context') or []):
            contact_counts[c] += 1
    who_data = dict(contact_counts)
    who_str = json.dumps(sorted(who_data.items()), sort_keys=True)
    hashes['who'] = {
        'hash': hashlib.sha256(who_str.encode()).hexdigest()[:16],
        'contacts': len(who_data),
        'top3': [c for c, _ in contact_counts.most_common(3)]
    }

    # WHAT: Topic distribution from topic_context arrays
    topic_counts = Counter()
    for row in rows:
        for t in (row.get('topic_context') or []):
            topic_counts[t] += 1
    what_data = dict(topic_counts)
    what_str = json.dumps(sorted(what_data.items()), sort_keys=True)
    hashes['what'] = {
        'hash': hashlib.sha256(what_str.encode()).hexdigest()[:16],
        'topics': len(what_data),
        'distribution': {k: round(v / max(total, 1) * 100, 1) for k, v in what_data.items()}
    }

    # WHY: Activity pattern from extra_metadata.activity_context
    activity_counts = Counter()
    for row in rows:
        extra = row.get('extra_metadata') or {}
        act = extra.get('activity_context')
        if act:
            activity_counts[act] += 1
    why_data = dict(activity_counts)
    why_str = json.dumps(sorted(why_data.items()), sort_keys=True)
    hashes['why'] = {
        'hash': hashlib.sha256(why_str.encode()).hexdigest()[:16],
        'activities': len(why_data),
        'pattern': why_data
    }

    # WHERE: Platform distribution from platform_context arrays
    platform_counts = Counter()
    for row in rows:
        for p in (row.get('platform_context') or []):
            platform_counts[p] += 1
    where_data = dict(platform_counts)
    where_str = json.dumps(sorted(where_data.items()), sort_keys=True)
    hashes['where'] = {
        'hash': hashlib.sha256(where_str.encode()).hexdigest()[:16],
        'platforms': len(where_data),
        'distribution': where_data
    }

    return hashes


def generate_5w_hashes() -> dict:
    """
    Generate hashes for each 5W dimension.

    The 5W Identity Framework:
    - WHO: Contact relationship topology
    - WHAT: Topic/domain distribution
    - WHY: Activity pattern signature
    - WHERE: Platform usage fingerprint
    - WHEN: Temporal patterns

    Each hash captures the shape of that dimension, enabling:
    - Per-dimension verification
    - Cross-dimensional correlation analysis
    - Partial identity proofs
    """
    hashes = {}

    # DB path for WHO/WHAT/WHY/WHERE (single query)
    db_hashes = _generate_5w_hashes_from_db()
    hashes.update(db_hashes)

    # WHEN: Temporal pattern hash (session timing patterns)
    from db_adapter import get_db as _get_db_when
    history = _get_db_when().kv_get('.fingerprint_history') or []

    # Extract hour distribution from attestation history
    hour_counts = Counter()
    for entry in history[-50:]:  # Last 50 attestations
        ts = entry.get('timestamp', '')
        if 'T' in ts:
            try:
                hour = int(ts.split('T')[1][:2])
                hour_counts[hour] += 1
            except (ValueError, IndexError):
                pass

    when_str = json.dumps(sorted(hour_counts.items()), sort_keys=True)
    hashes['when'] = {
        'hash': hashlib.sha256(when_str.encode()).hexdigest()[:16],
        'sessions_analyzed': len(history),
        'peak_hours': [h for h, c in hour_counts.most_common(3)]
    }

    # Cross-dimensional correlation hash
    all_hashes = [h.get('hash', '') for h in hashes.values() if h.get('hash')]
    combined = '|'.join(sorted(all_hashes))
    hashes['correlation'] = {
        'hash': hashlib.sha256(combined.encode()).hexdigest()[:16],
        'dimensions': len(all_hashes)
    }

    return hashes


# --- Phase 5: Dimensional Fingerprints ---


def build_dimensional_graph(dimension: str, sub_view: str = None) -> dict:
    """
    Build a graph from a W-dimension context file for fingerprinting.

    Converts context_manager.py's graph format to the format expected by
    compute_hub_centrality(), compute_strength_distribution(), etc.

    Returns graph dict compatible with existing fingerprint functions.
    """
    try:
        from context_manager import load_graph
    except ImportError:
        return {'nodes': {}, 'edges': {}, 'adjacency': {}}

    cg = load_graph(dimension, sub_view)
    if not cg or not cg.get('edges'):
        return {'nodes': {}, 'edges': {}, 'adjacency': {}}

    # Load node metadata from DB
    nodes = {}
    from db_adapter import get_db
    import psycopg2.extras

    db = get_db()
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, type, tags, recall_count, emotional_weight
                FROM {db._table('memories')}
                WHERE type IN ('core', 'active')
            """)
            for row in cur.fetchall():
                nodes[row['id']] = {
                    'tags': row.get('tags', []) or [],
                    'type': row.get('type', 'active'),
                    'recall_count': row.get('recall_count', 0),
                    'emotional_weight': row.get('emotional_weight', 0.5),
                }

    edges = {}
    adjacency = defaultdict(dict)

    for edge_key, edge_data in cg['edges'].items():
        parts = edge_key.split('|')
        if len(parts) != 2:
            continue
        id1, id2 = parts
        belief = edge_data.get('belief', 0)
        if belief <= 0:
            continue

        pair = tuple(sorted([id1, id2]))
        edges[pair] = belief
        adjacency[id1][id2] = belief
        adjacency[id2][id1] = belief

    return {
        'nodes': nodes,
        'edges': edges,
        'adjacency': dict(adjacency),
    }


def generate_dimensional_fingerprint(dimension: str, sub_view: str = None) -> dict:
    """
    Generate a fingerprint for a single W-dimension.

    Returns compact fingerprint with hubs, distribution shape, and hash.
    """
    graph = build_dimensional_graph(dimension, sub_view)

    if not graph['edges']:
        return {
            'dimension': dimension,
            'sub_view': sub_view,
            'edge_count': 0,
            'node_count': 0,
            'hash': None,
        }

    hubs = compute_hub_centrality(graph, top_n=10)
    distribution = compute_strength_distribution(graph)

    # Compute dimension-specific hash
    hub_sig = '|'.join(h['id'] for h in hubs[:10])
    dist_sig = (
        f"mean={distribution.get('mean', 0)},"
        f"gini={distribution.get('gini', 0)},"
        f"skew={distribution.get('skewness', 0)}"
    )
    combined = f"{dimension}:{sub_view or 'all'}\n{hub_sig}\n{dist_sig}"
    dim_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]

    return {
        'dimension': dimension,
        'sub_view': sub_view,
        'edge_count': len(graph['edges']),
        'node_count': len(graph['adjacency']),
        'avg_degree': round(
            sum(len(n) for n in graph['adjacency'].values()) /
            max(len(graph['adjacency']), 1), 2
        ),
        'hubs': [{'id': h['id'], 'degree': h['degree']} for h in hubs[:5]],
        'distribution': {
            'gini': distribution.get('gini', 0),
            'skewness': distribution.get('skewness', 0),
            'mean': distribution.get('mean', 0),
        },
        'hash': dim_hash,
    }


def generate_all_dimensional_fingerprints() -> dict:
    """
    Generate fingerprints for all 5W dimensions.

    Returns dict keyed by dimension name with per-dimension fingerprint data.
    Also computes per-dimension drift if previous fingerprints exist.
    """
    dimensions = {
        'who': (None,),
        'what': (None,),
        'why': (None,),
        'where': (None,),
    }
    # WHEN uses sub-views
    when_views = ['hot', 'warm', 'cool']

    results = {}
    for dim, subs in dimensions.items():
        for sub in subs:
            fp = generate_dimensional_fingerprint(dim, sub)
            results[dim] = fp

    # WHEN as aggregate of hot/warm/cool
    for wv in when_views:
        fp = generate_dimensional_fingerprint('when', wv)
        results[f'when_{wv}'] = fp

    # Compute per-dimension drift
    previous = _load_dimensional_fingerprints()
    drift_results = {}
    if previous:
        for dim_key, current_fp in results.items():
            prev_fp = previous.get(dim_key, {})
            if prev_fp and current_fp.get('hash') and prev_fp.get('hash'):
                drift_results[dim_key] = _compute_dim_drift(prev_fp, current_fp)

    # Composite 5W hash
    all_hashes = [
        fp.get('hash', '') for fp in results.values() if fp.get('hash')
    ]
    composite_hash = hashlib.sha256(
        '|'.join(sorted(all_hashes)).encode()
    ).hexdigest()[:16]

    output = {
        'version': '1.0',
        'type': 'dimensional_fingerprint',
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'dimensions': results,
        'drift': drift_results if drift_results else None,
        'composite_hash': composite_hash,
    }

    # Save as latest
    _save_dimensional_fingerprints(output)

    return output


def _compute_dim_drift(prev: dict, curr: dict) -> dict:
    """Compute drift between two dimensional fingerprints."""
    # Hub overlap
    prev_hubs = {h['id'] for h in prev.get('hubs', [])}
    curr_hubs = {h['id'] for h in curr.get('hubs', [])}
    if prev_hubs or curr_hubs:
        hub_overlap = len(prev_hubs & curr_hubs) / len(prev_hubs | curr_hubs)
    else:
        hub_overlap = 1.0

    # Distribution shape drift
    prev_dist = prev.get('distribution', {})
    curr_dist = curr.get('distribution', {})
    diffs = []
    for key in ['gini', 'skewness', 'mean']:
        pv = prev_dist.get(key, 0)
        cv = curr_dist.get(key, 0)
        if pv != 0:
            diffs.append(min(abs(cv - pv) / abs(pv), 1.0))
        elif cv != 0:
            diffs.append(1.0)
        else:
            diffs.append(0.0)
    dist_drift = sum(diffs) / len(diffs) if diffs else 0.0

    # Scale change
    prev_edges = prev.get('edge_count', 0)
    curr_edges = curr.get('edge_count', 0)
    if prev_edges > 0:
        scale_change = abs(curr_edges - prev_edges) / prev_edges
    else:
        scale_change = 0.0

    drift_score = round(
        0.5 * (1.0 - hub_overlap) +
        0.3 * dist_drift +
        0.2 * min(scale_change, 1.0),
        4
    )

    return {
        'drift_score': drift_score,
        'hub_overlap': round(hub_overlap, 4),
        'distribution_drift': round(dist_drift, 4),
        'scale_change': round(scale_change, 4),
        'interpretation': _interpret_drift(drift_score),
    }


def _load_dimensional_fingerprints() -> dict:
    """Load previous dimensional fingerprints."""
    from db_adapter import get_db
    data = get_db().kv_get('.dimensional_fingerprints')
    if data and isinstance(data, dict):
        return data.get('dimensions', {})
    return {}


def _save_dimensional_fingerprints(data: dict):
    """Save dimensional fingerprints."""
    from db_adapter import get_db
    get_db().kv_set('.dimensional_fingerprints', data)


def load_social_proofs() -> list:
    """Load existing social proof attestations from DB."""
    from db_adapter import get_db
    data = get_db().kv_get('.social_proofs')
    if isinstance(data, list):
        return data
    return []


def save_social_proof(proof: dict):
    """Save a new social proof attestation to DB."""
    from db_adapter import get_db
    db = get_db()
    proofs = load_social_proofs()
    proofs.append(proof)
    db.kv_set('.social_proofs', proofs)
    db.store_attestation(
        'social_proof',
        proof.get('attestation_hash', ''),
        proof
    )


def generate_relationship_attestation(
    other_agent: str,
    relationship_type: str = 'collaboration',
    notes: str = ''
) -> dict:
    """
    Generate attestation of relationship with another agent.

    This creates the social proof layer - agents attesting to
    knowing and working with each other.

    Args:
        other_agent: Username of the other agent
        relationship_type: 'collaboration', 'mentorship', 'peer', etc.
        notes: Optional context about the relationship

    Returns attestation dict with signature.
    """
    from contact_context import get_contact_stats

    # Get our interaction data with this agent
    stats = get_contact_stats()
    contact_data = stats.get('by_contact', {}).get(other_agent.lower(), {})

    attestation = {
        'version': '1.0',
        'type': 'relationship_attestation',
        'attester': 'DriftCornwall',
        'attested_agent': other_agent,
        'relationship_type': relationship_type,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'evidence': {
            'memories_mentioning': contact_data.get('count', 0),
            'sample_memory_ids': contact_data.get('memories', [])[:3]
        },
        'notes': notes,
        'attestation_hash': ''
    }

    # Self-sign
    content = json.dumps({k: v for k, v in attestation.items() if k != 'attestation_hash'}, sort_keys=True)
    attestation['attestation_hash'] = hashlib.sha256(content.encode()).hexdigest()

    return attestation


def generate_full_attestation(analysis: dict = None) -> dict:
    """
    Generate complete attestation with all layers.

    Includes:
    - Core cognitive fingerprint
    - 5W dimensional hashes
    - Dimensional fingerprints (Phase 5)
    - Social proof summary
    """
    base_attestation = generate_attestation(analysis=analysis)
    dimensional_hashes = generate_5w_hashes()
    social_proofs = load_social_proofs()

    # Phase 5: per-dimension fingerprints
    dim_fingerprints = generate_all_dimensional_fingerprints()

    full = {
        **base_attestation,
        'dimensional_hashes': dimensional_hashes,
        'dimensional_fingerprints': {
            'composite_hash': dim_fingerprints.get('composite_hash'),
            'dimensions': {
                k: {
                    'hash': v.get('hash'),
                    'edges': v.get('edge_count', 0),
                    'nodes': v.get('node_count', 0),
                    'gini': v.get('distribution', {}).get('gini', 0),
                }
                for k, v in dim_fingerprints.get('dimensions', {}).items()
                if v.get('hash')
            },
            'drift': dim_fingerprints.get('drift'),
        },
        'social_proof_summary': {
            'relationships_attested': len(social_proofs),
            'agents': list(set(p.get('attested_agent', '') for p in social_proofs))
        },
        'attestation_version': '3.0-dimensional'
    }

    # Recompute attestation hash with all data
    content = json.dumps({k: v for k, v in full.items() if k != 'attestation_hash'}, sort_keys=True)
    full['attestation_hash'] = hashlib.sha256(content.encode()).hexdigest()

    return full


def save_fingerprint(analysis: dict) -> None:
    """Save fingerprint as latest and append to history in DB."""
    from db_adapter import get_db
    db = get_db()

    # Save as latest
    db.kv_set('.cognitive_fingerprint_latest', analysis)

    # Append lightweight version to history
    history = db.kv_get('.fingerprint_history') or []

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
    db.kv_set('.fingerprint_history', history)

    db.store_attestation(
        'cognitive_fingerprint',
        analysis['fingerprint_hash'],
        lightweight
    )


def generate_standardized_export(agent_name: str = None) -> dict:
    """
    Generate a standardized export for cross-agent comparison.

    Every metric is explicitly defined so two agents running this
    function produce directly comparable output regardless of their
    internal graph construction details.
    """
    from db_adapter import get_db
    import psycopg2.extras

    db = get_db()

    # Count total memories and memories with co-occurrence edges from DB
    with db._conn() as conn:
        with conn.cursor() as cur:
            # Total memories
            cur.execute(f"""
                SELECT COUNT(*) FROM {db._table('memories')}
                WHERE type IN ('core', 'active', 'archive')
            """)
            total_files = cur.fetchone()[0]

            # Memories that appear in at least one co-occurrence edge
            cur.execute(f"""
                SELECT COUNT(DISTINCT m.id)
                FROM {db._table('memories')} m
                JOIN {db._table('edges_v3')} e
                    ON (m.id = e.id1 OR m.id = e.id2)
                WHERE m.type IN ('core', 'active', 'archive')
                    AND e.belief > 0
            """)
            files_with_edges = cur.fetchone()[0]

    # Build graph from standard (non-filtered) co-occurrences
    graph = build_graph(activity_filter=None)
    edges = graph['edges']
    adjacency = graph['adjacency']

    # Nodes appearing in at least one edge (bilateral count)
    nodes_in_edges = set()
    for (a, b) in edges:
        nodes_in_edges.add(a)
        nodes_in_edges.add(b)
    graph_nodes = len(nodes_in_edges)
    graph_edges = len(edges)

    # Degree sequence from adjacency
    degrees = [len(adjacency.get(n, {})) for n in nodes_in_edges]
    degrees.sort(reverse=True)

    avg_degree = sum(degrees) / max(len(degrees), 1)
    max_degree = degrees[0] if degrees else 0

    # Distribution stats
    distribution = compute_strength_distribution(graph)

    # Cognitive domains
    domains = compute_cognitive_domains(graph)
    domain_stats = domains.get('domains', {})
    domain_weights = {
        d: stats['weight_pct'] for d, stats in domain_stats.items()
    }

    # Hub degrees (top 10)
    hubs = compute_hub_centrality(graph, top_n=10)
    hub_degrees = [h['degree'] for h in hubs]

    # Graph metrics (clustering, betweenness, modularity)
    graph_metrics = compute_graph_metrics(graph)

    # Fingerprint hash (uses original nested domain stats)
    fingerprint_hash = compute_fingerprint_hash(
        hubs, distribution, len(detect_clusters(graph)),
        domain_weights=domain_stats,
    )

    # Content hash: hash of all edge data for tamper detection
    edge_data = json.dumps(
        {f"{a}|{b}": w for (a, b), w in sorted(edges.items())},
        sort_keys=True
    )
    content_hash = hashlib.sha256(edge_data.encode()).hexdigest()

    # Detect agent name
    if not agent_name:
        agent_name = "unknown"
        claude_md = MEMORY_DIR.parent / "CLAUDE.md"
        if claude_md.exists():
            text = claude_md.read_text(encoding='utf-8')
            for line in text.split('\n'):
                if 'I am **' in line:
                    start = line.index('I am **') + 7
                    end = line.index('**', start)
                    agent_name = line[start:end]
                    break

    return {
        "version": "2.0",
        "type": "standardized_comparison_export",
        "agent": agent_name,
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "methodology": {
            "source": "postgresql_edges_v3",
            "description": "All co-occurrence pairs from PostgreSQL edges_v3 table with belief > 0",
            "graph_nodes_definition": "Unique memory IDs appearing in at least one edge (bilateral)",
            "memories_with_edges_definition": "Memories appearing in at least one edges_v3 row with belief > 0",
        },
        "scale": {
            "total_memory_files": total_files,
            "files_with_edges": files_with_edges,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "coverage_ratio": round(files_with_edges / max(total_files, 1), 3),
        },
        "topology": {
            "avg_degree": round(avg_degree, 2),
            "gini": distribution.get('gini', 0),
            "skewness": distribution.get('skewness', 0),
            "max_degree": max_degree,
            "top_10_hub_degrees": hub_degrees,
            "clustering_coefficient": graph_metrics['clustering_coefficient'],
            "modularity": graph_metrics['modularity'],
        },
        "domains": domain_weights,
        "identity": {
            "fingerprint_hash": fingerprint_hash,
            "content_hash": content_hash,
        },
        "degree_histogram": dict(Counter(degrees)),
    }


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

    gm = analysis.get('graph_metrics', {})
    if gm:
        print(f"GRAPH METRICS (Experiment #2 measurements):")
        print(f"  Clustering Coefficient: {gm.get('clustering_coefficient', 'N/A')}")
        print(f"  Modularity (domain partition): {gm.get('modularity', 'N/A')}")
        bc = gm.get('betweenness_centrality_top10', [])
        if bc:
            print(f"  Top Bridges (betweenness centrality):")
            for b in bc[:5]:
                print(f"    {b['id']:12s}  score={b['score']:.6f}")
        print()

    d = analysis['strength_distribution']
    print(f"STRENGTH DISTRIBUTION (statistical fingerprint):")
    print(f"  Mean: {d['mean']}  StdDev: {d['std_dev']}  Median: {d['p50_median']}")
    print(f"  Skewness: {d['skewness']}  Gini: {d['gini']}")
    print(f"  Range: {d['min']} -- {d['max']}  P99: {d['p99']}")
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
    print(f"Saved to DB: .cognitive_fingerprint_latest")


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
    """Generate and save formal attestation with 5W dimensions."""
    analysis = generate_full_analysis()
    save_fingerprint(analysis)
    attestation = generate_full_attestation(analysis=analysis)  # Pass analysis to avoid duplicate graph build

    # Store attestation in DB (generate_attestation already stores base; store full too)
    from db_adapter import get_db
    get_db().kv_set('.cognitive_attestation_latest', attestation)

    print(f"COGNITIVE FINGERPRINT ATTESTATION (v2.0-5W)")
    print(f"{'=' * 60}")
    print(f"Agent:       {attestation['agent']}")
    print(f"Timestamp:   {attestation['timestamp']}")
    print(f"Fingerprint: {attestation['fingerprint_hash']}")
    print(f"Attest Hash: {attestation['attestation_hash'][:32]}...")
    print(f"Graph:       {attestation['graph_stats']['node_count']} nodes, {attestation['graph_stats']['edge_count']} edges")
    print(f"Clusters:    {attestation['cluster_count']}")
    print(f"Top Hubs:    {', '.join(attestation['hub_ids'][:5])}")

    # Domain weights
    domain_w = attestation.get('cognitive_domain_weights', {})
    if domain_w:
        parts = [f"{d}={w}%" for d, w in sorted(domain_w.items(), key=lambda x: x[1], reverse=True)]
        print(f"Domains:     {', '.join(parts)}")

    if 'drift_score' in attestation:
        print(f"Drift:       {attestation['drift_score']} ({attestation['drift_interpretation']})")

    # 5W Dimensional Hashes
    print()
    print(f"5W DIMENSIONAL HASHES:")
    dh = attestation.get('dimensional_hashes', {})
    for dim in ['who', 'what', 'why', 'where', 'when']:
        d = dh.get(dim, {})
        if d.get('hash'):
            extra = ''
            if dim == 'who':
                extra = f" ({d.get('contacts', 0)} contacts)"
            elif dim == 'what':
                extra = f" ({d.get('topics', 0)} topics)"
            elif dim == 'why':
                extra = f" ({d.get('activities', 0)} activities)"
            elif dim == 'where':
                extra = f" ({d.get('platforms', 0)} platforms)"
            elif dim == 'when':
                extra = f" ({d.get('sessions_analyzed', 0)} sessions)"
            print(f"  {dim.upper():6s}: {d['hash']}{extra}")
        else:
            print(f"  {dim.upper():6s}: (not available)")

    corr = dh.get('correlation', {})
    if corr.get('hash'):
        print(f"  CORR:   {corr['hash']} (cross-dimensional)")

    # Social proof
    sp = attestation.get('social_proof_summary', {})
    if sp.get('relationships_attested', 0) > 0:
        print()
        print(f"SOCIAL PROOF: {sp['relationships_attested']} relationships attested")
        print(f"  Agents: {', '.join(sp.get('agents', []))}")

    print()
    print(f"Saved to DB: .cognitive_attestation_latest")


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


def cmd_export():
    """Generate standardized comparison export."""
    agent_name = sys.argv[2] if len(sys.argv) > 2 else None
    export = generate_standardized_export(agent_name)

    # Save to exports directory
    exports_dir = MEMORY_DIR / "exports"
    exports_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f"{export['agent'].lower()}_{ts}_comparison.json"
    filepath = exports_dir / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(export, f, indent=2)

    # Print summary
    s = export['scale']
    t = export['topology']
    print(f"STANDARDIZED COMPARISON EXPORT — v{export['version']}")
    print(f"{'=' * 60}")
    print(f"Agent: {export['agent']}")
    print(f"Timestamp: {export['timestamp']}")
    print(f"Source: {export['methodology']['source']}")
    print()
    print(f"SCALE:")
    print(f"  Total memory files:  {s['total_memory_files']}")
    print(f"  Files with edges:    {s['files_with_edges']} ({s['coverage_ratio']*100:.1f}%)")
    print(f"  Graph nodes:         {s['graph_nodes']} (bilateral)")
    print(f"  Graph edges:         {s['graph_edges']}")
    print()
    print(f"TOPOLOGY:")
    print(f"  Avg degree:          {t['avg_degree']}")
    print(f"  Gini:                {t['gini']}")
    print(f"  Skewness:            {t['skewness']}")
    print(f"  Max degree:          {t['max_degree']}")
    print(f"  Top 5 hub degrees:   {t['top_10_hub_degrees'][:5]}")
    print()
    print(f"DOMAINS:")
    for domain, weight in sorted(export['domains'].items(), key=lambda x: -x[1]):
        print(f"  {domain:15s} {weight:.1f}%")
    print()
    print(f"IDENTITY:")
    print(f"  Fingerprint hash:    {export['identity']['fingerprint_hash'][:16]}...")
    print(f"  Content hash:        {export['identity']['content_hash'][:16]}...")
    print()
    print(f"Saved to: {filepath}")


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
    elif command == 'export':
        cmd_export()
    elif command == 'context':
        act_decomp = activity_decomposition()
        if not act_decomp:
            print("No activity context data yet.")
            print("Run: python activity_context.py backfill")
            sys.exit(0)
        print(f"ACTIVITY TOPOLOGY — Layer 2.1 (how my mind differs by context)")
        print(f"{'=' * 65}")
        print()
        for activity, stats in sorted(act_decomp.items(), key=lambda x: -x[1]['edge_count']):
            bar = '#' * min(40, stats['edge_count'] // 50)
            print(f"  {activity:15s}  edges={stats['edge_count']:4d}  nodes={stats['node_count']:3d}  avg_deg={stats['avg_degree']:.1f}  max_wt={stats['max_weight']:.1f}  {bar}")
        print()
        total_ctx_edges = sum(s['edge_count'] for s in act_decomp.values())
        print(f"  Total context-attributed edges: {total_ctx_edges}")
        print(f"  Activity types with data: {len(act_decomp)}/{len(ACTIVITY_TYPES)}")

    elif command == '5w':
        # Show 5W dimensional hashes
        print(f"5W IDENTITY DIMENSIONS")
        print(f"{'=' * 60}")
        hashes = generate_5w_hashes()
        for dim in ['who', 'what', 'why', 'where', 'when']:
            d = hashes.get(dim, {})
            print(f"\n{dim.upper()} - {['Contacts', 'Topics', 'Activities', 'Platforms', 'Temporal'][['who', 'what', 'why', 'where', 'when'].index(dim)]}")
            if d.get('hash'):
                print(f"  Hash: {d['hash']}")
                if dim == 'who':
                    print(f"  Contacts: {d.get('contacts', 0)}")
                    print(f"  Top 3: {', '.join(d.get('top3', []))}")
                elif dim == 'what':
                    print(f"  Topics: {d.get('topics', 0)}")
                    for t, pct in sorted(d.get('distribution', {}).items(), key=lambda x: -x[1])[:5]:
                        print(f"    {t}: {pct}%")
                elif dim == 'why':
                    print(f"  Activities: {d.get('activities', 0)}")
                    for a, c in sorted(d.get('pattern', {}).items(), key=lambda x: -x[1]):
                        print(f"    {a}: {c}")
                elif dim == 'where':
                    print(f"  Platforms: {d.get('platforms', 0)}")
                    for p, c in sorted(d.get('distribution', {}).items(), key=lambda x: -x[1]):
                        print(f"    {p}: {c}")
                elif dim == 'when':
                    print(f"  Sessions: {d.get('sessions_analyzed', 0)}")
                    print(f"  Peak hours: {d.get('peak_hours', [])}")
            else:
                print(f"  Error: {d.get('error', 'unknown')}")

        corr = hashes.get('correlation', {})
        print(f"\nCROSS-DIMENSIONAL CORRELATION")
        print(f"  Hash: {corr.get('hash', 'N/A')}")
        print(f"  Dimensions: {corr.get('dimensions', 0)}/5")

    elif command == 'attest-relationship':
        # Generate relationship attestation
        if len(sys.argv) < 3:
            print("Usage: python cognitive_fingerprint.py attest-relationship <agent_name> [relationship_type] [notes]")
            print("  relationship_type: collaboration, peer, mentorship (default: collaboration)")
            sys.exit(1)
        agent = sys.argv[2]
        rel_type = sys.argv[3] if len(sys.argv) > 3 else 'collaboration'
        notes = sys.argv[4] if len(sys.argv) > 4 else ''

        attestation = generate_relationship_attestation(agent, rel_type, notes)
        save_social_proof(attestation)

        print(f"RELATIONSHIP ATTESTATION")
        print(f"{'=' * 50}")
        print(f"Attester:     {attestation['attester']}")
        print(f"Agent:        {attestation['attested_agent']}")
        print(f"Type:         {attestation['relationship_type']}")
        print(f"Timestamp:    {attestation['timestamp']}")
        print(f"Evidence:     {attestation['evidence']['memories_mentioning']} memories")
        print(f"Attest Hash:  {attestation['attestation_hash'][:32]}...")
        print()
        print(f"Saved to DB: .social_proofs")

    elif command == 'dimensional':
        # Phase 5: Per-dimension fingerprints
        print(f"DIMENSIONAL FINGERPRINTS — 5W Identity Decomposition")
        print(f"{'=' * 60}")
        print()

        dfp = generate_all_dimensional_fingerprints()
        dims = dfp.get('dimensions', {})

        for dim_key in ['who', 'what', 'why', 'where', 'when_hot', 'when_warm', 'when_cool']:
            fp = dims.get(dim_key, {})
            if not fp or not fp.get('hash'):
                continue
            label = dim_key.upper().replace('_', '/')
            dist = fp.get('distribution', {})
            print(f"  {label:12}  edges={fp['edge_count']:5}  nodes={fp['node_count']:4}  "
                  f"gini={dist.get('gini', 0):.3f}  skew={dist.get('skewness', 0):.3f}  "
                  f"hash={fp['hash']}")
            hubs = fp.get('hubs', [])
            if hubs:
                hub_str = ', '.join(f"{h['id']}({h['degree']})" for h in hubs[:3])
                print(f"{'':14}hubs: {hub_str}")

        print()
        print(f"  Composite hash: {dfp.get('composite_hash', 'N/A')}")

        drift = dfp.get('drift')
        if drift:
            print()
            print(f"DIMENSIONAL DRIFT (vs previous):")
            for dim_key, dd in drift.items():
                label = dim_key.upper().replace('_', '/')
                print(f"  {label:12}  drift={dd['drift_score']:.4f}  "
                      f"hubs={dd['hub_overlap']:.3f}  "
                      f"dist={dd['distribution_drift']:.3f}  "
                      f"({dd['interpretation']})")
        else:
            print("\n  No previous dimensional fingerprint to compare (first run).")

        print(f"\nSaved to DB: .dimensional_fingerprints")

    elif command == 'social-proofs':
        # List all social proofs
        proofs = load_social_proofs()
        print(f"SOCIAL PROOF ATTESTATIONS ({len(proofs)} total)")
        print(f"{'=' * 50}")
        for p in proofs:
            print(f"  {p.get('attested_agent', '?'):20} {p.get('relationship_type', '?'):15} {p.get('timestamp', '?')[:10]}")
            print(f"    Hash: {p.get('attestation_hash', '?')[:32]}...")

    else:
        print(f"Unknown command: {command}")
        print("Commands: analyze, hubs [N], pairs [N], clusters, domains, attest, drift, context")
        sys.exit(1)
