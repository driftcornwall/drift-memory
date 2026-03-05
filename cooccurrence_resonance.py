#!/usr/bin/env python3
"""
Co-occurrence Resonance Detection — Tesla2 Proposal #3.

Finds self-reinforcing cycles (behavioral loops) in the co-occurrence graph.
Unlike the resonance scanner (#4) which probes embedding similarity,
this module analyzes the actual co-occurrence graph (edges_v3) to find
memories that chase each other through lived experience.

Key concepts:
- CYCLE: A closed loop in the co-occurrence graph (triangles = 3-cycles)
- CYCLE STRENGTH: Geometric mean of edge beliefs in the cycle
- CYCLING FREQUENCY: Distinct sessions where cycle members co-occurred
- CYCLE COUPLING: Two cycles sharing a node (bridge between thought patterns)

Usage:
    python cooccurrence_resonance.py scan        # Full cycle detection
    python cooccurrence_resonance.py cycles      # Show top cycles
    python cooccurrence_resonance.py coupling     # Show coupled cycles
    python cooccurrence_resonance.py history      # Track evolution
    python cooccurrence_resonance.py summary      # Compact for priming
"""

import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

from db_adapter import get_db

# --- Configuration ---
MIN_BELIEF_THRESHOLD = 1.0   # Ignore edges weaker than this
MAX_TRIANGLES = 50           # Keep top N strongest cycles
MAX_DEGREE_FOR_ENUM = 200    # Skip nodes with degree > this (hub dampening)

# --- KV Keys ---
RESONANCE_KV_KEY = '.cooccurrence_resonance_latest'
RESONANCE_HISTORY_KEY = '.cooccurrence_resonance_history'


# --- Core algorithms ---

def find_triangles(
    adj: dict[str, set],
    beliefs: dict[tuple, float],
    min_belief: float = MIN_BELIEF_THRESHOLD,
    max_triangles: int = MAX_TRIANGLES,
) -> list[dict]:
    """Find triangles in the co-occurrence graph, scored by strength.

    Algorithm: For each edge (A,B), find shared neighbors C.
    O(E * avg_degree) -- feasible for our graph size.

    Args:
        adj: {node_id: set(neighbor_ids)}
        beliefs: {(sorted_id1, sorted_id2): belief_value}
        min_belief: Minimum edge belief to consider
        max_triangles: Return at most this many triangles

    Returns:
        List of dicts sorted by strength descending:
        [{'nodes': [A,B,C], 'strength': float, 'edges': [(A,B,belief), ...]}]
    """
    # Filter edges by min_belief
    filtered_adj = defaultdict(set)
    for key, belief in beliefs.items():
        if belief >= min_belief:
            a, b = key
            filtered_adj[a].add(b)
            filtered_adj[b].add(a)

    seen = set()
    triangles = []

    for a in filtered_adj:
        if len(filtered_adj[a]) > MAX_DEGREE_FOR_ENUM:
            continue
        for b in filtered_adj[a]:
            if b <= a:
                continue
            if len(filtered_adj[b]) > MAX_DEGREE_FOR_ENUM:
                continue
            # Shared neighbors of a and b
            shared = filtered_adj[a] & filtered_adj[b]
            for c in shared:
                if c <= b:
                    continue
                tri_key = tuple(sorted([a, b, c]))
                if tri_key in seen:
                    continue
                seen.add(tri_key)

                # Get edge beliefs
                ab = beliefs.get(tuple(sorted([a, b])), 0)
                bc = beliefs.get(tuple(sorted([b, c])), 0)
                ac = beliefs.get(tuple(sorted([a, c])), 0)

                # Geometric mean of edge beliefs
                strength = (ab * bc * ac) ** (1 / 3)

                triangles.append({
                    'nodes': list(tri_key),
                    'strength': round(strength, 4),
                    'edges': [
                        (a, b, round(ab, 2)),
                        (b, c, round(bc, 2)),
                        (a, c, round(ac, 2)),
                    ],
                })

    # Sort by strength descending, keep top N
    triangles.sort(key=lambda t: t['strength'], reverse=True)
    return triangles[:max_triangles]


def compute_cycling_frequency(
    triangle: dict,
    edge_sessions: dict[tuple, list[str]],
) -> int:
    """Count sessions where >=2 edges of the triangle were observed.

    A high cycling frequency means the memories in this triangle
    consistently co-occur together across sessions -- a stable attractor.

    Args:
        triangle: {'nodes': [A, B, C]}
        edge_sessions: {(sorted_id1, sorted_id2): [session_id, ...]}

    Returns:
        Number of sessions where at least 2 edges were active
    """
    nodes = triangle['nodes']
    edges = [
        tuple(sorted([nodes[0], nodes[1]])),
        tuple(sorted([nodes[1], nodes[2]])),
        tuple(sorted([nodes[0], nodes[2]])),
    ]

    # Count sessions per edge
    session_counts = defaultdict(int)
    for edge in edges:
        for session_id in edge_sessions.get(edge, []):
            session_counts[session_id] += 1

    # Sessions where >=2 edges were observed
    return sum(1 for count in session_counts.values() if count >= 2)


def detect_coupling(triangles: list[dict]) -> list[dict]:
    """Find coupled cycles (triangles that share nodes).

    Shared nodes are bridges between thought patterns -- the connective
    tissue of identity.

    Args:
        triangles: List of triangle dicts with 'nodes' and 'strength'

    Returns:
        List of coupling dicts sorted by coupling_strength descending
    """
    if len(triangles) < 2:
        return []

    couplings = []
    for i in range(len(triangles)):
        nodes_i = set(triangles[i]['nodes'])
        for j in range(i + 1, len(triangles)):
            nodes_j = set(triangles[j]['nodes'])
            shared = nodes_i & nodes_j
            if shared:
                coupling_strength = min(
                    triangles[i].get('strength', 0),
                    triangles[j].get('strength', 0),
                )
                couplings.append({
                    'cycle_indices': (i, j),
                    'shared_nodes': sorted(shared),
                    'coupling_strength': round(coupling_strength, 4),
                })

    couplings.sort(key=lambda c: c['coupling_strength'], reverse=True)
    return couplings


# --- DB Integration ---

def _build_graph(min_belief: float = MIN_BELIEF_THRESHOLD) -> tuple[dict, dict]:
    """Load edges_v3 from DB, build adjacency list and belief dict."""
    db = get_db()
    raw_edges = db.get_all_edges()
    adj = defaultdict(set)
    beliefs = {}

    for key_str, edge_data in raw_edges.items():
        parts = key_str.split('|')
        if len(parts) != 2:
            continue
        belief = edge_data.get('belief', 0)
        if belief < min_belief:
            continue
        a, b = parts
        adj[a].add(b)
        adj[b].add(a)
        beliefs[tuple(sorted([a, b]))] = belief

    return dict(adj), beliefs


def _load_edge_sessions() -> dict[tuple, list[str]]:
    """Load session IDs per edge from edge_observations."""
    db = get_db()
    edge_sessions = defaultdict(list)
    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT edge_id1, edge_id2, session_id
                FROM {db._table('edge_observations')}
                WHERE session_id IS NOT NULL
                ORDER BY observed_at ASC
            """)
            for row in cur.fetchall():
                key = tuple(sorted([row['edge_id1'], row['edge_id2']]))
                sid = row['session_id']
                if sid and sid not in edge_sessions[key]:
                    edge_sessions[key].append(sid)
    return dict(edge_sessions)


def _get_content_preview(memory_id: str) -> str:
    """Get a content preview for a memory ID."""
    try:
        db = get_db()
        import psycopg2.extras
        with db._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT LEFT(content, 80) as preview
                    FROM {db._table('memories')}
                    WHERE id = %s
                """, (memory_id,))
                row = cur.fetchone()
                return row['preview'] if row else '?'
    except Exception:
        return '?'


def run_scan(verbose: bool = True) -> dict:
    """Run a full co-occurrence resonance scan.

    1. Build graph from edges_v3
    2. Find triangles
    3. Score cycling frequency from edge_observations
    4. Detect coupling between triangles
    5. Store results to DB KV
    """
    start = time.time()
    db = get_db()

    # Build graph
    adj, beliefs = _build_graph()
    if verbose:
        print(f"Graph: {len(adj)} nodes, {len(beliefs)} edges (belief >= {MIN_BELIEF_THRESHOLD})")

    # Find triangles
    triangles = find_triangles(adj, beliefs)
    if verbose:
        print(f"Triangles found: {len(triangles)}")

    # Score cycling frequency
    edge_sessions = _load_edge_sessions()
    for tri in triangles:
        tri['cycling_freq'] = compute_cycling_frequency(tri, edge_sessions)

    # Composite score: strength * log(1 + cycling_freq + 1)
    for tri in triangles:
        tri['composite_score'] = round(
            tri['strength'] * math.log(1 + tri['cycling_freq'] + 1), 4
        )
    triangles.sort(key=lambda t: t['composite_score'], reverse=True)

    # Detect coupling
    coupling = detect_coupling(triangles)

    # Find bridge nodes (nodes appearing in multiple triangles)
    node_triangle_count = defaultdict(int)
    for tri in triangles:
        for node in tri['nodes']:
            node_triangle_count[node] += 1
    bridge_nodes = [
        {'node': n, 'triangle_count': c}
        for n, c in sorted(node_triangle_count.items(), key=lambda x: -x[1])
        if c >= 2
    ][:20]

    elapsed = time.time() - start

    result = {
        'scanned_at': datetime.now(timezone.utc).isoformat(),
        'elapsed_s': round(elapsed, 1),
        'stats': {
            'nodes': len(adj),
            'edges': len(beliefs),
            'triangles_found': len(triangles),
            'coupled_pairs': len(coupling),
            'bridge_nodes': len(bridge_nodes),
        },
        'triangles': triangles,
        'coupling': coupling[:20],
        'bridge_nodes': bridge_nodes,
    }

    # Store to KV
    db.kv_set(RESONANCE_KV_KEY, result)
    _append_history(db, result)

    # Fire cognitive event
    try:
        from cognitive_state import process_event
        process_event('cycle_detected', {'triangles': len(triangles), 'bridges': len(bridge_nodes)})
    except Exception:
        pass

    if verbose:
        _print_results(result)

    return result


def _append_history(db, result: dict):
    """Append scan summary to history for tracking evolution."""
    raw = db.kv_get(RESONANCE_HISTORY_KEY)
    history = json.loads(raw) if isinstance(raw, str) else (raw or [])
    if not isinstance(history, list):
        history = []

    summary = {
        'scanned_at': result['scanned_at'],
        'stats': result['stats'],
        'top_triangle_strength': result['triangles'][0]['strength'] if result['triangles'] else 0,
        'top_composite': result['triangles'][0]['composite_score'] if result['triangles'] else 0,
    }
    history.append(summary)
    history = history[-50:]
    db.kv_set(RESONANCE_HISTORY_KEY, history)


def _print_results(result: dict):
    """Pretty-print scan results."""
    stats = result['stats']
    print(f"\n=== Co-occurrence Resonance Scan ===")
    print(f"  Graph: {stats['nodes']} nodes, {stats['edges']} edges")
    print(f"  Triangles: {stats['triangles_found']}")
    print(f"  Coupled pairs: {stats['coupled_pairs']}")
    print(f"  Bridge nodes: {stats['bridge_nodes']}")
    print(f"  Elapsed: {result['elapsed_s']}s")

    if result['triangles']:
        print(f"\n  Top cycles (by composite score):")
        for i, tri in enumerate(result['triangles'][:10]):
            nodes_str = ', '.join(n[:8] for n in tri['nodes'])
            print(f"    {i+1}. [{nodes_str}]")
            print(f"       strength={tri['strength']:.2f}  freq={tri['cycling_freq']}  "
                  f"composite={tri['composite_score']:.2f}")

    if result['bridge_nodes']:
        print(f"\n  Bridge nodes (in multiple cycles):")
        for bn in result['bridge_nodes'][:10]:
            preview = _get_content_preview(bn['node'])
            print(f"    [{bn['node'][:8]}] in {bn['triangle_count']} cycles: {preview}")

    if result['coupling']:
        print(f"\n  Coupled cycles:")
        for cp in result['coupling'][:5]:
            i, j = cp['cycle_indices']
            shared = ', '.join(n[:8] for n in cp['shared_nodes'])
            print(f"    Cycle {i+1} <-> Cycle {j+1} via [{shared}]  "
                  f"coupling={cp['coupling_strength']:.2f}")


def get_resonance_summary() -> dict:
    """Get compact summary for session priming injection."""
    db = get_db()
    raw = db.kv_get(RESONANCE_KV_KEY)
    if not raw:
        return {'status': 'no_scan'}
    result = json.loads(raw) if isinstance(raw, str) else raw
    return {
        'scanned_at': result.get('scanned_at'),
        'triangles': result['stats']['triangles_found'],
        'bridges': result['stats']['bridge_nodes'],
        'coupled': result['stats']['coupled_pairs'],
        'top_strength': result['triangles'][0]['strength'] if result.get('triangles') else 0,
        'top_composite': result['triangles'][0]['composite_score'] if result.get('triangles') else 0,
    }


# --- CLI ---

def _cli_scan():
    run_scan(verbose=True)


def _cli_cycles():
    db = get_db()
    raw = db.kv_get(RESONANCE_KV_KEY)
    if not raw:
        print("No scan data. Run: python cooccurrence_resonance.py scan")
        return
    result = json.loads(raw) if isinstance(raw, str) else raw
    print(f"Co-occurrence cycles from {result.get('scanned_at', '?')}")
    for i, tri in enumerate(result.get('triangles', [])[:20]):
        nodes_str = ', '.join(n[:8] for n in tri['nodes'])
        print(f"  {i+1}. [{nodes_str}] str={tri['strength']:.2f} freq={tri.get('cycling_freq', '?')} "
              f"comp={tri.get('composite_score', '?')}")
        for node in tri['nodes']:
            preview = _get_content_preview(node)
            print(f"      [{node[:8]}] {preview}")


def _cli_coupling():
    db = get_db()
    raw = db.kv_get(RESONANCE_KV_KEY)
    if not raw:
        print("No scan data. Run: python cooccurrence_resonance.py scan")
        return
    result = json.loads(raw) if isinstance(raw, str) else raw
    coupling = result.get('coupling', [])
    bridges = result.get('bridge_nodes', [])
    print(f"Cycle coupling from {result.get('scanned_at', '?')}")
    print(f"\n  Bridge nodes:")
    for bn in bridges[:15]:
        preview = _get_content_preview(bn['node'])
        print(f"    [{bn['node'][:8]}] in {bn['triangle_count']} cycles: {preview}")
    print(f"\n  Coupled pairs:")
    for cp in coupling[:15]:
        shared = ', '.join(n[:8] for n in cp['shared_nodes'])
        print(f"    Cycle {cp['cycle_indices'][0]+1} <-> {cp['cycle_indices'][1]+1} "
              f"via [{shared}] strength={cp['coupling_strength']:.2f}")


def _cli_history():
    db = get_db()
    raw = db.kv_get(RESONANCE_HISTORY_KEY)
    if not raw:
        print("No scan history.")
        return
    history = json.loads(raw) if isinstance(raw, str) else raw
    print(f"Co-occurrence resonance history ({len(history)} scans)")
    for entry in history[-10:]:
        s = entry.get('stats', {})
        print(f"  {entry['scanned_at'][:19]}: {s.get('triangles_found', 0)} triangles, "
              f"{s.get('bridge_nodes', 0)} bridges, top={entry.get('top_composite', 0):.2f}")


def _cli_summary():
    summary = get_resonance_summary()
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    commands = {
        'scan': _cli_scan,
        'cycles': _cli_cycles,
        'coupling': _cli_coupling,
        'history': _cli_history,
        'summary': _cli_summary,
    }
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'scan'
    if cmd in commands:
        commands[cmd]()
    else:
        print(f"Usage: python cooccurrence_resonance.py [{' | '.join(commands)}]")
