#!/usr/bin/env python3
"""
Context Manager -- 5W Multi-Graph Projection Engine

Phase 1 of Multi-Graph Architecture (RFC: GitHub Issue #19).
Materializes 5 dimensional views from L0 canonical graph (.edges_v3.json).

Dimensions:
  WHO   - contact-weighted (which memories co-fire around the same people)
  WHAT  - topic-weighted (which memories co-fire within the same domain)
  WHY   - activity-weighted (which memories co-fire during the same work)
  WHERE - platform-weighted (which memories co-fire on the same platform)
  WHEN  - temporal windows (recent vs historical co-occurrences)
  BRIDGE - cross-dimensional connections

Design principles:
  - L0 (.edges_v3.json) is the single source of truth
  - Context graphs are materialized views -- rebuilt, never written independently
  - Each W-dimension uses existing context modules for classification

Usage:
    python context_manager.py rebuild [--verbose]  # Full 5W rebuild
    python context_manager.py --json               # Rebuild, JSON output (for hooks)
    python context_manager.py stats [dimension]     # Stats for dimension(s)
    python context_manager.py hubs <dimension>      # Hub nodes in dimension
    python context_manager.py query <dim> <node>    # Neighbors in dimension
    python context_manager.py bridges [dim_a dim_b] # Cross-dimensional bridges

Credit: Joint design with SpindriftMind (Issue #19), 5W framework with Lex
"""

import io
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

if __name__ == '__main__':
    if sys.stdout and hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if sys.stderr and hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

MEMORY_ROOT = Path(__file__).parent
CONTEXT_DIR = MEMORY_ROOT / "context"
EDGES_FILE = MEMORY_ROOT / ".edges_v3.json"
SESSION_HOURS = 4  # approximate hours per session


# --- Graph I/O ---

def _ensure_context_dir():
    CONTEXT_DIR.mkdir(exist_ok=True)


def _save_graph(filename: str, data: dict):
    _ensure_context_dir()
    (CONTEXT_DIR / filename).write_text(json.dumps(data, indent=2), encoding='utf-8')


def load_graph(dimension: str, sub_view: str = None) -> dict:
    """Load a context graph from disk."""
    name = f"{dimension}_{sub_view}.json" if sub_view else f"{dimension}.json"
    path = CONTEXT_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def _load_l0() -> dict:
    if not EDGES_FILE.exists():
        return {}
    return json.loads(EDGES_FILE.read_text(encoding='utf-8'))


def _build_metadata_cache() -> dict:
    """Build memory_id -> metadata lookup from all memory files."""
    from memory_common import ALL_DIRS, parse_memory_file
    cache = {}
    for d in ALL_DIRS:
        if not d.exists():
            continue
        for fp in d.glob("*.md"):
            meta, content = parse_memory_file(fp)
            mid = meta.get('id')
            if mid:
                cache[mid] = meta
    return cache


# --- Graph utilities ---

def _compute_hubs(edges: dict, top_n: int = 10) -> list[str]:
    degree = defaultdict(int)
    for k in edges:
        parts = k.split('|')
        if len(parts) == 2:
            degree[parts[0]] += 1
            degree[parts[1]] += 1
    return [n for n, _ in sorted(degree.items(), key=lambda x: -x[1])[:top_n]]


def _node_set(edges: dict) -> set:
    nodes = set()
    for k in edges:
        parts = k.split('|')
        if len(parts) == 2:
            nodes.update(parts)
    return nodes


def _make_graph(dimension: str, sub_view: Optional[str], edges: dict) -> dict:
    nodes = _node_set(edges)
    n = len(nodes)
    max_e = n * (n - 1) / 2 if n > 1 else 1
    beliefs = [e.get('belief', 0) for e in edges.values()]
    return {
        'meta': {
            'dimension': dimension,
            'sub_view': sub_view,
            'last_rebuilt': datetime.now(timezone.utc).isoformat(),
            'edge_count': len(edges),
            'node_count': n,
        },
        'edges': edges,
        'hubs': _compute_hubs(edges),
        'stats': {
            'avg_belief': round(sum(beliefs) / len(beliefs), 3) if beliefs else 0,
            'density': round(len(edges) / max_e, 6) if max_e else 0,
        },
    }


# --- Projection Functions ---

def _project_who(l0: dict, cache: dict) -> dict:
    """WHO dimension: edges weighted by shared contacts."""
    out = {}
    for k, e in l0.items():
        parts = k.split('|')
        if len(parts) != 2:
            continue
        id1, id2 = parts

        edge_contacts = set(e.get('contact_context', []))
        c1 = set(cache.get(id1, {}).get('contact_context', []))
        c2 = set(cache.get(id2, {}).get('contact_context', []))
        all_c = edge_contacts | c1 | c2

        if not all_c:
            continue

        shared = c1 & c2
        overlap = len(shared) / len(all_c) if all_c else 0
        dw = round(0.3 + 0.7 * overlap, 3)

        out[k] = {
            'belief': round(e.get('belief', 0) * dw, 3),
            'dimension_weight': dw,
            'observation_count': len(e.get('observations', [])),
            'last_observed': e.get('last_updated', ''),
        }
    return out


def _project_what(l0: dict, cache: dict, topic: str = None) -> dict:
    """WHAT dimension: edges weighted by shared topics."""
    out = {}
    for k, e in l0.items():
        parts = k.split('|')
        if len(parts) != 2:
            continue
        id1, id2 = parts

        tc = e.get('topic_context', {})
        edge_shared = set(tc.get('shared', [])) if isinstance(tc, dict) else set()
        edge_union = set(tc.get('union', [])) if isinstance(tc, dict) else set()

        t1 = set(cache.get(id1, {}).get('topic_context', []))
        t2 = set(cache.get(id2, {}).get('topic_context', []))
        all_t = edge_union | t1 | t2
        all_shared = edge_shared | (t1 & t2)

        if topic:
            in1 = topic in t1 or topic in edge_union
            in2 = topic in t2 or topic in edge_union
            if not (in1 or in2):
                continue
            dw = 1.0 if topic in all_shared else 0.5
        else:
            if all_shared:
                dw = len(all_shared) / max(len(all_t), 1)
            elif all_t:
                dw = 0.3
            else:
                continue

        out[k] = {
            'belief': round(e.get('belief', 0) * dw, 3),
            'dimension_weight': round(dw, 3),
            'observation_count': len(e.get('observations', [])),
            'last_observed': e.get('last_updated', ''),
        }
    return out


def _project_why(l0: dict, activity: str = None) -> dict:
    """WHY dimension: edges weighted by activity context."""
    out = {}
    for k, e in l0.items():
        ac = e.get('activity_context', {})
        if not ac:
            continue

        if activity:
            count = ac.get(activity, 0)
            if count <= 0:
                continue
            total = sum(ac.values())
            dw = round(count / total, 3) if total > 0 else 0
        else:
            dw = 1.0

        if dw <= 0:
            continue

        out[k] = {
            'belief': round(e.get('belief', 0) * dw, 3),
            'dimension_weight': dw,
            'observation_count': len(e.get('observations', [])),
            'last_observed': e.get('last_updated', ''),
            'dominant_activity': max(ac, key=ac.get) if ac else None,
        }
    return out


def _project_where(l0: dict, platform: str = None) -> dict:
    """WHERE dimension: edges weighted by platform context."""
    out = {}
    for k, e in l0.items():
        pc = e.get('platform_context', {})
        counts = {p: v for p, v in pc.items() if not p.startswith('_')}
        if not counts:
            continue

        if platform:
            c = counts.get(platform, 0)
            if c <= 0:
                continue
            total = sum(counts.values())
            dw = round(c / total, 3) if total > 0 else 0
        else:
            dw = 1.0

        if dw <= 0:
            continue

        out[k] = {
            'belief': round(e.get('belief', 0) * dw, 3),
            'dimension_weight': dw,
            'observation_count': len(e.get('observations', [])),
            'last_observed': e.get('last_updated', ''),
            'dominant_platform': max(counts, key=counts.get) if counts else None,
        }
    return out


def _project_when(l0: dict, window_sessions: int = 3) -> dict:
    """WHEN dimension: edges filtered by observation recency."""
    now = datetime.now(timezone.utc)
    max_hours = window_sessions * SESSION_HOURS
    out = {}

    for k, e in l0.items():
        recent = []
        for obs in e.get('observations', []):
            try:
                ts = obs.get('observed_at', '')
                t = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                if (now - t).total_seconds() / 3600 <= max_hours:
                    recent.append(obs)
            except (ValueError, TypeError, AttributeError):
                continue

        if not recent:
            continue

        # Recompute belief from recent observations only
        from co_occurrence import aggregate_belief
        out[k] = {
            'belief': round(aggregate_belief(recent), 3),
            'dimension_weight': 1.0,
            'observation_count': len(recent),
            'last_observed': e.get('last_updated', ''),
        }
    return out


def _detect_bridges(w_graphs: dict) -> dict:
    """Find edges appearing in 2+ W-dimension graphs."""
    edge_dims = defaultdict(set)
    for dim, g in w_graphs.items():
        if dim == 'bridges':
            continue
        for ek in g.get('edges', {}):
            edge_dims[ek].add(dim)

    bridges = {}
    total_dims = len(w_graphs)
    for ek, dims in edge_dims.items():
        if len(dims) < 2:
            continue
        beliefs = {}
        for d in dims:
            edge = w_graphs[d].get('edges', {}).get(ek, {})
            beliefs[d] = edge.get('belief', 0)
        bridges[ek] = {
            'dimensions': sorted(dims),
            'dimension_count': len(dims),
            'bridge_score': round(len(dims) / total_dims, 3),
            'beliefs': beliefs,
        }
    return bridges


# --- Main Rebuild ---

def rebuild_all(verbose: bool = False) -> dict:
    """Full 5W projection from L0. Returns summary dict."""
    l0 = _load_l0()
    if not l0:
        return {'graphs_created': 0, 'total_l0_edges': 0, 'bridges': 0}

    cache = _build_metadata_cache()
    _ensure_context_dir()

    from topic_context import TOPIC_DEFINITIONS
    from activity_context import ACTIVITY_TYPES

    # Discover platforms from data
    platforms = set()
    for e in l0.values():
        for p in e.get('platform_context', {}):
            if not p.startswith('_'):
                platforms.add(p)

    created = 0
    w_graphs = {}

    # WHO
    who = _project_who(l0, cache)
    g = _make_graph('who', None, who)
    _save_graph('who.json', g)
    w_graphs['who'] = g
    created += 1
    if verbose:
        print(f"  WHO: {len(who)} edges, {g['meta']['node_count']} nodes")

    # WHAT aggregate + sub-views
    what = _project_what(l0, cache)
    g = _make_graph('what', None, what)
    _save_graph('what.json', g)
    w_graphs['what'] = g
    created += 1
    if verbose:
        print(f"  WHAT: {len(what)} edges, {g['meta']['node_count']} nodes")
    for topic in TOPIC_DEFINITIONS:
        te = _project_what(l0, cache, topic=topic)
        if te:
            tg = _make_graph('what', topic, te)
            _save_graph(f'what_{topic}.json', tg)
            created += 1
            if verbose:
                print(f"    WHAT/{topic}: {len(te)} edges")

    # WHY aggregate + sub-views
    why = _project_why(l0)
    g = _make_graph('why', None, why)
    _save_graph('why.json', g)
    w_graphs['why'] = g
    created += 1
    if verbose:
        print(f"  WHY: {len(why)} edges, {g['meta']['node_count']} nodes")
    for act in ACTIVITY_TYPES:
        ae = _project_why(l0, activity=act)
        if ae:
            ag = _make_graph('why', act, ae)
            _save_graph(f'why_{act}.json', ag)
            created += 1
            if verbose:
                print(f"    WHY/{act}: {len(ae)} edges")

    # WHERE aggregate + sub-views
    where = _project_where(l0)
    g = _make_graph('where', None, where)
    _save_graph('where.json', g)
    w_graphs['where'] = g
    created += 1
    if verbose:
        print(f"  WHERE: {len(where)} edges, {g['meta']['node_count']} nodes")
    for plat in sorted(platforms):
        pe = _project_where(l0, platform=plat)
        if pe:
            pg = _make_graph('where', plat, pe)
            _save_graph(f'where_{plat}.json', pg)
            created += 1
            if verbose:
                print(f"    WHERE/{plat}: {len(pe)} edges")

    # WHEN (hot=3, warm=7, cool=21 sessions)
    for name, sessions in [('hot', 3), ('warm', 7), ('cool', 21)]:
        we = _project_when(l0, window_sessions=sessions)
        wg = _make_graph('when', name, we)
        _save_graph(f'when_{name}.json', wg)
        w_graphs[f'when_{name}'] = wg
        created += 1
        if verbose:
            print(f"  WHEN/{name}: {len(we)} edges")

    # BRIDGES
    bridges = _detect_bridges(w_graphs)
    bg = _make_graph('bridges', None, bridges)
    _save_graph('bridges.json', bg)
    created += 1
    if verbose:
        print(f"  BRIDGES: {len(bridges)} cross-dimensional edges")

    return {
        'graphs_created': created,
        'total_l0_edges': len(l0),
        'bridges': len(bridges),
        'dimensions': {
            'who': len(who), 'what': len(what),
            'why': len(why), 'where': len(where),
        },
        'rebuilt_at': datetime.now(timezone.utc).isoformat(),
    }


# --- Query Functions ---

def get_stats(dimension: str = None) -> dict:
    """Get stats for one or all dimensions."""
    if dimension:
        g = load_graph(dimension)
        if not g:
            return {'error': f'No graph for {dimension}'}
        return {
            'dimension': dimension,
            'edges': g['meta']['edge_count'],
            'nodes': g['meta']['node_count'],
            'hubs': g.get('hubs', [])[:5],
            'stats': g.get('stats', {}),
            'last_rebuilt': g['meta']['last_rebuilt'],
        }

    results = {}
    for dim in ['who', 'what', 'why', 'where']:
        g = load_graph(dim)
        if g:
            results[dim] = {
                'edges': g['meta']['edge_count'],
                'nodes': g['meta']['node_count'],
                'hubs': g.get('hubs', [])[:3],
                'avg_belief': g.get('stats', {}).get('avg_belief', 0),
            }
    for w in ['hot', 'warm', 'cool']:
        g = load_graph('when', w)
        if g:
            results[f'when_{w}'] = {
                'edges': g['meta']['edge_count'],
                'nodes': g['meta']['node_count'],
            }
    g = load_graph('bridges')
    if g:
        results['bridges'] = {
            'edges': g['meta']['edge_count'],
            'nodes': g['meta']['node_count'],
        }
    return results


def query_neighbors(dimension: str, node_id: str,
                    sub_view: str = None, top_n: int = 10) -> list[dict]:
    """Top neighbors for a node in a dimension."""
    g = load_graph(dimension, sub_view)
    if not g:
        return []
    neighbors = []
    for k, ed in g.get('edges', {}).items():
        parts = k.split('|')
        if len(parts) == 2 and node_id in parts:
            other = parts[1] if parts[0] == node_id else parts[0]
            neighbors.append({
                'id': other,
                'belief': ed.get('belief', 0),
                'dimension_weight': ed.get('dimension_weight', 0),
            })
    neighbors.sort(key=lambda x: -x['belief'])
    return neighbors[:top_n]


def discover_bridges(dim_a: str, dim_b: str,
                     sub_a: str = None, sub_b: str = None) -> list[dict]:
    """Find edges appearing in both dimensions, ranked by combined belief."""
    ga = load_graph(dim_a, sub_a)
    gb = load_graph(dim_b, sub_b)
    if not ga or not gb:
        return []
    shared = set(ga.get('edges', {}).keys()) & set(gb.get('edges', {}).keys())
    results = []
    for ek in shared:
        ea = ga['edges'][ek]
        eb = gb['edges'][ek]
        results.append({
            'edge': ek,
            f'belief_{dim_a}': ea.get('belief', 0),
            f'belief_{dim_b}': eb.get('belief', 0),
            'combined': ea.get('belief', 0) + eb.get('belief', 0),
        })
    results.sort(key=lambda x: -x['combined'])
    return results[:20]


def get_session_dimensions() -> dict:
    """
    Collect active W-dimensions from current session state files.
    Used by Phase 2 dimensional decay to know which contexts are active.
    """
    dims = {}

    # WHERE: .session_platforms.json
    pf = MEMORY_ROOT / ".session_platforms.json"
    if pf.exists():
        try:
            dims['where'] = json.loads(pf.read_text(encoding='utf-8')).get('platforms', [])
        except Exception:
            pass

    # WHY: .session_activity.json
    af = MEMORY_ROOT / ".session_activity.json"
    if af.exists():
        try:
            data = json.loads(af.read_text(encoding='utf-8'))
            acts = []
            if data.get('dominant'):
                acts.append(data['dominant'])
            acts.extend(data.get('secondary', []))
            dims['why'] = acts
        except Exception:
            pass

    # WHO: .session_contacts.json
    cf = MEMORY_ROOT / ".session_contacts.json"
    if cf.exists():
        try:
            dims['who'] = json.loads(cf.read_text(encoding='utf-8')).get('contacts', [])
        except Exception:
            pass

    # WHAT: inferred from recalled memories' topic_context
    sf = MEMORY_ROOT / ".session_state.json"
    if sf.exists():
        try:
            state = json.loads(sf.read_text(encoding='utf-8'))
            recalled = state.get('retrieved', [])
            if recalled:
                from memory_common import ALL_DIRS, parse_memory_file
                topics = set()
                for mid in recalled:
                    for d in ALL_DIRS:
                        if not d.exists():
                            continue
                        for fp in d.glob(f"*-{mid}.md"):
                            meta, _ = parse_memory_file(fp)
                            topics.update(meta.get('topic_context', []))
                            break
                dims['what'] = list(topics)
        except Exception:
            pass

    return dims


# --- CLI ---

def main():
    if len(sys.argv) < 2:
        print("Context Manager -- 5W Multi-Graph Projection Engine")
        print()
        print("Usage:")
        print("  python context_manager.py rebuild [--verbose]")
        print("  python context_manager.py --json")
        print("  python context_manager.py stats [dimension]")
        print("  python context_manager.py hubs <dimension> [sub_view]")
        print("  python context_manager.py query <dimension> <node_id> [sub_view]")
        print("  python context_manager.py bridges [dim_a dim_b]")
        print("  python context_manager.py session-dims")
        return

    cmd = sys.argv[1]

    if cmd == '--json':
        summary = rebuild_all()
        print(json.dumps(summary))
        return

    if cmd == 'rebuild':
        verbose = '--verbose' in sys.argv or '-v' in sys.argv
        print("Rebuilding 5W context graphs from L0...")
        s = rebuild_all(verbose=verbose)
        print(f"\nDone: {s['graphs_created']} graphs from {s['total_l0_edges']} L0 edges")
        print(f"  WHO:     {s['dimensions']['who']} edges")
        print(f"  WHAT:    {s['dimensions']['what']} edges")
        print(f"  WHY:     {s['dimensions']['why']} edges")
        print(f"  WHERE:   {s['dimensions']['where']} edges")
        print(f"  BRIDGES: {s['bridges']} cross-dimensional")

    elif cmd == 'stats':
        dim = sys.argv[2] if len(sys.argv) > 2 else None
        stats = get_stats(dim)
        if dim:
            print(f"\n{dim.upper()} dimension:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
        else:
            print("\n5W CONTEXT GRAPH STATS")
            print("=" * 60)
            for dn, ds in stats.items():
                e = ds.get('edges', 0)
                n = ds.get('nodes', 0)
                h = ds.get('hubs', [])
                ab = ds.get('avg_belief', '')
                hs = f"  hubs: {', '.join(h[:3])}" if h else ""
                bs = f"  avg_belief: {ab}" if ab else ""
                print(f"  {dn:12} {e:5} edges, {n:4} nodes{bs}{hs}")

    elif cmd == 'hubs':
        if len(sys.argv) < 3:
            print("Usage: python context_manager.py hubs <dimension> [sub_view]")
            return
        dim = sys.argv[2]
        sub = sys.argv[3] if len(sys.argv) > 3 else None
        g = load_graph(dim, sub)
        if g:
            label = f"{dim}/{sub}" if sub else dim
            print(f"\nTop hubs in {label}:")
            for i, h in enumerate(g.get('hubs', [])):
                print(f"  {i+1}. {h}")
        else:
            print(f"No graph found. Run 'rebuild' first.")

    elif cmd == 'query':
        if len(sys.argv) < 4:
            print("Usage: python context_manager.py query <dimension> <node_id> [sub_view]")
            return
        dim, node = sys.argv[2], sys.argv[3]
        sub = sys.argv[4] if len(sys.argv) > 4 else None
        ns = query_neighbors(dim, node, sub)
        if ns:
            label = f"{dim}/{sub}" if sub else dim
            print(f"\nNeighbors of {node} in {label}:")
            for n in ns:
                print(f"  {n['id']:12}  belief={n['belief']:.3f}  weight={n['dimension_weight']:.3f}")
        else:
            print(f"No neighbors found.")

    elif cmd == 'bridges':
        if len(sys.argv) >= 4:
            da, db = sys.argv[2], sys.argv[3]
            rs = discover_bridges(da, db)
            if rs:
                print(f"\nBridges between {da} and {db}:")
                for r in rs[:10]:
                    ba = r.get(f'belief_{da}', 0)
                    bb = r.get(f'belief_{db}', 0)
                    print(f"  {r['edge'][:35]:35} {da}={ba:.3f}  {db}={bb:.3f}")
            else:
                print("No bridges found.")
        else:
            g = load_graph('bridges')
            if g:
                edges = g.get('edges', {})
                print(f"\nCross-dimensional bridges: {len(edges)}")
                ranked = sorted(
                    edges.items(),
                    key=lambda x: (-x[1].get('dimension_count', 0),
                                   -sum(x[1].get('beliefs', {}).values()))
                )
                for k, d in ranked[:15]:
                    dims = ', '.join(d.get('dimensions', []))
                    sc = d.get('bridge_score', 0)
                    print(f"  {k[:35]:35} dims=[{dims}] score={sc:.3f}")
            else:
                print("No bridge graph. Run 'rebuild' first.")

    elif cmd == 'session-dims':
        dims = get_session_dimensions()
        print("\nActive session dimensions:")
        for k, v in dims.items():
            print(f"  {k}: {v}")

    else:
        print(f"Unknown command: {cmd}")


if __name__ == '__main__':
    main()
