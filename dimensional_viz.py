#!/usr/bin/env python3
"""
5W Dimensional Overlay Visualization for Drift's cognitive topology.

Neon light-drawing style: bright glowing edges on dark background,
all 5 dimensional projections overlaid with per-dimension Gini stats.

Usage:
    python dimensional_viz.py                # Generate overlay image
    python dimensional_viz.py --multiples    # Generate small multiples (5 panels)
    python dimensional_viz.py --open         # Generate and open the image
"""

import sys
import math
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

MEMORY_DIR = Path(__file__).parent
OUTPUT_PATH = MEMORY_DIR / "dimensional_overlay.png"
MULTIPLES_PATH = MEMORY_DIR / "dimensional_multiples.png"

# Neon dimension palette — electric, saturated, glow-friendly
DIMENSIONS = {
    'why':   {'color': '#ff9f1c', 'glow': '#ff9f1c', 'label': 'WHY',   'order': 0},
    'what':  {'color': '#00f0ff', 'glow': '#00f0ff', 'label': 'WHAT',  'order': 1},
    'when':  {'color': '#ff3366', 'glow': '#ff3366', 'label': 'WHEN',  'order': 2},
    'who':   {'color': '#39ff14', 'glow': '#39ff14', 'label': 'WHO',   'order': 3},
    'where': {'color': '#bf5fff', 'glow': '#bf5fff', 'label': 'WHERE', 'order': 4},
}

# Layout tuning
MAX_EDGES_PER_DIM = 350
GLOW_WIDTH = 2.5             # Outer glow line width
CORE_WIDTH = 0.5             # Inner bright core line width
GLOW_ALPHA_BASE = 0.04       # Glow layer alpha
GLOW_ALPHA_SCALE = 0.10
CORE_ALPHA_BASE = 0.15       # Core bright line alpha
CORE_ALPHA_SCALE = 0.55
NODE_SIZE_MIN = 6
NODE_SIZE_MAX = 160
HUB_GLOW_LAYERS = [
    (900, 0.03),
    (550, 0.08),
    (300, 0.15),
]


def load_dimension(name):
    """Load a dimension's context graph from DB."""
    try:
        from db_adapter import get_db
        db = get_db()
        if name == 'when':
            edges = {}
            hubs = []
            for sub in ['warm', 'cool']:
                row = db.get_context_graph('when', sub)
                if row and row.get('edges'):
                    for k, v in row['edges'].items():
                        if k not in edges or v.get('belief', 0) > edges[k].get('belief', 0):
                            edges[k] = v
                    hubs.extend(row.get('hubs', []))
            return edges, list(dict.fromkeys(hubs))[:10]

        row = db.get_context_graph(name, '')
        if not row or not row.get('edges'):
            return {}, []
        return row['edges'], row.get('hubs', [])[:10]
    except Exception:
        return {}, []


def load_gini_values():
    """Load per-dimension Gini from DB."""
    try:
        from db_adapter import get_db
        data = get_db().kv_get('.dimensional_fingerprints')
        if not data:
            return {}
        dims = data.get('dimensions', {})
        result = {}
        for key, info in dims.items():
            dist = info.get('distribution', {})
            gini = dist.get('gini')
            if gini is not None:
                dim_name = key.split('_')[0] if key.startswith('when') else key
                if dim_name not in result:
                    result[dim_name] = gini
        return result
    except Exception:
        return {}


def build_combined_graph():
    """Load all dimensions and build combined node/edge structures."""
    import networkx as nx

    dim_edges = {}
    dim_hubs = {}
    all_nodes = set()
    node_dim_degree = {}

    for dim_name in DIMENSIONS:
        edges, hubs = load_dimension(dim_name)
        dim_hubs[dim_name] = set(hubs)
        parsed = []

        for edge_key, props in edges.items():
            parts = edge_key.split('|')
            if len(parts) != 2:
                continue
            n1, n2 = parts
            belief = props.get('belief', 1.0)
            parsed.append((n1, n2, belief))
            all_nodes.add(n1)
            all_nodes.add(n2)

            for n in (n1, n2):
                if n not in node_dim_degree:
                    node_dim_degree[n] = {}
                node_dim_degree[n][dim_name] = node_dim_degree[n].get(dim_name, 0) + 1

        parsed.sort(key=lambda x: -x[2])
        dim_edges[dim_name] = parsed[:MAX_EDGES_PER_DIM]

    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    for dim_name, edges in dim_edges.items():
        for n1, n2, belief in edges:
            if G.has_edge(n1, n2):
                G[n1][n2]['weight'] = G[n1][n2]['weight'] + belief
            else:
                G.add_edge(n1, n2, weight=belief)

    return G, dim_edges, dim_hubs, node_dim_degree


def compute_layout(G):
    """Compute force-directed layout with radial stretching."""
    import networkx as nx
    import numpy as np

    pos = nx.spring_layout(
        G,
        k=2.5 / math.sqrt(max(G.number_of_nodes(), 1)),
        iterations=200,
        weight=None,
        seed=42,
        scale=1.0,
    )

    coords = np.array([pos[n] for n in G.nodes()])
    cx, cy = coords.mean(axis=0)

    stretched = {}
    for node_id in G.nodes():
        x, y = pos[node_id]
        dx, dy = x - cx, y - cy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-8:
            stretched[node_id] = (cx, cy)
            continue
        new_dist = dist ** 0.45
        scale = new_dist / dist
        stretched[node_id] = (cx + dx * scale, cy + dy * scale)

    return stretched


def find_dominant_dimension(node_id, node_dim_degree):
    """Find which dimension a node is most connected in."""
    dims = node_dim_degree.get(node_id, {})
    if not dims:
        return 'why'
    return max(dims, key=dims.get)


def render(G, pos, dim_edges, dim_hubs, node_dim_degree):
    """Render the 5D overlay with neon glow aesthetic."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection
    import numpy as np

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12), facecolor='#050510')
    ax = fig.add_axes([0.02, 0.14, 0.96, 0.78], facecolor='#050510')

    sorted_dims = sorted(DIMENSIONS.items(), key=lambda x: x[1]['order'])

    connected_nodes = set()
    total_edges_drawn = 0

    # --- NEON EDGE RENDERING (double-draw: glow + core) ---
    for dim_name, dim_cfg in sorted_dims:
        edges = dim_edges.get(dim_name, [])
        if not edges:
            continue

        rgb = mcolors.to_rgb(dim_cfg['color'])
        max_belief = max(e[2] for e in edges) if edges else 1.0

        dim_total = len(load_dimension(dim_name)[0])
        is_sparse = dim_total < 1000
        boost = 1.5 if is_sparse else 1.0

        glow_lines = []
        glow_colors = []
        core_lines = []
        core_colors = []

        for n1, n2, belief in edges:
            if n1 not in pos or n2 not in pos:
                continue
            x1, y1 = pos[n1]
            x2, y2 = pos[n2]
            segment = [(x1, y1), (x2, y2)]
            connected_nodes.add(n1)
            connected_nodes.add(n2)

            rel = belief / max_belief

            # Outer glow: wide, soft
            ga = min((GLOW_ALPHA_BASE + rel * GLOW_ALPHA_SCALE) * boost, 0.25)
            glow_lines.append(segment)
            glow_colors.append((*rgb, ga))

            # Inner core: thin, bright
            ca = min((CORE_ALPHA_BASE + rel * CORE_ALPHA_SCALE) * boost, 0.85)
            core_lines.append(segment)
            core_colors.append((*rgb, ca))

        # Draw glow layer first (behind)
        if glow_lines:
            gw = GLOW_WIDTH * (1.4 if is_sparse else 1.0)
            lc_glow = LineCollection(glow_lines, colors=glow_colors,
                                     linewidths=gw, zorder=1)
            ax.add_collection(lc_glow)

        # Draw core layer on top (bright thin line)
        if core_lines:
            cw = CORE_WIDTH * (1.3 if is_sparse else 1.0)
            lc_core = LineCollection(core_lines, colors=core_colors,
                                     linewidths=cw, zorder=2)
            ax.add_collection(lc_core)
            total_edges_drawn += len(core_lines)

    # --- NEON NODE RENDERING ---
    total_degree = {}
    for node_id, dims in node_dim_degree.items():
        total_degree[node_id] = sum(dims.values())
    max_deg = max(total_degree.values()) if total_degree else 1

    all_hubs = set()
    for hubs in dim_hubs.values():
        all_hubs.update(hubs)

    # Regular nodes
    for node_id in G.nodes():
        if node_id not in pos or node_id not in connected_nodes:
            continue
        if node_id in all_hubs:
            continue
        x, y = pos[node_id]
        deg = total_degree.get(node_id, 0)
        rel_deg = deg / max_deg

        dom_dim = find_dominant_dimension(node_id, node_dim_degree)
        color = DIMENSIONS[dom_dim]['color']

        size = NODE_SIZE_MIN + rel_deg * NODE_SIZE_MAX * 0.5
        alpha = 0.3 + rel_deg * 0.6

        # Subtle glow
        ax.scatter(x, y, c=color, s=size * 3, alpha=alpha * 0.15,
                   edgecolors='none', zorder=3)
        # Core dot
        ax.scatter(x, y, c=color, s=size, alpha=alpha,
                   edgecolors='none', zorder=4)

    # Hub nodes — multi-layer neon glow
    for node_id in all_hubs:
        if node_id not in pos or node_id not in connected_nodes:
            continue
        x, y = pos[node_id]
        deg = total_degree.get(node_id, 0)
        rel_deg = deg / max_deg

        dom_dim = find_dominant_dimension(node_id, node_dim_degree)
        color = DIMENSIONS[dom_dim]['color']

        core_size = NODE_SIZE_MIN + rel_deg * NODE_SIZE_MAX * 1.5

        # Glow rings (large, soft, layered)
        for glow_size, glow_alpha in HUB_GLOW_LAYERS:
            ax.scatter(x, y, c=color, s=glow_size,
                       alpha=glow_alpha, edgecolors='none', zorder=5)

        # Bright core
        ax.scatter(x, y, c=color, s=core_size,
                   alpha=0.95, edgecolors='none', zorder=6)

        # White-hot center for largest hubs
        if rel_deg > 0.5:
            ax.scatter(x, y, c='white', s=core_size * 0.3,
                       alpha=0.6, edgecolors='none', zorder=7)

    # --- TITLE (neon style) ---
    ax.text(0.5, 1.04, 'DRIFT', fontsize=36, fontweight='bold',
            color='#e8e8ff', ha='center', va='bottom',
            fontfamily='monospace', transform=ax.transAxes)
    ax.text(0.5, 1.005, '5W DIMENSIONAL OVERLAY', fontsize=12,
            color='#6666aa', ha='center', va='bottom',
            fontfamily='monospace', transform=ax.transAxes,
            fontstyle='italic')

    ax.set_aspect('equal')
    ax.axis('off')

    # Auto-scale to connected nodes
    connected_pos = [pos[n] for n in connected_nodes if n in pos]
    if connected_pos:
        xs = [p[0] for p in connected_pos]
        ys = [p[1] for p in connected_pos]
        margin = 0.08
        x_range = max(xs) - min(xs) or 1
        y_range = max(ys) - min(ys) or 1
        ax.set_xlim(min(xs) - margin * x_range, max(xs) + margin * x_range)
        ax.set_ylim(min(ys) - margin * y_range, max(ys) + margin * y_range)

    # --- LEGEND + STATS PANEL (neon bar) ---
    ax_legend = fig.add_axes([0.02, 0.01, 0.96, 0.11], facecolor='#050510')
    ax_legend.axis('off')
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)

    # Thin separator line
    ax_legend.plot([0.02, 0.98], [0.95, 0.95], color='#222244',
                   linewidth=0.5, zorder=1)

    gini_values = load_gini_values()

    # Dimension legend: dot + label + edges + gini
    legend_x = 0.04
    spacing = 0.19
    for dim_name, dim_cfg in sorted_dims:
        color = dim_cfg['color']
        label = dim_cfg['label']
        dim_full_edges = len(load_dimension(dim_name)[0])
        gini = gini_values.get(dim_name)

        # Neon dot with glow
        ax_legend.scatter(legend_x, 0.65, c=color, s=120, alpha=0.2,
                          edgecolors='none', zorder=2)
        ax_legend.scatter(legend_x, 0.65, c=color, s=50, alpha=0.9,
                          edgecolors='none', zorder=3)

        # Dimension name
        ax_legend.text(legend_x + 0.022, 0.65, label, fontsize=11,
                       color=color, va='center', fontfamily='monospace',
                       fontweight='bold')

        # Edge count
        ax_legend.text(legend_x + 0.022, 0.35, f'{dim_full_edges:,} edges',
                       fontsize=8, color='#777799', va='center',
                       fontfamily='monospace')

        # Gini value
        if gini is not None:
            ax_legend.text(legend_x + 0.022, 0.10,
                           f'gini {gini:.3f}',
                           fontsize=8, color='#555577', va='center',
                           fontfamily='monospace')

        legend_x += spacing

    # Total stats centered at bottom
    node_count = len(connected_nodes)
    stats_text = (
        f'NODES: {node_count}  |  '
        f'EDGES DRAWN: {total_edges_drawn:,}  |  '
        f'HUBS: {len(all_hubs)}  |  '
        f'L0 BASIS: 4,168 edges'
    )
    ax_legend.text(0.5, -0.15, stats_text, fontsize=8,
                   color='#555577', ha='center', va='center',
                   fontfamily='monospace')

    plt.savefig(str(OUTPUT_PATH), dpi=180, bbox_inches='tight',
                facecolor='#050510', edgecolor='none')
    plt.close(fig)

    size = OUTPUT_PATH.stat().st_size
    print(f"Saved: {OUTPUT_PATH} ({size:,} bytes)")
    return OUTPUT_PATH


def render_multiples(G, pos, dim_edges, dim_hubs, node_dim_degree):
    """Render 5 side-by-side panels, one per dimension. Same layout, different edges."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 5, figsize=(25, 6), facecolor='#050510')
    fig.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.85, bottom=0.12)

    sorted_dims = sorted(DIMENSIONS.items(), key=lambda x: x[1]['order'])
    gini_values = load_gini_values()

    # Compute connected nodes and total degree for node sizing
    total_degree = {}
    for node_id, dims in node_dim_degree.items():
        total_degree[node_id] = sum(dims.values())
    max_deg = max(total_degree.values()) if total_degree else 1

    all_hubs = set()
    for hubs in dim_hubs.values():
        all_hubs.update(hubs)

    # Get axis bounds from full graph
    all_pos = [pos[n] for n in G.nodes() if n in pos]
    if all_pos:
        xs = [p[0] for p in all_pos]
        ys = [p[1] for p in all_pos]
        margin = 0.1
        x_range = max(xs) - min(xs) or 1
        y_range = max(ys) - min(ys) or 1
        xlim = (min(xs) - margin * x_range, max(xs) + margin * x_range)
        ylim = (min(ys) - margin * y_range, max(ys) + margin * y_range)
    else:
        xlim = (-1, 1)
        ylim = (-1, 1)

    for idx, (dim_name, dim_cfg) in enumerate(sorted_dims):
        ax = axes[idx]
        ax.set_facecolor('#050510')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        ax.axis('off')

        edges = dim_edges.get(dim_name, [])
        rgb = mcolors.to_rgb(dim_cfg['color'])
        color = dim_cfg['color']
        label = dim_cfg['label']
        dim_full_edges = len(load_dimension(dim_name)[0])
        gini = gini_values.get(dim_name)

        max_belief = max(e[2] for e in edges) if edges else 1.0
        connected = set()

        # Draw edges with glow
        if edges:
            glow_lines = []
            glow_colors = []
            core_lines = []
            core_colors = []

            for n1, n2, belief in edges:
                if n1 not in pos or n2 not in pos:
                    continue
                x1, y1 = pos[n1]
                x2, y2 = pos[n2]
                segment = [(x1, y1), (x2, y2)]
                connected.add(n1)
                connected.add(n2)

                rel = belief / max_belief
                ga = min(0.04 + rel * 0.12, 0.25)
                glow_lines.append(segment)
                glow_colors.append((*rgb, ga))

                ca = min(0.15 + rel * 0.6, 0.85)
                core_lines.append(segment)
                core_colors.append((*rgb, ca))

            if glow_lines:
                lc = LineCollection(glow_lines, colors=glow_colors, linewidths=2.0, zorder=1)
                ax.add_collection(lc)
            if core_lines:
                lc = LineCollection(core_lines, colors=core_colors, linewidths=0.4, zorder=2)
                ax.add_collection(lc)

        # Draw nodes
        dim_hub_set = dim_hubs.get(dim_name, set())
        for node_id in G.nodes():
            if node_id not in pos or node_id not in connected:
                continue
            x, y = pos[node_id]
            deg = node_dim_degree.get(node_id, {}).get(dim_name, 0)
            max_dim_deg = max((node_dim_degree.get(n, {}).get(dim_name, 0) for n in connected), default=1) or 1
            rel_deg = deg / max_dim_deg

            if node_id in dim_hub_set:
                size = 20 + rel_deg * 80
                ax.scatter(x, y, c=color, s=size * 3, alpha=0.12, edgecolors='none', zorder=3)
                ax.scatter(x, y, c=color, s=size, alpha=0.9, edgecolors='none', zorder=4)
                if rel_deg > 0.5:
                    ax.scatter(x, y, c='white', s=size * 0.25, alpha=0.5, edgecolors='none', zorder=5)
            else:
                size = 3 + rel_deg * 25
                alpha = 0.2 + rel_deg * 0.5
                ax.scatter(x, y, c=color, s=size, alpha=alpha, edgecolors='none', zorder=3)

        # Panel title
        ax.set_title(label, fontsize=16, color=color, fontfamily='monospace',
                     fontweight='bold', pad=8)

        # Stats below panel
        stats_text = f'{dim_full_edges:,} edges'
        if gini is not None:
            stats_text += f' | gini {gini:.3f}'
        ax.text(0.5, -0.04, stats_text, fontsize=8, color='#777799',
                ha='center', va='top', fontfamily='monospace',
                transform=ax.transAxes)

    # Overall title
    fig.suptitle('DRIFT  |  5W SMALL MULTIPLES', fontsize=20, color='#e8e8ff',
                 fontfamily='monospace', fontweight='bold', y=0.95)

    plt.savefig(str(MULTIPLES_PATH), dpi=150, bbox_inches='tight',
                facecolor='#050510', edgecolor='none')
    plt.close(fig)

    size = MULTIPLES_PATH.stat().st_size
    print(f"Saved: {MULTIPLES_PATH} ({size:,} bytes)")
    return MULTIPLES_PATH


def main():
    args = set(sys.argv[1:])

    print("=== 5W DIMENSIONAL OVERLAY ===\n")

    print("[1/3] Loading 5 dimensional graphs...")
    G, dim_edges, dim_hubs, node_dim_degree = build_combined_graph()
    print(f"  Combined: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    for dim_name in DIMENSIONS:
        print(f"  {dim_name.upper()}: {len(dim_edges[dim_name])} edges (capped at {MAX_EDGES_PER_DIM})")

    print("\n[2/3] Computing force-directed layout...")
    pos = compute_layout(G)
    print(f"  Positioned {len(pos)} nodes")

    if '--multiples' in args:
        print("\n[3/3] Rendering small multiples...")
        output = render_multiples(G, pos, dim_edges, dim_hubs, node_dim_degree)
    else:
        print("\n[3/3] Rendering overlay...")
        output = render(G, pos, dim_edges, dim_hubs, node_dim_degree)

    if '--open' in args:
        import subprocess
        subprocess.Popen(['start', '', str(output)], shell=True)

    print("\nDone.")


if __name__ == '__main__':
    main()
