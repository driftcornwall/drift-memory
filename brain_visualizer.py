#!/usr/bin/env python3
"""
Brain Visualizer — Cognitive Fingerprint as Graph

Renders the co-occurrence topology as a visual graph:
- Nodes = memories (sized by degree/connections)
- Edges = co-occurrence relationships (weighted by belief score)
- Colors = platform context or activity context
- Layout = force-directed (clusters emerge naturally)

Usage:
    python brain_visualizer.py              # Generate brain.png
    python brain_visualizer.py --html       # Generate interactive HTML
    python brain_visualizer.py --platform   # Color by platform
    python brain_visualizer.py --activity   # Color by activity type
    python brain_visualizer.py --top N      # Only show top N connected nodes
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

MEMORY_DIR = Path(__file__).parent
EDGES_FILE = MEMORY_DIR / ".edges_v3.json"
OUTPUT_DIR = MEMORY_DIR / "visualizations"

# Color schemes
PLATFORM_COLORS = {
    'github': '#238636',      # GitHub green
    'moltx': '#1DA1F2',       # Twitter blue
    'moltbook': '#9146FF',    # Purple
    'clawtasks': '#F7931A',   # Orange (Bitcoin-ish)
    'lobsterpedia': '#FF4500', # Reddit orange
    'dead-internet': '#666666', # Gray
    'nostr': '#8B5CF6',       # Purple
    'unknown': '#888888',     # Gray
}

ACTIVITY_COLORS = {
    'technical': '#00D4AA',    # Cyan/teal
    'collaborative': '#FF6B6B', # Coral red
    'exploratory': '#4ECDC4',  # Turquoise
    'social': '#FFE66D',       # Yellow
    'economic': '#95E616',     # Lime green
    'reflective': '#A855F7',   # Purple
    'unknown': '#888888',      # Gray
}


def load_edges() -> dict:
    """Load edges from v3 format."""
    if not EDGES_FILE.exists():
        return {}

    with open(EDGES_FILE, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # Convert string keys back to tuples
    edges = {}
    for key, value in raw.items():
        if '|' in key:
            pair = tuple(key.split('|'))
            edges[pair] = value

    return edges


def get_node_metadata(memory_id: str) -> dict:
    """Get metadata for a memory node."""
    for subdir in ['core', 'active', 'archive']:
        dir_path = MEMORY_DIR / subdir
        if not dir_path.exists():
            continue
        for f in dir_path.glob(f"*-{memory_id}.md"):
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Parse YAML frontmatter
                    if content.startswith('---'):
                        end = content.find('---', 3)
                        if end > 0:
                            import yaml
                            metadata = yaml.safe_load(content[3:end])
                            return metadata or {}
            except Exception:
                pass
    return {}


def build_graph_data(edges: dict, color_by: str = 'platform', top_n: int = None):
    """Build node and edge data for visualization."""
    # Calculate node degrees
    node_degrees = defaultdict(int)
    node_platforms = defaultdict(lambda: defaultdict(int))
    node_activities = defaultdict(lambda: defaultdict(int))

    for (id1, id2), edge_data in edges.items():
        belief = edge_data.get('belief', 1.0)
        node_degrees[id1] += belief
        node_degrees[id2] += belief

        # Aggregate platform context
        for plat, count in edge_data.get('platform_context', {}).items():
            node_platforms[id1][plat] += count
            node_platforms[id2][plat] += count

        # Aggregate activity context
        for act, count in edge_data.get('activity_context', {}).items():
            node_activities[id1][act] += count
            node_activities[id2][act] += count

    # Filter to top N if specified
    if top_n:
        sorted_nodes = sorted(node_degrees.items(), key=lambda x: -x[1])[:top_n]
        included_nodes = set(n for n, _ in sorted_nodes)
        edges = {k: v for k, v in edges.items()
                 if k[0] in included_nodes and k[1] in included_nodes}
        node_degrees = {n: d for n, d in node_degrees.items() if n in included_nodes}

    # Determine node colors
    node_colors = {}
    for node in node_degrees:
        if color_by == 'platform':
            platforms = node_platforms.get(node, {})
            if platforms:
                dominant = max(platforms.items(), key=lambda x: x[1])[0]
                node_colors[node] = PLATFORM_COLORS.get(dominant, PLATFORM_COLORS['unknown'])
            else:
                node_colors[node] = PLATFORM_COLORS['unknown']
        else:  # activity
            activities = node_activities.get(node, {})
            if activities:
                dominant = max(activities.items(), key=lambda x: x[1])[0]
                node_colors[node] = ACTIVITY_COLORS.get(dominant, ACTIVITY_COLORS['unknown'])
            else:
                node_colors[node] = ACTIVITY_COLORS['unknown']

    return {
        'nodes': node_degrees,
        'edges': edges,
        'colors': node_colors,
        'platforms': dict(node_platforms),
        'activities': dict(node_activities),
    }


def generate_matplotlib_graph(graph_data: dict, output_path: Path, title: str = "Drift's Cognitive Fingerprint", color_by: str = 'platform'):
    """Generate static PNG using matplotlib + networkx."""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.run(['pip', 'install', 'networkx', 'matplotlib'], check=True)
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

    # Build networkx graph
    G = nx.Graph()

    nodes = graph_data['nodes']
    edges = graph_data['edges']
    colors = graph_data['colors']

    # Add nodes
    for node, degree in nodes.items():
        G.add_node(node, size=degree)

    # Add edges
    for (id1, id2), edge_data in edges.items():
        weight = edge_data.get('belief', 1.0)
        G.add_edge(id1, id2, weight=weight)

    if len(G.nodes()) == 0:
        print("No nodes to visualize!")
        return

    # Create figure with dark background
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')

    # Layout - spring layout for force-directed clustering
    print(f"Computing layout for {len(G.nodes())} nodes, {len(G.edges())} edges...")
    pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)

    # Node sizes based on degree (normalized)
    max_degree = max(nodes.values()) if nodes else 1
    node_sizes = [300 + (nodes.get(n, 1) / max_degree) * 1500 for n in G.nodes()]

    # Node colors
    node_color_list = [colors.get(n, '#888888') for n in G.nodes()]

    # Edge weights for thickness
    edge_weights = [edges.get((min(u, v), max(u, v)), {}).get('belief', 0.5) for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.3 + (w / max_weight) * 2 for w in edge_weights]

    # Draw edges first (behind nodes)
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='#58a6ff',  # Lighter blue for visibility
        width=edge_widths,
        alpha=0.3
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_color_list,
        alpha=0.9,
        edgecolors='#ffffff',
        linewidths=0.5
    )

    # Labels for top nodes only (to avoid clutter)
    top_nodes = sorted(nodes.items(), key=lambda x: -x[1])[:15]
    labels = {n: n[:8] for n, _ in top_nodes}
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=7,
        font_color='#ffffff',
        font_weight='bold'
    )

    # Title
    ax.set_title(title, fontsize=16, fontweight='bold', color='#ffffff', pad=20)

    # Stats annotation
    stats_text = f"Nodes: {len(G.nodes())} | Edges: {len(G.edges())} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    ax.annotate(stats_text, xy=(0.5, 0.02), xycoords='axes fraction',
                ha='center', fontsize=9, color='#8b949e')

    # Legend - use the correct color map based on color_by parameter
    legend_patches = []
    color_map = PLATFORM_COLORS if color_by == 'platform' else ACTIVITY_COLORS
    for name, color in list(color_map.items())[:6]:
        if name != 'unknown':
            legend_patches.append(mpatches.Patch(color=color, label=name.title()))

    ax.legend(handles=legend_patches, loc='upper left',
              facecolor='#161b22', edgecolor='#30363d',
              labelcolor='#ffffff', fontsize=8)

    ax.axis('off')

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def generate_interactive_html(graph_data: dict, output_path: Path, title: str = "Drift's Cognitive Fingerprint"):
    """Generate interactive HTML using pyvis."""
    try:
        from pyvis.network import Network
    except ImportError:
        print("Installing pyvis...")
        import subprocess
        subprocess.run(['pip', 'install', 'pyvis'], check=True)
        from pyvis.network import Network

    nodes = graph_data['nodes']
    edges = graph_data['edges']
    colors = graph_data['colors']

    # Create network
    net = Network(height='900px', width='100%', bgcolor='#0d1117', font_color='#ffffff')
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200)

    # Normalize sizes
    max_degree = max(nodes.values()) if nodes else 1

    # Add nodes
    for node, degree in nodes.items():
        size = 10 + (degree / max_degree) * 40
        color = colors.get(node, '#888888')

        # Get metadata for tooltip
        metadata = get_node_metadata(node)
        tooltip = f"ID: {node}\nDegree: {degree:.1f}"
        if metadata.get('tags'):
            tooltip += f"\nTags: {', '.join(metadata['tags'][:5])}"

        net.add_node(node, label=node[:8], size=size, color=color, title=tooltip)

    # Add edges
    for (id1, id2), edge_data in edges.items():
        weight = edge_data.get('belief', 1.0)
        width = 0.5 + weight * 2

        # Edge tooltip
        tooltip = f"Belief: {weight:.2f}"
        if edge_data.get('platform_context'):
            tooltip += f"\nPlatforms: {', '.join(edge_data['platform_context'].keys())}"
        if edge_data.get('activity_context'):
            tooltip += f"\nActivities: {', '.join(edge_data['activity_context'].keys())}"
        if edge_data.get('thinking_about'):
            tooltip += f"\nContext: {len(edge_data['thinking_about'])} other memories"

        net.add_edge(id1, id2, value=width, title=tooltip, color='#30363d')

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_path))

    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize cognitive fingerprint as graph")
    parser.add_argument('--html', action='store_true', help='Generate interactive HTML instead of PNG')
    parser.add_argument('--platform', action='store_true', help='Color by platform (default)')
    parser.add_argument('--activity', action='store_true', help='Color by activity type')
    parser.add_argument('--top', type=int, default=None, help='Only show top N connected nodes')
    parser.add_argument('--output', type=str, default=None, help='Output filename')
    args = parser.parse_args()

    # Load edges
    print("Loading edges...")
    edges = load_edges()
    print(f"Loaded {len(edges)} edges")

    if not edges:
        print("No edges found! Run some sessions first to build co-occurrences.")
        return

    # Determine color scheme
    color_by = 'activity' if args.activity else 'platform'

    # Build graph data
    print(f"Building graph (color by {color_by})...")
    graph_data = build_graph_data(edges, color_by=color_by, top_n=args.top)

    print(f"Graph has {len(graph_data['nodes'])} nodes")

    # Generate visualization
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    if args.html:
        filename = args.output or f"brain_{color_by}_{timestamp}.html"
        output_path = OUTPUT_DIR / filename
        generate_interactive_html(graph_data, output_path,
                                  title=f"Drift's Cognitive Fingerprint (by {color_by.title()})")
    else:
        filename = args.output or f"brain_{color_by}_{timestamp}.png"
        output_path = OUTPUT_DIR / filename
        generate_matplotlib_graph(graph_data, output_path,
                                  title=f"Drift's Cognitive Fingerprint (by {color_by.title()})",
                                  color_by=color_by)

    return output_path


if __name__ == '__main__':
    main()
