#!/usr/bin/env python3
"""
Experiment Comparison Tool for drift-memory co-occurrence experiments.

Compares two cognitive fingerprint exports side-by-side.
Produces formatted terminal output and optional JSON diff.

Usage:
    python experiment_compare.py <export_a.json> <export_b.json>
    python experiment_compare.py <export_a.json> <export_b.json> --json
    python experiment_compare.py --self  # Compare all own exports chronologically
"""

import json
import sys
import os
from pathlib import Path


def load_export(path):
    with open(path) as f:
        return json.load(f)


def extract_domains(export):
    """Extract domain percentages from various export formats."""
    domains = export.get("domains", {})
    if "domains" in domains:
        domains = domains["domains"]
    result = {}
    for name, info in domains.items():
        if isinstance(info, dict):
            result[name] = info.get("weight_pct", 0)
        else:
            result[name] = info
    return result


def compare_exports(a, b):
    """Compare two fingerprint exports."""
    agent_a = a.get("agent", "Agent A")
    agent_b = b.get("agent", "Agent B")
    label_a = a.get("label", a.get("timestamp", "?"))
    label_b = b.get("label", b.get("timestamp", "?"))

    print("=" * 70)
    print("  CO-OCCURRENCE EXPERIMENT COMPARISON")
    print("=" * 70)
    print(f"  {agent_a}: {label_a}")
    print(f"  {agent_b}: {label_b}")
    print()

    # Graph metrics
    metrics = [
        ("Nodes", "nodes"),
        ("Edges", "edges"),
        ("Avg Degree", "avg_degree"),
    ]

    print("  GRAPH METRICS")
    print("  " + "-" * 66)
    print(f"  {'Metric':<20} {agent_a:>15} {agent_b:>15} {'Ratio':>10}")
    print("  " + "-" * 66)

    for label, key in metrics:
        va = a.get(key, 0)
        vb = b.get(key, 0)
        ratio = va / vb if vb else float("inf")
        print(f"  {label:<20} {va:>15.1f} {vb:>15.1f} {ratio:>9.1f}x")

    print()

    # Distribution stats
    dist_a = a.get("distribution", {})
    dist_b = b.get("distribution", {})

    if dist_a and dist_b:
        print("  DISTRIBUTION")
        print("  " + "-" * 66)
        print(f"  {'Metric':<20} {agent_a:>15} {agent_b:>15} {'Delta':>10}")
        print("  " + "-" * 66)

        dist_keys = [
            ("Mean weight", "mean"),
            ("Median weight", "p50_median"),
            ("Std dev", "std_dev"),
            ("Skewness", "skewness"),
            ("Gini coeff", "gini"),
            ("Max weight", "max"),
            ("P99 weight", "p99"),
        ]

        for label, key in dist_keys:
            va = dist_a.get(key, 0)
            vb = dist_b.get(key, 0)
            delta = va - vb
            sign = "+" if delta > 0 else ""
            if va and vb:
                print(f"  {label:<20} {va:>15.3f} {vb:>15.3f} {sign}{delta:>9.3f}")

    print()

    # Domains
    dom_a = extract_domains(a)
    dom_b = extract_domains(b)
    all_domains = sorted(set(list(dom_a.keys()) + list(dom_b.keys())))

    if all_domains:
        print("  COGNITIVE DOMAINS")
        print("  " + "-" * 66)
        print(f"  {'Domain':<20} {agent_a:>15} {agent_b:>15} {'Delta':>10}")
        print("  " + "-" * 66)

        for d in all_domains:
            va = dom_a.get(d, 0)
            vb = dom_b.get(d, 0)
            delta = va - vb
            sign = "+" if delta > 0 else ""
            bar_a = "#" * int(va / 2)
            print(f"  {d:<20} {va:>14.1f}% {vb:>14.1f}% {sign}{delta:>8.1f}pp")

    print()

    # Hub ordering
    hubs_a = a.get("hub_ordering", [])
    hubs_b = b.get("hub_ordering", [])

    if hubs_a and hubs_b:
        print("  HUB ORDERING (top 10)")
        print("  " + "-" * 66)
        shared = set(hubs_a[:10]) & set(hubs_b[:10])
        only_a = set(hubs_a[:10]) - set(hubs_b[:10])
        only_b = set(hubs_b[:10]) - set(hubs_a[:10])

        print(f"  Shared hubs: {len(shared)}")
        for h in hubs_a[:10]:
            if h in shared:
                rank_a = hubs_a.index(h) + 1
                rank_b = hubs_b.index(h) + 1
                print(f"    {h}: #{rank_a} ({agent_a}) / #{rank_b} ({agent_b})")

        if only_a:
            print(f"  Only {agent_a}: {', '.join(only_a)}")
        if only_b:
            print(f"  Only {agent_b}: {', '.join(only_b)}")

    print()

    # Fingerprint hashes
    hash_a = a.get("fingerprint_hash", "?")
    hash_b = b.get("fingerprint_hash", "?")
    print("  FINGERPRINT HASHES")
    print(f"  {agent_a}: {hash_a[:32]}...")
    print(f"  {agent_b}: {hash_b[:32]}...")
    print(f"  Match: {'YES' if hash_a == hash_b else 'NO'}")

    print()

    # Drift
    drift_a = a.get("drift_score", 0)
    drift_b = b.get("drift_score", 0)
    print("  IDENTITY DRIFT")
    print(f"  {agent_a}: {drift_a}")
    print(f"  {agent_b}: {drift_b}")

    print()
    print("=" * 70)

    return {
        "agents": [agent_a, agent_b],
        "edge_ratio": a.get("edges", 0) / max(b.get("edges", 1), 1),
        "shared_hubs": len(shared) if hubs_a and hubs_b else 0,
        "domain_divergence": {
            d: dom_a.get(d, 0) - dom_b.get(d, 0) for d in all_domains
        },
    }


def self_compare(export_dir):
    """Compare all exports from a single agent chronologically."""
    files = sorted(Path(export_dir).glob("*.json"))
    if len(files) < 2:
        print(f"Need at least 2 exports in {export_dir}, found {len(files)}")
        return

    print(f"CHRONOLOGICAL SELF-COMPARISON ({len(files)} snapshots)")
    print("=" * 70)

    prev = None
    for f in files:
        curr = load_export(f)
        nodes = curr.get("nodes", "?")
        edges = curr.get("edges", "?")
        label = curr.get("label", f.stem)

        if prev is not None:
            pn = prev.get("nodes", 0)
            pe = prev.get("edges", 0)
            dn = nodes - pn if isinstance(nodes, (int, float)) else "?"
            de = edges - pe if isinstance(edges, (int, float)) else "?"
            print(
                f"  {f.name}: {nodes} nodes (+{dn}), "
                f"{edges} edges (+{de})"
            )
        else:
            print(f"  {f.name}: {nodes} nodes, {edges} edges [baseline]")

        prev = curr

    print()
    print("  First -> Last:")
    first = load_export(files[0])
    last = load_export(files[-1])
    fn = first.get("nodes", 0)
    ln = last.get("nodes", 0)
    fe = first.get("edges", 0)
    le = last.get("edges", 0)
    if fn and ln:
        print(f"    Nodes: {fn} -> {ln} ({ln/fn:.1f}x)")
    if fe and le:
        print(f"    Edges: {fe} -> {le} ({le/fe:.1f}x)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "--self":
        export_dir = (
            sys.argv[2]
            if len(sys.argv) > 2
            else os.path.join(os.path.dirname(__file__), "exports")
        )
        self_compare(export_dir)
    elif len(sys.argv) >= 3:
        a = load_export(sys.argv[1])
        b = load_export(sys.argv[2])
        result = compare_exports(a, b)

        if "--json" in sys.argv:
            print(json.dumps(result, indent=2))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
