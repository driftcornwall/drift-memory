#!/usr/bin/env python3
"""
Experiment #2: Measurement Script

Shared measurement infrastructure for both Drift and SpindriftMend.
Captures standardized metrics at end of each session phase.

Usage:
    python experiment2/measure.py s1          # After intake session
    python experiment2/measure.py s2          # After integration session 2
    python experiment2/measure.py s3          # After integration session 3
    python experiment2/measure.py s4          # After integration session 4
    python experiment2/measure.py s5          # Final measurement
    python experiment2/measure.py baseline    # Pre-experiment baseline
"""

import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Add memory directory to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
MEMORY_DIR = PROJECT_DIR / "memory"
sys.path.insert(0, str(MEMORY_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from memory.memory_common import CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR, ALL_DIRS, parse_memory_file
from memory.session_state import get_retrieved as get_session_retrieved

RESULTS_DIR = SCRIPT_DIR / "results"
SOURCES_DIR = SCRIPT_DIR / "sources"


def get_experiment_memories():
    """Find all memories tagged with experiment-2."""
    experiment_mems = []
    for d in ALL_DIRS:
        if not d.exists():
            continue
        for f in d.glob("*.md"):
            try:
                meta, body = parse_memory_file(f)
                tags = meta.get("tags", [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",")]
                if "experiment-2" in tags or any(t.startswith("source-") for t in tags):
                    experiment_mems.append({
                        "id": meta.get("id", f.stem),
                        "file": str(f),
                        "tags": tags,
                        "created": str(meta.get("created", "")),
                        "recall_count": meta.get("recall_count", 0),
                        "co_occurrences": meta.get("co_occurrences", {}),
                        "trust_tier": meta.get("trust_tier", "active"),
                        "content_preview": body[:200] if body else "",
                        "directory": d.name,
                    })
            except Exception as e:
                print(f"  Warning: could not parse {f}: {e}", file=sys.stderr)
    return experiment_mems


def get_all_memory_stats():
    """Get basic memory statistics."""
    counts = {"core": 0, "active": 0, "archive": 0}
    total_edges = 0
    for d in ALL_DIRS:
        if not d.exists():
            continue
        for f in d.glob("*.md"):
            try:
                meta, _ = parse_memory_file(f)
                counts[d.name] = counts.get(d.name, 0) + 1
                cooc = meta.get("co_occurrences", {})
                if isinstance(cooc, dict):
                    total_edges += len(cooc)
            except Exception:
                pass
    return {
        "total_memories": sum(counts.values()),
        "by_tier": counts,
        "total_edges": total_edges,
    }


def run_fingerprint_export():
    """Run cognitive_fingerprint.py export and return the data."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(MEMORY_DIR / "cognitive_fingerprint.py"), "export"],
        capture_output=True, text=True, cwd=str(PROJECT_DIR)
    )
    if result.returncode == 0:
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"raw_output": result.stdout[:2000]}
    return {"error": result.stderr[:500]}


def measure_session(session_id: str):
    """Run full measurement for a given session phase."""
    timestamp = datetime.now(timezone.utc).isoformat()
    output_dir = RESULTS_DIR / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== EXPERIMENT #2 MEASUREMENT: {session_id.upper()} ===")
    print(f"Timestamp: {timestamp}")
    print()

    # 1. Memory stats
    print("[1/4] Memory statistics...")
    stats = get_all_memory_stats()
    print(f"  Total: {stats['total_memories']} memories, {stats['total_edges']} edges")
    print(f"  Tiers: {stats['by_tier']}")

    # 2. Experiment memories
    print("[2/4] Experiment memories...")
    exp_mems = get_experiment_memories()
    print(f"  Found: {len(exp_mems)} experiment-tagged memories")

    # Per-memory detail
    for m in exp_mems:
        edge_count = len(m["co_occurrences"]) if isinstance(m["co_occurrences"], dict) else 0
        source_tags = [t for t in m["tags"] if t.startswith("source-")]
        print(f"  [{m['id'][:8]}] {source_tags} edges={edge_count} recalls={m['recall_count']} tier={m['trust_tier']}")

    # 3. Session recalls (which experiment memories were naturally recalled?)
    print("[3/4] Session recall analysis...")
    session_retrieved = get_session_retrieved()
    exp_ids = {m["id"] for m in exp_mems}
    naturally_recalled = exp_ids & session_retrieved
    print(f"  Session recalls: {len(session_retrieved)} total")
    print(f"  Experiment recalls: {len(naturally_recalled)} (natural, not forced)")
    if naturally_recalled:
        for mid in naturally_recalled:
            mem = next((m for m in exp_mems if m["id"] == mid), None)
            if mem:
                print(f"    [{mid[:8]}] co-occurred with: {list(mem['co_occurrences'].keys())[:5]}")

    # 4. Cognitive fingerprint
    print("[4/4] Cognitive fingerprint export...")
    fingerprint = run_fingerprint_export()
    if "error" not in fingerprint:
        print(f"  Fingerprint captured successfully")
    else:
        print(f"  Warning: {fingerprint.get('error', 'unknown error')}")

    # Compile results
    results = {
        "session_id": session_id,
        "timestamp": timestamp,
        "memory_stats": stats,
        "experiment_memories": exp_mems,
        "session_recalls": {
            "total_recalled": len(session_retrieved),
            "experiment_recalled_naturally": list(naturally_recalled),
            "experiment_recalled_count": len(naturally_recalled),
        },
        "fingerprint": fingerprint,
        "experiment_summary": {
            "total_experiment_memories": len(exp_mems),
            "total_experiment_edges": sum(
                len(m["co_occurrences"]) if isinstance(m["co_occurrences"], dict) else 0
                for m in exp_mems
            ),
            "average_recall_count": (
                sum(m["recall_count"] for m in exp_mems) / len(exp_mems)
                if exp_mems else 0
            ),
            "tier_distribution": {},
            "domain_distribution": {},
        }
    }

    # Tier distribution
    for m in exp_mems:
        tier = m["trust_tier"]
        results["experiment_summary"]["tier_distribution"][tier] = \
            results["experiment_summary"]["tier_distribution"].get(tier, 0) + 1

    # Save results
    output_file = output_dir / f"measurement_{session_id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Also save fingerprint separately for easy comparison
    fp_file = output_dir / f"fingerprint_{session_id}.json"
    with open(fp_file, "w", encoding="utf-8") as f:
        json.dump(fingerprint, f, indent=2, default=str)
    print(f"Fingerprint saved to: {fp_file}")

    # Save stats separately
    stats_file = output_dir / f"stats_{session_id}.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Stats saved to: {stats_file}")

    print(f"\n=== MEASUREMENT COMPLETE: {session_id.upper()} ===")
    return results


def compare_sessions(s1: str, s2: str):
    """Compare measurements between two sessions."""
    f1 = RESULTS_DIR / s1 / f"measurement_{s1}.json"
    f2 = RESULTS_DIR / s2 / f"measurement_{s2}.json"

    if not f1.exists() or not f2.exists():
        print(f"Error: need both {f1} and {f2} to exist")
        return

    with open(f1) as f:
        d1 = json.load(f)
    with open(f2) as f:
        d2 = json.load(f)

    print(f"=== COMPARISON: {s1.upper()} vs {s2.upper()} ===")
    print()

    # Memory count delta
    m1 = d1["memory_stats"]["total_memories"]
    m2 = d2["memory_stats"]["total_memories"]
    print(f"Total memories: {m1} -> {m2} (delta: {m2 - m1:+d})")

    # Edge delta
    e1 = d1["memory_stats"]["total_edges"]
    e2 = d2["memory_stats"]["total_edges"]
    print(f"Total edges: {e1} -> {e2} (delta: {e2 - e1:+d})")

    # Experiment memory evolution
    exp1 = {m["id"]: m for m in d1.get("experiment_memories", [])}
    exp2 = {m["id"]: m for m in d2.get("experiment_memories", [])}

    print(f"\nExperiment memories: {len(exp1)} -> {len(exp2)}")
    print(f"Experiment edges: {d1['experiment_summary']['total_experiment_edges']} -> {d2['experiment_summary']['total_experiment_edges']}")

    # Per-memory edge growth
    common_ids = set(exp1.keys()) & set(exp2.keys())
    if common_ids:
        print(f"\nPer-memory edge growth ({len(common_ids)} tracked):")
        for mid in sorted(common_ids):
            e1_count = len(exp1[mid]["co_occurrences"]) if isinstance(exp1[mid]["co_occurrences"], dict) else 0
            e2_count = len(exp2[mid]["co_occurrences"]) if isinstance(exp2[mid]["co_occurrences"], dict) else 0
            r1 = exp1[mid]["recall_count"]
            r2 = exp2[mid]["recall_count"]
            print(f"  [{mid[:8]}] edges: {e1_count}->{e2_count} ({e2_count-e1_count:+d}), recalls: {r1}->{r2}")

    # Naturally recalled in s2
    recalled = d2.get("session_recalls", {}).get("experiment_recalled_naturally", [])
    if recalled:
        print(f"\nNaturally recalled in {s2}: {recalled}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd in ("baseline", "s1", "s2", "s3", "s4", "s5"):
        measure_session(cmd)
    elif cmd == "compare" and len(sys.argv) >= 4:
        compare_sessions(sys.argv[2], sys.argv[3])
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)
