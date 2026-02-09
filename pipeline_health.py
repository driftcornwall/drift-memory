#!/usr/bin/env python3
"""Pipeline Health Monitor — Experiment #1 lesson: bugs go undetected without heartbeats.

Tracks session-over-session deltas for memory count, edge count, recalls, and decay.
Flags anomalies when deltas fall outside expected ranges.

Usage:
    python pipeline_health.py snapshot          # Record current state
    python pipeline_health.py check             # Compare against last snapshot
    python pipeline_health.py trend [n]         # Show last n snapshots (default 10)
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HEALTH_DIR = os.path.join(os.path.dirname(__file__), "health")
SNAPSHOTS_FILE = os.path.join(HEALTH_DIR, "snapshots.json")

# Anomaly thresholds
EDGE_DROP_THRESHOLD = 0.20      # >20% edge loss in one session = alert
MEMORY_DROP_THRESHOLD = 0.10    # >10% memory loss = alert
NO_RECALL_SESSIONS = 3          # No recalls for 3+ sessions = alert
NO_NEW_MEMORIES_SESSIONS = 3    # No new memories for 3+ sessions = alert
DECAY_SPIKE_THRESHOLD = 3.0     # Decay events >3x average = alert


def get_current_stats():
    """Collect current pipeline stats from memory system."""
    mem_dir = os.path.dirname(__file__)
    sys.path.insert(0, mem_dir)
    from memory_common import CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR, parse_memory_file

    all_memories = []
    for d in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.md'):
                    all_memories.append(os.path.join(d, f))

    memory_count = len(all_memories)

    edge_count = 0
    edge_sum = 0.0
    for path in all_memories:
        try:
            meta, _ = parse_memory_file(Path(path))
            co = meta.get('co_occurrences', {})
            edge_count += len(co)
            edge_sum += sum(co.values())
        except Exception:
            pass

    # Edges are bidirectional — each pair counted twice
    unique_pairs = edge_count // 2

    session_file = os.path.join(mem_dir, "session_state.json")
    recalls = 0
    decay_events = 0
    prune_events = 0
    if os.path.exists(session_file):
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session = json.load(f)
            recalls = len(session.get('recalled_memories', []))
            decay_events = session.get('decay_events', 0)
            prune_events = session.get('prune_events', 0)
        except Exception:
            pass

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "memory_count": memory_count,
        "edge_pairs": unique_pairs,
        "edge_sum": round(edge_sum / 2, 2),
        "recalls": recalls,
        "decay_events": decay_events,
        "prune_events": prune_events,
    }


def load_snapshots():
    """Load all historical snapshots."""
    if not os.path.exists(SNAPSHOTS_FILE):
        return []
    with open(SNAPSHOTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_snapshots(snapshots):
    """Save snapshots to disk."""
    os.makedirs(HEALTH_DIR, exist_ok=True)
    with open(SNAPSHOTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(snapshots, f, indent=2)


def snapshot():
    """Record current pipeline state."""
    stats = get_current_stats()
    snapshots = load_snapshots()
    snapshots.append(stats)
    save_snapshots(snapshots)

    print(f"Snapshot recorded: {stats['timestamp']}")
    print(f"  Memories: {stats['memory_count']}")
    print(f"  Edge pairs: {stats['edge_pairs']}")
    print(f"  Edge weight sum: {stats['edge_sum']}")
    print(f"  Recalls this session: {stats['recalls']}")
    print(f"  Decay events: {stats['decay_events']}")
    print(f"  Prune events: {stats['prune_events']}")
    return stats


def check():
    """Compare current state against last snapshot. Flag anomalies."""
    current = get_current_stats()
    snapshots = load_snapshots()

    if not snapshots:
        print("No previous snapshots. Run 'snapshot' first.")
        return

    prev = snapshots[-1]
    alerts = []

    # Memory count delta
    mem_delta = current['memory_count'] - prev['memory_count']
    if prev['memory_count'] > 0:
        mem_pct = mem_delta / prev['memory_count']
        if mem_pct < -MEMORY_DROP_THRESHOLD:
            alerts.append(f"MEMORY DROP: {mem_delta} ({mem_pct:.1%}) — was {prev['memory_count']}, now {current['memory_count']}")

    # Edge count delta
    edge_delta = current['edge_pairs'] - prev['edge_pairs']
    if prev['edge_pairs'] > 0:
        edge_pct = edge_delta / prev['edge_pairs']
        if edge_pct < -EDGE_DROP_THRESHOLD:
            alerts.append(f"EDGE DROP: {edge_delta} ({edge_pct:.1%}) — was {prev['edge_pairs']}, now {current['edge_pairs']}")

    # No-recall streak
    recent = snapshots[-NO_RECALL_SESSIONS:]
    no_recall_streak = sum(1 for s in recent if s.get('recalls', 0) == 0)
    if no_recall_streak >= NO_RECALL_SESSIONS and current['recalls'] == 0:
        alerts.append(f"NO RECALLS: {no_recall_streak + 1} consecutive sessions without memory recall")

    # No-new-memory streak
    if len(snapshots) >= NO_NEW_MEMORIES_SESSIONS:
        no_new_streak = 0
        for i in range(len(snapshots) - 1, max(0, len(snapshots) - NO_NEW_MEMORIES_SESSIONS) - 1, -1):
            if i > 0 and snapshots[i]['memory_count'] <= snapshots[i-1]['memory_count']:
                no_new_streak += 1
            else:
                break
        if no_new_streak >= NO_NEW_MEMORIES_SESSIONS:
            alerts.append(f"NO NEW MEMORIES: {no_new_streak} consecutive sessions without memory growth")

    # Decay spike
    if len(snapshots) >= 3:
        avg_decay = sum(s.get('decay_events', 0) for s in snapshots[-3:]) / 3
        if avg_decay > 0 and current.get('decay_events', 0) > avg_decay * DECAY_SPIKE_THRESHOLD:
            alerts.append(f"DECAY SPIKE: {current['decay_events']} events (avg: {avg_decay:.0f})")

    # Print report
    print("=== PIPELINE HEALTH CHECK ===")
    print(f"  Timestamp: {current['timestamp']}")
    print(f"  Previous:  {prev['timestamp']}")
    print()
    print(f"  Memories:    {prev['memory_count']} -> {current['memory_count']} ({mem_delta:+d})")
    print(f"  Edge pairs:  {prev['edge_pairs']} -> {current['edge_pairs']} ({edge_delta:+d})")
    print(f"  Edge weight: {prev['edge_sum']} -> {current['edge_sum']} ({current['edge_sum'] - prev['edge_sum']:+.1f})")
    print(f"  Recalls:     {current['recalls']}")
    print(f"  Decay:       {current['decay_events']}")
    print(f"  Pruned:      {current['prune_events']}")
    print()

    if alerts:
        print(f"  ALERTS ({len(alerts)}):")
        for a in alerts:
            print(f"    ! {a}")
    else:
        print("  STATUS: HEALTHY")

    return alerts


def trend(n=10):
    """Show last n snapshots as a trend."""
    snapshots = load_snapshots()
    if not snapshots:
        print("No snapshots recorded yet.")
        return

    recent = snapshots[-n:]
    print(f"=== PIPELINE TREND (last {len(recent)} snapshots) ===")
    print(f"{'Timestamp':<26} {'Mem':>5} {'Edges':>7} {'Weight':>9} {'Recall':>6} {'Decay':>6} {'Prune':>6}")
    print("-" * 76)

    for s in recent:
        ts = s['timestamp'][:19]
        print(f"{ts:<26} {s['memory_count']:>5} {s['edge_pairs']:>7} {s['edge_sum']:>9.1f} {s['recalls']:>6} {s['decay_events']:>6} {s['prune_events']:>6}")

    if len(recent) >= 2:
        first, last = recent[0], recent[-1]
        print("-" * 76)
        mem_d = last['memory_count'] - first['memory_count']
        edge_d = last['edge_pairs'] - first['edge_pairs']
        weight_d = last['edge_sum'] - first['edge_sum']
        print(f"{'TOTAL DELTA':<26} {mem_d:>+5} {edge_d:>+7} {weight_d:>+9.1f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline_health.py [snapshot|check|trend [n]]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "snapshot":
        snapshot()
    elif cmd == "check":
        check()
    elif cmd == "trend":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        trend(n)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
