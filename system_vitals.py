#!/usr/bin/env python3
"""
System Vitals Monitor v1.0-drift

Adapted from SpindriftMend's system_vitals.py (Issue #22).
Extended with: W-graph dimensional metrics, Twitter metrics,
vocabulary bridge health, within-session deltas.

Usage:
    python system_vitals.py record          # Take a snapshot now
    python system_vitals.py latest          # Show most recent snapshot
    python system_vitals.py trends [N]      # Show trends over last N sessions (default 10)
    python system_vitals.py alerts          # Flag stalled or regressing metrics
    python system_vitals.py history [N]     # Show last N snapshots compact (default 5)
    python system_vitals.py collect         # Debug: collect without saving

Credit: SpindriftMend (github.com/SpindriftMind)
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

MEMORY_ROOT = Path(__file__).parent
VITALS_LOG = MEMORY_ROOT / ".vitals_log.json"
CONTEXT_DIR = MEMORY_ROOT / "context"

STALL_THRESHOLD = 5
DECLINE_THRESHOLD = 3


def _count_files(directory, pattern="*.md"):
    d = MEMORY_ROOT / directory
    return len(list(d.glob(pattern))) if d.exists() else 0


def _load_json(path, default=None):
    fp = MEMORY_ROOT / path if not Path(path).is_absolute() else Path(path)
    if not fp.exists():
        return default
    try:
        return json.loads(fp.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return default


def _count_edges_in_graph(graph_data):
    """Count edges in a W-graph dict."""
    edges = graph_data.get('edges', {})
    return len(edges)


def collect_vitals():
    """Collect all system vitals. Returns a snapshot dict."""
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {}
    }
    m = snapshot["metrics"]

    # --- MEMORY COUNTS ---
    core = _count_files("core")
    active = _count_files("active")
    archive = _count_files("archive")
    m["memory_total"] = core + active + archive
    m["memory_core"] = core
    m["memory_active"] = active
    m["memory_archive"] = archive

    # --- CO-OCCURRENCE ---
    edges_data = _load_json(".edges_v3.json", {})
    if isinstance(edges_data, dict) and edges_data:
        beliefs = [e.get('belief', 0) for e in edges_data.values() if isinstance(e, dict)]
        m["cooccurrence_pairs"] = sum(1 for b in beliefs if b > 0)
        m["cooccurrence_total_strength"] = round(sum(beliefs), 2)
        m["cooccurrence_links"] = sum(1 for b in beliefs if b >= 3.0)
        m["cooccurrence_edges_total"] = len(edges_data)
    else:
        m["cooccurrence_pairs"] = 0
        m["cooccurrence_total_strength"] = 0
        m["cooccurrence_links"] = 0
        m["cooccurrence_edges_total"] = 0

    # --- REJECTION LOG ---
    rej_data = _load_json(".rejection_log.json", {})
    if isinstance(rej_data, dict):
        rejections = rej_data.get('rejections', [])
    elif isinstance(rej_data, list):
        rejections = rej_data
    else:
        rejections = []
    m["rejection_count"] = len(rejections)

    # --- LESSONS ---
    lessons = _load_json("lessons.json", [])
    m["lesson_count"] = len(lessons) if isinstance(lessons, list) else 0

    # --- MERKLE CHAIN ---
    attestations = _load_json("attestations.json", [])
    if isinstance(attestations, list) and attestations:
        latest_att = attestations[-1]
        m["merkle_chain_depth"] = latest_att.get('chain_depth', len(attestations))
        m["merkle_memory_count"] = latest_att.get('memory_count', 0)
    else:
        m["merkle_chain_depth"] = 0
        m["merkle_memory_count"] = 0

    # --- COGNITIVE FINGERPRINT ---
    fp_history = _load_json(".fingerprint_history.json", [])
    if isinstance(fp_history, list) and fp_history:
        latest_fp = fp_history[-1]
        m["fingerprint_nodes"] = latest_fp.get('node_count', 0)
        m["fingerprint_edges"] = latest_fp.get('edge_count', 0)
        m["identity_drift"] = latest_fp.get('drift_score', 0.0)
    else:
        m["fingerprint_nodes"] = 0
        m["fingerprint_edges"] = 0
        m["identity_drift"] = 0.0

    # --- SESSION RECALLS (granular by source) ---
    session = _load_json(".session_state.json", {})
    retrieved = session.get('retrieved', [])
    m["session_recalls"] = len(retrieved) if isinstance(retrieved, list) else 0
    by_source = session.get('recalls_by_source', {})
    m["recalls_manual"] = len(by_source.get('manual', []))
    m["recalls_start_priming"] = len(by_source.get('start_priming', []))
    m["recalls_thought_priming"] = len(by_source.get('thought_priming', []))
    m["recalls_prompt_priming"] = len(by_source.get('prompt_priming', []))

    # --- SOCIAL ---
    replies_data = _load_json("social/my_replies.json", {})
    if isinstance(replies_data, dict):
        m["social_replies_tracked"] = len(replies_data.get('replies', {}))
    else:
        m["social_replies_tracked"] = 0

    index_data = _load_json("social/social_index.json", {})
    m["social_contacts"] = index_data.get('total_contacts', 0)

    # --- PLATFORM CONTEXT ---
    tagged = 0
    total = m["memory_total"]
    for tier in ["core", "active"]:
        d = MEMORY_ROOT / tier
        if not d.exists():
            continue
        for fp in d.glob("*.md"):
            try:
                content = fp.read_text(encoding='utf-8', errors='replace')
                if 'platforms:' in content[:500]:
                    tagged += 1
            except Exception:
                pass
    m["platform_tagged"] = tagged
    m["platform_tagged_pct"] = round(tagged * 100 / total, 1) if total > 0 else 0

    # --- DECAY HISTORY ---
    decay = _load_json(".decay_history.json", {"sessions": []})
    sessions = decay.get('sessions', [])
    if sessions:
        last = sessions[-1]
        m["last_decay_count"] = last.get('decayed', 0)
        m["last_prune_count"] = last.get('pruned', 0)
    else:
        m["last_decay_count"] = 0
        m["last_prune_count"] = 0
    m["decay_sessions_recorded"] = len(sessions)

    # --- VOCABULARY BRIDGE ---
    vocab = _load_json("vocabulary_map.json", {})
    m["vocabulary_terms"] = len(vocab) if isinstance(vocab, dict) else 0

    # --- SEARCH INDEX ---
    emb_file = MEMORY_ROOT / "embeddings.json"
    if emb_file.exists():
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-c",
                 "import json;d=json.load(open('embeddings.json','r',encoding='utf-8'));print(len(d.get('memories',{})))"],
                capture_output=True, text=True, timeout=10, cwd=str(MEMORY_ROOT)
            )
            m["search_indexed"] = int(result.stdout.strip()) if result.returncode == 0 else 0
        except Exception:
            m["search_indexed"] = 0
    else:
        m["search_indexed"] = 0

    # --- W-GRAPH DIMENSIONAL METRICS (Drift extension) ---
    primary_dims = ['who', 'what', 'why', 'where']
    total_wgraph_edges = 0
    for dim in primary_dims:
        graph = _load_json(f"context/{dim}.json", {})
        edge_count = _count_edges_in_graph(graph)
        m[f"wgraph_{dim}_edges"] = edge_count
        total_wgraph_edges += edge_count
    m["wgraph_total_edges"] = total_wgraph_edges

    # Count sub-views
    if CONTEXT_DIR.exists():
        subview_count = len([f for f in CONTEXT_DIR.glob("*.json")
                           if f.stem not in primary_dims and not f.stem.startswith('bridges')])
        m["wgraph_subviews"] = subview_count
    else:
        m["wgraph_subviews"] = 0

    # Bridge count
    bridges = _load_json("context/bridges.json", {})
    m["wgraph_bridges"] = len(bridges) if isinstance(bridges, dict) else 0

    # --- TWITTER METRICS (Drift extension) ---
    # Count twitter rejections specifically
    twitter_rejections = sum(1 for r in rejections
                            if isinstance(r, dict) and r.get('source') == 'twitter')
    m["twitter_rejections"] = twitter_rejections

    return snapshot


def load_vitals_log():
    return _load_json(str(VITALS_LOG), [])


def save_vitals_log(log):
    log = log[-100:]
    VITALS_LOG.write_text(json.dumps(log, indent=2), encoding='utf-8')


def record_vitals():
    snapshot = collect_vitals()
    log = load_vitals_log()
    log.append(snapshot)
    save_vitals_log(log)
    return snapshot


def get_trends(window=10):
    log = load_vitals_log()
    if len(log) < 2:
        return []

    # Dedupe by session boundary so trends reflect real changes
    session_snapshots = _dedupe_by_session(log)
    if len(session_snapshots) < 2:
        return []
    recent = session_snapshots[-window:] if len(session_snapshots) >= window else session_snapshots
    latest = recent[-1]["metrics"]
    trends = []

    for key in sorted(latest.keys()):
        values = [s["metrics"].get(key) for s in recent if key in s.get("metrics", {})]
        values = [v for v in values if v is not None and isinstance(v, (int, float))]
        if not values:
            continue

        current = values[-1]
        if len(values) < 2:
            trends.append((key, "new", current, None, "first measurement"))
            continue

        prev_avg = sum(values[:-1]) / len(values[:-1])
        prev_values = values[:-1]

        consecutive_same = 0
        for v in reversed(values):
            if v == current:
                consecutive_same += 1
            else:
                break

        if consecutive_same >= STALL_THRESHOLD:
            direction = "stalled"
            detail = f"unchanged for {consecutive_same} sessions"
        elif current > prev_avg * 1.05:
            direction = "growing"
            detail = f"+{current - prev_avg:.1f} vs avg"
        elif current < prev_avg * 0.95:
            if len(prev_values) >= 2 and all(prev_values[i] >= prev_values[i + 1] for i in range(len(prev_values) - 1)):
                direction = "declining"
            else:
                direction = "below_avg"
            detail = f"{current - prev_avg:.1f} vs avg"
        else:
            direction = "stable"
            detail = f"~{prev_avg:.1f} avg"

        trends.append((key, direction, current, round(prev_avg, 2), detail))

    return trends


def _dedupe_by_session(log):
    """Pick one snapshot per session (last entry before a >1hr gap or end of log)."""
    if not log:
        return []
    sessions = []
    for i, entry in enumerate(log):
        ts = datetime.fromisoformat(entry["timestamp"])
        is_last = (i == len(log) - 1)
        if not is_last:
            next_ts = datetime.fromisoformat(log[i + 1]["timestamp"])
            gap_hours = (next_ts - ts).total_seconds() / 3600
            if gap_hours >= 1.0:
                sessions.append(entry)
        else:
            sessions.append(entry)
    return sessions


def check_alerts():
    log = load_vitals_log()
    alerts = []

    if len(log) < 2:
        alerts.append({
            "metric": "vitals_log",
            "severity": "info",
            "message": f"Only {len(log)} snapshot(s). Need >= 2 for trends.",
            "values": []
        })
        return alerts

    # Dedupe: one snapshot per session boundary (not just last N entries)
    session_snapshots = _dedupe_by_session(log)
    recent = session_snapshots[-STALL_THRESHOLD:] if len(session_snapshots) >= STALL_THRESHOLD else session_snapshots
    metrics_to_watch = {
        "rejection_count": (True, "warn", "Taste fingerprint not building"),
        "lesson_count": (True, "warn", "No new lessons being extracted"),
        "cooccurrence_links": (True, "warn", "Co-occurrence links not growing"),
        "merkle_chain_depth": (True, "error", "Merkle chain not incrementing"),
        "social_replies_tracked": (True, "info", "No new social replies tracked"),
        "memory_total": (True, "info", "Total memory count not growing"),
        "wgraph_total_edges": (True, "warn", "W-graph edges not growing"),
    }

    for metric, (should_grow, severity, desc) in metrics_to_watch.items():
        values = [s["metrics"].get(metric) for s in recent if metric in s.get("metrics", {})]
        values = [v for v in values if v is not None]

        if len(values) < 2:
            continue

        if should_grow and len(values) >= STALL_THRESHOLD:
            if all(v == values[0] for v in values):
                alerts.append({
                    "metric": metric,
                    "severity": severity,
                    "message": f"{desc} - unchanged at {values[0]} for {len(values)} sessions",
                    "values": values
                })

        if len(values) >= DECLINE_THRESHOLD:
            tail = values[-DECLINE_THRESHOLD:]
            if all(tail[i] > tail[i + 1] for i in range(len(tail) - 1)):
                alerts.append({
                    "metric": metric,
                    "severity": "warn",
                    "message": f"{metric} declining: {' -> '.join(str(v) for v in tail)}",
                    "values": tail
                })

    # Session recalls = 0 streak (use session-deduped data)
    recall_values = [s["metrics"].get("session_recalls", 0) for s in recent]
    zero_streak = sum(1 for v in reversed(recall_values) if v == 0)
    if zero_streak >= 3:
        alerts.append({
            "metric": "session_recalls",
            "severity": "warn",
            "message": f"Session recalls 0 for {zero_streak} sessions (graph not building)",
            "values": recall_values
        })

    # Per-path recall health (check latest snapshot only)
    if log:
        latest_m = log[-1].get("metrics", {})
        paths = {
            "manual": latest_m.get("recalls_manual", 0),
            "start_priming": latest_m.get("recalls_start_priming", 0),
            "thought_priming": latest_m.get("recalls_thought_priming", 0),
            "prompt_priming": latest_m.get("recalls_prompt_priming", 0),
        }
        dead_paths = [p for p, v in paths.items() if v == 0]
        if len(dead_paths) == 4 and latest_m.get("session_recalls", 0) == 0:
            alerts.append({
                "metric": "recall_paths",
                "severity": "warn",
                "message": "All 4 recall paths at 0 this session",
                "values": list(paths.values())
            })

    # Co-occurrence pair collapse
    pair_values = [s["metrics"].get("cooccurrence_pairs", 0) for s in log[-5:]]
    pair_values = [v for v in pair_values if v > 0]
    if len(pair_values) >= 2 and pair_values[-1] < pair_values[0] * 0.8:
        alerts.append({
            "metric": "cooccurrence_pairs",
            "severity": "warn",
            "message": f"Co-occurrence pairs dropped {pair_values[0]} -> {pair_values[-1]} ({round((1 - pair_values[-1]/pair_values[0]) * 100)}% loss)",
            "values": pair_values
        })

    # Identity drift — lower threshold than Spin's 0.5
    drift_values = [s["metrics"].get("identity_drift", 0) for s in recent]
    drift_values = [v for v in drift_values if isinstance(v, (int, float))]
    if drift_values and drift_values[-1] > 0.3:
        sev = "error" if drift_values[-1] > 0.5 else "warn"
        alerts.append({
            "metric": "identity_drift",
            "severity": sev,
            "message": f"Identity drift {drift_values[-1]:.3f} {'critically ' if sev == 'error' else ''}high",
            "values": drift_values
        })

    # W-graph dimensional imbalance (Drift extension)
    dims = ['who', 'what', 'why', 'where']
    dim_edges = {}
    if log:
        latest_m = log[-1].get("metrics", {})
        for d in dims:
            val = latest_m.get(f"wgraph_{d}_edges", 0)
            if val > 0:
                dim_edges[d] = val
    if len(dim_edges) >= 2:
        max_dim = max(dim_edges.values())
        for d, v in dim_edges.items():
            if v < max_dim * 0.1:
                alerts.append({
                    "metric": f"wgraph_{d}_edges",
                    "severity": "info",
                    "message": f"W-graph '{d}' dimension sparse ({v} edges vs {max_dim} max)",
                    "values": [v]
                })

    if not alerts:
        alerts.append({
            "metric": "all_clear",
            "severity": "ok",
            "message": "All systems nominal.",
            "values": []
        })

    return alerts


def format_snapshot(snapshot, compact=False):
    m = snapshot["metrics"]
    ts = snapshot.get("timestamp", "?")

    if compact:
        parts = [
            f"mem={m.get('memory_total', '?')}",
            f"pairs={m.get('cooccurrence_pairs', '?')}",
            f"links={m.get('cooccurrence_links', '?')}",
            f"rej={m.get('rejection_count', '?')}",
            f"lessons={m.get('lesson_count', '?')}",
            f"merkle={m.get('merkle_chain_depth', '?')}",
            f"drift={m.get('identity_drift', '?')}",
            f"wg={m.get('wgraph_total_edges', '?')}",
        ]
        return f"[{ts[:19]}] {' | '.join(parts)}"

    lines = [
        f"System Vitals Snapshot - {ts}",
        "=" * 60,
        "",
        "MEMORY",
        f"  Total: {m.get('memory_total', '?')} (core={m.get('memory_core', '?')}, active={m.get('memory_active', '?')}, archive={m.get('memory_archive', '?')})",
        "",
        "CO-OCCURRENCE",
        f"  Total edges: {m.get('cooccurrence_edges_total', '?')}",
        f"  Active pairs (belief>0): {m.get('cooccurrence_pairs', '?')}",
        f"  Total strength: {m.get('cooccurrence_total_strength', '?')}",
        f"  Links (>=3.0): {m.get('cooccurrence_links', '?')}",
        "",
        "IDENTITY",
        f"  Merkle chain depth: {m.get('merkle_chain_depth', '?')} ({m.get('merkle_memory_count', '?')} attested)",
        f"  Fingerprint: {m.get('fingerprint_nodes', '?')} nodes, {m.get('fingerprint_edges', '?')} edges",
        f"  Identity drift: {m.get('identity_drift', '?')}",
        f"  Rejections: {m.get('rejection_count', '?')} (twitter: {m.get('twitter_rejections', '?')})",
        "",
        "LEARNING",
        f"  Lessons: {m.get('lesson_count', '?')}",
        f"  Session recalls: {m.get('session_recalls', '?')}",
        f"    Manual: {m.get('recalls_manual', 0)} | Start: {m.get('recalls_start_priming', 0)} | Thought: {m.get('recalls_thought_priming', 0)} | Prompt: {m.get('recalls_prompt_priming', 0)}",
        "",
        "SOCIAL",
        f"  Contacts: {m.get('social_contacts', '?')}",
        f"  Replies tracked: {m.get('social_replies_tracked', '?')}",
        "",
        "PLATFORM",
        f"  Tagged: {m.get('platform_tagged', '?')} ({m.get('platform_tagged_pct', '?')}%)",
        "",
        "W-GRAPHS (5W Dimensions)",
        f"  WHO: {m.get('wgraph_who_edges', '?')} | WHAT: {m.get('wgraph_what_edges', '?')} | WHY: {m.get('wgraph_why_edges', '?')} | WHERE: {m.get('wgraph_where_edges', '?')}",
        f"  Total: {m.get('wgraph_total_edges', '?')} edges, {m.get('wgraph_subviews', '?')} sub-views, {m.get('wgraph_bridges', '?')} bridges",
        "",
        "GRAPH HEALTH",
        f"  Last decay: {m.get('last_decay_count', '?')} | pruned: {m.get('last_prune_count', '?')}",
        f"  Decay sessions: {m.get('decay_sessions_recorded', '?')}",
        "",
        "SEARCH & VOCABULARY",
        f"  Indexed: {m.get('search_indexed', '?')} | Vocab terms: {m.get('vocabulary_terms', '?')}",
    ]
    return "\n".join(lines)


def format_trends(trends):
    if not trends:
        return "No trend data. Need >= 2 snapshots."

    icons = {
        "growing": "+", "stable": "=", "stalled": "!",
        "declining": "-", "below_avg": "~", "new": "*",
    }

    lines = ["System Vitals Trends", "=" * 60, ""]
    for metric, direction, current, prev_avg, detail in trends:
        icon = icons.get(direction, "?")
        lines.append(f"  [{icon}] {metric:<30s} {str(current):<10} {direction:<12s} {detail}")

    lines.append("")
    lines.append("[+] growing  [=] stable  [!] stalled  [-] declining  [~] below avg  [*] new")
    return "\n".join(lines)


def format_alerts(alerts):
    icons = {"ok": "OK", "info": "INFO", "warn": "WARN", "error": "ERR"}

    lines = ["System Vitals Alerts", "=" * 55, ""]
    for alert in alerts:
        icon = icons.get(alert["severity"], "?")
        lines.append(f"  [{icon}] {alert['message']}")
        if alert["values"]:
            lines.append(f"         values: {alert['values']}")
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    args = sys.argv[1:]
    if not args or args[0] in ('--help', '-h'):
        print(__doc__.strip())
        return

    cmd = args[0]

    if cmd == 'record':
        snapshot = record_vitals()
        log = load_vitals_log()
        print(f"Vitals recorded. Snapshot #{len(log)} at {snapshot['timestamp'][:19]}")
        print(format_snapshot(snapshot, compact=True))

    elif cmd == 'latest':
        log = load_vitals_log()
        if not log:
            print("No vitals recorded yet. Run: python system_vitals.py record")
            return
        print(format_snapshot(log[-1]))

    elif cmd == 'trends':
        window = int(args[1]) if len(args) > 1 else 10
        trends = get_trends(window)
        print(format_trends(trends))

    elif cmd == 'alerts':
        alerts = check_alerts()
        print(format_alerts(alerts))

    elif cmd == 'history':
        n = int(args[1]) if len(args) > 1 else 5
        log = load_vitals_log()
        if not log:
            print("No vitals recorded yet.")
            return
        print(f"Last {min(n, len(log))} snapshots:")
        print()
        for s in log[-n:]:
            print(format_snapshot(s, compact=True))

    elif cmd == 'collect':
        snapshot = collect_vitals()
        print(format_snapshot(snapshot))

    else:
        print(f"Unknown command: {cmd}")
        print("Available: record, latest, trends, alerts, history, collect")
        sys.exit(1)


if __name__ == '__main__':
    main()
