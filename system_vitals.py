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

    # --- MEMORY COUNTS (DB) ---
    from db_adapter import get_db
    import psycopg2.extras
    db = get_db()
    stats = db.comprehensive_stats()
    m["memory_total"] = stats.get('total_memories', 0)
    m["memory_core"] = stats.get('memories', {}).get('core', 0)
    m["memory_active"] = stats.get('memories', {}).get('active', 0)
    m["memory_archive"] = stats.get('memories', {}).get('archive', 0)

    # --- CO-OCCURRENCE (DB) ---
    edge_stats = db.edge_stats()
    m["cooccurrence_pairs"] = edge_stats.get('total_edges', 0)
    m["cooccurrence_total_strength"] = round(edge_stats.get('total_belief', 0), 2)
    m["cooccurrence_links"] = edge_stats.get('strong_links', 0)
    m["cooccurrence_edges_total"] = edge_stats.get('total_edges', 0)

    # --- REJECTION LOG (DB) ---
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {db._table('rejections')}")
            m["rejection_count"] = cur.fetchone()[0]

    # Load rejections for twitter filter below
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT source FROM {db._table('rejections')}")
            rejections = [dict(r) for r in cur.fetchall()]

    # --- LESSONS (DB) ---
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {db._table('lessons')}")
            m["lesson_count"] = cur.fetchone()[0]

    # --- MERKLE CHAIN (DB) ---
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT data FROM {db._table('attestations')}
                ORDER BY timestamp DESC LIMIT 1
            """)
            att_row = cur.fetchone()
    if att_row and att_row.get('data'):
        att_data = att_row['data']
        m["merkle_chain_depth"] = att_data.get('chain_depth', 0)
        m["merkle_memory_count"] = att_data.get('memory_count', 0)
    else:
        m["merkle_chain_depth"] = 0
        m["merkle_memory_count"] = 0

    # --- COGNITIVE FINGERPRINT (DB KV) ---
    fp_history_raw = db.kv_get('.fingerprint_history')
    fp_history = []
    if fp_history_raw:
        fp_history = json.loads(fp_history_raw) if isinstance(fp_history_raw, str) else fp_history_raw
    if isinstance(fp_history, list) and fp_history:
        latest_fp = fp_history[-1]
        m["fingerprint_nodes"] = latest_fp.get('node_count', 0)
        m["fingerprint_edges"] = latest_fp.get('edge_count', 0)
        m["identity_drift"] = latest_fp.get('drift_score', 0.0)
    else:
        m["fingerprint_nodes"] = 0
        m["fingerprint_edges"] = 0
        m["identity_drift"] = 0.0

    # --- SESSION RECALLS (granular by source, from DB KV) ---
    session_raw = db.kv_get('.session_state')
    session = {}
    if session_raw:
        session = json.loads(session_raw) if isinstance(session_raw, str) else session_raw
    retrieved = session.get('retrieved', [])
    m["session_recalls"] = len(retrieved) if isinstance(retrieved, list) else 0
    by_source = session.get('recalls_by_source', {})
    m["recalls_manual"] = len(by_source.get('manual', []))
    m["recalls_start_priming"] = len(by_source.get('start_priming', []))
    m["recalls_thought_priming"] = len(by_source.get('thought_priming', []))
    m["recalls_prompt_priming"] = len(by_source.get('prompt_priming', []))

    # --- SOCIAL (DB KV) ---
    # Keys match social_memory.py: KV_MY_REPLIES = '.social_my_replies', KV_INDEX = '.social_index'
    replies_data = db.kv_get('.social_my_replies') or {}
    if isinstance(replies_data, str):
        replies_data = json.loads(replies_data)
    m["social_replies_tracked"] = len(replies_data.get('replies', {})) if isinstance(replies_data, dict) else 0

    index_data = db.kv_get('.social_index') or {}
    if isinstance(index_data, str):
        index_data = json.loads(index_data)
    m["social_contacts"] = index_data.get('total_contacts', 0)

    # --- PLATFORM CONTEXT (DB) ---
    tagged = 0
    total = m["memory_total"]
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) FROM {db._table('memories')}
                WHERE type IN ('core', 'active')
                AND platform_context IS NOT NULL
                AND array_length(platform_context, 1) > 0
            """)
            tagged = cur.fetchone()[0]
    m["platform_tagged"] = tagged
    m["platform_tagged_pct"] = round(tagged * 100 / total, 1) if total > 0 else 0

    # --- DECAY HISTORY (DB KV) ---
    decay_raw = db.kv_get('.decay_history')
    decay = {"sessions": []}
    if decay_raw:
        decay = json.loads(decay_raw) if isinstance(decay_raw, str) else decay_raw
    decay_sessions = decay.get('sessions', [])
    if decay_sessions:
        last = decay_sessions[-1]
        m["last_decay_count"] = last.get('decayed', 0)
        m["last_prune_count"] = last.get('pruned', 0)
    else:
        m["last_decay_count"] = 0
        m["last_prune_count"] = 0
    m["decay_sessions_recorded"] = len(decay_sessions)

    # --- VOCABULARY BRIDGE ---
    vocab_raw = db.kv_get('vocabulary_map')
    if vocab_raw:
        vocab = json.loads(vocab_raw) if isinstance(vocab_raw, str) else vocab_raw
        m["vocabulary_terms"] = len(vocab) if isinstance(vocab, dict) else 0
    else:
        # Fallback: vocabulary_map might not be in KV yet, try file
        vocab = _load_json("vocabulary_map.json", {})
        m["vocabulary_terms"] = len(vocab) if isinstance(vocab, dict) else 0

    # --- SEARCH INDEX (DB) ---
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {db._table('text_embeddings')}")
            m["search_indexed"] = cur.fetchone()[0]

    # --- W-GRAPH DIMENSIONAL METRICS (DB) ---
    primary_dims = ['who', 'what', 'why', 'where']
    total_wgraph_edges = 0
    subview_count = 0
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT dimension, sub_view, edge_count
                FROM {db._table('context_graphs')}
            """)
            for row in cur.fetchall():
                dim = row['dimension']
                sub = row.get('sub_view', '')
                ec = row.get('edge_count', 0) or 0
                if dim in primary_dims and not sub:
                    m[f"wgraph_{dim}_edges"] = ec
                    total_wgraph_edges += ec
                elif dim not in ('bridges',) and sub:
                    subview_count += 1
    # Ensure all dims have a value
    for dim in primary_dims:
        if f"wgraph_{dim}_edges" not in m:
            m[f"wgraph_{dim}_edges"] = 0
    m["wgraph_total_edges"] = total_wgraph_edges
    m["wgraph_subviews"] = subview_count

    # Bridge count (DB)
    bridge_row = db.get_context_graph('bridges', '') if hasattr(db, 'get_context_graph') else None
    m["wgraph_bridges"] = len(bridge_row.get('edges', {})) if bridge_row and bridge_row.get('edges') else 0

    # --- TWITTER METRICS (Drift extension) ---
    # Count twitter rejections specifically
    twitter_rejections = sum(1 for r in rejections
                            if isinstance(r, dict) and r.get('source') == 'twitter')
    m["twitter_rejections"] = twitter_rejections

    return snapshot


def load_vitals_log():
    from db_adapter import get_db
    import psycopg2.extras

    db = get_db()
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT timestamp, metrics FROM {db._table('vitals_log')} ORDER BY timestamp DESC LIMIT 100")
            rows = cur.fetchall()
            if rows:
                return [{'timestamp': r['timestamp'].isoformat(), 'metrics': r['metrics']} for r in reversed(rows)]
    return []


def save_vitals_log(log):
    """Legacy file writer — kept for backward compat but DB is primary via record_vitals()."""
    pass  # DB is the only store now. record_vitals() handles persistence.


def record_vitals():
    snapshot = collect_vitals()
    # Write to DB
    from db_adapter import get_db
    get_db().record_vitals(snapshot['metrics'])
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
