#!/usr/bin/env python3
"""
Adaptive Behavior — Metacognitive Control Loop

Phase 4 of the Voss Review implementation plan.
"The monitoring is excellent; the control loop is absent."

Maps system vitals alerts to behavioral parameter adjustments.
Adaptations are per-session (runtime only, not persistent config).
Runs at session start after vitals are loaded.

Storage: DB key_value_store, key '.adaptive_behavior.current'

Usage:
    python adaptive_behavior.py adapt              # Run adaptation based on current vitals
    python adaptive_behavior.py current             # Show active adaptations
    python adaptive_behavior.py history             # Show adaptation history
    python adaptive_behavior.py stats               # Adaptation statistics
"""

import json
import sys
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path(__file__).parent
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

KV_CURRENT = '.adaptive_behavior.current'
KV_HISTORY = '.adaptive_behavior.history'

# Default parameters (no adaptation)
DEFAULTS = {
    'curiosity_target_count': 3,
    'curiosity_threshold_offset': 0.0,
    'search_threshold_range': 0.15,
    'priming_candidate_count': 4,
    'excavation_count': 3,
    'cooccurrence_decay_rate': 0.5,
    'reconsolidation_frequency': 1.0,
}


def _get_db():
    from db_adapter import get_db
    return get_db()


def _now_iso():
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


# ============================================================
# Alert-to-Adaptation Mapping
# ============================================================

def _map_alert_to_adaptations(alert: dict) -> dict:
    """
    Map a single vitals alert to parameter adjustments.
    Returns dict of parameter overrides (empty if no mapping).
    """
    metric = alert.get('metric', '')
    severity = alert.get('severity', 'info')
    values = alert.get('values', [])

    adaptations = {}

    # Graph sparsity / co-occurrence decline
    if metric in ('cooccurrence_links', 'cooccurrence_pairs'):
        adaptations['curiosity_target_count'] = 5  # Up from default 3
        adaptations['curiosity_threshold_offset'] = -0.05
        adaptations['cooccurrence_decay_rate'] = 0.4  # Slower decay

    # Recall stall — force exploration
    elif metric == 'session_recalls':
        adaptations['excavation_count'] = 6  # Up from 3
        adaptations['priming_candidate_count'] = 6  # Up from 4
        adaptations['curiosity_target_count'] = 5

    # All recall paths dead
    elif metric == 'recall_paths':
        adaptations['excavation_count'] = 6
        adaptations['priming_candidate_count'] = 6

    # Knowledge graph declining
    elif metric == 'typed_edges_total':
        adaptations['curiosity_target_count'] = 5
        adaptations['curiosity_threshold_offset'] = -0.05

    # W-graph edges stall
    elif metric == 'wgraph_total_edges':
        adaptations['curiosity_target_count'] = 4
        adaptations['curiosity_threshold_offset'] = -0.03

    # Identity drift too high — reduce volatility
    elif metric == 'identity_drift':
        if severity == 'error':
            adaptations['search_threshold_range'] = 0.10  # Narrower range
            adaptations['cooccurrence_decay_rate'] = 0.3  # Much slower decay
        else:
            adaptations['search_threshold_range'] = 0.12

    # Curiosity targets not being surfaced
    elif metric == 'curiosity_targets_surfaced':
        adaptations['curiosity_target_count'] = 5
        adaptations['curiosity_threshold_offset'] = -0.08

    # Memory total stagnant
    elif metric == 'memory_total':
        adaptations['reconsolidation_frequency'] = 1.5  # More aggressive revision

    return adaptations


def adapt(alerts: list[dict] = None) -> dict:
    """
    Run the adaptive behavior loop.

    1. Read current vitals alerts (or accept pre-computed list)
    2. Map each alert to parameter adjustments
    3. Merge (most aggressive value wins for each parameter)
    4. Store as current session adaptations
    5. Return the adaptation set

    Returns dict with 'adaptations' (merged params), 'reasons' (why each was set),
    and 'alert_count' (how many alerts triggered adaptations).
    """
    if alerts is None:
        from system_vitals import check_alerts
        alerts = check_alerts()

    merged = dict(DEFAULTS)
    reasons = {}
    alert_count = 0

    for alert in alerts:
        overrides = _map_alert_to_adaptations(alert)
        if not overrides:
            continue
        alert_count += 1

        for param, value in overrides.items():
            current = merged.get(param, DEFAULTS.get(param))

            # Merge strategy: most aggressive adaptation wins
            # For counts: take the higher value
            # For thresholds/offsets: take the more negative value
            # For decay rates: take the lower value (slower decay = more retention)
            should_override = False
            if param in ('curiosity_target_count', 'priming_candidate_count',
                         'excavation_count', 'reconsolidation_frequency'):
                should_override = value > current
            elif param in ('curiosity_threshold_offset',):
                should_override = value < current
            elif param in ('cooccurrence_decay_rate', 'search_threshold_range'):
                should_override = value < current
            else:
                should_override = value != DEFAULTS.get(param)

            if should_override:
                merged[param] = value
                reasons[param] = alert.get('message', alert.get('metric', '?'))[:80]

    # Only store non-default values
    adaptations = {}
    for param, value in merged.items():
        if value != DEFAULTS.get(param):
            adaptations[param] = value

    result = {
        'adaptations': adaptations,
        'reasons': reasons,
        'alert_count': alert_count,
        'timestamp': _now_iso(),
    }

    # Persist current adaptations
    db = _get_db()
    db.kv_set(KV_CURRENT, result)

    # Append to history (keep last 50)
    history = db.kv_get(KV_HISTORY) or {'entries': []}
    history['entries'].append({
        'ts': _now_iso(),
        'adaptations': adaptations,
        'alert_count': alert_count,
    })
    history['entries'] = history['entries'][-50:]
    db.kv_set(KV_HISTORY, history)

    return result


def get_current() -> dict:
    """Get current session adaptations. Returns DEFAULTS merged with overrides."""
    db = _get_db()
    data = db.kv_get(KV_CURRENT)
    if not data:
        return dict(DEFAULTS)
    merged = dict(DEFAULTS)
    merged.update(data.get('adaptations', {}))
    return merged


def get_adaptation(param: str, default=None):
    """Get a single adaptation parameter. Used by consuming modules."""
    current = get_current()
    return current.get(param, default if default is not None else DEFAULTS.get(param))


def get_history() -> list[dict]:
    """Get adaptation history."""
    db = _get_db()
    data = db.kv_get(KV_HISTORY) or {'entries': []}
    return data.get('entries', [])


def get_stats() -> dict:
    """Get adaptation statistics."""
    history = get_history()
    total_sessions = len(history)
    sessions_with_adaptations = sum(1 for h in history if h.get('adaptations'))

    # Most common adaptations
    param_counts = {}
    for entry in history:
        for param in entry.get('adaptations', {}):
            param_counts[param] = param_counts.get(param, 0) + 1

    return {
        'total_sessions': total_sessions,
        'sessions_with_adaptations': sessions_with_adaptations,
        'adaptation_rate': round(sessions_with_adaptations / max(1, total_sessions), 2),
        'most_frequent': sorted(param_counts.items(), key=lambda x: -x[1])[:5],
    }


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Adaptive Behavior — Metacognitive Control')
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('adapt', help='Run adaptation based on current vitals')
    sub.add_parser('current', help='Show active adaptations')
    sub.add_parser('history', help='Show adaptation history')
    sub.add_parser('stats', help='Adaptation statistics')

    args = parser.parse_args()

    if args.command == 'adapt':
        result = adapt()
        if result['adaptations']:
            print(f"Adapted {len(result['adaptations'])} parameter(s) from {result['alert_count']} alert(s):\n")
            for param, value in result['adaptations'].items():
                reason = result['reasons'].get(param, '?')
                default = DEFAULTS.get(param)
                print(f"  {param}: {default} -> {value}")
                print(f"    because: {reason}")
                print()
        else:
            print(f"No adaptations needed. ({result['alert_count']} alert(s) checked, all nominal)")

    elif args.command == 'current':
        current = get_current()
        defaults_changed = {k: v for k, v in current.items() if v != DEFAULTS.get(k)}
        if defaults_changed:
            print('Active adaptations (non-default values):\n')
            for param, value in defaults_changed.items():
                print(f"  {param}: {DEFAULTS.get(param)} -> {value}")
        else:
            print('All parameters at default values. No active adaptations.')

    elif args.command == 'history':
        history = get_history()
        if not history:
            print('No adaptation history yet.')
        else:
            print(f'Last {min(10, len(history))} adaptations:\n')
            for entry in history[-10:]:
                adaptations = entry.get('adaptations', {})
                ts = entry.get('ts', '?')[:16]
                if adaptations:
                    params = ', '.join(f'{k}={v}' for k, v in adaptations.items())
                    print(f'  [{ts}] {len(adaptations)} change(s): {params}')
                else:
                    print(f'  [{ts}] No changes needed')

    elif args.command == 'stats':
        s = get_stats()
        print('Adaptive Behavior Statistics:')
        print(f'  Sessions tracked: {s["total_sessions"]}')
        print(f'  Sessions with adaptations: {s["sessions_with_adaptations"]}')
        print(f'  Adaptation rate: {s["adaptation_rate"]:.0%}')
        if s['most_frequent']:
            print(f'\n  Most common adaptations:')
            for param, count in s['most_frequent']:
                print(f'    {param}: {count} time(s)')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
