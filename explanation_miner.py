#!/usr/bin/env python3
"""
Explanation Miner — Strategy Extraction from Search Traces

Phase 5 of the Voss Review implementation plan.
"A goldmine of metacognitive data that no module reads."

Reads explanation traces from the DB, correlates pipeline steps with
Q-value outcomes, and extracts strategy heuristics.

Storage: DB key_value_store, key pattern '.strategy.*'

Usage:
    python explanation_miner.py mine              # Extract strategies from traces
    python explanation_miner.py strategies         # Show active strategies
    python explanation_miner.py evaluate           # Check strategy validity
    python explanation_miner.py stats              # Mining statistics
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path(__file__).parent
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

KV_STRATEGIES = '.strategy.active'
KV_MINE_HISTORY = '.strategy.mine_history'

MIN_SAMPLES = 10     # Minimum traces for a pattern to be significant
MIN_CONFIDENCE = 0.6  # Minimum success rate for a strategy


def _get_db():
    from db_adapter import get_db
    return get_db()


def _now_iso():
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


# ============================================================
# Mining: Extract patterns from explanation traces
# ============================================================

def _get_traces(limit: int = 200) -> list[dict]:
    """Get recent explanation traces for search operations."""
    from explanation import get_store
    store = get_store()
    return store.get_last(n=limit, module='semantic_search', operation='search')


def _get_q_values_for_results(result_ids: list[str]) -> dict:
    """Get Q-values for search result memories."""
    try:
        from q_value_engine import get_q_values
        return get_q_values(result_ids)
    except Exception:
        return {}


def mine_strategies(limit: int = 200) -> list[dict]:
    """
    Extract strategy heuristics from explanation traces.

    Process:
    1. Load recent search explanation traces
    2. For each trace, get Q-values of the top results
    3. Correlate pipeline step factors with Q-value outcomes
    4. Identify patterns where specific steps correlate with better/worse outcomes

    Returns list of extracted strategy dicts.
    """
    traces = _get_traces(limit)
    if len(traces) < MIN_SAMPLES:
        return []

    # Collect per-factor statistics
    # factor_stats[factor_name] = {
    #   'present': [(avg_q, result_count), ...],
    #   'absent': [(avg_q, result_count), ...],
    #   'values': {value: [(avg_q, result_count), ...]}
    # }
    factor_stats = defaultdict(lambda: {
        'present': [],
        'absent': [],
        'high_value': [],
        'low_value': [],
    })

    # Track which factors appear in which traces
    all_factors = set()
    for trace in traces:
        for step in (trace.get('reasoning') or []):
            factor = step.get('factor', '')
            if factor:
                all_factors.add(factor)

    for trace in traces:
        output = trace.get('output') or {}
        top_ids = output.get('top_ids', [])
        top_scores = output.get('top_scores', [])

        if not top_ids:
            continue

        # Get Q-values for results
        q_vals = _get_q_values_for_results(top_ids)
        if not q_vals:
            continue

        avg_q = sum(q_vals.get(mid, 0.5) for mid in top_ids) / len(top_ids)
        result_count = output.get('result_count', len(top_ids))

        # Track factors present in this trace
        present_factors = set()
        for step in (trace.get('reasoning') or []):
            factor = step.get('factor', '')
            if not factor:
                continue
            present_factors.add(factor)

            # Track factor value for high/low analysis
            try:
                value = float(step.get('value', 0))
                if value > 0:
                    factor_stats[factor]['high_value'].append((avg_q, value))
                else:
                    factor_stats[factor]['low_value'].append((avg_q, value))
            except (ValueError, TypeError):
                pass

            factor_stats[factor]['present'].append((avg_q, result_count))

        # Track absence
        for factor in all_factors - present_factors:
            factor_stats[factor]['absent'].append((avg_q, result_count))

    # Extract strategies from statistical patterns
    strategies = []

    for factor, stats in factor_stats.items():
        present_qs = [q for q, _ in stats['present']]
        absent_qs = [q for q, _ in stats['absent']]

        if len(present_qs) < MIN_SAMPLES:
            continue

        avg_present = sum(present_qs) / len(present_qs)

        # Strategy 1: Factor presence vs absence
        if len(absent_qs) >= MIN_SAMPLES:
            avg_absent = sum(absent_qs) / len(absent_qs)
            diff = avg_present - avg_absent

            if abs(diff) > 0.02:  # Meaningful difference
                direction = 'positive' if diff > 0 else 'negative'
                strategies.append({
                    'type': 'factor_impact',
                    'factor': factor,
                    'direction': direction,
                    'avg_q_present': round(avg_present, 4),
                    'avg_q_absent': round(avg_absent, 4),
                    'delta': round(diff, 4),
                    'samples_present': len(present_qs),
                    'samples_absent': len(absent_qs),
                    'confidence': round(min(len(present_qs), len(absent_qs)) / MIN_SAMPLES, 2),
                    'description': (
                        f'When {factor} is active, avg Q-value is {avg_present:.3f} '
                        f'vs {avg_absent:.3f} without ({diff:+.3f})'
                    ),
                })

        # Strategy 2: High vs low factor values
        high_qs = [q for q, v in stats['high_value'] if v > 3]
        low_qs = [q for q, v in stats['high_value'] if v <= 3]

        if len(high_qs) >= 5 and len(low_qs) >= 5:
            avg_high = sum(high_qs) / len(high_qs)
            avg_low = sum(low_qs) / len(low_qs)
            diff = avg_high - avg_low

            if abs(diff) > 0.02:
                strategies.append({
                    'type': 'value_correlation',
                    'factor': factor,
                    'direction': 'positive' if diff > 0 else 'negative',
                    'avg_q_high': round(avg_high, 4),
                    'avg_q_low': round(avg_low, 4),
                    'delta': round(diff, 4),
                    'samples': len(high_qs) + len(low_qs),
                    'description': (
                        f'Higher {factor} values correlate with '
                        f'{"better" if diff > 0 else "worse"} outcomes ({diff:+.3f} Q-value)'
                    ),
                })

    # Sort by absolute impact
    strategies.sort(key=lambda s: abs(s.get('delta', 0)), reverse=True)

    # Store strategies
    db = _get_db()
    db.kv_set(KV_STRATEGIES, {
        'strategies': strategies,
        'mined_at': _now_iso(),
        'traces_analyzed': len(traces),
    })

    # Append to history
    history = db.kv_get(KV_MINE_HISTORY) or {'entries': []}
    history['entries'].append({
        'ts': _now_iso(),
        'traces': len(traces),
        'strategies_found': len(strategies),
    })
    history['entries'] = history['entries'][-50:]
    db.kv_set(KV_MINE_HISTORY, history)

    return strategies


def get_strategies() -> list[dict]:
    """Get currently active strategies."""
    db = _get_db()
    data = db.kv_get(KV_STRATEGIES) or {}
    return data.get('strategies', [])


def evaluate_strategies() -> list[dict]:
    """
    Evaluate which strategies are still valid by checking recent traces.
    Returns list of strategies with updated validity scores.
    """
    strategies = get_strategies()
    if not strategies:
        return []

    # Re-mine with recent data to check consistency
    recent = mine_strategies(limit=50)
    recent_map = {(s['type'], s['factor']): s for s in recent}

    evaluated = []
    for s in strategies:
        key = (s['type'], s['factor'])
        if key in recent_map:
            recent_s = recent_map[key]
            # Check if direction is consistent
            consistent = recent_s['direction'] == s['direction']
            evaluated.append({
                **s,
                'still_valid': consistent,
                'recent_delta': recent_s.get('delta', 0),
                'trend': 'stable' if consistent else 'reversed',
            })
        else:
            evaluated.append({
                **s,
                'still_valid': False,
                'trend': 'insufficient_data',
            })

    return evaluated


def get_stats() -> dict:
    """Get mining statistics."""
    db = _get_db()
    data = db.kv_get(KV_STRATEGIES) or {}
    history = db.kv_get(KV_MINE_HISTORY) or {'entries': []}

    strategies = data.get('strategies', [])

    return {
        'active_strategies': len(strategies),
        'last_mined': data.get('mined_at', 'never'),
        'traces_analyzed': data.get('traces_analyzed', 0),
        'mine_runs': len(history.get('entries', [])),
        'top_factors': [
            f"{s['factor']} ({s['direction']}, delta={s.get('delta', 0):+.3f})"
            for s in strategies[:5]
        ],
    }


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Explanation Miner — Strategy Extraction')
    sub = parser.add_subparsers(dest='command')

    p_mine = sub.add_parser('mine', help='Extract strategies from traces')
    p_mine.add_argument('--limit', type=int, default=200, help='Max traces to analyze')

    sub.add_parser('strategies', help='Show active strategies')
    sub.add_parser('evaluate', help='Check strategy validity')
    sub.add_parser('stats', help='Mining statistics')

    args = parser.parse_args()

    if args.command == 'mine':
        strategies = mine_strategies(limit=args.limit)
        if strategies:
            print(f'Extracted {len(strategies)} strategy heuristic(s):\n')
            for s in strategies:
                print(f'  [{s["type"]}] {s["factor"]}')
                print(f'    {s["description"]}')
                print(f'    Samples: {s.get("samples_present", s.get("samples", "?"))} | '
                      f'Confidence: {s.get("confidence", "?")}')
                print()
        else:
            print(f'Not enough data yet (need >= {MIN_SAMPLES} traces)')

    elif args.command == 'strategies':
        strategies = get_strategies()
        if not strategies:
            print('No strategies mined yet. Run: python explanation_miner.py mine')
        else:
            print(f'{len(strategies)} active strategy heuristic(s):\n')
            for s in strategies:
                print(f'  {s["factor"]}: {s["description"]}')

    elif args.command == 'evaluate':
        evaluated = evaluate_strategies()
        if not evaluated:
            print('No strategies to evaluate.')
        else:
            valid = [e for e in evaluated if e.get('still_valid')]
            invalid = [e for e in evaluated if not e.get('still_valid')]
            print(f'{len(valid)} valid, {len(invalid)} invalid/stale:\n')
            for e in evaluated:
                status = 'VALID' if e.get('still_valid') else 'STALE'
                print(f'  [{status}] {e["factor"]}: {e.get("trend", "?")}')

    elif args.command == 'stats':
        s = get_stats()
        print('Explanation Mining Statistics:')
        print(f'  Active strategies: {s["active_strategies"]}')
        print(f'  Last mined: {s["last_mined"]}')
        print(f'  Traces analyzed: {s["traces_analyzed"]}')
        print(f'  Total mine runs: {s["mine_runs"]}')
        if s['top_factors']:
            print(f'\n  Top factors:')
            for f in s['top_factors']:
                print(f'    {f}')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
