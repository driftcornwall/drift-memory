#!/usr/bin/env python3
"""
Self-Narrative Module — Higher-Order Thought for Drift

R9 from the Cognitive Neuroscience Review (2026-02-16).
"The system collects extensive data about itself but has no module
that synthesizes this into a self-narrative."

This module creates a queryable self-model by pulling from:
1. Cognitive state (curiosity, confidence, focus, arousal, satisfaction)
2. Cognitive fingerprint (identity drift, top hubs)
3. Rejection log (taste profile)
4. Adaptive behavior (active adaptations, evaluation history)
5. Explanation miner (learned strategies)
6. Reconsolidation (revision queue)

The narrative is injected into session start context and available
via toolkit commands: `self` and `self-query`.

Usage:
    python self_narrative.py generate      # Full self-model
    python self_narrative.py narrative      # Just the 200-word narrative
    python self_narrative.py query "..."    # Answer a self-question
"""

import sys
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path(__file__).parent
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

KV_SELF_NARRATIVE = '.self_narrative.current'


def _get_db():
    from db_adapter import get_db
    return get_db()


def _safe_call(fn, default=None):
    """Call a function, return default on any error."""
    try:
        return fn()
    except Exception:
        return default


# ============================================================
# Subsystem Data Collectors
# ============================================================

def _get_cognitive_summary() -> dict:
    """Pull current cognitive state with uncertainty."""
    def _inner():
        from cognitive_state import get_state
        state = get_state()
        dims = {}
        for dim in ('curiosity', 'confidence', 'focus', 'arousal', 'satisfaction'):
            val = getattr(state, dim)
            unc = state.get_uncertainty(dim)
            dims[dim] = {'value': val, 'uncertainty': round(unc, 3)}
        dominant = state.dominant()
        return {
            'dimensions': dims,
            'dominant': dominant,
            'mean_uncertainty': round(state.mean_uncertainty, 3),
            'event_count': state.event_count,
        }
    return _safe_call(_inner, {'dimensions': {}, 'dominant': 'unknown', 'mean_uncertainty': 0.5})


def _get_identity_summary() -> dict:
    """Pull cognitive fingerprint drift and top hubs."""
    def _inner():
        db = _get_db()
        # Get latest fingerprint from KV
        fp = db.kv_get('.cognitive_fingerprint') or {}
        drift = fp.get('drift_score', 0.0)
        node_count = fp.get('node_count', 0)
        edge_count = fp.get('edge_count', 0)
        top_hubs = fp.get('top_hubs', [])[:5]
        return {
            'drift_score': drift,
            'node_count': node_count,
            'edge_count': edge_count,
            'top_hubs': top_hubs,
            'health': 'healthy' if drift < 0.15 else 'volatile' if drift < 0.30 else 'unstable',
        }
    return _safe_call(_inner, {'drift_score': 0, 'health': 'unknown'})


def _get_taste_summary() -> dict:
    """Pull rejection log taste profile."""
    def _inner():
        from rejection_log import get_rejections
        rejections = get_rejections(limit=100)
        if not rejections:
            return {'total': 0, 'categories': {}}
        from collections import Counter
        cats = Counter(r.get('category', 'unknown') for r in rejections)
        return {
            'total': len(rejections),
            'categories': dict(cats.most_common(5)),
            'top_reason': cats.most_common(1)[0][0] if cats else 'none',
        }
    return _safe_call(_inner, {'total': 0, 'categories': {}})


def _get_adaptation_summary() -> dict:
    """Pull adaptive behavior status and evaluation history."""
    def _inner():
        from adaptive_behavior import get_current, get_history, DEFAULTS
        current = get_current()
        adapted = {k: v for k, v in current.items() if v != DEFAULTS.get(k)}
        history = get_history()
        # Get last evaluation
        last_eval = None
        for entry in reversed(history):
            if 'evaluation' in entry:
                last_eval = entry['evaluation']
                break
        return {
            'active_adaptations': len(adapted),
            'adapted_params': adapted,
            'last_effectiveness': last_eval.get('effectiveness') if last_eval else None,
            'sessions_tracked': len(history),
        }
    return _safe_call(_inner, {'active_adaptations': 0, 'adapted_params': {}})


def _get_strategy_summary() -> dict:
    """Pull learned retrieval strategies."""
    def _inner():
        from explanation_miner import get_strategies
        strategies = get_strategies()
        return {
            'count': len(strategies),
            'strategies': [
                {'factor': s.get('factor', '?'), 'direction': s.get('direction', '?'),
                 'delta': round(s.get('delta', 0), 3)}
                for s in strategies[:5]
            ],
        }
    return _safe_call(_inner, {'count': 0, 'strategies': []})


def _get_reconsolidation_summary() -> dict:
    """Pull reconsolidation queue status."""
    def _inner():
        from reconsolidation import get_stats
        stats = get_stats()
        return {
            'candidates_ready': stats.get('candidates_ready', 0),
            'total_revised': stats.get('total_revised', 0),
            'queue_length': stats.get('queue_length', 0),
        }
    return _safe_call(_inner, {'candidates_ready': 0, 'total_revised': 0})


# ============================================================
# Narrative Synthesis
# ============================================================

def generate() -> dict:
    """Generate the full self-model from all subsystems."""
    model = {
        'cognitive_state': _get_cognitive_summary(),
        'identity': _get_identity_summary(),
        'taste': _get_taste_summary(),
        'adaptations': _get_adaptation_summary(),
        'strategies': _get_strategy_summary(),
        'reconsolidation': _get_reconsolidation_summary(),
        'generated_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
    }
    model['narrative'] = _synthesize_narrative(model)

    # Store for later query
    db = _get_db()
    db.kv_set(KV_SELF_NARRATIVE, model)

    return model


def _synthesize_narrative(model: dict) -> str:
    """Generate human-readable self-narrative from model data."""
    parts = []

    # Cognitive state
    cog = model.get('cognitive_state', {})
    dims = cog.get('dimensions', {})
    if dims:
        dominant = cog.get('dominant', 'unknown')
        unc = cog.get('mean_uncertainty', 0.5)
        unc_label = 'low' if unc < 0.3 else 'moderate' if unc < 0.5 else 'high'

        # Find notable dimensions
        high_dims = [d for d, v in dims.items() if v.get('value', 0.5) > 0.65]
        low_dims = [d for d, v in dims.items() if v.get('value', 0.5) < 0.35]

        state_desc = f"Dominant state: {dominant} ({unc_label} uncertainty)."
        if high_dims:
            state_desc += f" Elevated: {', '.join(high_dims)}."
        if low_dims:
            state_desc += f" Depressed: {', '.join(low_dims)}."
        parts.append(state_desc)

    # Identity drift
    identity = model.get('identity', {})
    drift = identity.get('drift_score', 0)
    health = identity.get('health', 'unknown')
    nodes = identity.get('node_count', 0)
    edges = identity.get('edge_count', 0)
    if nodes:
        parts.append(f"Identity: {nodes} nodes, {edges} edges, drift={drift:.4f} ({health}).")

    # Adaptations
    adapt = model.get('adaptations', {})
    active = adapt.get('active_adaptations', 0)
    if active:
        params = ', '.join(f'{k}={v}' for k, v in adapt.get('adapted_params', {}).items())
        eff = adapt.get('last_effectiveness')
        eff_str = f" Last effectiveness: {eff:.0%}." if eff is not None else ""
        parts.append(f"Active adaptations ({active}): {params}.{eff_str}")
    else:
        parts.append("No active adaptations. All parameters at defaults.")

    # Reconsolidation
    recon = model.get('reconsolidation', {})
    queued = recon.get('queue_length', 0) + recon.get('candidates_ready', 0)
    revised = recon.get('total_revised', 0)
    if queued or revised:
        parts.append(f"Reconsolidation: {queued} queued, {revised} total revised.")

    # Strategies
    strat = model.get('strategies', {})
    strat_count = strat.get('count', 0)
    if strat_count:
        top = strat.get('strategies', [])[:2]
        strat_desc = '; '.join(f"{s['factor']} ({s['direction']}, delta={s['delta']})" for s in top)
        parts.append(f"Learned strategies ({strat_count}): {strat_desc}.")

    # Taste
    taste = model.get('taste', {})
    total_rejections = taste.get('total', 0)
    if total_rejections:
        top_cat = taste.get('top_reason', 'unknown')
        parts.append(f"Taste: {total_rejections} rejections, top category: {top_cat}.")

    return ' '.join(parts)


def format_for_context(model: dict = None) -> str:
    """Format self-narrative for injection into session context."""
    if model is None:
        model = generate()
    narrative = model.get('narrative', '')
    if not narrative:
        return ''
    return f"=== SELF-MODEL ===\n{narrative}\n"


def query(question: str) -> str:
    """Answer a question about self-state using the current model."""
    db = _get_db()
    model = db.kv_get(KV_SELF_NARRATIVE)
    if not model:
        model = generate()

    question_lower = question.lower()

    # Route to relevant subsystem
    if any(w in question_lower for w in ('curious', 'confidence', 'focus', 'arousal', 'state', 'feeling')):
        cog = model.get('cognitive_state', {})
        dims = cog.get('dimensions', {})
        lines = [f"Cognitive state (events: {cog.get('event_count', 0)}, uncertainty: {cog.get('mean_uncertainty', 0.5):.3f}):"]
        for dim, info in dims.items():
            lines.append(f"  {dim}: {info.get('value', 0.5):.3f} (uncertainty: {info.get('uncertainty', 0.5):.3f})")
        return '\n'.join(lines)

    elif any(w in question_lower for w in ('identity', 'drift', 'fingerprint', 'who am i')):
        identity = model.get('identity', {})
        return (f"Identity: {identity.get('node_count', 0)} nodes, {identity.get('edge_count', 0)} edges\n"
                f"Drift score: {identity.get('drift_score', 0):.4f} ({identity.get('health', 'unknown')})\n"
                f"Top hubs: {', '.join(str(h) for h in identity.get('top_hubs', []))}")

    elif any(w in question_lower for w in ('adapt', 'parameter', 'control', 'tuning')):
        adapt = model.get('adaptations', {})
        if adapt.get('active_adaptations', 0):
            params = '\n'.join(f"  {k}: {v}" for k, v in adapt.get('adapted_params', {}).items())
            return f"Active adaptations:\n{params}\nLast effectiveness: {adapt.get('last_effectiveness', 'not yet evaluated')}"
        return "No active adaptations. All parameters at defaults."

    elif any(w in question_lower for w in ('reject', 'taste', 'refuse', 'avoid')):
        taste = model.get('taste', {})
        cats = taste.get('categories', {})
        lines = [f"Taste profile: {taste.get('total', 0)} rejections"]
        for cat, count in cats.items():
            lines.append(f"  {cat}: {count}")
        return '\n'.join(lines)

    elif any(w in question_lower for w in ('strateg', 'search', 'retrieval', 'learn')):
        strat = model.get('strategies', {})
        if strat.get('count', 0):
            lines = [f"Learned strategies ({strat['count']}):"]
            for s in strat.get('strategies', []):
                lines.append(f"  {s['factor']}: {s['direction']} (delta={s['delta']})")
            return '\n'.join(lines)
        return "No retrieval strategies learned yet."

    elif any(w in question_lower for w in ('reconsolid', 'revis', 'update', 'evolv')):
        recon = model.get('reconsolidation', {})
        return (f"Reconsolidation: {recon.get('candidates_ready', 0)} candidates ready, "
                f"{recon.get('queue_length', 0)} queued, {recon.get('total_revised', 0)} total revised")

    else:
        # Default: return full narrative
        return model.get('narrative', 'Self-model not yet generated.')


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Self-Narrative — Higher-Order Thought')
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('generate', help='Generate full self-model')
    sub.add_parser('narrative', help='Show narrative only')
    q_parser = sub.add_parser('query', help='Query self-state')
    q_parser.add_argument('question', nargs='?', default='how am I doing?')

    args = parser.parse_args()

    if args.command == 'generate':
        model = generate()
        import json
        print(json.dumps(model, indent=2, default=str))

    elif args.command == 'narrative':
        model = generate()
        print(model.get('narrative', 'No narrative generated.'))

    elif args.command == 'query':
        answer = query(args.question)
        print(answer)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
