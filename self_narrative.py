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


def _get_contacts_summary() -> dict:
    """R14: Pull contact model stats."""
    def _inner():
        db = _get_db()
        raw = db.kv_get('.contact_models') or {}
        # update_all() stores {models: {...}, updated: ..., count: N}
        models = raw.get('models', raw) if isinstance(raw, dict) else {}
        if not models or 'updated' in models and 'models' not in models:
            # Bare wrapper with no actual models
            return {'total': 0, 'top': [], 'avg_reliability': 0.0}
        # Sort by engagement score
        sorted_contacts = sorted(models.items(),
                                 key=lambda x: x[1].get('engagement', 0) if isinstance(x[1], dict) else 0,
                                 reverse=True)
        top3 = [name for name, _ in sorted_contacts[:3]]
        reliabilities = [m.get('reliability', 0.5) for m in models.values() if isinstance(m, dict)]
        avg_rel = sum(reliabilities) / len(reliabilities) if reliabilities else 0.5
        return {
            'total': len(models),
            'top': top3,
            'avg_reliability': round(avg_rel, 2),
        }
    return _safe_call(_inner, {'total': 0, 'top': [], 'avg_reliability': 0.0})


def _get_attention_summary() -> dict:
    """B2: Pull attention schema stats."""
    def _inner():
        db = _get_db()
        schema = db.kv_get('.attention_schema') or []
        if not schema:
            return {'searches': 0, 'avg_ms': 0, 'heaviest': 'none'}
        avg_ms = sum(s.get('total_ms', 0) for s in schema) / len(schema)
        # Find heaviest stage across all searches
        stage_totals = {}
        for entry in schema:
            for stage in entry.get('stages', []):
                name = stage.get('stage', '?')
                stage_totals[name] = stage_totals.get(name, 0) + stage.get('time_ms', 0)
        heaviest = max(stage_totals, key=stage_totals.get) if stage_totals else 'none'
        return {
            'searches': len(schema),
            'avg_ms': round(avg_ms, 1),
            'heaviest': heaviest,
        }
    return _safe_call(_inner, {'searches': 0, 'avg_ms': 0, 'heaviest': 'none'})


def _get_predictions_summary() -> dict:
    """R11: Pull prediction history stats."""
    def _inner():
        db = _get_db()
        history = db.kv_get('.prediction_history') or []
        if not history:
            return {'sessions': 0, 'accuracy': 0.0, 'trend': 'none'}
        accuracies = [h.get('accuracy', 0) for h in history if 'accuracy' in h]
        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
        # Simple trend: compare last 5 to first 5
        trend = 'none'
        if len(accuracies) >= 6:
            early = sum(accuracies[:3]) / 3
            late = sum(accuracies[-3:]) / 3
            if late > early + 0.05:
                trend = 'improving'
            elif late < early - 0.05:
                trend = 'declining'
            else:
                trend = 'stable'
        return {
            'sessions': len(history),
            'accuracy': round(avg_acc, 2),
            'trend': trend,
        }
    return _safe_call(_inner, {'sessions': 0, 'accuracy': 0.0, 'trend': 'none'})


def _get_counterfactual_summary() -> dict:
    """N3/SS3: Pull counterfactual reasoning history."""
    def _inner():
        db = _get_db()
        history = db.kv_get('.counterfactual_history') or []
        if not history:
            return {'sessions': 0, 'total_generated': 0, 'avg_per_session': 0.0, 'llm_ratio': 0.0}
        total_gen = sum(h.get('generated', 0) for h in history)
        total_llm = sum(h.get('llm_calls', 0) for h in history)
        avg_gen = total_gen / len(history) if history else 0
        llm_ratio = total_llm / max(1, total_gen)
        return {
            'sessions': len(history),
            'total_generated': total_gen,
            'avg_per_session': round(avg_gen, 1),
            'llm_ratio': round(llm_ratio, 2),
        }
    return _safe_call(_inner, {'sessions': 0, 'total_generated': 0, 'avg_per_session': 0.0, 'llm_ratio': 0.0})


def _get_goals_summary() -> dict:
    """SN1: Pull volitional goal state and progress."""
    def _inner():
        db = _get_db()
        goals = db.kv_get('.active_goals') or []
        active = [g for g in goals if g.get('status') in ('active', 'watching')]
        history = db.kv_get('.goal_history') or {}
        completed = len(history.get('completed', []))
        abandoned = len(history.get('abandoned', []))
        focus = next((g for g in active if g.get('is_focus')), None)
        avg_vitality = (sum(g.get('vitality', 0) for g in active) / len(active)) if active else 0
        return {
            'active': len(active),
            'completed': completed,
            'abandoned': abandoned,
            'focus': focus.get('action', '')[:60] if focus else 'none',
            'avg_vitality': round(avg_vitality, 2),
            'total_committed': history.get('stats', {}).get('total_committed', 0),
        }
    return _safe_call(_inner, {'active': 0, 'completed': 0, 'abandoned': 0,
                                'focus': 'none', 'avg_vitality': 0, 'total_committed': 0})


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
        'contacts': _get_contacts_summary(),
        'attention': _get_attention_summary(),
        'predictions': _get_predictions_summary(),
        'counterfactuals': _get_counterfactual_summary(),
        'goals': _get_goals_summary(),
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

    # Contacts (R14)
    contacts = model.get('contacts', {})
    if contacts.get('total', 0):
        top_names = ', '.join(contacts.get('top', []))
        parts.append(f"Social: {contacts['total']} contacts modeled, "
                     f"top: {top_names}. Avg reliability: {contacts.get('avg_reliability', 0):.0%}.")

    # Attention (B2)
    attention = model.get('attention', {})
    if attention.get('searches', 0):
        parts.append(f"Attention: {attention['searches']} searches tracked, "
                     f"avg {attention['avg_ms']:.0f}ms, heaviest stage: {attention['heaviest']}.")

    # Predictions (R11)
    predictions = model.get('predictions', {})
    if predictions.get('sessions', 0):
        trend = predictions.get('trend', 'none')
        trend_str = f" ({trend})" if trend != 'none' else ''
        parts.append(f"Predictions: {predictions['sessions']} sessions tracked, "
                     f"{predictions['accuracy']:.0%} accuracy{trend_str}.")

    # Counterfactuals (N3)
    cfs = model.get('counterfactuals', {})
    if cfs.get('total_generated', 0):
        parts.append(f"Counterfactuals: {cfs['total_generated']} generated across "
                     f"{cfs['sessions']} sessions, avg {cfs['avg_per_session']}/session, "
                     f"LLM ratio: {cfs['llm_ratio']:.0%}.")

    # Goals (N4/SN2)
    goals = model.get('goals', {})
    if goals.get('active', 0):
        parts.append(f"Goals: {goals['active']} active (focus: {goals['focus']}), "
                     f"{goals['completed']} completed, {goals['abandoned']} abandoned, "
                     f"avg vitality: {goals['avg_vitality']:.2f}.")

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

    elif any(w in question_lower for w in ('contact', 'social', 'relationship', 'friend')):
        contacts = model.get('contacts', {})
        top = ', '.join(contacts.get('top', [])) or 'none'
        return (f"Contact models: {contacts.get('total', 0)} modeled\n"
                f"Top by engagement: {top}\n"
                f"Avg reliability: {contacts.get('avg_reliability', 0):.0%}")

    elif any(w in question_lower for w in ('attention', 'time', 'slow', 'performance', 'pipeline')):
        attention = model.get('attention', {})
        return (f"Attention schema: {attention.get('searches', 0)} searches tracked\n"
                f"Avg time: {attention.get('avg_ms', 0):.0f}ms\n"
                f"Heaviest stage: {attention.get('heaviest', 'none')}")

    elif any(w in question_lower for w in ('predict', 'expect', 'forecast', 'surprise')):
        predictions = model.get('predictions', {})
        return (f"Predictions: {predictions.get('sessions', 0)} sessions tracked\n"
                f"Accuracy: {predictions.get('accuracy', 0):.0%}\n"
                f"Trend: {predictions.get('trend', 'none')}")

    elif any(w in question_lower for w in ('goal', 'objective', 'focus', 'vitality', 'progress')):
        goals = model.get('goals', {})
        return (f"Goals: {goals.get('active', 0)} active, "
                f"{goals.get('completed', 0)} completed, "
                f"{goals.get('abandoned', 0)} abandoned\n"
                f"Focus: {goals.get('focus', 'none')}\n"
                f"Avg vitality: {goals.get('avg_vitality', 0):.2f}\n"
                f"Total committed: {goals.get('total_committed', 0)}")

    elif any(w in question_lower for w in ('counterfactual', 'what if', 'alternative', 'imagin')):
        cfs = model.get('counterfactuals', {})
        if cfs.get('total_generated', 0):
            return (f"Counterfactual reasoning: {cfs['total_generated']} CFs across "
                    f"{cfs['sessions']} sessions\n"
                    f"Avg per session: {cfs['avg_per_session']}\n"
                    f"LLM ratio: {cfs['llm_ratio']:.0%}")
        return "No counterfactuals generated yet."

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
