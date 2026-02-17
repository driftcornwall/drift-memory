#!/usr/bin/env python3
"""
Prediction Module — Forward Model for Drift (R11)

The system's most glaring theoretical absence: no forward model.
Never generates expectations, never experiences surprise, never learns
from prediction error.

This module generates heuristic predictions at session start and scores
them against actuals at session end. Prediction error feeds into:
1. cognitive_state (prediction_confirmed / prediction_violated events)
2. curiosity_engine (wrong predictions boost curiosity in that domain)
3. self_narrative (accuracy tracking over sessions)

No LLM required — pure heuristics from existing data.

Usage:
    python prediction_module.py generate    # Generate predictions for this session
    python prediction_module.py score       # Score predictions against actuals
    python prediction_module.py history     # Show prediction history
    python prediction_module.py calibration # Show calibration stats
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

MEMORY_DIR = Path(__file__).parent
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

# DB KV keys
KV_PREDICTIONS = '.session_predictions'
KV_PRED_HISTORY = '.prediction_history'


def _get_db():
    from db_adapter import get_db
    return get_db()


# ============================================================
# Prediction Generation (Session Start)
# ============================================================

def generate_predictions() -> list[dict]:
    """
    Generate 3-5 heuristic predictions for this session.
    Called at session start. No LLM — pure data-driven heuristics.

    Returns list of predictions, each with:
    {type, description, confidence, basis, reference}
    """
    predictions = []

    # Source 1: Pending intentions
    predictions.extend(_predict_from_intentions())

    # Source 2: Contact models
    predictions.extend(_predict_from_contacts())

    # Source 3: Platform usage patterns
    predictions.extend(_predict_from_platforms())

    # Source 4: Vitals-based outcome predictions
    predictions.extend(_predict_from_vitals())

    # Source 5: Causal hypothesis engine (N6 — learned action→outcome beliefs)
    try:
        from causal_model import generate_predictions as _causal_predictions
        predictions.extend(_causal_predictions())
    except ImportError:
        pass
    except Exception:
        pass

    # Limit to 5 most confident
    predictions.sort(key=lambda p: p.get('confidence', 0), reverse=True)
    predictions = predictions[:5]

    # Ensure minimum 3 if we have any
    if not predictions:
        predictions.append({
            'type': 'outcome',
            'description': 'At least 3 memories will be recalled this session',
            'confidence': 0.7,
            'basis': 'baseline expectation',
            'reference': None,
        })

    # Store for scoring at session end
    db = _get_db()
    db.kv_set(KV_PREDICTIONS, {
        'predictions': predictions,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'scored': False,
    })

    return predictions


def _predict_from_intentions() -> list[dict]:
    """Generate predictions from pending intentions."""
    predictions = []
    try:
        from temporal_intentions import get_pending
        pending = get_pending()
        for intent in pending[:2]:  # Top 2 by priority
            priority = intent.get('priority', 'M')
            conf = {'H': 0.7, 'M': 0.5, 'L': 0.3}.get(priority, 0.5)
            action = intent.get('action', 'unknown')[:80]
            predictions.append({
                'type': 'intention',
                'description': f'Will work on: {action}',
                'confidence': conf,
                'basis': f'pending intention (priority {priority})',
                'reference': intent.get('id'),
            })
    except Exception:
        pass
    return predictions


def _predict_from_contacts() -> list[dict]:
    """Generate predictions from contact engagement models."""
    predictions = []
    try:
        db = _get_db()
        models = db.kv_get('.contact_models') or {}
        # Find highest-engagement contacts
        top = sorted(models.items(),
                     key=lambda x: x[1].get('engagement', 0), reverse=True)
        for name, model in top[:2]:
            engagement = model.get('engagement', 0)
            if engagement > 1.5:
                predictions.append({
                    'type': 'contact',
                    'description': f'Will interact with {name}',
                    'confidence': min(0.8, engagement / 5.0),
                    'basis': f'engagement score {engagement:.1f}',
                    'reference': name,
                })
    except Exception:
        pass
    return predictions


def _predict_from_platforms() -> list[dict]:
    """Generate predictions from platform usage history."""
    predictions = []
    try:
        db = _get_db()
        history = db.kv_get(KV_PRED_HISTORY) or []
        if len(history) < 3:
            return predictions

        # Count platform usage across recent sessions
        platform_counts = {}
        for session in history[-10:]:
            for plat in session.get('actual_platforms', []):
                platform_counts[plat] = platform_counts.get(plat, 0) + 1

        for plat, count in platform_counts.items():
            if count >= 3:  # Used in 3+ of last 10 sessions
                predictions.append({
                    'type': 'platform',
                    'description': f'Will use {plat}',
                    'confidence': min(0.85, count / 10.0),
                    'basis': f'used in {count}/10 recent sessions',
                    'reference': plat,
                })
    except Exception:
        pass
    return predictions


def _predict_from_vitals() -> list[dict]:
    """Generate predictions from system vitals averages."""
    predictions = []
    try:
        db = _get_db()
        history = db.kv_get(KV_PRED_HISTORY) or []
        if len(history) < 3:
            # Default prediction
            predictions.append({
                'type': 'outcome',
                'description': 'Will recall 5-10 memories',
                'confidence': 0.6,
                'basis': 'baseline estimate',
                'reference': None,
            })
            return predictions

        # Average recall count from recent sessions
        recall_counts = [s.get('actual_recalls', 0) for s in history[-10:] if 'actual_recalls' in s]
        if recall_counts:
            avg_recalls = sum(recall_counts) / len(recall_counts)
            predictions.append({
                'type': 'outcome',
                'description': f'Will recall ~{int(avg_recalls)} memories',
                'confidence': 0.6,
                'basis': f'avg {avg_recalls:.1f} over {len(recall_counts)} sessions',
                'reference': round(avg_recalls),
            })
    except Exception:
        pass
    return predictions


# ============================================================
# Prediction Scoring (Session End)
# ============================================================

def score_predictions() -> dict:
    """
    Score this session's predictions against actuals.
    Called at session end.

    Returns dict with scores, accuracy, and calibration.
    """
    db = _get_db()
    pred_data = db.kv_get(KV_PREDICTIONS)
    if not pred_data or pred_data.get('scored'):
        return {'error': 'No unscored predictions found'}

    predictions = pred_data.get('predictions', [])
    if not predictions:
        return {'error': 'Empty predictions list'}

    # Gather actuals
    actuals = _gather_actuals()

    # Score each prediction
    scored = []
    for pred in predictions:
        score = _score_single(pred, actuals)
        scored.append({**pred, 'score': score, 'correct': score > 0.5})

    # Compute summary stats
    scores = [s['score'] for s in scored]
    correct_count = sum(1 for s in scored if s['correct'])
    accuracy = correct_count / len(scored) if scored else 0
    mean_error = sum(abs(s['confidence'] - s['score']) for s in scored) / len(scored) if scored else 0

    # Calibration: avg confidence of correct vs incorrect
    correct_confs = [s['confidence'] for s in scored if s['correct']]
    incorrect_confs = [s['confidence'] for s in scored if not s['correct']]
    calibration = {
        'correct_avg_conf': sum(correct_confs) / len(correct_confs) if correct_confs else 0,
        'incorrect_avg_conf': sum(incorrect_confs) / len(incorrect_confs) if incorrect_confs else 0,
    }

    result = {
        'predictions': scored,
        'accuracy': round(accuracy, 3),
        'mean_error': round(mean_error, 3),
        'calibration': calibration,
        'correct': correct_count,
        'total': len(scored),
        'scored_at': datetime.now(timezone.utc).isoformat(),
    }

    # Fire cognitive state events
    _fire_cognitive_events(scored)

    # N3: Generate prospective counterfactual context for violated predictions
    # (Actual counterfactual generation happens in stop.py Phase 2 via session_end_review)
    violated = [s for s in scored if not s.get('correct')]
    if violated:
        try:
            result['violated_count'] = len(violated)
            result['near_miss_count'] = sum(
                1 for s in violated
                if 0.3 <= s.get('confidence', 0) <= 0.7
            )
        except Exception:
            pass

    # Mark as scored
    pred_data['scored'] = True
    pred_data['result'] = result
    db.kv_set(KV_PREDICTIONS, pred_data)

    # Append to history (rolling 30 sessions)
    history_entry = {
        'accuracy': result['accuracy'],
        'mean_error': result['mean_error'],
        'correct': correct_count,
        'total': len(scored),
        'actual_recalls': actuals.get('recall_count', 0),
        'actual_platforms': actuals.get('platforms', []),
        'actual_contacts': actuals.get('contacts', []),
        'session_date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
    }
    history = db.kv_get(KV_PRED_HISTORY) or []
    history.append(history_entry)
    db.kv_set(KV_PRED_HISTORY, history[-30:])

    return result


def _gather_actuals() -> dict:
    """Gather actual session data for scoring predictions."""
    actuals = {'recall_count': 0, 'platforms': [], 'contacts': []}

    try:
        import session_state
        retrieved = session_state.get_retrieved()
        actuals['recall_count'] = len(retrieved)
    except Exception:
        pass

    try:
        from platform_context import get_session_platforms
        actuals['platforms'] = get_session_platforms()
    except Exception:
        pass

    try:
        db = _get_db()
        session_contacts = db.kv_get('.session_contacts') or []
        if isinstance(session_contacts, dict):
            session_contacts = list(session_contacts.keys())
        actuals['contacts'] = [c.lower() if isinstance(c, str) else c for c in session_contacts]
    except Exception:
        pass

    return actuals


def _score_single(prediction: dict, actuals: dict) -> float:
    """Score a single prediction against actuals. Returns 0.0-1.0."""
    pred_type = prediction.get('type', '')

    if pred_type == 'contact':
        ref = (prediction.get('reference') or '').lower()
        return 1.0 if ref in actuals.get('contacts', []) else 0.0

    elif pred_type == 'platform':
        ref = (prediction.get('reference') or '').lower()
        return 1.0 if ref in [p.lower() for p in actuals.get('platforms', [])] else 0.0

    elif pred_type == 'outcome':
        ref = prediction.get('reference')
        actual = actuals.get('recall_count', 0)
        if isinstance(ref, (int, float)) and ref > 0:
            # Score based on how close actual was to predicted
            ratio = min(actual, ref) / max(actual, ref) if max(actual, ref) > 0 else 0
            return round(ratio, 3)
        # Generic "will recall some" — true if any recalls happened
        return 1.0 if actual > 0 else 0.0

    elif pred_type == 'intention':
        # Intentions are hard to auto-score — conservative 0.5
        return 0.5

    elif pred_type == 'causal':
        # Causal hypothesis predictions: check if the predicted platform/contact/outcome appeared
        ref = (prediction.get('reference') or '').lower()
        desc = (prediction.get('description') or '').lower()
        # Check against all actuals for any match
        all_actuals = (
            [p.lower() for p in actuals.get('platforms', [])] +
            actuals.get('contacts', []) +
            [str(actuals.get('recall_count', 0))]
        )
        for actual in all_actuals:
            if actual and actual in desc:
                return 0.8  # Partial confirmation
        # Check keywords
        if any(k in desc for k in ['engagement', 'like', 'reply'] if actuals.get('contacts')):
            return 0.7
        return 0.4  # Weak — couldn't confirm or deny

    return 0.5  # Unknown type


def _fire_cognitive_events(scored: list[dict]):
    """Fire cognitive state events based on prediction outcomes."""
    try:
        from cognitive_state import process_event
        for s in scored:
            if s.get('correct'):
                process_event('prediction_confirmed')
            else:
                process_event('prediction_violated')
    except Exception:
        pass


# ============================================================
# History & Calibration
# ============================================================

def get_history(limit: int = 30) -> list[dict]:
    """Get prediction history."""
    db = _get_db()
    history = db.kv_get(KV_PRED_HISTORY) or []
    return history[-limit:]


def get_calibration() -> dict:
    """Get calibration stats across all sessions."""
    history = get_history()
    if not history:
        return {'sessions': 0, 'message': 'No prediction history yet'}

    accuracies = [h.get('accuracy', 0) for h in history]
    errors = [h.get('mean_error', 0) for h in history]

    return {
        'sessions': len(history),
        'overall_accuracy': round(sum(accuracies) / len(accuracies), 3),
        'overall_error': round(sum(errors) / len(errors), 3),
        'best_session': round(max(accuracies), 3),
        'worst_session': round(min(accuracies), 3),
        'recent_5_avg': round(sum(accuracies[-5:]) / min(5, len(accuracies)), 3),
    }


def health() -> dict:
    """Health check for toolkit."""
    db = _get_db()
    history = db.kv_get(KV_PRED_HISTORY) or []
    return {'ok': True, 'detail': f'{len(history)} sessions tracked'}


def format_predictions_context(predictions: list[dict]) -> str:
    """Format predictions for session start context injection."""
    if not predictions:
        return ''
    lines = ['=== SESSION PREDICTIONS ===']
    for i, p in enumerate(predictions, 1):
        conf_pct = f"{p['confidence']:.0%}"
        lines.append(f"  [{conf_pct}] {p['description']} (basis: {p['basis']})")
    lines.append('')
    return '\n'.join(lines)


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Prediction Module — Forward Model')
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('generate', help='Generate predictions for this session')
    sub.add_parser('score', help='Score predictions against actuals')
    h_parser = sub.add_parser('history', help='Show prediction history')
    h_parser.add_argument('--limit', type=int, default=10)
    sub.add_parser('calibration', help='Show calibration stats')

    args = parser.parse_args()

    if args.command == 'generate':
        predictions = generate_predictions()
        print(format_predictions_context(predictions))
        print(f'\n{len(predictions)} predictions generated.')

    elif args.command == 'score':
        result = score_predictions()
        if 'error' in result:
            print(f'Error: {result["error"]}')
        else:
            print(f'Accuracy: {result["accuracy"]:.0%} ({result["correct"]}/{result["total"]})')
            print(f'Mean error: {result["mean_error"]:.3f}')
            for p in result.get('predictions', []):
                mark = 'OK' if p.get('correct') else 'MISS'
                print(f'  [{mark}] {p["description"]} (conf={p["confidence"]:.0%}, score={p["score"]:.2f})')

    elif args.command == 'history':
        history = get_history(limit=args.limit)
        if not history:
            print('No prediction history yet.')
        else:
            for h in history:
                date = h.get('session_date', '?')
                acc = h.get('accuracy', 0)
                print(f'  {date}: {acc:.0%} accuracy ({h.get("correct", 0)}/{h.get("total", 0)})')

    elif args.command == 'calibration':
        cal = get_calibration()
        if cal.get('message'):
            print(cal['message'])
        else:
            print(f'Sessions: {cal["sessions"]}')
            print(f'Overall accuracy: {cal["overall_accuracy"]:.0%}')
            print(f'Overall error: {cal["overall_error"]:.3f}')
            print(f'Best: {cal["best_session"]:.0%}, Worst: {cal["worst_session"]:.0%}')
            print(f'Recent 5 avg: {cal["recent_5_avg"]:.0%}')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
