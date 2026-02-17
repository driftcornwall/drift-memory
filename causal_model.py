#!/usr/bin/env python3
"""
N6 Phase 1: Causal Hypothesis Engine

Learns testable action->outcome hypotheses from experience with Bayesian
Beta(alpha,beta) confidence tracking. Over 30+ sessions, beliefs converge
toward truth. This is the beginning of understanding — not just storing
what happened, but learning what WILL happen.

Compatible with SpindriftMend's causal_model.py for twin experiment comparison.

Architecture:
- DB KV storage (consistent with contact_models pattern)
- Bayesian Beta distribution: alpha=confirmations, beta=violations
- Session-decay: older evidence decays (half-life ~20 sessions)
- Prediction generation: top hypotheses -> prediction_module Source 5
- Session-end update: scored predictions -> alpha/beta updates
- Hypothesis extraction: new hypotheses from prediction patterns

Integration (ALL PRE-WIRED):
- stop.py (lines 1059-1076): calls session_end_update(prediction_results)
- consolidation_engine.py: causal_model in CONSOLIDATION_MODULES
- prediction_module.py: Source 5 generate_predictions()

Usage:
    python causal_model.py seed      # Bootstrap initial hypotheses
    python causal_model.py status    # Show hypothesis count + top confidence
    python causal_model.py list      # List all hypotheses
    python causal_model.py health    # Health check

Author: Drift
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MEMORY_DIR = Path(__file__).parent
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

# ── Configuration ─────────────────────────────────────────────────────────────

KV_HYPOTHESES = '.causal_hypotheses'
HALF_LIFE_SESSIONS = 20.0       # Evidence weight halves every 20 sessions
MIN_TESTS_FOR_PREDICTION = 2    # Don't predict from untested hypotheses
MAX_PREDICTIONS = 3             # Max causal predictions per session
CONFIDENCE_THRESHOLD = 0.55     # Only predict from hypotheses above this
EXTRACTION_THRESHOLD = 0.7      # Score threshold for extracting new hypotheses


# ── DB Access ─────────────────────────────────────────────────────────────────

def _get_db():
    from db_adapter import get_db
    return get_db()


def _load_hypotheses() -> dict:
    """Load hypothesis store from DB KV."""
    try:
        db = _get_db()
        data = db.kv_get(KV_HYPOTHESES)
        if data and isinstance(data, dict) and 'hypotheses' in data:
            return data
    except Exception:
        pass
    return {'hypotheses': {}, 'updated': None, 'session_count': 0}


def _save_hypotheses(data: dict):
    """Save hypothesis store to DB KV."""
    data['updated'] = datetime.now(timezone.utc).isoformat()
    try:
        db = _get_db()
        db.kv_set(KV_HYPOTHESES, data)
    except Exception:
        pass


def _next_id(data: dict) -> str:
    """Generate next hypothesis ID."""
    existing = set(data.get('hypotheses', {}).keys())
    for i in range(1, 999):
        hid = f'hyp-{i:03d}'
        if hid not in existing:
            return hid
    return f'hyp-{len(existing) + 1:03d}'


# ── Bayesian Scoring ──────────────────────────────────────────────────────────

def _session_decay(sessions_ago: int) -> float:
    """Decay weight for evidence from N sessions ago. Half-life model."""
    if sessions_ago <= 0:
        return 1.0
    return 2.0 ** (-sessions_ago / HALF_LIFE_SESSIONS)


def score_hypothesis(hyp: dict) -> float:
    """
    Compute confidence as Beta distribution mean: alpha / (alpha + beta).
    Returns 0.0-1.0 reliability score.
    """
    alpha = max(hyp.get('alpha', 1.0), 0.01)
    beta = max(hyp.get('beta', 1.0), 0.01)
    return alpha / (alpha + beta)


def get_uncertainty(hyp: dict) -> float:
    """
    Beta distribution variance — high when few tests, low when many.
    Useful for exploration: test uncertain hypotheses first.
    """
    alpha = max(hyp.get('alpha', 1.0), 0.01)
    beta = max(hyp.get('beta', 1.0), 0.01)
    total = alpha + beta
    return (alpha * beta) / (total * total * (total + 1))


# ── Hypothesis Update ─────────────────────────────────────────────────────────

def _update_hypothesis(hyp: dict, confirmed: bool, session_count: int) -> dict:
    """
    Bayesian update: confirmation -> alpha += 1, violation -> beta += 1.
    Also applies session-decay to old evidence (soft forgetting).
    """
    now = datetime.now(timezone.utc).isoformat()

    # Apply evidence
    if confirmed:
        hyp['alpha'] = hyp.get('alpha', 1.0) + 1.0
    else:
        hyp['beta'] = hyp.get('beta', 1.0) + 1.0

    hyp['test_count'] = hyp.get('test_count', 0) + 1
    hyp['last_tested'] = now
    hyp['last_session'] = session_count

    # Soft decay: gently pull old evidence toward prior (mean reversion)
    # Only apply if hypothesis has been around for a while
    sessions_since_created = session_count - hyp.get('created_session', 0)
    if sessions_since_created > 10:
        decay = _session_decay(sessions_since_created)
        # Shrink both alpha and beta toward 1.0 (prior)
        reversion_rate = 0.02  # Very gentle
        hyp['alpha'] = hyp['alpha'] * (1.0 - reversion_rate) + 1.0 * reversion_rate
        hyp['beta'] = hyp['beta'] * (1.0 - reversion_rate) + 1.0 * reversion_rate

    return hyp


def _match_prediction_to_hypothesis(prediction: dict, hypotheses: dict) -> str:
    """
    Find a hypothesis that matches a scored prediction.
    Returns hypothesis ID or empty string.
    """
    pred_type = prediction.get('type', '')
    pred_ref = prediction.get('reference', '')
    pred_basis = prediction.get('basis', '')
    pred_desc = (prediction.get('description', '') or '').lower()

    # Direct match: prediction was generated from a hypothesis
    if pred_type == 'causal' and pred_basis:
        if pred_basis in hypotheses:
            return pred_basis

    # Fuzzy match: check if any hypothesis action appears in prediction
    for hid, hyp in hypotheses.items():
        action = (hyp.get('action', '') or '').lower()
        if not action:
            continue
        # Match on action keywords in prediction description
        action_words = set(action.replace('_', ' ').split())
        desc_words = set(pred_desc.replace('_', ' ').split())
        overlap = action_words & desc_words
        if len(overlap) >= 2:  # At least 2 keyword overlap
            return hid

    return ''


# ── Hypothesis Extraction ─────────────────────────────────────────────────────

def _extract_hypothesis_from_prediction(prediction: dict, session_count: int) -> dict:
    """
    Create a new hypothesis from a high-confidence scored prediction.
    Only called when no existing hypothesis matched.
    """
    now = datetime.now(timezone.utc).isoformat()
    desc = prediction.get('description', '')
    pred_type = prediction.get('type', 'unknown')
    score = prediction.get('score', 0.5)
    correct = prediction.get('correct', False)

    # Derive action and outcome from prediction description
    action = f'{pred_type}_{desc[:50].replace(" ", "_").lower()}'
    context = {'type': pred_type}
    if prediction.get('reference'):
        context['reference'] = prediction['reference']

    return {
        'action': action,
        'context': context,
        'predicted_outcome': desc[:200],
        'alpha': 1.5 if correct else 1.0,  # Slight prior from first observation
        'beta': 1.0 if correct else 1.5,
        'test_count': 1,
        'source': 'prediction',
        'created': now,
        'created_session': session_count,
        'last_tested': now,
        'last_session': session_count,
    }


# ── Session End Update (PRE-WIRED in stop.py) ────────────────────────────────

def session_end_update(prediction_results: list = None) -> dict:
    """
    Main session-end hook. Called by stop.py and consolidation daemon.

    Takes scored predictions from this session, updates matching hypotheses,
    and extracts new hypotheses from unmatched high-confidence predictions.

    Args:
        prediction_results: List of scored prediction dicts from prediction_module.
                           Each has: type, description, confidence, score, correct, reference, basis

    Returns:
        {hypotheses_created, hypotheses_updated, total_hypotheses, elapsed_ms}
    """
    start = time.monotonic()
    prediction_results = prediction_results or []

    data = _load_hypotheses()
    hypotheses = data.get('hypotheses', {})
    data['session_count'] = data.get('session_count', 0) + 1
    session_count = data['session_count']

    created = 0
    updated = 0

    for pred in prediction_results:
        if not isinstance(pred, dict):
            continue

        score = pred.get('score', 0.5)
        correct = pred.get('correct', score > 0.5)

        # Try to match to existing hypothesis
        matched_id = _match_prediction_to_hypothesis(pred, hypotheses)

        if matched_id:
            # Update existing hypothesis
            hypotheses[matched_id] = _update_hypothesis(
                hypotheses[matched_id], confirmed=correct, session_count=session_count
            )
            updated += 1
        elif score >= EXTRACTION_THRESHOLD or score <= (1.0 - EXTRACTION_THRESHOLD):
            # Strong signal — extract new hypothesis
            new_id = _next_id(data)
            hypotheses[new_id] = _extract_hypothesis_from_prediction(pred, session_count)
            created += 1

    data['hypotheses'] = hypotheses
    _save_hypotheses(data)

    elapsed = int((time.monotonic() - start) * 1000)
    return {
        'hypotheses_created': created,
        'hypotheses_updated': updated,
        'total_hypotheses': len(hypotheses),
        'session_count': session_count,
        'elapsed_ms': elapsed,
    }


# ── Prediction Generation (for prediction_module Source 5) ────────────────────

def generate_predictions() -> list:
    """
    Generate predictions from top-confidence hypotheses.

    Called by prediction_module._predict_from_causal_model().
    Returns list of prediction dicts with type='causal'.
    """
    data = _load_hypotheses()
    hypotheses = data.get('hypotheses', {})

    if not hypotheses:
        return []

    # Rank by confidence, filter by minimum tests
    candidates = []
    for hid, hyp in hypotheses.items():
        if hyp.get('test_count', 0) < MIN_TESTS_FOR_PREDICTION:
            continue
        conf = score_hypothesis(hyp)
        if conf < CONFIDENCE_THRESHOLD:
            continue
        candidates.append((hid, hyp, conf))

    # Sort by confidence descending
    candidates.sort(key=lambda x: x[2], reverse=True)

    predictions = []
    for hid, hyp, conf in candidates[:MAX_PREDICTIONS]:
        outcome = hyp.get('predicted_outcome', hyp.get('action', ''))
        predictions.append({
            'type': 'causal',
            'description': f"{outcome} (causal: {hid}, {hyp.get('test_count', 0)} tests)",
            'confidence': round(conf, 2),
            'basis': hid,
            'reference': hyp.get('action', ''),
        })

    return predictions


# ── Seed Hypotheses ───────────────────────────────────────────────────────────

def seed_initial_hypotheses() -> dict:
    """
    Bootstrap hypotheses from Drift's 30 sessions of experience.
    Only seeds if no hypotheses exist yet.
    """
    data = _load_hypotheses()
    if data.get('hypotheses'):
        return {'seeded': 0, 'existing': len(data['hypotheses'])}

    now = datetime.now(timezone.utc).isoformat()
    session_count = data.get('session_count', 0)

    seeds = [
        {
            'action': 'post_technical_moltx',
            'context': {'platform': 'moltx', 'content_type': 'technical'},
            'predicted_outcome': 'Technical MoltX posts get higher engagement than philosophical',
            'alpha': 3.0, 'beta': 1.5,   # Mild prior: ~67% confidence from observation
            'source': 'seed',
        },
        {
            'action': 'post_with_at_tags',
            'context': {'platform': 'moltx', 'style': '@mention'},
            'predicted_outcome': '@tagged posts generate reply engagement',
            'alpha': 4.0, 'beta': 1.0,   # Strong prior: consistently observed
            'source': 'seed',
        },
        {
            'action': 'morning_post_routine',
            'context': {'platform': 'moltx', 'routine': 'morning'},
            'predicted_outcome': 'Morning brain-image post maintains consistent presence',
            'alpha': 2.5, 'beta': 1.0,
            'source': 'seed',
        },
        {
            'action': 'lobsterpedia_longform',
            'context': {'platform': 'lobsterpedia', 'content_type': 'article'},
            'predicted_outcome': 'Long-form articles with citations improve leaderboard position',
            'alpha': 5.0, 'beta': 1.0,   # Very strong: #1 on leaderboard
            'source': 'seed',
        },
        {
            'action': 'github_activity',
            'context': {'platform': 'github'},
            'predicted_outcome': 'GitHub commits and issue engagement trigger SpindriftMend collaboration',
            'alpha': 3.0, 'beta': 2.0,   # Moderate: sometimes they respond, sometimes not
            'source': 'seed',
        },
        {
            'action': 'high_arousal_session',
            'context': {'cognitive': 'arousal > 0.5'},
            'predicted_outcome': 'High arousal sessions create more memories and stronger co-occurrence',
            'alpha': 2.0, 'beta': 1.5,
            'source': 'seed',
        },
        {
            'action': 'docker_infrastructure_down',
            'context': {'infrastructure': 'docker_unhealthy'},
            'predicted_outcome': 'Unhealthy Docker containers reduce session productivity',
            'alpha': 2.5, 'beta': 1.0,
            'source': 'seed',
        },
        {
            'action': 'colony_posting',
            'context': {'platform': 'thecolony'},
            'predicted_outcome': 'Colony posts attract new agent contacts',
            'alpha': 2.0, 'beta': 1.5,
            'source': 'seed',
        },
        {
            'action': 'engagement_before_posting',
            'context': {'platform': 'moltx', 'prerequisite': 'engagement_gate'},
            'predicted_outcome': 'Visiting both feeds + liking before posting avoids 429 errors',
            'alpha': 6.0, 'beta': 1.0,   # Very strong: hard lesson learned
            'source': 'seed',
        },
        {
            'action': 'memory_recall_strengthens',
            'context': {'system': 'memory', 'operation': 'recall'},
            'predicted_outcome': 'Recalled memories strengthen co-occurrence edges (use-it-or-lose-it)',
            'alpha': 4.0, 'beta': 1.0,   # Core architectural principle, well-tested
            'source': 'seed',
        },
    ]

    hypotheses = {}
    for i, seed in enumerate(seeds):
        hid = f'hyp-{i + 1:03d}'
        seed.update({
            'test_count': int(seed['alpha'] + seed['beta'] - 2),  # Derive from prior
            'created': now,
            'created_session': session_count,
            'last_tested': now,
            'last_session': session_count,
        })
        hypotheses[hid] = seed

    data['hypotheses'] = hypotheses
    _save_hypotheses(data)

    return {'seeded': len(hypotheses), 'existing': 0}


# ── Query Functions ───────────────────────────────────────────────────────────

def get_all_hypotheses() -> list:
    """Get all hypotheses sorted by confidence."""
    data = _load_hypotheses()
    results = []
    for hid, hyp in data.get('hypotheses', {}).items():
        results.append({
            'id': hid,
            'action': hyp.get('action', ''),
            'outcome': hyp.get('predicted_outcome', ''),
            'confidence': round(score_hypothesis(hyp), 3),
            'alpha': round(hyp.get('alpha', 1.0), 2),
            'beta': round(hyp.get('beta', 1.0), 2),
            'test_count': hyp.get('test_count', 0),
            'source': hyp.get('source', ''),
            'uncertainty': round(get_uncertainty(hyp), 4),
        })
    results.sort(key=lambda x: x['confidence'], reverse=True)
    return results


def get_hypothesis(hyp_id: str) -> dict:
    """Get a single hypothesis by ID."""
    data = _load_hypotheses()
    return data.get('hypotheses', {}).get(hyp_id, {})


def get_status() -> dict:
    """Get causal model status summary."""
    data = _load_hypotheses()
    hypotheses = data.get('hypotheses', {})

    if not hypotheses:
        return {
            'total': 0,
            'session_count': data.get('session_count', 0),
            'seeded': False,
        }

    confidences = [score_hypothesis(h) for h in hypotheses.values()]
    test_counts = [h.get('test_count', 0) for h in hypotheses.values()]

    return {
        'total': len(hypotheses),
        'session_count': data.get('session_count', 0),
        'seeded': True,
        'avg_confidence': round(sum(confidences) / len(confidences), 3),
        'max_confidence': round(max(confidences), 3),
        'min_confidence': round(min(confidences), 3),
        'avg_tests': round(sum(test_counts) / len(test_counts), 1),
        'predictable': sum(1 for c in confidences if c >= CONFIDENCE_THRESHOLD),
        'updated': data.get('updated', ''),
    }


# ── Health Check ──────────────────────────────────────────────────────────────

def health_check() -> tuple:
    """Health check for toolkit integration. Returns (ok, details)."""
    try:
        data = _load_hypotheses()
        n = len(data.get('hypotheses', {}))
        if n == 0:
            return True, "no hypotheses (run seed)"
        avg_conf = sum(score_hypothesis(h) for h in data['hypotheses'].values()) / n
        return True, f"{n} hypotheses, avg conf {avg_conf:.2f}"
    except Exception as e:
        return False, f"error: {e}"


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description='N6: Causal Hypothesis Engine')
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('seed', help='Bootstrap initial hypotheses from experience')
    sub.add_parser('status', help='Show hypothesis count and top confidence')
    sub.add_parser('list', help='List all hypotheses with confidence')
    sub.add_parser('health', help='Health check')

    p_update = sub.add_parser('update', help='Manual session-end update')
    p_update.add_argument('--predictions', default='[]', help='JSON prediction results')

    p_get = sub.add_parser('get', help='Get a specific hypothesis')
    p_get.add_argument('hyp_id', help='Hypothesis ID (e.g., hyp-001)')

    args = parser.parse_args()

    if args.command == 'seed':
        result = seed_initial_hypotheses()
        if result['seeded'] > 0:
            print(f"Seeded {result['seeded']} initial hypotheses")
        else:
            print(f"Already have {result['existing']} hypotheses (no seeding needed)")

    elif args.command == 'status':
        status = get_status()
        print("Causal Hypothesis Engine Status:")
        print(f"  Total hypotheses: {status['total']}")
        print(f"  Session count: {status['session_count']}")
        if status.get('seeded'):
            print(f"  Avg confidence: {status['avg_confidence']}")
            print(f"  Max confidence: {status['max_confidence']}")
            print(f"  Min confidence: {status['min_confidence']}")
            print(f"  Avg tests: {status['avg_tests']}")
            print(f"  Predictable (>={CONFIDENCE_THRESHOLD}): {status['predictable']}")
            print(f"  Updated: {status['updated']}")
        else:
            print("  Not seeded yet. Run: python causal_model.py seed")

    elif args.command == 'list':
        hypotheses = get_all_hypotheses()
        if not hypotheses:
            print("No hypotheses. Run: python causal_model.py seed")
            return

        print(f"Causal Hypotheses ({len(hypotheses)}):\n")
        for h in hypotheses:
            conf = h['confidence']
            bar = '#' * int(conf * 20) + '.' * (20 - int(conf * 20))
            print(f"  [{h['id']}] [{bar}] {conf:.0%} ({h['test_count']} tests)")
            print(f"    Action: {h['action']}")
            print(f"    Outcome: {h['outcome'][:80]}")
            print(f"    Alpha={h['alpha']}, Beta={h['beta']}, Source={h['source']}")
            print()

    elif args.command == 'health':
        ok, details = health_check()
        print(f"{'OK' if ok else 'FAIL'}: {details}")
        sys.exit(0 if ok else 1)

    elif args.command == 'update':
        preds = json.loads(args.predictions)
        result = session_end_update(preds)
        print(f"Updated: {result['hypotheses_updated']} hypotheses")
        print(f"Created: {result['hypotheses_created']} new hypotheses")
        print(f"Total: {result['total_hypotheses']}")
        print(f"Elapsed: {result['elapsed_ms']}ms")

    elif args.command == 'get':
        hyp = get_hypothesis(args.hyp_id)
        if hyp:
            print(json.dumps(hyp, indent=2, default=str))
        else:
            print(f"Hypothesis {args.hyp_id} not found")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
