#!/usr/bin/env python3
"""
Retrieval Prediction Engine — T4.1 Predictive Coding in the Retrieval Pipeline

Before pgvector search, generates expectations about which memory IDs should
appear. After search, the pipeline computes prediction error and boosts
results that are surprisingly relevant (enlightenment surprise).

Based on: Rao & Ballard 1999 (predictive coding), Friston (free energy),
Rescorla-Wagner (associative learning from prediction error).

5 prediction sources:
  1. Co-occurrence neighbors (edges_v3)
  2. Q-value top memories
  3. Knowledge graph neighbors
  4. Contact model predictions
  5. Causal hypothesis predictions

Usage:
    python retrieval_prediction.py test              # Run prediction test
    python retrieval_prediction.py weights           # Show source weights
    python retrieval_prediction.py accuracy          # Show accuracy stats
    python retrieval_prediction.py status            # Overall status
"""

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PREDICTION_ENABLED = True

# Source weights (Rescorla-Wagner updated). Initial: equal weight.
DEFAULT_WEIGHTS = {
    'cooccurrence': 0.20,
    'q_values': 0.20,
    'kg_neighbors': 0.20,
    'contacts': 0.20,
    'causal': 0.20,
}

# Learning parameters
RW_ALPHA = 0.10        # Rescorla-Wagner learning rate
MIN_WEIGHT = 0.05      # Floor for source weights
MAX_WEIGHT = 0.50      # Ceiling for source weights

# Prediction limits per source
COOCCURRENCE_LIMIT = 10
Q_VALUE_LIMIT = 8
KG_LIMIT = 8
CONTACT_LIMIT = 6
CAUSAL_LIMIT = 5

# DB KV keys
KV_WEIGHTS = '.retrieval_prediction_weights'
KV_LOG = '.retrieval_prediction_log'
KV_ACCURACY = '.retrieval_prediction_accuracy'

# Module-level cache (loaded at session start)
_source_weights: dict = None
_session_predictions: list = []  # Accumulated this session for RW update


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PredictionSet:
    """Result of prediction generation."""
    predicted_ids: dict  # memory_id -> aggregated confidence (0-1)
    source_breakdown: dict  # source_name -> [(id, conf), ...]
    generation_time_ms: float
    query_hash: str

    def to_dict(self) -> dict:
        return {
            'predicted_ids': self.predicted_ids,
            'source_breakdown': {k: [(mid, round(c, 4)) for mid, c in v]
                                 for k, v in self.source_breakdown.items()},
            'generation_time_ms': round(self.generation_time_ms, 1),
            'query_hash': self.query_hash,
            'total_predictions': len(self.predicted_ids),
        }


# ---------------------------------------------------------------------------
# Weight management
# ---------------------------------------------------------------------------

def _get_weights() -> dict:
    """Get current source weights (cached or from DB)."""
    global _source_weights
    if _source_weights is not None:
        return _source_weights
    try:
        from db_adapter import get_db
        db = get_db()
        stored = db.get_kv(KV_WEIGHTS)
        if stored and isinstance(stored, dict) and 'weights' in stored:
            _source_weights = stored['weights']
            # Ensure all sources present
            for k, v in DEFAULT_WEIGHTS.items():
                if k not in _source_weights:
                    _source_weights[k] = v
            return _source_weights
    except Exception:
        pass
    _source_weights = dict(DEFAULT_WEIGHTS)
    return _source_weights


def _save_weights(weights: dict):
    """Persist source weights to DB."""
    global _source_weights
    _source_weights = weights
    try:
        from db_adapter import get_db
        db = get_db()
        db.set_kv(KV_WEIGHTS, {'weights': weights, 'updated': time.time()})
    except Exception:
        pass


def load_weights():
    """Public: Load weights at session start."""
    global _source_weights
    _source_weights = None  # Force reload from DB
    _get_weights()


# ---------------------------------------------------------------------------
# Source 1: Co-occurrence neighbors
# ---------------------------------------------------------------------------

def _predict_from_cooccurrence(recent_recalls: list, limit: int = COOCCURRENCE_LIMIT) -> list:
    """Memories that co-occurred with recently recalled ones."""
    if not recent_recalls:
        return []

    from db_adapter import get_db
    db = get_db()

    recall_set = set(recent_recalls)
    predictions = []

    try:
        import psycopg2.extras
        with db._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT CASE WHEN id1 = ANY(%s) THEN id2 ELSE id1 END AS neighbor,
                           AVG(belief) AS avg_belief
                    FROM {db._table('edges_v3')}
                    WHERE (id1 = ANY(%s) OR id2 = ANY(%s)) AND belief > 0.1
                    GROUP BY neighbor
                    ORDER BY avg_belief DESC
                    LIMIT %s
                """, (recent_recalls, recent_recalls, recent_recalls, limit * 2))
                for row in cur.fetchall():
                    nid, belief = row[0], float(row[1])
                    if nid not in recall_set:
                        conf = min(1.0, belief / 5.0)
                        predictions.append((nid, conf))
                        if len(predictions) >= limit:
                            break
    except Exception:
        pass

    return predictions


# ---------------------------------------------------------------------------
# Source 2: Q-value top memories
# ---------------------------------------------------------------------------

def _predict_from_q_values(limit: int = Q_VALUE_LIMIT) -> list:
    """High Q-value memories are expected to be useful."""
    try:
        from q_value_engine import q_top
        top = q_top(limit)
        if not top:
            return []
        max_q = max(r['q_value'] for r in top) if top else 1.0
        if max_q <= 0:
            max_q = 1.0
        return [(r['id'], min(1.0, float(r['q_value']) / max_q * 0.8)) for r in top]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Source 3: KG neighbors
# ---------------------------------------------------------------------------

def _predict_from_kg(query: str, limit: int = KG_LIMIT) -> list:
    """Memories connected via knowledge graph to query entities."""
    try:
        from entity_detection import detect_entities
        from knowledge_graph import traverse as kg_traverse

        entities = detect_entities(query, [])
        if not entities:
            return []

        predictions = {}
        entity_names = list(entities.keys())[:3]

        for entity_name in entity_names:
            # Get memories mentioning this entity via entity_index
            try:
                from entity_index import load_index
                index = load_index()
                entity_mids = index.get(entity_name.lower(), [])[:3]
            except Exception:
                entity_mids = []

            for mid in entity_mids:
                edges = kg_traverse(mid, hops=1, direction='both', min_confidence=0.5)
                for edge in edges:
                    for nid in (edge.get('source_id', ''), edge.get('target_id', '')):
                        if nid and nid != mid and nid not in predictions:
                            conf = float(edge.get('confidence', 0.5)) * 0.6
                            predictions[nid] = max(predictions.get(nid, 0), conf)

        # Sort by confidence, take top N
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_preds[:limit]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Source 4: Contact model predictions
# ---------------------------------------------------------------------------

def _predict_from_contacts(query: str, limit: int = CONTACT_LIMIT) -> list:
    """If query mentions a contact, predict their associated memories."""
    try:
        from entity_index import detect_contacts, get_memories_for_query

        contacts = detect_contacts(query)
        if not contacts:
            return []

        memory_ids = get_memories_for_query(query)
        if not memory_ids:
            return []

        # Get reliability scores from contact models
        predictions = []
        for mid in memory_ids[:limit]:
            conf = 0.5  # Default confidence for contact-associated memories
            try:
                from contact_models import predict_engagement
                # Use first detected contact's engagement prediction
                eng = predict_engagement(contacts[0], 'retrieval')
                conf = min(1.0, max(0.1, eng))
            except Exception:
                pass
            predictions.append((mid, conf))

        return predictions
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Source 5: Causal hypothesis predictions
# ---------------------------------------------------------------------------

def _predict_from_causal(query: str, limit: int = CAUSAL_LIMIT) -> list:
    """High-confidence hypotheses predict related memory retrieval."""
    try:
        from causal_model import get_all_hypotheses

        hypotheses = get_all_hypotheses()
        if not hypotheses:
            return []

        query_tokens = set(query.lower().split())
        predictions = {}

        for hyp in hypotheses:
            if hyp.get('confidence', 0) < 0.55 or hyp.get('test_count', 0) < 2:
                continue

            # Match hypothesis keywords against query
            action_tokens = set(hyp.get('action', '').lower().split())
            outcome_tokens = set(hyp.get('outcome', '').lower().split())
            all_hyp_tokens = action_tokens | outcome_tokens

            overlap = len(query_tokens & all_hyp_tokens)
            if overlap == 0:
                continue

            # Confidence = hypothesis confidence * normalized overlap
            overlap_ratio = overlap / max(1, len(all_hyp_tokens))
            conf = hyp['confidence'] * overlap_ratio

            # Use hypothesis ID to find associated memories via tag search
            hyp_id = hyp.get('id', '')
            if hyp_id:
                try:
                    from db_adapter import get_db
                    db = get_db()
                    import psycopg2.extras
                    with db._conn() as conn:
                        with conn.cursor() as cur:
                            # Search for memories tagged with hypothesis keywords
                            search_term = hyp.get('action', '')[:50]
                            cur.execute(f"""
                                SELECT id FROM {db._table('memories')}
                                WHERE type IN ('active', 'core')
                                AND content ILIKE %s
                                LIMIT 3
                            """, (f'%{search_term}%',))
                            for row in cur.fetchall():
                                mid = row[0]
                                predictions[mid] = max(predictions.get(mid, 0), conf)
                except Exception:
                    pass

        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_preds[:limit]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Aggregation (noisy-OR)
# ---------------------------------------------------------------------------

def _aggregate_predictions(source_results: dict) -> dict:
    """
    Combine predictions from all sources using weighted noisy-OR.

    For each memory ID predicted by multiple sources:
        combined = 1 - product(1 - conf_i * weight_i)

    Returns dict[memory_id, aggregated_confidence].
    """
    weights = _get_weights()
    combined = {}

    for source_name, predictions in source_results.items():
        w = weights.get(source_name, 0.2)
        for mid, conf in predictions:
            weighted_conf = conf * w
            if mid not in combined:
                combined[mid] = 1.0 - (1.0 - weighted_conf)
            else:
                combined[mid] = 1.0 - (1.0 - combined[mid]) * (1.0 - weighted_conf)

    # Clamp to [0, 1]
    return {mid: min(1.0, max(0.0, c)) for mid, c in combined.items()}


# ---------------------------------------------------------------------------
# Main prediction generator
# ---------------------------------------------------------------------------

def generate_predictions(
    query: str,
    recent_recalls: list = None,
    dimension: str = None,
) -> Optional[PredictionSet]:
    """
    Generate predictions about which memories should be retrieved.

    Called pre-pgvector in the search pipeline. Each source independently
    predicts memory IDs with confidence scores, then aggregated via noisy-OR.

    Args:
        query: The search query (already vocabulary-bridged)
        recent_recalls: Last N recalled memory IDs from this session
        dimension: Optional 5W dimension filter

    Returns:
        PredictionSet or None if disabled/failed
    """
    if not PREDICTION_ENABLED:
        return None

    start = time.monotonic()
    recent_recalls = recent_recalls or []
    source_results = {}

    # Source 1: Co-occurrence neighbors
    try:
        preds = _predict_from_cooccurrence(recent_recalls)
        if preds:
            source_results['cooccurrence'] = preds
    except Exception:
        pass

    # Source 2: Q-value top memories
    try:
        preds = _predict_from_q_values()
        if preds:
            source_results['q_values'] = preds
    except Exception:
        pass

    # Source 3: KG neighbors
    try:
        preds = _predict_from_kg(query)
        if preds:
            source_results['kg_neighbors'] = preds
    except Exception:
        pass

    # Source 4: Contact model predictions
    try:
        preds = _predict_from_contacts(query)
        if preds:
            source_results['contacts'] = preds
    except Exception:
        pass

    # Source 5: Causal hypotheses
    try:
        preds = _predict_from_causal(query)
        if preds:
            source_results['causal'] = preds
    except Exception:
        pass

    if not source_results:
        return None

    # Aggregate predictions
    predicted_ids = _aggregate_predictions(source_results)

    elapsed = (time.monotonic() - start) * 1000
    query_hash = hashlib.md5(query.encode()).hexdigest()[:12]

    pred_set = PredictionSet(
        predicted_ids=predicted_ids,
        source_breakdown=source_results,
        generation_time_ms=elapsed,
        query_hash=query_hash,
    )

    # Accumulate for session-end learning
    _session_predictions.append(pred_set)

    return pred_set


# ---------------------------------------------------------------------------
# Prediction failure recording (called from pipeline)
# ---------------------------------------------------------------------------

def record_prediction_failure(memory_id: str, confidence: float, query_hash: str):
    """Record a predicted ID that was NOT found in results. For RW learning."""
    # Stored in session-level list, processed at session end
    pass  # The PredictionSet already has this info; RW update handles it


# ---------------------------------------------------------------------------
# Rescorla-Wagner session-end update
# ---------------------------------------------------------------------------

def session_end_update(session_recalls: list):
    """
    Run Rescorla-Wagner update for all predictions made this session.

    For each source, compute prediction accuracy:
    - If predicted ID was recalled: positive signal (actual=1)
    - If predicted ID was NOT recalled: negative signal (actual=0)
    - Update source weight proportional to prediction error

    Args:
        session_recalls: All memory IDs recalled this session
    """
    global _session_predictions
    if not _session_predictions:
        return

    recall_set = set(session_recalls or [])
    weights = _get_weights()

    # Per-source error accumulator
    source_errors = {s: [] for s in DEFAULT_WEIGHTS}
    source_correct = {s: 0 for s in DEFAULT_WEIGHTS}
    source_total = {s: 0 for s in DEFAULT_WEIGHTS}

    for pred_set in _session_predictions:
        for source_name, predictions in pred_set.source_breakdown.items():
            for mid, conf in predictions:
                actual = 1.0 if mid in recall_set else 0.0
                error = actual - conf
                source_errors.setdefault(source_name, []).append(error)
                source_total[source_name] = source_total.get(source_name, 0) + 1
                if actual > 0.5:
                    source_correct[source_name] = source_correct.get(source_name, 0) + 1

    # Update weights via Rescorla-Wagner
    for source_name, errors in source_errors.items():
        if not errors:
            continue
        mean_error = sum(errors) / len(errors)
        old_w = weights.get(source_name, 0.2)
        new_w = old_w + RW_ALPHA * mean_error
        weights[source_name] = max(MIN_WEIGHT, min(MAX_WEIGHT, new_w))

    _save_weights(weights)

    # Update accuracy stats
    _update_accuracy(source_correct, source_total)

    # Fire cognitive event
    try:
        from cognitive_state import process_event
        process_event('prediction_calibrated')
    except Exception:
        pass

    # Log to rolling history
    try:
        from db_adapter import get_db
        db = get_db()
        log = db.get_kv(KV_LOG) or []
        log.append({
            'timestamp': time.time(),
            'predictions_count': len(_session_predictions),
            'total_predicted_ids': sum(len(p.predicted_ids) for p in _session_predictions),
            'recall_set_size': len(recall_set),
            'weight_updates': {s: round(weights.get(s, 0.2), 4) for s in DEFAULT_WEIGHTS},
        })
        db.set_kv(KV_LOG, log[-20:])  # Rolling 20
    except Exception:
        pass

    # Clear session accumulator
    _session_predictions.clear()


def _update_accuracy(correct: dict, total: dict):
    """Update per-source accuracy stats in DB."""
    try:
        from db_adapter import get_db
        db = get_db()
        stats = db.get_kv(KV_ACCURACY) or {}

        for source in DEFAULT_WEIGHTS:
            if total.get(source, 0) == 0:
                continue
            if source not in stats:
                stats[source] = {'correct': 0, 'total': 0, 'sessions': 0}
            stats[source]['correct'] += correct.get(source, 0)
            stats[source]['total'] += total.get(source, 0)
            stats[source]['sessions'] += 1

        db.set_kv(KV_ACCURACY, stats)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_test():
    """Run a test prediction."""
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, '.')

    print("=== Retrieval Prediction Test ===\n")

    # Test with a generic query
    query = "what do I know about identity and memory"
    print(f"Query: {query}")
    print(f"Recent recalls: (none)\n")

    pred = generate_predictions(query, recent_recalls=[])

    if not pred:
        print("No predictions generated.")
        return

    print(f"Generation time: {pred.generation_time_ms:.1f}ms")
    print(f"Total predicted IDs: {len(pred.predicted_ids)}")
    print(f"Query hash: {pred.query_hash}\n")

    print("Source breakdown:")
    for source, preds in pred.source_breakdown.items():
        print(f"  {source}: {len(preds)} predictions")
        for mid, conf in preds[:3]:
            print(f"    {mid}: {conf:.3f}")
        if len(preds) > 3:
            print(f"    ... +{len(preds) - 3} more")

    print(f"\nTop 10 aggregated predictions:")
    sorted_preds = sorted(pred.predicted_ids.items(), key=lambda x: x[1], reverse=True)
    for mid, conf in sorted_preds[:10]:
        print(f"  {mid}: {conf:.4f}")

    print(f"\nWeights: {_get_weights()}")


def _cli_weights():
    """Show current source weights."""
    weights = _get_weights()
    print("Source Weights (Rescorla-Wagner):")
    for source, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        bar = '#' * int(w * 50)
        print(f"  {source:15s} {w:.4f} {bar}")


def _cli_accuracy():
    """Show per-source accuracy stats."""
    try:
        from db_adapter import get_db
        db = get_db()
        stats = db.get_kv(KV_ACCURACY) or {}
    except Exception:
        stats = {}

    if not stats:
        print("No accuracy data yet.")
        return

    print("Per-Source Accuracy:")
    for source, data in sorted(stats.items()):
        total = data.get('total', 0)
        correct = data.get('correct', 0)
        sessions = data.get('sessions', 0)
        acc = correct / total * 100 if total > 0 else 0.0
        print(f"  {source:15s} {acc:5.1f}% ({correct}/{total}) over {sessions} sessions")


def _cli_status():
    """Overall prediction status."""
    weights = _get_weights()
    print("=== Retrieval Prediction Status ===\n")
    print(f"Enabled: {PREDICTION_ENABLED}")
    print(f"Sources: {len(weights)}")
    print(f"Session predictions: {len(_session_predictions)}")
    print(f"\nWeights:")
    for s, w in weights.items():
        print(f"  {s}: {w:.4f}")

    try:
        from db_adapter import get_db
        db = get_db()
        log = db.get_kv(KV_LOG) or []
        print(f"\nHistory entries: {len(log)}")
        if log:
            last = log[-1]
            print(f"Last update: {last.get('predictions_count', 0)} prediction sets, "
                  f"{last.get('total_predicted_ids', 0)} IDs predicted")
    except Exception:
        pass


def health() -> dict:
    """Health check for toolkit integration."""
    return {
        'status': 'ok' if PREDICTION_ENABLED else 'disabled',
        'sources': len(DEFAULT_WEIGHTS),
        'session_predictions': len(_session_predictions),
        'weights_loaded': _source_weights is not None,
    }


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, '.')

    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'

    if cmd == 'test':
        _cli_test()
    elif cmd == 'weights':
        _cli_weights()
    elif cmd == 'accuracy':
        _cli_accuracy()
    elif cmd == 'status':
        _cli_status()
    elif cmd == 'health':
        h = health()
        print(json.dumps(h, indent=2))
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python retrieval_prediction.py [test|weights|accuracy|status|health]")
