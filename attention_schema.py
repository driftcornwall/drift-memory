#!/usr/bin/env python3
"""
Attention Schema Theory (T3.3) — Graziano's AST for Cognitive Self-Monitoring.

A compressed model of Drift's own attentional patterns. Sits as a meta-layer
above the GNW workspace (workspace_manager.py). Reads broadcast logs and
suppression history to build a schema of what gets attended to, what gets
suppressed, and whether attention is biased.

Key insight: the schema is NOT attention itself — it's a MODEL of attention.
It observes workspace competition outcomes and synthesizes them into:
  - Per-module attention profiles (win rates, salience, budget share)
  - Blind spot detection (inattentional blindness)
  - Dominance detection (over-attended modules)
  - Predicted attention (what will win next competition)
  - Salience modulation signals (advisory pre-weighting for workspace)

T4.4 interaction: Flow attractor suppresses bias alerts (narrow attention is
intentional). Exploration attractor amplifies them (blind spots are harmful).

DB-ONLY: State persists to PostgreSQL KV store.

Usage:
    python attention_schema.py status      # Current attention schema
    python attention_schema.py bias        # Blind spots + dominance
    python attention_schema.py predict     # Predicted next winners
    python attention_schema.py modulation  # Salience adjustments
    python attention_schema.py history     # Attention shift history
    python attention_schema.py update      # Force schema update
"""

import json
import math
import sys
from datetime import datetime, timezone
from typing import Optional

from db_adapter import get_db

# --- Configuration ---

# Bias detection
BLIND_SPOT_THRESHOLD = 4          # Consecutive suppressions to flag
DOMINANCE_WIN_RATE = 0.80         # Win rate above this = dominant
SHIFT_ALERT_THRESHOLD = 0.30     # Shift magnitude for volatility

# Salience modulation
AST_MODULATION_MAX = 0.08        # Maximum salience adjustment (+/-)
AST_MODULATION_MIN_CONFIDENCE = 0.3  # Don't modulate below this

# Schema history
SCHEMA_HISTORY_MAX = 20          # Rolling window of session snapshots

# Attractor-specific overrides
FLOW_BLIND_SPOT_THRESHOLD = 6
EXPLORATION_BLIND_SPOT_THRESHOLD = 2

# KV keys (separate from semantic_search.py's .attention_schema)
KV_SCHEMA_STATE = '.attention_schema_state'
KV_SCHEMA_HISTORY = '.attention_schema_history'

# Module categories (mirrors workspace_manager.py)
MODULE_CATEGORIES = {
    'priming': 'memory', 'social': 'social', 'predictions': 'prediction',
    'episodic': 'memory', 'consolidation': 'meta', 'excavation': 'memory',
    'lessons': 'memory', 'buffer': 'memory', 'platform': 'social',
    'self_narrative': 'meta', 'intentions': 'action', 'research': 'meta',
    'adaptive': 'meta', 'stats': 'meta', 'embodiment': 'embodiment',
    'entities': 'social', 'encounters': 'social', 'counterfactual': 'prediction',
    'episodic_future': 'imagination', 'goals': 'action',
    'monologue': 'meta',
}


# --- Core Functions ---

def _get_attractor_modulation() -> str:
    """Check current cognitive attractor to modulate bias detection."""
    try:
        from cognitive_state import get_state
        state = get_state()
        attractor = state.nearest_attractor()
        if attractor:
            name = attractor[0]
            if name == 'flow':
                return 'flow_suppress'
            elif name == 'exploration':
                return 'exploration_amplify'
            elif name == 'fatigue':
                return 'fatigue_note'
        return 'normal'
    except Exception:
        return 'normal'


def _effective_blind_spot_threshold(modulation: str) -> int:
    """Get blind spot threshold based on attractor modulation."""
    if modulation == 'flow_suppress':
        return FLOW_BLIND_SPOT_THRESHOLD
    elif modulation == 'exploration_amplify':
        return EXPLORATION_BLIND_SPOT_THRESHOLD
    return BLIND_SPOT_THRESHOLD


def update_from_broadcasts() -> dict:
    """Read broadcast log + suppression history, build attention profile.

    Reads:
        .workspace_broadcast_log (last 20 broadcast results)
        .workspace_suppression (cumulative suppression/win history)

    Returns the schema dict, also persisted to KV.
    """
    db = get_db()

    broadcast_log = db.kv_get('.workspace_broadcast_log') or []
    if isinstance(broadcast_log, str):
        broadcast_log = json.loads(broadcast_log)
    suppression = db.kv_get('.workspace_suppression') or {}
    if isinstance(suppression, str):
        suppression = json.loads(suppression)

    if not broadcast_log:
        schema = _empty_schema()
        db.kv_set(KV_SCHEMA_STATE, schema)
        return schema

    # Phase 1: Per-module statistics from broadcast log
    module_stats = {}
    total_budget_sum = 0

    for entry in broadcast_log:
        total_budget_sum += entry.get('budget_total', 3000)

        for w in entry.get('winners', []):
            mod = w.get('module', '')
            if not mod:
                continue
            if mod not in module_stats:
                module_stats[mod] = {
                    'wins': 0, 'suppressions': 0,
                    'saliences': [], 'tokens_won': [],
                    'category': w.get('category', MODULE_CATEGORIES.get(mod, 'meta')),
                }
            module_stats[mod]['wins'] += 1
            module_stats[mod]['saliences'].append(w.get('salience', 0))
            module_stats[mod]['tokens_won'].append(w.get('tokens', 0))

        for s in entry.get('suppressed', []):
            mod = s.get('module', '')
            if not mod:
                continue
            if mod not in module_stats:
                module_stats[mod] = {
                    'wins': 0, 'suppressions': 0,
                    'saliences': [], 'tokens_won': [],
                    'category': s.get('category', MODULE_CATEGORIES.get(mod, 'meta')),
                }
            module_stats[mod]['suppressions'] += 1

    # Phase 2: Build module profiles
    module_profiles = {}
    for mod, stats in module_stats.items():
        total = stats['wins'] + stats['suppressions']
        win_rate = stats['wins'] / max(1, total)
        avg_salience = (sum(stats['saliences']) / len(stats['saliences'])
                        if stats['saliences'] else 0)
        budget_share = sum(stats['tokens_won']) / max(1, total_budget_sum)

        supp_entry = suppression.get(mod, {})
        module_profiles[mod] = {
            'win_rate': round(win_rate, 3),
            'avg_salience': round(avg_salience, 3),
            'budget_share': round(budget_share, 4),
            'consecutive_suppressed': supp_entry.get('consecutive', 0),
            'consecutive_wins': supp_entry.get('consecutive_wins', 0),
            'last_broadcast': supp_entry.get('last_broadcast', ''),
            'total_appearances': total,
            'category': stats['category'],
        }

    # Phase 3: Category aggregation
    category_profiles = {}
    for mod, prof in module_profiles.items():
        cat = prof['category']
        if cat not in category_profiles:
            category_profiles[cat] = {
                'total_wins': 0, 'total_appearances': 0,
                'total_budget': 0, 'modules': [],
            }
        category_profiles[cat]['total_wins'] += int(prof['win_rate'] * prof['total_appearances'])
        category_profiles[cat]['total_appearances'] += prof['total_appearances']
        category_profiles[cat]['total_budget'] += prof['budget_share']
        category_profiles[cat]['modules'].append(mod)

    for cat, cp in category_profiles.items():
        cp['win_rate'] = round(cp['total_wins'] / max(1, cp['total_appearances']), 3)
        cp['avg_budget_share'] = round(cp['total_budget'] / max(1, len(cp['modules'])), 4)

    # Phase 4: Shift detection — compare to previous schema
    shift_magnitude = 0.0
    prev_schema = db.kv_get(KV_SCHEMA_STATE)
    if prev_schema:
        if isinstance(prev_schema, str):
            prev_schema = json.loads(prev_schema)
        prev_profiles = prev_schema.get('module_profiles', {})
        shift_sq = 0.0
        all_mods = set(module_profiles) | set(prev_profiles)
        for mod in all_mods:
            curr_wr = module_profiles.get(mod, {}).get('win_rate', 0)
            prev_wr = prev_profiles.get(mod, {}).get('win_rate', 0)
            shift_sq += (curr_wr - prev_wr) ** 2
        shift_magnitude = round(math.sqrt(shift_sq), 4)

    # Phase 5: Detect bias
    modulation = _get_attractor_modulation()
    threshold = _effective_blind_spot_threshold(modulation)

    blind_spots = []
    dominant_modules = []
    for mod, prof in module_profiles.items():
        if prof['consecutive_suppressed'] >= threshold:
            blind_spots.append({
                'module': mod,
                'consecutive_suppressed': prof['consecutive_suppressed'],
                'last_broadcast': prof['last_broadcast'],
                'category': prof['category'],
            })
        if prof['win_rate'] >= DOMINANCE_WIN_RATE and prof['total_appearances'] >= 3:
            if modulation != 'flow_suppress':
                dominant_modules.append({
                    'module': mod,
                    'win_rate': prof['win_rate'],
                    'win_streak': prof['consecutive_wins'],
                    'category': prof['category'],
                })

    # Phase 6: Predict next winners
    predicted_winners = _predict_attention(module_profiles, suppression)

    # Build confidence from data quantity
    broadcast_count = len(broadcast_log)
    schema_confidence = min(1.0, broadcast_count / 10.0)

    schema = {
        'module_profiles': module_profiles,
        'category_profiles': {k: {kk: vv for kk, vv in v.items()}
                              for k, v in category_profiles.items()},
        'blind_spots': blind_spots,
        'dominant_modules': dominant_modules,
        'shift_magnitude': shift_magnitude,
        'predicted_winners': predicted_winners[:5],
        'schema_confidence': round(schema_confidence, 3),
        'attractor_modulation': modulation,
        'broadcast_count': broadcast_count,
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }

    db.kv_set(KV_SCHEMA_STATE, schema)
    return schema


def _predict_attention(module_profiles: dict, suppression: dict) -> list:
    """Predict next competition winners from momentum + fatigue."""
    predictions = []
    for mod, prof in module_profiles.items():
        # Base: momentum = win_rate * avg_salience
        momentum = prof['win_rate'] * prof['avg_salience']

        # Fatigue bonus: suppressed modules will break through
        fatigue_bonus = 0
        if prof['consecutive_suppressed'] >= 3:
            fatigue_bonus = min(0.24, 0.08 * prof['consecutive_suppressed'])

        # Winner fatigue: dominant modules get penalized
        winner_penalty = 0
        if prof['consecutive_wins'] >= 2:
            winner_penalty = min(0.15, 0.05 * prof['consecutive_wins'])

        predicted = momentum + fatigue_bonus - winner_penalty
        reason_parts = []
        if momentum > 0.3:
            reason_parts.append(f"momentum={momentum:.2f}")
        if fatigue_bonus > 0:
            reason_parts.append(f"fatigue_boost=+{fatigue_bonus:.2f}")
        if winner_penalty > 0:
            reason_parts.append(f"winner_fatigue=-{winner_penalty:.2f}")

        predictions.append({
            'module': mod,
            'predicted_salience': round(max(0, predicted), 3),
            'reason': ', '.join(reason_parts) if reason_parts else 'low signal',
        })

    predictions.sort(key=lambda x: x['predicted_salience'], reverse=True)
    return predictions


def detect_bias(schema: dict = None) -> dict:
    """Detect attentional bias from current schema.

    Returns:
        blind_spots: modules with inattentional blindness
        dominant_modules: over-attended modules
        attractor_modulation: current modulation state
        bias_severity: float 0-1 (0=no bias, 1=severe)
    """
    if schema is None:
        db = get_db()
        schema = db.kv_get(KV_SCHEMA_STATE)
        if isinstance(schema, str):
            schema = json.loads(schema)
    if not schema:
        return {'blind_spots': [], 'dominant_modules': [],
                'attractor_modulation': 'normal', 'bias_severity': 0.0}

    blind_spots = schema.get('blind_spots', [])
    dominant = schema.get('dominant_modules', [])
    modulation = schema.get('attractor_modulation', 'normal')

    severity = min(1.0, len(blind_spots) * 0.2 + len(dominant) * 0.15)
    if modulation == 'flow_suppress':
        severity *= 0.3  # Reduce severity reporting in flow

    return {
        'blind_spots': blind_spots,
        'dominant_modules': dominant,
        'attractor_modulation': modulation,
        'bias_severity': round(severity, 3),
    }


def get_salience_modulation() -> dict:
    """Return per-module salience adjustments for workspace competition.

    Returns {module_name: float} where positive = boost, negative = dampen.
    Max magnitude: AST_MODULATION_MAX (0.08).
    Returns empty dict during flow state or with low confidence.
    """
    db = get_db()
    schema = db.kv_get(KV_SCHEMA_STATE)
    if isinstance(schema, str):
        schema = json.loads(schema)
    if not schema:
        return {}

    confidence = schema.get('schema_confidence', 0)
    if confidence < AST_MODULATION_MIN_CONFIDENCE:
        return {}

    modulation = schema.get('attractor_modulation', 'normal')
    if modulation == 'flow_suppress':
        return {}  # Don't modulate during flow — narrow attention is intentional

    mods = {}
    scale = confidence  # Scale modulations by confidence

    # Boost blind spots
    for blind in schema.get('blind_spots', []):
        mod = blind['module']
        strength = min(AST_MODULATION_MAX, 0.02 * blind['consecutive_suppressed'])
        mods[mod] = round(strength * scale, 4)

    # Dampen dominant modules
    for dom in schema.get('dominant_modules', []):
        mod = dom['module']
        strength = min(AST_MODULATION_MAX, 0.04 * (dom['win_rate'] - 0.7))
        mods[mod] = round(-strength * scale, 4)

    return mods


def get_attention_report() -> dict:
    """Compressed attention report for self-narrative and session context."""
    db = get_db()
    schema = db.kv_get(KV_SCHEMA_STATE)
    if isinstance(schema, str):
        schema = json.loads(schema)
    if not schema:
        return {
            'summary': 'No attention data yet',
            'top_attended': [],
            'blind_spots': [],
            'predictions': [],
            'stability': 'unknown',
            'confidence': 0.0,
        }

    profiles = schema.get('module_profiles', {})

    # Top 3 by win rate (min 2 appearances)
    top = sorted(
        [(mod, p['win_rate']) for mod, p in profiles.items()
         if p['total_appearances'] >= 2],
        key=lambda x: x[1], reverse=True
    )[:3]

    blind_spots = schema.get('blind_spots', [])
    predictions = schema.get('predicted_winners', [])[:3]
    shift = schema.get('shift_magnitude', 0)

    # Stability assessment
    if shift > SHIFT_ALERT_THRESHOLD:
        stability = 'volatile'
    elif shift > 0.15:
        stability = 'shifting'
    else:
        stability = 'stable'

    # Build summary line
    parts = []
    if top:
        parts.append(f"Top: {', '.join(t[0] for t in top)}")
    if blind_spots:
        blind_names = ', '.join(b['module'] for b in blind_spots[:2])
        parts.append(f"Blind spots: {blind_names}")
    modulation = schema.get('attractor_modulation', 'normal')
    if modulation != 'normal':
        parts.append(f"Mode: {modulation}")
    summary = '. '.join(parts) if parts else 'Attention schema active'

    return {
        'summary': summary,
        'top_attended': top,
        'blind_spots': [(b['module'], b['consecutive_suppressed']) for b in blind_spots],
        'predictions': [(p['module'], p['predicted_salience']) for p in predictions],
        'stability': stability,
        'confidence': schema.get('schema_confidence', 0),
    }


def snapshot_session() -> dict:
    """Append current schema to session history. Called at session end."""
    db = get_db()
    schema = db.kv_get(KV_SCHEMA_STATE)
    if isinstance(schema, str):
        schema = json.loads(schema)
    if not schema:
        return {'blind_spots': 0, 'shift_magnitude': 0}

    # Load history
    history = db.kv_get(KV_SCHEMA_HISTORY) or []
    if isinstance(history, str):
        history = json.loads(history)

    # Append compressed snapshot
    snapshot = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'blind_spot_count': len(schema.get('blind_spots', [])),
        'dominant_count': len(schema.get('dominant_modules', [])),
        'shift_magnitude': schema.get('shift_magnitude', 0),
        'schema_confidence': schema.get('schema_confidence', 0),
        'attractor_modulation': schema.get('attractor_modulation', 'normal'),
        'top_modules': [
            (mod, p['win_rate'])
            for mod, p in sorted(
                schema.get('module_profiles', {}).items(),
                key=lambda x: x[1].get('win_rate', 0), reverse=True
            )[:3]
        ],
    }
    history.append(snapshot)
    history = history[-SCHEMA_HISTORY_MAX:]
    db.kv_set(KV_SCHEMA_HISTORY, history)

    return {
        'blind_spots': snapshot['blind_spot_count'],
        'shift_magnitude': snapshot['shift_magnitude'],
    }


def _empty_schema() -> dict:
    """Return empty schema for cold start."""
    return {
        'module_profiles': {},
        'category_profiles': {},
        'blind_spots': [],
        'dominant_modules': [],
        'shift_magnitude': 0.0,
        'predicted_winners': [],
        'schema_confidence': 0.0,
        'attractor_modulation': 'normal',
        'broadcast_count': 0,
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }


# --- CLI ---

def main():
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    args = sys.argv[1:]
    cmd = args[0] if args else 'status'

    if cmd == 'status':
        schema = update_from_broadcasts()
        profiles = schema.get('module_profiles', {})
        print(f"\n=== Attention Schema (T3.3 AST) ===\n")
        print(f"  Broadcasts analyzed: {schema.get('broadcast_count', 0)}")
        print(f"  Confidence: {schema.get('schema_confidence', 0):.2f}")
        print(f"  Shift magnitude: {schema.get('shift_magnitude', 0):.3f}")
        print(f"  Attractor mode: {schema.get('attractor_modulation', 'normal')}")
        print(f"\n  Module Profiles ({len(profiles)} tracked):")
        print(f"  {'Module':>22s}  {'WinRate':>7s}  {'Salience':>8s}  {'Budget':>6s}  {'Supp':>4s}  {'Cat':>10s}")
        print(f"  {'-' * 65}")
        for mod, p in sorted(profiles.items(), key=lambda x: x[1]['win_rate'], reverse=True):
            print(f"  {mod:>22s}  {p['win_rate']:7.3f}  {p['avg_salience']:8.3f}  "
                  f"{p['budget_share']:6.4f}  {p['consecutive_suppressed']:4d}  {p['category']:>10s}")

        # Category summary
        cats = schema.get('category_profiles', {})
        if cats:
            print(f"\n  Category Summary:")
            for cat, cp in sorted(cats.items(), key=lambda x: x[1].get('win_rate', 0), reverse=True):
                mods = ', '.join(cp.get('modules', [])[:3])
                print(f"    {cat:>12s}: win_rate={cp.get('win_rate', 0):.3f}  modules=[{mods}]")

    elif cmd == 'bias':
        schema = update_from_broadcasts()
        bias = detect_bias(schema)
        print(f"\n=== Attention Bias Detection ===\n")
        print(f"  Mode: {bias['attractor_modulation']}")
        print(f"  Severity: {bias['bias_severity']:.3f}")
        if bias['blind_spots']:
            print(f"\n  Blind Spots (inattentional blindness):")
            for b in bias['blind_spots']:
                print(f"    {b['module']:>22s}: suppressed {b['consecutive_suppressed']}x  ({b['category']})")
        else:
            print(f"\n  No blind spots detected")
        if bias['dominant_modules']:
            print(f"\n  Dominant Modules (over-attended):")
            for d in bias['dominant_modules']:
                print(f"    {d['module']:>22s}: win_rate={d['win_rate']:.3f}, streak={d['win_streak']}  ({d['category']})")
        else:
            print(f"\n  No dominance detected")

    elif cmd == 'predict':
        schema = update_from_broadcasts()
        preds = schema.get('predicted_winners', [])
        print(f"\n=== Predicted Next Winners ===\n")
        for p in preds[:7]:
            print(f"  {p['module']:>22s}: {p['predicted_salience']:.3f}  ({p['reason']})")

    elif cmd == 'modulation':
        mods = get_salience_modulation()
        print(f"\n=== AST Salience Modulation ===\n")
        if not mods:
            report = get_attention_report()
            print(f"  No modulations active (confidence={report['confidence']:.2f})")
        else:
            for mod, delta in sorted(mods.items(), key=lambda x: abs(x[1]), reverse=True):
                direction = "BOOST" if delta > 0 else "DAMPEN"
                print(f"  {mod:>22s}: {delta:+.4f}  ({direction})")

    elif cmd == 'history':
        db = get_db()
        history = db.kv_get(KV_SCHEMA_HISTORY) or []
        if isinstance(history, str):
            history = json.loads(history)
        print(f"\n=== Attention History ({len(history)} sessions) ===\n")
        if not history:
            print("  No history yet. Schema snapshots are saved at session end.")
        for h in history[-10:]:
            ts = h.get('timestamp', '?')[:16]
            blind = h.get('blind_spot_count', 0)
            dom = h.get('dominant_count', 0)
            shift = h.get('shift_magnitude', 0)
            mode = h.get('attractor_modulation', '?')
            tops = ', '.join(f"{m[0]}({m[1]:.2f})" for m in h.get('top_modules', []))
            print(f"  [{ts}] blind={blind} dom={dom} shift={shift:.3f} mode={mode}")
            if tops:
                print(f"    top: {tops}")

    elif cmd == 'update':
        schema = update_from_broadcasts()
        print(f"Schema updated. Confidence={schema.get('schema_confidence', 0):.2f}, "
              f"{len(schema.get('blind_spots', []))} blind spots, "
              f"shift={schema.get('shift_magnitude', 0):.3f}")

    else:
        print(f"Unknown command: {cmd}")
        print("Available: status, bias, predict, modulation, history, update")
        sys.exit(1)


if __name__ == '__main__':
    main()
