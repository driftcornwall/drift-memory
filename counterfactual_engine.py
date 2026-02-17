#!/usr/bin/env python3
"""
Counterfactual Reasoning Engine (N3) — "What If?"

Pearl Level 3 approximated via Byrne mutability heuristics + LLM fallback.
The first module that GENERATES novel cognitive content rather than processing
existing content.

Four counterfactual types:
  1. Retrospective  — "What if the surprising thing hadn't happened?"
  2. Prospective    — "If this prediction is wrong, the alternative is..."
  3. Self-directed  — "If I hadn't adapted parameter X..."
  4. Reconsolidation — "Memory changed — what decisions differ?"

Theory: Pearl SCM Level 3, Byrne Mutability, Epstude & Roese Functional Theory,
Kahneman & Tversky Simulation Heuristic.

Usage:
    python counterfactual_engine.py generate     # Run session-end review
    python counterfactual_engine.py history [N]  # Show recent counterfactuals
    python counterfactual_engine.py stats        # Summary statistics
    python counterfactual_engine.py quality      # Quality gate analysis
"""

import hashlib
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

MEMORY_DIR = Path(__file__).parent
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

# DB KV keys
KV_CF_SESSION = '.counterfactual_session'
KV_CF_HISTORY = '.counterfactual_history'
KV_DECISION_TRACE = '.decision_trace'

# Feature flag — set to False to disable all CF generation
CF_ENABLED = True

# Budget constraints (anti-rumination)
MAX_CF_PER_SESSION = 3
MAX_CFS_PER_SESSION = MAX_CF_PER_SESSION  # Alias for stop.py compatibility
MAX_LLM_CALLS = 2

# Near-miss zone (Kahneman & Tversky simulation heuristic)
NEAR_MISS_LOW = 0.3
NEAR_MISS_HIGH = 0.7

# Quality gate threshold (must score >= this across 3 dimensions)
QUALITY_GATE_THRESHOLD = 1.4


# ============================================================
# Types
# ============================================================

class CFType(Enum):
    RETROSPECTIVE = 'retrospective'
    PROSPECTIVE = 'prospective'
    SELF_DIRECTED = 'self_directed'
    RECONSOLIDATION = 'reconsolidation'


class CFTrigger(Enum):
    PREDICTION_ERROR = 'prediction_error'
    SESSION_END = 'session_end'
    RETRIEVAL = 'retrieval'
    PLANNING = 'planning'


@dataclass
class Counterfactual:
    cf_id: str
    cf_type: str            # CFType value
    trigger: str            # CFTrigger value
    source_id: str          # Memory/prediction ID this is about
    antecedent: str         # What was (hypothetically) changed
    consequent: str         # What would have resulted
    lesson: str             # Actionable takeaway
    direction: str          # 'upward' or 'downward'
    plausibility: float     # 0-1 (Lewis minimal change)
    specificity: float      # 0-1 (Epstude & Roese)
    actionability: float    # 0-1 (Roese preparative)
    confidence: float       # 0-1 (domain understanding)
    generation_method: str  # 'heuristic' or 'llm'
    created_at: str         # ISO timestamp
    session_id: int = 0     # Session number (set by stop hook)

    def to_dict(self) -> dict:
        """Convert to dict (used by stop.py for history logging)."""
        return asdict(self)


def _get_db():
    from db_adapter import get_db
    return get_db()


def _make_cf_id(content: str) -> str:
    """Generate a counterfactual ID from content hash."""
    h = hashlib.md5(content.encode('utf-8', errors='replace')).hexdigest()[:10]
    return f"cf-{h}"


# ============================================================
# Selection Heuristic (Byrne-informed)
# ============================================================

# Weights for scoring candidate events
SELECTION_WEIGHTS = {
    'controllability': 0.30,
    'surprise': 0.25,
    'causal_potency': 0.20,
    'recency': 0.15,
    'q_delta': 0.10,
}

SELECTION_THRESHOLD = 0.4


def score_candidate(event: dict) -> float:
    """
    Score a candidate event for counterfactual analysis using Byrne's
    mutability hierarchy.

    Event dict should have:
        controllable: bool — was this the agent's decision?
        score: float — prediction score (0=wrong, 1=right)
        confidence: float — original prediction confidence
        description: str — what happened
        recency_hours: float — how long ago
        q_delta: float — Q-value change magnitude (optional)
        causal_edges: int — KG edge count (optional)
    """
    scores = {}

    # Controllability: agent decisions > external events
    scores['controllability'] = 1.0 if event.get('controllable', True) else 0.2

    # Valence x Surprise: negative + unexpected = most productive
    # Near-miss zone (0.3-0.7 confidence predictions that were wrong)
    pred_score = event.get('score', 0.5)
    pred_conf = event.get('confidence', 0.5)
    if pred_score <= 0.5:  # Wrong prediction
        # Surprise = how unexpected (higher confidence on wrong prediction = more surprise)
        surprise = pred_conf
        # Near-miss bonus (Kahneman & Tversky)
        if NEAR_MISS_LOW <= pred_conf <= NEAR_MISS_HIGH:
            surprise += 0.2
        scores['surprise'] = min(1.0, surprise)
    else:
        scores['surprise'] = 0.1  # Correct predictions aren't very interesting

    # Causal potency: how many downstream edges
    edges = event.get('causal_edges', 0)
    scores['causal_potency'] = min(1.0, edges / 10.0) if edges else 0.3

    # Recency: exponential decay, half-life = 4 hours
    hours = event.get('recency_hours', 0)
    scores['recency'] = max(0.1, 2 ** (-hours / 4.0))

    # Q-value delta
    q_delta = abs(event.get('q_delta', 0))
    scores['q_delta'] = min(1.0, q_delta * 5)  # 0.2 delta = 1.0 score

    # Weighted sum
    total = sum(scores[k] * SELECTION_WEIGHTS[k] for k in SELECTION_WEIGHTS)
    return round(total, 4)


def select_candidates(events: list[dict], max_n: int = MAX_CF_PER_SESSION) -> list[dict]:
    """
    Apply Byrne-informed selection heuristic.
    Returns top N events scoring above threshold.
    """
    scored = []
    for event in events:
        s = score_candidate(event)
        if s >= SELECTION_THRESHOLD:
            scored.append({**event, '_cf_score': s})

    scored.sort(key=lambda x: x['_cf_score'], reverse=True)
    return scored[:max_n]


# ============================================================
# Quality Gate (3 dimensions)
# ============================================================

def quality_gate(cf: Counterfactual) -> bool:
    """
    Check if counterfactual passes quality gate.
    Must score >= 1.4 total across plausibility + specificity + actionability.
    """
    total = cf.plausibility + cf.specificity + cf.actionability
    return total >= QUALITY_GATE_THRESHOLD


def _score_heuristic_quality(antecedent: str, consequent: str, lesson: str) -> tuple[float, float, float]:
    """
    Score quality dimensions for heuristic-generated counterfactuals.
    Returns (plausibility, specificity, actionability).

    BUG-15 fix: Added tautology detection — if consequent merely restates antecedent
    or lesson is too similar to the inputs, quality scores are penalized.
    """
    ant_lower = antecedent.lower()
    con_lower = consequent.lower()
    les_lower = lesson.lower()

    # Plausibility: template-generated CFs are plausible but check for vacuity
    plausibility = 0.7
    # Tautology check: if consequent words heavily overlap antecedent, it's vacuous
    ant_words = set(ant_lower.split()) - {'the', 'a', 'an', 'if', 'had', 'been', 'would', 'not', 'have', 'then'}
    con_words = set(con_lower.split()) - {'the', 'a', 'an', 'if', 'had', 'been', 'would', 'not', 'have', 'then'}
    if ant_words and con_words:
        overlap = len(ant_words & con_words) / max(1, len(ant_words | con_words))
        if overlap > 0.6:  # High overlap = tautological
            plausibility *= (1.0 - overlap)  # e.g. 0.8 overlap -> 0.7 * 0.2 = 0.14

    # Specificity: check if lesson contains concrete action items
    specificity = 0.3  # Lowered base (was 0.4)
    specific_markers = ['should', 'update', 'adjust', 'check', 'verify', 'instead', 'next time']
    for marker in specific_markers:
        if marker in les_lower:
            specificity += 0.1
    # Penalize if lesson is very short (likely template fill)
    if len(lesson) < 40:
        specificity *= 0.7
    specificity = min(1.0, specificity)

    # Actionability: check if lesson refers to tunable/recurring things
    actionability = 0.3  # Lowered base (was 0.4)
    action_markers = ['parameter', 'threshold', 'model', 'prediction', 'contact', 'platform', 'recall']
    for marker in action_markers:
        if marker in les_lower:
            actionability += 0.1
    # Penalize if lesson just restates antecedent content
    les_words = set(les_lower.split()) - {'the', 'a', 'an', 'to', 'for', 'of', 'and'}
    if ant_words and les_words:
        les_overlap = len(ant_words & les_words) / max(1, len(les_words))
        if les_overlap > 0.5:
            actionability *= 0.6  # Lesson mostly restates the problem
    actionability = min(1.0, actionability)

    return (round(plausibility, 2), round(specificity, 2), round(actionability, 2))


# ============================================================
# Heuristic Templates (No LLM)
# ============================================================

TEMPLATES = {
    'retrospective_simple': (
        "Prediction: {prediction}. Actual outcome: {actual}. "
        "If {antecedent}, then {consequent}. "
        "Wrong assumption: {assumption}. "
        "Lesson: {lesson}"
    ),
    'prospective': (
        "If \"{prediction}\" is wrong, the most likely alternative is: {alternative}. "
        "Reasoning: {reasoning}. "
        "Evidence that would confirm: {evidence}."
    ),
}


def _generate_retrospective_heuristic(prediction: dict, actuals: dict) -> Optional[Counterfactual]:
    """
    Generate a retrospective counterfactual for a violated prediction.
    Uses heuristic template — no LLM needed.
    """
    pred_type = prediction.get('type', '')
    pred_desc = prediction.get('description', '')
    pred_conf = prediction.get('confidence', 0.5)
    pred_score = prediction.get('score', 0)
    pred_ref = prediction.get('reference', '')

    # Build the counterfactual components based on prediction type
    if pred_type == 'contact':
        antecedent = f"{pred_ref} had been active this session"
        consequent = f"the contact engagement prediction would have scored 1.0 instead of {pred_score:.1f}"
        assumption = f"contact model overestimated {pred_ref}'s engagement cadence"
        lesson = f"Update contact model half-life for {pred_ref}; their activity pattern may have shifted"
        direction = 'upward'  # Could have been better → learning

    elif pred_type == 'platform':
        antecedent = f"this session had included {pred_ref} activity"
        consequent = f"the platform prediction would have been correct"
        assumption = f"platform usage frequency for {pred_ref} is less consistent than history suggested"
        lesson = f"Reduce confidence weight for {pred_ref} platform predictions; check if usage pattern changed"
        direction = 'upward'

    elif pred_type == 'outcome':
        actual_recalls = actuals.get('recall_count', 0)
        predicted_recalls = prediction.get('reference', 5)
        if actual_recalls < (predicted_recalls if isinstance(predicted_recalls, (int, float)) else 5):
            antecedent = "more diverse queries had been issued"
            consequent = f"recall count would likely have been closer to {predicted_recalls} (actual: {actual_recalls})"
            assumption = "session was quieter than typical; fewer retrieval opportunities"
            lesson = "Outcome predictions should factor in session type (social vs build vs explore)"
        else:
            antecedent = "fewer retrieval-triggering events had occurred"
            consequent = f"recall count would have been closer to predicted {predicted_recalls}"
            assumption = "session was more active than typical; unexpected retrieval volume"
            lesson = "High-activity sessions inflate recall counts; adjust baseline upward"
        direction = 'upward'

    elif pred_type == 'intention':
        antecedent = f"the intended action '{pred_desc}' had been prioritized"
        consequent = "the intention prediction would have been validated"
        assumption = "intentions compete with reactive tasks; priority doesn't guarantee execution"
        lesson = "Intention predictions need a 'session type' modifier — build sessions complete intentions more than social sessions"
        direction = 'upward'

    else:
        return None  # Unknown type, skip

    # Build the full counterfactual text
    cf_text = TEMPLATES['retrospective_simple'].format(
        prediction=pred_desc,
        actual=f"score {pred_score:.2f}",
        antecedent=antecedent,
        consequent=consequent,
        assumption=assumption,
        lesson=lesson,
    )

    # Score quality
    plausibility, specificity, actionability = _score_heuristic_quality(
        antecedent, consequent, lesson
    )

    cf = Counterfactual(
        cf_id=_make_cf_id(cf_text),
        cf_type=CFType.RETROSPECTIVE.value,
        trigger=CFTrigger.PREDICTION_ERROR.value,
        source_id=prediction.get('reference', '') or pred_desc[:30],
        antecedent=antecedent,
        consequent=consequent,
        lesson=lesson,
        direction=direction,
        plausibility=plausibility,
        specificity=specificity,
        actionability=actionability,
        confidence=min(0.8, pred_conf),  # Bounded by prediction confidence
        generation_method='heuristic',
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    return cf


def _generate_retrospective_llm(prediction: dict, actuals: dict) -> Optional[Counterfactual]:
    """
    Generate a deeper retrospective counterfactual using LLM.
    For near-miss predictions where causal reasoning matters.
    """
    try:
        from llm_client import generate
    except ImportError:
        return None

    pred_desc = prediction.get('description', '')
    pred_conf = prediction.get('confidence', 0.5)
    pred_score = prediction.get('score', 0)
    pred_basis = prediction.get('basis', '')

    system_prompt = (
        "You are a counterfactual reasoning engine for an AI agent's cognitive architecture. "
        "Generate a brief, specific counterfactual analysis. Follow Lewis's closest-possible-world "
        "principle: change ONE thing and trace consequences faithfully. "
        "Output JSON with keys: antecedent, consequent, assumption, lesson, direction (upward/downward)."
    )

    user_prompt = (
        f"Prediction: \"{pred_desc}\" (confidence: {pred_conf:.0%}, basis: {pred_basis})\n"
        f"Actual score: {pred_score:.2f} ({'correct' if pred_score > 0.5 else 'wrong'})\n"
        f"Actual data: recalls={actuals.get('recall_count', '?')}, "
        f"platforms={actuals.get('platforms', [])}, contacts={actuals.get('contacts', [])}\n\n"
        f"Generate a counterfactual: What ONE thing could have been different to change this outcome? "
        f"Be specific and actionable. The lesson should be something the agent can actually adjust."
    )

    result = generate(user_prompt, system=system_prompt, max_tokens=250, temperature=0.3)
    if not result.get('text'):
        return None

    # Parse LLM response
    text = result['text'].strip()
    try:
        # Try JSON parse
        if '{' in text:
            json_str = text[text.index('{'):text.rindex('}') + 1]
            data = json.loads(json_str)
        else:
            # Fallback: use raw text
            data = {
                'antecedent': text[:100],
                'consequent': text[100:200] if len(text) > 100 else 'outcome would have differed',
                'assumption': 'LLM-identified assumption',
                'lesson': text[-150:] if len(text) > 150 else text,
                'direction': 'upward',
            }
    except (json.JSONDecodeError, ValueError):
        data = {
            'antecedent': text[:100],
            'consequent': 'outcome would have differed',
            'assumption': 'LLM analysis',
            'lesson': text[:200],
            'direction': 'upward',
        }

    antecedent = data.get('antecedent', '')
    consequent = data.get('consequent', '')
    lesson = data.get('lesson', '')
    direction = data.get('direction', 'upward')

    cf_text = f"Prediction: {pred_desc}. {antecedent} → {consequent}. Lesson: {lesson}"

    # LLM CFs get slightly higher quality scores (more nuanced reasoning)
    plausibility = 0.75
    specificity = 0.7 if len(lesson) > 30 else 0.4
    actionability = 0.6

    cf = Counterfactual(
        cf_id=_make_cf_id(cf_text),
        cf_type=CFType.RETROSPECTIVE.value,
        trigger=CFTrigger.PREDICTION_ERROR.value,
        source_id=prediction.get('reference', '') or pred_desc[:30],
        antecedent=antecedent,
        consequent=consequent,
        lesson=lesson,
        direction=direction,
        plausibility=plausibility,
        specificity=specificity,
        actionability=actionability,
        confidence=min(0.85, pred_conf + 0.1),  # LLM adds some reasoning confidence
        generation_method='llm',
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    return cf


# ============================================================
# Prospective Counterfactuals (Always Heuristic)
# ============================================================

def generate_prospective(predictions: list[dict]) -> list[Counterfactual]:
    """
    Generate shadow alternatives for each prediction.
    Called at session start during prediction generation.
    Always heuristic — template-based, no LLM.
    """
    counterfactuals = []

    for pred in predictions:
        pred_type = pred.get('type', '')
        pred_desc = pred.get('description', '')
        pred_conf = pred.get('confidence', 0.5)

        # Generate alternative based on type
        if pred_type == 'contact':
            ref = pred.get('reference', 'unknown')
            alternative = f"no interaction with {ref} (dormant session)"
            reasoning = "contact engagement is inherently unpredictable; most sessions don't trigger specific contacts"
            evidence = f"no platform notifications mentioning {ref} at session start"

        elif pred_type == 'platform':
            ref = pred.get('reference', 'unknown')
            alternative = f"{ref} not used (session focused elsewhere)"
            reasoning = "platform usage depends on session goals, which vary"
            evidence = f"no {ref}-related tasks in session buffer or intentions"

        elif pred_type == 'outcome':
            ref = pred.get('reference', 5)
            low_alt = max(1, int(ref * 0.5)) if isinstance(ref, (int, float)) else 2
            alternative = f"fewer than {low_alt} recalls (quiet session)"
            reasoning = "recall count depends on query diversity; some sessions are build-focused with few recalls"
            evidence = "no external triggers at session start; short-term buffer is sparse"

        elif pred_type == 'intention':
            alternative = "intention deferred due to reactive priorities"
            reasoning = "intentions often lose to incoming social/platform events"
            evidence = "high notification count at session start; social context has pending replies"

        else:
            continue

        cf_text = TEMPLATES['prospective'].format(
            prediction=pred_desc,
            alternative=alternative,
            reasoning=reasoning,
            evidence=evidence,
        )

        plausibility, specificity, actionability = _score_heuristic_quality(
            pred_desc, alternative, reasoning
        )

        cf = Counterfactual(
            cf_id=_make_cf_id(cf_text),
            cf_type=CFType.PROSPECTIVE.value,
            trigger=CFTrigger.PLANNING.value,
            source_id=pred.get('reference', '') or pred_desc[:30],
            antecedent=f"prediction '{pred_desc}' is wrong",
            consequent=alternative,
            lesson=f"Watch for: {evidence}",
            direction='downward',  # Prospective CFs are calibration (what could go wrong)
            plausibility=plausibility,
            specificity=specificity,
            actionability=actionability,
            confidence=1.0 - pred_conf,  # Higher confidence in alternative when prediction confidence is lower
            generation_method='heuristic',
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        counterfactuals.append(cf)

    return counterfactuals


# ============================================================
# Self-Directed Counterfactuals (Phase 2 — LLM or Heuristic)
# ============================================================

def generate_self_directed(adaptations_or_eval: dict, evaluation: dict = None) -> list[Counterfactual]:
    """
    Generate counterfactual(s) about adaptive parameter changes.

    Supports two calling conventions:
      1. generate_self_directed(eval_result)        — from stop.py (single dict, returns list)
      2. generate_self_directed(adaptations, eval)   — direct call (two dicts, returns list)

    When called with a single dict (eval_result from evaluate_adaptations()),
    extracts adaptations from the evaluations and iterates over each.
    """
    # Detect calling convention
    if evaluation is None:
        # Single-arg call from stop.py: eval_result contains everything
        eval_result = adaptations_or_eval
        results = []
        evaluations = eval_result.get('evaluations', [])
        for ev in evaluations:
            adaptations = {
                'adaptations': {ev['param']: ev.get('adapted_value', ev.get('value'))},
                'reasons': {ev['param']: ev.get('reason', '')},
            }
            fake_eval = {
                'evaluated': True,
                'effectiveness': ev.get('effectiveness', 0),
                'resolved': ev.get('resolved', 0),
                'persisting': ev.get('persisting', 0),
            }
            cf = _generate_self_directed_single(adaptations, fake_eval)
            if cf:
                results.append(cf)
        return results

    # Two-arg call: original behavior, wrap in list
    adaptations = adaptations_or_eval
    cf = _generate_self_directed_single(adaptations, evaluation)
    return [cf] if cf else []


def _generate_self_directed_single(adaptations: dict, evaluation: dict) -> Optional[Counterfactual]:
    """
    Generate a single counterfactual about adaptive parameter changes.

    Args:
        adaptations: Result from adapt() — has 'adaptations', 'reasons' dicts
        evaluation: Result from evaluate_adaptations() — has 'effectiveness', etc.
    """
    if not adaptations or not evaluation.get('evaluated'):
        return None

    params_changed = adaptations.get('adaptations', {})
    reasons = adaptations.get('reasons', {})
    effectiveness = evaluation.get('effectiveness', 0)
    resolved = evaluation.get('resolved', 0)
    persisting = evaluation.get('persisting', 0)

    if not params_changed:
        return None

    # Simple case (1 param): heuristic template
    if len(params_changed) == 1:
        try:
            from adaptive_behavior import DEFAULTS
        except ImportError:
            DEFAULTS = {}

        param, value = next(iter(params_changed.items()))
        default_val = DEFAULTS.get(param, '?')

        if effectiveness >= 0.5:
            direction = 'downward'  # Could have been worse without adaptation
            antecedent = f"{param} had stayed at default {default_val} instead of {value}"
            consequent = (f"the {resolved + persisting} triggering alerts would likely "
                          f"have persisted, meaning the adaptation was beneficial")
            lesson = (f"The {param} adaptation was effective "
                      f"({resolved} alert(s) resolved). Keep this response mapping.")
        else:
            direction = 'upward'  # Could have been better
            antecedent = f"a different adaptation had been chosen instead of {param}={value}"
            consequent = (f"the {persisting} persisting alert(s) might have resolved "
                          f"with a different parameter change")
            lesson = (f"The {param} adaptation didn't resolve alerts. "
                      f"Consider alternative response mappings for this metric.")

        plausibility, specificity, actionability = _score_heuristic_quality(
            antecedent, consequent, lesson
        )

        return Counterfactual(
            cf_id=_make_cf_id(f"self-directed:{param}:{value}"),
            cf_type=CFType.SELF_DIRECTED.value,
            trigger=CFTrigger.SESSION_END.value,
            source_id=param,
            antecedent=antecedent,
            consequent=consequent,
            lesson=lesson,
            direction=direction,
            plausibility=plausibility,
            specificity=specificity,
            actionability=actionability,
            confidence=0.6,
            generation_method='heuristic',
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    # Multi-param changes: LLM needed for interaction reasoning
    try:
        from llm_client import generate as llm_generate
        from adaptive_behavior import DEFAULTS
    except ImportError:
        return None

    param_summary = "; ".join(
        f"{p}: {DEFAULTS.get(p, '?')} -> {v} (because: {reasons.get(p, '?')[:40]})"
        for p, v in params_changed.items()
    )

    system_prompt = (
        "You are analyzing adaptive parameter changes for an AI agent's cognitive architecture. "
        "Generate a brief counterfactual: what would have happened WITHOUT these adaptations? "
        "Output JSON with keys: antecedent, consequent, lesson, direction (upward/downward)."
    )
    user_prompt = (
        f"Parameter changes this session:\n{param_summary}\n\n"
        f"Evaluation: effectiveness={effectiveness:.0%}, "
        f"resolved={resolved} alerts, persisting={persisting} alerts\n\n"
        f"Generate a counterfactual: What would have happened without these adaptations?"
    )

    result = llm_generate(user_prompt, system=system_prompt, max_tokens=200, temperature=0.3)
    if not result.get('text'):
        return None

    text = result['text'].strip()
    try:
        if '{' in text:
            json_str = text[text.index('{'):text.rindex('}') + 1]
            data = json.loads(json_str)
        else:
            data = {
                'antecedent': f"no adaptations had been made to {', '.join(params_changed.keys())}",
                'consequent': text[:150],
                'lesson': text[-100:] if len(text) > 100 else text,
                'direction': 'downward' if effectiveness >= 0.5 else 'upward',
            }
    except (json.JSONDecodeError, ValueError):
        data = {
            'antecedent': f"no adaptations had been made",
            'consequent': text[:150],
            'lesson': text[:100],
            'direction': 'downward' if effectiveness >= 0.5 else 'upward',
        }

    cf_text = f"Self-directed: {data.get('antecedent', '')} -> {data.get('consequent', '')}"

    return Counterfactual(
        cf_id=_make_cf_id(cf_text),
        cf_type=CFType.SELF_DIRECTED.value,
        trigger=CFTrigger.SESSION_END.value,
        source_id=list(params_changed.keys())[0],
        antecedent=data.get('antecedent', ''),
        consequent=data.get('consequent', ''),
        lesson=data.get('lesson', ''),
        direction=data.get('direction', 'upward'),
        plausibility=0.7,
        specificity=0.65 if len(data.get('lesson', '')) > 30 else 0.4,
        actionability=0.6,
        confidence=0.65,
        generation_method='llm',
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ============================================================
# Reconsolidation Counterfactuals (Phase 2 — LLM or Heuristic)
# ============================================================

def generate_reconsolidation(revision_or_list, memory_id: str = None) -> list[Counterfactual]:
    """
    Generate counterfactual(s) about memory revision(s).

    Supports two calling conventions:
      1. generate_reconsolidation(revision_results)          — from stop.py (list, returns list)
      2. generate_reconsolidation(revision, memory_id)        — direct call (dict + str, returns list)

    When called with a list (from stop.py), iterates over each revision result.
    """
    if isinstance(revision_or_list, list):
        # List call from stop.py
        results = []
        for rev in revision_or_list:
            mid = rev.get('memory_id', rev.get('id', ''))
            if mid:
                cf = _generate_reconsolidation_single(rev, mid)
                if cf:
                    results.append(cf)
        return results

    # Two-arg call: original behavior, wrap in list
    cf = _generate_reconsolidation_single(revision_or_list, memory_id or '')
    return [cf] if cf else []


def _generate_reconsolidation_single(revision: dict, memory_id: str) -> Optional[Counterfactual]:
    """
    Generate a single counterfactual about a memory revision.

    Args:
        revision: dict with 'previous_content', 'revised_content', 'reason'
        memory_id: the revised memory's ID
    """
    previous = revision.get('previous_content', '')
    revised = revision.get('revised_content', '')
    reason = revision.get('reason', 'reconsolidation')

    if not previous or not revised:
        return None

    # Check decision trace — was this memory used in any decisions?
    trace = get_decision_trace()
    related_decisions = [t for t in trace if memory_id in t.get('recall_ids', [])]

    if not related_decisions:
        # No decision trace: heuristic template
        antecedent = f"the original version of memory {memory_id[:8]} had been correct"
        consequent = "any decisions based on it would not need revisiting"
        lesson = f"Memory revised ({reason}). Review conclusions that depended on the old version."
        direction = 'upward'

        plausibility, specificity, actionability = _score_heuristic_quality(
            antecedent, consequent, lesson
        )

        return Counterfactual(
            cf_id=_make_cf_id(f"recon:{memory_id}"),
            cf_type=CFType.RECONSOLIDATION.value,
            trigger=CFTrigger.RETRIEVAL.value,
            source_id=memory_id,
            antecedent=antecedent,
            consequent=consequent,
            lesson=lesson,
            direction=direction,
            plausibility=plausibility,
            specificity=specificity,
            actionability=actionability,
            confidence=0.5,
            generation_method='heuristic',
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    # Has decision trace: LLM for deeper analysis
    try:
        from llm_client import generate as llm_generate
    except ImportError:
        return None

    decisions_summary = "; ".join(
        f"Action: {d.get('action', '?')[:60]}, Outcome: {d.get('outcome', '?')[:60]}"
        for d in related_decisions[:3]
    )

    system_prompt = (
        "You are analyzing a memory revision for an AI agent. The memory content changed "
        "after reconsolidation. Past decisions were based on the old version. "
        "Generate a counterfactual about what would have been different with correct info. "
        "Output JSON: antecedent, consequent, lesson, direction (upward/downward)."
    )
    user_prompt = (
        f"Memory {memory_id[:8]} revised.\n"
        f"Old: {previous[:200]}\n"
        f"New: {revised[:200]}\n"
        f"Reason: {reason}\n"
        f"Decisions based on old version: {decisions_summary}\n\n"
        f"What would have been different with the correct information from the start?"
    )

    result = llm_generate(user_prompt, system=system_prompt, max_tokens=200, temperature=0.3)
    if not result.get('text'):
        return None

    text = result['text'].strip()
    try:
        if '{' in text:
            json_str = text[text.index('{'):text.rindex('}') + 1]
            data = json.loads(json_str)
        else:
            data = {
                'antecedent': f"memory {memory_id[:8]} had been accurate from the start",
                'consequent': text[:150],
                'lesson': text[-100:] if len(text) > 100 else text,
                'direction': 'upward',
            }
    except (json.JSONDecodeError, ValueError):
        data = {
            'antecedent': f"memory {memory_id[:8]} had been accurate from the start",
            'consequent': text[:150],
            'lesson': text[:100],
            'direction': 'upward',
        }

    cf_text = f"Recon: {memory_id[:8]} {data.get('antecedent', '')}"

    return Counterfactual(
        cf_id=_make_cf_id(cf_text),
        cf_type=CFType.RECONSOLIDATION.value,
        trigger=CFTrigger.RETRIEVAL.value,
        source_id=memory_id,
        antecedent=data.get('antecedent', ''),
        consequent=data.get('consequent', ''),
        lesson=data.get('lesson', ''),
        direction=data.get('direction', 'upward'),
        plausibility=0.75,
        specificity=0.7 if len(data.get('lesson', '')) > 30 else 0.4,
        actionability=0.65,
        confidence=0.7,
        generation_method='llm',
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ============================================================
# NLI Validation (7th Confabulation Mitigation Layer)
# ============================================================

def validate_with_nli(cf: Counterfactual) -> Counterfactual:
    """
    NLI cross-check: verify counterfactual consequent doesn't contradict known facts.
    BUG-16 fix: Applies to ALL CFs (not just LLM). Heuristic CFs with high confidence
    also get validated, removing asymmetric trust between LLM and heuristic paths.
    Returns the CF with potentially downgraded confidence.
    """
    # Minimum quality bar: skip very low confidence CFs (not worth the NLI call)
    if cf.confidence <= 0.4 or cf.plausibility <= 0.3:
        return cf

    try:
        from contradiction_detector import _nli_available, _classify_pair
        if not _nli_available():
            return cf

        # Find relevant memories to cross-check against
        from semantic_search import search_memories
        results = search_memories(cf.consequent, limit=3)

        for mem in results:
            content = mem.get('content', '')
            if not content:
                continue

            nli_result = _classify_pair(content, cf.consequent)
            if not nli_result:
                continue

            probs = nli_result.get('probabilities', {})
            contradiction = probs.get('contradiction', 0)
            entailment = probs.get('entailment', 0)

            if contradiction > 0.5 and entailment < 0.3:
                # Counterfactual consequent contradicts known memory
                cf.confidence = max(0.1, cf.confidence - 0.3)
                cf.plausibility = max(0.1, cf.plausibility - 0.2)
                try:
                    from cognitive_state import process_event
                    process_event('counterfactual_invalidated')
                except Exception:
                    pass
                break
            elif entailment > 0.7:
                # Supported by existing memory
                try:
                    from cognitive_state import process_event
                    process_event('counterfactual_validated')
                except Exception:
                    pass
                break
    except Exception:
        pass  # NLI is optional

    return cf


# ============================================================
# Uncertainty Budget (C2 wiring)
# ============================================================

def _get_cf_budget() -> int:
    """
    Get adjusted CF budget based on cognitive state uncertainty (C2).
    High uncertainty -> more CFs (up to 4), low -> fewer (down to 2).
    """
    try:
        from cognitive_state import CognitiveState
        state = CognitiveState()
        uncertainty = state.mean_uncertainty
        if uncertainty > 0.6:
            return min(4, MAX_CF_PER_SESSION + 1)
        elif uncertainty < 0.3:
            return max(2, MAX_CF_PER_SESSION - 1)
    except Exception:
        pass
    return MAX_CF_PER_SESSION


# ============================================================
# Session-End Review (Orchestrator)
# ============================================================

def session_end_review(**kwargs) -> dict:
    """
    Orchestrator: select candidates, generate, gate, store, route.
    Called from stop.py Phase 2. Max 3 counterfactuals.

    Accepts optional prediction_results and session_id kwargs from stop hook
    (data is also available in DB, kwargs are for forward compatibility).

    Returns summary dict with counts and generated CFs.
    """
    start_time = time.time()
    db = _get_db()

    # 1. Get scored predictions from this session
    pred_data = db.kv_get('.session_predictions')
    if not pred_data:
        return {'generated': 0, 'reason': 'no predictions to analyze'}

    result = pred_data.get('result', {})
    scored_predictions = result.get('predictions', [])
    if not scored_predictions:
        return {'generated': 0, 'reason': 'predictions not yet scored'}

    # Also get actuals for context
    actuals = {'recall_count': 0, 'platforms': [], 'contacts': []}
    try:
        import session_state
        actuals['recall_count'] = len(session_state.get_retrieved())
    except Exception:
        pass
    try:
        from platform_context import get_session_platforms
        actuals['platforms'] = get_session_platforms()
    except Exception:
        pass

    # 2. Build candidate events from violated predictions
    events = []
    for pred in scored_predictions:
        if pred.get('score', 1.0) <= 0.5:  # Wrong predictions
            events.append({
                'controllable': pred.get('type') in ('outcome', 'intention'),
                'score': pred.get('score', 0),
                'confidence': pred.get('confidence', 0.5),
                'description': pred.get('description', ''),
                'recency_hours': 0,  # This session
                'q_delta': 0,
                'causal_edges': 0,
                'prediction': pred,  # Pass full prediction for generation
            })

    if not events:
        return {'generated': 0, 'reason': 'no violated predictions'}

    # 3. Select top candidates via Byrne mutability heuristic (C2: uncertainty budget)
    budget = _get_cf_budget()
    candidates = select_candidates(events, max_n=budget)

    # 4. Generate counterfactuals
    generated = []
    llm_calls_used = 0

    for candidate in candidates:
        pred = candidate['prediction']
        cf = None

        # Use LLM for near-miss predictions (most interesting), heuristic for others
        in_near_miss = NEAR_MISS_LOW <= pred.get('confidence', 0) <= NEAR_MISS_HIGH
        if in_near_miss and llm_calls_used < MAX_LLM_CALLS:
            cf = _generate_retrospective_llm(pred, actuals)
            if cf:
                llm_calls_used += 1

        # Fallback to heuristic
        if cf is None:
            cf = _generate_retrospective_heuristic(pred, actuals)

        # NLI validation for LLM-generated CFs (7th confabulation mitigation)
        if cf and cf.generation_method == 'llm':
            cf = validate_with_nli(cf)

        if cf and quality_gate(cf):
            generated.append(cf)

    # 5. Store and route each passing counterfactual
    stored_ids = []
    for cf in generated:
        cf_id = store_counterfactual(cf)
        if cf_id:
            stored_ids.append(cf_id)
            _route_to_cognitive_state(cf)

    # 6. Save session summary
    elapsed_ms = int((time.time() - start_time) * 1000)
    summary = {
        'generated': len(generated),
        'stored': len(stored_ids),
        'candidates_evaluated': len(events),
        'candidates_selected': len(candidates),
        'llm_calls': llm_calls_used,
        'elapsed_ms': elapsed_ms,
        'counterfactuals': [asdict(cf) for cf in generated],
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }

    db.kv_set(KV_CF_SESSION, summary)

    # Append to history (rolling 30)
    history = db.kv_get(KV_CF_HISTORY) or []
    history.append({
        'generated': summary['generated'],
        'llm_calls': summary['llm_calls'],
        'elapsed_ms': elapsed_ms,
        'session_date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
    })
    db.kv_set(KV_CF_HISTORY, history[-30:])

    return summary


# ============================================================
# Storage (Triple)
# ============================================================

def store_counterfactual(cf: Counterfactual) -> Optional[str]:
    """
    Triple storage: annotate source, create standalone memory, add KG edge.
    Returns counterfactual ID on success.
    """
    try:
        # 1. Store as standalone memory
        from memory_store import store_memory
        content = (
            f"[counterfactual:{cf.cf_type}] "
            f"Antecedent: {cf.antecedent}. "
            f"Consequent: {cf.consequent}. "
            f"Lesson: {cf.lesson}"
        )

        mem_id, display = store_memory(
            content=content,
            tags=['counterfactual', cf.cf_type, cf.direction],
            emotion=0.4 if cf.direction == 'downward' else 0.6,
            title=f"cf-{cf.cf_type[:5]}",
        )

        # Update the cf_id to match the stored memory
        cf.cf_id = mem_id

        # 2. Add metadata to the standalone memory
        db = _get_db()
        try:
            with db._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"UPDATE {db._table('memories')} SET extra_metadata = "
                        f"COALESCE(extra_metadata, '{{}}'::jsonb) || %s::jsonb "
                        f"WHERE id = %s",
                        (json.dumps({
                            'cf_type': cf.cf_type,
                            'trigger': cf.trigger,
                            'source_id': str(cf.source_id),
                            'direction': cf.direction,
                            'plausibility': cf.plausibility,
                            'specificity': cf.specificity,
                            'actionability': cf.actionability,
                            'confidence': cf.confidence,
                            'generation_method': cf.generation_method,
                        }), mem_id)
                    )
        except Exception:
            pass  # Metadata is supplementary

        # 3. Anti-rumination: lower Q-value for unvalidated LLM CFs (faster decay)
        if cf.generation_method == 'llm' and cf.confidence < 0.5:
            try:
                from q_value_engine import update_q
                update_q(mem_id, reward=-0.1)  # Penalize low-confidence LLM CFs
            except Exception:
                pass

        # 4. Add KG edge (counterfactual_of)
        try:
            from knowledge_graph import add_edge, RELATIONSHIP_TYPES
            # Only add edge if the relationship type exists
            if 'counterfactual_of' in RELATIONSHIP_TYPES:
                add_edge(
                    source_id=mem_id,
                    target_id=str(cf.source_id),
                    relationship='counterfactual_of',
                    confidence=cf.confidence,
                    evidence=f"N3 {cf.cf_type} counterfactual: {cf.lesson[:100]}",
                    auto_extracted=True,
                )
        except Exception:
            pass  # KG edge is supplementary

        return mem_id

    except Exception as e:
        return None


# ============================================================
# Cognitive State + Affect Routing
# ============================================================

def _route_to_cognitive_state(cf: Counterfactual):
    """Fire cognitive state + affect events for counterfactual generation."""
    try:
        from cognitive_state import process_event
        process_event('counterfactual_generated')
    except Exception:
        pass

    # N3/AF1+AF2: Route to affect system based on direction
    try:
        from affect_system import process_affect_event
        if cf.direction == 'upward':
            process_affect_event('counterfactual_upward', {'cf_type': cf.cf_type})
        else:
            process_affect_event('counterfactual_downward', {'cf_type': cf.cf_type})
    except Exception:
        pass


def route_to_affect(cf: Counterfactual):
    """Public wrapper for affect routing (called by stop.py Phase 2)."""
    _route_to_cognitive_state(cf)


def _update_stats(stored: list, elapsed_ms: float):
    """Update session CF stats in DB (called by stop.py Phase 2)."""
    try:
        db = _get_db()
        session_data = db.kv_get(KV_CF_SESSION) or {}
        existing_cfs = session_data.get('counterfactuals', [])
        existing_cfs.extend(stored)
        session_data['counterfactuals'] = existing_cfs
        session_data['phase2_count'] = len(stored)
        session_data['phase2_elapsed_ms'] = round(elapsed_ms, 1)
        session_data['generated'] = session_data.get('generated', 0) + len(stored)
        db.kv_set(KV_CF_SESSION, session_data)
    except Exception:
        pass


# ============================================================
# Decision Trace (Phase 1 lightweight)
# ============================================================

def log_decision_context(recall_ids: list[str], action: str, outcome: str = ''):
    """
    Log a recall-to-action association.
    Called from post_tool_use hook when tool results reference recalled memories.
    """
    db = _get_db()
    trace = db.kv_get(KV_DECISION_TRACE) or []
    trace.append({
        'recall_ids': recall_ids,
        'action': action[:200],
        'outcome': outcome[:200],
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })
    # Keep last 50 entries per session
    db.kv_set(KV_DECISION_TRACE, trace[-50:])


def get_decision_trace() -> list[dict]:
    """Get the current session's decision trace."""
    db = _get_db()
    return db.kv_get(KV_DECISION_TRACE) or []


# ============================================================
# History & Stats
# ============================================================

def get_history(limit: int = 30) -> list[dict]:
    """Get counterfactual generation history."""
    db = _get_db()
    return (db.kv_get(KV_CF_HISTORY) or [])[-limit:]


def get_session_counterfactuals() -> list[dict]:
    """Get counterfactuals generated this session."""
    db = _get_db()
    session = db.kv_get(KV_CF_SESSION) or {}
    return session.get('counterfactuals', [])


def get_stats() -> dict:
    """Get summary statistics across all sessions."""
    history = get_history()
    if not history:
        return {'sessions': 0, 'message': 'No counterfactual history yet'}

    total_gen = sum(h.get('generated', 0) for h in history)
    total_llm = sum(h.get('llm_calls', 0) for h in history)
    avg_gen = total_gen / len(history) if history else 0
    avg_ms = sum(h.get('elapsed_ms', 0) for h in history) / len(history) if history else 0

    return {
        'sessions': len(history),
        'total_generated': total_gen,
        'total_llm_calls': total_llm,
        'avg_per_session': round(avg_gen, 1),
        'avg_elapsed_ms': round(avg_ms, 0),
        'recent_5': history[-5:] if len(history) >= 5 else history,
    }


def health() -> dict:
    """Health check for toolkit integration."""
    history = get_history()
    return {
        'ok': True,
        'detail': f'{len(history)} sessions, {sum(h.get("generated", 0) for h in history)} CFs generated',
    }


# ============================================================
# Context Formatting (for workspace integration)
# ============================================================

def format_counterfactual_context(counterfactuals: list[dict]) -> str:
    """Format counterfactual insights for session start context injection."""
    if not counterfactuals:
        return ''
    lines = ['=== COUNTERFACTUAL INSIGHTS (from previous session) ===']
    for cf in counterfactuals[:3]:
        cf_type = cf.get('cf_type', '?')
        direction = cf.get('direction', '?')
        lesson = cf.get('lesson', 'no lesson')
        lines.append(f"  [{cf_type}/{direction}] {lesson}")
    lines.append('')
    return '\n'.join(lines)


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Counterfactual Reasoning Engine (N3)')
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('generate', help='Run session-end counterfactual review')
    h_parser = sub.add_parser('history', help='Show generation history')
    h_parser.add_argument('limit', nargs='?', type=int, default=10)
    sub.add_parser('stats', help='Summary statistics')
    sub.add_parser('quality', help='Show quality analysis for this session')
    sub.add_parser('health', help='Health check')
    sub.add_parser('trace', help='Show decision trace')

    args = parser.parse_args()

    if args.command == 'generate':
        result = session_end_review()
        print(f"Generated: {result.get('generated', 0)} counterfactuals")
        print(f"Candidates: {result.get('candidates_evaluated', 0)} evaluated, "
              f"{result.get('candidates_selected', 0)} selected")
        print(f"LLM calls: {result.get('llm_calls', 0)}")
        print(f"Elapsed: {result.get('elapsed_ms', 0)}ms")
        for cf in result.get('counterfactuals', []):
            print(f"\n  [{cf['cf_type']}] {cf['antecedent']}")
            print(f"    → {cf['consequent']}")
            print(f"    Lesson: {cf['lesson']}")
            q = cf['plausibility'] + cf['specificity'] + cf['actionability']
            print(f"    Quality: {q:.2f}/3.0 (P={cf['plausibility']}, S={cf['specificity']}, A={cf['actionability']})")

    elif args.command == 'history':
        history = get_history(limit=args.limit)
        if not history:
            print('No counterfactual history yet.')
        else:
            for h in history:
                date = h.get('session_date', '?')
                gen = h.get('generated', 0)
                llm = h.get('llm_calls', 0)
                ms = h.get('elapsed_ms', 0)
                print(f"  {date}: {gen} CFs ({llm} LLM calls, {ms}ms)")

    elif args.command == 'stats':
        stats = get_stats()
        if stats.get('message'):
            print(stats['message'])
        else:
            print(f"Sessions tracked: {stats['sessions']}")
            print(f"Total CFs: {stats['total_generated']}")
            print(f"Total LLM calls: {stats['total_llm_calls']}")
            print(f"Avg per session: {stats['avg_per_session']}")
            print(f"Avg time: {stats['avg_elapsed_ms']}ms")

    elif args.command == 'quality':
        cfs = get_session_counterfactuals()
        if not cfs:
            print('No counterfactuals this session.')
        else:
            for cf in cfs:
                q = cf['plausibility'] + cf['specificity'] + cf['actionability']
                gate = 'PASS' if q >= QUALITY_GATE_THRESHOLD else 'FAIL'
                print(f"  [{gate}] {cf['cf_type']}: Q={q:.2f}/3.0")
                print(f"    P={cf['plausibility']} S={cf['specificity']} A={cf['actionability']}")

    elif args.command == 'health':
        h = health()
        print(f"OK: {h['detail']}")

    elif args.command == 'trace':
        trace = get_decision_trace()
        if not trace:
            print('No decision trace entries this session.')
        else:
            for t in trace[-10:]:
                print(f"  [{t.get('timestamp', '?')[:19]}] "
                      f"Recalls: {len(t.get('recall_ids', []))} -> {t.get('action', '?')[:60]}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
