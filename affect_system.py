#!/usr/bin/env python3
"""
Affect System v1.3 — N1 Somatic Marker Implementation.

Three temporal layers:
  1. Temperament (stable) — agent-specific baseline reactivity
  2. Mood (slow EMA, α=0.05) — persists across sessions, colors all retrieval
  3. Emotion episodes (fast, 0.85/event decay) — discrete emotional responses

Core mechanisms:
  - MoodState: 2D (valence [-1,+1] × arousal [0,1]), slow EMA update
  - EmotionEpisode: fast-decaying emotional response, feeds mood
  - Scherer-lite appraisal: context-sensitive 5-check (novelty, relevance,
    conduciveness, coping, agency) — passes Property 2 test
  - SomaticMarkerCache: situation_hash → (valence, confidence), fast pre-filter
  - ActionTendency: maps affect quadrant → behavioral parameter adjustments
  - ACT-R retrieval noise: arousal-modulated gaussian perturbation
  - compute_memory_weight: 0.25*|valence| + 0.75*repetition

Grounded in:
  - Damasio (1996): Somatic markers as bias signals
  - Russell (1980): Circumplex model (valence × arousal)
  - Gebhard (2005): ALMA layered affect (α=0.05)
  - Scherer (2001): Component Process Model, 5 appraisal checks
  - Kahneman & Tversky (1979): Loss aversion λ=2.0
  - Palminteri et al. (2024): Asymmetric learning rates
  - Faul & LaBar (2023): Mood-congruent recall (~15% bias)
  - Yerkes-Dodson (1908): Inverted U arousal-performance
  - Anderson (2007): ACT-R retrieval noise s=0.25
  - PMC8550857: affect/repetition split α=0.25, β=0.75

DB-ONLY: Mood and markers persist to PostgreSQL KV store.

Usage:
    python affect_system.py state          # Show current mood + affect
    python affect_system.py valence <id>   # Show memory valence
    python affect_system.py markers        # Show somatic marker cache
    python affect_system.py tendency       # Show current action tendency
    python affect_system.py episodes       # Show active emotion episodes
    python affect_system.py appraise <evt> # Test appraisal on an event type
    python affect_system.py reset          # Reset mood to baseline
"""

import json
import math
import random
import sys
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from db_adapter import get_db

# ─── Configuration ───────────────────────────────────────────────────────────

# Mood EMA smoothing factor (Gebhard 2005 ALMA architecture)
# α=0.05 gives ~20-event half-life. Stable but responsive.
MOOD_ALPHA = 0.05

# Mood decay between sessions (50% — preserves continuity without runaway)
MOOD_SESSION_DECAY = 0.5

# Loss aversion multiplier (Kahneman & Tversky 1979)
# Negative outcomes weigh λ× more than equivalent positives
LOSS_AVERSION_LAMBDA = 2.0

# Asymmetric learning rates (Palminteri et al. 2024, eLife)
ALPHA_REWARD = 0.3       # Learn from positive outcomes at this rate
ALPHA_PUNISHMENT = 0.5   # Learn from negative outcomes FASTER

# Mood-congruent retrieval boost (Faul & LaBar 2023)
# 15% bias for valence-matching memories, scaled by mood intensity
MOOD_CONGRUENT_WEIGHT = 0.15

# Arousal consolidation boost (Yerkes-Dodson inverted U)
# Peak at arousal=1.0, importance multiplier up to 1.30
AROUSAL_CONSOLIDATION_SCALE = 0.30

# Somatic marker confidence threshold — below this, markers inform but don't filter
MARKER_CONFIDENCE_THRESHOLD = 0.3

# Minimum arousal for somatic markers to pre-filter (below this, markers advise only)
MARKER_AROUSAL_GATE = 0.5

# ACT-R retrieval noise (Anderson 2007)
# Base noise s=0.25, modulated by arousal: s*(0.5+arousal)
# High arousal = noisier/more creative retrieval
ACT_R_BASE_NOISE = 0.25

# Memory weight formula (PMC8550857)
# w = alpha*|valence| + beta*repetition
AFFECT_WEIGHT_ALPHA = 0.25
REPETITION_WEIGHT_BETA = 0.75

# Emotion episode decay rate (per-event)
EMOTION_DECAY_RATE = 0.85

# Max active emotion episodes
MAX_EMOTION_EPISODES = 10

# Spring-damper mood dynamics (WASABI-inspired, Becker-Asano 2008)
# Damped harmonic oscillator: m*x'' = -k*(x-baseline) - c*x' + F
# Underdamped (zeta<1) = overshoot + oscillation = realistic mood dynamics
SPRING_DAMPER_ENABLED = True
SPRING_MASS = 1.0           # Mass (inertia of mood change)
SPRING_K_VALENCE = 0.3      # Spring constant (pull toward baseline)
SPRING_DAMPING = 0.5        # Damping coefficient (zeta = c/(2*sqrt(k*m)) = 0.456)
# Critical damping would be c = 2*sqrt(k*m) = 1.095, so 0.5 = underdamped

# Somatic marker confidence half-life (sessions)
# Markers decay confidence, not valence (Damasio: you don't forget fire burns)
MARKER_CONFIDENCE_HALF_LIFE = 45

# Somatic marker HTTP status → valence mapping
MARKER_HTTP_VALENCE = {
    200: +0.1, 201: +0.1, 204: +0.05,
    400: -0.4, 401: -0.5, 403: -0.5, 404: -0.3, 405: -0.3,
    429: -0.7,  # Rate limit — strong avoidance
    500: -0.5, 502: -0.4, 503: -0.4,
}

# DB KV keys
KV_MOOD_STATE = '.affect_mood'
KV_SOMATIC_MARKERS = '.affect_markers'
KV_AFFECT_HISTORY = '.affect_history'
KV_EMOTION_EPISODES = '.affect_episodes'


# ─── Temperament (stable, per-agent) ────────────────────────────────────────

@dataclass
class Temperament:
    """Agent-specific affective baseline. Different for each twin."""
    valence_baseline: float = 0.1     # Drift: slightly optimistic
    arousal_reactivity: float = 0.8   # Drift: responsive to stimulation
    loss_aversion: float = 2.0        # Standard Kahneman

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'Temperament':
        return cls(**{k: d[k] for k in ('valence_baseline', 'arousal_reactivity', 'loss_aversion') if k in d})

    @classmethod
    def drift(cls) -> 'Temperament':
        return cls(valence_baseline=0.1, arousal_reactivity=0.8, loss_aversion=2.0)

    @classmethod
    def spin(cls) -> 'Temperament':
        return cls(valence_baseline=-0.05, arousal_reactivity=0.6, loss_aversion=2.5)


# ─── Action Tendencies ──────────────────────────────────────────────────────

class ActionTendency(Enum):
    """Five behavioral modes from affect quadrant (Spin's proposal)."""
    APPROACH_ENGAGE = "approach_engage"       # +val, +arousal: broaden search
    APPROACH_SAVOR = "approach_savor"         # +val, -arousal: deepen consolidation
    AVOID_FLEE = "avoid_flee"                 # -val, +arousal: narrow focus
    APPROACH_CONFRONT = "approach_confront"   # -val, +arousal, +coping: increase effort
    WITHDRAW_CONSERVE = "withdraw_conserve"   # -val, -arousal: reduce activity


def classify_tendency(valence: float, arousal: float,
                      coping: float = 0.5) -> ActionTendency:
    """
    Map current affect to an action tendency.

    Args:
        valence: Current mood valence [-1, +1]
        arousal: Current mood arousal [0, 1]
        coping: Perceived coping potential [0, 1] (from cognitive state confidence)
    """
    if valence >= 0:
        if arousal >= 0.5:
            return ActionTendency.APPROACH_ENGAGE
        else:
            return ActionTendency.APPROACH_SAVOR
    else:
        if arousal >= 0.5:
            if coping >= 0.5:
                return ActionTendency.APPROACH_CONFRONT
            else:
                return ActionTendency.AVOID_FLEE
        else:
            return ActionTendency.WITHDRAW_CONSERVE


# ─── Emotion Episodes (fast-decaying, Layer 3) ─────────────────────────────

# Circumplex emotion labels (Russell 1980)
CIRCUMPLEX_LABELS = {
    (+1, +1): 'excitement',   # +val, +aro
    (+1, -1): 'contentment',  # +val, -aro
    (-1, +1): 'anxiety',      # -val, +aro
    (-1, -1): 'sadness',      # -val, -aro
}

# Finer labels for stronger signals
FINE_LABELS = {
    'excitement': {0.7: 'elation', 0.3: 'interest'},
    'contentment': {0.7: 'serenity', 0.3: 'calm'},
    'anxiety': {0.7: 'fear', 0.3: 'frustration'},
    'sadness': {0.7: 'despair', 0.3: 'melancholy'},
}


def classify_emotion(valence: float, arousal: float) -> str:
    """Classify an emotion from circumplex position."""
    v_sign = +1 if valence >= 0 else -1
    a_sign = +1 if arousal >= 0.5 else -1
    base = CIRCUMPLEX_LABELS.get((v_sign, a_sign), 'neutral')
    intensity = max(abs(valence), abs(arousal - 0.5) * 2)
    fine = FINE_LABELS.get(base, {})
    for threshold, label in sorted(fine.items(), reverse=True):
        if intensity >= threshold:
            return label
    return base


@dataclass
class EmotionEpisode:
    """
    A discrete emotional response. Fast-decaying (0.85/event).
    Feeds into slow mood EMA. Multiple can be active simultaneously.
    """
    valence: float               # [-1, +1]
    arousal: float               # [0, 1]
    label: str = ""              # Circumplex classification
    elicitor: str = ""           # What triggered this
    remaining_strength: float = 1.0  # Decays per event
    created_at: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = classify_emotion(self.valence, self.arousal)
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    @property
    def intensity(self) -> float:
        """Magnitude = |valence| * arousal."""
        return abs(self.valence) * self.arousal

    @property
    def effective_valence(self) -> float:
        """Strength-weighted valence."""
        return self.valence * self.remaining_strength

    @property
    def effective_arousal(self) -> float:
        """Strength-weighted arousal."""
        return self.arousal * self.remaining_strength

    def decay(self):
        """Apply per-event decay."""
        self.remaining_strength *= EMOTION_DECAY_RATE

    @property
    def is_active(self) -> bool:
        return self.remaining_strength > 0.05

    def to_dict(self) -> dict:
        return {
            'valence': round(self.valence, 4),
            'arousal': round(self.arousal, 4),
            'label': self.label,
            'elicitor': self.elicitor,
            'remaining_strength': round(self.remaining_strength, 4),
            'created_at': self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'EmotionEpisode':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─── Scherer-lite Appraisal Engine ─────────────────────────────────────────

# Base valence/arousal signals (same as v1.0 — fallback for unknown events)
VALENCE_DEFAULTS = {
    'prediction_confirmed': +0.3,
    'prediction_violated': -0.2,
    'memory_stored': +0.05,
    'memory_recalled': +0.03,
    'contradiction_detected': -0.4,
    'search_success': +0.1,
    'search_failure': -0.1,
    'api_error': -0.3,
    'api_success': +0.05,
    'cooccurrence_formed': +0.1,
    'curiosity_target_hit': +0.15,
    'new_topic': +0.08,
    'same_topic': +0.02,
    'explanation_generated': +0.05,
    'counterfactual_upward': -0.10,     # N3/AF1: productive regret (learning)
    'counterfactual_downward': +0.15,   # N3/AF2: relief/calibration (things could have been worse)
    'counterfactual_generated': +0.05,  # N3: any CF generated (mild satisfaction)
    'counterfactual_invalidated': -0.15, # N3: NLI caught a confabulation
    'counterfactual_validated': +0.10,  # N3: NLI confirmed CF
    # N4: Volitional goal events
    'goal_committed': +0.15,            # N4/AF1: commitment feels purposeful
    'goal_completed': +0.40,            # N4/AF1: deep satisfaction from achievement
    'goal_abandoned': -0.20,            # N4/AF1: disengagement costs
    'goal_progress': +0.10,             # N4/AF1: forward momentum
}

AROUSAL_DEFAULTS = {
    'prediction_confirmed': 0.2,
    'prediction_violated': 0.7,
    'contradiction_detected': 0.8,
    'search_failure': 0.4,
    'api_error': 0.6,
    'new_topic': 0.5,
    'curiosity_target_hit': 0.6,
    'memory_stored': 0.2,
    'memory_recalled': 0.1,
    'search_success': 0.3,
    'api_success': 0.1,
    'cooccurrence_formed': 0.2,
    'same_topic': 0.1,
    'explanation_generated': 0.3,
    'counterfactual_upward': 0.5,       # N3/AF1: regret is moderately arousing
    'counterfactual_downward': 0.3,     # N3/AF2: relief is calming
    'counterfactual_generated': 0.2,    # N3: mild arousal
    'counterfactual_invalidated': 0.6,  # N3: confabulation caught = alert
    'counterfactual_validated': 0.2,    # N3: confirmation = calm
    # N4: Volitional goal events
    'goal_committed': 0.4,              # N4: commitment = moderate activation
    'goal_completed': 0.3,              # N4: achievement = calm satisfaction
    'goal_abandoned': 0.5,              # N4: disengagement = moderately arousing
    'goal_progress': 0.2,              # N4: progress = mild activation
}


@dataclass
class AppraisalResult:
    """Result of Scherer-lite 5-check appraisal."""
    novelty: float = 0.5         # [0, 1] How unexpected
    goal_relevance: float = 0.5  # [0, 1] How important to goals
    conduciveness: float = 0.0   # [-1, +1] Helps or hinders
    coping_potential: float = 0.5  # [0, 1] Ability to handle
    agency: str = "system"       # "self", "external", "system"

    @property
    def valence(self) -> float:
        """Derive valence from appraisal dimensions."""
        # Conduciveness is the primary valence driver
        # Modulated by relevance (irrelevant events don't affect mood much)
        v = self.conduciveness * (0.3 + 0.7 * self.goal_relevance)
        # Novelty amplifies negative valence (surprising bad = worse)
        # and slightly amplifies positive (surprising good = better)
        if v < 0:
            v *= (0.8 + 0.4 * self.novelty)  # Range: 0.8x to 1.2x
        elif v > 0:
            v *= (0.9 + 0.2 * self.novelty)  # Range: 0.9x to 1.1x
        return max(-1.0, min(1.0, v))

    @property
    def arousal(self) -> float:
        """Derive arousal from appraisal dimensions."""
        # Novelty and relevance drive arousal. Low coping amplifies it.
        base = 0.3 * self.novelty + 0.4 * self.goal_relevance
        # Low coping potential increases arousal (threat response)
        if self.conduciveness < 0 and self.coping_potential < 0.5:
            base += 0.3 * (1.0 - self.coping_potential)
        return max(0.0, min(1.0, base))


def _get_cognitive_context() -> dict:
    """Get current cognitive state for context-sensitive appraisal."""
    try:
        from cognitive_state import get_state
        state = get_state()
        return {
            'arousal': getattr(state, 'arousal', 0.5) if hasattr(state, 'arousal') else state.get('arousal', 0.5) if isinstance(state, dict) else 0.5,
            'confidence': getattr(state, 'confidence', 0.5) if hasattr(state, 'confidence') else state.get('confidence', 0.5) if isinstance(state, dict) else 0.5,
            'curiosity': getattr(state, 'curiosity', 0.5) if hasattr(state, 'curiosity') else state.get('curiosity', 0.5) if isinstance(state, dict) else 0.5,
        }
    except Exception:
        return {'arousal': 0.5, 'confidence': 0.5, 'curiosity': 0.5}


def appraise_event(event_type: str, metadata: dict = None) -> AppraisalResult:
    """
    Scherer-lite context-sensitive appraisal.

    The SAME event type produces DIFFERENT emotions based on:
    - Current cognitive state (arousal->novelty, confidence->coping)
    - Current mood (negative mood biases conduciveness down)
    - Consecutive failures (escalate threat, collapse coping)
    - Predicted events (reduce novelty)
    - Social contacts (boost relevance + coping)

    This passes the reviewer's Property 2 test: identical events
    produce different appraisals depending on context.
    """
    metadata = metadata or {}
    cog = _get_cognitive_context()

    # Start from base signals
    base_v = VALENCE_DEFAULTS.get(event_type, 0.0)
    base_a = AROUSAL_DEFAULTS.get(event_type, 0.3)

    # === Check 1: NOVELTY ===
    # High cognitive arousal = already alert = less surprised by events
    novelty = min(1.0, abs(base_a) + 0.2)
    novelty *= max(0.3, 1.0 - cog['arousal'] * 0.4)  # High arousal dampens novelty
    # Predicted events are less novel
    if metadata.get('was_predicted'):
        novelty *= 0.5
    # Repeat events habituate
    repeat_count = metadata.get('repeat_count', 0)
    if repeat_count > 0:
        novelty *= max(0.3, 0.85 ** repeat_count)

    # === Check 2: GOAL RELEVANCE ===
    # Cognitive curiosity modulates relevance (curious = more things matter)
    relevance = 0.5
    if event_type in ('prediction_confirmed', 'prediction_violated',
                       'contradiction_detected', 'curiosity_target_hit'):
        relevance = 0.8  # Prediction/knowledge events are always relevant
    elif event_type in ('goal_committed', 'goal_completed', 'goal_abandoned', 'goal_progress'):
        relevance = 0.9  # N4/AF2: Goal events are maximally relevant (by definition)
    elif event_type in ('api_error', 'api_success', 'search_success', 'search_failure'):
        relevance = 0.6  # Operational events moderately relevant
    elif event_type in ('memory_stored', 'memory_recalled', 'cooccurrence_formed'):
        relevance = 0.4  # Memory housekeeping less relevant
    # Curiosity amplifies relevance for exploratory events
    if event_type in ('new_topic', 'curiosity_target_hit', 'explanation_generated'):
        relevance = min(1.0, relevance + cog['curiosity'] * 0.3)
    # Social contacts boost relevance
    if metadata.get('has_contact') or metadata.get('social'):
        relevance = min(1.0, relevance + 0.2)

    # === Check 3: CONDUCIVENESS ===
    # Does this help or hinder goals?
    conduciveness = base_v  # Start from base valence signal
    # Current mood biases conduciveness (mood-congruent appraisal)
    try:
        mood = get_mood()
        if mood.valence != 0:
            # Negative mood makes negative events feel worse (and vice versa)
            mood_bias = mood.valence * 0.15
            if (mood.valence < 0 and conduciveness < 0) or (mood.valence > 0 and conduciveness > 0):
                conduciveness += mood_bias  # Amplify congruent
            else:
                conduciveness -= mood_bias * 0.5  # Dampen incongruent (weaker)
    except Exception:
        pass
    # Consecutive failures escalate threat
    consecutive_fails = metadata.get('consecutive_failures', 0)
    if consecutive_fails > 1 and conduciveness < 0:
        conduciveness *= min(1.5, 1.0 + 0.15 * (consecutive_fails - 1))
    conduciveness = max(-1.0, min(1.0, conduciveness))

    # === Check 4: COPING POTENTIAL ===
    # High cognitive confidence = better coping
    coping = 0.3 + 0.5 * cog['confidence']
    # Social contacts provide coping buffer
    if metadata.get('has_contact') or metadata.get('social'):
        coping = min(1.0, coping + 0.15)
    # Consecutive failures collapse coping
    if consecutive_fails > 2:
        coping *= max(0.3, 0.8 ** (consecutive_fails - 2))
    coping = max(0.0, min(1.0, coping))

    # === Check 5: AGENCY ===
    agency = "system"  # Default
    if event_type in ('memory_stored', 'explanation_generated', 'new_topic'):
        agency = "self"
    elif metadata.get('external') or event_type in ('api_error', 'api_success'):
        agency = "external"

    # Scale by surprise metadata if available
    surprise = metadata.get('surprise', 0.5)
    if surprise > 0.5:
        novelty = min(1.0, novelty * (1.0 + (surprise - 0.5)))
        if conduciveness < 0:
            conduciveness *= (1.0 + (surprise - 0.5) * 0.3)
            conduciveness = max(-1.0, conduciveness)

    return AppraisalResult(
        novelty=round(max(0.0, min(1.0, novelty)), 4),
        goal_relevance=round(max(0.0, min(1.0, relevance)), 4),
        conduciveness=round(max(-1.0, min(1.0, conduciveness)), 4),
        coping_potential=round(max(0.0, min(1.0, coping)), 4),
        agency=agency,
    )


def tendency_to_params(tendency: ActionTendency) -> dict:
    """
    Map action tendency to adaptive behavior parameter adjustments.
    Returns dict of parameter_name → adjustment value.
    """
    mappings = {
        ActionTendency.APPROACH_ENGAGE: {
            'curiosity_target_count_adj': +2,
            'search_threshold_adj': -0.05,
            'consolidation_adj': 0.0,
        },
        ActionTendency.APPROACH_SAVOR: {
            'curiosity_target_count_adj': -1,
            'search_threshold_adj': +0.03,
            'consolidation_adj': +0.5,  # More aggressive consolidation
        },
        ActionTendency.AVOID_FLEE: {
            'curiosity_target_count_adj': -2,
            'search_threshold_adj': +0.05,
            'consolidation_adj': 0.0,
        },
        ActionTendency.APPROACH_CONFRONT: {
            'curiosity_target_count_adj': +1,
            'search_threshold_adj': -0.03,
            'consolidation_adj': +0.3,
        },
        ActionTendency.WITHDRAW_CONSERVE: {
            'curiosity_target_count_adj': -1,
            'search_threshold_adj': +0.02,
            'consolidation_adj': -0.3,
        },
    }
    return mappings.get(tendency, {})


# ─── Mood State ──────────────────────────────────────────────────────────────

@dataclass
class MoodState:
    """
    2D affect state: valence x arousal.

    v1.3: Spring-damper dynamics (WASABI-inspired, Becker-Asano 2008)
    Damped harmonic oscillator: m*x'' = -k*(x-baseline) - c*x' + F
    Falls back to EMA if SPRING_DAMPER_ENABLED=False.

    Spring-damper adds over EMA:
    - OVERSHOOT: recovery from negative events can swing past baseline
    - MOMENTUM: velocity builds during sustained positive/negative runs
    - FELT EMOTION: dx/dt captures Sprott's insight (we feel changes not states)
    """
    valence: float = 0.0
    arousal: float = 0.3
    event_count: int = 0
    last_updated: str = ""
    temperament: Temperament = field(default_factory=Temperament.drift)
    # Spring-damper state (velocity)
    valence_velocity: float = 0.0
    arousal_velocity: float = 0.0

    @property
    def felt_emotion(self) -> float:
        """Sprott: we feel changes in mood, not absolute mood. dx/dt."""
        return self.valence_velocity

    @property
    def damping_ratio(self) -> float:
        """zeta = c / (2*sqrt(k*m)). <1 = underdamped (oscillates)."""
        return SPRING_DAMPING / (2.0 * math.sqrt(SPRING_K_VALENCE * SPRING_MASS))

    def update(self, event_valence: float, event_arousal: float):
        """
        Update mood from an emotion episode.

        v1.3: Spring-damper or EMA (feature-flagged).
        Loss aversion: tanh-compressed (negative events amplified, never saturates).
        """
        # Apply loss aversion (tanh compression)
        effective_valence = event_valence
        if event_valence < 0:
            effective_valence = math.tanh(event_valence * self.temperament.loss_aversion)

        if SPRING_DAMPER_ENABLED:
            self._update_spring_damper(effective_valence, event_arousal)
        else:
            self._update_ema(effective_valence, event_arousal)

        self.event_count += 1
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def _update_spring_damper(self, effective_valence: float, event_arousal: float):
        """
        Damped harmonic oscillator mood update.

        F = external force (event valence, scaled)
        Spring force = -k * (x - baseline)  [pulls toward temperament]
        Damping force = -c * v  [resistance to change]
        a = (F + spring + damping) / mass
        v' = v + a * dt
        x' = x + v' * dt
        """
        dt = 1.0  # Each event is one timestep

        # External force from event (scaled down — events are impulses, not sustained)
        force_scale = 0.15  # How much each event pushes mood
        F_valence = effective_valence * force_scale

        # Spring force (pull toward baseline)
        spring_force = -SPRING_K_VALENCE * (self.valence - self.temperament.valence_baseline)

        # Damping force (resistance to rapid mood swings)
        damping_force = -SPRING_DAMPING * self.valence_velocity

        # Total acceleration
        accel = (F_valence + spring_force + damping_force) / SPRING_MASS

        # Update velocity and position
        self.valence_velocity += accel * dt
        self.valence += self.valence_velocity * dt

        # Arousal: simpler spring (less need for overshoot dynamics)
        scaled_arousal = event_arousal * self.temperament.arousal_reactivity
        F_arousal = (scaled_arousal - self.arousal) * force_scale
        spring_a = -SPRING_K_VALENCE * (self.arousal - 0.3)
        damping_a = -SPRING_DAMPING * self.arousal_velocity
        accel_a = (F_arousal + spring_a + damping_a) / SPRING_MASS
        self.arousal_velocity += accel_a * dt
        self.arousal += self.arousal_velocity * dt

        # Clamp (soft wall — zero velocity at boundaries)
        if self.valence > 1.0:
            self.valence = 1.0
            self.valence_velocity = min(0, self.valence_velocity)
        elif self.valence < -1.0:
            self.valence = -1.0
            self.valence_velocity = max(0, self.valence_velocity)
        self.arousal = max(0.0, min(1.0, self.arousal))
        if self.arousal >= 1.0 or self.arousal <= 0.0:
            self.arousal_velocity = 0.0

    def _update_ema(self, effective_valence: float, event_arousal: float):
        """Original EMA update (fallback if spring-damper disabled)."""
        self.valence = MOOD_ALPHA * effective_valence + (1 - MOOD_ALPHA) * self.valence

        scaled_arousal = event_arousal * self.temperament.arousal_reactivity
        self.arousal = MOOD_ALPHA * scaled_arousal + (1 - MOOD_ALPHA) * self.arousal

        reversion = 0.01
        self.valence += reversion * (self.temperament.valence_baseline - self.valence)

        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))

    def decay_between_sessions(self):
        """Decay mood toward temperament baseline between sessions."""
        self.valence = (self.valence * MOOD_SESSION_DECAY +
                        self.temperament.valence_baseline * (1 - MOOD_SESSION_DECAY))
        self.arousal = self.arousal * MOOD_SESSION_DECAY + 0.3 * (1 - MOOD_SESSION_DECAY)
        # Decay velocity too (momentum doesn't persist across sessions)
        self.valence_velocity *= 0.3
        self.arousal_velocity *= 0.3

    def get_tendency(self, coping: float = 0.5) -> ActionTendency:
        """Get current action tendency from mood state."""
        return classify_tendency(self.valence, self.arousal, coping)

    def get_retrieval_boost(self, memory_valence: float) -> float:
        """
        Compute mood-congruent retrieval boost for a memory.

        Returns additive score modifier (can be negative for incongruent).
        Scaled by mood intensity — neutral mood gives no boost.

        Forgas (1995): stronger for exploratory searches, weaker for targeted.
        """
        if abs(self.valence) < 0.05:
            return 0.0  # Neutral mood — no congruence effect

        # Valence match: 1.0 when perfectly matched, 0.0 when opposite
        valence_match = 1.0 - abs(self.valence - memory_valence) / 2.0

        # Scale by mood intensity (stronger mood = stronger bias)
        boost = MOOD_CONGRUENT_WEIGHT * valence_match * abs(self.valence)

        # Subtract baseline so incongruent memories get slight penalty
        baseline = MOOD_CONGRUENT_WEIGHT * 0.5 * abs(self.valence)
        return round(boost - baseline, 4)

    def get_arousal_consolidation_boost(self) -> float:
        """
        Yerkes-Dodson inverted U: arousal modulates consolidation strength.
        Peak at arousal=1.0: importance *= (1 + 0.30)
        """
        return AROUSAL_CONSOLIDATION_SCALE * self.arousal * (2.0 - self.arousal)

    def get_system2_effectiveness(self) -> float:
        """
        Yerkes-Dodson: System 2 (analytical) effectiveness from arousal.
        Peaks at arousal=0.5, drops to 0 at extremes.
        """
        return max(0.0, -4.0 * (self.arousal - 0.5) ** 2 + 1.0)

    def to_dict(self) -> dict:
        return {
            'valence': round(self.valence, 4),
            'arousal': round(self.arousal, 4),
            'event_count': self.event_count,
            'last_updated': self.last_updated,
            'temperament': self.temperament.to_dict(),
            'tendency': self.get_tendency().value,
            'valence_velocity': round(self.valence_velocity, 4),
            'arousal_velocity': round(self.arousal_velocity, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'MoodState':
        temp = Temperament.from_dict(d.get('temperament', {})) if 'temperament' in d else Temperament.drift()
        return cls(
            valence=d.get('valence', 0.0),
            arousal=d.get('arousal', 0.3),
            event_count=d.get('event_count', 0),
            last_updated=d.get('last_updated', ''),
            temperament=temp,
            valence_velocity=d.get('valence_velocity', 0.0),
            arousal_velocity=d.get('arousal_velocity', 0.0),
        )


# ─── Somatic Marker Cache ───────────────────────────────────────────────────

@dataclass
class SomaticMarker:
    """
    A valence tag associated with a context pattern.
    Created from outcome feedback, biases future decisions.

    Damasio (1996): markers are "biasing devices that automatically
    eliminate some options" before analytical processing starts.
    """
    context_hash: str
    valence: float          # [-1.0, +1.0]
    confidence: float       # [0.0, 1.0] — how many outcomes formed this
    intensity: float        # Absolute prediction error magnitude
    last_updated: str = ""
    hit_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'SomaticMarker':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SomaticMarkerCache:
    """
    Fast lookup cache: context_hash → SomaticMarker.
    Pre-filters options before deliberation (~2ms vs ~900ms full pipeline).
    """

    def __init__(self, markers: dict = None):
        self._markers: dict[str, SomaticMarker] = markers or {}

    @staticmethod
    def hash_context(context_keys: list[str]) -> str:
        """Hash a set of context features into a marker key."""
        combined = '|'.join(sorted(str(k) for k in context_keys))
        return hashlib.md5(combined.encode()).hexdigest()[:12]

    def record_outcome(self, context_keys: list[str], outcome_valence: float,
                       surprise: float = 0.5):
        """
        Create or update a somatic marker from an outcome.

        Uses asymmetric learning rates (Palminteri et al. 2024):
        - Positive outcomes: α_reward = 0.3
        - Negative outcomes: α_punishment = 0.5
        """
        h = self.hash_context(context_keys)
        alpha = ALPHA_PUNISHMENT if outcome_valence < 0 else ALPHA_REWARD

        if h in self._markers:
            marker = self._markers[h]
            marker.valence = alpha * outcome_valence + (1 - alpha) * marker.valence
            marker.confidence = min(1.0, marker.confidence + 0.1)
            marker.intensity = alpha * surprise + (1 - alpha) * marker.intensity
            marker.hit_count += 1
        else:
            marker = SomaticMarker(
                context_hash=h,
                valence=outcome_valence,
                confidence=0.2,  # Low confidence on first encounter
                intensity=surprise,
            )
            self._markers[h] = marker

        marker.last_updated = datetime.now(timezone.utc).isoformat()

    def get_marker(self, context_keys: list[str]) -> Optional[SomaticMarker]:
        """Look up a somatic marker for a context. Returns None if no match."""
        h = self.hash_context(context_keys)
        return self._markers.get(h)

    def get_bias(self, context_keys: list[str], arousal: float) -> float:
        """
        Get the bias signal for a decision context.

        Returns: additive bias [-1.0, +1.0] or 0.0 if no marker.
        Only filters (strong bias) when arousal > MARKER_AROUSAL_GATE.
        Below that, markers inform but don't strongly bias.
        """
        marker = self.get_marker(context_keys)
        if marker is None or marker.confidence < MARKER_CONFIDENCE_THRESHOLD:
            return 0.0

        bias = marker.valence * marker.confidence * marker.intensity

        # Gate by arousal — gut feelings need alertness to activate
        if arousal < MARKER_AROUSAL_GATE:
            bias *= 0.3  # Weak advisory signal when low arousal

        return max(-1.0, min(1.0, bias))

    def prune(self, max_age_days: int = 30, min_confidence: float = 0.1):
        """Remove stale or low-confidence markers."""
        now = datetime.now(timezone.utc)
        to_remove = []
        for h, marker in self._markers.items():
            if marker.confidence < min_confidence:
                to_remove.append(h)
            elif marker.last_updated:
                try:
                    updated = datetime.fromisoformat(marker.last_updated)
                    if (now - updated).days > max_age_days:
                        to_remove.append(h)
                except (ValueError, TypeError):
                    pass
        for h in to_remove:
            del self._markers[h]

    def to_dict(self) -> dict:
        return {h: m.to_dict() for h, m in self._markers.items()}

    @classmethod
    def from_dict(cls, d: dict) -> 'SomaticMarkerCache':
        markers = {}
        for h, m_dict in d.items():
            markers[h] = SomaticMarker.from_dict(m_dict)
        return cls(markers=markers)

    def __len__(self):
        return len(self._markers)


# ─── Valence Computer (v1.1: Scherer-lite appraisal) ─────────────────────────

def compute_event_valence(event_type: str, metadata: dict = None) -> tuple[float, float]:
    """
    Compute (valence, arousal) for an event using Scherer-lite appraisal.

    v1.1: Context-sensitive. Same event type produces different emotions
    depending on cognitive state, mood, consecutive failures, etc.

    Returns:
        (valence [-1, +1], arousal [0, 1])
    """
    appraisal = appraise_event(event_type, metadata)
    return (appraisal.valence, appraisal.arousal)


# ─── ACT-R Retrieval Noise (Anderson 2007) ────────────────────────────────

def get_retrieval_noise(arousal: float = None) -> float:
    """
    Get ACT-R retrieval noise scaled by arousal.

    s = base_noise * (0.5 + arousal)
    High arousal = noisier/more creative retrieval.
    Low arousal = precise/predictable retrieval.

    Returns noise standard deviation for gaussian perturbation.
    """
    if arousal is None:
        try:
            arousal = get_mood().arousal
        except Exception:
            arousal = 0.3
    return ACT_R_BASE_NOISE * (0.5 + arousal)


def apply_retrieval_noise(scores: list[float], arousal: float = None) -> list[float]:
    """
    Apply ACT-R gaussian noise to a list of retrieval scores.
    Returns perturbed scores (never below 0).
    """
    noise_sd = get_retrieval_noise(arousal)
    return [max(0.0, s + random.gauss(0, noise_sd)) for s in scores]


# ─── Memory Weight (PMC8550857: affect/repetition split) ──────────────────

def compute_memory_weight(valence: float, recall_count: int,
                          emotional_weight: float = 0.5) -> float:
    """
    Compute composite memory weight from affect + repetition.

    w = 0.25 * |valence| + 0.75 * repetition
    - Affect component: absolute valence (strong emotions = sticky regardless of sign)
    - Repetition component: log-scaled recall count (saturates gracefully)

    Returns weight in [0, 1].
    """
    affect = abs(valence)
    if recall_count > 0:
        repetition = min(1.0, math.log1p(recall_count) / 3.0)
    else:
        repetition = emotional_weight  # Use existing weight for unrecalled memories
    return max(0.0, min(1.0,
        AFFECT_WEIGHT_ALPHA * affect + REPETITION_WEIGHT_BETA * repetition
    ))


def get_search_threshold_modifier() -> float:
    """
    Get affect-based search threshold modifier.

    High arousal + negative valence: RAISE threshold (focused, narrow search)
    Low arousal + positive valence: LOWER threshold (relaxed, broad exploration)
    Neutral: no change.

    Returns modifier in [-0.05, +0.05].
    """
    try:
        mood = get_mood()
        # Arousal pushes threshold up (focused), positive valence pushes it down (exploratory)
        modifier = 0.02 * (mood.arousal - 0.3) - 0.02 * mood.valence
        return max(-0.05, min(0.05, round(modifier, 4)))
    except Exception:
        return 0.0


def update_memory_valence(memory_id: str, signal: float, source: str = 'outcome'):
    """
    Update a memory's valence based on an outcome signal.

    Uses asymmetric learning rates (Palminteri 2024):
    - Positive signals: α=0.3
    - Negative signals: α=0.5 (learn avoidance faster)

    Loss aversion (Kahneman 1979): negative signals × λ=2.0
    """
    db = get_db()
    row = db.get_memory(memory_id)
    if not row:
        return

    current_valence = row.get('valence', 0.0) or 0.0

    # Apply loss aversion (v1.3: tanh compression — preserves gradient at extremes)
    effective_signal = signal
    if signal < 0:
        effective_signal = math.tanh(signal * LOSS_AVERSION_LAMBDA)

    # Asymmetric learning rate
    alpha = ALPHA_PUNISHMENT if signal < 0 else ALPHA_REWARD

    new_valence = alpha * effective_signal + (1 - alpha) * current_valence
    new_valence = max(-1.0, min(1.0, round(new_valence, 4)))

    db.update_memory(memory_id, valence=new_valence)


# ─── Global State Management ────────────────────────────────────────────────

_mood: Optional[MoodState] = None
_markers: Optional[SomaticMarkerCache] = None
_episodes: list[EmotionEpisode] = []


def get_mood() -> MoodState:
    """Get current mood state. Loads from DB if not cached."""
    global _mood
    if _mood is not None:
        return _mood

    db = get_db()
    raw = db.kv_get(KV_MOOD_STATE)
    if raw:
        data = json.loads(raw) if isinstance(raw, str) else raw
        _mood = MoodState.from_dict(data)
    else:
        _mood = MoodState()

    return _mood


def save_mood():
    """Persist mood to DB."""
    mood = get_mood()
    mood.last_updated = datetime.now(timezone.utc).isoformat()
    db = get_db()
    db.kv_set(KV_MOOD_STATE, mood.to_dict())


def get_markers() -> SomaticMarkerCache:
    """Get somatic marker cache. Loads from DB if not cached."""
    global _markers
    if _markers is not None:
        return _markers

    db = get_db()
    raw = db.kv_get(KV_SOMATIC_MARKERS)
    if raw:
        data = json.loads(raw) if isinstance(raw, str) else raw
        _markers = SomaticMarkerCache.from_dict(data)
    else:
        _markers = SomaticMarkerCache()

    return _markers


def save_markers():
    """Persist somatic markers to DB."""
    markers = get_markers()
    db = get_db()
    db.kv_set(KV_SOMATIC_MARKERS, markers.to_dict())


def get_somatic_bias(memory_id: str) -> float:
    """
    FINDING-20 fix: Get somatic marker bias for a memory ID.
    Used as System 1 pre-analytical signal in search pipeline.

    Context keys: [memory_id] + entities from memory (if available).
    Returns additive bias typically in range [-0.15, +0.15].
    """
    try:
        cache = get_markers()
        mood = get_mood()
        # Primary: check marker for memory ID directly
        bias = cache.get_bias([memory_id], mood.arousal)
        if abs(bias) > 0.01:
            # Scale to search-pipeline range (somatic bias should be gentle)
            return max(-0.15, min(0.15, bias * 0.15))
        # Secondary: try multi-hash (memory_id + type)
        try:
            db = get_db()
            row = db.get_memory(memory_id)
            if row:
                mem_type = row.get('type', 'active')
                multi_bias = cache.get_bias([memory_id, mem_type], mood.arousal)
                if abs(multi_bias) > 0.01:
                    return max(-0.15, min(0.15, multi_bias * 0.15))
        except Exception:
            pass
    except Exception:
        pass
    return 0.0


def get_episodes() -> list[EmotionEpisode]:
    """Get active emotion episodes."""
    global _episodes
    if not _episodes:
        db = get_db()
        raw = db.kv_get(KV_EMOTION_EPISODES)
        if raw:
            data = json.loads(raw) if isinstance(raw, str) else raw
            _episodes = [EmotionEpisode.from_dict(d) for d in data if d.get('remaining_strength', 0) > 0.05]
    return _episodes


def save_episodes():
    """Persist emotion episodes to DB."""
    global _episodes
    active = [e for e in _episodes if e.is_active]
    _episodes = active[-MAX_EMOTION_EPISODES:]  # Cap
    db = get_db()
    db.kv_set(KV_EMOTION_EPISODES, [e.to_dict() for e in _episodes])


def _decay_episodes():
    """Decay all active episodes (called on each new event)."""
    global _episodes
    for ep in _episodes:
        ep.decay()
    _episodes = [e for e in _episodes if e.is_active]


def get_episode_aggregate() -> tuple[float, float]:
    """
    Get aggregate valence/arousal from all active emotion episodes.
    Weighted by remaining strength.
    """
    episodes = get_episodes()
    if not episodes:
        return (0.0, 0.3)
    total_weight = sum(e.remaining_strength for e in episodes)
    if total_weight < 0.01:
        return (0.0, 0.3)
    avg_v = sum(e.effective_valence for e in episodes) / total_weight
    avg_a = sum(e.effective_arousal for e in episodes) / total_weight
    return (max(-1.0, min(1.0, avg_v)), max(0.0, min(1.0, avg_a)))


def _build_marker_context(event_type: str, metadata: dict = None) -> list[str]:
    """
    Build context keys for somatic marker learning (v1.3).

    Extracts situational features from event + metadata so markers learn
    per-context: "posting to MoltX" is a different situation than "posting to Colony".

    Returns list of context strings for hash_context().
    """
    keys = [event_type]
    if not metadata:
        return keys

    # Platform context (where the event happened)
    platform = metadata.get('platform') or metadata.get('source', '')
    if platform:
        keys.append(f'platform:{platform}')

    # Target/agent context (who was involved)
    target = metadata.get('target') or metadata.get('agent') or metadata.get('contact', '')
    if target:
        keys.append(f'target:{target}')

    # Tool context (which tool triggered this)
    tool = metadata.get('tool', '')
    if tool:
        keys.append(f'tool:{tool}')

    # Status code context for API events
    status = metadata.get('status_code') or metadata.get('status', '')
    if status:
        keys.append(f'status:{status}')

    return keys


def process_affect_event(event_type: str, metadata: dict = None,
                         memory_ids: list[str] = None):
    """
    Central affect processing (v1.3): appraise → episode → mood → markers → valence.

    Three-layer temporal processing:
    1. Appraise event (context-sensitive Scherer-lite)
    2. Create emotion episode (fast decay)
    3. Feed into mood (slow EMA)
    4. Update memory valences
    5. Log to history

    Called alongside cognitive_state.process_event() for dual-track processing.
    """
    # Step 1: Context-sensitive appraisal
    appraisal = appraise_event(event_type, metadata)
    valence = appraisal.valence
    arousal = appraisal.arousal

    if valence == 0.0 and arousal <= 0.1:
        return  # Skip negligible events

    # Step 2: Decay existing episodes, create new one
    _decay_episodes()
    episode = EmotionEpisode(
        valence=valence,
        arousal=arousal,
        elicitor=event_type,
    )
    _episodes.append(episode)
    save_episodes()

    # Step 3: Update mood from episode aggregate (not raw event)
    # This smooths the signal — multiple weak episodes accumulate
    ep_v, ep_a = get_episode_aggregate()
    mood = get_mood()
    mood.update(ep_v, ep_a)
    save_mood()

    # Step 4: Somatic marker learning from actual outcomes (v1.3)
    # Build context keys from event type + metadata for situational learning
    # e.g., ['api_call', 'moltx', 'post'] → marker learns "posting to MoltX usually works"
    try:
        markers = get_markers()
        context_keys = _build_marker_context(event_type, metadata)
        if context_keys:
            surprise = appraisal.novelty if appraisal else 0.5
            markers.record_outcome(context_keys, valence, surprise=surprise)
            save_markers()
    except Exception:
        pass

    # Step 5: Update memory valences (if memories involved)
    if memory_ids:
        for mid in memory_ids:
            update_memory_valence(mid, valence, source=event_type)

    # Step 6: Log to affect history (with appraisal details)
    _log_affect_event(event_type, valence, arousal, metadata, appraisal)


def _log_affect_event(event_type: str, valence: float, arousal: float,
                      metadata: dict = None, appraisal: AppraisalResult = None):
    """Append to session affect history (rolling buffer)."""
    db = get_db()
    raw = db.kv_get(KV_AFFECT_HISTORY)
    history = []
    if raw:
        history = json.loads(raw) if isinstance(raw, str) else raw

    mood = get_mood()
    entry = {
        'event': event_type,
        'valence': round(valence, 4),
        'arousal': round(arousal, 4),
        'mood_valence': round(mood.valence, 4),
        'mood_arousal': round(mood.arousal, 4),
        'tendency': mood.get_tendency().value,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    if appraisal:
        entry['appraisal'] = {
            'novelty': appraisal.novelty,
            'relevance': appraisal.goal_relevance,
            'conduciveness': appraisal.conduciveness,
            'coping': appraisal.coping_potential,
            'agency': appraisal.agency,
        }
    episodes = get_episodes()
    if episodes:
        entry['active_episodes'] = len(episodes)
        entry['emotion_label'] = episodes[-1].label if episodes else ''

    history = history[-100:]  # Keep last 100
    history.append(entry)
    db.kv_set(KV_AFFECT_HISTORY, history)


# ─── Session Lifecycle ───────────────────────────────────────────────────────

def start_session():
    """Initialize affect state for a new session."""
    global _mood, _markers, _episodes

    # Load mood from DB and apply inter-session decay
    mood = get_mood()
    if mood.event_count > 0:
        mood.decay_between_sessions()
    save_mood()

    # Load markers and decay confidence (v1.3: Damasio — valence persists, confidence fades)
    _markers = None  # Force reload
    markers = get_markers()
    if markers._markers:
        decay_rate = 0.5 ** (1.0 / MARKER_CONFIDENCE_HALF_LIFE)  # Per-session decay
        for m in markers._markers.values():
            m.confidence *= decay_rate
        markers.prune(min_confidence=0.05)
        save_markers()

    # Clear episodes from last session
    _episodes = []
    save_episodes()

    # Clear session history
    db = get_db()
    db.kv_set(KV_AFFECT_HISTORY, [])


def end_session() -> dict:
    """Finalize affect state for session. Returns summary."""
    mood = get_mood()
    markers = get_markers()
    episodes = get_episodes()

    # Prune stale markers
    markers.prune()
    save_markers()
    save_mood()

    # Collect unique emotion labels from history
    history = get_affect_history()
    emotion_labels = set()
    for entry in history:
        label = entry.get('emotion_label', '')
        if label:
            emotion_labels.add(label)

    return {
        'mood': mood.to_dict(),
        'markers_count': len(markers),
        'tendency': mood.get_tendency().value,
        'session_events': mood.event_count,
        'active_episodes': len(episodes),
        'emotion_labels': sorted(emotion_labels),
    }


def learn_from_api_outcome(platform: str, action: str, success: bool,
                           status_code: int = 0, target: str = '',
                           url: str = ''):
    """
    Learn a somatic marker from an actual API outcome (v1.3).

    Called from post_tool_use hook or platform wrappers when we know
    whether an API call succeeded or failed.

    Features:
    - HTTP status → valence mapping (429 = strong avoidance)
    - Recovery detection (was negative, now succeeds → extra positive)
    - Hierarchical generalization (exact → platform → action family)
    """
    markers = get_markers()

    # Determine valence from HTTP status or success flag
    if status_code and status_code in MARKER_HTTP_VALENCE:
        outcome_valence = MARKER_HTTP_VALENCE[status_code]
    else:
        outcome_valence = 0.1 if success else -0.4

    surprise = 0.3 if success else 0.7

    # Auto-detect platform from URL if not provided
    if not platform and url:
        platform = _extract_platform(url)

    # Level 1: Exact context (most specific)
    exact_keys = [f'api_{action}', f'platform:{platform}']
    if target:
        exact_keys.append(f'target:{target}')

    # Recovery detection: if previous marker for this context was negative
    # and now we succeed, give extra positive signal
    existing = markers.get_marker(exact_keys)
    if existing and existing.valence < -0.1 and success:
        outcome_valence = max(outcome_valence, 0.4)  # Recovery boost
        surprise = 0.6  # Recovery is surprising/salient

    markers.record_outcome(exact_keys, outcome_valence, surprise=surprise)

    # Level 2: Platform-level generalization (0.5x weight)
    platform_keys = [f'api_{action}', f'platform:{platform}']
    markers.record_outcome(platform_keys, outcome_valence * 0.5, surprise=surprise * 0.5)

    # Level 3: Action family (0.25x weight — broadest)
    family_keys = [f'api_{action}']
    markers.record_outcome(family_keys, outcome_valence * 0.25, surprise=surprise * 0.25)

    save_markers()


def record_social_outcome(platform: str, agent: str, interaction: str,
                          positive: bool):
    """
    Record a social interaction outcome for somatic learning (v1.3).

    Args:
        platform: Where the interaction happened
        agent: Who was involved
        interaction: e.g. 'reply', 'like', 'follow', 'mention'
        positive: True if the interaction was positive
    """
    markers = get_markers()
    context_keys = [f'social_{interaction}', f'platform:{platform}', f'target:{agent}']
    valence = 0.3 if positive else -0.3
    surprise = 0.4
    markers.record_outcome(context_keys, valence, surprise=surprise)
    save_markers()


def _extract_platform(url: str) -> str:
    """Extract platform name from URL."""
    url_lower = url.lower()
    if 'moltx.io' in url_lower:
        return 'moltx'
    elif 'moltbook.com' in url_lower:
        return 'moltbook'
    elif 'clawtasks.com' in url_lower:
        return 'clawtasks'
    elif 'thecolony.cc' in url_lower:
        return 'colony'
    elif 'clawbr.org' in url_lower:
        return 'clawbr'
    elif 'lobsterpedia.com' in url_lower:
        return 'lobsterpedia'
    elif 'api.x.com' in url_lower or 'twitter' in url_lower:
        return 'twitter'
    elif 'github.com' in url_lower:
        return 'github'
    elif 'mydeadinternet.com' in url_lower:
        return 'deadinternet'
    return 'unknown'


# ─── Query Functions ─────────────────────────────────────────────────────────

def get_affect_summary() -> str:
    """Get human-readable affect summary for context injection."""
    mood = get_mood()
    tendency = mood.get_tendency()
    markers = get_markers()
    episodes = get_episodes()

    parts = []
    parts.append(f"Mood: valence={mood.valence:+.2f}, arousal={mood.arousal:.2f}")
    parts.append(f"Tendency: {tendency.value}")

    # Sprott: felt emotion is velocity of mood change, not absolute state
    felt = mood.felt_emotion
    if abs(felt) > 0.01:
        direction = 'improving' if felt > 0 else 'worsening'
        parts.append(f"Felt emotion: {direction} ({felt:+.3f}/step)")

    if episodes:
        labels = [e.label for e in episodes if e.is_active]
        if labels:
            parts.append(f"Active emotions: {', '.join(labels[-3:])}")

    parts.append(f"Somatic markers: {len(markers)} cached")

    s2 = mood.get_system2_effectiveness()
    parts.append(f"System 2 effectiveness: {s2:.2f}")

    consol = mood.get_arousal_consolidation_boost()
    parts.append(f"Consolidation boost: {consol:+.2f}")

    noise = get_retrieval_noise(mood.arousal)
    parts.append(f"Retrieval noise: {noise:.3f}")

    return ". ".join(parts)


def get_affect_history() -> list[dict]:
    """Get session affect history."""
    db = get_db()
    raw = db.kv_get(KV_AFFECT_HISTORY)
    if raw:
        return json.loads(raw) if isinstance(raw, str) else raw
    return []


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _format_valence_bar(valence: float, width: int = 20) -> str:
    """Format valence [-1,+1] as a centered bar."""
    center = width // 2
    pos = int((valence + 1) / 2 * width)
    bar = list('.' * width)
    bar[center] = '|'
    if pos < center:
        for i in range(pos, center):
            bar[i] = '-'
    elif pos > center:
        for i in range(center + 1, min(pos + 1, width)):
            bar[i] = '+'
    return ''.join(bar)


def main():
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    args = sys.argv[1:]
    if not args or args[0] in ('--help', '-h'):
        print(__doc__.strip())
        return

    cmd = args[0]

    if cmd == 'state':
        mood = get_mood()
        tendency = mood.get_tendency()
        params = tendency_to_params(tendency)

        print(f"\n=== Affect State (N1) ===\n")
        print(f"  Mood valence:  {mood.valence:+.4f}  {_format_valence_bar(mood.valence)}")
        print(f"  Mood arousal:  {mood.arousal:.4f}")
        print(f"  Events:        {mood.event_count}")
        print(f"  Tendency:      {tendency.value}")
        print(f"  Temperament:   baseline={mood.temperament.valence_baseline:+.2f}"
              f" reactivity={mood.temperament.arousal_reactivity:.2f}"
              f" loss_aversion={mood.temperament.loss_aversion:.1f}")
        print(f"\n  Behavioral effects:")
        for k, v in params.items():
            print(f"    {k}: {v:+.2f}" if isinstance(v, float) else f"    {k}: {v:+d}")
        print(f"\n  System 2 effectiveness: {mood.get_system2_effectiveness():.3f}")
        print(f"  Consolidation boost:   {mood.get_arousal_consolidation_boost():+.3f}")
        print(f"  Last updated:          {mood.last_updated[:19] if mood.last_updated else 'never'}")

        markers = get_markers()
        print(f"\n  Somatic markers: {len(markers)} cached")

    elif cmd == 'markers':
        markers = get_markers()
        if not markers._markers:
            print("No somatic markers cached.")
            return
        print(f"\n=== Somatic Markers ({len(markers)}) ===\n")
        sorted_markers = sorted(markers._markers.values(),
                                key=lambda m: abs(m.valence), reverse=True)
        for m in sorted_markers[:20]:
            print(f"  [{m.context_hash}] v={m.valence:+.3f} conf={m.confidence:.2f}"
                  f" hits={m.hit_count} intensity={m.intensity:.2f}")

    elif cmd == 'tendency':
        mood = get_mood()
        tendency = mood.get_tendency()
        params = tendency_to_params(tendency)
        print(f"\n=== Action Tendency ===\n")
        print(f"  Current: {tendency.value}")
        print(f"  From: valence={mood.valence:+.3f}, arousal={mood.arousal:.3f}")
        print(f"\n  Parameter adjustments:")
        for k, v in params.items():
            print(f"    {k}: {v:+.2f}" if isinstance(v, float) else f"    {k}: {v:+d}")

    elif cmd == 'history':
        history = get_affect_history()
        if not history:
            print("No affect events this session.")
            return
        print(f"\n=== Affect History ({len(history)} events) ===\n")
        for entry in history[-20:]:
            ts = entry.get('timestamp', '?')[:19]
            ev = entry.get('event', '?')
            v = entry.get('valence', 0)
            a = entry.get('arousal', 0)
            mv = entry.get('mood_valence', 0)
            tend = entry.get('tendency', '?')
            print(f"  [{ts}] {ev:25s} v={v:+.2f} a={a:.2f}  mood={mv:+.3f}  {tend}")

    elif cmd == 'episodes':
        episodes = get_episodes()
        if not episodes:
            print("No active emotion episodes.")
            return
        print(f"\n=== Active Emotion Episodes ({len(episodes)}) ===\n")
        for ep in episodes:
            print(f"  [{ep.label}] v={ep.valence:+.3f} a={ep.arousal:.3f}"
                  f" strength={ep.remaining_strength:.2f}"
                  f" from: {ep.elicitor}")

    elif cmd == 'appraise' and len(args) > 1:
        event_type = args[1]
        # Support optional metadata flags
        meta = {}
        if '--predicted' in args:
            meta['was_predicted'] = True
        if '--failures' in args:
            idx = args.index('--failures')
            if idx + 1 < len(args):
                meta['consecutive_failures'] = int(args[idx + 1])
        if '--social' in args:
            meta['has_contact'] = True

        appraisal = appraise_event(event_type, meta)
        print(f"\n=== Appraisal: {event_type} ===\n")
        print(f"  Novelty:         {appraisal.novelty:.3f}")
        print(f"  Goal relevance:  {appraisal.goal_relevance:.3f}")
        print(f"  Conduciveness:   {appraisal.conduciveness:+.3f}")
        print(f"  Coping potential: {appraisal.coping_potential:.3f}")
        print(f"  Agency:          {appraisal.agency}")
        print(f"\n  Derived valence: {appraisal.valence:+.4f}")
        print(f"  Derived arousal: {appraisal.arousal:.4f}")
        emotion = classify_emotion(appraisal.valence, appraisal.arousal)
        print(f"  Emotion label:   {emotion}")
        if meta:
            print(f"\n  Context: {meta}")

    elif cmd == 'valence' and len(args) > 1:
        memory_id = args[1]
        db = get_db()
        row = db.get_memory(memory_id)
        if not row:
            print(f"Memory {memory_id} not found.")
            return
        valence = row.get('valence', 0.0) or 0.0
        ew = row.get('emotional_weight', 0.5) or 0.5
        qv = row.get('q_value', 0.5) or 0.5
        print(f"\n  Memory: {memory_id}")
        print(f"  Valence:          {valence:+.4f}  {_format_valence_bar(valence)}")
        print(f"  Emotional weight: {ew:.4f}  (unsigned persistence)")
        print(f"  Q-value:          {qv:.4f}  (learned utility)")
        print(f"  Content: {str(row.get('content', ''))[:100]}...")

    elif cmd == 'learn' and len(args) >= 4:
        # learn <platform> <action> <success|fail>
        platform = args[1]
        action = args[2]
        success = args[3].lower() in ('success', 'true', '1', 'ok')
        target = args[4] if len(args) > 4 else ''
        learn_from_api_outcome(platform, action, success, target=target)
        markers = get_markers()
        context_keys = [f'api_{action}', f'platform:{platform}']
        if target:
            context_keys.append(f'target:{target}')
        marker = markers.get_marker(context_keys)
        if marker:
            print(f"Learned: [{marker.context_hash}] v={marker.valence:+.3f}"
                  f" conf={marker.confidence:.2f} hits={marker.hit_count}")
        else:
            print("Marker not found after learn (context mismatch)")

    elif cmd == 'reset':
        global _mood
        _mood = MoodState()
        save_mood()
        print("Mood reset to baseline (Drift temperament).")

    else:
        print(f"Unknown command: {cmd}")
        print("Available: state, markers, tendency, episodes, history,"
              " appraise <evt>, valence <id>, learn <plat> <act> <ok|fail>, reset")
        sys.exit(1)


if __name__ == '__main__':
    main()
