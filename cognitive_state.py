#!/usr/bin/env python3
"""
Cognitive State Tracker v2.0 — Beta Distribution Uncertainty Quantification.

Tracks Drift's cognitive state during a session across 5 dimensions:
- Curiosity:    hunger for new information (exploration drive)
- Confidence:   trust in current knowledge (exploitation signal)
- Focus:        depth vs breadth of attention (topic coherence)
- Arousal:      processing intensity (activity level)
- Satisfaction:  quality of session progress (reward signal)

v2.0 UPGRADE: Each dimension is a Beta(alpha, beta) distribution instead of
a scalar. This captures UNCERTAINTY — not just "how curious am I?" but
"how certain am I about my curiosity level?"

Key insight: Two states with identical curiosity MEANS but different
uncertainty should behave differently. High uncertainty = explore more.
Low uncertainty = commit to current strategy.

Behavioral modifiers now scale with distribution spread (WaterFall's insight).

DB-ONLY: State persists to PostgreSQL KV store.

Usage:
    python cognitive_state.py state          # Show current state + uncertainty
    python cognitive_state.py uncertainty    # Detailed distribution view
    python cognitive_state.py history        # State changes this session
    python cognitive_state.py trend          # Cross-session trends
    python cognitive_state.py reset          # Reset to neutral
    python cognitive_state.py simulate       # Run event simulation
"""

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from db_adapter import get_db

# --- Configuration ---

# Evidence scale: converts event deltas to Beta distribution evidence.
# Calibrated so delta=0.10 produces ~0.018 mean shift (similar to old EMA).
EVIDENCE_SCALE = 3.0

# Decay rate per event: erodes old evidence via multiplicative decay.
# 0.98 = ~35-event half-life. Prevents distribution from peaking too hard.
EVIDENCE_DECAY = 0.98

# Mean reversion: adds pseudo-counts biased toward default mean each event.
# Prevents runaway drift. 0.15 tuned session 18: curiosity was stuck at 0.28
# because success events (-0.05 each) vastly outnumber novelty events.
# 0.10 wasn't enough pull toward 0.5 to keep curiosity healthy.
MEAN_REVERSION_STRENGTH = 0.15

# Parameter clamps
MIN_PARAM = 0.5    # Prevent degenerate distributions
MAX_PARAM = 50.0   # Prevent overly peaked distributions

# State clamp range (for backward-compat mean values)
STATE_MIN = 0.01
STATE_MAX = 0.99

# Default initial state (target means)
DEFAULT_STATE = {
    'curiosity': 0.5,
    'confidence': 0.5,
    'focus': 0.5,
    'arousal': 0.3,       # Starts low, builds during session
    'satisfaction': 0.5,
}

# Initial concentration (alpha + beta) — lower = more uncertain at session start
INITIAL_CONCENTRATION = 4.0

# Dimension names (ordered)
DIMENSIONS = ['curiosity', 'confidence', 'focus', 'arousal', 'satisfaction']

# DB KV keys
KV_COGNITIVE_STATE = '.cognitive_state'
KV_COGNITIVE_HISTORY = '.cognitive_history'
KV_COGNITIVE_SESSIONS = '.cognitive_sessions'

# --- Event Deltas ---
# Same events and magnitudes as v1. Sign determines alpha vs beta update.
EVENT_DELTAS = {
    'search_success': {
        'curiosity': -0.05,
        'confidence': +0.10,
        'focus': +0.05,
        'arousal': +0.02,
        'satisfaction': +0.05,
    },
    'search_failure': {
        'curiosity': +0.15,
        'confidence': -0.10,
        'focus': -0.05,
        'arousal': +0.05,
        'satisfaction': -0.05,
    },
    'memory_stored': {
        'curiosity': -0.02,
        'confidence': +0.05,
        'focus': +0.02,
        'arousal': +0.03,
        'satisfaction': +0.08,
    },
    'memory_recalled': {
        'curiosity': -0.03,
        'confidence': +0.05,
        'focus': +0.03,
        'arousal': +0.01,
        'satisfaction': +0.03,
    },
    'api_error': {
        'curiosity': +0.05,
        'confidence': -0.15,
        'focus': -0.10,
        'arousal': +0.10,
        'satisfaction': -0.10,
    },
    'api_success': {
        'curiosity': 0,
        'confidence': +0.03,
        'focus': +0.02,
        'arousal': +0.01,
        'satisfaction': +0.02,
    },
    'new_topic': {
        'curiosity': +0.10,
        'confidence': -0.05,
        'focus': -0.15,
        'arousal': +0.05,
        'satisfaction': 0,
    },
    'same_topic': {
        'curiosity': -0.05,
        'confidence': +0.05,
        'focus': +0.15,
        'arousal': -0.02,
        'satisfaction': +0.03,
    },
    'cooccurrence_formed': {
        'curiosity': -0.03,
        'confidence': +0.08,
        'focus': +0.03,
        'arousal': +0.02,
        'satisfaction': +0.10,
    },
    'explanation_generated': {
        'curiosity': -0.02,
        'confidence': +0.05,
        'focus': +0.02,
        'arousal': +0.01,
        'satisfaction': +0.04,
    },
    'curiosity_target_hit': {
        'curiosity': +0.08,
        'confidence': -0.02,
        'focus': -0.05,
        'arousal': +0.05,
        'satisfaction': +0.06,
    },
    'contradiction_detected': {
        'curiosity': +0.10,      # Contradiction drives exploration
        'confidence': -0.15,     # Confidence in current knowledge drops
        'focus': +0.05,          # Attention sharpens on conflict
        'arousal': +0.08,        # Processing intensity rises
        'satisfaction': -0.05,   # Dissatisfaction drives resolution
    },
}


# --- Beta Distribution Dimension ---

@dataclass
class DimensionDist:
    """Beta distribution modeling a single cognitive dimension.

    Mean = alpha / (alpha + beta)
    Variance = a*b / ((a+b)^2 * (a+b+1))
    Uncertainty = inverse of concentration, normalized to [0, 1]
    """
    alpha: float = 2.0
    beta: float = 2.0

    @property
    def mean(self) -> float:
        """Expected value of the distribution."""
        total = self.alpha + self.beta
        if total <= 0:
            return 0.5
        return self.alpha / total

    @property
    def variance(self) -> float:
        """Variance of the distribution."""
        a, b = self.alpha, self.beta
        total = a + b
        if total <= 0 or total + 1 <= 0:
            return 0.25
        return (a * b) / (total * total * (total + 1))

    @property
    def std(self) -> float:
        """Standard deviation."""
        return math.sqrt(max(0, self.variance))

    @property
    def concentration(self) -> float:
        """Total evidence (alpha + beta). Higher = more certain."""
        return self.alpha + self.beta

    @property
    def uncertainty(self) -> float:
        """Normalized uncertainty in [0, 1].

        Based on concentration (total evidence):
        conc = 2  -> unc = 1.0  (just the prior, no evidence)
        conc = 8  -> unc = 0.67 (some evidence)
        conc = 14 -> unc = 0.33 (moderate evidence)
        conc = 20 -> unc = 0.0  (strong evidence, very certain)
        """
        MIN_CONC = 2.0
        MAX_CONC = 20.0
        raw = 1.0 - (self.concentration - MIN_CONC) / (MAX_CONC - MIN_CONC)
        return round(max(0.0, min(1.0, raw)), 4)

    def update(self, delta: float):
        """Update distribution with signed evidence from an event delta.

        delta > 0: evidence the dimension should be HIGHER -> increase alpha
        delta < 0: evidence the dimension should be LOWER -> increase beta
        delta = 0: no evidence (skip)
        """
        if delta == 0:
            return
        evidence = abs(delta) * EVIDENCE_SCALE
        if delta > 0:
            self.alpha += evidence
        else:
            self.beta += evidence

    def decay(self):
        """Multiplicative decay - erodes old evidence, prevents over-certainty."""
        self.alpha = max(MIN_PARAM, self.alpha * EVIDENCE_DECAY)
        self.beta = max(MIN_PARAM, self.beta * EVIDENCE_DECAY)

    def revert_toward(self, target_mean: float):
        """Add pseudo-counts biased toward target mean (mean reversion).

        This pulls the distribution mean toward the default over time,
        preventing runaway drift from one-sided event sequences.
        """
        self.alpha += target_mean * MEAN_REVERSION_STRENGTH
        self.beta += (1 - target_mean) * MEAN_REVERSION_STRENGTH

    def clamp(self):
        """Ensure parameters stay in valid range."""
        self.alpha = max(MIN_PARAM, min(MAX_PARAM, self.alpha))
        self.beta = max(MIN_PARAM, min(MAX_PARAM, self.beta))

    def to_dict(self) -> dict:
        return {
            'alpha': round(self.alpha, 4),
            'beta': round(self.beta, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'DimensionDist':
        return cls(alpha=d.get('alpha', 2.0), beta=d.get('beta', 2.0))

    @classmethod
    def from_scalar(cls, mean: float, concentration: float = None) -> 'DimensionDist':
        """Create from a legacy scalar mean value."""
        if concentration is None:
            concentration = INITIAL_CONCENTRATION
        mean = max(0.01, min(0.99, mean))
        return cls(alpha=mean * concentration, beta=(1 - mean) * concentration)


# --- Core State ---

def _default_distributions() -> dict:
    """Create default distributions for all dimensions."""
    return {
        dim: DimensionDist.from_scalar(DEFAULT_STATE[dim])
        for dim in DIMENSIONS
    }


class CognitiveState:
    """5-dimensional cognitive state with Beta distribution uncertainty.

    Backward compatible: .curiosity, .confidence etc. return float means.
    New: .get_dist(dim) returns full DimensionDist with uncertainty info.
    """

    def __init__(self, distributions: dict = None, timestamp: str = "",
                 event_count: int = 0):
        self._dists = distributions or _default_distributions()
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.event_count = event_count

    # --- Backward-compatible float properties ---

    @property
    def curiosity(self) -> float:
        return round(self._dists['curiosity'].mean, 4)

    @property
    def confidence(self) -> float:
        return round(self._dists['confidence'].mean, 4)

    @property
    def focus(self) -> float:
        return round(self._dists['focus'].mean, 4)

    @property
    def arousal(self) -> float:
        return round(self._dists['arousal'].mean, 4)

    @property
    def satisfaction(self) -> float:
        return round(self._dists['satisfaction'].mean, 4)

    # --- Distribution access ---

    def get_dist(self, dim: str) -> DimensionDist:
        """Get the full Beta distribution for a dimension."""
        return self._dists[dim]

    def get_uncertainty(self, dim: str) -> float:
        """Get normalized uncertainty for a dimension."""
        return self._dists[dim].uncertainty

    @property
    def uncertainties(self) -> dict:
        """All dimension uncertainties as a dict."""
        return {dim: self._dists[dim].uncertainty for dim in DIMENSIONS}

    @property
    def mean_uncertainty(self) -> float:
        """Average uncertainty across all dimensions."""
        vals = [self._dists[dim].uncertainty for dim in DIMENSIONS]
        return round(sum(vals) / len(vals), 4)

    # --- Existing methods (preserved for all consumers) ---

    def vector(self) -> tuple:
        """Return state as a 5-tuple of means for comparison."""
        return (self.curiosity, self.confidence, self.focus,
                self.arousal, self.satisfaction)

    def dominant(self) -> str:
        """Which dimension is highest right now?"""
        dims = {dim: getattr(self, dim) for dim in DIMENSIONS}
        return max(dims, key=dims.get)

    def magnitude(self) -> float:
        """Overall activation magnitude (L2 norm of mean vector)."""
        v = self.vector()
        return round(math.sqrt(sum(x * x for x in v)), 4)

    def volatility_from(self, other: 'CognitiveState') -> float:
        """How much did the mean state change? (Euclidean distance)"""
        if other is None:
            return 0.0
        v1 = self.vector()
        v2 = other.vector()
        return round(math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2))), 4)

    def to_dict(self) -> dict:
        """Serialize state. Backward-compatible scalar means + new distribution data."""
        return {
            # Scalar means (backward compatible — all consumers read these)
            'curiosity': self.curiosity,
            'confidence': self.confidence,
            'focus': self.focus,
            'arousal': self.arousal,
            'satisfaction': self.satisfaction,
            'timestamp': self.timestamp,
            'event_count': self.event_count,
            'dominant': self.dominant(),
            'magnitude': self.magnitude(),
            # v2.0: distribution parameters
            'distributions': {
                dim: self._dists[dim].to_dict()
                for dim in DIMENSIONS
            },
            # v2.0: uncertainty values
            'uncertainties': self.uncertainties,
            'mean_uncertainty': self.mean_uncertainty,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CognitiveState':
        """Deserialize state. Handles both v1 (scalar) and v2 (distribution) formats."""
        dists = {}

        if 'distributions' in d:
            # v2 format: full distribution params
            for dim in DIMENSIONS:
                if dim in d['distributions']:
                    dists[dim] = DimensionDist.from_dict(d['distributions'][dim])
                else:
                    dists[dim] = DimensionDist.from_scalar(
                        d.get(dim, DEFAULT_STATE[dim]))
        else:
            # v1 format: convert scalars to weak-prior distributions
            for dim in DIMENSIONS:
                scalar = d.get(dim, DEFAULT_STATE[dim])
                dists[dim] = DimensionDist.from_scalar(scalar)

        return cls(
            distributions=dists,
            timestamp=d.get('timestamp', ''),
            event_count=d.get('event_count', 0),
        )


# --- State Management ---

_current_state: Optional[CognitiveState] = None


def get_state() -> CognitiveState:
    """Get the current cognitive state. Loads from DB if not cached."""
    global _current_state
    if _current_state is not None:
        return _current_state

    db = get_db()
    raw = db.kv_get(KV_COGNITIVE_STATE)
    if raw:
        data = json.loads(raw) if isinstance(raw, str) else raw
        _current_state = CognitiveState.from_dict(data)
    else:
        _current_state = CognitiveState()

    return _current_state


def save_state():
    """Persist current state to DB."""
    state = get_state()
    state.timestamp = datetime.now(timezone.utc).isoformat()
    db = get_db()
    db.kv_set(KV_COGNITIVE_STATE, state.to_dict())


def reset_state():
    """Reset to default neutral state. Called at session start."""
    global _current_state
    _current_state = CognitiveState()
    save_state()
    return _current_state


# --- Event Processing ---

def process_event(event_type: str, metadata: dict = None) -> CognitiveState:
    """
    Update cognitive state based on an event.

    v2.0: Uses Bayesian Beta distribution updates instead of EMA.
    For each dimension with a non-zero delta:
      1. Decay old evidence (multiplicative)
      2. Mean reversion (pseudo-count toward default)
      3. Apply new evidence (delta sign -> alpha or beta)
      4. Clamp parameters

    Args:
        event_type: One of the EVENT_DELTAS keys
        metadata: Optional event-specific data

    Returns:
        Updated CognitiveState
    """
    deltas = EVENT_DELTAS.get(event_type)
    if not deltas:
        return get_state()

    state = get_state()

    for dim in DIMENSIONS:
        delta = deltas.get(dim, 0)
        dist = state._dists[dim]

        # 1. Decay old evidence (prevents over-certainty in long sessions)
        dist.decay()

        # 2. Mean reversion (prevents runaway drift from one-sided events)
        dist.revert_toward(DEFAULT_STATE[dim])

        # 3. Apply new evidence from this event
        dist.update(delta)

        # 4. Clamp parameters to valid range
        dist.clamp()

    state.event_count += 1
    state.timestamp = datetime.now(timezone.utc).isoformat()

    # Log to history
    _log_history_entry(event_type, state)

    # Persist
    save_state()

    return state


def _log_history_entry(event_type: str, state: CognitiveState):
    """Append a state change to the session history."""
    db = get_db()
    raw = db.kv_get(KV_COGNITIVE_HISTORY)
    history = []
    if raw:
        history = json.loads(raw) if isinstance(raw, str) else raw

    history.append({
        'event': event_type,
        'state': state.to_dict(),
        'timestamp': state.timestamp,
    })

    # Keep last 200 entries per session
    history = history[-200:]
    db.kv_set(KV_COGNITIVE_HISTORY, history)


# --- Session Management ---

def start_session():
    """
    Initialize cognitive state for a new session.
    Saves previous session's final state for trend analysis, then resets.
    """
    db = get_db()

    # Save previous session's final state to session log
    current = get_state()
    if current.event_count > 0:
        raw = db.kv_get(KV_COGNITIVE_SESSIONS)
        sessions = []
        if raw:
            sessions = json.loads(raw) if isinstance(raw, str) else raw
        sessions.append(current.to_dict())
        sessions = sessions[-50:]  # Keep last 50 sessions
        db.kv_set(KV_COGNITIVE_SESSIONS, sessions)

    # Clear session history
    db.kv_set(KV_COGNITIVE_HISTORY, [])

    # Reset state
    reset_state()


def end_session() -> dict:
    """
    Finalize cognitive state for the session.
    Returns summary for vitals recording.
    """
    state = get_state()
    db = get_db()

    # Get history for volatility calculation
    raw = db.kv_get(KV_COGNITIVE_HISTORY)
    history = []
    if raw:
        history = json.loads(raw) if isinstance(raw, str) else raw

    # Calculate session volatility (average distance between consecutive states)
    volatility = 0.0
    if len(history) >= 2:
        total_dist = 0.0
        for i in range(1, len(history)):
            s1 = CognitiveState.from_dict(history[i - 1].get('state', {}))
            s2 = CognitiveState.from_dict(history[i].get('state', {}))
            total_dist += s2.volatility_from(s1)
        volatility = round(total_dist / (len(history) - 1), 4)

    return {
        'final_state': state.to_dict(),
        'event_count': state.event_count,
        'volatility': volatility,
        'dominant': state.dominant(),
        'mean_uncertainty': state.mean_uncertainty,
        'history_length': len(history),
    }


# --- Query Functions ---

def get_session_history() -> list[dict]:
    """Get state change history for current session."""
    db = get_db()
    raw = db.kv_get(KV_COGNITIVE_HISTORY)
    if raw:
        return json.loads(raw) if isinstance(raw, str) else raw
    return []


def get_session_trends() -> list[dict]:
    """Get final states from previous sessions for trend analysis."""
    db = get_db()
    raw = db.kv_get(KV_COGNITIVE_SESSIONS)
    if raw:
        return json.loads(raw) if isinstance(raw, str) else raw
    return []


# --- Behavioral Modifiers (v2.0 — uncertainty-aware) ---

def get_search_threshold_modifier() -> float:
    """
    Return a modifier for semantic search similarity threshold.

    v2.0: Scales with UNCERTAINTY, not just mean value.

    High curiosity + high curiosity uncertainty -> lower threshold (explore aggressively)
    High confidence + low confidence uncertainty -> higher threshold (commit to strategy)
    High overall uncertainty -> slight exploration bonus

    Returns: additive modifier (-0.15 to +0.15)
    """
    state = get_state()

    cur_mean = state.curiosity
    conf_mean = state.confidence
    cur_unc = state.get_uncertainty('curiosity')
    conf_unc = state.get_uncertainty('confidence')

    # Base modifier (same direction as v1)
    base = -0.05 * (cur_mean - 0.5) + 0.05 * (conf_mean - 0.5)

    # Uncertainty scaling:
    # High curiosity uncertainty amplifies exploration signal
    # High confidence uncertainty dampens exploitation signal
    uncertainty_scale = 1.0 + 0.5 * cur_unc - 0.3 * conf_unc

    modifier = base * uncertainty_scale

    # Pure uncertainty bonus: if very uncertain overall, lean toward exploration
    avg_unc = state.mean_uncertainty
    if avg_unc > 0.6:
        modifier -= 0.03

    return round(max(-0.15, min(0.15, modifier)), 3)


def get_priming_modifier() -> dict:
    """
    Return modifiers for priming strategy.

    v2.0: Curiosity targets consider uncertainty signal.
    High uncertainty in curiosity = "I don't know how curious I should be" = EXPLORE.

    Returns dict with:
    - curiosity_targets: extra curiosity targets (0-3)
    - cooccurrence_expand: co-occurrence expansion multiplier (0.5-2.0x)
    - max_candidates: total candidate multiplier (0.5-1.5x)
    - uncertainty_mode: 'explore' | 'exploit' | 'balanced'
    """
    state = get_state()

    cur_mean = state.curiosity
    cur_unc = state.get_uncertainty('curiosity')
    focus_mean = state.focus
    focus_unc = state.get_uncertainty('focus')

    # Effective curiosity: mean PLUS uncertainty bonus
    # "I'm uncertain about my curiosity" -> explore to resolve it
    # v2.0.1: Increased uncertainty weight from 0.3 to 0.4 and lowered thresholds.
    # With 88% isolated memories, we were in a dead zone where neither condition fired.
    effective_curiosity = cur_mean + cur_unc * 0.4

    curiosity_targets = 0
    if effective_curiosity > 0.6 and focus_mean < 0.4:
        curiosity_targets = 3
    elif effective_curiosity > 0.45:
        curiosity_targets = 1
    elif cur_unc > 0.6:
        # High uncertainty even with low mean -> explore anyway
        curiosity_targets = 2

    # Co-occurrence expansion: considers focus certainty
    cooccurrence_expand = 1.0
    if focus_mean > 0.7 and focus_unc < 0.4:
        cooccurrence_expand = 1.5  # Certain and focused -> deep associations
    elif focus_mean < 0.3:
        cooccurrence_expand = 0.7
    elif focus_unc > 0.6:
        cooccurrence_expand = 1.2  # Uncertain focus -> slightly broader

    # Candidate count considers arousal
    max_candidates = 1.0
    if state.arousal > 0.8:
        max_candidates = 0.7
    elif state.arousal < 0.3:
        max_candidates = 1.3

    # Overall uncertainty mode
    avg_unc = state.mean_uncertainty
    if avg_unc > 0.6:
        mode = 'explore'
    elif avg_unc < 0.3:
        mode = 'exploit'
    else:
        mode = 'balanced'

    return {
        'curiosity_targets': curiosity_targets,
        'cooccurrence_expand': round(cooccurrence_expand, 2),
        'max_candidates': round(max_candidates, 2),
        'uncertainty_mode': mode,
    }


# --- CLI ---

def _format_bar(value: float, width: int = 20) -> str:
    """Format a 0-1 value as a visual bar."""
    filled = int(value * width)
    return '#' * filled + '.' * (width - filled)


def _format_uncertainty(unc: float) -> str:
    """Format uncertainty as a compact indicator."""
    if unc > 0.7:
        return f"???  {unc:.2f}"
    elif unc > 0.4:
        return f"??   {unc:.2f}"
    elif unc > 0.2:
        return f"?    {unc:.2f}"
    else:
        return f"     {unc:.2f}"


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
        state = get_state()
        print(f"\n=== Cognitive State v2.0 (Beta Distribution) ===\n")
        for dim in DIMENSIONS:
            mean = getattr(state, dim)
            unc = state.get_uncertainty(dim)
            dist = state.get_dist(dim)
            label = f"{dim.capitalize():>14s}"
            bar = _format_bar(mean)
            unc_str = _format_uncertainty(unc)
            print(f"  {label}: {mean:.3f} {bar}  unc={unc_str}  a={dist.alpha:.1f} b={dist.beta:.1f}")

        print()
        print(f"  Dominant: {state.dominant()}")
        print(f"  Magnitude: {state.magnitude()}")
        print(f"  Mean uncertainty: {state.mean_uncertainty:.3f}")
        print(f"  Events: {state.event_count}")
        print(f"  Updated: {state.timestamp[:19]}")

        # Show behavioral modifiers
        threshold_mod = get_search_threshold_modifier()
        priming_mod = get_priming_modifier()
        print(f"\n  Behavioral modifiers:")
        print(f"    Search threshold: {threshold_mod:+.3f} (range: +/-0.15)")
        print(f"    Curiosity targets: +{priming_mod['curiosity_targets']}")
        print(f"    Co-occurrence expand: {priming_mod['cooccurrence_expand']}x")
        print(f"    Max candidates: {priming_mod['max_candidates']}x")
        print(f"    Uncertainty mode: {priming_mod['uncertainty_mode']}")

    elif cmd == 'uncertainty':
        state = get_state()
        print(f"\n=== Distribution Detail ===\n")
        for dim in DIMENSIONS:
            dist = state.get_dist(dim)
            print(f"  {dim.capitalize()}")
            print(f"    Alpha: {dist.alpha:.4f}  Beta: {dist.beta:.4f}")
            print(f"    Mean:  {dist.mean:.4f}  Std:  {dist.std:.4f}")
            print(f"    Concentration: {dist.concentration:.2f}  Uncertainty: {dist.uncertainty:.4f}")
            print()

        print(f"  Overall mean uncertainty: {state.mean_uncertainty:.4f}")

        threshold_mod = get_search_threshold_modifier()
        print(f"\n  Search modifier: {threshold_mod:+.3f}")
        print(f"    (v1 range was +/-0.10, v2 range is +/-0.15)")

    elif cmd == 'history':
        history = get_session_history()
        if not history:
            print("No state changes this session.")
            return
        print(f"\n=== Cognitive State History ({len(history)} events) ===\n")
        for entry in history[-20:]:  # Show last 20
            event = entry.get('event', '?')
            s = entry.get('state', {})
            ts = entry.get('timestamp', '?')[:19]
            unc = s.get('mean_uncertainty', '?')
            unc_str = f"{unc:.2f}" if isinstance(unc, (int, float)) else str(unc)
            print(f"  [{ts}] {event:20s} C={s.get('curiosity', 0):.2f} "
                  f"Conf={s.get('confidence', 0):.2f} F={s.get('focus', 0):.2f} "
                  f"A={s.get('arousal', 0):.2f} S={s.get('satisfaction', 0):.2f} "
                  f"unc={unc_str}")

    elif cmd == 'trend':
        sessions = get_session_trends()
        if not sessions:
            print("No previous session data. Trends need 2+ sessions.")
            return
        print(f"\n=== Cognitive State Trends ({len(sessions)} sessions) ===\n")
        dims = DIMENSIONS
        print(f"  {'Sess':>4s}  {'Evts':>4s}  "
              + "  ".join(f"{d[:5]:>5s}" for d in dims)
              + "   Unc   Dominant")
        for i, s in enumerate(sessions[-10:], 1):
            vals = "  ".join(f"{s.get(d, 0):.2f}" for d in dims)
            unc = s.get('mean_uncertainty', '?')
            unc_str = f"{unc:.2f}" if isinstance(unc, (int, float)) else str(unc)
            print(f"  {i:4d}  {s.get('event_count', 0):4d}  {vals}  "
                  f"{unc_str:>5s}  {s.get('dominant', '?')}")

        # Averages
        if len(sessions) >= 2:
            print()
            avgs = "  ".join(
                f"{sum(s.get(d, 0.5) for s in sessions) / len(sessions):.2f}"
                for d in dims
            )
            print(f"  {'avg':>4s}  {'':>4s}  {avgs}")

    elif cmd == 'reset':
        reset_state()
        print("Cognitive state reset to neutral (Beta distributions, concentration=4.0).")

    elif cmd == 'simulate':
        # Simulate a sequence of events to verify behavior
        # NOTE: Uses isolated state — does NOT affect production DB
        events = ['search_success', 'search_success', 'new_topic',
                  'memory_stored', 'api_error', 'search_failure',
                  'same_topic', 'cooccurrence_formed', 'search_success']
        print(f"\n=== Simulating {len(events)} events (Beta Distribution v2.0) ===\n")

        # Save production state
        saved_state = get_state()
        saved_dict = saved_state.to_dict()

        reset_state()
        for event in events:
            state = process_event(event)
            unc = state.mean_uncertainty
            print(f"  {event:25s} -> C={state.curiosity:.3f} Conf={state.confidence:.3f} "
                  f"F={state.focus:.3f} A={state.arousal:.3f} S={state.satisfaction:.3f} "
                  f"unc={unc:.3f}")
        print()
        # Show final distribution detail
        print("  Final distributions:")
        for dim in DIMENSIONS:
            dist = state.get_dist(dim)
            print(f"    {dim:>14s}: a={dist.alpha:.2f} b={dist.beta:.2f} "
                  f"mean={dist.mean:.3f} unc={dist.uncertainty:.3f}")

        # Restore production state
        global _current_state
        _current_state = CognitiveState.from_dict(saved_dict)
        save_state()
        print("\n  (Production state restored — simulation was isolated)")

    else:
        print(f"Unknown command: {cmd}")
        print("Available: state, uncertainty, history, trend, reset, simulate")
        sys.exit(1)


if __name__ == '__main__':
    main()
