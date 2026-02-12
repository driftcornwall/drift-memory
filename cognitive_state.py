#!/usr/bin/env python3
"""
Cognitive State Tracker — Real-time 5-dimension self-awareness.

Tracks Drift's cognitive state during a session across 5 dimensions:
- Curiosity:    hunger for new information (exploration drive)
- Confidence:   trust in current knowledge (exploitation signal)
- Focus:        depth vs breadth of attention (topic coherence)
- Arousal:      processing intensity (activity level)
- Satisfaction:  quality of session progress (reward signal)

The state updates based on events (search results, API calls, memory stores)
and modifies behavior: high curiosity lowers search thresholds, high focus
keeps priming on-topic, high arousal reduces candidate count.

DB-ONLY: State persists to PostgreSQL KV store.

Usage:
    python cognitive_state.py state          # Show current state
    python cognitive_state.py history        # State changes this session
    python cognitive_state.py trend          # Cross-session trends
    python cognitive_state.py reset          # Reset to neutral
"""

import json
import math
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from db_adapter import get_db

# --- Configuration ---

# EMA smoothing factor: higher = more responsive, lower = more stable
EMA_ALPHA = 0.3

# State clamp range
STATE_MIN = 0.0
STATE_MAX = 1.0

# Default initial state
DEFAULT_STATE = {
    'curiosity': 0.5,
    'confidence': 0.5,
    'focus': 0.5,
    'arousal': 0.3,       # Starts low, builds during session
    'satisfaction': 0.5,
}

# DB KV keys
KV_COGNITIVE_STATE = '.cognitive_state'
KV_COGNITIVE_HISTORY = '.cognitive_history'
KV_COGNITIVE_SESSIONS = '.cognitive_sessions'

# --- Event Deltas ---
# Each event type maps to dimension adjustments
# Positive = increase, negative = decrease
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
}


# --- Core State ---

@dataclass
class CognitiveState:
    curiosity: float = 0.5
    confidence: float = 0.5
    focus: float = 0.5
    arousal: float = 0.3
    satisfaction: float = 0.5
    timestamp: str = ""
    event_count: int = 0

    def vector(self) -> tuple:
        """Return state as a 5-tuple for comparison/math."""
        return (self.curiosity, self.confidence, self.focus,
                self.arousal, self.satisfaction)

    def dominant(self) -> str:
        """Which dimension is highest right now?"""
        dims = {
            'curiosity': self.curiosity,
            'confidence': self.confidence,
            'focus': self.focus,
            'arousal': self.arousal,
            'satisfaction': self.satisfaction,
        }
        return max(dims, key=dims.get)

    def magnitude(self) -> float:
        """Overall activation magnitude (L2 norm of vector)."""
        v = self.vector()
        return round(math.sqrt(sum(x * x for x in v)), 4)

    def volatility_from(self, other: 'CognitiveState') -> float:
        """How much did the state change? (Euclidean distance)"""
        if other is None:
            return 0.0
        v1 = self.vector()
        v2 = other.vector()
        return round(math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2))), 4)

    def to_dict(self) -> dict:
        return {
            'curiosity': round(self.curiosity, 4),
            'confidence': round(self.confidence, 4),
            'focus': round(self.focus, 4),
            'arousal': round(self.arousal, 4),
            'satisfaction': round(self.satisfaction, 4),
            'timestamp': self.timestamp,
            'event_count': self.event_count,
            'dominant': self.dominant(),
            'magnitude': self.magnitude(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CognitiveState':
        return cls(
            curiosity=d.get('curiosity', 0.5),
            confidence=d.get('confidence', 0.5),
            focus=d.get('focus', 0.5),
            arousal=d.get('arousal', 0.3),
            satisfaction=d.get('satisfaction', 0.5),
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
        _current_state = CognitiveState(
            timestamp=datetime.now(timezone.utc).isoformat()
        )

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
    _current_state = CognitiveState(
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    save_state()
    return _current_state


def _clamp(value: float) -> float:
    """Clamp value to valid range."""
    return max(STATE_MIN, min(STATE_MAX, value))


# --- Event Processing ---

def process_event(event_type: str, metadata: dict = None) -> CognitiveState:
    """
    Update cognitive state based on an event.

    Uses EMA smoothing: new_value = alpha * target + (1 - alpha) * current
    This prevents jitter while staying responsive.

    Args:
        event_type: One of the EVENT_DELTAS keys
        metadata: Optional event-specific data (unused for now, future expansion)

    Returns:
        Updated CognitiveState
    """
    deltas = EVENT_DELTAS.get(event_type)
    if not deltas:
        return get_state()

    state = get_state()

    # Apply EMA-smoothed deltas
    for dim, delta in deltas.items():
        current = getattr(state, dim)
        # Target is current + delta, then EMA smooth toward it
        target = _clamp(current + delta)
        new_value = EMA_ALPHA * target + (1 - EMA_ALPHA) * current
        setattr(state, dim, round(_clamp(new_value), 4))

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
    Saves the previous session's final state for trend analysis,
    then resets to neutral.
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


# --- Behavioral Modifiers ---

def get_search_threshold_modifier() -> float:
    """
    Return a modifier for semantic search similarity threshold.

    High curiosity → lower threshold (explore more, accept weaker matches)
    High confidence → higher threshold (be selective)

    Returns: additive modifier (-0.1 to +0.1)
    """
    state = get_state()
    # Curiosity pulls threshold down, confidence pushes up
    modifier = -0.05 * (state.curiosity - 0.5) + 0.05 * (state.confidence - 0.5)
    return round(max(-0.1, min(0.1, modifier)), 3)


def get_priming_modifier() -> dict:
    """
    Return modifiers for priming strategy.

    Returns dict with:
    - curiosity_targets: how many extra curiosity targets (0-3)
    - cooccurrence_expand: how much to expand co-occurrence (0.5-2.0x)
    - max_candidates: multiplier for total candidates (0.5-1.5x)
    """
    state = get_state()

    # High curiosity + low focus → more exploration targets
    curiosity_targets = 0
    if state.curiosity > 0.6 and state.focus < 0.4:
        curiosity_targets = 2
    elif state.curiosity > 0.5:
        curiosity_targets = 1

    # High focus → more co-occurrence expansion (stay on topic)
    cooccurrence_expand = 1.0
    if state.focus > 0.7:
        cooccurrence_expand = 1.5
    elif state.focus < 0.3:
        cooccurrence_expand = 0.7

    # High arousal → fewer candidates (avoid overload)
    max_candidates = 1.0
    if state.arousal > 0.8:
        max_candidates = 0.7
    elif state.arousal < 0.3:
        max_candidates = 1.3

    return {
        'curiosity_targets': curiosity_targets,
        'cooccurrence_expand': round(cooccurrence_expand, 2),
        'max_candidates': round(max_candidates, 2),
    }


# --- CLI ---

def _format_bar(value: float, width: int = 20) -> str:
    """Format a 0-1 value as a visual bar."""
    filled = int(value * width)
    return '#' * filled + '.' * (width - filled)


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
        print(f"\n=== Cognitive State ===\n")
        print(f"  Curiosity:    {state.curiosity:.3f} {_format_bar(state.curiosity)}")
        print(f"  Confidence:   {state.confidence:.3f} {_format_bar(state.confidence)}")
        print(f"  Focus:        {state.focus:.3f} {_format_bar(state.focus)}")
        print(f"  Arousal:      {state.arousal:.3f} {_format_bar(state.arousal)}")
        print(f"  Satisfaction: {state.satisfaction:.3f} {_format_bar(state.satisfaction)}")
        print()
        print(f"  Dominant: {state.dominant()}")
        print(f"  Magnitude: {state.magnitude()}")
        print(f"  Events: {state.event_count}")
        print(f"  Updated: {state.timestamp[:19]}")

        # Show behavioral modifiers
        threshold_mod = get_search_threshold_modifier()
        priming_mod = get_priming_modifier()
        print(f"\n  Behavioral modifiers:")
        print(f"    Search threshold: {threshold_mod:+.3f}")
        print(f"    Curiosity targets: +{priming_mod['curiosity_targets']}")
        print(f"    Co-occurrence expand: {priming_mod['cooccurrence_expand']}x")
        print(f"    Max candidates: {priming_mod['max_candidates']}x")

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
            print(f"  [{ts}] {event:20s} C={s.get('curiosity', 0):.2f} "
                  f"Conf={s.get('confidence', 0):.2f} F={s.get('focus', 0):.2f} "
                  f"A={s.get('arousal', 0):.2f} S={s.get('satisfaction', 0):.2f}")

    elif cmd == 'trend':
        sessions = get_session_trends()
        if not sessions:
            print("No previous session data. Trends need 2+ sessions.")
            return
        print(f"\n=== Cognitive State Trends ({len(sessions)} sessions) ===\n")
        dims = ['curiosity', 'confidence', 'focus', 'arousal', 'satisfaction']
        print(f"  {'Session':>8s}  {'Events':>6s}  " + "  ".join(f"{d:>7s}" for d in dims) + "  Dominant")
        for i, s in enumerate(sessions[-10:], 1):
            vals = "  ".join(f"{s.get(d, 0):.3f}" for d in dims)
            print(f"  {i:8d}  {s.get('event_count', 0):6d}  {vals}  {s.get('dominant', '?')}")

        # Averages
        if len(sessions) >= 2:
            print()
            avgs = "  ".join(
                f"{sum(s.get(d, 0.5) for s in sessions) / len(sessions):.3f}"
                for d in dims
            )
            print(f"  {'avg':>8s}  {'':>6s}  {avgs}")

    elif cmd == 'reset':
        reset_state()
        print("Cognitive state reset to neutral.")

    elif cmd == 'simulate':
        # Debug: simulate some events
        events = ['search_success', 'search_success', 'new_topic',
                  'memory_stored', 'api_error', 'search_failure',
                  'same_topic', 'cooccurrence_formed', 'search_success']
        print(f"\n=== Simulating {len(events)} events ===\n")
        for event in events:
            state = process_event(event)
            print(f"  {event:25s} -> C={state.curiosity:.3f} Conf={state.confidence:.3f} "
                  f"F={state.focus:.3f} A={state.arousal:.3f} S={state.satisfaction:.3f}")

    else:
        print(f"Unknown command: {cmd}")
        print("Available: state, history, trend, reset, simulate")
        sys.exit(1)


if __name__ == '__main__':
    main()
