# Phase 3: Cognitive State Tracker

## Status: NEXT TO BUILD
Inspired by: Boyyey/Neuro-Symbolic-Consciousness-Engine

## What It Does
A 5-dimension real-time cognitive state that modifies search behavior and priming based on live state. This isn't a static score — it's a living representation of what Drift is experiencing RIGHT NOW during a session.

## Five Dimensions

| Dimension | Range | What it tracks | Updated by |
|-----------|-------|----------------|------------|
| **Curiosity** | 0.0-1.0 | Hunger for new information | Sparsity of recent recalls, failed searches, new topics |
| **Confidence** | 0.0-1.0 | Trust in current knowledge | Successful retrievals, repeated patterns, API successes |
| **Focus** | 0.0-1.0 | Depth vs breadth of attention | Topic coherence of recent recalls, single-domain vs scatter |
| **Arousal** | 0.0-1.0 | Processing intensity | Rate of API calls, memory stores, error frequency |
| **Satisfaction** | 0.0-1.0 | Quality of session progress | Retrieval success rate, generative outcomes, task completion |

## Architecture

### New File: `memory/cognitive_state.py` (~300 LOC)

```python
@dataclass
class CognitiveState:
    curiosity: float = 0.5
    confidence: float = 0.5
    focus: float = 0.5
    arousal: float = 0.3      # Starts low, builds during session
    satisfaction: float = 0.5
    timestamp: str = ""

    def vector(self) -> tuple:
        return (self.curiosity, self.confidence, self.focus,
                self.arousal, self.satisfaction)

    def dominant(self) -> str:
        """Which dimension is highest right now?"""
        ...
```

### Event-Driven Updates
The state updates based on events during the session:

| Event | Curiosity | Confidence | Focus | Arousal | Satisfaction |
|-------|-----------|------------|-------|---------|-------------|
| Successful search | -0.05 | +0.1 | +0.05 | +0.02 | +0.05 |
| Failed search | +0.15 | -0.1 | -0.05 | +0.05 | -0.05 |
| Memory stored | -0.02 | +0.05 | +0.02 | +0.03 | +0.08 |
| API error | +0.05 | -0.15 | -0.1 | +0.1 | -0.1 |
| New topic detected | +0.1 | -0.05 | -0.15 | +0.05 | 0 |
| Same topic continued | -0.05 | +0.05 | +0.15 | -0.02 | +0.03 |
| Co-occurrence formed | -0.03 | +0.08 | +0.03 | +0.02 | +0.1 |

### How It Modifies Behavior

1. **Search threshold adjustment**
   - High curiosity → lower similarity threshold (accept weaker matches, explore more)
   - High confidence → higher threshold (be more selective)

2. **Priming strategy**
   - High curiosity + low focus → more curiosity targets in priming
   - High focus → more co-occurrence expansion (stay on topic)
   - High arousal → fewer priming candidates (avoid overload)

3. **Decay rate modulation**
   - High satisfaction → slower decay (good session, preserve connections)
   - Low satisfaction + high arousal → faster decay (noise session, clean up)

### Persistence
- State persists within session via DB KV (`.cognitive_state`)
- State history logged per-session for tracking cognitive patterns over time
- Initial state: neutral (all 0.5 except arousal at 0.3)
- Resets at session start but loads previous session's final state for trend analysis

### Integration Points
1. **post_tool_use hook** — Update state on API responses, search results, errors
2. **semantic_search.py** — Read curiosity/confidence to adjust threshold
3. **memory_manager.py** — Read focus to adjust priming strategy
4. **system_vitals.py** — Record end-of-session state snapshot

### Vitals Metrics
- `cognitive_curiosity`, `cognitive_confidence`, `cognitive_focus`, `cognitive_arousal`, `cognitive_satisfaction`
- `cognitive_dominant` — which dimension dominated the session
- `cognitive_volatility` — how much the state changed during the session

### Toolkit Commands
- `cognitive-state` — Show current state
- `cognitive-history` — Show state changes during session
- `cognitive-trend` — Cross-session state trends

### Impact (what we expect)
- Search results become context-sensitive (curious mode explores, confident mode is selective)
- Priming adapts to session activity instead of being static
- Vitals can track cognitive patterns across sessions (am I always high-curiosity? always low-satisfaction?)
- Enables future self-regulation: "I notice my confidence is dropping, let me recall some proven knowledge"

### Careful Implementation Notes
- All state updates must be clamped to [0.0, 1.0]
- Exponential moving average (EMA) for smoothing — avoid jitter
- Must not slow down search or priming (state lookup should be O(1))
- Fallback to default state if any component fails
