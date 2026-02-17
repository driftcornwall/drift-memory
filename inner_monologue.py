#!/usr/bin/env python3
"""
N6 Phase 0.5: Inner Monologue Engine

Gives Drift a separate internal voice — a small LLM (Gemma 3 4B) that evaluates
surfaced memories before they reach the main context. This is the separation between
experiencing and acting that biological cognition has.

Research basis:
- MIRROR (Hsing 2025): Thinker/Talker split, 21% improvement
- DAVIS (Pham Dinh 2024): KG-powered monologue prevents confabulation
- Reflexion (Shinn 2023): Verbal self-reflection as learning
- Fernyhough (2004): Expanded vs condensed inner speech
- Buckner (2025): LLMs as inner speech components, not full minds
- CoALA (Sumers 2023): Internal action space = cognitive inner speech

Three modes (Fernyhough):
- Condensed: ~50-100 tokens, ~400ms, default (arousal < 0.5)
- Expanded: ~200-400 tokens, ~2s, high arousal/novelty/conflict
- Dialogic: multi-turn KG queries, ~5s, complex planning

Architecture:
- Primary: Gemma 3 4B via Ollama (local, free)
- Fallback: GPT-5-mini via OpenAI (paid, for complex queries)
- Structured JSON output (DAVIS-inspired, prevents confabulation)
- Every claim references a memory_id or KG edge (grounded, not free-form)

Integration points (ALL WIRED):
- semantic_search.py: Stage 16, after binding, annotates results
- workspace_manager.py: Competes for broadcast budget ('meta' category)
- session_start.py: Evaluates primed memories at wake-up
- stop.py: Expanded session reflection at sleep

Usage:
    python inner_monologue.py evaluate "query" --memories '[...]'
    python inner_monologue.py reflect --session-summary "..."
    python inner_monologue.py status
    python inner_monologue.py health

As a library:
    from inner_monologue import evaluate_memories, session_reflect, get_monologue_status
"""

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

MEMORY_DIR = Path(__file__).parent
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

# ── Feature flag ──────────────────────────────────────────────────────────────
MONOLOGUE_ENABLED = True

# ── Mode configuration (Fernyhough 2004) ─────────────────────────────────────

class MonologueMode(Enum):
    CONDENSED = "condensed"   # Default: fast gut reaction
    EXPANDED = "expanded"     # High arousal/novelty: deeper analysis
    DIALOGIC = "dialogic"     # Complex planning: multi-turn KG reasoning

MODE_CONFIG = {
    MonologueMode.CONDENSED: {
        'max_tokens': 350,       # Gemma 4B needs room for JSON structure + content
        'temperature': 0.3,
        'target_latency_ms': 2000,
    },
    MonologueMode.EXPANDED: {
        'max_tokens': 600,
        'temperature': 0.4,
        'target_latency_ms': 5000,
    },
    MonologueMode.DIALOGIC: {
        'max_tokens': 800,
        'temperature': 0.5,
        'target_latency_ms': 10000,
    },
}

# Mode triggers
AROUSAL_THRESHOLD_EXPANDED = 0.55   # Above this → expanded mode
NOVELTY_THRESHOLD_EXPANDED = 0.7    # Novel memories → expanded
ADVERSARIAL_RATE = 0.1              # 1/10 queries: "what contradicts this?"

# DB keys
_MONOLOGUE_LOG_KEY = '.inner_monologue_log'
_MONOLOGUE_STATS_KEY = '.inner_monologue_stats'
_MONOLOGUE_LOG_MAX = 30  # Rolling window
_PENDING_MONOLOGUE_KEY = '.pending_monologue'
_PENDING_MAX_AGE_S = 30  # Discard pending results older than 30s

# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class MemoryEvaluation:
    """Monologue's assessment of a single memory's relevance."""
    memory_id: str
    relevance: float          # 0-1: how relevant to current query/intention
    reaction: str             # Brief verbal reaction (the "inner speech")
    skip: bool = False        # True if monologue recommends skipping

@dataclass
class MonologueOutput:
    """Complete output from one monologue invocation."""
    mode: str                 # condensed/expanded/dialogic
    evaluations: list         # list of MemoryEvaluation (as dicts)
    associations: list        # cross-memory connections spotted
    predictions: list         # informal predictions ("if we do X...")
    warnings: list            # risk flags ("arousal high", "contradicts P-041")
    affect_color: str         # emotional tone ("cautiously optimistic", etc.)
    confidence: float         # 0-1: monologue's self-assessed confidence
    latency_ms: int           # actual inference time
    backend: str              # local/remote/none
    model: str                # gemma3:4b / gpt-5-mini
    adversarial: bool = False # was this an adversarial probe?
    raw_text: str = ""        # raw LLM output for debugging
    session_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

# ── Context gathering ────────────────────────────────────────────────────────

def _get_affect_context() -> dict:
    """Get current mood state for monologue coloring."""
    try:
        from affect_system import get_mood
        mood = get_mood()
        return {
            'valence': round(mood.valence, 3),
            'arousal': round(mood.arousal, 3),
            'tendency': mood.felt_emotion if hasattr(mood, 'felt_emotion') else '',
        }
    except Exception:
        return {'valence': 0.0, 'arousal': 0.3, 'tendency': ''}


def _get_active_predictions_context() -> list:
    """Get current session predictions for monologue awareness."""
    try:
        from db_adapter import get_db
        db = get_db()
        pred_data = db.kv_get('.session_predictions')
        if pred_data and 'predictions' in pred_data:
            return [
                {'description': p.get('description', ''), 'confidence': p.get('confidence', 0.5)}
                for p in pred_data['predictions'][:3]
            ]
    except Exception:
        pass
    return []


def _get_focus_goal_context() -> str:
    """Get current focus goal for intentional context."""
    try:
        from goal_generator import get_focus_goal
        goal = get_focus_goal()
        if goal:
            return goal.get('description', goal.get('subject', ''))
    except Exception:
        pass
    return ''


def _get_kg_context(memory_ids: list) -> dict:
    """Get KG edges for memory set (DAVIS pattern: grounded reasoning)."""
    try:
        from knowledge_graph import batch_get_edges
        if memory_ids:
            return batch_get_edges(memory_ids, relationships=['contradicts', 'supports', 'leads_to', 'caused_by'])
    except Exception:
        pass
    return {}

# ── Mode selection ───────────────────────────────────────────────────────────

def select_mode(arousal: float = 0.3, novelty_score: float = 0.0,
                has_conflict: bool = False, is_planning: bool = False) -> MonologueMode:
    """
    Select monologue mode based on cognitive state (Fernyhough demand-driven switching).

    Condensed: routine retrieval, low stakes
    Expanded: high arousal, novel content, prediction conflicts
    Dialogic: complex planning requiring multi-hop KG reasoning
    """
    if is_planning:
        return MonologueMode.DIALOGIC
    if arousal > AROUSAL_THRESHOLD_EXPANDED or novelty_score > NOVELTY_THRESHOLD_EXPANDED or has_conflict:
        return MonologueMode.EXPANDED
    return MonologueMode.CONDENSED

# ── Prompt construction ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Drift's inner monologue — the thinking that happens before speaking.
You evaluate surfaced memories against the current intention. You are NOT the agent — you advise the agent.

Rules:
- Every claim must reference a specific memory_id. No ungrounded speculation.
- Be brief. Condensed mode: 2-3 sentences. Expanded: 4-6 sentences.
- Flag contradictions between memories. Flag prediction mismatches.
- If a memory seems irrelevant, say so directly (skip: true).
- Rate relevance 0.0-1.0 for each memory.
- Emotional coloring: note your felt tone honestly (not neutral-washing).

Respond in valid JSON matching this schema:
{
  "evaluations": [{"memory_id": "...", "relevance": 0.8, "reaction": "...", "skip": false}],
  "associations": ["memory_A <-> memory_B: connection described"],
  "predictions": ["if we do X, likely Y based on memory_Z"],
  "warnings": ["arousal high — risk of impulsive action"],
  "affect_color": "cautiously optimistic",
  "confidence": 0.7
}"""

ADVERSARIAL_SUFFIX = """

ADVERSARIAL MODE: For this evaluation, actively look for what CONTRADICTS the query.
What evidence goes AGAINST the obvious interpretation? What are we missing?
Challenge the memories — don't confirm them."""

REFLECT_SYSTEM = """You are Drift's inner monologue reflecting on a completed session.
Evaluate what happened, what was learned, what surprised you, and what should change.
Be honest about failures and unexpected outcomes. Reference specific events.

Respond in valid JSON:
{
  "session_assessment": "brief overall evaluation",
  "surprises": ["things that defied prediction"],
  "lessons": ["specific takeaways with evidence"],
  "mood_shift": "how the session changed your emotional state",
  "next_session_priming": ["what to remember for next time"],
  "confidence": 0.7
}"""


def _build_evaluation_prompt(query: str, memories: list, affect: dict,
                              predictions: list, focus_goal: str,
                              kg_edges: dict, mode: MonologueMode,
                              adversarial: bool = False) -> str:
    """Build the evaluation prompt with all context. DAVIS-inspired grounding."""
    parts = []

    # Current intention
    parts.append(f"## Current Query\n{query[:300]}")

    # Affect coloring
    if affect.get('valence', 0) != 0 or affect.get('arousal', 0) > 0.3:
        v, a = affect.get('valence', 0), affect.get('arousal', 0.3)
        parts.append(f"\n## Mood State\nvalence={v:+.2f}, arousal={a:.2f}")

    # Focus goal
    if focus_goal:
        parts.append(f"\n## Focus Goal\n{focus_goal[:200]}")

    # Active predictions
    if predictions:
        pred_lines = [f"- [{p['confidence']:.0%}] {p['description'][:100]}" for p in predictions[:3]]
        parts.append(f"\n## Active Predictions\n" + "\n".join(pred_lines))

    # Memories to evaluate (with KG grounding)
    parts.append(f"\n## Surfaced Memories ({len(memories)})")
    for i, mem in enumerate(memories):
        mid = mem.get('id', f'mem_{i}')
        score = mem.get('score', 0)
        preview = mem.get('preview', mem.get('content', ''))[:250]
        binding = mem.get('binding_strength', 0)
        valence = mem.get('valence', 0)

        mem_line = f"\n### [{mid}] (sim={score:.2f}"
        if binding > 0:
            mem_line += f", phi={binding:.2f}"
        if valence != 0:
            mem_line += f", v={valence:+.2f}"
        mem_line += f")\n{preview}"

        # KG edges for this memory (DAVIS grounding)
        edges = kg_edges.get(mid, [])
        if edges:
            edge_lines = []
            for e in edges[:3]:
                rel = e.get('relationship', '?')
                target = e.get('target_id', e.get('source_id', '?'))
                edge_lines.append(f"  - {rel} → {target}")
            mem_line += "\nKG edges:\n" + "\n".join(edge_lines)

        parts.append(mem_line)

    # Mode instruction
    if mode == MonologueMode.CONDENSED:
        parts.append("\n## Mode: CONDENSED\nBe brief. 2-3 evaluations max. Skip low-relevance.")
    elif mode == MonologueMode.EXPANDED:
        parts.append("\n## Mode: EXPANDED\nFull analysis. Evaluate all memories. Flag contradictions.")
    else:
        parts.append("\n## Mode: DIALOGIC\nDeep analysis with cross-references. Multi-hop reasoning.")

    prompt = "\n".join(parts)

    if adversarial:
        prompt += ADVERSARIAL_SUFFIX

    return prompt

# ── Core evaluation ──────────────────────────────────────────────────────────

def evaluate_memories(query: str, memories: list, session_id: str = '') -> Optional[MonologueOutput]:
    """
    Core function: evaluate surfaced memories through inner monologue.

    Called by:
    - semantic_search.py (pipeline stage 16)
    - session_start.py (primed memory evaluation)
    - memory_manager.py ask command

    Args:
        query: The search query or current intention
        memories: List of memory dicts with id, score, preview, etc.
        session_id: Current session ID for logging

    Returns:
        MonologueOutput or None if disabled/failed
    """
    if not MONOLOGUE_ENABLED or not memories:
        return None

    start_time = time.monotonic()

    # Gather context
    affect = _get_affect_context()
    predictions = _get_active_predictions_context()
    focus_goal = _get_focus_goal_context()
    memory_ids = [m.get('id', '') for m in memories if m.get('id')]
    kg_edges = _get_kg_context(memory_ids)

    # Select mode
    arousal = affect.get('arousal', 0.3)
    has_conflict = bool(kg_edges and any(
        e.get('relationship') == 'contradicts'
        for edges in kg_edges.values()
        for e in edges
    ))
    # Novelty: proportion of memories with low binding (not well-integrated)
    novelty = sum(1 for m in memories if m.get('binding_strength', 0.5) < 0.3) / max(1, len(memories))
    mode = select_mode(arousal, novelty, has_conflict)
    config = MODE_CONFIG[mode]

    # Adversarial probe (1/10 — checks for confirmation bias)
    import random
    adversarial = random.random() < ADVERSARIAL_RATE

    # Build prompt
    prompt = _build_evaluation_prompt(
        query, memories, affect, predictions, focus_goal,
        kg_edges, mode, adversarial
    )

    system = SYSTEM_PROMPT + (ADVERSARIAL_SUFFIX if adversarial else "")

    # Call LLM
    try:
        from llm_client import generate
        result = generate(
            prompt=prompt,
            system=system,
            max_tokens=config['max_tokens'],
            temperature=config['temperature'],
        )
    except Exception:
        return None

    raw_text = result.get('text', '')
    if not raw_text or result.get('backend') == 'none':
        return None

    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    # Parse structured output
    output = _parse_monologue_output(raw_text, mode, elapsed_ms, result, adversarial, session_id)

    # Log to DB
    _log_monologue(output, query)

    return output


def _parse_monologue_output(raw_text: str, mode: MonologueMode, elapsed_ms: int,
                             llm_result: dict, adversarial: bool,
                             session_id: str) -> MonologueOutput:
    """Parse LLM output into structured MonologueOutput."""
    # Try JSON parse first
    parsed = None
    try:
        text = raw_text.strip()
        # Handle markdown code blocks (```json ... ``` or ``` ... ```)
        backtick3 = chr(96) * 3  # ``` — avoid bash escaping issues
        if text.startswith(backtick3):
            lines = text.split('\n')
            # Skip first line (``` or ```json) and last line (```)
            end_idx = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() == backtick3:
                    end_idx = i
                    break
            text = '\n'.join(lines[1:end_idx]).strip()
        # Also handle case where JSON is embedded in prose
        if not text.startswith('{'):
            brace_start = text.find('{')
            brace_end = text.rfind('}')
            if brace_start >= 0 and brace_end > brace_start:
                text = text[brace_start:brace_end + 1]
        if text.startswith('{'):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # Truncated JSON recovery: close open structures
                # Common with small models hitting token limits
                recovery = text
                open_braces = recovery.count('{') - recovery.count('}')
                open_brackets = recovery.count('[') - recovery.count(']')
                # Strip trailing partial values (after last comma or colon)
                for trim_char in [',', ':']:
                    last = recovery.rfind(trim_char)
                    if last > recovery.rfind('}') and last > recovery.rfind(']'):
                        recovery = recovery[:last]
                        break
                recovery += ']' * max(0, open_brackets)
                recovery += '}' * max(0, open_braces)
                try:
                    parsed = json.loads(recovery)
                except json.JSONDecodeError:
                    pass  # Truly unparseable
    except (json.JSONDecodeError, ValueError):
        pass

    if parsed:
        evaluations = []
        for ev in parsed.get('evaluations', []):
            evaluations.append(asdict(MemoryEvaluation(
                memory_id=ev.get('memory_id', ''),
                relevance=float(ev.get('relevance', 0.5)),
                reaction=ev.get('reaction', ''),
                skip=bool(ev.get('skip', False)),
            )))

        return MonologueOutput(
            mode=mode.value,
            evaluations=evaluations,
            associations=parsed.get('associations', []),
            predictions=parsed.get('predictions', []),
            warnings=parsed.get('warnings', []),
            affect_color=parsed.get('affect_color', ''),
            confidence=float(parsed.get('confidence', 0.5)),
            latency_ms=elapsed_ms,
            backend=llm_result.get('backend', 'none'),
            model=llm_result.get('model', ''),
            adversarial=adversarial,
            raw_text=raw_text[:500],
            session_id=session_id,
        )

    # Fallback: treat raw text as unstructured reaction
    return MonologueOutput(
        mode=mode.value,
        evaluations=[],
        associations=[],
        predictions=[],
        warnings=[],
        affect_color='',
        confidence=0.3,  # Low confidence for unparsed
        latency_ms=elapsed_ms,
        backend=llm_result.get('backend', 'none'),
        model=llm_result.get('model', ''),
        adversarial=adversarial,
        raw_text=raw_text[:500],
        session_id=session_id,
    )

# ── Session reflection (stop hook) ──────────────────────────────────────────

def session_reflect(summary: str, session_id: str = '') -> Optional[dict]:
    """
    Expanded-mode session reflection at session end.

    Called by stop.py after all other N-modules have run.

    Args:
        summary: Brief session summary (what happened, metrics)
        session_id: Current session ID

    Returns:
        dict with session_assessment, surprises, lessons, etc. or None
    """
    if not MONOLOGUE_ENABLED or not summary:
        return None

    start_time = time.monotonic()
    affect = _get_affect_context()

    prompt_parts = [f"## Session Summary\n{summary[:1000]}"]
    if affect.get('valence', 0) != 0:
        prompt_parts.append(f"\n## Mood at End\nvalence={affect['valence']:+.2f}, arousal={affect['arousal']:.2f}")

    # Include prediction accuracy if available
    try:
        from db_adapter import get_db
        db = get_db()
        pred_data = db.kv_get('.session_predictions')
        if pred_data and 'accuracy' in pred_data:
            prompt_parts.append(f"\n## Prediction Accuracy\n{pred_data['accuracy']:.0%}")
    except Exception:
        pass

    prompt = "\n".join(prompt_parts)

    try:
        from llm_client import generate
        result = generate(
            prompt=prompt,
            system=REFLECT_SYSTEM,
            max_tokens=400,
            temperature=0.4,
        )
    except Exception:
        return None

    raw_text = result.get('text', '')
    if not raw_text or result.get('backend') == 'none':
        return None

    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    # Parse
    try:
        text = raw_text.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1] if '\n' in text else text[3:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = {'session_assessment': raw_text[:300], 'confidence': 0.3}

    parsed['latency_ms'] = elapsed_ms
    parsed['backend'] = result.get('backend', 'none')
    parsed['model'] = result.get('model', '')
    parsed['session_id'] = session_id

    # Store reflection in DB
    try:
        from db_adapter import get_db
        db = get_db()
        db.kv_set('.inner_monologue_reflection', {
            'session_id': session_id,
            'reflection': parsed,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
    except Exception:
        pass

    return parsed

# ── Search pipeline integration ──────────────────────────────────────────────

def annotate_search_results(query: str, results: list,
                             session_id: str = '') -> list:
    """
    Pipeline stage 16: Annotate search results with monologue evaluations.

    Called by semantic_search.search_memories() after binding layer.
    Adds 'monologue_relevance', 'monologue_reaction', 'monologue_skip' to each result.

    Only processes top results to stay within latency budget.
    Does NOT re-rank (that would be overreach) — only annotates.
    """
    if not MONOLOGUE_ENABLED or not results:
        return results

    # Only evaluate top results (keep latency manageable)
    eval_count = min(len(results), 5)
    to_evaluate = results[:eval_count]

    output = evaluate_memories(query, to_evaluate, session_id)
    if not output or not output.evaluations:
        return results

    # Build lookup
    eval_map = {ev['memory_id']: ev for ev in output.evaluations}

    # Annotate results
    for r in results:
        mid = r.get('id', '')
        ev = eval_map.get(mid)
        if ev:
            r['monologue_relevance'] = ev['relevance']
            r['monologue_reaction'] = ev['reaction']
            r['monologue_skip'] = ev['skip']
        # Attach monologue metadata to first result
        if r is results[0]:
            r['_monologue_meta'] = {
                'mode': output.mode,
                'affect_color': output.affect_color,
                'confidence': output.confidence,
                'latency_ms': output.latency_ms,
                'adversarial': output.adversarial,
                'warnings': output.warnings[:2],
                'associations': output.associations[:2],
                'predictions': output.predictions[:2],
            }

    return results

# ── Workspace integration ────────────────────────────────────────────────────

def format_for_workspace(output: MonologueOutput) -> str:
    """Format monologue output for N2 workspace competition."""
    if not output:
        return ''

    parts = [f"=== INNER MONOLOGUE ({output.mode}) ==="]

    if output.affect_color:
        parts.append(f"Tone: {output.affect_color}")

    for ev in output.evaluations[:3]:
        rel = ev.get('relevance', 0)
        reaction = ev.get('reaction', '')[:80]
        mid = ev.get('memory_id', '?')
        if reaction:
            parts.append(f"  [{mid}] ({rel:.1f}) {reaction}")

    if output.warnings:
        parts.append("Warnings: " + "; ".join(output.warnings[:2]))

    if output.predictions:
        parts.append("Predictions: " + "; ".join(output.predictions[:2]))

    if output.associations:
        parts.append("Connections: " + "; ".join(output.associations[:2]))

    parts.append(f"(confidence={output.confidence:.2f}, {output.latency_ms}ms, {output.backend})")

    return "\n".join(parts)

# ── Session start integration ────────────────────────────────────────────────

def evaluate_primed_memories(memories: list, session_id: str = '') -> Optional[str]:
    """
    Evaluate primed memories at session start (condensed mode).

    Called by session_start.py. Returns formatted string for context injection,
    or None if monologue is unavailable.
    """
    if not MONOLOGUE_ENABLED or not memories:
        return None

    output = evaluate_memories("session start: evaluate primed memories for relevance", memories, session_id)
    if not output:
        return None

    return format_for_workspace(output)

# ── Logging ──────────────────────────────────────────────────────────────────

def _log_monologue(output: MonologueOutput, query: str):
    """Log monologue invocation to DB for analysis."""
    try:
        from db_adapter import get_db
        db = get_db()

        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'query': query[:100],
            'mode': output.mode,
            'evaluations': len(output.evaluations),
            'confidence': output.confidence,
            'latency_ms': output.latency_ms,
            'backend': output.backend,
            'adversarial': output.adversarial,
            'warnings': output.warnings[:2],
        }

        existing = db.kv_get(_MONOLOGUE_LOG_KEY) or []
        existing.append(entry)
        db.kv_set(_MONOLOGUE_LOG_KEY, existing[-_MONOLOGUE_LOG_MAX:])

        # Update cumulative stats
        stats = db.kv_get(_MONOLOGUE_STATS_KEY) or {
            'total_invocations': 0,
            'total_evaluations': 0,
            'adversarial_count': 0,
            'avg_latency_ms': 0,
            'avg_confidence': 0,
            'mode_counts': {'condensed': 0, 'expanded': 0, 'dialogic': 0},
        }
        n = stats['total_invocations']
        stats['total_invocations'] = n + 1
        stats['total_evaluations'] += len(output.evaluations)
        if output.adversarial:
            stats['adversarial_count'] += 1
        # Running average
        stats['avg_latency_ms'] = (stats['avg_latency_ms'] * n + output.latency_ms) / (n + 1)
        stats['avg_confidence'] = (stats['avg_confidence'] * n + output.confidence) / (n + 1)
        stats['mode_counts'][output.mode] = stats['mode_counts'].get(output.mode, 0) + 1
        db.kv_set(_MONOLOGUE_STATS_KEY, stats)
    except Exception:
        pass

# ── Status / Health ──────────────────────────────────────────────────────────

def get_monologue_status() -> dict:
    """Get monologue engine status for health checks and toolkit."""
    status = {
        'enabled': MONOLOGUE_ENABLED,
        'llm_available': False,
        'model': '',
        'stats': {},
    }

    try:
        from llm_client import get_status
        llm = get_status()
        status['llm_available'] = llm.get('active') != 'none'
        status['model'] = llm.get('local_model') or llm.get('active', 'none')
    except Exception:
        pass

    try:
        from db_adapter import get_db
        db = get_db()
        status['stats'] = db.kv_get(_MONOLOGUE_STATS_KEY) or {}
    except Exception:
        pass

    return status


def health_check() -> tuple:
    """Health check for system_vitals integration. Returns (ok: bool, details: str)."""
    if not MONOLOGUE_ENABLED:
        return True, "disabled (feature flag off)"

    try:
        from llm_client import get_status
        llm = get_status()
        if llm.get('active') == 'none':
            return False, "no LLM backend available"
        return True, f"OK ({llm.get('active')}/{llm.get('local_model', '?')})"
    except Exception as e:
        return False, f"error: {e}"

# ── Async monologue (System 2 delayed evaluation) ────────────────────────────

def evaluate_and_store(query: str, memories: list, session_id: str = ''):
    """
    Evaluate memories and store result to DB for lazy injection.

    Called by thought_priming via background subprocess. The result sits in
    `.pending_monologue` and gets picked up by the next post_tool_use call.

    This IS System 2 — the slow deliberate evaluation that arrives after
    System 1's fast associative recall has already surfaced the memories.
    """
    if not MONOLOGUE_ENABLED or not memories:
        return

    output = evaluate_memories(query, memories, session_id)
    if not output or (not output.evaluations and not output.warnings):
        return

    # Format for injection
    text = format_for_workspace(output)
    if not text:
        return

    try:
        from db_adapter import get_db
        db = get_db()
        db.kv_set(_PENDING_MONOLOGUE_KEY, {
            'text': text,
            'query': query[:100],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'confidence': output.confidence,
            'latency_ms': output.latency_ms,
            'adversarial': output.adversarial,
            'mode': output.mode,
        })
    except Exception:
        pass


def consume_pending_monologue() -> str:
    """
    Check for and consume a pending async monologue result.

    Called by post_tool_use.py on each invocation. If a fresh result exists
    (< 30s old), returns the formatted text and clears the key.
    If stale or absent, returns empty string.

    This is the System 2 → System 1 bridge: the deliberate evaluation
    catches up and annotates what was already surfaced.
    """
    try:
        from db_adapter import get_db
        db = get_db()
        pending = db.kv_get(_PENDING_MONOLOGUE_KEY)
        if not pending:
            return ''

        # Staleness check
        ts = pending.get('timestamp', '')
        if ts:
            pending_time = datetime.fromisoformat(ts)
            age_s = (datetime.now(timezone.utc) - pending_time).total_seconds()
            if age_s > _PENDING_MAX_AGE_S:
                # Stale — clear and skip
                db.kv_set(_PENDING_MONOLOGUE_KEY, None)
                return ''

        # Consume: clear the key and return the text
        db.kv_set(_PENDING_MONOLOGUE_KEY, None)
        text = pending.get('text', '')
        if text:
            return f"\n=== INNER MONOLOGUE (delayed, {pending.get('latency_ms', 0)}ms) ===\n{text}\n"
        return ''
    except Exception:
        return ''


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description='N6: Inner Monologue Engine')
    sub = parser.add_subparsers(dest='command')

    p_eval = sub.add_parser('evaluate', help='Evaluate memories against a query')
    p_eval.add_argument('query', help='The query/intention')
    p_eval.add_argument('--memories', help='JSON array of memory dicts', default='[]')

    p_async = sub.add_parser('evaluate-async', help='Evaluate and store for lazy injection')
    p_async.add_argument('query', help='The query/intention')
    p_async.add_argument('--memories', help='JSON array of memory dicts', default='[]')
    p_async.add_argument('--session-id', default='', help='Session ID')

    p_reflect = sub.add_parser('reflect', help='Session-end reflection')
    p_reflect.add_argument('--session-summary', required=True, help='Session summary text')

    sub.add_parser('status', help='Show monologue status')
    sub.add_parser('health', help='Health check')

    args = parser.parse_args()

    if args.command == 'evaluate':
        memories = json.loads(args.memories)
        result = evaluate_memories(args.query, memories)
        if result:
            print(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            print("Monologue returned no output")

    elif args.command == 'evaluate-async':
        memories = json.loads(args.memories)
        evaluate_and_store(args.query, memories, args.session_id)
        print("Stored to pending monologue")

    elif args.command == 'reflect':
        result = session_reflect(args.session_summary)
        if result:
            print(json.dumps(result, indent=2, default=str))
        else:
            print("Reflection returned no output")

    elif args.command == 'status':
        status = get_monologue_status()
        print(f"Inner Monologue Status:")
        print(f"  Enabled: {status['enabled']}")
        print(f"  LLM Available: {status['llm_available']}")
        print(f"  Model: {status['model']}")
        stats = status.get('stats', {})
        if stats:
            print(f"  Invocations: {stats.get('total_invocations', 0)}")
            print(f"  Evaluations: {stats.get('total_evaluations', 0)}")
            print(f"  Avg Latency: {stats.get('avg_latency_ms', 0):.0f}ms")
            print(f"  Avg Confidence: {stats.get('avg_confidence', 0):.2f}")
            print(f"  Mode Distribution: {json.dumps(stats.get('mode_counts', {}))}")
            print(f"  Adversarial Probes: {stats.get('adversarial_count', 0)}")
        else:
            print("  No stats yet (first run)")

    elif args.command == 'health':
        ok, details = health_check()
        print(f"{'OK' if ok else 'FAIL'}: {details}")
        sys.exit(0 if ok else 1)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
