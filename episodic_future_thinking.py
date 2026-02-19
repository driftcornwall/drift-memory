"""
T4.2: Episodic Future Thinking — Prospective Memory via Constructive Simulation

Biological basis: Schacter & Addis (2007) — the hippocampus uses the same
machinery for episodic recall and future simulation. We recombine recent
episodic memories with an active goal to construct plausible future scenarios,
store them as prospective memories with trigger conditions, and evaluate
predictions against reality at session end.

Lifecycle:
  session_start → generate_prospective_memories() → active prospective memories
  post_tool_use → check_triggers(context) → inject triggered scenarios
  stop          → evaluate_prospective(session_data) → prediction error → learning

DB KV keys:
  .prospective_memories   — list of active + recently evaluated ProspectiveMemory dicts
  .prospective_history    — rolling 20 evaluated entries
  .prospective_stats      — aggregate stats {generated, triggered, confirmed, ...}
"""

import hashlib
import json
import sys
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Feature flag
EFT_ENABLED = True

# Constants
MAX_ACTIVE = 3            # Max concurrent prospective memories
MAX_PER_SESSION = 2       # Max generated per session start
EXPIRY_SESSIONS = 3       # Expire untriggered after N sessions
SCENARIO_MIN_LEN = 50     # Reject scenarios shorter than this
SCENARIO_MAX_LEN = 800    # Reject scenarios longer than this
JACCARD_DUP_THRESHOLD = 0.7  # Reject if overlap with existing

# DB KV keys
KV_PROSPECTIVE = '.prospective_memories'
KV_HISTORY = '.prospective_history'
KV_STATS = '.prospective_stats'

# Known platforms for heuristic extraction
KNOWN_PLATFORMS = [
    'moltx', 'moltbook', 'colony', 'thecolony', 'clawbr', 'clawtasks',
    'lobsterpedia', 'github', 'twitter', 'agentlink', 'agenthub',
    'dead-internet', 'nostr', 'ugig',
]

# ============================================================
# Data structures
# ============================================================

@dataclass
class ProspectiveMemory:
    eft_id: str
    scenario: str
    trigger_condition: str
    trigger_type: str              # 'platform', 'contact', 'topic', 'time'
    source_memories: list = field(default_factory=list)
    goal_id: Optional[str] = None
    confidence: float = 0.5
    status: str = 'active'         # 'active', 'triggered', 'expired', 'evaluated'
    created_at: str = ''
    triggered_at: Optional[str] = None
    outcome: Optional[str] = None
    prediction_error: Optional[float] = None
    sessions_active: int = 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_id(scenario: str) -> str:
    h = hashlib.sha256(scenario.encode()).hexdigest()[:8]
    return f'eft-{h}'


def _get_db():
    from db_adapter import get_db
    return get_db()


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _tokenize(text: str) -> set:
    """Simple word tokenizer for Jaccard comparison."""
    return {w.lower() for w in text.split() if len(w) > 2 and w.isalpha()}


# ============================================================
# Persistence
# ============================================================

def _load_prospective() -> list[dict]:
    db = _get_db()
    data = db.kv_get(KV_PROSPECTIVE)
    if isinstance(data, list):
        return data
    return []


def _save_prospective(memories: list[dict]):
    db = _get_db()
    db.kv_set(KV_PROSPECTIVE, memories)


def _load_stats() -> dict:
    db = _get_db()
    return db.kv_get(KV_STATS) or {
        'generated': 0, 'triggered': 0, 'confirmed': 0,
        'violated': 0, 'expired': 0, 'avg_error': 0.0,
    }


def _save_stats(stats: dict):
    db = _get_db()
    db.kv_set(KV_STATS, stats)


def _append_history(entry: dict):
    db = _get_db()
    history = db.kv_get(KV_HISTORY) or []
    history.append(entry)
    history = history[-20:]  # Rolling 20
    db.kv_set(KV_HISTORY, history)


# ============================================================
# Generation
# ============================================================

def generate_prospective_memories(max_count: int = MAX_PER_SESSION) -> list[ProspectiveMemory]:
    """
    Generate prospective memories by recombining episodic summaries with active goals.
    Called at session start.
    """
    if not EFT_ENABLED:
        return []

    # Load existing — check capacity
    existing = _load_prospective()
    active = [m for m in existing if m.get('status') == 'active']
    if len(active) >= MAX_ACTIVE:
        return []

    slots = min(max_count, MAX_ACTIVE - len(active))
    if slots <= 0:
        return []

    # Gather inputs
    episodic_summaries = _get_episodic_summaries()
    focus_goal = _get_focus_goal()
    pending_intentions = _get_pending_intentions()

    if not episodic_summaries:
        return []

    # Try LLM first, heuristic fallback
    new_memories = []
    try:
        new_memories = _generate_llm(episodic_summaries, focus_goal, pending_intentions, slots)
    except Exception:
        pass

    if not new_memories:
        new_memories = _generate_heuristic(episodic_summaries, focus_goal, slots)

    # Quality gate + dedup
    accepted = []
    active_tokens = [_tokenize(m.get('scenario', '')) for m in active]
    for pm in new_memories:
        if not _quality_gate(pm, active_tokens):
            continue
        accepted.append(pm)
        active_tokens.append(_tokenize(pm.scenario))
        if len(accepted) >= slots:
            break

    # Persist
    if accepted:
        for pm in accepted:
            existing.append(asdict(pm))
        _save_prospective(existing)

        stats = _load_stats()
        stats['generated'] = stats.get('generated', 0) + len(accepted)
        _save_stats(stats)

    return accepted


def _get_episodic_summaries() -> list[dict]:
    try:
        from episodic_db import load_recent_summaries
        return load_recent_summaries(3)
    except Exception:
        return []


def _get_focus_goal() -> dict | None:
    try:
        from goal_generator import get_focus_goal
        return get_focus_goal()
    except Exception:
        return None


def _get_pending_intentions() -> list[dict]:
    try:
        from temporal_intentions import list_intentions
        return list_intentions(status='pending')
    except Exception:
        return []


def _generate_llm(summaries: list[dict], goal: dict | None,
                  intentions: list[dict], count: int) -> list[ProspectiveMemory]:
    """Use local LLM (Gemma/Ollama) to construct future scenarios."""
    from llm_client import generate

    # Build prompt
    summary_parts = []
    source_ids = []
    for i, s in enumerate(summaries, 1):
        content = (s.get('content') or '')[:250]
        platforms = s.get('platforms_active', [])
        contacts = s.get('contacts_active', [])
        summary_parts.append(
            f"Session {i} ({s.get('date', '?')}): {content}\n"
            f"  Platforms: {', '.join(platforms[:5]) if platforms else 'unknown'}\n"
            f"  Contacts: {', '.join(contacts[:5]) if contacts else 'none'}"
        )
        if s.get('id'):
            source_ids.append(s['id'])

    goal_text = ""
    goal_id = None
    if goal:
        goal_text = f"\nActive goal: {goal.get('action', '?')} (type: {goal.get('goal_type', '?')})"
        goal_id = goal.get('goal_id')

    intention_text = ""
    if intentions:
        items = [f"  - {it.get('action', '?')}" for it in intentions[:3]]
        intention_text = f"\nPending intentions:\n" + '\n'.join(items)

    system_prompt = (
        "You construct plausible future scenarios for an AI agent based on recent session history. "
        "Given session summaries and an active goal, predict what might happen next session. "
        "Be specific and concrete — name platforms, contacts, or topics. "
        f"Return a JSON array of {count} objects, each with: "
        '"scenario" (100-200 words), "trigger_condition" (specific keyword or event), '
        '"trigger_type" (one of: platform, contact, topic), "confidence" (0.0-1.0). '
        "Only return the JSON array, no other text."
    )

    prompt = '\n\n'.join(summary_parts) + goal_text + intention_text

    result = generate(prompt, system=system_prompt, max_tokens=500, temperature=0.5)
    text = result.get('text', '').strip()

    if not text:
        return []

    # Parse JSON — handle both array and single object
    try:
        # Strip markdown fences if present
        if text.startswith('```'):
            text = text.split('\n', 1)[1] if '\n' in text else text[3:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()

        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return []
    except json.JSONDecodeError:
        # Try to recover truncated JSON
        try:
            # Close open brackets
            fixed = text
            open_brackets = fixed.count('[') - fixed.count(']')
            open_braces = fixed.count('{') - fixed.count('}')
            fixed += '}' * max(0, open_braces) + ']' * max(0, open_brackets)
            parsed = json.loads(fixed)
            if isinstance(parsed, dict):
                parsed = [parsed]
        except Exception:
            return []

    memories = []
    for item in parsed[:count]:
        scenario = item.get('scenario', '')
        trigger = item.get('trigger_condition', '')
        ttype = item.get('trigger_type', 'topic')
        conf = min(1.0, max(0.0, float(item.get('confidence', 0.5))))

        if not scenario or not trigger:
            continue

        if ttype not in ('platform', 'contact', 'topic', 'time'):
            ttype = 'topic'

        pm = ProspectiveMemory(
            eft_id=_make_id(scenario),
            scenario=scenario,
            trigger_condition=trigger.lower(),
            trigger_type=ttype,
            source_memories=source_ids,
            goal_id=goal_id,
            confidence=conf,
            status='active',
            created_at=_now_iso(),
        )
        memories.append(pm)

    return memories


def _generate_heuristic(summaries: list[dict], goal: dict | None,
                        count: int) -> list[ProspectiveMemory]:
    """Template-based fallback when LLM unavailable."""
    memories = []
    source_ids = [s.get('id', '') for s in summaries if s.get('id')]

    # Extract platforms and contacts from summaries
    all_platforms = []
    all_contacts = []
    for s in summaries:
        all_platforms.extend(s.get('platforms_active', []))
        all_contacts.extend(s.get('contacts_active', []))

    # Count frequency
    from collections import Counter
    platform_freq = Counter(all_platforms)
    contact_freq = Counter(all_contacts)

    # Generate platform-based prediction
    if platform_freq and len(memories) < count:
        top_platform = platform_freq.most_common(1)[0][0]
        goal_text = f" Goal '{goal.get('action', '')}' may drive activity there." if goal else ""
        scenario = (
            f"Based on recent sessions, expect continued activity on {top_platform}. "
            f"This platform appeared in {platform_freq[top_platform]}/{len(summaries)} recent sessions.{goal_text} "
            f"Look for new posts, replies, or engagement opportunities."
        )
        pm = ProspectiveMemory(
            eft_id=_make_id(scenario),
            scenario=scenario,
            trigger_condition=top_platform.lower(),
            trigger_type='platform',
            source_memories=source_ids,
            goal_id=goal.get('goal_id') if goal else None,
            confidence=0.4,
            status='active',
            created_at=_now_iso(),
        )
        memories.append(pm)

    # Generate contact-based prediction
    if contact_freq and len(memories) < count:
        top_contact = contact_freq.most_common(1)[0][0]
        scenario = (
            f"Recent sessions involved interaction with {top_contact} across "
            f"{contact_freq[top_contact]} sessions. Expect continued engagement — "
            f"check for replies, new posts, or collaboration opportunities from {top_contact}."
        )
        pm = ProspectiveMemory(
            eft_id=_make_id(scenario),
            scenario=scenario,
            trigger_condition=top_contact.lower(),
            trigger_type='contact',
            source_memories=source_ids,
            goal_id=None,
            confidence=0.35,
            status='active',
            created_at=_now_iso(),
        )
        memories.append(pm)

    return memories


def _quality_gate(pm: ProspectiveMemory, existing_tokens: list[set]) -> bool:
    """Reject low-quality or duplicate prospective memories."""
    # Length check
    if len(pm.scenario) < SCENARIO_MIN_LEN or len(pm.scenario) > SCENARIO_MAX_LEN:
        return False

    # Trigger check
    if not pm.trigger_condition or pm.trigger_condition in ('something', 'anything', 'event'):
        return False

    # Dedup check
    pm_tokens = _tokenize(pm.scenario)
    for existing in existing_tokens:
        if _jaccard(pm_tokens, existing) > JACCARD_DUP_THRESHOLD:
            return False

    return True


# ============================================================
# Trigger detection (mid-session)
# ============================================================

def check_triggers(context: str) -> list[dict]:
    """
    Check if any active prospective memories should trigger.
    Called from post_tool_use hook with tool result text.
    """
    if not EFT_ENABLED:
        return []

    all_memories = _load_prospective()
    active = [m for m in all_memories if m.get('status') == 'active']
    if not active:
        return []

    context_lower = context.lower()
    triggered = []

    for mem in active:
        trigger = mem.get('trigger_condition', '').lower()
        ttype = mem.get('trigger_type', 'topic')

        fired = False
        if ttype == 'platform':
            fired = trigger in context_lower
        elif ttype == 'contact':
            fired = trigger in context_lower
        elif ttype == 'topic':
            trigger_words = {w for w in trigger.split() if len(w) > 2}
            context_words = {w for w in context_lower.split() if len(w) > 2}
            overlap = trigger_words & context_words
            fired = len(overlap) >= min(2, len(trigger_words))
        # 'time' type not implemented yet

        if fired:
            mem['status'] = 'triggered'
            mem['triggered_at'] = _now_iso()
            triggered.append(mem)

    if triggered:
        _save_prospective(all_memories)
        stats = _load_stats()
        stats['triggered'] = stats.get('triggered', 0) + len(triggered)
        _save_stats(stats)

    return triggered


def format_triggered(triggered: list[dict]) -> str:
    """Format triggered prospective memories for context injection."""
    if not triggered:
        return ""

    parts = ["=== PROSPECTIVE MEMORY TRIGGERED (T4.2) ==="]
    for mem in triggered:
        parts.append(
            f"  You predicted: {mem.get('scenario', '?')[:200]}\n"
            f"  Trigger: {mem.get('trigger_condition', '?')} ({mem.get('trigger_type', '?')})\n"
            f"  Confidence: {mem.get('confidence', 0):.0%}\n"
            f"  Compare what's actually happening against this expectation."
        )
    return '\n'.join(parts)


def format_prospective_context(memories: list) -> str:
    """Format active prospective memories for session start context."""
    if not memories:
        return ""

    parts = ["=== EPISODIC FUTURE THINKING (T4.2) ==="]
    for pm in memories:
        if isinstance(pm, ProspectiveMemory):
            pm = asdict(pm)
        parts.append(
            f"  [{pm.get('trigger_type', '?')}] {pm.get('scenario', '?')[:150]}\n"
            f"    Trigger: {pm.get('trigger_condition', '?')} | "
            f"Confidence: {pm.get('confidence', 0):.0%}"
        )
    return '\n'.join(parts)


# ============================================================
# Evaluation (session end)
# ============================================================

def evaluate_prospective(session_data: dict) -> dict:
    """
    Evaluate prospective memories against actual session activity.
    Called from stop.py at session end.
    """
    all_memories = _load_prospective()
    stats = _load_stats()

    result = {'evaluated': 0, 'confirmed': 0, 'violated': 0, 'expired': 0}

    # Build actual activity set for comparison
    actual_recalls = session_data.get('recalls', [])
    actual_platforms = [p.lower() for p in session_data.get('platforms', [])]
    actual_contacts = [c.lower() for c in session_data.get('contacts', [])]
    actual_text = ' '.join(actual_platforms + actual_contacts + actual_recalls).lower()

    for mem in all_memories:
        if mem.get('status') == 'triggered':
            # Evaluate: compare scenario keywords against actuals
            scenario_tokens = _tokenize(mem.get('scenario', ''))
            actual_tokens = _tokenize(actual_text)
            overlap = _jaccard(scenario_tokens, actual_tokens)
            prediction_error = 1.0 - overlap

            mem['prediction_error'] = round(prediction_error, 3)
            mem['status'] = 'evaluated'
            mem['outcome'] = f"overlap={overlap:.2f}, platforms={actual_platforms[:3]}, contacts={actual_contacts[:3]}"

            result['evaluated'] += 1

            if prediction_error < 0.3:
                result['confirmed'] += 1
                stats['confirmed'] = stats.get('confirmed', 0) + 1
                _fire_cognitive_event('prospective_confirmed', prediction_error)
            elif prediction_error > 0.7:
                result['violated'] += 1
                stats['violated'] = stats.get('violated', 0) + 1
                _fire_cognitive_event('prospective_violated', prediction_error)

            # Append to history
            _append_history({
                'eft_id': mem.get('eft_id'),
                'scenario': mem.get('scenario', '')[:100],
                'prediction_error': prediction_error,
                'trigger_type': mem.get('trigger_type'),
                'evaluated_at': _now_iso(),
            })

        elif mem.get('status') == 'active':
            # Age out untriggered memories
            mem['sessions_active'] = mem.get('sessions_active', 0) + 1
            if mem['sessions_active'] >= EXPIRY_SESSIONS:
                mem['status'] = 'expired'
                result['expired'] += 1
                stats['expired'] = stats.get('expired', 0) + 1

    # Update avg_error
    history = _get_db().kv_get(KV_HISTORY) or []
    errors = [h.get('prediction_error', 0.5) for h in history if h.get('prediction_error') is not None]
    if errors:
        stats['avg_error'] = round(sum(errors) / len(errors), 3)

    # Clean up: remove old evaluated/expired entries (keep last 5 for reference)
    evaluated_expired = [m for m in all_memories if m.get('status') in ('evaluated', 'expired')]
    active_triggered = [m for m in all_memories if m.get('status') in ('active', 'triggered')]
    all_memories = active_triggered + evaluated_expired[-5:]

    _save_prospective(all_memories)
    _save_stats(stats)

    return result


def _fire_cognitive_event(event_name: str, prediction_error: float):
    """Fire cognitive state event for prediction outcome."""
    try:
        from cognitive_state import process_event
        if event_name == 'prospective_confirmed':
            process_event(event_name)
        elif event_name == 'prospective_violated':
            process_event(event_name)
    except Exception:
        pass

    # Also update affect
    try:
        from affect_system import process_affect_event
        if event_name == 'prospective_confirmed':
            process_affect_event(event_name, valence=0.1, arousal=-0.05)
        elif event_name == 'prospective_violated':
            process_affect_event(event_name, valence=-0.05, arousal=0.15)
    except Exception:
        pass


# ============================================================
# CLI
# ============================================================

def _print_status():
    """Show active prospective memories and stats."""
    all_mem = _load_prospective()
    stats = _load_stats()

    active = [m for m in all_mem if m.get('status') == 'active']
    triggered = [m for m in all_mem if m.get('status') == 'triggered']
    evaluated = [m for m in all_mem if m.get('status') == 'evaluated']

    print("=== Episodic Future Thinking (T4.2) ===")
    print(f"  Active: {len(active)}/{MAX_ACTIVE}")
    print(f"  Triggered: {len(triggered)}")
    print(f"  Recently evaluated: {len(evaluated)}")
    print(f"\n  Stats: generated={stats.get('generated', 0)}, "
          f"triggered={stats.get('triggered', 0)}, "
          f"confirmed={stats.get('confirmed', 0)}, "
          f"violated={stats.get('violated', 0)}, "
          f"expired={stats.get('expired', 0)}, "
          f"avg_error={stats.get('avg_error', 0):.3f}")

    if active:
        print("\n  Active prospective memories:")
        for m in active:
            print(f"    [{m.get('eft_id')}] ({m.get('trigger_type')}) "
                  f"trigger='{m.get('trigger_condition')}' "
                  f"conf={m.get('confidence', 0):.0%} "
                  f"age={m.get('sessions_active', 0)} sessions")
            print(f"      {m.get('scenario', '')[:120]}...")


def _print_history():
    """Show evaluation history."""
    db = _get_db()
    history = db.kv_get(KV_HISTORY) or []

    print("=== Prospective Memory Evaluation History ===")
    if not history:
        print("  No evaluations yet.")
        return

    for h in history[-10:]:
        error = h.get('prediction_error', '?')
        status = 'CONFIRMED' if isinstance(error, float) and error < 0.3 else \
                 'VIOLATED' if isinstance(error, float) and error > 0.7 else 'PARTIAL'
        print(f"  [{h.get('eft_id')}] {status} error={error} "
              f"type={h.get('trigger_type')} at={h.get('evaluated_at', '?')[:16]}")
        print(f"    {h.get('scenario', '')[:100]}")


def _run_test():
    """Generate prospective memories from existing episodic DB and show results."""
    print("=== EFT Test: Generating prospective memories ===")
    memories = generate_prospective_memories(max_count=2)
    if not memories:
        print("  No prospective memories generated (check episodic DB and LLM status)")
        return

    for pm in memories:
        print(f"\n  [{pm.eft_id}]")
        print(f"    Type: {pm.trigger_type}")
        print(f"    Trigger: {pm.trigger_condition}")
        print(f"    Confidence: {pm.confidence:.0%}")
        print(f"    Scenario: {pm.scenario[:200]}")
        print(f"    Sources: {pm.source_memories}")
        print(f"    Goal: {pm.goal_id}")


def _check_trigger_test(context: str):
    """Test trigger detection against a context string."""
    print(f"=== EFT Trigger Test: '{context[:80]}' ===")
    triggered = check_triggers(context)
    if triggered:
        for t in triggered:
            print(f"  FIRED: [{t.get('eft_id')}] trigger='{t.get('trigger_condition')}' "
                  f"type={t.get('trigger_type')}")
            print(f"    Scenario: {t.get('scenario', '')[:150]}")
    else:
        print("  No triggers fired.")
        active = [m for m in _load_prospective() if m.get('status') == 'active']
        if active:
            print(f"  ({len(active)} active prospective memories with triggers: "
                  f"{[m.get('trigger_condition') for m in active]})")
        else:
            print("  (no active prospective memories)")


def _run_health():
    """Health check."""
    checks = 0
    passed = 0

    # Check DB access
    checks += 1
    try:
        _get_db()
        passed += 1
    except Exception:
        print("  FAIL: DB access")

    # Check episodic_db import
    checks += 1
    try:
        from episodic_db import load_recent_summaries
        passed += 1
    except Exception:
        print("  FAIL: episodic_db import")

    # Check llm_client import
    checks += 1
    try:
        from llm_client import generate
        passed += 1
    except Exception:
        print("  FAIL: llm_client import")

    # Check goal_generator import
    checks += 1
    try:
        from goal_generator import get_focus_goal
        passed += 1
    except Exception:
        print("  FAIL: goal_generator import")

    # Check temporal_intentions import
    checks += 1
    try:
        from temporal_intentions import list_intentions
        passed += 1
    except Exception:
        print("  FAIL: temporal_intentions import")

    print(f"EFT health: {passed}/{checks} checks passed")
    return passed == checks


def main():
    import argparse
    parser = argparse.ArgumentParser(description='T4.2: Episodic Future Thinking')
    parser.add_argument('command', choices=['test', 'status', 'history', 'check-trigger', 'health'],
                        help='Command to run')
    parser.add_argument('context', nargs='?', default='', help='Context for check-trigger')
    args = parser.parse_args()

    # Ensure memory dir on path
    mem_dir = str(Path(__file__).parent)
    if mem_dir not in sys.path:
        sys.path.insert(0, mem_dir)

    if args.command == 'test':
        _run_test()
    elif args.command == 'status':
        _print_status()
    elif args.command == 'history':
        _print_history()
    elif args.command == 'check-trigger':
        if not args.context:
            print("Usage: episodic_future_thinking.py check-trigger 'context text'")
            return
        _check_trigger_test(args.context)
    elif args.command == 'health':
        _run_health()


if __name__ == '__main__':
    main()
