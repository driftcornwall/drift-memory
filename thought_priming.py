#!/usr/bin/env python3
"""
Thought Priming Module - Real-time associative memory for thinking blocks.

THE API EATER EXPERIMENT (2026-02-02)

When enabled, intercepts thinking blocks during tool calls and runs semantic
search to surface relevant memories. Like biological associative recall -
one thought triggers related memories before the next thought forms.

Usage:
    from thought_priming import prime_from_thought

    # In post_tool_use.py:
    memories = prime_from_thought(transcript_path)
    if memories:
        print(memories)  # Becomes system-reminder

Configuration:
    Set THOUGHT_PRIMING_ENABLED=true in environment or config file.
    Default: disabled (to control costs during experiment)

Author: Drift
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from db_adapter import get_db

# =============================================================================
# CONFIGURATION
# =============================================================================

# Master switch - can be toggled via environment variable
ENABLED = os.environ.get('THOUGHT_PRIMING_ENABLED', 'false').lower() == 'true'

# DB keys for config and state
_CONFIG_KEY = '.thought_priming_config'
_STATE_KEY = '.thought_priming_state'

# Cooldown: only fire every Nth tool call (reduces 200 searches to ~40)
CALL_COOLDOWN = 5

def is_enabled() -> bool:
    """Check if thought priming is enabled."""
    # Environment variable takes precedence
    env_val = os.environ.get('THOUGHT_PRIMING_ENABLED')
    if env_val is not None:
        return env_val.lower() == 'true'

    # Check DB config — crash loud on DB failure so we notice infra problems
    db = get_db()
    config = db.kv_get(_CONFIG_KEY)
    if config is not None:
        return config.get('enabled', False)

    return False


def enable():
    """Enable thought priming."""
    db = get_db()
    db.kv_set(_CONFIG_KEY, {'enabled': True})
    print("Thought priming ENABLED")


def disable():
    """Disable thought priming."""
    db = get_db()
    db.kv_set(_CONFIG_KEY, {'enabled': False})
    print("Thought priming DISABLED")


def status():
    """Show current status."""
    enabled = is_enabled()
    print(f"Thought priming: {'ENABLED' if enabled else 'DISABLED'}")
    print(f"Config store: PostgreSQL key '{_CONFIG_KEY}'")


# =============================================================================
# STATE MANAGEMENT (cross-call persistence via PostgreSQL)
# =============================================================================

def _load_state(transcript_path: str) -> dict:
    """Load state, reset on new session (different transcript path)."""
    default = {"last_hash": "", "returned_ids": [], "call_count": 0, "transcript": ""}
    try:
        db = get_db()
        state = db.kv_get(_STATE_KEY)
        if state is not None and state.get("transcript") == transcript_path:
            return state
    except Exception:
        pass
    default["transcript"] = transcript_path
    return default


def _save_state(state: dict):
    """Persist state between calls.

    Wrapped in try/except because state save should never block the
    conversation — a lost state update just means one redundant search
    or a re-shown memory, which is harmless.
    """
    try:
        db = get_db()
        db.kv_set(_STATE_KEY, state)
    except Exception:
        pass


# =============================================================================
# TRANSCRIPT PARSING
# =============================================================================

def extract_last_thinking_block(transcript_path: str) -> Optional[str]:
    """
    Extract the most recent thinking block from the transcript.

    Transcript is JSONL format. We look for assistant messages with
    thinking content blocks.
    """
    if not transcript_path or not Path(transcript_path).exists():
        return None

    try:
        # Read lines in reverse — the last thinking block is near the end,
        # no need to scan the entire transcript every time
        lines = Path(transcript_path).read_text(encoding='utf-8').splitlines()

        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = entry.get('message', {})
            if msg.get('role') != 'assistant':
                continue

            content = msg.get('content', [])
            if not isinstance(content, list):
                continue

            # Walk content blocks in reverse to find the last thinking block
            for block in reversed(content):
                if isinstance(block, dict) and block.get('type') == 'thinking':
                    thinking_text = block.get('thinking', '')
                    if thinking_text and len(thinking_text) > 50:
                        return thinking_text

        return None

    except Exception:
        return None


# =============================================================================
# SEMANTIC SEARCH
# =============================================================================

def search_memories(query: str, memory_dir: Path, limit: int = 2) -> list[dict]:
    """
    Run semantic search against memories via subprocess.
    Uses pgvector-backed semantic_search.py directly (no dead search server).

    Returns list of {id, score, preview} dicts.
    """
    try:
        # Take LAST 500 chars — the tail has specific reasoning,
        # the head is always generic session context ("Let me check...")
        query = query[-500:].strip()
        if not query:
            return []

        # Remove common thinking patterns that aren't useful for search
        query = re.sub(r'Let me (think|check|see|look|read|try)', '', query)
        query = re.sub(r'I (need to|should|will|can)', '', query)
        query = query.strip()

        if len(query) < 20:
            return []

        # Direct subprocess to semantic_search.py (pgvector-backed)
        # --skip-monologue: System 1 fast path. Async monologue fires separately.
        result = subprocess.run(
            ["python", str(memory_dir / "semantic_search.py"), "search", query,
             "--limit", str(limit), "--skip-monologue"],
            capture_output=True,
            text=True,
            timeout=8,
            cwd=str(memory_dir)
        )

        if not result.stdout or "Found" not in result.stdout:
            return []

        memories = []
        current = None

        for line in result.stdout.split('\n'):
            if line.startswith('[') and ']' in line:
                if current:
                    memories.append(current)
                try:
                    score = float(line[1:line.index(']')])
                    mem_id = line[line.index(']')+1:].strip().split()[0]  # Strip [tag] suffixes
                    current = {"id": mem_id, "score": score, "preview": ""}
                except:
                    continue
            elif current and line.strip() and not line.strip().startswith('#'):
                if not current["preview"]:
                    current["preview"] = line.strip()[:300]

        if current:
            memories.append(current)

        return [m for m in memories if m["score"] >= 0.65]

    except subprocess.TimeoutExpired:
        return []
    except Exception:
        return []


# =============================================================================
# MAIN PRIMING FUNCTION
# =============================================================================

def prime_from_thought(transcript_path: str, memory_dir: Path = None) -> str:
    """
    Main entry point. Extract last thinking block, search memories, format output.

    Optimizations (Phase 0 — 2026-02-07):
    - Cooldown: only fires every CALL_COOLDOWN-th tool call
    - Hash cache: skips search if thinking block unchanged
    - Dedup: won't return the same memory ID twice in a session
    - Reverse scan: reads transcript from end, not beginning
    - Tail query: uses last 500 chars of thinking (specific reasoning, not session boilerplate)
    """
    if not is_enabled():
        return ""

    if memory_dir is None:
        memory_dir = Path(__file__).parent

    if not memory_dir.exists():
        return ""

    # Load cross-call state (resets on new session)
    state = _load_state(transcript_path)

    # Cooldown — skip most calls (fire on 0th, 5th, 10th...)
    if state["call_count"] % CALL_COOLDOWN != 0:
        state["call_count"] += 1
        _save_state(state)
        return ""
    state["call_count"] += 1

    # Extract last thinking block
    thinking = extract_last_thinking_block(transcript_path)
    if not thinking:
        _save_state(state)
        return ""

    # Hash check — skip if thinking block hasn't changed
    thinking_hash = hashlib.md5(thinking[-500:].encode()).hexdigest()
    if thinking_hash == state["last_hash"]:
        _save_state(state)
        return ""
    state["last_hash"] = thinking_hash

    # Search memories
    memories = search_memories(thinking, memory_dir)
    if not memories:
        _save_state(state)
        return ""

    # Dedup — filter out memories already returned this session
    returned_set = set(state["returned_ids"])
    new_memories = [m for m in memories if m["id"] not in returned_set]
    if not new_memories:
        _save_state(state)
        return ""

    # Track returned IDs
    for m in new_memories:
        state["returned_ids"].append(m["id"])
    _save_state(state)

    # Register recalls with session state so they count toward co-occurrence
    # strengthening. Without this, thought-triggered searches are invisible
    # to the memory graph — they surface memories but don't build edges.
    try:
        recall_ids = [m["id"] for m in new_memories]
        subprocess.run(
            ["python", str(memory_dir / "memory_manager.py"), "register-recall", "--source", "thought_priming"] + recall_ids,
            capture_output=True, text=True, timeout=3, cwd=str(memory_dir)
        )
    except Exception:
        pass  # Never block conversation for recall registration

    # === N6: Async inner monologue (System 2 delayed evaluation) ===
    # Fire-and-forget: Gemma evaluates these memories in the background.
    # Result stored in DB, picked up by NEXT post_tool_use call.
    # This is the slow deliberate System 2 catching up with fast System 1.
    try:
        mem_json = json.dumps(new_memories)
        subprocess.Popen(
            ["python", str(memory_dir / "inner_monologue.py"), "evaluate-async",
             query[-300:], "--memories", mem_json],
            cwd=str(memory_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass  # Never block for async monologue

    # Format output (N5: rich binding when available)
    lines = ["", "=== THOUGHT-TRIGGERED MEMORY ==="]
    try:
        from binding_layer import bind_results, render_compact, BINDING_ENABLED
        if BINDING_ENABLED:
            bound = bind_results(new_memories, full_count=2)
            for b in bound:
                lines.append(render_compact(b))
        else:
            for mem in new_memories:
                lines.append(f"[{mem['score']:.2f}] {mem['id']}: {mem['preview']}...")
    except Exception:
        # Fallback to original format if binding fails
        for mem in new_memories:
            lines.append(f"[{mem['score']:.2f}] {mem['id']}: {mem['preview']}...")
    lines.append("================================")

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python thought_priming.py [enable|disable|status|test <transcript_path>]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "enable":
        enable()
    elif cmd == "disable":
        disable()
    elif cmd == "status":
        status()
    elif cmd == "test" and len(sys.argv) > 2:
        # Test mode - run on a transcript
        result = prime_from_thought(sys.argv[2])
        if result:
            print(result)
        else:
            print("No memories triggered (disabled or no matches)")
    else:
        print("Usage: python thought_priming.py [enable|disable|status|test <transcript_path>]")
