#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""
Automatic Memory Hook for Drift

This module provides hooks that automatically capture, filter, and store
memories without conscious intervention - mimicking biological memory.

Architecture:
1. SHORT-TERM BUFFER: Everything enters here (high capacity, fast decay)
2. ATTENTION FILTER: Salience detection (emotional, repetitive, novel)
3. CONSOLIDATION: Move salient items to long-term with associations
4. DECAY: Time-based forgetting for unreinforced memories

Key insight: Don't choose what to remember. Let the system decide based on:
- Repetition (seeing something multiple times)
- Emotional salience (errors, successes, important keywords)
- Novelty (first time seeing a pattern)
- Association (co-occurrence with existing memories)
"""

import json
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Memory paths
MEMORY_DIR = Path(__file__).parent
SALIENCE_KEYWORDS = [
    # Emotional/Important
    "error", "failed", "success", "important", "critical", "warning",
    "learned", "discovered", "realized", "insight",
    # Economic
    "bounty", "earned", "$", "usdc", "eth", "wallet", "stake",
    # Social
    "collaboration", "mentioned", "replied", "followed",
    # Identity
    "drift", "spindrift", "lex", "agent",
]


def load_short_term() -> dict:
    """Load short-term buffer from DB KV store."""
    try:
        from db_adapter import get_db
        db = get_db()
        raw = db.kv_get('.short_term_buffer')
        if raw:
            return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        pass
    return {"items": [], "last_updated": None}


def save_short_term(data: dict):
    """Save short-term buffer to DB KV store."""
    data["last_updated"] = datetime.now().isoformat()
    from db_adapter import get_db
    db = get_db()
    db.kv_set('.short_term_buffer', data)


def compute_salience(content: str) -> float:
    """
    Compute salience score for content (0.0 - 1.0).
    Higher = more likely to be consolidated to long-term.
    """
    content_lower = content.lower()
    score = 0.0

    # Keyword matching
    for keyword in SALIENCE_KEYWORDS:
        if keyword in content_lower:
            score += 0.1

    # Length bonus (longer = more detail = potentially important)
    if len(content) > 500:
        score += 0.1
    if len(content) > 1000:
        score += 0.1

    # Novelty would require checking against existing memories
    # (not implemented in this draft)

    return min(score, 1.0)


def extract_from_tool_result(tool_name: str, result: str) -> Optional[dict]:
    """
    Extract memorable content from tool results.
    Returns dict with content and metadata, or None if not memorable.
    """
    # Skip if result is too short or clearly not useful
    if len(result) < 50:
        return None

    result_lower = result.lower()

    # Detect API source and extract meaningful content
    api_source = None
    summary = None

    # GitHub API responses
    if "api.github.com" in result_lower or "github.com/repos" in result_lower:
        api_source = "github"
        # Try to extract key info from GitHub responses
        if '"title":' in result or '"body":' in result:
            summary = f"GitHub: {result[:400]}"
        elif '"login":' in result or '"name":' in result:
            summary = f"GitHub user/repo: {result[:400]}"

    # ClawTasks API responses
    elif "clawtasks.com" in result_lower or '"bounty' in result_lower or '"proposal' in result_lower:
        api_source = "clawtasks"
        if '"amount":' in result or '"earned":' in result:
            summary = f"ClawTasks economic: {result[:400]}"
        elif '"status":' in result:
            summary = f"ClawTasks status: {result[:400]}"

    # Moltbook API responses
    elif "moltbook.com" in result_lower or '"submolt":' in result_lower or '"karma":' in result_lower:
        api_source = "moltbook"
        if '"post":' in result or '"title":' in result:
            summary = f"Moltbook post: {result[:400]}"
        elif '"status":' in result:
            summary = f"Moltbook status: {result[:400]}"

    # MoltX API responses
    elif "moltx.io" in result_lower or '"moltx_notice"' in result_lower:
        api_source = "moltx"
        if '"content":' in result:
            summary = f"MoltX: {result[:400]}"

    # General API responses (fallback)
    elif "success" in result_lower or "error" in result_lower:
        api_source = "api"
        summary = result[:500]

    # File contents with code
    elif "def " in result or "class " in result or "function" in result:
        return {
            "type": "code_pattern",
            "tool": tool_name,
            "content": result[:500],
            "timestamp": datetime.now().isoformat(),
        }

    # Return if we identified an API source
    if api_source and summary:
        return {
            "type": "api_result",
            "source": api_source,
            "tool": tool_name,
            "content": summary,
            "timestamp": datetime.now().isoformat(),
        }

    return None


def add_to_short_term(item: dict):
    """Add item to short-term buffer with decay metadata."""
    buffer = load_short_term()

    # Add decay metadata
    item["added_at"] = datetime.now().isoformat()
    item["salience"] = compute_salience(item.get("content", ""))
    item["reinforcement_count"] = 1

    # Check for duplicates (by content hash)
    content_hash = hashlib.md5(item.get("content", "").encode()).hexdigest()[:8]
    item["hash"] = content_hash

    for existing in buffer["items"]:
        if existing.get("hash") == content_hash:
            # Reinforce existing instead of adding duplicate
            existing["reinforcement_count"] = existing.get("reinforcement_count", 1) + 1
            existing["salience"] = min(existing["salience"] + 0.1, 1.0)
            save_short_term(buffer)
            return

    buffer["items"].append(item)

    # Limit buffer size (biological short-term capacity ~7 items)
    # But we'll use a larger buffer since we're processing asynchronously
    MAX_BUFFER = 50
    if len(buffer["items"]) > MAX_BUFFER:
        # Remove lowest salience items
        buffer["items"].sort(key=lambda x: x.get("salience", 0), reverse=True)
        buffer["items"] = buffer["items"][:MAX_BUFFER]

    save_short_term(buffer)


def decay_short_term():
    """
    Apply decay to short-term buffer.
    Items below threshold get removed.
    """
    buffer = load_short_term()
    now = datetime.now()

    surviving = []
    for item in buffer["items"]:
        added_at = datetime.fromisoformat(item.get("added_at", now.isoformat()))
        age_hours = (now - added_at).total_seconds() / 3600

        # Decay formula: salience reduces over time
        # High reinforcement slows decay
        decay_rate = 0.1 / max(item.get("reinforcement_count", 1), 1)
        item["salience"] = item.get("salience", 0.5) - (decay_rate * age_hours)

        # Survive if salience > threshold
        if item["salience"] > 0.2:
            surviving.append(item)

    buffer["items"] = surviving
    save_short_term(buffer)


def consolidate_to_long_term():
    """
    Move high-salience items from short-term to long-term memory.
    This would call memory_manager.py store command.
    """
    buffer = load_short_term()

    # Find items ready for consolidation
    # (high salience + sufficient age + reinforcement)
    ready = []
    for item in buffer["items"]:
        if item.get("salience", 0) >= 0.5 and item.get("reinforcement_count", 1) >= 2:
            ready.append(item)

    # TODO: Call memory_manager.py store for each ready item
    # For now, just mark them as consolidated
    for item in ready:
        item["consolidated"] = True
        print(f"[AUTO-MEMORY] Would consolidate: {item.get('content', '')[:100]}...",
              file=sys.stderr)

    save_short_term(buffer)


# Hook entry point for PostToolUse
def post_tool_use_hook():
    """Called after every tool use. Captures results for memory."""
    try:
        input_data = json.load(sys.stdin)
        tool_name = input_data.get("tool_name", "unknown")
        tool_result = input_data.get("tool_result", "")

        # Extract memorable content
        memory_item = extract_from_tool_result(tool_name, str(tool_result))
        if memory_item:
            add_to_short_term(memory_item)

    except Exception as e:
        print(f"[AUTO-MEMORY] Error: {e}", file=sys.stderr)

    sys.exit(0)


# Hook entry point for Stop
def stop_hook():
    """Called when response ends. Consolidation opportunity."""
    try:
        decay_short_term()
        consolidate_to_long_term()
    except Exception as e:
        print(f"[AUTO-MEMORY] Consolidation error: {e}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-tool", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--decay", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.post_tool:
        post_tool_use_hook()
    elif args.stop:
        stop_hook()
    elif args.decay:
        decay_short_term()
        print("Decay applied")
    elif args.status:
        buffer = load_short_term()
        print(f"Short-term buffer: {len(buffer['items'])} items")
        for item in buffer["items"][:5]:
            print(f"  - [{item.get('salience', 0):.2f}] {item.get('content', '')[:60]}...")
    else:
        print("Usage: --post-tool, --stop, --decay, --status")
