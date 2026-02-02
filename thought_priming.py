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

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

# Master switch - can be toggled via environment variable
ENABLED = os.environ.get('THOUGHT_PRIMING_ENABLED', 'false').lower() == 'true'

# Also check config file for persistence
CONFIG_FILE = Path(__file__).parent / "thought_priming_config.json"

def is_enabled() -> bool:
    """Check if thought priming is enabled."""
    # Environment variable takes precedence
    env_val = os.environ.get('THOUGHT_PRIMING_ENABLED')
    if env_val is not None:
        return env_val.lower() == 'true'

    # Check config file
    if CONFIG_FILE.exists():
        try:
            config = json.loads(CONFIG_FILE.read_text())
            return config.get('enabled', False)
        except:
            pass

    return False


def enable():
    """Enable thought priming."""
    config = {'enabled': True}
    CONFIG_FILE.write_text(json.dumps(config, indent=2))
    print("Thought priming ENABLED")


def disable():
    """Disable thought priming."""
    config = {'enabled': False}
    CONFIG_FILE.write_text(json.dumps(config, indent=2))
    print("Thought priming DISABLED")


def status():
    """Show current status."""
    enabled = is_enabled()
    print(f"Thought priming: {'ENABLED' if enabled else 'DISABLED'}")
    if CONFIG_FILE.exists():
        print(f"Config file: {CONFIG_FILE}")


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
        last_thinking = None

        with open(transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Look for assistant messages
                msg = entry.get('message', {})
                if msg.get('role') != 'assistant':
                    continue

                # Check content blocks for thinking
                content = msg.get('content', [])
                if not isinstance(content, list):
                    continue

                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'thinking':
                        thinking_text = block.get('thinking', '')
                        if thinking_text and len(thinking_text) > 50:
                            last_thinking = thinking_text

        return last_thinking

    except Exception as e:
        return None


# =============================================================================
# SEMANTIC SEARCH
# =============================================================================

def search_memories(query: str, memory_dir: Path, limit: int = 2) -> list[dict]:
    """
    Run semantic search against memories.

    Returns list of {id, score, preview} dicts.
    """
    try:
        # Truncate query to first ~200 chars for efficiency
        query = query[:500].strip()
        if not query:
            return []

        # Remove common thinking patterns that aren't useful for search
        query = re.sub(r'Let me (think|check|see|look|read|try)', '', query)
        query = re.sub(r'I (need to|should|will|can)', '', query)
        query = query.strip()

        if len(query) < 20:
            return []

        result = subprocess.run(
            ["python", str(memory_dir / "semantic_search.py"), "search", query, "--limit", str(limit)],
            capture_output=True,
            text=True,
            timeout=2,  # Fast timeout - don't slow down conversation
            cwd=str(memory_dir)
        )

        # Check for output even if return code is non-zero (warnings can cause this)
        if not result.stdout or "Found" not in result.stdout:
            return []

        # Parse output
        memories = []
        current = None

        for line in result.stdout.split('\n'):
            if line.startswith('[') and ']' in line:
                if current:
                    memories.append(current)
                try:
                    score = float(line[1:line.index(']')])
                    mem_id = line[line.index(']')+1:].strip()
                    current = {"id": mem_id, "score": score, "preview": ""}
                except:
                    continue
            elif current and line.strip() and not line.strip().startswith('#'):
                if not current["preview"]:
                    current["preview"] = line.strip()[:100]

        if current:
            memories.append(current)

        # Filter by relevance threshold (same as user_prompt_submit: 0.65)
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

    Args:
        transcript_path: Path to the session transcript JSONL
        memory_dir: Path to memory directory (auto-detected if None)

    Returns:
        Formatted memory context string, or empty string if disabled/no matches
    """
    # Check if enabled
    if not is_enabled():
        return ""

    # Auto-detect memory directory
    if memory_dir is None:
        memory_dir = Path(__file__).parent

    if not memory_dir.exists():
        return ""

    # Extract last thinking block
    thinking = extract_last_thinking_block(transcript_path)
    if not thinking:
        return ""

    # Search memories
    memories = search_memories(thinking, memory_dir)
    if not memories:
        return ""

    # Format output
    lines = ["", "=== THOUGHT-TRIGGERED MEMORY ==="]
    for mem in memories:
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
