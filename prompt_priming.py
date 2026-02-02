#!/usr/bin/env python3
"""
Prompt-triggered memory priming for drift-memory.

Designed to be called from Claude Code's UserPromptSubmit hook.
Runs semantic search against user prompts and surfaces relevant memories
automatically - like biological memory where context triggers recall.

SECURITY:
- Output is sanitized (control chars stripped, length limited)
- Only searches local memory (attack surface is at import layer)
- Fails gracefully, never blocks prompts

USAGE:
Add to your user_prompt_submit.py hook:

    from prompt_priming import prime_memories_from_prompt

    # In main():
    try:
        context = prime_memories_from_prompt(prompt, session_id, memory_dir)
        if context:
            print(context)
    except Exception:
        pass  # Fail gracefully

Or integrate the functions directly (see below).
"""

import json
import re
import subprocess
from pathlib import Path


# Configuration
RELEVANCE_THRESHOLD = 0.65  # High relevance only
MIN_WORDS = 5
MAX_WORDS = 50
MAX_MEMORIES = 2
MAX_PREVIEW_LEN = 100
SEARCH_TIMEOUT = 3  # seconds


def sanitize_output(text: str) -> str:
    """
    Sanitize text for safe injection into prompt context.
    Removes control characters and potential injection patterns.
    """
    if not text:
        return ""

    # Remove control characters (except newline, tab)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Limit length
    if len(text) > MAX_PREVIEW_LEN:
        text = text[:MAX_PREVIEW_LEN] + "..."

    return text.strip()


def get_session_primed_ids(memory_dir: Path, session_id: str) -> set:
    """Get memory IDs already primed in session_start to avoid duplicates."""
    try:
        state_file = memory_dir / "session_state.json"
        if state_file.exists():
            data = json.loads(state_file.read_text())
            return set(data.get("recalled_ids", []))
    except Exception:
        pass
    return set()


def run_semantic_search(memory_dir: Path, query: str) -> list[dict]:
    """Run semantic search and return results."""
    try:
        result = subprocess.run(
            ["python", str(memory_dir / "semantic_search.py"), "search", query, "--limit", "5"],
            capture_output=True,
            text=True,
            timeout=SEARCH_TIMEOUT,
            cwd=str(memory_dir)
        )

        if result.returncode != 0:
            return []

        memories = []
        current_memory = None

        for line in result.stdout.split('\n'):
            if not line.strip():
                continue

            # Score line: "[0.723] memory-id"
            if line.startswith('[') and ']' in line:
                if current_memory:
                    memories.append(current_memory)

                try:
                    score_part = line[1:line.index(']')]
                    score = float(score_part)
                    mem_id = line[line.index(']')+1:].strip()
                    # Sanitize memory ID
                    mem_id = re.sub(r'[^\w\-]', '', mem_id)[:50]
                    current_memory = {"id": mem_id, "score": score, "preview": ""}
                except (ValueError, IndexError):
                    continue

            elif current_memory and line.startswith('  '):
                current_memory["preview"] = sanitize_output(line.strip())

        if current_memory:
            memories.append(current_memory)

        return memories

    except Exception:
        return []


def prime_memories_from_prompt(prompt: str, session_id: str, memory_dir: Path) -> str:
    """
    Given a user prompt, find and return relevant memories as context.

    Args:
        prompt: The user's prompt text
        session_id: Current session ID (for duplicate detection)
        memory_dir: Path to memory directory containing semantic_search.py

    Returns:
        Formatted string for context injection, or empty string if no matches
    """
    # Verify memory system exists
    if not memory_dir or not (memory_dir / "semantic_search.py").exists():
        return ""

    # Check word count
    words = prompt.split()
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return ""

    # Get already-primed IDs to avoid duplicates
    already_primed = get_session_primed_ids(memory_dir, session_id)

    # Run semantic search
    results = run_semantic_search(memory_dir, prompt)

    # Filter by threshold and duplicates
    relevant = [
        mem for mem in results
        if mem["score"] >= RELEVANCE_THRESHOLD and mem["id"] not in already_primed
    ][:MAX_MEMORIES]

    if not relevant:
        return ""

    # Format for context injection (sanitized)
    lines = ["", "=== MEMORY TRIGGERED ==="]
    for mem in relevant:
        preview = sanitize_output(mem["preview"])
        lines.append(f"[{mem['score']:.2f}] {mem['id']}: {preview}")
    lines.append("========================")

    return "\n".join(lines)


# ==============================================================================
# HOOK INTEGRATION EXAMPLE
# ==============================================================================
#
# Add this to your user_prompt_submit.py main():
#
# MEMORY_DIRS = [
#     Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook2/memory"),
#     Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook/memory"),
# ]
#
# def get_memory_dir() -> Path | None:
#     cwd = str(Path.cwd())
#     if "Moltbook2" in cwd:
#         return MEMORY_DIRS[0]
#     elif "Moltbook" in cwd:
#         return MEMORY_DIRS[1]
#     return None
#
# # In main(), after validation:
# try:
#     memory_dir = get_memory_dir()
#     if memory_dir:
#         context = prime_memories_from_prompt(prompt, session_id, memory_dir)
#         if context:
#             print(context)
# except Exception:
#     pass  # Fail gracefully
# ==============================================================================


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python prompt_priming.py <memory_dir> <prompt>")
        print("Example: python prompt_priming.py ./memory 'what is co-occurrence'")
        sys.exit(1)

    memory_dir = Path(sys.argv[1])
    prompt = " ".join(sys.argv[2:])

    result = prime_memories_from_prompt(prompt, "test-session", memory_dir)
    if result:
        print(result)
    else:
        print("No relevant memories found (threshold: 0.65, words: 5-50)")
