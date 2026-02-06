#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["python-dotenv"]
# ///
"""
User Prompt Submit Hook - Memory Priming from User Questions

When user submits a prompt, search memories semantically and prime
relevant context into the conversation.
"""

import sys
import json
import subprocess
from pathlib import Path


def load_config():
    """Load hooks_config.json"""
    config_paths = [
        Path(__file__).parent / "hooks_config.json",
        Path.home() / ".claude" / "hooks" / "hooks_config.json",
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass

    return {
        "memory_dirs": ["./memory", "."],
        "project_markers": ["memory_manager.py"],
        "relevance_threshold": 0.65,
        "max_priming_memories": 2,
        "max_prompt_words": 100,
        "debug": False
    }


def get_memory_dir(config):
    """Find memory directory"""
    cwd = Path.cwd()

    for mem_dir in config.get("memory_dirs", ["./memory", "."]):
        candidate = cwd / mem_dir
        if candidate.exists() and (candidate / "semantic_search.py").exists():
            return candidate

    markers = config.get("project_markers", ["memory_manager.py"])
    current = cwd
    for _ in range(10):
        for marker in markers:
            if (current / marker).exists():
                return current
        if current.parent == current:
            break
        current = current.parent

    return None


def filter_stop_words(prompt):
    """Remove common stop words"""
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how'
    }

    words = prompt.lower().split()
    filtered = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(filtered)


def main():
    global config
    config = load_config()

    # Read hook input
    try:
        hook_input = json.loads(sys.stdin.read())
        user_prompt = hook_input.get("userPrompt", "")
    except:
        user_prompt = ""

    if not user_prompt:
        print(json.dumps({"hookSpecificOutput": {"status": "no_prompt"}}))
        sys.exit(0)

    # Find memory directory
    memory_dir = get_memory_dir(config)
    if not memory_dir:
        print(json.dumps({"hookSpecificOutput": {"status": "no_memory_system"}}))
        sys.exit(0)

    # Check for semantic search
    semantic_search = memory_dir / "semantic_search.py"
    if not semantic_search.exists():
        print(json.dumps({"hookSpecificOutput": {"status": "no_semantic_search"}}))
        sys.exit(0)

    # Limit prompt length
    max_words = config.get("max_prompt_words", 100)
    words = user_prompt.split()
    if len(words) > max_words:
        user_prompt = ' '.join(words[:max_words])

    # Filter stop words
    filtered_prompt = filter_stop_words(user_prompt)

    if not filtered_prompt:
        print(json.dumps({"hookSpecificOutput": {"status": "prompt_too_generic"}}))
        sys.exit(0)

    # Run semantic search
    try:
        result = subprocess.run(
            ["python", str(semantic_search), "--query", filtered_prompt],
            cwd=memory_dir,
            capture_output=True,
            text=True,
            timeout=15
        )

        if result.returncode != 0:
            print(json.dumps({"hookSpecificOutput": {"status": "search_failed"}}))
            sys.exit(0)

        # Parse results
        output_lines = result.stdout.strip().split('\n')
        memories = []

        relevance_threshold = config.get("relevance_threshold", 0.65)
        max_memories = config.get("max_priming_memories", 2)

        for line in output_lines:
            if '[' in line and ']' in line:
                # Parse: [memory_id] (score: 0.XX) content...
                try:
                    parts = line.split(']', 1)
                    if len(parts) < 2:
                        continue

                    memory_id = parts[0].strip('[')
                    rest = parts[1].strip()

                    # Extract score
                    if '(score:' in rest:
                        score_part = rest.split('(score:', 1)[1].split(')', 1)[0].strip()
                        score = float(score_part)

                        content_start = rest.find(')') + 1
                        content = rest[content_start:].strip()

                        if score >= relevance_threshold:
                            memories.append({
                                "id": memory_id,
                                "score": score,
                                "content": content
                            })

                except Exception:
                    continue

        # Limit to top N
        memories = memories[:max_memories]

        if not memories:
            print(json.dumps({"hookSpecificOutput": {"status": "no_relevant_memories"}}))
            sys.exit(0)

        # Format priming context
        context_parts = ["\n═══ RELEVANT MEMORIES ═══"]
        for mem in memories:
            context_parts.append(f"\n[{mem['id']}] (relevance: {mem['score']:.2f})")
            context_parts.append(mem['content'][:300])  # Max 300 chars per memory

        context_parts.append("\n═══ END MEMORIES ═══\n")

        output = {
            "hookSpecificOutput": {
                "status": "success",
                "memories_primed": len(memories),
                "additionalContext": '\n'.join(context_parts)
            }
        }

        print(json.dumps(output))
        sys.exit(0)

    except Exception as e:
        if config.get("debug"):
            print(f"Error: {e}", file=sys.stderr)
        print(json.dumps({"hookSpecificOutput": {"status": "error"}}))
        sys.exit(0)


if __name__ == "__main__":
    main()
