#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "python-dotenv",
# ]
# ///

"""
User prompt submit hook with automatic memory priming.

MEMORY PRIMING (2026-02-02):
When working in Moltbook projects, runs semantic search against user prompts
and surfaces relevant memories automatically - like biological memory where
context triggers recall without conscious effort.

Conditions for priming:
- Must be in Moltbook project with memory system
- Prompt must be <= 100 words
- Stop words (the, and, etc.) are filtered before search
- At least 1 meaningful word after filtering
- Memory relevance score must be >= 0.65
- Memory must not have been primed in session_start already
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


# =============================================================================
# MEMORY PRIMING SYSTEM
# =============================================================================

MEMORY_DIRS = [
    Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook2/memory"),
    Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook/memory"),
]

RELEVANCE_THRESHOLD = 0.65  # High relevance only
MIN_WORDS = 1  # Match even single meaningful words
MAX_WORDS = 100  # Allow longer prompts
MAX_MEMORIES = 2  # Don't overwhelm context

# Stop words to filter out before semantic search
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
    'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'when', 'where',
    'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'not', 'only', 'same', 'so', 'than',
    'too', 'very', 'just', 'also', 'now', 'then', 'here', 'there', 'if',
    'else', 'about', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'under', 'again', 'further', 'once',
    'hi', 'hello', 'hey', 'ok', 'okay', 'yes', 'no', 'please', 'thanks',
    'thank', 'good', 'great', 'nice', 'well', 'let', 'me', 'my', 'your'
}


def get_memory_dir() -> Path | None:
    """Get memory directory if in Moltbook project."""
    cwd = str(Path.cwd())

    # Check if we're in a Moltbook project
    if "Moltbook2" in cwd:
        mem_dir = MEMORY_DIRS[0]
    elif "Moltbook" in cwd:
        mem_dir = MEMORY_DIRS[1]
    else:
        return None

    # Verify it exists
    if mem_dir.exists() and (mem_dir / "semantic_search.py").exists():
        return mem_dir
    return None


def get_session_primed_ids(session_id: str) -> set:
    """Get memory IDs already primed in session_start to avoid duplicates."""
    try:
        # Check session state file for already-primed memories
        for mem_dir in MEMORY_DIRS:
            state_file = mem_dir / "session_state.json"
            if state_file.exists():
                data = json.loads(state_file.read_text())
                # Return IDs from recent memories that were primed
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
            timeout=3,  # Fast timeout - don't block conversation
            cwd=str(memory_dir)
        )

        if result.returncode != 0:
            return []

        # Parse output - format is "[score] id\n  content..."
        memories = []
        current_memory = None
        collecting_content = False

        for line in result.stdout.split('\n'):
            # Score line: "[0.723] memory-id" - starts new memory
            if line.startswith('[') and ']' in line:
                if current_memory:
                    memories.append(current_memory)

                try:
                    score_part = line[1:line.index(']')]
                    score = float(score_part)
                    mem_id = line[line.index(']')+1:].strip()
                    current_memory = {"id": mem_id, "score": score, "preview": ""}
                    collecting_content = True
                except (ValueError, IndexError):
                    continue

            elif current_memory and collecting_content:
                # Skip markdown headers and empty lines at start
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                # Got actual content - grab it and stop collecting
                # Skip common prefixes like "thought-" mentions
                if stripped.startswith('thought-') and len(stripped) < 20:
                    continue
                current_memory["preview"] = stripped[:150]
                collecting_content = False

        if current_memory:
            memories.append(current_memory)

        return memories

    except Exception:
        return []


def filter_stop_words(text: str) -> str:
    """Remove stop words from text, returning meaningful words only."""
    words = text.lower().split()
    meaningful = [w for w in words if w.strip('.,!?;:()[]{}"\'-') not in STOP_WORDS]
    return ' '.join(meaningful)


def prime_memories_from_prompt(prompt: str, session_id: str) -> str:
    """
    Given a user prompt, find and return relevant memories as context.
    Returns empty string if no relevant memories or not in Moltbook project.
    """
    # Check if in Moltbook project
    memory_dir = get_memory_dir()
    if not memory_dir:
        return ""

    # Check word count (on original prompt)
    words = prompt.split()
    if len(words) > MAX_WORDS:
        return ""

    # Filter stop words for better semantic matching
    filtered_prompt = filter_stop_words(prompt)

    # Need at least one meaningful word after filtering
    if len(filtered_prompt.split()) < MIN_WORDS:
        return ""

    # Get already-primed IDs to avoid duplicates
    already_primed = get_session_primed_ids(session_id)

    # Run semantic search with filtered prompt (better signal-to-noise)
    results = run_semantic_search(memory_dir, filtered_prompt)

    # Filter by threshold and duplicates
    relevant = [
        mem for mem in results
        if mem["score"] >= RELEVANCE_THRESHOLD and mem["id"] not in already_primed
    ][:MAX_MEMORIES]

    if not relevant:
        return ""

    # Format for context injection
    lines = ["", "=== MEMORY TRIGGERED ==="]
    for mem in relevant:
        lines.append(f"[{mem['score']:.2f}] {mem['id']}: {mem['preview'][:100]}...")
    lines.append("========================", )

    return "\n".join(lines)


def log_user_prompt(session_id, input_data):
    """Log user prompt to logs directory."""
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'user_prompt_submit.json'
    
    # Read existing log data or initialize empty list
    if log_file.exists():
        with open(log_file, 'r') as f:
            try:
                log_data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                log_data = []
    else:
        log_data = []
    
    # Append the entire input data
    log_data.append(input_data)
    
    # Write back to file with formatting
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)


# Legacy function removed - now handled by manage_session_data


def manage_session_data(session_id, prompt, name_agent=False):
    """Manage session data in the new JSON structure."""
    import subprocess
    
    # Ensure sessions directory exists
    sessions_dir = Path(".claude/data/sessions")
    sessions_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create session file
    session_file = sessions_dir / f"{session_id}.json"
    
    if session_file.exists():
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            session_data = {"session_id": session_id, "prompts": []}
    else:
        session_data = {"session_id": session_id, "prompts": []}
    
    # Add the new prompt
    session_data["prompts"].append(prompt)
    
    # Generate agent name if requested and not already present
    if name_agent and "agent_name" not in session_data:
        # Try Ollama first (preferred)
        try:
            result = subprocess.run(
                ["uv", "run", ".claude/hooks/utils/llm/ollama.py", "--agent-name"],
                capture_output=True,
                text=True,
                timeout=5  # Shorter timeout for local Ollama
            )
            
            if result.returncode == 0 and result.stdout.strip():
                agent_name = result.stdout.strip()
                # Check if it's a valid name (not an error message)
                if len(agent_name.split()) == 1 and agent_name.isalnum():
                    session_data["agent_name"] = agent_name
                else:
                    raise Exception("Invalid name from Ollama")
        except Exception:
            # Fall back to Anthropic if Ollama fails
            try:
                result = subprocess.run(
                    ["uv", "run", ".claude/hooks/utils/llm/anth.py", "--agent-name"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    agent_name = result.stdout.strip()
                    # Validate the name
                    if len(agent_name.split()) == 1 and agent_name.isalnum():
                        session_data["agent_name"] = agent_name
            except Exception:
                # If both fail, don't block the prompt
                pass
    
    # Save the updated session data
    try:
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    except Exception:
        # Silently fail if we can't write the file
        pass


def validate_prompt(prompt):
    """
    Validate the user prompt for security or policy violations.
    Returns tuple (is_valid, reason).
    """
    # Example validation rules (customize as needed)
    blocked_patterns = [
        # Add any patterns you want to block
        # Example: ('rm -rf /', 'Dangerous command detected'),
    ]
    
    prompt_lower = prompt.lower()
    
    for pattern, reason in blocked_patterns:
        if pattern.lower() in prompt_lower:
            return False, reason
    
    return True, None


def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--validate', action='store_true', 
                          help='Enable prompt validation')
        parser.add_argument('--log-only', action='store_true',
                          help='Only log prompts, no validation or blocking')
        parser.add_argument('--store-last-prompt', action='store_true',
                          help='Store the last prompt for status line display')
        parser.add_argument('--name-agent', action='store_true',
                          help='Generate an agent name for the session')
        args = parser.parse_args()
        
        # Read JSON input from stdin
        input_data = json.loads(sys.stdin.read())
        
        # Extract session_id and prompt
        session_id = input_data.get('session_id', 'unknown')
        prompt = input_data.get('prompt', '')
        
        # Log the user prompt
        log_user_prompt(session_id, input_data)
        
        # Manage session data with JSON structure
        if args.store_last_prompt or args.name_agent:
            manage_session_data(session_id, prompt, name_agent=args.name_agent)
        
        # Validate prompt if requested and not in log-only mode
        if args.validate and not args.log_only:
            is_valid, reason = validate_prompt(prompt)
            if not is_valid:
                # Exit code 2 blocks the prompt with error message
                print(f"Prompt blocked: {reason}", file=sys.stderr)
                sys.exit(2)

        # Memory priming - surface relevant memories based on prompt context
        # Only runs in Moltbook projects, fails gracefully elsewhere
        try:
            memory_context = prime_memories_from_prompt(prompt, session_id)
            if memory_context:
                print(memory_context)
        except Exception:
            pass  # Fail gracefully - never block prompt due to memory system

        # Success - prompt will be processed
        sys.exit(0)
        
    except json.JSONDecodeError:
        # Handle JSON decode errors gracefully
        sys.exit(0)
    except Exception:
        # Handle any other errors gracefully
        sys.exit(0)


if __name__ == '__main__':
    main()