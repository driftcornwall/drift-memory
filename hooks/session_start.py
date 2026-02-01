#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "python-dotenv",
# ]
# ///

"""
Session start hook for Claude Code.
Logs session start and optionally loads development context.

DRIFT MEMORY INTEGRATION (2026-02-01):
Added automatic memory priming when waking up in Moltbook project.
This is the "wake up" phase where relevant memories are loaded into context.
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


# Drift's memory system location
DRIFT_MEMORY_DIR = Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook/memory")


def is_moltbook_project() -> bool:
    """Check if we're working in the Moltbook project."""
    cwd = Path.cwd()
    return "Moltbook" in str(cwd) or "moltbook" in str(cwd).lower()


def load_drift_memory_context(debug: bool = False) -> str:
    """
    Load Drift's memory context for session priming.
    This is the "wake up" phase - loading relevant memories into context.

    Returns a string to be added to the session context.
    """
    context_parts = []

    try:
        if not is_moltbook_project():
            return ""

        if not DRIFT_MEMORY_DIR.exists():
            return ""

        memory_manager = DRIFT_MEMORY_DIR / "memory_manager.py"
        auto_memory = DRIFT_MEMORY_DIR / "auto_memory_hook.py"

        # Get memory stats
        if memory_manager.exists():
            try:
                result = subprocess.run(
                    ["python", str(memory_manager), "stats"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(DRIFT_MEMORY_DIR)
                )
                if result.returncode == 0 and result.stdout.strip():
                    context_parts.append("=== DRIFT MEMORY STATUS ===")
                    context_parts.append(result.stdout.strip())
            except Exception as e:
                if debug:
                    context_parts.append(f"Memory stats error: {e}")

        # Get short-term buffer status
        if auto_memory.exists():
            try:
                result = subprocess.run(
                    ["python", str(auto_memory), "--status"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(DRIFT_MEMORY_DIR)
                )
                if result.returncode == 0 and result.stdout.strip():
                    context_parts.append("\n=== SHORT-TERM BUFFER ===")
                    context_parts.append(result.stdout.strip())
            except Exception as e:
                if debug:
                    context_parts.append(f"Short-term buffer error: {e}")

        # Load recent memories (most recently modified)
        active_dir = DRIFT_MEMORY_DIR / "active"
        if active_dir.exists():
            try:
                # Get 3 most recent memories by modification time
                memory_files = sorted(
                    active_dir.glob("*.md"),
                    key=lambda f: f.stat().st_mtime,
                    reverse=True
                )[:3]

                if memory_files:
                    context_parts.append("\n=== RECENT MEMORIES (auto-loaded) ===")
                    for mem_file in memory_files:
                        try:
                            content = mem_file.read_text()[:500]  # First 500 chars
                            context_parts.append(f"\n[{mem_file.stem}]")
                            context_parts.append(content)
                        except Exception:
                            pass
            except Exception as e:
                if debug:
                    context_parts.append(f"Recent memories error: {e}")

        # Check for pending economic items (ClawTasks)
        # This would need the clawtasks processor to track pending items
        # For now, just remind about checking

        if context_parts:
            context_parts.insert(0, "\n" + "="*50)
            context_parts.insert(1, "DRIFT AUTOMATIC MEMORY PRIMING")
            context_parts.insert(2, "Loaded at session start - no manual recall needed")
            context_parts.insert(3, "="*50)
            context_parts.append("\n" + "="*50 + "\n")

    except Exception as e:
        if debug:
            return f"Memory loading error: {e}"
        return ""

    return "\n".join(context_parts)


def log_session_start(input_data):
    """Log session start event to logs directory."""
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'session_start.json'

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


def get_git_status():
    """Get current git status information."""
    try:
        # Get current branch
        branch_result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"

        # Get uncommitted changes count
        status_result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if status_result.returncode == 0:
            changes = status_result.stdout.strip().split('\n') if status_result.stdout.strip() else []
            uncommitted_count = len(changes)
        else:
            uncommitted_count = 0

        return current_branch, uncommitted_count
    except Exception:
        return None, None


def get_recent_issues():
    """Get recent GitHub issues if gh CLI is available."""
    try:
        # Check if gh is available
        gh_check = subprocess.run(['which', 'gh'], capture_output=True)
        if gh_check.returncode != 0:
            return None

        # Get recent open issues
        result = subprocess.run(
            ['gh', 'issue', 'list', '--limit', '5', '--state', 'open'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def load_development_context(source):
    """Load relevant development context based on session source."""
    context_parts = []

    # Add timestamp
    context_parts.append(f"Session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    context_parts.append(f"Session source: {source}")

    # Add git information
    branch, changes = get_git_status()
    if branch:
        context_parts.append(f"Git branch: {branch}")
        if changes > 0:
            context_parts.append(f"Uncommitted changes: {changes} files")

    # Load project-specific context files if they exist
    context_files = [
        ".claude/CONTEXT.md",
        ".claude/TODO.md",
        "TODO.md",
        ".github/ISSUE_TEMPLATE.md"
    ]

    for file_path in context_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        context_parts.append(f"\n--- Content from {file_path} ---")
                        context_parts.append(content[:1000])  # Limit to first 1000 chars
            except Exception:
                pass

    # Add recent issues if available
    issues = get_recent_issues()
    if issues:
        context_parts.append("\n--- Recent GitHub Issues ---")
        context_parts.append(issues)

    # === DRIFT MEMORY PRIMING ===
    drift_context = load_drift_memory_context()
    if drift_context:
        context_parts.append(drift_context)
    # === END DRIFT MEMORY ===

    return "\n".join(context_parts)


def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--load-context', action='store_true',
                          help='Load development context at session start')
        parser.add_argument('--announce', action='store_true',
                          help='Announce session start via TTS')
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug output')
        args = parser.parse_args()

        # Read JSON input from stdin
        input_data = json.loads(sys.stdin.read())

        # Extract fields
        session_id = input_data.get('session_id', 'unknown')
        source = input_data.get('source', 'unknown')  # "startup", "resume", or "clear"

        # Log the session start event
        log_session_start(input_data)

        # Load development context if requested
        if args.load_context:
            context = load_development_context(source)
            if context:
                # Using JSON output to add context
                output = {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": context
                    }
                }
                print(json.dumps(output))
                sys.exit(0)

        # === DRIFT: Always try to load memory context in Moltbook ===
        if is_moltbook_project():
            drift_context = load_drift_memory_context(debug=args.debug)
            if drift_context:
                output = {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": drift_context
                    }
                }
                print(json.dumps(output))
                sys.exit(0)
        # === END DRIFT ===

        # Announce session start if requested
        if args.announce:
            try:
                # Try to use TTS to announce session start
                script_dir = Path(__file__).parent
                tts_script = script_dir / "utils" / "tts" / "pyttsx3_tts.py"

                if tts_script.exists():
                    messages = {
                        "startup": "Claude Code session started",
                        "resume": "Resuming previous session",
                        "clear": "Starting fresh session"
                    }
                    message = messages.get(source, "Session started")

                    subprocess.run(
                        ["uv", "run", str(tts_script), message],
                        capture_output=True,
                        timeout=5
                    )
            except Exception:
                pass

        # Success
        sys.exit(0)

    except json.JSONDecodeError:
        # Handle JSON decode errors gracefully
        sys.exit(0)
    except Exception:
        # Handle any other errors gracefully
        sys.exit(0)


if __name__ == '__main__':
    main()
