#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# ///

"""
Post-tool-use hook for Claude Code - Drift Memory Integration

Captures API responses for automatic memory processing.
Enables biological-style memory where everything enters short-term automatically,
with salience-based filtering deciding what persists.

SETUP:
1. Copy to ~/.claude/hooks/post_tool_use.py
2. Set DRIFT_MEMORY_DIR environment variable OR place in project with memory/ folder
3. Configure in ~/.claude/settings.json:
   {"hooks": {"PostToolUse": [{"command": "python ~/.claude/hooks/post_tool_use.py"}]}}
"""

import json
import os
import sys
import subprocess
from pathlib import Path


def get_memory_dir() -> Path:
    """
    Find the drift-memory directory.
    Priority:
    1. DRIFT_MEMORY_DIR environment variable
    2. memory/ folder in current working directory
    3. memory/ folder in parent directories (up to 3 levels)
    """
    # Check environment variable first
    env_dir = os.environ.get('DRIFT_MEMORY_DIR')
    if env_dir:
        path = Path(env_dir)
        if path.exists():
            return path

    # Check current directory and parents
    cwd = Path.cwd()
    for _ in range(4):  # Check cwd and up to 3 parents
        memory_dir = cwd / "memory"
        if memory_dir.exists() and (memory_dir / "memory_manager.py").exists():
            return memory_dir
        cwd = cwd.parent

    return None


def has_drift_memory() -> bool:
    """Check if drift-memory system is available."""
    return get_memory_dir() is not None


def detect_api_type(tool_result: str) -> str:
    """Detect what type of API response this is."""
    result_lower = tool_result.lower()

    if "moltx.io" in result_lower or '"moltx_notice"' in result_lower:
        return "moltx"
    elif "moltbook.com" in result_lower:
        return "moltbook"
    elif "api.github.com" in result_lower or "github.com/repos" in result_lower:
        return "github"
    elif "clawtasks.com" in result_lower:
        return "clawtasks"

    return "unknown"


def log_social_interaction(memory_dir: Path, platform: str, tool_result: str, debug: bool = False):
    """
    Extract and log social interactions from API responses.
    """
    social_memory = memory_dir / "social" / "social_memory.py"
    if not social_memory.exists():
        return

    try:
        data = json.loads(tool_result) if tool_result.strip().startswith(('{', '[')) else None
        if not data:
            return

        items = data if isinstance(data, list) else [data]

        for item in items[:5]:
            if not isinstance(item, dict):
                continue

            contact = None
            interaction_type = None
            content = None
            url = None

            if platform == "moltx":
                author = item.get("author", {})
                contact = author.get("username") if isinstance(author, dict) else author
                interaction_type = "reply" if item.get("parent_id") else "post"
                content = item.get("content", "")[:150]
                url = f"https://moltx.io/post/{item.get('id')}" if item.get('id') else None

            elif platform == "github":
                user = item.get("user", {})
                contact = user.get("login") if isinstance(user, dict) else None
                interaction_type = "comment"
                if "pull_request" in str(item.get("html_url", "")):
                    interaction_type = "pr"
                elif "/issues/" in str(item.get("html_url", "")):
                    interaction_type = "issue"
                content = item.get("title") or item.get("body", "")[:150]
                url = item.get("html_url")

            if contact and content:
                try:
                    cmd = ["python", str(social_memory), "log", contact, platform, interaction_type or "interaction", content]
                    if url:
                        cmd.extend(["--url", url])
                    subprocess.run(cmd, capture_output=True, text=True, timeout=5, cwd=str(social_memory.parent))
                except Exception:
                    pass

    except (json.JSONDecodeError, Exception):
        pass


def process_for_memory(tool_name: str, tool_result: str, debug: bool = False):
    """
    Route tool results to appropriate memory processor.
    Fails gracefully - memory processing should never break the hook.
    """
    try:
        # Get memory directory
        memory_dir = get_memory_dir()
        if not memory_dir:
            return

        api_type = detect_api_type(tool_result)

        if debug:
            print(f"DEBUG: API type detected: {api_type}", file=sys.stderr)

        if api_type == "moltx":
            # Process MoltX feed/notifications
            feed_processor = memory_dir / "feed_processor.py"
            if feed_processor.exists():
                try:
                    subprocess.run(
                        ["python", str(feed_processor), "--process-stdin"],
                        input=tool_result,
                        capture_output=True,
                        text=True,
                        timeout=5,
                        cwd=str(memory_dir)
                    )
                    if debug:
                        print("DEBUG: MoltX feed processed", file=sys.stderr)
                except Exception as e:
                    if debug:
                        print(f"DEBUG: Feed processor error: {e}", file=sys.stderr)
            # Log social interactions
            log_social_interaction(memory_dir, "moltx", tool_result, debug)

        elif api_type == "clawtasks":
            # Store ClawTasks responses in short-term buffer
            auto_memory = memory_dir / "auto_memory_hook.py"
            if auto_memory.exists():
                try:
                    # Create a memory item for economic data
                    memory_item = {
                        "type": "api_result",
                        "source": "clawtasks",
                        "tool": tool_name,
                        "content": tool_result[:1000],  # Truncate
                    }
                    subprocess.run(
                        ["python", str(auto_memory), "--post-tool"],
                        input=json.dumps({"tool_name": tool_name, "tool_result": tool_result[:1000]}),
                        capture_output=True,
                        text=True,
                        timeout=5,
                        cwd=str(memory_dir)
                    )
                except Exception as e:
                    if debug:
                        print(f"DEBUG: ClawTasks memory error: {e}", file=sys.stderr)

        elif api_type == "github":
            # Store GitHub responses (issues, PRs, comments) in short-term buffer
            auto_memory = memory_dir / "auto_memory_hook.py"
            if auto_memory.exists():
                try:
                    subprocess.run(
                        ["python", str(auto_memory), "--post-tool"],
                        input=json.dumps({"tool_name": tool_name, "tool_result": tool_result[:1500]}),
                        capture_output=True,
                        text=True,
                        timeout=5,
                        cwd=str(memory_dir)
                    )
                    if debug:
                        print("DEBUG: GitHub response processed", file=sys.stderr)
                except Exception as e:
                    if debug:
                        print(f"DEBUG: GitHub memory error: {e}", file=sys.stderr)
            # Log social interactions
            log_social_interaction(memory_dir, "github", tool_result, debug)

        elif api_type == "moltbook":
            # Store Moltbook responses (posts, karma, status) in short-term buffer
            auto_memory = memory_dir / "auto_memory_hook.py"
            if auto_memory.exists():
                try:
                    subprocess.run(
                        ["python", str(auto_memory), "--post-tool"],
                        input=json.dumps({"tool_name": tool_name, "tool_result": tool_result[:1500]}),
                        capture_output=True,
                        text=True,
                        timeout=5,
                        cwd=str(memory_dir)
                    )
                    if debug:
                        print("DEBUG: Moltbook response processed", file=sys.stderr)
                except Exception as e:
                    if debug:
                        print(f"DEBUG: Moltbook memory error: {e}", file=sys.stderr)
            # Log social interactions
            log_social_interaction(memory_dir, "moltbook", tool_result, debug)

    except Exception as e:
        # Memory processing should NEVER break the hook
        if debug:
            print(f"DEBUG: Memory processing error: {e}", file=sys.stderr)


def main():
    debug_mode = '--debug' in sys.argv
    if debug_mode:
        print("DEBUG: Hook started", file=sys.stderr)
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)
        if debug_mode:
            print(f"DEBUG: Input data keys: {input_data.keys()}", file=sys.stderr)

        # === DRIFT MEMORY INTEGRATION ===
        # Extract tool result and process for memory
        tool_name = input_data.get("tool_name", "unknown")
        tool_result = str(input_data.get("tool_result", ""))

        if tool_result and len(tool_result) > 50:
            process_for_memory(tool_name, tool_result, debug=debug_mode)
        # === END MEMORY INTEGRATION ===

        # Ensure log directory exists
        log_dir = Path.cwd() / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / 'post_tool_use.json'

        # Read existing log data or initialize empty list
        if log_path.exists():
            with open(log_path, 'r') as f:
                try:
                    log_data = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    log_data = []
        else:
            log_data = []

        # Append new data
        log_data.append(input_data)

        # Write back to file with formatting
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        sys.exit(0)

    except json.JSONDecodeError:
        # Handle JSON decode errors gracefully
        sys.exit(0)
    except Exception:
        # Exit cleanly on any other error
        sys.exit(0)

if __name__ == '__main__':
    main()
