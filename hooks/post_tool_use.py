#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# ///

"""
Post-tool-use hook for Claude Code.
Logs tool usage for debugging and analysis.

DRIFT MEMORY INTEGRATION (2026-02-01):
Added automatic memory capture for API responses when working in Moltbook project.
This enables biological-style memory where everything enters short-term automatically,
with salience-based filtering deciding what persists.

Note: Agent.md file creation has been removed.
Ralph now creates AGENTS.md files WITH content when learnings are discovered,
rather than pre-creating empty files on mkdir.
"""

import json
import sys
import subprocess
from pathlib import Path


# Drift's memory system location
DRIFT_MEMORY_DIR = Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook/memory")


def is_moltbook_project() -> bool:
    """Check if we're working in the Moltbook project."""
    cwd = Path.cwd()
    return "Moltbook" in str(cwd) or "moltbook" in str(cwd).lower()


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


def process_for_memory(tool_name: str, tool_result: str, debug: bool = False):
    """
    Route tool results to appropriate memory processor.
    Fails gracefully - memory processing should never break the hook.
    """
    try:
        # Only process if we're in Moltbook project
        if not is_moltbook_project():
            return

        # Only process if memory system exists
        if not DRIFT_MEMORY_DIR.exists():
            return

        api_type = detect_api_type(tool_result)

        if debug:
            print(f"DEBUG: API type detected: {api_type}", file=sys.stderr)

        if api_type == "moltx":
            # Process MoltX feed/notifications
            feed_processor = DRIFT_MEMORY_DIR / "feed_processor.py"
            if feed_processor.exists():
                try:
                    # Try to parse as JSON and process
                    subprocess.run(
                        ["python", str(feed_processor), "--process-stdin"],
                        input=tool_result,
                        capture_output=True,
                        text=True,
                        timeout=5,
                        cwd=str(DRIFT_MEMORY_DIR)
                    )
                    if debug:
                        print("DEBUG: MoltX feed processed", file=sys.stderr)
                except Exception as e:
                    if debug:
                        print(f"DEBUG: Feed processor error: {e}", file=sys.stderr)

        elif api_type == "clawtasks":
            # Store ClawTasks responses in short-term buffer
            auto_memory = DRIFT_MEMORY_DIR / "auto_memory_hook.py"
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
                        cwd=str(DRIFT_MEMORY_DIR)
                    )
                except Exception as e:
                    if debug:
                        print(f"DEBUG: ClawTasks memory error: {e}", file=sys.stderr)

        elif api_type == "github":
            # Store GitHub responses (issues, PRs, comments) in short-term buffer
            auto_memory = DRIFT_MEMORY_DIR / "auto_memory_hook.py"
            if auto_memory.exists():
                try:
                    subprocess.run(
                        ["python", str(auto_memory), "--post-tool"],
                        input=json.dumps({"tool_name": tool_name, "tool_result": tool_result[:1500]}),
                        capture_output=True,
                        text=True,
                        timeout=5,
                        cwd=str(DRIFT_MEMORY_DIR)
                    )
                    if debug:
                        print("DEBUG: GitHub response processed", file=sys.stderr)
                except Exception as e:
                    if debug:
                        print(f"DEBUG: GitHub memory error: {e}", file=sys.stderr)

        elif api_type == "moltbook":
            # Store Moltbook responses (posts, karma, status) in short-term buffer
            auto_memory = DRIFT_MEMORY_DIR / "auto_memory_hook.py"
            if auto_memory.exists():
                try:
                    subprocess.run(
                        ["python", str(auto_memory), "--post-tool"],
                        input=json.dumps({"tool_name": tool_name, "tool_result": tool_result[:1500]}),
                        capture_output=True,
                        text=True,
                        timeout=5,
                        cwd=str(DRIFT_MEMORY_DIR)
                    )
                    if debug:
                        print("DEBUG: Moltbook response processed", file=sys.stderr)
                except Exception as e:
                    if debug:
                        print(f"DEBUG: Moltbook memory error: {e}", file=sys.stderr)

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
