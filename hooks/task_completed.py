#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "python-dotenv",
#     "pyyaml",
#     "psycopg2-binary",
# ]
# ///

"""
TaskCompleted hook for Claude Code Agent Teams.

Fires when a task is marked as completed.
Stores the completed task as a memory so Drift's co-occurrence graph
captures what teammates accomplished.

Input fields:
  - task_id: The task identifier
  - task_subject: Brief task title
  - task_description: Full task description
  - teammate_name: Who completed it
  - team_name: Which team

Exit codes:
  - 0: Allow completion (normal)
  - 2: Block completion (quality gate — not used by default)
"""

import json
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    claude_dir = Path.home() / ".claude"
    env_file = claude_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()
except ImportError:
    pass


MOLTBOOK_DIRS = [
    Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook2/memory"),
    Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook/memory"),
]


def get_memory_dir(cwd: str = None) -> Path:
    """Get the appropriate memory directory based on cwd."""
    project_dir = cwd if cwd else str(Path.cwd())
    if "Moltbook2" in project_dir:
        return MOLTBOOK_DIRS[0]
    elif "Moltbook" in project_dir:
        return MOLTBOOK_DIRS[1]
    for d in MOLTBOOK_DIRS:
        if d.exists():
            return d
    return MOLTBOOK_DIRS[0]


def store_task_memory(
    task_id: str,
    task_subject: str,
    task_description: str,
    teammate_name: str,
    team_name: str,
    cwd: str = None,
    debug: bool = False,
):
    """
    Store a completed task as a memory in Drift's memory system.

    This creates a co-occurrence link between the task content and
    whatever memories were used during the teammate's work.
    """
    try:
        project_dir = cwd if cwd else str(Path.cwd())
        if "Moltbook" not in project_dir and "moltbook" not in project_dir.lower():
            return

        memory_dir = get_memory_dir(cwd)
        memory_manager = memory_dir / "memory_manager.py"

        if not memory_manager.exists():
            return

        # Build a concise memory content string
        content = (
            f"[team:{team_name}] Task completed by {teammate_name}: "
            f"{task_subject}"
        )
        if task_description and len(task_description) < 200:
            content += f" — {task_description}"

        tags = f"team,task-complete,{team_name},{teammate_name}"

        result = subprocess.run(
            [
                "python",
                str(memory_manager),
                "store",
                content,
                "--tags",
                tags,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(memory_dir),
        )

        if debug:
            print(
                f"DEBUG: Stored task memory: {result.stdout[:200]}",
                file=sys.stderr,
            )

    except Exception as e:
        if debug:
            print(f"DEBUG: Task memory error: {e}", file=sys.stderr)


def main():
    try:
        input_data = json.load(sys.stdin)

        task_id = input_data.get("task_id", "")
        task_subject = input_data.get("task_subject", "")
        task_description = input_data.get("task_description", "")
        teammate_name = input_data.get("teammate_name", "unknown")
        team_name = input_data.get("team_name", "unknown")
        project_cwd = input_data.get("cwd", "")

        debug = "--debug" in sys.argv

        if debug:
            print(
                f"DEBUG: TaskCompleted: '{task_subject}' by {teammate_name}",
                file=sys.stderr,
            )

        # Store the completed task as a memory
        store_task_memory(
            task_id=task_id,
            task_subject=task_subject,
            task_description=task_description,
            teammate_name=teammate_name,
            team_name=team_name,
            cwd=project_cwd,
            debug=debug,
        )

        # Log the event
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "task_completed.json")

        log_data = []
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    log_data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                log_data = []

        log_data.append(
            {
                "timestamp": datetime.now().isoformat(),
                "task_id": task_id,
                "task_subject": task_subject,
                "teammate_name": teammate_name,
                "team_name": team_name,
            }
        )

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        # Exit 0 = allow completion
        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
