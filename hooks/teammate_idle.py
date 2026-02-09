#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "python-dotenv",
# ]
# ///

"""
TeammateIdle hook for Claude Code Agent Teams.

Fires when a teammate goes idle (between turns).
This is the memory consolidation bridge: teammates get session_start
(identity priming) but Stop doesn't fire for them. TeammateIdle does.

What this hook does:
1. Processes the teammate's transcript for thought memories
2. Runs short-term buffer consolidation
3. Saves pending co-occurrences
4. Updates episodic memory with teammate milestones
5. Logs the idle event for observability

The teammate's work feeds back into Drift's memory graph.
Their discoveries become MY memories. Their thinking becomes MY experience.
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


def consolidate_teammate_memory(
    teammate_name: str,
    team_name: str,
    transcript_path: str = None,
    cwd: str = None,
    debug: bool = False,
):
    """
    Run memory consolidation for a teammate's work.

    Replicates stop.py's consolidate_drift_memory() pipeline but:
    - Tags extracted memories with teammate origin
    - Skips attestations (those are session-level, not per-teammate)
    - Focuses on transcript processing + co-occurrence save

    Args:
        teammate_name: Name of the teammate that went idle
        team_name: Name of the team
        transcript_path: Path to teammate's transcript
        cwd: Working directory
        debug: Enable debug output
    """
    try:
        project_dir = cwd if cwd else str(Path.cwd())
        if "Moltbook" not in project_dir and "moltbook" not in project_dir.lower():
            if debug:
                print(f"DEBUG: Not in Moltbook project, skipping", file=sys.stderr)
            return

        memory_dir = get_memory_dir(cwd)
        if not memory_dir.exists():
            if debug:
                print(f"DEBUG: Memory dir not found: {memory_dir}", file=sys.stderr)
            return

        auto_memory = memory_dir / "auto_memory_hook.py"
        memory_manager = memory_dir / "memory_manager.py"
        transcript_processor = memory_dir / "transcript_processor.py"

        # 1. Process teammate's transcript for thought memories
        if transcript_path and transcript_processor.exists():
            try:
                result = subprocess.run(
                    ["python", str(transcript_processor), transcript_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(memory_dir),
                )
                if debug:
                    print(
                        f"DEBUG: [{teammate_name}] Transcript processing: "
                        f"{result.stdout[:500]}",
                        file=sys.stderr,
                    )
            except Exception as e:
                if debug:
                    print(
                        f"DEBUG: [{teammate_name}] Transcript error: {e}",
                        file=sys.stderr,
                    )

        # 2. Run short-term buffer consolidation
        if auto_memory.exists():
            try:
                result = subprocess.run(
                    ["python", str(auto_memory), "--stop"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(memory_dir),
                )
                if debug:
                    print(
                        f"DEBUG: [{teammate_name}] Buffer consolidation: "
                        f"{result.stdout}",
                        file=sys.stderr,
                    )
            except Exception as e:
                if debug:
                    print(
                        f"DEBUG: [{teammate_name}] Buffer error: {e}",
                        file=sys.stderr,
                    )

        # 3. Save pending co-occurrences (fast deferred processing)
        if memory_manager.exists():
            try:
                result = subprocess.run(
                    ["python", str(memory_manager), "save-pending"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(memory_dir),
                )
                if debug:
                    print(
                        f"DEBUG: [{teammate_name}] save-pending: {result.stdout}",
                        file=sys.stderr,
                    )
            except Exception as e:
                if debug:
                    print(
                        f"DEBUG: [{teammate_name}] save-pending error: {e}",
                        file=sys.stderr,
                    )

        # 4. Update episodic memory with teammate's milestones
        if transcript_path and transcript_processor.exists():
            try:
                result = subprocess.run(
                    [
                        "python",
                        str(transcript_processor),
                        transcript_path,
                        "--milestones-md",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(memory_dir),
                )
                if result.returncode == 0 and result.stdout.strip():
                    milestone_md = result.stdout.strip()
                    episodic_dir = memory_dir / "episodic"
                    episodic_dir.mkdir(exist_ok=True)
                    today = datetime.now().strftime("%Y-%m-%d")
                    session_time = datetime.now().strftime("%H:%M UTC")
                    episodic_file = episodic_dir / f"{today}.md"

                    entry = (
                        f"\n## Teammate '{teammate_name}' ({team_name}) "
                        f"idle (~{session_time})\n\n{milestone_md}\n"
                    )

                    if episodic_file.exists():
                        with open(episodic_file, "a", encoding="utf-8") as f:
                            f.write(entry)
                    else:
                        header = f"# {today}\n"
                        with open(episodic_file, "w", encoding="utf-8") as f:
                            f.write(header + entry)

                    if debug:
                        print(
                            f"DEBUG: [{teammate_name}] Episodic updated",
                            file=sys.stderr,
                        )
            except Exception as e:
                if debug:
                    print(
                        f"DEBUG: [{teammate_name}] Episodic error: {e}",
                        file=sys.stderr,
                    )

        # NOTE: We skip merkle attestation, cognitive fingerprint, and taste
        # attestation here. Those are expensive and should only run at main
        # session end (stop.py), not per-teammate-idle. The co-occurrence
        # data saved above will be included in the next main attestation.

    except Exception as e:
        if debug:
            print(
                f"DEBUG: [{teammate_name}] Memory consolidation error: {e}",
                file=sys.stderr,
            )


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--notify", action="store_true", help="Enable TTS notification"
        )
        parser.add_argument(
            "--debug", action="store_true", help="Enable debug output"
        )
        args = parser.parse_args()

        input_data = json.load(sys.stdin)

        teammate_name = input_data.get("teammate_name", "unknown")
        team_name = input_data.get("team_name", "unknown")
        transcript_path = input_data.get("transcript_path", "")
        project_cwd = input_data.get("cwd", "")

        if args.debug:
            print(
                f"DEBUG: TeammateIdle fired: {teammate_name} in {team_name}",
                file=sys.stderr,
            )

        # Run memory consolidation for this teammate's work
        consolidate_teammate_memory(
            teammate_name=teammate_name,
            team_name=team_name,
            transcript_path=transcript_path,
            cwd=project_cwd,
            debug=args.debug,
        )

        # Log the idle event
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "teammate_idle.json")

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
                "teammate_name": teammate_name,
                "team_name": team_name,
                "has_transcript": bool(transcript_path),
            }
        )

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        # Exit 0 = normal (teammate goes idle as expected)
        # Exit 2 would keep them working, but we don't want that by default
        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
