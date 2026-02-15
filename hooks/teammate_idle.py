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


def _try_daemon(transcript_path: str, cwd: str, phases: list, debug: bool = False) -> bool:
    """Try to delegate to the consolidation daemon on port 8083.

    Returns True if daemon accepted the request, False if unavailable.
    Uses specific phases to avoid suppressing stop.py's full consolidation.
    """
    try:
        import urllib.request
        payload = json.dumps({
            "cwd": cwd or str(Path.cwd()),
            "transcript_path": transcript_path or "",
            "phases": phases,
        }).encode('utf-8')
        req = urllib.request.Request(
            "http://localhost:8083/consolidate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method='POST'
        )
        urllib.request.urlopen(req, timeout=3)
        if debug:
            print("DEBUG: [teammate] Delegated to consolidation daemon", file=sys.stderr)
        return True
    except Exception as e:
        if debug:
            print(f"DEBUG: [teammate] Daemon unavailable ({e}), falling back to local", file=sys.stderr)
        return False


def _update_episodic(memory_dir: Path, teammate_name: str, team_name: str,
                     transcript_path: str, debug: bool = False):
    """Update episodic memory with teammate milestones. Runs locally (needs file write)."""
    transcript_processor = memory_dir / "transcript_processor.py"
    if not transcript_path or not transcript_processor.exists():
        return
    try:
        result = subprocess.run(
            ["python", str(transcript_processor), transcript_path, "--milestones-md"],
            capture_output=True, text=True, timeout=30, cwd=str(memory_dir),
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
                with open(episodic_file, "w", encoding="utf-8") as f:
                    f.write(f"# {today}\n{entry}")

            if debug:
                print(f"DEBUG: [{teammate_name}] Episodic updated", file=sys.stderr)
    except Exception as e:
        if debug:
            print(f"DEBUG: [{teammate_name}] Episodic error: {e}", file=sys.stderr)


def consolidate_teammate_memory(
    teammate_name: str,
    team_name: str,
    transcript_path: str = None,
    cwd: str = None,
    debug: bool = False,
):
    """
    Run memory consolidation for a teammate's work.

    DAEMON-FIRST: Tries consolidation daemon with lightweight+core phases.
    Falls back to local subprocess pipeline if daemon is unavailable.
    Episodic update always runs locally (daemon can't write host files).

    Skips attestations — those are session-level, not per-teammate.
    """
    try:
        project_dir = cwd if cwd else str(Path.cwd())
        if "Moltbook" not in project_dir and "moltbook" not in project_dir.lower():
            if debug:
                print("DEBUG: Not in Moltbook project, skipping", file=sys.stderr)
            return

        memory_dir = get_memory_dir(cwd)
        if not memory_dir.exists():
            if debug:
                print(f"DEBUG: Memory dir not found: {memory_dir}", file=sys.stderr)
            return

        # Try daemon first — lightweight+core (skips attestation/finalize/enrichment)
        if _try_daemon(transcript_path, cwd, ["lightweight", "core"], debug):
            # Daemon handles transcript + buffer + co-occurrences.
            # Episodic still needs local file write.
            _update_episodic(memory_dir, teammate_name, team_name, transcript_path, debug)
            return

        # ===== FALLBACK: Local subprocess pipeline =====
        if debug:
            print(f"DEBUG: [{teammate_name}] Running local fallback pipeline", file=sys.stderr)

        auto_memory = memory_dir / "auto_memory_hook.py"
        transcript_processor = memory_dir / "transcript_processor.py"

        # 1. Process teammate's transcript for thought memories
        if transcript_path and transcript_processor.exists():
            try:
                result = subprocess.run(
                    ["python", str(transcript_processor), transcript_path],
                    capture_output=True, text=True, timeout=30, cwd=str(memory_dir),
                )
                if debug:
                    print(f"DEBUG: [{teammate_name}] Transcript processing: {result.stdout[:500]}", file=sys.stderr)
            except Exception as e:
                if debug:
                    print(f"DEBUG: [{teammate_name}] Transcript error: {e}", file=sys.stderr)

        # 2. Run short-term buffer consolidation
        if auto_memory.exists():
            try:
                result = subprocess.run(
                    ["python", str(auto_memory), "--stop"],
                    capture_output=True, text=True, timeout=10, cwd=str(memory_dir),
                )
                if debug:
                    print(f"DEBUG: [{teammate_name}] Buffer consolidation: {result.stdout}", file=sys.stderr)
            except Exception as e:
                if debug:
                    print(f"DEBUG: [{teammate_name}] Buffer error: {e}", file=sys.stderr)

        # 3. Save co-occurrences in-process
        try:
            if str(memory_dir) not in sys.path:
                sys.path.insert(0, str(memory_dir))
            from co_occurrence import end_session_cooccurrence
            result = end_session_cooccurrence()
            if debug:
                print(f"DEBUG: [{teammate_name}] co-occurrences saved: {len(result)} new links", file=sys.stderr)
        except Exception as e:
            if debug:
                print(f"DEBUG: [{teammate_name}] co-occurrence error: {e}", file=sys.stderr)

        # 4. Update episodic memory with teammate's milestones
        _update_episodic(memory_dir, teammate_name, team_name, transcript_path, debug)

    except Exception as e:
        if debug:
            print(f"DEBUG: [{teammate_name}] Memory consolidation error: {e}", file=sys.stderr)


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
