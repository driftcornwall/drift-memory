#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "python-dotenv",
# ]
# ///

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


# Memory system locations - check both Moltbook and Moltbook2
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


def log_pre_compact(input_data):
    """Log pre-compact event to logs directory."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'pre_compact.json'

    if log_file.exists():
        with open(log_file, 'r') as f:
            try:
                log_data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                log_data = []
    else:
        log_data = []

    log_data.append(input_data)

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)


def backup_transcript(transcript_path, trigger):
    """Create a backup of the transcript before compaction."""
    try:
        if not os.path.exists(transcript_path):
            return

        backup_dir = Path("logs") / "transcript_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = Path(transcript_path).stem
        backup_name = f"{session_name}_pre_compact_{trigger}_{timestamp}.jsonl"
        backup_path = backup_dir / backup_name

        import shutil
        shutil.copy2(transcript_path, backup_path)

        return str(backup_path)
    except Exception:
        return None


def process_memories_before_compaction(transcript_path: str, cwd: str = None):
    """
    Extract memories from transcript BEFORE compaction destroys them.

    When context compaction fires, thinking blocks and detailed tool
    results from earlier in the session get compressed into summaries.
    If we don't extract memories here, those thoughts are lost forever.

    The transcript_processor has hash-based dedup, so running it here
    AND again at session end (stop.py) is safe — no double-counting.

    Also runs save-pending to persist any co-occurrences accumulated
    during the session so far.
    """
    try:
        project_dir = cwd if cwd else str(Path.cwd())
        if "Moltbook" not in project_dir and "moltbook" not in project_dir.lower():
            return

        memory_dir = get_memory_dir(cwd)
        if not memory_dir.exists():
            return

        transcript_processor = memory_dir / "transcript_processor.py"
        memory_manager = memory_dir / "memory_manager.py"
        auto_memory = memory_dir / "auto_memory_hook.py"

        # 1. Process transcript for thought memories
        if transcript_path and transcript_processor.exists():
            try:
                subprocess.run(
                    ["python", str(transcript_processor), transcript_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(memory_dir)
                )
            except Exception:
                pass

        # 2. Consolidate short-term buffer
        if auto_memory.exists():
            try:
                subprocess.run(
                    ["python", str(auto_memory), "--stop"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(memory_dir)
                )
            except Exception:
                pass

        # 3. Save pending co-occurrences
        if memory_manager.exists():
            try:
                subprocess.run(
                    ["python", str(memory_manager), "save-pending"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(memory_dir)
                )
            except Exception:
                pass

        # 4. Mine lessons from ALL sources (v4.5)
        # Compaction = mid-session checkpoint. New learnings from MEMORY.md edits,
        # rejection patterns, and co-occurrence hubs should be captured NOW
        # before context compression potentially loses the reasoning that led to them.
        lesson_script = memory_dir / "lesson_extractor.py"
        if lesson_script.exists():
            for mine_cmd in ["mine-memory", "mine-rejections", "mine-hubs"]:
                try:
                    subprocess.run(
                        ["python", str(lesson_script), mine_cmd],
                        capture_output=True,
                        text=True,
                        timeout=15,
                        cwd=str(memory_dir)
                    )
                except Exception:
                    pass

    except Exception:
        pass  # Never break compaction


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--backup', action='store_true',
                          help='Create backup of transcript before compaction')
        parser.add_argument('--verbose', action='store_true',
                          help='Print verbose output')
        args = parser.parse_args()

        input_data = json.loads(sys.stdin.read())

        session_id = input_data.get('session_id', 'unknown')
        transcript_path = input_data.get('transcript_path', '')
        trigger = input_data.get('trigger', 'unknown')
        custom_instructions = input_data.get('custom_instructions', '')
        cwd = input_data.get('cwd', '')

        # Log the pre-compact event
        log_pre_compact(input_data)

        # CRITICAL: Process memories before compaction loses them
        # Thinking blocks and detailed tool results are about to be
        # compressed into summaries. Extract memories NOW.
        if transcript_path:
            process_memories_before_compaction(transcript_path, cwd)

        # Create backup if requested
        backup_path = None
        if args.backup and transcript_path:
            backup_path = backup_transcript(transcript_path, trigger)

        # Provide feedback
        message = "Pre-compaction memory extraction complete."
        if trigger == "auto":
            message = f"Auto-compaction: memories extracted before compression. {message}"

        print(message)

        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == '__main__':
    main()