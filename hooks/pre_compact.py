#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["python-dotenv"]
# ///
"""
Pre-Compact Hook - Context Compaction Logging

Logs when context is about to be compacted.
Optionally backs up transcript before compaction.
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime


def main():
    # Read hook input
    try:
        hook_input = json.loads(sys.stdin.read())
        transcript_path = hook_input.get("transcriptPath")
        reason = hook_input.get("reason", "unknown")
    except:
        transcript_path = None
        reason = "unknown"

    # Check for --backup flag in command line args
    backup = "--backup" in sys.argv

    if backup and transcript_path:
        transcript_file = Path(transcript_path)
        if transcript_file.exists():
            # Create backup directory
            backup_dir = transcript_file.parent / "transcript_backups"
            backup_dir.mkdir(exist_ok=True)

            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"{transcript_file.stem}_{timestamp}_precompact{transcript_file.suffix}"

            try:
                shutil.copy2(transcript_file, backup_file)
                print(f"Transcript backed up to: {backup_file}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Failed to backup transcript: {e}", file=sys.stderr)

    # Log the compaction event
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "pre_compact.json"

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "reason": reason,
        "transcript_path": str(transcript_path) if transcript_path else None,
        "backed_up": backup
    }

    logs = []
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except:
            logs = []

    logs.append(log_entry)

    # Keep only last 50 entries
    if len(logs) > 50:
        logs = logs[-50:]

    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2)

    output = {
        "hookSpecificOutput": {
            "status": "success",
            "reason": reason,
            "backed_up": backup
        }
    }

    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
