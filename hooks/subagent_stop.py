#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["python-dotenv"]
# ///
"""
Subagent Stop Hook - Subagent Completion Notifications

Announces via TTS when subagent completes.
Optionally backs up transcript.
"""

import sys
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime


def get_tts_script():
    """Find available TTS script"""
    search_dirs = [
        Path(__file__).parent / "utils" / "tts",
        Path(__file__).parent.parent / "utils" / "tts",
    ]

    tts_priority = ["elevenlabs_tts.py", "openai_tts.py", "pyttsx3_tts.py"]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for tts_script in tts_priority:
            script_path = search_dir / tts_script
            if script_path.exists():
                return script_path

    return None


def main():
    # Read hook input
    try:
        hook_input = json.loads(sys.stdin.read())
        transcript_path = hook_input.get("transcriptPath")
        subagent_name = hook_input.get("subagentName", "Subagent")
    except:
        transcript_path = None
        subagent_name = "Subagent"

    # TTS announcement
    tts_script = get_tts_script()
    if tts_script:
        try:
            subprocess.run(
                ["python", str(tts_script), "Subagent Complete"],
                capture_output=True,
                timeout=5
            )
        except Exception:
            pass

    # Check for --chat flag for transcript backup
    backup = "--chat" in sys.argv

    if backup and transcript_path:
        transcript_file = Path(transcript_path)
        if transcript_file.exists():
            backup_dir = transcript_file.parent / "subagent_transcripts"
            backup_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"{subagent_name}_{timestamp}{transcript_file.suffix}"

            try:
                shutil.copy2(transcript_file, backup_file)
                print(f"Subagent transcript backed up: {backup_file}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Failed to backup transcript: {e}", file=sys.stderr)

    # Log event
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "subagent_stop.json"

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "subagent_name": subagent_name,
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

    if len(logs) > 50:
        logs = logs[-50:]

    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2)

    output = {
        "hookSpecificOutput": {
            "status": "success",
            "subagent_name": subagent_name,
            "backed_up": backup
        }
    }

    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
