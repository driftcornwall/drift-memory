#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["python-dotenv"]
# ///
"""
Notification Hook - TTS Announcements

Announces via TTS when agent needs input.
Detects available TTS system (ElevenLabs > OpenAI > pyttsx3).
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime


def get_tts_script():
    """Find available TTS script in utils/tts/ directory"""
    # Look in hook directory and parent
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
    # Check for --notify flag
    if "--notify" not in sys.argv:
        # TTS disabled
        sys.exit(0)

    # Read hook input
    try:
        hook_input = json.loads(sys.stdin.read())
        reason = hook_input.get("reason", "input_needed")
    except:
        reason = "input_needed"

    # Find TTS script
    tts_script = get_tts_script()
    if not tts_script:
        # No TTS available
        sys.exit(0)

    # Prepare message
    messages = {
        "input_needed": "Input needed",
        "task_complete": "Task complete",
        "error": "Error occurred",
    }

    message = messages.get(reason, "Notification")

    # Run TTS
    try:
        subprocess.run(
            ["python", str(tts_script), message],
            capture_output=True,
            timeout=5
        )
    except Exception:
        pass

    # Log notification
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "notification.json"

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "reason": reason,
        "message": message,
        "tts_script": str(tts_script)
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
            "message": message,
            "tts_available": True
        }
    }

    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
