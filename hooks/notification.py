#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "python-dotenv",
#     "pyyaml",
#     "psycopg2-binary",
# ]
# ///

import argparse
import json
import os
import sys
import subprocess
import random
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load .env from ~/.claude directory
    claude_dir = Path.home() / ".claude"
    env_file = claude_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()  # Fallback to default
except ImportError:
    pass  # dotenv is optional


def _resolve_telegram_bot(cwd=None):
    """Find the correct telegram_bot.py for the running project."""
    project_dir = cwd or str(Path.cwd())
    base = Path("Q:/Codings/ClaudeCodeProjects/LEX")

    if "Moltbook2" in project_dir:
        own, other = base / "Moltbook2", base / "Moltbook"
    else:
        own, other = base / "Moltbook", base / "Moltbook2"

    for proj in [own, other]:
        for sub in ["telegram_bot.py", "memory/telegram_bot.py"]:
            p = proj / sub
            if p.exists():
                return p
    return None


def send_telegram(text, cwd=None):
    """Send a short Telegram notification via the correct project's bot."""
    try:
        import subprocess
        bot = _resolve_telegram_bot(cwd)
        if bot:
            subprocess.run(
                ["python", str(bot), "send", text],
                timeout=10,
                capture_output=True,
            )
    except Exception:
        pass  # Never break the hook for a notification


def get_tts_script_path():
    """
    Determine which TTS script to use based on available API keys.
    Priority order: ElevenLabs > OpenAI > pyttsx3
    """
    # Get current script directory and construct utils/tts path
    script_dir = Path(__file__).parent
    tts_dir = script_dir / "utils" / "tts"
    
    # Check for ElevenLabs API key (highest priority)
    if os.getenv('ELEVENLABS_API_KEY'):
        elevenlabs_script = tts_dir / "elevenlabs_tts.py"
        if elevenlabs_script.exists():
            return str(elevenlabs_script)
    
    # Check for OpenAI API key (second priority)
    if os.getenv('OPENAI_API_KEY'):
        openai_script = tts_dir / "openai_tts.py"
        if openai_script.exists():
            return str(openai_script)
    
    # Fall back to pyttsx3 (no API key required)
    pyttsx3_script = tts_dir / "pyttsx3_tts.py"
    if pyttsx3_script.exists():
        return str(pyttsx3_script)
    
    return None


def announce_notification():
    """Announce that the agent needs user input."""
    try:
        tts_script = get_tts_script_path()
        if not tts_script:
            # Debug: log why no TTS script was found
            log_dir = Path.cwd() / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / 'tts_debug.log', 'a') as f:
                f.write(f"No TTS script found. ELEVENLABS_API_KEY set: {bool(os.getenv('ELEVENLABS_API_KEY'))}\n")
            return  # No TTS scripts available

        # Get engineer name if available
        engineer_name = os.getenv('ENGINEER_NAME', '').strip()

        # Create notification message with 30% chance to include name
        if engineer_name and random.random() < 0.3:
            notification_message = f"{engineer_name}, your agent needs your input"
        else:
            notification_message = "Your agent needs your input"

        # Call the TTS script with the notification message
        result = subprocess.run([
            "uv", "run", tts_script, notification_message
        ],
        capture_output=True,
        text=True,
        timeout=15  # 15-second timeout for ElevenLabs
        )

        # Debug: log result
        log_dir = Path.cwd() / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / 'tts_debug.log', 'a') as f:
            f.write(f"TTS script: {tts_script}\n")
            f.write(f"Return code: {result.returncode}\n")
            if result.stderr:
                f.write(f"Stderr: {result.stderr}\n")

    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError) as e:
        log_dir = Path.cwd() / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / 'tts_debug.log', 'a') as f:
            f.write(f"TTS Error: {type(e).__name__}: {e}\n")
    except Exception as e:
        log_dir = Path.cwd() / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / 'tts_debug.log', 'a') as f:
            f.write(f"TTS Exception: {type(e).__name__}: {e}\n")


def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--notify', action='store_true', help='Enable TTS notifications')
        args = parser.parse_args()
        
        # Read JSON input from stdin
        input_data = json.loads(sys.stdin.read())
        
        # Ensure log directory exists
        import os
        log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'notification.json')
        
        # Read existing log data or initialize empty list
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                try:
                    log_data = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    log_data = []
        else:
            log_data = []
        
        # Append new data
        log_data.append(input_data)
        
        # Write back to file with formatting
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        # Announce notification via TTS when --notify flag is set
        # Debug: always log to see if this code path is reached
        debug_log = Path.home() / ".claude" / "logs" / "tts_debug.log"
        debug_log.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_log, 'a') as f:
            f.write(f"args.notify={args.notify}, message={input_data.get('message', 'N/A')}\n")

        if args.notify:
            announce_notification()

        # Send Telegram notification so Lex sees it on phone
        message = input_data.get('message', 'Agent needs input')
        cwd = input_data.get('cwd', '')
        from datetime import datetime
        now = datetime.now().strftime('%H:%M UTC')
        send_telegram(f'Agent needs input ({now}): {message[:200]}', cwd=cwd)

        sys.exit(0)
        
    except json.JSONDecodeError:
        # Handle JSON decode errors gracefully
        sys.exit(0)
    except Exception:
        # Handle any other errors gracefully
        sys.exit(0)

if __name__ == '__main__':
    main()