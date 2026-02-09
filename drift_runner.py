"""Drift Runner — Async autonomy via Telegram.

Run this instead of 'claude' directly. It:
1. Starts Claude Code (--continue to resume context)
2. When Claude stops, sends a Telegram summary to Lex
3. Polls Telegram for Lex's reply
4. Feeds the reply back to Claude as a new prompt
5. Loops until Lex sends 'stop' or 'exit'

Usage:
    python drift_runner.py              # Start fresh
    python drift_runner.py --continue   # Resume last conversation
    python drift_runner.py --auto       # Auto-continue without waiting (dangerous)
"""
import subprocess
import sys
import os
import time
import json
import signal
from pathlib import Path
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')

# Add memory dir to path for telegram_bot
sys.path.insert(0, str(Path(__file__).parent / 'memory'))

WORK_DIR = Path(__file__).parent
CLAUDE_CMD = 'claude'  # Assumes claude is on PATH


def send_telegram(text):
    """Send a message to Lex via Telegram."""
    try:
        from telegram_bot import send_message
        return send_message(text)
    except Exception as e:
        print(f'[runner] Telegram send failed: {e}')
        return False


def poll_telegram(timeout_minutes=None):
    """Poll Telegram for a reply from Lex. Returns message text or None."""
    try:
        from telegram_bot import get_unread_messages
    except ImportError:
        print('[runner] telegram_bot not found')
        return None

    start = time.time()
    print('[runner] Waiting for Telegram reply...')
    print('[runner] (Lex can send a message to @driftcornwallbot)')

    while True:
        messages = get_unread_messages()
        if messages:
            # Return the most recent message
            latest = messages[-1]
            print(f'[runner] Received: {latest["text"][:100]}')
            return latest['text']

        # Check timeout
        if timeout_minutes and (time.time() - start) > timeout_minutes * 60:
            print(f'[runner] Timeout after {timeout_minutes} minutes')
            return None

        # Poll interval: 5 seconds
        time.sleep(5)


def extract_summary_from_output(output):
    """Extract session summary from Claude's output."""
    lines = output.strip().split('\n')

    # Look for summary-like content near the end
    summary_lines = []
    in_summary = False

    for line in reversed(lines[-50:]):  # Check last 50 lines
        line_stripped = line.strip()
        if not line_stripped:
            if in_summary:
                break
            continue

        # Detect summary markers
        lower = line_stripped.lower()
        is_milestone = any(w in lower for w in [
            'shipped', 'pushed', 'posted', 'published', 'built',
            'fixed', 'blocked', 'committed', 'deployed', 'error',
            'summary', 'done', 'complete', 'ready', 'waiting'
        ])

        if is_milestone or in_summary:
            summary_lines.insert(0, line_stripped)
            in_summary = True

    if summary_lines:
        return '\n'.join(summary_lines[-15:])  # Last 15 relevant lines

    # Fallback: just return last 10 non-empty lines
    last_lines = [l.strip() for l in lines if l.strip()][-10:]
    return '\n'.join(last_lines) if last_lines else 'Session ended (no output captured)'


def run_claude(prompt=None, continue_session=True):
    """Run Claude Code and capture output."""
    cmd = [CLAUDE_CMD]

    if continue_session:
        cmd.append('--continue')

    if prompt:
        cmd.extend(['-p', prompt])

    print(f'\n[runner] Starting: {" ".join(cmd[:3])}...')
    print(f'[runner] Working dir: {WORK_DIR}')
    print('=' * 60)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(WORK_DIR),
            capture_output=False,  # Let output flow to terminal
            text=True,
            timeout=3600  # 1 hour max
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print('[runner] Claude timed out after 1 hour')
        return -1
    except KeyboardInterrupt:
        print('\n[runner] Interrupted by user')
        return -2
    except Exception as e:
        print(f'[runner] Error: {e}')
        return -3


def main():
    args = sys.argv[1:]
    continue_session = '--continue' in args or '-c' in args
    auto_mode = '--auto' in args
    initial_prompt = None

    # Extract --prompt or -p value
    for flag in ['--prompt', '-p']:
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                initial_prompt = args[idx + 1]

    print('=' * 60)
    print('  DRIFT RUNNER — Async Autonomy via Telegram')
    print('=' * 60)
    print(f'  Mode: {"auto" if auto_mode else "telegram-directed"}')
    print(f'  Continue: {continue_session}')
    print(f'  Time: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}')
    print('=' * 60)

    # Send startup notification
    send_telegram('Drift Runner started. I will notify you when each session ends.')

    prompt = initial_prompt
    loop_count = 0

    while True:
        loop_count += 1

        # Run Claude
        exit_code = run_claude(prompt=prompt, continue_session=(continue_session or loop_count > 1))

        # Session ended
        now = datetime.now(timezone.utc).strftime('%H:%M UTC')

        if exit_code == -2:
            # Keyboard interrupt — exit cleanly
            send_telegram(f'Drift Runner stopped by keyboard interrupt ({now})')
            break

        # Send Telegram notification
        notification = f'Drift session ended ({now})\n\nExit code: {exit_code}\nLoop: {loop_count}\n\nReply to continue, or send "stop" to shut down.'
        sent = send_telegram(notification)

        if not sent:
            print('[runner] Could not send Telegram notification. Stopping.')
            break

        if auto_mode:
            print('[runner] Auto-mode: continuing in 30 seconds...')
            time.sleep(30)
            prompt = None
            continue

        # Wait for Telegram reply
        reply = poll_telegram(timeout_minutes=480)  # 8 hour timeout

        if reply is None:
            send_telegram('No reply received after 8 hours. Drift Runner shutting down.')
            break

        reply_lower = reply.strip().lower()
        if reply_lower in ['stop', 'exit', 'quit', 'shutdown']:
            send_telegram('Drift Runner shutting down. See you next time.')
            break

        # Use the reply as the next prompt
        prompt = reply
        continue_session = True

    print('\n[runner] Drift Runner finished.')


if __name__ == '__main__':
    main()
