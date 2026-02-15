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
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
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
        bot = _resolve_telegram_bot(cwd)
        if bot:
            subprocess.run(
                ["python", str(bot), "send", text],
                timeout=10,
                capture_output=True,
            )
    except Exception:
        pass  # Never break the hook for a notification


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


def _run_script(memory_dir, script_name, args, timeout=15):
    """Run a memory script and return (returncode, stdout, stderr)."""
    script = memory_dir / script_name
    if not script.exists():
        return (-1, "", f"{script_name} not found")
    try:
        result = subprocess.run(
            ["python", str(script)] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(memory_dir)
        )
        return (result.returncode, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return (-2, "", f"{script_name} timed out after {timeout}s")
    except Exception as e:
        return (-3, "", str(e))


def _try_daemon(transcript_path: str, cwd: str, phases: list) -> bool:
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
        return True
    except Exception:
        return False


def process_memories_before_compaction(transcript_path: str, cwd: str = None):
    """
    Extract memories from transcript BEFORE compaction destroys them.

    DAEMON-FIRST: Tries consolidation daemon with lightweight+core+enrichment phases.
    Falls back to local subprocess pipeline if daemon is unavailable.

    The transcript_processor has hash-based dedup, so running it here
    AND again at session end (stop.py) is safe — no double-counting.
    """
    try:
        project_dir = cwd if cwd else str(Path.cwd())
        if "Moltbook" not in project_dir and "moltbook" not in project_dir.lower():
            return

        memory_dir = get_memory_dir(cwd)
        if not memory_dir.exists():
            return

        # Try daemon first — lightweight+core+enrichment (skips attestation/finalize)
        if _try_daemon(transcript_path, cwd, ["lightweight", "core", "enrichment"]):
            return  # Daemon handles it

        # ===== FALLBACK: Local subprocess pipeline =====

        # Phase 1: transcript + consolidation + save-pending (all independent)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = []
            if transcript_path:
                futures.append(pool.submit(
                    _run_script, memory_dir, "transcript_processor.py",
                    [transcript_path], 30
                ))
            futures.append(pool.submit(
                _run_script, memory_dir, "auto_memory_hook.py", ["--stop"], 10
            ))
            # Save co-occurrences in-process (save-pending was removed in DB migration)
            def _save_cooccurrences(mdir):
                try:
                    if str(mdir) not in sys.path:
                        sys.path.insert(0, str(mdir))
                    from co_occurrence import end_session_cooccurrence
                    result = end_session_cooccurrence()
                    return (0, f"Co-occurrences saved: {len(result)} new links", "")
                except Exception as e:
                    return (-3, "", f"Co-occurrence error: {e}")
            futures.append(pool.submit(_save_cooccurrences, memory_dir))

        # Phase 2: lesson mining x3 (after phase 1, all independent)
        with ThreadPoolExecutor(max_workers=3) as pool:
            pool.submit(_run_script, memory_dir, "lesson_extractor.py", ["mine-memory"], 15)
            pool.submit(_run_script, memory_dir, "lesson_extractor.py", ["mine-rejections"], 15)
            pool.submit(_run_script, memory_dir, "lesson_extractor.py", ["mine-hubs"], 15)

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

        transcript_path = input_data.get('transcript_path', '')
        trigger = input_data.get('trigger', 'unknown')
        cwd = input_data.get('cwd', '')

        # Run log + telegram + memory extraction + backup in parallel
        # Telegram and logging are independent of memory processing
        with ThreadPoolExecutor(max_workers=4) as pool:
            pool.submit(log_pre_compact, input_data)

            now = datetime.now().strftime('%H:%M UTC')
            pool.submit(
                send_telegram,
                f'Context compacting ({now}) — extracting memories before compression',
                cwd
            )

            if transcript_path:
                pool.submit(process_memories_before_compaction, transcript_path, cwd)

            if args.backup and transcript_path:
                pool.submit(backup_transcript, transcript_path, trigger)

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