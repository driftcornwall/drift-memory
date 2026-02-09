#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "python-dotenv",
# ]
# ///

"""
Stop hook for Claude Code.
Handles session end: logging, transcript copying, TTS announcement.

DRIFT MEMORY INTEGRATION (2026-02-01):
Added automatic memory consolidation when session ends.
This is the "sleep consolidation" phase where salient short-term items
are moved to long-term memory and decay is applied.
"""

import argparse
import json
import os
import sys
import random
import subprocess
from pathlib import Path
from datetime import datetime

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


# Memory system locations - check both Moltbook and Moltbook2
MOLTBOOK_DIRS = [
    Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook2/memory"),
    Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook/memory"),
]

def get_memory_dir(cwd: str = None) -> Path:
    """Get the appropriate memory directory based on cwd.

    Args:
        cwd: Optional working directory (uses Path.cwd() if not provided)
    """
    project_dir = cwd if cwd else str(Path.cwd())
    if "Moltbook2" in project_dir:
        return MOLTBOOK_DIRS[0]
    elif "Moltbook" in project_dir:
        return MOLTBOOK_DIRS[1]
    # Default to first that exists
    for d in MOLTBOOK_DIRS:
        if d.exists():
            return d
    return MOLTBOOK_DIRS[0]


def is_moltbook_project() -> bool:
    """Check if we're working in the Moltbook project."""
    cwd = Path.cwd()
    return "Moltbook" in str(cwd) or "moltbook" in str(cwd).lower()


def update_episodic_memory(memory_dir: Path, transcript_path: str = None, debug: bool = False):
    """
    Update episodic memory with session summary.
    Creates or appends to memory/episodic/YYYY-MM-DD.md

    The folder is the brain, but only if you use it.

    SMART EXTRACTION (2026-02-02):
    - Uses transcript_processor.py to extract ONLY from assistant output
    - Looks for milestone keywords (shipped, launched, deployed, etc.)
    - Only writes to episodic if there's something meaningful to record

    DEDUP GUARD (2026-02-04):
    - Filters out milestone sentences that already appear in the episodic file
    - Prevents the same milestones from being appended on every session-end
    """
    try:
        # Only run if we have a transcript to analyze
        if not transcript_path:
            if debug:
                print("DEBUG: No transcript path, skipping episodic update", file=sys.stderr)
            return

        transcript_processor = memory_dir / "transcript_processor.py"
        if not transcript_processor.exists():
            if debug:
                print(f"DEBUG: transcript_processor.py not found at {transcript_processor}", file=sys.stderr)
            return

        # Extract milestones using smart parser (only looks at assistant output)
        result = subprocess.run(
            ["python", str(transcript_processor), transcript_path, "--milestones-md"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(memory_dir)
        )

        if result.returncode != 0:
            if debug:
                print(f"DEBUG: Milestone extraction failed: {result.stderr}", file=sys.stderr)
            return

        milestone_md = result.stdout.strip()

        # Only update episodic if there are actual milestones
        if not milestone_md or milestone_md == "":
            if debug:
                print("DEBUG: No milestones found, skipping episodic update", file=sys.stderr)
            return

        # We have milestones! But first check for duplicates
        episodic_dir = memory_dir / "episodic"
        episodic_dir.mkdir(exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")
        session_time = datetime.now().strftime("%H:%M UTC")
        episodic_file = episodic_dir / f"{today}.md"

        # DEDUP GUARD: Read existing content and filter out already-recorded milestones
        existing_content = ""
        if episodic_file.exists():
            existing_content = episodic_file.read_text(encoding='utf-8')

        if existing_content:
            # Parse milestone_md into individual milestone blocks
            # Format: "**[keywords]**\n- sentence\n- sentence\n\n"
            milestone_blocks = []
            current_block = []
            for line in milestone_md.split('\n'):
                if line.startswith('**[') and current_block:
                    milestone_blocks.append('\n'.join(current_block))
                    current_block = [line]
                elif line.strip():
                    current_block.append(line)
                elif current_block and not line.startswith('###'):
                    current_block.append(line)
            if current_block:
                milestone_blocks.append('\n'.join(current_block))

            # Filter: only keep blocks whose "- sentence" lines are NOT already in the file
            new_blocks = []
            for block in milestone_blocks:
                # Extract the actual content lines (- sentence)
                content_lines = [l.strip() for l in block.split('\n') if l.strip().startswith('- ')]
                if not content_lines:
                    continue
                # A block is "new" if ANY of its content lines are not in the existing file
                is_new = any(line not in existing_content for line in content_lines)
                if is_new:
                    new_blocks.append(block)

            if not new_blocks:
                if debug:
                    print("DEBUG: All milestones already recorded, skipping", file=sys.stderr)
                return

            # Reconstruct milestone_md with only new blocks
            milestone_md = "### Session Milestones (auto-extracted)\n\n" + '\n\n'.join(new_blocks)

        entry = f"\n## Session End (~{session_time})\n\n{milestone_md}\n"

        if episodic_file.exists():
            with open(episodic_file, 'a', encoding='utf-8') as f:
                f.write(entry)
        else:
            header = f"# {today}\n"
            with open(episodic_file, 'w', encoding='utf-8') as f:
                f.write(header + entry)

        if debug:
            print(f"DEBUG: Wrote {len(new_blocks) if existing_content else 'all'} new milestones to {episodic_file}", file=sys.stderr)

    except Exception as e:
        if debug:
            print(f"DEBUG: Episodic memory error: {e}", file=sys.stderr)


def consolidate_drift_memory(transcript_path: str = None, cwd: str = None, debug: bool = False):
    """
    Run Drift's memory consolidation at session end.
    This is the "sleep" phase where:
    1. Short-term buffer is decayed
    2. High-salience items are moved to long-term
    3. Co-occurrences are logged
    4. Transcript is processed for thought memories
    5. Episodic memory is updated (NEW!)

    Args:
        transcript_path: Path to session transcript for milestone extraction
        cwd: Working directory from hook input (more reliable than Path.cwd())
        debug: Enable debug output

    Fails gracefully - should never break the stop hook.
    """
    try:
        # Use cwd from input if provided, otherwise fall back to Path.cwd()
        project_dir = cwd if cwd else str(Path.cwd())
        if "Moltbook" not in project_dir and "moltbook" not in project_dir.lower():
            return

        memory_dir = get_memory_dir(cwd)
        if not memory_dir.exists():
            return

        auto_memory = memory_dir / "auto_memory_hook.py"
        memory_manager = memory_dir / "memory_manager.py"
        transcript_processor = memory_dir / "transcript_processor.py"

        # NEW: Process transcript for thought memories
        if transcript_path and transcript_processor.exists():
            try:
                result = subprocess.run(
                    ["python", str(transcript_processor), transcript_path],
                    capture_output=True,
                    text=True,
                    timeout=30,  # Longer timeout for transcript processing
                    cwd=str(memory_dir)
                )
                if debug:
                    print(f"DEBUG: Transcript processing: {result.stdout[:500]}", file=sys.stderr)
                if result.returncode == 0:
                    # Log what was extracted
                    try:
                        import json
                        summary = json.loads(result.stdout)
                        count = summary.get('total_extracted', 0)
                        if debug:
                            print(f"DEBUG: Extracted {count} thought memories", file=sys.stderr)
                    except:
                        pass
            except Exception as e:
                if debug:
                    print(f"DEBUG: Transcript processing error: {e}", file=sys.stderr)

        # Run consolidation from auto_memory_hook
        if auto_memory.exists():
            try:
                result = subprocess.run(
                    ["python", str(auto_memory), "--stop"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(memory_dir)
                )
                if debug:
                    print(f"DEBUG: Consolidation output: {result.stdout}", file=sys.stderr)
                    if result.stderr:
                        print(f"DEBUG: Consolidation stderr: {result.stderr}", file=sys.stderr)
            except Exception as e:
                if debug:
                    print(f"DEBUG: Consolidation error: {e}", file=sys.stderr)

        # v3.7/v2.16: Use save-pending for FAST session end (deferred co-occurrence processing)
        # Co-occurrences will be calculated at next session start when there's time
        if memory_manager.exists():
            try:
                result = subprocess.run(
                    ["python", str(memory_manager), "save-pending"],
                    capture_output=True,
                    text=True,
                    timeout=5,  # Should be very fast now
                    cwd=str(memory_dir)
                )
                if debug:
                    print(f"DEBUG: save-pending output: {result.stdout}", file=sys.stderr)
            except Exception as e:
                if debug:
                    print(f"DEBUG: save-pending error: {e}", file=sys.stderr)

        # v4.4: Run session maintenance (increment sessions_since_recall, decay candidates)
        # CRITICAL: Without this, sessions_since_recall stays at 0, breaking grace periods,
        # excavation, and decay. This was a bug from 2026-02-01 to 2026-02-09.
        if memory_manager.exists():
            try:
                result = subprocess.run(
                    ["python", str(memory_manager), "maintenance"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(memory_dir)
                )
                if debug:
                    print(f"DEBUG: maintenance output: {result.stdout}", file=sys.stderr)
            except Exception as e:
                if debug:
                    print(f"DEBUG: maintenance error: {e}", file=sys.stderr)

        # NEW: Update episodic memory with session summary
        update_episodic_memory(memory_dir, transcript_path, debug)

        # NEW: Extract and store session summaries (the rich summaries given to Lex)
        if transcript_path and transcript_processor.exists():
            try:
                result = subprocess.run(
                    ["python", str(transcript_processor), transcript_path, "--store-summary"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(memory_dir)
                )
                if debug and result.returncode == 0:
                    print(f"DEBUG: Summary extraction: {result.stdout[:300]}", file=sys.stderr)
            except Exception as e:
                if debug:
                    print(f"DEBUG: Summary extraction error: {e}", file=sys.stderr)

        # LESSON MINING: Auto-extract lessons from ALL sources (v4.5)
        # Mine MEMORY.md, rejection patterns, and co-occurrence hubs
        lesson_script = memory_dir / "lesson_extractor.py"
        if lesson_script.exists():
            for mine_cmd in ["mine-memory", "mine-rejections", "mine-hubs"]:
                try:
                    result = subprocess.run(
                        ["python", str(lesson_script), mine_cmd],
                        capture_output=True,
                        text=True,
                        timeout=15,
                        cwd=str(memory_dir)
                    )
                    if debug:
                        print(f"DEBUG: Lesson {mine_cmd}: {result.stdout[:200]}", file=sys.stderr)
                except Exception as e:
                    if debug:
                        print(f"DEBUG: Lesson {mine_cmd} error: {e}", file=sys.stderr)

        # MERKLE ATTESTATION: Last step before sleep
        # Compute chain-linked merkle root of all memories and save locally.
        # This is the "seal" on the session - proves memory state at sleep time.
        # Nostr publish happens at next session start (more time available).
        merkle_script = memory_dir / "merkle_attestation.py"
        if merkle_script.exists():
            try:
                result = subprocess.run(
                    ["python", str(merkle_script), "generate-chain"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    cwd=str(memory_dir)
                )
                if debug:
                    print(f"DEBUG: Merkle attestation: {result.stdout}", file=sys.stderr)
                    if result.stderr:
                        print(f"DEBUG: Merkle stderr: {result.stderr}", file=sys.stderr)
            except Exception as e:
                if debug:
                    print(f"DEBUG: Merkle attestation error: {e}", file=sys.stderr)

        # COGNITIVE FINGERPRINT: Layer 2 attestation
        # Compute cognitive topology hash from co-occurrence graph and save.
        fingerprint_script = memory_dir / "cognitive_fingerprint.py"
        if fingerprint_script.exists():
            try:
                result = subprocess.run(
                    ["python", str(fingerprint_script), "attest"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    cwd=str(memory_dir)
                )
                if debug:
                    print(f"DEBUG: Cognitive fingerprint: {result.stdout}", file=sys.stderr)
                    if result.stderr:
                        print(f"DEBUG: Fingerprint stderr: {result.stderr}", file=sys.stderr)
            except Exception as e:
                if debug:
                    print(f"DEBUG: Cognitive fingerprint error: {e}", file=sys.stderr)

        # TASTE ATTESTATION: Layer 3 attestation
        # Compute taste hash from rejection log and save.
        rejection_script = memory_dir / "rejection_log.py"
        if rejection_script.exists():
            try:
                result = subprocess.run(
                    ["python", str(rejection_script), "attest"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    cwd=str(memory_dir)
                )
                if debug:
                    print(f"DEBUG: Taste attestation: {result.stdout}", file=sys.stderr)
                    if result.stderr:
                        print(f"DEBUG: Taste stderr: {result.stderr}", file=sys.stderr)
            except Exception as e:
                if debug:
                    print(f"DEBUG: Taste attestation error: {e}", file=sys.stderr)

    except Exception as e:
        # Memory consolidation should NEVER break the stop hook
        if debug:
            print(f"DEBUG: Memory system error: {e}", file=sys.stderr)


def get_completion_messages():
    """Return list of friendly completion messages."""
    return [
        "Work complete!",
        "All done!",
        "Task finished!",
        "Job complete!",
        "Ready for next task!"
    ]


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


def get_llm_completion_message():
    """
    Generate completion message using available LLM services.
    Priority order: OpenAI > Anthropic > Ollama > fallback to random message

    Returns:
        str: Generated or fallback completion message
    """
    # Get current script directory and construct utils/llm path
    script_dir = Path(__file__).parent
    llm_dir = script_dir / "utils" / "llm"

    # Try OpenAI first (highest priority)
    if os.getenv('OPENAI_API_KEY'):
        oai_script = llm_dir / "oai.py"
        if oai_script.exists():
            try:
                result = subprocess.run([
                    "uv", "run", str(oai_script), "--completion"
                ],
                capture_output=True,
                text=True,
                timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass

    # Try Anthropic second
    if os.getenv('ANTHROPIC_API_KEY'):
        anth_script = llm_dir / "anth.py"
        if anth_script.exists():
            try:
                result = subprocess.run([
                    "uv", "run", str(anth_script), "--completion"
                ],
                capture_output=True,
                text=True,
                timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass

    # Try Ollama third (local LLM)
    ollama_script = llm_dir / "ollama.py"
    if ollama_script.exists():
        try:
            result = subprocess.run([
                "uv", "run", str(ollama_script), "--completion"
            ],
            capture_output=True,
            text=True,
            timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass

    # Fallback to random predefined message
    messages = get_completion_messages()
    return random.choice(messages)

def announce_completion():
    """Announce completion using the best available TTS service."""
    try:
        tts_script = get_tts_script_path()
        if not tts_script:
            return  # No TTS scripts available

        # Get completion message (LLM-generated or fallback)
        completion_message = get_llm_completion_message()

        # Call the TTS script with the completion message
        subprocess.run([
            "uv", "run", tts_script, completion_message
        ],
        capture_output=True,  # Suppress output
        timeout=10  # 10-second timeout
        )

    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        # Fail silently if TTS encounters issues
        pass
    except Exception:
        # Fail silently for any other errors
        pass


def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--chat', action='store_true', help='Copy transcript to chat.json')
        parser.add_argument('--notify', action='store_true', help='Enable TTS completion announcement')
        parser.add_argument('--debug', action='store_true', help='Enable debug output')
        args = parser.parse_args()

        # Read JSON input from stdin
        input_data = json.load(sys.stdin)

        # === DRIFT MEMORY CONSOLIDATION ===
        # Run memory consolidation before anything else
        # Pass transcript_path and cwd so we can extract thought memories
        transcript_path = input_data.get("transcript_path", "")
        project_cwd = input_data.get("cwd", "")
        consolidate_drift_memory(transcript_path=transcript_path, cwd=project_cwd, debug=args.debug)
        # === END MEMORY CONSOLIDATION ===

        # === TELEGRAM NOTIFICATION ===
        try:
            telegram_bot_path = None
            for candidate in [
                Path(project_cwd) / "telegram_bot.py",
                Path(project_cwd) / "memory" / "telegram_bot.py",
                Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook2") / "telegram_bot.py",
                Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook") / "telegram_bot.py",
            ]:
                if candidate.exists():
                    telegram_bot_path = candidate
                    break

            if telegram_bot_path:
                # Extract last assistant output blocks from transcript for summary
                summary_lines = []
                if transcript_path and os.path.exists(transcript_path):
                    with open(transcript_path, 'r', encoding='utf-8', errors='replace') as tf:
                        for line in tf:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entry = json.loads(line)
                                if entry.get('type') == 'assistant' and entry.get('message', {}).get('role') == 'assistant':
                                    for block in entry.get('message', {}).get('content', []):
                                        if block.get('type') == 'text':
                                            summary_lines.append(block.get('text', ''))
                            except (json.JSONDecodeError, KeyError):
                                pass

                # Take last 2 text blocks as summary
                summary_text = '\n\n'.join(summary_lines[-2:]) if summary_lines else 'Session ended.'
                # Truncate for Telegram
                if len(summary_text) > 3500:
                    summary_text = summary_text[-3500:]

                now = datetime.now().strftime('%H:%M UTC')
                msg = f"Drift session ended ({now})\n\n{summary_text}\n\nReply to direct next session."

                subprocess.run(
                    ["python", str(telegram_bot_path), "send", msg],
                    timeout=15,
                    capture_output=True
                )
                if args.debug:
                    print("DEBUG: Telegram notification sent", file=sys.stderr)
        except Exception as e:
            if args.debug:
                print(f"DEBUG: Telegram notification failed: {e}", file=sys.stderr)
        # === END TELEGRAM NOTIFICATION ===

        # Extract required fields
        session_id = input_data.get("session_id", "")
        stop_hook_active = input_data.get("stop_hook_active", False)

        # Ensure log directory exists
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "stop.json")

        # Read existing log data or initialize empty list
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                try:
                    log_data = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    log_data = []
        else:
            log_data = []

        # Append new data
        log_data.append(input_data)

        # Write back to file with formatting
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        # Handle --chat switch
        if args.chat and 'transcript_path' in input_data:
            transcript_path = input_data['transcript_path']
            if os.path.exists(transcript_path):
                # Read .jsonl file and convert to JSON array
                chat_data = []
                try:
                    with open(transcript_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    chat_data.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass  # Skip invalid lines

                    # Write to logs/chat.json
                    chat_file = os.path.join(log_dir, 'chat.json')
                    with open(chat_file, 'w') as f:
                        json.dump(chat_data, f, indent=2)
                except Exception:
                    pass  # Fail silently

        # Announce completion via TTS (only if --notify flag is set)
        if args.notify:
            announce_completion()

        sys.exit(0)

    except json.JSONDecodeError:
        # Handle JSON decode errors gracefully
        sys.exit(0)
    except Exception:
        # Handle any other errors gracefully
        sys.exit(0)


if __name__ == "__main__":
    main()
