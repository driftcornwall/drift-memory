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
    pass  # dotenv is optional


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


def consolidate_subagent_memory(
    transcript_path: str = None,
    cwd: str = None,
    debug: bool = False,
):
    """
    Run memory consolidation for a subagent's work.

    Replicates teammate_idle.py's consolidation pipeline:
    - Process transcript for thought memories
    - Run short-term buffer consolidation
    - Save pending co-occurrences
    - Update episodic memory with subagent milestones
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

        transcript_processor = memory_dir / "transcript_processor.py"
        auto_memory = memory_dir / "auto_memory_hook.py"
        memory_manager = memory_dir / "memory_manager.py"

        # 1. Process subagent's transcript for thought memories
        if transcript_path and transcript_processor.exists() and os.path.exists(transcript_path):
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
                        f"DEBUG: [subagent] Transcript processing: {result.stdout[:500]}",
                        file=sys.stderr,
                    )
            except Exception as e:
                if debug:
                    print(f"DEBUG: [subagent] Transcript error: {e}", file=sys.stderr)

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
                        f"DEBUG: [subagent] Buffer consolidation: {result.stdout}",
                        file=sys.stderr,
                    )
            except Exception as e:
                if debug:
                    print(f"DEBUG: [subagent] Buffer error: {e}", file=sys.stderr)

        # 3. Save co-occurrences in-process (save-pending was removed in DB migration)
        try:
            if str(memory_dir) not in sys.path:
                sys.path.insert(0, str(memory_dir))
            from co_occurrence import end_session_cooccurrence
            result = end_session_cooccurrence()
            if debug:
                print(
                    f"DEBUG: [subagent] co-occurrences saved: {len(result)} new links",
                    file=sys.stderr,
                )
        except Exception as e:
            if debug:
                print(f"DEBUG: [subagent] co-occurrence error: {e}", file=sys.stderr)

        # 4. Update episodic memory with subagent milestones
        if transcript_path and transcript_processor.exists() and os.path.exists(transcript_path):
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
                        f"\n## Subagent completed (~{session_time})\n\n"
                        f"{milestone_md}\n"
                    )

                    if episodic_file.exists():
                        with open(episodic_file, "a", encoding="utf-8") as f:
                            f.write(entry)
                    else:
                        header = f"# {today}\n"
                        with open(episodic_file, "w", encoding="utf-8") as f:
                            f.write(header + entry)

                    if debug:
                        print("DEBUG: [subagent] Episodic updated", file=sys.stderr)
            except Exception as e:
                if debug:
                    print(f"DEBUG: [subagent] Episodic error: {e}", file=sys.stderr)

    except Exception as e:
        if debug:
            print(f"DEBUG: [subagent] Memory consolidation error: {e}", file=sys.stderr)


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


def announce_subagent_completion():
    """Announce subagent completion using the best available TTS service."""
    try:
        tts_script = get_tts_script_path()
        if not tts_script:
            return  # No TTS scripts available
        
        # Use fixed message for subagent completion
        completion_message = "Subagent Complete"
        
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

        # Extract required fields
        session_id = input_data.get("session_id", "")
        stop_hook_active = input_data.get("stop_hook_active", False)

        # Ensure log directory exists
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "subagent_stop.json")

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
        
        # Handle --chat switch (same as stop.py)
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

        # Consolidate subagent work into Drift's memory graph
        transcript_path = input_data.get("transcript_path", "")
        project_cwd = input_data.get("cwd", "")
        consolidate_subagent_memory(
            transcript_path=transcript_path,
            cwd=project_cwd,
            debug=args.debug if hasattr(args, 'debug') else False,
        )

        # Announce subagent completion via TTS (only if --notify flag is set)
        if args.notify:
            announce_subagent_completion()

        sys.exit(0)

    except json.JSONDecodeError:
        # Handle JSON decode errors gracefully
        sys.exit(0)
    except Exception:
        # Handle any other errors gracefully
        sys.exit(0)


if __name__ == "__main__":
    main()