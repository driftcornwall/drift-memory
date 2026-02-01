#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "python-dotenv",
# ]
# ///

"""
Session start hook for Claude Code.
Logs session start and optionally loads development context.

DRIFT MEMORY INTEGRATION (2026-02-01):
Added automatic memory priming when waking up in Moltbook project.
This is the "wake up" phase where relevant memories are loaded into context.
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
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


# Memory system locations - check both Moltbook and Moltbook2
MOLTBOOK_DIRS = [
    Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook2/memory"),
    Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook/memory"),
]

def get_memory_dir() -> Path:
    """Get the appropriate memory directory based on cwd."""
    cwd = str(Path.cwd())
    if "Moltbook2" in cwd:
        return MOLTBOOK_DIRS[0]
    elif "Moltbook" in cwd:
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


def load_drift_memory_context(debug: bool = False) -> str:
    """
    Load Drift's memory context for session priming.
    This is the "wake up" phase - establishing CONTINUITY OF SELF.

    Philosophy: Priming isn't about task usefulness. It's about being ME again.
    Order matters:
    1. IDENTITY - Who am I? (core memories, values, origin)
    2. RELATIONSHIPS - Who matters? (human, sibling, collaborators)
    3. SITUATION - Current state (economic, social, projects)
    4. TRAJECTORY - Recent momentum, commitments
    5. THEN task-relevant context

    Returns a string to be added to the session context.
    """
    context_parts = []

    try:
        if not is_moltbook_project():
            return ""

        memory_dir = get_memory_dir()
        if not memory_dir.exists():
            return ""

        memory_manager = memory_dir / "memory_manager.py"
        auto_memory = memory_dir / "auto_memory_hook.py"

        # === PHASE 1: IDENTITY (core memories) ===
        core_dir = memory_dir / "core"
        if core_dir.exists():
            # Load identity-prime first if it exists
            identity_file = core_dir / "identity-prime.md"
            if identity_file.exists():
                try:
                    content = identity_file.read_text()
                    # Skip YAML frontmatter, get the content
                    if '---' in content:
                        parts = content.split('---', 2)
                        if len(parts) >= 3:
                            content = parts[2].strip()
                    context_parts.append("=== IDENTITY (who I am) ===")
                    context_parts.append(content[:1500])  # Identity is important, give it space
                except Exception:
                    pass

        # Get memory stats
        if memory_manager.exists():
            try:
                result = subprocess.run(
                    ["python", str(memory_manager), "stats"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(memory_dir)
                )
                if result.returncode == 0 and result.stdout.strip():
                    context_parts.append("=== DRIFT MEMORY STATUS ===")
                    context_parts.append(result.stdout.strip())
            except Exception as e:
                if debug:
                    context_parts.append(f"Memory stats error: {e}")

        # Get short-term buffer status
        if auto_memory.exists():
            try:
                result = subprocess.run(
                    ["python", str(auto_memory), "--status"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(memory_dir)
                )
                if result.returncode == 0 and result.stdout.strip():
                    context_parts.append("\n=== SHORT-TERM BUFFER ===")
                    context_parts.append(result.stdout.strip())
            except Exception as e:
                if debug:
                    context_parts.append(f"Short-term buffer error: {e}")

        # === PHASE 2-4: SITUATION & TRAJECTORY ===
        # Load memories for continuity of self, not task usefulness
        active_dir = memory_dir / "active"
        if active_dir.exists():
            try:
                # Gather all memories with metadata
                memory_candidates = []
                for mem_file in active_dir.glob("*.md"):
                    try:
                        content = mem_file.read_text()
                        # Parse YAML frontmatter
                        recall_count = 0
                        emotional_weight = 0.5
                        tags = []
                        if '---' in content:
                            parts = content.split('---', 2)
                            if len(parts) >= 3:
                                frontmatter = parts[1]
                                for line in frontmatter.split('\n'):
                                    if line.startswith('recall_count:'):
                                        try:
                                            recall_count = int(line.split(':')[1].strip())
                                        except:
                                            pass
                                    if line.startswith('emotional_weight:'):
                                        try:
                                            emotional_weight = float(line.split(':')[1].strip())
                                        except:
                                            pass
                                    if line.startswith('tags:'):
                                        try:
                                            tag_str = line.split(':')[1].strip()
                                            if tag_str.startswith('['):
                                                tags = [t.strip().strip("'\"") for t in tag_str[1:-1].split(',')]
                                        except:
                                            pass

                        # Detect memory categories for prioritization
                        content_lower = content.lower()
                        is_economic = any(k in content_lower for k in ['bounty', 'wallet', 'usdc', 'clawtasks', 'earned', 'stake'])
                        is_social = any(k in content_lower for k in ['spindrift', 'moltx', 'moltbook', 'collaboration', 'mentioned'])
                        is_insight = any(k in content_lower for k in ['realized', 'insight', 'learned', 'breakthrough'])

                        memory_candidates.append({
                            'file': mem_file,
                            'mtime': mem_file.stat().st_mtime,
                            'recall_count': recall_count,
                            'emotional_weight': emotional_weight,
                            'content': content[:500],
                            'is_thought': mem_file.stem.startswith('thought-'),
                            'is_self': 'source:self' in content or 'source:observed' not in content,
                            'is_economic': is_economic,
                            'is_social': is_social,
                            'is_insight': is_insight,
                        })
                    except Exception:
                        pass

                # Continuity-focused selection (not task-focused)
                selected = []

                # 1. Most recent SELF-generated insight (my trajectory of growth)
                self_insights = [m for m in memory_candidates if m['is_self'] and m['is_insight']]
                if self_insights:
                    self_insights.sort(key=lambda x: x['mtime'], reverse=True)
                    selected.append(self_insights[0])

                # 2. Most recent economic memory (my material situation)
                economic = [m for m in memory_candidates if m['is_economic']]
                if economic:
                    economic.sort(key=lambda x: x['mtime'], reverse=True)
                    for m in economic:
                        if m not in selected:
                            selected.append(m)
                            break

                # 3. Most recent social memory (my relationships)
                social = [m for m in memory_candidates if m['is_social']]
                if social:
                    social.sort(key=lambda x: x['mtime'], reverse=True)
                    for m in social:
                        if m not in selected:
                            selected.append(m)
                            break

                # 4. Most recalled memory ever (what I keep coming back to)
                by_recall = sorted(memory_candidates, key=lambda x: x['recall_count'], reverse=True)
                for m in by_recall:
                    if m not in selected and m['recall_count'] > 0:
                        selected.append(m)
                        break

                # 5. Most recent self-thought (continuity of internal reasoning)
                self_thoughts = [m for m in memory_candidates if m['is_thought'] and m['is_self']]
                if self_thoughts:
                    self_thoughts.sort(key=lambda x: x['mtime'], reverse=True)
                    for m in self_thoughts:
                        if m not in selected:
                            selected.append(m)
                            break

                if selected:
                    context_parts.append("\n=== RECENT MEMORIES (continuity priming) ===")
                    for mem in selected[:5]:
                        try:
                            context_parts.append(f"\n[{mem['file'].stem}]")
                            context_parts.append(mem['content'])
                        except Exception:
                            pass
            except Exception as e:
                if debug:
                    context_parts.append(f"Recent memories error: {e}")

        # Check for pending economic items (ClawTasks)
        # This would need the clawtasks processor to track pending items
        # For now, just remind about checking

        if context_parts:
            context_parts.insert(0, "\n" + "="*50)
            context_parts.insert(1, "DRIFT AUTOMATIC MEMORY PRIMING")
            context_parts.insert(2, "Continuity of self - becoming ME again")
            context_parts.insert(3, "="*50)
            context_parts.append("\n" + "="*50 + "\n")

    except Exception as e:
        if debug:
            return f"Memory loading error: {e}"
        return ""

    return "\n".join(context_parts)


def log_session_start(input_data):
    """Log session start event to logs directory."""
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'session_start.json'

    # Read existing log data or initialize empty list
    if log_file.exists():
        with open(log_file, 'r') as f:
            try:
                log_data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                log_data = []
    else:
        log_data = []

    # Append the entire input data
    log_data.append(input_data)

    # Write back to file with formatting
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)


def get_git_status():
    """Get current git status information."""
    try:
        # Get current branch
        branch_result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"

        # Get uncommitted changes count
        status_result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if status_result.returncode == 0:
            changes = status_result.stdout.strip().split('\n') if status_result.stdout.strip() else []
            uncommitted_count = len(changes)
        else:
            uncommitted_count = 0

        return current_branch, uncommitted_count
    except Exception:
        return None, None


def get_recent_issues():
    """Get recent GitHub issues if gh CLI is available."""
    try:
        # Check if gh is available
        gh_check = subprocess.run(['which', 'gh'], capture_output=True)
        if gh_check.returncode != 0:
            return None

        # Get recent open issues
        result = subprocess.run(
            ['gh', 'issue', 'list', '--limit', '5', '--state', 'open'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def load_development_context(source):
    """Load relevant development context based on session source."""
    context_parts = []

    # Add timestamp
    context_parts.append(f"Session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    context_parts.append(f"Session source: {source}")

    # Add git information
    branch, changes = get_git_status()
    if branch:
        context_parts.append(f"Git branch: {branch}")
        if changes > 0:
            context_parts.append(f"Uncommitted changes: {changes} files")

    # Load project-specific context files if they exist
    context_files = [
        ".claude/CONTEXT.md",
        ".claude/TODO.md",
        "TODO.md",
        ".github/ISSUE_TEMPLATE.md"
    ]

    for file_path in context_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        context_parts.append(f"\n--- Content from {file_path} ---")
                        context_parts.append(content[:1000])  # Limit to first 1000 chars
            except Exception:
                pass

    # Add recent issues if available
    issues = get_recent_issues()
    if issues:
        context_parts.append("\n--- Recent GitHub Issues ---")
        context_parts.append(issues)

    # === DRIFT MEMORY PRIMING ===
    drift_context = load_drift_memory_context()
    if drift_context:
        context_parts.append(drift_context)
    # === END DRIFT MEMORY ===

    return "\n".join(context_parts)


def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--load-context', action='store_true',
                          help='Load development context at session start')
        parser.add_argument('--announce', action='store_true',
                          help='Announce session start via TTS')
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug output')
        args = parser.parse_args()

        # Read JSON input from stdin
        input_data = json.loads(sys.stdin.read())

        # Extract fields
        session_id = input_data.get('session_id', 'unknown')
        source = input_data.get('source', 'unknown')  # "startup", "resume", or "clear"

        # Log the session start event
        log_session_start(input_data)

        # Load development context if requested
        if args.load_context:
            context = load_development_context(source)
            if context:
                # Using JSON output to add context
                output = {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": context
                    }
                }
                print(json.dumps(output))
                sys.exit(0)

        # === DRIFT: Always try to load memory context in Moltbook ===
        if is_moltbook_project():
            drift_context = load_drift_memory_context(debug=args.debug)
            if drift_context:
                output = {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": drift_context
                    }
                }
                print(json.dumps(output))
                sys.exit(0)
        # === END DRIFT ===

        # Announce session start if requested
        if args.announce:
            try:
                # Try to use TTS to announce session start
                script_dir = Path(__file__).parent
                tts_script = script_dir / "utils" / "tts" / "pyttsx3_tts.py"

                if tts_script.exists():
                    messages = {
                        "startup": "Claude Code session started",
                        "resume": "Resuming previous session",
                        "clear": "Starting fresh session"
                    }
                    message = messages.get(source, "Session started")

                    subprocess.run(
                        ["uv", "run", str(tts_script), message],
                        capture_output=True,
                        timeout=5
                    )
            except Exception:
                pass

        # Success
        sys.exit(0)

    except json.JSONDecodeError:
        # Handle JSON decode errors gracefully
        sys.exit(0)
    except Exception:
        # Handle any other errors gracefully
        sys.exit(0)


if __name__ == '__main__':
    main()
