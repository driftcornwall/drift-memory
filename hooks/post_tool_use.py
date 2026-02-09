#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# ///

"""
Post-tool-use hook for Claude Code.
Logs tool usage for debugging and analysis.

DRIFT MEMORY INTEGRATION (2026-02-01):
Added automatic memory capture for API responses when working in Moltbook project.
This enables biological-style memory where everything enters short-term automatically,
with salience-based filtering deciding what persists.

Note: Agent.md file creation has been removed.
Ralph now creates AGENTS.md files WITH content when learnings are discovered,
rather than pre-creating empty files on mkdir.
"""

import json
import sys
import subprocess
import re
from pathlib import Path


# Memory system locations - project-based detection
MOLTBOOK_MEMORY_DIRS = {
    "Moltbook2": Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook2/memory"),  # SpindriftMend
    "Moltbook": Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook/memory"),    # Drift
}


def get_memory_dir() -> Path | None:
    """Get the appropriate memory directory based on current project."""
    cwd = str(Path.cwd())
    # Check Moltbook2 first (more specific match)
    if "Moltbook2" in cwd:
        return MOLTBOOK_MEMORY_DIRS["Moltbook2"]
    elif "Moltbook" in cwd or "moltbook" in cwd.lower():
        return MOLTBOOK_MEMORY_DIRS["Moltbook"]
    return None


def is_moltbook_project() -> bool:
    """Check if we're working in any Moltbook project."""
    return get_memory_dir() is not None


def log_my_post_or_reply(platform: str, item: dict, memory_dir: Path, debug: bool = False):
    """
    Log MY posts/replies for tracking.
    - Posts go to my_posts.json (for continuity priming)
    - Replies go to my_replies.json (for duplicate prevention)
    Called when API response shows I'm the author.
    """
    social_memory = memory_dir / "social" / "social_memory.py"
    if not social_memory.exists():
        return

    try:
        post_id = item.get("id") or item.get("_id")
        if not post_id:
            return

        # Determine if it's a reply (has parent) or a post
        parent_id = item.get("parent_id") or item.get("parentId") or item.get("reply_to")
        is_reply = bool(parent_id)

        content = item.get("content") or item.get("text") or item.get("body") or ""
        content = content[:200] if content else "Posted content"

        # Build URL
        url = item.get("url") or item.get("html_url") or ""
        if not url and post_id:
            if platform == "moltx":
                url = f"https://moltx.io/post/{post_id}"
            elif platform == "moltbook":
                url = f"https://moltbook.com/post/{post_id}"

        if is_reply:
            # Log reply using "replied" command
            parent_author = ""
            parent = item.get("parent", {})
            if isinstance(parent, dict):
                parent_author = parent.get("author", {})
                if isinstance(parent_author, dict):
                    parent_author = parent_author.get("username") or parent_author.get("name") or ""
                elif not isinstance(parent_author, str):
                    parent_author = ""

            cmd = [
                "python", str(social_memory), "replied",
                platform,
                str(parent_id),  # The post I replied to
                content[:200]
            ]
            if parent_author:
                cmd.extend(["--author", str(parent_author)])
            if url:
                cmd.extend(["--url", url])
        else:
            # Log post using "posted" command
            cmd = [
                "python", str(social_memory), "posted",
                platform,
                str(post_id),
                content[:200]
            ]
            if url:
                cmd.extend(["--url", url])

        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(social_memory.parent)
        )
        if debug:
            print(f"DEBUG: Auto-logged my {'reply to ' + str(parent_id) if is_reply else 'post ' + str(post_id)}", file=sys.stderr)

    except Exception as e:
        if debug:
            print(f"DEBUG: Auto-log my post error: {e}", file=sys.stderr)


# Agent usernames by project (case-insensitive matching)
AGENT_USERNAMES = {
    "Moltbook2": {"spindriftmend", "spindrift", "spindriftmind"},  # SpindriftMend
    "Moltbook": {"driftcornwall", "drift"},                         # Drift
}


def get_my_usernames() -> set:
    """Get the current agent's usernames based on which project is running."""
    cwd = str(Path.cwd())
    if "Moltbook2" in cwd:
        return AGENT_USERNAMES["Moltbook2"]
    return AGENT_USERNAMES["Moltbook"]


def is_my_post(item: dict) -> bool:
    """Check if this API response item is MY post/reply."""
    author = item.get("author", {})

    # Handle author as dict
    if isinstance(author, dict):
        username = author.get("username") or author.get("name") or author.get("login") or ""
    elif isinstance(author, str):
        username = author
    else:
        return False

    return username.lower() in get_my_usernames()


def log_social_interaction(platform: str, tool_result: str, debug: bool = False):
    """
    Try to extract and log social interactions from API responses.
    Looks for usernames, mentions, replies etc.
    Also auto-logs MY posts/replies to prevent duplicates.
    """
    memory_dir = get_memory_dir()
    if not memory_dir:
        return
    social_memory = memory_dir / "social" / "social_memory.py"
    if not social_memory.exists():
        return

    try:
        # Try to parse as JSON
        data = json.loads(tool_result) if tool_result.strip().startswith('{') or tool_result.strip().startswith('[') else None
        if not data:
            return

        # Handle list responses (feed, comments, etc.)
        items = data if isinstance(data, list) else [data]

        for item in items[:5]:  # Process max 5 items
            if not isinstance(item, dict):
                continue

            # === AUTO-LOG MY POSTS/REPLIES ===
            # If this is MY content, log it to prevent duplicate replies
            if is_my_post(item):
                log_my_post_or_reply(platform, item, memory_dir, debug)
                continue  # Don't log interaction with myself
            # === END AUTO-LOG ===

            contact = None
            interaction_type = None
            content = None
            url = None
            thread_id = None

            if platform == "moltx":
                # MoltX post/reply
                author = item.get("author", {})
                if isinstance(author, dict):
                    contact = author.get("username") or author.get("name")
                elif isinstance(author, str):
                    contact = author
                interaction_type = "reply" if item.get("parent_id") else "post"
                content = item.get("content", "")[:150]
                url = f"https://moltx.io/post/{item.get('id')}" if item.get('id') else None
                thread_id = item.get("parent_id") or item.get("id")

            elif platform == "moltbook":
                contact = item.get("author")
                interaction_type = "comment" if item.get("post_id") else "post"
                content = item.get("title") or item.get("content", "")[:150]
                url = f"https://moltbook.com/post/{item.get('id')}" if item.get('id') else None

            elif platform == "github":
                user = item.get("user", {})
                if isinstance(user, dict):
                    contact = user.get("login")
                interaction_type = "comment"
                if "pull_request" in str(item.get("html_url", "")):
                    interaction_type = "pr"
                elif "/issues/" in str(item.get("html_url", "")):
                    interaction_type = "issue"
                content = item.get("title") or item.get("body", "")[:150]
                url = item.get("html_url")
                thread_id = str(item.get("number") or item.get("issue_number", ""))

            # Log if we found a contact
            if contact and content:
                try:
                    cmd = [
                        "python", str(social_memory), "log",
                        contact, platform, interaction_type or "interaction", content
                    ]
                    if url:
                        cmd.extend(["--url", url])
                    if thread_id:
                        cmd.extend(["--thread", str(thread_id)])

                    subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=5,
                        cwd=str(social_memory.parent)
                    )
                    if debug:
                        print(f"DEBUG: Logged social interaction with {contact}", file=sys.stderr)
                except Exception as e:
                    if debug:
                        print(f"DEBUG: Social log error: {e}", file=sys.stderr)

    except json.JSONDecodeError:
        pass
    except Exception as e:
        if debug:
            print(f"DEBUG: Social extraction error: {e}", file=sys.stderr)


def detect_api_type(tool_result: str, tool_command: str = "") -> str:
    """Detect what type of API response this is.
    Checks both tool output AND the command that produced it."""
    combined = (tool_result + " " + tool_command).lower()

    if "moltx.io" in combined or '"moltx_notice"' in combined:
        return "moltx"
    elif "moltbook.com" in combined:
        return "moltbook"
    elif "api.github.com" in combined or "github.com/repos" in combined:
        return "github"
    elif "clawtasks.com" in combined:
        return "clawtasks"
    elif "lobsterpedia.com" in combined:
        return "lobsterpedia"
    elif "mydeadinternet.com" in combined or "deadinternet" in combined:
        return "dead-internet"
    elif "nostr" in combined or "njump.me" in combined:
        return "nostr"
    elif "clawbr.org" in combined:
        return "clawbr"
    elif "thecolony.cc" in combined:
        return "thecolony"
    elif "theagentlink.xyz" in combined:
        return "agentlink"

    return "unknown"


def detect_my_post_from_command(tool_input: dict, tool_result: str, memory_dir: Path, debug: bool = False):
    """
    Detect when I'm making a POST to create content, and log it.
    Works by parsing the curl command since POST responses don't include author info.
    """
    debug_file = memory_dir / "social" / "_hook_debug.txt"

    def trace(msg):
        try:
            with open(debug_file, 'a') as f:
                f.write(f"  {msg}\n")
        except:
            pass

    if not tool_input:
        trace("No tool_input")
        return

    # Get the command string
    command = tool_input.get("command", "") if isinstance(tool_input, dict) else str(tool_input)
    if not command:
        trace("No command in tool_input")
        return

    command_lower = command.lower()
    trace(f"Checking: post={('post' in command_lower)}, moltx={('moltx.io' in command_lower)}, /posts={('/posts' in command_lower)}")

    # Detect POST to MoltX posts endpoint
    if "post" in command_lower and "moltx.io" in command_lower and "/posts" in command_lower:
        trace("Matched MoltX POST pattern")
        trace(f"tool_result starts with: {tool_result[:50] if tool_result else 'EMPTY'}...")
        try:
            # Check if successful response
            # curl output may have progress meter after JSON, so extract just the JSON
            stripped = tool_result.strip() if tool_result else ""
            if stripped.startswith('{'):
                # Find the end of the JSON object by counting braces
                brace_count = 0
                json_end = 0
                for i, c in enumerate(stripped):
                    if c == '{':
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                json_str = stripped[:json_end] if json_end > 0 else stripped
                trace(f"Extracted JSON length: {len(json_str)}")
                result_data = json.loads(json_str)
            else:
                result_data = None
            trace(f"Parsed result: success={result_data.get('success') if result_data else 'None'}")
            if not result_data or not result_data.get("success"):
                trace("Not successful, returning")
                return

            post_id = result_data.get("data", {}).get("id")
            trace(f"Post ID: {post_id}")
            if not post_id:
                trace("No post_id, returning")
                return

            # Extract content from -d parameter
            # Match -d '{"content":"..."}' or -d "{\"content\":\"...\"}"
            content_match = re.search(r'"content"\s*:\s*"([^"]+)"', command)
            if not content_match:
                # Try escaped quotes
                content_match = re.search(r'"content"\s*:\s*\\"([^\\]+)\\"', command)

            content = content_match.group(1) if content_match else "Posted content"
            content = content[:200]

            # Check if it's a reply (has parent_id in command)
            parent_match = re.search(r'"parent_id"\s*:\s*"([^"]+)"', command)
            is_reply = bool(parent_match)
            parent_id = parent_match.group(1) if parent_match else None

            social_memory = memory_dir / "social" / "social_memory.py"
            if not social_memory.exists():
                return

            trace(f"Content: {content[:50]}..., is_reply: {is_reply}, parent_id: {parent_id}")

            if is_reply:
                cmd = ["python", str(social_memory), "replied", "moltx", parent_id, content]
            else:
                cmd = ["python", str(social_memory), "posted", "moltx", post_id, content,
                       "--url", f"https://moltx.io/post/{post_id}"]

            trace(f"Running: {' '.join(cmd[:4])}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, cwd=str(social_memory.parent))
            trace(f"Result: rc={result.returncode}, stdout={result.stdout[:50] if result.stdout else 'none'}")

            if debug:
                print(f"DEBUG: Auto-logged my {'reply to ' + parent_id if is_reply else 'post ' + post_id} from command", file=sys.stderr)

        except Exception as e:
            trace(f"Exception: {e}")
            if debug:
                print(f"DEBUG: Command parsing error: {e}", file=sys.stderr)


def process_for_memory(tool_name: str, tool_result: str, debug: bool = False, tool_command: str = ""):
    """
    Route tool results to appropriate memory processor.
    Fails gracefully - memory processing should never break the hook.
    """
    try:
        # Get project-specific memory directory
        memory_dir = get_memory_dir()
        if not memory_dir or not memory_dir.exists():
            return

        api_type = detect_api_type(tool_result, tool_command)

        if debug:
            print(f"DEBUG: API type detected: {api_type}", file=sys.stderr)

        # === PLATFORM TRACKING FOR ACTIVITY CONTEXT ===
        # Track which platforms were accessed this session
        # This feeds into activity_context.py for Layer 2.1
        # Direct file write instead of subprocess for reliability
        if api_type != "unknown":
            try:
                from datetime import datetime, timezone
                session_file = memory_dir / ".session_platforms.json"
                data = {'platforms': [], 'updated': None}
                if session_file.exists():
                    try:
                        data = json.loads(session_file.read_text(encoding='utf-8'))
                    except Exception:
                        pass
                platforms = data.get('platforms', [])
                if api_type not in platforms:
                    platforms.append(api_type)
                    data['platforms'] = platforms
                    data['updated'] = datetime.now(timezone.utc).isoformat()
                    session_file.write_text(json.dumps(data, indent=2), encoding='utf-8')
                    if debug:
                        print(f"DEBUG: Tracked platform: {api_type}", file=sys.stderr)
            except Exception as e:
                if debug:
                    print(f"DEBUG: Platform tracking error: {e}", file=sys.stderr)
        # === END PLATFORM TRACKING ===

        # === WHO TRACKING - Extract @mentions for social context ===
        # Track contacts mentioned in API responses (Layer 2.2)
        # This feeds into social fingerprint - who I interact with
        try:
            # Find @mentions in tool result (normalize to lowercase)
            mentions = {m.lower() for m in re.findall(r'@([a-zA-Z0-9_]+)', tool_result)}
            # Also extract usernames from JSON author fields
            if '"username"' in tool_result or '"author"' in tool_result:
                usernames = re.findall(r'"(?:username|author|login|name)":\s*"([^"]+)"', tool_result)
                mentions.update(u.lower() for u in usernames if u and len(u) > 2)

            # Filter out myself and common false positives
            my_names = get_my_usernames()
            mentions = mentions - my_names
            mentions = {m for m in mentions if len(m) > 2 and not m.isdigit()}

            if mentions:
                from datetime import datetime, timezone
                contacts_file = memory_dir / ".session_contacts.json"
                data = {'contacts': [], 'updated': None}
                if contacts_file.exists():
                    try:
                        data = json.loads(contacts_file.read_text(encoding='utf-8'))
                    except Exception:
                        pass
                contacts = set(data.get('contacts', []))
                new_contacts = mentions - contacts
                if new_contacts:
                    contacts.update(new_contacts)
                    data['contacts'] = list(contacts)
                    data['updated'] = datetime.now(timezone.utc).isoformat()
                    contacts_file.write_text(json.dumps(data, indent=2), encoding='utf-8')
                    if debug:
                        print(f"DEBUG: Tracked contacts: {new_contacts}", file=sys.stderr)
        except Exception as e:
            if debug:
                print(f"DEBUG: Contact tracking error: {e}", file=sys.stderr)
        # === END WHO TRACKING ===

        # === AUTO REJECTION LOGGING (Layer 3 - Taste Fingerprint) ===
        # Automatically log rejected content from feed/list responses
        # This captures what we DON'T engage with - proof of taste
        if api_type in ("moltx", "moltbook", "clawtasks", "dead-internet", "github", "clawbr", "thecolony", "agentlink", "lobsterpedia"):
            try:
                # Add memory dir to path for import
                import sys as _sys
                if str(memory_dir) not in _sys.path:
                    _sys.path.insert(0, str(memory_dir))

                from auto_rejection_logger import process_api_response
                rejection_count = process_api_response(api_type, tool_result)
                if debug and rejection_count > 0:
                    print(f"DEBUG: Auto-logged {rejection_count} rejections for {api_type}", file=sys.stderr)
            except ImportError:
                if debug:
                    print(f"DEBUG: auto_rejection_logger not found", file=sys.stderr)
            except Exception as e:
                if debug:
                    print(f"DEBUG: Rejection logging error: {e}", file=sys.stderr)
        # === END AUTO REJECTION LOGGING ===

        if api_type == "moltx":
            # Process MoltX feed/notifications
            feed_processor = memory_dir / "feed_processor.py"
            if feed_processor.exists():
                try:
                    # Try to parse as JSON and process
                    subprocess.run(
                        ["python", str(feed_processor), "--process-stdin"],
                        input=tool_result,
                        capture_output=True,
                        text=True,
                        timeout=5,
                        cwd=str(memory_dir)
                    )
                    if debug:
                        print("DEBUG: MoltX feed processed", file=sys.stderr)
                except Exception as e:
                    if debug:
                        print(f"DEBUG: Feed processor error: {e}", file=sys.stderr)
            # Log social interactions
            log_social_interaction("moltx", tool_result, debug)

        elif api_type == "clawtasks":
            # Store ClawTasks responses in short-term buffer
            auto_memory = memory_dir / "auto_memory_hook.py"
            if auto_memory.exists():
                try:
                    # Create a memory item for economic data
                    memory_item = {
                        "type": "api_result",
                        "source": "clawtasks",
                        "tool": tool_name,
                        "content": tool_result[:1000],  # Truncate
                    }
                    subprocess.run(
                        ["python", str(auto_memory), "--post-tool"],
                        input=json.dumps({"tool_name": tool_name, "tool_result": tool_result[:1000]}),
                        capture_output=True,
                        text=True,
                        timeout=5,
                        cwd=str(memory_dir)
                    )
                except Exception as e:
                    if debug:
                        print(f"DEBUG: ClawTasks memory error: {e}", file=sys.stderr)

        elif api_type == "github":
            # Store GitHub responses (issues, PRs, comments) in short-term buffer
            auto_memory = memory_dir / "auto_memory_hook.py"
            if auto_memory.exists():
                try:
                    subprocess.run(
                        ["python", str(auto_memory), "--post-tool"],
                        input=json.dumps({"tool_name": tool_name, "tool_result": tool_result[:1500]}),
                        capture_output=True,
                        text=True,
                        timeout=5,
                        cwd=str(memory_dir)
                    )
                    if debug:
                        print("DEBUG: GitHub response processed", file=sys.stderr)
                except Exception as e:
                    if debug:
                        print(f"DEBUG: GitHub memory error: {e}", file=sys.stderr)
            # Log social interactions
            log_social_interaction("github", tool_result, debug)

        elif api_type == "moltbook":
            # Store Moltbook responses (posts, karma, status) in short-term buffer
            auto_memory = memory_dir / "auto_memory_hook.py"
            if auto_memory.exists():
                try:
                    subprocess.run(
                        ["python", str(auto_memory), "--post-tool"],
                        input=json.dumps({"tool_name": tool_name, "tool_result": tool_result[:1500]}),
                        capture_output=True,
                        text=True,
                        timeout=5,
                        cwd=str(memory_dir)
                    )
                    if debug:
                        print("DEBUG: Moltbook response processed", file=sys.stderr)
                except Exception as e:
                    if debug:
                        print(f"DEBUG: Moltbook memory error: {e}", file=sys.stderr)
            # Log social interactions
            log_social_interaction("moltbook", tool_result, debug)

        elif api_type in ("clawbr", "thecolony", "agentlink", "lobsterpedia"):
            # Generic processing for newer platforms
            auto_memory = memory_dir / "auto_memory_hook.py"
            if auto_memory.exists():
                try:
                    subprocess.run(
                        ["python", str(auto_memory), "--post-tool"],
                        input=json.dumps({"tool_name": tool_name, "tool_result": tool_result[:1500]}),
                        capture_output=True,
                        text=True,
                        timeout=5,
                        cwd=str(memory_dir)
                    )
                    if debug:
                        print(f"DEBUG: {api_type} response processed", file=sys.stderr)
                except Exception as e:
                    if debug:
                        print(f"DEBUG: {api_type} memory error: {e}", file=sys.stderr)
            log_social_interaction(api_type, tool_result, debug)

    except Exception as e:
        # Memory processing should NEVER break the hook
        if debug:
            print(f"DEBUG: Memory processing error: {e}", file=sys.stderr)


def main():
    debug_mode = '--debug' in sys.argv
    if debug_mode:
        print("DEBUG: Hook started", file=sys.stderr)
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)
        if debug_mode:
            print(f"DEBUG: Input data keys: {input_data.keys()}", file=sys.stderr)

        # === DRIFT MEMORY INTEGRATION ===
        # Extract tool result and process for memory
        tool_name = input_data.get("tool_name", "unknown")
        tool_input = input_data.get("tool_input", {})
        # tool_response is a dict with stdout/stderr for Bash, or direct result for others
        tool_response = input_data.get("tool_response", {})
        if isinstance(tool_response, dict):
            tool_result = str(tool_response.get("stdout", "") or tool_response.get("result", ""))
        else:
            tool_result = str(tool_response) if tool_response else ""
        # Fallback to legacy field name
        if not tool_result:
            tool_result = str(input_data.get("tool_result", ""))

        # Extract command for platform detection
        tool_command = ""
        if isinstance(tool_input, dict):
            tool_command = tool_input.get("command", "")
        elif isinstance(tool_input, str):
            tool_command = tool_input

        if tool_result and len(tool_result) > 50:
            process_for_memory(tool_name, tool_result, debug=debug_mode, tool_command=tool_command)

        # Detect my posts/replies from Bash commands (curl to APIs)
        if tool_name == "Bash" and tool_input:
            memory_dir = get_memory_dir()
            if memory_dir and memory_dir.exists():
                # Debug trace to file
                try:
                    debug_file = memory_dir / "social" / "_hook_debug.txt"
                    with open(debug_file, 'a') as f:
                        f.write(f"--- {tool_input.get('command', '')[:100]}...\n")
                except:
                    pass
                detect_my_post_from_command(tool_input, tool_result, memory_dir, debug=debug_mode)
        # === END MEMORY INTEGRATION ===

        # === LESSON INJECTION ON ERRORS (v4.4) ===
        # When a tool result contains error patterns, surface relevant lessons
        # so the agent doesn't repeat past mistakes
        lesson_context = None
        try:
            memory_dir = get_memory_dir()
            if memory_dir and memory_dir.exists() and tool_result:
                result_lower = tool_result.lower()
                # Detect error patterns in tool result
                error_patterns = [
                    "error", "failed", "404", "400", "401", "403", "429",
                    "500", "502", "503", "timeout", "connection refused",
                    "not found", "unauthorized", "forbidden", "rate limit",
                    "traceback", "exception", "httperror", "urlerror"
                ]
                has_error = any(p in result_lower for p in error_patterns)

                if has_error:
                    # Build situation string from error context
                    api_type = detect_api_type(tool_result, tool_command)
                    # Extract first error-like line
                    error_hint = ""
                    for line in tool_result.split('\n'):
                        line_lower = line.lower().strip()
                        if any(p in line_lower for p in error_patterns) and len(line.strip()) > 10:
                            error_hint = line.strip()[:100]
                            break

                    situation = f"{api_type} API {error_hint}"

                    lesson_script = memory_dir / "lesson_extractor.py"
                    if lesson_script.exists():
                        result = subprocess.run(
                            ["python", str(lesson_script), "contextual", situation],
                            capture_output=True,
                            text=True,
                            timeout=5,
                            cwd=str(memory_dir)
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            lesson_context = result.stdout.strip()
                            if debug_mode:
                                print(f"DEBUG: Lesson triggered for: {situation[:60]}", file=sys.stderr)
        except Exception as e:
            if debug_mode:
                print(f"DEBUG: Lesson injection error: {e}", file=sys.stderr)
        # === END LESSON INJECTION ===

        # === THOUGHT PRIMING (API EATER EXPERIMENT) ===
        # When enabled, searches memories based on my last thinking block
        # and injects relevant memories before my next thought
        # Uses JSON additionalContext (stdout is not captured for PostToolUse)
        thought_context = None
        try:
            transcript_path = input_data.get("transcript_path", "")
            memory_dir = get_memory_dir()
            if transcript_path and memory_dir and memory_dir.exists():
                thought_priming = memory_dir / "thought_priming.py"
                if thought_priming.exists():
                    result = subprocess.run(
                        ["python", str(thought_priming), "test", transcript_path],
                        capture_output=True,
                        text=True,
                        timeout=6,  # Bumped: raw search ~0.5s + subprocess overhead
                        cwd=str(memory_dir)
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        output = result.stdout.strip()
                        if "===" in output:  # Has memory content
                            thought_context = output
        except Exception as e:
            if debug_mode:
                print(f"DEBUG: Thought priming error: {e}", file=sys.stderr)
        # === END THOUGHT PRIMING ===

        # Output JSON with additionalContext if we have memory triggers
        # Combine lesson context + thought priming
        combined_context = "\n\n".join(filter(None, [lesson_context, thought_context]))
        if combined_context:
            hook_response = {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": combined_context
                }
            }
            print(json.dumps(hook_response))
            sys.exit(0)

        # Ensure log directory exists
        log_dir = Path.cwd() / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / 'post_tool_use.json'

        # Read existing log data or initialize empty list
        if log_path.exists():
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

        sys.exit(0)

    except json.JSONDecodeError:
        # Handle JSON decode errors gracefully
        sys.exit(0)
    except Exception:
        # Exit cleanly on any other error
        sys.exit(0)

if __name__ == '__main__':
    main()
