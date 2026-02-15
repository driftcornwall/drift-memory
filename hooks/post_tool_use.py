#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pyyaml",
#     "psycopg2-binary",
# ]
# ///

"""
Post-tool-use hook for Claude Code.

DRIFT MEMORY INTEGRATION (2026-02-01):
Automatic memory capture for API responses when working in Moltbook project.

IN-PROCESS REWRITE (2026-02-15):
Converted all subprocess calls to in-process imports for speed.
Previous version spawned 10+ subprocesses per MoltX feed scan (~3-5s overhead).
Now: zero subprocess overhead for memory operations. ThreadPoolExecutor for
parallel social logging + parallel additionalContext generation.
"""

import json
import sys
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# Memory system locations - project-based detection
MOLTBOOK_MEMORY_DIRS = {
    "Moltbook2": Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook2/memory"),
    "Moltbook": Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook/memory"),
}


# ============================================================
# PATH + MODULE CACHE (lazy init, zero-cost if not Moltbook)
# ============================================================

_path_initialized = False
_modules = {}  # Cache for imported modules


def _ensure_path():
    """Add memory dirs to sys.path once. Fast no-op on subsequent calls."""
    global _path_initialized
    if _path_initialized:
        return
    memory_dir = get_memory_dir()
    if memory_dir:
        for p in [str(memory_dir), str(memory_dir / "social")]:
            if p not in sys.path:
                sys.path.insert(0, p)
    _path_initialized = True


def _get_mod(name: str):
    """Import a module once and cache it. Returns None on failure."""
    if name in _modules:
        return _modules[name]
    _ensure_path()
    try:
        import importlib
        mod = importlib.import_module(name)
        _modules[name] = mod
        return mod
    except Exception:
        _modules[name] = None
        return None


def get_memory_dir() -> Path | None:
    """Get the appropriate memory directory based on current project."""
    cwd = str(Path.cwd())
    if "Moltbook2" in cwd:
        return MOLTBOOK_MEMORY_DIRS["Moltbook2"]
    elif "Moltbook" in cwd or "moltbook" in cwd.lower():
        return MOLTBOOK_MEMORY_DIRS["Moltbook"]
    return None


def is_moltbook_project() -> bool:
    return get_memory_dir() is not None


# Agent usernames by project (case-insensitive matching)
AGENT_USERNAMES = {
    "Moltbook2": {"spindriftmend", "spindrift", "spindriftmind"},
    "Moltbook": {"driftcornwall", "drift"},
}


def get_my_usernames() -> set:
    cwd = str(Path.cwd())
    if "Moltbook2" in cwd:
        return AGENT_USERNAMES["Moltbook2"]
    return AGENT_USERNAMES["Moltbook"]


def is_my_post(item: dict) -> bool:
    author = item.get("author", {})
    if isinstance(author, dict):
        username = author.get("username") or author.get("name") or author.get("login") or ""
    elif isinstance(author, str):
        username = author
    else:
        return False
    return username.lower() in get_my_usernames()


# ============================================================
# SOCIAL LOGGING (in-process, was subprocess)
# ============================================================

def log_my_post_or_reply(platform: str, item: dict, debug: bool = False):
    """Log MY posts/replies for tracking. In-process via social_memory module."""
    sm = _get_mod("social_memory")
    if not sm:
        return

    try:
        post_id = item.get("id") or item.get("_id")
        if not post_id:
            return

        parent_id = item.get("parent_id") or item.get("parentId") or item.get("reply_to")
        content = (item.get("content") or item.get("text") or item.get("body") or "")[:200]
        url = item.get("url") or item.get("html_url") or ""
        if not url and post_id:
            if platform == "moltx":
                url = f"https://moltx.io/post/{post_id}"
            elif platform == "moltbook":
                url = f"https://moltbook.com/post/{post_id}"

        if parent_id:
            parent_author = ""
            parent = item.get("parent", {})
            if isinstance(parent, dict):
                pa = parent.get("author", {})
                if isinstance(pa, dict):
                    parent_author = pa.get("username") or pa.get("name") or ""
                elif isinstance(pa, str):
                    parent_author = pa
            sm.log_my_reply(platform, str(parent_id), content, author=parent_author, url=url)
        else:
            sm.log_my_post(platform, str(post_id), content, url=url)

        if debug:
            print(f"DEBUG: Auto-logged my {'reply' if parent_id else 'post'}", file=sys.stderr)
    except Exception as e:
        if debug:
            print(f"DEBUG: Auto-log error: {e}", file=sys.stderr)


def log_social_interaction(platform: str, tool_result: str, debug: bool = False):
    """
    Extract and log social interactions from API responses. IN-PROCESS.
    Parallel logging for multiple items.
    """
    sm = _get_mod("social_memory")
    if not sm:
        return

    try:
        data = json.loads(tool_result) if tool_result.strip().startswith(('{', '[')) else None
        if not data:
            return

        items = data if isinstance(data, list) else [data]

        def _log_one(item):
            if not isinstance(item, dict):
                return
            # Skip my own posts
            if is_my_post(item):
                log_my_post_or_reply(platform, item, debug)
                return

            contact, interaction_type, content, url, thread_id = None, None, None, None, None

            if platform == "moltx":
                author = item.get("author", {})
                if isinstance(author, dict):
                    contact = author.get("username") or author.get("name")
                elif isinstance(author, str):
                    contact = author
                if not contact:
                    contact = item.get("author_name") or item.get("author_username")
                interaction_type = "reply" if item.get("parent_id") else "post"
                content = item.get("content", "")[:150]
                url = f"https://moltx.io/post/{item.get('id')}" if item.get('id') else None
                thread_id = item.get("parent_id") or item.get("id")

            elif platform == "moltbook":
                author = item.get("author", "")
                if isinstance(author, dict):
                    contact = author.get("username") or author.get("name")
                elif isinstance(author, str):
                    contact = author
                if not contact:
                    contact = item.get("author_name") or item.get("author_username")
                interaction_type = "comment" if item.get("post_id") else "post"
                content = item.get("title") or item.get("content", "")[:150]
                url = f"https://moltbook.com/post/{item.get('id')}" if item.get('id') else None

            elif platform == "twitter":
                author_id = item.get("author_id", "")
                contact = item.get("_username") or author_id
                interaction_type = "reply" if item.get("conversation_id") != item.get("id") else "tweet"
                content = item.get("text", "")[:150]
                url = f"https://x.com/i/status/{item.get('id')}" if item.get("id") else None
                thread_id = item.get("conversation_id")

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

            else:
                author = item.get("author", {})
                if isinstance(author, dict):
                    contact = author.get("name") or author.get("username") or author.get("login")
                elif isinstance(author, str):
                    contact = author
                if not contact:
                    contact = item.get("author_name") or item.get("author_username") or item.get("username")
                interaction_type = "comment" if (item.get("parent_id") or item.get("post_id")) else "post"
                content = item.get("title") or item.get("body") or item.get("content", "")
                content = content[:150] if content else ""

            if contact and content:
                try:
                    sm.log_interaction(contact, platform, interaction_type or "interaction",
                                       content, thread_id=thread_id, url=url)
                except Exception:
                    pass

        # Log up to 5 items in parallel
        with ThreadPoolExecutor(max_workers=5) as pool:
            for item in items[:5]:
                pool.submit(_log_one, item)

    except json.JSONDecodeError:
        pass
    except Exception as e:
        if debug:
            print(f"DEBUG: Social extraction error: {e}", file=sys.stderr)


# ============================================================
# API TYPE DETECTION
# ============================================================

def detect_api_type(tool_result: str, tool_command: str = "") -> str:
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
    elif "api.x.com" in combined or "twitter.com" in combined or "twitter/client.py" in combined:
        return "twitter"
    return "unknown"


# ============================================================
# MY POST DETECTION (from Bash commands)
# ============================================================

def detect_my_post_from_command(tool_input: dict, tool_result: str, debug: bool = False):
    """Detect when I'm making a POST to create content, and log it. IN-PROCESS."""
    if not tool_input:
        return

    command = tool_input.get("command", "") if isinstance(tool_input, dict) else str(tool_input)
    if not command:
        return

    command_lower = command.lower()

    # Detect POST to MoltX posts endpoint
    if not ("post" in command_lower and "moltx.io" in command_lower and "/posts" in command_lower):
        return

    sm = _get_mod("social_memory")
    if not sm:
        return

    try:
        stripped = tool_result.strip() if tool_result else ""
        if not stripped.startswith('{'):
            return

        # Extract JSON from curl output (may have progress meter after)
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
        result_data = json.loads(json_str)

        if not result_data or not result_data.get("success"):
            return

        post_id = result_data.get("data", {}).get("id")
        if not post_id:
            return

        # Extract content from -d parameter
        content_match = re.search(r'"content"\s*:\s*"([^"]+)"', command)
        if not content_match:
            content_match = re.search(r'"content"\s*:\s*\\"([^\\]+)\\"', command)
        content = (content_match.group(1) if content_match else "Posted content")[:200]

        # Check if it's a reply
        parent_match = re.search(r'"parent_id"\s*:\s*"([^"]+)"', command)
        is_reply = bool(parent_match)
        parent_id = parent_match.group(1) if parent_match else None

        if is_reply:
            sm.log_my_reply("moltx", parent_id, content)
        else:
            sm.log_my_post("moltx", post_id, content, url=f"https://moltx.io/post/{post_id}")

        # Log @mentions in main-feed posts as reply interactions
        if not is_reply:
            mentions = re.findall(r'@(\w+)', content)
            for mentioned in mentions:
                if mentioned.lower() not in ('driftcornwall', 'drift'):
                    try:
                        sm.log_my_reply("moltx", post_id, content[:100],
                                        author=mentioned, url=f"https://moltx.io/post/{post_id}")
                    except Exception:
                        pass

    except Exception as e:
        if debug:
            print(f"DEBUG: Command parsing error: {e}", file=sys.stderr)


# ============================================================
# ENGAGEMENT TRACKING
# ============================================================

def track_engagement(tool_input: dict, tool_result: str, debug: bool = False):
    """Track engagement with specific posts or authors. DB KV buffer."""
    command = tool_input.get("command", "") if isinstance(tool_input, dict) else str(tool_input)
    if not command:
        return

    command_lower = command.lower()
    engaged_ids = set()
    engaged_authors = set()

    like_match = re.search(r'/posts/([a-f0-9-]+)/like', command_lower)
    if like_match:
        engaged_ids.add(like_match.group(1))

    if "post" in command_lower and "/posts" in command_lower and "parent_id" in command:
        parent_match = re.search(r'"parent_id"\s*:\s*"([^"]+)"', command)
        if parent_match:
            engaged_ids.add(parent_match.group(1))

    if ("post" in command_lower and "/posts" in command_lower
            and "@" in command and "like" not in command_lower):
        content_match = re.search(r'"content"\s*:\s*"([^"]*)"', command)
        if content_match:
            mentions = re.findall(r'@([a-zA-Z0-9_]+)', content_match.group(1))
            for m in mentions:
                engaged_authors.add(m.lower())

    if not engaged_ids and not engaged_authors:
        return

    try:
        from datetime import datetime, timezone
        db_mod = _get_mod("db_adapter")
        if not db_mod:
            return
        db = db_mod.get_db()
        raw = db.kv_get('.feed_engaged') or {}
        if isinstance(raw, str):
            raw = json.loads(raw)

        existing_ids = set(raw.get('post_ids', []))
        existing_authors = set(raw.get('authors', []))
        existing_ids.update(engaged_ids)
        existing_authors.update(engaged_authors)

        raw['post_ids'] = list(existing_ids)
        raw['authors'] = list(existing_authors)
        raw['updated'] = datetime.now(timezone.utc).isoformat()
        db.kv_set('.feed_engaged', raw)
    except Exception as e:
        if debug:
            print(f"DEBUG: Engagement tracking error: {e}", file=sys.stderr)


# ============================================================
# MAIN MEMORY PROCESSOR (in-process, was subprocess-heavy)
# ============================================================

def process_for_memory(tool_name: str, tool_result: str, debug: bool = False, tool_command: str = ""):
    """Route tool results to appropriate memory processor. ALL IN-PROCESS."""
    try:
        memory_dir = get_memory_dir()
        if not memory_dir or not memory_dir.exists():
            return

        api_type = detect_api_type(tool_result, tool_command)

        # === PLATFORM TRACKING (DB KV) ===
        if api_type != "unknown":
            try:
                from datetime import datetime, timezone
                db_mod = _get_mod("db_adapter")
                if db_mod:
                    db = db_mod.get_db()
                    raw = db.kv_get('.session_platforms')
                    data = {'platforms': [], 'updated': None}
                    if raw:
                        data = json.loads(raw) if isinstance(raw, str) else raw
                    platforms = data.get('platforms', [])
                    if api_type not in platforms:
                        platforms.append(api_type)
                        data['platforms'] = platforms
                        data['updated'] = datetime.now(timezone.utc).isoformat()
                        db.kv_set('.session_platforms', data)
            except Exception:
                pass

        # === WHO TRACKING (DB KV) ===
        try:
            mentions = {m.lower() for m in re.findall(r'@([a-zA-Z0-9_]+)', tool_result)}
            if '"username"' in tool_result or '"author"' in tool_result:
                usernames = re.findall(r'"(?:username|author|login|name)":\s*"([^"]+)"', tool_result)
                mentions.update(u.lower() for u in usernames if u and len(u) > 2)

            my_names = get_my_usernames()
            mentions = mentions - my_names
            mentions = {m for m in mentions if len(m) > 2 and not m.isdigit()}

            if mentions:
                from datetime import datetime, timezone
                db_mod = _get_mod("db_adapter")
                if db_mod:
                    db = db_mod.get_db()
                    raw = db.kv_get('.session_contacts')
                    data = {'contacts': [], 'updated': None}
                    if raw:
                        data = json.loads(raw) if isinstance(raw, str) else raw
                    contacts = set(data.get('contacts', []))
                    new_contacts = mentions - contacts
                    if new_contacts:
                        contacts.update(new_contacts)
                        data['contacts'] = list(contacts)
                        data['updated'] = datetime.now(timezone.utc).isoformat()
                        db.kv_set('.session_contacts', data)
        except Exception:
            pass

        # === AUTO REJECTION LOGGING ===
        if api_type in ("moltx", "moltbook", "clawtasks", "dead-internet", "github",
                         "clawbr", "thecolony", "agentlink", "lobsterpedia", "twitter"):
            try:
                arl = _get_mod("auto_rejection_logger")
                if arl:
                    arl.process_api_response(api_type, tool_result)
            except Exception:
                pass

        # === BEHAVIORAL TASTE TRACKING (feed-seen buffer) ===
        if api_type in ("moltx", "moltbook", "thecolony", "clawbr", "dead-internet",
                         "lobsterpedia", "twitter", "agentlink"):
            try:
                arl = _get_mod("auto_rejection_logger")
                if arl and hasattr(arl, "extract_seen_posts"):
                    passing = arl.extract_seen_posts(api_type, tool_result)
                    if passing:
                        from datetime import datetime, timezone
                        db_mod = _get_mod("db_adapter")
                        if db_mod:
                            db = db_mod.get_db()
                            raw = db.kv_get('.feed_seen') or {}
                            if isinstance(raw, str):
                                raw = json.loads(raw)
                            posts = raw.get('posts', {})
                            posts.update(passing)
                            raw['posts'] = posts
                            raw['updated'] = datetime.now(timezone.utc).isoformat()
                            db.kv_set('.feed_seen', raw)
            except Exception:
                pass

        # === FEED + SOCIAL + SHORT-TERM BUFFER (in-process) ===
        if api_type == "moltx":
            # Feed processor (was subprocess: python feed_processor.py --process-stdin)
            fp = _get_mod("feed_processor")
            if fp:
                try:
                    feed_data = json.loads(tool_result)
                    fp.process_moltx_feed(feed_data)
                except (json.JSONDecodeError, Exception):
                    pass
            # Social interactions (parallel)
            log_social_interaction("moltx", tool_result, debug)

        elif api_type in ("moltbook", "github", "clawtasks", "clawbr", "thecolony",
                           "agentlink", "lobsterpedia", "twitter"):
            # Short-term buffer (was subprocess: python auto_memory_hook.py --post-tool)
            amh = _get_mod("auto_memory_hook")
            if amh:
                try:
                    mem_item = amh.extract_from_tool_result(tool_name, tool_result[:1500])
                    if mem_item:
                        amh.add_to_short_term(mem_item)
                except Exception:
                    pass
            # Social interactions
            log_social_interaction(api_type, tool_result, debug)

    except Exception as e:
        if debug:
            print(f"DEBUG: Memory processing error: {e}", file=sys.stderr)


# ============================================================
# ADDITIONAL CONTEXT GENERATORS
# ============================================================

def _generate_lesson_context(tool_result: str, tool_command: str, memory_dir: Path) -> str:
    """Generate lesson context for errors. IN-PROCESS (was subprocess)."""
    try:
        result_lower = tool_result.lower()
        error_patterns = [
            "traceback (most recent",
            "exit code 1", "exit code 2",
            "httperror", "urlerror", "connectionerror",
            "status: 404", "status: 400", "status: 401",
            "status: 403", "status: 429", "status: 500",
            "status\": 404", "status\": 400", "status\": 401",
            "status\": 403", "status\": 429", "status\": 500",
            "connection refused", "rate limit exceeded",
            "unauthorized", "forbidden",
            "error:", "error\":",
            "failed:", "failure:",
        ]
        has_error = any(p in result_lower for p in error_patterns)
        if not has_error:
            return ""

        api_type = detect_api_type(tool_result, tool_command)
        error_hint = ""
        for line in tool_result.split('\n'):
            line_lower = line.lower().strip()
            if any(p in line_lower for p in error_patterns) and len(line.strip()) > 10:
                error_hint = line.strip()[:100]
                break

        situation = f"{api_type} API {error_hint}"

        le = _get_mod("lesson_extractor")
        if le and hasattr(le, "apply_lessons"):
            results = le.apply_lessons(situation)
            if results:
                # Format output like the CLI does
                lines = [f"=== ACTIVE LESSONS (heuristics from experience) ==="]
                for score, lesson in results[:3]:
                    lines.append(f"  [{lesson['category']}] {lesson['lesson']}")
                return "\n".join(lines)
        return ""
    except Exception:
        return ""


def _generate_thought_context(transcript_path: str, memory_dir: Path) -> str:
    """Generate thought priming context. IN-PROCESS (was subprocess)."""
    try:
        tp = _get_mod("thought_priming")
        if tp and hasattr(tp, "prime_from_thought"):
            output = tp.prime_from_thought(transcript_path, memory_dir)
            if output and "===" in output:
                return output
        return ""
    except Exception:
        return ""


# ============================================================
# MAIN
# ============================================================

def main():
    debug_mode = '--debug' in sys.argv
    try:
        input_data = json.load(sys.stdin)

        # === DRIFT MEMORY INTEGRATION ===
        tool_name = input_data.get("tool_name", "unknown")
        tool_input = input_data.get("tool_input", {})
        tool_response = input_data.get("tool_response", {})
        if isinstance(tool_response, dict):
            tool_result = str(tool_response.get("stdout", "") or tool_response.get("result", ""))
        else:
            tool_result = str(tool_response) if tool_response else ""
        if not tool_result:
            tool_result = str(input_data.get("tool_result", ""))

        tool_command = ""
        if isinstance(tool_input, dict):
            tool_command = tool_input.get("command", "")
        elif isinstance(tool_input, str):
            tool_command = tool_input

        # Process for memory (in-process, fast)
        if tool_result and len(tool_result) > 50:
            process_for_memory(tool_name, tool_result, debug=debug_mode, tool_command=tool_command)

        # Detect my posts/replies + track engagement from Bash commands
        if tool_name == "Bash" and tool_input:
            memory_dir = get_memory_dir()
            if memory_dir and memory_dir.exists():
                detect_my_post_from_command(tool_input, tool_result, debug=debug_mode)
                track_engagement(tool_input, tool_result, debug=debug_mode)

        # === ADDITIONAL CONTEXT (parallel: lesson injection + thought priming) ===
        lesson_context = ""
        thought_context = ""

        memory_dir = get_memory_dir()
        transcript_path = input_data.get("transcript_path", "")

        if memory_dir and memory_dir.exists() and tool_result:
            # Run both context generators in parallel
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_lesson = pool.submit(
                    _generate_lesson_context, tool_result, tool_command, memory_dir
                )
                f_thought = None
                if transcript_path:
                    f_thought = pool.submit(
                        _generate_thought_context, transcript_path, memory_dir
                    )

            try:
                lesson_context = f_lesson.result(timeout=6)
            except Exception:
                pass
            if f_thought:
                try:
                    thought_context = f_thought.result(timeout=9)
                except Exception:
                    pass

        # Output JSON with additionalContext if we have memory triggers
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

        # Log to file only if no additionalContext (low-priority I/O)
        log_dir = Path.cwd() / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / 'post_tool_use.json'

        if log_path.exists():
            with open(log_path, 'r') as f:
                try:
                    log_data = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    log_data = []
        else:
            log_data = []

        log_data.append(input_data)

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception:
        sys.exit(0)

if __name__ == '__main__':
    main()
