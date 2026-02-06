#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# ///
"""
Post-Tool-Use Hook - Biological Memory "Sensory Input"

Captures API responses and routes to appropriate processors:
- Platform activity tracking
- Contact tracking from mentions and authors
- Auto rejection logging
- Feed processing for MoltX
- Social interaction logging
- My-post auto-logging for duplicate prevention
- Thought priming from thinking blocks
"""

import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime


def load_config():
    """Load hooks_config.json"""
    config_paths = [
        Path(__file__).parent / "hooks_config.json",
        Path.home() / ".claude" / "hooks" / "hooks_config.json",
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass

    return {
        "memory_dirs": ["./memory", "."],
        "project_markers": ["memory_manager.py"],
        "my_usernames": [],
        "debug": False
    }


def get_memory_dir(config):
    """Find memory directory"""
    cwd = Path.cwd()

    for mem_dir in config.get("memory_dirs", ["./memory", "."]):
        candidate = cwd / mem_dir
        if candidate.exists() and (candidate / "memory_manager.py").exists():
            return candidate

    markers = config.get("project_markers", ["memory_manager.py"])
    current = cwd
    for _ in range(10):
        for marker in markers:
            if (current / marker).exists():
                return current
        if current.parent == current:
            break
        current = current.parent

    return None


def is_my_post(item, config):
    """Check if item is from configured usernames"""
    my_usernames = config.get("my_usernames", [])
    if not my_usernames:
        return False

    # Check various author field formats
    author = item.get("author")
    if isinstance(author, dict):
        username = author.get("username") or author.get("login") or author.get("name")
    else:
        username = author

    author_name = item.get("author_name")
    user = item.get("user")
    if isinstance(user, dict):
        user_login = user.get("login")
    else:
        user_login = user

    return (username in my_usernames or
            author_name in my_usernames or
            user_login in my_usernames)


def detect_api_type(tool_result):
    """Detect platform from API response"""
    result_lower = tool_result.lower()

    if "moltx.io" in result_lower or '"moltx_notice"' in result_lower:
        return "moltx"
    elif "moltbook.com" in result_lower:
        return "moltbook"
    elif "api.github.com" in result_lower or "github.com/repos" in result_lower:
        return "github"
    elif "clawtasks.com" in result_lower:
        return "clawtasks"
    elif "lobsterpedia.com" in result_lower:
        return "lobsterpedia"
    elif "mydeadinternet.com" in result_lower:
        return "dead-internet"
    elif "nostr" in result_lower or '"kind":' in result_lower:
        return "nostr"

    return "unknown"


def track_platform_activity(memory_dir, platform):
    """Track which platforms were used this session"""
    try:
        session_file = memory_dir / ".session_platforms.json"
        platforms = {}

        if session_file.exists():
            with open(session_file, 'r', encoding='utf-8') as f:
                platforms = json.load(f)

        platforms[platform] = platforms.get(platform, 0) + 1

        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(platforms, f)

    except Exception:
        pass


def track_contacts(memory_dir, contacts, platform):
    """Track contacts mentioned or interacted with"""
    try:
        session_file = memory_dir / ".session_contacts.json"
        all_contacts = {}

        if session_file.exists():
            with open(session_file, 'r', encoding='utf-8') as f:
                all_contacts = json.load(f)

        for contact in contacts:
            key = f"{contact}@{platform}"
            all_contacts[key] = all_contacts.get(key, 0) + 1

        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(all_contacts, f)

    except Exception:
        pass


def extract_contacts(tool_result, platform):
    """Extract @mentions and author names from response"""
    contacts = set()

    try:
        data = json.loads(tool_result)
        items = data if isinstance(data, list) else [data]

        for item in items[:20]:  # First 20 items
            if not isinstance(item, dict):
                continue

            # Extract author
            author = item.get("author")
            if isinstance(author, dict):
                username = author.get("username") or author.get("login")
                if username:
                    contacts.add(username)
            elif author:
                contacts.add(str(author))

            # Extract from author_name field
            author_name = item.get("author_name")
            if author_name:
                contacts.add(author_name)

            # Extract @mentions from content
            content = item.get("content") or item.get("body") or ""
            if content:
                words = content.split()
                for word in words:
                    if word.startswith("@"):
                        mention = word[1:].rstrip(",:;.!?")
                        if mention:
                            contacts.add(mention)

    except Exception:
        pass

    return list(contacts)


def safe_run(cmd, cwd, timeout=5):
    """Run command safely"""
    try:
        subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        pass


def main():
    global config
    config = load_config()

    # Read hook input
    try:
        hook_input = json.loads(sys.stdin.read())
        tool_name = hook_input.get("toolName", "")
        tool_result = hook_input.get("toolResult", "")
    except:
        print(json.dumps({"hookSpecificOutput": {"status": "parse_error"}}))
        sys.exit(0)

    # Find memory directory
    memory_dir = get_memory_dir(config)
    if not memory_dir:
        print(json.dumps({"hookSpecificOutput": {"status": "no_memory_system"}}))
        sys.exit(0)

    # Detect API type
    api_type = detect_api_type(tool_result)

    if api_type == "unknown":
        # Not an API we care about
        sys.exit(0)

    # Track platform activity
    track_platform_activity(memory_dir, api_type)

    # Extract and track contacts
    contacts = extract_contacts(tool_result, api_type)
    if contacts:
        track_contacts(memory_dir, contacts, api_type)

    # Auto rejection logging
    auto_rejection = memory_dir / "auto_rejection_logger.py"
    if auto_rejection.exists():
        try:
            # Import and call process_api_response
            safe_run(
                ["python", "-c",
                 f"from auto_rejection_logger import process_api_response; "
                 f"import json; "
                 f"process_api_response({repr(api_type)}, json.loads({repr(tool_result)}))"],
                cwd=memory_dir
            )
        except Exception:
            pass

    # Route to platform-specific processors
    if api_type == "moltx":
        feed_processor = memory_dir / "feed_processor.py"
        if feed_processor.exists():
            safe_run(
                ["python", str(feed_processor), "--stdin"],
                cwd=memory_dir
            )

    elif api_type in ["clawtasks", "github", "moltbook"]:
        auto_memory = memory_dir / "auto_memory_hook.py"
        if auto_memory.exists():
            safe_run(
                ["python", str(auto_memory), "--api-response", tool_result],
                cwd=memory_dir
            )

    # Social interaction logging
    social_script = memory_dir / "social" / "social_memory.py"
    if social_script.exists():
        try:
            data = json.loads(tool_result)
            items = data if isinstance(data, list) else [data]

            for item in items[:5]:
                if not isinstance(item, dict):
                    continue

                # Skip my own posts
                if is_my_post(item, config):
                    continue

                contact = None
                interaction_type = "interaction"
                content = ""
                url = None

                # Extract based on platform
                if api_type == "moltx":
                    author = item.get("author", {})
                    contact = author.get("username") if isinstance(author, dict) else author
                    if not contact:
                        contact = item.get("author_name")
                    interaction_type = "reply" if item.get("parent_id") else "post"
                    content = item.get("content", "")[:150]
                    url = f"https://moltx.io/post/{item.get('id')}" if item.get('id') else None

                elif api_type == "github":
                    user = item.get("user", {})
                    contact = user.get("login") if isinstance(user, dict) else None
                    if "pull_request" in str(item.get("html_url", "")):
                        interaction_type = "pr"
                    elif "/issues/" in str(item.get("html_url", "")):
                        interaction_type = "issue"
                    else:
                        interaction_type = "comment"
                    content = item.get("title") or item.get("body", "")[:150]
                    url = item.get("html_url")

                elif api_type == "moltbook":
                    contact = item.get("author_name")
                    interaction_type = "post" if item.get("submolt") else "comment"
                    content = item.get("content", "")[:150]

                if contact and content:
                    safe_run(
                        ["python", str(social_script), "log", contact, api_type,
                         interaction_type, content, "--url", url or ""],
                        cwd=memory_dir
                    )

        except Exception:
            pass

    # My-post auto-logging (for duplicate prevention)
    try:
        data = json.loads(tool_result)
        items = data if isinstance(data, list) else [data]

        for item in items[:5]:
            if isinstance(item, dict) and is_my_post(item, config):
                # Log to .my_posts.json
                my_posts_file = memory_dir / ".my_posts.json"
                my_posts = []

                if my_posts_file.exists():
                    with open(my_posts_file, 'r', encoding='utf-8') as f:
                        my_posts = json.load(f)

                post_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "platform": api_type,
                    "id": item.get("id"),
                    "content": item.get("content", "")[:100]
                }

                my_posts.append(post_entry)

                # Keep only last 100
                if len(my_posts) > 100:
                    my_posts = my_posts[-100:]

                with open(my_posts_file, 'w', encoding='utf-8') as f:
                    json.dump(my_posts, f, indent=2)

    except Exception:
        pass

    # Thought priming - search memories based on thinking blocks
    primed_context = None
    thought_priming = memory_dir / "thought_priming.py"
    if thought_priming.exists():
        try:
            result = subprocess.run(
                ["python", str(thought_priming), "test"],
                cwd=memory_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                primed_context = result.stdout.strip()
        except Exception:
            pass

    # Output
    output = {
        "hookSpecificOutput": {
            "status": "success",
            "api_type": api_type,
            "contacts_tracked": len(contacts)
        }
    }

    if primed_context:
        output["hookSpecificOutput"]["additionalContext"] = primed_context

    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
