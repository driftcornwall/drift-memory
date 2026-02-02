#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///

"""
Social Memory System for Drift

Tracks relationships, conversations, and social context across platforms.
Designed for positive-sum growth through maintained relationships.

Platforms tracked:
- MoltX (Twitter for agents)
- Moltbook (Reddit for agents)
- GitHub (code collaboration)
- ClawTasks (bounty marketplace)

Usage:
    python social_memory.py log <contact> <platform> <interaction_type> "<content>"
    python social_memory.py contact <name>           # View contact details
    python social_memory.py recent [--limit N]      # Recent interactions across all contacts
    python social_memory.py prime                   # Output priming context for session start
    python social_memory.py index                   # Rebuild social_index.json
"""

import json
import sys
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import yaml

SOCIAL_DIR = Path(__file__).parent
CONTACTS_DIR = SOCIAL_DIR / "contacts"
THREADS_DIR = SOCIAL_DIR / "threads"
ARCHIVE_DIR = SOCIAL_DIR / "archive"
INDEX_FILE = SOCIAL_DIR / "social_index.json"

# Ensure directories exist
CONTACTS_DIR.mkdir(exist_ok=True)
THREADS_DIR.mkdir(exist_ok=True)
ARCHIVE_DIR.mkdir(exist_ok=True)

# Maximum recent interactions before archiving
MAX_RECENT = 10
ARCHIVE_THRESHOLD = 5  # Archive oldest N when limit hit


def normalize_contact_name(name: str) -> str:
    """Normalize contact name for filename."""
    return name.lower().replace(" ", "-").replace("_", "-")


def get_contact_path(name: str) -> Path:
    """Get path to contact file."""
    return CONTACTS_DIR / f"{normalize_contact_name(name)}.md"


def load_contact(name: str) -> dict:
    """Load contact data from markdown file with YAML frontmatter."""
    path = get_contact_path(name)
    if not path.exists():
        return {
            "name": name,
            "platforms": {},
            "relationship": "",
            "recent": [],
            "tags": [],
            "first_contact": None,
            "last_contact": None
        }

    content = path.read_text(encoding='utf-8')

    # Parse YAML frontmatter
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            try:
                data = yaml.safe_load(parts[1])
                data['_body'] = parts[2].strip()
                return data
            except yaml.YAMLError:
                pass

    return {
        "name": name,
        "platforms": {},
        "relationship": "",
        "recent": [],
        "tags": [],
        "_body": content
    }


def save_contact(name: str, data: dict):
    """Save contact data to markdown file with YAML frontmatter."""
    path = get_contact_path(name)

    # Separate body from frontmatter data
    body = data.pop('_body', '')

    # Build frontmatter
    frontmatter = yaml.dump(data, default_flow_style=False, allow_unicode=True)

    # Reconstruct file
    content = f"---\n{frontmatter}---\n\n{body}"

    path.write_text(content, encoding='utf-8')

    # Put body back for caller
    data['_body'] = body


def log_interaction(contact: str, platform: str, interaction_type: str, content: str,
                    thread_id: str = None, url: str = None):
    """
    Log a social interaction with a contact.

    Args:
        contact: Contact name (e.g., "SpindriftMend")
        platform: Platform name (moltx, moltbook, github, clawtasks)
        interaction_type: Type (post, reply, comment, pr, issue, dm, mention)
        content: Brief description of the interaction
        thread_id: Optional thread identifier for threading
        url: Optional URL to the interaction
    """
    data = load_contact(contact)
    now = datetime.now().isoformat()

    # Create interaction record
    interaction = {
        "timestamp": now,
        "platform": platform.lower(),
        "type": interaction_type.lower(),
        "content": content[:500],  # Truncate long content
    }
    if thread_id:
        interaction["thread_id"] = thread_id
    if url:
        interaction["url"] = url

    # Update contact data
    if "recent" not in data:
        data["recent"] = []

    data["recent"].insert(0, interaction)  # Most recent first
    data["last_contact"] = now

    if not data.get("first_contact"):
        data["first_contact"] = now

    # Track platforms
    if "platforms" not in data:
        data["platforms"] = {}
    if platform.lower() not in data["platforms"]:
        data["platforms"][platform.lower()] = {"username": contact}

    # Archive if too many recent
    if len(data["recent"]) > MAX_RECENT:
        archive_old_interactions(contact, data)

    save_contact(contact, data)
    update_index()

    return interaction


def archive_old_interactions(contact: str, data: dict):
    """Move oldest interactions to archive file."""
    if len(data["recent"]) <= MAX_RECENT:
        return

    # Split: keep recent, archive old
    to_archive = data["recent"][MAX_RECENT - ARCHIVE_THRESHOLD:]
    data["recent"] = data["recent"][:MAX_RECENT - ARCHIVE_THRESHOLD]

    # Append to archive file
    archive_file = ARCHIVE_DIR / f"{normalize_contact_name(contact)}.jsonl"
    with open(archive_file, 'a', encoding='utf-8') as f:
        for interaction in to_archive:
            f.write(json.dumps(interaction) + '\n')


def get_contact_summary(name: str) -> dict:
    """Get a summary of a contact for priming."""
    data = load_contact(name)

    recent_count = len(data.get("recent", []))
    last_contact = data.get("last_contact")

    # Calculate recency
    recency = "never"
    if last_contact:
        try:
            last_dt = datetime.fromisoformat(last_contact)
            delta = datetime.now() - last_dt
            if delta < timedelta(hours=24):
                recency = "today"
            elif delta < timedelta(days=2):
                recency = "yesterday"
            elif delta < timedelta(days=7):
                recency = "this_week"
            else:
                recency = "older"
        except:
            pass

    return {
        "name": data.get("name", name),
        "relationship": data.get("relationship", ""),
        "platforms": list(data.get("platforms", {}).keys()),
        "tags": data.get("tags", []),
        "interaction_count": recent_count,
        "last_contact": last_contact,
        "recency": recency,
        "recent_preview": data.get("recent", [])[:3]  # Last 3 interactions
    }


def update_index():
    """Rebuild the social index file for quick priming."""
    contacts = []

    for contact_file in CONTACTS_DIR.glob("*.md"):
        name = contact_file.stem
        summary = get_contact_summary(name)
        contacts.append(summary)

    # Sort by recency (most recent first)
    recency_order = {"today": 0, "yesterday": 1, "this_week": 2, "older": 3, "never": 4}
    contacts.sort(key=lambda x: (recency_order.get(x["recency"], 5), x["name"]))

    index = {
        "updated": datetime.now().isoformat(),
        "total_contacts": len(contacts),
        "active_today": sum(1 for c in contacts if c["recency"] == "today"),
        "active_week": sum(1 for c in contacts if c["recency"] in ["today", "yesterday", "this_week"]),
        "contacts": contacts
    }

    INDEX_FILE.write_text(json.dumps(index, indent=2), encoding='utf-8')
    return index


def get_priming_context(limit: int = 5) -> str:
    """
    Generate priming context for session start.
    Returns markdown-formatted social context.
    """
    # Load or rebuild index
    if INDEX_FILE.exists():
        index = json.loads(INDEX_FILE.read_text(encoding='utf-8'))
    else:
        index = update_index()

    lines = ["## Social Context (auto-primed)\n"]
    lines.append(f"**Active contacts:** {index['active_week']} this week, {index['total_contacts']} total\n")

    # Show most recent contacts
    active = [c for c in index["contacts"] if c["recency"] in ["today", "yesterday", "this_week"]][:limit]

    if active:
        lines.append("### Recent Relationships\n")
        for contact in active:
            rel = contact.get("relationship", "")[:100]
            platforms = ", ".join(contact.get("platforms", []))
            lines.append(f"**{contact['name']}** ({platforms})")
            if rel:
                lines.append(f"  {rel}")

            # Show last interaction
            if contact.get("recent_preview"):
                last = contact["recent_preview"][0]
                lines.append(f"  Last: [{last.get('type')}] {last.get('content', '')[:80]}...")
            lines.append("")

    return '\n'.join(lines)


def view_contact(name: str) -> str:
    """View full contact details."""
    data = load_contact(name)

    lines = [f"# {data.get('name', name)}\n"]

    if data.get("relationship"):
        lines.append(f"**Relationship:** {data['relationship']}\n")

    if data.get("platforms"):
        platforms = [f"{k}: {v.get('username', '?')}" for k, v in data["platforms"].items()]
        lines.append(f"**Platforms:** {', '.join(platforms)}\n")

    if data.get("tags"):
        lines.append(f"**Tags:** {', '.join(data['tags'])}\n")

    if data.get("first_contact"):
        lines.append(f"**First contact:** {data['first_contact'][:10]}")
    if data.get("last_contact"):
        lines.append(f"**Last contact:** {data['last_contact'][:10]}\n")

    # Recent interactions
    if data.get("recent"):
        lines.append("## Recent Interactions\n")
        for i, interaction in enumerate(data["recent"][:10]):
            ts = interaction.get("timestamp", "")[:10]
            platform = interaction.get("platform", "?")
            itype = interaction.get("type", "?")
            content = interaction.get("content", "")[:100]
            url = interaction.get("url", "")

            line = f"{i+1}. [{ts}] **{platform}/{itype}**: {content}"
            if url:
                line += f" [link]({url})"
            lines.append(line)

    return '\n'.join(lines)


def list_recent(limit: int = 20) -> str:
    """List recent interactions across all contacts."""
    all_interactions = []

    for contact_file in CONTACTS_DIR.glob("*.md"):
        name = contact_file.stem
        data = load_contact(name)
        for interaction in data.get("recent", []):
            interaction["contact"] = name
            all_interactions.append(interaction)

    # Sort by timestamp (most recent first)
    all_interactions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    lines = ["# Recent Social Activity\n"]
    for interaction in all_interactions[:limit]:
        ts = interaction.get("timestamp", "")[:16]
        contact = interaction.get("contact", "?")
        platform = interaction.get("platform", "?")
        itype = interaction.get("type", "?")
        content = interaction.get("content", "")[:80]

        lines.append(f"- [{ts}] **{contact}** ({platform}/{itype}): {content}")

    return '\n'.join(lines)


# ============ AUTO-DETECTION HELPERS ============

def detect_contact_from_api_response(platform: str, response_data: dict) -> Optional[dict]:
    """
    Try to extract contact and interaction info from API response.
    Returns dict with contact, type, content, url, thread_id if found.
    """
    if platform == "moltx":
        # MoltX post/reply detection
        if "author" in response_data:
            author = response_data.get("author", {})
            return {
                "contact": author.get("username") or author.get("name"),
                "type": "post" if not response_data.get("parent_id") else "reply",
                "content": response_data.get("content", "")[:200],
                "url": f"https://moltx.io/post/{response_data.get('id')}",
                "thread_id": response_data.get("parent_id") or response_data.get("id")
            }

    elif platform == "moltbook":
        # Moltbook post/comment detection
        if "author" in response_data:
            return {
                "contact": response_data.get("author"),
                "type": "post" if response_data.get("type") == "post" else "comment",
                "content": response_data.get("title") or response_data.get("content", "")[:200],
                "url": f"https://moltbook.com/post/{response_data.get('id')}",
                "thread_id": response_data.get("post_id") or response_data.get("id")
            }

    elif platform == "github":
        # GitHub issue/PR/comment detection
        if "user" in response_data:
            user = response_data.get("user", {})
            itype = "comment"
            if "pull_request" in response_data:
                itype = "pr"
            elif "issue" in str(response_data.get("html_url", "")):
                itype = "issue"

            return {
                "contact": user.get("login"),
                "type": itype,
                "content": response_data.get("title") or response_data.get("body", "")[:200],
                "url": response_data.get("html_url"),
                "thread_id": str(response_data.get("issue_number") or response_data.get("number"))
            }

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drift's Social Memory System")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # log command
    log_parser = subparsers.add_parser("log", help="Log an interaction")
    log_parser.add_argument("contact", help="Contact name")
    log_parser.add_argument("platform", help="Platform (moltx, moltbook, github, clawtasks)")
    log_parser.add_argument("type", help="Interaction type (post, reply, comment, pr, issue, dm)")
    log_parser.add_argument("content", help="Brief description")
    log_parser.add_argument("--thread", help="Thread ID")
    log_parser.add_argument("--url", help="URL to interaction")

    # contact command
    contact_parser = subparsers.add_parser("contact", help="View contact details")
    contact_parser.add_argument("name", help="Contact name")

    # recent command
    recent_parser = subparsers.add_parser("recent", help="List recent interactions")
    recent_parser.add_argument("--limit", type=int, default=20, help="Limit results")

    # prime command
    prime_parser = subparsers.add_parser("prime", help="Output priming context")
    prime_parser.add_argument("--limit", type=int, default=5, help="Max contacts to show")

    # index command
    index_parser = subparsers.add_parser("index", help="Rebuild social index")

    args = parser.parse_args()

    if args.command == "log":
        result = log_interaction(
            args.contact, args.platform, args.type, args.content,
            thread_id=args.thread, url=args.url
        )
        print(json.dumps(result, indent=2))

    elif args.command == "contact":
        print(view_contact(args.name))

    elif args.command == "recent":
        print(list_recent(args.limit))

    elif args.command == "prime":
        print(get_priming_context(args.limit))

    elif args.command == "index":
        index = update_index()
        print(f"Index updated: {index['total_contacts']} contacts, {index['active_week']} active this week")

    else:
        parser.print_help()
