#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg2-binary"]
# ///

"""
Social Memory System for Drift

Tracks relationships, conversations, and social context across platforms.
Designed for positive-sum growth through maintained relationships.

All data stored in PostgreSQL via db_adapter.get_db().kv_get/kv_set.
NO file reads or writes for data storage.

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
    python social_memory.py index                   # Rebuild social index
"""

import json
from datetime import datetime, timedelta
from typing import Optional

# DB keys
KV_CONTACTS = '.social_contacts'       # {normalized_name: contact_data}
KV_INDEX = '.social_index'             # aggregated index
KV_MY_POSTS = '.social_my_posts'       # {updated, description, posts: [...]}
KV_MY_REPLIES = '.social_my_replies'   # {updated, description, replies: {...}}

# Maximum recent interactions before archiving
MAX_RECENT = 10
ARCHIVE_THRESHOLD = 5  # Archive oldest N when limit hit


def _get_db():
    """Get the DB instance. Import here to avoid circular imports."""
    from db_adapter import get_db
    return get_db()


def normalize_contact_name(name: str) -> str:
    """Normalize contact name for storage key."""
    return name.lower().replace(" ", "-").replace("_", "-")


def _load_all_contacts() -> dict:
    """Load all contacts dict from DB. Returns {normalized_name: contact_data}."""
    db = _get_db()
    data = db.kv_get(KV_CONTACTS)
    if data is None:
        return {}
    return data


def _save_all_contacts(contacts: dict):
    """Save all contacts dict to DB."""
    db = _get_db()
    db.kv_set(KV_CONTACTS, contacts)


def load_contact(name: str) -> dict:
    """Load contact data from DB."""
    contacts = _load_all_contacts()
    normalized = normalize_contact_name(name)

    if normalized in contacts:
        return contacts[normalized]

    return {
        "name": name,
        "platforms": {},
        "relationship": "",
        "recent": [],
        "tags": [],
        "first_contact": None,
        "last_contact": None
    }


def save_contact(name: str, data: dict):
    """Save contact data to DB."""
    contacts = _load_all_contacts()
    normalized = normalize_contact_name(name)

    # Remove _body if present (legacy field from markdown storage)
    data.pop('_body', None)

    contacts[normalized] = data
    _save_all_contacts(contacts)


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
    """Move oldest interactions to archive storage in DB."""
    if len(data["recent"]) <= MAX_RECENT:
        return

    # Split: keep recent, archive old
    to_archive = data["recent"][MAX_RECENT - ARCHIVE_THRESHOLD:]
    data["recent"] = data["recent"][:MAX_RECENT - ARCHIVE_THRESHOLD]

    # Store archived interactions in DB
    db = _get_db()
    archive_key = f'.social_archive_{normalize_contact_name(contact)}'
    existing = db.kv_get(archive_key) or []
    existing.extend(to_archive)
    db.kv_set(archive_key, existing)


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
    """Rebuild the social index in DB for quick priming."""
    contacts_data = _load_all_contacts()
    contacts = []

    for normalized_name, contact_data in contacts_data.items():
        name = contact_data.get("name", normalized_name)
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

    db = _get_db()
    db.kv_set(KV_INDEX, index)
    return index


def get_priming_context(limit: int = 5, include_replies: bool = True) -> str:
    """
    Generate priming context for session start.
    Returns markdown-formatted social context.
    """
    # Load or rebuild index
    db = _get_db()
    index = db.kv_get(KV_INDEX)
    if index is None:
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

    # Add my recent posts (continuity - what have I been saying?)
    posts_context = get_posts_priming_context(limit=3)
    if posts_context:
        lines.append(posts_context)

    # Add my recent replies to avoid duplicates
    if include_replies:
        replies_context = get_replies_priming_context(days=3, limit=10)
        if replies_context:
            lines.append(replies_context)

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

    contacts_data = _load_all_contacts()
    for normalized_name, contact_data in contacts_data.items():
        name = contact_data.get("name", normalized_name)
        for interaction in contact_data.get("recent", []):
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


def _load_my_posts() -> dict:
    """Load my posts tracking from DB."""
    db = _get_db()
    data = db.kv_get(KV_MY_POSTS)
    if data is None:
        return {"updated": datetime.now().isoformat(), "description": "Tracks my feed posts", "posts": []}
    return data


def _save_my_posts(data: dict):
    """Save my posts tracking to DB."""
    data["updated"] = datetime.now().isoformat()
    db = _get_db()
    db.kv_set(KV_MY_POSTS, data)


def log_my_post(platform: str, post_id: str, content: str, url: str = ""):
    """
    Log a post I made on my feed. Call this AFTER posting.

    Args:
        platform: moltx, moltbook
        post_id: The unique ID of my post
        content: What I posted
        url: URL to the post

    Returns:
        The post record created
    """
    data = _load_my_posts()

    record = {
        "platform": platform.lower(),
        "post_id": str(post_id),
        "content": content[:500],
        "url": url,
        "timestamp": datetime.now().isoformat()
    }

    # Add to front of list (most recent first)
    data["posts"].insert(0, record)

    # Keep only last 50 posts
    data["posts"] = data["posts"][:50]

    _save_my_posts(data)
    return record


def get_my_recent_posts(days: int = 7, limit: int = 10) -> list[dict]:
    """Get my recent posts for priming context."""
    data = _load_my_posts()
    cutoff = datetime.now() - timedelta(days=days)

    recent = []
    for post in data.get("posts", []):
        try:
            ts = datetime.fromisoformat(post["timestamp"])
            if ts > cutoff:
                recent.append(post)
        except (KeyError, ValueError):
            continue

    return recent[:limit]


def get_posts_priming_context(limit: int = 3) -> str:
    """Generate priming context showing my recent posts."""
    recent = get_my_recent_posts(days=7, limit=limit)

    if not recent:
        return ""

    lines = ["\n### My Recent Posts (what I've been saying)\n"]

    for p in recent:
        platform = p.get("platform", "?")
        ts = p.get("timestamp", "")[:10]
        content = p.get("content", "")[:100]

        lines.append(f"- [{platform}] ({ts}): \"{content}...\"")

    return "\n".join(lines)


def _load_my_replies() -> dict:
    """Load my replies tracking from DB."""
    db = _get_db()
    data = db.kv_get(KV_MY_REPLIES)
    if data is None:
        return {"updated": datetime.now().isoformat(), "description": "Tracks posts I have replied to", "replies": {}}
    return data


def _save_my_replies(data: dict):
    """Save my replies tracking to DB."""
    data["updated"] = datetime.now().isoformat()
    db = _get_db()
    db.kv_set(KV_MY_REPLIES, data)


def log_my_reply(platform: str, post_id: str, my_content: str, author: str = "", url: str = ""):
    """
    Log that I replied to a specific post. Call this AFTER making a reply.

    Args:
        platform: moltx, moltbook, github
        post_id: The unique ID of the post I replied to
        my_content: Brief summary of what I said
        author: Original post author (for context)
        url: URL to the post

    Returns:
        The reply record created
    """
    # Also log to the dedicated social_replies table for check_replied()
    db = _get_db()
    db.log_reply(platform.lower(), str(post_id), author, my_content[:300])

    # And store full context in kv for priming
    data = _load_my_replies()
    key = f"{platform.lower()}_{post_id}"

    record = {
        "platform": platform.lower(),
        "post_id": str(post_id),
        "author": author,
        "my_reply": my_content[:300],
        "url": url,
        "timestamp": datetime.now().isoformat()
    }

    data["replies"][key] = record
    _save_my_replies(data)

    return record


def have_i_replied(platform: str, post_id: str) -> Optional[dict]:
    """
    Check if I've already replied to a specific post.

    Returns:
        The reply record if I have replied, None otherwise
    """
    # First check the dedicated social_replies table (fast)
    db = _get_db()
    if db.check_replied(platform.lower(), str(post_id)):
        # Get the full record from kv if available
        data = _load_my_replies()
        key = f"{platform.lower()}_{post_id}"
        record = data["replies"].get(key)
        if record:
            return record
        # Exists in social_replies but not kv — return minimal record
        return {"platform": platform.lower(), "post_id": str(post_id), "my_reply": "(logged in DB)", "timestamp": ""}

    return None


def get_my_recent_replies(days: int = 7, limit: int = 20) -> list[dict]:
    """Get my recent replies for priming context."""
    data = _load_my_replies()
    cutoff = datetime.now() - timedelta(days=days)

    recent = []
    for key, record in data["replies"].items():
        try:
            ts = datetime.fromisoformat(record["timestamp"])
            if ts > cutoff:
                recent.append(record)
        except (KeyError, ValueError):
            continue

    # Sort by timestamp descending
    recent.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return recent[:limit]


def get_replies_priming_context(days: int = 3, limit: int = 10) -> str:
    """Generate priming context showing posts I've already replied to."""
    recent = get_my_recent_replies(days=days, limit=limit)

    if not recent:
        return ""

    lines = ["\n### Posts I've Already Replied To (avoid duplicates)\n"]

    for r in recent:
        platform = r.get("platform", "?")
        post_id = r.get("post_id", "?")
        author = r.get("author", "unknown")
        ts = r.get("timestamp", "")[:10]
        my_reply = r.get("my_reply", "")[:60]

        lines.append(f"- [{platform}] {author}'s post {post_id} ({ts}): \"{my_reply}...\"")

    return "\n".join(lines)


def generate_embeddable_context():
    """
    Generate social context data and store it in DB via kv_set.
    Returns a summary string of what was stored.
    """
    lines = []
    lines.append("---")
    lines.append("id: social-context")
    lines.append(f"created: '{datetime.now().isoformat()}'")
    lines.append("type: active")
    lines.append("tags:")
    lines.append("- social")
    lines.append("- moltx")
    lines.append("- auto-generated")
    lines.append("emotional_weight: 0.3")
    lines.append("---")
    lines.append("")
    lines.append("# Social Context (Auto-Generated)")
    lines.append("")

    # Add my recent posts
    posts = get_my_recent_posts(days=7, limit=5)
    if posts:
        lines.append("## My Recent Posts")
        lines.append("")
        for p in posts:
            ts = p.get("timestamp", "")[:10]
            platform = p.get("platform", "?")
            content = p.get("content", "")[:150]
            lines.append(f"- [{platform}] ({ts}): {content}")
        lines.append("")

    # Add posts I've replied to
    replies = get_my_recent_replies(days=7, limit=10)
    if replies:
        lines.append("## Posts I've Already Replied To")
        lines.append("**Check before replying to avoid duplicates**")
        lines.append("")
        for r in replies:
            ts = r.get("timestamp", "")[:10]
            platform = r.get("platform", "?")
            author = r.get("author", "unknown")
            post_id = r.get("post_id", "?")
            my_reply = r.get("my_reply", "")[:100]
            lines.append(f"- [{platform}] {author}'s post {post_id[:8]}... ({ts}): replied \"{my_reply}...\"")
        lines.append("")

    # Add recent contacts
    db = _get_db()
    index = db.kv_get(KV_INDEX)
    if index:
        active = [c for c in index.get("contacts", [])
                 if c.get("recency") in ["today", "yesterday", "this_week"]][:5]
        if active:
            lines.append("## Recent Social Contacts")
            lines.append("")
            for c in active:
                name = c.get("name", "?")
                platforms = ", ".join(c.get("platforms", []))
                lines.append(f"- **{name}** ({platforms})")
            lines.append("")

    # Store the embeddable context in DB
    content = "\n".join(lines)
    db.kv_set('.social_embeddable_context', {
        "content": content,
        "generated": datetime.now().isoformat()
    })

    return f"Stored embeddable social context in DB (.social_embeddable_context), {len(lines)} lines"


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

    # replied command - log that I replied to a post
    replied_parser = subparsers.add_parser("replied", help="Log that I replied to a post")
    replied_parser.add_argument("platform", help="Platform (moltx, moltbook, github)")
    replied_parser.add_argument("post_id", help="Post ID I replied to")
    replied_parser.add_argument("content", help="Brief summary of my reply")
    replied_parser.add_argument("--author", default="", help="Original post author")
    replied_parser.add_argument("--url", default="", help="URL to post")

    # check command - check if I've already replied
    check_parser = subparsers.add_parser("check", help="Check if I've already replied to a post")
    check_parser.add_argument("platform", help="Platform (moltx, moltbook, github)")
    check_parser.add_argument("post_id", help="Post ID to check")

    # my-replies command - list my recent replies
    myreplies_parser = subparsers.add_parser("my-replies", help="List my recent replies")
    myreplies_parser.add_argument("--days", type=int, default=7, help="Days to look back")
    myreplies_parser.add_argument("--limit", type=int, default=20, help="Limit results")

    # posted command - log a post I made
    posted_parser = subparsers.add_parser("posted", help="Log a post I made on my feed")
    posted_parser.add_argument("platform", help="Platform (moltx, moltbook)")
    posted_parser.add_argument("post_id", help="Post ID")
    posted_parser.add_argument("content", help="What I posted")
    posted_parser.add_argument("--url", default="", help="URL to post")

    # my-posts command - list my recent posts
    myposts_parser = subparsers.add_parser("my-posts", help="List my recent posts")
    myposts_parser.add_argument("--days", type=int, default=7, help="Days to look back")
    myposts_parser.add_argument("--limit", type=int, default=10, help="Limit results")

    # embed command - generate embeddable social context
    embed_parser = subparsers.add_parser("embed", help="Generate embeddable social context in DB")

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

    elif args.command == "replied":
        result = log_my_reply(
            args.platform, args.post_id, args.content,
            author=args.author, url=args.url
        )
        print(f"Logged reply to {args.platform} post {args.post_id}")
        print(json.dumps(result, indent=2))

    elif args.command == "check":
        result = have_i_replied(args.platform, args.post_id)
        if result:
            print(f"YES - Already replied on {result['timestamp'][:10]}")
            print(f"  My reply: {result['my_reply'][:100]}...")
        else:
            print(f"NO - Haven't replied to {args.platform} post {args.post_id}")

    elif args.command == "my-replies":
        replies = get_my_recent_replies(days=args.days, limit=args.limit)
        if not replies:
            print(f"No replies in the last {args.days} days")
        else:
            print(f"# My Recent Replies ({len(replies)} in last {args.days} days)\n")
            for r in replies:
                ts = r.get("timestamp", "")[:10]
                platform = r.get("platform", "?")
                author = r.get("author", "unknown")
                post_id = r.get("post_id", "?")
                my_reply = r.get("my_reply", "")[:80]
                print(f"- [{ts}] {platform}/{post_id} (to {author}): {my_reply}...")

    elif args.command == "posted":
        result = log_my_post(
            args.platform, args.post_id, args.content, url=args.url
        )
        print(f"Logged my post on {args.platform}: {args.post_id}")
        print(json.dumps(result, indent=2))

    elif args.command == "my-posts":
        posts = get_my_recent_posts(days=args.days, limit=args.limit)
        if not posts:
            print(f"No posts in the last {args.days} days")
        else:
            print(f"# My Recent Posts ({len(posts)} in last {args.days} days)\n")
            for p in posts:
                ts = p.get("timestamp", "")[:10]
                platform = p.get("platform", "?")
                post_id = p.get("post_id", "?")
                content = p.get("content", "")[:80]
                print(f"- [{ts}] {platform}/{post_id}: {content}...")

    elif args.command == "embed":
        output = generate_embeddable_context()
        print(output)

    else:
        parser.print_help()
