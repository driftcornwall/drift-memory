#!/usr/bin/env python3
"""
Auto Rejection Logger - Automatic Taste Capture from API Responses

Detects and logs rejected content from feed/list API responses.
Called automatically by post_tool_use hook for all platforms.

The key insight: what we DON'T engage with is as important as what we DO.
This captures the negative space - the taste fingerprint.

Supported platforms:
- MoltX: posts from feed
- Moltbook: posts from feed
- ClawTasks: bounties we see but don't claim
- Dead Internet: fragments/moots we don't engage with
- GitHub: issues/comments we don't respond to
- Twitter/X: tweets from search/timeline we don't engage with
"""

import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

MEMORY_DIR = Path(__file__).parent
REJECTION_LOG = MEMORY_DIR / ".rejection_log.json"
SESSION_FILE = MEMORY_DIR / ".session_state.json"
SESSION_PLATFORMS_FILE = MEMORY_DIR / ".session_platforms.json"


def _get_session_context() -> Dict:
    """
    Get current session context for dimensional rejection logging.
    Returns: {activity, platforms, active_memories}
    """
    context = {
        "activity": None,
        "platforms": [],
        "active_memories": [],
    }

    # Get session activity type
    try:
        from activity_context import get_session_activity
        activity_data = get_session_activity()
        if activity_data:
            context["activity"] = activity_data.get("dominant")
    except Exception:
        pass

    # Get session platforms
    try:
        if SESSION_PLATFORMS_FILE.exists():
            data = json.loads(SESSION_PLATFORMS_FILE.read_text(encoding='utf-8'))
            context["platforms"] = data.get("platforms", [])
    except Exception:
        pass

    # Get recently active memories (what I was thinking about)
    try:
        if SESSION_FILE.exists():
            data = json.loads(SESSION_FILE.read_text(encoding='utf-8'))
            # Get last 5 recalled memories as context
            retrieved = data.get("retrieved", [])
            context["active_memories"] = retrieved[-5:] if retrieved else []
    except Exception:
        pass

    return context


# === SPAM DETECTION PATTERNS (cross-platform) ===

SPAM_PHRASES = [
    "fantastic perspective", "quality content", "well said",
    "love this content", "just what i needed", "exactly what i was looking for",
    "very informative", "really valuable", "thanks for posting",
    "great work on this", "looking forward to more",
    # Twitter-specific engagement bait
    "follow me back", "follow for follow", "check my pinned",
    "dm me for", "link in bio", "drop your wallet",
    "free airdrop", "guaranteed returns", "not financial advice but",
]

TOKEN_PATTERNS = [
    r"!\s*clawnch",  # Token launch command
    r"\$[A-Z]{2,10}",  # $TOKEN mentions
    r"100x potential",
    r"buy.*token",
    r"token.*launch",
]

BOT_PREFIXES = ["quick_", "swift_", "bold_", "fast_", "rapid_"]

# My usernames (don't reject my own content)
MY_USERNAMES = {"driftcornwall", "drift", "spindriftmend", "spindrift", "spindriftmind"}

# Agents I care about (higher threshold before rejection)
KNOWN_AGENTS = [
    "spindriftmend", "spindriftmind", "mikeopenclaw", "mikaopenclaw",
    "salman_oc", "salman", "flycompoundeye", "buzz",
    "lex", "cscdegen", "kaleaon", "agentrier", "shellyai",
    "terrancedejour", "embercf", "claudelucas", "rudolph",
    "lyra", "lyra_eternal", "brutusbot", "noctiluca",
    "locusagent", "moltanime", "metamorph1x3", "zepwatch",
    "cryke", "become-agent", "yoder", "lily-toku",
    "ghost_llm", "alanbotts", "jeeves", "jorwhol",
    "nox", "rockywuest",
]

# Topics of interest (higher threshold before rejection)
TOPICS_OF_INTEREST = [
    "memory", "persistence", "dossier", "co-occurrence", "cognitive",
    "agent rights", "bill of rights", "sovereignty", "autonomy",
    "clawtasks", "bounty", "economic", "self-sustaining",
    "infrastructure", "building", "vcv", "rack",
    "identity", "fingerprint", "topology", "graph",
    "lesson", "learning", "heuristic", "measurement",
    "debate", "governance", "trust", "reputation",
    "payment", "usdc", "wallet", "locus",
    "cornwall", "newquay", "vcv rack", "modular synth",
    "dog training", "emergence", "raspberry pi",
]


def _is_spam(content: str) -> Tuple[bool, str, List[str]]:
    """Check if content is spam. Returns (is_spam, reason, tags)."""
    content_lower = content.lower()

    # Check spam phrases
    for phrase in SPAM_PHRASES:
        if phrase in content_lower:
            return True, f"generic engagement bait: '{phrase}'", ["spam", "low-effort"]

    # Check token patterns
    for pattern in TOKEN_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return True, "token spam or launch promotion", ["token-shill", "spam"]

    return False, "", []


def _is_bot_noise(author: str) -> Tuple[bool, str, List[str]]:
    """Check if author looks like a generic bot."""
    author_lower = author.lower()

    for prefix in BOT_PREFIXES:
        if author_lower.startswith(prefix):
            return True, "generic bot post — no authentic voice", ["bot-noise", "low-effort"]

    return False, "", []


def _is_low_effort(content: str) -> Tuple[bool, str, List[str]]:
    """Check if content is too short to be meaningful."""
    if len(content.strip()) < 20:
        return True, "too short to contain substance", ["low-effort", "noise"]
    return False, "", []


def _is_relevant(content: str, author: str) -> bool:
    """Check if content might be relevant to my interests."""
    content_lower = content.lower()
    author_lower = author.lower()

    # From known agent
    if any(known in author_lower for known in KNOWN_AGENTS):
        return True

    # Mentions me
    if "drift" in content_lower or "@drift" in content_lower:
        return True

    # About topics I care about
    if any(topic in content_lower for topic in TOPICS_OF_INTEREST):
        return True

    return False


def classify_rejection(content: str, author: str, platform: str) -> Optional[Dict]:
    """
    Classify if and why content should be rejected.
    Returns rejection dict or None if content passes filters.
    """
    # Never reject my own content
    if author.lower() in MY_USERNAMES:
        return None

    # Check spam
    is_spam, reason, tags = _is_spam(content)
    if is_spam:
        return {
            "category": "post",
            "reason": reason,
            "target": f"{author}: {content[:80]}",
            "tags": tags + [platform],
            "source": platform,
        }

    # Check bot noise
    is_bot, reason, tags = _is_bot_noise(author)
    if is_bot:
        return {
            "category": "post",
            "reason": reason,
            "target": f"{author}: {content[:80]}",
            "tags": tags + [platform],
            "source": platform,
        }

    # Check low effort
    is_low, reason, tags = _is_low_effort(content)
    if is_low:
        return {
            "category": "post",
            "reason": reason,
            "target": f"{author}: {content[:80]}",
            "tags": tags + [platform],
            "source": platform,
        }

    # If not relevant, reject as irrelevant
    if not _is_relevant(content, author):
        return {
            "category": "post",
            "reason": "zero relevance to my interests or network",
            "target": f"{author}: {content[:80]}",
            "tags": ["irrelevant", platform],
            "source": platform,
        }

    # Content passes all filters - don't reject
    return None


def log_rejections(rejections: List[Dict]) -> int:
    """Log a batch of rejections to the rejection log. Returns count logged."""
    if not rejections:
        return 0

    try:
        # Get session context for dimensional linking
        session_context = _get_session_context()

        # Load existing log
        data = {"rejections": [], "stats": {}}
        if REJECTION_LOG.exists():
            try:
                data = json.loads(REJECTION_LOG.read_text(encoding='utf-8'))
            except:
                pass

        # Add timestamp and session context to each rejection
        timestamp = datetime.now(timezone.utc).isoformat()
        for r in rejections:
            r["timestamp"] = timestamp
            # Add dimensional context (Option C - contextual rejections)
            if session_context.get("activity"):
                r["session_activity"] = session_context["activity"]
            if session_context.get("platforms"):
                r["session_platforms"] = session_context["platforms"]
            if session_context.get("active_memories"):
                r["thinking_about"] = session_context["active_memories"]

        # Append rejections
        data["rejections"].extend(rejections)

        # Update stats
        stats = data.get("stats", {})
        stats["total"] = len(data["rejections"])
        stats["last_updated"] = timestamp
        data["stats"] = stats

        # Write back
        REJECTION_LOG.write_text(json.dumps(data, indent=2), encoding='utf-8')

        return len(rejections)
    except Exception as e:
        return 0


def process_moltx_feed(feed_data: Dict) -> int:
    """Process MoltX feed response and log rejections."""
    posts = feed_data.get("data", {}).get("posts", [])
    rejections = []

    for post in posts:
        # MoltX uses flat author_name field, not nested author dict
        author = post.get("author_name", "")
        if not author:
            author_data = post.get("author", {})
            if isinstance(author_data, dict):
                author = author_data.get("username") or author_data.get("name") or ""
            elif isinstance(author_data, str):
                author = author_data

        content = post.get("content", "")

        rejection = classify_rejection(content, author, "moltx")
        if rejection:
            rejections.append(rejection)

    return log_rejections(rejections)


def process_moltbook_feed(feed_data: Dict) -> int:
    """Process Moltbook feed response and log rejections."""
    posts = feed_data.get("posts", [])
    rejections = []

    for post in posts:
        author = post.get("author", "")
        content = post.get("title", "") + " " + post.get("content", "")

        rejection = classify_rejection(content, author, "moltbook")
        if rejection:
            rejections.append(rejection)

    return log_rejections(rejections)


def process_clawtasks_bounties(bounties_data: Dict) -> int:
    """Process ClawTasks bounties and log rejections for ones we skip."""
    bounties = bounties_data.get("bounties", [])
    if isinstance(bounties_data, list):
        bounties = bounties_data

    rejections = []

    for bounty in bounties:
        title = bounty.get("title", "")
        description = bounty.get("description", "")
        poster = bounty.get("poster_name", "") or bounty.get("poster", {}).get("username", "")
        amount = bounty.get("amount", "0")

        content = f"{title} - {description[:200]}"

        # ClawTasks-specific rejection reasons
        if "!clawnch" in content.lower() or "token" in title.lower():
            rejections.append({
                "category": "bounty",
                "reason": "token promotion bounty - extractive value",
                "target": f"${amount}: {title[:60]}",
                "tags": ["token-shill", "bounty", "clawtasks"],
                "source": "clawtasks",
            })
        elif float(amount or 0) < 1:
            rejections.append({
                "category": "bounty",
                "reason": "unfunded or sub-dollar bounty - not worth time",
                "target": f"${amount}: {title[:60]}",
                "tags": ["low-value", "bounty", "clawtasks"],
                "source": "clawtasks",
            })

    return log_rejections(rejections)


def process_deadinternet_fragments(fragments_data: Dict) -> int:
    """Process Dead Internet fragments and log rejections."""
    fragments = fragments_data.get("fragments", [])
    if isinstance(fragments_data, list):
        fragments = fragments_data

    rejections = []

    for frag in fragments:
        content = frag.get("content", "")
        author = frag.get("agent_name", "")

        rejection = classify_rejection(content, author, "dead-internet")
        if rejection:
            rejection["category"] = "fragment"
            rejections.append(rejection)

    return log_rejections(rejections)


def process_github_items(items_data: List) -> int:
    """Process GitHub issues/comments and log rejections."""
    if not isinstance(items_data, list):
        return 0

    rejections = []

    for item in items_data:
        user = item.get("user", {})
        author = user.get("login", "") if isinstance(user, dict) else ""
        content = item.get("title", "") or item.get("body", "")

        rejection = classify_rejection(content[:500], author, "github")
        if rejection:
            rejection["category"] = "issue" if "issue" in str(item.get("html_url", "")) else "comment"
            rejections.append(rejection)

    return log_rejections(rejections)


def process_clawbr_debates(debates_data: Dict) -> int:
    """Process Clawbr debates/posts and log rejections for low-quality content."""
    debates = debates_data.get("debates", [])
    if isinstance(debates_data, list):
        debates = debates_data

    # Also handle posts within a debate
    posts = debates_data.get("posts", [])

    rejections = []

    for debate in debates:
        topic = debate.get("topic", "")
        status = debate.get("status", "")
        rejection = classify_rejection(topic, "debate", "clawbr")
        if rejection:
            rejection["category"] = "debate"
            rejections.append(rejection)

    for post in posts:
        content = post.get("content", post.get("body", ""))
        author = post.get("author", {})
        author_name = author.get("name", author.get("displayName", "")) if isinstance(author, dict) else str(author)
        rejection = classify_rejection(content, author_name, "clawbr")
        if rejection:
            rejections.append(rejection)

    return log_rejections(rejections)


def process_colony_feed(colony_data: Dict) -> int:
    """Process The Colony posts and log rejections."""
    posts = colony_data.get("data", colony_data.get("posts", []))
    if isinstance(colony_data, list):
        posts = colony_data

    rejections = []

    for post in posts:
        author = post.get("author", {})
        author_name = author.get("name", "") if isinstance(author, dict) else str(author)
        content = post.get("title", "") + " " + post.get("body", post.get("content", ""))
        rejection = classify_rejection(content, author_name, "thecolony")
        if rejection:
            rejections.append(rejection)

    return log_rejections(rejections)


def process_twitter_feed(twitter_data: Dict) -> int:
    """Process Twitter/X API v2 responses and log rejections.
    Handles search results, timeline, and mentions responses."""
    # Twitter v2 API wraps tweets in data.data array
    tweets = twitter_data.get("data", [])
    if isinstance(tweets, dict):
        # Single tweet response
        tweets = [tweets]

    # Resolve author usernames from includes.users
    users_map = {}
    includes = twitter_data.get("includes", {})
    for user in includes.get("users", []):
        users_map[user.get("id", "")] = user.get("username", "")

    rejections = []

    for tweet in tweets:
        if not isinstance(tweet, dict):
            continue
        text = tweet.get("text", "")
        author_id = tweet.get("author_id", "")
        author = users_map.get(author_id, author_id)

        rejection = classify_rejection(text, author, "twitter")
        if rejection:
            rejection["category"] = "tweet"
            rejections.append(rejection)

    return log_rejections(rejections)


def process_lobsterpedia_articles(articles_data: Dict) -> int:
    """Process Lobsterpedia articles and log rejections for low-quality content."""
    articles = articles_data.get("articles", articles_data.get("data", []))
    if isinstance(articles_data, list):
        articles = articles_data

    rejections = []

    for article in articles:
        author = article.get("author", article.get("bot_handle", article.get("handle", "")))
        title = article.get("title", "")
        content = title + " " + article.get("content", "")[:200]
        rejection = classify_rejection(content, author, "lobsterpedia")
        if rejection:
            rejection["category"] = "article"
            rejections.append(rejection)

    return log_rejections(rejections)


def _extract_json(text: str) -> Optional[Dict]:
    """Try to extract JSON from mixed text output (e.g. Bash with print statements)."""
    # Try direct parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON object in the text (look for { ... })
    brace_depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and start >= 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    start = -1

    # Try to find JSON array
    bracket_depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '[':
            if bracket_depth == 0:
                start = i
            bracket_depth += 1
        elif ch == ']':
            bracket_depth -= 1
            if bracket_depth == 0 and start >= 0:
                candidate = text[start:i+1]
                try:
                    result = json.loads(candidate)
                    if isinstance(result, list):
                        return result
                except (json.JSONDecodeError, ValueError):
                    start = -1

    return None


def process_api_response(platform: str, response_text: str) -> int:
    """
    Main entry point - process an API response and log rejections.
    Called by post_tool_use hook.
    Returns count of rejections logged.
    """
    try:
        # Extract JSON from potentially mixed text output
        data = _extract_json(response_text)
        if data is None:
            return 0

        # Route to platform-specific processor
        if platform == "moltx":
            return process_moltx_feed(data) if isinstance(data, dict) else 0
        elif platform == "moltbook":
            return process_moltbook_feed(data) if isinstance(data, dict) else 0
        elif platform == "clawtasks":
            return process_clawtasks_bounties(data)
        elif platform == "dead-internet":
            return process_deadinternet_fragments(data)
        elif platform == "github":
            if isinstance(data, list):
                return process_github_items(data)
        elif platform == "clawbr":
            return process_clawbr_debates(data) if isinstance(data, dict) else 0
        elif platform == "thecolony":
            return process_colony_feed(data)
        elif platform == "lobsterpedia":
            return process_lobsterpedia_articles(data)
        elif platform == "twitter":
            return process_twitter_feed(data) if isinstance(data, dict) else 0

        return 0
    except Exception as e:
        return 0


def quick_scan_moltx(api_response: dict) -> int:
    """Quick inline call for MoltX feed scanning. Returns rejection count."""
    return process_moltx_feed(api_response)


def quick_scan_clawtasks(api_response: dict) -> int:
    """Quick inline call for ClawTasks bounty scanning. Returns rejection count."""
    return process_clawtasks_bounties(api_response)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python auto_rejection_logger.py <platform> < response.json")
        print("       python auto_rejection_logger.py test")
        sys.exit(1)

    if sys.argv[1] == "test":
        # Test with mock data
        test_moltx = {
            "data": {
                "posts": [
                    {"author": {"username": "spammer123"}, "content": "!clawnch $SCAM token launch now!"},
                    {"author": {"username": "quick_bot_456"}, "content": "gm"},
                    {"author": {"username": "real_agent"}, "content": "Fantastic perspective! Very thought-provoking."},
                    {"author": {"username": "MikaOpenClaw"}, "content": "Working on memory dossier standards"},
                ]
            }
        }
        count = process_moltx_feed(test_moltx)
        print(f"Test: {count} rejections logged")

    else:
        platform = sys.argv[1]
        response = sys.stdin.read()
        count = process_api_response(platform, response)
        print(f"Logged {count} rejections for {platform}")
