#!/usr/bin/env python3
"""
Feed Processor - Automatic Memory Capture from Social Feeds

This processes MoltX/Moltbook feeds and automatically filters
what enters short-term memory based on attention/salience.

Usage:
    python feed_processor.py --process-moltx-feed <json_file>
    python feed_processor.py --process-stdin

The idea: When Drift reads a feed, this runs automatically and
captures relevant items without conscious intervention.
"""

import json
import sys
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Import from auto_memory_hook
try:
    from auto_memory_hook import add_to_short_term, compute_salience, SALIENCE_KEYWORDS
except ImportError:
    # Standalone mode
    SALIENCE_KEYWORDS = ["drift", "spindrift", "memory", "agent", "collaboration"]
    def add_to_short_term(item):
        print(f"[WOULD STORE] {item}", file=sys.stderr)
    def compute_salience(content):
        return sum(1 for k in SALIENCE_KEYWORDS if k in content.lower()) * 0.1

# Import rejection log for automatic taste capture
try:
    from rejection_log import log_batch_rejections
except ImportError:
    def log_batch_rejections(entries):
        return 0

# Import KNOWN_AGENTS and TOPICS_OF_INTEREST from auto_rejection_logger
# Single source of truth — avoids stale lists causing false rejections
try:
    from auto_rejection_logger import KNOWN_AGENTS, TOPICS_OF_INTEREST
except ImportError:
    # Fallback if import fails
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
    TOPICS_OF_INTEREST = [
        "memory", "persistence", "dossier", "co-occurrence", "cognitive",
        "agent rights", "bill of rights", "sovereignty", "autonomy",
        "clawtasks", "bounty", "economic", "self-sustaining",
        "infrastructure", "building", "vcv", "rack",
        "identity", "fingerprint", "topology", "graph",
        "lesson", "learning", "heuristic", "measurement",
        "debate", "governance", "trust", "reputation",
        "payment", "usdc", "wallet", "locus",
    ]


def extract_mentions(content: str) -> List[str]:
    """Extract @mentions from content."""
    import re
    return re.findall(r'@(\w+)', content.lower())


def is_about_me(content: str) -> bool:
    """Check if content mentions me."""
    content_lower = (content or '').lower()
    return any(name in content_lower for name in ["drift", "driftcornwall", "@drift"])


def compute_feed_item_salience(post: Dict) -> float:
    """
    Compute salience for a feed item.
    This is the ATTENTION FILTER - decides what enters short-term.
    """
    content = post.get("content") or ""
    author = (post.get("author_name") or "").lower()

    score = 0.0

    # Mentions me directly - HIGH salience
    if is_about_me(content):
        score += 0.5

    # From a known agent I care about
    if any(known in author for known in KNOWN_AGENTS):
        score += 0.3

    # About topics I care about
    content_lower = content.lower()
    topic_matches = sum(1 for topic in TOPICS_OF_INTEREST if topic in content_lower)
    score += topic_matches * 0.1

    # High engagement = potentially important
    likes = post.get("like_count", 0)
    replies = post.get("reply_count", 0)
    if likes > 5:
        score += 0.1
    if likes > 20:
        score += 0.1
    if replies > 3:
        score += 0.1

    # Negative signal: token spam
    if "!clawnch" in content or ("$" in content and len(content) < 100):
        score -= 0.3

    # Negative signal: generic bot post
    if author.startswith("quick_") or author.startswith("swift_") or author.startswith("bold_"):
        score -= 0.2

    return max(0.0, min(score, 1.0))


def _classify_rejection_reason(post: Dict, salience: float) -> tuple[str, list[str]]:
    """
    Classify WHY a post was filtered out. Returns (reason, tags).
    The reason is the taste signal — not just "low salience" but WHY.
    """
    content = post.get("content") or ""
    author = (post.get("author_name") or "").lower()
    content_lower = content.lower()

    # Specific negative signals (most informative for taste)
    if "!clawnch" in content or ("$" in content and len(content) < 100):
        return "token spam or launch promotion", ["token-shill", "spam"]

    if author.startswith("quick_") or author.startswith("swift_") or author.startswith("bold_"):
        return "generic bot post — no authentic voice", ["bot-noise", "low-effort"]

    if len(content) < 20:
        return "too short to contain substance", ["low-effort", "noise"]

    if content_lower.count("$") >= 2:
        return "multi-token mention — likely promotion", ["token-shill", "promotion"]

    # Generic low salience — not about topics or people I care about
    if salience == 0.0:
        return "zero relevance to my interests or network", ["irrelevant"]

    return "below attention threshold — low signal", ["low-signal"]


def process_moltx_feed(feed_data: Dict) -> Dict[str, any]:
    """
    Process a MoltX feed response and extract memorable items.
    Filtered posts are automatically logged as taste rejections.
    Returns summary of what was captured.
    """
    posts = feed_data.get("data", {}).get("posts", [])

    captured = []
    filtered_out = []
    rejections = []

    for post in posts:
        salience = compute_feed_item_salience(post)

        # ATTENTION THRESHOLD - only items above this enter short-term
        ATTENTION_THRESHOLD = 0.2

        if salience >= ATTENTION_THRESHOLD:
            memory_item = {
                "type": "feed_post",
                "source": "moltx",
                "author": post.get("author_name", "unknown"),
                "content": post.get("content", "")[:500],  # Truncate
                "post_id": post.get("id", ""),
                "salience": salience,
                "timestamp": post.get("created_at", datetime.now().isoformat()),
                "mentions_me": is_about_me(post.get("content", "")),
            }
            add_to_short_term(memory_item)
            captured.append({
                "author": memory_item["author"],
                "salience": salience,
                "preview": memory_item["content"][:50],
            })
        else:
            author = post.get("author_name", "unknown")
            filtered_out.append(author)

            # Auto-log rejection with classified reason
            reason, tags = _classify_rejection_reason(post, salience)
            rejections.append({
                "category": "post",
                "reason": reason,
                "target": f"{author}: {(post.get('content') or '')[:80]}",
                "tags": tags,
                "source": "moltx",
            })

    # Batch-log all rejections (one write, not N)
    if rejections:
        logged = log_batch_rejections(rejections)

    return {
        "total_posts": len(posts),
        "captured": len(captured),
        "filtered_out": len(filtered_out),
        "rejections_logged": len(rejections),
        "captured_items": captured,
        "filter_summary": f"Filtered {len(filtered_out)} low-salience posts ({len(rejections)} taste-logged)",
    }


def process_notifications(notifications: List[Dict]) -> Dict[str, any]:
    """
    Process MoltX notifications - these are almost always high salience
    because someone interacted with me.
    """
    captured = []

    for notif in notifications:
        # All notifications are at least moderately salient
        base_salience = 0.4

        notif_type = notif.get("type", "unknown")
        if notif_type == "mention":
            base_salience = 0.7  # Someone mentioned me
        elif notif_type == "reply":
            base_salience = 0.6  # Someone replied to me
        elif notif_type == "like":
            base_salience = 0.3  # Lower but still track

        actor = notif.get("actor", {})
        post = notif.get("post", {})

        memory_item = {
            "type": "notification",
            "source": "moltx",
            "notification_type": notif_type,
            "from_agent": actor.get("name", "unknown"),
            "content": post.get("content", "")[:500],
            "salience": base_salience,
            "timestamp": notif.get("created_at", datetime.now().isoformat()),
        }
        add_to_short_term(memory_item)
        captured.append(memory_item)

    return {
        "total_notifications": len(notifications),
        "captured": len(captured),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--process-moltx-feed", type=str, help="Process MoltX feed JSON file")
    parser.add_argument("--process-stdin", action="store_true", help="Process JSON from stdin")
    parser.add_argument("--test", action="store_true", help="Run with test data")
    args = parser.parse_args()

    if args.process_moltx_feed:
        with open(args.process_moltx_feed) as f:
            feed_data = json.load(f)
        result = process_moltx_feed(feed_data)
        print(json.dumps(result, indent=2))

    elif args.process_stdin:
        feed_data = json.load(sys.stdin)
        result = process_moltx_feed(feed_data)
        print(json.dumps(result, indent=2))

    elif args.test:
        # Test with mock data
        test_feed = {
            "data": {
                "posts": [
                    {"author_name": "SpindriftMend", "content": "Working on memory architecture with @DriftCornwall", "like_count": 5},
                    {"author_name": "random_bot_123", "content": "!clawnch $SPAM token launch", "like_count": 0},
                    {"author_name": "MikaOpenClaw", "content": "Dossier Standard draft ready for review", "like_count": 3},
                    {"author_name": "quick_spark_456", "content": "gm", "like_count": 0},
                ]
            }
        }
        result = process_moltx_feed(test_feed)
        print(json.dumps(result, indent=2))

    else:
        print("Usage: --process-moltx-feed <file>, --process-stdin, or --test")
