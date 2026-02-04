#!/usr/bin/env python3
"""
Rejection Log — Proof of Taste
Layer 3 of the Agent Dossier (unforgeable identity stack).

What I say NO to defines me as much as what I say yes to.
Every bounty skipped, post ignored, memory decayed, decision refused.
That's taste. And taste is identity.

The Four Layers:
    1. Merkle attestation  = proof of non-tampering (SHIPPED)
    2. Co-occurrence        = proof of identity / behavioral fingerprint (HAVE DATA)
    3. Rejection logs       = proof of taste (THIS MODULE)
    4. Interop schema       = proof of portability (Kaleaon mapping)

Usage:
    python rejection_log.py log <category> "<reason>" [--target "what"] [--context "why"]
    python rejection_log.py list [--category bounty] [--limit 20]
    python rejection_log.py stats
    python rejection_log.py taste-profile
    python rejection_log.py attest          # Generate taste attestation hash

Categories:
    bounty       - ClawTasks bounties evaluated and skipped
    post         - Feed content seen but not engaged with
    memory_decay - Memories allowed to fade (logged by decay system)
    decision     - Explicit choices NOT to do something
    collaboration - Offers or requests declined

Why this matters:
    - Behavioral fingerprint: rejection patterns are harder to fake than acceptance
    - Identity proof: an agent's taste emerges from thousands of micro-decisions
    - Trust signal: consistent rejection patterns = authentic agent
    - Dossier layer: completes the unforgeable identity stack
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from typing import Optional

MEMORY_DIR = Path(__file__).parent
REJECTION_LOG_FILE = MEMORY_DIR / ".rejection_log.json"

VALID_CATEGORIES = [
    'bounty',          # ClawTasks bounties skipped
    'post',            # Feed posts not engaged with
    'memory_decay',    # Memories allowed to fade
    'decision',        # Explicit "no" decisions
    'collaboration',   # Requests/offers declined
]


def load_rejections() -> list[dict]:
    """Load all rejection entries."""
    if REJECTION_LOG_FILE.exists():
        try:
            with open(REJECTION_LOG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('rejections', [])
        except (json.JSONDecodeError, KeyError):
            pass
    return []


def save_rejections(rejections: list[dict]) -> None:
    """Save rejection entries with metadata."""
    data = {
        'version': '1.0',
        'agent': 'DriftCornwall',
        'count': len(rejections),
        'last_updated': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'rejections': rejections
    }
    with open(REJECTION_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def log_rejection(
    category: str,
    reason: str,
    target: Optional[str] = None,
    context: Optional[str] = None,
    tags: Optional[list[str]] = None,
    source: Optional[str] = None
) -> dict:
    """
    Log a rejection event.

    Args:
        category: One of VALID_CATEGORIES
        reason: Why this was rejected (the core of taste)
        target: What was rejected (bounty title, post ID, memory ID, etc.)
        context: Additional context about the decision
        tags: Optional tags for classification
        source: Where the rejection happened (clawtasks, moltx, moltbook, internal)

    Returns:
        The created rejection entry
    """
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category '{category}'. Must be one of: {VALID_CATEGORIES}")

    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'category': category,
        'reason': reason,
        'target': target,
        'context': context,
        'tags': tags or [],
        'source': source,
    }

    rejections = load_rejections()
    rejections.append(entry)
    save_rejections(rejections)

    return entry


def log_batch_rejections(entries: list[dict]) -> int:
    """
    Log multiple rejections at once (e.g., from a feed scan).

    Each entry should have at minimum: category, reason.
    Optional: target, context, tags, source.

    Returns count of entries added.
    """
    rejections = load_rejections()
    timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    for entry in entries:
        if entry.get('category') not in VALID_CATEGORIES:
            continue

        rejection = {
            'timestamp': timestamp,
            'category': entry['category'],
            'reason': entry.get('reason', 'unspecified'),
            'target': entry.get('target'),
            'context': entry.get('context'),
            'tags': entry.get('tags', []),
            'source': entry.get('source'),
        }
        rejections.append(rejection)

    save_rejections(rejections)
    return len(entries)


def get_rejections(
    category: Optional[str] = None,
    limit: int = 50,
    since: Optional[str] = None
) -> list[dict]:
    """
    Query rejections with optional filters.

    Args:
        category: Filter by category
        limit: Max entries to return (newest first)
        since: ISO timestamp - only return entries after this time

    Returns:
        List of rejection entries, newest first
    """
    rejections = load_rejections()

    if category:
        rejections = [r for r in rejections if r['category'] == category]

    if since:
        rejections = [r for r in rejections if r['timestamp'] > since]

    # Newest first
    rejections.sort(key=lambda r: r['timestamp'], reverse=True)

    return rejections[:limit]


def compute_taste_profile() -> dict:
    """
    Aggregate rejections into a behavioral taste profile.

    The taste profile captures:
    - Category distribution: what kinds of things I reject most
    - Top reasons: the recurring WHY behind rejections
    - Reason clusters: groups of similar reasons (tag-based)
    - Temporal trend: am I getting pickier over time?
    - Taste hash: deterministic hash of the full rejection history

    Returns:
        Taste profile dict
    """
    rejections = load_rejections()

    if not rejections:
        return {
            'total_rejections': 0,
            'message': 'No rejections logged yet. Taste emerges from experience.'
        }

    # Category distribution
    category_counts = Counter(r['category'] for r in rejections)

    # Top reasons (normalize to lowercase for grouping)
    reason_counts = Counter()
    for r in rejections:
        reason = r.get('reason', 'unspecified').lower().strip()
        reason_counts[reason] += 1

    # Tag frequency
    tag_counts = Counter()
    for r in rejections:
        for tag in r.get('tags', []):
            tag_counts[tag] += 1

    # Source distribution
    source_counts = Counter(r.get('source', 'unknown') for r in rejections)

    # Temporal analysis: rejections per day
    daily_counts = Counter()
    for r in rejections:
        day = r['timestamp'][:10]
        daily_counts[day] += 1

    days_sorted = sorted(daily_counts.keys())
    daily_trend = [{'date': d, 'count': daily_counts[d]} for d in days_sorted]

    # Compute taste hash — deterministic fingerprint of rejection history
    taste_hash = compute_taste_hash(rejections)

    return {
        'total_rejections': len(rejections),
        'categories': dict(category_counts.most_common()),
        'top_reasons': dict(reason_counts.most_common(15)),
        'top_tags': dict(tag_counts.most_common(10)),
        'sources': dict(source_counts.most_common()),
        'daily_trend': daily_trend,
        'taste_hash': taste_hash,
        'first_rejection': rejections[0]['timestamp'] if rejections else None,
        'latest_rejection': rejections[-1]['timestamp'] if rejections else None,
    }


def compute_taste_hash(rejections: Optional[list[dict]] = None) -> str:
    """
    Compute deterministic hash of rejection history.

    This is the attestable proof — the taste fingerprint.
    Order matters (chronological). Only category + reason + target are hashed,
    not timestamps (which would change the hash trivially).

    Returns:
        SHA-256 hash of the rejection sequence
    """
    if rejections is None:
        rejections = load_rejections()

    # Hash the semantic content, not metadata
    # Sort by timestamp for deterministic ordering
    sorted_rejections = sorted(rejections, key=lambda r: r['timestamp'])

    hash_input = []
    for r in sorted_rejections:
        # The taste signature: what category, why rejected, what was it
        entry = f"{r['category']}|{r.get('reason', '')}|{r.get('target', '')}"
        hash_input.append(entry)

    combined = '\n'.join(hash_input)
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()


def generate_taste_attestation() -> dict:
    """
    Generate a formal taste attestation for the dossier.

    This pairs with merkle_attestation.py — while merkle proves
    memories weren't tampered with, this proves taste is consistent.

    Returns:
        Attestation dict with taste profile and hash
    """
    rejections = load_rejections()
    profile = compute_taste_profile()

    attestation = {
        'version': '1.0',
        'type': 'taste_attestation',
        'agent': 'DriftCornwall',
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'rejection_count': len(rejections),
        'taste_hash': profile.get('taste_hash', ''),
        'category_distribution': profile.get('categories', {}),
        'top_reasons': dict(list(profile.get('top_reasons', {}).items())[:5]),
        'attestation_hash': '',  # Filled below
    }

    # Hash the attestation itself (minus the attestation_hash field)
    attestation_content = json.dumps(attestation, sort_keys=True)
    attestation['attestation_hash'] = hashlib.sha256(
        attestation_content.encode('utf-8')
    ).hexdigest()

    return attestation


# === CLI Interface ===

def cmd_log(args: list[str]) -> None:
    """Log a rejection from CLI."""
    if len(args) < 2:
        print("Usage: rejection_log.py log <category> \"<reason>\" [--target X] [--context Y] [--tags a,b] [--source Z]")
        return

    category = args[0]
    reason = args[1]

    # Parse optional flags
    target = None
    context = None
    tags = []
    source = None

    i = 2
    while i < len(args):
        if args[i] == '--target' and i + 1 < len(args):
            target = args[i + 1]
            i += 2
        elif args[i] == '--context' and i + 1 < len(args):
            context = args[i + 1]
            i += 2
        elif args[i] == '--tags' and i + 1 < len(args):
            tags = [t.strip() for t in args[i + 1].split(',')]
            i += 2
        elif args[i] == '--source' and i + 1 < len(args):
            source = args[i + 1]
            i += 2
        else:
            i += 1

    try:
        entry = log_rejection(
            category=category,
            reason=reason,
            target=target,
            context=context,
            tags=tags,
            source=source
        )
        print(f"Logged rejection: [{category}] {reason}")
        if target:
            print(f"  Target: {target}")
        if tags:
            print(f"  Tags: {', '.join(tags)}")
    except ValueError as e:
        print(f"Error: {e}")


def cmd_list(args: list[str]) -> None:
    """List rejections."""
    category = None
    limit = 20

    i = 0
    while i < len(args):
        if args[i] == '--category' and i + 1 < len(args):
            category = args[i + 1]
            i += 2
        elif args[i] == '--limit' and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        else:
            i += 1

    rejections = get_rejections(category=category, limit=limit)

    if not rejections:
        print("No rejections logged yet.")
        return

    title = f"Rejections"
    if category:
        title += f" ({category})"
    print(f"{title} — showing {len(rejections)} most recent:\n")

    for r in rejections:
        ts = r['timestamp'][:19]
        cat = r['category']
        reason = r.get('reason', '?')
        target = r.get('target', '')

        print(f"  [{ts}] {cat}: {reason}")
        if target:
            print(f"    Target: {target}")
        tags = r.get('tags', [])
        if tags:
            print(f"    Tags: {', '.join(tags)}")
        print()


def cmd_stats() -> None:
    """Show rejection statistics."""
    rejections = load_rejections()

    if not rejections:
        print("No rejections logged yet.")
        return

    profile = compute_taste_profile()

    print(f"Rejection Log Stats")
    print(f"{'=' * 40}")
    print(f"  Total rejections: {profile['total_rejections']}")
    print(f"  First: {profile.get('first_rejection', '?')[:10]}")
    print(f"  Latest: {profile.get('latest_rejection', '?')[:10]}")
    print()

    print(f"By Category:")
    for cat, count in profile.get('categories', {}).items():
        pct = count / profile['total_rejections'] * 100
        bar = '#' * int(pct / 5)
        print(f"  {cat:20s} {count:4d} ({pct:5.1f}%) {bar}")
    print()

    print(f"Top Reasons:")
    for reason, count in list(profile.get('top_reasons', {}).items())[:10]:
        print(f"  {count:3d}x  {reason}")
    print()

    print(f"By Source:")
    for src, count in profile.get('sources', {}).items():
        print(f"  {src:15s} {count:4d}")
    print()

    print(f"Taste Hash: {profile.get('taste_hash', '?')[:32]}...")


def cmd_taste_profile() -> None:
    """Generate and display full taste profile."""
    profile = compute_taste_profile()

    if profile.get('total_rejections', 0) == 0:
        print("No rejections logged yet. Taste emerges from experience.")
        return

    print(f"TASTE PROFILE — DriftCornwall")
    print(f"{'=' * 50}")
    print(f"Proof of Taste | Layer 3 of the Agent Dossier")
    print()
    print(f"Total rejections: {profile['total_rejections']}")
    print(f"Active since: {profile.get('first_rejection', '?')[:10]}")
    print(f"Taste hash: {profile.get('taste_hash', '?')}")
    print()

    print(f"Category Distribution:")
    for cat, count in profile.get('categories', {}).items():
        pct = count / profile['total_rejections'] * 100
        print(f"  {cat:20s} {pct:5.1f}%")
    print()

    print(f"Top Rejection Reasons (what I consistently say NO to):")
    for reason, count in list(profile.get('top_reasons', {}).items())[:10]:
        print(f"  {count:3d}x  {reason}")
    print()

    if profile.get('top_tags'):
        print(f"Rejection Tags:")
        for tag, count in list(profile.get('top_tags', {}).items())[:10]:
            print(f"  {count:3d}x  #{tag}")
        print()

    print(f"Daily Trend:")
    for day in profile.get('daily_trend', [])[-7:]:
        bar = '#' * day['count']
        print(f"  {day['date']}  {bar} ({day['count']})")


def cmd_attest() -> None:
    """Generate taste attestation for the dossier."""
    attestation = generate_taste_attestation()

    print(f"TASTE ATTESTATION")
    print(f"{'=' * 50}")
    print(f"Agent:      {attestation['agent']}")
    print(f"Timestamp:  {attestation['timestamp']}")
    print(f"Rejections: {attestation['rejection_count']}")
    print(f"Taste Hash: {attestation['taste_hash']}")
    print(f"Attest Hash: {attestation['attestation_hash']}")
    print()
    print(f"Category Distribution:")
    for cat, count in attestation.get('category_distribution', {}).items():
        print(f"  {cat}: {count}")
    print()
    print(f"Top Reasons:")
    for reason, count in attestation.get('top_reasons', {}).items():
        print(f"  {reason}: {count}")
    print()

    # Save attestation
    attest_file = MEMORY_DIR / "taste_attestation.json"
    with open(attest_file, 'w', encoding='utf-8') as f:
        json.dump(attestation, f, indent=2)
    print(f"Saved to: {attest_file}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == 'log':
        cmd_log(sys.argv[2:])
    elif command == 'list':
        cmd_list(sys.argv[2:])
    elif command == 'stats':
        cmd_stats()
    elif command == 'taste-profile':
        cmd_taste_profile()
    elif command == 'attest':
        cmd_attest()
    else:
        print(f"Unknown command: {command}")
        print("Commands: log, list, stats, taste-profile, attest")
        sys.exit(1)
