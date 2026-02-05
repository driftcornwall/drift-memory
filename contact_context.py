#!/usr/bin/env python3
"""
Contact Context - Layer 2.2: WHO dimension for cognitive fingerprint.

Tracks which contacts/agents are associated with memories and edges.
Enables filtered fingerprints like "my cognitive patterns when interacting with Spin".

Part of the 5W Identity Framework:
- WHO: contacts (this file)
- WHAT: topics (topic_context.py)
- WHY: activity (activity_context.py)
- WHERE: platform (platform_context.py)
- WHEN: temporal (observation timestamps)

Usage:
    python contact_context.py session          # Show this session's contacts
    python contact_context.py backfill         # Tag memories with contact mentions
    python contact_context.py backfill-edges   # Add contact_context to edges
    python contact_context.py stats            # Show contact distribution

Credit: 5W framework discussion with Lex (2026-02-05)
"""

import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict

MEMORY_ROOT = Path(__file__).parent
CORE_DIR = MEMORY_ROOT / "core"
ACTIVE_DIR = MEMORY_ROOT / "active"
ARCHIVE_DIR = MEMORY_ROOT / "archive"

# My known usernames (to exclude from contact extraction)
MY_NAMES = {'driftcornwall', 'drift', 'spindriftmend', 'spindrift'}

# False positives to filter out
BLOCKLIST = {
    'tags', 'tag', 'gmail', 'com', 'json', 'yaml', 'file', 'path', 'http',
    'https', 'api', 'url', 'get', 'post', 'put', 'delete', 'the', 'and',
    'for', 'you', 'type', 'data', 'content', 'status', 'error', 'success'
}

# Known contacts and their aliases
KNOWN_CONTACTS = {
    'spindriftmind': ['spindriftmind', 'spindrift', 'spin'],
    'kaleaon': ['kaleaon'],
    'flycompoundeye': ['flycompoundeye', 'buzz'],
    'mikaopenclaw': ['mikaopenclaw', 'mika'],
    'mogra': ['mogra', 'mogradev', 'mograflower'],
}


def parse_memory_file(filepath: Path) -> tuple[dict, str]:
    """Parse a memory file into metadata and content."""
    try:
        text = filepath.read_text(encoding='utf-8')
    except Exception:
        return {}, ""

    if not text.startswith('---'):
        return {}, text

    parts = text.split('---', 2)
    if len(parts) < 3:
        return {}, text

    import yaml
    try:
        metadata = yaml.safe_load(parts[1]) or {}
    except Exception:
        metadata = {}

    content = parts[2].strip()
    return metadata, content


def write_memory_file(filepath: Path, metadata: dict, content: str):
    """Write metadata and content back to a memory file."""
    import yaml

    yaml_str = yaml.dump(metadata, default_flow_style=False, allow_unicode=True, sort_keys=False)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('---\n')
        f.write(yaml_str)
        f.write('---\n\n')
        f.write(content)


def extract_contacts(content: str) -> list[str]:
    """
    Extract contact mentions from content.
    Returns normalized contact names.
    """
    content_lower = content.lower()
    contacts = set()

    # Find @mentions
    mentions = re.findall(r'@([a-zA-Z0-9_]+)', content_lower)
    for mention in mentions:
        if mention not in MY_NAMES and mention not in BLOCKLIST and len(mention) > 2:
            # Normalize to known contact if possible
            for known, aliases in KNOWN_CONTACTS.items():
                if mention in aliases:
                    contacts.add(known)
                    break
            else:
                contacts.add(mention)

    # Also check for known contact names without @
    for known, aliases in KNOWN_CONTACTS.items():
        for alias in aliases:
            if alias in content_lower:
                contacts.add(known)
                break

    return list(contacts)


def get_session_contacts() -> list[str]:
    """Get contacts from current session."""
    session_file = MEMORY_ROOT / ".session_contacts.json"
    if not session_file.exists():
        return []
    try:
        data = json.loads(session_file.read_text(encoding='utf-8'))
        return data.get('contacts', [])
    except Exception:
        return []


def clear_session_contacts():
    """Clear session contacts (called at session end)."""
    session_file = MEMORY_ROOT / ".session_contacts.json"
    if session_file.exists():
        session_file.unlink()


def backfill_contact_context():
    """
    Backfill contact_context into all existing memories.
    """
    updated = 0
    skipped = 0

    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue

        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)

            if not metadata.get('id'):
                continue

            # Skip if already has contact_context
            if 'contact_context' in metadata:
                skipped += 1
                continue

            # Extract contacts
            tags = metadata.get('tags', [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(',')]
            full_content = content + ' ' + ' '.join(tags)
            contacts = extract_contacts(full_content)

            if contacts:
                metadata['contact_context'] = contacts
                write_memory_file(filepath, metadata, content)
                updated += 1

    print(f"Backfill complete: {updated} memories updated, {skipped} already tagged")
    return updated


def backfill_edges():
    """
    Backfill contact_context into existing .edges_v3.json edges.
    """
    edges_file = MEMORY_ROOT / ".edges_v3.json"
    if not edges_file.exists():
        print("No .edges_v3.json found.")
        return 0

    # Load edges
    with open(edges_file, 'r', encoding='utf-8') as f:
        edges_raw = json.load(f)

    # Build contact cache for all memories
    contact_cache = {}
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            memory_id = metadata.get('id')
            if memory_id:
                if 'contact_context' in metadata:
                    contact_cache[memory_id] = metadata['contact_context']
                else:
                    tags = metadata.get('tags', [])
                    if isinstance(tags, str):
                        tags = [t.strip() for t in tags.split(',')]
                    full_content = content + ' ' + ' '.join(tags)
                    contact_cache[memory_id] = extract_contacts(full_content)

    updated = 0
    for pair_key, edge_data in edges_raw.items():
        ids = pair_key.split('|')
        if len(ids) != 2:
            continue

        id1, id2 = ids
        contacts1 = set(contact_cache.get(id1, []))
        contacts2 = set(contact_cache.get(id2, []))

        # Edge gets contacts from either memory
        all_contacts = list(contacts1 | contacts2)

        if all_contacts:
            if 'contact_context' not in edge_data:
                edge_data['contact_context'] = all_contacts
                updated += 1

    # Save
    with open(edges_file, 'w', encoding='utf-8') as f:
        json.dump(edges_raw, f, indent=2)

    print(f"Edge backfill complete: {updated} edges updated")
    return updated


def get_contact_stats() -> dict:
    """Get statistics on contact distribution."""
    stats = defaultdict(lambda: {'count': 0, 'memories': []})
    total = 0

    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue

        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            memory_id = metadata.get('id')
            if not memory_id:
                continue

            total += 1
            contacts = metadata.get('contact_context', [])
            if not contacts:
                tags = metadata.get('tags', [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(',')]
                full_content = content + ' ' + ' '.join(tags)
                contacts = extract_contacts(full_content)

            for contact in contacts:
                stats[contact]['count'] += 1
                if len(stats[contact]['memories']) < 3:
                    stats[contact]['memories'].append(memory_id)

    return {'total': total, 'by_contact': dict(stats)}


def print_stats():
    """Print contact distribution stats."""
    stats = get_contact_stats()
    total = stats['total']

    print(f"\nCONTACT DISTRIBUTION — WHO dimension")
    print("=" * 50)
    print(f"Total memories: {total}\n")

    # Sort by count
    by_contact = sorted(stats['by_contact'].items(), key=lambda x: -x[1]['count'])

    if not by_contact:
        print("  No contacts found in memories.")
        return

    max_count = max(c[1]['count'] for c in by_contact) if by_contact else 1

    for contact, data in by_contact[:15]:
        count = data['count']
        pct = (count / total * 100) if total > 0 else 0
        bar_len = int(count / max_count * 30)
        bar = '#' * bar_len
        print(f"  {contact:20} {count:4} ({pct:5.1f}%)  {bar}")


def print_session():
    """Print current session contacts."""
    contacts = get_session_contacts()
    print(f"\nSESSION CONTACTS — WHO this session")
    print("=" * 50)
    if contacts:
        for c in contacts:
            print(f"  - {c}")
    else:
        print("  No contacts tracked this session yet.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python contact_context.py session")
        print("  python contact_context.py backfill")
        print("  python contact_context.py backfill-edges")
        print("  python contact_context.py stats")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "session":
        print_session()

    elif cmd == "backfill":
        backfill_contact_context()

    elif cmd == "backfill-edges":
        backfill_edges()

    elif cmd == "stats":
        print_stats()

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
