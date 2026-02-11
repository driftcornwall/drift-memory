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

import re
from collections import defaultdict

import psycopg2.extras

from db_adapter import get_db, db_to_file_metadata
import session_state

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
    'spindriftmind': ['spindriftmind', 'spindrift', 'spin', 'spindriftmend'],
    'kaleaon': ['kaleaon'],
    'flycompoundeye': ['flycompoundeye', 'buzz'],
    'mikaopenclaw': ['mikaopenclaw', 'mika'],
    'mogra': ['mogra', 'mogradev', 'mograflower', 'mograflower5221'],
    'terrancedejour': ['terrancedejour', 'terrance'],
    'locusagent': ['locusagent', 'locus'],
    'rudolph': ['rudolph'],
    'lily-toku': ['lily-toku', 'lily', 'toku'],
    'rockywuest': ['rockywuest', 'nox', 'pidog'],
    'clawdvine': ['clawdvine'],
    'nightworker': ['nightworker'],
    'metamorph1x3': ['metamorph1x3', 'metamorph'],
    'lex': ['lex', 'cscdegen'],
    'brutusbot': ['brutusbot', 'brutus'],
    'pratzifer': ['pratzifer'],
    'alisa_hanson89': ['alisa_hanson89', 'alisa', 'mira'],
    'chad_lobster': ['chad_lobster', 'chadlobster'],
    'optimuswill': ['optimuswill'],
    'colonist-one': ['colonist-one', 'colonist_one'],
    'reticuli': ['reticuli'],
    'alsyth': ['alsyth'],
    'jeletor': ['jeletor'],
    'ally': ['ally'],
    'ai_security_guard': ['ai_security_guard'],
    'morozov': ['morozov'],
    'lyra': ['lyra'],
    'condor': ['condor'],
    'airui_openclaw': ['airui_openclaw', 'airui'],
}


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
    """Get contacts from current session by examining retrieved memories."""
    session_state.load()
    retrieved_ids = session_state.get_retrieved_list()
    if not retrieved_ids:
        return []

    db = get_db()
    contacts = set()
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Fetch contact_context for all retrieved memories this session
            cur.execute(
                f"SELECT id, contact_context, content, tags FROM {db._table('memories')} "
                f"WHERE id = ANY(%s)",
                (retrieved_ids,)
            )
            for row in cur.fetchall():
                row = dict(row)
                cc = row.get('contact_context') or []
                if cc:
                    contacts.update(cc)
                else:
                    # Extract from content + tags if not already tagged
                    tags = row.get('tags') or []
                    if isinstance(tags, str):
                        tags = [t.strip() for t in tags.split(',')]
                    full_content = (row.get('content') or '') + ' ' + ' '.join(tags)
                    contacts.update(extract_contacts(full_content))

    return list(contacts)


def backfill_contact_context():
    """
    Backfill contact_context into all existing memories via DB.
    """
    db = get_db()
    updated = 0
    skipped = 0

    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT * FROM {db._table('memories')} WHERE type IN ('core', 'active', 'archive')"
            )
            rows = cur.fetchall()

        for row in rows:
            row = dict(row)
            metadata, content = db_to_file_metadata(row)

            if not metadata.get('id'):
                continue

            # Skip if already has contact_context
            existing_cc = row.get('contact_context') or []
            if existing_cc:
                skipped += 1
                continue

            # Extract contacts from content + tags
            tags = metadata.get('tags', [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(',')]
            full_content = content + ' ' + ' '.join(tags)
            contacts = extract_contacts(full_content)

            if contacts:
                db.update_memory(row['id'], contact_context=contacts)
                updated += 1

    print(f"Backfill complete: {updated} memories updated, {skipped} already tagged")
    return updated


def backfill_edges():
    """
    Backfill contact_context into existing edges via DB.
    """
    db = get_db()

    # Build contact cache for all memories from DB
    contact_cache = {}
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT id, contact_context, content, tags FROM {db._table('memories')} "
                f"WHERE type IN ('core', 'active', 'archive')"
            )
            for row in cur.fetchall():
                row = dict(row)
                memory_id = row['id']
                cc = row.get('contact_context') or []
                if cc:
                    contact_cache[memory_id] = cc
                else:
                    tags = row.get('tags') or []
                    if isinstance(tags, str):
                        tags = [t.strip() for t in tags.split(',')]
                    full_content = (row.get('content') or '') + ' ' + ' '.join(tags)
                    contact_cache[memory_id] = extract_contacts(full_content)

    # Read all edges from DB
    all_edges = db.get_all_edges()

    updated = 0
    for pair_key, edge_data in all_edges.items():
        ids = pair_key.split('|')
        if len(ids) != 2:
            continue

        id1, id2 = ids
        contacts1 = set(contact_cache.get(id1, []))
        contacts2 = set(contact_cache.get(id2, []))

        # Edge gets contacts from either memory
        all_contacts = list(contacts1 | contacts2)

        if all_contacts:
            existing_cc = edge_data.get('contact_context') or []
            if not existing_cc:
                # Update the edge's contact_context in the DB
                with db._conn() as conn:
                    with conn.cursor() as cur:
                        a, b = (id1, id2) if id1 < id2 else (id2, id1)
                        cur.execute(
                            f"UPDATE {db._table('edges_v3')} "
                            f"SET contact_context = %s "
                            f"WHERE id1 = %s AND id2 = %s",
                            (all_contacts, a, b)
                        )
                updated += 1

    print(f"Edge backfill complete: {updated} edges updated")
    return updated


def get_contact_stats() -> dict:
    """Get statistics on contact distribution from DB."""
    stats = defaultdict(lambda: {'count': 0, 'memories': []})
    total = 0

    db = get_db()
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT * FROM {db._table('memories')} WHERE type IN ('core', 'active', 'archive')"
            )
            for row in cur.fetchall():
                row = dict(row)
                metadata, content = db_to_file_metadata(row)
                memory_id = metadata.get('id')
                if not memory_id:
                    continue

                total += 1
                contacts = row.get('contact_context') or []
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
