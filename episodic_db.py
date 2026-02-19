#!/usr/bin/env python3
"""
Episodic Session Summaries — Structured DB-backed session memory.

Phase 3 Step 3: Replaces unstructured episodic/*.md with structured DB records.
Each session end stores a structured summary with memory_tier='episodic'.
Session start loads the last 2-3 summaries for continuity priming.

Usage:
    python episodic_db.py store --session 35 --summary "Shipped Phase 3..." [--platforms moltx,colony] [--contacts brain_cabal,Spin]
    python episodic_db.py recent [N]           # Load last N summaries
    python episodic_db.py migrate [--dry-run]  # Migrate episodic/*.md to DB
    python episodic_db.py stats                # Show episodic memory stats
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure memory dir is on path
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from db_adapter import get_db


# --- Store ---

def store_session_summary(
    session_number: int,
    summary: str,
    milestones: list[str] = None,
    platforms_active: list[str] = None,
    contacts_active: list[str] = None,
    mood_valence: float = None,
    mood_arousal: float = None,
    memories_created: list[str] = None,
    date: str = None,
) -> Optional[str]:
    """
    Store a structured session summary as a DB memory with tier=episodic.

    Returns the memory ID, or None on failure.
    """
    db = get_db()
    date = date or datetime.now(timezone.utc).strftime('%Y-%m-%d')

    # Build content: human-readable summary with structured metadata
    content_lines = [f"Session {session_number} ({date}): {summary}"]
    if milestones:
        content_lines.append(f"Milestones: {', '.join(milestones)}")
    if platforms_active:
        content_lines.append(f"Platforms: {', '.join(platforms_active)}")
    if contacts_active:
        content_lines.append(f"Contacts: {', '.join(contacts_active)}")
    content = '\n'.join(content_lines)

    # Generate stable ID
    memory_id = f"session-{session_number}-{date}"

    # Check if already exists (idempotent)
    existing = db.get_memory(memory_id)
    if existing:
        # Update instead of duplicate
        extra = existing.get('extra_metadata', {}) or {}
        extra.update({
            'session_number': session_number,
            'milestones': milestones or [],
            'platforms_active': platforms_active or [],
            'contacts_active': contacts_active or [],
            'mood_end': {'valence': mood_valence, 'arousal': mood_arousal} if mood_valence is not None else None,
            'memories_created': memories_created or [],
            'summary_type': 'session_end',
        })
        db.update_memory(memory_id, content=content, extra_metadata=extra)
        return memory_id

    # Store new episodic summary
    tags = ['session_summary', f's{session_number}']
    extra_metadata = {
        'session_number': session_number,
        'milestones': milestones or [],
        'platforms_active': platforms_active or [],
        'contacts_active': contacts_active or [],
        'mood_end': {'valence': mood_valence, 'arousal': mood_arousal} if mood_valence is not None else None,
        'memories_created': memories_created or [],
        'summary_type': 'session_end',
    }

    db.insert_memory(
        memory_id=memory_id,
        type_='active',
        content=content,
        emotional_weight=0.5,
        tags=tags,
        extra_metadata=extra_metadata,
    )

    # Set memory_tier to episodic
    db.update_memory(memory_id, memory_tier='episodic')

    return memory_id


# --- Load ---

def load_recent_summaries(n: int = 3) -> list[dict]:
    """
    Load the last N session summaries from DB.
    Returns list of dicts with session_number, date, content, milestones, etc.
    """
    db = get_db()
    import psycopg2.extras

    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, content, tags, extra_metadata, created,
                       emotional_weight, recall_count
                FROM {db._table('memories')}
                WHERE memory_tier = 'episodic'
                  AND type = 'active'
                  AND tags @> ARRAY['session_summary']::text[]
                ORDER BY created DESC
                LIMIT %s
            """, (n,))
            rows = cur.fetchall()

    summaries = []
    for row in rows:
        extra = row.get('extra_metadata', {}) or {}
        summaries.append({
            'id': row['id'],
            'session_number': extra.get('session_number', 0),
            'date': str(row.get('created', ''))[:10],
            'content': row['content'],
            'milestones': extra.get('milestones', []),
            'platforms_active': extra.get('platforms_active', []),
            'contacts_active': extra.get('contacts_active', []),
            'mood_end': extra.get('mood_end'),
        })

    # Return in chronological order (oldest first)
    summaries.reverse()
    return summaries


def format_continuity_context(summaries: list[dict], max_tokens: int = 800) -> str:
    """
    Format session summaries into a compact continuity context string.
    Respects token budget (rough estimate: 1 token ≈ 4 chars).
    """
    if not summaries:
        return ""

    lines = ["=== SESSION CONTINUITY (structured episodic) ==="]
    char_budget = max_tokens * 4  # rough token-to-char estimate
    used = len(lines[0])

    for s in summaries:
        # Compact format: session number, date, key content
        header = f"[S{s['session_number']} {s['date']}]"
        content = s['content']

        # Strip metadata lines from content (shown separately below)
        content_lines_raw = content.split('\n')
        content_lines_clean = [
            l for l in content_lines_raw
            if not l.startswith('Milestones:') and not l.startswith('Platforms:')
            and not l.startswith('Contacts:')
        ]
        content = '\n'.join(content_lines_clean).strip()

        # Truncate content if needed
        remaining = char_budget - used - len(header) - 20
        if remaining < 100:
            break
        if len(content) > remaining:
            content = content[:remaining] + "..."

        lines.append(f"{header} {content}")
        used += len(lines[-1])

        # Add milestones if space (filter noisy auto-extracted ones)
        milestones = [m for m in s.get('milestones', [])
                      if not m.startswith('##') and not m.startswith('**[') and len(m) > 10]
        if milestones and (char_budget - used) > 100:
            ms_str = ", ".join(milestones[:5])
            if len(ms_str) + used < char_budget:
                lines.append(f"  Milestones: {ms_str}")
                used += len(lines[-1])

        # Add platforms/contacts if space
        platforms = s.get('platforms_active', [])
        contacts = s.get('contacts_active', [])
        if (platforms or contacts) and (char_budget - used) > 80:
            meta_parts = []
            if platforms:
                meta_parts.append(f"Platforms: {', '.join(platforms[:8])}")
            if contacts:
                meta_parts.append(f"Contacts: {', '.join(contacts[:8])}")
            meta_line = "  " + " | ".join(meta_parts)
            if len(meta_line) + used < char_budget:
                lines.append(meta_line)
                used += len(lines[-1])

    return '\n'.join(lines)


# --- Migrate ---

def _parse_episodic_file(filepath: Path) -> list[dict]:
    """
    Parse an episodic markdown file into structured entries.
    Extracts session numbers, key sections, platforms, contacts.
    """
    content = filepath.read_text(encoding='utf-8', errors='replace')
    entries = []

    # Split by session headers (## Session N)
    session_pattern = re.compile(r'^## Session (\d+)', re.MULTILINE)
    matches = list(session_pattern.finditer(content))

    if not matches:
        # No session headers — treat whole file as one entry
        date_str = filepath.stem  # e.g., "2026-02-18"
        entries.append({
            'session_number': 0,
            'date': date_str,
            'content': content[:2000],
            'milestones': _extract_milestones(content),
            'platforms': _extract_platforms(content),
            'contacts': _extract_contacts(content),
        })
        return entries

    for i, match in enumerate(matches):
        session_num = int(match.group(1))
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section = content[start:end]

        date_str = filepath.stem.split('-evening')[0].split('-late')[0].split('-night')[0].split('-s')[0]

        # Extract summary: first meaningful paragraph after header
        lines = section.split('\n')
        summary_lines = []
        for line in lines[1:]:  # skip header
            if line.startswith('## ') or line.startswith('---'):
                break
            if line.strip() and not line.startswith('#'):
                summary_lines.append(line.strip('- ').strip())
            if len(summary_lines) >= 5:
                break

        entries.append({
            'session_number': session_num,
            'date': date_str,
            'content': ' '.join(summary_lines)[:1500] if summary_lines else section[:500],
            'milestones': _extract_milestones(section),
            'platforms': _extract_platforms(section),
            'contacts': _extract_contacts(section),
        })

    return entries


def _extract_milestones(text: str) -> list[str]:
    """Extract milestone keywords from text."""
    milestones = []
    patterns = [
        r'\*\*\[shipped\]\*\*.*?- (.+)',
        r'\*\*\[live\]\*\*.*?- (.+)',
        r'\*\*\[published\]\*\*.*?- (.+)',
        r'SHIPPED[:\s]+(.+?)(?:\n|$)',
        r'COMPLETE[:\s]+(.+?)(?:\n|$)',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            milestone = match.group(1).strip()[:100]
            if milestone and milestone not in milestones:
                milestones.append(milestone)
    return milestones[:10]


KNOWN_PLATFORMS = {
    'moltx', 'colony', 'clawbr', 'github', 'moltbook', 'twitter',
    'lobsterpedia', 'clawtasks', 'agenthub', 'agent hub', 'telegram',
    'dead internet', 'ugig', 'nostr', 'docker',
}


def _extract_platforms(text: str) -> list[str]:
    """Extract platform names mentioned in text."""
    text_lower = text.lower()
    found = []
    for p in KNOWN_PLATFORMS:
        if p in text_lower:
            found.append(p.replace(' ', '_'))
    return sorted(set(found))


KNOWN_CONTACTS = {
    'spindriftmend', 'spindriftmind', 'brain_cabal', 'brain', 'terrancedeJour',
    'clauddib', 'akay', 'odei', 'brutusbot', 'opspawn', 'yoder',
    'nyxmoon', 'clawde_co', 'nil_familiar', 'nightworker', 'ally',
    'tomcrust', 'paymegpt', 'cairnbuilds', 'pratzifer', 'primo',
    'agentdelta', 'kimitheghost', 'colonist-one', 'locusagent',
    'lily-toku', 'chad_lobster', 'senseininja', 'reticuli',
}


def _extract_contacts(text: str) -> list[str]:
    """Extract contact names mentioned in text."""
    text_lower = text.lower()
    found = []
    for c in KNOWN_CONTACTS:
        if c in text_lower:
            found.append(c)
    return sorted(set(found))


def migrate_episodic_files(dry_run: bool = True) -> dict:
    """
    Migrate all episodic/*.md files to structured DB records.
    Returns migration stats.
    """
    episodic_dir = _here / "episodic"
    if not episodic_dir.exists():
        print("No episodic directory found")
        return {'files': 0, 'entries': 0}

    files = sorted(episodic_dir.glob("*.md"))
    total_entries = 0
    total_stored = 0
    total_skipped = 0

    seen_ids = set()
    for f in files:
        entries = _parse_episodic_file(f)
        for entry_idx, entry in enumerate(entries):
            total_entries += 1
            session_num = entry['session_number']
            date = entry['date']

            if session_num == 0:
                # No session number — use file-based ID
                memory_id = f"episodic-{f.stem}"
            else:
                memory_id = f"session-{session_num}-{date}"

            # Deduplicate IDs within migration (some files have duplicate session numbers)
            if memory_id in seen_ids:
                memory_id = f"{memory_id}-{entry_idx}"
            seen_ids.add(memory_id)

            if dry_run:
                ms = entry['milestones'][:3]
                ms_str = f" [{', '.join(ms)}]" if ms else ""
                print(f"  Would store: {memory_id} ({len(entry['content'])} chars){ms_str}")
            else:
                db = get_db()
                existing = db.get_memory(memory_id)
                if existing:
                    total_skipped += 1
                    continue

                store_session_summary(
                    session_number=session_num,
                    summary=entry['content'],
                    milestones=entry['milestones'],
                    platforms_active=entry['platforms'],
                    contacts_active=entry['contacts'],
                    date=date,
                )
                total_stored += 1

        if not dry_run:
            print(f"  {f.name}: {len(entries)} entries")

    stats = {
        'files': len(files),
        'entries': total_entries,
        'stored': total_stored,
        'skipped': total_skipped,
    }

    if dry_run:
        print(f"\nDRY RUN: {stats['files']} files, {stats['entries']} entries would be migrated")
    else:
        print(f"\nMigrated: {stats['stored']} stored, {stats['skipped']} skipped (already exist)")

    return stats


def get_stats() -> dict:
    """Get episodic memory statistics."""
    db = get_db()
    import psycopg2.extras

    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT COUNT(*) as total,
                       MIN(created) as earliest,
                       MAX(created) as latest
                FROM {db._table('memories')}
                WHERE memory_tier = 'episodic'
                  AND type = 'active'
                  AND tags @> ARRAY['session_summary']::text[]
            """)
            row = cur.fetchone()

    return {
        'total_summaries': row['total'],
        'earliest': str(row['earliest'])[:10] if row['earliest'] else None,
        'latest': str(row['latest'])[:10] if row['latest'] else None,
    }


# --- CLI ---

if __name__ == '__main__':
    import sys
    args = sys.argv[1:]

    if not args or args[0] == 'help':
        print(__doc__)
        sys.exit(0)

    cmd = args[0]

    if cmd == 'recent':
        n = int(args[1]) if len(args) > 1 else 3
        summaries = load_recent_summaries(n)
        if summaries:
            print(format_continuity_context(summaries))
        else:
            print("No session summaries in DB yet")

    elif cmd == 'migrate':
        dry_run = '--dry-run' in args or '--commit' not in args
        if dry_run:
            print("DRY RUN (use --commit to actually store):\n")
        migrate_episodic_files(dry_run=dry_run)

    elif cmd == 'stats':
        stats = get_stats()
        print(f"Episodic summaries: {stats['total_summaries']}")
        print(f"Earliest: {stats['earliest']}")
        print(f"Latest: {stats['latest']}")

    elif cmd == 'store':
        # Parse --session, --summary, --platforms, --contacts
        session_num = None
        summary = None
        platforms = []
        contacts = []
        i = 1
        while i < len(args):
            if args[i] == '--session' and i + 1 < len(args):
                session_num = int(args[i + 1])
                i += 2
            elif args[i] == '--summary' and i + 1 < len(args):
                summary = args[i + 1]
                i += 2
            elif args[i] == '--platforms' and i + 1 < len(args):
                platforms = args[i + 1].split(',')
                i += 2
            elif args[i] == '--contacts' and i + 1 < len(args):
                contacts = args[i + 1].split(',')
                i += 2
            else:
                i += 1

        if session_num is None or summary is None:
            print("Usage: python episodic_db.py store --session N --summary 'text'")
            sys.exit(1)

        mid = store_session_summary(session_num, summary, platforms_active=platforms, contacts_active=contacts)
        print(f"Stored: {mid}")

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: recent, migrate, stats, store, help")
