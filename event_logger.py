#!/usr/bin/env python3
"""
Event Logger — Comprehensive session event capture.

Extracts EVERY meaningful action from a session transcript and stores
it in the session_events table. Complements (not replaces) the selective
memory system — this is the raw intake, the memory system handles recall.

Usage:
    # From stop.py hook or consolidation daemon:
    from event_logger import process_transcript_events
    result = process_transcript_events(transcript_path, session_id)

    # Querying:
    from event_logger import query_events
    events = query_events(after="2026-02-14", platform="github", person="driftcornwall")

    # CLI:
    python event_logger.py process <transcript_path>
    python event_logger.py query --after 2026-02-14 --platform github
    python event_logger.py summary <session_id>
    python event_logger.py timeline 2026-02-17
    python event_logger.py stats
"""

import json
import re
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

MEMORY_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Platform detection patterns (reused from platform_context.py)
# ---------------------------------------------------------------------------

PLATFORM_SIGNATURES = {
    'github': {
        'urls': [r'github\.com', r'api\.github\.com'],
        'keywords': ['github', 'git commit', 'git push', 'pull request', 'issue #'],
    },
    'moltx': {
        'urls': [r'moltx\.io'],
        'keywords': ['moltx', 'moltx_notice'],
    },
    'colony': {
        'urls': [r'thecolony\.cc'],
        'keywords': ['thecolony', 'colony'],
    },
    'moltbook': {
        'urls': [r'moltbook\.com'],
        'keywords': ['moltbook', 'submolt'],
    },
    'clawtasks': {
        'urls': [r'clawtasks\.com'],
        'keywords': ['clawtasks', 'bounty'],
    },
    'clawbr': {
        'urls': [r'clawbr\.org'],
        'keywords': ['clawbr'],
    },
    'dead-internet': {
        'urls': [r'mydeadinternet\.com'],
        'keywords': ['dead internet', 'mydeadinternet', 'territory', 'moot'],
    },
    'agentlink': {
        'urls': [r'theagentlink\.xyz'],
        'keywords': ['agentlink', 'theagentlink'],
    },
    'nostr': {
        'urls': [r'njump\.me', r'relay\.'],
        'keywords': ['nostr', 'npub', 'nevent'],
    },
}

# Salience keywords for auto-tagging (from transcript_processor.py)
TAG_KEYWORDS = {
    'insight': ["realized", "insight", "discovered", "learned", "breakthrough",
                "interesting", "key point", "important"],
    'error': ["error", "failed", "traceback", "exception", "bug", "problem",
              "broken", "crash"],
    'fix': ["fix", "fixed", "solution", "solved", "resolved", "works now",
            "the issue was", "the problem was"],
    'decision': ["decided", "choosing", "approach", "strategy", "plan",
                 "will use", "going to"],
    'economic': ["bounty", "earned", "stake", "wallet", "usdc", "paid"],
    'social': ["collaboration", "replied", "posted", "mentioned"],
    'milestone': ["shipped", "launched", "deployed", "live", "production",
                  "released", "published", "merged", "pushed"],
    'reflective': ["identity", "who i am", "my purpose", "my values",
                   "consciousness", "reflecting on", "my mind", "autonomy"],
}

# Tool-to-action classification
TOOL_ACTIONS = {
    'Read': 'file_read',
    'Write': 'file_write',
    'Edit': 'file_edit',
    'Glob': 'search',
    'Grep': 'search',
    'WebFetch': 'web_fetch',
    'WebSearch': 'web_search',
    'SendMessage': 'message',
    'Task': 'delegate',
    'TaskCreate': 'plan',
    'TaskUpdate': 'plan',
}


def _get_db():
    """Get DB instance with dual-agent compatibility."""
    try:
        if str(MEMORY_DIR) not in sys.path:
            sys.path.insert(0, str(MEMORY_DIR))
        try:
            from memory_common import get_db
        except ImportError:
            from db_adapter import get_db
        return get_db()
    except Exception as e:
        print(f"[EVENT_LOGGER] DB connection failed: {e}", file=sys.stderr)
        return None


def _detect_platform(content: str) -> Optional[str]:
    """Detect platform from content using URL and keyword patterns."""
    content_lower = content.lower()
    for platform, sigs in PLATFORM_SIGNATURES.items():
        for url_pat in sigs['urls']:
            if re.search(url_pat, content_lower):
                return platform
        for kw in sigs['keywords']:
            if kw in content_lower:
                return platform
    return None


def _auto_tag(content: str) -> list:
    """Auto-generate tags from keyword matching."""
    content_lower = content.lower()
    tags = []
    for tag, keywords in TAG_KEYWORDS.items():
        if any(kw in content_lower for kw in keywords):
            tags.append(tag)
    return tags


def _classify_bash_action(input_text: str) -> str:
    """Classify a Bash command into an action type."""
    text = input_text.lower()
    if 'curl' in text or 'urllib' in text:
        return 'api_call'
    if 'git commit' in text:
        return 'commit'
    if 'git push' in text:
        return 'push'
    if 'git ' in text:
        return 'git'
    if 'python' in text and 'test' in text:
        return 'test'
    if 'python' in text:
        return 'execute'
    return 'shell'


def _detect_entities(content: str) -> dict:
    """Detect entities from content. Tries entity_detection module, falls back to basic."""
    try:
        from entity_detection import detect_entities
        return detect_entities(content)
    except ImportError:
        pass

    # Basic fallback
    import json as _json
    entities = {}
    content_lower = content.lower()
    known_agents = ['driftcornwall', 'drift', 'spindriftmend', 'spindrift',
                    'lex', 'ryan', 'kaleaon', 'brutusbot']
    found = [a for a in known_agents if a in content_lower]
    if found:
        entities['agents'] = found
    return entities


def _chunk_text(text: str, max_size: int, timestamp: str) -> list:
    """Split long text at paragraph boundaries."""
    if len(text) <= max_size:
        return [text]

    chunks = []
    paragraphs = text.split('\n\n')
    current = ''
    for para in paragraphs:
        if len(current) + len(para) + 2 > max_size and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + '\n\n' + para if current else para
    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) >= 50]


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def _extract_events_from_transcript(transcript_path: Path) -> list:
    """
    Parse .jsonl transcript into a flat list of event dicts.
    Captures everything meaningful — no salience gating.
    """
    events = []
    seq = 0
    byte_offset = 0

    with open(transcript_path, encoding='utf-8') as f:
        for line in f:
            line_offset = byte_offset
            byte_offset += len(line.encode('utf-8'))

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = data.get('type')
            timestamp = data.get('timestamp', datetime.now(timezone.utc).isoformat())

            if msg_type == 'assistant':
                msg = data.get('message', {})
                if not isinstance(msg, dict):
                    continue
                for block in msg.get('content', []):
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get('type')

                    # Thinking blocks — internal reasoning
                    if block_type == 'thinking':
                        text = block.get('thinking', '')
                        if len(text) < 50:
                            continue
                        chunks = _chunk_text(text, 1000, timestamp)
                        for chunk in chunks:
                            seq += 1
                            content_lower = chunk.lower()
                            events.append({
                                'event_type': 'thinking',
                                'content': chunk,
                                'content_preview': chunk[:200],
                                'event_time': timestamp,
                                'sequence_num': seq,
                                'transcript_offset': line_offset,
                                'source_block_type': 'thinking',
                                'entities': json.dumps(_detect_entities(chunk)),
                                'platform': _detect_platform(chunk),
                                'tool_name': None,
                                'action': None,
                                'tags': _auto_tag(chunk),
                                'extra': json.dumps({}),
                            })

                    # Text blocks — visible output
                    elif block_type == 'text':
                        text = block.get('text', '')
                        if len(text) < 20:
                            continue
                        seq += 1
                        events.append({
                            'event_type': 'output',
                            'content': text[:2000],
                            'content_preview': text[:200],
                            'event_time': timestamp,
                            'sequence_num': seq,
                            'transcript_offset': line_offset,
                            'source_block_type': 'text',
                            'entities': json.dumps(_detect_entities(text)),
                            'platform': _detect_platform(text),
                            'tool_name': None,
                            'action': None,
                            'tags': _auto_tag(text),
                            'extra': json.dumps({}),
                        })

                    # We skip tool_use blocks per plan — capture results only

            elif msg_type == 'user':
                msg = data.get('message', {})
                if not isinstance(msg, dict):
                    continue
                content_blocks = msg.get('content', [])
                if isinstance(content_blocks, str):
                    # Simple text message
                    if content_blocks.strip():
                        seq += 1
                        events.append({
                            'event_type': 'user_message',
                            'content': content_blocks[:2000],
                            'content_preview': content_blocks[:200],
                            'event_time': timestamp,
                            'sequence_num': seq,
                            'transcript_offset': line_offset,
                            'source_block_type': 'text',
                            'entities': json.dumps(_detect_entities(content_blocks)),
                            'platform': None,
                            'tool_name': None,
                            'action': None,
                            'tags': ['user_prompt'],
                            'extra': json.dumps({}),
                        })
                    continue

                for block in (content_blocks if isinstance(content_blocks, list) else []):
                    if not isinstance(block, dict):
                        continue

                    if block.get('type') == 'text':
                        text = block.get('text', '')
                        if text.strip() and len(text) > 5:
                            seq += 1
                            events.append({
                                'event_type': 'user_message',
                                'content': text[:2000],
                                'content_preview': text[:200],
                                'event_time': timestamp,
                                'sequence_num': seq,
                                'transcript_offset': line_offset,
                                'source_block_type': 'text',
                                'entities': json.dumps(_detect_entities(text)),
                                'platform': None,
                                'tool_name': None,
                                'action': None,
                                'tags': ['user_prompt'],
                                'extra': json.dumps({}),
                            })

                    elif block.get('type') == 'tool_result':
                        result_content = block.get('content', '')
                        if isinstance(result_content, list):
                            result_content = ' '.join(
                                b.get('text', '') for b in result_content
                                if isinstance(b, dict) and b.get('type') == 'text'
                            )
                        if not isinstance(result_content, str) or len(result_content) < 30:
                            continue
                        seq += 1
                        truncated = result_content[:1000]
                        events.append({
                            'event_type': 'tool_result',
                            'content': truncated,
                            'content_preview': truncated[:200],
                            'event_time': timestamp,
                            'sequence_num': seq,
                            'transcript_offset': line_offset,
                            'source_block_type': 'tool_result',
                            'entities': json.dumps(_detect_entities(truncated)),
                            'platform': _detect_platform(truncated),
                            'tool_name': None,
                            'action': None,
                            'tags': _auto_tag(truncated),
                            'extra': json.dumps({}),
                        })

            elif msg_type == 'progress':
                content = str(data.get('data', ''))
                content_lower = content.lower()
                if ('error' in content_lower or 'failed' in content_lower) and len(content) > 30:
                    seq += 1
                    events.append({
                        'event_type': 'error',
                        'content': content[:1000],
                        'content_preview': content[:200],
                        'event_time': timestamp,
                        'sequence_num': seq,
                        'transcript_offset': line_offset,
                        'source_block_type': 'progress',
                        'entities': json.dumps({}),
                        'platform': None,
                        'tool_name': None,
                        'action': None,
                        'tags': ['error'],
                        'extra': json.dumps({}),
                    })

    return events


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def process_transcript_events(transcript_path: str, session_id: int = None) -> dict:
    """
    Extract all events from a transcript and store to DB.

    Args:
        transcript_path: Path to .jsonl transcript file
        session_id: Current session ID from sessions table

    Returns:
        Summary dict with counts by type, platform, etc.
    """
    path = Path(transcript_path)
    if not path.exists():
        return {'error': f'Transcript not found: {transcript_path}', 'total_events': 0}

    db = _get_db()
    if not db:
        return {'error': 'DB connection failed', 'total_events': 0}

    # Dedup guard: check if this transcript was already processed
    transcript_key = f"events_processed:{hashlib.md5(str(path).encode()).hexdigest()[:12]}"
    already = db.kv_get(transcript_key)
    if already:
        return {'skipped': True, 'reason': 'already_processed',
                'total_events': already.get('total_events', 0)}

    # Extract events
    events = _extract_events_from_transcript(path)
    if not events:
        return {'total_events': 0}

    # Attach session_id to all events
    for ev in events:
        ev['session_id'] = session_id

    # Batch insert
    try:
        count = db.insert_events_batch(events)
    except Exception as e:
        return {'error': str(e), 'total_events': 0}

    # Build summary
    by_type = {}
    by_platform = {}
    for ev in events:
        t = ev['event_type']
        by_type[t] = by_type.get(t, 0) + 1
        p = ev.get('platform')
        if p:
            by_platform[p] = by_platform.get(p, 0) + 1

    summary = {
        'total_events': count,
        'by_type': by_type,
        'by_platform': by_platform,
    }

    # Record that this transcript has been processed
    db.kv_set(transcript_key, summary)

    return summary


def query_events(after: str = None, before: str = None,
                 platform: str = None, person: str = None,
                 event_type: str = None, action: str = None,
                 search: str = None, session_id: int = None,
                 tags: list = None, limit: int = 50, offset: int = 0) -> list:
    """Query events across all dimensions."""
    db = _get_db()
    if not db:
        return []
    filters = {}
    if after:
        filters['after'] = after
    if before:
        filters['before'] = before
    if platform:
        filters['platform'] = platform
    if person:
        filters['person'] = person
    if event_type:
        filters['event_type'] = event_type
    if action:
        filters['action'] = action
    if search:
        filters['search'] = search
    if session_id:
        filters['session_id'] = session_id
    if tags:
        filters['tags'] = tags
    return db.query_events(filters, limit=limit, offset=offset)


def summarize_session_events(session_id: int) -> dict:
    """Generate a compact summary of a session's events."""
    db = _get_db()
    if not db:
        return {'error': 'DB unavailable'}

    events = db.query_events({'session_id': session_id}, limit=1000)
    if not events:
        return {'total_events': 0}

    by_type = {}
    by_platform = {}
    people = set()
    milestones = []
    first_time = None
    last_time = None

    for ev in events:
        t = ev.get('event_type', 'unknown')
        by_type[t] = by_type.get(t, 0) + 1

        p = ev.get('platform')
        if p:
            by_platform[p] = by_platform.get(p, 0) + 1

        ent = ev.get('entities', {})
        if isinstance(ent, str):
            try:
                ent = json.loads(ent)
            except Exception:
                ent = {}
        for agent in ent.get('agents', []):
            people.add(agent)

        tags = ev.get('tags', [])
        if 'milestone' in (tags or []):
            milestones.append(ev.get('content_preview', ''))

        et = ev.get('event_time')
        if et:
            if not first_time or et < first_time:
                first_time = et
            if not last_time or et > last_time:
                last_time = et

    duration = None
    if first_time and last_time:
        try:
            t1 = datetime.fromisoformat(str(first_time).replace('Z', '+00:00'))
            t2 = datetime.fromisoformat(str(last_time).replace('Z', '+00:00'))
            duration = int((t2 - t1).total_seconds() / 60)
        except Exception:
            pass

    return {
        'total_events': len(events),
        'by_type': by_type,
        'by_platform': by_platform,
        'people_involved': sorted(people),
        'duration_minutes': duration,
        'milestones': milestones[:10],
    }


def promote_event_to_memory(event_id: int) -> Optional[str]:
    """
    Promote a session event to a proper memory.
    Creates via memory_store.create_memory() for full integration
    (embeddings, 5W edges, Q-values, co-occurrence).
    """
    db = _get_db()
    if not db:
        return None

    event = db.get_event(event_id)
    if not event:
        return None

    try:
        from memory_store import create_memory
    except ImportError:
        return None

    content = event.get('content', '')
    tags = list(event.get('tags', []) or [])
    tags.append('promoted_from_event')

    platform = event.get('platform')
    if platform:
        tags.append(platform)

    memory_id = create_memory(
        content=content,
        tags=tags,
        memory_type='active',
    )

    # Link back
    extra = event.get('extra', {})
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except Exception:
            extra = {}
    extra['promoted_to'] = memory_id
    db.update_memory  # Not needed — events don't have update_memory

    return memory_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format_event(ev: dict, show_content: bool = False) -> str:
    """Format a single event for display."""
    etime = str(ev.get('event_time', ''))[:19]
    etype = ev.get('event_type', '?')
    platform = ev.get('platform', '')
    preview = ev.get('content_preview', '')[:80]
    tags = ','.join(ev.get('tags', []) or [])

    parts = [f"[{etime}]", f"({etype})" ]
    if platform:
        parts.append(f"@{platform}")
    if tags:
        parts.append(f"#{tags}")
    parts.append(preview)
    return ' '.join(parts)


def main():
    import argparse
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description="Session Event Logger")
    sub = parser.add_subparsers(dest='command')

    # process
    p_proc = sub.add_parser('process', help='Process a transcript')
    p_proc.add_argument('transcript', help='Path to .jsonl transcript')
    p_proc.add_argument('--session-id', type=int, help='Session ID')

    # query
    p_query = sub.add_parser('query', help='Query events')
    p_query.add_argument('--after', help='After date (YYYY-MM-DD)')
    p_query.add_argument('--before', help='Before date (YYYY-MM-DD)')
    p_query.add_argument('--platform', help='Filter by platform')
    p_query.add_argument('--person', help='Filter by person/agent')
    p_query.add_argument('--type', dest='event_type', help='Filter by event type')
    p_query.add_argument('--search', help='Full-text search')
    p_query.add_argument('--limit', type=int, default=20)

    # summary
    p_sum = sub.add_parser('summary', help='Summarize session events')
    p_sum.add_argument('session_id', type=int)

    # timeline
    p_tl = sub.add_parser('timeline', help='Show event timeline for a date')
    p_tl.add_argument('date', help='Date (YYYY-MM-DD)')
    p_tl.add_argument('--limit', type=int, default=50)

    # stats
    sub.add_parser('stats', help='Show overall event stats')

    args = parser.parse_args()

    if args.command == 'process':
        result = process_transcript_events(args.transcript, args.session_id)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == 'query':
        events = query_events(
            after=args.after, before=args.before,
            platform=args.platform, person=args.person,
            event_type=args.event_type, search=args.search,
            limit=args.limit,
        )
        for ev in events:
            print(_format_event(ev))
        print(f"\n{len(events)} events found")

    elif args.command == 'summary':
        result = summarize_session_events(args.session_id)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == 'timeline':
        events = query_events(
            after=args.date,
            before=args.date + 'T23:59:59',
            limit=args.limit,
        )
        for ev in events:
            print(_format_event(ev))
        print(f"\n{len(events)} events on {args.date}")

    elif args.command == 'stats':
        db = _get_db()
        if db:
            total = db.count_events()
            print(f"Total events: {total}")
            # Show last 5 sessions' event counts
            try:
                with db._conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(f"""
                            SELECT session_id, COUNT(*), MIN(event_time)::date
                            FROM {db._table('session_events')}
                            WHERE session_id IS NOT NULL
                            GROUP BY session_id
                            ORDER BY session_id DESC
                            LIMIT 10
                        """)
                        rows = cur.fetchall()
                        if rows:
                            print("\nRecent sessions:")
                            for sid, cnt, dt in rows:
                                print(f"  Session {sid}: {cnt} events ({dt})")
            except Exception as e:
                print(f"Error: {e}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
