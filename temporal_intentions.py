#!/usr/bin/env python3
"""
Temporal Intentions — Prospective Memory for Drift

"Remember to do X when Y happens."

Implements forward-looking temporal intentions that persist across sessions.
Intentions are checked at session start and surfaced when triggered.

Phase 2 of the Voss Review implementation plan (Gap 3: No Prospective Memory).

Storage: DB key_value_store, key pattern '.intention.{uuid}'

Usage:
    python temporal_intentions.py create "Follow up with opspawn" --trigger-type time --trigger "2026-03-01"
    python temporal_intentions.py create "Check Hedera hackathon results" --trigger-type event --trigger "hedera apex"
    python temporal_intentions.py list                    # Show all pending
    python temporal_intentions.py check                   # Check what's triggered now
    python temporal_intentions.py complete <id> --outcome "Done"
    python temporal_intentions.py expire                  # Expire past-due intentions
"""

import json
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

MEMORY_DIR = Path(__file__).parent

# Add memory dir to path for imports
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))


def _get_db():
    from db_adapter import get_db
    return get_db()


def _now_iso():
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


def _intention_key(intention_id: str) -> str:
    return f'.intention.{intention_id}'


# ============================================================
# CRUD Operations
# ============================================================

def create_intention(action: str, trigger_type: str = 'time',
                     trigger_condition: str = '', priority: str = 'medium',
                     expiry_days: int = 30) -> dict:
    """
    Create a new temporal intention.

    Args:
        action: What to do when triggered (free text instruction)
        trigger_type: 'time' (datetime), 'event' (keyword match), 'condition' (structured)
        trigger_condition: When to trigger:
            - time: ISO date or datetime (e.g., '2026-03-01', '2026-02-20T15:00')
            - event: keyword or phrase to match in session context
            - condition: JSON string with structured conditions
        priority: 'high', 'medium', or 'low'
        expiry_days: Days until auto-expiry (default 30)

    Returns:
        The created intention dict
    """
    db = _get_db()
    intention_id = uuid.uuid4().hex[:8]

    # Parse trigger condition for time type
    parsed_trigger = trigger_condition
    if trigger_type == 'time' and trigger_condition:
        # Normalize date-only to datetime
        if len(trigger_condition) <= 10:  # date only like '2026-03-01'
            parsed_trigger = trigger_condition + 'T00:00:00'

    # Calculate expiry
    expiry = None
    if expiry_days > 0:
        expiry = (datetime.now() + timedelta(days=expiry_days)).strftime('%Y-%m-%dT%H:%M:%S')

    intention = {
        'id': intention_id,
        'action': action,
        'trigger_type': trigger_type,
        'trigger_condition': parsed_trigger,
        'priority': priority,
        'created': _now_iso(),
        'status': 'pending',
        'expiry': expiry,
        'outcome': None,
        'triggered_at': None,
    }

    db.kv_set(_intention_key(intention_id), intention)
    return intention


def get_intention(intention_id: str) -> dict | None:
    """Get a single intention by ID."""
    db = _get_db()
    return db.kv_get(_intention_key(intention_id))


def list_intentions(status: str = None) -> list[dict]:
    """
    List all intentions, optionally filtered by status.

    Args:
        status: Filter by 'pending', 'triggered', 'completed', 'expired', or None for all
    """
    db = _get_db()
    intentions = []

    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT key, value FROM {db._table('key_value_store')} "
                f"WHERE key LIKE '.intention.%'"
            )
            for row in cur.fetchall():
                val = row[1] if isinstance(row[1], dict) else json.loads(row[1])
                if status is None or val.get('status') == status:
                    intentions.append(val)

    # Sort by priority (high first) then by created date
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    intentions.sort(key=lambda x: (priority_order.get(x.get('priority', 'medium'), 1),
                                   x.get('created', '')))
    return intentions


def complete_intention(intention_id: str, outcome: str = '') -> dict | None:
    """Mark an intention as completed with optional outcome notes."""
    db = _get_db()
    intention = db.kv_get(_intention_key(intention_id))
    if not intention:
        return None

    intention['status'] = 'completed'
    intention['outcome'] = outcome
    intention['completed_at'] = _now_iso()
    db.kv_set(_intention_key(intention_id), intention)
    return intention


def expire_intention(intention_id: str) -> dict | None:
    """Mark an intention as expired."""
    db = _get_db()
    intention = db.kv_get(_intention_key(intention_id))
    if not intention:
        return None

    intention['status'] = 'expired'
    db.kv_set(_intention_key(intention_id), intention)
    return intention


def delete_intention(intention_id: str) -> bool:
    """Permanently delete an intention."""
    db = _get_db()
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {db._table('key_value_store')} WHERE key = %s",
                (_intention_key(intention_id),)
            )
            conn.commit()
            return cur.rowcount > 0


# ============================================================
# Trigger Checking (called at session start)
# ============================================================

def check_triggers(context: str = '') -> list[dict]:
    """
    Check all pending intentions against current context.
    Returns list of triggered intentions (already updated to 'triggered' status).

    Args:
        context: Current session context string (date, platform info, etc.)
            Used for event-type trigger matching.
    """
    db = _get_db()
    pending = list_intentions(status='pending')
    now = _now_iso()
    triggered = []

    for intention in pending:
        trigger_type = intention.get('trigger_type', 'time')
        condition = intention.get('trigger_condition', '')

        # Check expiry first
        expiry = intention.get('expiry')
        if expiry and now > expiry:
            intention['status'] = 'expired'
            db.kv_set(_intention_key(intention['id']), intention)
            continue

        match = False

        if trigger_type == 'time':
            # Time trigger: fire if current datetime >= trigger datetime
            if condition and now >= condition:
                match = True

        elif trigger_type == 'event':
            # Event trigger: fire if trigger keywords appear in context
            if condition and context:
                keywords = condition.lower().split()
                context_lower = context.lower()
                # All keywords must appear
                if all(kw in context_lower for kw in keywords):
                    match = True

        elif trigger_type == 'condition':
            # Structured condition: check JSON fields against context
            try:
                cond = json.loads(condition) if isinstance(condition, str) else condition
                if isinstance(cond, dict) and context:
                    context_lower = context.lower()
                    match = all(
                        str(v).lower() in context_lower
                        for v in cond.values()
                    )
            except (json.JSONDecodeError, TypeError):
                pass

        if match:
            intention['status'] = 'triggered'
            intention['triggered_at'] = now
            db.kv_set(_intention_key(intention['id']), intention)
            triggered.append(intention)

    return triggered


def check_and_format(context: str = '') -> str:
    """
    Check triggers and return formatted context string for session priming.
    Returns empty string if no intentions are triggered.
    """
    triggered = check_triggers(context)
    pending = list_intentions(status='pending')

    parts = []

    if triggered:
        parts.append('=== PROSPECTIVE MEMORY (triggered intentions) ===')
        for i in triggered:
            priority_marker = '[!]' if i.get('priority') == 'high' else '[ ]'
            parts.append(f"  {priority_marker} {i['action']}")
            parts.append(f"      id: {i['id']} | created: {i.get('created', '?')[:10]} | "
                         f"trigger: {i.get('trigger_type')}={i.get('trigger_condition', '?')[:40]}")
        parts.append(f'  ({len(triggered)} intention(s) triggered — act on these)')
        parts.append('')

    if pending:
        parts.append(f'=== PENDING INTENTIONS ({len(pending)} waiting) ===')
        for i in pending[:5]:  # Show at most 5
            trigger_preview = i.get('trigger_condition', '?')[:30]
            parts.append(f"  [{i.get('priority', '?')[0].upper()}] {i['action'][:60]}"
                         f" (when: {i.get('trigger_type')}={trigger_preview})")
        if len(pending) > 5:
            parts.append(f'  ... and {len(pending) - 5} more')
        parts.append('')

    return '\n'.join(parts)


# ============================================================
# Session End: Extract intentions from transcript (Phase 2b)
# ============================================================

def extract_from_transcript(transcript_path: str, max_intentions: int = 3) -> list[dict]:
    """
    Extract temporal intentions from session transcript.
    Looks for patterns like "I should...", "next time...", "follow up on...", "by March..."

    This is Phase 2b — starts with simple pattern matching, not LLM.
    Returns list of created intentions (already stored).
    """
    import re

    if not transcript_path or not Path(transcript_path).exists():
        return []

    # Read transcript (JSONL format)
    text_blocks = []
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # Extract assistant thinking and output text
                    if entry.get('type') == 'assistant':
                        msg = entry.get('message', {})
                        content = msg.get('content', [])
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get('type') == 'thinking':
                                    text_blocks.append(block.get('thinking', ''))
                                elif isinstance(block, dict) and block.get('type') == 'text':
                                    text_blocks.append(block.get('text', ''))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []

    if not text_blocks:
        return []

    full_text = '\n'.join(text_blocks)

    # Pattern matching for temporal intentions
    # These patterns indicate forward-looking commitments
    patterns = [
        # "I should follow up on X by DATE"
        r'(?:I should|need to|must|have to|want to)\s+(.{10,80}?)(?:\.|$)',
        # "Next session: X"
        r'(?:next session|next time|tomorrow|later)[:,]?\s+(.{10,80}?)(?:\.|$)',
        # "Follow up on/with X"
        r'(?:follow up (?:on|with))\s+(.{10,80}?)(?:\.|$)',
        # "Remember to X"
        r'(?:remember to|don\'t forget to)\s+(.{10,80}?)(?:\.|$)',
    ]

    candidates = []
    seen = set()
    for pattern in patterns:
        matches = re.finditer(pattern, full_text, re.IGNORECASE)
        for m in matches:
            action = m.group(1).strip()
            # Filter out noise
            if len(action) < 15 or len(action) > 100:
                continue
            # Dedup by first 30 chars
            key = action[:30].lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(action)

    # Extract date references for trigger conditions
    date_pattern = r'(?:by|before|on|until)\s+(\d{4}-\d{2}-\d{2}|(?:March|April|May|June|July|August|September|October|November|December|January|February)\s+\d{1,2}(?:,?\s+\d{4})?)'

    created = []
    for action in candidates[:max_intentions]:
        # Try to find a date in the action text
        trigger_type = 'event'
        trigger_condition = action[:30].lower()  # Default: event trigger on keywords

        date_match = re.search(date_pattern, action, re.IGNORECASE)
        if date_match:
            trigger_type = 'time'
            date_str = date_match.group(1)
            # Try to parse the date
            try:
                for fmt in ['%Y-%m-%d', '%B %d, %Y', '%B %d %Y', '%B %d']:
                    try:
                        parsed = datetime.strptime(date_str, fmt)
                        if parsed.year < 2000:  # No year specified
                            parsed = parsed.replace(year=datetime.now().year)
                        trigger_condition = parsed.strftime('%Y-%m-%dT00:00:00')
                        break
                    except ValueError:
                        continue
            except Exception:
                pass

        intention = create_intention(
            action=action,
            trigger_type=trigger_type,
            trigger_condition=trigger_condition,
            priority='medium',
            expiry_days=14,  # Auto-extracted intentions expire sooner
        )
        created.append(intention)

    return created


# ============================================================
# CLI
# ============================================================

def _format_intention(i: dict) -> str:
    """Format an intention for display."""
    status_icons = {
        'pending': '  ',
        'triggered': '>>',
        'completed': 'ok',
        'expired': 'xx',
    }
    icon = status_icons.get(i.get('status', 'pending'), '??')
    priority = i.get('priority', 'medium')[0].upper()
    trigger = f"{i.get('trigger_type', '?')}={i.get('trigger_condition', '?')[:35]}"
    return (f"[{icon}] [{priority}] {i['id']}  {i.get('action', '?')[:55]}\n"
            f"         trigger: {trigger}  |  created: {i.get('created', '?')[:10]}  |  "
            f"status: {i.get('status', '?')}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Temporal Intentions — Prospective Memory')
    sub = parser.add_subparsers(dest='command')

    # create
    p_create = sub.add_parser('create', help='Create a new intention')
    p_create.add_argument('action', help='What to do when triggered')
    p_create.add_argument('--trigger-type', '-t', choices=['time', 'event', 'condition'],
                          default='time', help='Trigger type')
    p_create.add_argument('--trigger', '-w', default='', help='Trigger condition')
    p_create.add_argument('--priority', '-p', choices=['high', 'medium', 'low'],
                          default='medium', help='Priority level')
    p_create.add_argument('--expiry-days', type=int, default=30, help='Days until expiry')

    # list
    p_list = sub.add_parser('list', help='List intentions')
    p_list.add_argument('--status', '-s', choices=['pending', 'triggered', 'completed', 'expired'],
                        help='Filter by status')
    p_list.add_argument('--json', action='store_true', help='Output as JSON')

    # check
    p_check = sub.add_parser('check', help='Check triggered intentions')
    p_check.add_argument('--context', '-c', default='', help='Context string for event matching')

    # complete
    p_complete = sub.add_parser('complete', help='Mark intention as completed')
    p_complete.add_argument('id', help='Intention ID')
    p_complete.add_argument('--outcome', '-o', default='', help='Completion notes')

    # expire
    sub.add_parser('expire', help='Expire all past-due intentions')

    # delete
    p_delete = sub.add_parser('delete', help='Delete an intention')
    p_delete.add_argument('id', help='Intention ID')

    # extract (Phase 2b)
    p_extract = sub.add_parser('extract', help='Extract intentions from transcript')
    p_extract.add_argument('transcript', help='Path to transcript JSONL file')
    p_extract.add_argument('--max', type=int, default=3, help='Max intentions to extract')

    args = parser.parse_args()

    if args.command == 'create':
        i = create_intention(args.action, args.trigger_type, args.trigger,
                             args.priority, args.expiry_days)
        print(f"Created intention [{i['id']}]: {i['action']}")
        print(f"  Trigger: {i['trigger_type']}={i['trigger_condition']}")
        print(f"  Priority: {i['priority']}, Expires: {i['expiry'][:10] if i['expiry'] else 'never'}")

    elif args.command == 'list':
        intentions = list_intentions(status=args.status)
        if args.json:
            print(json.dumps(intentions, indent=2))
        elif not intentions:
            print('No intentions found.')
        else:
            for i in intentions:
                print(_format_intention(i))
                print()

    elif args.command == 'check':
        context = args.context or f"date={datetime.now().strftime('%Y-%m-%d')}"
        output = check_and_format(context)
        if output:
            print(output)
        else:
            print('No intentions triggered.')

    elif args.command == 'complete':
        result = complete_intention(args.id, args.outcome)
        if result:
            print(f"Completed [{args.id}]: {result['action']}")
            if args.outcome:
                print(f"  Outcome: {args.outcome}")
        else:
            print(f"Intention {args.id} not found.")

    elif args.command == 'expire':
        pending = list_intentions(status='pending')
        now = _now_iso()
        expired_count = 0
        for i in pending:
            expiry = i.get('expiry')
            if expiry and now > expiry:
                expire_intention(i['id'])
                expired_count += 1
                print(f"  Expired [{i['id']}]: {i['action'][:50]}")
        print(f'\n{expired_count} intention(s) expired.')

    elif args.command == 'delete':
        if delete_intention(args.id):
            print(f"Deleted intention {args.id}.")
        else:
            print(f"Intention {args.id} not found.")

    elif args.command == 'extract':
        created = extract_from_transcript(args.transcript, args.max)
        if created:
            print(f'Extracted {len(created)} intention(s) from transcript:')
            for i in created:
                print(f"  [{i['id']}] {i['action'][:60]}")
                print(f"    trigger: {i['trigger_type']}={i['trigger_condition'][:40]}")
        else:
            print('No temporal intentions found in transcript.')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
