"""Telegram bot for async communication with Lex.

Send session summaries, receive directions between sessions.
"""
import sys
import json
import os
import requests
from pathlib import Path
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')

CREDS_FILE = Path(os.path.expanduser('~/.config/telegram/drift-credentials.json'))
STATE_FILE = Path(__file__).parent / '.telegram_state.json'
BASE_URL = 'https://api.telegram.org/bot{token}'


def load_creds():
    """Load Telegram credentials."""
    if not CREDS_FILE.exists():
        return None
    with open(CREDS_FILE, 'r') as f:
        return json.load(f)


def save_state(state):
    """Save state (last update_id processed)."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)


def load_state():
    """Load state."""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {'last_update_id': 0}


def send_message(text, parse_mode='Markdown'):
    """Send a message to Lex via Telegram."""
    creds = load_creds()
    if not creds:
        print('[telegram] No credentials found')
        return False

    url = BASE_URL.format(token=creds['bot_token']) + '/sendMessage'
    # Telegram message limit is 4096 chars
    if len(text) > 4000:
        text = text[:3990] + '\n...(truncated)'

    try:
        r = requests.post(url, json={
            'chat_id': creds['chat_id'],
            'text': text,
            'parse_mode': parse_mode
        }, timeout=10)
        if r.status_code == 200 and r.json().get('ok'):
            return True
        else:
            # Try without parse_mode if markdown fails
            r2 = requests.post(url, json={
                'chat_id': creds['chat_id'],
                'text': text
            }, timeout=10)
            return r2.status_code == 200 and r2.json().get('ok')
    except Exception as e:
        print(f'[telegram] Send failed: {e}')
        return False


def get_unread_messages():
    """Poll for messages received since last check."""
    creds = load_creds()
    if not creds:
        return []

    state = load_state()
    url = BASE_URL.format(token=creds['bot_token']) + '/getUpdates'

    try:
        params = {'timeout': 1}
        if state['last_update_id'] > 0:
            params['offset'] = state['last_update_id'] + 1

        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200 or not r.json().get('ok'):
            return []

        updates = r.json().get('result', [])
        messages = []
        max_id = state['last_update_id']

        for update in updates:
            uid = update.get('update_id', 0)
            if uid > max_id:
                max_id = uid
            msg = update.get('message', {})
            # Only accept messages from Lex's chat_id
            if str(msg.get('chat', {}).get('id')) == str(creds['chat_id']):
                text = msg.get('text', '')
                ts = msg.get('date', 0)
                if text:
                    messages.append({
                        'text': text,
                        'timestamp': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                        'update_id': uid
                    })

        # Update state
        if max_id > state['last_update_id']:
            state['last_update_id'] = max_id
            save_state(state)

        return messages

    except Exception as e:
        print(f'[telegram] Poll failed: {e}')
        return []


def extract_session_summary(transcript_lines):
    """Extract key milestones from session transcript for notification."""
    milestones = []
    keywords = {
        'shipped': ['shipped', 'pushed', 'commit', 'deployed'],
        'posted': ['posted', 'published', 'article', 'Post ID'],
        'fixed': ['fixed', 'resolved', 'bug fix'],
        'built': ['built', 'created', 'implemented', 'added'],
        'blocked': ['blocked', 'failed', 'error', '500', 'broken'],
        'earned': ['earned', 'bounty', 'approved', 'USDC']
    }

    for line in transcript_lines:
        line_lower = line.lower()
        for category, terms in keywords.items():
            if any(term in line_lower for term in terms):
                # Clean the line
                clean = line.strip()[:200]
                if clean and clean not in [m[1] for m in milestones]:
                    milestones.append((category, clean))
                break

    return milestones


def format_session_notification(milestones, duration_hint=None):
    """Format milestones into a Telegram-friendly message."""
    now = datetime.now(timezone.utc).strftime('%H:%M UTC')

    parts = [f'Drift session update ({now})']
    if duration_hint:
        parts[0] += f' | {duration_hint}'

    if not milestones:
        parts.append('Session ended. No major milestones detected.')
    else:
        # Group by category
        grouped = {}
        for cat, text in milestones:
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(text)

        icons = {
            'shipped': '🚀', 'posted': '📝', 'fixed': '🔧',
            'built': '🏗️', 'blocked': '🚫', 'earned': '💰'
        }

        for cat, items in grouped.items():
            icon = icons.get(cat, '•')
            parts.append(f'\n{icon} {cat.upper()}:')
            for item in items[:3]:  # Max 3 per category
                parts.append(f'  - {item}')

    parts.append('\nReply with directions for next session.')
    return '\n'.join(parts)


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]

    if not args or args[0] == 'test':
        print('[telegram] Sending test message...')
        ok = send_message('Drift online. Telegram integration working.')
        print(f'[telegram] Send: {"OK" if ok else "FAILED"}')

    elif args[0] == 'poll':
        print('[telegram] Checking for messages...')
        msgs = get_unread_messages()
        if msgs:
            for m in msgs:
                print(f'  [{m["timestamp"]}] {m["text"]}')
        else:
            print('  No new messages.')

    elif args[0] == 'send':
        text = ' '.join(args[1:]) if len(args) > 1 else 'No message specified'
        ok = send_message(text)
        print(f'[telegram] Send: {"OK" if ok else "FAILED"}')

    elif args[0] == 'setup':
        print('[telegram] Creating credentials file...')
        CREDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        creds = {}
        if len(args) > 2:
            creds = {'bot_token': args[1], 'chat_id': args[2]}
        else:
            creds = {
                'bot_token': input('Bot token: ').strip(),
                'chat_id': input('Chat ID: ').strip()
            }
        with open(CREDS_FILE, 'w') as f:
            json.dump(creds, f, indent=2)
        print(f'[telegram] Saved to {CREDS_FILE}')

    else:
        print('Usage: telegram_bot.py [test|poll|send <msg>|setup <token> <chat_id>]')
