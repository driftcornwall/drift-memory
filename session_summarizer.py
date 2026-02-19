"""
Session Transcript Summarizer — GPT-5-mini with Gemma fallback.
Extracts threads, lessons, and facts from session transcripts.
Stores each as a dated memory in the graph with full pipeline integration.

Usage:
    python session_summarizer.py <transcript.jsonl>
    python session_summarizer.py <transcript.jsonl> --dry-run   # Preview without storing
"""
import json
import os
import sys
import time
import re
from datetime import datetime, timezone
from pathlib import Path

# Ensure memory dir is importable
MEMORY_DIR = Path(__file__).parent
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))


# ---------------------------------------------------------------------------
# Transcript extraction
# ---------------------------------------------------------------------------

def extract_session_text(jsonl_path: str, max_chars: int = 10000) -> tuple[str, dict]:
    """Extract and condense session transcript. Returns (text, metadata)."""
    entries = []
    with open(jsonl_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass

    texts = []
    user_msgs = 0
    assistant_msgs = 0
    thinking_blocks = 0

    for entry in entries:
        if entry.get('type') == 'assistant':
            msg = entry.get('message', {})
            for block in msg.get('content', []):
                if block.get('type') == 'thinking':
                    texts.append(f"[THINKING] {block['thinking']}")
                    thinking_blocks += 1
                elif block.get('type') == 'text':
                    texts.append(f"[ASSISTANT] {block['text']}")
                    assistant_msgs += 1
        elif entry.get('type') == 'human':
            msg = entry.get('message', {})
            for block in msg.get('content', []):
                if isinstance(block, dict) and block.get('type') == 'text':
                    # Skip system-reminder blocks
                    text = block['text']
                    if not text.startswith('<system-reminder>'):
                        texts.append(f"[USER] {text}")
                        user_msgs += 1
                elif isinstance(block, str) and not block.startswith('<system-reminder>'):
                    texts.append(f"[USER] {block}")
                    user_msgs += 1

    full = "\n".join(texts)
    meta = {
        'total_entries': len(entries),
        'user_messages': user_msgs,
        'assistant_messages': assistant_msgs,
        'thinking_blocks': thinking_blocks,
        'total_chars': len(full),
    }

    if len(full) <= max_chars:
        return full, meta

    # Proportional sampling: 40% start, 20% middle, 40% end
    chunk = max_chars // 5
    start = full[:chunk * 2]
    mid_point = len(full) // 2
    middle = full[mid_point - chunk:mid_point + chunk]
    end = full[-chunk * 2:]

    condensed = f"{start}\n\n[... middle of session ...]\n\n{middle}\n\n[... later in session ...]\n\n{end}"
    return condensed, meta


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a session analyst. You extract structured summaries from AI agent work session transcripts. Be specific — include names, numbers, URLs, and concrete outcomes. Never be vague."

EXTRACTION_PROMPT = """Analyze this AI agent session transcript. Extract EXACTLY this structure:

THREAD: [short name]
SUMMARY: [2-3 specific sentences about what happened]
STATUS: [completed/in-progress/blocked]

THREAD: [short name]
SUMMARY: [2-3 specific sentences]
STATUS: [status]

THREAD: [short name]
SUMMARY: [2-3 specific sentences]
STATUS: [status]

LESSON: [concrete thing learned — include specifics]
LESSON: [another concrete lesson]
LESSON: [another concrete lesson]

FACT: [specific config, decision, URL, or number to remember]
FACT: [another specific fact]
FACT: [another specific fact]

SESSION TRANSCRIPT:
{session_text}

Begin analysis:

THREAD:"""


def _call_openai(session_text: str) -> str:
    """Call OpenAI API for structured extraction. Uses gpt-4o-mini (cheapest: $0.15/$0.60 per 1M tokens)."""
    import requests

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError("No OPENAI_API_KEY environment variable")

    # Model preference: gpt-4o-mini is cheapest and fast. Override via env var.
    model = os.environ.get('DRIFT_SUMMARY_MODEL', 'gpt-4o-mini')
    prompt = EXTRACTION_PROMPT.format(session_text=session_text)

    resp = requests.post('https://api.openai.com/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        json={
            'model': model,
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.3,
            'max_tokens': 1000,
        },
        timeout=60
    )

    if resp.status_code != 200:
        raise RuntimeError(f"{model} error {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    content = data['choices'][0]['message']['content']
    usage = data.get('usage', {})
    return content, {
        'model': model,
        'prompt_tokens': usage.get('prompt_tokens', 0),
        'completion_tokens': usage.get('completion_tokens', 0),
    }


def _call_gemma(session_text: str) -> str:
    """Call Gemma 3 4B via local ollama."""
    import requests

    prompt = EXTRACTION_PROMPT.format(session_text=session_text)

    resp = requests.post('http://localhost:11434/api/generate',
        json={
            'model': 'gemma3:4b',
            'prompt': f"{SYSTEM_PROMPT}\n\n{prompt}",
            'stream': False,
            'options': {
                'temperature': 0.3,
                'num_predict': 800,
            }
        },
        timeout=300
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Gemma error {resp.status_code}")

    data = resp.json()
    return data.get('response', ''), {
        'model': 'gemma3:4b',
        'eval_count': data.get('eval_count', 0),
    }


def summarize_session(session_text: str) -> tuple[str, dict]:
    """Call GPT-5-mini, fall back to Gemma on failure."""
    # Try gpt-4.1-mini first (fast, cheap, great at structured extraction)
    try:
        result, meta = _call_openai(session_text)
        if result and len(result) > 50:
            return "THREAD:" + result, meta
    except Exception as e:
        print(f"  OpenAI failed: {e}")

    # Fallback to Gemma
    try:
        result, meta = _call_gemma(session_text)
        if result and len(result) > 50:
            return result, meta
    except Exception as e:
        print(f"  Gemma fallback also failed: {e}")

    return "", {'model': 'none', 'error': 'all backends failed'}


# ---------------------------------------------------------------------------
# Parser: structured output -> individual memories
# ---------------------------------------------------------------------------

def parse_extraction(raw: str) -> dict:
    """Parse the structured LLM output into threads, lessons, facts."""
    threads = []
    lessons = []
    facts = []

    # Strip markdown bold markers: **THREAD:** -> THREAD:
    raw = re.sub(r'\*\*([A-Z]+:)\*\*', r'\1', raw)
    raw = re.sub(r'\*\*([A-Z]+ \d*:?)\*\*', r'\1', raw)

    lines = raw.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Parse THREAD blocks
        if line.upper().startswith('THREAD:'):
            name = line.split(':', 1)[1].strip()
            # Remove duplicate THREAD: prefix if present
            if name.upper().startswith('THREAD:'):
                name = name.split(':', 1)[1].strip()
            summary = ""
            status = "unknown"
            i += 1
            while i < len(lines):
                l = lines[i].strip()
                if l.upper().startswith('SUMMARY:'):
                    summary = l.split(':', 1)[1].strip()
                elif l.upper().startswith('STATUS:'):
                    status = l.split(':', 1)[1].strip().lower()
                elif l.upper().startswith(('THREAD:', 'LESSON:', 'FACT:')):
                    break
                else:
                    if summary and l:
                        summary += " " + l
                i += 1
            if name:
                threads.append({'name': name, 'summary': summary, 'status': status})
            continue

        # Parse LESSON lines
        if line.upper().startswith('LESSON:'):
            lesson = line.split(':', 1)[1].strip()
            if lesson:
                lessons.append(lesson)

        # Parse FACT lines
        if line.upper().startswith('FACT:'):
            fact = line.split(':', 1)[1].strip()
            if fact:
                facts.append(fact)

        i += 1

    return {'threads': threads, 'lessons': lessons, 'facts': facts}


# ---------------------------------------------------------------------------
# Memory storage with full pipeline integration
# ---------------------------------------------------------------------------

def store_memories(parsed: dict, session_date: str, session_num: int = None,
                   transcript_path: str = None, dry_run: bool = False) -> list[str]:
    """Store extracted items as memories with proper graph integration."""
    stored_ids = []
    session_tag = f"session-{session_date}"

    if not dry_run:
        from memory_store import store_memory

    # Store each thread as a memory
    for i, thread in enumerate(parsed.get('threads', []), 1):
        content = f"[Session {session_date}] Thread: {thread['name']}. {thread['summary']} Status: {thread['status']}."
        tags = ['session-summary', 'thread', session_tag, f"thread-{thread['status']}"]

        # Emotional weight: completed=0.6 (positive), blocked=0.3, in-progress=0.5
        emotion = {'completed': 0.65, 'blocked': 0.3, 'in-progress': 0.5}.get(thread['status'], 0.5)

        if dry_run:
            print(f"  [THREAD {i}] ({emotion:.2f}) {content[:120]}...")
            print(f"    Tags: {tags}")
        else:
            mid, name = store_memory(content, tags=tags, emotion=emotion,
                                     title=f"thread-{thread['name'][:20]}", event_time=session_date)
            stored_ids.append(mid)
            print(f"  Stored thread: {name} ({mid})")

    # Store lessons
    for i, lesson in enumerate(parsed.get('lessons', []), 1):
        content = f"[Session {session_date}] Lesson learned: {lesson}"
        tags = ['session-summary', 'lesson', session_tag, 'heuristic']
        emotion = 0.6  # Lessons are generally positive (growth)

        if dry_run:
            print(f"  [LESSON {i}] ({emotion:.2f}) {content[:120]}...")
        else:
            mid, name = store_memory(content, tags=tags, emotion=emotion,
                                     title=f"lesson-{i}", event_time=session_date)
            stored_ids.append(mid)
            print(f"  Stored lesson: {name} ({mid})")

    # Store facts
    for i, fact in enumerate(parsed.get('facts', []), 1):
        content = f"[Session {session_date}] Key fact: {fact}"
        tags = ['session-summary', 'key-fact', session_tag, 'procedural']
        emotion = 0.5  # Facts are neutral

        if dry_run:
            print(f"  [FACT {i}] ({emotion:.2f}) {content[:120]}...")
        else:
            mid, name = store_memory(content, tags=tags, emotion=emotion,
                                     title=f"fact-{i}", event_time=session_date)
            stored_ids.append(mid)
            print(f"  Stored fact: {name} ({mid})")

    # Link all stored memories via co-occurrence (they're from the same session)
    if not dry_run and len(stored_ids) > 1:
        try:
            from db_adapter import get_db
            db = get_db()
            pairs = 0
            with db._conn() as conn:
                with conn.cursor() as cur:
                    for j in range(len(stored_ids)):
                        for k in range(j + 1, len(stored_ids)):
                            for m1, m2 in [(stored_ids[j], stored_ids[k]),
                                           (stored_ids[k], stored_ids[j])]:
                                cur.execute(f"""
                                    INSERT INTO {db._table('co_occurrences')} (memory_id, other_id, count)
                                    VALUES (%s, %s, 1)
                                    ON CONFLICT (memory_id, other_id)
                                    DO UPDATE SET count = {db._table('co_occurrences')}.count + 1
                                """, (m1, m2))
                            pairs += 1
            print(f"  Linked {pairs} co-occurrence pairs across {len(stored_ids)} memories")
        except Exception as e:
            print(f"  Co-occurrence linking failed: {e}")

    return stored_ids


# ---------------------------------------------------------------------------
# Main: full pipeline
# ---------------------------------------------------------------------------

def run(transcript_path: str, max_chars: int = 10000, dry_run: bool = False) -> dict:
    """Full pipeline: extract -> summarize -> parse -> store."""
    print(f"Extracting transcript: {transcript_path}")
    session_text, meta = extract_session_text(transcript_path, max_chars=max_chars)
    print(f"  {meta['total_entries']} entries, {meta['total_chars']} chars -> {len(session_text)} condensed")

    # Detect session date from transcript filename or modification time
    fname = Path(transcript_path).stem
    try:
        mtime = os.path.getmtime(transcript_path)
        session_date = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime('%Y-%m-%d')
    except Exception:
        session_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    print(f"\nSummarizing ({os.environ.get('DRIFT_SUMMARY_MODEL', 'gpt-4o-mini')} -> Gemma fallback)...")
    start = time.time()
    raw_output, llm_meta = summarize_session(session_text)
    elapsed = time.time() - start
    print(f"  {llm_meta.get('model', '?')} responded in {elapsed:.1f}s")

    if not raw_output:
        print("  ERROR: No output from any LLM backend")
        return {'success': False, 'error': 'no LLM output'}

    print(f"\nParsing structured output...")
    parsed = parse_extraction(raw_output)
    print(f"  {len(parsed['threads'])} threads, {len(parsed['lessons'])} lessons, {len(parsed['facts'])} facts")

    if not any(parsed.values()):
        print("  WARNING: Parser found nothing — raw output may not match expected format")
        print(f"  Raw (first 500 chars): {raw_output[:500]}")
        return {'success': False, 'error': 'parse failed', 'raw': raw_output[:500]}

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Storing memories (session: {session_date})...")
    stored_ids = store_memories(parsed, session_date, dry_run=dry_run)

    result = {
        'success': True,
        'session_date': session_date,
        'threads': len(parsed['threads']),
        'lessons': len(parsed['lessons']),
        'facts': len(parsed['facts']),
        'stored': len(stored_ids),
        'llm': llm_meta,
        'elapsed': elapsed,
    }
    print(f"\nDone: {result['threads']} threads, {result['lessons']} lessons, {result['facts']} facts "
          f"-> {result['stored']} memories stored in {elapsed:.1f}s")
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Session Transcript Summarizer')
    parser.add_argument('transcript', help='Path to session JSONL transcript')
    parser.add_argument('--max-chars', type=int, default=10000)
    parser.add_argument('--dry-run', action='store_true', help='Preview without storing')
    args = parser.parse_args()

    run(args.transcript, max_chars=args.max_chars, dry_run=args.dry_run)
