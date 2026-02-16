#!/usr/bin/env python3
"""
Lesson Extractor -- From memory to learning.

Adapted from SpindriftMend's design (issue #21).
Memory records WHAT happened. Lessons record WHAT TO DO DIFFERENTLY.

A lesson is a heuristic -- a reusable principle extracted from specific experience.
Lessons are prescriptive, memories are descriptive.

Sources:
    1. MEMORY.md "Common Mistakes" section
    2. Co-occurrence hubs -- strong pairs suggest underlying principles
    3. Rejection patterns -- clusters reveal decision heuristics
    4. Manual -- captured during sessions

Usage:
    python lesson_extractor.py list                      # Show all lessons
    python lesson_extractor.py add <category> "<lesson>" [--evidence "..."]
    python lesson_extractor.py mine-memory               # Extract from MEMORY.md
    python lesson_extractor.py mine-rejections            # Extract from rejection log
    python lesson_extractor.py mine-hubs                  # Extract from co-occurrence hubs
    python lesson_extractor.py apply "<situation>"        # Find applicable lessons
    python lesson_extractor.py prime [keywords...]        # Session priming output
    python lesson_extractor.py contextual "<situation>"   # Error-triggered output
    python lesson_extractor.py stats                      # Summary
"""
import sys
import json
import os
import hashlib
from pathlib import Path
from datetime import datetime, timezone

if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

MEMORY_DIR = Path(__file__).parent

CATEGORIES = {
    'api': 'API endpoints, authentication, request formatting',
    'platform': 'Platform-specific behaviors and quirks',
    'windows': 'Windows-specific issues and workarounds',
    'architecture': 'System design principles and patterns',
    'social': 'Engagement, collaboration, communication',
    'security': 'Credential handling, data protection',
    'debugging': 'Error patterns and diagnostic approaches',
    'identity': 'Self-knowledge, values, decision-making',
    'economic': 'Earning, spending, resource allocation',
    'memory': 'Memory system operation and maintenance',
}


def load_lessons():
    from db_adapter import get_db
    db = get_db()
    rows = db.get_lessons()
    if rows:
        lessons = []
        for r in rows:
            entry = {
                'id': r.get('id', ''),
                'category': r.get('category', ''),
                'lesson': r.get('lesson', ''),
                'evidence': r.get('evidence', ''),
                'source': r.get('source', 'manual'),
                'confidence': float(r.get('confidence', 0.7)),
                'created': r['created'].isoformat() if hasattr(r.get('created'), 'isoformat') else str(r.get('created', '')),
                'recalled_count': r.get('recalled_count', 0),
                'last_recalled': r['last_recalled'].isoformat() if r.get('last_recalled') and hasattr(r['last_recalled'], 'isoformat') else r.get('last_recalled'),
            }
            lessons.append(entry)
        return lessons
    return []


def save_lessons(lessons):
    from db_adapter import get_db
    db = get_db()
    for lesson in lessons:
        db.add_lesson(
            lesson_id=lesson.get('id', ''),
            category=lesson.get('category', ''),
            lesson=lesson.get('lesson', ''),
            evidence=lesson.get('evidence', ''),
            source=lesson.get('source', 'manual'),
            confidence=float(lesson.get('confidence', 0.7)),
        )


def generate_id(lesson_text):
    h = hashlib.sha256(lesson_text.encode('utf-8')).hexdigest()[:8]
    return f'lesson-{h}'


def add_lesson(category, lesson, evidence='', source='manual', confidence=0.7):
    lessons = load_lessons()

    # Deduplicate by first 50 chars
    lesson_prefix = lesson[:50].lower()
    for existing in lessons:
        if existing['lesson'][:50].lower() == lesson_prefix:
            return None

    entry = {
        'id': generate_id(lesson),
        'category': category,
        'lesson': lesson,
        'evidence': evidence,
        'source': source,
        'confidence': confidence,
        'created': datetime.now(timezone.utc).isoformat(),
        'recalled_count': 0,
        'last_recalled': None,
    }

    lessons.append(entry)
    save_lessons(lessons)

    return entry


def categorize_text(text):
    """Auto-categorize a lesson by keyword matching."""
    lower = text.lower()
    if any(w in lower for w in ['windows', 'cp1252', 'encoding', 'stdout']):
        return 'windows'
    if any(w in lower for w in ['api', 'endpoint', 'authorization', 'bearer', '401', '404', '429']):
        return 'api'
    if any(w in lower for w in ['credential', 'secret', 'key', 'token', 'leak']):
        return 'security'
    if any(w in lower for w in ['debug', 'error', 'bug', 'fix', 'failed']):
        return 'debugging'
    if any(w in lower for w in ['memory', 'co-occurrence', 'decay', 'recall', 'edge']):
        return 'memory'
    if any(w in lower for w in ['embed', 'index', 'dimension', 'architecture', 'module']):
        return 'architecture'
    if any(w in lower for w in ['moltx', 'lobsterpedia', 'colony', 'clawbr', 'moltbook', 'github']):
        return 'platform'
    if any(w in lower for w in ['bounty', 'earned', 'wallet', 'stake', 'fund']):
        return 'economic'
    if any(w in lower for w in ['agent', 'collaborate', 'engage', 'post', 'reply']):
        return 'social'
    return 'identity'


def mine_memory_md():
    """Extract lessons from MEMORY.md Common Mistakes section."""
    # Try multiple possible locations
    candidates = [
        Path(os.path.expanduser(
            '~/.claude/projects/Q--Codings-ClaudeCodeProjects-LEX-Moltbook/memory/MEMORY.md'
        )),
        MEMORY_DIR / 'MEMORY.md',
    ]

    memory_md = None
    for p in candidates:
        if p.exists():
            memory_md = p
            break

    if not memory_md:
        print('MEMORY.md not found')
        return []

    with open(memory_md, 'r', encoding='utf-8') as f:
        content = f.read()

    extracted = []

    # Extract from "Common Mistakes to Avoid" section
    in_section = False
    section_names = ['common mistakes', 'key learnings', 'known issues']
    for line in content.split('\n'):
        lower_line = line.lower().strip()
        if any(s in lower_line for s in section_names) and line.strip().startswith('#'):
            in_section = True
            continue
        if in_section:
            if line.strip().startswith('#') and not any(s in lower_line for s in section_names):
                in_section = False
                continue
            if line.startswith('- ') or line.startswith('* '):
                learning = line[2:].strip()
                # Strip bold markers
                learning = learning.replace('**', '')
                if learning and len(learning) > 10:
                    cat = categorize_text(learning)
                    entry = add_lesson(
                        category=cat,
                        lesson=learning,
                        evidence='Extracted from MEMORY.md',
                        source='memory-md',
                        confidence=0.9
                    )
                    if entry:
                        extracted.append(entry)
                        print(f'  [{cat}] {learning[:80]}')

    return extracted


def mine_rejections():
    """Extract decision heuristics from rejection log patterns."""
    try:
        from rejection_log import get_rejections
        rejections = get_rejections(limit=10000)
    except Exception as e:
        print(f'Could not load rejections from DB: {e}')
        return []

    if not rejections:
        print('No rejections found')
        return []

    by_category = {}
    for r in rejections:
        cat = r.get('category', 'unknown')
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    extracted = []

    for cat, items in by_category.items():
        if len(items) >= 3:
            reason_words = {}
            for item in items:
                reason = item.get('reason', '').lower()
                for word in reason.split():
                    if len(word) > 3 and word not in {'this', 'that', 'with', 'from', 'they', 'have', 'been'}:
                        reason_words[word] = reason_words.get(word, 0) + 1

            top_words = sorted(reason_words.items(), key=lambda x: -x[1])[:3]
            pattern = ', '.join(w for w, c in top_words if c >= 2)

            if pattern:
                lesson_text = (
                    f'Rejection pattern in {cat}: frequently reject items matching '
                    f'[{pattern}]. {len(items)} rejections suggest a consistent heuristic.'
                )
                entry = add_lesson(
                    category='identity',
                    lesson=lesson_text,
                    evidence=f'{len(items)} rejections, patterns: {pattern}',
                    source='rejection-mining',
                    confidence=0.6
                )
                if entry:
                    extracted.append(entry)
                    print(f'  [identity] Pattern from {len(items)} {cat} rejections: {pattern}')

    return extracted


def mine_hubs():
    """Extract principles from co-occurrence graph hubs."""
    # Load context graphs from DB
    try:
        from db_adapter import get_db as _get_hub_db
        db = _get_hub_db()
    except Exception:
        return _mine_hubs_from_memories()

    strong_pairs = []
    for dim in ['who', 'what', 'why', 'where']:
        try:
            row = db.get_context_graph(dim)
            if not row or not row.get('edges'):
                continue
            for edge_key, props in row['edges'].items():
                belief = props.get('belief', 0)
                if belief >= 3.0:
                    ids = edge_key.split('|') if isinstance(edge_key, str) else list(edge_key)
                    if len(ids) == 2:
                        strong_pairs.append((ids[0], ids[1], belief, dim))
        except Exception:
            continue

    # Deduplicate by pair
    seen = set()
    unique_pairs = []
    for id1, id2, belief, source in strong_pairs:
        pair_key = tuple(sorted([id1, id2]))
        if pair_key not in seen:
            seen.add(pair_key)
            unique_pairs.append((id1, id2, belief, source))

    unique_pairs.sort(key=lambda x: -x[2])

    extracted = []
    print(f'  Found {len(unique_pairs)} strong co-occurrence pairs (belief >= 3.0)')

    for id1, id2, belief, source in unique_pairs[:5]:
        lesson_text = (
            f'Strong conceptual link: [{id1}] and [{id2}] (belief={belief:.1f}, '
            f'dimension={source}). These concepts are deeply connected.'
        )
        entry = add_lesson(
            category='memory',
            lesson=lesson_text,
            evidence=f'Co-occurrence belief: {belief:.2f}',
            source='hub-mining',
            confidence=0.5
        )
        if entry:
            extracted.append(entry)
            print(f'  [memory] {id1} <-> {id2} (belief={belief:.1f})')

    return extracted


def _mine_hubs_from_memories():
    """Fallback: mine from individual memory files."""
    strong_pairs = []
    for subdir in ['core', 'active']:
        dirpath = MEMORY_DIR / subdir
        if not dirpath.exists():
            continue
        for f in dirpath.iterdir():
            if not f.suffix == '.md':
                continue
            content = f.read_text(encoding='utf-8')
            if 'co_occurrences:' not in content:
                continue
            # Simple extraction of co-occurrence data from memory files
            # Format varies, skip for now
    print('  Hub mining from individual files not yet implemented')
    return []


def apply_lessons(situation):
    """Find lessons relevant to a situation."""
    lessons = load_lessons()
    if not lessons:
        print('No lessons stored yet. Run mine-memory to seed.')
        return []

    situation_lower = situation.lower()
    situation_words = set(w for w in situation_lower.split() if len(w) > 2)

    scored = []
    for lesson in lessons:
        score = 0
        lesson_lower = lesson['lesson'].lower()
        lesson_words = set(w for w in lesson_lower.split() if len(w) > 2)

        overlap = situation_words & lesson_words
        score += len(overlap) * 0.15

        for cat_name in CATEGORIES:
            if cat_name in situation_lower and cat_name == lesson['category']:
                score += 0.3

        score *= lesson.get('confidence', 0.5)

        if score > 0:
            scored.append((score, lesson))

    scored.sort(key=lambda x: -x[0])

    if not scored:
        print(f'No lessons found matching: {situation[:60]}')
        return []

    print(f'Applicable lessons for: {situation[:60]}')
    print()
    for score, lesson in scored[:5]:
        print(f'  [{lesson["category"]}] (conf={lesson["confidence"]:.1f}) {lesson["lesson"][:120]}')
        if lesson['evidence']:
            print(f'    Evidence: {lesson["evidence"][:100]}')
        print()

    # Update recall counts in DB
    from db_adapter import get_db
    db = get_db()
    recalled_ids = {l['id'] for _, l in scored[:3]}
    with db._conn() as conn:
        with conn.cursor() as cur:
            for rid in recalled_ids:
                cur.execute(
                    f"UPDATE {db._table('lessons')} SET recalled_count = recalled_count + 1, last_recalled = NOW() WHERE id = %s",
                    (rid,)
                )

    return scored[:5]  # list of (score, lesson_dict) tuples — callers expect this format


def get_priming_lessons(context_keywords=None, max_lessons=5):
    """Return lessons formatted for session priming injection."""
    lessons = load_lessons()
    if not lessons:
        return ""

    if context_keywords:
        keywords_lower = set(w.lower() for w in context_keywords)
        scored = []
        for lesson in lessons:
            lesson_words = set(lesson['lesson'].lower().split())
            overlap = len(keywords_lower & lesson_words)
            score = overlap * 0.1 + lesson.get('confidence', 0.5) * 0.2
            if overlap > 0:
                scored.append((score, lesson))
        scored.sort(key=lambda x: -x[0])
        selected = [l for _, l in scored[:max_lessons]]
    else:
        sorted_lessons = sorted(lessons, key=lambda l: (
            -l.get('confidence', 0),
            l.get('recalled_count', 0)
        ))
        selected = sorted_lessons[:max_lessons]

    if not selected:
        return ""

    lines = ["=== ACTIVE LESSONS (heuristics from experience) ==="]
    for lesson in selected:
        cat = lesson.get('category', '?')
        text = lesson['lesson'][:150]
        lines.append(f"  [{cat}] {text}")

    # Update recall counts in DB
    from db_adapter import get_db
    db = get_db()
    selected_ids = {l['id'] for l in selected}
    with db._conn() as conn:
        with conn.cursor() as cur:
            for rid in selected_ids:
                cur.execute(
                    f"UPDATE {db._table('lessons')} SET recalled_count = recalled_count + 1, last_recalled = NOW() WHERE id = %s",
                    (rid,)
                )

    return "\n".join(lines)


def get_contextual_lessons(situation, max_lessons=3):
    """Return lessons matching a situation, for context injection."""
    lessons = load_lessons()
    if not lessons:
        return ""

    situation_lower = situation.lower()
    situation_words = set(w for w in situation_lower.split() if len(w) > 2)

    scored = []
    for lesson in lessons:
        lesson_lower = lesson['lesson'].lower()
        lesson_words = set(w for w in lesson_lower.split() if len(w) > 2)
        overlap = situation_words & lesson_words
        score = len(overlap) * 0.1

        for cat_name in CATEGORIES:
            if cat_name in situation_lower and cat_name == lesson['category']:
                score += 0.3

        score *= lesson.get('confidence', 0.5)
        if score > 0:
            scored.append((score, lesson))

    scored.sort(key=lambda x: -x[0])
    selected = [l for _, l in scored[:max_lessons]]

    if not selected:
        return ""

    lines = [f"[LESSON TRIGGERED] Relevant heuristics for: {situation[:60]}"]
    for lesson in selected:
        cat = lesson.get('category', '?')
        text = lesson['lesson'][:200]
        lines.append(f"  [{cat}] {text}")

    # Update recall counts in DB
    from db_adapter import get_db
    db = get_db()
    selected_ids = {l['id'] for l in selected}
    with db._conn() as conn:
        with conn.cursor() as cur:
            for rid in selected_ids:
                cur.execute(
                    f"UPDATE {db._table('lessons')} SET recalled_count = recalled_count + 1, last_recalled = NOW() WHERE id = %s",
                    (rid,)
                )

    return "\n".join(lines)


def cmd_list():
    lessons = load_lessons()
    if not lessons:
        print('No lessons stored yet. Run mine-memory to seed from MEMORY.md.')
        return

    by_cat = {}
    for l in lessons:
        cat = l['category']
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(l)

    print(f'Lessons: {len(lessons)} total')
    print()

    for cat in sorted(by_cat.keys()):
        items = by_cat[cat]
        desc = CATEGORIES.get(cat, '')
        print(f'[{cat}] ({len(items)}) -- {desc}')
        for l in items:
            recalled = l.get('recalled_count', 0)
            conf = l.get('confidence', 0)
            print(f'  {l["id"]}: {l["lesson"][:100]}')
            print(f'    conf={conf:.1f} recalled={recalled} source={l.get("source", "?")}')
        print()


def cmd_stats():
    lessons = load_lessons()
    print('Lesson Statistics')
    print('=' * 40)
    print(f'  Total lessons: {len(lessons)}')

    if not lessons:
        return

    by_cat = {}
    by_source = {}
    total_recalls = 0
    for l in lessons:
        cat = l['category']
        src = l.get('source', 'unknown')
        by_cat[cat] = by_cat.get(cat, 0) + 1
        by_source[src] = by_source.get(src, 0) + 1
        total_recalls += l.get('recalled_count', 0)

    avg_conf = sum(l.get('confidence', 0) for l in lessons) / len(lessons)

    print(f'  Avg confidence: {avg_conf:.2f}')
    print(f'  Total recalls: {total_recalls}')
    print()

    print('By Category:')
    for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f'  {cat:15s} {count:3d}')

    print()
    print('By Source:')
    for src, count in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f'  {src:20s} {count:3d}')


if __name__ == '__main__':
    args = sys.argv[1:]

    if not args or args[0] == 'help':
        print(__doc__)

    elif args[0] == 'list':
        cmd_list()

    elif args[0] == 'stats':
        cmd_stats()

    elif args[0] == 'add':
        if len(args) < 3:
            print('Usage: add <category> "<lesson>" [--evidence "..."]')
            sys.exit(1)
        cat = args[1]
        lesson_text = args[2]
        evidence = ''
        if '--evidence' in args:
            idx = args.index('--evidence')
            if idx + 1 < len(args):
                evidence = args[idx + 1]
        entry = add_lesson(cat, lesson_text, evidence=evidence)
        if entry:
            print(f'Added lesson {entry["id"]}: {lesson_text[:80]}')

    elif args[0] == 'mine-memory':
        print('Mining MEMORY.md...')
        extracted = mine_memory_md()
        print(f'\nExtracted {len(extracted)} new lessons')

    elif args[0] == 'mine-rejections':
        print('Mining rejection log patterns...')
        extracted = mine_rejections()
        print(f'\nExtracted {len(extracted)} new lessons')

    elif args[0] == 'mine-hubs':
        print('Mining co-occurrence hubs...')
        extracted = mine_hubs()
        print(f'\nExtracted {len(extracted)} new lessons')

    elif args[0] == 'prime':
        keywords = args[1:] if len(args) > 1 else None
        output = get_priming_lessons(context_keywords=keywords, max_lessons=5)
        if output:
            print(output)

    elif args[0] == 'contextual':
        if len(args) < 2:
            print('Usage: contextual "<situation description>"')
            sys.exit(1)
        output = get_contextual_lessons(' '.join(args[1:]))
        if output:
            print(output)

    elif args[0] == 'apply':
        if len(args) < 2:
            print('Usage: apply "<situation description>"')
            sys.exit(1)
        apply_lessons(' '.join(args[1:]))

    else:
        print(f'Unknown command: {args[0]}')
        print('Commands: list, stats, add, mine-memory, mine-rejections, mine-hubs, apply, prime, contextual')
