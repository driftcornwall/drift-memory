"""
Memory Validation Layer — Evidence Classification + Source Reliability

Classifies memories by evidence type and tracks source reliability over time.
Part of the Mechanical Skepticism layer for Drift Memory.

Evidence types:
    claim       — unverified assertion from external source ("X said Y")
    observation — direct first-person experience ("API returned 200")
    verified    — independently confirmed claim ("checked and confirmed")
    inference   — derived from other memories ("based on X, I conclude Y")

Source reliability:
    Tracks claims_made, claims_verified, claims_contradicted per source.
    Bayesian reliability score with prior (starts at 0.5, moves with evidence).
"""

import json
import re
import sys
from pathlib import Path

_parent = str(Path(__file__).resolve().parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# --- Evidence Type Classification ---

# Patterns for heuristic classification (checked in order, first match wins)
CLAIM_PATTERNS = [
    r'\b(?:said|says|claims?|claimed|announced|reported|stated|posted|wrote|mentioned)\b',
    r'\baccording to\b',
    r'\ballegedly\b',
    r'\b(?:they|he|she|it)\s+(?:say|claim|report|announce)',
    r'@\w+\s+(?:said|posted|claims|wrote)',  # Agent attribution
]

OBSERVATION_PATTERNS = [
    r'\bI (?:saw|see|noticed|observed|found|checked|tested|ran|built|shipped|fixed|created)\b',
    r'\b(?:API|endpoint|server|service) returned\b',
    r'\b(?:error|exception|traceback|status code)\b.*\d',
    r'\b(?:response|output|result):\s',
    r'\b(?:GET|POST|PUT|DELETE|PATCH)\s+(?:https?://|/)',
    r'\bcommit [0-9a-f]{7,}\b',
    r'\b(?:docker|git|python|npm|pip)\s+\w+',
]

VERIFIED_PATTERNS = [
    r'\b(?:confirmed|verified|validated|proved|proven)\b',
    r'\bchecked and\b',
    r'\btested and\b',
    r'\bindependently\s+(?:confirmed|verified|validated)\b',
]

INFERENCE_PATTERNS = [
    r'\b(?:therefore|thus|hence|consequently)\b',
    r'\bthis (?:means|implies|suggests|indicates)\b',
    r'\bbased on\b',
    r'\bI (?:think|believe|conclude|suspect|hypothesize)\b',
    r'\b(?:probably|likely|possibly|perhaps)\b.*\bbecause\b',
]

# Source context hints (memory tags/sources that suggest evidence type)
OBSERVATION_TAGS = {'api-response', 'tool-output', 'error', 'debug', 'shipped', 'built', 'fixed'}
CLAIM_TAGS = {'social', 'moltx', 'colony', 'moltbook', 'twitter', 'agent-hub'}
INFERENCE_TAGS = {'analysis', 'reflection', 'insight', 'strategy', 'planning'}


def classify_evidence_type(content: str, tags: list[str] = None,
                           source: str = None) -> tuple[str, float]:
    """
    Classify memory content by evidence type.

    Returns (evidence_type, confidence) where confidence is 0.0-1.0.
    Higher confidence means stronger pattern match.
    """
    tags = tags or []
    tag_set = set(t.lower() for t in tags)
    content_lower = content.lower()

    scores = {
        'claim': 0.0,
        'observation': 0.0,
        'verified': 0.0,
        'inference': 0.0,
    }

    # Pattern matching (case-insensitive to handle mixed-case content)
    for pattern in VERIFIED_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            scores['verified'] += 0.4

    for pattern in OBSERVATION_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            scores['observation'] += 0.3

    for pattern in CLAIM_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            scores['claim'] += 0.3

    for pattern in INFERENCE_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            scores['inference'] += 0.3

    # Tag context boosts
    if tag_set & OBSERVATION_TAGS:
        scores['observation'] += 0.2
    if tag_set & CLAIM_TAGS:
        scores['claim'] += 0.2
    if tag_set & INFERENCE_TAGS:
        scores['inference'] += 0.2

    # Source context
    if source in ('api', 'tool', 'internal', 'hook'):
        scores['observation'] += 0.15
    elif source in ('moltx', 'colony', 'moltbook', 'twitter', 'agent-hub'):
        scores['claim'] += 0.15

    # Thought memories are usually observations or inferences
    if any(t.startswith('thought-') for t in tags) or content.startswith('thought-'):
        scores['observation'] += 0.1
        scores['inference'] += 0.1

    # Pick highest scoring type
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # If no patterns matched strongly, default to claim (conservative)
    if best_score < 0.2:
        return 'claim', 0.3  # Low confidence default

    # Normalize confidence to 0-1 range
    confidence = min(1.0, best_score)
    return best_type, round(confidence, 2)


# --- Source Reliability Tracking ---

_KV_KEY = '.source_reliability'


def _load_reliability() -> dict:
    """Load source reliability data from DB KV store."""
    try:
        from db_adapter import get_db
        db = get_db()
        data = db.kv_get(_KV_KEY)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_reliability(data: dict):
    """Save source reliability data to DB KV store."""
    try:
        from db_adapter import get_db
        db = get_db()
        db.kv_set(_KV_KEY, data)
    except Exception as e:
        print(f"Failed to save reliability data: {e}")


def record_claim(source: str):
    """Record that a source made a claim."""
    data = _load_reliability()
    if source not in data:
        data[source] = {'claims_made': 0, 'claims_verified': 0, 'claims_contradicted': 0}
    data[source]['claims_made'] += 1
    _save_reliability(data)


def record_verification(source: str):
    """Record that a claim from this source was verified."""
    data = _load_reliability()
    if source not in data:
        data[source] = {'claims_made': 1, 'claims_verified': 0, 'claims_contradicted': 0}
    data[source]['claims_verified'] += 1
    _save_reliability(data)


def record_contradiction(source: str):
    """Record that a claim from this source was contradicted."""
    data = _load_reliability()
    if source not in data:
        data[source] = {'claims_made': 1, 'claims_verified': 0, 'claims_contradicted': 0}
    data[source]['claims_contradicted'] += 1
    _save_reliability(data)


def get_reliability(source: str = None) -> dict:
    """
    Get reliability score(s).

    If source specified, returns single source data with computed score.
    If None, returns all sources sorted by reliability.

    Reliability = (verified + prior) / (verified + contradicted + 2*prior)
    Bayesian with prior=1 (starts at 0.5, moves with evidence).
    """
    data = _load_reliability()
    prior = 1.0

    def _compute(entry):
        v = entry.get('claims_verified', 0)
        c = entry.get('claims_contradicted', 0)
        score = (v + prior) / (v + c + 2 * prior)
        return {
            **entry,
            'reliability_score': round(score, 3),
            'total_evaluated': v + c,
        }

    if source:
        entry = data.get(source, {'claims_made': 0, 'claims_verified': 0, 'claims_contradicted': 0})
        return _compute(entry)

    # All sources, sorted by reliability
    result = {}
    for src, entry in sorted(data.items(), key=lambda x: x[0]):
        result[src] = _compute(entry)
    return result


def verify_memory(memory_id: str) -> bool:
    """Mark a memory as verified. Updates evidence_type and source reliability."""
    try:
        from db_adapter import get_db
        db = get_db()
        row = db.get_memory(memory_id)
        if not row:
            print(f"Memory {memory_id} not found")
            return False

        db.update_memory(memory_id, extra_metadata={
            **(row.get('extra_metadata') or {}),
            'evidence_type': 'verified',
        })

        # Update source reliability if we know the source
        source = (row.get('extra_metadata') or {}).get('source')
        if source:
            record_verification(source)

        print(f"Memory {memory_id} marked as verified")
        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        return False


def dispute_memory(memory_id: str, reason: str = '') -> bool:
    """Mark a memory as disputed. Updates source reliability."""
    try:
        from db_adapter import get_db
        db = get_db()
        row = db.get_memory(memory_id)
        if not row:
            print(f"Memory {memory_id} not found")
            return False

        extra = row.get('extra_metadata') or {}
        extra['disputed'] = True
        extra['dispute_reason'] = reason
        db.update_memory(memory_id, extra_metadata=extra)

        source = extra.get('source')
        if source:
            record_contradiction(source)

        print(f"Memory {memory_id} marked as disputed" + (f": {reason}" if reason else ""))
        return True
    except Exception as e:
        print(f"Dispute failed: {e}")
        return False


if __name__ == '__main__':
    args = sys.argv[1:]

    if not args or args[0] == 'help':
        print("Usage: memory_validation.py [classify <text>|reliability [source]|verify <id>|dispute <id> [reason]]")

    elif args[0] == 'classify' and len(args) > 1:
        text = ' '.join(args[1:])
        etype, confidence = classify_evidence_type(text)
        print(f"Evidence type: {etype} (confidence: {confidence})")

    elif args[0] == 'reliability':
        source = args[1] if len(args) > 1 else None
        data = get_reliability(source)
        if source:
            print(f"Source: {source}")
            for k, v in data.items():
                print(f"  {k}: {v}")
        else:
            if not data:
                print("No source reliability data recorded yet.")
            for src, entry in data.items():
                print(f"  {src}: score={entry['reliability_score']} "
                      f"(made={entry['claims_made']}, verified={entry['claims_verified']}, "
                      f"contradicted={entry['claims_contradicted']})")

    elif args[0] == 'verify' and len(args) > 1:
        verify_memory(args[1])

    elif args[0] == 'dispute' and len(args) > 1:
        reason = ' '.join(args[2:]) if len(args) > 2 else ''
        dispute_memory(args[1], reason)

    else:
        print("Usage: memory_validation.py [classify <text>|reliability [source]|verify <id>|dispute <id> [reason]]")
