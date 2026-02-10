#!/usr/bin/env python3
"""
Session State Management — Extracted from memory_manager.py (Phase 1)

Owns session-scoped state: which memories were retrieved during this session.
File-backed persistence so state survives across Python invocations within
the same session (SESSION_TIMEOUT_HOURS window).

Handles deferred processing from previous sessions:
- Pending co-occurrence calculation
- Pending semantic indexing
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

MEMORY_ROOT = Path(__file__).parent
SESSION_TIMEOUT_HOURS = 4

# Module-level state
_session_retrieved: set[str] = set()
_recalls_by_source: dict[str, list[str]] = {}
_session_loaded: bool = False

# Valid recall sources
RECALL_SOURCES = ("manual", "start_priming", "thought_priming", "prompt_priming")


def _start_db_session() -> None:
    """Start a DB session. Raises on failure."""
    global _db_session_id
    from db_adapter import get_db
    _db_session_id = get_db().start_session()

_db_session_id = None


def get_db_session_id():
    """Get the current DB session ID (or None if not using DB)."""
    return _db_session_id


def load() -> None:
    """Load session state from file. Idempotent — only loads once per process."""
    global _session_retrieved, _recalls_by_source, _session_loaded

    if _session_loaded:
        return

    _session_loaded = True

    # Start a DB session
    _start_db_session()

    # v2.16: Process pending co-occurrences from previous session (deferred)
    # v3.0: Check DB KV store instead of file
    try:
        from db_adapter import get_db as _get_db_for_pending
        pending = _get_db_for_pending().kv_get('.pending_cooccurrence')
        if pending:
            from co_occurrence import process_pending_cooccurrence
            process_pending_cooccurrence()
    except Exception as e:
        print(f"Warning: Could not process pending co-occurrences: {e}")

    # v2.16: Process pending semantic indexing from previous session
    pending_index_file = MEMORY_ROOT / ".pending_index"
    if pending_index_file.exists():
        try:
            semantic_search = MEMORY_ROOT / "semantic_search.py"
            if semantic_search.exists():
                print("Indexing new memories from previous session...")
                result = subprocess.run(
                    ["python", str(semantic_search), "index"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(MEMORY_ROOT)
                )
                if result.returncode == 0:
                    print("Semantic index updated.")
            pending_index_file.unlink(missing_ok=True)
        except Exception as e:
            print(f"Warning: Could not process pending index: {e}")

    # Load session state from DB KV store
    try:
        from db_adapter import get_db as _get_db_for_session
        raw = _get_db_for_session().kv_get('.session_state')
        if not raw:
            _session_retrieved = set()
            return

        data = json.loads(raw) if isinstance(raw, str) else raw
        session_start = datetime.fromisoformat(data.get('started', '2000-01-01'))

        hours_old = (datetime.now(timezone.utc) - session_start).total_seconds() / 3600
        if hours_old > SESSION_TIMEOUT_HOURS:
            _session_retrieved = set()
            _get_db_for_session().kv_set('.session_state', None)
        else:
            _session_retrieved = set(data.get('retrieved', []))
            _recalls_by_source = data.get('recalls_by_source', {})
    except (json.JSONDecodeError, KeyError, ValueError):
        _session_retrieved = set()
        _recalls_by_source = {}


def save() -> None:
    """Save session state to DB KV store."""
    from db_adapter import get_db as _get_db_for_save
    db = _get_db_for_save()

    started = datetime.now(timezone.utc).isoformat()
    existing = db.kv_get('.session_state')
    if existing:
        prev = json.loads(existing) if isinstance(existing, str) else existing
        started = prev.get('started', started)

    data = {
        'started': started,
        'retrieved': list(_session_retrieved),
        'recalls_by_source': _recalls_by_source,
        'last_updated': datetime.now(timezone.utc).isoformat()
    }
    db.kv_set('.session_state', data)


def get_retrieved() -> set[str]:
    """Get set of memory IDs retrieved this session (copy)."""
    load()
    return _session_retrieved.copy()


def add_retrieved(memory_id: str, source: str = "manual") -> None:
    """Add a memory ID to the session's retrieved set with source tracking."""
    load()
    _session_retrieved.add(memory_id)
    if source in RECALL_SOURCES:
        if source not in _recalls_by_source:
            _recalls_by_source[source] = []
        if memory_id not in _recalls_by_source[source]:
            _recalls_by_source[source].append(memory_id)


def get_retrieved_list() -> list[str]:
    """Get retrieved memory IDs as a list (for co-occurrence processing)."""
    load()
    return list(_session_retrieved)


def get_recalls_by_source() -> dict[str, list[str]]:
    """Get recall counts keyed by source."""
    load()
    return {s: list(ids) for s, ids in _recalls_by_source.items()}


def clear() -> None:
    """Clear session state."""
    global _session_retrieved, _session_loaded, _recalls_by_source, _db_session_id
    _session_retrieved = set()
    _recalls_by_source = {}
    _session_loaded = False

    # Clear DB KV
    from db_adapter import get_db
    db = get_db()
    db.kv_set('.session_state', None)

    # End DB session
    if _db_session_id is not None:
        db.end_session(_db_session_id)
        _db_session_id = None
