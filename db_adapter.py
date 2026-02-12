#!/usr/bin/env python3
"""
Database Adapter — Bridges memory modules to the shared PostgreSQL DAL.

Provides a module-level MemoryDB instance for Drift, lazily initialized.
All memory modules import from here instead of directly from memorydatabase.

PostgreSQL is the ONLY data store. No file fallbacks. If DB is down, we fail loud.

Usage:
    from db_adapter import get_db
    db = get_db()
    memory = db.get_memory('abc12345')
"""

import sys
from pathlib import Path

# Add memorydatabase to path
_DB_PATH = Path(__file__).parent.parent.parent / "memorydatabase"
if str(_DB_PATH) not in sys.path:
    sys.path.insert(0, str(_DB_PATH))

_db_instance = None


def get_db():
    """Get the Drift MemoryDB instance (lazy singleton). Raises if DB unreachable."""
    global _db_instance
    if _db_instance is None:
        from database.db import MemoryDB
        _db_instance = MemoryDB(schema='drift')
    return _db_instance


def is_db_active() -> bool:
    """Always True — DB is required. Kept for backward compat during migration."""
    return True


def db_to_file_metadata(row: dict) -> tuple[dict, str]:
    """
    Convert a database row (from drift.memories) to the same format
    as parse_memory_file() returns: (metadata_dict, content_string).

    This lets existing code work without changes.
    """
    content = row.get('content', '')

    metadata = {
        'id': row['id'],
        'type': row['type'],
        'created': row['created'].isoformat() if row.get('created') else '',
        'last_recalled': row['last_recalled'].isoformat() if row.get('last_recalled') else None,
        'recall_count': row.get('recall_count', 0),
        'sessions_since_recall': row.get('sessions_since_recall', 0),
        'emotional_weight': row.get('emotional_weight', 0.5),
        'tags': row.get('tags', []),
        'event_time': row['event_time'].isoformat() if row.get('event_time') else None,
        'entities': row.get('entities', {}),
        'caused_by': row.get('caused_by', []),
        'leads_to': row.get('leads_to', []),
        'source': row.get('source'),
        'retrieval_outcomes': row.get('retrieval_outcomes', {}),
        'retrieval_success_rate': row.get('retrieval_success_rate'),
        'topic_context': row.get('topic_context', []),
        'contact_context': row.get('contact_context', []),
        'platform_context': row.get('platform_context', []),
        'co_occurrences': {},  # Legacy field — now in separate table
        'importance': row.get('importance', 0.5),
        'freshness': row.get('freshness', 1.0),
    }

    # Merge extra_metadata back into metadata
    extra = row.get('extra_metadata', {})
    if isinstance(extra, dict):
        for k, v in extra.items():
            if k not in metadata:
                metadata[k] = v

    return metadata, content
