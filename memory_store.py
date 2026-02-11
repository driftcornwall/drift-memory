#!/usr/bin/env python3
"""
Memory Store — Write operations for creating and linking memories.

Extracted from memory_manager.py (Phase 3).
All writes go directly to PostgreSQL. No file system. No fallbacks.
"""

import hashlib
import random
import string
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from db_adapter import get_db, db_to_file_metadata
from entity_detection import detect_entities, detect_event_time
import session_state


def generate_id() -> str:
    """Generate a short, readable memory ID."""
    return hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:8]


def calculate_emotional_weight(
    surprise: float = 0.0,
    goal_relevance: float = 0.0,
    social_significance: float = 0.0,
    utility: float = 0.0
) -> float:
    """
    Calculate emotional weight from factors (0-1 each).

    - surprise: contradicted my model (high = sticky)
    - goal_relevance: connected to self-sustainability, collaboration
    - social_significance: interactions with respected agents
    - utility: proved useful when recalled later
    """
    weights = [0.2, 0.35, 0.2, 0.25]  # goal_relevance weighted highest
    factors = [surprise, goal_relevance, social_significance, utility]
    return sum(w * f for w, f in zip(weights, factors))


def create_memory(
    content: str,
    tags: list[str],
    memory_type: str = "active",
    emotional_factors: Optional[dict] = None,
    links: Optional[list[str]] = None
) -> str:
    """
    Create a new memory with proper metadata. DB-only.

    Args:
        content: The memory content (markdown)
        tags: Keywords for associative retrieval
        memory_type: "core", "active", or "archive"
        emotional_factors: Dict with surprise, goal_relevance, social_significance, utility
        links: List of other memory IDs this links to

    Returns:
        The memory ID
    """
    memory_id = generate_id()

    emotional_factors = emotional_factors or {}
    emotional_weight = calculate_emotional_weight(**emotional_factors)

    db = get_db()
    db.insert_memory(
        memory_id=memory_id,
        type_=memory_type,
        content=content,
        tags=tags,
        emotional_weight=round(emotional_weight, 3),
        extra_metadata={'links': links or []},
    )

    print(f"Created memory: {memory_id} (type={memory_type})")
    return memory_id


def store_memory(
    content: str,
    tags: list[str] = None,
    emotion: float = 0.5,
    title: str = None,
    caused_by: list[str] = None,
    event_time: str = None
) -> str:
    """
    Store a new memory to PostgreSQL. DB-only, no file writes.

    Args:
        content: The memory content
        tags: Keywords for retrieval
        emotion: Emotional weight (0-1)
        title: Optional title for filename
        caused_by: List of memory IDs that caused/led to this memory (CAUSAL EDGES)
        event_time: When the event happened (ISO format). Defaults to now. (BI-TEMPORAL v2.10)

    Returns:
        Tuple of (memory_id, display_name)
    """
    memory_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    if title:
        slug = title.lower().replace(' ', '-')[:30]
    else:
        slug = content.split()[:4]
        slug = '-'.join(slug).lower()[:30]
    slug = ''.join(c for c in slug if c.isalnum() or c == '-')
    display_name = f"{slug}-{memory_id}"

    caused_by = caused_by or []
    auto_causal = session_state.get_retrieved_list()
    all_causal = list(set(caused_by + auto_causal))

    tags = tags or []
    now = datetime.now(timezone.utc)
    created = now.isoformat()

    if event_time:
        event = event_time
    else:
        detected = detect_event_time(content)
        event = detected if detected else created

    detected_entities = detect_entities(content, tags)

    db = get_db()
    db.insert_memory(
        memory_id=memory_id,
        type_='active',
        content=content,
        tags=tags,
        entities=detected_entities or {},
        emotional_weight=emotion,
        extra_metadata={
            'caused_by': all_causal,
            'leads_to': [],
            'event_time': event,
        },
        created=now,
    )

    # Update leads_to links on cause memories
    for cause_id in all_causal:
        _add_leads_to_link(cause_id, memory_id)

    # Auto-embed for semantic search
    try:
        from vocabulary_bridge import bridge_content
        from semantic_search import embed_single
        bridged = bridge_content(content)
        embed_single(memory_id, bridged)
    except Exception:
        pass  # Embedding failure shouldn't block store

    # Auto-classify WHAT topics via Gemma (background thread — ~6s inference)
    try:
        from gemma_bridge import _ollama_available
        if _ollama_available():
            import threading
            def _classify_bg(mid, text):
                try:
                    from gemma_bridge import classify_topics
                    from db_adapter import get_db as _get_db
                    topics = classify_topics(text)
                    if topics:
                        _get_db().update_memory(mid, topic_context=topics)
                except Exception:
                    pass
            threading.Thread(target=_classify_bg, args=(memory_id, content), daemon=True).start()
    except Exception:
        pass  # Classification failure shouldn't block store

    return memory_id, display_name


def _add_leads_to_link(source_id: str, target_id: str) -> bool:
    """Add a leads_to link from source memory to target memory. DB-only."""
    db = get_db()
    row = db.get_memory(source_id)
    if not row:
        return False

    extra = row.get('extra_metadata', {}) or {}
    leads_to = extra.get('leads_to', [])
    if target_id not in leads_to:
        leads_to.append(target_id)
        extra['leads_to'] = leads_to
        db.update_memory(source_id, extra_metadata=extra)
        return True
    return False


def find_causal_chain(memory_id: str, direction: str = "both", max_depth: int = 5) -> dict:
    """
    Trace the causal chain from a memory. DB-only.

    Args:
        memory_id: Starting memory
        direction: "causes" (what this led to), "effects" (what caused this), or "both"
        max_depth: Maximum chain depth to traverse

    Returns:
        Dict with 'causes' (upstream) and 'effects' (downstream) chains
    """
    result = {"causes": [], "effects": [], "root": memory_id}
    db = get_db()

    def get_memory_meta(mid: str) -> Optional[dict]:
        row = db.get_memory(mid)
        if row:
            meta, _ = db_to_file_metadata(row)
            return meta
        return None

    def trace_causes(mid: str, depth: int, visited: set) -> list:
        if depth > max_depth or mid in visited:
            return []
        visited.add(mid)

        meta = get_memory_meta(mid)
        if not meta:
            return []

        caused_by = meta.get('caused_by', [])
        chain = []
        for cause_id in caused_by:
            chain.append({"id": cause_id, "depth": depth})
            chain.extend(trace_causes(cause_id, depth + 1, visited))
        return chain

    def trace_effects(mid: str, depth: int, visited: set) -> list:
        if depth > max_depth or mid in visited:
            return []
        visited.add(mid)

        meta = get_memory_meta(mid)
        if not meta:
            return []

        leads_to = meta.get('leads_to', [])
        chain = []
        for effect_id in leads_to:
            chain.append({"id": effect_id, "depth": depth})
            chain.extend(trace_effects(effect_id, depth + 1, visited))
        return chain

    if direction in ["causes", "both"]:
        result["causes"] = trace_causes(memory_id, 1, set())

    if direction in ["effects", "both"]:
        result["effects"] = trace_effects(memory_id, 1, set())

    return result
