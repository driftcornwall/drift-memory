#!/usr/bin/env python3
"""
Memory Store — Write operations for creating and linking memories.

Extracted from memory_manager.py (Phase 3).
Functions here create new memory files or modify existing ones
to establish causal links.
"""

import hashlib
import random
import string
import uuid
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from memory_common import (
    CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR, ALL_DIRS,
    parse_memory_file, write_memory_file,
)
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
    Create a new memory with proper metadata.

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
    now = datetime.now(timezone.utc).isoformat()

    emotional_factors = emotional_factors or {}
    emotional_weight = calculate_emotional_weight(**emotional_factors)

    metadata = {
        'id': memory_id,
        'created': now,
        'last_recalled': now,
        'recall_count': 1,
        'emotional_weight': round(emotional_weight, 3),
        'tags': tags,
        'links': links or [],
        'sessions_since_recall': 0
    }

    if memory_type == "core":
        target_dir = CORE_DIR
    elif memory_type == "archive":
        target_dir = ARCHIVE_DIR
    else:
        target_dir = ACTIVE_DIR

    target_dir.mkdir(parents=True, exist_ok=True)

    safe_tag = tags[0].replace(' ', '-').lower() if tags else 'memory'
    filename = f"{safe_tag}-{memory_id}.md"
    filepath = target_dir / filename

    write_memory_file(filepath, metadata, content)
    print(f"Created memory: {filepath}")
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
    Store a new memory to the active directory.

    Args:
        content: The memory content
        tags: Keywords for retrieval
        emotion: Emotional weight (0-1)
        title: Optional title for filename
        caused_by: List of memory IDs that caused/led to this memory (CAUSAL EDGES)
        event_time: When the event happened (ISO format). Defaults to now. (BI-TEMPORAL v2.10)
                    Distinct from 'created' which is ingestion time.

    Returns:
        Tuple of (memory_id, filename)
    """
    memory_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    if title:
        slug = title.lower().replace(' ', '-')[:30]
    else:
        slug = content.split()[:4]
        slug = '-'.join(slug).lower()[:30]
    slug = ''.join(c for c in slug if c.isalnum() or c == '-')

    filename = f"{slug}-{memory_id}.md"
    filepath = ACTIVE_DIR / filename

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

    metadata = {
        'id': memory_id,
        'type': 'active',
        'created': created,
        'event_time': event,
        'tags': tags,
        'emotional_weight': emotion,
        'recall_count': 0,
        'co_occurrences': {},
        'caused_by': all_causal,
        'leads_to': []
    }

    if detected_entities:
        metadata['entities'] = detected_entities

    yaml_str = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
    frontmatter = f"---\n{yaml_str}---\n\n"

    full_content = frontmatter + content
    filepath.write_text(full_content, encoding='utf-8')

    for cause_id in all_causal:
        _add_leads_to_link(cause_id, memory_id)

    try:
        from vocabulary_bridge import bridge_content
        from semantic_search import embed_single
        bridged = bridge_content(content)
        embed_single(memory_id, bridged)
    except Exception:
        pass

    return memory_id, filepath.name


def _add_leads_to_link(source_id: str, target_id: str) -> bool:
    """Add a leads_to link from source memory to target memory."""
    for directory in ALL_DIRS:
        if not directory.exists():
            continue
        for filepath in directory.glob(f"*-{source_id}.md"):
            metadata, content = parse_memory_file(filepath)

            leads_to = metadata.get('leads_to', [])
            if target_id not in leads_to:
                leads_to.append(target_id)
                metadata['leads_to'] = leads_to
                write_memory_file(filepath, metadata, content)
                return True
    return False


def find_causal_chain(memory_id: str, direction: str = "both", max_depth: int = 5) -> dict:
    """
    Trace the causal chain from a memory.

    Args:
        memory_id: Starting memory
        direction: "causes" (what this led to), "effects" (what caused this), or "both"
        max_depth: Maximum chain depth to traverse

    Returns:
        Dict with 'causes' (upstream) and 'effects' (downstream) chains
    """
    result = {"causes": [], "effects": [], "root": memory_id}

    def get_memory_meta(mid: str) -> Optional[dict]:
        for directory in ALL_DIRS:
            if not directory.exists():
                continue
            for filepath in directory.glob(f"*-{mid}.md"):
                metadata, _ = parse_memory_file(filepath)
                return metadata
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
