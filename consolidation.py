#!/usr/bin/env python3
"""
Memory Consolidation — Merge semantically similar memories.

Extracted from memory_manager.py (Phase 6).
Credit: Mem0 consolidation, MemEvolve self-organization.
"""

from datetime import datetime, timezone
from typing import Optional

from memory_common import (
    CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR,
    parse_memory_file, write_memory_file,
)


def consolidate_memories(id1: str, id2: str, merged_content: Optional[str] = None) -> Optional[str]:
    """
    Consolidate two similar memories into one.

    The consolidation process:
    1. Merges content (or uses provided merged_content)
    2. Takes the higher emotional_weight
    3. Unions all tags
    4. Sums recall_counts
    5. Merges co-occurrence counts (union, sum overlaps)
    6. Keeps the older created date
    7. Unions causal links
    8. Archives the absorbed memory (doesn't delete - preserves history)
    9. Updates the embedding index

    Args:
        id1: First memory ID (will be kept and updated)
        id2: Second memory ID (will be archived/absorbed)
        merged_content: Optional custom merged content. If None, concatenates both.

    Returns:
        The surviving memory ID (id1), or None if failed.
    """
    # Find both memories
    mem1_data = None
    mem2_data = None
    mem1_path = None
    mem2_path = None

    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            mid = metadata.get('id')
            if mid == id1:
                mem1_data = (metadata, content)
                mem1_path = filepath
            elif mid == id2:
                mem2_data = (metadata, content)
                mem2_path = filepath

    if not mem1_data or not mem2_data:
        print(f"Error: Could not find both memories ({id1}, {id2})")
        return None

    meta1, content1 = mem1_data
    meta2, content2 = mem2_data

    # Merge content
    if merged_content:
        final_content = merged_content
    else:
        final_content = f"{content1}\n\n---\n[Consolidated from {id2}]\n\n{content2}"

    # Take higher emotional weight
    final_weight = max(
        meta1.get('emotional_weight', 0.5),
        meta2.get('emotional_weight', 0.5)
    )

    # Union tags
    tags1 = set(meta1.get('tags', []))
    tags2 = set(meta2.get('tags', []))
    final_tags = list(tags1 | tags2)

    # Sum recall counts
    final_recalls = meta1.get('recall_count', 0) + meta2.get('recall_count', 0)

    # Merge co-occurrences (union, sum overlapping counts)
    co1 = meta1.get('co_occurrences', {})
    co2 = meta2.get('co_occurrences', {})
    final_co = dict(co1)
    for other_id, count in co2.items():
        if other_id == id1:
            continue
        final_co[other_id] = final_co.get(other_id, 0) + count
    final_co.pop(id2, None)

    # Keep older created date
    created1 = str(meta1.get('created', ''))
    created2 = str(meta2.get('created', ''))
    final_created = min(created1, created2) if created1 and created2 else created1 or created2

    # Union causal links
    caused_by1 = set(meta1.get('caused_by', []))
    caused_by2 = set(meta2.get('caused_by', []))
    final_caused_by = list((caused_by1 | caused_by2) - {id1, id2})

    leads_to1 = set(meta1.get('leads_to', []))
    leads_to2 = set(meta2.get('leads_to', []))
    final_leads_to = list((leads_to1 | leads_to2) - {id1, id2})

    # Update surviving memory (id1)
    meta1['emotional_weight'] = round(final_weight, 3)
    meta1['tags'] = final_tags
    meta1['recall_count'] = final_recalls
    meta1['co_occurrences'] = final_co
    meta1['created'] = final_created
    meta1['caused_by'] = final_caused_by
    meta1['leads_to'] = final_leads_to
    meta1['consolidated_from'] = meta1.get('consolidated_from', []) + [id2]
    meta1['consolidated_at'] = datetime.now(timezone.utc).isoformat()

    write_memory_file(mem1_path, meta1, final_content)

    # Archive the absorbed memory (id2)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    meta2['archived_at'] = datetime.now(timezone.utc).isoformat()
    meta2['archived_reason'] = f'consolidated_into:{id1}'
    archive_path = ARCHIVE_DIR / mem2_path.name
    write_memory_file(archive_path, meta2, content2)
    mem2_path.unlink()

    # Update embedding index
    try:
        from semantic_search import embed_single, remove_from_index
        embed_single(id1, final_content)
        remove_from_index(id2)
    except Exception:
        pass

    # Update co-occurrence references in other memories (replace id2 with id1)
    for directory in [CORE_DIR, ACTIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            mid = metadata.get('id')
            if mid in (id1, id2):
                continue

            co = metadata.get('co_occurrences', {})
            if id2 in co:
                co[id1] = co.get(id1, 0) + co[id2]
                del co[id2]
                metadata['co_occurrences'] = co
                write_memory_file(filepath, metadata, content)

    print(f"Consolidated: {id2} -> {id1}")
    print(f"  Final weight: {final_weight:.3f}")
    print(f"  Final tags: {final_tags}")
    print(f"  Final recalls: {final_recalls}")
    return id1


def find_consolidation_candidates(threshold: float = 0.85, limit: int = 10) -> list[dict]:
    """
    Find pairs of memories that are candidates for consolidation.

    Args:
        threshold: Minimum similarity (0.85 = very similar)
        limit: Max candidates to return

    Returns:
        List of candidate pairs with similarity scores
    """
    try:
        from semantic_search import find_similar_pairs
        return find_similar_pairs(threshold=threshold, limit=limit)
    except ImportError:
        print("Semantic search not available")
        return []
