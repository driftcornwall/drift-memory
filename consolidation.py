#!/usr/bin/env python3
"""
Memory Consolidation — Merge semantically similar memories.

Extracted from memory_manager.py (Phase 6).
All operations go through PostgreSQL. No file system.
Credit: Mem0 consolidation, MemEvolve self-organization.
"""

from datetime import datetime, timezone
from typing import Optional

from db_adapter import get_db, db_to_file_metadata


def consolidate_memories(id1: str, id2: str, merged_content: Optional[str] = None) -> Optional[str]:
    """
    Consolidate two similar memories into one. DB-only.

    The consolidation process:
    1. Merges content (or uses provided merged_content)
    2. Takes the higher emotional_weight
    3. Unions all tags
    4. Sums recall_counts
    5. Keeps the older created date
    6. Unions causal links
    7. Archives the absorbed memory (type -> 'archive')
    8. Updates the embedding index

    Args:
        id1: First memory ID (will be kept and updated)
        id2: Second memory ID (will be archived/absorbed)
        merged_content: Optional custom merged content. If None, concatenates both.

    Returns:
        The surviving memory ID (id1), or None if failed.
    """
    db = get_db()

    row1 = db.get_memory(id1)
    row2 = db.get_memory(id2)

    if not row1 or not row2:
        print(f"Error: Could not find both memories ({id1}, {id2})")
        return None

    meta1, content1 = db_to_file_metadata(row1)
    meta2, content2 = db_to_file_metadata(row2)

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

    # Update surviving memory (id1) in DB
    extra = row1.get('extra_metadata', {}) or {}
    extra['caused_by'] = final_caused_by
    extra['leads_to'] = final_leads_to
    extra['consolidated_from'] = extra.get('consolidated_from', []) + [id2]
    extra['consolidated_at'] = datetime.now(timezone.utc).isoformat()

    db.update_memory(id1,
        content=final_content,
        emotional_weight=round(final_weight, 3),
        tags=final_tags,
        recall_count=final_recalls,
        extra_metadata=extra,
    )

    # Archive the absorbed memory (id2) — change type to 'archive'
    extra2 = row2.get('extra_metadata', {}) or {}
    extra2['archived_at'] = datetime.now(timezone.utc).isoformat()
    extra2['archived_reason'] = f'consolidated_into:{id1}'
    db.update_memory(id2, type='archive', extra_metadata=extra2)

    # Update embedding index
    try:
        from semantic_search import embed_single, remove_from_index
        try:
            from vocabulary_bridge import bridge_content
            final_content_bridged = bridge_content(final_content)
        except ImportError:
            final_content_bridged = final_content
        embed_single(id1, final_content_bridged)
        remove_from_index(id2)
    except Exception:
        pass

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
