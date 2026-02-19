#!/usr/bin/env python3
"""
Memory Consolidation — Tier-aware merge of semantically similar memories.

Extracted from memory_manager.py (Phase 6).
Phase 3 Step 5: Tier-aware consolidation (episodic/semantic/procedural).
All operations go through PostgreSQL. No file system.
Credit: Mem0 consolidation, MemEvolve self-organization.
"""

from datetime import datetime, timezone
from typing import Optional

from db_adapter import get_db, db_to_file_metadata


# --- Tier-specific consolidation parameters ---
# Episodic: time-bound events, higher threshold to avoid losing temporal context
# Semantic: abstracted knowledge, standard threshold, NLI dedup
# Procedural: skill chunks, NEVER auto-consolidated
TIER_CONSOLIDATION_THRESHOLD = {
    'episodic': 0.92,    # High — don't merge events unless near-duplicate
    'semantic': 0.85,    # Standard — merge similar knowledge freely
    'procedural': 1.01,  # Impossible — procedural memories never consolidate
}

# When two episodic memories merge, check if result should promote to semantic
EPISODIC_TO_SEMANTIC_MIN_SOURCES = 3  # Need 3+ consolidated sources to promote


def _get_memory_tier(row: dict) -> str:
    """Get memory tier from DB row, defaulting to episodic."""
    return row.get('memory_tier') or 'episodic'


def consolidate_memories(id1: str, id2: str, merged_content: Optional[str] = None) -> Optional[str]:
    """
    Consolidate two similar memories into one. DB-only. Tier-aware.

    Tier-specific behavior:
    - PROCEDURAL: Refuses to consolidate. Returns None.
    - EPISODIC: Takes max recall_count (not sum — time-bound events).
      If 3+ sources consolidated, auto-promotes to semantic.
    - SEMANTIC: Sums recall_counts. NLI check before merge if available.
    - CROSS-TIER: Lower tier absorbs into higher (episodic → semantic).

    Args:
        id1: First memory ID (will be kept and updated)
        id2: Second memory ID (will be archived/absorbed)
        merged_content: Optional custom merged content. If None, uses LLM/concat.

    Returns:
        The surviving memory ID (id1), or None if failed.
    """
    db = get_db()

    row1 = db.get_memory(id1)
    row2 = db.get_memory(id2)

    if not row1 or not row2:
        print(f"Error: Could not find both memories ({id1}, {id2})")
        return None

    tier1 = _get_memory_tier(row1)
    tier2 = _get_memory_tier(row2)

    # PROCEDURAL: Never consolidate
    if tier1 == 'procedural' or tier2 == 'procedural':
        print(f"Skipped: procedural memories cannot be consolidated ({id1}={tier1}, {id2}={tier2})")
        return None

    meta1, content1 = db_to_file_metadata(row1)
    meta2, content2 = db_to_file_metadata(row2)

    # SEMANTIC: NLI dedup check — only merge if entailment is strong
    if tier1 == 'semantic' or tier2 == 'semantic':
        try:
            from contradiction_detector import _classify_pair, _nli_available
            if _nli_available():
                nli_result = _classify_pair(content1, content2)
                entailment = nli_result.get('entailment', 0.0) if nli_result else 0.0
                if entailment < 0.7:
                    print(f"Skipped: semantic NLI entailment too low ({entailment:.2f} < 0.7)")
                    return None
        except Exception:
            pass  # NLI unavailable — proceed with LLM merge

    # Merge content — R12: try LLM-mediated consolidation first
    if merged_content:
        final_content = merged_content
    else:
        try:
            from llm_client import consolidate_memories_llm
            llm_result = consolidate_memories_llm(content1, content2, id1, id2, meta1, meta2)
            final_content = llm_result['merged_content']
            if llm_result.get('used_llm'):
                print(f"  LLM consolidation: {llm_result['backend']}/{llm_result['model']} "
                      f"({llm_result['elapsed_ms']}ms)")
        except Exception:
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

    # Tier-aware recall count merging
    if tier1 == 'episodic' and tier2 == 'episodic':
        # Episodic: take max (time-bound events don't "stack")
        final_recalls = max(meta1.get('recall_count', 0), meta2.get('recall_count', 0))
    else:
        # Semantic / cross-tier: sum (knowledge accumulates)
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

    # Determine surviving tier (cross-tier: promote to higher)
    tier_rank = {'episodic': 0, 'semantic': 1}
    surviving_tier = tier1 if tier_rank.get(tier1, 0) >= tier_rank.get(tier2, 0) else tier2

    # Update surviving memory (id1) in DB
    extra = row1.get('extra_metadata', {}) or {}
    extra['caused_by'] = final_caused_by
    extra['leads_to'] = final_leads_to
    extra['consolidated_from'] = extra.get('consolidated_from', []) + [id2]
    extra['consolidated_at'] = datetime.now(timezone.utc).isoformat()

    # Auto-promote: episodic with 3+ consolidated sources → semantic
    source_count = len(extra.get('consolidated_from', []))
    if surviving_tier == 'episodic' and source_count >= EPISODIC_TO_SEMANTIC_MIN_SOURCES:
        surviving_tier = 'semantic'
        extra['promoted_from'] = 'episodic'
        extra['promoted_at'] = datetime.now(timezone.utc).isoformat()
        extra['promotion_reason'] = f'consolidated {source_count} episodic sources'
        print(f"  PROMOTED: episodic -> semantic ({source_count} sources consolidated)")

    update_kwargs = dict(
        content=final_content,
        emotional_weight=round(final_weight, 3),
        tags=final_tags,
        recall_count=final_recalls,
        extra_metadata=extra,
    )

    # Set surviving tier if changed
    if surviving_tier != tier1:
        update_kwargs['memory_tier'] = surviving_tier

    db.update_memory(id1, **update_kwargs)

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

    print(f"Consolidated: {id2} -> {id1} (tier: {surviving_tier})")
    print(f"  Final weight: {final_weight:.3f}")
    print(f"  Final tags: {final_tags}")
    print(f"  Final recalls: {final_recalls}")
    return id1


def find_consolidation_candidates(threshold: float = None, limit: int = 10) -> list[dict]:
    """
    Find pairs of memories that are candidates for consolidation.
    Tier-aware: uses different similarity thresholds per tier.

    Phase 3 Step 5:
    - Episodic pairs need 0.92+ similarity (preserve temporal specificity)
    - Semantic pairs need 0.85+ similarity (standard knowledge merge)
    - Procedural pairs are NEVER candidates
    - Cross-tier: uses the HIGHER threshold of the two tiers

    N5 v1.2: High-binding-strength memories resist consolidation.

    Args:
        threshold: Override minimum similarity (None = tier-aware defaults)
        limit: Max candidates to return

    Returns:
        List of candidate pairs with similarity scores
    """
    # Use the lowest tier threshold as initial filter, then refine per-pair
    min_threshold = threshold if threshold is not None else min(TIER_CONSOLIDATION_THRESHOLD.values())

    try:
        from semantic_search import find_similar_pairs
        candidates = find_similar_pairs(threshold=min_threshold, limit=limit * 3)
    except ImportError:
        print("Semantic search not available")
        return []

    if not candidates:
        return []

    # Phase 3: Tier-aware filtering — look up tiers and apply per-pair thresholds
    db = get_db()
    all_ids = set()
    for pair in candidates:
        all_ids.add(pair.get('id1', ''))
        all_ids.add(pair.get('id2', ''))
    all_ids.discard('')

    # Batch fetch tiers
    tier_map = {}
    if all_ids:
        import psycopg2.extras
        with db._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT id, COALESCE(memory_tier, 'episodic') as memory_tier
                    FROM {db._table('memories')}
                    WHERE id = ANY(%s)
                """, (list(all_ids),))
                for row in cur.fetchall():
                    tier_map[row['id']] = row['memory_tier']

    # Apply tier-specific thresholds
    tier_filtered = []
    for pair in candidates:
        t1 = tier_map.get(pair.get('id1', ''), 'episodic')
        t2 = tier_map.get(pair.get('id2', ''), 'episodic')

        # Procedural: never consolidate
        if t1 == 'procedural' or t2 == 'procedural':
            continue

        # Cross-tier or same-tier: use the HIGHER threshold
        required = max(
            TIER_CONSOLIDATION_THRESHOLD.get(t1, 0.85),
            TIER_CONSOLIDATION_THRESHOLD.get(t2, 0.85)
        )

        if pair.get('similarity', 0) >= required:
            pair['tier1'] = t1
            pair['tier2'] = t2
            pair['tier_threshold'] = required
            tier_filtered.append(pair)

    candidates = tier_filtered

    # N5 v1.2: Filter out pairs where either memory has high binding_strength
    PHI_RESISTANCE_THRESHOLD = 0.5  # Memories above this resist consolidation
    try:
        from binding_layer import bind_results, BINDING_ENABLED
        if BINDING_ENABLED:
            cand_ids = set()
            for pair in candidates:
                cand_ids.add(pair.get('id1', ''))
                cand_ids.add(pair.get('id2', ''))
            cand_ids.discard('')

            if cand_ids:
                # Bind all candidate memories
                fake_results = [{'id': mid, 'score': 0.5} for mid in cand_ids]
                bound = bind_results(fake_results, full_count=min(len(fake_results), 15))
                phi_map = {b.id: b.binding_strength for b in bound}

                # Filter: skip pairs where either memory has high Phi
                for pair in candidates:
                    phi1 = phi_map.get(pair.get('id1', ''), 0)
                    phi2 = phi_map.get(pair.get('id2', ''), 0)
                    if phi1 >= PHI_RESISTANCE_THRESHOLD or phi2 >= PHI_RESISTANCE_THRESHOLD:
                        pair['binding_resistant'] = True
                        pair['phi1'] = round(phi1, 3)
                        pair['phi2'] = round(phi2, 3)

                candidates = [p for p in candidates if not p.get('binding_resistant')]
    except Exception:
        pass  # Binding unavailable — proceed without filtering

    return candidates[:limit]
