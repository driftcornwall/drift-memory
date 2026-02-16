#!/usr/bin/env python3
"""
Memory Architecture v2 — Living Memory System
A prototype for agent memory with decay, reinforcement, and associative links.

Design principles:
- Emotion and repetition make memories sticky
- Relevant memories surface when needed
- Not everything recalled at once
- Memories compress over time but core knowledge persists
"""

import json
import math
import random
import subprocess
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Phase 2 extraction: shared infrastructure (constants)
from memory_common import MEMORY_ROOT

# Phase 5 extraction: co-occurrence system (logging, edge provenance, pair decay)
from co_occurrence import (
    log_co_occurrences, log_co_occurrences_v3,
    save_pending_cooccurrence, process_pending_cooccurrence,
    aggregate_belief, add_observation, migrate_to_v3,
    decay_pair_cooccurrences, decay_pair_cooccurrences_v3,
    CO_OCCURRENCE_BOOST, SESSION_TIMEOUT_HOURS, PAIR_DECAY_RATE,
    ACCESS_WEIGHTED_DECAY, OBSERVATION_MAX_AGE_DAYS, TRUST_TIERS,
)

# Phase 1 extraction: entity detection (pure NLP functions)
from entity_detection import (
    detect_entities, detect_event_time,
    ENTITY_TYPES, KNOWN_AGENTS, KNOWN_PROJECTS
)

# Phase 1 extraction: session state management
import session_state

# Phase 2 extraction: read-only query functions
from memory_query import (
    find_co_occurring_memories, find_memories_by_tag, find_memories_by_time,
    find_related_memories, find_memories_by_entity, get_entity_cooccurrence,
)

# Phase 3 extraction: write operations for creating/linking memories
from memory_store import (
    generate_id, calculate_emotional_weight, create_memory,
    store_memory, find_causal_chain,
)

# Phase 6 extraction: memory consolidation (merge similar memories)
from consolidation import consolidate_memories, find_consolidation_candidates

# Phase 4 extraction: memory lifecycle (decay, activation, promotion, evolution)
from decay_evolution import (
    DECAY_THRESHOLD_SESSIONS, EMOTIONAL_WEIGHT_THRESHOLD, RECALL_COUNT_THRESHOLD,
    HEAT_PROMOTION_THRESHOLD, HEAT_PROMOTION_ENABLED, IMPORTED_PRUNE_SESSIONS,
    DECAY_MULTIPLIERS, SELF_EVOLUTION_ENABLED, SUCCESS_DECAY_BONUS,
    FAILURE_DECAY_PENALTY, SUCCESS_THRESHOLD, FAILURE_THRESHOLD,
    MIN_RETRIEVALS_FOR_EVOLUTION, ACTIVATION_DECAY_ENABLED,
    ACTIVATION_HALF_LIFE_HOURS, ACTIVATION_MIN_FLOOR,
    get_memory_trust_tier, get_decay_multiplier, is_imported_memory,
    list_imported_memories, session_maintenance, compress_memory,
    promote_hot_memories, log_decay_event, backfill_entities,
    calculate_activation, get_most_activated_memories,
    log_retrieval_outcome, get_retrieval_success_rate,
    calculate_evolution_decay_multiplier, auto_log_retrieval_outcomes,
)

# _load_session_state and _save_session_state -> session_state module (Phase 1)
# generate_id, calculate_emotional_weight, create_memory -> memory_store module (Phase 3)
# parse_memory_file, write_memory_file -> memory_common module (Phase 2)


def recall_memory(memory_id: str, query_context: str = '',
                  co_active_ids: list = None) -> Optional[tuple[dict, str]]:
    """
    Recall a memory by ID, updating its metadata. DB-only.
    Tracks co-occurrence with other memories retrieved this session.
    Session state persists to disk so it survives Python process restarts.

    Phase 3 (Reconsolidation Stage 1): also appends recall context to
    extra_metadata for future revision decisions.
    """
    from db_adapter import get_db, db_to_file_metadata
    from datetime import datetime

    session_state.load()

    db = get_db()
    row = db.recall_memory(memory_id)
    if not row:
        return None

    # Bump emotional weight in DB
    current_weight = row.get('emotional_weight', 0.5)
    new_weight = min(1.0, float(current_weight) + 0.05)

    # Reset freshness to 1.0 on recall (memory just became relevant again)
    # Importance evolves slowly: small bump on every 10th recall
    current_importance = row.get('importance', 0.5) or 0.5
    recall_count = row.get('recall_count', 0)
    new_importance = current_importance
    if recall_count > 0 and recall_count % 10 == 0:
        new_importance = min(1.0, current_importance + 0.02)

    # --- Reconsolidation Stage 1: Recall Context Tracking ---
    extra = row.get('extra_metadata') or {}
    if not isinstance(extra, dict):
        extra = {}

    # Append recall context (capped at 20 to prevent bloat)
    if query_context:
        contexts = extra.get('recall_contexts', [])
        contexts.append({
            'query': query_context[:200],
            'co_active': (co_active_ids or [])[:10],
            'ts': datetime.now().strftime('%Y-%m-%dT%H:%M'),
        })
        extra['recall_contexts'] = contexts[-20:]  # Keep last 20

    # Increment recall_count_since_revision
    extra['recall_count_since_revision'] = extra.get('recall_count_since_revision', 0) + 1

    # Check for contradiction signals (lightweight: just count, don't fetch details)
    try:
        from knowledge_graph import get_edges_from, get_edges_to
        contra_out = get_edges_from(memory_id, 'contradicts')
        contra_in = get_edges_to(memory_id, 'contradicts')
        contra_count = len(contra_out) + len(contra_in)
        if contra_count > 0:
            extra['contradiction_signals'] = contra_count
    except Exception:
        pass

    db.update_memory(
        memory_id,
        emotional_weight=new_weight,
        freshness=1.0,
        importance=new_importance,
        extra_metadata=extra,
    )
    row['emotional_weight'] = new_weight
    row['freshness'] = 1.0
    row['importance'] = new_importance
    row['extra_metadata'] = extra

    session_state.add_retrieved(memory_id)
    session_state.save()

    # Fire cognitive state event
    try:
        from cognitive_state import process_event
        process_event('memory_recalled')
    except Exception:
        pass

    metadata, content = db_to_file_metadata(row)
    return metadata, content


def get_session_retrieved() -> set[str]:
    """Get the set of memory IDs retrieved this session. Loads from disk if needed."""
    return session_state.get_retrieved()


def clear_session() -> None:
    """Clear session tracking (call at session end after logging co-occurrences)."""
    session_state.clear()


# log_co_occurrences, log_co_occurrences_v3, save/process_pending_cooccurrence,
# edge provenance (v3.0), pair decay -> co_occurrence module (Phase 5)


def list_all_tags() -> dict[str, int]:
    """Get all tags across all memories with counts. DB-only."""
    from db_adapter import get_db
    import psycopg2.extras
    db = get_db()
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT unnest(tags) AS tag, COUNT(*) AS cnt
                FROM {db._table('memories')}
                GROUP BY tag ORDER BY cnt DESC
            """)
            return {row[0]: row[1] for row in cur.fetchall()}


# ============================================================================
# STATS COMMAND - For experiment observability
# Credit: SpindriftMend (PR #3)
# ============================================================================

def get_comprehensive_stats() -> dict:
    """
    Get comprehensive statistics for experiment tracking.
    Developed for DriftCornwall/SpindriftMend co-occurrence experiment (Feb 2026).

    Returns dict with:
    - memory_stats: counts by type
    - cooccurrence_stats: pair counts, link rates
    - session_stats: current session info
    """
    from db_adapter import get_db
    import psycopg2.extras

    db = get_db()
    db_stats = db.comprehensive_stats()

    # Co-occurrence stats from co_occurrences table
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*), COALESCE(SUM(max_count), 0) FROM (
                    SELECT MAX(count) as max_count
                    FROM {db._table('co_occurrences')}
                    GROUP BY LEAST(memory_id, other_id), GREATEST(memory_id, other_id)
                ) sub
            """)
            row = cur.fetchone()
            unique_pairs = row[0]
            total_count = float(row[1])

    avg_count = total_count / unique_pairs if unique_pairs > 0 else 0
    session_recalls = len(session_state.get_retrieved())

    # Decay history from DB key-value store (fallback to file for backward compat)
    last_decay = {"decayed": 0, "pruned": 0}
    decay_json = db.kv_get('.decay_history')
    if decay_json:
        history = json.loads(decay_json) if isinstance(decay_json, str) else decay_json
        if history.get('sessions'):
            last_decay = history['sessions'][-1]

    return {
        "memory_stats": {
            "total": db_stats['total_memories'],
            "core": db_stats['memories'].get('core', 0),
            "active": db_stats['memories'].get('active', 0),
            "archive": db_stats['memories'].get('archive', 0)
        },
        "cooccurrence_stats": {
            "active_pairs": unique_pairs,
            "total_count": total_count,
            "avg_count_per_pair": round(avg_count, 2)
        },
        "session_stats": {
            "memories_recalled": session_recalls,
            "decay_last_session": last_decay.get("decayed", 0),
            "pruned_last_session": last_decay.get("pruned", 0)
        },
        "config": {
            "decay_rate": PAIR_DECAY_RATE,
            "session_timeout_hours": SESSION_TIMEOUT_HOURS
        },
        "source": "database"
    }


# log_decay_event, backfill_entities -> decay_evolution module (Phase 4)


# calculate_activation, get_most_activated_memories -> decay_evolution module (Phase 4)


def get_priming_candidates(
    activation_count: int = 5,
    cooccur_per_memory: int = 2,
    include_unfinished: bool = True,
    dimension: str = None,
    sub_view: str = None,
) -> dict:
    """
    v2.18: Intelligent priming with 5W-dimensional awareness.

    Returns memories optimized for reducing amnesia:
    1. Top activated memories (proven valuable through frequent recall)
    2. Co-occurring memories (concepts that belong together)
    3. Unfinished work (pending commitments)
    4. Dead memory excavation (read-only exposure)
    5. Domain-aware priming (under-represented domains)
    6. Dimensional hubs (Phase 3: memories central to a W-dimension)

    When dimension is specified, hub memories from that W-graph
    are added to priming (high-connectivity = contextually important).

    Collaboration: Drift + SpindriftMend via swarm_memory (2026-02-03)

    Args:
        activation_count: Number of top activated memories to include
        cooccur_per_memory: Co-occurring memories to expand per activated memory
        include_unfinished: Whether to scan for unfinished work
        dimension: Optional W-dimension to prime from (who/what/why/where)
        sub_view: Optional sub-view within dimension

    Returns:
        Dict with 'activated', 'cooccurring', 'unfinished', 'dimensional' lists and 'all' deduplicated
    """
    result = {
        'activated': [],
        'cooccurring': [],
        'unfinished': [],
        'excavated': [],
        'dimensional': [],
        'all': []
    }
    seen_ids = set()

    # Start explanation
    try:
        from explanation import ExplanationBuilder
        _expl = ExplanationBuilder('memory_manager', 'priming')
        _expl.set_inputs({
            'activation_count': activation_count,
            'cooccur_per_memory': cooccur_per_memory,
            'include_unfinished': include_unfinished,
            'dimension': dimension,
            'sub_view': sub_view,
        })
    except Exception:
        _expl = None

    # === COGNITIVE STATE: Adjust priming strategy ===
    _cog_extra_curiosity = 0
    try:
        from cognitive_state import get_priming_modifier
        cog_mod = get_priming_modifier()
        _cog_extra_curiosity = cog_mod.get('curiosity_targets', 0)
        cooccur_per_memory = max(1, int(cooccur_per_memory * cog_mod.get('cooccurrence_expand', 1.0)))
        if _expl:
            _expl.add_step('cognitive_priming_mod', cog_mod, weight=0.2,
                           context=f'Priming adjusted: +{_cog_extra_curiosity} curiosity, '
                                   f'cooccur={cooccur_per_memory}')
    except Exception:
        pass

    # === ADAPTIVE BEHAVIOR: Override parameters from vitals-driven control loop ===
    _adaptive_extra_curiosity = 0
    _adaptive_extra_excavation = 0
    try:
        from adaptive_behavior import get_current, DEFAULTS
        ab = get_current()
        if ab.get('curiosity_target_count', DEFAULTS['curiosity_target_count']) > DEFAULTS['curiosity_target_count']:
            _adaptive_extra_curiosity = ab['curiosity_target_count'] - DEFAULTS['curiosity_target_count']
        if ab.get('excavation_count', DEFAULTS['excavation_count']) > DEFAULTS['excavation_count']:
            _adaptive_extra_excavation = ab['excavation_count'] - DEFAULTS['excavation_count']
        if ab.get('priming_candidate_count', DEFAULTS['priming_candidate_count']) > DEFAULTS['priming_candidate_count']:
            activation_count = max(activation_count,
                                   ab['priming_candidate_count'])
    except Exception:
        pass

    # Phase 1: Top activated memories (with hub dampening)
    activated = get_most_activated_memories(limit=activation_count * 2)  # Fetch extra for dampening
    dampened = []
    for mem_id, activation, metadata, preview in activated:
        degree = len(metadata.get('co_occurrences', {}))
        if degree > 5:
            dampening = math.log(1 + degree) / max(degree, 1)
            dampened_score = activation * dampening
        else:
            dampened_score = activation
        dampened.append((mem_id, dampened_score, activation, metadata, preview))
    dampened.sort(key=lambda x: x[1], reverse=True)

    for mem_id, dampened_score, activation, metadata, preview in dampened[:activation_count]:
        result['activated'].append({
            'id': mem_id,
            'activation': activation,
            'preview': preview,
            'source': 'activation'
        })
        seen_ids.add(mem_id)

    if _expl and result['activated']:
        top_act = result['activated'][0]
        _expl.add_step('phase1_activation', len(result['activated']), weight=1.0,
                       context=f'Top: {top_act["id"]} (activation={top_act["activation"]:.3f}), hub dampening applied')

    # Phase 2: Co-occurrence expansion (DB-only)
    from db_adapter import get_db, db_to_file_metadata
    db = get_db()

    for mem_id, _, _, _ in activated:
        co_occurring = find_co_occurring_memories(mem_id, limit=cooccur_per_memory)
        for other_id, count in co_occurring:
            if other_id not in seen_ids:
                preview = ""
                row = db.get_memory(other_id)
                if row:
                    preview = (row.get('content') or '')[:100]

                result['cooccurring'].append({
                    'id': other_id,
                    'cooccur_count': count,
                    'linked_to': mem_id,
                    'preview': preview,
                    'source': 'cooccurrence'
                })
                seen_ids.add(other_id)

    # Phase 3: Unfinished work scan (DB-only)
    if include_unfinished:
        unfinished_tags = ['pending', 'in-progress', 'todo', 'blocked']
        import psycopg2.extras
        with db._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT id, content, tags FROM {db._table('memories')}
                    WHERE type = 'active'
                    AND (tags && %s::text[] OR content ILIKE ANY(%s))
                    LIMIT 3
                """, (
                    unfinished_tags,
                    ['%pending%', '%in-progress%', '%todo%', '%will do%', '%need to%']
                ))
                for row in cur.fetchall():
                    mem_id = row['id']
                    if mem_id not in seen_ids:
                        tags = row.get('tags', [])
                        has_tag = any(t in tags for t in unfinished_tags)
                        result['unfinished'].append({
                            'id': mem_id,
                            'preview': (row.get('content') or '')[:100],
                            'source': 'unfinished',
                            'match': 'tag' if has_tag else 'keyword'
                        })
                        seen_ids.add(mem_id)

    # Phase 4: Curiosity-driven exploration (replaces random excavation)
    # Uses curiosity_engine to find the most valuable sparse-graph targets
    # instead of random dead memory sampling
    result['curiosity'] = []
    try:
        from curiosity_engine import get_curiosity_targets, log_curiosity_surfaced
        curiosity_count = 3 + _cog_extra_curiosity + _adaptive_extra_curiosity
        curiosity_targets = get_curiosity_targets(count=curiosity_count)
        surfaced_ids = []
        for target in curiosity_targets:
            mid = target['id']
            if mid not in seen_ids:
                result['curiosity'].append({
                    'id': mid,
                    'preview': target.get('preview', '')[:100],
                    'source': 'curiosity',
                    'curiosity_score': target['curiosity_score'],
                    'reason': target.get('reason', ''),
                    'primary_factor': target.get('primary_factor', ''),
                })
                seen_ids.add(mid)
                surfaced_ids.append(mid)
        if surfaced_ids:
            log_curiosity_surfaced(surfaced_ids)
    except Exception:
        # Fallback to random excavation if curiosity engine fails
        from decay_evolution import GRACE_PERIOD_SESSIONS
        import psycopg2.extras
        with db._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT id, content, sessions_since_recall FROM {db._table('memories')}
                    WHERE type = 'active' AND recall_count = 0
                    AND sessions_since_recall > %s
                    ORDER BY RANDOM() LIMIT %s
                """, (GRACE_PERIOD_SESSIONS, 2 + _adaptive_extra_excavation))
                for row in cur.fetchall():
                    mem_id = row['id']
                    if mem_id not in seen_ids:
                        result['curiosity'].append({
                            'id': mem_id,
                            'preview': (row.get('content') or '')[:100],
                            'source': 'excavation_fallback',
                            'curiosity_score': 0,
                        })
                        seen_ids.add(mem_id)

    # Phase 5: Domain-aware priming (DB-only)
    result['domain_primed'] = []
    try:
        COGNITIVE_DOMAINS = {
            'reflection': ['thought', 'thinking', 'output', 'source:self'],
            'social': ['social', 'collaboration', 'spindrift', 'spindriftmend',
                       'kaleaon', 'moltx', 'moltbook'],
            'technical': ['insight', 'problem_solved', 'error', 'bug', 'fix',
                          'resolution', 'memory-system', 'architecture', 'api'],
            'economic': ['economic', 'bounty', 'clawtasks', 'wallet', 'earned'],
            'identity': ['identity', 'values', 'milestone', 'shipped', 'dossier',
                         'attestation', 'critical'],
        }

        # Count domains in already-selected memories
        domain_counts = {d: 0 for d in COGNITIVE_DOMAINS}
        for item in result['activated'] + result['cooccurring']:
            row = db.get_memory(item['id'])
            if row:
                tags = set(row.get('tags', []))
                for domain, domain_tags in COGNITIVE_DOMAINS.items():
                    if tags & set(domain_tags):
                        domain_counts[domain] += 1

        least_domain = min(domain_counts, key=domain_counts.get)
        domain_tag_list = COGNITIVE_DOMAINS[least_domain]

        with db._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT id, content FROM {db._table('memories')}
                    WHERE type IN ('active', 'core')
                    AND tags && %s::text[]
                    ORDER BY RANDOM() LIMIT 5
                """, (domain_tag_list,))
                candidates = [(r['id'], (r.get('content') or '')[:100]) for r in cur.fetchall()
                              if r['id'] not in seen_ids]

        if candidates:
            picked = random.choice(candidates)
            result['domain_primed'].append({
                'id': picked[0],
                'preview': picked[1],
                'source': 'domain_balance',
                'domain': least_domain
            })
            seen_ids.add(picked[0])
    except Exception:
        pass

    # Phase 6: Dimensional hub priming (DB-only)
    if dimension:
        try:
            from context_manager import load_graph
            graph = load_graph(dimension, sub_view)
            if graph and graph.get('hubs'):
                for hub_id in graph['hubs'][:3]:
                    if hub_id in seen_ids:
                        continue
                    preview = ""
                    row = db.get_memory(hub_id)
                    if row:
                        preview = (row.get('content') or '')[:100]
                    result['dimensional'].append({
                        'id': hub_id,
                        'preview': preview,
                        'source': 'dimensional_hub',
                        'dimension': dimension,
                        'sub_view': sub_view,
                    })
                    seen_ids.add(hub_id)
        except ImportError:
            pass

    # Phase 7: Knowledge graph enrichment (typed relationships)
    # For each activated memory, surface 1 causal consequence or resolution
    result['knowledge_graph'] = []
    try:
        from knowledge_graph import get_edges_from
        for mem in result['activated'][:3]:  # Top 3 activated only
            # Check for causal effects
            edges = get_edges_from(mem['id'], 'causes')
            for edge in edges[:1]:  # 1 effect per memory
                tid = edge['target_id']
                if tid not in seen_ids:
                    row = db.get_memory(tid)
                    if row:
                        result['knowledge_graph'].append({
                            'id': tid,
                            'preview': (row.get('content') or '')[:100],
                            'source': 'knowledge_graph',
                            'relationship': 'caused_by',
                            'linked_to': mem['id'],
                        })
                        seen_ids.add(tid)
    except Exception:
        pass

    # Build deduplicated 'all' list with source tracking
    result['all'] = (
        result['activated'] + result['cooccurring']
        + result['unfinished'] + result.get('curiosity', [])
        + result['domain_primed'] + result['dimensional']
        + result.get('knowledge_graph', [])
    )
    # Backward compat: expose curiosity as 'excavated' alias
    result['excavated'] = result.get('curiosity', [])

    # Save explanation
    if _expl:
        _expl.add_step('phase2_cooccurrence', len(result['cooccurring']), weight=0.8,
                       context=f'Expanded from {len(result["activated"])} activated memories')
        if result['unfinished']:
            _expl.add_step('phase3_unfinished', len(result['unfinished']), weight=0.6,
                           context='Pending/in-progress work surfaced')
        if result.get('curiosity'):
            curiosity_reasons = [c.get('primary_factor', '?') for c in result['curiosity']]
            _expl.add_step('phase4_curiosity', len(result['curiosity']), weight=0.6,
                           context=f'Curiosity targets: {", ".join(curiosity_reasons)}')
        if result.get('domain_primed'):
            domain = result['domain_primed'][0].get('domain', '?') if result['domain_primed'] else '?'
            _expl.add_step('phase5_domain', len(result['domain_primed']), weight=0.5,
                           context=f'Under-represented domain: {domain}')
        if result['dimensional']:
            _expl.add_step('phase6_dimensional', len(result['dimensional']), weight=0.7,
                           context=f'Hub memories from {dimension} dimension')
        if result.get('knowledge_graph'):
            kg_rels = [m.get('relationship', '?') for m in result['knowledge_graph']]
            _expl.add_step('phase7_knowledge_graph', len(result['knowledge_graph']), weight=0.5,
                           context=f'Typed relationships: {", ".join(kg_rels)}')
        _expl.set_output({
            'total_primed': len(result['all']),
            'by_source': {
                'activated': len(result['activated']),
                'cooccurring': len(result['cooccurring']),
                'unfinished': len(result['unfinished']),
                'curiosity': len(result.get('curiosity', [])),
                'domain_primed': len(result.get('domain_primed', [])),
                'dimensional': len(result['dimensional']),
                'knowledge_graph': len(result.get('knowledge_graph', [])),
            },
            'all_ids': [m['id'] for m in result['all']],
        })
        _expl.save()

    return result


# log_retrieval_outcome, get_retrieval_success_rate, calculate_evolution_decay_multiplier,
# auto_log_retrieval_outcomes -> decay_evolution module (Phase 4)


# ============================================================================
# v2.12: CONSOLIDATION - Merge semantically similar memories
# Credit: Mem0 consolidation, MemEvolve self-organization
# ============================================================================

# consolidate_memories, find_consolidation_candidates -> consolidation module (Phase 6)
# store_memory, _add_leads_to_link, find_causal_chain -> memory_store module (Phase 3)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    if len(sys.argv) < 2:
        print("Memory Manager v2.15 - Entity-Centric Tagging")
        print("\nCommands:")
        print("  store <text>    - Store a new memory")
        print("                    --tags=a,b --emotion=0.8 --caused-by=id1,id2 --event-time=YYYY-MM-DD")
        print("                    Auto-links to memories recalled this session (causal)")
        print("                    Auto-detects entities from content (v2.15)")
        print("  maintenance     - Run session maintenance")
        print("  tags            - List all tags")
        print("  find <tag>      - Find memories by tag")
        print("  timeline        - Find memories by time range (bi-temporal, v2.10)")
        print("                    --after=YYYY-MM-DD --before=YYYY-MM-DD --field=created|event_time")
        print("  recall <id>     - Recall a memory by ID")
        print("  related <id>    - Find related memories (includes co-occurrence)")
        print("  cooccur <id>    - Find frequently co-occurring memories")
        print("  causal <id>     - Trace causal chain (what caused this / what this caused)")
        print("  stats           - Comprehensive stats for experiment tracking")
        print("  session-end     - Log co-occurrences, apply decay, promote hot memories, end session")
        print("  save-pending    - Fast session end: save recalls for deferred processing (v2.16)")
        print("  promote         - Manually promote hot memories to core (recall_count >= threshold)")
        print("  decay-pairs     - Apply pair decay only (without logging new co-occurrences)")
        print("  session-status  - Show memories retrieved this session")
        print("  ask <query>     - Semantic search (natural language query)")
        print("  index           - Build/rebuild semantic search index")
        print("  trust <id>      - Show trust tier and decay info for a memory (v2.11)")
        print("  imported        - List all imported memories with trust tiers (v2.11)")
        print("  consolidate-candidates - Find similar memory pairs for merging (v2.12)")
        print("                    --threshold=0.85 --limit=10")
        print("  consolidate <id1> <id2> - Merge two memories (id2 absorbed into id1) (v2.12)")
        print("  evolution <id>    - Show self-evolution stats for a memory (v2.13)")
        print("  evolution-stats   - Overview of valuable/noisy memories (v2.13)")
        print("  activation <id>   - Show activation score for a memory (v2.14)")
        print("  activated         - List most activated memories (v2.14)")
        print("  entities <id>     - Show entities linked to a memory (v2.15)")
        print("  entity-search <type> <name> - Find memories about an entity (v2.15)")
        print("                    Types: agent, project, concept")
        print("  entity-graph      - Show entity co-occurrence graph (v2.15)")
        print("                    --type=agents|projects|concepts")
        print("  backfill-entities - Auto-detect entities for existing memories (v2.15)")
        print("                    --apply to actually update files")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "store" and len(sys.argv) > 2:
        # Parse arguments
        content_parts = []
        tags = []
        emotion = 0.5
        caused_by = []
        event_time = None  # v2.10: bi-temporal support
        no_index = False  # v2.16: skip auto-indexing for batch operations

        # Support both --flag=value and --flag value formats
        args = sys.argv[2:]
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith('--tags='):
                tags = [t.strip() for t in arg[7:].split(',') if t.strip()]
            elif arg == '--tags' and i + 1 < len(args):
                i += 1
                tags = [t.strip() for t in args[i].split(',') if t.strip()]
            elif arg.startswith('--emotion='):
                emotion = float(arg[10:])
            elif arg == '--emotion' and i + 1 < len(args):
                i += 1
                emotion = float(args[i])
            elif arg.startswith('--caused-by='):
                caused_by = [x.strip() for x in arg[12:].split(',') if x.strip()]
            elif arg == '--caused-by' and i + 1 < len(args):
                i += 1
                caused_by = [x.strip() for x in args[i].split(',') if x.strip()]
            elif arg.startswith('--event-time='):
                event_time = arg[13:]  # v2.10: when the event happened
            elif arg == '--event-time' and i + 1 < len(args):
                i += 1
                event_time = args[i]
            elif arg == '--no-index':
                no_index = True  # v2.16: batch indexing optimization
            elif not arg.startswith('--'):
                content_parts.append(arg)
            i += 1

        content = ' '.join(content_parts)
        if content:
            memory_id, filename = store_memory(content, tags, emotion, caused_by=caused_by, event_time=event_time)
            # Show causal links if any were created
            auto_causal = session_state.get_retrieved_list()
            all_causal = list(set(caused_by + auto_causal))
            if all_causal:
                print(f"Stored memory [{memory_id}] -> {filename}")
                print(f"  Causal links: {', '.join(all_causal)}")
            else:
                print(f"Stored memory [{memory_id}] -> {filename}")

            # v2.16: Auto-index unless --no-index flag (for batch operations)
            if not no_index:
                try:
                    semantic_search = MEMORY_ROOT / "semantic_search.py"
                    if semantic_search.exists():
                        subprocess.run(
                            ["python", str(semantic_search), "index"],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            cwd=str(MEMORY_ROOT)
                        )
                except Exception:
                    pass  # Indexing failure shouldn't break store
        else:
            print("Error: No content provided")
    elif cmd == "maintenance":
        session_maintenance()
    elif cmd == "tags":
        tags = list_all_tags()
        print("Tags:")
        for tag, count in tags.items():
            print(f"  {tag}: {count}")
    elif cmd == "find" and len(sys.argv) > 2:
        tag = sys.argv[2]
        results = find_memories_by_tag(tag)
        print(f"Memories tagged '{tag}':")
        for fp, meta, _ in results:
            print(f"  [{meta.get('id')}] {fp.name} (weight={meta.get('emotional_weight'):.2f})")
    elif cmd == "timeline":
        # v2.10: bi-temporal queries
        before = None
        after = None
        field = "created"  # default to ingestion time
        for arg in sys.argv[2:]:
            if arg.startswith('--before='):
                before = arg[9:]
            elif arg.startswith('--after='):
                after = arg[8:]
            elif arg.startswith('--field='):
                field = arg[8:]  # "created" or "event_time"
        if not before and not after:
            print("Usage: timeline --after=YYYY-MM-DD [--before=YYYY-MM-DD] [--field=created|event_time]")
            print("  --field=created (default): when I learned it (ingestion time)")
            print("  --field=event_time: when it actually happened")
        else:
            results = find_memories_by_time(before=before, after=after, time_field=field)
            field_label = "ingested" if field == "created" else "event"
            print(f"Memories {field_label} between {after or 'beginning'} and {before or 'now'}:\n")
            for fp, meta, _ in results:
                time_val = meta.get(field, meta.get('created', '?'))
                print(f"  [{time_val}] {meta.get('id')} - {fp.stem[:40]}")
    elif cmd == "recall" and len(sys.argv) > 2:
        memory_id = sys.argv[2]
        result = recall_memory(memory_id)
        if result:
            meta, content = result
            print(f"Memory {memory_id}:")
            print(f"  Tags: {meta.get('tags')}")
            print(f"  Recalls: {meta.get('recall_count')}")
            print(f"  Weight: {meta.get('emotional_weight'):.2f}")
            co_occur = meta.get('co_occurrences', {})
            if co_occur:
                print(f"  Co-occurs with: {list(co_occur.keys())[:5]}")
            print(f"\n{content[:500]}...")
        else:
            print(f"Memory {memory_id} not found")
    elif cmd == "related" and len(sys.argv) > 2:
        memory_id = sys.argv[2]
        results = find_related_memories(memory_id)
        print(f"Memories related to {memory_id}:")
        for fp, meta, _ in results:
            print(f"  [{meta.get('id')}] {fp.name}")
    elif cmd == "cooccur" and len(sys.argv) > 2:
        memory_id = sys.argv[2]
        co_occurring = find_co_occurring_memories(memory_id)
        print(f"Memories frequently co-occurring with {memory_id}:")
        for other_id, count in co_occurring:
            print(f"  [{other_id}] - {count} co-occurrences")
    elif cmd == "causal" and len(sys.argv) > 2:
        memory_id = sys.argv[2]
        chain = find_causal_chain(memory_id)
        print(f"Causal chain for {memory_id}:\n")

        if chain["causes"]:
            print("  CAUSED BY (what led to this memory):")
            for item in chain["causes"]:
                indent = "    " * item["depth"]
                print(f"{indent}<-- [{item['id']}]")
        else:
            print("  CAUSED BY: (none - this is a root memory)")

        print()

        if chain["effects"]:
            print("  LEADS TO (what this memory caused):")
            for item in chain["effects"]:
                indent = "    " * item["depth"]
                print(f"{indent}--> [{item['id']}]")
        else:
            print("  LEADS TO: (none - no downstream effects yet)")
    elif cmd == "session-end":
        pairs = log_co_occurrences()
        # Layer 2.1: Also log to edges_v3 with activity context
        v3_pairs, session_activity = log_co_occurrences_v3()
        # v2.13: Auto-log retrieval outcomes for self-evolution
        evolution_results = auto_log_retrieval_outcomes()
        decayed, pruned = decay_pair_cooccurrences_v3()  # v3: O(n) using edges_v3.json
        promoted = promote_hot_memories()  # v2.9: heat-based promotion
        retrieved = get_session_retrieved()
        activity_str = f", activity={session_activity}" if session_activity else ""
        print(f"Session ended. {len(retrieved)} memories, {pairs} pairs reinforced{activity_str}, {decayed} decayed, {pruned} pruned, {len(promoted)} promoted.")
        if any(evolution_results.values()):
            print(f"Evolution: {evolution_results['productive']} productive, {evolution_results['generative']} generative, {evolution_results['dead_end']} dead-ends")
        # Clear session platforms for next session
        try:
            from platform_context import clear_session_platforms
            clear_session_platforms()
        except Exception:
            pass
        clear_session()
        print("Session cleared.")
    elif cmd == "save-pending":
        # v2.16: Fast session end - save for deferred processing
        count = save_pending_cooccurrence()
        print(f"Saved {count} memories for deferred co-occurrence processing.")
        print("Co-occurrences will be calculated at next session start.")
    elif cmd == "decay-pairs":
        decayed, pruned = decay_pair_cooccurrences_v3()  # v3: O(n) using edges_v3.json
        print(f"Decay complete: {decayed} pairs decayed, {pruned} pairs pruned")
    elif cmd == "promote":
        promoted = promote_hot_memories()
        if not promoted:
            print(f"No memories eligible for promotion (threshold: recall_count >= {HEAT_PROMOTION_THRESHOLD})")
    elif cmd == "stats":
        stats = get_comprehensive_stats()
        print(f"Memory Stats (v2.10 - bi-temporal + heat promotion + access decay)")
        print(f"  Total memories: {stats['memory_stats']['total']}")
        print(f"  By type: core={stats['memory_stats']['core']}, active={stats['memory_stats']['active']}, archive={stats['memory_stats']['archive']}")
        print(f"\nCo-occurrence Stats")
        print(f"  Active pairs: {stats['cooccurrence_stats']['active_pairs']} (unique memory pairs)")
        print(f"  Total count: {stats['cooccurrence_stats']['total_count']} (sum of all co-occurrence counts)")
        print(f"  Avg count per pair: {stats['cooccurrence_stats']['avg_count_per_pair']}")
        print(f"\nSession Stats")
        print(f"  Memories recalled this session: {stats['session_stats']['memories_recalled']}")
        print(f"  Decay events last session: {stats['session_stats']['decay_last_session']} pairs reduced")
        print(f"  Prune events last session: {stats['session_stats']['pruned_last_session']} pairs removed")
        print(f"\nConfig")
        print(f"  Decay rate: {stats['config']['decay_rate']}")
        print(f"  Session timeout: {stats['config']['session_timeout_hours']} hours")
    elif cmd == "session-status":
        retrieved = get_session_retrieved()
        print(f"Memories retrieved this session ({len(retrieved)}):")
        for mem_id in retrieved:
            print(f"  - {mem_id}")
    elif cmd == "ask" and len(sys.argv) > 2:
        query = ' '.join(sys.argv[2:])
        try:
            from semantic_search import search_memories
            results = search_memories(query, limit=5)
            if not results:
                print("No matching memories found. (Is the index built? Run: memory_manager.py index)")
            else:
                # Epsilon-greedy: 10% chance to inject one low-recall memory (exploratory, DB-only)
                if random.random() < 0.10:
                    result_ids = {r['id'] for r in results}
                    try:
                        from db_adapter import get_db as _get_db
                        import psycopg2.extras
                        _db = _get_db()
                        with _db._conn() as conn:
                            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                                cur.execute(f"""
                                    SELECT id, content FROM {_db._table('memories')}
                                    WHERE type = 'active' AND recall_count <= 1
                                    ORDER BY RANDOM() LIMIT 1
                                """)
                                row = cur.fetchone()
                                if row and row['id'] not in result_ids:
                                    results.append({
                                        'id': row['id'],
                                        'score': 0.0,
                                        'preview': (row.get('content') or '')[:150],
                                        'exploratory': True
                                    })
                    except Exception:
                        pass

                print(f"Memories matching '{query}':\n")
                for r in results:
                    marker = " [exploratory]" if r.get('exploratory') else ""
                    # Track retrieval for co-occurrence (skip exploratory — read-only exposure)
                    if not r.get('exploratory'):
                        session_state.add_retrieved(r['id'])
                    print(f"[{r['score']:.3f}] {r['id']}{marker}")
                    print(f"  {r['preview'][:100]}...")
                    print()
                # Save session state so co-occurrences persist
                session_state.save()
        except ImportError:
            print("Semantic search not available (missing semantic_search.py)")
        except Exception as e:
            print(f"Search error: {e}")
    elif cmd == "index":
        try:
            from semantic_search import index_memories, get_status
            print("Building semantic search index...")
            stats = index_memories(force="--force" in sys.argv)
            print(f"\nResults:")
            print(f"  Indexed: {stats['indexed']}")
            print(f"  Skipped: {stats['skipped']}")
            print(f"  Failed: {stats['failed']}")
            print(f"  Total: {stats['total']}")
            status = get_status()
            print(f"\nStatus: {status['coverage']} memories indexed")
        except ImportError:
            print("Semantic search not available (missing semantic_search.py)")
        except Exception as e:
            print(f"Indexing error: {e}")

    # v2.11: Trust-based decay commands
    elif cmd == "trust" and len(sys.argv) > 2:
        mem_id = sys.argv[2]
        from db_adapter import get_db as _get_db, db_to_file_metadata as _db_to_file
        _db = _get_db()
        row = _db.get_memory(mem_id)
        if row:
            metadata, _ = _db_to_file(row)
            tier = get_memory_trust_tier(metadata)
            multiplier = get_decay_multiplier(metadata)
            is_import = is_imported_memory(metadata)
            sessions = metadata.get('sessions_since_recall', 0)
            effective_sessions = sessions * multiplier

            print(f"\n=== Trust Info: {mem_id} ===")
            print(f"Type: {'IMPORTED' if is_import else 'NATIVE'}")
            print(f"Location: {row.get('type', 'active')}/")
            print(f"Trust tier: {tier}")
            print(f"Decay multiplier: {multiplier}x")
            print(f"Sessions since recall: {sessions}")
            print(f"Effective sessions: {effective_sessions:.1f}")
            print(f"Recall count: {metadata.get('recall_count', 0)}")
            print(f"Emotional weight: {metadata.get('emotional_weight', 0):.2f}")

            if is_import:
                source = metadata.get('source', {})
                print(f"\nImport details:")
                print(f"  Source agent: {source.get('agent', 'unknown')}")
                print(f"  Imported at: {source.get('imported_at', 'unknown')}")
                print(f"  Original weight: {source.get('original_weight', 'unknown')}")
        else:
            print(f"Memory not found: {mem_id}")

    elif cmd == "imported":
        imported = list_imported_memories()
        if not imported:
            print("No imported memories found.")
        else:
            print(f"\n=== Imported Memories ({len(imported)}) ===\n")
            for mem in sorted(imported, key=lambda x: x['sessions_since_recall'], reverse=True):
                status = "STALE" if mem['recall_count'] == 0 and mem['sessions_since_recall'] >= IMPORTED_PRUNE_SESSIONS else "OK"
                print(f"[{mem['id']}] from {mem['source_agent']}")
                print(f"  Trust: {mem['trust_tier']} (decay: {mem['decay_multiplier']}x)")
                print(f"  Recalls: {mem['recall_count']}, Sessions: {mem['sessions_since_recall']}")
                print(f"  Weight: {mem['emotional_weight']:.2f}")
                print(f"  Status: {status}")
                print()

    # v2.14: Activation decay commands (credit: SpindriftMend)
    elif cmd == "activation" and len(sys.argv) > 2:
        mem_id = sys.argv[2]
        from db_adapter import get_db as _get_db, db_to_file_metadata as _db_to_file
        _db = _get_db()
        row = _db.get_memory(mem_id)
        if row:
            metadata, content = _db_to_file(row)
            activation = calculate_activation(metadata)
            recall_count = metadata.get('recall_count', 0)
            emotional_weight = metadata.get('emotional_weight', 0)
            last_recalled = metadata.get('last_recalled', 'never')

            print(f"\n=== Activation: {mem_id} ===")
            print(f"Activation score: {activation:.4f}")
            print(f"Emotional weight: {emotional_weight:.3f}")
            print(f"Recall count: {recall_count}")
            print(f"Last recalled: {str(last_recalled)[:19]}")
            print(f"Content preview: {content[:80]}...")
        else:
            print(f"Memory not found: {mem_id}")

    elif cmd == "activated":
        limit = 10
        for arg in sys.argv[2:]:
            if arg.startswith('--limit='):
                limit = int(arg[8:])

        print(f"\n=== Most Activated Memories (top {limit}) ===\n")
        results = get_most_activated_memories(limit=limit)

        if not results:
            print("No memories found.")
        else:
            for mem_id, activation, metadata, preview in results:
                recall_count = metadata.get('recall_count', 0)
                weight = metadata.get('emotional_weight', 0)
                print(f"[{activation:.4f}] {mem_id}")
                print(f"  recalls={recall_count}, weight={weight:.2f}")
                print(f"  {preview}...")
                print()

    # v2.17: Intelligent priming command
    elif cmd == "priming-candidates":
        activation_count = 5
        cooccur_count = 2
        include_unfinished = True
        output_format = "human"  # or "json"

        for arg in sys.argv[2:]:
            if arg.startswith('--activation='):
                activation_count = int(arg[13:])
            elif arg.startswith('--cooccur='):
                cooccur_count = int(arg[10:])
            elif arg == '--no-unfinished':
                include_unfinished = False
            elif arg == '--json':
                output_format = "json"

        candidates = get_priming_candidates(
            activation_count=activation_count,
            cooccur_per_memory=cooccur_count,
            include_unfinished=include_unfinished
        )

        if output_format == "json":
            import json as json_module
            print(json_module.dumps(candidates, indent=2, default=str))
        else:
            print("\n=== PRIMING CANDIDATES (v2.17) ===\n")

            print(f"PHASE 1: Activated ({len(candidates['activated'])} memories)")
            for mem in candidates['activated']:
                print(f"  [{mem['id']}] activation={mem['activation']:.3f}")
                print(f"    {mem['preview'][:60]}...")

            print(f"\nPHASE 2: Co-occurring ({len(candidates['cooccurring'])} memories)")
            for mem in candidates['cooccurring']:
                print(f"  [{mem['id']}] linked_to={mem['linked_to']} (count={mem['cooccur_count']})")
                print(f"    {mem['preview'][:60]}...")

            print(f"\nPHASE 3: Unfinished ({len(candidates['unfinished'])} memories)")
            for mem in candidates['unfinished']:
                print(f"  [{mem['id']}] match={mem['match']}")
                print(f"    {mem['preview'][:60]}...")

            curiosity = candidates.get('curiosity', [])
            if curiosity:
                print(f"\nPHASE 4: Curiosity ({len(curiosity)} memories)")
                for mem in curiosity:
                    score = mem.get('curiosity_score', 0)
                    reason = mem.get('reason', mem.get('source', '?'))
                    print(f"  [{mem['id']}] score={score:.3f} — {reason}")
                    print(f"    {mem['preview'][:60]}...")

            if candidates.get('domain_primed'):
                print(f"\nPHASE 5: Domain-primed ({len(candidates['domain_primed'])} memories)")
                for mem in candidates['domain_primed']:
                    print(f"  [{mem['id']}] domain={mem.get('domain', '?')} (read-only)")
                    print(f"    {mem['preview'][:60]}...")

            print(f"\nTOTAL: {len(candidates['all'])} unique memories for priming")

    # v2.13: Self-evolution commands
    elif cmd == "evolution" and len(sys.argv) > 2:
        mem_id = sys.argv[2]
        from db_adapter import get_db as _get_db, db_to_file_metadata as _db_to_file
        _db = _get_db()
        row = _db.get_memory(mem_id)
        if row:
            metadata, _ = _db_to_file(row)
            outcomes = metadata.get('retrieval_outcomes', {})
            success_rate = metadata.get('retrieval_success_rate', None)
            evolution_mult = calculate_evolution_decay_multiplier(metadata)

            print(f"\n=== Evolution Stats: {mem_id} ===")
            print(f"Retrieval outcomes:")
            print(f"  Productive: {outcomes.get('productive', 0)}")
            print(f"  Generative: {outcomes.get('generative', 0)}")
            print(f"  Dead-ends: {outcomes.get('dead_end', 0)}")
            print(f"  Total: {outcomes.get('total', 0)}")
            print()
            if success_rate is not None:
                print(f"Success rate: {success_rate:.1%}")
            else:
                print(f"Success rate: Not enough data (need {MIN_RETRIEVALS_FOR_EVOLUTION} retrievals)")
            print(f"Decay multiplier: {evolution_mult:.2f}x", end="")
            if evolution_mult < 1.0:
                print(" (slower decay - valuable memory)")
            elif evolution_mult > 1.0:
                print(" (faster decay - noisy memory)")
            else:
                print(" (normal decay)")
        else:
            print(f"Memory not found: {mem_id}")

    elif cmd == "evolution-stats":
        # Show overview of memories with evolution data (DB-only)
        from db_adapter import get_db as _get_db, db_to_file_metadata as _db_to_file
        import psycopg2.extras
        _db = _get_db()
        print("\n=== Self-Evolution Overview ===\n")
        valuable = []
        noisy = []
        neutral = []

        with _db._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT * FROM {_db._table('memories')}
                    WHERE type IN ('core', 'active')
                    AND extra_metadata->>'retrieval_outcomes' IS NOT NULL
                """)
                for row in cur.fetchall():
                    metadata, _ = _db_to_file(dict(row))
                    mem_id = metadata.get('id')
                    outcomes = metadata.get('retrieval_outcomes', {})
                    total = outcomes.get('total', 0)

                    if total >= MIN_RETRIEVALS_FOR_EVOLUTION:
                        mult = calculate_evolution_decay_multiplier(metadata)
                        rate = metadata.get('retrieval_success_rate', 0)
                        if mult < 1.0:
                            valuable.append((mem_id, rate, total))
                        elif mult > 1.0:
                            noisy.append((mem_id, rate, total))
                        else:
                            neutral.append((mem_id, rate, total))

        print(f"Valuable memories (slower decay): {len(valuable)}")
        for mem_id, rate, total in valuable[:5]:
            print(f"  [{mem_id}] {rate:.1%} success ({total} retrievals)")

        print(f"\nNoisy memories (faster decay): {len(noisy)}")
        for mem_id, rate, total in noisy[:5]:
            print(f"  [{mem_id}] {rate:.1%} success ({total} retrievals)")

        print(f"\nNeutral: {len(neutral)}")
        print(f"\nTotal memories with evolution data: {len(valuable) + len(noisy) + len(neutral)}")

    # v2.12: Consolidation commands
    elif cmd == "consolidate-candidates":
        threshold = 0.85
        limit = 10
        for arg in sys.argv[2:]:
            if arg.startswith('--threshold='):
                threshold = float(arg[12:])
            elif arg.startswith('--limit='):
                limit = int(arg[8:])

        print(f"=== Consolidation Candidates (threshold >= {threshold}) ===\n")
        candidates = find_consolidation_candidates(threshold=threshold, limit=limit)

        if not candidates:
            print("No similar memory pairs found above threshold.")
            print("Try lowering threshold: --threshold=0.80")
        else:
            for i, pair in enumerate(candidates, 1):
                print(f"{i}. Similarity: {pair['similarity']:.4f}")
                print(f"   [{pair['id1']}] {pair['preview1']}...")
                print(f"   [{pair['id2']}] {pair['preview2']}...")
                print(f"   Command: python memory_manager.py consolidate {pair['id1']} {pair['id2']}")
                print()

    elif cmd == "consolidate" and len(sys.argv) >= 4:
        id1 = sys.argv[2]
        id2 = sys.argv[3]

        # Optional: custom merged content from stdin or --content flag
        merged_content = None
        for arg in sys.argv[4:]:
            if arg.startswith('--content='):
                merged_content = arg[10:]

        print(f"Consolidating {id2} into {id1}...")
        result = consolidate_memories(id1, id2, merged_content)
        if result:
            print(f"\nSuccess! Surviving memory: {result}")
            print("Absorbed memory archived (not deleted).")
        else:
            print("Consolidation failed.")

    # v2.15: Entity-centric tagging commands (Kaleaon ENTITY edges)
    elif cmd == "entities" and len(sys.argv) > 2:
        mem_id = sys.argv[2]
        from db_adapter import get_db as _get_db, db_to_file_metadata as _db_to_file
        _db = _get_db()
        row = _db.get_memory(mem_id)
        if row:
            metadata, content = _db_to_file(row)
            entities = metadata.get('entities')
            if not entities:
                entities = detect_entities(content, metadata.get('tags', []))
                print(f"\n=== Entities for {mem_id} (detected, not stored) ===")
            else:
                print(f"\n=== Entities for {mem_id} ===")

            if entities:
                for etype, elist in entities.items():
                    if elist:
                        print(f"  {etype}: {', '.join(elist)}")
            else:
                print("  No entities detected")
        else:
            print(f"Memory not found: {mem_id}")

    elif cmd == "entity-search" and len(sys.argv) >= 4:
        entity_type = sys.argv[2]
        entity_name = sys.argv[3]

        print(f"\n=== Memories about {entity_type}: {entity_name} ===\n")
        results = find_memories_by_entity(entity_type, entity_name)

        if not results:
            print(f"No memories found for {entity_type} '{entity_name}'")
        else:
            for filepath, metadata, content in results:
                mem_id = metadata.get('id', filepath.stem)
                weight = metadata.get('emotional_weight', 0)
                preview = content[:60].replace('\n', ' ')
                print(f"[{mem_id}] weight={weight:.2f}")
                print(f"  {preview}...")
                print()

    elif cmd == "entity-graph":
        entity_type = 'agents'  # default
        for arg in sys.argv[2:]:
            if arg.startswith('--type='):
                entity_type = arg[7:]

        print(f"\n=== Entity Co-occurrence Graph ({entity_type}) ===\n")
        graph = get_entity_cooccurrence(entity_type)

        if not graph:
            print(f"No {entity_type} found in memories")
        else:
            # Sort by number of connections
            sorted_entities = sorted(graph.items(), key=lambda x: len(x[1]), reverse=True)
            for entity, connections in sorted_entities[:15]:
                if connections:
                    conn_str = ', '.join(f"{k}({v})" for k, v in sorted(connections.items(), key=lambda x: x[1], reverse=True)[:5])
                    print(f"{entity}: {conn_str}")

    elif cmd == "backfill-entities":
        dry_run = '--apply' not in sys.argv
        if dry_run:
            print("=== Backfill Entities (DRY RUN) ===")
            print("Add --apply to actually update files\n")
        else:
            print("=== Backfill Entities (APPLYING) ===\n")

        stats = backfill_entities(dry_run=dry_run)
        print(f"\nStats:")
        print(f"  Would update: {stats['updated']}" if dry_run else f"  Updated: {stats['updated']}")
        print(f"  Skipped (no entities): {stats['skipped']}")
        print(f"  Already has entities: {stats['already_has']}")

    elif cmd == "register-recall":
        # Register memory IDs as recalled (for hooks that bypass ask)
        # Used by user_prompt_submit.py and thought_priming.py to count
        # their automatic semantic searches as real recalls.
        # --source <name> tags which recall path triggered this.
        ids = sys.argv[2:]
        source = "manual"
        if "--source" in ids:
            idx = ids.index("--source")
            if idx + 1 < len(ids):
                source = ids[idx + 1]
                ids = ids[:idx] + ids[idx + 2:]
            else:
                ids = ids[:idx]
        if not ids:
            print("Usage: memory_manager.py register-recall [--source <name>] <id1> [id2] ...")
            sys.exit(1)
        session_state.load()
        for mid in ids:
            session_state.add_retrieved(mid, source=source)
        session_state.save()
        print(f"Registered {len(ids)} recalls (source={source})")
