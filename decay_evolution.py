#!/usr/bin/env python3
"""
Decay Evolution — Memory lifecycle operations.

Extracted from memory_manager.py (Phase 4).
Handles trust tiers, activation scoring, session maintenance,
compression, promotion, retrieval outcomes, and adaptive decay.
"""

import json
import math
from datetime import datetime, timezone
from typing import Optional

from memory_common import (
    MEMORY_ROOT, CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR, ALL_DIRS,
    parse_memory_file, write_memory_file,
)
from entity_detection import detect_entities
import session_state

# --- Configuration constants (decay/evolution-specific) ---

DECAY_THRESHOLD_SESSIONS = 7
EMOTIONAL_WEIGHT_THRESHOLD = 0.6
RECALL_COUNT_THRESHOLD = 5
HEAT_PROMOTION_THRESHOLD = 10
HEAT_PROMOTION_ENABLED = True
IMPORTED_PRUNE_SESSIONS = 14

# Trust-based decay for imported memories (v2.11, credit: SpindriftMend)
DECAY_MULTIPLIERS = {
    'self': 1.0,
    'verified_agent': 1.5,
    'platform': 2.0,
    'unknown': 3.0,
}

# Self-evolution — canonical value lives in memory_common.py (shared config)
from memory_common import SELF_EVOLUTION_ENABLED
SUCCESS_DECAY_BONUS = 0.7
FAILURE_DECAY_PENALTY = 1.5
SUCCESS_THRESHOLD = 0.6
FAILURE_THRESHOLD = 0.3
MIN_RETRIEVALS_FOR_EVOLUTION = 3

# Activation decay — Hebbian time-based (v2.14, credit: SpindriftMend v3.1)
ACTIVATION_DECAY_ENABLED = True
ACTIVATION_HALF_LIFE_HOURS = 24 * 7
ACTIVATION_MIN_FLOOR = 0.01


# --- Trust tier functions ---

def get_memory_trust_tier(metadata: dict) -> str:
    """
    Extract trust tier from memory metadata.

    Priority:
    1. Explicit source.trust_tier field (from import)
    2. Presence of 'imported:AgentName' tag -> verified_agent
    3. Default to 'self' (my own memories)
    """
    source = metadata.get('source', {})
    if isinstance(source, dict):
        tier = source.get('trust_tier')
        if tier and tier in DECAY_MULTIPLIERS:
            return tier

    for tag in metadata.get('tags', []):
        if isinstance(tag, str) and tag.startswith('imported:'):
            return 'verified_agent'

    return 'self'


def get_decay_multiplier(metadata: dict) -> float:
    """Get decay rate multiplier based on trust tier."""
    tier = get_memory_trust_tier(metadata)
    return DECAY_MULTIPLIERS.get(tier, 1.0)


def is_imported_memory(metadata: dict) -> bool:
    """Check if memory was imported from another agent."""
    source = metadata.get('source', {})
    if isinstance(source, dict) and source.get('agent'):
        return True

    for tag in metadata.get('tags', []):
        if isinstance(tag, str) and tag.startswith('imported:'):
            return True

    return False


def list_imported_memories() -> list:
    """List all imported memories with their trust tiers."""
    imported = []
    for directory in [ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            if is_imported_memory(metadata):
                imported.append({
                    'filepath': filepath,
                    'id': metadata.get('id', filepath.stem),
                    'trust_tier': get_memory_trust_tier(metadata),
                    'decay_multiplier': get_decay_multiplier(metadata),
                    'recall_count': metadata.get('recall_count', 0),
                    'sessions_since_recall': metadata.get('sessions_since_recall', 0),
                    'source_agent': metadata.get('source', {}).get('agent', 'unknown'),
                    'emotional_weight': metadata.get('emotional_weight', 0.5),
                })
    return imported


# --- Session maintenance ---

def session_maintenance():
    """
    Run at the start of each session to:
    1. Increment sessions_since_recall for all active memories
    2. Identify decay candidates (with trust-based decay multipliers for imports)
    3. Prune never-recalled imports after IMPORTED_PRUNE_SESSIONS
    4. Report status

    v2.11: Trust-based decay for imported memories
    """
    print("\n=== Memory Session Maintenance ===\n")

    decay_candidates = []
    reinforced = []
    prune_candidates = []

    for filepath in ACTIVE_DIR.glob("*.md") if ACTIVE_DIR.exists() else []:
        metadata, content = parse_memory_file(filepath)

        sessions = metadata.get('sessions_since_recall', 0) + 1
        metadata['sessions_since_recall'] = sessions

        decay_multiplier = get_decay_multiplier(metadata)
        effective_sessions = sessions * decay_multiplier

        emotional_weight = metadata.get('emotional_weight', 0.5)
        recall_count = metadata.get('recall_count', 0)

        should_resist_decay = (
            emotional_weight >= EMOTIONAL_WEIGHT_THRESHOLD or
            recall_count >= RECALL_COUNT_THRESHOLD
        )

        is_import = is_imported_memory(metadata)
        if is_import and recall_count == 0 and sessions >= IMPORTED_PRUNE_SESSIONS:
            prune_candidates.append((filepath, metadata, content))
        elif effective_sessions >= DECAY_THRESHOLD_SESSIONS and not should_resist_decay:
            decay_candidates.append((filepath, metadata, content, decay_multiplier))
        elif should_resist_decay:
            reinforced.append((filepath, metadata))

        write_memory_file(filepath, metadata, content)

    print(f"Active memories: {len(list(ACTIVE_DIR.glob('*.md'))) if ACTIVE_DIR.exists() else 0}")
    print(f"Core memories: {len(list(CORE_DIR.glob('*.md'))) if CORE_DIR.exists() else 0}")
    print(f"Archived memories: {len(list(ARCHIVE_DIR.glob('*.md'))) if ARCHIVE_DIR.exists() else 0}")

    if decay_candidates:
        print(f"\nDecay candidates ({len(decay_candidates)}):")
        for item in decay_candidates:
            fp, meta = item[0], item[1]
            multiplier = item[3] if len(item) > 3 else 1.0
            eff_sessions = meta.get('sessions_since_recall', 0) * multiplier
            import_marker = " [IMPORT]" if is_imported_memory(meta) else ""
            print(f"  - {fp.name}: {meta.get('sessions_since_recall')} sessions (eff: {eff_sessions:.1f}), weight={meta.get('emotional_weight'):.2f}{import_marker}")

    if prune_candidates:
        print(f"\nPrune candidates - never-recalled imports ({len(prune_candidates)}):")
        for fp, meta, _ in prune_candidates:
            source = meta.get('source', {}).get('agent', 'unknown')
            print(f"  - {fp.name}: from {source}, {meta.get('sessions_since_recall')} sessions, 0 recalls")

    if reinforced:
        print(f"\nReinforced (resist decay):")
        for fp, meta in reinforced[:5]:
            print(f"  - {fp.name}: recalls={meta.get('recall_count')}, weight={meta.get('emotional_weight'):.2f}")

    return decay_candidates, prune_candidates


# --- Compression and promotion ---

def compress_memory(memory_id: str, compressed_content: str):
    """
    Compress a memory - move to archive with reduced content.
    The original content is lost but can be referenced.
    """
    for filepath in ACTIVE_DIR.glob(f"*-{memory_id}.md"):
        metadata, original_content = parse_memory_file(filepath)

        metadata['compressed_at'] = datetime.now(timezone.utc).isoformat()
        metadata['original_length'] = len(original_content)

        new_path = ARCHIVE_DIR / filepath.name
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        write_memory_file(new_path, metadata, compressed_content)

        filepath.unlink()
        print(f"Compressed: {filepath.name} -> {new_path}")
        return new_path

    return None


def promote_hot_memories() -> list[str]:
    """
    Promote frequently-accessed memories from active to core.
    Called at session-end to elevate important memories.

    Credit: memU heat-based promotion pattern (v2.9)
    """
    if not HEAT_PROMOTION_ENABLED:
        return []

    if not ACTIVE_DIR.exists():
        return []

    promoted = []
    CORE_DIR.mkdir(parents=True, exist_ok=True)

    for filepath in ACTIVE_DIR.glob("*.md"):
        metadata, content = parse_memory_file(filepath)
        recall_count = metadata.get('recall_count', 0)

        if recall_count >= HEAT_PROMOTION_THRESHOLD:
            memory_id = metadata.get('id', 'unknown')

            metadata['type'] = 'core'
            metadata['promoted_at'] = datetime.now(timezone.utc).isoformat()
            metadata['promoted_reason'] = f'recall_count={recall_count} >= {HEAT_PROMOTION_THRESHOLD}'

            new_path = CORE_DIR / filepath.name
            write_memory_file(new_path, metadata, content)
            filepath.unlink()

            promoted.append(memory_id)
            print(f"Promoted to core: {memory_id} (recall_count={recall_count})")

    if promoted:
        print(f"Heat promotion: {len(promoted)} memories promoted to core")
    return promoted


# --- Decay event logging ---

def log_decay_event(decayed: int, pruned: int):
    """Log a decay event for stats tracking."""
    decay_file = MEMORY_ROOT / ".decay_history.json"
    history = {"sessions": []}
    if decay_file.exists():
        try:
            history = json.loads(decay_file.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, KeyError):
            pass

    history["sessions"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decayed": decayed,
        "pruned": pruned
    })

    history["sessions"] = history["sessions"][-20:]
    decay_file.write_text(json.dumps(history, indent=2), encoding='utf-8')


# --- Entity backfill (maintenance) ---

def backfill_entities(dry_run: bool = True) -> dict:
    """
    Backfill entities field for existing memories that don't have it.

    Args:
        dry_run: If True, just report what would be updated

    Returns:
        Stats dict with counts
    """
    stats = {'updated': 0, 'skipped': 0, 'already_has': 0}

    for directory in [CORE_DIR, ACTIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)

            if metadata.get('entities'):
                stats['already_has'] += 1
                continue

            detected = detect_entities(content, metadata.get('tags', []))

            if not detected:
                stats['skipped'] += 1
                continue

            if dry_run:
                print(f"Would update {filepath.name}: {detected}")
                stats['updated'] += 1
            else:
                metadata['entities'] = detected
                write_memory_file(filepath, metadata, content)
                stats['updated'] += 1

    return stats


# --- Activation scoring (v2.14) ---

def calculate_activation(metadata: dict) -> float:
    """
    Calculate memory activation score using exponential time decay.

    Inspired by SpindriftMend's Hebbian learning implementation.
    Formula: A(t) = A_0 * e^(-lambda*t)

    Returns:
        Activation score (0.0 to 1.0+, can exceed 1.0 for highly reinforced memories)
    """
    if not ACTIVATION_DECAY_ENABLED:
        return metadata.get('emotional_weight', 0.5)

    emotional_weight = metadata.get('emotional_weight', 0.5)

    recall_count = metadata.get('recall_count', 1)
    recall_bonus = math.log(recall_count + 1) / 5

    base_activation = emotional_weight + recall_bonus

    last_recalled_str = metadata.get('last_recalled')
    if last_recalled_str:
        try:
            if hasattr(last_recalled_str, 'isoformat'):
                last_recalled_str = last_recalled_str.isoformat()
            last_recalled = datetime.fromisoformat(str(last_recalled_str).replace('Z', '+00:00'))
            if last_recalled.tzinfo is None:
                last_recalled = last_recalled.replace(tzinfo=timezone.utc)
            hours_since_recall = (datetime.now(timezone.utc) - last_recalled).total_seconds() / 3600
        except (ValueError, TypeError):
            hours_since_recall = ACTIVATION_HALF_LIFE_HOURS
    else:
        hours_since_recall = ACTIVATION_HALF_LIFE_HOURS

    lambda_rate = math.log(2) / ACTIVATION_HALF_LIFE_HOURS
    decay_factor = math.exp(-lambda_rate * hours_since_recall)

    activation = base_activation * decay_factor

    min_floor = 0.1 if emotional_weight >= EMOTIONAL_WEIGHT_THRESHOLD else ACTIVATION_MIN_FLOOR

    return max(min_floor, round(activation, 4))


def get_most_activated_memories(limit: int = 10) -> list[tuple[str, float, dict]]:
    """
    Get the most activated memories (highest time-weighted activation).

    Returns:
        List of (memory_id, activation_score, metadata, preview) tuples
    """
    results = []

    for directory in [CORE_DIR, ACTIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            memory_id = metadata.get('id', filepath.stem)
            activation = calculate_activation(metadata)
            results.append((memory_id, activation, metadata, content[:100]))

    results.sort(key=lambda x: x[1], reverse=True)
    return [(r[0], r[1], r[2], r[3]) for r in results[:limit]]


# --- Retrieval outcomes (v2.13 self-evolution) ---

def log_retrieval_outcome(memory_id: str, outcome: str) -> bool:
    """
    Log the outcome of a memory retrieval for self-evolution.

    Outcomes:
    - "productive": Memory led to another recall or useful work
    - "generative": Memory led to creation of new memory (highest value)
    - "dead_end": Memory was recalled but nothing followed
    """
    valid_outcomes = {"productive", "generative", "dead_end"}
    if outcome not in valid_outcomes:
        print(f"Invalid outcome: {outcome}. Must be one of {valid_outcomes}")
        return False

    for directory in ALL_DIRS:
        if not directory.exists():
            continue
        for filepath in directory.glob(f"*-{memory_id}.md"):
            metadata, content = parse_memory_file(filepath)

            outcomes = metadata.get('retrieval_outcomes', {
                'productive': 0,
                'generative': 0,
                'dead_end': 0,
                'total': 0
            })

            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            outcomes['total'] = outcomes.get('total', 0) + 1
            metadata['retrieval_outcomes'] = outcomes

            total = outcomes['total']
            if total > 0:
                successes = outcomes.get('productive', 0) + (outcomes.get('generative', 0) * 2)
                metadata['retrieval_success_rate'] = round(successes / (total + outcomes.get('generative', 0)), 3)

            write_memory_file(filepath, metadata, content)
            return True

    return False


def get_retrieval_success_rate(memory_id: str) -> Optional[float]:
    """Get the retrieval success rate for a memory."""
    for directory in ALL_DIRS:
        if not directory.exists():
            continue
        for filepath in directory.glob(f"*-{memory_id}.md"):
            metadata, _ = parse_memory_file(filepath)
            return metadata.get('retrieval_success_rate')
    return None


def calculate_evolution_decay_multiplier(metadata: dict) -> float:
    """
    Calculate decay multiplier based on retrieval success (self-evolution).

    High success rate -> slower decay (memory is valuable)
    Low success rate -> faster decay (memory is noise)
    Insufficient data -> no adjustment (multiplier = 1.0)
    """
    if not SELF_EVOLUTION_ENABLED:
        return 1.0

    outcomes = metadata.get('retrieval_outcomes', {})
    total = outcomes.get('total', 0)

    if total < MIN_RETRIEVALS_FOR_EVOLUTION:
        return 1.0

    success_rate = metadata.get('retrieval_success_rate', 0.5)

    if success_rate >= SUCCESS_THRESHOLD:
        return SUCCESS_DECAY_BONUS
    elif success_rate <= FAILURE_THRESHOLD:
        return FAILURE_DECAY_PENALTY
    else:
        return 1.0


def auto_log_retrieval_outcomes() -> dict:
    """
    Automatically infer retrieval outcomes from session patterns.
    Call at session-end to update success rates.
    """
    retrieved = session_state.get_retrieved_list()

    if not retrieved:
        return {"productive": 0, "generative": 0, "dead_end": 0}

    results = {"productive": 0, "generative": 0, "dead_end": 0}

    for i, mem_id in enumerate(retrieved[:-1]):
        log_retrieval_outcome(mem_id, "productive")
        results["productive"] += 1

    today = datetime.now(timezone.utc).date().isoformat()
    for directory in [ACTIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, _ = parse_memory_file(filepath)
            created = metadata.get('created', '')
            if hasattr(created, 'isoformat'):
                created = created.isoformat()
            created = str(created)[:10]
            if created != today:
                continue

            caused_by = metadata.get('caused_by', [])
            for mem_id in retrieved:
                if mem_id in caused_by:
                    log_retrieval_outcome(mem_id, "generative")
                    results["generative"] += 1

    if retrieved:
        last_mem = retrieved[-1]
        for directory in [CORE_DIR, ACTIVE_DIR]:
            if not directory.exists():
                continue
            for filepath in directory.glob(f"*-{last_mem}.md"):
                metadata, _ = parse_memory_file(filepath)
                outcomes = metadata.get('retrieval_outcomes', {})
                if outcomes.get('generative', 0) == 0:
                    log_retrieval_outcome(last_mem, "dead_end")
                    results["dead_end"] += 1
                break

    return results
