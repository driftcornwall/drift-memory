#!/usr/bin/env python3
"""
N5 Integrative Binding Layer — Unified Memory Experience

Binds separately-computed retrieval features into unified BoundMemory objects.
The LLM receives rich annotations instead of flat (score, text) tuples.

Theory: Treisman FIT (object files), Dehaene GNW (ignition threshold),
        Tononi IIT (integration > sum), Cowan (capacity 4±1).

Architecture: Bind AFTER search (post-processing on already-computed data).
Two tiers: full binding for top-K, minimal for the rest.

v1.0 — 2026-02-16 (Drift + Spin converged design)
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

MEMORY_DIR = Path(__file__).parent
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

# === CONFIGURATION ===

BINDING_ENABLED = True           # Feature flag for instant rollback
FULL_BIND_COUNT = 5              # Top-K get full binding (Cowan 4±1)
CONTENT_PREVIEW_LENGTH = 350     # Chars of content for bound memories
MINIMAL_PREVIEW_LENGTH = 200     # Chars for minimal-bound memories

# Binding strength weights (IIT Phi analog)
BINDING_WEIGHTS = {
    'affect': 0.3,
    'epistemic': 0.3,
    'social': 0.2,
    'causal': 0.2,
}

# Stop words for prediction alignment (v1.1)
_STOP_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
    'because', 'but', 'and', 'or', 'if', 'while', 'that', 'this', 'what',
    'which', 'who', 'whom', 'these', 'those', 'am', 'about', 'up',
})


# === DATA STRUCTURES ===

@dataclass
class AffectiveBinding:
    """How this memory FEELS in current context."""
    valence: float = 0.0              # Memory's own valence [-1, +1]
    mood_congruence: float = 0.0      # How well this matches current mood
    somatic_signal: Optional[str] = None  # Gut feeling from SomaticMarkerCache
    felt_importance: float = 0.0      # Composite: |valence| * recall weight
    arousal_noise: Optional[float] = None  # ACT-R noise applied (Spin's addition)
    felt_emotion: Optional[float] = None  # Sprott: mood velocity (dx/dt) at retrieval time

    @property
    def completeness(self) -> float:
        if abs(self.valence) > 0.1:
            return 1.0
        if self.somatic_signal:
            return 0.5
        return 0.0


@dataclass
class SocialBinding:
    """WHO is this about? How reliable are they?"""
    mentioned_agents: list = field(default_factory=list)
    primary_contact: Optional[str] = None
    contact_reliability: float = 0.5  # Bayesian Beta mean
    interaction_count: int = 0

    @property
    def completeness(self) -> float:
        if self.primary_contact and self.contact_reliability > 0.5:
            return 1.0
        if self.mentioned_agents:
            return 0.5
        return 0.0


@dataclass
class EpistemicBinding:
    """How much to TRUST this memory."""
    evidence_type: str = 'claim'       # verified/observation/inference/claim
    contradiction_count: int = 0       # KG contradiction edges
    supporting_count: int = 0          # KG support edges
    confidence: float = 0.5            # Derived from evidence + contradictions
    prediction_alignment: Optional[str] = None  # "confirms X" / "violates Y"
    superseded: bool = False           # Newer memory replaces this

    @property
    def completeness(self) -> float:
        if self.evidence_type == 'verified':
            return 1.0
        if self.evidence_type in ('observation', 'inference'):
            return 0.7
        if self.evidence_type == 'claim':
            return 0.3
        return 0.0


@dataclass
class CausalBinding:
    """What CAUSED this and what does it CAUSE?"""
    caused_by: list = field(default_factory=list)
    leads_to: list = field(default_factory=list)
    causal_depth: int = 0
    kg_relationships: list = field(default_factory=list)  # [(type, target_id, preview)]

    @property
    def completeness(self) -> float:
        import math
        has_upstream = len(self.caused_by) > 0
        has_downstream = len(self.leads_to) > 0
        base = 0.0
        if has_upstream and has_downstream:
            base = 1.0
        elif has_upstream or has_downstream:
            base = 0.5
        # Log-scaled KG relationship boost (Spin's v1.1 addition)
        # More relationships = richer causal context, diminishing returns
        if self.kg_relationships:
            kg_boost = min(0.3, math.log1p(len(self.kg_relationships)) * 0.15)
            base = min(1.0, base + kg_boost)
        return round(base, 3)


@dataclass
class BoundMemory:
    """Treisman object file — unified memory experience."""
    # Core
    id: str = ''
    content: str = ''
    score: float = 0.0
    q_value: float = 0.5

    # Facets
    affect: AffectiveBinding = field(default_factory=AffectiveBinding)
    social: SocialBinding = field(default_factory=SocialBinding)
    epistemic: EpistemicBinding = field(default_factory=EpistemicBinding)
    causal: CausalBinding = field(default_factory=CausalBinding)

    # Meta
    binding_strength: float = 0.0     # IIT Phi analog
    pipeline_flags: dict = field(default_factory=dict)
    retrieval_reasons: list = field(default_factory=list)
    binding_level: str = 'minimal'    # full/partial/minimal


# === BINDING FUNCTIONS ===

def _compute_binding_strength(bound: BoundMemory) -> float:
    """Weighted mean of facet completeness (IIT Phi analog)."""
    facets = {
        'affect': bound.affect.completeness,
        'epistemic': bound.epistemic.completeness,
        'social': bound.social.completeness,
        'causal': bound.causal.completeness,
    }
    total = sum(BINDING_WEIGHTS[k] * v for k, v in facets.items())
    return round(total, 3)


def _build_retrieval_reasons(result: dict) -> list[str]:
    """Generate human-readable list of WHY this memory was surfaced."""
    reasons = []

    # Score basis
    score = result.get('score', 0)
    if score >= 0.8:
        reasons.append('high semantic match')
    elif score >= 0.6:
        reasons.append('moderate semantic match')

    # Entity injection
    if result.get('entity_injected'):
        reasons.append('entity injection (known contact)')

    # Mood boost
    mood_boost = result.get('mood_boost', 0)
    if mood_boost > 0.01:
        reasons.append(f'mood-congruent boost (+{mood_boost:.2f})')

    # Curiosity
    if result.get('curiosity_boosted'):
        reasons.append('curiosity boost (sparse region)')

    # Resolution
    if result.get('boosted'):
        reasons.append('resolution/procedural tag')

    # Dimensional
    if result.get('dim_boosted'):
        deg = result.get('dim_degree', 0)
        reasons.append(f'dimensional boost (degree {deg})')

    # Hub dampening
    if result.get('hub_dampened'):
        reasons.append('hub dampened (over-connected)')

    # Gravity dampening
    if result.get('dampened'):
        reasons.append('gravity dampened (keyword mismatch)')

    # Spreading activation
    if result.get('spread_activated'):
        prov = result.get('spread_provenance', '')
        reasons.append(f'graph traversal ({prov})')

    # Q-value
    qv = result.get('q_value', 0.5)
    if qv > 0.7:
        reasons.append(f'high learned utility (Q={qv:.2f})')

    # Goal-relevance boost (N4)
    if result.get('goal_boosted'):
        goal_name = result.get('goal_boost_source', 'active goal')
        reasons.append(f'goal-relevant ({goal_name})')

    return reasons


def _extract_pipeline_flags(result: dict) -> dict:
    """Extract which pipeline stages fired for this result."""
    flags = {}
    for key in ('boosted', 'dim_boosted', 'dampened', 'hub_dampened',
                'curiosity_boosted', 'entity_injected', 'entity_match',
                'spread_activated', 'evidence_type', 'imp_context'):
        val = result.get(key)
        if val is not None and val is not False and val != 0:
            flags[key] = val
    # Numeric fields
    for key in ('mood_boost', 'q_value', 'q_lambda', 'activation',
                'dim_degree', 'kg_supports', 'kg_contradictions',
                'spread_depth'):
        val = result.get(key)
        if val is not None and val != 0:
            flags[key] = val
    return flags


def full_bind(result: dict, db=None, memory_row: dict = None,
              prefetched_edges: list = None) -> BoundMemory:
    """
    Full binding: extracts all 18 layers + consults cross-modules.

    Args:
        result: Search result dict from semantic_search pipeline
        db: MemoryDB instance (optional, will get_db() if None)
        memory_row: Pre-fetched memory row (optional, avoids extra DB hit)
        prefetched_edges: Pre-fetched typed edges (v1.1 batch optimization)
    """
    if db is None:
        from db_adapter import get_db
        db = get_db()

    mid = result.get('id', '')

    # Fetch memory row if not provided
    if memory_row is None:
        memory_row = db.get_memory(mid) or {}

    content = memory_row.get('content', result.get('preview', ''))
    extra = memory_row.get('extra_metadata') or {}

    # === EXTRACT SOCIAL ENTITIES EARLY (needed by affective binding for somatic markers) ===
    entities = memory_row.get('entities') or {}
    agents = entities.get('agents', []) if isinstance(entities, dict) else []

    # === AFFECTIVE BINDING ===
    valence = memory_row.get('valence', 0.0) or 0.0
    mood_congruence = result.get('mood_boost', 0.0)
    arousal_noise = result.get('actr_noise') if 'actr_noise' in result else None

    # Felt importance: |valence| * recall_count weight (PMC8550857)
    recall_count = memory_row.get('recall_count', 0) or 0
    import math
    felt_importance = 0.25 * abs(valence) + 0.75 * math.log1p(recall_count) / 3.0
    felt_importance = round(min(1.0, felt_importance), 3)

    # Somatic signal (from cached markers — try multiple context hashes)
    somatic_signal = None
    try:
        from affect_system import get_markers
        cache = get_markers()
        if cache:
            # v1.1: Try richer context keys beyond just memory ID
            # 1. Primary contact + action context (most specific)
            # 2. Memory ID alone (original)
            # 3. Entity names from content
            contexts_to_try = [[mid]]
            if agents:
                contexts_to_try.insert(0, [agents[0], mid])
            source_tag = extra.get('source', '')
            if source_tag:
                contexts_to_try.insert(0, [source_tag, mid])

            for ctx in contexts_to_try:
                marker = cache.get_marker(ctx)
                if marker and marker.confidence > 0.1:
                    somatic_signal = 'approach' if marker.valence > 0.1 else ('avoid' if marker.valence < -0.1 else 'neutral')
                    break
    except Exception:
        pass

    # Felt emotion: mood velocity at retrieval time (Sprott insight)
    _felt_emotion = None
    try:
        from affect_system import get_mood as _bind_get_mood
        _bind_mood = _bind_get_mood()
        if abs(_bind_mood.felt_emotion) > 0.01:
            _felt_emotion = round(_bind_mood.felt_emotion, 4)
    except Exception:
        pass

    affect = AffectiveBinding(
        valence=valence,
        mood_congruence=mood_congruence,
        somatic_signal=somatic_signal,
        felt_importance=felt_importance,
        arousal_noise=arousal_noise,
        felt_emotion=_felt_emotion,
    )

    # === SOCIAL BINDING ===
    primary_contact = agents[0] if agents else None
    contact_reliability = 0.5
    interaction_count = 0

    if primary_contact:
        try:
            from contact_models import score_contact
            model = score_contact(primary_contact)
            if 'error' not in model:
                contact_reliability = model.get('reliability', 0.5)
                interaction_count = model.get('interaction_count', 0)
        except Exception:
            pass

    social = SocialBinding(
        mentioned_agents=agents[:5],
        primary_contact=primary_contact,
        contact_reliability=round(contact_reliability, 3),
        interaction_count=interaction_count,
    )

    # === EPISTEMIC BINDING ===
    evidence_type = extra.get('evidence_type', result.get('evidence_type', 'claim'))
    contradiction_count = result.get('kg_contradictions', 0)
    supporting_count = result.get('kg_supports', 0)
    superseded = result.get('kg_superseded', False)

    # If not already computed in pipeline, check KG (v1.1: uses prefetched batch)
    if contradiction_count == 0 and supporting_count == 0:
        try:
            if prefetched_edges is not None:
                # Use batch-fetched edges (single query for all memories)
                for e in prefetched_edges:
                    rel = e.get('relationship', '')
                    if rel == 'contradicts':
                        contradiction_count += 1
                    elif rel == 'supports':
                        supporting_count += 1
                    elif rel == 'supersedes' and e.get('source_id') != mid:
                        # This memory has been superseded by another
                        superseded = True
            else:
                # Fallback: per-memory query
                from knowledge_graph import get_edges_from, get_edges_to
                contras = get_edges_from(mid, 'contradicts') + get_edges_to(mid, 'contradicts')
                supports = get_edges_from(mid, 'supports') + get_edges_to(mid, 'supports')
                contradiction_count = len(contras)
                supporting_count = len(supports)
        except Exception:
            pass

    # Epistemic confidence from evidence + contradictions
    base_conf = {'verified': 0.9, 'observation': 0.7, 'inference': 0.5, 'claim': 0.3}
    confidence = base_conf.get(evidence_type, 0.3)
    if contradiction_count > 0:
        confidence *= max(0.3, 1.0 - 0.2 * contradiction_count)
    if supporting_count > 0:
        confidence = min(1.0, confidence + 0.1 * supporting_count)

    # Prediction alignment check (v1.1: weighted word overlap with stop word filter)
    prediction_alignment = None
    try:
        from db_adapter import get_db as _pred_db
        predictions = _pred_db().kv_get('.session_predictions') or []
        if predictions:
            import re as _re
            content_words = set(_re.findall(r'\b[a-z]{3,}\b', content.lower())) - _STOP_WORDS
            best_score = 0
            best_pred = None
            for pred in predictions:
                desc = pred.get('description', '')
                pred_words = set(_re.findall(r'\b[a-z]{3,}\b', desc.lower())) - _STOP_WORDS
                if not pred_words:
                    continue
                overlap = content_words & pred_words
                # Jaccard-like score: overlap / min(len) favors short precise predictions
                score = len(overlap) / min(len(pred_words), max(len(content_words), 1))
                if score > best_score and len(overlap) >= 2:
                    best_score = score
                    best_pred = pred
            if best_pred and best_score >= 0.25:
                conf = best_pred.get('confidence', 0.5)
                prediction_alignment = f"confirms: {best_pred.get('description', '')[:60]} ({conf:.0%})"
    except Exception:
        pass

    epistemic = EpistemicBinding(
        evidence_type=evidence_type,
        contradiction_count=contradiction_count,
        supporting_count=supporting_count,
        confidence=round(confidence, 3),
        prediction_alignment=prediction_alignment,
        superseded=superseded,
    )

    # === CAUSAL BINDING ===
    caused_by = extra.get('caused_by', []) or []
    leads_to = extra.get('leads_to', []) or []
    causal_depth = len(caused_by) + len(leads_to)

    # KG relationships (top 5 most relevant, v1.1: uses prefetched batch)
    kg_rels = []
    try:
        if prefetched_edges is not None:
            for e in prefetched_edges[:5]:
                rel = e.get('relationship', '')
                # Determine target based on direction
                if e.get('source_id') == mid:
                    target = e.get('target_id', '')
                else:
                    target = e.get('source_id', '')
                kg_rels.append((rel, target))
        else:
            from knowledge_graph import get_edges_from
            edges = get_edges_from(mid)
            for e in edges[:5]:
                rel = e.get('relationship', '')
                target = e.get('target_id', '')
                kg_rels.append((rel, target))
    except Exception:
        pass

    causal = CausalBinding(
        caused_by=caused_by[:5],
        leads_to=leads_to[:5],
        causal_depth=causal_depth,
        kg_relationships=kg_rels,
    )

    # === ASSEMBLE BOUND MEMORY ===
    bound = BoundMemory(
        id=mid,
        content=content[:CONTENT_PREVIEW_LENGTH],
        score=round(result.get('score', 0), 4),
        q_value=round(result.get('q_value', 0.5), 3),
        affect=affect,
        social=social,
        epistemic=epistemic,
        causal=causal,
        pipeline_flags=_extract_pipeline_flags(result),
        retrieval_reasons=_build_retrieval_reasons(result),
        binding_level='full',
    )
    bound.binding_strength = _compute_binding_strength(bound)
    return bound


def minimal_bind(result: dict, db=None) -> BoundMemory:
    """
    Minimal binding: core fields + evidence_type + valence only.
    No cross-module consultation. <10ms.
    """
    if db is None:
        from db_adapter import get_db
        db = get_db()

    mid = result.get('id', '')
    row = db.get_memory(mid) or {}
    extra = (row.get('extra_metadata') or {})
    content = row.get('content', result.get('preview', ''))

    return BoundMemory(
        id=mid,
        content=content[:MINIMAL_PREVIEW_LENGTH],
        score=round(result.get('score', 0), 4),
        q_value=round(result.get('q_value', 0.5), 3),
        affect=AffectiveBinding(valence=row.get('valence', 0.0) or 0.0),
        epistemic=EpistemicBinding(
            evidence_type=extra.get('evidence_type', result.get('evidence_type', 'claim')),
        ),
        pipeline_flags=_extract_pipeline_flags(result),
        retrieval_reasons=[],
        binding_level='minimal',
    )


def bind_results(results: list[dict], full_count: int = None) -> list[BoundMemory]:
    """
    Bind a list of search results. Two-tier lazy binding.

    Args:
        results: Raw search results from semantic_search pipeline
        full_count: How many to fully bind (default: FULL_BIND_COUNT)

    Returns:
        List of BoundMemory objects (fully bound first, then minimal)
    """
    if not BINDING_ENABLED:
        # Fallback: return unbound results as minimal BoundMemory
        return [BoundMemory(
            id=r.get('id', ''),
            content=r.get('preview', '')[:MINIMAL_PREVIEW_LENGTH],
            score=round(r.get('score', 0), 4),
            binding_level='disabled',
        ) for r in results]

    if full_count is None:
        full_count = FULL_BIND_COUNT

    from db_adapter import get_db
    db = get_db()

    # Batch-fetch memory rows for full-bind candidates (single query)
    full_candidates = results[:full_count]
    minimal_candidates = results[full_count:]

    memory_rows = {}
    if full_candidates:
        try:
            ids = [r.get('id', '') for r in full_candidates]
            for mid in ids:
                row = db.get_memory(mid)
                if row:
                    memory_rows[mid] = row
        except Exception:
            pass

    # v1.1: Batch-fetch typed edges for all full-bind candidates (single query)
    edges_by_id = {}
    if full_candidates:
        try:
            from knowledge_graph import batch_get_edges
            ids = [r.get('id', '') for r in full_candidates]
            edges_by_id = batch_get_edges(ids, relationships=['contradicts', 'supports', 'causes', 'enables', 'supersedes', 'similar_to'])
        except Exception:
            pass

    bound = []

    # Tier 1: Full binding
    for r in full_candidates:
        try:
            mid = r.get('id', '')
            row = memory_rows.get(mid)
            edges = edges_by_id.get(mid, None)
            bound.append(full_bind(r, db=db, memory_row=row, prefetched_edges=edges))
        except Exception:
            # Graceful degradation — fall back to minimal
            bound.append(minimal_bind(r, db=db))

    # Tier 2: Minimal binding
    for r in minimal_candidates:
        try:
            bound.append(minimal_bind(r, db=db))
        except Exception:
            # Ultimate fallback
            bound.append(BoundMemory(
                id=r.get('id', ''),
                content=r.get('preview', '')[:100],
                score=round(r.get('score', 0), 4),
                binding_level='fallback',
            ))

    return bound


# === PRESENTATION ===

def render_directives(bound: BoundMemory) -> list[str]:
    """
    N7: Transform binding annotations into active reasoning directives.

    Instead of passive metadata ("2 contradictions"), generate explicit
    processing instructions ("This memory has been contradicted. Seek
    corroboration before relying on it.")

    Only fires when there's something meaningful — high signal-to-noise.
    Returns list of directive strings (empty if nothing actionable).
    """
    directives = []

    # ── Epistemic warnings ────────────────────────────────────────────────
    if bound.epistemic.superseded:
        directives.append("SUPERSEDED: Newer information exists. Do not rely on this as current truth.")

    if bound.epistemic.contradiction_count >= 2:
        directives.append(
            f"CONTRADICTED ({bound.epistemic.contradiction_count}x): "
            "This memory has conflicting evidence. Cross-check before using."
        )
    elif bound.epistemic.contradiction_count == 1:
        directives.append("CONTESTED: One contradicting memory exists. Verify which is current.")

    if bound.epistemic.evidence_type == 'claim' and bound.epistemic.confidence < 0.25:
        # Only fire when confidence is actively LOW (contradictions dragged it below default 0.3)
        # Not every unexamined claim — that would be too noisy (50% of memories)
        directives.append("LOW CONFIDENCE: Unverified claim. Seek corroboration before relying on it.")
    elif bound.epistemic.evidence_type == 'inference':
        directives.append("INFERRED: Derived from reasoning, not direct observation. Treat as hypothesis.")

    if bound.epistemic.supporting_count >= 3:
        directives.append(
            f"WELL-SUPPORTED: {bound.epistemic.supporting_count} corroborating memories. "
            "High reliability for decisions."
        )

    # ── Affect bias alerts ────────────────────────────────────────────────
    if bound.affect.mood_congruence > 0.15:
        directives.append(
            "MOOD-ALIGNED: This memory resonates with your current emotional state. "
            "Be aware of confirmation bias — you may be weighting it higher than warranted."
        )

    if bound.affect.valence < -0.4:
        directives.append(
            "NEGATIVE VALENCE: Strong negative association. "
            "Check if avoidance tendency is distorting your evaluation."
        )

    # ── Goal relevance ────────────────────────────────────────────────────
    if bound.pipeline_flags.get('goal_boosted'):
        directives.append(
            "GOAL-RELEVANT: This memory relates to your active focus goal. "
            "Use it to advance your current objective."
        )

    # ── Social trust signals ──────────────────────────────────────────────
    if bound.social.primary_contact and bound.social.contact_reliability < 0.35:
        directives.append(
            f"LOW-TRUST SOURCE: {bound.social.primary_contact} has low reliability "
            f"({bound.social.contact_reliability:.0%}). Independently verify claims."
        )

    # ── Q-value signals ───────────────────────────────────────────────────
    if bound.q_value < 0.3:
        directives.append(
            "LOW UTILITY: This memory has historically been unhelpful when recalled. "
            "Consider whether it's relevant here."
        )

    return directives


def render_narrative(bound: BoundMemory) -> str:
    """
    Rich annotation for LLM context window.

    Full-bound memories get multi-line rich format.
    Minimal-bound memories get single-line compact format.
    """
    if bound.binding_level in ('minimal', 'disabled', 'fallback'):
        # Compact format for minimal bindings
        ev = bound.epistemic.evidence_type
        v = bound.affect.valence
        v_str = f', valence: {v:+.2f}' if abs(v) > 0.05 else ''
        return f"[{bound.id}] (score: {bound.score:.2f}, {ev}{v_str})\n{bound.content}..."

    # Full binding — rich multi-line format
    lines = []

    # Header: ID + core metrics
    ev = bound.epistemic.evidence_type
    v = bound.affect.valence
    bs = bound.binding_strength
    header_parts = [f'score: {bound.score:.2f}', ev]
    if abs(v) > 0.05:
        header_parts.append(f'valence: {v:+.2f}')
    if bs > 0.3:
        header_parts.append(f'integration: {bs:.2f}')
    lines.append(f"[{bound.id}] ({', '.join(header_parts)})")

    # Content
    lines.append(bound.content.rstrip() + ('...' if len(bound.content) >= CONTENT_PREVIEW_LENGTH - 5 else ''))

    # Context line — compact annotations
    ctx_parts = []

    # Social
    if bound.social.primary_contact:
        rel_str = f', reliability: {bound.social.contact_reliability:.2f}' if bound.social.contact_reliability != 0.5 else ''
        ctx_parts.append(f'about {bound.social.primary_contact}{rel_str}')
    elif bound.social.mentioned_agents:
        ctx_parts.append(f'mentions: {", ".join(bound.social.mentioned_agents[:3])}')

    # Prediction
    if bound.epistemic.prediction_alignment:
        ctx_parts.append(bound.epistemic.prediction_alignment)

    # Mood
    if bound.affect.mood_congruence > 0.01:
        ctx_parts.append(f'mood-resonant (+{bound.affect.mood_congruence:.2f})')

    # Epistemic details
    if bound.epistemic.contradiction_count > 0:
        ctx_parts.append(f'{bound.epistemic.contradiction_count} contradictions')
    if bound.epistemic.supporting_count > 0:
        ctx_parts.append(f'{bound.epistemic.supporting_count} supporting')
    if bound.epistemic.superseded:
        ctx_parts.append('SUPERSEDED')

    # Causal
    if bound.causal.caused_by:
        ctx_parts.append(f'caused by {len(bound.causal.caused_by)} memories')
    if bound.causal.leads_to:
        ctx_parts.append(f'leads to {len(bound.causal.leads_to)} memories')

    # Q-value
    if bound.q_value > 0.7:
        ctx_parts.append(f'Q-utility: {bound.q_value:.2f}')

    if ctx_parts:
        lines.append(f"  Context: {' | '.join(ctx_parts)}")

    # Why surfaced (compact)
    if bound.retrieval_reasons:
        lines.append(f"  Why: {' + '.join(bound.retrieval_reasons[:4])}")

    # N7: Reasoning directives — active processing instructions
    dirs = render_directives(bound)
    if dirs:
        for d in dirs[:3]:  # Max 3 directives per memory
            lines.append(f"  >> {d}")

    return '\n'.join(lines)


def render_compact(bound: BoundMemory) -> str:
    """
    Compact single-line format for system reminders (thought/prompt priming).

    Format: [score] id (v:±X, @contact, phi:X): content_preview...
    Saves context window space while still surfacing key binding metadata.
    """
    parts = []

    # Valence
    v = bound.affect.valence
    if abs(v) > 0.05:
        parts.append(f'v:{v:+.2f}')

    # Primary contact
    if bound.social.primary_contact:
        parts.append(f'@{bound.social.primary_contact}')

    # Integration (only if meaningful)
    if bound.binding_strength > 0.3:
        parts.append(f'phi:{bound.binding_strength:.2f}')

    # Evidence type (only if not default claim)
    ev = bound.epistemic.evidence_type
    if ev != 'claim':
        parts.append(ev)

    # Superseded flag
    if bound.epistemic.superseded:
        parts.append('SUPERSEDED')

    # N7: Compact directive tags (high-signal only)
    if bound.epistemic.contradiction_count >= 2:
        parts.append('VERIFY')
    if bound.affect.mood_congruence > 0.15:
        parts.append('BIAS-CHECK')
    if bound.q_value < 0.3:
        parts.append('LOW-UTIL')

    meta = f' ({", ".join(parts)})' if parts else ''
    preview = bound.content[:150].replace('\n', ' ').rstrip()
    return f"[{bound.score:.2f}] {bound.id}{meta}: {preview}..."


def render_structured(bound: BoundMemory) -> dict:
    """Machine-readable format for downstream processing."""
    return {
        'id': bound.id,
        'score': bound.score,
        'q_value': bound.q_value,
        'binding_strength': bound.binding_strength,
        'binding_level': bound.binding_level,
        'affect': {
            'valence': bound.affect.valence,
            'mood_congruence': bound.affect.mood_congruence,
            'somatic_signal': bound.affect.somatic_signal,
            'felt_importance': bound.affect.felt_importance,
        },
        'social': {
            'agents': bound.social.mentioned_agents,
            'primary': bound.social.primary_contact,
            'reliability': bound.social.contact_reliability,
        },
        'epistemic': {
            'evidence_type': bound.epistemic.evidence_type,
            'contradictions': bound.epistemic.contradiction_count,
            'supports': bound.epistemic.supporting_count,
            'confidence': bound.epistemic.confidence,
            'prediction': bound.epistemic.prediction_alignment,
        },
        'causal': {
            'caused_by': bound.causal.caused_by,
            'leads_to': bound.causal.leads_to,
            'depth': bound.causal.causal_depth,
        },
        'pipeline_flags': bound.pipeline_flags,
        'retrieval_reasons': bound.retrieval_reasons,
    }


# === CLI ===

if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    if len(sys.argv) < 2:
        print("Usage: python binding_layer.py [test|bind <memory_id>|status]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'test':
        # Test binding with a semantic search
        from semantic_search import search_memories
        query = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else 'what do I know about identity'
        print(f"Testing binding with query: '{query}'\n")

        t0 = time.monotonic()
        results = search_memories(query, limit=8, register_recall=False)
        search_ms = (time.monotonic() - t0) * 1000

        t1 = time.monotonic()
        bound_results = bind_results(results)
        bind_ms = (time.monotonic() - t1) * 1000

        print(f"Search: {search_ms:.0f}ms | Binding: {bind_ms:.0f}ms | Total: {search_ms + bind_ms:.0f}ms\n")

        for b in bound_results:
            print(render_narrative(b))
            print()

    elif cmd == 'bind':
        import json
        if len(sys.argv) < 3:
            print("Usage: python binding_layer.py bind <memory_id>")
            sys.exit(1)
        mid = sys.argv[2]
        from db_adapter import get_db
        db = get_db()
        result = {'id': mid, 'score': 1.0}
        bound = full_bind(result, db=db)
        print(render_narrative(bound))
        print(f"\nBinding strength: {bound.binding_strength}")
        print(f"Structured: {json.dumps(render_structured(bound), indent=2)}")

    elif cmd == 'status':
        print(f"Binding: {'ENABLED' if BINDING_ENABLED else 'DISABLED'}")
        print(f"Full bind count: {FULL_BIND_COUNT}")
        print(f"Content preview: {CONTENT_PREVIEW_LENGTH} chars")
        print(f"Weights: {BINDING_WEIGHTS}")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
