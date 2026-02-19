#!/usr/bin/env python3
"""
Semantic Search for Drift's Memory System

Enables natural language queries like "what do I know about bounties?"
instead of requiring exact memory IDs.

Supports:
- OpenAI embeddings (requires OPENAI_API_KEY)
- Local models via HTTP endpoint (for Docker-based free option)

Usage:
    python semantic_search.py index          # Build/rebuild index
    python semantic_search.py search "query" # Search memories
    python semantic_search.py status         # Check index status
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional
import math

MEMORY_DIR = Path(__file__).parent

# Embedding dimensions vary by model:
# - OpenAI text-embedding-3-small: 1536
# - Qwen3-Embedding-8B: 4096
# We don't enforce dimension - just compare what we have

# Ablation hooks — set by ablation_framework.py StageDisabler, never by production code
_ABLATION_SKIP_GRAVITY = False
_ABLATION_SKIP_CURIOSITY = False


def get_embedding_openai(text: str, model: str = "text-embedding-3-small") -> Optional[list[float]]:
    """Get embedding from OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        import urllib.request
        import urllib.error

        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = json.dumps({
            "input": text[:8000],  # Truncate to avoid token limits
            "model": model
        }).encode('utf-8')

        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result['data'][0]['embedding']
    except Exception as e:
        print(f"OpenAI embedding error: {e}", file=sys.stderr)
        return None


def get_embedding_local(text: str, endpoint: str = "http://localhost:8080/embed") -> Optional[list[float]]:
    """
    Get embedding from local model endpoint.
    Supports both TEI format ({"inputs": "..."}) and generic format ({"text": "..."}).
    """
    try:
        import urllib.request

        # Try TEI format first (Hugging Face text-embeddings-inference)
        data = json.dumps({"inputs": text[:4000]}).encode('utf-8')
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            # TEI returns [[0.1, 0.2, ...]] for single input
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    return result[0]
                return result
            # Generic format
            return result.get('embedding') or result.get('embeddings', [[]])[0]
    except Exception as e:
        return None


def get_embedding(text: str) -> Optional[list[float]]:
    """
    Get embedding using best available method.
    Priority: Local (free) > OpenAI (paid)
    """
    # Check for local endpoint first (free, private)
    local_endpoint = os.getenv("LOCAL_EMBEDDING_ENDPOINT", "").strip()
    if not local_endpoint:
        # Default to localhost if docker service might be running
        local_endpoint = "http://localhost:8080/embed"

    # Try local first
    emb = get_embedding_local(text, local_endpoint)
    if emb:
        return emb

    # Fall back to OpenAI if local unavailable
    return get_embedding_openai(text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)




def _is_noise_content(content: str) -> bool:
    """
    Detect noise memories: raw API responses, notification dumps, session logs.
    Used by hub dampening and spreading activation to filter low-quality candidates.
    (Spin's spreading activation noise fix, adapted for shared use.)
    """
    if not content or len(content) < 80:
        return True
    c = content.strip()
    # Raw JSON / API responses
    if c.startswith('{') or c.startswith('['):
        return True
    # Section headers / session logs
    if c.startswith('===') or c.startswith('---'):
        return True
    # Platform dumps (GitHub Notifications, Colony Feed, etc.)
    noise_prefixes = ('GitHub Notifications', 'Colony Feed', 'Global feed status',
                      'thought-', 'session-', 'Session ')
    if any(c.startswith(p) for p in noise_prefixes):
        return True
    # Very short after stripping
    if len(c.split()) < 10:
        return True
    return False


def detect_embedding_source() -> str:
    """Detect which embedding source will be used."""
    local_endpoint = os.getenv("LOCAL_EMBEDDING_ENDPOINT", "http://localhost:8080/embed")
    try:
        import urllib.request
        req = urllib.request.Request(f"{local_endpoint.rsplit('/embed', 1)[0]}/info", method='GET')
        with urllib.request.urlopen(req, timeout=5) as response:
            info = json.loads(response.read().decode('utf-8'))
            return info.get('model_id', 'local-unknown')
    except:
        pass

    if os.getenv("OPENAI_API_KEY"):
        return "openai/text-embedding-3-small"
    return "unknown"


def index_memories(force: bool = False) -> dict:
    """
    Index all memories by generating embeddings. DB-only.

    Reads memories from PostgreSQL, generates embeddings, stores back to DB.

    Args:
        force: If True, re-index all memories. Otherwise, only index unembedded ones.

    Returns:
        Summary of indexing results.
    """
    from db_adapter import get_db
    import psycopg2.extras

    db = get_db()
    model_source = detect_embedding_source()

    # Get all memories from DB
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT id, content FROM {db._table('memories')} WHERE type IN ('core', 'active')")
            all_memories = cur.fetchall()

    # Get already-embedded IDs (unless forcing)
    existing = set()
    if not force:
        with db._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT memory_id FROM {db._table('text_embeddings')}")
                existing = {row[0] for row in cur.fetchall()}

    stats = {"indexed": 0, "skipped": 0, "failed": 0, "total": len(all_memories)}

    for row in all_memories:
        memory_id = row['id']
        content = row.get('content', '')
        if not memory_id or not content:
            stats["failed"] += 1
            continue

        if memory_id in existing and not force:
            stats["skipped"] += 1
            continue

        # Apply vocabulary bridge before embedding
        try:
            from vocabulary_bridge import bridge_content
            bridged = bridge_content(content)
        except ImportError:
            bridged = content

        # Generate embedding and store to DB
        embedding = get_embedding(bridged)
        if embedding:
            db.store_embedding(
                memory_id=memory_id,
                embedding=embedding,
                preview=content[:200],
                model=model_source,
            )
            stats["indexed"] += 1
            print(f"  Indexed: {memory_id}")
        else:
            stats["failed"] += 1
            print(f"  Failed: {memory_id}")

    return stats


def load_memory_tags(memory_id: str) -> list[str]:
    """Load tags from DB."""
    from db_adapter import get_db
    row = get_db().get_memory(memory_id)
    if row and row.get('tags'):
        return row['tags']
    return []


# Resolution/procedural tags that indicate actionable knowledge
RESOLUTION_TAGS = {'resolution', 'procedural', 'fix', 'solution', 'howto', 'api', 'endpoint'}
RESOLUTION_BOOST = 1.25  # 25% score boost for resolution memories
DIMENSION_BOOST_SCALE = 0.1  # Dimensional connectivity boost: score *= (1 + 0.1 * log(1+degree))


def search_memories(query: str, limit: int = 5, threshold: float = 0.3,
                    register_recall: bool = True,
                    dimension: str = None, sub_view: str = None,
                    skip_monologue: bool = False) -> list[dict]:
    """
    Search memories by semantic similarity with resolution + dimensional boosting.

    v2.20: Uses pgvector when available for O(1) indexed search instead of
    loading all embeddings into Python.

    Args:
        query: Natural language query
        limit: Maximum results to return
        threshold: Minimum similarity score (0-1)
        register_recall: If True, register results as "recalled" for decay system
        dimension: Optional W-dimension to boost by (who/what/why/where)
        sub_view: Optional sub-view within dimension (e.g. topic name)

    Returns:
        List of matching memories with scores.
    """
    # === ATTENTION SCHEMA TIMING ===
    import time as _time
    _attention_stages = []
    _t0 = _time.monotonic()

    # === PER-STAGE Q-LEARNING: Initialize tracker + learner ===
    _stage_tracker = None
    _stage_learner = None
    _query_type = 'what'
    try:
        from stage_q_learning import StageTracker, get_learner, classify_query
        _query_type = classify_query(query)
        _stage_learner = get_learner()
    except Exception:
        pass  # Stage Q-learning is optional

    # Bidirectional vocabulary bridge
    _ts = _time.monotonic()
    try:
        from vocabulary_bridge import bridge_query
        bridged_query = bridge_query(query)
    except ImportError:
        bridged_query = query
    _attention_stages.append({'stage': 'vocab_bridge', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})

    # Get query embedding
    _ts = _time.monotonic()
    query_embedding = get_embedding(bridged_query)
    _attention_stages.append({'stage': 'query_embedding', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})
    if not query_embedding:
        print("Failed to get query embedding", file=sys.stderr)
        return []

    # === DYNAMIC THRESHOLD ADJUSTMENT (N1 v1.2: 3-source stacking) ===
    # Sources: cognitive_state + affect_system + adaptive_behavior
    _cog_modifier = 0.0
    _affect_modifier = 0.0
    _adapt_modifier = 0.0
    try:
        from cognitive_state import get_search_threshold_modifier, process_event
        _cog_modifier = get_search_threshold_modifier()
    except Exception:
        pass
    try:
        from affect_system import get_search_threshold_modifier as _affect_thresh
        _affect_modifier = _affect_thresh()
    except Exception:
        pass
    try:
        from adaptive_behavior import get_adaptation
        _adapt_modifier = get_adaptation('curiosity_threshold_offset', 0.0)
    except Exception:
        pass
    # BUG-4 fix: System 2 effectiveness (Yerkes-Dodson) — extreme arousal degrades
    # analytical precision, broadening search (lower threshold)
    _s2_modifier = 0.0
    try:
        from affect_system import get_mood as _s2_mood
        _s2 = _s2_mood()
        s2_eff = _s2.get_system2_effectiveness()  # 0-1, peaks at arousal=0.5
        if s2_eff < 0.8:  # Only adjust when S2 is meaningfully degraded
            _s2_modifier = -(0.8 - s2_eff) * 0.05  # Up to -0.04 threshold reduction
    except Exception:
        pass
    _total_modifier = _cog_modifier + _affect_modifier + _adapt_modifier + _s2_modifier
    threshold = max(0.1, min(0.6, threshold + _total_modifier))

    results = []

    # Start explanation
    try:
        from explanation import ExplanationBuilder
        _expl = ExplanationBuilder('semantic_search', 'search')
        _expl.set_inputs({
            'query': query[:200],
            'bridged_query': bridged_query[:200] if bridged_query != query else '(unchanged)',
            'limit': limit,
            'threshold': threshold,
            'dimension': dimension,
            'sub_view': sub_view,
            'cognitive_threshold_modifier': _cog_modifier,
            'affect_threshold_modifier': _affect_modifier,
            'adaptive_threshold_modifier': _adapt_modifier,
            'total_threshold_modifier': _total_modifier,
        })
    except Exception:
        _expl = None

    # === PREDICTION GENERATION (T4.1: Predictive Coding) ===
    # Before pgvector, generate expectations about which memories should appear.
    # Prediction error (surprising results) gets boosted after pgvector returns.
    # Note: _sq_before/after not available yet (defined after pgvector), so manual timing.
    _ts = _time.monotonic()
    _prediction_set = None
    try:
        from retrieval_prediction import generate_predictions, PREDICTION_ENABLED
        if PREDICTION_ENABLED:
            import session_state as _pred_ss
            _pred_ss.load()
            _prediction_set = generate_predictions(
                query=bridged_query,
                recent_recalls=_pred_ss.get_retrieved_list()[-10:],
                dimension=dimension,
            )
    except Exception:
        pass
    _attention_stages.append({
        'stage': 'prediction_generation',
        'time_ms': round((_time.monotonic() - _ts) * 1000, 1),
        'predictions': len(_prediction_set.predicted_ids) if _prediction_set else 0,
    })

    # pgvector search — DB-only, no file fallback
    _ts = _time.monotonic()
    from db_adapter import get_db
    db = get_db()
    rows = db.search_embeddings(query_embedding, limit=limit * 3)
    for row in rows:
        score = float(row.get('similarity', 0))
        if score >= threshold:
            results.append({
                "id": row['id'],
                "score": score,
                "preview": (row.get('preview') or row.get('content', ''))[:150],
                "path": f"db://{row.get('type', 'active')}/{row['id']}.md"
            })
    _attention_stages.append({'stage': 'pgvector_search', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})

    # BUG-22: Store original score for cumulative bias cap
    for r in results:
        r['original_score'] = r['score']

    if _expl:
        _expl.add_step('pgvector_candidates', len(results), weight=1.0,
                        context=f'{len(rows)} raw rows, {len(results)} above threshold {threshold}')

    # === PER-STAGE Q-LEARNING: Initialize tracker now that results exist ===
    if results and _stage_learner is not None:
        try:
            from stage_q_learning import StageTracker
            _stage_tracker = StageTracker(results)
        except Exception:
            pass

    # T1.2 (from Spin): Incremental bias cap enforcement after EACH bias-adding stage.
    # Without this, scores grow unbounded through stacked boosts (entity 2x * resolution 1.25x
    # * goal 1.3x * mood 1.15x = 3.74x), distorting intermediate sort orders.
    MAX_CUMULATIVE_BIAS = 3.0
    EXCESS_COMPRESSION = 0.3  # Compress excess beyond cap to 30% (soft cap, not hard cutoff)

    def _enforce_bias_cap():
        """Cap scores after each bias-adding stage. Dampening stages don't need this."""
        for r in results:
            original = r.get('original_score', r['score'])
            if original <= 0:
                continue
            max_allowed = original * MAX_CUMULATIVE_BIAS
            if r['score'] > max_allowed:
                excess = r['score'] - max_allowed
                r['score'] = max_allowed + excess * EXCESS_COMPRESSION
                r['score_capped_incremental'] = True

    # Helper: track stage before/after and check Q-based skip
    def _sq_before(stage_name):
        if _stage_tracker:
            _stage_tracker.before(stage_name)

    def _sq_after(stage_name):
        if _stage_tracker:
            _stage_tracker.after(stage_name)

    def _sq_should_skip(stage_name):
        if _stage_learner:
            return _stage_learner.should_skip(stage_name, _query_type)
        return False

    # === SOMATIC MARKER FAST-PATH (FINDING-20: System 1 pre-analytical bias) ===
    # Runs BEFORE analytical stages. Somatic markers provide fast gut-feeling
    # pre-screening: strongly-marked contexts get an early score shift.
    # This is the closest to genuine System 1 processing: a fast, pre-attentive
    # signal that biases results before the 10+ analytical stages run.
    # DENSITY GATE: Skip if < 15 somatic markers cached (saves ~20ms when sparse)
    _ts = _time.monotonic()
    _somatic_applied = 0
    _sq_before('somatic_prefilter')
    try:
        from affect_system import get_somatic_bias, get_markers as _sm_get_markers
        _sm_count = len(_sm_get_markers())
        if _sm_count < 15:
            raise ImportError(f'Density gate: {_sm_count} somatic markers < 15 threshold')
        for result in results:
            bias = get_somatic_bias(result['id'])
            if abs(bias) > 0.0001:  # T2.3: low threshold — weak but real signals
                result['score'] *= (1.0 + bias)
                result['somatic_bias'] = round(bias, 6)
                _somatic_applied += 1
    except (ImportError, Exception):
        pass  # Somatic markers optional or density gate
    if _expl and _somatic_applied:
        _expl.add_step('somatic_prefilter', _somatic_applied, weight=0.1,
                       context=f'{_somatic_applied} results with somatic marker bias (System 1)')
    _sq_after('somatic_prefilter')
    _attention_stages.append({'stage': 'somatic_prefilter', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})
    _enforce_bias_cap()

    # === PREDICTION SURPRISE BOOST (T4.1: Predictive Coding) ===
    # Compare pgvector results against predictions. Boost surprisingly relevant
    # results (high similarity but NOT predicted = enlightenment surprise).
    _ts = _time.monotonic()
    _sq_before('prediction_surprise')
    _surprise_boosts = 0
    if _prediction_set and _prediction_set.predicted_ids:
        predicted_ids = _prediction_set.predicted_ids
        result_ids = {r['id'] for r in results}
        SURPRISE_WEIGHT = 0.15
        MIN_SURPRISE = 0.3
        _pred_mean_conf = sum(predicted_ids.values()) / max(1, len(predicted_ids))

        for r in results:
            if r['id'] in predicted_ids:
                # Confirmed prediction — mark for learning, no boost
                r['prediction_status'] = 'confirmed'
                r['prediction_confidence'] = predicted_ids[r['id']]
            else:
                # Not predicted — compute surprise boost
                error = r.get('score', 0.0)
                if error > MIN_SURPRISE and _pred_mean_conf > 0.1:
                    boost = error * _pred_mean_conf * SURPRISE_WEIGHT
                    r['score'] += boost
                    r['prediction_status'] = 'surprising'
                    r['surprise_boost'] = round(boost, 4)
                    _surprise_boosts += 1
                else:
                    r['prediction_status'] = 'neutral'

        # Track prediction failures for Rescorla-Wagner learning
        for pid, conf in predicted_ids.items():
            if pid not in result_ids:
                try:
                    from retrieval_prediction import record_prediction_failure
                    record_prediction_failure(pid, conf, _prediction_set.query_hash)
                except Exception:
                    pass
    _sq_after('prediction_surprise')
    _attention_stages.append({
        'stage': 'prediction_surprise',
        'time_ms': round((_time.monotonic() - _ts) * 1000, 1),
        'surprises': _surprise_boosts,
    })
    _enforce_bias_cap()
    if _expl and _surprise_boosts:
        _expl.add_step('prediction_surprise', _surprise_boosts, weight=0.15,
                       context=f'{_surprise_boosts} results boosted by prediction error (enlightenment surprise)')

    # === ENTITY INDEX INJECTION (Fix for WHO dimension) ===
    # When query mentions a known contact, inject their memories into candidates
    # This bridges the gap between contact names and memory embeddings
    _ts = _time.monotonic()
    _sq_before('entity_injection')
    try:
        from entity_index import get_memories_for_query, detect_contacts
        entity_mem_ids = get_memories_for_query(query)
        if entity_mem_ids:
            existing_ids = {r["id"] for r in results}
            injected = 0
            for mem_id in entity_mem_ids[:10]:  # Cap at 10 injected
                if mem_id not in existing_ids:
                    # Load this memory's embedding from DB and compute actual similarity
                    mem_score = threshold  # Default to threshold
                    preview = ""
                    path = ""
                    try:
                        emb_row = db.get_embedding(mem_id) if hasattr(db, 'get_embedding') else None
                        if emb_row and emb_row.get('embedding'):
                            mem_score = cosine_similarity(query_embedding, emb_row['embedding'])
                            preview = (emb_row.get('preview') or '')[:150]
                        else:
                            # No embedding — get preview from memory content
                            mem_row = db.get_memory(mem_id)
                            if mem_row:
                                preview = (mem_row.get('content') or '')[:150]
                    except Exception:
                        pass

                    # Strong boost for entity-matched memories (2x)
                    # These are KNOWN contacts mentioned in the query — high confidence
                    boosted_score = max(mem_score * 2.0, threshold + 0.3)
                    results.append({
                        "id": mem_id,
                        "score": boosted_score,
                        "preview": preview,
                        "path": path,
                        "entity_injected": True,
                    })
                    injected += 1
                else:
                    # Strong boost for existing results matching entity (1.8x)
                    for r in results:
                        if r["id"] == mem_id:
                            r["score"] *= 1.8
                            r["entity_match"] = True
                            break
    except ImportError:
        pass
    except Exception:
        pass
    _sq_after('entity_injection')
    _attention_stages.append({'stage': 'entity_injection', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})
    _enforce_bias_cap()

    if _expl:
        injected_count = sum(1 for r in results if r.get('entity_injected'))
        if injected_count:
            _expl.add_step('entity_injection', injected_count, weight=0.5,
                           context=f'{injected_count} memories injected from entity index')

    # === MOOD-CONGRUENT SCORING (N1: Affect System) ===
    # Memories whose valence matches current mood get a retrieval boost.
    # Faul & LaBar (2023): ~15% mood-congruent retrieval bias.
    # Forgas (1995): stronger for exploratory (generative) searches.
    _ts = _time.monotonic()
    _sq_before('mood_congruent')
    try:
        from affect_system import get_mood
        mood = get_mood()
        if abs(mood.valence) > 0.05:  # Only apply when mood is non-neutral
            mood_boosted = 0
            for result in results:
                mem_row = db.get_memory(result['id'])
                if mem_row:
                    mem_valence = mem_row.get('valence', 0.0) or 0.0
                    boost = mood.get_retrieval_boost(mem_valence)
                    if boost != 0.0:
                        result['score'] += boost
                        result['mood_boost'] = round(boost, 4)
                        mood_boosted += 1
            if _expl and mood_boosted:
                _expl.add_step('mood_congruent', mood_boosted, weight=0.15,
                               context=f'mood v={mood.valence:+.2f} a={mood.arousal:.2f}, '
                                       f'{mood_boosted} memories boosted')
    except Exception:
        pass
    _sq_after('mood_congruent')
    _attention_stages.append({'stage': 'mood_congruent', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})
    _enforce_bias_cap()

    # === ACT-R RETRIEVAL NOISE (N1 v1.1: arousal-modulated perturbation) ===
    # Anderson (2007): retrieval noise s=0.25, scaled by arousal.
    # High arousal = noisier = more creative/serendipitous retrieval.
    # Low arousal = precise = predictable retrieval.
    _ts = _time.monotonic()
    _sq_before('actr_noise')
    try:
        from affect_system import get_retrieval_noise, get_mood as _actr_get_mood
        _actr_mood = _actr_get_mood()
        noise_sd = get_retrieval_noise(_actr_mood.arousal)
        if noise_sd > 0.01:  # Only apply meaningful noise
            import random as _rand
            noise_applied = 0
            for result in results:
                perturbation = _rand.gauss(0, noise_sd)
                result['score'] = max(0.0, result['score'] + perturbation)
                result['actr_noise'] = round(perturbation, 4)  # BUG-7 fix: store for binding layer
                if abs(perturbation) > 0.01:
                    noise_applied += 1
            if _expl and noise_applied:
                _expl.add_step('actr_retrieval_noise', noise_applied, weight=0.05,
                               context=f'arousal={_actr_mood.arousal:.2f}, '
                                       f'noise_sd={noise_sd:.3f}, '
                                       f'{noise_applied} scores perturbed')
    except Exception:
        pass
    _sq_after('actr_noise')
    _attention_stages.append({'stage': 'actr_noise', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})
    _enforce_bias_cap()

    # === GRAVITY WELL DAMPENING ===
    # Penalize memories where query key terms don't appear in the preview
    # This catches hub memories that match everything via embedding similarity
    # but don't actually contain the relevant information
    _ts = _time.monotonic()
    _sq_before('gravity_dampening')
    query_terms = set(query.lower().split())
    # Filter out stopwords
    stopwords = {'what', 'is', 'my', 'the', 'a', 'an', 'do', 'i', 'know', 'about',
                 'have', 'did', 'on', 'in', 'for', 'to', 'of', 'and', 'or', 'how',
                 'why', 'where', 'when', 'who', 'should', 'today', 'done', 'been'}
    key_terms = query_terms - stopwords
    if key_terms and len(key_terms) >= 1 and not _ABLATION_SKIP_GRAVITY:
        for result in results:
            preview_lower = result.get("preview", "").lower()
            # Check if ANY key term appears in the preview
            term_overlap = sum(1 for t in key_terms if t in preview_lower)
            if term_overlap == 0 and result["score"] > threshold:
                # No key terms found in preview — dampen by 50%
                # This is aggressive but necessary to break gravity wells
                result["score"] *= 0.5
                result["dampened"] = True

    _sq_after('gravity_dampening')
    _attention_stages.append({'stage': 'gravity_dampening', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})

    if _expl:
        dampened_count = sum(1 for r in results if r.get('dampened'))
        if dampened_count:
            _expl.add_step('gravity_dampening', dampened_count, weight=-0.5,
                           context=f'{dampened_count} results dampened (no key terms in preview)')

    # === HUB DEGREE DAMPENING (from SpindriftMend #27) ===
    # Structural penalty for high-degree co-occurrence hubs (P90+)
    # Complements keyword dampening: catches hubs whose preview naturally
    # contains common terms but are still too general
    _ts = _time.monotonic()
    _sq_before('hub_dampening')
    try:
        from curiosity_engine import _build_degree_map
        _hub_degree_map = _build_degree_map()
        if _hub_degree_map:
            degrees = sorted(_hub_degree_map.values())
            p90_idx = int(len(degrees) * 0.9)
            p90_threshold = degrees[p90_idx] if p90_idx < len(degrees) else degrees[-1]
            max_deg = degrees[-1] if degrees else 1
            hub_dampened = 0
            for result in results:
                if result.get('entity_injected') or result.get('entity_match'):
                    continue  # Entity-matched memories exempt
                deg = _hub_degree_map.get(result['id'], 0)
                if deg > p90_threshold and max_deg > p90_threshold:
                    # Scale from 1.0 at P90 to 0.4 at max degree (was 0.6, Spin fix)
                    ratio = (deg - p90_threshold) / (max_deg - p90_threshold)
                    penalty = 1.0 - 0.6 * ratio  # Floor at 0.4x (was 0.6x)
                    # Content quality: noise hubs get additional penalty
                    content = result.get('preview', '')
                    if _is_noise_content(content):
                        penalty *= 0.5  # Effective floor 0.2x for noise hubs
                    result['score'] *= max(0.2, penalty)
                    result['hub_dampened'] = True
                    hub_dampened += 1
            if _expl and hub_dampened:
                _expl.add_step('hub_degree_dampening', hub_dampened, weight=-0.3,
                               context=f'{hub_dampened} high-degree hubs dampened (P90={p90_threshold})')
    except Exception:
        pass
    _sq_after('hub_dampening')
    _attention_stages.append({'stage': 'hub_dampening', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})

    # === Q-VALUE RE-RANKING (Phase 5: MemRL) ===
    # Blend similarity score with learned Q-value utility
    _ts = _time.monotonic()
    _sq_before('q_rerank')
    try:
        from q_value_engine import get_q_values, get_lambda, Q_RERANKING_ENABLED
        if Q_RERANKING_ENABLED:
            result_ids = [r['id'] for r in results]
            q_vals = get_q_values(result_ids)
            lam = get_lambda()
            q_reranked = 0
            for result in results:
                q = q_vals.get(result['id'], 0.5)
                if q != 0.5:  # Only rerank trained memories
                    old_score = result['score']
                    result['score'] = lam * old_score + (1 - lam) * q
                    result['q_value'] = q
                    result['q_lambda'] = lam
                    q_reranked += 1
            if _expl and q_reranked:
                _expl.add_step('q_value_reranking', q_reranked, weight=0.4,
                               context=f'{q_reranked} results reranked (lambda={lam:.3f})')
    except Exception:
        pass
    _sq_after('q_rerank')
    _attention_stages.append({'stage': 'q_rerank', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})
    _enforce_bias_cap()

    # === STRATEGY-GUIDED ADJUSTMENT (explanation mining feedback) ===
    # DENSITY GATE: Skip if < 5 learned strategies (saves ~15ms when sparse)
    _ts = _time.monotonic()
    _sq_before('strategy_resolution')
    try:
        from explanation_miner import get_strategies
        strategies = get_strategies()
        strategy_adjusted = 0
        if strategies and len(strategies) >= 5:
            for strategy in strategies:
                if strategy.get('type') != 'factor_impact':
                    continue
                delta = strategy.get('delta', 0)
                if abs(delta) < 0.03:
                    continue  # Skip weak signals
                factor = strategy.get('factor', '')
                direction = strategy.get('direction', '')
                for result in results:
                    # Check if this result's provenance mentions the factor
                    prov = str(result.get('provenance', '')) + str(result.get('boosted_by', ''))
                    if factor.lower() in prov.lower():
                        multiplier = 1.0 + min(0.15, abs(delta))
                        if direction == 'negative':
                            multiplier = 1.0 / multiplier
                        result['score'] *= multiplier
                        strategy_adjusted += 1
            if _expl and strategy_adjusted:
                _expl.add_step('strategy_adjustment', strategy_adjusted, weight=0.2,
                               context=f'{strategy_adjusted} results adjusted by {len(strategies)} learned strategies')
    except Exception:
        pass
    _sq_after('strategy_resolution')
    _attention_stages.append({'stage': 'strategy_resolution', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})
    _enforce_bias_cap()

    # === RESOLUTION BOOSTING ===
    _ts = _time.monotonic()
    _sq_before('importance_freshness')
    for result in results:
        tags = load_memory_tags(result["id"])
        if tags and RESOLUTION_TAGS.intersection(set(t.lower() for t in tags)):
            result["score"] *= RESOLUTION_BOOST
            result["boosted"] = True

    if _expl:
        boosted_count = sum(1 for r in results if r.get('boosted'))
        if boosted_count:
            _expl.add_step('resolution_boost', boosted_count, weight=0.25,
                           context=f'{boosted_count} results boosted ({RESOLUTION_BOOST}x) for resolution tags')

    # === EVIDENCE TYPE SCORING (epistemic weight) ===
    EVIDENCE_MULTIPLIERS = {'verified': 1.20, 'observation': 1.10, 'inference': 1.00, 'claim': 0.85}
    try:
        evidence_scored = 0
        for result in results:
            row = db.get_memory(result['id'])
            if row:
                extra = row.get('extra_metadata') or {}
                etype = extra.get('evidence_type', 'claim')
                mult = EVIDENCE_MULTIPLIERS.get(etype, 1.0)
                if mult != 1.0:
                    result['score'] *= mult
                    result['evidence_type'] = etype
                    evidence_scored += 1
        if _expl and evidence_scored:
            _expl.add_step('evidence_type_scoring', evidence_scored, weight=0.15,
                           context=f'{evidence_scored} results weighted by evidence type')
    except Exception:
        pass

    # === IMPORTANCE/FRESHNESS SCORING ===
    # Detect query context and apply importance/freshness weights to boost scores
    try:
        from decay_evolution import calculate_activation
        from db_adapter import get_db as _imp_db, db_to_file_metadata as _imp_meta

        # Detect context from query keywords
        recent_keywords = {'today', 'recent', 'latest', 'just', 'last', 'new', 'current'}
        foundational_keywords = {'core', 'fundamental', 'always', 'identity', 'values', 'principle'}
        query_words = set(query.lower().split())

        if query_words & recent_keywords:
            imp_context = 'recent'
        elif query_words & foundational_keywords:
            imp_context = 'foundational'
        else:
            imp_context = 'general'

        _db = _imp_db()
        for result in results:
            row = _db.get_memory(result['id'])
            if row:
                meta, _ = _imp_meta(row)
                activation = calculate_activation(meta, context=imp_context)
                # Blend cosine similarity with activation: 70% cosine, 30% activation
                result['score'] = 0.7 * result['score'] + 0.3 * activation
                result['activation'] = activation
                result['imp_context'] = imp_context
    except Exception:
        pass  # Don't break search if activation scoring fails

    _sq_after('importance_freshness')
    _attention_stages.append({'stage': 'resolution_evidence_importance', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})
    _enforce_bias_cap()

    if _expl and any(r.get('activation') for r in results):
        _expl.add_step('importance_freshness', imp_context, weight=0.3,
                       context=f'context={imp_context}, blended 70% cosine + 30% activation')

    # === INHIBITION OF RETURN (IOR) — Biological novelty bias ===
    # Penalize memories recalled in recent sessions to encourage exploration.
    # In neuroscience, IOR prevents re-attending to recently-fixated locations.
    # Here: memories recalled in last 3 sessions get a small score penalty
    # so the system surfaces different memories over time.
    _ts = _time.monotonic()
    _sq_before('inhibition_of_return')
    _ior_applied = 0
    try:
        _ior_db = db
        _ior_session_id = None
        try:
            import session_state as _ior_ss
            _ior_session_id = _ior_ss.get_db_session_id()
        except Exception:
            pass
        if _ior_session_id and results:
            import psycopg2.extras as _ior_extras
            with _ior_db._conn() as conn:
                with conn.cursor(cursor_factory=_ior_extras.RealDictCursor) as cur:
                    # Get memory IDs recalled in last 3 sessions (excluding current)
                    cur.execute(f"""
                        SELECT DISTINCT sr.memory_id, COUNT(DISTINCT sr.session_id) as session_count
                        FROM {_ior_db._table('session_recalls')} sr
                        JOIN {_ior_db._table('sessions')} s ON sr.session_id = s.id
                        WHERE sr.session_id != %s
                        AND s.started_at > NOW() - INTERVAL '3 days'
                        GROUP BY sr.memory_id
                    """, (_ior_session_id,))
                    _recent_recalls = {row['memory_id']: row['session_count'] for row in cur.fetchall()}

            if _recent_recalls:
                for result in results:
                    recall_count = _recent_recalls.get(result['id'], 0)
                    if recall_count > 0:
                        # -5% per recent session, max -15% penalty
                        ior_penalty = min(0.15, 0.05 * recall_count)
                        result['score'] *= (1.0 - ior_penalty)
                        result['ior_penalty'] = round(ior_penalty, 3)
                        _ior_applied += 1
    except Exception:
        pass  # Don't break search if IOR fails
    if _expl and _ior_applied:
        _expl.add_step('inhibition_of_return', _ior_applied, weight=0.1,
                       context=f'{_ior_applied} results penalized for recent recall (novelty bias)')
    _sq_after('inhibition_of_return')
    _attention_stages.append({'stage': 'inhibition_of_return', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})

    # === CURIOSITY BOOST (Phase 2: sparse-region exploration) ===
    # Memories with few co-occurrence edges get a small boost to encourage
    # the system to build connections in sparse graph regions
    _ts = _time.monotonic()
    _sq_before('curiosity_boost')
    try:
        if _ABLATION_SKIP_CURIOSITY:
            raise ImportError('Ablation: curiosity_boost disabled')
        from curiosity_engine import _build_degree_map, LOW_DEGREE_THRESHOLD
        _degree_map = _build_degree_map()
        if _degree_map:
            max_degree = max(_degree_map.values())
            curiosity_boosted = 0
            for result in results:
                degree = _degree_map.get(result['id'], 0)
                if degree <= LOW_DEGREE_THRESHOLD and max_degree > 0:
                    # Small boost: up to 10% for completely isolated memories
                    boost = 0.10 * (1.0 - degree / max(LOW_DEGREE_THRESHOLD + 1, 1))
                    result['score'] *= (1.0 + boost)
                    result['curiosity_boosted'] = True
                    curiosity_boosted += 1
            if _expl and curiosity_boosted:
                _expl.add_step('curiosity_boost', curiosity_boosted, weight=0.1,
                               context=f'{curiosity_boosted} results boosted for sparse-graph exploration')
    except Exception:
        pass  # Don't break search if curiosity engine unavailable
    _sq_after('curiosity_boost')
    _attention_stages.append({'stage': 'curiosity_boost', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})

    # === GOAL-RELEVANCE BOOST (SR1/SR2: N4 goal-directed retrieval bias) ===
    # FINDING-19 fix: Improved from naive keyword to phrase-aware matching with
    # punctuation stripping and full content search (not just 150-char preview)
    _ts = _time.monotonic()
    _sq_before('goal_relevance')
    try:
        import re as _re_goal
        from goal_generator import get_active_goals, get_focus_goal
        _active_goals = get_active_goals()
        if _active_goals and results:
            # SR2: Extract keywords AND meaningful phrases from goals
            _goal_stopwords = {'the', 'a', 'an', 'to', 'for', 'of', 'and', 'or', 'in', 'on',
                               'with', 'is', 'at', 'by', 'from', 'that', 'this', 'my', 'i',
                               'be', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'do'}
            _goal_keywords = set()
            _focus_keywords = set()
            _focus_phrases = []  # Multi-word phrases for stronger matching
            _goal_phrases = []
            for g in _active_goals:
                action = g.get('action', '')
                # Strip punctuation for cleaner matching
                clean = _re_goal.sub(r'[^\w\s]', '', action.lower())
                words = set(clean.split()) - _goal_stopwords
                _goal_keywords.update(words)
                # Extract 2-3 word phrases (consecutive non-stopwords)
                action_words = clean.split()
                for i in range(len(action_words) - 1):
                    if action_words[i] not in _goal_stopwords and action_words[i+1] not in _goal_stopwords:
                        phrase = f"{action_words[i]} {action_words[i+1]}"
                        _goal_phrases.append(phrase)
                if g.get('is_focus'):
                    _focus_keywords.update(words)
                    for i in range(len(action_words) - 1):
                        if action_words[i] not in _goal_stopwords and action_words[i+1] not in _goal_stopwords:
                            _focus_phrases.append(f"{action_words[i]} {action_words[i+1]}")

            if _goal_keywords:
                goal_boosted = 0
                for result in results:
                    # Use full content, not just preview (FINDING-19: was truncated to 150 chars)
                    content_lower = result.get('content', result.get('preview', '')).lower()
                    # Focus goal: phrase match (2x weight) + keyword match
                    focus_phrase_hits = sum(1 for p in _focus_phrases if p in content_lower) if _focus_phrases else 0
                    focus_kw_hits = sum(1 for t in _focus_keywords if t in content_lower) if _focus_keywords else 0
                    # Weight: phrases worth 2x keywords
                    focus_score = focus_phrase_hits * 2 + focus_kw_hits

                    # Other goals
                    other_kw = _goal_keywords - _focus_keywords
                    other_phrase_hits = sum(1 for p in _goal_phrases if p not in _focus_phrases and p in content_lower)
                    other_kw_hits = sum(1 for t in other_kw if t in content_lower)
                    other_score = other_phrase_hits * 2 + other_kw_hits

                    if focus_score > 0:
                        boost = min(0.15, 0.03 * focus_score)  # Finer granularity
                        result['score'] *= (1.0 + boost)
                        result['goal_boosted'] = True
                        result['goal_boost_source'] = 'focus goal'
                        goal_boosted += 1
                    elif other_score > 0:
                        boost = min(0.05, 0.015 * other_score)
                        result['score'] *= (1.0 + boost)
                        result['goal_boosted'] = True
                        result['goal_boost_source'] = 'active goal'
                        goal_boosted += 1

                if _expl and goal_boosted:
                    _expl.add_step('goal_relevance_boost', goal_boosted, weight=0.15,
                                   context=f'{goal_boosted} results boosted for goal alignment '
                                   f'({len(_focus_keywords)} focus kw, {len(_focus_phrases)} focus phrases, {len(_goal_keywords)} total)')
    except Exception:
        pass
    _sq_after('goal_relevance')
    _attention_stages.append({'stage': 'goal_relevance', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})
    _enforce_bias_cap()

    # === AUTO-DETECT DIMENSION from session context ===
    _ts = _time.monotonic()
    _sq_before('dimensional_boost')
    if not dimension and results:
        try:
            from context_manager import get_session_dimensions
            session_dims = get_session_dimensions()
            # Pick strongest signal: WHO > WHERE > WHY > WHAT
            if session_dims.get('who'):
                dimension = 'who'
            elif session_dims.get('where'):
                dimension = 'where'
                sub_view = session_dims['where'][0] if session_dims['where'] else None
            elif session_dims.get('why'):
                dimension = 'why'
            elif session_dims.get('what'):
                dimension = 'what'
        except Exception:
            pass

    # === DIMENSIONAL BOOSTING (Phase 3: 5W-aware search) ===
    if dimension and results:
        result_ids = [r['id'] for r in results]
        dim_degree = db.get_dimension_degree(dimension, sub_view or '', result_ids)
        for result in results:
            degree = dim_degree.get(result['id'], 0)
            if degree > 0:
                result['score'] *= (1 + DIMENSION_BOOST_SCALE * math.log(1 + degree))
                result['dim_boosted'] = True
                result['dim_degree'] = degree

    _sq_after('dimensional_boost')
    _attention_stages.append({'stage': 'dimensional_boost', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})

    if _expl and dimension:
        dim_boosted = sum(1 for r in results if r.get('dim_boosted'))
        if dim_boosted:
            _expl.add_step('dimensional_boost', dim_boosted, weight=0.1,
                           context=f'{dim_boosted} results boosted in {dimension} dimension')

    _enforce_bias_cap()  # T1.2: after dimensional_boost

    # === TYPED EDGE EXPANSION + INFERENCE (Phase 4: knowledge graph reasoning) ===
    # Annotate results with typed edges AND apply inference rules:
    #   - 'contradicts' edges REDUCE score (conflicting memories less reliable)
    #   - 'supports' edges BOOST score (corroborated memories more reliable)
    #   - 'supersedes' edges PENALIZE the superseded memory
    # DENSITY GATE: Early exit if first 3 results have zero typed edges
    _ts = _time.monotonic()
    _sq_before('kg_expansion')
    try:
        from knowledge_graph import get_edges_from, get_edges_to
        kg_annotated = 0
        kg_inferred = 0
        result_ids = {r['id'] for r in results}
        # Early exit: probe first 3 results for any edges
        _kg_probe_hits = 0
        for _probe in results[:3]:
            if get_edges_from(_probe['id'], 'causes') or get_edges_from(_probe['id'], 'supports'):
                _kg_probe_hits += 1
        if _kg_probe_hits == 0:
            raise ImportError('Density gate: no typed edges on top 3 results')
        for result in results[:20]:  # Only check top 20 candidates
            edges = get_edges_from(result['id'], 'causes')
            if edges:
                result['kg_causes'] = len(edges)
                kg_annotated += 1
            res_edges = get_edges_from(result['id'], 'resolves')
            if res_edges:
                result['kg_resolves'] = len(res_edges)
                kg_annotated += 1

            # Inference: contradicts edges penalize
            contra_out = get_edges_from(result['id'], 'contradicts')
            contra_in = get_edges_to(result['id'], 'contradicts')
            if contra_out or contra_in:
                # Penalize based on how many contradictions touch other results
                contra_targets = {e['target_id'] for e in contra_out} | {e['source_id'] for e in contra_in}
                in_result_set = contra_targets & result_ids
                if in_result_set:
                    result['score'] *= max(0.6, 0.85 ** len(in_result_set))
                    result['kg_contradictions'] = len(contra_out) + len(contra_in)
                    kg_inferred += 1

            # Inference: supports edges boost
            support_out = get_edges_from(result['id'], 'supports')
            support_in = get_edges_to(result['id'], 'supports')
            support_count = len(support_out) + len(support_in)
            if support_count > 0:
                result['score'] *= min(1.5, 1.0 + 0.10 * support_count)
                result['kg_supports'] = support_count
                kg_inferred += 1

            # Inference: superseded memories penalized
            superseded_by = get_edges_to(result['id'], 'supersedes')
            if superseded_by:
                result['score'] *= 0.5  # Heavy penalty — this memory is outdated
                result['kg_superseded'] = True
                kg_inferred += 1

        if _expl and (kg_annotated or kg_inferred):
            _expl.add_step('knowledge_graph_inference', kg_annotated + kg_inferred, weight=0.15,
                           context=f'{kg_annotated} annotated, {kg_inferred} scored by edge type')
    except Exception:
        pass  # Don't break search if knowledge graph unavailable
    _sq_after('kg_expansion')
    _attention_stages.append({'stage': 'kg_expansion', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})

    # BUG-22 fix: Cap cumulative multiplicative bias at 3.0x original score.
    # Pipeline stages can stack entity(2x) * resolution(1.25x) * evidence(1.2x) *
    # goal(1.15x) * KG(1.5x) = 5.2x uncapped. Soft cap at 3.0x preserves ranking
    # while preventing extreme outliers from dominating.
    MAX_SCORE_MULTIPLIER = 3.0
    for result in results:
        original = result.get('original_score', result['score'])
        if result['score'] > original * MAX_SCORE_MULTIPLIER:
            result['score'] = original * MAX_SCORE_MULTIPLIER
            result['score_capped'] = True

    # Sort by (boosted) score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:limit]

    # === STEP 11: SPREADING ACTIVATION (Phase 5: graph-driven candidate generation) ===
    # Walk KG edges from top results to discover memories that are IMPLIED
    # by the results but don't match the query embedding directly.
    # Spread candidates are APPENDED to results (not competing for slots) because
    # their value is discovering non-obvious connections, not matching embeddings.
    # DENSITY GATE: Probe top seed first; if traverse returns nothing, skip all 3
    _ts = _time.monotonic()
    _sq_before('spreading_activation')
    try:
        from knowledge_graph import traverse as kg_traverse
        from q_value_engine import get_q_values as _sa_get_q_values

        existing_ids = {r['id'] for r in top_results}

        # Density gate: probe first seed before committing to full traversal
        if top_results:
            _sa_probe = kg_traverse(top_results[0]['id'], hops=2, direction='both', min_confidence=0.5)
            if not _sa_probe:
                raise ImportError('Density gate: spreading activation probe empty')

        # Collect best path to each reachable node across all seeds
        # Key: node_id, Value: (score, provenance, relationship, depth)
        reachable = {}

        # Two-pass spreading activation with FAN EFFECT (Anderson 1974):
        # Pass 1: Collect all edges and build degree map
        _all_seed_edges = []  # (seed, edge)
        _sa_degree_map = {}   # node_id -> edge count (proxy for KG degree)
        for seed in top_results[:3]:
            edges = kg_traverse(seed['id'], hops=2, direction='both', min_confidence=0.5)
            for edge in edges:
                _all_seed_edges.append((seed, edge))
                for nid in (edge['source_id'], edge['target_id']):
                    _sa_degree_map[nid] = _sa_degree_map.get(nid, 0) + 1

        # Pass 2: Compute spread scores with fan effect dampening
        for seed, edge in _all_seed_edges:
            for node_id in (edge['source_id'], edge['target_id']):
                if node_id == seed['id'] or node_id in existing_ids:
                    continue
                # Score decays with hops: parent_score * confidence * 0.5^depth
                # R13: Relationship type modifies spread score
                rel_type = edge.get('relationship', '')
                type_mult = 1.0
                if rel_type == 'contradicts':
                    type_mult = -0.3  # Contradictions spread as negative signal
                elif rel_type == 'supports':
                    type_mult = 1.3
                elif rel_type == 'supersedes':
                    type_mult = 0.3  # Superseded paths are weak
                spread_score = seed['score'] * edge['confidence'] * (0.5 ** edge['depth']) * abs(type_mult)
                # FAN EFFECT: More associations = weaker each individual activation.
                # Divide by sqrt(degree) so hub nodes don't dominate spread.
                fan_degree = _sa_degree_map.get(node_id, 1)
                spread_score /= math.sqrt(max(fan_degree, 1))
                if type_mult < 0:
                    continue  # Don't spread through contradictions
                if node_id not in reachable or spread_score > reachable[node_id][0]:
                    provenance = f"spread_from:{seed['id']} via:{edge['relationship']}"
                    reachable[node_id] = (spread_score, provenance,
                                          edge['relationship'], edge['depth'])

        if reachable:
            spread_candidates = [(nid, *vals) for nid, vals in reachable.items()]
            # Filter by Q-value: only inject candidates with Q >= 0.4
            candidate_ids = [c[0] for c in spread_candidates]
            q_vals = _sa_get_q_values(candidate_ids)
            filtered = [(cid, score, prov, rel, depth) for cid, score, prov, rel, depth
                        in spread_candidates if q_vals.get(cid, 0.5) >= 0.4]

            # Cap at 5 spread candidates (highest score first)
            filtered.sort(key=lambda x: x[1], reverse=True)
            filtered = filtered[:5]

            if filtered:
                # Batch-fetch content previews + noise filter (Spin fix)
                fetch_ids = [c[0] for c in filtered]
                previews = {}
                try:
                    import psycopg2.extras as _sa_extras
                    with db._conn() as conn:
                        with conn.cursor(cursor_factory=_sa_extras.RealDictCursor) as cur:
                            cur.execute(
                                f"SELECT id, content FROM {db._table('memories')} WHERE id = ANY(%s)",
                                (fetch_ids,)
                            )
                            for row in cur.fetchall():
                                previews[row['id']] = (row.get('content') or '')[:150]
                except Exception:
                    pass

                # Filter out noise memories from spread candidates
                filtered = [(cid, score, prov, rel, depth) for cid, score, prov, rel, depth
                            in filtered if not _is_noise_content(previews.get(cid, ''))]

                for cid, score, prov, rel, depth in filtered:
                    top_results.append({
                        'id': cid,
                        'score': score,
                        'preview': previews.get(cid, ''),
                        'path': f'db://spread/{cid}.md',
                        'spread_activated': True,
                        'spread_provenance': prov,
                        'spread_relationship': rel,
                        'spread_depth': depth,
                        'q_value': q_vals.get(cid, 0.5),
                    })

                if _expl:
                    _expl.add_step('spreading_activation', len(filtered), weight=0.15,
                                   context=f'{len(filtered)} spread candidates from '
                                           f'{len(reachable)} reachable neighbors (Q>=0.4, top 5)')
    except ImportError:
        pass  # KG or Q-value modules not available
    except Exception:
        pass  # Don't break search if spreading activation fails
    _sq_after('spreading_activation')
    _attention_stages.append({'stage': 'spreading_activation', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})

    # Register recalls with the memory system (+ reconsolidation context)
    if register_recall and top_results:
        try:
            from memory_manager import recall_memory
            co_active = [r['id'] for r in top_results[:10]]
            for r in top_results:
                recall_memory(r["id"], query_context=query,
                              co_active_ids=[x for x in co_active if x != r['id']])
        except Exception:
            pass

    # === COGNITIVE STATE + AFFECT: Fire search event (dual-track) ===
    try:
        from cognitive_state import process_event
        if top_results:
            process_event('search_success')
        else:
            process_event('search_failure')
    except Exception:
        pass

    try:
        from affect_system import process_affect_event
        event = 'search_success' if top_results else 'search_failure'
        memory_ids = [r['id'] for r in top_results[:5]] if top_results else None
        process_affect_event(event, memory_ids=memory_ids)
    except Exception:
        pass

    # Save explanation
    if _expl:
        if _cog_modifier != 0.0:
            _expl.add_step('cognitive_threshold', _cog_modifier, weight=0.1,
                           context=f'Threshold adjusted by {_cog_modifier:+.3f} from cognitive state')
        _expl.set_output({
            'result_count': len(top_results),
            'top_ids': [r['id'] for r in top_results[:5]],
            'top_scores': [round(r['score'], 4) for r in top_results[:5]],
            'total_candidates': len(results),
        })
        _expl.save()

    # === N5: INTEGRATIVE BINDING (BUG-18 fix: wire into production pipeline) ===
    # Apply binding to search results so downstream consumers (memory_manager,
    # thought_priming, session hooks) get bound representations, not raw dicts.
    _ts = _time.monotonic()
    _bind_count = 0
    try:
        from binding_layer import full_bind, minimal_bind, render_compact, BINDING_ENABLED
        if BINDING_ENABLED and top_results:
            for i, r in enumerate(top_results):
                try:
                    if i < 5:
                        bound = full_bind(r)
                    else:
                        bound = minimal_bind(r)
                    if bound:
                        r['binding_strength'] = getattr(bound, 'binding_strength', 0.0)
                        r['binding_level'] = getattr(bound, 'binding_level', 'minimal')
                        r['retrieval_reasons'] = getattr(bound, 'retrieval_reasons', [])
                        # Compact rendering for LLM context
                        r['bound_context'] = render_compact(bound)
                        _bind_count += 1
                except Exception:
                    pass
    except ImportError:
        pass  # Binding layer not available
    except Exception:
        pass
    _attention_stages.append({'stage': 'integrative_binding', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})

    # === N6: INNER MONOLOGUE (Stage 16 — verbal evaluation of surfaced memories) ===
    # Gemma 3 4B evaluates memories against query BEFORE they reach Claude's context.
    # Adds monologue_relevance, monologue_reaction, monologue_skip annotations.
    # Does NOT re-rank — only annotates. Fail-safe: if monologue fails, results pass through.
    # skip_monologue=True for thought priming (System 1 fast path — async monologue fires separately).
    _ts = _time.monotonic()
    _monologue_count = 0
    try:
        from inner_monologue import annotate_search_results, MONOLOGUE_ENABLED
        if MONOLOGUE_ENABLED and top_results and not skip_monologue:
            _session_id = ''
            try:
                import session_state as _ss_mono
                _session_id = _ss_mono.get_session_id() if hasattr(_ss_mono, 'get_session_id') else ''
            except Exception:
                pass
            top_results = annotate_search_results(query, top_results, _session_id)
            _monologue_count = sum(1 for r in top_results if r.get('monologue_relevance') is not None)
    except ImportError:
        pass  # Inner monologue not available
    except Exception:
        pass  # Never break search for monologue failure
    _attention_stages.append({'stage': 'inner_monologue', 'time_ms': round((_time.monotonic() - _ts) * 1000, 1)})

    # Attention schema: store timing for self-narrative (AFTER all stages complete)
    _total_ms = round((_time.monotonic() - _t0) * 1000, 1)
    try:
        from datetime import datetime, timezone
        from db_adapter import get_db as _attn_db
        _adb = _attn_db()
        for s in _attention_stages:
            s['pct'] = round(s['time_ms'] / max(_total_ms, 0.1) * 100, 1)
        schema_entry = {
            'query': query[:100],
            'stages': _attention_stages,
            'total_ms': _total_ms,
            'result_count': len(top_results),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        existing = _adb.kv_get('.attention_schema') or []
        existing.append(schema_entry)
        _adb.kv_set('.attention_schema', existing[-20:])
    except Exception:
        pass

    # === PER-STAGE Q-LEARNING: Record search deltas for credit assignment ===
    try:
        from stage_q_learning import record_search_deltas
        record_search_deltas(top_results, _query_type)
    except Exception:
        pass

    return top_results


def get_status() -> dict:
    """Get status of the semantic search index. DB-only."""
    from db_adapter import get_db
    db = get_db()

    memory_count = db.count_memories()

    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {db._table('text_embeddings')}")
            indexed_count = cur.fetchone()[0]

    return {
        "indexed": indexed_count,
        "total_memories": memory_count,
        "coverage": f"{indexed_count}/{memory_count}",
        "model": "qwen3-embedding",
        "store": "postgresql/pgvector",
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "local_endpoint": os.getenv("LOCAL_EMBEDDING_ENDPOINT", "not configured")
    }


def embed_single(memory_id: str, content: str) -> bool:
    """
    Embed a single memory (call this when storing new memories).
    Returns True if successful. DB-only.
    """
    embedding = get_embedding(content)
    if not embedding:
        return False

    from db_adapter import get_db
    get_db().store_embedding(
        memory_id=memory_id,
        embedding=embedding,
        preview=content[:200],
        model="qwen3-embedding",
    )

    return True


# ============================================================================
# v2.12: CONSOLIDATION - Find and merge semantically similar memories
# Credit: Mem0 consolidation pattern, MemEvolve self-organization
# ============================================================================

def find_similar_pairs(threshold: float = 0.85, limit: int = 20) -> list[dict]:
    """
    Find pairs of memories that are semantically similar.
    These are candidates for consolidation (merging).

    Args:
        threshold: Minimum cosine similarity to consider (0.85 = very similar)
        limit: Maximum pairs to return

    Returns:
        List of dicts with {id1, id2, similarity, preview1, preview2}
    """
    # Load all embeddings from DB
    from db_adapter import get_db
    import psycopg2.extras
    db = get_db()

    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT memory_id, embedding, preview FROM {db._table('text_embeddings')}")
            rows = cur.fetchall()

    if len(rows) < 2:
        return []

    # Compare all pairs
    pairs = []
    for i, r1 in enumerate(rows):
        emb1 = r1.get('embedding')
        if not emb1:
            continue
        for r2 in rows[i+1:]:
            emb2 = r2.get('embedding')
            if not emb2:
                continue

            sim = cosine_similarity(list(emb1), list(emb2))
            if sim >= threshold:
                pairs.append({
                    "id1": r1['memory_id'],
                    "id2": r2['memory_id'],
                    "similarity": round(sim, 4),
                    "preview1": (r1.get("preview") or "")[:80],
                    "preview2": (r2.get("preview") or "")[:80]
                })

    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return pairs[:limit]


def get_memory_embedding(memory_id: str) -> Optional[list[float]]:
    """Get the embedding for a specific memory. DB-only."""
    from db_adapter import get_db
    db = get_db()
    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT embedding FROM {db._table('text_embeddings')} WHERE memory_id = %s",
                (memory_id,)
            )
            row = cur.fetchone()
            return list(row['embedding']) if row else None


def remove_from_index(memory_id: str) -> bool:
    """Remove a memory from the embedding index. DB-only."""
    from db_adapter import get_db
    db = get_db()
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {db._table('text_embeddings')} WHERE memory_id = %s",
                (memory_id,)
            )
            return cur.rowcount > 0


def get_temporal_successors(memory_id: str, limit: int = 5) -> list[dict]:
    """
    R10: Get memories that typically follow the given one in recall sequences.
    Aggregates direction_weight from edge_observations to find temporal flow.
    Positive aggregate = this memory tends to be recalled AFTER memory_id.

    Returns list of {id, content_preview, aggregate_direction, observation_count}
    """
    from db_adapter import get_db
    db = get_db()
    import psycopg2.extras
    results = []
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Find all edges involving this memory, aggregate direction_weight
            # For edges where memory_id is id1: positive dw means id2 comes after
            # For edges where memory_id is id2: negative dw means id1 comes after
            cur.execute(f"""
                WITH successors AS (
                    SELECT edge_id2 AS other_id,
                           SUM(direction_weight) AS agg_dir,
                           COUNT(*) AS obs_count
                    FROM {db._table('edge_observations')}
                    WHERE edge_id1 = %s AND direction_weight != 0
                    GROUP BY edge_id2
                    UNION ALL
                    SELECT edge_id1 AS other_id,
                           SUM(-direction_weight) AS agg_dir,
                           COUNT(*) AS obs_count
                    FROM {db._table('edge_observations')}
                    WHERE edge_id2 = %s AND direction_weight != 0
                    GROUP BY edge_id1
                )
                SELECT other_id, SUM(agg_dir) AS aggregate_direction,
                       SUM(obs_count) AS observation_count
                FROM successors
                GROUP BY other_id
                HAVING SUM(agg_dir) > 0
                ORDER BY SUM(agg_dir) DESC
                LIMIT %s
            """, (memory_id, memory_id, limit))
            rows = cur.fetchall()

    for row in rows:
        other = db.get_memory(row['other_id'])
        preview = (other.get('content', '') or '')[:100] if other else ''
        results.append({
            'id': row['other_id'],
            'content_preview': preview,
            'aggregate_direction': float(row['aggregate_direction']),
            'observation_count': int(row['observation_count']),
        })
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic search for Drift's memories")
    parser.add_argument("command", choices=["index", "search", "status", "temporal-successors"],
                       help="Command to run")
    parser.add_argument("query", nargs="?", help="Search query (for search command)")
    parser.add_argument("--limit", type=int, default=5, help="Max results")
    parser.add_argument("--force", action="store_true", help="Force re-index all")
    parser.add_argument("--threshold", type=float, default=0.3, help="Min similarity")
    parser.add_argument("--dimension", type=str, default=None,
                       help="W-dimension to boost by (who/what/why/where)")
    parser.add_argument("--sub-view", type=str, default=None,
                       help="Sub-view within dimension")
    parser.add_argument("--skip-monologue", action="store_true",
                       help="Skip N6 inner monologue stage (for fast System 1 paths)")

    args = parser.parse_args()

    if args.command == "status":
        status = get_status()
        print("Semantic Search Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")

    elif args.command == "index":
        print("Indexing memories...")
        stats = index_memories(force=args.force)
        print(f"\nResults:")
        print(f"  Indexed: {stats['indexed']}")
        print(f"  Skipped: {stats['skipped']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Total: {stats['total']}")

    elif args.command == "search":
        if not args.query:
            print("Error: search requires a query")
            sys.exit(1)

        results = search_memories(
            args.query, limit=args.limit, threshold=args.threshold,
            dimension=args.dimension, sub_view=getattr(args, 'sub_view', None),
            skip_monologue=getattr(args, 'skip_monologue', False),
        )

        if not results:
            print("No matching memories found.")
        else:
            dim_label = f" (dimension: {args.dimension}" + (f"/{args.sub_view}" if getattr(args, 'sub_view', None) else "") + ")" if args.dimension else ""
            print(f"Found {len(results)} matching memories{dim_label}:\n")
            # N5: Rich binding when available
            try:
                from binding_layer import bind_results, render_narrative, BINDING_ENABLED
                if BINDING_ENABLED:
                    bound = bind_results(results)
                    for b in bound:
                        print(render_narrative(b))
                        print()
                else:
                    raise ImportError("disabled")
            except Exception:
                # Fallback to original format
                for r in results:
                    flags = []
                    if r.get('boosted'):
                        flags.append('resolution')
                    if r.get('dim_boosted'):
                        flags.append(f'dim:{r.get("dim_degree", 0)}')
                    flag_str = f" [{', '.join(flags)}]" if flags else ""
                    print(f"[{r['score']:.3f}] {r['id']}{flag_str}")
                    print(f"  {r['preview']}...")
                    print()

    elif args.command == "temporal-successors":
        if not args.query:
            print("Error: temporal-successors requires a memory ID")
            sys.exit(1)
        results = get_temporal_successors(args.query, limit=args.limit)
        if not results:
            print(f"No temporal successors found for {args.query}")
            print("(Direction data accumulates over sessions)")
        else:
            print(f"Memories that tend to follow {args.query}:\n")
            for r in results:
                print(f"  {r['id']} (dir={r['aggregate_direction']:.1f}, obs={r['observation_count']})")
                print(f"    {r['content_preview'][:80]}...")
                print()
