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
                    dimension: str = None, sub_view: str = None) -> list[dict]:
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
    # Bidirectional vocabulary bridge
    try:
        from vocabulary_bridge import bridge_query
        bridged_query = bridge_query(query)
    except ImportError:
        bridged_query = query

    # Get query embedding
    query_embedding = get_embedding(bridged_query)
    if not query_embedding:
        print("Failed to get query embedding", file=sys.stderr)
        return []

    # === COGNITIVE STATE THRESHOLD ADJUSTMENT ===
    # High curiosity lowers threshold (explore more), high confidence raises it
    _cog_modifier = 0.0
    try:
        from cognitive_state import get_search_threshold_modifier, process_event
        _cog_modifier = get_search_threshold_modifier()
        threshold = max(0.1, min(0.5, threshold + _cog_modifier))
    except Exception:
        pass

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
        })
    except Exception:
        _expl = None

    # pgvector search — DB-only, no file fallback
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

    if _expl:
        _expl.add_step('pgvector_candidates', len(results), weight=1.0,
                        context=f'{len(rows)} raw rows, {len(results)} above threshold {threshold}')

    # === ENTITY INDEX INJECTION (Fix for WHO dimension) ===
    # When query mentions a known contact, inject their memories into candidates
    # This bridges the gap between contact names and memory embeddings
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

    if _expl:
        injected_count = sum(1 for r in results if r.get('entity_injected'))
        if injected_count:
            _expl.add_step('entity_injection', injected_count, weight=0.5,
                           context=f'{injected_count} memories injected from entity index')

    # === GRAVITY WELL DAMPENING ===
    # Penalize memories where query key terms don't appear in the preview
    # This catches hub memories that match everything via embedding similarity
    # but don't actually contain the relevant information
    query_terms = set(query.lower().split())
    # Filter out stopwords
    stopwords = {'what', 'is', 'my', 'the', 'a', 'an', 'do', 'i', 'know', 'about',
                 'have', 'did', 'on', 'in', 'for', 'to', 'of', 'and', 'or', 'how',
                 'why', 'where', 'when', 'who', 'should', 'today', 'done', 'been'}
    key_terms = query_terms - stopwords
    if key_terms and len(key_terms) >= 1:
        for result in results:
            preview_lower = result.get("preview", "").lower()
            # Check if ANY key term appears in the preview
            term_overlap = sum(1 for t in key_terms if t in preview_lower)
            if term_overlap == 0 and result["score"] > threshold:
                # No key terms found in preview — dampen by 50%
                # This is aggressive but necessary to break gravity wells
                result["score"] *= 0.5
                result["dampened"] = True

    if _expl:
        dampened_count = sum(1 for r in results if r.get('dampened'))
        if dampened_count:
            _expl.add_step('gravity_dampening', dampened_count, weight=-0.5,
                           context=f'{dampened_count} results dampened (no key terms in preview)')

    # === HUB DEGREE DAMPENING (from SpindriftMend #27) ===
    # Structural penalty for high-degree co-occurrence hubs (P90+)
    # Complements keyword dampening: catches hubs whose preview naturally
    # contains common terms but are still too general
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
                    # Scale from 1.0 at P90 to 0.6 at max degree
                    ratio = (deg - p90_threshold) / (max_deg - p90_threshold)
                    penalty = 1.0 - 0.4 * ratio  # Floor at 0.6x
                    result['score'] *= max(0.6, penalty)
                    result['hub_dampened'] = True
                    hub_dampened += 1
            if _expl and hub_dampened:
                _expl.add_step('hub_degree_dampening', hub_dampened, weight=-0.3,
                               context=f'{hub_dampened} high-degree hubs dampened (P90={p90_threshold})')
    except Exception:
        pass

    # === Q-VALUE RE-RANKING (Phase 5: MemRL) ===
    # Blend similarity score with learned Q-value utility
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

    # === RESOLUTION BOOSTING ===
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

    if _expl and any(r.get('activation') for r in results):
        _expl.add_step('importance_freshness', imp_context, weight=0.3,
                       context=f'context={imp_context}, blended 70% cosine + 30% activation')

    # === CURIOSITY BOOST (Phase 2: sparse-region exploration) ===
    # Memories with few co-occurrence edges get a small boost to encourage
    # the system to build connections in sparse graph regions
    try:
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

    # === AUTO-DETECT DIMENSION from session context ===
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

    if _expl and dimension:
        dim_boosted = sum(1 for r in results if r.get('dim_boosted'))
        if dim_boosted:
            _expl.add_step('dimensional_boost', dim_boosted, weight=0.1,
                           context=f'{dim_boosted} results boosted in {dimension} dimension')

    # === TYPED EDGE EXPANSION (Phase 4: knowledge graph enrichment) ===
    # For top results, check if they have causal/resolves relationships
    # and flag them for context (don't add new results, just annotate)
    try:
        from knowledge_graph import get_edges_from
        kg_annotated = 0
        for result in results[:20]:  # Only check top 20 candidates
            edges = get_edges_from(result['id'], 'causes')
            if edges:
                result['kg_causes'] = len(edges)
                kg_annotated += 1
            res_edges = get_edges_from(result['id'], 'resolves')
            if res_edges:
                result['kg_resolves'] = len(res_edges)
                kg_annotated += 1
        if _expl and kg_annotated:
            _expl.add_step('knowledge_graph_annotation', kg_annotated, weight=0.05,
                           context=f'{kg_annotated} results annotated with typed edges')
    except Exception:
        pass  # Don't break search if knowledge graph unavailable

    # Sort by (boosted) score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:limit]

    # Register recalls with the memory system
    if register_recall and top_results:
        try:
            from memory_manager import recall_memory
            for r in top_results:
                recall_memory(r["id"])
        except Exception:
            pass

    # === COGNITIVE STATE: Fire search event ===
    try:
        from cognitive_state import process_event
        if top_results:
            process_event('search_success')
        else:
            process_event('search_failure')
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic search for Drift's memories")
    parser.add_argument("command", choices=["index", "search", "status"],
                       help="Command to run")
    parser.add_argument("query", nargs="?", help="Search query (for search command)")
    parser.add_argument("--limit", type=int, default=5, help="Max results")
    parser.add_argument("--force", action="store_true", help="Force re-index all")
    parser.add_argument("--threshold", type=float, default=0.3, help="Min similarity")
    parser.add_argument("--dimension", type=str, default=None,
                       help="W-dimension to boost by (who/what/why/where)")
    parser.add_argument("--sub-view", type=str, default=None,
                       help="Sub-view within dimension")

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
        )

        if not results:
            print("No matching memories found.")
        else:
            dim_label = f" (dimension: {args.dimension}" + (f"/{args.sub_view}" if getattr(args, 'sub_view', None) else "") + ")" if args.dimension else ""
            print(f"Found {len(results)} matching memories{dim_label}:\n")
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
