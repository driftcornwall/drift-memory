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

    results = []

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

    # === RESOLUTION BOOSTING ===
    for result in results:
        tags = load_memory_tags(result["id"])
        if tags and RESOLUTION_TAGS.intersection(set(t.lower() for t in tags)):
            result["score"] *= RESOLUTION_BOOST
            result["boosted"] = True

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
