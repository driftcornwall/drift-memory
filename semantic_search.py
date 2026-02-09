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
ACTIVE_DIR = MEMORY_DIR / "active"
CORE_DIR = MEMORY_DIR / "core"
EMBEDDINGS_FILE = MEMORY_DIR / "embeddings.json"

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


def load_embeddings() -> dict:
    """Load embeddings index from disk."""
    if EMBEDDINGS_FILE.exists():
        try:
            with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"memories": {}, "model": "text-embedding-3-small"}


def save_embeddings(data: dict):
    """Save embeddings index to disk."""
    with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def parse_memory_file(path: Path) -> tuple[Optional[str], Optional[str]]:
    """Parse a memory file and return (id, content)."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse YAML frontmatter
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                import re
                # Extract ID from frontmatter
                id_match = re.search(r'^id:\s*(.+)$', parts[1], re.MULTILINE)
                memory_id = id_match.group(1).strip() if id_match else path.stem
                # Content is everything after frontmatter
                body = parts[2].strip()
                return memory_id, body

        return path.stem, content
    except Exception as e:
        return None, None


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
    Index all memories by generating embeddings.

    Args:
        force: If True, re-index all memories. Otherwise, only index new ones.

    Returns:
        Summary of indexing results.
    """
    model_source = detect_embedding_source()
    data = load_embeddings() if not force else {"memories": {}, "model": model_source}
    existing = set(data["memories"].keys())

    stats = {"indexed": 0, "skipped": 0, "failed": 0, "total": 0}

    # Collect all memory files
    memory_files = []
    for directory in [CORE_DIR, ACTIVE_DIR]:
        if directory.exists():
            memory_files.extend(directory.glob("*.md"))

    stats["total"] = len(memory_files)

    for path in memory_files:
        memory_id, content = parse_memory_file(path)
        if not memory_id or not content:
            stats["failed"] += 1
            continue

        # Skip if already indexed (unless forcing)
        if memory_id in existing and not force:
            stats["skipped"] += 1
            continue

        # Apply vocabulary bridge before embedding (cross-register matching)
        try:
            from vocabulary_bridge import bridge_content
            bridged = bridge_content(content)
        except ImportError:
            bridged = content

        # Generate embedding
        embedding = get_embedding(bridged)
        if embedding:
            data["memories"][memory_id] = {
                "embedding": embedding,
                "path": str(path),
                "preview": content[:200]
            }
            stats["indexed"] += 1
            print(f"  Indexed: {memory_id}")
        else:
            stats["failed"] += 1
            print(f"  Failed: {memory_id}")

    # Update model info
    data["model"] = model_source
    data["embedding_dim"] = len(next(iter(data["memories"].values()), {}).get("embedding", [])) if data["memories"] else 0

    save_embeddings(data)
    return stats


def load_memory_tags(memory_id: str) -> list[str]:
    """Load tags from a memory file."""
    for directory in [ACTIVE_DIR, CORE_DIR]:
        # Try both naming patterns
        for pattern in [f"{memory_id}.md", f"*{memory_id}*.md"]:
            matches = list(directory.glob(pattern))
            if matches:
                try:
                    content = matches[0].read_text(encoding='utf-8')
                    if content.startswith('---'):
                        import re
                        tags_match = re.search(r'^tags:\s*\[([^\]]*)\]', content, re.MULTILINE)
                        if tags_match:
                            return [t.strip().strip("'\"") for t in tags_match.group(1).split(',')]
                except:
                    pass
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

    When register_recall=True (default), retrieved memories are registered
    with the decay/co-occurrence system, strengthening accessed memories
    and building associative links between concepts retrieved together.

    Resolution memories (tagged with 'resolution', 'procedural', 'fix', etc.)
    get a score boost so solutions surface before problem descriptions.

    When dimension is specified, memories well-connected in that W-graph
    get an additional score boost proportional to their degree.

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
    data = load_embeddings()
    if not data["memories"]:
        return []

    # Bidirectional vocabulary bridge:
    # - Reverse bridge: operational query -> append academic terms
    # - Forward bridge: academic query -> append operational terms
    try:
        from vocabulary_bridge import bridge_query
        bridged_query = bridge_query(query)
    except ImportError:
        bridged_query = query

    # Get query embedding (bridged for cross-register matching)
    query_embedding = get_embedding(bridged_query)
    if not query_embedding:
        print("Failed to get query embedding", file=sys.stderr)
        return []

    # Score all memories
    results = []
    for memory_id, info in data["memories"].items():
        score = cosine_similarity(query_embedding, info["embedding"])
        if score >= threshold:
            results.append({
                "id": memory_id,
                "score": score,
                "preview": info.get("preview", "")[:150],
                "path": info.get("path", "")
            })

    # === RESOLUTION BOOSTING ===
    # Boost memories tagged as resolution/procedural so solutions surface first
    for result in results:
        tags = load_memory_tags(result["id"])
        if tags and RESOLUTION_TAGS.intersection(set(t.lower() for t in tags)):
            result["score"] *= RESOLUTION_BOOST
            result["boosted"] = True

    # === DIMENSIONAL BOOSTING (Phase 3: 5W-aware search) ===
    # When a dimension is specified, boost memories with high connectivity
    # in that W-graph. Well-connected = contextually important.
    if dimension and results:
        try:
            from context_manager import load_graph
            graph = load_graph(dimension, sub_view)
            if graph and graph.get('edges'):
                # Build degree map for all nodes in this dimension
                dim_degree = {}
                for edge_key in graph['edges']:
                    parts = edge_key.split('|')
                    if len(parts) == 2:
                        dim_degree[parts[0]] = dim_degree.get(parts[0], 0) + 1
                        dim_degree[parts[1]] = dim_degree.get(parts[1], 0) + 1

                for result in results:
                    degree = dim_degree.get(result['id'], 0)
                    if degree > 0:
                        result['score'] *= (1 + DIMENSION_BOOST_SCALE * math.log(1 + degree))
                        result['dim_boosted'] = True
                        result['dim_degree'] = degree
        except ImportError:
            pass

    # Sort by (boosted) score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:limit]

    # Register recalls with the memory system (strengthens memories + builds co-occurrence)
    if register_recall and top_results:
        try:
            from memory_manager import recall_memory
            for r in top_results:
                # This updates recall_count and adds to session tracking for co-occurrence
                recall_memory(r["id"])
        except Exception:
            pass  # Fail gracefully if memory_manager unavailable

    return top_results


def get_status() -> dict:
    """Get status of the semantic search index."""
    data = load_embeddings()

    # Count actual memory files
    memory_count = 0
    for directory in [CORE_DIR, ACTIVE_DIR]:
        if directory.exists():
            memory_count += len(list(directory.glob("*.md")))

    indexed_count = len(data.get("memories", {}))

    return {
        "indexed": indexed_count,
        "total_memories": memory_count,
        "coverage": f"{indexed_count}/{memory_count}",
        "model": data.get("model", "unknown"),
        "index_file": str(EMBEDDINGS_FILE),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "local_endpoint": os.getenv("LOCAL_EMBEDDING_ENDPOINT", "not configured")
    }


def embed_single(memory_id: str, content: str) -> bool:
    """
    Embed a single memory (call this when storing new memories).
    Returns True if successful.
    """
    embedding = get_embedding(content)
    if not embedding:
        return False

    data = load_embeddings()
    data["memories"][memory_id] = {
        "embedding": embedding,
        "preview": content[:200]
    }
    save_embeddings(data)
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
    data = load_embeddings()
    memories = data.get("memories", {})

    if len(memories) < 2:
        return []

    # Compare all pairs
    pairs = []
    memory_ids = list(memories.keys())

    for i, id1 in enumerate(memory_ids):
        emb1 = memories[id1].get("embedding")
        if not emb1:
            continue

        for id2 in memory_ids[i+1:]:
            emb2 = memories[id2].get("embedding")
            if not emb2:
                continue

            sim = cosine_similarity(emb1, emb2)
            if sim >= threshold:
                pairs.append({
                    "id1": id1,
                    "id2": id2,
                    "similarity": round(sim, 4),
                    "preview1": memories[id1].get("preview", "")[:80],
                    "preview2": memories[id2].get("preview", "")[:80]
                })

    # Sort by similarity descending
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return pairs[:limit]


def get_memory_embedding(memory_id: str) -> Optional[list[float]]:
    """Get the embedding for a specific memory."""
    data = load_embeddings()
    mem = data.get("memories", {}).get(memory_id)
    return mem.get("embedding") if mem else None


def remove_from_index(memory_id: str) -> bool:
    """Remove a memory from the embedding index (after consolidation)."""
    data = load_embeddings()
    if memory_id in data.get("memories", {}):
        del data["memories"][memory_id]
        save_embeddings(data)
        return True
    return False


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
