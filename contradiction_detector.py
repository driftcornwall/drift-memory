"""
Contradiction Detector — Mechanical Skepticism for Drift Memory

Auto-detects when new memories contradict existing ones using NLI inference.
Creates 'contradicts' and 'supports' edges in the knowledge graph automatically.

Architecture:
    New memory → find top-5 similar via pgvector → NLI classify each pair
    → contradiction > 0.7 → auto KG edge + rejection log + cognitive event
    → entailment > 0.8 → auto KG 'supports' edge

Requires: NLI Docker service on port 8082 (cross-encoder/nli-deberta-v3-xsmall)
Degrades gracefully if service unavailable.
"""

import json
import sys
import urllib.request
from pathlib import Path

_parent = str(Path(__file__).resolve().parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

NLI_ENDPOINT = "http://localhost:8082"
CONTRADICTION_THRESHOLD = 0.7
ENTAILMENT_THRESHOLD = 0.8
MAX_SIMILAR_CHECK = 5


def _nli_available() -> bool:
    """Check if NLI service is reachable."""
    try:
        req = urllib.request.Request(f"{NLI_ENDPOINT}/health", method='GET')
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data.get('status') == 'ready'
    except Exception:
        return False


def _classify_pair(premise: str, hypothesis: str) -> dict | None:
    """Classify NLI relationship between two texts. Returns None on failure."""
    try:
        payload = json.dumps({
            "premise": premise[:2000],
            "hypothesis": hypothesis[:2000],
        }).encode('utf-8')
        req = urllib.request.Request(
            f"{NLI_ENDPOINT}/classify",
            data=payload,
            headers={"Content-Type": "application/json"},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"NLI classify failed: {e}")
        return None


def _classify_batch(pairs: list[tuple[str, str]]) -> list[dict | None]:
    """Batch classify NLI pairs. Returns list of results (None for failures)."""
    try:
        payload = json.dumps({
            "pairs": [
                {"premise": p[:2000], "hypothesis": h[:2000]}
                for p, h in pairs
            ]
        }).encode('utf-8')
        req = urllib.request.Request(
            f"{NLI_ENDPOINT}/classify-batch",
            data=payload,
            headers={"Content-Type": "application/json"},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data.get('results', [])
    except Exception as e:
        print(f"NLI batch classify failed: {e}")
        return [None] * len(pairs)


def check_contradictions(new_content: str, new_id: str) -> list[dict]:
    """
    Check if a new memory contradicts existing memories.

    Finds top-N semantically similar memories, then NLI-classifies each pair.
    Auto-creates KG edges and logs contradictions.

    Returns list of detected relationships:
        [{"memory_id": str, "type": "contradicts"|"supports", "confidence": float, "content_preview": str}]
    """
    if not _nli_available():
        return []

    # Find similar memories via pgvector
    try:
        from semantic_search import search_memories
        similar = search_memories(
            new_content,
            limit=MAX_SIMILAR_CHECK,
            threshold=0.3,
            register_recall=False,  # Don't pollute recall tracking
        )
    except Exception as e:
        print(f"Contradiction check: search failed: {e}")
        return []

    if not similar:
        return []

    # Filter out self-reference
    candidates = [m for m in similar if m.get('id') != new_id]
    if not candidates:
        return []

    # Batch NLI classification (search_memories returns 'preview' for content)
    pairs = [(m.get('preview', m.get('content', '')), new_content) for m in candidates]
    results = _classify_batch(pairs)

    detected = []
    for memory, nli_result in zip(candidates, results):
        if nli_result is None:
            continue

        scores = nli_result.get('scores', {})
        mem_id = memory.get('id', '')
        content_preview = memory.get('preview', memory.get('content', ''))[:100]

        # Contradiction detected
        if scores.get('contradiction', 0) > CONTRADICTION_THRESHOLD:
            confidence = scores['contradiction']
            detected.append({
                "memory_id": mem_id,
                "type": "contradicts",
                "confidence": round(confidence, 3),
                "content_preview": content_preview,
            })
            _create_kg_edge(new_id, mem_id, 'contradicts', confidence)
            _log_contradiction(new_id, mem_id, confidence, new_content, content_preview)
            _fire_cognitive_event('contradiction_detected')

        # Strong entailment — supports
        elif scores.get('entailment', 0) > ENTAILMENT_THRESHOLD:
            confidence = scores['entailment']
            detected.append({
                "memory_id": mem_id,
                "type": "supports",
                "confidence": round(confidence, 3),
                "content_preview": content_preview,
            })
            _create_kg_edge(new_id, mem_id, 'supports', confidence)

    if detected:
        contradictions = [d for d in detected if d['type'] == 'contradicts']
        supports = [d for d in detected if d['type'] == 'supports']
        print(f"Contradiction check: {len(contradictions)} contradictions, {len(supports)} supports detected")

    return detected


def _create_kg_edge(source_id: str, target_id: str, rel_type: str, conf: float):
    """Create a typed edge in the knowledge graph."""
    try:
        from knowledge_graph import add_edge
        add_edge(source_id, target_id, rel_type, confidence=conf,
                 evidence='nli-auto-detected', auto_extracted=True)
    except Exception as e:
        print(f"KG edge creation failed: {e}")


def _log_contradiction(new_id: str, existing_id: str, confidence: float,
                       new_content: str, existing_preview: str):
    """Log contradiction to rejection log for taste fingerprint."""
    try:
        from rejection_log import log_rejection
        log_rejection(
            category='contradiction_detected',
            reason=f'New memory [{new_id}] contradicts existing [{existing_id}] '
                   f'(confidence: {confidence:.2f})',
            target=f'new: {new_content[:80]}... vs existing: {existing_preview[:80]}...',
            tags=['auto-contradiction', 'nli-detected', 'mechanical-skepticism'],
            source='internal',
        )
    except Exception as e:
        print(f"Contradiction rejection log failed: {e}")


def _fire_cognitive_event(event_type: str):
    """Update cognitive state with contradiction event."""
    try:
        from cognitive_state import process_event
        process_event(event_type)
    except Exception:
        pass


def scan_memories(limit: int = 50) -> list[dict]:
    """
    Batch scan recent memories for contradictions among themselves.
    Useful for finding existing contradictions in the knowledge base.
    """
    if not _nli_available():
        print("NLI service not available")
        return []

    from db_adapter import get_db
    db = get_db()

    # Get recent memories
    rows = db.list_memories(limit=limit)
    if not rows:
        print("No memories to scan")
        return []

    all_detected = []
    checked = 0

    for i, mem in enumerate(rows):
        content = mem.get('content', '')
        mem_id = mem.get('id', '')
        if not content or not mem_id:
            continue

        # Check against remaining memories (avoid duplicate pairs)
        for other in rows[i + 1:]:
            other_content = other.get('content', '')
            other_id = other.get('id', '')
            if not other_content or not other_id:
                continue

            result = _classify_pair(content, other_content)
            checked += 1
            if result and result.get('scores', {}).get('contradiction', 0) > CONTRADICTION_THRESHOLD:
                confidence = result['scores']['contradiction']
                all_detected.append({
                    "memory_a": mem_id,
                    "memory_b": other_id,
                    "confidence": round(confidence, 3),
                    "preview_a": content[:80],
                    "preview_b": other_content[:80],
                })
                _create_kg_edge(mem_id, other_id, 'contradicts', confidence)

    print(f"Scanned {checked} pairs, found {len(all_detected)} contradictions")
    return all_detected


def report() -> str:
    """Generate a contradiction report from KG edges."""
    try:
        from db_adapter import get_db
        import psycopg2.extras
        db = get_db()
        with db._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT * FROM {db._table('typed_edges')}
                    WHERE relationship = 'contradicts'
                    ORDER BY confidence DESC
                """)
                edges = [dict(r) for r in cur.fetchall()]

        if not edges:
            return "No contradictions detected in knowledge graph."

        lines = [f"=== Contradiction Report ({len(edges)} detected) ===\n"]
        for edge in edges[:20]:
            lines.append(
                f"  [{edge.get('source_id', '?')}] contradicts [{edge.get('target_id', '?')}] "
                f"(confidence: {edge.get('confidence', 0):.2f})"
            )
        if len(edges) > 20:
            lines.append(f"  ... and {len(edges) - 20} more")
        return '\n'.join(lines)
    except Exception as e:
        return f"Report failed: {e}"


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]

    if not args or args[0] == 'status':
        available = _nli_available()
        print(f"NLI service: {'READY' if available else 'UNAVAILABLE'}")
        if available:
            try:
                req = urllib.request.Request(f"{NLI_ENDPOINT}/info", method='GET')
                with urllib.request.urlopen(req, timeout=5) as resp:
                    info = json.loads(resp.read().decode('utf-8'))
                    print(f"  Model: {info.get('model_id')}")
                    print(f"  Device: {info.get('device')}")
            except Exception:
                pass

    elif args[0] == 'check' and len(args) > 1:
        memory_id = args[1]
        from db_adapter import get_db
        db = get_db()
        row = db.get_memory(memory_id)
        if row:
            results = check_contradictions(row['content'], memory_id)
            for r in results:
                print(f"  {r['type']}: [{r['memory_id']}] (confidence: {r['confidence']}) "
                      f"— {r['content_preview']}")
            if not results:
                print("  No contradictions or strong entailments found.")
        else:
            print(f"Memory {memory_id} not found")

    elif args[0] == 'scan':
        limit = int(args[1]) if len(args) > 1 else 50
        results = scan_memories(limit=limit)
        for r in results:
            print(f"  [{r['memory_a']}] contradicts [{r['memory_b']}] "
                  f"(confidence: {r['confidence']})")

    elif args[0] == 'report':
        print(report())

    else:
        print("Usage: contradiction_detector.py [status|check <id>|scan [limit]|report]")
