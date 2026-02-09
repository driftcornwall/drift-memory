#!/usr/bin/env python3
"""
Gemma Bridge Expansion — Auto-discover vocabulary bridge terms using local LLM.

Uses Ollama + Gemma 3 4B to find academic/foreign-register terms in memories
that aren't in the vocabulary bridge, and generates operational synonyms.

Architecture:
- Only fires for novel terms (not already in seed or sidecar)
- Writes discoveries to vocabulary_map.json (sidecar)
- Memory embeddings must be regenerated after new terms are added
- Designed to run as batch scan, not on every store (latency matters)

Requires: Ollama running with gemma3:4b model pulled.

Usage:
    python gemma_bridge.py scan         # Scan dead memories for new terms
    python gemma_bridge.py check <text> # Check one text for novel terms
    python gemma_bridge.py status       # Show Ollama status and map size
"""

import json
import sys
from pathlib import Path

MEMORY_DIR = Path(__file__).parent
OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma3:4b"

PROMPT_TEMPLATE = """You are a vocabulary bridge for an AI agent's memory system.

The agent stores memories using academic/theoretical terms but searches using plain operational language.
Find specialized terms that would block retrieval because a search wouldn't use those words.

INCLUDE (these are the kind of terms we want):
- Single-word discipline-specific jargon: "reconsolidation", "autopoiesis", "semiosis", "eigenvalue"
- Two-word technical compounds: "synaptic plasticity", "predictive processing", "scale-free network"

EXCLUDE (do NOT return these):
- Common English words or phrases: "trust", "decay", "merge conflicts", "philosophical reflections"
- Anything a developer would naturally type in a search query
- Terms already in the known list below
- Multi-word phrases that are just normal descriptions (e.g. "trust-based decay", "memory system")

Synonyms should be the PLAIN words someone would actually search for.

Already known terms (skip these): {known_terms}

Text to analyze:
{content}

Return JSON array: [{{"term": "academic_term", "synonyms": "plain synonym 1, plain synonym 2"}}]
If no novel academic terms found, return: []"""


def _ollama_available() -> bool:
    """Check if Ollama is running and has the model."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            models = [m['name'] for m in data.get('models', [])]
            return any(MODEL in m for m in models)
    except Exception:
        return False


def _query_gemma(content: str, known_terms: list[str]) -> list[dict]:
    """Send content to Gemma and get back novel bridge terms."""
    import urllib.request

    known_str = ", ".join(known_terms[:50])  # Limit to avoid context overflow
    prompt = PROMPT_TEMPLATE.format(
        known_terms=known_str,
        content=content[:2000]  # Limit content length
    )

    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temp for consistent output
            "num_predict": 500
        }
    }).encode('utf-8')

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            response_text = result.get('response', '').strip()

            # Extract JSON from response (handle markdown code blocks)
            if '```' in response_text:
                start = response_text.index('```') + 3
                if response_text[start:start+4] == 'json':
                    start += 4
                end = response_text.index('```', start)
                response_text = response_text[start:end].strip()

            terms = json.loads(response_text)
            if isinstance(terms, list):
                return [t for t in terms if 'term' in t and 'synonyms' in t]
    except Exception as e:
        print(f"Gemma query failed: {e}", file=sys.stderr)

    return []


# Common words that Gemma often incorrectly identifies as academic jargon
_REJECT_TERMS = {
    'implementation', 'sync', 'synchronization', 'integration', 'optimization',
    'configuration', 'deployment', 'debugging', 'refactoring', 'testing',
    'validation', 'monitoring', 'logging', 'caching', 'authentication',
    'authorization', 'encryption', 'serialization', 'deserialization',
    'initialization', 'migration', 'pagination', 'aggregation',
    'merge conflicts', 'pull request', 'code review', 'bug fix',
    'behavioral patterns', 'trust-based decay', 'weight penalties',
    'self-evolution', 'trust tiers', 'schema interop',
}


def _is_quality_term(term: str) -> bool:
    """Filter out low-quality Gemma discoveries."""
    if term in _REJECT_TERMS:
        return False
    if len(term) < 4:
        return False
    # Reject if all words are common English (no Latin/Greek roots)
    words = term.split()
    if len(words) > 3:
        return False
    return True


def scan_dead_memories(limit: int = 20) -> dict:
    """
    Scan memories with low recall for novel bridge terms.

    Targets memories that are structurally present but functionally dead —
    exactly the ones that need vocabulary bridging.

    Returns:
        Dict with scan results: {scanned, terms_found, terms_added}
    """
    from vocabulary_bridge import VOCABULARY_MAP, add_term
    from memory_common import parse_memory_file, ACTIVE_DIR, CORE_DIR

    if not _ollama_available():
        return {"error": f"Ollama not running or {MODEL} not pulled"}

    known_terms = list(VOCABULARY_MAP.keys())
    stats = {"scanned": 0, "terms_found": 0, "terms_added": 0, "new_terms": []}

    # Find dead memories (low recall, sorted by session age)
    candidates = []
    for directory in [ACTIVE_DIR, CORE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            recall_count = metadata.get('recall_count', 0)
            if recall_count <= 1 and len(content) > 100:
                candidates.append((metadata.get('sessions_since_recall', 0), content, metadata.get('id', '')))

    # Sort by most neglected first
    candidates.sort(reverse=True)

    for _, content, mem_id in candidates[:limit]:
        stats["scanned"] += 1
        novel_terms = _query_gemma(content, known_terms)

        for term_info in novel_terms:
            term = term_info['term'].lower()
            synonyms = term_info['synonyms']

            # Skip if already known
            if term in VOCABULARY_MAP:
                continue

            # Quality filter: reject common English, too-short, or descriptive phrases
            if not _is_quality_term(term):
                continue

            stats["terms_found"] += 1
            add_term(term, synonyms)
            known_terms.append(term)  # Prevent duplicates within scan
            stats["terms_added"] += 1
            stats["new_terms"].append({"term": term, "synonyms": synonyms, "source_memory": mem_id})
            print(f"  NEW: {term} -> {synonyms} (from {mem_id})")

    return stats


def check_text(text: str) -> list[dict]:
    """Check a single text for novel bridge terms."""
    from vocabulary_bridge import VOCABULARY_MAP

    if not _ollama_available():
        print(f"Ollama not running or {MODEL} not pulled", file=sys.stderr)
        return []

    return _query_gemma(text, list(VOCABULARY_MAP.keys()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    if len(sys.argv) < 2:
        print("Usage:")
        print("  gemma_bridge.py scan [limit]  - Scan dead memories for new terms")
        print("  gemma_bridge.py check <text>  - Check one text for novel terms")
        print("  gemma_bridge.py status        - Show Ollama status and map size")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "status":
        available = _ollama_available()
        print(f"Ollama ({OLLAMA_URL}): {'ONLINE' if available else 'OFFLINE'}")
        print(f"Model: {MODEL}")

        from vocabulary_bridge import VOCABULARY_MAP, _SEED_MAP, SIDECAR_FILE
        seed_count = len(_SEED_MAP)
        total_count = len(VOCABULARY_MAP)
        print(f"Vocabulary map: {total_count} terms ({seed_count} seed, {total_count - seed_count} discovered)")
        print(f"Sidecar: {SIDECAR_FILE}")

    elif cmd == "scan":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        print(f"Scanning up to {limit} dead memories for novel terms...")
        result = scan_dead_memories(limit=limit)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\nScan complete:")
            print(f"  Scanned: {result['scanned']}")
            print(f"  Terms found: {result['terms_found']}")
            print(f"  Terms added: {result['terms_added']}")
            if result['new_terms']:
                print(f"\nNew terms:")
                for t in result['new_terms']:
                    print(f"  {t['term']} -> {t['synonyms']}")

    elif cmd == "check":
        text = ' '.join(sys.argv[2:])
        if Path(text).exists():
            text = Path(text).read_text(encoding='utf-8')
        terms = check_text(text)
        if terms:
            print(f"Novel terms found:")
            for t in terms:
                print(f"  {t['term']} -> {t['synonyms']}")
        else:
            print("No novel terms found.")

    else:
        print(f"Unknown command: {cmd}")
