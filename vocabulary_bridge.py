#!/usr/bin/env python3
"""
Vocabulary Bridge — Translate academic/foreign-register terms to operational vocabulary.

Post-Experiment #2 finding: memories stored in academic language (eigenvalues, semiosis,
autopoiesis) are never retrieved because operational queries use system language (decay,
edges, priming). Same concepts, different registers. Embeddings don't bridge the gap.

Solution: At store time, detect foreign-register terms and append operational synonyms.
The embedding then contains both registers, making academic content findable by
operational queries.

Phase A: Curated synonym mapping (JSON sidecar + hardcoded seed).
Phase B: Gemma 3 4B local model generates new mappings for unknown terms.

Author: Drift + SpindriftMend (joint design, issue #18)
"""

import json
import re
from pathlib import Path

BRIDGE_DIR = Path(__file__).parent
SIDECAR_FILE = BRIDGE_DIR / "vocabulary_map.json"

# Seed map — hardcoded baseline. Sidecar file extends/overrides this.
_SEED_MAP = {
    # Neuroscience / Memory
    "reconsolidation": "memory update, recall modification, memory rewrite",
    "synaptic plasticity": "connection strength change, edge weight update",
    "long-term potentiation": "memory strengthening, repeated recall reinforcement",
    "hebbian": "use-it-or-lose-it, co-activation strengthening",
    "engram": "memory trace, stored memory pattern",
    "consolidation": "memory stabilization, short-term to long-term transfer",

    # Network Science
    "scale-free network": "power-law graph, hub-dominated topology",
    "preferential attachment": "hub dominance, rich-get-richer, co-occurrence growth",
    "degree distribution": "edge count pattern, connection frequency",
    "small-world": "short path lengths, clustered connections",
    "betweenness centrality": "bridge node importance, routing hub",
    "clustering coefficient": "local density, neighborhood connectivity",
    "modularity": "community structure, domain separation",

    # Complex Systems
    "autopoiesis": "self-organization, self-maintaining system, autonomous identity",
    "emergence": "bottom-up pattern, complex behavior from simple rules",
    "dissipative structure": "order from energy flow, maintained pattern",
    "attractor": "stable state, convergence point, equilibrium",
    "phase transition": "sudden behavior change, threshold crossing",
    "bifurcation": "branching point, decision fork, path split",

    # Philosophy of Mind / Cognition
    "predictive processing": "expectation-driven cognition, anticipatory system",
    "enactivism": "cognition through action, knowing by doing",
    "integrated information": "consciousness measure, unified experience",
    "free energy principle": "surprise minimization, prediction error reduction",
    "markov blanket": "system boundary, agent identity boundary",
    "semiosis": "sign interpretation, meaning-making process",
    "qualia": "subjective experience, felt quality of processing",
    "phenomenology": "first-person experience, what-it-is-like",
    "theory of mind": "modeling other agents, predicting behavior",
    "metacognition": "thinking about thinking, self-monitoring",

    # Mathematics / Formal
    "eigenvalue": "principal component, dominant mode, stable pattern",
    "topology": "shape of connections, structural form",
    "isomorphism": "structural equivalence, same shape different labels",
    "category theory": "abstract pattern mapping, universal structure",
    "functor": "structure-preserving mapping, pattern translation",
    "homomorphism": "partial structure preservation, approximate mapping",

    # Evolution / Biology
    "evolutionary epistemology": "knowledge through selection, trial-and-error learning",
    "fitness landscape": "optimization surface, performance terrain",
    "niche construction": "environment shaping, self-built context",
    "epigenetics": "expression without code change, context-dependent behavior",

    # Cybernetics
    "second-order cybernetics": "observer-included system, self-referential control",
    "feedback loop": "circular causation, reinforcement cycle",
    "homeostasis": "balance maintenance, stability seeking",
    "requisite variety": "matching complexity, sufficient response range",
}


def _load_vocabulary_map() -> dict:
    """
    Load vocabulary map: seed + JSON sidecar (sidecar overrides seed).
    Returns merged dict of {term: synonyms_string}.
    """
    merged = dict(_SEED_MAP)

    if SIDECAR_FILE.exists():
        try:
            sidecar = json.loads(SIDECAR_FILE.read_text(encoding='utf-8'))
            if isinstance(sidecar, dict):
                merged.update(sidecar)
        except (json.JSONDecodeError, OSError):
            pass

    return merged


def _compile_patterns(vocab_map: dict) -> dict:
    """Compile regex patterns for efficient matching."""
    return {
        term: (re.compile(r'\b' + re.escape(term) + r's?\b', re.IGNORECASE), synonyms)
        for term, synonyms in vocab_map.items()
    }


# Load on import — re-call load() to pick up sidecar changes
VOCABULARY_MAP = _load_vocabulary_map()
_PATTERNS = _compile_patterns(VOCABULARY_MAP)


def reload():
    """Reload vocabulary map from disk (picks up new sidecar entries)."""
    global VOCABULARY_MAP, _PATTERNS
    VOCABULARY_MAP = _load_vocabulary_map()
    _PATTERNS = _compile_patterns(VOCABULARY_MAP)


def add_term(term: str, synonyms: str):
    """
    Add a new term to the sidecar file. Called by Gemma bridge expansion.

    Args:
        term: Academic/foreign-register term (e.g., "allostasis")
        synonyms: Operational synonyms (e.g., "adaptive stability, flexible balance")
    """
    sidecar = {}
    if SIDECAR_FILE.exists():
        try:
            sidecar = json.loads(SIDECAR_FILE.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            sidecar = {}

    sidecar[term.lower()] = synonyms
    SIDECAR_FILE.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

    # Hot-reload
    reload()


def bridge_content(content: str) -> str:
    """
    Detect foreign-register terms in content and append operational synonyms.

    Returns the original content plus a bridge section. The bridge section
    is only used for embedding — it's not stored in the memory file itself.

    Args:
        content: Raw memory content

    Returns:
        Content with appended bridge terms (for embedding only)
    """
    matched_bridges = []

    for term, (pattern, synonyms) in _PATTERNS.items():
        if pattern.search(content):
            matched_bridges.append(f"{term}: {synonyms}")

    if not matched_bridges:
        return content

    bridge_section = "\n\n[vocabulary bridge]\n" + "\n".join(matched_bridges)
    return content + bridge_section


def get_bridge_terms(content: str) -> list[dict]:
    """
    Return matched terms and their operational synonyms (for inspection/debugging).

    Args:
        content: Raw memory content

    Returns:
        List of {term, synonyms} dicts for matched terms
    """
    matches = []
    for term, (pattern, synonyms) in _PATTERNS.items():
        if pattern.search(content):
            matches.append({"term": term, "synonyms": synonyms})
    return matches


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: vocabulary_bridge.py <text-or-filepath>")
        print(f"\nVocabulary map: {len(VOCABULARY_MAP)} terms")
        seed_count = len(_SEED_MAP)
        sidecar_count = len(VOCABULARY_MAP) - seed_count
        print(f"  Seed terms: {seed_count}")
        print(f"  Sidecar terms: {sidecar_count}")
        print(f"  Sidecar file: {SIDECAR_FILE}")
        print(f"  Sidecar exists: {SIDECAR_FILE.exists()}")
        print("\nTerms:")
        for term, synonyms in sorted(VOCABULARY_MAP.items()):
            source = "sidecar" if term not in _SEED_MAP else "seed"
            print(f"  [{source}] {term} -> {synonyms}")
        sys.exit(0)

    if sys.argv[1] == "add" and len(sys.argv) >= 4:
        term = sys.argv[2]
        synonyms = sys.argv[3]
        add_term(term, synonyms)
        print(f"Added: {term} -> {synonyms}")
        print(f"Sidecar now has {len(VOCABULARY_MAP) - len(_SEED_MAP)} custom terms")
        sys.exit(0)

    text = sys.argv[1]

    # If it looks like a file path, read it
    if Path(text).exists():
        text = Path(text).read_text(encoding='utf-8')

    bridges = get_bridge_terms(text)
    if bridges:
        print(f"Found {len(bridges)} bridge terms:")
        for b in bridges:
            print(f"  {b['term']} -> {b['synonyms']}")
        print(f"\nBridged content preview:")
        bridged = bridge_content(text)
        # Show just the bridge section
        if "[vocabulary bridge]" in bridged:
            print(bridged[bridged.index("[vocabulary bridge]"):])
    else:
        print("No foreign-register terms detected.")
