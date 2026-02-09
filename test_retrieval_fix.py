#!/usr/bin/env python3
"""
Test: Does the vocabulary bridge + re-indexing make experiment sources findable?

Run after re-indexing with `python semantic_search.py index --force`.
Tests whether operational queries can find academic-register memories.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from semantic_search import search_memories

# Experiment 2 source memory IDs
SOURCE_IDS = {
    'ollzdo2n', '6w59yrfn', 'zdzoori9', 'db39vhky', 'd71h77hb',
    'vp2fm7fy', 'cchafa9d', 'gnqxh39g', '2z2ztnbi', '9av5bwg2',
    '9go71n9d', 'ccep2dfq', 'xoxfa7sv', '6zibj9dp', '53asfqlh',
    '14amo6cj', 'ce5mhz60', 'vbxsl4ek'
}

# Tests: operational query -> expected source that should match
TESTS = [
    # Bridge: reconsolidation -> "memory update, recall modification, memory rewrite"
    ("memory update", "9av5bwg2", "Reconsolidation (Nader & Hardt)"),

    # Bridge: metacognition -> "thinking about thinking, self-monitoring"
    ("thinking about thinking", "vp2fm7fy", "Theory of Mind and Metacognition"),

    # Bridge: autopoiesis -> "self-organization, self-maintaining system"
    ("self-organization autonomous identity", "vbxsl4ek", "Enactivism"),

    # Bridge: scale-free network -> "power-law graph, hub-dominated topology"
    ("power-law graph hub topology", "gnqxh39g", "Scale-Free Networks (Barabasi)"),

    # Bridge: free energy principle -> "surprise minimization, prediction error reduction"
    ("surprise minimization prediction error", "zdzoori9", "Free Energy Principle"),

    # Bridge: integrated information -> "consciousness measure, unified experience"
    ("consciousness measure unified experience", "53asfqlh", "IIT 4.0 (Tononi)"),

    # Bridge: eigenvalue -> "principal component, dominant mode, stable pattern"
    ("stable pattern dominant mode", "6w59yrfn", "Constructing a Reality (von Foerster)"),

    # Bridge: synaptic plasticity -> "connection strength change, edge weight update"
    ("edge weight update connection strength", "ccep2dfq", "Synaptic Plasticity"),

    # Generic: should find ANY experiment source
    ("embodied cognition agent architecture", None, "Any experiment source"),
]

print("=" * 70)
print("RETRIEVAL FIX VALIDATION TEST")
print("=" * 70)
print(f"Testing {len(TESTS)} operational queries against {len(SOURCE_IDS)} experiment sources")
print()

hits = 0
misses = 0

for query, expected_id, description in TESTS:
    results = search_memories(query, limit=10, threshold=0.2, register_recall=False)
    result_ids = {r['id'] for r in results}
    source_hits = [(r['score'], r['id']) for r in results if r['id'] in SOURCE_IDS]

    if source_hits:
        hits += 1
        best_score, best_id = max(source_hits)
        expected_match = best_id == expected_id if expected_id else True
        marker = "EXACT" if expected_match else "DIFFERENT"
        print(f"HIT [{marker}] Query: \"{query}\"")
        print(f"  Target: {description}")
        for score, mid in sorted(source_hits, reverse=True):
            print(f"  Source: [{score:.3f}] {mid}")
    else:
        misses += 1
        top = results[0] if results else None
        top_info = f"(top non-source: [{top['score']:.3f}] {top['id']})" if top else ""
        print(f"MISS Query: \"{query}\"")
        print(f"  Target: {description} {top_info}")
    print()

print("=" * 70)
print(f"RESULTS: {hits}/{len(TESTS)} queries found experiment sources")
print(f"  Hits: {hits}")
print(f"  Misses: {misses}")
if hits > 0:
    print(f"  VERDICT: Vocabulary bridge is {'WORKING' if hits >= len(TESTS) // 2 else 'PARTIALLY WORKING'}")
else:
    print(f"  VERDICT: Vocabulary bridge NOT WORKING — investigate embeddings")
print("=" * 70)
