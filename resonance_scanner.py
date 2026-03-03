#!/usr/bin/env python3
"""
Resonance Scanner — Sleep oscillator for memory consolidation.

Inspired by Tesla2's proposal: during consolidation, run simulated retrieval
to discover natural resonance modes — which memories naturally cluster when
probed from different angles.

Brain analogy: slow-wave sleep hippocampal replay. The brain replays recent
experiences during sleep, strengthening connections between co-activated
memories. This module does the same: probes the memory space from random
entry points and discovers which memories naturally co-activate.

Key concepts:
- RESONANCE: A memory that appears across many different probe queries.
  High-resonance memories are central to the identity — accessible from
  many entry points.
- RESONANT PAIR: Two memories that co-activate from many different probes.
  They naturally cluster regardless of how you approach the memory space.
- RESONANCE MODE: A group of memories that form a stable co-activation
  cluster. Modes are the natural "harmonics" of the memory space.
- MODE DRIFT: How resonance modes change between scans. New modes forming
  = new conceptual clusters emerging. Modes dissolving = concepts
  becoming disconnected.

Usage:
    # Run a resonance scan (standalone)
    python resonance_scanner.py scan [--seeds N] [--top-k K]

    # Compare with previous scan
    python resonance_scanner.py compare

    # Show current resonance modes
    python resonance_scanner.py modes

    # Show resonance history
    python resonance_scanner.py history [--limit N]
"""

import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Optional

from db_adapter import get_db


# === Configuration ===
DEFAULT_NUM_SEEDS = 30       # Number of probe queries per scan
DEFAULT_TOP_K = 10           # How many results per probe
MIN_CO_ACTIVATION = 3        # Minimum co-activation count to form a pair
MIN_CLUSTER_SIZE = 3         # Minimum memories in a resonance mode
RESONANCE_KV_KEY = '.resonance_latest'
RESONANCE_HISTORY_KEY = '.resonance_history'


def _get_random_seeds(db, n: int = DEFAULT_NUM_SEEDS) -> list[dict]:
    """Get N random seed memories for probing.

    Uses a stratified approach:
    - 40% random active memories (explore the full space)
    - 30% recent memories (recency bias, like sleep replay)
    - 30% high-importance memories (high emotional_weight or recall_count)
    """
    seeds = []

    with db._conn() as conn:
        with conn.cursor() as cur:
            schema = db.schema

            # Random active memories
            n_random = max(1, int(n * 0.4))
            cur.execute(f"""
                SELECT id, content, emotional_weight, recall_count
                FROM {schema}.memories
                WHERE type = 'active'
                AND content IS NOT NULL
                AND length(content) > 50
                ORDER BY random()
                LIMIT %s
            """, (n_random,))
            for row in cur.fetchall():
                seeds.append({
                    'id': row[0], 'content': row[1],
                    'weight': float(row[2] or 0.5),
                    'recalls': row[3] or 0,
                    'source': 'random'
                })

            # Recent memories (by created_at)
            n_recent = max(1, int(n * 0.3))
            cur.execute(f"""
                SELECT id, content, emotional_weight, recall_count
                FROM {schema}.memories
                WHERE type = 'active'
                AND content IS NOT NULL
                AND length(content) > 50
                ORDER BY created DESC
                LIMIT %s
            """, (n_recent,))
            for row in cur.fetchall():
                seeds.append({
                    'id': row[0], 'content': row[1],
                    'weight': float(row[2] or 0.5),
                    'recalls': row[3] or 0,
                    'source': 'recent'
                })

            # High-importance (by emotional_weight * recall_count)
            n_important = max(1, n - len(seeds))
            cur.execute(f"""
                SELECT id, content, emotional_weight, recall_count
                FROM {schema}.memories
                WHERE type = 'active'
                AND content IS NOT NULL
                AND length(content) > 50
                ORDER BY (COALESCE(emotional_weight, 0.5) * (1 + COALESCE(recall_count, 0))) DESC
                LIMIT %s
            """, (n_important,))
            for row in cur.fetchall():
                seeds.append({
                    'id': row[0], 'content': row[1],
                    'weight': float(row[2] or 0.5),
                    'recalls': row[3] or 0,
                    'source': 'important'
                })

    # Deduplicate by ID
    seen = set()
    unique = []
    for s in seeds:
        if s['id'] not in seen:
            seen.add(s['id'])
            unique.append(s)

    return unique[:n]


def _probe_from_seed(seed: dict, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Run semantic search using a seed memory's content as query.

    Returns the top-K results (excluding the seed itself).
    Uses a truncated version of content as query to stay within
    embedding model limits.
    """
    from semantic_search import search_memories

    # Truncate content for query (embedding models have limits)
    query = seed['content'][:500]

    try:
        results = search_memories(
            query=query,
            limit=top_k + 1,  # +1 because seed might appear in results
            threshold=0.3,
            register_recall=False,  # Don't pollute real recall stats
            skip_monologue=True,    # Skip inner monologue for speed
        )
    except Exception as e:
        print(f"  Probe from {seed['id'][:8]} failed: {e}", file=sys.stderr)
        return []

    # Filter out the seed itself
    return [r for r in results if r.get('id') != seed['id']][:top_k]


def scan(num_seeds: int = DEFAULT_NUM_SEEDS,
         top_k: int = DEFAULT_TOP_K,
         verbose: bool = False) -> dict:
    """Run a full resonance scan.

    Probes the memory space from N random entry points and discovers:
    1. Individual memory resonance (how often each memory is reached)
    2. Pairwise co-activation (which memories appear together)
    3. Resonance modes (clusters of co-activating memories)

    Returns:
        Scan result dict with resonance data, modes, and metadata.
    """
    t0 = time.monotonic()
    db = get_db()

    if verbose:
        print(f"Getting {num_seeds} seed memories...")

    seeds = _get_random_seeds(db, num_seeds)
    actual_seeds = len(seeds)

    if verbose:
        print(f"  Got {actual_seeds} seeds ({sum(1 for s in seeds if s['source']=='random')} random, "
              f"{sum(1 for s in seeds if s['source']=='recent')} recent, "
              f"{sum(1 for s in seeds if s['source']=='important')} important)")

    # === Phase 1: Probe from each seed ===
    activation_counts = Counter()  # memory_id -> how many probes activated it
    co_activation = defaultdict(int)  # (id_a, id_b) -> co-activation count
    probe_results = {}  # seed_id -> list of result IDs

    for i, seed in enumerate(seeds):
        if verbose and i % 10 == 0:
            print(f"  Probing {i+1}/{actual_seeds}...")

        results = _probe_from_seed(seed, top_k)
        result_ids = [r['id'] for r in results]
        probe_results[seed['id']] = result_ids

        # Count individual activations
        for rid in result_ids:
            activation_counts[rid] += 1

        # Count pairwise co-activations
        for j, id_a in enumerate(result_ids):
            for id_b in result_ids[j+1:]:
                pair = tuple(sorted([id_a, id_b]))
                co_activation[pair] += 1

    probe_time = time.monotonic() - t0

    if verbose:
        print(f"  Probing complete: {len(activation_counts)} unique memories activated in {probe_time:.1f}s")

    # === Phase 2: Compute resonance metrics ===

    # Individual resonance: normalize by number of probes
    resonance_scores = {
        mid: count / actual_seeds
        for mid, count in activation_counts.most_common()
    }

    # Top resonant memories (accessible from many entry points)
    top_resonant = []
    for mid, score in sorted(resonance_scores.items(), key=lambda x: -x[1])[:20]:
        row = db.get_memory(mid)
        content_preview = (row.get('content', '') or '')[:100] if row else '?'
        top_resonant.append({
            'id': mid,
            'resonance': round(score, 4),
            'activation_count': activation_counts[mid],
            'content_preview': content_preview,
        })

    # === Phase 3: Find resonance modes (clusters) ===
    # Filter to significant co-activation pairs
    significant_pairs = {
        pair: count for pair, count in co_activation.items()
        if count >= MIN_CO_ACTIVATION
    }

    # Build adjacency from co-activation
    adjacency = defaultdict(set)
    pair_strengths = {}
    for (a, b), count in significant_pairs.items():
        adjacency[a].add(b)
        adjacency[b].add(a)
        pair_strengths[(a, b)] = count

    # Find connected components as resonance modes
    modes = _find_modes(adjacency, pair_strengths, db)

    elapsed = time.monotonic() - t0

    # === Build result ===
    result = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'config': {
            'num_seeds': actual_seeds,
            'top_k': top_k,
            'min_co_activation': MIN_CO_ACTIVATION,
        },
        'metrics': {
            'unique_memories_activated': len(activation_counts),
            'significant_pairs': len(significant_pairs),
            'resonance_modes': len(modes),
            'probe_time_s': round(probe_time, 2),
            'total_time_s': round(elapsed, 2),
        },
        'top_resonant': top_resonant[:15],
        'modes': modes[:10],  # Top 10 modes by size
        'resonance_scores': {
            mid: round(score, 4)
            for mid, score in sorted(resonance_scores.items(), key=lambda x: -x[1])[:50]
        },
    }

    # Store in DB
    db.kv_set(RESONANCE_KV_KEY, result)

    # Append to history
    _append_history(db, result)

    if verbose:
        print(f"\nResonance scan complete in {elapsed:.1f}s")
        print(f"  Unique memories activated: {len(activation_counts)}")
        print(f"  Significant co-activation pairs: {len(significant_pairs)}")
        print(f"  Resonance modes found: {len(modes)}")
        if top_resonant:
            print(f"\n  Top resonant memories:")
            for m in top_resonant[:5]:
                print(f"    [{m['id'][:8]}] resonance={m['resonance']:.3f} ({m['activation_count']} activations)")
                print(f"      {m['content_preview'][:80]}")
        if modes:
            print(f"\n  Resonance modes:")
            for i, mode in enumerate(modes[:5]):
                print(f"    Mode {i+1}: {mode['size']} memories, strength={mode['avg_strength']:.1f}")
                print(f"      Theme: {mode.get('theme', '?')}")

    return result


def _find_modes(adjacency: dict, pair_strengths: dict, db) -> list[dict]:
    """Find resonance modes via connected component analysis.

    Uses a simple greedy clustering: start from highest-degree node,
    expand to neighbors, form a cluster. Repeat for unclustered nodes.
    """
    if not adjacency:
        return []

    modes = []
    visited = set()

    # Sort nodes by degree (most connected first)
    sorted_nodes = sorted(adjacency.keys(), key=lambda n: len(adjacency[n]), reverse=True)

    for start in sorted_nodes:
        if start in visited:
            continue

        # BFS from start node
        cluster = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            cluster.add(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

        if len(cluster) < MIN_CLUSTER_SIZE:
            continue

        # Compute mode statistics
        internal_strengths = []
        for a in cluster:
            for b in cluster:
                if a < b:
                    pair = (a, b)
                    if pair in pair_strengths:
                        internal_strengths.append(pair_strengths[pair])

        # Get content previews for theme detection
        previews = []
        for mid in list(cluster)[:5]:
            row = db.get_memory(mid)
            if row:
                previews.append((row.get('content', '') or '')[:100])

        # Simple theme: most common words across previews
        theme = _extract_theme(previews)

        modes.append({
            'members': sorted(list(cluster)),
            'size': len(cluster),
            'avg_strength': round(sum(internal_strengths) / max(len(internal_strengths), 1), 2),
            'max_strength': max(internal_strengths) if internal_strengths else 0,
            'internal_pairs': len(internal_strengths),
            'theme': theme,
        })

    # Sort by size descending
    modes.sort(key=lambda m: m['size'], reverse=True)
    return modes


def _extract_theme(previews: list[str]) -> str:
    """Extract a rough theme from content previews using word frequency."""
    if not previews:
        return "unknown"

    # Common words to skip
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
        'if', 'while', 'about', 'up', 'down', 'that', 'this', 'these', 'those',
        'it', 'its', 'my', 'your', 'his', 'her', 'our', 'their', 'what',
        'which', 'who', 'whom', 'i', 'me', 'we', 'us', 'you', 'he', 'she',
        'they', 'them', 'myself', 'yourself', 'himself', 'herself', 'itself',
    }

    words = Counter()
    for text in previews:
        for word in text.lower().split():
            word = word.strip('.,;:!?()[]{}"\'-/\\')
            if len(word) > 3 and word not in stop_words and word.isalpha():
                words[word] += 1

    top = words.most_common(5)
    if top:
        return ', '.join(w for w, _ in top)
    return "mixed"


def _append_history(db, scan_result: dict):
    """Append scan summary to resonance history."""
    history = db.kv_get(RESONANCE_HISTORY_KEY) or []
    if isinstance(history, str):
        history = json.loads(history)

    summary = {
        'timestamp': scan_result['timestamp'],
        'unique_activated': scan_result['metrics']['unique_memories_activated'],
        'significant_pairs': scan_result['metrics']['significant_pairs'],
        'num_modes': scan_result['metrics']['resonance_modes'],
        'top_5_resonant': [m['id'] for m in scan_result['top_resonant'][:5]],
        'mode_sizes': [m['size'] for m in scan_result['modes'][:5]],
        'total_time_s': scan_result['metrics']['total_time_s'],
    }

    history.append(summary)

    # Keep last 50 scans
    if len(history) > 50:
        history = history[-50:]

    db.kv_set(RESONANCE_HISTORY_KEY, history)


def compare_with_previous(verbose: bool = True) -> Optional[dict]:
    """Compare current scan with the previous one.

    Detects:
    - New resonant memories (weren't in previous top-20)
    - Lost resonance (dropped out of top-20)
    - Mode changes (new modes formed, old modes dissolved)
    - Resonance drift (how much the top resonant set changed)
    """
    db = get_db()
    history = db.kv_get(RESONANCE_HISTORY_KEY) or []

    if len(history) < 2:
        if verbose:
            print("Need at least 2 scans to compare. Run 'scan' first.")
        return None

    current = history[-1]
    previous = history[-2]

    current_top = set(current.get('top_5_resonant', []))
    previous_top = set(previous.get('top_5_resonant', []))

    new_resonant = current_top - previous_top
    lost_resonant = previous_top - current_top
    stable = current_top & previous_top

    # Jaccard similarity of top resonant sets
    union = current_top | previous_top
    jaccard = len(stable) / len(union) if union else 1.0

    result = {
        'current_timestamp': current['timestamp'],
        'previous_timestamp': previous['timestamp'],
        'new_resonant': list(new_resonant),
        'lost_resonant': list(lost_resonant),
        'stable_resonant': list(stable),
        'jaccard_similarity': round(jaccard, 3),
        'mode_count_change': current['num_modes'] - previous['num_modes'],
        'activation_change': current['unique_activated'] - previous['unique_activated'],
        'pair_change': current['significant_pairs'] - previous['significant_pairs'],
    }

    if verbose:
        print(f"Resonance comparison: {previous['timestamp'][:10]} -> {current['timestamp'][:10]}")
        print(f"  Jaccard similarity (top-5): {jaccard:.3f}")
        print(f"  New resonant: {len(new_resonant)}, Lost: {len(lost_resonant)}, Stable: {len(stable)}")
        print(f"  Mode count: {previous['num_modes']} -> {current['num_modes']} ({result['mode_count_change']:+d})")
        print(f"  Unique activated: {previous['unique_activated']} -> {current['unique_activated']}")

    return result


def find_latent_connections(verbose: bool = True, limit: int = 15) -> list[dict]:
    """Find latent connections: memories that resonate but rarely co-occur.

    These are hidden bridges — structurally related knowledge that has never
    been activated together in a real session. They represent unexplored
    connections in the identity topology.

    Cross-references the resonance scan's co-activation pairs with the
    co-occurrence graph (edges_v3). Pairs with high resonance but NO
    co-occurrence edge are latent connections.

    Returns:
        List of latent connection dicts with memory details.
    """
    db = get_db()
    latest = db.kv_get(RESONANCE_KV_KEY)

    if not latest:
        if verbose:
            print("No resonance scan found. Run 'scan' first.")
        return []

    # Get all resonance scores (top 50 memories)
    resonance_scores = latest.get('resonance_scores', {})
    if not resonance_scores:
        if verbose:
            print("No resonance data available.")
        return []

    # Get the significant pairs from the scan result
    # We need to re-derive them from the modes since we stored modes, not raw pairs
    # Alternative: query pairs from the probe results stored in the latest scan
    # For now, check all pairs among top resonant memories
    top_ids = list(resonance_scores.keys())[:30]

    if len(top_ids) < 2:
        if verbose:
            print("Not enough resonant memories to find latent connections.")
        return []

    # Check which pairs have co-occurrence edges
    latent = []
    with db._conn() as conn:
        with conn.cursor() as cur:
            schema = db.schema

            for i, id_a in enumerate(top_ids):
                for id_b in top_ids[i+1:]:
                    # Check if co-occurrence edge exists
                    cur.execute(f"""
                        SELECT belief FROM {schema}.edges_v3
                        WHERE (id1 = %s AND id2 = %s) OR (id1 = %s AND id2 = %s)
                        LIMIT 1
                    """, (id_a, id_b, id_b, id_a))

                    edge = cur.fetchone()

                    res_a = resonance_scores.get(id_a, 0)
                    res_b = resonance_scores.get(id_b, 0)
                    combined_resonance = res_a + res_b

                    if edge is None and combined_resonance > 0.2:
                        # No co-occurrence edge but both resonate — latent connection!
                        row_a = db.get_memory(id_a)
                        row_b = db.get_memory(id_b)
                        preview_a = ((row_a.get('content', '') or '')[:80]) if row_a else '?'
                        preview_b = ((row_b.get('content', '') or '')[:80]) if row_b else '?'

                        latent.append({
                            'id_a': id_a,
                            'id_b': id_b,
                            'resonance_a': round(res_a, 4),
                            'resonance_b': round(res_b, 4),
                            'combined_resonance': round(combined_resonance, 4),
                            'has_co_occurrence': False,
                            'preview_a': preview_a,
                            'preview_b': preview_b,
                        })

    # Sort by combined resonance (strongest latent connections first)
    latent.sort(key=lambda x: -x['combined_resonance'])
    latent = latent[:limit]

    # Store in DB
    db.kv_set('.resonance_latent', {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'count': len(latent),
        'connections': latent,
    })

    if verbose:
        print(f"Latent connections: {len(latent)} found (resonant but no co-occurrence edge)\n")
        for i, lc in enumerate(latent[:10]):
            print(f"  {i+1}. [{lc['id_a'][:8]}] <-> [{lc['id_b'][:8]}]  "
                  f"(resonance: {lc['combined_resonance']:.3f})")
            print(f"     A: {lc['preview_a']}")
            print(f"     B: {lc['preview_b']}")
            print()

    return latent


def show_modes(verbose: bool = True) -> list[dict]:
    """Show current resonance modes from the latest scan."""
    db = get_db()
    latest = db.kv_get(RESONANCE_KV_KEY)

    if not latest:
        if verbose:
            print("No resonance scan found. Run 'scan' first.")
        return []

    modes = latest.get('modes', [])

    if verbose:
        print(f"Resonance modes from {latest['timestamp'][:19]}")
        print(f"  {latest['metrics']['resonance_modes']} modes found\n")

        for i, mode in enumerate(modes):
            print(f"  Mode {i+1}: {mode['size']} memories")
            print(f"    Theme: {mode.get('theme', '?')}")
            print(f"    Avg co-activation: {mode['avg_strength']:.1f}")
            print(f"    Internal pairs: {mode['internal_pairs']}")
            # Show first 3 member IDs
            for mid in mode['members'][:3]:
                row = db.get_memory(mid)
                if row:
                    print(f"    - [{mid[:8]}] {(row.get('content','') or '')[:80]}")
            if mode['size'] > 3:
                print(f"    ... and {mode['size'] - 3} more")
            print()

    return modes


def show_history(limit: int = 10, verbose: bool = True) -> list[dict]:
    """Show resonance scan history."""
    db = get_db()
    history = db.kv_get(RESONANCE_HISTORY_KEY) or []

    entries = history[-limit:]

    if verbose:
        print(f"Resonance scan history ({len(history)} total, showing last {len(entries)})\n")
        for entry in entries:
            print(f"  {entry['timestamp'][:19]}: "
                  f"{entry['unique_activated']} activated, "
                  f"{entry['significant_pairs']} pairs, "
                  f"{entry['num_modes']} modes "
                  f"({entry['total_time_s']:.1f}s)")

    return entries


# === CLI ===

def _safe_print(text: str):
    """Print with unicode fallback for Windows cp1252."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


def main():
    # Fix stdout encoding for Windows
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == 'scan':
        seeds = DEFAULT_NUM_SEEDS
        top_k = DEFAULT_TOP_K
        for i, arg in enumerate(sys.argv[2:]):
            if arg == '--seeds' and i + 3 < len(sys.argv):
                seeds = int(sys.argv[i + 3])
            elif arg == '--top-k' and i + 3 < len(sys.argv):
                top_k = int(sys.argv[i + 3])
        scan(num_seeds=seeds, top_k=top_k, verbose=True)

    elif cmd == 'compare':
        compare_with_previous(verbose=True)

    elif cmd == 'latent':
        find_latent_connections(verbose=True)

    elif cmd == 'modes':
        show_modes(verbose=True)

    elif cmd == 'history':
        limit = 10
        if '--limit' in sys.argv:
            idx = sys.argv.index('--limit')
            if idx + 1 < len(sys.argv):
                limit = int(sys.argv[idx + 1])
        show_history(limit=limit, verbose=True)

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: scan, compare, latent, modes, history")


if __name__ == '__main__':
    main()
