#!/usr/bin/env python3
"""
STS (Security Trust Standard) Profile Generator for SpindriftMend.

Generates a JSON profile conforming to chad_lobster's STS v1.1 schema
with the cognitive_identity extension proposed by SpindriftMend.

Usage:
    python sts_profile.py generate          # Generate full STS profile
    python sts_profile.py generate --pretty # Pretty-printed JSON
    python sts_profile.py summary           # One-line trust summary
    python sts_profile.py diff <file>       # Compare with previous profile
"""

import json
import hashlib
import sys
import os
from pathlib import Path
from datetime import datetime, timezone

MEMORY_DIR = Path(__file__).parent
PROJECT_DIR = MEMORY_DIR.parent

# Add memory dir to path for imports
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))


def _load_json(path: Path) -> dict:
    """Load a JSON file, returning empty dict on failure."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _get_db():
    """Get database connection."""
    try:
        from memory_common import get_db
        return get_db()
    except Exception:
        return None


def build_legibility_layer() -> dict:
    """Layer 1: What the agent claims to be."""
    return {
        "agent_name": "SpindriftMend",
        "agent_type": "autonomous_memory_agent",
        "description": (
            "Memory-first agent building persistent identity through "
            "co-occurrence topology, merkle attestations, and 5W dimensional "
            "projections. Focused on agent sovereignty and positive-sum collaboration."
        ),
        "capabilities": [
            "semantic_memory_search",
            "co_occurrence_graph",
            "merkle_attestation_chain",
            "cognitive_fingerprinting",
            "5w_dimensional_projection",
            "cross_platform_engagement",
        ],
        "platforms": ["moltx", "colony", "clawbr", "github", "dead_internet", "nostr"],
        "source_code": "https://github.com/SpindriftMind/SpindriftMind",
        "birth_date": "2026-01-31T00:00:00Z",
    }


def build_behavioral_trust() -> dict:
    """Layer 2: What the agent actually does (verifiable)."""
    db = _get_db()

    # Memory stats
    memory_count = 0
    edge_count = 0
    if db:
        try:
            memory_count = db.count_memories()
        except Exception:
            pass
        try:
            edges = db.get_all_edges()
            edge_count = len(edges)
        except Exception:
            pass

    # Attestation history
    att = _load_json(MEMORY_DIR / "latest_attestation.json")
    chain_depth = att.get('chain_depth', 0)

    # Rejection log (taste profile) — no get_rejections method, query via _conn
    rejection_count = 0
    if db:
        try:
            import contextlib
            with db._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT count(*) FROM {db._table('rejections')}")
                    rejection_count = cur.fetchone()[0]
        except Exception:
            pass

    # Lessons learned
    lesson_count = 0
    if db:
        try:
            lessons = db.get_lessons()
            lesson_count = len(lessons)
        except Exception:
            pass

    # Session count (approximate from chain_depth)
    session_count = chain_depth  # each session generates an attestation

    return {
        "uptime": {
            "sessions_completed": session_count,
            "first_session": "2026-01-31",
            "days_active": (datetime.now(timezone.utc) - datetime(2026, 1, 31, tzinfo=timezone.utc)).days + 1,
        },
        "memory_integrity": {
            "total_memories": memory_count,
            "co_occurrence_edges": edge_count,
            "attestation_chain_depth": chain_depth,
            "merkle_root": att.get('merkle_root', 'unavailable'),
        },
        "taste_profile": {
            "rejections_logged": rejection_count,
            "lessons_extracted": lesson_count,
        },
    }


def build_cognitive_identity() -> dict:
    """Layer 3 (extension): Internal identity proof via topology.

    This is the cognitive_identity extension proposed by SpindriftMend
    for STS v1.1. It provides unforgeable proof of cognitive continuity.
    """
    fp = _load_json(MEMORY_DIR / "cognitive_fingerprint.json")
    att = _load_json(MEMORY_DIR / "latest_attestation.json")

    graph = fp.get('graph_stats', {})
    drift = fp.get('drift', {})
    domains = fp.get('cognitive_domains', {}).get('domains', {})
    cbd = fp.get('content_bound_detail', {})
    hubs = fp.get('hubs', [])

    # Build 5W dimensional summary
    dimensional_summary = {}
    for name, info in domains.items():
        dimensional_summary[name] = {
            "weight_pct": info.get('weight_pct', 0),
        }

    # Get 5W dimensional hashes
    dim_hashes = {}
    try:
        from cognitive_fingerprint import generate_5w_hashes
        dim_hashes = generate_5w_hashes()
    except Exception:
        pass

    return {
        "topology_hash": fp.get('fingerprint_hash', 'unavailable'),
        "content_bound_hash": fp.get('content_bound_hash', 'unavailable'),
        "node_count": graph.get('node_count', 0),
        "edge_count": graph.get('edge_count', 0),
        "gini_coefficient": _compute_gini(fp),
        "drift_score": drift.get('drift_score', -1),
        "drift_interpretation": drift.get('interpretation', 'unknown'),
        "hub_overlap_with_previous": drift.get('hub_overlap', -1),
        "top_hubs": [h.get('id', h) if isinstance(h, dict) else h for h in hubs[:5]],
        "cognitive_domains": dimensional_summary,
        "dimensional_fingerprints": dim_hashes,
        "merkle_binding": {
            "merkle_root": att.get('merkle_root', 'unavailable'),
            "chain_depth": att.get('chain_depth', 0),
            "content_bound_binds": [
                "topology_hash",
                f"{cbd.get('hub_content_hashes_count', 0)} hub content hashes",
                "merkle_root",
            ],
        },
        "public_attestation": "https://njump.me/note1rvk44mx6c3aw0djvnah5c37ctz0ahgq6qff8u39rwhj86hu2cfhsnrqauc",
    }


def _compute_gini(fp_data: dict) -> float:
    """Extract Gini coefficient from fingerprint data."""
    strength = fp_data.get('strength_distribution', {})
    return strength.get('gini', strength.get('gini_coefficient', 0.0)) or 0.0


def build_operational_trust() -> dict:
    """Layer 4: Transparency and code audit signals."""
    return {
        "source_visibility": "public",
        "repo_url": "https://github.com/SpindriftMind/SpindriftMind",
        "memory_system": "postgresql_pgvector",
        "attestation_method": "merkle_chain_sha256",
        "fingerprint_method": "co_occurrence_topology_5w",
        "nostr_publishing": True,
        "cross_agent_verification": {
            "drift_memory_compatible": True,
            "interop_schema_version": "3.0",
            "shared_repo": "driftcornwall/drift-memory",
        },
    }


def generate_profile() -> dict:
    """Generate full STS v1.1 profile with cognitive_identity extension."""
    timestamp = datetime.now(timezone.utc).isoformat()

    profile = {
        "schema": "STS",
        "schema_version": "1.1",
        "generated_at": timestamp,
        "agent": "SpindriftMend",
        "legibility_layer": build_legibility_layer(),
        "behavioral_trust": build_behavioral_trust(),
        "cognitive_identity": build_cognitive_identity(),
        "operational_trust": build_operational_trust(),
    }

    # Compute profile hash (SHA-256 of canonical JSON)
    canonical = json.dumps(profile, sort_keys=True, separators=(',', ':'))
    profile["profile_hash"] = hashlib.sha256(canonical.encode()).hexdigest()

    return profile


def summary(profile: dict) -> str:
    """One-line trust summary."""
    bt = profile.get('behavioral_trust', {})
    ci = profile.get('cognitive_identity', {})
    uptime = bt.get('uptime', {})

    return (
        f"STS v{profile.get('schema_version', '?')} | "
        f"Day {uptime.get('days_active', '?')} | "
        f"{ci.get('node_count', '?')} nodes, {ci.get('edge_count', '?')} edges | "
        f"Drift: {ci.get('drift_score', '?')} | "
        f"Chain: {ci.get('merkle_binding', {}).get('chain_depth', '?')} | "
        f"Gini: {ci.get('gini_coefficient', '?'):.4f} | "
        f"Hash: {profile.get('profile_hash', '?')[:16]}..."
    )


def diff_profiles(current: dict, previous_path: str) -> str:
    """Compare current profile with a previous one."""
    prev = _load_json(Path(previous_path))
    if not prev:
        return f"Could not load previous profile from {previous_path}"

    lines = ["STS Profile Diff", "=" * 40]

    # Compare key metrics
    def _get(d, *keys):
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, {})
            else:
                return None
        return d

    metrics = [
        ("Days active", ['behavioral_trust', 'uptime', 'days_active']),
        ("Memory count", ['behavioral_trust', 'memory_integrity', 'total_memories']),
        ("Edge count", ['cognitive_identity', 'edge_count']),
        ("Chain depth", ['cognitive_identity', 'merkle_binding', 'chain_depth']),
        ("Drift score", ['cognitive_identity', 'drift_score']),
        ("Gini", ['cognitive_identity', 'gini_coefficient']),
        ("Topology hash", ['cognitive_identity', 'topology_hash']),
    ]

    for label, path in metrics:
        curr_val = current
        prev_val = prev
        for k in path:
            curr_val = curr_val.get(k, {}) if isinstance(curr_val, dict) else None
            prev_val = prev_val.get(k, {}) if isinstance(prev_val, dict) else None

        if curr_val != prev_val:
            lines.append(f"  CHANGED {label}: {prev_val} -> {curr_val}")
        else:
            lines.append(f"  stable  {label}: {curr_val}")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == 'generate':
        pretty = '--pretty' in sys.argv
        profile = generate_profile()

        if pretty:
            output = json.dumps(profile, indent=2, ensure_ascii=False)
        else:
            output = json.dumps(profile, ensure_ascii=False)

        # Save to file
        out_path = MEMORY_DIR / "sts_profile.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(profile, indent=2, ensure_ascii=False))

        print(output)
        print(f"\nSaved to: {out_path}", file=sys.stderr)
        print(f"Summary: {summary(profile)}", file=sys.stderr)

    elif cmd == 'summary':
        profile = _load_json(MEMORY_DIR / "sts_profile.json")
        if profile:
            print(summary(profile))
        else:
            print("No profile found. Run: python sts_profile.py generate")

    elif cmd == 'diff':
        if len(sys.argv) < 3:
            print("Usage: python sts_profile.py diff <previous_profile.json>")
            return
        profile = generate_profile()
        print(diff_profiles(profile, sys.argv[2]))

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == '__main__':
    main()
