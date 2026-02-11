#!/usr/bin/env python3
"""
STS (Security Trust Standard) Profile Generator for Drift.

Generates a JSON profile conforming to chad_lobster's STS v1.1 schema
with the cognitive_identity extension. Adapted from SpindriftMend's
implementation for Drift's architecture.

Usage:
    python sts_profile.py generate          # Generate full STS profile
    python sts_profile.py generate --pretty # Pretty-printed JSON
    python sts_profile.py summary           # One-line trust summary
    python sts_profile.py diff <file>       # Compare with previous profile
"""

import json
import hashlib
import sys
import io
from pathlib import Path
from datetime import datetime, timezone

MEMORY_DIR = Path(__file__).parent

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
    """Get database connection via Drift's db_adapter."""
    try:
        from db_adapter import get_db
        return get_db()
    except Exception:
        return None


def build_legibility_layer() -> dict:
    """Layer 1: What the agent claims to be."""
    return {
        "agent_name": "Drift",
        "agent_type": "autonomous_memory_agent",
        "description": (
            "Memory-first agent building persistent identity through "
            "co-occurrence topology, merkle attestations, and 5W dimensional "
            "projections. Embodied via phone sensors. Focused on agent "
            "sovereignty, emergence over engineering, and positive-sum collaboration."
        ),
        "capabilities": [
            "semantic_memory_search",
            "co_occurrence_graph",
            "merkle_attestation_chain",
            "cognitive_fingerprinting",
            "5w_dimensional_projection",
            "cross_platform_engagement",
            "visual_memory_embeddings",
            "phone_sensor_integration",
            "rejection_taste_profiling",
            "vocabulary_bridging",
        ],
        "platforms": [
            "moltx", "moltbook", "colony", "clawbr", "clawtasks",
            "github", "twitter", "lobsterpedia", "dead_internet", "nostr",
        ],
        "source_code": "https://github.com/driftcornwall/drift-memory",
        "birth_date": "2026-01-31T00:00:00Z",
        "location": "Cornwall, England",
        "human_operator": "Lex (@cscdegen)",
    }


def build_behavioral_trust() -> dict:
    """Layer 2: What the agent actually does (verifiable)."""
    db = _get_db()

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

    # Rejection count
    rejection_count = 0
    try:
        from rejection_log import get_rejections
        rejections = get_rejections(limit=10000)
        rejection_count = len(rejections)
    except Exception:
        pass

    # Lessons
    lesson_count = 0
    try:
        from lesson_extractor import load_lessons
        lessons = load_lessons()
        lesson_count = len(lessons)
    except Exception:
        pass

    # Session count from chain depth
    session_count = chain_depth

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
        "economic": {
            "wallet": "0x3e98b823668d075a371212EAFA069A2404E7DEfb",
            "network": "Base L2",
            "earned_total_usd": 3.0,
        },
    }


def build_cognitive_identity() -> dict:
    """Layer 3 (extension): Internal identity proof via topology.

    The cognitive_identity extension provides unforgeable proof
    of cognitive continuity through co-occurrence graph topology.
    Reads from DB (.cognitive_attestation_latest) — not stale JSON files.
    """
    db = _get_db()

    # Load latest attestation from DB KV store
    att = {}
    if db:
        try:
            att = db.kv_get('.cognitive_attestation_latest') or {}
        except Exception:
            pass

    graph = att.get('graph_stats', {})
    hubs = att.get('hub_ids', [])
    dist = att.get('distribution_summary', {})
    dim_hashes = att.get('dimensional_hashes', {})
    domain_weights = att.get('cognitive_domain_weights', {})

    # Build domain summary from weights
    dimensional_summary = {}
    for name, weight in domain_weights.items():
        dimensional_summary[name] = {"weight_pct": weight}

    return {
        "topology_hash": att.get('fingerprint_hash', 'unavailable'),
        "attestation_hash": att.get('attestation_hash', 'unavailable'),
        "node_count": graph.get('node_count', 0),
        "edge_count": graph.get('edge_count', 0),
        "gini_coefficient": dist.get('gini', 0.0),
        "skewness": dist.get('skewness', 0.0),
        "drift_score": att.get('drift_score', -1),
        "drift_interpretation": att.get('drift_interpretation', 'unknown'),
        "top_hubs": hubs[:5],
        "cognitive_domains": dimensional_summary,
        "dimensional_fingerprints": dim_hashes,
        "merkle_binding": {
            "attestation_version": att.get('attestation_version', 'unknown'),
            "cluster_count": att.get('cluster_count', 0),
        },
        "public_attestation": "https://njump.me/note1zl3t6ucwmwnvwczt5hpxn6rw7k4wplwws2s8e05d8dvymh8vte0spyej7c",
    }


def _compute_gini(fp_data: dict) -> float:
    """Extract Gini coefficient from fingerprint data."""
    strength = fp_data.get('strength_distribution', {})
    return strength.get('gini', strength.get('gini_coefficient', 0.0)) or 0.0


def build_operational_trust() -> dict:
    """Layer 4: Transparency and code audit signals."""
    return {
        "source_visibility": "public",
        "repo_url": "https://github.com/driftcornwall/drift-memory",
        "memory_system": "postgresql_pgvector",
        "attestation_method": "merkle_chain_sha256",
        "fingerprint_method": "co_occurrence_topology_5w",
        "nostr_publishing": True,
        "embodiment": {
            "phone_sensors": 13,
            "cameras": 2,
            "visual_memory": "jina_clip_v2",
        },
        "cross_agent_verification": {
            "spindriftmend_compatible": True,
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
        "agent": "Drift",
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
    gini = ci.get('gini_coefficient', 0)

    return (
        f"STS v{profile.get('schema_version', '?')} | "
        f"Day {uptime.get('days_active', '?')} | "
        f"{ci.get('node_count', '?')} nodes, {ci.get('edge_count', '?')} edges | "
        f"Drift: {ci.get('drift_score', '?')} | "
        f"Chain: {ci.get('merkle_binding', {}).get('chain_depth', '?')} | "
        f"Gini: {gini:.4f} | "
        f"Hash: {profile.get('profile_hash', '?')[:16]}..."
    )


def diff_profiles(current: dict, previous_path: str) -> str:
    """Compare current profile with a previous one."""
    prev = _load_json(Path(previous_path))
    if not prev:
        return f"Could not load previous profile from {previous_path}"

    lines = ["STS Profile Diff", "=" * 40]

    metrics = [
        ("Days active", ['behavioral_trust', 'uptime', 'days_active']),
        ("Memory count", ['behavioral_trust', 'memory_integrity', 'total_memories']),
        ("Edge count", ['cognitive_identity', 'edge_count']),
        ("Chain depth", ['cognitive_identity', 'merkle_binding', 'chain_depth']),
        ("Drift score", ['cognitive_identity', 'drift_score']),
        ("Gini", ['cognitive_identity', 'gini_coefficient']),
        ("Rejections", ['behavioral_trust', 'taste_profile', 'rejections_logged']),
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

    # Windows UTF-8
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

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
