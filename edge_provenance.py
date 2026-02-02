#!/usr/bin/env python3
"""
Edge Provenance System v3.0
Based on BrutusBot's security recommendations for multi-agent memory sharing.

Core concepts:
- OBSERVATIONS: Immutable records of when/why two memories were seen together
- BELIEFS: Aggregated confidence scores computed from observations
- TRUST TIERS: Weight observations differently based on source trustworthiness

This module can be used standalone or integrated with memory_manager.py.
"""

import json
import uuid
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# Trust tiers for multi-agent memory sharing
TRUST_TIERS = {
    'self': 1.0,           # Direct observation by this agent
    'verified_agent': 0.8,  # Observation from known, verified agent
    'platform': 0.6,        # Data from platform (Moltbook, MoltX, etc.)
    'unknown': 0.3          # Unverified source
}

# Configuration
OBSERVATION_MAX_AGE_DAYS = 30  # Observations older than this contribute less
RATE_LIMIT_NEW_SOURCES = 3     # Diminishing returns after this many from same source
BELIEF_DECAY_HALF_LIFE_DAYS = 14  # Time for observation weight to halve


def create_observation(
    source_type: str,
    weight: float = 1.0,
    trust_tier: str = 'self',
    session_id: Optional[str] = None,
    agent: str = 'unknown',
    platform: Optional[str] = None,
    artifact_id: Optional[str] = None
) -> dict:
    """
    Create a new observation record.

    Args:
        source_type: Type of source (session_recall, api_mention, manual_link, etc.)
        weight: Base weight of this observation (default 1.0)
        trust_tier: Trust level (self, verified_agent, platform, unknown)
        session_id: Optional session identifier
        agent: Agent name making the observation
        platform: Platform where observation occurred (moltbook, github, etc.)
        artifact_id: Optional reference to external artifact (post ID, commit hash, etc.)

    Returns:
        Observation dict with all metadata
    """
    return {
        'id': str(uuid.uuid4()),
        'observed_at': datetime.now(timezone.utc).isoformat(),
        'source': {
            'type': source_type,
            'session_id': session_id,
            'agent': agent,
            'platform': platform,
            'artifact_id': artifact_id
        },
        'weight': weight,
        'trust_tier': trust_tier
    }


def aggregate_belief(observations: list[dict]) -> float:
    """
    Compute belief score from observations with time decay and trust weighting.

    The algorithm:
    1. Apply time decay (older observations contribute less)
    2. Apply trust tier weighting
    3. Rate limit per source (diminishing returns from same source)
    4. Sum weighted observations

    Args:
        observations: List of observation dicts

    Returns:
        Aggregated belief score
    """
    if not observations:
        return 0.0

    now = datetime.now(timezone.utc)
    source_counts: dict[str, int] = {}
    total_belief = 0.0

    for obs in observations:
        # Parse observation time
        try:
            obs_time = datetime.fromisoformat(obs['observed_at'].replace('Z', '+00:00'))
        except (KeyError, ValueError):
            obs_time = now

        # Time decay: exponential decay based on age
        age_days = (now - obs_time).total_seconds() / 86400
        time_factor = math.exp(-age_days * math.log(2) / BELIEF_DECAY_HALF_LIFE_DAYS)

        # Trust tier weight
        trust_tier = obs.get('trust_tier', 'unknown')
        trust_weight = TRUST_TIERS.get(trust_tier, TRUST_TIERS['unknown'])

        # Rate limit per source (sqrt scaling after threshold)
        source_key = f"{obs.get('source', {}).get('type', 'unknown')}:{obs.get('source', {}).get('agent', 'unknown')}"
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        count = source_counts[source_key]

        if count <= RATE_LIMIT_NEW_SOURCES:
            rate_factor = 1.0
        else:
            # Diminishing returns: sqrt scaling after threshold
            rate_factor = math.sqrt(RATE_LIMIT_NEW_SOURCES / count)

        # Compute weighted contribution
        base_weight = obs.get('weight', 1.0)
        contribution = base_weight * time_factor * trust_weight * rate_factor
        total_belief += contribution

    return round(total_belief, 3)


class EdgeStore:
    """
    Manages edges (memory pair relationships) with provenance tracking.

    File format (.edges_v3.json):
    {
        "version": "3.0",
        "edges": {
            "mem1|mem2": {
                "observations": [...],
                "belief": 2.5,
                "status": "active",
                "last_updated": "2026-02-02T00:00:00Z"
            }
        }
    }
    """

    LINK_THRESHOLD = 3.0  # Belief score needed to form automatic link

    def __init__(self, storage_path: Path):
        """Initialize edge store with storage path."""
        self.storage_path = storage_path
        self.edges: dict = {}
        self._load()

    def _load(self) -> None:
        """Load edges from storage."""
        if not self.storage_path.exists():
            self.edges = {}
            return

        try:
            data = json.loads(self.storage_path.read_text(encoding='utf-8'))
            if data.get('version') == '3.0':
                self.edges = data.get('edges', {})
            else:
                # Legacy format - needs migration
                self.edges = {}
        except (json.JSONDecodeError, KeyError):
            self.edges = {}

    def _save(self) -> None:
        """Save edges to storage."""
        data = {
            'version': '3.0',
            'edges': self.edges
        }
        self.storage_path.write_text(json.dumps(data, indent=2), encoding='utf-8')

    @staticmethod
    def _make_key(id1: str, id2: str) -> str:
        """Create canonical key for memory pair (sorted for consistency)."""
        return '|'.join(sorted([id1, id2]))

    def get_edge(self, id1: str, id2: str) -> Optional[dict]:
        """Get edge between two memories, or None if not exists."""
        key = self._make_key(id1, id2)
        return self.edges.get(key)

    def add_observation(
        self,
        id1: str,
        id2: str,
        source_type: str = 'session_recall',
        weight: float = 1.0,
        trust_tier: str = 'self',
        session_id: Optional[str] = None,
        agent: str = 'unknown',
        platform: Optional[str] = None,
        artifact_id: Optional[str] = None
    ) -> dict:
        """
        Add observation to edge and recompute belief.

        Creates edge if it doesn't exist.
        Returns the updated edge.
        """
        key = self._make_key(id1, id2)

        # Get or create edge
        if key not in self.edges:
            self.edges[key] = {
                'observations': [],
                'belief': 0.0,
                'status': 'pending',
                'last_updated': datetime.now(timezone.utc).isoformat()
            }

        edge = self.edges[key]

        # Create and add observation
        obs = create_observation(
            source_type=source_type,
            weight=weight,
            trust_tier=trust_tier,
            session_id=session_id,
            agent=agent,
            platform=platform,
            artifact_id=artifact_id
        )
        edge['observations'].append(obs)

        # Recompute belief
        edge['belief'] = aggregate_belief(edge['observations'])
        edge['last_updated'] = datetime.now(timezone.utc).isoformat()

        # Update status based on belief threshold
        if edge['belief'] >= self.LINK_THRESHOLD:
            edge['status'] = 'active'

        self._save()
        return edge

    def decay_beliefs(self) -> tuple[int, int]:
        """
        Decay all beliefs and prune empty edges.

        Returns:
            (decayed_count, pruned_count)
        """
        decayed = 0
        pruned = 0
        to_remove = []

        for key, edge in self.edges.items():
            old_belief = edge['belief']
            edge['belief'] = aggregate_belief(edge['observations'])

            if edge['belief'] != old_belief:
                decayed += 1

            # Prune edges with no belief and no recent observations
            if edge['belief'] < 0.1 and not edge['observations']:
                to_remove.append(key)

        for key in to_remove:
            del self.edges[key]
            pruned += 1

        if decayed or pruned:
            self._save()

        return decayed, pruned

    def get_linked_memories(self, memory_id: str) -> list[str]:
        """Get IDs of memories linked to the given memory (belief >= threshold)."""
        linked = []
        for key, edge in self.edges.items():
            if edge['status'] == 'active' and edge['belief'] >= self.LINK_THRESHOLD:
                ids = key.split('|')
                if memory_id in ids:
                    other_id = ids[0] if ids[1] == memory_id else ids[1]
                    linked.append(other_id)
        return linked

    def list_edges(self) -> list[dict]:
        """List all edges with summary info."""
        result = []
        for key, edge in self.edges.items():
            ids = key.split('|')
            result.append({
                'id1': ids[0],
                'id2': ids[1],
                'belief': edge['belief'],
                'status': edge['status'],
                'observation_count': len(edge['observations']),
                'last_updated': edge['last_updated']
            })
        return sorted(result, key=lambda x: -x['belief'])

    def prune_old_observations(self, max_age_days: int = OBSERVATION_MAX_AGE_DAYS) -> int:
        """
        Remove observations older than max_age_days.

        Returns count of removed observations.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=max_age_days)
        removed = 0

        for edge in self.edges.values():
            original_count = len(edge['observations'])
            edge['observations'] = [
                obs for obs in edge['observations']
                if datetime.fromisoformat(obs['observed_at'].replace('Z', '+00:00')) > cutoff
            ]
            removed += original_count - len(edge['observations'])

            # Recompute belief after pruning
            edge['belief'] = aggregate_belief(edge['observations'])
            if edge['belief'] < self.LINK_THRESHOLD:
                edge['status'] = 'pending'

        if removed:
            self._save()

        return removed


def migrate_from_v2(cooccurrence_yaml_path: Path, edges_v3_path: Path, agent: str = 'unknown') -> int:
    """
    Migrate from v2.x .cooccurrence.yaml to v3.0 edges format.

    Args:
        cooccurrence_yaml_path: Path to existing .cooccurrence.yaml
        edges_v3_path: Path for new .edges_v3.json
        agent: Agent name for attribution

    Returns:
        Number of edges migrated
    """
    import yaml

    if not cooccurrence_yaml_path.exists():
        return 0

    # Load legacy data
    data = yaml.safe_load(cooccurrence_yaml_path.read_text(encoding='utf-8'))
    if not data:
        return 0

    pairs = data.get('pairs', {})

    # Create new edge store
    store = EdgeStore(edges_v3_path)

    migrated = 0
    for pair_key, pair_data in pairs.items():
        ids = pair_key.split('|')
        if len(ids) != 2:
            continue

        count = pair_data.get('count', 0)
        if count <= 0:
            continue

        # Create observation for the legacy count
        # We create a single observation with weight = count
        store.add_observation(
            id1=ids[0],
            id2=ids[1],
            source_type='legacy_migration',
            weight=float(count),
            trust_tier='self',
            agent=agent
        )
        migrated += 1

    return migrated


# CLI interface
if __name__ == '__main__':
    import sys

    # Default paths - adjust as needed
    MEMORY_ROOT = Path(__file__).parent
    EDGES_FILE = MEMORY_ROOT / '.edges_v3.json'
    LEGACY_FILE = MEMORY_ROOT / '.cooccurrence.yaml'

    if len(sys.argv) < 2:
        print("Edge Provenance System v3.0")
        print()
        print("Commands:")
        print("  edges              - List all edges")
        print("  edge <id1> <id2>   - Show specific edge details")
        print("  add <id1> <id2>    - Add observation to edge")
        print("  decay              - Decay beliefs and prune")
        print("  migrate            - Migrate from v2 .cooccurrence.yaml")
        print("  prune              - Remove old observations")
        sys.exit(0)

    cmd = sys.argv[1]
    store = EdgeStore(EDGES_FILE)

    if cmd == 'edges':
        edges = store.list_edges()
        print(f"Total edges: {len(edges)}")
        print()
        for e in edges[:20]:
            status_icon = '✓' if e['status'] == 'active' else '○'
            print(f"{status_icon} {e['id1']} <-> {e['id2']}")
            print(f"  Belief: {e['belief']:.3f} | Observations: {e['observation_count']}")

    elif cmd == 'edge' and len(sys.argv) >= 4:
        id1, id2 = sys.argv[2], sys.argv[3]
        edge = store.get_edge(id1, id2)
        if not edge:
            print(f"No edge between {id1} and {id2}")
        else:
            print(f"Edge: {id1} <-> {id2}")
            print(f"Belief: {edge['belief']:.3f}")
            print(f"Status: {edge['status']}")
            print(f"Last Updated: {edge['last_updated']}")
            print(f"Observations ({len(edge['observations'])}):")
            for obs in edge['observations'][-5:]:
                src = obs.get('source', {})
                print(f"  [{obs['id'][:8]}] {obs['observed_at']}")
                print(f"    Source: {src.get('type', '?')} | Agent: {src.get('agent', '?')}")
                print(f"    Weight: {obs['weight']:.2f} | Trust: {obs['trust_tier']}")

    elif cmd == 'add' and len(sys.argv) >= 4:
        id1, id2 = sys.argv[2], sys.argv[3]
        edge = store.add_observation(id1, id2, agent='cli')
        print(f"Added observation. Belief now: {edge['belief']:.3f}")

    elif cmd == 'decay':
        decayed, pruned = store.decay_beliefs()
        print(f"Decayed: {decayed} edges | Pruned: {pruned} edges")

    elif cmd == 'migrate':
        count = migrate_from_v2(LEGACY_FILE, EDGES_FILE, agent='migration')
        print(f"Migrated {count} edges from v2 format")

    elif cmd == 'prune':
        removed = store.prune_old_observations()
        print(f"Removed {removed} old observations")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
