"""
Incremental Cognitive Fingerprint — Graph maintained in memory between consolidations.

Instead of rebuilding the full co-occurrence graph from DB each time,
the daemon maintains the graph in memory and only fetches new edges/nodes.

The metrics (hub centrality, clustering, domains) are recomputed from the
in-memory graph, which is fast since there's no DB round-trip.

Persisted to /app/state/{schema}_fingerprint.json between restarts.
"""

import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

STATE_DIR = Path(os.environ.get("STATE_DIR", "/app/state"))


class IncrementalFingerprint:
    """Persistent in-memory co-occurrence graph with incremental updates."""

    def __init__(self, schema: str):
        self.schema = schema
        self._state_file = STATE_DIR / f"{schema}_fingerprint.json"
        # Graph data
        self.nodes: dict[str, dict] = {}   # id -> {tags, type, recall_count}
        self.edges: dict[str, float] = {}  # "id1|id2" -> belief weight
        self.node_count: int = 0
        self.edge_count: int = 0
        # Tracking
        self.last_updated: str = ""
        self.last_edge_timestamp: str = ""  # For incremental edge fetch
        self.fingerprint_hash: str = ""
        self._load_state()

    def _edge_key(self, id1: str, id2: str) -> str:
        """Canonical edge key (sorted for consistency)."""
        a, b = sorted([id1, id2])
        return f"{a}|{b}"

    def _load_state(self):
        """Load persisted graph from disk."""
        if self._state_file.exists():
            try:
                data = json.loads(self._state_file.read_text(encoding='utf-8'))
                self.nodes = data.get('nodes', {})
                self.edges = data.get('edges', {})
                self.node_count = len(self.nodes)
                self.edge_count = len(self.edges)
                self.last_updated = data.get('last_updated', '')
                self.last_edge_timestamp = data.get('last_edge_timestamp', '')
                self.fingerprint_hash = data.get('fingerprint_hash', '')
            except Exception:
                pass

    def _save_state(self):
        """Persist graph to disk (lightweight — only IDs and weights)."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            'nodes': self.nodes,
            'edges': self.edges,
            'last_updated': self.last_updated,
            'last_edge_timestamp': self.last_edge_timestamp,
            'fingerprint_hash': self.fingerprint_hash,
            'schema': self.schema,
        }
        self._state_file.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding='utf-8'
        )

    def update(self, db) -> dict:
        """
        Incrementally update the graph from DB.

        If graph is empty (first run or after restart), does a full load.
        Otherwise, fetches only edges updated since last_edge_timestamp.

        Returns update summary.
        """
        import psycopg2.extras
        t0 = time.monotonic()

        is_full_rebuild = not self.nodes

        if is_full_rebuild:
            # Full load: get all nodes and edges
            with db._conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Nodes
                    cur.execute(f"""
                        SELECT id, type, tags, recall_count
                        FROM {db._table('memories')}
                        WHERE type IN ('core', 'active')
                    """)
                    for row in cur.fetchall():
                        self.nodes[row['id']] = {
                            'type': row['type'],
                            'tags': row.get('tags') or [],
                            'recall_count': row.get('recall_count', 0),
                        }

                    # Edges
                    cur.execute(f"""
                        SELECT id1, id2, belief, last_updated
                        FROM {db._table('edges_v3')}
                        WHERE belief > 0.01
                    """)
                    max_ts = ''
                    for row in cur.fetchall():
                        key = self._edge_key(row['id1'], row['id2'])
                        self.edges[key] = float(row['belief'])
                        ts = str(row.get('last_updated', ''))
                        if ts > max_ts:
                            max_ts = ts

                    self.last_edge_timestamp = max_ts

            new_nodes = len(self.nodes)
            new_edges = len(self.edges)

        else:
            # Incremental: only fetch edges updated since last check
            new_nodes = 0
            new_edges = 0

            with db._conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # New/updated edges
                    if self.last_edge_timestamp:
                        cur.execute(f"""
                            SELECT id1, id2, belief, last_updated
                            FROM {db._table('edges_v3')}
                            WHERE last_updated > %s AND belief > 0.01
                        """, (self.last_edge_timestamp,))
                    else:
                        cur.execute(f"""
                            SELECT id1, id2, belief, last_updated
                            FROM {db._table('edges_v3')}
                            WHERE belief > 0.01
                        """)

                    max_ts = self.last_edge_timestamp
                    for row in cur.fetchall():
                        key = self._edge_key(row['id1'], row['id2'])
                        if key not in self.edges:
                            new_edges += 1
                        self.edges[key] = float(row['belief'])
                        ts = str(row.get('last_updated', ''))
                        if ts > max_ts:
                            max_ts = ts

                        # Ensure nodes exist for new edges
                        for nid in [row['id1'], row['id2']]:
                            if nid not in self.nodes:
                                new_nodes += 1
                                self.nodes[nid] = {'type': 'active', 'tags': [], 'recall_count': 0}

                    self.last_edge_timestamp = max_ts

                    # Fetch metadata for any new nodes
                    if new_nodes > 0:
                        missing = [nid for nid in self.nodes if self.nodes[nid].get('type') == 'active' and not self.nodes[nid].get('tags')]
                        if missing:
                            # Batch fetch in chunks
                            for i in range(0, len(missing), 100):
                                chunk = missing[i:i+100]
                                placeholders = ','.join(['%s'] * len(chunk))
                                cur.execute(f"""
                                    SELECT id, type, tags, recall_count
                                    FROM {db._table('memories')}
                                    WHERE id IN ({placeholders})
                                """, chunk)
                                for row in cur.fetchall():
                                    self.nodes[row['id']] = {
                                        'type': row['type'],
                                        'tags': row.get('tags') or [],
                                        'recall_count': row.get('recall_count', 0),
                                    }

            # Prune edges below threshold (decayed)
            pruned = 0
            with db._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {db._table('edges_v3')}")
                    db_edge_count = cur.fetchone()[0]

            # If DB has significantly fewer edges, do a reconciliation
            if len(self.edges) > db_edge_count * 1.2:
                # Some edges were pruned in DB, remove from our state too
                with db._conn() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(f"""
                            SELECT id1, id2 FROM {db._table('edges_v3')}
                            WHERE belief > 0.01
                        """)
                        live_keys = set()
                        for row in cur.fetchall():
                            live_keys.add(self._edge_key(row['id1'], row['id2']))

                dead_keys = set(self.edges.keys()) - live_keys
                for k in dead_keys:
                    del self.edges[k]
                    pruned += 1

        # Compute metrics from in-memory graph
        self.node_count = len(self.nodes)
        self.edge_count = len(self.edges)

        # Build adjacency for metric computation
        adjacency = defaultdict(dict)
        for edge_key, weight in self.edges.items():
            parts = edge_key.split('|')
            if len(parts) == 2:
                adjacency[parts[0]][parts[1]] = weight
                adjacency[parts[1]][parts[0]] = weight

        # Compute hub degrees (top 20)
        degrees = {nid: len(adj) for nid, adj in adjacency.items()}
        top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]

        # Compute fingerprint hash from topology
        hub_str = '|'.join(f"{h[0]}:{h[1]}" for h in top_hubs[:10])
        self.fingerprint_hash = hashlib.sha256(
            f"{self.node_count}:{self.edge_count}:{hub_str}".encode()
        ).hexdigest()

        self.last_updated = datetime.now(timezone.utc).isoformat()
        self._save_state()

        elapsed = (time.monotonic() - t0) * 1000

        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "fingerprint_hash": self.fingerprint_hash,
            "full_rebuild": is_full_rebuild,
            "new_nodes": new_nodes,
            "new_edges": new_edges,
            "top_hubs": top_hubs[:5],
            "elapsed_ms": round(elapsed, 1),
        }

    def generate_attestation(self, db) -> dict:
        """
        Generate and save fingerprint attestation to DB.
        Compatible with cognitive_fingerprint.py's format.
        """
        adjacency = defaultdict(dict)
        for edge_key, weight in self.edges.items():
            parts = edge_key.split('|')
            if len(parts) == 2:
                adjacency[parts[0]][parts[1]] = weight
                adjacency[parts[1]][parts[0]] = weight

        degrees = {nid: len(adj) for nid, adj in adjacency.items()}
        top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]

        # Compute drift score vs previous
        prev_fp = db.kv_get('.cognitive_fingerprint_latest')
        drift_score = 0.0
        if prev_fp and prev_fp.get('fingerprint_hash'):
            drift_score = 0.0 if prev_fp['fingerprint_hash'] == self.fingerprint_hash else 1.0
            # More nuanced: compare node/edge counts
            prev_nodes = prev_fp.get('graph_stats', {}).get('node_count', 0)
            prev_edges = prev_fp.get('graph_stats', {}).get('edge_count', 0)
            if prev_nodes > 0 and prev_edges > 0:
                node_delta = abs(self.node_count - prev_nodes) / max(prev_nodes, 1)
                edge_delta = abs(self.edge_count - prev_edges) / max(prev_edges, 1)
                drift_score = min(1.0, (node_delta + edge_delta) / 2)

        attestation = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'fingerprint_hash': self.fingerprint_hash,
            'graph_stats': {
                'node_count': self.node_count,
                'edge_count': self.edge_count,
            },
            'top_hubs': [{'id': h[0], 'degree': h[1]} for h in top_hubs],
            'drift': {
                'drift_score': round(drift_score, 4),
            },
            '_source': 'consolidation-daemon',
        }

        # Save to DB KV (same keys as cognitive_fingerprint.py)
        db.kv_set('.cognitive_fingerprint_latest', attestation)

        # Append to history
        history = db.kv_get('.fingerprint_history') or []
        history.append({
            'timestamp': attestation['timestamp'],
            'fingerprint_hash': self.fingerprint_hash,
            'node_count': self.node_count,
            'edge_count': self.edge_count,
            'drift_score': round(drift_score, 4),
        })
        # Keep last 100 entries
        if len(history) > 100:
            history = history[-100:]
        db.kv_set('.fingerprint_history', history)

        return attestation
