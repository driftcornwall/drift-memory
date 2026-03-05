"""Tests for co-occurrence resonance detection (Tesla2 Proposal #3)."""

import pytest
from unittest.mock import MagicMock, patch


def _make_adj(edges):
    """Build adjacency dict from edge list [(a,b,belief), ...]"""
    adj = {}
    beliefs = {}
    for a, b, belief in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
        key = tuple(sorted([a, b]))
        beliefs[key] = belief
    return adj, beliefs


class TestFindTriangles:
    def test_single_triangle(self):
        from cooccurrence_resonance import find_triangles
        adj, beliefs = _make_adj([('A', 'B', 5.0), ('B', 'C', 3.0), ('A', 'C', 4.0)])
        triangles = find_triangles(adj, beliefs)
        assert len(triangles) == 1
        assert set(triangles[0]['nodes']) == {'A', 'B', 'C'}

    def test_no_triangle(self):
        from cooccurrence_resonance import find_triangles
        adj, beliefs = _make_adj([('A', 'B', 5.0), ('C', 'D', 3.0)])
        triangles = find_triangles(adj, beliefs)
        assert len(triangles) == 0

    def test_strength_is_geometric_mean(self):
        from cooccurrence_resonance import find_triangles
        adj, beliefs = _make_adj([('A', 'B', 8.0), ('B', 'C', 2.0), ('A', 'C', 4.0)])
        triangles = find_triangles(adj, beliefs)
        assert len(triangles) == 1
        # geometric mean of 8, 2, 4 = (8*2*4)^(1/3) = 64^(1/3) = 4.0
        assert abs(triangles[0]['strength'] - 4.0) < 0.01

    def test_two_triangles_sorted_by_strength(self):
        from cooccurrence_resonance import find_triangles
        adj, beliefs = _make_adj([
            ('A', 'B', 10.0), ('B', 'C', 10.0), ('A', 'C', 10.0),  # strong
            ('D', 'E', 2.0), ('E', 'F', 2.0), ('D', 'F', 2.0),     # weak
        ])
        triangles = find_triangles(adj, beliefs)
        assert len(triangles) == 2
        assert triangles[0]['strength'] > triangles[1]['strength']

    def test_min_belief_filters_weak_edges(self):
        from cooccurrence_resonance import find_triangles
        adj, beliefs = _make_adj([('A', 'B', 0.5), ('B', 'C', 0.5), ('A', 'C', 0.5)])
        triangles = find_triangles(adj, beliefs, min_belief=1.0)
        assert len(triangles) == 0

    def test_max_triangles_limit(self):
        from cooccurrence_resonance import find_triangles
        # Build a clique of 5 nodes = 10 triangles
        nodes = ['A', 'B', 'C', 'D', 'E']
        edges = []
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                edges.append((a, b, 5.0))
        adj, beliefs = _make_adj(edges)
        triangles = find_triangles(adj, beliefs, max_triangles=3)
        assert len(triangles) == 3


class TestCyclingFrequency:
    def test_counts_distinct_sessions(self):
        from cooccurrence_resonance import compute_cycling_frequency
        obs = {
            ('A', 'B'): ['s1', 's2', 's3'],
            ('B', 'C'): ['s1', 's3'],
            ('A', 'C'): ['s2', 's3'],
        }
        triangle = {'nodes': ['A', 'B', 'C']}
        freq = compute_cycling_frequency(triangle, obs)
        # s1: AB+BC=2, s2: AB+AC=2, s3: AB+BC+AC=3
        assert freq == 3

    def test_single_session_no_overlap(self):
        from cooccurrence_resonance import compute_cycling_frequency
        obs = {
            ('A', 'B'): ['s1'],
            ('B', 'C'): ['s2'],
            ('A', 'C'): ['s3'],
        }
        triangle = {'nodes': ['A', 'B', 'C']}
        freq = compute_cycling_frequency(triangle, obs)
        assert freq == 0

    def test_empty_observations(self):
        from cooccurrence_resonance import compute_cycling_frequency
        triangle = {'nodes': ['A', 'B', 'C']}
        freq = compute_cycling_frequency(triangle, {})
        assert freq == 0


class TestCycleCoupling:
    def test_two_coupled_triangles(self):
        from cooccurrence_resonance import detect_coupling
        triangles = [
            {'nodes': ['A', 'B', 'C'], 'strength': 5.0},
            {'nodes': ['B', 'C', 'D'], 'strength': 3.0},
        ]
        coupling = detect_coupling(triangles)
        assert len(coupling) == 1
        assert set(coupling[0]['shared_nodes']) == {'B', 'C'}
        assert coupling[0]['cycle_indices'] == (0, 1)

    def test_uncoupled_triangles(self):
        from cooccurrence_resonance import detect_coupling
        triangles = [
            {'nodes': ['A', 'B', 'C'], 'strength': 5.0},
            {'nodes': ['D', 'E', 'F'], 'strength': 3.0},
        ]
        coupling = detect_coupling(triangles)
        assert len(coupling) == 0

    def test_bridge_node(self):
        from cooccurrence_resonance import detect_coupling
        triangles = [
            {'nodes': ['A', 'B', 'C'], 'strength': 5.0},
            {'nodes': ['C', 'D', 'E'], 'strength': 3.0},
        ]
        coupling = detect_coupling(triangles)
        assert len(coupling) == 1
        assert coupling[0]['shared_nodes'] == ['C']

    def test_no_triangles(self):
        from cooccurrence_resonance import detect_coupling
        assert detect_coupling([]) == []


class TestScan:
    """Integration test with mocked DB."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.schema = 'drift'
        db._table = lambda t: f'drift.{t}'
        db.get_all_edges.return_value = {
            'A|B': {'belief': 5.0, 'platform_context': {}, 'activity_context': {},
                    'topic_context': {}, 'contact_context': [], 'first_formed': '', 'last_updated': ''},
            'B|C': {'belief': 3.0, 'platform_context': {}, 'activity_context': {},
                    'topic_context': {}, 'contact_context': [], 'first_formed': '', 'last_updated': ''},
            'A|C': {'belief': 4.0, 'platform_context': {}, 'activity_context': {},
                    'topic_context': {}, 'contact_context': [], 'first_formed': '', 'last_updated': ''},
        }
        db.kv_get.return_value = None
        return db

    def test_scan_returns_expected_structure(self, mock_db):
        from cooccurrence_resonance import run_scan
        with patch('cooccurrence_resonance.get_db', return_value=mock_db), \
             patch('cooccurrence_resonance._load_edge_sessions', return_value={}):
            result = run_scan(verbose=False)
        assert 'triangles' in result
        assert 'coupling' in result
        assert 'scanned_at' in result
        assert 'stats' in result
        assert len(result['triangles']) == 1

    def test_scan_stores_to_kv(self, mock_db):
        from cooccurrence_resonance import run_scan
        with patch('cooccurrence_resonance.get_db', return_value=mock_db), \
             patch('cooccurrence_resonance._load_edge_sessions', return_value={}):
            run_scan(verbose=False)
        mock_db.kv_set.assert_called()

    def test_scan_with_cycling_frequency(self, mock_db):
        from cooccurrence_resonance import run_scan
        edge_sessions = {
            ('A', 'B'): ['s1', 's2'],
            ('A', 'C'): ['s1', 's2'],
            ('B', 'C'): ['s1'],
        }
        with patch('cooccurrence_resonance.get_db', return_value=mock_db), \
             patch('cooccurrence_resonance._load_edge_sessions', return_value=edge_sessions):
            result = run_scan(verbose=False)
        tri = result['triangles'][0]
        assert tri['cycling_freq'] > 0
        assert tri['composite_score'] > 0


class TestLiveIntegration:
    """Run against real DB -- skip if unavailable."""

    @pytest.fixture
    def live_db(self):
        try:
            from db_adapter import get_db
            db = get_db()
            stats = db.edge_stats()
            if stats.get('total_edges', 0) == 0:
                pytest.skip("No edges in DB")
            return db
        except Exception as e:
            pytest.skip(f"DB unavailable: {e}")

    def test_live_scan(self, live_db):
        from cooccurrence_resonance import run_scan
        result = run_scan(verbose=False)
        assert result['stats']['nodes'] > 0
        assert result['stats']['edges'] > 0
        assert result['stats']['triangles_found'] > 0

    def test_live_summary(self, live_db):
        from cooccurrence_resonance import get_resonance_summary
        summary = get_resonance_summary()
        assert summary.get('status') != 'no_scan'
        assert 'triangles' in summary
