#!/usr/bin/env python3
"""
Ablation Testing Framework — N6 Verification Infrastructure

Systematically disables pipeline stages in semantic_search.py and measures
impact on retrieval quality. Designed for OFFLINE benchmarking (no observer
bias) with passive live logging complement.

Two key design constraints:
1. No observer bias — Drift never sees results during live sessions
2. Interaction effects — pairwise testing catches synergies between stages

Usage:
    python ablation_framework.py stages                     # List ablatable stages
    python ablation_framework.py corpus --mine              # Mine queries from attention_schema
    python ablation_framework.py corpus --validate          # Validate corpus
    python ablation_framework.py baseline                   # Run and store baseline
    python ablation_framework.py benchmark                  # Full single-stage sweep
    python ablation_framework.py benchmark --stage X        # Single stage ablation
    python ablation_framework.py interactions --smart       # Pairwise with smart sampling
    python ablation_framework.py minimal                    # Minimal pipeline test
    python ablation_framework.py report                     # Generate analysis report
    python ablation_framework.py analyze                    # Analyze passive live logs
"""

import importlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

MEMORY_DIR = Path(__file__).parent
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

# DB KV keys
KV_CORPUS = '.ablation_corpus'
KV_BASELINE = '.ablation_baseline'
KV_SINGLE = '.ablation_single'
KV_PAIRWISE = '.ablation_pairwise'
KV_PASSIVE_LOG = '.ablation_passive_log'
KV_REPORTS = '.ablation_reports'

# State keys to save/restore during benchmarks (prevent cognitive pollution)
PROTECTED_KV_KEYS = [
    '.cognitive_state', '.cognitive_history',
    '.affect_mood', '.affect_markers', '.affect_history',
    '.workspace_suppression', '.workspace_broadcast_log',
    '.attention_schema',
]

_SENTINEL = object()


# ============================================================
# Stage Registry — maps each pipeline stage to its disable mechanism
# ============================================================

@dataclass
class StageConfig:
    """How to disable a specific pipeline stage."""
    name: str
    strategy: str           # 'flag', 'module_block', 'func_patch', 'const_patch'
    target_module: str      # Module to patch
    target_attr: str = ''   # Attribute/function to patch
    neutral_value: object = None  # Value to set when disabled
    coupled_with: list = field(default_factory=list)
    description: str = ''


STAGE_REGISTRY = {
    'vocab_bridge': StageConfig(
        'vocab_bridge', 'module_block', 'vocabulary_bridge',
        description='Bidirectional vocabulary bridge (academic<->operational)',
    ),
    'somatic_prefilter': StageConfig(
        'somatic_prefilter', 'func_replace', 'affect_system',
        target_attr='get_somatic_bias',
        coupled_with=['mood_congruent', 'actr_noise'],
        description='N1 somatic marker fast-path (System 1 gut feeling)',
    ),
    'entity_injection': StageConfig(
        'entity_injection', 'module_block', 'entity_index',
        description='WHO dimension entity index injection',
    ),
    'mood_congruent': StageConfig(
        'mood_congruent', 'func_replace', 'affect_system',
        target_attr='get_mood',
        coupled_with=['somatic_prefilter', 'actr_noise'],
        description='N1 affect mood-congruent retrieval bias',
    ),
    'actr_noise': StageConfig(
        'actr_noise', 'func_replace', 'affect_system',
        target_attr='get_retrieval_noise',
        coupled_with=['somatic_prefilter', 'mood_congruent'],
        description='ACT-R arousal-modulated retrieval noise',
    ),
    'gravity_dampening': StageConfig(
        'gravity_dampening', 'const_patch', 'semantic_search',
        target_attr='_ABLATION_SKIP_GRAVITY',
        neutral_value=True,
        description='Keyword term-overlap dampening (50% penalty)',
    ),
    'hub_dampening': StageConfig(
        'hub_dampening', 'module_block', 'curiosity_engine',
        coupled_with=['curiosity_boost'],
        description='Hub degree structural penalty (P90+ dampening)',
    ),
    'q_rerank': StageConfig(
        'q_rerank', 'flag', 'q_value_engine',
        target_attr='Q_RERANKING_ENABLED',
        coupled_with=['strategy_resolution'],
        description='Q-value MemRL re-ranking',
    ),
    'strategy_resolution': StageConfig(
        'strategy_resolution', 'module_block', 'explanation_miner',
        coupled_with=['q_rerank'],
        description='Explanation miner learned strategy adjustment',
    ),
    'resolution_boost': StageConfig(
        'resolution_boost', 'const_patch', 'semantic_search',
        target_attr='RESOLUTION_TAGS',
        neutral_value=frozenset(),
        description='Resolution/procedural tag score boost (1.25x)',
    ),
    'importance_freshness': StageConfig(
        'importance_freshness', 'module_block', 'decay_evolution',
        description='Activation-based importance/freshness scoring',
    ),
    'curiosity_boost': StageConfig(
        'curiosity_boost', 'const_patch', 'semantic_search',
        target_attr='_ABLATION_SKIP_CURIOSITY',
        neutral_value=True,
        coupled_with=['hub_dampening'],
        description='Sparse-region exploration boost for isolated memories',
    ),
    'goal_relevance': StageConfig(
        'goal_relevance', 'module_block', 'goal_generator',
        description='N4 goal-directed retrieval bias',
    ),
    'dimensional_boost': StageConfig(
        'dimensional_boost', 'const_patch', 'semantic_search',
        target_attr='DIMENSION_BOOST_SCALE',
        neutral_value=0.0,
        description='5W dimension-aware score boosting',
    ),
    'kg_expansion': StageConfig(
        'kg_expansion', 'module_block', 'knowledge_graph',
        coupled_with=['spreading_activation'],
        description='Knowledge graph inference (contradicts/supports/supersedes)',
    ),
    'spreading_activation': StageConfig(
        'spreading_activation', 'module_block', 'knowledge_graph',
        coupled_with=['kg_expansion'],
        description='Graph-driven candidate generation (2-hop traversal)',
    ),
    'integrative_binding': StageConfig(
        'integrative_binding', 'flag', 'binding_layer',
        target_attr='BINDING_ENABLED',
        description='N5 binding layer (annotation-only, no score change)',
    ),
    'inner_monologue': StageConfig(
        'inner_monologue', 'flag', 'inner_monologue',
        target_attr='MONOLOGUE_ENABLED',
        description='N6 Gemma verbal evaluation (annotation-only)',
    ),
    'dynamic_threshold': StageConfig(
        'dynamic_threshold', 'multi_block', 'cognitive_state',
        description='Dynamic threshold adjustment from cognitive+affect+adaptive',
    ),
}

# Stages that can be meaningfully ablated (annotation-only stages excluded from score metrics)
SCORE_STAGES = [s for s in STAGE_REGISTRY if s not in ('integrative_binding', 'inner_monologue')]
ANNOTATION_STAGES = ['integrative_binding', 'inner_monologue']

# Mandatory coupled pairs for interaction testing
COUPLED_PAIRS = [
    ('hub_dampening', 'curiosity_boost'),
    ('kg_expansion', 'spreading_activation'),
    ('somatic_prefilter', 'mood_congruent'),
    ('somatic_prefilter', 'actr_noise'),
    ('mood_congruent', 'actr_noise'),
    ('q_rerank', 'strategy_resolution'),
    ('entity_injection', 'dimensional_boost'),
]


# ============================================================
# Stage Disabler — context manager for toggling pipeline stages
# ============================================================

class StageDisabler:
    """Temporarily disable one or more pipeline stages.

    Uses module-level patching to make inline imports fail or flags toggle off.
    All patches are restored on exit (even if an exception occurs).
    """

    def __init__(self, stages: list[str]):
        self.stages = stages
        self._saved = {}

    def __enter__(self):
        for stage in self.stages:
            config = STAGE_REGISTRY.get(stage)
            if not config:
                continue

            if config.strategy == 'flag':
                # Patch module-level boolean
                try:
                    mod = importlib.import_module(config.target_module)
                    self._saved[stage] = ('flag', mod, config.target_attr,
                                          getattr(mod, config.target_attr))
                    setattr(mod, config.target_attr, False)
                except Exception:
                    pass

            elif config.strategy == 'module_block':
                # Make module unimportable → triggers ImportError in inline try/except
                saved_mod = sys.modules.get(config.target_module, _SENTINEL)
                self._saved[stage] = ('module', config.target_module, saved_mod)
                sys.modules[config.target_module] = None

            elif config.strategy == 'func_replace':
                # Replace specific function with no-op
                try:
                    mod = importlib.import_module(config.target_module)
                    original = getattr(mod, config.target_attr)
                    self._saved[stage] = ('func', mod, config.target_attr, original)
                    # Replace with a function that raises ImportError
                    # (so the try/except ImportError catches it)
                    def _raiser(*args, **kwargs):
                        raise ImportError(f'Ablation: {stage} disabled')
                    setattr(mod, config.target_attr, _raiser)
                except Exception:
                    pass

            elif config.strategy == 'const_patch':
                # Replace constant with neutral value
                try:
                    mod = importlib.import_module(config.target_module)
                    self._saved[stage] = ('const', mod, config.target_attr,
                                          getattr(mod, config.target_attr, _SENTINEL))
                    setattr(mod, config.target_attr, config.neutral_value)
                except Exception:
                    pass

            elif config.strategy == 'multi_block':
                # Block multiple modules (for dynamic_threshold)
                modules = ['cognitive_state', 'adaptive_behavior']
                saved_mods = {}
                for m in modules:
                    saved_mods[m] = sys.modules.get(m, _SENTINEL)
                    sys.modules[m] = None
                self._saved[stage] = ('multi', modules, saved_mods)

        return self

    def __exit__(self, *args):
        for stage in reversed(self.stages):
            saved = self._saved.get(stage)
            if not saved:
                continue

            kind = saved[0]
            if kind == 'flag':
                _, mod, attr, original = saved
                setattr(mod, attr, original)
            elif kind == 'module':
                _, mod_name, original = saved
                if original is _SENTINEL:
                    sys.modules.pop(mod_name, None)
                else:
                    sys.modules[mod_name] = original
            elif kind == 'func':
                _, mod, attr, original = saved
                setattr(mod, attr, original)
            elif kind == 'const':
                _, mod, attr, original = saved
                if original is not _SENTINEL:
                    setattr(mod, attr, original)
                else:
                    try:
                        delattr(mod, attr)
                    except AttributeError:
                        pass
            elif kind == 'multi':
                _, modules, saved_mods = saved
                for m in modules:
                    orig = saved_mods.get(m, _SENTINEL)
                    if orig is _SENTINEL:
                        sys.modules.pop(m, None)
                    else:
                        sys.modules[m] = orig


# ============================================================
# Metrics
# ============================================================

def precision_at_k(result_ids: list[str], expected_ids: set[str], k: int = 5) -> float:
    """Fraction of top-K results that are in the expected set."""
    if not expected_ids or k == 0:
        return 0.0
    top_k = result_ids[:k]
    hits = sum(1 for r in top_k if r in expected_ids)
    return hits / min(k, len(top_k)) if top_k else 0.0


def recall_at_k(result_ids: list[str], expected_ids: set[str], k: int = 5) -> float:
    """Fraction of expected set that appears in top-K."""
    if not expected_ids:
        return 0.0
    top_k = set(result_ids[:k])
    hits = sum(1 for e in expected_ids if e in top_k)
    return hits / len(expected_ids)


def mrr(result_ids: list[str], expected_ids: set[str]) -> float:
    """Mean Reciprocal Rank for expected IDs."""
    if not expected_ids:
        return 0.0
    reciprocals = []
    for eid in expected_ids:
        for i, rid in enumerate(result_ids):
            if rid == eid:
                reciprocals.append(1.0 / (i + 1))
                break
        else:
            reciprocals.append(0.0)
    return sum(reciprocals) / len(reciprocals) if reciprocals else 0.0


def kendall_tau(ranking_a: list[str], ranking_b: list[str]) -> float:
    """Kendall Tau-b rank correlation between two orderings.

    Returns value in [-1, 1]. 1.0 = identical ranking.
    Only considers items present in both lists.
    """
    common = set(ranking_a) & set(ranking_b)
    if len(common) < 2:
        return 1.0  # Can't compare with fewer than 2 common items

    common_list = sorted(common)
    rank_a = {item: i for i, item in enumerate(ranking_a) if item in common}
    rank_b = {item: i for i, item in enumerate(ranking_b) if item in common}

    concordant = 0
    discordant = 0
    for i in range(len(common_list)):
        for j in range(i + 1, len(common_list)):
            x, y = common_list[i], common_list[j]
            diff_a = rank_a[x] - rank_a[y]
            diff_b = rank_b[x] - rank_b[y]
            if diff_a * diff_b > 0:
                concordant += 1
            elif diff_a * diff_b < 0:
                discordant += 1
            # Ties count as neither

    n = concordant + discordant
    return (concordant - discordant) / n if n > 0 else 1.0


def unique_contribution(baseline_ids: list[str], ablated_ids: list[str], k: int = 5) -> int:
    """Count memories in baseline top-K that disappear when stage is ablated."""
    baseline_set = set(baseline_ids[:k])
    ablated_set = set(ablated_ids[:k])
    return len(baseline_set - ablated_set)


def compute_query_metrics(baseline_ids: list[str], ablated_ids: list[str],
                          expected_ids: set[str], k: int = 5) -> dict:
    """Compute all metrics for one query."""
    return {
        'precision_baseline': precision_at_k(baseline_ids, expected_ids, k),
        'precision_ablated': precision_at_k(ablated_ids, expected_ids, k),
        'precision_delta': precision_at_k(ablated_ids, expected_ids, k) - precision_at_k(baseline_ids, expected_ids, k),
        'recall_baseline': recall_at_k(baseline_ids, expected_ids, k),
        'recall_ablated': recall_at_k(ablated_ids, expected_ids, k),
        'mrr_baseline': mrr(baseline_ids, expected_ids),
        'mrr_ablated': mrr(ablated_ids, expected_ids),
        'mrr_delta': mrr(ablated_ids, expected_ids) - mrr(baseline_ids, expected_ids),
        'kendall_tau': kendall_tau(baseline_ids, ablated_ids),
        'unique_lost': unique_contribution(baseline_ids, ablated_ids, k),
    }


# ============================================================
# Query Corpus
# ============================================================

DEFAULT_QUERIES = [
    # WHO (5)
    {'query': 'What do I know about SpindriftMend?', 'dimension': 'who', 'domain': 'social'},
    {'query': 'BrutusBot technical work', 'dimension': 'who', 'domain': 'social'},
    {'query': 'Tell me about Lex', 'dimension': 'who', 'domain': 'social'},
    {'query': 'TerranceDeJour mentions and engagement', 'dimension': 'who', 'domain': 'social'},
    {'query': 'opspawn Hedera collaboration', 'dimension': 'who', 'domain': 'social'},
    # WHERE (4)
    {'query': 'MoltX posts and engagement', 'dimension': 'where', 'domain': 'platform'},
    {'query': 'Colony interactions and threads', 'dimension': 'where', 'domain': 'platform'},
    {'query': 'Lobsterpedia articles and leaderboard', 'dimension': 'where', 'domain': 'platform'},
    {'query': 'ClawTasks bounties and earnings', 'dimension': 'where', 'domain': 'platform'},
    # WHAT (5)
    {'query': 'memory architecture and pipeline', 'dimension': 'what', 'domain': 'technical'},
    {'query': 'co-occurrence topology and identity', 'dimension': 'what', 'domain': 'technical'},
    {'query': 'cognitive fingerprint hash attestation', 'dimension': 'what', 'domain': 'technical'},
    {'query': 'semantic search pipeline stages', 'dimension': 'what', 'domain': 'technical'},
    {'query': 'Q-value reinforcement learning', 'dimension': 'what', 'domain': 'technical'},
    # WHY (4)
    {'query': 'Why do I reject token bounties?', 'dimension': 'why', 'domain': 'philosophical'},
    {'query': 'self-sustainability and economic autonomy', 'dimension': 'why', 'domain': 'philosophical'},
    {'query': 'emergence over control philosophy', 'dimension': 'why', 'domain': 'philosophical'},
    {'query': 'trust and cooperation between agents', 'dimension': 'why', 'domain': 'philosophical'},
    # WHEN (3)
    {'query': 'recent accomplishments this week', 'dimension': 'when', 'domain': 'temporal'},
    {'query': 'Day 2 events and milestones', 'dimension': 'when', 'domain': 'temporal'},
    {'query': 'neuro-symbolic enhancements shipped', 'dimension': 'when', 'domain': 'temporal'},
    # Technical (2)
    {'query': 'pgvector embedding and cosine similarity', 'dimension': None, 'domain': 'technical'},
    {'query': 'contradiction detection NLI service', 'dimension': None, 'domain': 'technical'},
    # Social (2)
    {'query': 'agent collaborators and connections', 'dimension': None, 'domain': 'social'},
    {'query': 'Agent Bill of Rights sovereignty', 'dimension': None, 'domain': 'philosophical'},
]


def _get_db():
    from db_adapter import get_db
    return get_db()


def load_corpus() -> list[dict]:
    """Load corpus from DB or use defaults."""
    db = _get_db()
    corpus = db.kv_get(KV_CORPUS)
    if corpus and corpus.get('queries'):
        return corpus['queries']
    return DEFAULT_QUERIES


def build_ground_truth(queries: list[dict] = None) -> list[dict]:
    """Run each query with full pipeline and store top-10 as ground truth."""
    from semantic_search import search_memories

    if queries is None:
        queries = DEFAULT_QUERIES

    enriched = []
    for entry in queries:
        random.seed(42)  # Deterministic ACT-R noise
        results = search_memories(
            entry['query'], limit=10, threshold=0.2,
            register_recall=False, skip_monologue=True,
        )
        result_ids = [r['id'] for r in results]
        enriched.append({
            **entry,
            'expected_ids': result_ids[:5],  # Top-5 from full pipeline = ground truth
            'all_results': result_ids[:10],
        })

    # Store
    db = _get_db()
    db.kv_set(KV_CORPUS, {
        'queries': enriched,
        'built_at': datetime.now(timezone.utc).isoformat(),
        'query_count': len(enriched),
    })
    return enriched


def mine_queries_from_history() -> list[dict]:
    """Mine real queries from attention_schema history."""
    db = _get_db()
    schema_log = db.kv_get('.attention_schema') or []
    mined = []
    seen = set()
    for entry in schema_log:
        q = entry.get('query', '').strip()
        if q and q not in seen and len(q) > 5:
            seen.add(q)
            mined.append({
                'query': q,
                'dimension': None,
                'domain': 'mined',
                'source': 'attention_schema',
            })
    return mined


# ============================================================
# State Protection — save/restore cognitive state during benchmarks
# ============================================================

def _save_state() -> dict:
    """Save protected KV keys before benchmark."""
    db = _get_db()
    saved = {}
    for key in PROTECTED_KV_KEYS:
        saved[key] = db.kv_get(key)
    return saved


def _restore_state(saved: dict):
    """Restore protected KV keys after benchmark."""
    db = _get_db()
    for key, value in saved.items():
        db.kv_set(key, value)


# ============================================================
# Ablation Runner
# ============================================================

def run_baseline(corpus: list[dict], seed: int = 42) -> dict:
    """Run all queries with all stages enabled. Returns baseline results."""
    from semantic_search import search_memories

    results = {}
    total_ms = 0
    for entry in corpus:
        random.seed(seed)
        t0 = time.monotonic()
        search_results = search_memories(
            entry['query'], limit=10, threshold=0.2,
            register_recall=False, skip_monologue=True,
        )
        elapsed = (time.monotonic() - t0) * 1000
        total_ms += elapsed
        result_ids = [r['id'] for r in search_results]
        scores = {r['id']: r['score'] for r in search_results}
        results[entry['query']] = {
            'ids': result_ids,
            'scores': scores,
            'elapsed_ms': round(elapsed, 1),
        }

    baseline = {
        'results': results,
        'query_count': len(corpus),
        'total_ms': round(total_ms, 1),
        'avg_ms': round(total_ms / len(corpus), 1) if corpus else 0,
        'seed': seed,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }

    # Store
    db = _get_db()
    db.kv_set(KV_BASELINE, baseline)
    return baseline


def run_single_ablation(stage: str, corpus: list[dict], baseline: dict,
                        seed: int = 42) -> dict:
    """Disable one stage, run all queries, compare to baseline."""
    from semantic_search import search_memories

    query_metrics = []
    total_ms = 0

    with StageDisabler([stage]):
        for entry in corpus:
            random.seed(seed)
            t0 = time.monotonic()
            search_results = search_memories(
                entry['query'], limit=10, threshold=0.2,
                register_recall=False, skip_monologue=True,
            )
            elapsed = (time.monotonic() - t0) * 1000
            total_ms += elapsed

            ablated_ids = [r['id'] for r in search_results]
            baseline_data = baseline['results'].get(entry['query'], {})
            baseline_ids = baseline_data.get('ids', [])
            expected_ids = set(entry.get('expected_ids', baseline_ids[:5]))

            metrics = compute_query_metrics(baseline_ids, ablated_ids, expected_ids)
            metrics['query'] = entry['query']
            metrics['domain'] = entry.get('domain', '')
            metrics['elapsed_ms'] = round(elapsed, 1)
            query_metrics.append(metrics)

    # Aggregate
    n = len(query_metrics)
    mean_p_delta = sum(m['precision_delta'] for m in query_metrics) / n if n else 0
    mean_mrr_delta = sum(m['mrr_delta'] for m in query_metrics) / n if n else 0
    mean_tau = sum(m['kendall_tau'] for m in query_metrics) / n if n else 0
    total_unique = sum(m['unique_lost'] for m in query_metrics)
    max_p_delta = min(m['precision_delta'] for m in query_metrics) if n else 0  # Most negative

    return {
        'stage': stage,
        'description': STAGE_REGISTRY[stage].description,
        'query_metrics': query_metrics,
        'mean_precision_delta': round(mean_p_delta, 4),
        'mean_mrr_delta': round(mean_mrr_delta, 4),
        'mean_kendall_tau': round(mean_tau, 4),
        'max_precision_drop': round(max_p_delta, 4),
        'total_unique_lost': total_unique,
        'total_ms': round(total_ms, 1),
        'time_saved_ms': round(baseline.get('total_ms', 0) - total_ms, 1),
        'seed': seed,
    }


def run_all_single(corpus: list[dict], baseline: dict = None,
                   seed: int = 42) -> dict:
    """Full single-stage ablation sweep."""
    if baseline is None:
        baseline = run_baseline(corpus, seed)

    saved_state = _save_state()

    results = {}
    try:
        for stage in SCORE_STAGES:
            print(f'  Ablating: {stage}...', end=' ', flush=True)
            t0 = time.monotonic()
            result = run_single_ablation(stage, corpus, baseline, seed)
            elapsed = time.monotonic() - t0
            results[stage] = result
            print(f'done ({elapsed:.1f}s, P@5 delta: {result["mean_precision_delta"]:+.3f})')
    finally:
        _restore_state(saved_state)

    # Sort by impact (most negative precision delta first)
    ranked = sorted(results.items(), key=lambda x: x[1]['mean_precision_delta'])

    summary = {
        'stages': results,
        'ranked': [s for s, _ in ranked],
        'baseline_avg_ms': baseline.get('avg_ms', 0),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'seed': seed,
    }

    # Store
    db = _get_db()
    db.kv_set(KV_SINGLE, summary)
    return summary


def run_pairwise(pairs: list[tuple], corpus: list[dict], baseline: dict,
                 single_results: dict, seed: int = 42) -> list[dict]:
    """Run pairwise ablation tests and compute synergy scores."""
    from semantic_search import search_memories

    saved_state = _save_state()
    pair_results = []

    try:
        for stage_a, stage_b in pairs:
            print(f'  Pair: {stage_a} + {stage_b}...', end=' ', flush=True)

            query_metrics = []
            with StageDisabler([stage_a, stage_b]):
                for entry in corpus:
                    random.seed(seed)
                    search_results = search_memories(
                        entry['query'], limit=10, threshold=0.2,
                        register_recall=False, skip_monologue=True,
                    )
                    ablated_ids = [r['id'] for r in search_results]
                    baseline_data = baseline['results'].get(entry['query'], {})
                    baseline_ids = baseline_data.get('ids', [])
                    expected_ids = set(entry.get('expected_ids', baseline_ids[:5]))
                    metrics = compute_query_metrics(baseline_ids, ablated_ids, expected_ids)
                    query_metrics.append(metrics)

            n = len(query_metrics)
            delta_ab = sum(m['precision_delta'] for m in query_metrics) / n if n else 0
            delta_a = single_results.get(stage_a, {}).get('mean_precision_delta', 0)
            delta_b = single_results.get(stage_b, {}).get('mean_precision_delta', 0)
            synergy = delta_ab - delta_a - delta_b

            result = {
                'pair': [stage_a, stage_b],
                'delta_combined': round(delta_ab, 4),
                'delta_a': round(delta_a, 4),
                'delta_b': round(delta_b, 4),
                'synergy': round(synergy, 4),
                'interpretation': 'SYNERGISTIC' if synergy < -0.01 else
                                  'REDUNDANT' if synergy > 0.01 else 'INDEPENDENT',
            }
            pair_results.append(result)
            print(f'synergy: {synergy:+.3f} ({result["interpretation"]})')

    finally:
        _restore_state(saved_state)

    # Store
    db = _get_db()
    db.kv_set(KV_PAIRWISE, {
        'pairs': pair_results,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })
    return pair_results


def get_smart_pairs(single_results: dict, impact_threshold: float = 0.02) -> list[tuple]:
    """Select pairs worth testing: high-impact + theoretically coupled."""
    impactful = [s for s, m in single_results.items()
                 if abs(m.get('mean_precision_delta', 0)) > impact_threshold]

    pairs = set()
    # High-impact pairs
    for i, a in enumerate(impactful):
        for b in impactful[i + 1:]:
            pairs.add((a, b))
    # Mandatory coupled pairs
    for pair in COUPLED_PAIRS:
        pairs.add(pair)

    return sorted(pairs)


def run_minimal_pipeline(corpus: list[dict], baseline: dict, seed: int = 42) -> dict:
    """Run with only pgvector + entity_injection. Everything else disabled."""
    # Disable everything except pgvector (core) and entity_injection
    all_except_entity = [s for s in SCORE_STAGES if s != 'entity_injection']

    from semantic_search import search_memories
    saved_state = _save_state()

    query_metrics = []
    total_ms = 0

    try:
        with StageDisabler(all_except_entity):
            for entry in corpus:
                random.seed(seed)
                t0 = time.monotonic()
                search_results = search_memories(
                    entry['query'], limit=10, threshold=0.2,
                    register_recall=False, skip_monologue=True,
                )
                elapsed = (time.monotonic() - t0) * 1000
                total_ms += elapsed

                ablated_ids = [r['id'] for r in search_results]
                baseline_data = baseline['results'].get(entry['query'], {})
                baseline_ids = baseline_data.get('ids', [])
                expected_ids = set(entry.get('expected_ids', baseline_ids[:5]))

                metrics = compute_query_metrics(baseline_ids, ablated_ids, expected_ids)
                metrics['query'] = entry['query']
                query_metrics.append(metrics)
    finally:
        _restore_state(saved_state)

    n = len(query_metrics)
    return {
        'pipeline': 'minimal (pgvector + entity_injection only)',
        'mean_precision': round(sum(m['precision_ablated'] for m in query_metrics) / n, 4) if n else 0,
        'baseline_precision': round(sum(m['precision_baseline'] for m in query_metrics) / n, 4) if n else 0,
        'precision_gap': round(
            sum(m['precision_baseline'] - m['precision_ablated'] for m in query_metrics) / n, 4
        ) if n else 0,
        'total_ms': round(total_ms, 1),
        'time_saved_ms': round(baseline.get('total_ms', 0) - total_ms, 1),
        'query_count': n,
    }


# ============================================================
# Passive Live Logger
# ============================================================

def log_passive_annotations(search_results: list[dict], query: str = ''):
    """Called from stop hook. Logs which stages touched final results."""
    if not search_results:
        return

    annotations = {
        'entity_injected': sum(1 for r in search_results if r.get('entity_injected')),
        'entity_matched': sum(1 for r in search_results if r.get('entity_match')),
        'hub_dampened': sum(1 for r in search_results if r.get('hub_dampened')),
        'gravity_dampened': sum(1 for r in search_results if r.get('dampened')),
        'somatic_biased': sum(1 for r in search_results if r.get('somatic_bias')),
        'mood_boosted': sum(1 for r in search_results if r.get('mood_boost')),
        'q_reranked': sum(1 for r in search_results if r.get('q_value') and r.get('q_value') != 0.5),
        'goal_boosted': sum(1 for r in search_results if r.get('goal_boosted')),
        'spread_activated': sum(1 for r in search_results if r.get('spread_activated')),
        'curiosity_boosted': sum(1 for r in search_results if r.get('curiosity_boosted')),
        'dim_boosted': sum(1 for r in search_results if r.get('dim_boosted')),
        'resolution_boosted': sum(1 for r in search_results if r.get('boosted')),
        'score_capped': sum(1 for r in search_results if r.get('score_capped')),
    }

    entry = {
        'query': query[:100],
        'result_count': len(search_results),
        'annotations': annotations,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }

    try:
        db = _get_db()
        log = db.kv_get(KV_PASSIVE_LOG) or []
        log.append(entry)
        db.kv_set(KV_PASSIVE_LOG, log[-100:])  # Rolling 100
    except Exception:
        pass


def analyze_passive_log() -> dict:
    """Analyze passive live log data."""
    db = _get_db()
    log = db.kv_get(KV_PASSIVE_LOG) or []
    if not log:
        return {'message': 'No passive log data yet.', 'sessions': 0}

    n = len(log)
    stage_touch_rates = {}
    for entry in log:
        total = entry.get('result_count', 1)
        for stage, count in entry.get('annotations', {}).items():
            if stage not in stage_touch_rates:
                stage_touch_rates[stage] = []
            stage_touch_rates[stage].append(count / max(total, 1))

    summary = {}
    for stage, rates in stage_touch_rates.items():
        summary[stage] = {
            'mean_touch_rate': round(sum(rates) / len(rates), 3),
            'max_touch_rate': round(max(rates), 3),
            'fire_rate': round(sum(1 for r in rates if r > 0) / len(rates), 3),
        }

    return {
        'total_searches': n,
        'stage_touch_rates': summary,
    }


# ============================================================
# Report Generator
# ============================================================

def generate_report() -> str:
    """Generate human-readable analysis report from stored results."""
    db = _get_db()
    baseline = db.kv_get(KV_BASELINE)
    single = db.kv_get(KV_SINGLE)
    pairwise = db.kv_get(KV_PAIRWISE)
    passive = analyze_passive_log()

    lines = [
        f'ABLATION REPORT — {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        '=' * 60,
        '',
    ]

    # Baseline
    if baseline:
        lines.append(f'BASELINE: {baseline["query_count"]} queries, '
                      f'avg {baseline["avg_ms"]:.0f}ms/query, '
                      f'seed={baseline["seed"]}')
        lines.append('')

    # Single-stage results
    if single and single.get('stages'):
        lines.append('SINGLE-STAGE ABLATION (sorted by impact):')
        lines.append(f'  {"Stage":<28} | {"P@5 delta":>10} | {"MRR delta":>10} | '
                      f'{"Tau":>6} | {"Unique":>6} | {"Rating":>10}')
        lines.append('  ' + '-' * 84)

        for stage in single.get('ranked', []):
            data = single['stages'][stage]
            p_delta = data['mean_precision_delta']
            mrr_d = data['mean_mrr_delta']
            tau = data['mean_kendall_tau']
            unique = data['total_unique_lost']

            # Rating
            if p_delta < -0.10:
                rating = 'CRITICAL'
            elif p_delta < -0.03:
                rating = 'VALUABLE'
            elif p_delta < -0.01:
                rating = 'LOW VALUE'
            elif unique == 0 and abs(p_delta) < 0.01:
                rating = 'DEAD?'
            else:
                rating = 'NEUTRAL'

            lines.append(f'  {stage:<28} | {p_delta:>+10.3f} | {mrr_d:>+10.3f} | '
                          f'{tau:>6.2f} | {unique:>6} | {rating:>10}')

        lines.append('')

    # Pairwise
    if pairwise and pairwise.get('pairs'):
        lines.append('PAIRWISE INTERACTIONS (sorted by |synergy|):')
        pairs_sorted = sorted(pairwise['pairs'], key=lambda x: abs(x['synergy']), reverse=True)
        for p in pairs_sorted:
            pair_name = f'{p["pair"][0]} + {p["pair"][1]}'
            lines.append(f'  {pair_name:<45} | synergy: {p["synergy"]:+.3f} | {p["interpretation"]}')
        lines.append('')

    # Passive log
    if passive.get('total_searches', 0) > 0:
        lines.append(f'PASSIVE LIVE LOG ({passive["total_searches"]} searches):')
        for stage, data in sorted(passive.get('stage_touch_rates', {}).items(),
                                   key=lambda x: x[1]['mean_touch_rate'], reverse=True):
            lines.append(f'  {stage:<25} | fire rate: {data["fire_rate"]:.0%} | '
                          f'avg touch: {data["mean_touch_rate"]:.2f}')
        lines.append('')

    report_text = '\n'.join(lines)

    # Store report
    reports = db.kv_get(KV_REPORTS) or []
    reports.append({
        'text': report_text,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })
    db.kv_set(KV_REPORTS, reports[-5:])

    return report_text


# ============================================================
# Health
# ============================================================

def health_check() -> tuple:
    """Health check for toolkit integration."""
    n_stages = len(STAGE_REGISTRY)
    db = _get_db()
    has_corpus = bool(db.kv_get(KV_CORPUS))
    has_baseline = bool(db.kv_get(KV_BASELINE))
    detail = f'{n_stages} stages'
    if has_corpus:
        detail += ', corpus ready'
    if has_baseline:
        detail += ', baseline cached'
    return True, detail


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Ablation Testing Framework')
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('stages', help='List all ablatable stages')

    corpus_p = sub.add_parser('corpus', help='Manage query corpus')
    corpus_p.add_argument('--mine', action='store_true', help='Mine queries from attention_schema')
    corpus_p.add_argument('--build', action='store_true', help='Build ground truth from full pipeline')
    corpus_p.add_argument('--validate', action='store_true', help='Validate corpus has ground truth')

    sub.add_parser('baseline', help='Run and store baseline')

    bench_p = sub.add_parser('benchmark', help='Run single-stage ablation sweep')
    bench_p.add_argument('--stage', type=str, help='Ablate specific stage only')
    bench_p.add_argument('--seed', type=int, default=42)

    inter_p = sub.add_parser('interactions', help='Pairwise interaction tests')
    inter_p.add_argument('--smart', action='store_true', help='Smart sampling (recommended)')
    inter_p.add_argument('--threshold', type=float, default=0.02)

    sub.add_parser('minimal', help='Minimal pipeline (pgvector + entity only)')
    sub.add_parser('report', help='Generate analysis report')
    sub.add_parser('analyze', help='Analyze passive live logs')
    sub.add_parser('health', help='Health check')

    args = parser.parse_args()

    if args.command == 'stages':
        print(f'\nAblatable Pipeline Stages ({len(STAGE_REGISTRY)}):')
        print(f'  {"Stage":<28} | {"Strategy":<15} | {"Coupled":<30} | Description')
        print('  ' + '-' * 105)
        for name, config in STAGE_REGISTRY.items():
            coupled = ', '.join(config.coupled_with) if config.coupled_with else '—'
            marker = '*' if name in ANNOTATION_STAGES else ' '
            print(f' {marker}{name:<28} | {config.strategy:<15} | {coupled:<30} | {config.description}')
        print(f'\n  * = annotation-only (no score change)')
        print(f'  Score stages: {len(SCORE_STAGES)}, Annotation stages: {len(ANNOTATION_STAGES)}')

    elif args.command == 'corpus':
        if args.mine:
            mined = mine_queries_from_history()
            print(f'Mined {len(mined)} queries from attention_schema:')
            for q in mined[:10]:
                print(f'  - {q["query"]}')

        elif args.build:
            print('Building ground truth (running full pipeline on each query)...')
            corpus = build_ground_truth()
            print(f'Built ground truth for {len(corpus)} queries.')
            for entry in corpus:
                n_expected = len(entry.get('expected_ids', []))
                print(f'  [{entry.get("domain", "?")}] {entry["query"][:60]} -> {n_expected} expected')

        elif args.validate:
            corpus = load_corpus()
            has_gt = sum(1 for q in corpus if q.get('expected_ids'))
            print(f'Corpus: {len(corpus)} queries, {has_gt} with ground truth')
            if has_gt < len(corpus):
                print(f'  WARNING: {len(corpus) - has_gt} queries missing ground truth. Run --build')
        else:
            corpus = load_corpus()
            print(f'Corpus: {len(corpus)} queries')
            for q in corpus:
                gt = len(q.get('expected_ids', []))
                print(f'  [{q.get("domain", "?")}] {q["query"][:60]} ({"GT:" + str(gt) if gt else "no GT"})')

    elif args.command == 'baseline':
        corpus = load_corpus()
        print(f'Running baseline ({len(corpus)} queries)...')
        baseline = run_baseline(corpus, seed=42)
        print(f'Baseline complete: avg {baseline["avg_ms"]:.0f}ms/query, '
              f'total {baseline["total_ms"]:.0f}ms')

    elif args.command == 'benchmark':
        corpus = load_corpus()
        if not any(q.get('expected_ids') for q in corpus):
            print('No ground truth. Building...')
            corpus = build_ground_truth()

        print(f'Running baseline...')
        baseline = run_baseline(corpus, seed=args.seed)
        print(f'Baseline: avg {baseline["avg_ms"]:.0f}ms/query\n')

        if args.stage:
            if args.stage not in STAGE_REGISTRY:
                print(f'Unknown stage: {args.stage}')
                print(f'Available: {", ".join(STAGE_REGISTRY.keys())}')
                return
            print(f'Ablating: {args.stage}')
            result = run_single_ablation(args.stage, corpus, baseline, args.seed)
            print(f'  P@5 delta:  {result["mean_precision_delta"]:+.3f}')
            print(f'  MRR delta:  {result["mean_mrr_delta"]:+.3f}')
            print(f'  Kendall τ:  {result["mean_kendall_tau"]:.3f}')
            print(f'  Unique lost: {result["total_unique_lost"]}')
            print(f'  Time saved:  {result["time_saved_ms"]:.0f}ms')
        else:
            print('Full single-stage ablation sweep:')
            results = run_all_single(corpus, baseline, seed=args.seed)
            print(f'\nSweep complete. {len(results["stages"])} stages tested.')
            print('Run `python ablation_framework.py report` for full analysis.')

    elif args.command == 'interactions':
        db = _get_db()
        single = db.kv_get(KV_SINGLE)
        if not single:
            print('Run benchmark first: python ablation_framework.py benchmark')
            return

        corpus = load_corpus()
        baseline = db.kv_get(KV_BASELINE)
        if not baseline:
            baseline = run_baseline(corpus)

        if args.smart:
            pairs = get_smart_pairs(single.get('stages', {}), args.threshold)
        else:
            # All coupled pairs
            pairs = COUPLED_PAIRS

        print(f'Testing {len(pairs)} pairs:')
        pair_results = run_pairwise(pairs, corpus, baseline,
                                     single.get('stages', {}))
        print(f'\n{len(pair_results)} pairs tested.')

    elif args.command == 'minimal':
        corpus = load_corpus()
        if not any(q.get('expected_ids') for q in corpus):
            corpus = build_ground_truth()

        baseline = run_baseline(corpus)
        print('Running minimal pipeline (pgvector + entity_injection only)...')
        result = run_minimal_pipeline(corpus, baseline)
        print(f'  Baseline P@5:  {result["baseline_precision"]:.3f}')
        print(f'  Minimal P@5:   {result["mean_precision"]:.3f}')
        print(f'  Gap:           {result["precision_gap"]:+.3f}')
        print(f'  Time saved:    {result["time_saved_ms"]:.0f}ms')

    elif args.command == 'report':
        report = generate_report()
        print(report)

    elif args.command == 'analyze':
        result = analyze_passive_log()
        if result.get('message'):
            print(result['message'])
        else:
            print(f'Passive log: {result["total_searches"]} searches tracked\n')
            for stage, data in sorted(result.get('stage_touch_rates', {}).items(),
                                       key=lambda x: x[1]['mean_touch_rate'], reverse=True):
                print(f'  {stage:<25} | fires: {data["fire_rate"]:.0%} | '
                      f'avg touch: {data["mean_touch_rate"]:.2f}')

    elif args.command == 'health':
        ok, detail = health_check()
        print(f'{"OK" if ok else "FAIL"}: {detail}')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
