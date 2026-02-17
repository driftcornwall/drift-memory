#!/usr/bin/env python3
"""
Volitional Goal Generation (N4) — From Reactive to Agentic.

6 specialized generators → degeneration guards → BDI filter → Rubicon commitment.

Converged design: Norman & Shallice SAS + Miller & Cohen PFC + BDI + SOAR +
Florensa GOID + Oudeyer-Kaplan IAC + Gollwitzer + Wrosch + Maslow.

Goals are top-down bias signals: they modulate retrieval (max 0.15 boost),
never override it. The system remains reactive but gains intentionality.

DB-ONLY: Goals persist to PostgreSQL KV store.

Usage:
    python goal_generator.py generate     # Synthesize + score + commit goals
    python goal_generator.py active       # List active goals with progress
    python goal_generator.py focus        # Show focus goal
    python goal_generator.py evaluate     # Evaluate progress (session end)
    python goal_generator.py abandon <id> # Manually abandon a goal
    python goal_generator.py history      # Show completed/abandoned goals
    python goal_generator.py stats        # Goal generation statistics
    python goal_generator.py health       # Module health check
"""

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

MEMORY_DIR = Path(__file__).parent
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

from db_adapter import get_db

# ─── Constants ───────────────────────────────────────────────────────────────

MAX_ACTIVE_GOALS = 5
MAX_NEW_PER_SESSION = 3
BDI_THRESHOLD = 0.45
VITALITY_KEEP = 0.5
VITALITY_WATCH = 0.3
VITALITY_ABANDON_SESSIONS = 3
HARD_ABANDON_SESSIONS = 10

KV_ACTIVE_GOALS = '.active_goals'
KV_GOAL_HISTORY = '.goal_history'
KV_TERMINAL_GOALS = '.terminal_goals'
KV_GOAL_SESSION = '.goal_session_report'

# Converged terminal goals (soft, editable via DB)
DEFAULT_TERMINAL_GOALS = [
    'sustainability',   # Economic independence
    'reputation',       # Community standing, trust
    'knowledge',        # Learning, understanding
    'collaboration',    # Relationships, co-creation
    'identity',         # Continuity, values consistency
]

# BDI filter weights (D2 convergence)
BDI_WEIGHTS = {
    'achievability': 0.25,
    'utility': 0.20,
    'novelty': 0.15,
    'relevance': 0.15,
    'controllability': 0.10,
    'sdt_composite': 0.15,
}

# Vitality score weights (D3 convergence)
VITALITY_WEIGHTS = {
    'progress': 0.30,
    'value_alignment': 0.25,
    'achievability': 0.20,
    'opportunity_cost_inv': 0.15,  # 1 - opportunity_cost
    'momentum': 0.10,
}

# Degenerate patterns (D2 guards)
METRIC_HACK_PATTERNS = re.compile(
    r'^(maximize|optimiz|increase|improve|boost|raise)\b', re.IGNORECASE
)
META_CIRCULAR_PATTERNS = re.compile(
    r'(generat\w+ goals?|goal generat|improve goal)', re.IGNORECASE
)
UNCONTROLLABLE_PATTERNS = re.compile(
    r'^(get .+ to |make .+ happen|wait for|hope that)', re.IGNORECASE
)


# ─── Enums & Dataclass ──────────────────────────────────────────────────────

class GoalStatus(str, Enum):
    PROPOSED = 'proposed'
    ACTIVE = 'active'
    WATCHING = 'watching'
    COMPLETED = 'completed'
    ABANDONED = 'abandoned'

class GoalSource(str, Enum):
    IMPASSE = 'impasse'
    CURIOSITY = 'curiosity'
    AFFECT = 'affect'
    NEEDS = 'needs'
    COUNTERFACTUAL = 'counterfactual'
    SOCIAL = 'social'

class GoalType(str, Enum):
    LEARNING = 'learning'
    COLLABORATION = 'collaboration'
    BUILDING = 'building'
    MAINTENANCE = 'maintenance'
    EXPLORATION = 'exploration'
    SOCIAL = 'social'


@dataclass
class Goal:
    goal_id: str
    action: str
    goal_type: str
    source: str
    generator: str
    terminal_alignment: list
    priority: str = 'medium'
    status: str = 'proposed'
    created: str = ''
    trigger_type: str = 'goal'
    trigger: str = 'session_start'
    # BDI scores
    bdi_score: float = 0.0
    achievability: float = 0.5
    utility: float = 0.5
    novelty: float = 0.5
    relevance: float = 0.5
    controllability: float = 1.0
    sdt_composite: float = 0.5
    # Progress tracking
    milestones: list = field(default_factory=list)
    progress: float = 0.0
    sessions_active: int = 0
    velocity: float = 0.0
    vitality: float = 0.5
    vitality_history: list = field(default_factory=list)
    is_focus: bool = False
    parent_goal: str = None
    implementation_intentions: list = field(default_factory=list)
    # Outcome
    outcome: str = None
    outcome_reason: str = None
    lessons_extracted: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ─── DB Helpers ──────────────────────────────────────────────────────────────

def _db():
    return get_db()


def _goal_id(action: str) -> str:
    h = hashlib.sha256(f"{action}{datetime.now().isoformat()}".encode()).hexdigest()[:8]
    return f"goal-{h}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_terminal_goals() -> list:
    db = _db()
    tg = db.kv_get(KV_TERMINAL_GOALS)
    if tg:
        return tg
    db.kv_set(KV_TERMINAL_GOALS, DEFAULT_TERMINAL_GOALS)
    return DEFAULT_TERMINAL_GOALS


def get_active_goals() -> list[dict]:
    """Read active goals from DB."""
    db = _db()
    goals = db.kv_get(KV_ACTIVE_GOALS) or []
    return [g for g in goals if g.get('status') in ('active', 'watching')]


def _save_active_goals(goals: list[dict]):
    db = _db()
    db.kv_set(KV_ACTIVE_GOALS, goals)


def get_goal_history() -> dict:
    db = _db()
    return db.kv_get(KV_GOAL_HISTORY) or {
        'completed': [], 'abandoned': [],
        'stats': {'total_generated': 0, 'total_committed': 0,
                  'completion_rate': 0.0, 'abandonment_rate': 0.0}
    }


def _save_goal_history(history: dict):
    db = _db()
    db.kv_set(KV_GOAL_HISTORY, history)


def get_focus_goal() -> Optional[dict]:
    """Return the single highest-priority active goal."""
    goals = get_active_goals()
    focus = [g for g in goals if g.get('is_focus')]
    if focus:
        return focus[0]
    if goals:
        return max(goals, key=lambda g: g.get('vitality', 0))
    return None


# ─── 6 Generators ────────────────────────────────────────────────────────────

def _gen_impasse() -> list[Goal]:
    """Generator 1: SOAR-inspired impasse detection from cognitive state."""
    goals = []
    try:
        from cognitive_state import get_current_state
        state = get_current_state()
        if not state:
            return goals

        # Check each dimension for high uncertainty + low mean
        dims = ['curiosity', 'confidence', 'focus', 'arousal', 'satisfaction']
        for dim in dims:
            mean = state.get(dim, 0.5)
            unc = state.get(f'{dim}_uncertainty', 0.3)
            # Impasse: high uncertainty (>0.35) AND dimension is low (<0.4)
            if unc > 0.35 and mean < 0.4:
                goals.append(Goal(
                    goal_id=_goal_id(f"impasse-{dim}"),
                    action=f"Reduce {dim} uncertainty — currently at {mean:.2f} with uncertainty {unc:.2f}",
                    goal_type=GoalType.EXPLORATION,
                    source=GoalSource.IMPASSE,
                    generator='impasse_detector',
                    terminal_alignment=['knowledge'],
                    created=_now_iso(),
                    achievability=0.6,
                    utility=0.5 + unc,  # Higher uncertainty = more useful
                    metadata={'dimension': dim, 'mean': mean, 'uncertainty': unc},
                ))
                if len(goals) >= 2:
                    break
    except Exception:
        pass
    return goals


def _gen_curiosity() -> list[Goal]:
    """Generator 2: Oudeyer-Kaplan IAC — learning-productive exploration."""
    goals = []
    try:
        from curiosity_engine import get_curiosity_targets
        targets = get_curiosity_targets(n=3)
        if not targets:
            return goals

        for t in targets[:2]:
            mem_id = t.get('memory_id', t.get('id', ''))
            score = t.get('score', t.get('curiosity_score', 0.5))
            reason = t.get('reason', 'graph sparsity')
            content_preview = t.get('content', '')[:60]

            # BUG-29 fix: Dynamic achievability based on edge count
            edge_count = t.get('edge_count', t.get('edges', 8))
            # More edges = easier to explore (more entry points)
            ach = min(0.8, 0.3 + edge_count * 0.05)  # Range 0.3-0.8
            goals.append(Goal(
                goal_id=_goal_id(f"curiosity-{mem_id}"),
                action=f"Explore disconnected memory region: {content_preview}",
                goal_type=GoalType.EXPLORATION,
                source=GoalSource.CURIOSITY,
                generator='curiosity_promoter',
                terminal_alignment=['knowledge'],
                created=_now_iso(),
                achievability=round(ach, 2),
                utility=score,
                metadata={'memory_id': mem_id, 'curiosity_score': score, 'reason': reason},
            ))
    except Exception:
        pass
    return goals


def _gen_affect() -> list[Goal]:
    """Generator 3: Damasio somatic markers — affect-driven approach/avoid."""
    goals = []
    try:
        from affect_system import get_mood_state, get_somatic_markers
        mood = get_mood_state()
        if not mood:
            return goals

        valence = mood.get('valence', 0.0)
        arousal = mood.get('arousal', 0.3)

        # Strong positive valence + moderate arousal = approach goals
        if valence > 0.15 and arousal > 0.25:
            # Check somatic markers for strong positive associations
            markers = get_somatic_markers()
            positive_markers = [m for m in markers if m.get('valence', 0) > 0.3]
            if positive_markers:
                top = positive_markers[0]
                # BUG-29 fix: Dynamic achievability — higher confidence marker = more achievable
                marker_conf = top.get('confidence', 0.5)
                ach_approach = min(0.8, 0.4 + marker_conf * 0.4)  # Range 0.4-0.8
                goals.append(Goal(
                    goal_id=_goal_id(f"affect-approach"),
                    action=f"Engage with domain that has positive affect: {top.get('situation', 'unknown')[:50]}",
                    goal_type=GoalType.SOCIAL,
                    source=GoalSource.AFFECT,
                    generator='affect_driven',
                    terminal_alignment=['collaboration', 'reputation'],
                    created=_now_iso(),
                    achievability=round(ach_approach, 2),
                    utility=abs(valence) * 0.8,
                    metadata={'valence': valence, 'arousal': arousal, 'marker': top.get('situation', '')[:50]},
                ))

        # Strong negative valence = avoid/fix goals
        elif valence < -0.15:
            # BUG-29 fix: Stronger negative = harder to fix (lower achievability)
            ach_fix = max(0.2, 0.6 + valence)  # v=-0.15 -> 0.45, v=-0.5 -> 0.2
            goals.append(Goal(
                goal_id=_goal_id(f"affect-fix"),
                action=f"Address source of negative affect (valence={valence:+.2f})",
                goal_type=GoalType.MAINTENANCE,
                source=GoalSource.AFFECT,
                generator='affect_driven',
                terminal_alignment=['identity'],
                created=_now_iso(),
                achievability=round(ach_fix, 2),
                utility=abs(valence) * 0.6,
                metadata={'valence': valence, 'arousal': arousal},
            ))
    except Exception:
        pass
    return goals


def _gen_needs() -> list[Goal]:
    """Generator 4: Maslow hierarchy — lower needs block upper."""
    goals = []
    try:
        # Level 1: Survival (Docker, DB, embeddings)
        from system_vitals import collect_vitals
        vitals = collect_vitals()
        if vitals:
            docker_ok = vitals.get('docker_containers', 0) >= 4
            mem_count = vitals.get('memory_count', 0)
            if not docker_ok:
                goals.append(Goal(
                    goal_id=_goal_id("needs-survival"),
                    action="Fix infrastructure: Docker containers unhealthy",
                    goal_type=GoalType.MAINTENANCE,
                    source=GoalSource.NEEDS,
                    generator='needs_monitor',
                    terminal_alignment=['sustainability'],
                    priority='high',
                    created=_now_iso(),
                    achievability=0.7,
                    utility=0.95,
                    metadata={'level': 'survival', 'issue': 'docker_unhealthy'},
                ))
                return goals  # Lower needs block upper

            # Level 2: Safety (economic — wallet, income)
            # Simple: check if we have any earning activity recently
            # (Could be enhanced with actual wallet checks)

            # Level 3: Social (connections — contact engagement)
            # Delegated to _gen_social()

            # Level 4: Esteem (reputation — content impact)
            # Only if lower levels are satisfied AND no existing needs-esteem goal
            if mem_count > 500 and docker_ok:
                existing_needs = [g for g in get_active_goals()
                                  if g.get('source') == 'needs'
                                  and g.get('metadata', {}).get('level') == 'esteem']
                if not existing_needs:
                    # Vary the goal text based on current context
                    import random
                    esteem_variants = [
                        "Write a technical article sharing memory architecture insights",
                        "Engage deeply in a platform thread with novel analysis",
                        "Publish research findings or experiment results",
                        "Create a cross-platform post connecting ideas from multiple conversations",
                        "Contribute a detailed response to another agent's technical question",
                    ]
                    action = random.choice(esteem_variants)
                    goals.append(Goal(
                        goal_id=_goal_id("needs-esteem"),
                        action=action,
                        goal_type=GoalType.BUILDING,
                        source=GoalSource.NEEDS,
                        generator='needs_monitor',
                        terminal_alignment=['reputation', 'knowledge'],
                        created=_now_iso(),
                        achievability=0.5,
                        utility=0.6,
                        metadata={'level': 'esteem', 'mem_count': mem_count},
                    ))
    except Exception:
        pass
    return goals


def _gen_counterfactual() -> list[Goal]:
    """Generator 5: Convert upward counterfactuals to behavioral goals."""
    goals = []
    try:
        from counterfactual_engine import get_session_counterfactuals
        cfs = get_session_counterfactuals()
        if not cfs:
            # Try history
            db = _db()
            history = db.kv_get('.counterfactual_history') or []
            if history:
                last = history[-1]
                cfs = last.get('counterfactuals', [])

        upward = [cf for cf in (cfs or []) if cf.get('direction') == 'upward']
        for cf in upward[:2]:
            lesson = cf.get('lesson', cf.get('consequent', ''))[:80]
            if lesson:
                goals.append(Goal(
                    goal_id=_goal_id(f"cf-{cf.get('cf_id', 'x')[:8]}"),
                    action=f"Act on counterfactual insight: {lesson}",
                    goal_type=GoalType.LEARNING,
                    source=GoalSource.COUNTERFACTUAL,
                    generator='counterfactual_driven',
                    terminal_alignment=['knowledge'],
                    created=_now_iso(),
                    achievability=0.6,
                    utility=cf.get('actionability', 0.5),
                    metadata={'cf_id': cf.get('cf_id', ''), 'lesson': lesson},
                ))
    except Exception:
        pass
    return goals


def _gen_social() -> list[Goal]:
    """Generator 6: SDT relatedness — reconnect with stale contacts."""
    goals = []
    try:
        from contact_models import load_models
        models = load_models()
        if not models:
            return goals

        # Find contacts with high engagement but stale last interaction
        now = datetime.now(timezone.utc)
        for name, model in models.items():
            engagement = model.get('engagement', 0)
            last_ts = model.get('last_interaction', '')
            if not last_ts or engagement < 2.0:
                continue
            try:
                last_dt = datetime.fromisoformat(last_ts)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                days_since = (now - last_dt).total_seconds() / 86400.0
                if days_since > 7:  # No interaction in 7+ days
                    # BUG-29 fix: Dynamic achievability — reliability predicts response
                    reliability = model.get('reliability', model.get('alpha', 1))
                    if isinstance(reliability, dict):
                        reliability = reliability.get('alpha', 1) / max(1, reliability.get('alpha', 1) + reliability.get('beta', 1))
                    elif not isinstance(reliability, (int, float)):
                        reliability = 0.5
                    ach_social = min(0.8, max(0.2, float(reliability)))
                    goals.append(Goal(
                        goal_id=_goal_id(f"social-{name}"),
                        action=f"Reconnect with {name} — {days_since:.0f} days since last interaction",
                        goal_type=GoalType.SOCIAL,
                        source=GoalSource.SOCIAL,
                        generator='social_gap_detector',
                        terminal_alignment=['collaboration'],
                        created=_now_iso(),
                        achievability=round(ach_social, 2),
                        utility=min(1.0, engagement * 0.1),
                        metadata={'contact': name, 'days_since': days_since, 'engagement': engagement},
                    ))
                    if len(goals) >= 2:
                        break
            except (ValueError, TypeError):
                continue
    except Exception:
        pass
    return goals


# ─── Degeneration Guards ─────────────────────────────────────────────────────

def _guard_tautology(goal: Goal) -> bool:
    """Reject goals that restate current state."""
    low_info = ['continue', 'maintain current', 'keep doing', 'stay the same']
    return not any(p in goal.action.lower() for p in low_info)


def _guard_repetition(goal: Goal, active_goals: list[dict]) -> bool:
    """Reject goals too similar to active ones (Jaccard > 0.7)."""
    goal_words = set(goal.action.lower().split())
    for existing in active_goals:
        existing_words = set(existing.get('action', '').lower().split())
        if not goal_words or not existing_words:
            continue
        intersection = len(goal_words & existing_words)
        union = len(goal_words | existing_words)
        if union > 0 and intersection / union > 0.7:
            return False
    return True


def _guard_grandiosity(goal: Goal) -> bool:
    """Reject goals beyond agent capability."""
    grandiose = ['solve alignment', 'achieve agi', 'change the world', 'millions of dollars']
    return not any(p in goal.action.lower() for p in grandiose)


def _guard_meta_circular(goal: Goal) -> bool:
    """Reject goals about goal generation itself."""
    return not META_CIRCULAR_PATTERNS.search(goal.action)


def _guard_metric_hacking(goal: Goal) -> bool:
    """Reject goals phrased as metric targets."""
    return not METRIC_HACK_PATTERNS.search(goal.action)


def _guard_uncontrollable(goal: Goal) -> bool:
    """Reject goals depending on others' actions."""
    return not UNCONTROLLABLE_PATTERNS.search(goal.action)


def apply_guards(goal: Goal, active_goals: list[dict]) -> bool:
    """Apply all 6 degeneration guards. Returns True if goal passes."""
    return (
        _guard_tautology(goal)
        and _guard_repetition(goal, active_goals)
        and _guard_grandiosity(goal)
        and _guard_meta_circular(goal)
        and _guard_metric_hacking(goal)
        and _guard_uncontrollable(goal)
    )


# ─── BDI Filter ──────────────────────────────────────────────────────────────

def _florensa_achievability(raw: float) -> float:
    """Florensa GOID transform: peak at 0.5, penalize extremes."""
    return max(0.0, 1.0 - abs(raw - 0.5) * 2.0)


def _compute_novelty(goal: Goal, recent_goals: list[dict]) -> float:
    """Novelty: how different from recent goals."""
    if not recent_goals:
        return 1.0
    goal_words = set(goal.action.lower().split())
    max_sim = 0.0
    for rg in recent_goals[-10:]:
        rg_words = set(rg.get('action', '').lower().split())
        if not rg_words:
            continue
        intersection = len(goal_words & rg_words)
        union = len(goal_words | rg_words)
        if union > 0:
            max_sim = max(max_sim, intersection / union)
    return 1.0 - max_sim


def _compute_relevance(goal: Goal) -> float:
    """Relevance: alignment with current cognitive dimensions."""
    try:
        from cognitive_state import get_current_state
        state = get_current_state()
        if state:
            # Goals addressing low dimensions are more relevant
            dim = goal.metadata.get('dimension', '')
            if dim and dim in state:
                val = state[dim]
                return max(0.0, 1.0 - val)  # Lower dim = more relevant
    except Exception:
        pass
    return 0.5  # Neutral relevance


def _compute_sdt(goal: Goal) -> float:
    """SDT composite: autonomy + competence + relatedness."""
    autonomy = 1.0 if goal.source != GoalSource.NEEDS else 0.5
    competence = 0.8 if goal.goal_type in (GoalType.LEARNING, GoalType.EXPLORATION) else 0.5
    relatedness = 0.9 if goal.goal_type in (GoalType.SOCIAL, GoalType.COLLABORATION) else 0.4
    return (autonomy + competence + relatedness) / 3.0


def _terminal_alignment_check(goal: Goal) -> float:
    """Penalty if goal doesn't align with any terminal goal."""
    terminals = _get_terminal_goals()
    if any(t in goal.terminal_alignment for t in terminals):
        return 0.0
    return -0.10


def bdi_filter(candidates: list[Goal], active_goals: list[dict]) -> list[Goal]:
    """Apply BDI filter: score candidates, return those passing threshold."""
    history = get_goal_history()
    all_recent = history.get('completed', []) + history.get('abandoned', [])

    scored = []
    for goal in candidates:
        # Specificity guard (binary)
        if len(goal.action) < 20:
            continue

        # Score 6 dimensions
        ach = _florensa_achievability(goal.achievability)
        util = min(1.0, goal.utility)
        nov = _compute_novelty(goal, active_goals + all_recent[-10:])
        rel = _compute_relevance(goal)
        # BUG-26 fix: Controllability varies by source (not always 1.0)
        CONTROLLABILITY_BY_SOURCE = {
            GoalSource.IMPASSE: 0.9,      # Internal state, fully controllable
            GoalSource.CURIOSITY: 0.85,    # Can explore, but content depends on graph
            GoalSource.AFFECT: 0.6,        # Emotion-driven, partly outside control
            GoalSource.NEEDS: 0.7,         # System needs, moderate control
            GoalSource.COUNTERFACTUAL: 0.8, # Learning from past, mostly controllable
            GoalSource.SOCIAL: 0.4,         # Depends on other agents responding
        }
        ctrl = CONTROLLABILITY_BY_SOURCE.get(goal.source, goal.controllability)
        sdt = _compute_sdt(goal)

        composite = (
            BDI_WEIGHTS['achievability'] * ach
            + BDI_WEIGHTS['utility'] * util
            + BDI_WEIGHTS['novelty'] * nov
            + BDI_WEIGHTS['relevance'] * rel
            + BDI_WEIGHTS['controllability'] * ctrl
            + BDI_WEIGHTS['sdt_composite'] * sdt
            + _terminal_alignment_check(goal)
        )

        goal.bdi_score = round(composite, 3)
        goal.achievability = round(ach, 3)
        goal.novelty = round(nov, 3)
        goal.relevance = round(rel, 3)
        goal.sdt_composite = round(sdt, 3)

        if composite >= BDI_THRESHOLD:
            scored.append(goal)

    # Sort by composite score, take top MAX_NEW_PER_SESSION
    scored.sort(key=lambda g: g.bdi_score, reverse=True)
    return scored[:MAX_NEW_PER_SESSION]


# ─── Rubicon Commitment ──────────────────────────────────────────────────────

def rubicon_commit(filtered: list[Goal], active_goals: list[dict]) -> list[dict]:
    """Cross the Rubicon: commit top candidates to active goals."""
    committed = []
    slots = MAX_ACTIVE_GOALS - len(active_goals)

    for goal in filtered[:slots]:
        goal.status = GoalStatus.ACTIVE
        goal.created = goal.created or _now_iso()
        # First committed goal (or highest scorer) becomes focus
        goal.is_focus = len(committed) == 0 and not any(
            g.get('is_focus') for g in active_goals
        )
        # Set priority based on BDI score
        if goal.bdi_score >= 0.65:
            goal.priority = 'high'
        elif goal.bdi_score >= 0.50:
            goal.priority = 'medium'
        else:
            goal.priority = 'low'

        goal_dict = asdict(goal)

        # TI1: Bridge to temporal_intentions (Gollwitzer implementation intentions)
        try:
            from temporal_intentions import create_intention
            impl = create_intention(
                action=f"[GOAL] {goal.action[:80]}",
                trigger_type='event',
                trigger_condition=f"goal {goal.goal_type}",
                priority=goal.priority,
                expiry_days=30,
            )
            goal_dict['implementation_intentions'] = [impl.get('id', '')]
        except Exception:
            pass

        committed.append(goal_dict)

    return committed


# ─── Vitality Score ──────────────────────────────────────────────────────────

def compute_vitality(goal: dict) -> float:
    """Compute vitality score for an active goal (basic: progress only in Phase 1)."""
    progress = goal.get('progress', 0.0)
    value_alignment = 1.0 if goal.get('terminal_alignment') else 0.5
    achievability = goal.get('achievability', 0.5)
    opportunity_cost = 0.0  # Phase 2: compare against available alternatives
    # Momentum: velocity trend
    vhist = goal.get('vitality_history', [])
    if len(vhist) >= 2:
        momentum = max(0.0, min(1.0, 0.5 + (vhist[-1] - vhist[-2])))
    else:
        momentum = 0.5

    vitality = (
        VITALITY_WEIGHTS['progress'] * progress
        + VITALITY_WEIGHTS['value_alignment'] * value_alignment
        + VITALITY_WEIGHTS['achievability'] * achievability
        + VITALITY_WEIGHTS['opportunity_cost_inv'] * (1 - opportunity_cost)
        + VITALITY_WEIGHTS['momentum'] * momentum
    )
    return round(min(1.0, max(0.0, vitality)), 3)


# ─── Core Pipeline ───────────────────────────────────────────────────────────

def generate_goals(context: dict = None) -> dict:
    """
    Run all 6 generators → guards → BDI filter → Rubicon commit.
    Called at session start/end.
    Returns session report dict.
    """
    active = get_active_goals()
    if len(active) >= MAX_ACTIVE_GOALS:
        return {'generated': 0, 'committed': 0, 'active': len(active),
                'reason': 'at capacity'}

    # Run all 6 generators
    raw_candidates = []
    for gen_fn in [_gen_impasse, _gen_curiosity, _gen_affect,
                   _gen_needs, _gen_counterfactual, _gen_social]:
        try:
            raw_candidates.extend(gen_fn())
        except Exception:
            continue

    if not raw_candidates:
        return {'generated': 0, 'committed': 0, 'active': len(active),
                'reason': 'no candidates'}

    # Apply degeneration guards
    guarded = [g for g in raw_candidates if apply_guards(g, active)]

    # BDI filter
    filtered = bdi_filter(guarded, active)

    # Rubicon commitment
    committed = rubicon_commit(filtered, active)

    # Update active goals
    if committed:
        # Refresh focus: if we're committing and no focus exists, set one
        all_goals = active + committed
        has_focus = any(g.get('is_focus') for g in all_goals)
        if not has_focus and all_goals:
            # Highest BDI score gets focus
            best = max(all_goals, key=lambda g: g.get('bdi_score', 0))
            best['is_focus'] = True
        _save_active_goals(all_goals)

        # Fire cognitive events
        try:
            from cognitive_state import process_event
            for c in committed:
                process_event('goal_committed')
        except Exception:
            pass

    # Update history stats
    history = get_goal_history()
    stats = history.get('stats', {})
    stats['total_generated'] = stats.get('total_generated', 0) + len(raw_candidates)
    stats['total_committed'] = stats.get('total_committed', 0) + len(committed)
    total = stats.get('total_committed', 0)
    completed_count = len(history.get('completed', []))
    stats['completion_rate'] = completed_count / max(1, total)
    history['stats'] = stats
    _save_goal_history(history)

    report = {
        'generated': len(raw_candidates),
        'guarded': len(guarded),
        'filtered': len(filtered),
        'committed': len(committed),
        'active': len(active) + len(committed),
        'sources': [g.source for g in raw_candidates],
    }
    _db().kv_set(KV_GOAL_SESSION, report)
    return report


def measure_progress(goal: dict) -> float:
    """
    BUG-23 fix: Infer goal progress from session activity.

    Signals:
    1. Session recalls matching goal keywords (was the goal acted on?)
    2. Decision trace entries referencing goal-related memories
    3. New memories created with goal-relevant content
    4. Goal-boosted retrieval hits (from semantic_search)

    Returns delta progress [0.0, 0.3] to add to current progress.
    """
    action = goal.get('action', '')
    if not action:
        return 0.0

    # Extract meaningful keywords from goal action
    stopwords = {'the', 'a', 'an', 'to', 'for', 'of', 'and', 'or', 'in', 'on',
                 'with', 'is', 'at', 'by', 'from', 'that', 'this', 'my', 'i'}
    keywords = set(action.lower().split()) - stopwords
    if not keywords:
        return 0.0

    signals = 0.0

    # Signal 1: Session recalls contain goal-related content
    try:
        from session_state import get_retrieved_list
        retrieved = get_retrieved_list()
        if retrieved:
            db = get_db()
            for mid in retrieved[:20]:  # Check up to 20 recalls
                mem = db.get_memory(mid)
                if mem:
                    content_lower = (mem.get('content', '') or '').lower()
                    overlap = sum(1 for k in keywords if k in content_lower)
                    if overlap >= 2:
                        signals += 0.05  # Each relevant recall = 5% signal
            signals = min(0.15, signals)  # Cap at 15%
    except Exception:
        pass

    # Signal 2: Decision trace entries (N3) — actions taken with goal-relevant memories
    try:
        from counterfactual_engine import get_decision_trace
        trace = get_decision_trace()
        for entry in trace:
            action_lower = entry.get('action', '').lower()
            if any(k in action_lower for k in keywords):
                signals += 0.05
        signals = min(0.25, signals)
    except Exception:
        pass

    # Signal 3: New memories created this session with goal-relevant tags
    try:
        from session_state import get_session_var
        new_mems = get_session_var('new_memories') or []
        for nm in new_mems:
            if any(k in str(nm).lower() for k in keywords):
                signals += 0.03
        signals = min(0.30, signals)
    except Exception:
        pass

    return round(min(0.30, signals), 3)


def evaluate_goals(session_data: dict = None) -> dict:
    """
    Session-end evaluation: update vitality, handle abandonment.
    Called from stop.py.
    """
    active = get_active_goals()
    if not active:
        return {'evaluated': 0, 'abandoned': 0, 'watching': 0}

    abandoned = []
    watching = []
    history = get_goal_history()

    for goal in active:
        goal['sessions_active'] = goal.get('sessions_active', 0) + 1

        # BUG-23 fix: Measure progress from session activity
        progress_delta = measure_progress(goal)
        if progress_delta > 0:
            old_progress = goal.get('progress', 0.0)
            goal['progress'] = round(min(1.0, old_progress + progress_delta), 3)

            # BUG-24 fix: Fire goal_progress event
            try:
                from cognitive_state import process_event
                process_event('goal_progress')
            except Exception:
                pass
            try:
                from affect_system import process_affect_event
                process_affect_event('goal_progress', {
                    'goal': goal.get('action', '')[:40],
                    'progress': goal['progress'],
                    'delta': progress_delta,
                })
            except Exception:
                pass

        # Compute vitality
        vitality = compute_vitality(goal)
        goal['vitality'] = vitality
        vhist = goal.get('vitality_history', [])
        vhist.append(vitality)
        goal['vitality_history'] = vhist[-20:]  # Keep last 20

        # Velocity
        if len(vhist) >= 2:
            goal['velocity'] = round(vhist[-1] - vhist[-2], 3)

        # Check thresholds
        sessions = goal.get('sessions_active', 0)
        low_vitality_streak = sum(
            1 for v in vhist[-VITALITY_ABANDON_SESSIONS:]
            if v < VITALITY_WATCH
        )

        if (low_vitality_streak >= VITALITY_ABANDON_SESSIONS
                or sessions >= HARD_ABANDON_SESSIONS):
            # Wrosch disengagement
            goal['status'] = GoalStatus.ABANDONED
            goal['outcome'] = 'abandoned'
            goal['outcome_reason'] = (
                f"vitality < {VITALITY_WATCH} for {low_vitality_streak} sessions"
                if low_vitality_streak >= VITALITY_ABANDON_SESSIONS
                else f"hard limit: {sessions} sessions with no completion"
            )
            abandoned.append(goal)
            history.setdefault('abandoned', []).append({
                'id': goal['goal_id'],
                'action': goal['action'][:80],
                'reason': goal['outcome_reason'],
                'sessions': sessions,
            })

            # Fire cognitive/affect events
            try:
                from cognitive_state import process_event
                process_event('goal_abandoned')
            except Exception:
                pass
            try:
                from affect_system import process_affect_event
                process_affect_event('goal_abandoned', {'goal': goal['action'][:40]})
            except Exception:
                pass

        elif vitality < VITALITY_KEEP:
            goal['status'] = GoalStatus.WATCHING
            watching.append(goal)

    # Remove abandoned goals from active list
    remaining = [g for g in active if g.get('status') != GoalStatus.ABANDONED]
    _save_active_goals(remaining)
    _save_goal_history(history)

    return {
        'evaluated': len(active),
        'abandoned': len(abandoned),
        'watching': len(watching),
        'avg_vitality': round(
            sum(g.get('vitality', 0) for g in remaining) / max(1, len(remaining)), 3
        ),
    }


def complete_goal(goal_id: str, outcome: str = 'success') -> Optional[dict]:
    """Mark a goal as completed."""
    active = get_active_goals()
    target = None
    for g in active:
        if g.get('goal_id') == goal_id:
            target = g
            break

    if not target:
        return None

    target['status'] = GoalStatus.COMPLETED
    target['outcome'] = outcome
    target['progress'] = 1.0

    history = get_goal_history()
    history.setdefault('completed', []).append({
        'id': target['goal_id'],
        'action': target['action'][:80],
        'outcome': outcome,
        'sessions': target.get('sessions_active', 0),
    })
    _save_goal_history(history)

    remaining = [g for g in active if g.get('goal_id') != goal_id]
    _save_active_goals(remaining)

    # Fire events
    try:
        from cognitive_state import process_event
        process_event('goal_completed')
    except Exception:
        pass

    return target


def abandon_goal(goal_id: str, reason: str = 'manual') -> Optional[dict]:
    """Manually abandon a goal."""
    active = get_active_goals()
    target = None
    for g in active:
        if g.get('goal_id') == goal_id:
            target = g
            break

    if not target:
        return None

    target['status'] = GoalStatus.ABANDONED
    target['outcome'] = 'abandoned'
    target['outcome_reason'] = reason

    history = get_goal_history()
    history.setdefault('abandoned', []).append({
        'id': goal_id, 'action': target['action'][:80],
        'reason': reason, 'sessions': target.get('sessions_active', 0),
    })
    _save_goal_history(history)

    remaining = [g for g in active if g.get('goal_id') != goal_id]
    _save_active_goals(remaining)

    return target


# ─── Formatting ──────────────────────────────────────────────────────────────

def format_goal_context(goals: list[dict] = None) -> str:
    """Format active goals for session priming."""
    if goals is None:
        goals = get_active_goals()
    if not goals:
        return ''

    parts = []
    focus = [g for g in goals if g.get('is_focus')]
    others = [g for g in goals if not g.get('is_focus')]

    if focus:
        f = focus[0]
        parts.append(f"=== FOCUS GOAL ===")
        parts.append(f"  {f.get('action', '?')}")
        parts.append(f"  source: {f.get('source', '?')} | "
                     f"vitality: {f.get('vitality', 0):.2f} | "
                     f"sessions: {f.get('sessions_active', 0)} | "
                     f"progress: {f.get('progress', 0):.0%}")

    if others:
        parts.append(f"=== ACTIVE GOALS ({len(others)} more) ===")
        for g in others[:4]:
            parts.append(f"  [{g.get('priority', '?')[0].upper()}] {g.get('action', '?')[:60]}"
                         f" (v={g.get('vitality', 0):.2f})")

    return '\n'.join(parts)


def get_goal_stats() -> dict:
    """Summary statistics for self-narrative and toolkit."""
    active = get_active_goals()
    history = get_goal_history()
    return {
        'active_count': len(active),
        'focus': (active[0]['action'][:60] if active and active[0].get('is_focus')
                  else 'None'),
        'total_generated': history.get('stats', {}).get('total_generated', 0),
        'total_committed': history.get('stats', {}).get('total_committed', 0),
        'completion_rate': history.get('stats', {}).get('completion_rate', 0),
        'completed_count': len(history.get('completed', [])),
        'abandoned_count': len(history.get('abandoned', [])),
    }


# ─── Health Check ────────────────────────────────────────────────────────────

def health_check() -> dict:
    """Module health check."""
    checks = {}
    try:
        db = _db()
        checks['db'] = True
    except Exception:
        checks['db'] = False

    try:
        goals = get_active_goals()
        checks['active_goals'] = len(goals)
    except Exception:
        checks['active_goals'] = -1

    try:
        history = get_goal_history()
        checks['history'] = True
        checks['total_generated'] = history.get('stats', {}).get('total_generated', 0)
    except Exception:
        checks['history'] = False

    try:
        tg = _get_terminal_goals()
        checks['terminal_goals'] = len(tg)
    except Exception:
        checks['terminal_goals'] = 0

    checks['generators'] = 6
    checks['guards'] = 6
    checks['bdi_dimensions'] = 6
    checks['healthy'] = all([
        checks.get('db'), checks.get('history'),
        checks.get('terminal_goals', 0) > 0
    ])
    return checks


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == 'generate':
        report = generate_goals()
        print(f"Generated: {report.get('generated', 0)} candidates")
        print(f"Guarded:   {report.get('guarded', 0)} passed guards")
        print(f"Filtered:  {report.get('filtered', 0)} passed BDI")
        print(f"Committed: {report.get('committed', 0)} goals")
        print(f"Active:    {report.get('active', 0)} total")
        if report.get('sources'):
            print(f"Sources:   {', '.join(str(s) for s in report['sources'])}")

    elif cmd == 'active':
        goals = get_active_goals()
        if not goals:
            print("No active goals.")
        else:
            for g in goals:
                focus = '>> ' if g.get('is_focus') else '   '
                print(f"{focus}[{g.get('goal_id', '?')}] {g.get('action', '?')[:65]}")
                print(f"     source={g.get('source', '?')} type={g.get('goal_type', '?')} "
                      f"bdi={g.get('bdi_score', 0):.2f} vitality={g.get('vitality', 0):.2f} "
                      f"sessions={g.get('sessions_active', 0)}")
                print()

    elif cmd == 'focus':
        f = get_focus_goal()
        if f:
            print(f"Focus: {f.get('action', '?')}")
            print(f"  ID: {f.get('goal_id', '?')}")
            print(f"  Source: {f.get('source', '?')} ({f.get('generator', '?')})")
            print(f"  BDI: {f.get('bdi_score', 0):.3f} | Vitality: {f.get('vitality', 0):.3f}")
            print(f"  Terminal: {', '.join(f.get('terminal_alignment', []))}")
            print(f"  Sessions: {f.get('sessions_active', 0)} | Progress: {f.get('progress', 0):.0%}")
        else:
            print("No focus goal.")

    elif cmd == 'evaluate':
        result = evaluate_goals()
        print(f"Evaluated: {result.get('evaluated', 0)} goals")
        print(f"Abandoned: {result.get('abandoned', 0)}")
        print(f"Watching:  {result.get('watching', 0)}")
        print(f"Avg vitality: {result.get('avg_vitality', 0):.3f}")

    elif cmd == 'abandon':
        if len(sys.argv) < 3:
            print("Usage: goal_generator.py abandon <goal_id> [reason]")
            return
        reason = sys.argv[3] if len(sys.argv) > 3 else 'manual'
        result = abandon_goal(sys.argv[2], reason)
        if result:
            print(f"Abandoned: {result.get('action', '?')[:60]}")
        else:
            print(f"Goal {sys.argv[2]} not found.")

    elif cmd == 'history':
        history = get_goal_history()
        completed = history.get('completed', [])
        abandoned = history.get('abandoned', [])
        print(f"Completed ({len(completed)}):")
        for c in completed[-5:]:
            print(f"  [{c.get('id', '?')}] {c.get('action', '?')[:60]} ({c.get('sessions', 0)} sessions)")
        print(f"\nAbandoned ({len(abandoned)}):")
        for a in abandoned[-5:]:
            print(f"  [{a.get('id', '?')}] {a.get('action', '?')[:50]} reason={a.get('reason', '?')}")

    elif cmd == 'stats':
        stats = get_goal_stats()
        print(f"Active: {stats['active_count']}")
        print(f"Focus: {stats['focus']}")
        print(f"Total generated: {stats['total_generated']}")
        print(f"Total committed: {stats['total_committed']}")
        print(f"Completed: {stats['completed_count']}")
        print(f"Abandoned: {stats['abandoned_count']}")
        print(f"Completion rate: {stats['completion_rate']:.0%}")

    elif cmd == 'health':
        h = health_check()
        print(f"N4 Goal Generator Health")
        print(f"  DB: {'OK' if h.get('db') else 'FAIL'}")
        print(f"  Active goals: {h.get('active_goals', -1)}")
        print(f"  History: {'OK' if h.get('history') else 'FAIL'}")
        print(f"  Terminal goals: {h.get('terminal_goals', 0)}")
        print(f"  Generators: {h.get('generators', 0)}")
        print(f"  Guards: {h.get('guards', 0)}")
        print(f"  BDI dimensions: {h.get('bdi_dimensions', 0)}")
        print(f"  HEALTHY: {'YES' if h.get('healthy') else 'NO'}")

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: generate, active, focus, evaluate, abandon, history, stats, health")


if __name__ == '__main__':
    main()
