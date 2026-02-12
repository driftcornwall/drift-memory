#!/usr/bin/env python3
"""
Q-Value Learning Engine for Memory Retrieval (MemRL-inspired)

Gives each memory a learned utility score that converges toward its true
expected retrieval value. Memories that consistently produce useful output
get higher Q-values; memories recalled but never contributing get lower.

Based on: MemRL (arXiv:2601.03192), adapted from SpindriftMend's implementation.
Ported to db_adapter pattern for Drift's PostgreSQL-only architecture.

Core update rule: Q <- Q + alpha(r - Q)
Composite score:  lambda * similarity + (1-lambda) * Q-value

Dynamic LAMBDA: cognitive state modifies the similarity/Q balance.
  High confidence -> exploit proven memories (lower LAMBDA, more Q weight)
  High curiosity  -> explore new matches (higher LAMBDA, more similarity weight)

Usage:
    python q_value_engine.py stats          # Q-value distribution
    python q_value_engine.py top [N]        # Top N by Q-value
    python q_value_engine.py bottom [N]     # Bottom N by Q-value
    python q_value_engine.py history <id>   # Q trajectory for memory
    python q_value_engine.py convergence    # Convergence report
"""

import sys
from db_adapter import get_db

# ---------------------------------------------------------------------------
# Core Parameters (from MemRL paper, tuned for our scale)
# ---------------------------------------------------------------------------

ALPHA = 0.1              # Learning rate
BASE_LAMBDA = 0.5        # Base similarity-utility balance
DEFAULT_Q = 0.5          # Optimistic initialization
Q_MIN, Q_MAX = 0.0, 1.0  # Clamp bounds

# Feature flags
Q_RERANKING_ENABLED = True
Q_UPDATES_ENABLED = True

# Reward signals
REWARD_RE_RECALL = 1.0        # Recalled by 2+ different sources
REWARD_DOWNSTREAM = 0.8       # Led to new memory creation
REWARD_DEAD_END = -0.3        # Recalled but unused

# Reward weights for composite
WEIGHT_RE_RECALL = 0.4
WEIGHT_DOWNSTREAM = 0.4
WEIGHT_EXPLICIT = 0.2


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def update_q(old_q: float, reward: float) -> float:
    """Q <- Q + alpha(r - Q). Converges toward the reward signal."""
    return clamp(old_q + ALPHA * (reward - old_q), Q_MIN, Q_MAX)


def get_lambda() -> float:
    """
    Dynamic LAMBDA based on cognitive state.
    High confidence -> lower LAMBDA (trust Q-values more, exploit)
    High curiosity  -> higher LAMBDA (trust similarity more, explore)
    """
    try:
        from cognitive_state import get_state
        state = get_state()
        # Shift LAMBDA by up to ±0.15 based on cognitive state
        confidence_shift = -0.15 * state.confidence  # More confidence = less LAMBDA
        curiosity_shift = 0.15 * state.curiosity      # More curiosity = more LAMBDA
        return clamp(BASE_LAMBDA + confidence_shift + curiosity_shift, 0.2, 0.8)
    except Exception:
        return BASE_LAMBDA


def composite_score(similarity: float, q_value: float, lam: float = None) -> float:
    """Two-phase retrieval: lambda*sim + (1-lambda)*Q."""
    if lam is None:
        lam = get_lambda()
    return lam * similarity + (1 - lam) * q_value


def get_q_value(memory_id: str) -> float:
    """Get Q-value for a single memory."""
    db = get_db()
    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT q_value FROM {db._table('memories')} WHERE id = %s",
                (memory_id,)
            )
            row = cur.fetchone()
            return float(row[0]) if row and row[0] is not None else DEFAULT_Q


def get_q_values(memory_ids: list) -> dict:
    """Batch-fetch Q-values for multiple memories."""
    if not memory_ids:
        return {}
    db = get_db()
    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, q_value FROM {db._table('memories')} WHERE id = ANY(%s)",
                (list(memory_ids),)
            )
            return {row[0]: float(row[1]) if row[1] is not None else DEFAULT_Q
                    for row in cur.fetchall()}


def compute_reward(memory_id: str, session_recalls: dict,
                   created_this_session: set) -> tuple:
    """
    Compute composite reward from session evidence.

    Returns:
        (reward, source_label) tuple
    """
    signals = []

    # Signal 1: Re-recall — recalled by 2+ different sources
    sources_that_recalled = 0
    for source, ids in session_recalls.items():
        if memory_id in ids:
            sources_that_recalled += 1
    if sources_that_recalled >= 2:
        signals.append((REWARD_RE_RECALL, WEIGHT_RE_RECALL, "re_recall"))

    # Signal 2: Downstream — appears in caused_by of a new memory
    db = get_db()
    if created_this_session:
        import psycopg2.extras
        with db._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                for new_id in created_this_session:
                    cur.execute(
                        f"SELECT extra_metadata FROM {db._table('memories')} WHERE id = %s",
                        (new_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        extra = row.get('extra_metadata') or {}
                        caused_by = extra.get('caused_by', [])
                        if memory_id in caused_by:
                            signals.append((REWARD_DOWNSTREAM, WEIGHT_DOWNSTREAM, "downstream"))
                            break

    # Compute weighted reward
    if signals:
        total_weight = sum(w for _, w, _ in signals)
        weighted_reward = sum(r * w for r, w, _ in signals) / total_weight
        source = "+".join(s for _, _, s in signals)
        return (weighted_reward, source)

    # No evidence of utility — dead end
    return (REWARD_DEAD_END, "dead_end")


# ---------------------------------------------------------------------------
# Session-End Batch Update
# ---------------------------------------------------------------------------

def session_end_q_update(session_id: int = None) -> dict:
    """
    Main entry point: compute rewards and batch-update all recalled memories.
    Called from stop.py hook at session end.
    """
    if not Q_UPDATES_ENABLED:
        return {"updated": 0, "skipped": True, "reason": "disabled"}

    import session_state
    session_state.load()

    db = get_db()

    # Get session data
    retrieved = session_state.get_retrieved_list()
    recalls_by_source = session_state.get_recalls_by_source()

    if not retrieved:
        return {"updated": 0, "reason": "no_recalls"}

    # Get memories created this session (from DB — most reliable)
    created_this_session = set()
    try:
        import psycopg2.extras
        with db._conn() as conn:
            with conn.cursor() as cur:
                # Find memories created in the last 4 hours (session window)
                cur.execute(f"""
                    SELECT id FROM {db._table('memories')}
                    WHERE created > NOW() - INTERVAL '4 hours'
                    AND type IN ('active', 'core')
                """)
                created_this_session = {row[0] for row in cur.fetchall()}
    except Exception:
        pass

    # Batch-fetch current Q-values
    q_vals = get_q_values(retrieved)

    results = {
        "updated": 0,
        "total_reward": 0.0,
        "avg_reward": 0.0,
        "by_source": {},
        "updates": [],
    }

    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor() as cur:
            for mem_id in retrieved:
                old_q = q_vals.get(mem_id, DEFAULT_Q)
                reward, source = compute_reward(
                    mem_id, recalls_by_source, created_this_session
                )
                new_q = update_q(old_q, reward)

                # Update memory's Q-value
                cur.execute(
                    f"UPDATE {db._table('memories')} SET q_value = %s WHERE id = %s",
                    (new_q, mem_id)
                )

                # Log to history
                cur.execute(f"""
                    INSERT INTO {db._table('q_value_history')}
                    (memory_id, session_id, old_q, new_q, reward, reward_source)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (mem_id, session_id or 0, old_q, new_q, reward, source))

                results["updated"] += 1
                results["total_reward"] += reward
                results["by_source"][source] = results["by_source"].get(source, 0) + 1
                results["updates"].append({
                    "id": mem_id,
                    "old_q": round(old_q, 4),
                    "new_q": round(new_q, 4),
                    "reward": round(reward, 4),
                    "source": source,
                })

        conn.commit()

    if results["updated"] > 0:
        results["avg_reward"] = round(
            results["total_reward"] / results["updated"], 4
        )

    # Fire cognitive state event based on average reward
    try:
        from cognitive_state import process_event
        if results["avg_reward"] > 0.3:
            process_event('memory_stored')  # Positive session
        elif results["avg_reward"] < -0.1:
            process_event('search_failure')  # Lots of dead ends
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------------

def q_stats() -> dict:
    """Q-value distribution summary."""
    db = get_db()
    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE q_value != 0.5) as trained,
                    COUNT(*) FILTER (WHERE q_value >= 0.7) as high,
                    COUNT(*) FILTER (WHERE q_value <= 0.3) as low,
                    AVG(q_value) as avg_q,
                    MIN(q_value) as min_q,
                    MAX(q_value) as max_q
                FROM {db._table('memories')}
                WHERE type IN ('active', 'core')
            """)
            row = cur.fetchone()
            return {
                "total": row[0],
                "trained": row[1],
                "high_q": row[2],
                "low_q": row[3],
                "avg_q": round(float(row[4] or 0.5), 4),
                "min_q": round(float(row[5] or 0.0), 4),
                "max_q": round(float(row[6] or 1.0), 4),
            }


def q_top(n: int = 10) -> list:
    """Top N memories by Q-value."""
    db = get_db()
    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, q_value, type, recall_count,
                       LEFT(content, 100) as preview
                FROM {db._table('memories')}
                WHERE type IN ('active', 'core')
                ORDER BY q_value DESC
                LIMIT %s
            """, (n,))
            return [dict(r) for r in cur.fetchall()]


def q_bottom(n: int = 10) -> list:
    """Bottom N by Q-value (excluding untrained)."""
    db = get_db()
    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, q_value, type, recall_count,
                       LEFT(content, 100) as preview
                FROM {db._table('memories')}
                WHERE type IN ('active', 'core')
                  AND q_value != 0.5
                ORDER BY q_value ASC
                LIMIT %s
            """, (n,))
            return [dict(r) for r in cur.fetchall()]


def q_history(memory_id: str) -> list:
    """Q-value trajectory for a specific memory."""
    db = get_db()
    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT session_id, old_q, new_q, reward, reward_source, updated_at
                FROM {db._table('q_value_history')}
                WHERE memory_id = %s
                ORDER BY updated_at ASC
            """, (memory_id,))
            return [dict(r) for r in cur.fetchall()]


def convergence_report() -> dict:
    """How well has the Q-value system converged?"""
    db = get_db()
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) FROM {db._table('memories')}
                WHERE type IN ('active', 'core') AND q_value != 0.5
            """)
            trained = cur.fetchone()[0]

            cur.execute(f"""
                SELECT COUNT(*) FROM {db._table('memories')}
                WHERE type IN ('active', 'core')
            """)
            total = cur.fetchone()[0]

            # Average update magnitude (last 50 updates)
            cur.execute(f"""
                SELECT AVG(ABS(new_q - old_q))
                FROM (
                    SELECT new_q, old_q
                    FROM {db._table('q_value_history')}
                    ORDER BY updated_at DESC
                    LIMIT 50
                ) recent
            """)
            avg_delta = cur.fetchone()[0]

    return {
        "total_active_core": total,
        "trained_count": trained,
        "trained_pct": round(trained * 100 / total, 1) if total > 0 else 0,
        "avg_update_magnitude": round(float(avg_delta or 0), 4),
        "converging": (avg_delta or 1) < 0.05 if avg_delta is not None else False,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    args = sys.argv[1:]
    if not args or args[0] in ('--help', '-h'):
        print(__doc__.strip())
        return

    cmd = args[0]

    if cmd == 'stats':
        stats = q_stats()
        print("Q-Value Distribution")
        print("=" * 40)
        print(f"  Total (active+core): {stats.get('total', 0)}")
        print(f"  Trained (Q != 0.5):  {stats.get('trained', 0)}")
        print(f"  High Q (>= 0.7):    {stats.get('high_q', 0)}")
        print(f"  Low Q (<= 0.3):     {stats.get('low_q', 0)}")
        print(f"  Avg Q: {stats.get('avg_q', 0.5)}")
        print(f"  Range: [{stats.get('min_q', 0)}, {stats.get('max_q', 1)}]")
        lam = get_lambda()
        print(f"\n  Dynamic LAMBDA: {lam:.3f} (base={BASE_LAMBDA})")

    elif cmd == 'top':
        n = int(args[1]) if len(args) > 1 else 10
        results = q_top(n)
        print(f"Top {n} memories by Q-value:")
        for r in results:
            print(f"  [Q:{r['q_value']:.3f}] {r['id']} (recalls={r['recall_count']}) {r['preview'][:60]}...")

    elif cmd == 'bottom':
        n = int(args[1]) if len(args) > 1 else 10
        results = q_bottom(n)
        if not results:
            print("No memories with non-default Q-values yet.")
        else:
            print(f"Bottom {n} memories by Q-value:")
            for r in results:
                print(f"  [Q:{r['q_value']:.3f}] {r['id']} (recalls={r['recall_count']}) {r['preview'][:60]}...")

    elif cmd == 'history':
        if len(args) < 2:
            print("Usage: python q_value_engine.py history <memory_id>")
            sys.exit(1)
        mem_id = args[1]
        history = q_history(mem_id)
        if not history:
            print(f"No Q-value history for {mem_id}")
        else:
            print(f"Q-value trajectory for {mem_id}:")
            for h in history:
                ts = str(h['updated_at'])[:19]
                print(f"  [{ts}] {h['old_q']:.3f} -> {h['new_q']:.3f} (r={h['reward']:.3f}, {h['reward_source']})")

    elif cmd == 'convergence':
        report = convergence_report()
        print("Q-Value Convergence Report")
        print("=" * 40)
        print(f"  Active+Core: {report.get('total_active_core', 0)}")
        print(f"  Trained:     {report.get('trained_count', 0)} ({report.get('trained_pct', 0)}%)")
        print(f"  Avg |delta|: {report.get('avg_update_magnitude', 0)}")
        print(f"  Converging:  {'Yes' if report.get('converging') else 'Not yet'}")

    else:
        print(f"Unknown command: {cmd}")
        print("Available: stats, top, bottom, history, convergence")
        sys.exit(1)


if __name__ == '__main__':
    main()
