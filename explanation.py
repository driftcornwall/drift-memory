#!/usr/bin/env python3
"""
Explainability Interface — Auditable reasoning chains for memory operations.

Every search, priming, decay, and consolidation decision can be explained
with structured reasoning: what factors contributed, their weights, and
why the final decision was made.

Usage:
    from explanation import ExplanationBuilder, get_store

    # During a search operation:
    builder = ExplanationBuilder('semantic_search', 'search')
    builder.set_inputs({'query': 'what do I know about trust?', 'limit': 5})
    builder.add_step('cosine_similarity', 0.82, weight=1.0, context='raw embedding match')
    builder.add_step('resolution_boost', 1.25, weight=0.25, context='memory has "api" tag')
    builder.add_step('gravity_dampening', 0.5, weight=-0.5, context='no key terms in preview')
    builder.set_output({'id': 'abc123', 'final_score': 0.51})
    builder.save()

CLI:
    python explanation.py last [N]          # Show last N explanations
    python explanation.py search [query]    # Show explanations for search operations
    python explanation.py priming           # Show last priming explanation
    python explanation.py why <memory_id>   # Why was this memory selected/rejected?
    python explanation.py stats             # Coverage and depth statistics
"""

import json
import sys
from datetime import datetime, timezone
from typing import Optional


# Session ID for grouping explanations within a session
_session_id = None


def get_session_id() -> str:
    """Get or create a session ID for grouping explanations."""
    global _session_id
    if _session_id is None:
        _session_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    return _session_id


def set_session_id(sid: str):
    """Set session ID (called by session_start hook)."""
    global _session_id
    _session_id = sid


class ExplanationBuilder:
    """
    Build a structured explanation for a single operation.

    Each explanation captures:
    - module: which module performed the operation (e.g., 'semantic_search')
    - operation: what was done (e.g., 'search', 'priming', 'decay')
    - inputs: what went in (query, parameters, memory IDs)
    - reasoning: ordered list of steps, each with factor/value/weight/context
    - output: what came out (selected memories, scores, decisions)
    """

    def __init__(self, module: str, operation: str):
        self.module = module
        self.operation = operation
        self.inputs = {}
        self.reasoning = []
        self.output = {}
        self.timestamp = datetime.now(timezone.utc)

    def set_inputs(self, inputs: dict) -> 'ExplanationBuilder':
        """Set the operation inputs."""
        self.inputs = inputs
        return self

    def add_step(self, factor: str, value, weight: float = 1.0,
                 context: str = '') -> 'ExplanationBuilder':
        """
        Add a reasoning step.

        Args:
            factor: What was evaluated (e.g., 'cosine_similarity', 'entity_boost')
            value: The raw value of this factor
            weight: How much this factor contributed (negative = penalty)
            context: Human-readable explanation of why this matters
        """
        self.reasoning.append({
            'factor': factor,
            'value': _serialize(value),
            'weight': weight,
            'context': context,
        })
        return self

    def set_output(self, output: dict) -> 'ExplanationBuilder':
        """Set the operation output."""
        self.output = output
        return self

    def save(self) -> Optional[int]:
        """Persist this explanation to the database. Returns the row ID."""
        try:
            return get_store().save(self)
        except Exception:
            return None

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            'module': self.module,
            'operation': self.operation,
            'inputs': self.inputs,
            'reasoning': self.reasoning,
            'output': self.output,
            'timestamp': self.timestamp.isoformat(),
            'session_id': get_session_id(),
        }


class ExplanationStore:
    """Persist and query explanations via PostgreSQL."""

    def __init__(self):
        self._db = None

    def _get_db(self):
        if self._db is None:
            from db_adapter import get_db
            self._db = get_db()
        return self._db

    def save(self, builder: ExplanationBuilder) -> Optional[int]:
        """Save an explanation to the DB. Returns the row ID."""
        import psycopg2.extras
        db = self._get_db()
        with db._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {db._table('explanations')}
                    (session_id, module, operation, inputs, output, reasoning, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    get_session_id(),
                    builder.module,
                    builder.operation,
                    psycopg2.extras.Json(builder.inputs),
                    psycopg2.extras.Json(builder.output),
                    psycopg2.extras.Json(builder.reasoning),
                    builder.timestamp,
                ))
                row = cur.fetchone()
                return row[0] if row else None

    def get_last(self, n: int = 5, module: str = None,
                 operation: str = None) -> list[dict]:
        """Get the last N explanations, optionally filtered."""
        import psycopg2.extras
        db = self._get_db()

        where_clauses = []
        params = []
        if module:
            where_clauses.append("module = %s")
            params.append(module)
        if operation:
            where_clauses.append("operation = %s")
            params.append(operation)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        params.append(n)

        with db._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT * FROM {db._table('explanations')}
                    {where_sql}
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, params)
                return [dict(r) for r in cur.fetchall()]

    def explain_memory(self, memory_id: str, n: int = 10) -> list[dict]:
        """Find explanations that mention a specific memory ID."""
        import psycopg2.extras
        db = self._get_db()
        with db._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Search in both inputs and output JSONB for the memory ID
                cur.execute(f"""
                    SELECT * FROM {db._table('explanations')}
                    WHERE inputs::text ILIKE %s
                       OR output::text ILIKE %s
                       OR reasoning::text ILIKE %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (
                    f'%{memory_id}%',
                    f'%{memory_id}%',
                    f'%{memory_id}%',
                    n,
                ))
                return [dict(r) for r in cur.fetchall()]

    def get_stats(self) -> dict:
        """Get explanation statistics."""
        import psycopg2.extras
        db = self._get_db()
        with db._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT
                        COUNT(*) as total_explanations,
                        COUNT(DISTINCT module) as modules_covered,
                        COUNT(DISTINCT operation) as operations_covered,
                        COUNT(DISTINCT session_id) as sessions_with_explanations,
                        AVG(jsonb_array_length(reasoning)) as avg_reasoning_depth,
                        MAX(timestamp) as last_explanation
                    FROM {db._table('explanations')}
                """)
                row = cur.fetchone()
                if row:
                    return {
                        'total_explanations': row['total_explanations'],
                        'modules_covered': row['modules_covered'],
                        'operations_covered': row['operations_covered'],
                        'sessions_with_explanations': row['sessions_with_explanations'],
                        'avg_reasoning_depth': round(float(row['avg_reasoning_depth'] or 0), 1),
                        'last_explanation': row['last_explanation'].isoformat() if row['last_explanation'] else None,
                    }
                return {'total_explanations': 0}

    def get_session_explanations(self, session_id: str = None) -> list[dict]:
        """Get all explanations for a session."""
        import psycopg2.extras
        db = self._get_db()
        sid = session_id or get_session_id()
        with db._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT * FROM {db._table('explanations')}
                    WHERE session_id = %s
                    ORDER BY timestamp ASC
                """, (sid,))
                return [dict(r) for r in cur.fetchall()]


# Module-level singleton
_store = None


def get_store() -> ExplanationStore:
    """Get the singleton ExplanationStore."""
    global _store
    if _store is None:
        _store = ExplanationStore()
    return _store


def _serialize(value) -> str:
    """Serialize a value for storage in reasoning steps."""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, default=str)
    return str(value)


# --- Formatting ---

def format_explanation(exp: dict, verbose: bool = False) -> str:
    """Format a single explanation for display."""
    lines = []
    ts = exp.get('timestamp', '?')
    if hasattr(ts, 'isoformat'):
        ts = ts.isoformat()

    lines.append(f"[{str(ts)[:19]}] {exp['module']}.{exp['operation']}")

    inputs = exp.get('inputs', {})
    if inputs:
        # Show key inputs concisely
        input_parts = []
        for k, v in inputs.items():
            sv = str(v)[:60]
            input_parts.append(f"{k}={sv}")
        lines.append(f"  Inputs: {', '.join(input_parts[:4])}")

    reasoning = exp.get('reasoning', [])
    if reasoning:
        lines.append(f"  Reasoning ({len(reasoning)} steps):")
        display_steps = reasoning if verbose else reasoning[:6]
        for step in display_steps:
            weight_sign = '+' if step.get('weight', 0) >= 0 else ''
            context = f" — {step['context']}" if step.get('context') else ''
            lines.append(
                f"    {step['factor']}: {step['value']} "
                f"(w={weight_sign}{step.get('weight', 1.0):.2f}){context}"
            )
        if not verbose and len(reasoning) > 6:
            lines.append(f"    ... +{len(reasoning) - 6} more steps")

    output = exp.get('output', {})
    if output:
        out_parts = []
        for k, v in output.items():
            sv = str(v)[:50]
            out_parts.append(f"{k}={sv}")
        lines.append(f"  Output: {', '.join(out_parts[:4])}")

    return '\n'.join(lines)


def format_stats(stats: dict) -> str:
    """Format explanation statistics."""
    lines = [
        "Explainability Statistics",
        "=" * 40,
        f"  Total explanations: {stats.get('total_explanations', 0)}",
        f"  Modules covered: {stats.get('modules_covered', 0)}",
        f"  Operations covered: {stats.get('operations_covered', 0)}",
        f"  Sessions with explanations: {stats.get('sessions_with_explanations', 0)}",
        f"  Avg reasoning depth: {stats.get('avg_reasoning_depth', 0)} steps",
        f"  Last explanation: {stats.get('last_explanation', 'never')}",
    ]
    return '\n'.join(lines)


# --- CLI ---

def main():
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    args = sys.argv[1:]
    if not args or args[0] in ('--help', '-h'):
        print(__doc__.strip())
        return

    cmd = args[0]
    store = get_store()

    if cmd == 'last':
        n = int(args[1]) if len(args) > 1 else 5
        explanations = store.get_last(n)
        if not explanations:
            print("No explanations recorded yet.")
            return
        for exp in reversed(explanations):
            print(format_explanation(exp))
            print()

    elif cmd == 'search':
        explanations = store.get_last(10, module='semantic_search', operation='search')
        if not explanations:
            print("No search explanations found.")
            return
        for exp in reversed(explanations):
            print(format_explanation(exp))
            print()

    elif cmd == 'priming':
        explanations = store.get_last(1, module='memory_manager', operation='priming')
        if not explanations:
            print("No priming explanations found.")
            return
        print(format_explanation(explanations[0], verbose=True))

    elif cmd == 'why' and len(args) > 1:
        memory_id = args[1]
        explanations = store.explain_memory(memory_id)
        if not explanations:
            print(f"No explanations found mentioning {memory_id}.")
            return
        print(f"Explanations involving {memory_id}:\n")
        for exp in explanations:
            print(format_explanation(exp, verbose=True))
            print()

    elif cmd == 'stats':
        stats = store.get_stats()
        print(format_stats(stats))

    elif cmd == 'session':
        sid = args[1] if len(args) > 1 else None
        explanations = store.get_session_explanations(sid)
        if not explanations:
            print(f"No explanations for session {sid or 'current'}.")
            return
        print(f"Session explanations ({len(explanations)}):\n")
        for exp in explanations:
            print(format_explanation(exp))
            print()

    else:
        print(f"Unknown command: {cmd}")
        print("Available: last [N], search, priming, why <memory_id>, stats, session")
        sys.exit(1)


if __name__ == '__main__':
    main()
