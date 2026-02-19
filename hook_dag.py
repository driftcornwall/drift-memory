#!/usr/bin/env python3
"""
Hook Task DAG — T1.3 Error Isolation for Session Hooks

Provides a lightweight DAG executor for session hook tasks (stop.py, session_start.py).
Tasks declare dependencies; the executor topologically sorts them, runs independent
tasks in parallel, and propagates failure status to dependents.

Key behaviors:
- Independent tasks run concurrently via ThreadPoolExecutor
- If a task fails, all tasks that depend on it are SKIPPED (not run with stale data)
- Each task returns (rc, stdout, stderr); rc != 0 = failure
- Skipped tasks get a "degraded" marker visible in the output log
- Timing is tracked per-task for performance monitoring

Usage:
    from hook_dag import DAGExecutor, Task

    dag = DAGExecutor(debug=True)
    dag.add(Task("save_pending", fn, args=(...,)))
    dag.add(Task("homeostasis", fn, args=(...,), depends_on=["save_pending"]))
    dag.add(Task("kg_enrichment", fn, args=(...,)))  # independent
    results = dag.run(max_workers=3)

Each result is a TaskResult with: name, status, rc, stdout, stderr, elapsed_ms, skipped_reason
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Callable, Optional, Any


@dataclass
class Task:
    """A task node in the DAG."""
    name: str
    fn: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    depends_on: list = field(default_factory=list)
    timeout_s: float = 30.0
    critical: bool = False  # If True, DAG aborts on failure


@dataclass
class TaskResult:
    """Result of executing a task."""
    name: str
    status: str  # 'ok', 'failed', 'skipped', 'timeout', 'error'
    rc: int = 0
    stdout: str = ''
    stderr: str = ''
    elapsed_ms: float = 0.0
    skipped_reason: str = ''

    @property
    def success(self) -> bool:
        return self.status == 'ok'


class DAGExecutor:
    """Execute tasks respecting dependency ordering with parallel independent tasks."""

    def __init__(self, debug: bool = False, max_workers: int = 4):
        self._tasks: dict[str, Task] = {}
        self._debug = debug
        self._max_workers = max_workers

    def add(self, task: Task) -> 'DAGExecutor':
        """Add a task to the DAG. Returns self for chaining."""
        self._tasks[task.name] = task
        return self

    def _topo_levels(self) -> list[list[str]]:
        """Topological sort into execution levels (Kahn's algorithm).

        Returns list of levels, where each level contains tasks that can
        run in parallel (all dependencies satisfied by previous levels).
        """
        # Build adjacency and in-degree
        in_degree = {name: 0 for name in self._tasks}
        dependents = {name: [] for name in self._tasks}

        for name, task in self._tasks.items():
            for dep in task.depends_on:
                if dep in self._tasks:
                    in_degree[name] += 1
                    dependents[dep].append(name)

        # Kahn's algorithm — group by levels
        levels = []
        queue = [n for n, d in in_degree.items() if d == 0]

        while queue:
            levels.append(sorted(queue))  # Sort for deterministic ordering
            next_queue = []
            for n in queue:
                for dep in dependents[n]:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        next_queue.append(dep)
            queue = next_queue

        # Check for cycles
        scheduled = sum(len(level) for level in levels)
        if scheduled < len(self._tasks):
            missing = set(self._tasks.keys()) - {n for level in levels for n in level}
            raise ValueError(f"Circular dependency detected in tasks: {missing}")

        return levels

    def _run_task(self, task: Task) -> TaskResult:
        """Execute a single task, capturing its result."""
        start = time.monotonic()
        try:
            result = task.fn(*task.args, **task.kwargs)

            elapsed = (time.monotonic() - start) * 1000

            if result is None:
                return TaskResult(task.name, 'ok', 0, '', '', elapsed)

            if isinstance(result, tuple) and len(result) == 3:
                rc, stdout, stderr = result
                status = 'ok' if rc == 0 else 'failed'
                return TaskResult(task.name, status, rc, str(stdout), str(stderr), elapsed)

            # Non-standard return — treat as OK
            return TaskResult(task.name, 'ok', 0, str(result), '', elapsed)

        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return TaskResult(task.name, 'error', -1, '', str(e), elapsed)

    def run(self) -> list[TaskResult]:
        """Execute all tasks respecting dependencies.

        Returns list of TaskResult in execution order.
        """
        levels = self._topo_levels()
        results: dict[str, TaskResult] = {}
        all_results: list[TaskResult] = []
        failed_tasks: set[str] = set()

        for level in levels:
            # Filter out tasks whose dependencies failed
            runnable = []
            for name in level:
                task = self._tasks[name]
                failed_deps = [d for d in task.depends_on if d in failed_tasks]
                if failed_deps:
                    # Skip — dependency failed
                    reason = f"dependency failed: {', '.join(failed_deps)}"
                    skip_result = TaskResult(
                        name, 'skipped', -2, '', '', 0.0, reason
                    )
                    results[name] = skip_result
                    all_results.append(skip_result)
                    failed_tasks.add(name)
                    if self._debug:
                        print(f"  SKIP {name}: {reason}", file=sys.stderr)
                else:
                    runnable.append(name)

            if not runnable:
                continue

            # Run this level's tasks in parallel
            if len(runnable) == 1:
                # Single task — no thread overhead
                task = self._tasks[runnable[0]]
                result = self._run_task(task)
                results[task.name] = result
                all_results.append(result)
                if not result.success:
                    failed_tasks.add(task.name)
                if self._debug:
                    status_icon = 'OK' if result.success else 'FAIL'
                    print(f"  {status_icon} {task.name} ({result.elapsed_ms:.0f}ms) {result.stdout[:100] if result.success else result.stderr[:100]}", file=sys.stderr)
            else:
                # Multiple tasks — run in parallel
                workers = min(self._max_workers, len(runnable))
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures: dict[Future, str] = {}
                    for name in runnable:
                        task = self._tasks[name]
                        future = pool.submit(self._run_task, task)
                        futures[future] = name

                    for future in as_completed(futures):
                        name = futures[future]
                        try:
                            result = future.result(timeout=self._tasks[name].timeout_s)
                        except Exception as e:
                            result = TaskResult(name, 'timeout', -1, '', str(e), 0.0)

                        results[name] = result
                        all_results.append(result)
                        if not result.success:
                            failed_tasks.add(name)
                        if self._debug:
                            status_icon = 'OK' if result.success else 'FAIL'
                            print(f"  {status_icon} {name} ({result.elapsed_ms:.0f}ms) {result.stdout[:100] if result.success else result.stderr[:100]}", file=sys.stderr)

            # Check for critical task failure
            for name in runnable:
                task = self._tasks[name]
                if task.critical and name in failed_tasks:
                    if self._debug:
                        print(f"  ABORT: Critical task '{name}' failed", file=sys.stderr)
                    return all_results

        return all_results

    def summary(self, results: list[TaskResult]) -> dict:
        """Generate a summary dict from results."""
        ok = [r for r in results if r.status == 'ok']
        failed = [r for r in results if r.status == 'failed']
        skipped = [r for r in results if r.status == 'skipped']
        errors = [r for r in results if r.status in ('error', 'timeout')]
        total_ms = sum(r.elapsed_ms for r in results)

        return {
            'total': len(results),
            'ok': len(ok),
            'failed': len(failed),
            'skipped': len(skipped),
            'errors': len(errors),
            'total_ms': round(total_ms, 1),
            'degraded': [r.name for r in results if r.status in ('failed', 'skipped', 'error')],
            'detail': {r.name: {
                'status': r.status,
                'rc': r.rc,
                'ms': round(r.elapsed_ms, 1),
                'msg': r.stdout[:100] if r.success else (r.skipped_reason or r.stderr[:100]),
            } for r in results},
        }
