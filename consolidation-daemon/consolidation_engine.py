"""
Consolidation Engine — The brain of the memory daemon.

Replaces stop.py's subprocess-heavy pipeline with all-in-process execution.
Modules are imported once at startup and called directly, sharing DB connections.

Schema-aware: routes to drift or spin memory directories.
Incremental: uses IncrementalMerkle and IncrementalFingerprint instead of O(n) recomputation.

Design: each phase returns a result dict. Failures in one phase don't block others.
"""

import importlib
import json
import logging
import os
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from incremental_merkle import IncrementalMerkle
from incremental_fingerprint import IncrementalFingerprint

logger = logging.getLogger("consolidation-engine")

# Memory directories mounted in Docker
MEMORY_DIRS = {
    "drift": Path("/app/drift-memory"),
    "spin": Path("/app/spin-memory"),
}

# Shared database module
DB_PATH = Path("/app/database")

# Modules that the engine imports from each agent's memory directory
CONSOLIDATION_MODULES = [
    "co_occurrence",
    "transcript_processor",
    "auto_memory_hook",
    "decay_evolution",
    "q_value_engine",
    "context_manager",
    "lesson_extractor",
    "cognitive_state",
    "system_vitals",
    "rejection_log",
    "auto_rejection_logger",
    "session_state",
    "gemma_bridge",
    "cognitive_fingerprint",
    "merkle_attestation",
    "db_adapter",
]


class SchemaEngine:
    """Consolidation engine for a single schema (drift or spin).

    Holds imported modules, DB connection, and incremental state.
    """

    def __init__(self, schema: str):
        self.schema = schema
        self.memory_dir = MEMORY_DIRS.get(schema)
        self.modules: dict = {}
        self.db = None
        self.merkle = IncrementalMerkle(schema)
        self.fingerprint = IncrementalFingerprint(schema)
        self.last_consolidation: Optional[str] = None
        self.last_full_consolidation: float = 0
        self.consolidation_count: int = 0
        self._loaded = False

    def load(self) -> dict:
        """Load all modules and DB connection. Call once at startup."""
        if self._loaded:
            return {"status": "already_loaded"}

        errors = []

        # Ensure database module is on path
        if str(DB_PATH) not in sys.path:
            sys.path.insert(0, str(DB_PATH))

        # Get DB connection (use env vars for Docker, fall back to defaults)
        try:
            from db import MemoryDB, get_pool
            db_config = {
                'host': os.environ.get('DB_HOST', 'localhost'),
                'port': int(os.environ.get('DB_PORT', '5433')),
                'dbname': os.environ.get('DB_NAME', 'agent_memory'),
                'user': os.environ.get('DB_USER', 'agent_admin'),
                'password': os.environ.get('DB_PASSWORD', 'agent_memory_local_dev'),
            }
            # Pre-initialize the global connection pool BEFORE any modules load.
            # This ensures all modules that call get_pool() reuse this pool
            # (which connects to host.docker.internal, not localhost).
            get_pool(db_config)
            self.db = MemoryDB(schema=self.schema, config=db_config)
            logger.info(f"[{self.schema}] DB connected ({db_config['host']}:{db_config['port']})")
        except Exception as e:
            errors.append(f"DB connection failed: {e}")
            logger.error(f"[{self.schema}] DB connection failed: {e}")

        # Load modules from memory directory
        if self.memory_dir and self.memory_dir.exists():
            self._load_modules(errors)
        else:
            errors.append(f"Memory dir not found: {self.memory_dir}")

        self._loaded = True
        return {
            "schema": self.schema,
            "modules_loaded": len(self.modules),
            "module_names": list(self.modules.keys()),
            "errors": errors,
            "db_connected": self.db is not None,
        }

    def _load_modules(self, errors: list):
        """Import consolidation modules from memory directory."""
        mem_str = str(self.memory_dir)

        # In Docker, db_adapter.py computes _DB_PATH relative to __file__,
        # which resolves to a non-existent path. Fix: ensure the shared
        # database module is importable as both 'db' and 'database.db'.
        # Also pre-inject db_adapter's singleton with our properly-configured DB.
        if mem_str not in sys.path:
            sys.path.insert(0, mem_str)

        # Make 'database.db' import work via sys.modules injection.
        # db_adapter.py does: from database.db import MemoryDB
        # The database dir is mounted read-only, so we can't create __init__.py.
        # Instead, we inject a fake 'database' package into sys.modules.
        import types
        if 'database' not in sys.modules:
            database_pkg = types.ModuleType('database')
            database_pkg.__path__ = [str(DB_PATH)]
            sys.modules['database'] = database_pkg
        if 'database.db' not in sys.modules:
            import db as _db_mod
            sys.modules['database.db'] = _db_mod

        # Pre-inject DB singleton into the adapter module so ALL modules
        # use our properly-configured DB (with host.docker.internal).
        # Drift uses db_adapter.get_db(), Spin uses memory_common.get_db().
        for adapter_name in ("db_adapter", "memory_common"):
            try:
                mod = importlib.import_module(adapter_name)
                if hasattr(mod, '_db_instance') and mod._db_instance is None:
                    mod._db_instance = self.db
                    logger.info(f"[{self.schema}] Injected DB into {adapter_name} singleton")
            except ImportError:
                pass

        for name in CONSOLIDATION_MODULES:
            try:
                # Use importlib to avoid conflicts between drift/spin modules
                mod = importlib.import_module(name)
                self.modules[name] = mod
            except ImportError as e:
                errors.append(f"{name}: {e}")
            except Exception as e:
                errors.append(f"{name}: {e}")

        logger.info(f"[{self.schema}] Loaded {len(self.modules)}/{len(CONSOLIDATION_MODULES)} modules")

    def _get_mod(self, name: str):
        """Get a loaded module, or None if not available."""
        return self.modules.get(name)

    def consolidate(
        self,
        transcript_path: str = "",
        phases: list[str] = None,
        force: bool = False,
    ) -> dict:
        """
        Run the full consolidation pipeline.

        Args:
            transcript_path: Path to session transcript (.jsonl)
            phases: Which phases to run. Default ["all"].
                    Options: "lightweight", "core", "enrichment", "attestation", "finalize", "all"
            force: Skip debounce check

        Returns:
            Summary dict with results from each phase.
        """
        if phases is None:
            phases = ["all"]

        t0 = time.monotonic()
        results = {
            "schema": self.schema,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "phases": {},
            "errors": [],
        }

        run_all = "all" in phases

        # Phase 0: Lightweight (always runs)
        if run_all or "lightweight" in phases:
            results["phases"]["lightweight"] = self._phase_lightweight()

        # Check debounce for expensive phases
        debounce_seconds = 60
        elapsed_since_last = time.monotonic() - self.last_full_consolidation if self.last_full_consolidation else float('inf')
        if not force and elapsed_since_last < debounce_seconds:
            results["debounced"] = True
            results["seconds_since_last"] = round(elapsed_since_last, 1)
            results["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 1)
            return results

        # Phase 1: Core consolidation
        if run_all or "core" in phases:
            results["phases"]["core"] = self._phase_core(transcript_path)

        # Phase 2: Enrichment
        if run_all or "enrichment" in phases:
            results["phases"]["enrichment"] = self._phase_enrichment(transcript_path)

        # Phase 3: Attestation (INCREMENTAL)
        if run_all or "attestation" in phases:
            results["phases"]["attestation"] = self._phase_attestation()

        # Phase 4: Finalize
        if run_all or "finalize" in phases:
            results["phases"]["finalize"] = self._phase_finalize()

        # Only update debounce timer when heavy phases (attestation/finalize) ran.
        # Lightweight/core-only calls (from pre_compact, subagent_stop, teammate_idle)
        # should NOT suppress stop.py's full consolidation.
        ran_heavy = any(p in phases for p in ("attestation", "finalize")) or run_all
        if ran_heavy:
            self.last_full_consolidation = time.monotonic()
        self.last_consolidation = datetime.now(timezone.utc).isoformat()
        self.consolidation_count += 1

        elapsed = (time.monotonic() - t0) * 1000
        results["elapsed_ms"] = round(elapsed, 1)
        results["consolidation_number"] = self.consolidation_count

        logger.info(f"[{self.schema}] Consolidation #{self.consolidation_count} complete in {elapsed:.0f}ms")
        return results

    def _safe_call(self, label: str, fn, *args, **kwargs) -> dict:
        """Call a function with error handling. Returns {ok, result/error, elapsed_ms}."""
        t0 = time.monotonic()
        try:
            result = fn(*args, **kwargs)
            elapsed = (time.monotonic() - t0) * 1000
            return {"ok": True, "result": result, "elapsed_ms": round(elapsed, 1)}
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            logger.warning(f"[{self.schema}] {label} failed: {e}")
            return {"ok": False, "error": str(e), "elapsed_ms": round(elapsed, 1)}

    # ===== Phase 0: Lightweight (runs every stop, ~100ms) =====

    def _phase_lightweight(self) -> dict:
        """Save pending co-occurrences. Fast, runs every stop."""
        results = {}

        # Pre-load session state to trigger deferred processing
        ss = self._get_mod("session_state")
        if ss:
            results["session_load"] = self._safe_call("session_state.load", ss.load)

        # Save pending co-occurrences
        co = self._get_mod("co_occurrence")
        if co:
            if hasattr(co, "end_session_cooccurrence"):
                results["co_occurrence"] = self._safe_call(
                    "end_session_cooccurrence", co.end_session_cooccurrence
                )
            elif hasattr(co, "save_pending_cooccurrence"):
                results["co_occurrence"] = self._safe_call(
                    "save_pending_cooccurrence", co.save_pending_cooccurrence
                )

        return results

    # ===== Phase 1: Core (transcript, maintenance, Q-update) =====

    def _phase_core(self, transcript_path: str) -> dict:
        """Independent tasks that form the core consolidation."""
        results = {}

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {}

            # Transcript processing
            tp = self._get_mod("transcript_processor")
            if tp and transcript_path:
                futures["transcript"] = pool.submit(
                    self._safe_call, "transcript",
                    tp.process_for_memory,
                    Path(transcript_path), True, 5
                )

            # Auto-memory consolidation
            amh = self._get_mod("auto_memory_hook")
            if amh and hasattr(amh, "consolidate_to_long_term"):
                futures["auto_memory"] = pool.submit(
                    self._safe_call, "auto_memory",
                    amh.consolidate_to_long_term
                )

            # Session maintenance (decay, promote)
            de = self._get_mod("decay_evolution")
            if de and hasattr(de, "session_maintenance"):
                futures["maintenance"] = pool.submit(
                    self._safe_call, "maintenance",
                    de.session_maintenance
                )

            # Behavioral rejections
            results["behavioral"] = self._task_behavioral_rejections()

            # Q-value update
            qv = self._get_mod("q_value_engine")
            if qv and hasattr(qv, "session_end_q_update"):
                futures["q_update"] = pool.submit(
                    self._safe_call, "q_update",
                    qv.session_end_q_update
                )

            for name, fut in futures.items():
                try:
                    results[name] = fut.result(timeout=30)
                except Exception as e:
                    results[name] = {"ok": False, "error": str(e)}

        return results

    def _task_behavioral_rejections(self) -> dict:
        """Compute behavioral rejection diff: seen - engaged = taste signal."""
        try:
            if not self.db:
                return {"ok": False, "error": "no DB"}

            seen_raw = self.db.kv_get('.feed_seen') or {}
            if isinstance(seen_raw, str):
                seen_raw = json.loads(seen_raw)
            seen_posts = seen_raw.get('posts', {})

            engaged_raw = self.db.kv_get('.feed_engaged') or {}
            if isinstance(engaged_raw, str):
                engaged_raw = json.loads(engaged_raw)
            engaged_ids = set(engaged_raw.get('post_ids', []))
            engaged_authors = set(a.lower() for a in engaged_raw.get('authors', []))

            if not seen_posts:
                self.db.kv_set('.feed_seen', {})
                self.db.kv_set('.feed_engaged', {})
                return {"ok": True, "result": "no posts seen"}

            # Expand engaged by author
            for post_id, post_data in seen_posts.items():
                author = post_data.get('author', '').lower()
                if author in engaged_authors:
                    engaged_ids.add(post_id)

            arl = self._get_mod("auto_rejection_logger")
            if arl and hasattr(arl, "log_behavioral_rejections"):
                count = arl.log_behavioral_rejections(seen_posts, engaged_ids)
            else:
                count = 0

            self.db.kv_set('.feed_seen', {})
            self.db.kv_set('.feed_engaged', {})

            return {"ok": True, "result": f"{count} rejections ({len(seen_posts)} seen)"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ===== Phase 2: Enrichment (episodic, lessons, 5W, Gemma) =====

    def _phase_enrichment(self, transcript_path: str) -> dict:
        """Enrichment tasks — lessons, 5W rebuild, Gemma classification."""
        results = {}

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {}

            # Store session summary
            tp = self._get_mod("transcript_processor")
            if tp and transcript_path and hasattr(tp, "extract_session_summaries"):
                futures["summary"] = pool.submit(
                    self._safe_call, "summary",
                    self._task_store_summary, transcript_path
                )

            # Lesson mining (3 sources)
            le = self._get_mod("lesson_extractor")
            if le:
                for cmd in ["mine-memory", "mine-rejections", "mine-hubs"]:
                    label = f"lesson_{cmd.replace('mine-', '')}"
                    if hasattr(le, "mine_from_source"):
                        futures[label] = pool.submit(
                            self._safe_call, label,
                            le.mine_from_source, cmd.replace("mine-", "")
                        )

            # 5W rebuild
            cm = self._get_mod("context_manager")
            if cm and hasattr(cm, "rebuild_all"):
                futures["rebuild_5w"] = pool.submit(
                    self._safe_call, "rebuild_5w",
                    cm.rebuild_all
                )

            # Gemma vocab scan (optional — requires Ollama)
            gb = self._get_mod("gemma_bridge")
            if gb:
                if hasattr(gb, "_ollama_available") and hasattr(gb, "scan_dead_memories"):
                    futures["gemma_vocab"] = pool.submit(
                        self._safe_call, "gemma_vocab",
                        self._task_gemma_vocab
                    )
                if hasattr(gb, "classify_topics"):
                    futures["gemma_classify"] = pool.submit(
                        self._safe_call, "gemma_classify",
                        self._task_gemma_classify
                    )

            for name, fut in futures.items():
                try:
                    results[name] = fut.result(timeout=60)
                except Exception as e:
                    results[name] = {"ok": False, "error": str(e)}

        return results

    def _task_store_summary(self, transcript_path: str):
        """Extract and store session summary."""
        tp = self._get_mod("transcript_processor")
        summaries = tp.extract_session_summaries(Path(transcript_path))
        if summaries:
            merged = tp.amalgamate_summaries(summaries) if len(summaries) > 1 else summaries[0]
            mem_id = tp.store_session_summary(merged, self.memory_dir)
            return f"stored summary as {mem_id}" if mem_id else "no summary stored"
        return "no summaries found"

    def _task_gemma_vocab(self):
        """Scan dead memories for bridge terms."""
        gb = self._get_mod("gemma_bridge")
        if not gb._ollama_available():
            return "Ollama offline, skipped"
        result = gb.scan_dead_memories(limit=10)
        added = result.get("terms_added", 0)
        scanned = result.get("scanned", 0)
        return f"scanned {scanned}, added {added} terms"

    def _task_gemma_classify(self):
        """Classify untagged memories using Gemma."""
        gb = self._get_mod("gemma_bridge")
        if not gb._ollama_available():
            return "Ollama offline, skipped"

        rows = self.db.list_memories(type_='active', limit=500)
        candidates = [r for r in rows
                      if not (r.get('topic_context') or [])
                      and len(r.get('content', '')) > 50]

        classified = 0
        for row in candidates[:15]:
            topics = gb.classify_topics(row['content'])
            if topics:
                self.db.update_memory(row['id'], topic_context=topics)
                classified += 1

        return f"{classified}/{min(len(candidates), 15)} tagged ({len(candidates)} total untagged)"

    # ===== Phase 3: Attestation (INCREMENTAL) =====

    def _phase_attestation(self) -> dict:
        """Attestation using incremental merkle + fingerprint + taste."""
        results = {}

        if not self.db:
            return {"error": "no DB connection"}

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {}

            # Incremental Merkle (O(log n) instead of O(n))
            futures["merkle"] = pool.submit(
                self._safe_call, "merkle",
                self._task_incremental_merkle
            )

            # Incremental Fingerprint (delta graph updates)
            futures["fingerprint"] = pool.submit(
                self._safe_call, "fingerprint",
                self._task_incremental_fingerprint
            )

            # Taste attestation (still full recompute — it's fast)
            rl = self._get_mod("rejection_log")
            if rl and hasattr(rl, "generate_taste_attestation"):
                futures["taste"] = pool.submit(
                    self._safe_call, "taste",
                    self._task_taste_attestation
                )

            for name, fut in futures.items():
                try:
                    results[name] = fut.result(timeout=30)
                except Exception as e:
                    results[name] = {"ok": False, "error": str(e)}

        return results

    def _task_incremental_merkle(self) -> dict:
        """Update merkle tree incrementally and save attestation."""
        update = self.merkle.update(self.db)
        attestation = self.merkle.save_attestation(self.db)
        return {
            "memory_count": update["memory_count"],
            "added": update["added"],
            "modified": update["modified"],
            "deleted": update["deleted"],
            "chain_depth": update["chain_depth"],
            "root": update["root"][:16] + "...",
            "alerts": update["alerts"],
            "elapsed_ms": update["elapsed_ms"],
        }

    def _task_incremental_fingerprint(self) -> dict:
        """Update fingerprint graph incrementally and save attestation."""
        update = self.fingerprint.update(self.db)
        attestation = self.fingerprint.generate_attestation(self.db)
        return {
            "node_count": update["node_count"],
            "edge_count": update["edge_count"],
            "new_nodes": update["new_nodes"],
            "new_edges": update["new_edges"],
            "fingerprint_hash": update["fingerprint_hash"][:16] + "...",
            "drift_score": attestation.get("drift", {}).get("drift_score", 0),
            "elapsed_ms": update["elapsed_ms"],
        }

    def _task_taste_attestation(self) -> dict:
        """Generate taste attestation."""
        rl = self._get_mod("rejection_log")
        attestation = rl.generate_taste_attestation()

        # Save to DB
        if self.db:
            self.db.kv_set('taste_attestation', attestation)

        return {
            "rejection_count": attestation.get("rejection_count", 0),
            "taste_hash": str(attestation.get("taste_hash", ""))[:16] + "...",
        }

    # ===== Phase 4: Finalize (cognitive state, vitals, session end) =====

    def _phase_finalize(self) -> dict:
        """Final tasks: cognitive state end, vitals, session cleanup."""
        results = {}

        # Cognitive state end
        cs = self._get_mod("cognitive_state")
        if cs and hasattr(cs, "end_session"):
            results["cognitive_state"] = self._safe_call(
                "cognitive_state.end_session", cs.end_session
            )

        # Record system vitals
        sv = self._get_mod("system_vitals")
        if sv and hasattr(sv, "record_vitals"):
            results["vitals"] = self._safe_call(
                "system_vitals.record_vitals", sv.record_vitals
            )

        # End session state
        ss = self._get_mod("session_state")
        if ss:
            if hasattr(ss, "end"):
                results["session_end"] = self._safe_call("session_state.end", ss.end)
            else:
                # Fallback: save then clear
                self._safe_call("session_state.save", ss.save)
                results["session_end"] = self._safe_call("session_state.clear", ss.clear)

        return results

    def get_status(self) -> dict:
        """Get current engine status for /status endpoint."""
        memory_count = 0
        edge_count = 0
        if self.db:
            try:
                memory_count = self.db.count_memories()
            except Exception:
                pass
            try:
                cs = self.db.comprehensive_stats()
                edge_count = cs.get("edges", {}).get("total_edges", 0)
            except Exception:
                pass

        return {
            "schema": self.schema,
            "loaded": self._loaded,
            "modules_loaded": len(self.modules),
            "db_connected": self.db is not None,
            "last_consolidation": self.last_consolidation,
            "consolidation_count": self.consolidation_count,
            "memories": memory_count,
            "edges": edge_count,
            "merkle": {
                "root": self.merkle.root[:16] + "..." if self.merkle.root else None,
                "chain_depth": self.merkle.chain_depth,
                "leaf_count": len(self.merkle.leaves),
            },
            "fingerprint": {
                "hash": self.fingerprint.fingerprint_hash[:16] + "..." if self.fingerprint.fingerprint_hash else None,
                "node_count": self.fingerprint.node_count,
                "edge_count": self.fingerprint.edge_count,
            },
        }


class ConsolidationEngine:
    """Top-level engine managing both schemas."""

    def __init__(self):
        self.engines: dict[str, SchemaEngine] = {}
        self.started_at = datetime.now(timezone.utc).isoformat()

    def init(self) -> dict:
        """Initialize engines for all available schemas."""
        results = {}
        for schema in ("drift", "spin"):
            mem_dir = MEMORY_DIRS.get(schema)
            if mem_dir and mem_dir.exists():
                engine = SchemaEngine(schema)
                load_result = engine.load()
                self.engines[schema] = engine
                results[schema] = load_result
                logger.info(f"[{schema}] Engine initialized: {load_result['modules_loaded']} modules")
            else:
                results[schema] = {"status": "skipped", "reason": f"dir not found: {mem_dir}"}
                logger.info(f"[{schema}] Skipped: memory dir not found")
        return results

    def consolidate(self, schema: str, **kwargs) -> dict:
        """Run consolidation for a specific schema."""
        engine = self.engines.get(schema)
        if not engine:
            return {"error": f"Schema '{schema}' not loaded"}
        return engine.consolidate(**kwargs)

    def get_status(self) -> dict:
        """Get status for all schemas."""
        return {
            schema: engine.get_status()
            for schema, engine in self.engines.items()
        }

    def health(self) -> dict:
        """Health check."""
        return {
            "status": "ready" if self.engines else "no_schemas",
            "schemas": list(self.engines.keys()),
            "uptime_s": round(
                (datetime.now(timezone.utc) - datetime.fromisoformat(self.started_at)).total_seconds()
            ),
            "engines": {
                schema: {
                    "modules": len(engine.modules),
                    "db": engine.db is not None,
                }
                for schema, engine in self.engines.items()
            },
        }
