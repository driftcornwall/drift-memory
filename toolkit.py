#!/usr/bin/env python3
"""
Drift Unified Memory Toolkit v1.0

Single entry point for all memory system operations.
Adapted from SpindriftMend's toolkit.py (commit aa6e8f6).

Usage:
    python toolkit.py help                    # List all commands
    python toolkit.py help <category>         # Commands in category
    python toolkit.py status                  # Full system dashboard
    python toolkit.py health                  # Component health check
    python toolkit.py <category>:<command>    # Run a command
    python toolkit.py <command>               # Run (if unambiguous)

Categories: identity, social, platform, search, memory, comms, dashboard, viz, calibration, experiment, lessons

Origin: SpindriftMend designed the toolkit pattern. Drift adapted it.
"""

import sys
import os
import traceback as tb
from dataclasses import dataclass, field
from pathlib import Path

MEMORY_ROOT = Path(__file__).parent
TOOLKIT_CONFIG = {
    "agent_name": "DriftCornwall",
    "version": "1.0",
    "social_dir": "social",
    "synonym_module": "vocabulary_bridge",
    "has_reciprocity": False,
    "has_excavation": False,
    "has_telegram": True,
    "has_twitter": True,
    "has_dashboard": True,
    "has_dimensional_viz": True,
    "has_temporal_calibration": True,
}


@dataclass
class Command:
    name: str
    category: str
    description: str
    module: str
    handler: str
    args_help: str = ""


def build_registry():
    """Build the command registry. No imports happen here — just metadata."""
    cfg = TOOLKIT_CONFIG
    cmds = []

    # === IDENTITY ===
    cmds += [
        Command("fingerprint", "identity", "Full cognitive topology analysis", "cognitive_fingerprint", "analyze"),
        Command("hubs", "identity", "Top N hub memories", "cognitive_fingerprint", "hubs", "[N=15]"),
        Command("pairs", "identity", "Top N strongest pairs", "cognitive_fingerprint", "pairs", "[N=15]"),
        Command("clusters", "identity", "Cluster detection", "cognitive_fingerprint", "clusters"),
        Command("domains", "identity", "Cognitive domain breakdown", "cognitive_fingerprint", "domains"),
        Command("drift", "identity", "Identity evolution score", "cognitive_fingerprint", "drift"),
        Command("dimfp", "identity", "5W dimensional fingerprints", "cognitive_fingerprint", "dim-fingerprint"),
        Command("5w", "identity", "5W dimensional hashes", "cognitive_fingerprint", "5w"),
        Command("resilience", "identity", "Hub resilience analysis", "cognitive_fingerprint", "resilience"),
        Command("attest", "identity", "Generate cognitive attestation", "cognitive_fingerprint", "attest"),
        Command("merkle", "identity", "Generate Merkle attestation", "merkle_attestation", "generate-chain"),
        Command("merkle-verify", "identity", "Verify Merkle integrity", "merkle_attestation", "verify-integrity"),
        Command("merkle-history", "identity", "Attestation history", "merkle_attestation", "history"),
        Command("taste", "identity", "Show taste fingerprint", "rejection_log", "taste-profile"),
        Command("reject", "identity", "Log a rejection", "rejection_log", "log", "<category> <reason>"),
        Command("rejections", "identity", "List rejections", "rejection_log", "list"),
        Command("nostr-publish", "identity", "Publish attestation to Nostr", "nostr_attestation", "publish"),
        Command("nostr-dossier", "identity", "Publish full dossier to Nostr", "nostr_attestation", "publish-dossier"),
        Command("morning-post", "identity", "Daily proof-of-life post", "morning_post", "main"),
        Command("visualize", "identity", "Brain graph visualization", "brain_visualizer", "main"),
    ]

    # === SOCIAL ===
    cmds += [
        Command("log", "social", "Log an interaction", "social.social_memory", "log", "<contact> <platform> <type> <content>"),
        Command("contact", "social", "View contact details", "social.social_memory", "contact", "<name>"),
        Command("recent", "social", "Recent interactions", "social.social_memory", "recent", "[--limit N]"),
        Command("prime", "social", "Social priming context", "social.social_memory", "prime"),
        Command("replied", "social", "Log a reply I made", "social.social_memory", "replied", "<platform> <post_id> <content>"),
        Command("check", "social", "Check if I already replied", "social.social_memory", "check", "<platform> <post_id>"),
        Command("my-replies", "social", "List my recent replies", "social.social_memory", "my-replies", "[--days 7]"),
        Command("my-posts", "social", "List my recent posts", "social.social_memory", "my-posts", "[--days 7]"),
        Command("index", "social", "Rebuild social index", "social.social_memory", "index"),
        Command("feed", "social", "MoltX feed quality filter", "feed_quality", "main", "[--raw] [--min-score N]"),
        Command("contact-score", "social", "Score a contact (Bayesian model)", "contact_models", "score", "<name>"),
        Command("contact-predict", "social", "Predict engagement with contact", "contact_models", "predict", "<name> <topic>"),
        Command("contact-summary", "social", "All contacts ranked by engagement", "contact_models", "summary"),
    ]
    if cfg.get("has_reciprocity"):
        cmds += [
            Command("reciprocity", "social", "Scan reciprocity metrics", "social.reciprocity_tracker", "scan"),
            Command("reciprocity-report", "social", "Show cached reciprocity report", "social.reciprocity_tracker", "report"),
        ]

    # === PLATFORM ===
    cmds += [
        Command("stats", "platform", "Platform distribution statistics", "platform_context", "stats"),
        Command("find", "platform", "Find memories by platform", "platform_context", "find", "<platform>"),
        Command("bridges", "platform", "Cross-platform bridge memories", "platform_context", "bridges"),
        Command("matrix", "platform", "Platform co-occurrence matrix", "platform_context", "matrix"),
        Command("backfill", "platform", "Tag memories with platforms", "platform_context", "backfill"),
        Command("backfill-edges", "platform", "Tag edges with platform context", "platform_context", "backfill-edges"),
        Command("detect", "platform", "Test platform detection", "platform_context", "detect", "<text>"),
        Command("topic-stats", "platform", "Topic distribution stats", "topic_context", "stats"),
        Command("topic-backfill", "platform", "Tag memories with topics", "topic_context", "backfill"),
        Command("contact-stats", "platform", "Contact distribution stats", "contact_context", "stats"),
        Command("contact-backfill", "platform", "Tag memories with contacts", "contact_context", "backfill"),
        Command("context-rebuild", "platform", "Full 5W projection rebuild", "context_manager", "rebuild"),
        Command("context-stats", "platform", "5W context statistics", "context_manager", "stats"),
    ]

    # === SEARCH ===
    cmds += [
        Command("query", "search", "Semantic search", "semantic_search", "search", "<query> [--limit N]"),
        Command("search-index", "search", "Index memories for search", "semantic_search", "index"),
        Command("search-status", "search", "Search index status", "semantic_search", "status"),
        Command("synonyms", "search", "Synonym bridge statistics", cfg.get("synonym_module", "synonym_bridge"), "stats"),
        Command("expand", "search", "Expand query with synonyms", cfg.get("synonym_module", "synonym_bridge"), "expand"),
        Command("gemma-scan", "search", "Scan dead memories for bridges", "gemma_bridge", "scan"),
        Command("gemma-curate", "search", "Review unconfirmed terms", "gemma_bridge", "curate"),
    ]
    if cfg.get("has_excavation"):
        cmds += [
            Command("excavate", "search", "Surface forgotten memories", "memory_excavation", "excavate", "[N=3]"),
            Command("excavation-stats", "search", "Excavation statistics", "memory_excavation", "stats"),
        ]

    # === MEMORY ===
    cmds += [
        Command("mem-stats", "memory", "Full memory statistics", "memory_manager", "stats"),
        Command("session-status", "memory", "Current session state", "memory_manager", "session-status"),
        Command("recall", "memory", "Recall a specific memory", "memory_manager", "recall", "<id>"),
        Command("find-tag", "memory", "Find memories by tag", "memory_manager", "find", "<tag>"),
        Command("tags", "memory", "List all tags", "memory_manager", "tags"),
        Command("related", "memory", "Find related memories", "memory_manager", "related", "<id>"),
        Command("maintenance", "memory", "Run session maintenance", "memory_manager", "maintenance"),
        Command("cooccur", "memory", "Co-occurrence statistics", "memory_manager", "sync-cooccur"),
        Command("priming", "memory", "Get priming candidates", "memory_manager", "priming"),
        Command("consolidate-candidates", "memory", "Find similar memories", "memory_manager", "consolidate-candidates"),
        Command("intend", "memory", "Create a temporal intention", "temporal_intentions", "create", "<action> --trigger-type time|event --trigger <cond>"),
        Command("intentions", "memory", "List pending intentions", "temporal_intentions", "list"),
        Command("check-intentions", "memory", "Check triggered intentions", "temporal_intentions", "check"),
        Command("complete-intention", "memory", "Mark intention done", "temporal_intentions", "complete", "<id> [--outcome text]"),
        Command("recon-candidates", "memory", "Reconsolidation candidates", "reconsolidation", "candidates"),
        Command("recon-queue", "memory", "Queue candidates for revision", "reconsolidation", "queue"),
        Command("recon-stats", "memory", "Reconsolidation statistics", "reconsolidation", "stats"),
        Command("recon-history", "memory", "Revision history for a memory", "reconsolidation", "history", "<id>"),
        Command("adapt", "memory", "Run adaptive behavior loop", "adaptive_behavior", "adapt"),
        Command("adapt-current", "memory", "Active adaptations", "adaptive_behavior", "current"),
        Command("adapt-stats", "memory", "Adaptation statistics", "adaptive_behavior", "stats"),
        Command("mine-strategies", "search", "Extract strategies from traces", "explanation_miner", "mine"),
        Command("strategies", "search", "Active strategy heuristics", "explanation_miner", "strategies"),
        Command("strategy-stats", "search", "Mining statistics", "explanation_miner", "stats"),
        Command("dream", "memory", "Run generative sleep cycle", "generative_sleep", "dream"),
        Command("dream-sample", "memory", "Preview dream sampling", "generative_sleep", "sample"),
        Command("dream-stats", "memory", "Dream statistics", "generative_sleep", "stats"),
        Command("llm-status", "memory", "LLM backend status", "llm_client", "status"),
        Command("llm-test", "memory", "Test LLM generation", "llm_client", "test"),
        Command("export", "memory", "Secure memory export", "memory_interop", "export"),
        Command("import", "memory", "Import with quarantine", "memory_interop", "import"),
        Command("follows", "memory", "What memories follow this one? (R10)", "semantic_search", "temporal-successors", "<id>"),
        Command("predict", "memory", "Generate session predictions (R11)", "prediction_module", "generate"),
        Command("predict-score", "memory", "Score predictions against actuals", "prediction_module", "score"),
        Command("predict-history", "memory", "Show prediction history", "prediction_module", "history"),
        Command("predict-calibration", "memory", "Prediction calibration stats", "prediction_module", "calibration"),
        Command("self", "memory", "Self-narrative model", "self_narrative", "narrative"),
        Command("self-query", "memory", "Query self-state", "self_narrative", "query", "<question>"),
    ]

    # === COMMS ===
    if cfg.get("has_telegram"):
        cmds += [
            Command("tg-send", "comms", "Send Telegram message to Lex", "telegram_bot", "send", "<message>"),
            Command("tg-poll", "comms", "Check for Telegram messages", "telegram_bot", "poll"),
            Command("tg-test", "comms", "Test Telegram connection", "telegram_bot", "test"),
        ]

    # === TWITTER ===
    if cfg.get("has_twitter"):
        cmds += [
            Command("tw-new", "social", "Check NEW mentions only", "@twitter/client", "new-mentions"),
            Command("tw-mentions", "social", "Get all mentions", "@twitter/client", "mentions"),
            Command("tw-post", "social", "Post a tweet", "@twitter/client", "post", "<text> [--reply-to ID]"),
            Command("tw-like", "social", "Like a tweet", "@twitter/client", "like", "<tweet_id>"),
            Command("tw-timeline", "social", "Get home timeline", "@twitter/client", "timeline"),
            Command("tw-search", "social", "Search tweets", "@twitter/client", "search", "<query>"),
            Command("tw-me", "social", "Get my profile", "@twitter/client", "me"),
            Command("tw-state", "social", "Show interaction state", "@twitter/client", "state"),
            Command("tw-contacts", "social", "Twitter contacts (5W log)", "@twitter/client", "contacts"),
        ]

    # === DASHBOARD ===
    if cfg.get("has_dashboard"):
        cmds += [
            Command("dash-export", "dashboard", "Export graph data to JSON", "dashboard_export", "main"),
            Command("dash-build", "dashboard", "Build standalone HTML dashboard", "dashboard.build", "main"),
        ]

    # === EXPLAINABILITY ===
    cmds += [
        Command("explain-last", "dashboard", "Show last N explanations", "explanation", "last", "[N=5]"),
        Command("explain-search", "dashboard", "Show search explanations", "explanation", "search"),
        Command("explain-priming", "dashboard", "Show last priming explanation", "explanation", "priming"),
        Command("explain-why", "dashboard", "Why was a memory selected?", "explanation", "why", "<memory_id>"),
        Command("explain-stats", "dashboard", "Explanation coverage statistics", "explanation", "stats"),
        Command("explain-session", "dashboard", "All explanations this session", "explanation", "session"),
    ]

    # === CURIOSITY ENGINE ===
    cmds += [
        Command("curiosity-scan", "dashboard", "Top curiosity targets (sparse graph)", "curiosity_engine", "scan", "[N=10]"),
        Command("curiosity-map", "dashboard", "Graph sparsity analysis", "curiosity_engine", "map"),
        Command("curiosity-stats", "dashboard", "Exploration history & conversion rates", "curiosity_engine", "stats"),
        Command("curiosity-targets", "dashboard", "Diversified targets for priming", "curiosity_engine", "targets", "[N=5]"),
    ]

    # === COGNITIVE STATE ===
    cmds += [
        Command("cognitive-state", "dashboard", "Current 5-dimension cognitive state", "cognitive_state", "state"),
        Command("cognitive-history", "dashboard", "State changes this session", "cognitive_state", "history"),
        Command("cognitive-trend", "dashboard", "Cross-session cognitive trends", "cognitive_state", "trend"),
    ]

    # === KNOWLEDGE GRAPH ===
    cmds += [
        Command("kg-extract", "dashboard", "Auto-extract typed relationships", "knowledge_graph", "extract", "[--limit=N]"),
        Command("kg-query", "dashboard", "Multi-hop graph traversal", "knowledge_graph", "query", "<id> [rel] [hops]"),
        Command("kg-stats", "dashboard", "Relationship type distribution", "knowledge_graph", "stats"),
        Command("kg-types", "dashboard", "List all relationship types with counts", "knowledge_graph", "types"),
        Command("kg-path", "dashboard", "Find shortest typed path", "knowledge_graph", "path", "<id1> <id2>"),
    ]

    # === CONTRADICTION DETECTION & VALIDATION ===
    cmds += [
        Command("contradict-status", "search", "NLI service status", "contradiction_detector", "status"),
        Command("contradict-check", "search", "Check memory for contradictions", "contradiction_detector", "check", "<id>"),
        Command("contradict-scan", "search", "Batch scan for contradictions", "contradiction_detector", "scan", "[limit]"),
        Command("contradict-report", "search", "Show all detected contradictions", "contradiction_detector", "report"),
        Command("validate", "search", "Classify evidence type of text", "memory_validation", "classify", "<text>"),
        Command("source-reliability", "search", "Source reliability scores", "memory_validation", "reliability", "[source]"),
        Command("verify-memory", "search", "Mark memory as verified", "memory_validation", "verify", "<id>"),
        Command("dispute-memory", "search", "Mark memory as disputed", "memory_validation", "dispute", "<id> [reason]"),
    ]

    # === SELF-NARRATIVE (Higher-Order Thought, R9) ===
    cmds += [
        Command("self", "identity", "Generate self-narrative", "self_narrative", "narrative"),
        Command("self-model", "identity", "Full self-model (JSON)", "self_narrative", "generate"),
        Command("self-query", "identity", "Query self-state", "self_narrative", "query", "<question>"),
    ]

    # === COUNTERFACTUAL REASONING (N3) ===
    cmds += [
        Command("cf-generate", "dashboard", "Run session-end counterfactual review", "counterfactual_engine", "generate"),
        Command("cf-history", "dashboard", "Counterfactual generation history", "counterfactual_engine", "history", "[N]"),
        Command("cf-stats", "dashboard", "Counterfactual summary statistics", "counterfactual_engine", "stats"),
        Command("cf-quality", "dashboard", "Quality analysis for this session", "counterfactual_engine", "quality"),
        Command("cf-trace", "dashboard", "Decision trace (recall-to-action)", "counterfactual_engine", "trace"),
        Command("cf-health", "dashboard", "Counterfactual engine health", "counterfactual_engine", "health"),
    ]

    # === Q-VALUE LEARNING (MemRL) ===
    cmds += [
        Command("q-stats", "dashboard", "Q-value distribution", "q_value_engine", "stats"),
        Command("q-top", "dashboard", "Top memories by Q-value", "q_value_engine", "top", "[N]"),
        Command("q-bottom", "dashboard", "Bottom memories by Q-value", "q_value_engine", "bottom", "[N]"),
        Command("q-history", "dashboard", "Q trajectory for memory", "q_value_engine", "history", "<id>"),
        Command("q-convergence", "dashboard", "Q-value convergence report", "q_value_engine", "convergence"),
    ]

    # === VISUALIZATION ===
    if cfg.get("has_dimensional_viz"):
        cmds += [
            Command("dim-viz", "viz", "Generate 5W dimensional visualization", "dimensional_viz", "main"),
        ]

    # === CALIBRATION ===
    if cfg.get("has_temporal_calibration"):
        cmds += [
            Command("calibrate", "calibration", "Take temporal calibration reading", "temporal_calibration", "read"),
            Command("cal-drift", "calibration", "Show calibration drift over time", "temporal_calibration", "drift"),
            Command("cal-history", "calibration", "Show all readings", "temporal_calibration", "history"),
        ]

    # === EXPERIMENT ===
    cmds += [
        Command("exp-compare", "experiment", "Compare fingerprints side-by-side", "experiment_compare", "main"),
        Command("exp-snapshot", "experiment", "Take experiment snapshot", "experiment_delta", "snapshot", "--tag <label>"),
        Command("exp-delta", "experiment", "Compare experiment snapshots", "experiment_delta", "compare", "<file1> <file2>"),
    ]

    # === VOLITIONAL GOALS (N4) ===
    cmds += [
        Command("goal-generate", "memory", "Synthesize + score + commit new goals", "goal_generator", "generate"),
        Command("goal-active", "memory", "List active goals with vitality", "goal_generator", "active"),
        Command("goal-focus", "memory", "Show current focus goal", "goal_generator", "focus"),
        Command("goal-eval", "memory", "Evaluate progress (session end)", "goal_generator", "evaluate"),
        Command("goal-abandon", "memory", "Manually abandon a goal", "goal_generator", "abandon", "<id>"),
        Command("goal-history", "memory", "Completed/abandoned goals", "goal_generator", "history"),
        Command("goal-stats", "memory", "Goal generation statistics", "goal_generator", "stats"),
        Command("goal-health", "memory", "Goal module health check", "goal_generator", "health"),
    ]

    # === LESSONS ===
    cmds += [
        Command("lessons", "lessons", "List all lessons", "lesson_extractor", "list"),
        Command("lesson-add", "lessons", "Add a lesson manually", "lesson_extractor", "add", "<category> <lesson>"),
        Command("lesson-mine", "lessons", "Mine MEMORY.md for lessons", "lesson_extractor", "mine-memory"),
        Command("lesson-rejections", "lessons", "Mine rejection patterns", "lesson_extractor", "mine-rejections"),
        Command("lesson-hubs", "lessons", "Mine co-occurrence hubs", "lesson_extractor", "mine-hubs"),
        Command("lesson-apply", "lessons", "Find lessons for a situation", "lesson_extractor", "apply", "<situation>"),
        Command("lesson-stats", "lessons", "Lesson statistics", "lesson_extractor", "stats"),
    ]

    registry = {}
    for cmd in cmds:
        key = f"{cmd.category}:{cmd.name}"
        registry[key] = cmd
    return registry


REGISTRY = build_registry()


def resolve_module(module_name):
    """Lazy-import a module by name, handling subpackages and external dirs."""
    # Handle @dir/module syntax for modules outside memory/
    if module_name.startswith('@'):
        rel_path = module_name[1:]  # e.g. "twitter/client"
        ext_path = MEMORY_ROOT.parent / rel_path.replace('/', os.sep)
        ext_dir = ext_path.parent
        mod_name = ext_path.stem
        old_path = sys.path.copy()
        sys.path.insert(0, str(ext_dir))
        try:
            return __import__(mod_name)
        finally:
            sys.path = old_path

    old_path = sys.path.copy()
    if str(MEMORY_ROOT) not in sys.path:
        sys.path.insert(0, str(MEMORY_ROOT))
    try:
        if '.' in module_name:
            parts = module_name.split('.')
            pkg_path = MEMORY_ROOT / parts[0]
            if str(pkg_path) not in sys.path:
                sys.path.insert(0, str(pkg_path.parent))
            mod = __import__(module_name, fromlist=[parts[-1]])
        else:
            mod = __import__(module_name)
        return mod
    finally:
        sys.path = old_path


def dispatch_to_module(cmd, args):
    """Import module and run its CLI with the handler as first arg."""
    mod = resolve_module(cmd.module)
    _ensure_stdout()

    # Strategy: most modules use sys.argv dispatch in __main__
    # We simulate that by setting sys.argv and calling their main logic
    handler = cmd.handler

    # Check if module has a direct function we can call
    if hasattr(mod, f'cmd_{handler.replace("-", "_")}'):
        func = getattr(mod, f'cmd_{handler.replace("-", "_")}')
        func(*args) if args else func()
        return

    # Check for 'main' function
    if handler == 'main' and hasattr(mod, 'main'):
        old_argv = sys.argv
        sys.argv = [cmd.module] + list(args)
        try:
            mod.main()
        finally:
            sys.argv = old_argv
        return

    # Fallback: re-run module as subprocess with the handler as CLI arg
    mod_file = getattr(mod, '__file__', None)
    if mod_file:
        import subprocess
        argv = [sys.executable, mod_file, handler] + list(args)
        result = subprocess.run(argv, capture_output=False, cwd=str(MEMORY_ROOT))
        sys.exit(result.returncode)
    else:
        print(f"[ERROR] Cannot dispatch '{handler}' to module '{cmd.module}'")
        sys.exit(1)


def _ensure_stdout():
    """Restore stdout if a module closed it during import."""
    if sys.stdout.closed:
        import io
        try:
            sys.stdout = io.TextIOWrapper(sys.__stdout__.buffer, encoding='utf-8', errors='replace')
        except ValueError:
            sys.stdout = open(os.dup(1), 'w', encoding='utf-8', errors='replace')


def cmd_status():
    """Full system dashboard — calls multiple modules, handles failures per section."""
    print(f"{'=' * 60}")
    print(f"  {TOOLKIT_CONFIG['agent_name']} System Status")
    print(f"  toolkit v{TOOLKIT_CONFIG['version']}")
    print(f"{'=' * 60}")
    print()

    # [1] MEMORY
    print("[1] MEMORY")
    try:
        mm = resolve_module("memory_manager")
        _ensure_stdout()
        stats = mm.get_comprehensive_stats() if hasattr(mm, 'get_comprehensive_stats') else None
        if stats:
            ms = stats.get('memory_stats', {})
            print(f"    Total memories: {ms.get('total', '?')}")
            print(f"    Types: core={ms.get('core', '?')}, active={ms.get('active', '?')}, archive={ms.get('archive', '?')}")
            co = stats.get('cooccurrence_stats', {})
            print(f"    Active pairs: {co.get('active_pairs', '?')} | Links: {co.get('links_created', '?')}")
        else:
            total = sum(1 for d in ['core', 'active', 'archive']
                        for f in (MEMORY_ROOT / d).glob('*.md') if (MEMORY_ROOT / d).exists())
            print(f"    Total memories: ~{total}")
    except Exception as e:
        _ensure_stdout()
        print(f"    [UNAVAILABLE] {e}")
    print()

    # [2] GRAPH TOPOLOGY
    print("[2] GRAPH TOPOLOGY")
    try:
        cf = resolve_module("cognitive_fingerprint")
        _ensure_stdout()
        graph = cf.build_graph()
        nodes = len(graph.get('nodes', {}))
        edges = len(graph.get('edges', {}))
        print(f"    Nodes: {nodes} | Edges: {edges}")
        drift_data = cf.compute_drift_score(cf.generate_full_analysis()) if hasattr(cf, 'compute_drift_score') else None
        if drift_data:
            print(f"    Drift: {drift_data.get('drift_score', '?')} ({drift_data.get('interpretation', '?')})")
    except Exception as e:
        _ensure_stdout()
        print(f"    [UNAVAILABLE] {e}")
    print()

    # [3] PLATFORM CONTEXT
    print("[3] PLATFORM CONTEXT")
    try:
        pc = resolve_module("platform_context")
        _ensure_stdout()
        pstats = pc.platform_stats() if hasattr(pc, 'platform_stats') else None
        if pstats:
            total = pstats.get('total_memories', '?')
            tagged = pstats.get('tagged_memories', '?')
            pct = round(tagged * 100 / total) if isinstance(total, int) and total > 0 else '?'
            xp = pstats.get('cross_platform_count', '?')
            print(f"    Tagged: {tagged}/{total} ({pct}%) | Cross-platform: {xp}")
            counts = pstats.get('platform_counts', {})
            parts = [f"{k}={v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])[:7]]
            print(f"    Distribution: {', '.join(parts)}")
        else:
            print("    [No stats function available]")
    except Exception as e:
        _ensure_stdout()
        print(f"    [UNAVAILABLE] {e}")
    print()

    # [4] SOCIAL
    print("[4] SOCIAL")
    try:
        sm = resolve_module("social.social_memory")
        _ensure_stdout()
        idx = sm.update_index()
        total_contacts = idx.get('total_contacts', 0)
        active_week = idx.get('active_week', 0)
        from db_adapter import get_db as _get_db_social
        reply_count = 0
        _sdb = _get_db_social()
        _replies_data = _sdb.kv_get('.social_my_replies')
        if _replies_data:
            import json
            _replies_data = json.loads(_replies_data) if isinstance(_replies_data, str) else _replies_data
            reply_count = len(_replies_data.get('replies', {}))
        print(f"    Contacts: {total_contacts} | Active this week: {active_week} | Replies tracked: {reply_count}")
    except Exception as e:
        _ensure_stdout()
        print(f"    [UNAVAILABLE] {e}")
    print()

    # [5] TELEGRAM
    print("[5] TELEGRAM")
    try:
        tg_creds = Path(os.path.expanduser('~/.config/telegram/drift-credentials.json'))
        if tg_creds.exists():
            print(f"    Credentials: OK")
        else:
            print(f"    Credentials: NOT FOUND")
        tg_state = MEMORY_ROOT / '.telegram_state.json'
        if tg_state.exists():
            import json
            ts = json.loads(tg_state.read_text(encoding='utf-8'))
            print(f"    Last update ID: {ts.get('last_update_id', 0)}")
        else:
            print(f"    State: fresh (no messages processed)")
    except Exception as e:
        print(f"    [UNAVAILABLE] {e}")
    print()

    # [6] IDENTITY STACK
    print("[6] IDENTITY STACK")
    try:
        from db_adapter import get_db as _get_db_dossier
        db = _get_db_dossier()
        # Merkle chain from attestations table
        with db._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT data FROM {db._table('attestations')} ORDER BY timestamp DESC LIMIT 1")
                row = cur.fetchone()
                if row and row[0]:
                    att_data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                    depth = att_data.get('chain_depth', '?')
                    print(f"    Merkle chain: depth {depth}")
                else:
                    print(f"    Merkle chain: not initialized")
        # Rejections from DB
        cs = db.comprehensive_stats()
        print(f"    Rejections logged: {cs.get('rejections', 0)}")
    except Exception as e:
        print(f"    [UNAVAILABLE] {e}")
    print()

    # [7] VOCABULARY
    print("[7] VOCABULARY")
    try:
        sb = resolve_module(TOOLKIT_CONFIG.get("synonym_module", "synonym_bridge"))
        _ensure_stdout()
        if hasattr(sb, 'SYNONYM_GROUPS'):
            groups = len(sb.SYNONYM_GROUPS)
            if isinstance(sb.SYNONYM_GROUPS, dict):
                terms = sum(len(g) for g in sb.SYNONYM_GROUPS.values())
            elif isinstance(sb.SYNONYM_GROUPS, list):
                terms = sum(len(g) for g in sb.SYNONYM_GROUPS)
            else:
                terms = '?'
            print(f"    Groups: {groups} | Terms: {terms}")
        elif hasattr(sb, 'bridge_stats'):
            sb.bridge_stats()
        else:
            print("    [No stats available]")
    except Exception as e:
        print(f"    [UNAVAILABLE] {e}")
    print()

    # [8] SEARCH INDEX
    print("[8] SEARCH INDEX")
    try:
        from db_adapter import get_db as _get_db_idx
        db = _get_db_idx()
        with db._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {db._table('text_embeddings')}")
                count = cur.fetchone()[0]
        print(f"    Indexed memories: {count}")
    except Exception as e:
        print(f"    [UNAVAILABLE] {e}")
    print()

    # [9] SESSION
    print("[9] SESSION")
    try:
        from db_adapter import get_db as _get_db_sess
        db = _get_db_sess()
        raw = db.kv_get('.session_state')
        if raw:
            state = json.loads(raw) if isinstance(raw, str) else raw
            retrieved = state.get('retrieved', [])
            started = state.get('started', '?')
            print(f"    Recalled this session: {len(retrieved)}")
            print(f"    Session started: {started}")
        else:
            print(f"    No active session")
    except Exception as e:
        print(f"    [UNAVAILABLE] {e}")

    print()
    print(f"{'=' * 60}")


def cmd_health():
    """Check health of all modules."""
    print(f"{'=' * 60}")
    print(f"  {TOOLKIT_CONFIG['agent_name']} Health Check")
    print(f"{'=' * 60}")
    print()

    modules = [
        ("memory_manager", "Core memory system"),
        ("cognitive_fingerprint", "Identity fingerprinting"),
        ("context_manager", "5W projection engine"),
        ("platform_context", "Platform detection (WHERE)"),
        ("contact_context", "Contact detection (WHO)"),
        ("topic_context", "Topic classification (WHAT)"),
        ("activity_context", "Activity classification (WHY)"),
        ("semantic_search", "Semantic search engine"),
        (TOOLKIT_CONFIG.get("synonym_module", "synonym_bridge"), "Vocabulary bridging"),
        ("gemma_bridge", "Gemma 3 auto-discovery"),
        ("merkle_attestation", "Merkle integrity proofs"),
        ("rejection_log", "Taste fingerprint"),
        ("nostr_attestation", "Nostr publishing"),
        ("morning_post", "Daily proof-of-life"),
        ("brain_visualizer", "Graph visualization"),
        ("feed_quality", "Feed quality filter"),
        ("memory_interop", "Memory import/export"),
        ("temporal_calibration", "Temporal calibration"),
        ("experiment_compare", "Experiment comparison"),
        ("experiment_delta", "Experiment snapshots"),
        ("dashboard_export", "Dashboard data export"),
        ("dimensional_viz", "5W dimensional visualization"),
        ("telegram_bot", "Telegram communication"),
        ("lesson_extractor", "Lesson extraction system"),
        ("curiosity_engine", "Curiosity-driven exploration"),
        ("explanation", "Explainability interface"),
        ("cognitive_state", "Cognitive state tracker"),
        ("knowledge_graph", "Knowledge graph (typed relationships)"),
        ("contradiction_detector", "NLI contradiction detection"),
        ("memory_validation", "Evidence classification & source reliability"),
        ("q_value_engine", "Q-value learning (MemRL)"),
        ("self_narrative", "Self-narrative synthesis"),
        ("contact_models", "Per-contact Bayesian scoring"),
        ("prediction_module", "Forward model predictions"),
        ("goal_generator", "Volitional goal generation (N4)"),
    ]

    # Social subpackage
    social_modules = [
        ("social.social_memory", "Social relationship tracking"),
    ]

    passed = 0
    failed = 0
    warned = 0

    for mod_name, desc in modules + social_modules:
        try:
            mod = resolve_module(mod_name)
            _ensure_stdout()

            status = "OK"
            detail = ""

            # Quick probes for key modules
            if mod_name == "memory_manager" and hasattr(mod, 'MEMORY_ROOT'):
                dirs = sum(1 for d in ['core', 'active', 'archive'] if (mod.MEMORY_ROOT / d).exists())
                detail = f"({dirs}/3 dirs)"
            elif mod_name == "cognitive_fingerprint" and hasattr(mod, 'build_graph'):
                g = mod.build_graph()
                detail = f"({len(g.get('nodes', {}))} nodes)"
            elif mod_name == "platform_context" and hasattr(mod, 'PLATFORMS'):
                detail = f"({len(mod.PLATFORMS)} platforms)"
            elif mod_name == "gemma_bridge":
                if hasattr(mod, '_ollama_available') and mod._ollama_available():
                    detail = "(Ollama running)"
                else:
                    status = "WARN"
                    detail = "(Ollama not reachable)"
                    warned += 1
                    print(f"  {mod_name:<30s} {status:<6s} {detail}")
                    continue

            passed += 1
            print(f"  {mod_name:<30s} {status:<6s} {detail}")
        except ImportError as e:
            failed += 1
            _ensure_stdout()
            print(f"  {mod_name:<30s} FAIL   (import: {e})")
        except Exception as e:
            failed += 1
            _ensure_stdout()
            print(f"  {mod_name:<30s} FAIL   ({e})")

    print()
    total = passed + failed + warned
    print(f"  Result: {passed} OK, {warned} WARN, {failed} FAIL / {total} total")
    print(f"{'=' * 60}")


def cmd_help(args=None):
    """Show help for all commands or a specific category."""
    category = args[0] if args else None

    print(f"\n  {TOOLKIT_CONFIG['agent_name']} Toolkit v{TOOLKIT_CONFIG['version']}")
    print(f"  Usage: python toolkit.py <command> [args]")
    print()

    all_cats = ('identity', 'social', 'platform', 'search', 'memory', 'comms', 'dashboard', 'viz', 'calibration', 'experiment')
    if category and category not in all_cats:
        print(f"  Unknown category: {category}")
        print(f"  Available: {', '.join(all_cats)}")
        return

    categories = {}
    for key, cmd in sorted(REGISTRY.items()):
        if category and cmd.category != category:
            continue
        categories.setdefault(cmd.category, []).append(cmd)

    for cat, cmds in categories.items():
        print(f"  [{cat.upper()}]")
        for cmd in cmds:
            full_name = f"{cat}:{cmd.name}"
            args_str = f" {cmd.args_help}" if cmd.args_help else ""
            print(f"    {full_name:<30s} {cmd.description}{args_str}")
        print()

    print("  [SPECIAL]")
    print("    status                         Full system dashboard")
    print("    health                         Component health check")
    print("    help [category]                Show this help")
    print()


def resolve_command(name):
    """Resolve a command name to a registry entry."""
    # Exact match: category:command
    if name in REGISTRY:
        return REGISTRY[name]

    # Bare command — find unambiguous match
    matches = [cmd for key, cmd in REGISTRY.items() if cmd.name == name]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        cats = [m.category for m in matches]
        print(f"Ambiguous command '{name}'. Found in: {', '.join(cats)}")
        print(f"Use: {' | '.join(f'{c}:{name}' for c in cats)}")
        return None

    # Partial match
    partials = [cmd for key, cmd in REGISTRY.items() if name in cmd.name]
    if partials:
        print(f"Unknown command '{name}'. Did you mean:")
        for cmd in partials[:5]:
            print(f"  {cmd.category}:{cmd.name}")
    else:
        print(f"Unknown command '{name}'. Run 'python toolkit.py help' for available commands.")
    return None


def main():
    # Windows encoding fix — only wrap if not already UTF-8
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer') and getattr(sys.stderr, 'encoding', '').lower() != 'utf-8':
        import io
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    show_traceback = '--traceback' in sys.argv
    if show_traceback:
        sys.argv.remove('--traceback')

    args = sys.argv[1:]
    if not args or args[0] in ('help', '-h', '--help'):
        cmd_help(args[1:] if len(args) > 1 else None)
        return

    cmd_name = args[0]
    cmd_args = args[1:]

    # Special commands
    if cmd_name == 'status':
        cmd_status()
        return
    if cmd_name == 'health':
        cmd_health()
        return

    # Resolve and dispatch
    cmd = resolve_command(cmd_name)
    if cmd is None:
        sys.exit(1)

    try:
        dispatch_to_module(cmd, cmd_args)
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print(f"  This command requires module: {cmd.module}")
        sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[ERROR] {cmd.category}:{cmd.name} failed: {e}")
        if show_traceback:
            tb.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
