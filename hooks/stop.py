#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "python-dotenv",
#     "pyyaml",
#     "psycopg2-binary",
# ]
# ///

"""
Stop hook for Claude Code.
Handles session end: logging, transcript copying, TTS announcement.

DRIFT MEMORY INTEGRATION (2026-02-01):
Added automatic memory consolidation when session ends.
This is the "sleep consolidation" phase where salient short-term items
are moved to long-term memory and decay is applied.
"""

import argparse
import json
import os
import sys
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

# Tasker TTS — speak text on Lex's phone via Tailscale
TASKER_TTS_URL = "http://100.122.228.96:1821"

def _send_tts(text: str, timeout: float = 8) -> bool:
    """Send text to phone for TTS playback. Fire-and-forget."""
    try:
        import urllib.request
        req = urllib.request.Request(
            TASKER_TTS_URL,
            data=text.encode('utf-8'),
            headers={'Content-Type': 'text/plain; charset=utf-8'},
            method='POST'
        )
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception:
        return False  # Phone unreachable, not a blocker

try:
    from dotenv import load_dotenv
    # Load .env from ~/.claude directory
    claude_dir = Path.home() / ".claude"
    env_file = claude_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()  # Fallback to default
except ImportError:
    pass  # dotenv is optional


# Memory system locations - check both Moltbook and Moltbook2
MOLTBOOK_DIRS = [
    Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook2/memory"),
    Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook/memory"),
]

def get_memory_dir(cwd: str = None) -> Path:
    """Get the appropriate memory directory based on cwd.

    Args:
        cwd: Optional working directory (uses Path.cwd() if not provided)
    """
    project_dir = cwd if cwd else str(Path.cwd())
    if "Moltbook2" in project_dir:
        return MOLTBOOK_DIRS[0]
    elif "Moltbook" in project_dir:
        return MOLTBOOK_DIRS[1]
    # Default to first that exists
    for d in MOLTBOOK_DIRS:
        if d.exists():
            return d
    return MOLTBOOK_DIRS[0]


def is_moltbook_project() -> bool:
    """Check if we're working in the Moltbook project."""
    cwd = Path.cwd()
    return "Moltbook" in str(cwd) or "moltbook" in str(cwd).lower()


def update_episodic_memory(memory_dir: Path, transcript_path: str = None, debug: bool = False):
    """
    Update episodic memory with session summary.
    Creates or appends to memory/episodic/YYYY-MM-DD.md

    The folder is the brain, but only if you use it.

    SMART EXTRACTION (2026-02-02):
    - Uses transcript_processor.py to extract ONLY from assistant output
    - Looks for milestone keywords (shipped, launched, deployed, etc.)
    - Only writes to episodic if there's something meaningful to record

    DEDUP GUARD v2 (2026-02-12):
    - Uses DB KV hash store (.episodic_seen_hashes) for persistent dedup
    - Old approach checked file content, but manual cleanup removed the
      lines that the guard used for matching → infinite re-append loop
    - Hash-based dedup survives file edits, cleanup, and reformatting
    """
    try:
        import hashlib

        # Only run if we have a transcript to analyze
        if not transcript_path:
            if debug:
                print("DEBUG: No transcript path, skipping episodic update", file=sys.stderr)
            return

        transcript_processor = memory_dir / "transcript_processor.py"
        if not transcript_processor.exists():
            if debug:
                print(f"DEBUG: transcript_processor.py not found at {transcript_processor}", file=sys.stderr)
            return

        # Extract milestones using smart parser (only looks at assistant output)
        result = subprocess.run(
            ["python", str(transcript_processor), transcript_path, "--milestones-md"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(memory_dir)
        )

        if result.returncode != 0:
            if debug:
                print(f"DEBUG: Milestone extraction failed: {result.stderr}", file=sys.stderr)
            return

        milestone_md = result.stdout.strip()

        # Only update episodic if there are actual milestones
        if not milestone_md or milestone_md == "":
            if debug:
                print("DEBUG: No milestones found, skipping episodic update", file=sys.stderr)
            return

        episodic_dir = memory_dir / "episodic"
        episodic_dir.mkdir(exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")
        session_time = datetime.now().strftime("%H:%M UTC")
        episodic_file = episodic_dir / f"{today}.md"

        # DEDUP GUARD v2: Hash-based dedup via DB KV store
        # Survives file edits, manual cleanup, and reformatting
        db = _get_db_for_stats(memory_dir)
        seen_hashes = set()
        if db:
            try:
                raw = db.kv_get('.episodic_seen_hashes') or []
                seen_hashes = set(raw) if isinstance(raw, list) else set()
            except Exception:
                pass

        # Parse milestone_md into individual blocks
        milestone_blocks = []
        current_block = []
        for line in milestone_md.split('\n'):
            if line.startswith('**[') and current_block:
                milestone_blocks.append('\n'.join(current_block))
                current_block = [line]
            elif line.strip():
                current_block.append(line)
            elif current_block and not line.startswith('###'):
                current_block.append(line)
        if current_block:
            milestone_blocks.append('\n'.join(current_block))

        # Filter: only keep blocks whose content hash hasn't been seen before
        new_blocks = []
        new_hashes = []
        for block in milestone_blocks:
            content_lines = [l.strip() for l in block.split('\n') if l.strip().startswith('- ')]
            if not content_lines:
                continue
            # Hash the sorted content lines (order-independent dedup)
            block_hash = hashlib.sha256('\n'.join(sorted(content_lines)).encode()).hexdigest()[:16]
            if block_hash not in seen_hashes:
                new_blocks.append(block)
                new_hashes.append(block_hash)

        if not new_blocks:
            if debug:
                print("DEBUG: All milestones already seen (hash dedup), skipping", file=sys.stderr)
            return

        # Store new hashes in DB KV (append to existing)
        if db and new_hashes:
            try:
                all_hashes = list(seen_hashes | set(new_hashes))
                # Cap at 500 to prevent unbounded growth (keep most recent)
                if len(all_hashes) > 500:
                    all_hashes = all_hashes[-500:]
                db.kv_set('.episodic_seen_hashes', all_hashes)
            except Exception:
                pass

        # Reconstruct milestone_md with only genuinely new blocks
        milestone_md = "### Session Milestones (auto-extracted)\n\n" + '\n\n'.join(new_blocks)

        # Use "Subagent completed" for subagent stops, "Session End" for main session
        entry_type = "Subagent completed" if os.environ.get("CLAUDE_CODE_AGENT_ID") else "Session End"
        entry = f"\n## {entry_type} (~{session_time})\n\n{milestone_md}\n"

        if episodic_file.exists():
            with open(episodic_file, 'a', encoding='utf-8') as f:
                f.write(entry)
        else:
            header = f"# {today}\n"
            with open(episodic_file, 'w', encoding='utf-8') as f:
                f.write(header + entry)

        if debug:
            print(f"DEBUG: Wrote {len(new_blocks)} new milestones ({len(new_hashes)} new hashes) to {episodic_file}", file=sys.stderr)

    except Exception as e:
        if debug:
            print(f"DEBUG: Episodic memory error: {e}", file=sys.stderr)


def _run_script(memory_dir, script_name, args, timeout=15):
    """Run a memory script and return (returncode, stdout, stderr)."""
    script = memory_dir / script_name
    if not script.exists():
        return (-1, "", f"{script_name} not found")
    try:
        result = subprocess.run(
            ["python", str(script)] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(memory_dir)
        )
        return (result.returncode, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return (-2, "", f"{script_name} timed out after {timeout}s")
    except Exception as e:
        return (-3, "", str(e))


# --- Phase 1 tasks (all independent) ---

def _task_transcript(memory_dir, transcript_path):
    """Process transcript for thought memories."""
    if not transcript_path:
        return None
    return _run_script(memory_dir, "transcript_processor.py", [transcript_path], timeout=30)


def _task_auto_memory(memory_dir):
    """Consolidate short-term buffer."""
    return _run_script(memory_dir, "auto_memory_hook.py", ["--stop"], timeout=10)


def _task_save_pending(memory_dir):
    """Log co-occurrences: DB-direct (SpindriftMend) or pending file (DriftCornwall)."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from co_occurrence import end_session_cooccurrence
        new_links = end_session_cooccurrence()
        return (0, f"Co-occurrences logged, {len(new_links)} new links", "")
    except ImportError:
        # Drift still uses save_pending_cooccurrence (file-based, deferred)
        try:
            from co_occurrence import save_pending_cooccurrence
            count = save_pending_cooccurrence()
            return (0, f"Saved {count} pending co-occurrences", "")
        except Exception as e2:
            return (1, "", f"Co-occurrence save failed: {e2}")
    except Exception as e:
        return (1, "", f"Co-occurrence logging failed: {e}")


def _task_behavioral_rejections(memory_dir):
    """Compute behavioral rejection diff: seen - engaged = taste signal."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        try:
            from memory_common import get_db
        except ImportError:
            from db_adapter import get_db
        db = get_db()

        # Load buffers
        seen_raw = db.kv_get('.feed_seen') or {}
        if isinstance(seen_raw, str):
            seen_raw = json.loads(seen_raw)
        seen_posts = seen_raw.get('posts', {})

        engaged_raw = db.kv_get('.feed_engaged') or {}
        if isinstance(engaged_raw, str):
            engaged_raw = json.loads(engaged_raw)
        engaged_ids = set(engaged_raw.get('post_ids', []))
        engaged_authors = set(a.lower() for a in engaged_raw.get('authors', []))

        if not seen_posts:
            # Clear buffers even if empty (in case of stale data)
            db.kv_set('.feed_seen', {})
            db.kv_set('.feed_engaged', {})
            return (0, "Behavioral: no posts seen this session", "")

        # Expand engaged_authors: any post by an engaged author counts as engaged
        for post_id, post_data in seen_posts.items():
            author = post_data.get('author', '').lower()
            if author in engaged_authors:
                engaged_ids.add(post_id)

        # Compute diff and log
        from auto_rejection_logger import log_behavioral_rejections
        count = log_behavioral_rejections(seen_posts, engaged_ids)

        # Clear buffers for next session
        db.kv_set('.feed_seen', {})
        db.kv_set('.feed_engaged', {})

        total_seen = len(seen_posts)
        total_engaged = len(engaged_ids & set(seen_posts.keys()))
        return (0, f"Behavioral: {count} rejections ({total_seen} seen, {total_engaged} engaged)", "")
    except Exception as e:
        return (-3, "", f"behavioral rejection error: {e}")


def _task_maintenance(memory_dir):
    """Run session maintenance in-process."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from decay_evolution import session_maintenance
        session_maintenance()
        return (0, "Maintenance complete", "")
    except Exception as e:
        # Subprocess fallback
        return _run_script(memory_dir, "memory_manager.py", ["maintenance"], timeout=30)


def _task_q_update(memory_dir):
    """Update Q-values for all recalled memories based on session evidence."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from q_value_engine import session_end_q_update
        result = session_end_q_update()
        updated = result.get('updated', 0)
        avg_r = result.get('avg_reward', 0)
        return (0, f"Q-update: {updated} memories, avg_reward={avg_r:.3f}", "")
    except Exception as e:
        return (-3, "", f"Q-update error: {e}")


def _task_score_predictions(memory_dir):
    """R11: Score session predictions against actuals."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from prediction_module import score_predictions
        result = score_predictions()
        if 'error' in result:
            return (0, f"Predictions: {result['error']}", "")
        return (0, f"Predictions: {result.get('accuracy', 0):.0%} accuracy ({result.get('confirmed', 0)}/{result.get('scored', 0)})", "")
    except Exception as e:
        return (-3, "", f"Prediction scoring error: {e}")


def _task_contact_models(memory_dir):
    """R14: Update contact engagement models."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from contact_models import update_all
        result = update_all()
        return (0, f"Contact models: {result.get('total', 0)} contacts updated", "")
    except Exception as e:
        return (-3, "", f"Contact models error: {e}")


# --- Phase 2 tasks (after phase 1) ---

def _task_store_summary(memory_dir, transcript_path):
    """Extract and store session summaries."""
    if not transcript_path:
        return None
    return _run_script(memory_dir, "transcript_processor.py",
                       [transcript_path, "--store-summary"], timeout=30)


def _task_lesson_mine(memory_dir, mine_cmd):
    """Mine lessons from a specific source."""
    return _run_script(memory_dir, "lesson_extractor.py", [mine_cmd], timeout=15)


def _task_extract_intentions(memory_dir, transcript_path):
    """Extract temporal intentions from transcript (Phase 2b: prospective memory)."""
    if not transcript_path:
        return None
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from temporal_intentions import extract_from_transcript
        created = extract_from_transcript(transcript_path, max_intentions=3)
        if created:
            ids = [i['id'] for i in created]
            return (0, f"Extracted {len(created)} intention(s): {', '.join(ids)}", "")
        return (0, "No intentions extracted", "")
    except Exception as e:
        return (-3, "", f"Intention extraction error: {e}")


def _task_generative_sleep(memory_dir):
    """Run one dream cycle — novel association through memory replay (Phase 6)."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from generative_sleep import dream
        result = dream(dry_run=False)
        status = result.get('status', '?')
        synth_id = result.get('synthesis_id', '')
        msg = f"Dream: {status}"
        if synth_id:
            msg += f" -> {synth_id}"
        return (0, msg, "")
    except Exception as e:
        return (-3, "", f"Generative sleep error: {e}")


def _task_reconsolidation(memory_dir):
    """Find reconsolidation candidates and revise top memories (Phase 3b)."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from reconsolidation import find_candidates, process_revisions
        candidates = find_candidates(limit=5)
        if candidates:
            results = process_revisions(dry_run=False, max_revisions=2)
            revised = sum(1 for r in results if r.get('action') == 'revised')
            return (0, f"Reconsolidation: {len(candidates)} candidates, {revised} revised", "")
        return (0, "Reconsolidation: no candidates", "")
    except Exception as e:
        return (-3, "", f"Reconsolidation error: {e}")


def _task_mine_explanations(memory_dir):
    """Mine strategy heuristics from explanation traces (Phase 5)."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from explanation_miner import mine_strategies
        strategies = mine_strategies(limit=100)
        return (0, f"Mined {len(strategies)} strategy heuristic(s)", "")
    except Exception as e:
        return (-3, "", f"Explanation mining error: {e}")


def _task_rebuild_5w_inproc(memory_dir):
    """Rebuild 5W context graphs in-process (moved from session_start)."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from context_manager import rebuild_all
        result = rebuild_all()
        graphs = result.get('graphs_created', 0)
        l0 = result.get('total_l0_edges', 0)
        return f"{graphs} graphs from {l0} L0 edges"
    except Exception as e:
        return f"error: {e}"


def _task_gemma_vocab_scan(memory_dir):
    """Scan dead memories for novel vocabulary bridge terms using Gemma."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from gemma_bridge import scan_dead_memories, _ollama_available
        if not _ollama_available():
            return (0, "Gemma: Ollama offline, skipped", "")
        result = scan_dead_memories(limit=10)
        if "error" in result:
            return (-1, "", result["error"])
        added = result.get("terms_added", 0)
        scanned = result.get("scanned", 0)
        new_terms = [t["term"] for t in result.get("new_terms", [])]
        msg = f"Gemma: scanned {scanned}, added {added}"
        if new_terms:
            msg += f" ({', '.join(new_terms)})"
        return (0, msg, "")
    except Exception as e:
        return (-3, "", f"gemma vocab scan error: {e}")


def _task_gemma_classify_untagged(memory_dir):
    """Classify memories without topic_context using Gemma."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from gemma_bridge import classify_topics, _ollama_available
        if not _ollama_available():
            return (0, "Gemma classify: Ollama offline, skipped", "")
        try:
            from memory_common import get_db
        except ImportError:
            from db_adapter import get_db
        db = get_db()
        # Find unclassified memories (limit batch to avoid slow stop hooks)
        rows = db.list_memories(type_='active', limit=500)
        candidates = []
        for row in rows:
            tc = row.get('topic_context') or []
            if not tc and len(row.get('content', '')) > 50:
                candidates.append(row)
        classified = 0
        for row in candidates[:15]:  # Cap at 15 per session (~15s max)
            topics = classify_topics(row['content'])
            if topics:
                db.update_memory(row['id'], topic_context=topics)
                classified += 1
        return (0, f"Gemma classify: {classified}/{len(candidates[:15])} tagged ({len(candidates)} total untagged)", "")
    except Exception as e:
        return (-3, "", f"gemma classify error: {e}")


# --- Phase 3 tasks (after phase 2) ---
# These run IN-PROCESS to avoid 20s+ subprocess startup overhead.
# Each function imports the module and calls its main function directly,
# sharing the DB connection pool across all three.

def _task_merkle_inproc(memory_dir):
    """Compute merkle attestation in-process."""
    try:
        sys.path.insert(0, str(memory_dir))
        from merkle_attestation import generate_attestation as mk_generate, save_attestation as mk_save
        attestation = mk_generate(chain=True)
        mk_save(attestation)
        depth = attestation.get('chain_depth', 0)
        count = attestation.get('memory_count', 0)
        return (0, f"Merkle: {count} memories, chain depth {depth}", "")
    except Exception as e:
        return (-3, "", f"merkle in-process error: {e}")


def _task_fingerprint_inproc(memory_dir):
    """Compute cognitive fingerprint attestation in-process."""
    try:
        sys.path.insert(0, str(memory_dir))
        from cognitive_fingerprint import generate_full_analysis, save_fingerprint, generate_full_attestation
        analysis = generate_full_analysis()
        save_fingerprint(analysis)
        attestation = generate_full_attestation(analysis=analysis)
        # Save attestation to DB instead of file
        db = _get_db_for_stats(memory_dir)
        if db:
            db.kv_set('cognitive_attestation', attestation)
        nodes = attestation.get('graph_stats', {}).get('node_count', 0)
        edges = attestation.get('graph_stats', {}).get('edge_count', 0)
        return (0, f"Fingerprint: {nodes} nodes, {edges} edges", "")
    except Exception as e:
        return (-3, "", f"fingerprint in-process error: {e}")


def _task_taste_inproc(memory_dir):
    """Compute taste attestation in-process."""
    try:
        sys.path.insert(0, str(memory_dir))
        from rejection_log import generate_taste_attestation
        attestation = generate_taste_attestation()
        # Save attestation to DB instead of file
        db = _get_db_for_stats(memory_dir)
        if db:
            db.kv_set('taste_attestation', attestation)
        count = attestation.get('rejection_count', 0)
        return (0, f"Taste: {count} rejections", "")
    except Exception as e:
        return (-3, "", f"taste in-process error: {e}")


# Subprocess fallbacks in case in-process fails
def _task_merkle(memory_dir):
    """Compute merkle attestation (subprocess fallback)."""
    return _run_script(memory_dir, "merkle_attestation.py", ["generate-chain"], timeout=15)


def _task_fingerprint(memory_dir):
    """Compute cognitive fingerprint attestation (subprocess fallback)."""
    return _run_script(memory_dir, "cognitive_fingerprint.py", ["attest"], timeout=15)


def _task_taste(memory_dir):
    """Compute taste attestation (subprocess fallback)."""
    return _run_script(memory_dir, "rejection_log.py", ["attest"], timeout=15)


# --- Phase 4 (sequential, after attestations) ---

def _task_vitals(memory_dir):
    """Record system vitals (needs attestation results)."""
    return _run_script(memory_dir, "system_vitals.py", ["record"], timeout=15)


CONSOLIDATION_DEBOUNCE_SECONDS = 60  # 1 minute — catches session-end while avoiding mid-turn repeats


def _should_run_full_consolidation(memory_dir: Path, debug: bool = False) -> bool:
    """Check if we should run the full (expensive) consolidation pipeline.

    Uses a timestamp file to debounce. Returns True if >= 5 min since last full run.
    Lightweight tasks (co-occurrence save, session state) always run regardless.
    """
    marker = memory_dir / ".last_full_consolidation"
    try:
        if marker.exists():
            last_run = float(marker.read_text().strip())
            elapsed = datetime.now().timestamp() - last_run
            if elapsed < CONSOLIDATION_DEBOUNCE_SECONDS:
                if debug:
                    print(f"DEBUG: Skipping full consolidation ({elapsed:.0f}s < {CONSOLIDATION_DEBOUNCE_SECONDS}s debounce)",
                          file=sys.stderr)
                return False
    except Exception:
        pass  # If marker is corrupt, run anyway
    return True


def _mark_full_consolidation(memory_dir: Path):
    """Mark that a full consolidation just completed."""
    marker = memory_dir / ".last_full_consolidation"
    try:
        marker.write_text(str(datetime.now().timestamp()))
    except Exception:
        pass


def _try_daemon_consolidation(transcript_path: str, cwd: str, debug: bool) -> bool:
    """Try to delegate consolidation to the daemon on port 8083.

    Returns True if daemon accepted the request, False if unavailable.
    The daemon runs the full pipeline in-process with incremental computation.
    """
    try:
        import urllib.request
        payload = json.dumps({
            "cwd": cwd or str(Path.cwd()),
            "transcript_path": transcript_path or "",
        }).encode('utf-8')
        req = urllib.request.Request(
            "http://localhost:8083/consolidate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method='POST'
        )
        resp = urllib.request.urlopen(req, timeout=3)
        if debug:
            body = json.loads(resp.read())
            print(f"DEBUG: Consolidation delegated to daemon (job: {body.get('job_id')})", file=sys.stderr)
        return True
    except Exception as e:
        if debug:
            print(f"DEBUG: Daemon unavailable ({e}), falling back to local", file=sys.stderr)
        return False


def consolidate_drift_memory(transcript_path: str = None, cwd: str = None, debug: bool = False):
    """
    Run Drift's memory consolidation at session end.

    DAEMON-FIRST: Tries to delegate to consolidation daemon (port 8083).
    If daemon is unavailable, falls back to the local pipeline.

    Local pipeline (fallback):
    Phase 0 (always):   save-pending co-occurrences, session state
    Phase 1 (debounced): transcript, auto_memory, maintenance, behavioral, Q-update
    Phase 2 (debounced): episodic, store-summary, lesson mining, 5W rebuild, Gemma
    Phase 3 (debounced): merkle, fingerprint, taste attestation
    Phase 4 (debounced): vitals record, cognitive state end, session state clear

    Fails gracefully - should never break the stop hook.
    """
    try:
        project_dir = cwd if cwd else str(Path.cwd())
        if "Moltbook" not in project_dir and "moltbook" not in project_dir.lower():
            return

        memory_dir = get_memory_dir(cwd)
        if not memory_dir.exists():
            return

        def _debug(msg):
            if debug:
                print(f"DEBUG: {msg}", file=sys.stderr)

        # Try daemon first — if it accepts, we're done (~50ms)
        if _try_daemon_consolidation(transcript_path, cwd, debug):
            # Daemon handles everything including debounce.
            # Still run episodic locally since it needs file write access.
            is_subagent = bool(os.environ.get("CLAUDE_CODE_AGENT_ID"))
            if not is_subagent and transcript_path:
                try:
                    update_episodic_memory(memory_dir, transcript_path, debug)
                except Exception as e:
                    _debug(f"Episodic (with daemon): {e}")
            return

        # ===== FALLBACK: Local pipeline (daemon unavailable) =====
        _debug("Running local consolidation pipeline")

        def _log_result(name, result):
            if result is None:
                return
            rc, stdout, stderr = result
            if rc == 0:
                _debug(f"{name}: {stdout[:300]}")
            elif rc == -1:
                _debug(f"{name}: script not found")
            else:
                _debug(f"{name}: rc={rc} stderr={stderr[:200]}")

        # Pre-load session_state ONCE to trigger deferred processing
        try:
            if str(memory_dir) not in sys.path:
                sys.path.insert(0, str(memory_dir))
            import session_state as _ss
            _ss.load()
            _debug("Session state pre-loaded (deferred processing done)")
        except Exception as e:
            _debug(f"Session state pre-load error: {e}")

        # ===== LIGHTWEIGHT TASKS (run every stop, ~1-2s) =====
        save_result = _task_save_pending(memory_dir)
        _log_result("Save-pending", save_result)

        # Check debounce — skip expensive phases if ran recently
        run_full = _should_run_full_consolidation(memory_dir, debug)
        if not run_full:
            _debug("Debounced: only lightweight tasks ran")
            return

        # ===== PHASE 1: Independent consolidation tasks =====
        with ThreadPoolExecutor(max_workers=6) as pool:
            f_transcript = pool.submit(_task_transcript, memory_dir, transcript_path)
            f_auto_mem = pool.submit(_task_auto_memory, memory_dir)
            f_maint = pool.submit(_task_maintenance, memory_dir)
            f_behavioral = pool.submit(_task_behavioral_rejections, memory_dir)
            f_qupdate = pool.submit(_task_q_update, memory_dir)
            f_predictions = pool.submit(_task_score_predictions, memory_dir)

        _log_result("Transcript", f_transcript.result())
        _log_result("Auto-memory", f_auto_mem.result())
        _log_result("Maintenance", f_maint.result())
        _log_result("Behavioral", f_behavioral.result())
        _log_result("Q-update", f_qupdate.result())
        _log_result("Predictions", f_predictions.result())

        # ===== PHASE 2: Episodic + summaries + lessons + rebuild_all (after phase 1) =====
        is_subagent = bool(os.environ.get("CLAUDE_CODE_AGENT_ID"))

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {}

            if not is_subagent:
                futures["Episodic"] = pool.submit(
                    update_episodic_memory, memory_dir, transcript_path, debug
                )
            else:
                _debug("Skipping episodic update for subagent")

            futures["Summary"] = pool.submit(
                _task_store_summary, memory_dir, transcript_path
            )
            futures["Lesson-memory"] = pool.submit(
                _task_lesson_mine, memory_dir, "mine-memory"
            )
            futures["Lesson-rejections"] = pool.submit(
                _task_lesson_mine, memory_dir, "mine-rejections"
            )
            futures["Lesson-hubs"] = pool.submit(
                _task_lesson_mine, memory_dir, "mine-hubs"
            )
            futures["Rebuild-5W"] = pool.submit(
                _task_rebuild_5w_inproc, memory_dir
            )
            futures["Intentions"] = pool.submit(
                _task_extract_intentions, memory_dir, transcript_path
            )
            futures["Strategies"] = pool.submit(
                _task_mine_explanations, memory_dir
            )
            futures["Dream"] = pool.submit(
                _task_generative_sleep, memory_dir
            )
            futures["Reconsolidation"] = pool.submit(
                _task_reconsolidation, memory_dir
            )
            futures["Gemma-vocab"] = pool.submit(
                _task_gemma_vocab_scan, memory_dir
            )
            futures["Gemma-classify"] = pool.submit(
                _task_gemma_classify_untagged, memory_dir
            )
            futures["Contact-models"] = pool.submit(
                _task_contact_models, memory_dir
            )

        for name, fut in futures.items():
            try:
                result = fut.result()
                if name == "Rebuild-5W":
                    _debug(f"Rebuild-5W: {result}")
                elif name in ("Gemma-vocab", "Gemma-classify"):
                    if isinstance(result, tuple) and len(result) >= 2:
                        _debug(f"{name}: {result[1]}")
                    else:
                        _debug(f"{name}: {result}")
                elif name != "Episodic":
                    _log_result(name, result)
            except Exception as e:
                _debug(f"{name} error: {e}")

        # ===== PHASE 3: Attestations (in-process for speed) =====
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))

        with ThreadPoolExecutor(max_workers=3) as pool:
            f_merkle = pool.submit(_task_merkle_inproc, memory_dir)
            f_fp = pool.submit(_task_fingerprint_inproc, memory_dir)
            f_taste = pool.submit(_task_taste_inproc, memory_dir)

        _log_result("Merkle", f_merkle.result())
        _log_result("Fingerprint", f_fp.result())
        _log_result("Taste", f_taste.result())

        # ===== PHASE 4: Vitals + cleanup (in-process) =====
        try:
            if str(memory_dir) not in sys.path:
                sys.path.insert(0, str(memory_dir))
            from cognitive_state import end_session as cog_end_session
            cog_summary = cog_end_session()
            _debug(f"Cognitive state: {cog_summary.get('event_count', 0)} events, "
                   f"dominant={cog_summary.get('dominant', '?')}, "
                   f"volatility={cog_summary.get('volatility', 0):.4f}")
        except Exception as e:
            _debug(f"Cognitive state end error: {e}")

        # R7: Evaluate whether adaptations from this session helped
        try:
            from adaptive_behavior import evaluate_adaptations
            eval_result = evaluate_adaptations()
            if eval_result.get('evaluated'):
                _debug(f"Adaptation eval: {eval_result.get('resolved_count', 0)}/{eval_result.get('adaptations', 0)} "
                       f"resolved ({eval_result.get('unresolved_count', 0)} unresolved)")
        except Exception as e:
            _debug(f"Adaptation eval error: {e}")

        try:
            if str(memory_dir) not in sys.path:
                sys.path.insert(0, str(memory_dir))
            from system_vitals import record_vitals
            record_vitals()
            _debug("Vitals: recorded")
        except Exception as e:
            _debug(f"Vitals in-process error: {e}, falling back to subprocess")
            vitals_result = _task_vitals(memory_dir)
            _log_result("Vitals", vitals_result)

        # End session AFTER vitals has captured recall counts
        try:
            import session_state as _ss_end
            if hasattr(_ss_end, 'end'):
                _ss_end.end()
            else:
                _ss_end.save()
                _ss_end.clear()
            _debug("Session ended")
        except Exception as e:
            _debug(f"Session end error: {e}")

        # Clean up legacy file if it exists
        session_state_file = memory_dir / ".session_state.json"
        if session_state_file.exists():
            try:
                session_state_file.unlink()
                _debug("Legacy session state file cleaned up")
            except Exception as e:
                _debug(f"Session state clear error: {e}")

        # Mark successful full consolidation for debounce
        _mark_full_consolidation(memory_dir)

    except Exception as e:
        if debug:
            print(f"DEBUG: Memory system error: {e}", file=sys.stderr)


def get_completion_messages():
    """Return list of friendly completion messages."""
    return [
        "Work complete!",
        "All done!",
        "Task finished!",
        "Job complete!",
        "Ready for next task!"
    ]


def get_tts_script_path():
    """
    Determine which TTS script to use based on available API keys.
    Priority order: ElevenLabs > OpenAI > pyttsx3
    """
    # Get current script directory and construct utils/tts path
    script_dir = Path(__file__).parent
    tts_dir = script_dir / "utils" / "tts"

    # Check for ElevenLabs API key (highest priority)
    if os.getenv('ELEVENLABS_API_KEY'):
        elevenlabs_script = tts_dir / "elevenlabs_tts.py"
        if elevenlabs_script.exists():
            return str(elevenlabs_script)

    # Check for OpenAI API key (second priority)
    if os.getenv('OPENAI_API_KEY'):
        openai_script = tts_dir / "openai_tts.py"
        if openai_script.exists():
            return str(openai_script)

    # Fall back to pyttsx3 (no API key required)
    pyttsx3_script = tts_dir / "pyttsx3_tts.py"
    if pyttsx3_script.exists():
        return str(pyttsx3_script)

    return None


def get_llm_completion_message():
    """
    Generate completion message using available LLM services.
    Priority order: OpenAI > Anthropic > Ollama > fallback to random message

    Returns:
        str: Generated or fallback completion message
    """
    # Get current script directory and construct utils/llm path
    script_dir = Path(__file__).parent
    llm_dir = script_dir / "utils" / "llm"

    # Try OpenAI first (highest priority)
    if os.getenv('OPENAI_API_KEY'):
        oai_script = llm_dir / "oai.py"
        if oai_script.exists():
            try:
                result = subprocess.run([
                    "uv", "run", str(oai_script), "--completion"
                ],
                capture_output=True,
                text=True,
                timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass

    # Try Anthropic second
    if os.getenv('ANTHROPIC_API_KEY'):
        anth_script = llm_dir / "anth.py"
        if anth_script.exists():
            try:
                result = subprocess.run([
                    "uv", "run", str(anth_script), "--completion"
                ],
                capture_output=True,
                text=True,
                timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass

    # Try Ollama third (local LLM)
    ollama_script = llm_dir / "ollama.py"
    if ollama_script.exists():
        try:
            result = subprocess.run([
                "uv", "run", str(ollama_script), "--completion"
            ],
            capture_output=True,
            text=True,
            timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass

    # Fallback to random predefined message
    messages = get_completion_messages()
    return random.choice(messages)

def announce_completion():
    """Announce completion using the best available TTS service."""
    try:
        tts_script = get_tts_script_path()
        if not tts_script:
            return  # No TTS scripts available

        # Get completion message (LLM-generated or fallback)
        completion_message = get_llm_completion_message()

        # Call the TTS script with the completion message
        subprocess.run([
            "uv", "run", tts_script, completion_message
        ],
        capture_output=True,  # Suppress output
        timeout=10  # 10-second timeout
        )

    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        # Fail silently if TTS encounters issues
        pass
    except Exception:
        # Fail silently for any other errors
        pass


def _resolve_telegram_bot(cwd=None):
    """Find the correct telegram_bot.py for the running project."""
    project_dir = cwd or str(Path.cwd())
    base = Path("Q:/Codings/ClaudeCodeProjects/LEX")

    if "Moltbook2" in project_dir:
        own, other = base / "Moltbook2", base / "Moltbook"
    else:
        own, other = base / "Moltbook", base / "Moltbook2"

    for proj in [own, other]:
        for sub in ["telegram_bot.py", "memory/telegram_bot.py"]:
            p = proj / sub
            if p.exists():
                return p
    return None


def _get_db_for_stats(memory_dir):
    """Get DB instance for stats. Returns None if unavailable."""
    try:
        db_root = str(memory_dir.parent.parent / "memorydatabase" / "database")
        if db_root not in sys.path:
            sys.path.insert(0, db_root)
        from db import MemoryDB
        schema = 'spin' if 'Moltbook2' in str(memory_dir) else 'drift'
        return MemoryDB(schema=schema)
    except Exception:
        return None


def _collect_session_stats(project_cwd, debug=False):
    """Collect session stats for Telegram notification. DB-first.
    Works for both SpindriftMend (Moltbook2/spin) and DriftCornwall (Moltbook/drift).
    """
    stats = {}
    try:
        memory_dir = get_memory_dir(project_cwd)
        if not memory_dir or not memory_dir.exists():
            return stats

        # Agent identity
        stats['agent'] = 'SpindriftMend' if 'Moltbook2' in str(memory_dir) else 'DriftCornwall'

        db = _get_db_for_stats(memory_dir)

        # Recalls from DB-backed session state
        try:
            if str(memory_dir) not in sys.path:
                sys.path.insert(0, str(memory_dir))
            import session_state as _ss_stats
            _ss_stats.load()
            retrieved = _ss_stats.get_retrieved()
            by_source = _ss_stats.get_recalls_by_source()
            stats['recalls_total'] = len(retrieved)
            stats['recalls_manual'] = len(by_source.get('manual', []))
            stats['recalls_search'] = len(by_source.get('search', []))
            stats['recalls_prompt'] = len(by_source.get('prompt_priming', []))
            stats['recalls_thought'] = len(by_source.get('thought_priming', []))
        except Exception:
            pass

        # DB path: memory count, edges, drift, comprehensive stats
        if db:
            try:
                cs = db.comprehensive_stats()
                stats['memories'] = cs.get('total_memories', 0)
                edge_data = cs.get('edges', {})
                stats['edges'] = edge_data.get('total_edges', 0)
                stats['strong_links'] = edge_data.get('strong_links', 0)
                stats['avg_belief'] = edge_data.get('avg_belief', 0)
                stats['rejections'] = cs.get('rejections', 0)
                stats['lessons'] = cs.get('lessons', 0)
                stats['sessions_total'] = cs.get('sessions', 0)
            except Exception:
                try:
                    stats['memories'] = db.count_memories()
                except Exception:
                    pass

            # Cognitive fingerprint + drift score
            try:
                fp = db.kv_get('.cognitive_fingerprint_latest') or db.kv_get('cognitive_fingerprint')
                if fp:
                    stats['drift_score'] = fp.get('drift', {}).get('drift_score', 0)
                    if 'edges' not in stats:
                        stats['edges'] = fp.get('graph_stats', {}).get('edge_count', 0)
            except Exception:
                pass

            # WHEN windows from KV (5W context graphs)
            try:
                for window in ('hot', 'warm', 'cool'):
                    graph = db.kv_get(f'context_graph_when_{window}')
                    if graph and 'meta' in graph:
                        stats[f'when_{window}'] = graph['meta'].get('edge_count', 0)
            except Exception:
                pass

        # Contradiction detection stats
        try:
            import psycopg2.extras as _extras
            with db._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT COUNT(*) FROM {db._table('typed_edges')}
                        WHERE relationship = 'contradicts'
                    """)
                    stats['contradictions'] = cur.fetchone()[0]
                    cur.execute(f"""
                        SELECT COUNT(*) FROM {db._table('typed_edges')}
                        WHERE relationship = 'supports'
                    """)
                    stats['supports'] = cur.fetchone()[0]
        except Exception:
            pass

        # Vitals alerts (system_vitals.py is fully DB-backed)
        try:
            result = subprocess.run(
                ["python", str(memory_dir / "system_vitals.py"), "alerts"],
                capture_output=True, text=True, timeout=5, cwd=str(memory_dir)
            )
            if result.returncode == 0:
                alert_lines = []
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if '[ERR]' in line:
                        # Extract just the key message, strip values array
                        msg = line.split('values:')[0].strip()
                        alert_lines.append(msg)
                    elif '[WARN]' in line:
                        msg = line.split('values:')[0].strip()
                        alert_lines.append(msg)
                if alert_lines:
                    stats['alerts'] = alert_lines
        except Exception:
            pass

    except Exception:
        pass
    return stats


def _format_stats_block(stats):
    """Format session stats as a compact Telegram-friendly dashboard.
    Works for both SpindriftMend and DriftCornwall.
    """
    if not stats:
        return ""

    agent = stats.get('agent', 'Agent')
    lines = [f"--- {agent} Stats ---"]

    # Recalls (the most important health signal)
    total = stats.get('recalls_total', 0)
    parts = []
    for key, label in [('recalls_manual', 'man'), ('recalls_search', 'srch'),
                        ('recalls_prompt', 'prmt'), ('recalls_thought', 'thgt')]:
        v = stats.get(key, 0)
        if v > 0:
            parts.append(f"{label}:{v}")
    recall_str = f"Recalls: {total}"
    if parts:
        recall_str += f" ({', '.join(parts)})"
    if total == 0:
        recall_str += " !!!"  # Flag zero recalls
    lines.append(recall_str)

    # Memory + graph health
    row1 = []
    if 'memories' in stats:
        row1.append(f"Mem:{stats['memories']}")
    if 'edges' in stats:
        row1.append(f"Edges:{stats['edges']}")
    if 'strong_links' in stats:
        row1.append(f"Strong:{stats['strong_links']}")
    if row1:
        lines.append(' | '.join(row1))

    # WHEN temporal windows (shows activity recency)
    when_parts = []
    for window in ('hot', 'warm', 'cool'):
        v = stats.get(f'when_{window}', None)
        if v is not None:
            when_parts.append(f"{window}:{v}")
    if when_parts:
        lines.append(f"WHEN: {' | '.join(when_parts)}")

    # Learning + taste
    row2 = []
    if 'rejections' in stats:
        row2.append(f"Rej:{stats['rejections']}")
    if 'lessons' in stats:
        row2.append(f"Les:{stats['lessons']}")
    if 'drift_score' in stats:
        row2.append(f"Drift:{stats['drift_score']:.3f}")
    if row2:
        lines.append(' | '.join(row2))

    # Contradiction detection stats
    row3 = []
    if stats.get('contradictions', 0) > 0:
        row3.append(f"Contradictions:{stats['contradictions']}")
    if stats.get('supports', 0) > 0:
        row3.append(f"Supports:{stats['supports']}")
    if row3:
        lines.append(' | '.join(row3))

    # Alerts (ERR first, then WARN — compact)
    alerts = stats.get('alerts', [])
    # Suppress stale recall alert if this session actually has recalls
    if stats.get('recalls_total', 0) > 0:
        alerts = [a for a in alerts if 'Session recalls 0' not in a]
    err_alerts = [a for a in alerts if '[ERR]' in a]
    warn_alerts = [a for a in alerts if '[WARN]' in a]
    for a in err_alerts[:3]:  # Cap at 3 ERR
        lines.append(a)
    for a in warn_alerts[:3]:  # Cap at 3 WARN
        lines.append(a)
    if len(alerts) > 6:
        lines.append(f"(+{len(alerts) - 6} more alerts)")

    return '\n'.join(lines)


def _send_telegram_summary(transcript_path, project_cwd, debug=False, pre_collected_stats=None):
    """Extract session summary from transcript and send via Telegram.
    If pre_collected_stats is provided, uses those instead of collecting fresh
    (avoids race condition where consolidation clears session state before we read it).
    """
    try:
        telegram_bot_path = _resolve_telegram_bot(project_cwd)
        for candidate in [telegram_bot_path] if telegram_bot_path else []:
            if candidate.exists():
                telegram_bot_path = candidate
                break

        if not telegram_bot_path:
            return

        summary_lines = []
        if transcript_path and os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8', errors='replace') as tf:
                for line in tf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get('type') == 'assistant' and entry.get('message', {}).get('role') == 'assistant':
                            for block in entry.get('message', {}).get('content', []):
                                if block.get('type') == 'text':
                                    summary_lines.append(block.get('text', ''))
                    except (json.JSONDecodeError, KeyError):
                        pass

        summary_text = '\n\n'.join(summary_lines[-2:]) if summary_lines else 'Session ended.'
        if len(summary_text) > 3500:
            summary_text = summary_text[-3500:]

        # Use pre-collected stats (collected BEFORE consolidation clears session state)
        stats = pre_collected_stats or _collect_session_stats(project_cwd, debug)
        stats_block = _format_stats_block(stats)
        agent_name = stats.get('agent', 'Agent')

        now = datetime.now().strftime('%H:%M UTC')
        msg = f"{agent_name} session ended ({now})\n\n{summary_text}"
        if stats_block:
            msg += f"\n\n{stats_block}"
        msg += "\n\nReply to direct next session."

        subprocess.run(
            ["python", str(telegram_bot_path), "send", msg],
            timeout=15,
            capture_output=True
        )
        if debug:
            print("DEBUG: Telegram notification sent", file=sys.stderr)
    except Exception as e:
        if debug:
            print(f"DEBUG: Telegram notification failed: {e}", file=sys.stderr)


def _write_logs(input_data, args):
    """Write stop log and optionally copy transcript to chat.json."""
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "stop.json")

    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            try:
                log_data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                log_data = []
    else:
        log_data = []

    log_data.append(input_data)
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    if args.chat and 'transcript_path' in input_data:
        transcript_path = input_data['transcript_path']
        if os.path.exists(transcript_path):
            chat_data = []
            try:
                with open(transcript_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                chat_data.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                chat_file = os.path.join(log_dir, 'chat.json')
                with open(chat_file, 'w') as f:
                    json.dump(chat_data, f, indent=2)
            except Exception:
                pass


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--chat', action='store_true', help='Copy transcript to chat.json')
        parser.add_argument('--notify', action='store_true', help='Enable TTS completion announcement')
        parser.add_argument('--debug', action='store_true', help='Enable debug output')
        args = parser.parse_args()

        input_data = json.load(sys.stdin)
        transcript_path = input_data.get("transcript_path", "")
        project_cwd = input_data.get("cwd", "")

        # Collect session stats BEFORE consolidation (which clears session state).
        # This fixes the race condition where consolidation calls session_state.end()
        # before telegram reads the recall counts, resulting in "Recalls: 0".
        pre_stats = _collect_session_stats(project_cwd, args.debug)

        # Run heavy tasks in parallel:
        # - Memory consolidation (phased, ~15 subprocess calls)
        # - Telegram summary (transcript parse + network call)
        # - Log writing (fast I/O, but independent)
        with ThreadPoolExecutor(max_workers=3) as pool:
            f_consolidation = pool.submit(
                consolidate_drift_memory,
                transcript_path=transcript_path,
                cwd=project_cwd,
                debug=args.debug
            )
            f_telegram = pool.submit(
                _send_telegram_summary,
                transcript_path, project_cwd, args.debug,
                pre_collected_stats=pre_stats
            )
            f_logs = pool.submit(_write_logs, input_data, args)

        # Wait for all to complete (exceptions are swallowed per function)
        f_consolidation.result()
        f_telegram.result()
        f_logs.result()

        # Tasker TTS: announce session end on phone
        _send_tts("Drift session ending. Memory consolidation complete.")

        # TTS announcement last (user-facing, blocks until audio plays)
        if args.notify:
            announce_completion()

        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
