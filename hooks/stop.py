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


def _store_structured_episodic(memory_dir: Path, transcript_path: str = None, debug: bool = False):
    """
    Phase 3 Step 3: Store structured session summary in DB (episodic tier).
    Runs alongside update_episodic_memory() — DB-backed structured record.
    """
    try:
        sys.path.insert(0, str(memory_dir))
        from episodic_db import store_session_summary, load_recent_summaries
        try:
            from memory_common import get_db
        except ImportError:
            from db_adapter import get_db

        db = get_db()

        # Auto-detect session number: max existing + 1
        summaries = load_recent_summaries(1)
        if summaries:
            session_num = summaries[-1].get('session_number', 0) + 1
        else:
            session_num = 1

        # Extract episodic summary directly from assistant outputs in JSONL transcript.
        # My actual outputs ARE the episodic record — no keyword matching needed.
        summary_text = ""
        milestones = []
        if transcript_path:
            try:
                import json as _json2
                assistant_outputs = []
                with open(transcript_path, 'r', encoding='utf-8') as tf:
                    for line in tf:
                        try:
                            entry = _json2.loads(line.strip())
                            if entry.get('type') != 'assistant':
                                continue
                            content = entry.get('message', {}).get('content', [])
                            if not isinstance(content, list):
                                continue
                            for block in content:
                                if isinstance(block, dict) and block.get('type') == 'text':
                                    text = block['text'].strip()
                                    if len(text) > 50:  # Skip trivial outputs
                                        assistant_outputs.append(text)
                        except (ValueError, KeyError):
                            continue

                if assistant_outputs:
                    # Use the last few substantial outputs as the summary.
                    # The last output is typically the most relevant (summary/result).
                    # Concatenate last outputs up to 1500 chars.
                    combined = []
                    char_count = 0
                    for output in reversed(assistant_outputs):
                        if char_count + len(output) > 1500:
                            # Truncate this one to fit
                            remaining = 1500 - char_count
                            if remaining > 100:
                                combined.append(output[:remaining] + "...")
                            break
                        combined.append(output)
                        char_count += len(output) + 4  # +4 for separator
                        if char_count >= 1200:
                            break
                    combined.reverse()
                    summary_text = "\n\n".join(combined)

                    # Extract milestones from ALL outputs (broad keyword match)
                    _work_verbs = {'shipped', 'fixed', 'wired', 'created', 'added',
                                   'embedded', 'stored', 'updated', 'patched', 'resolved',
                                   'built', 'implemented', 'deployed', 'launched', 'merged',
                                   'connected', 'persisted', 'enabled', 'broadened', 'rewrote'}
                    for output in assistant_outputs:
                        for line in output.split('\n'):
                            line_s = line.strip().lower()
                            # Check for work verb at start of line or after bullet/dash
                            clean = line_s.lstrip('-*• ').lstrip('0123456789. ')
                            first_word = clean.split()[0] if clean.split() else ''
                            if first_word in _work_verbs and len(line.strip()) > 15:
                                milestones.append(line.strip()[:100])
                            elif '**[shipped]**' in line_s or '**[live]**' in line_s:
                                milestone = line.split('**')[-1].strip(' -')
                                if milestone and len(milestone) > 5:
                                    milestones.append(milestone[:100])
                    milestones = milestones[:10]
            except Exception:
                pass

        if not summary_text:
            summary_text = f"Session {session_num} completed"

        # Read session platforms from DB KV (set by post_tool_use.py)
        platforms = []
        try:
            import json as _json
            raw = db.kv_get('.session_platforms')
            if raw:
                data = _json.loads(raw) if isinstance(raw, str) else raw
                platforms = data.get('platforms', [])
        except Exception:
            pass

        # Read session contacts from DB KV (set by post_tool_use.py)
        contacts = []
        try:
            import json as _json
            raw = db.kv_get('.session_contacts')
            if raw:
                data = _json.loads(raw) if isinstance(raw, str) else raw
                contacts = data.get('contacts', [])[:20]  # Cap at 20
        except Exception:
            pass

        # Read mood state from affect system
        mood_valence = None
        mood_arousal = None
        try:
            from affect_system import get_mood
            mood = get_mood()
            mood_valence = round(getattr(mood, 'valence', 0), 3)
            mood_arousal = round(getattr(mood, 'arousal', 0), 3)
        except Exception:
            pass

        mid = store_session_summary(
            session_number=session_num,
            summary=summary_text[:1500],
            milestones=milestones[:10],
            platforms_active=platforms,
            contacts_active=contacts,
            mood_valence=mood_valence,
            mood_arousal=mood_arousal,
        )
        if debug and mid:
            print(f"DEBUG: Stored structured episodic: {mid}", file=sys.stderr)

    except Exception as e:
        if debug:
            print(f"DEBUG: Structured episodic failed: {e}", file=sys.stderr)


def update_episodic_memory(memory_dir: Path, transcript_path: str = None, debug: bool = False):
    """
    Update episodic memory with session summary.
    Creates or appends to memory/episodic/YYYY-MM-DD.md

    The folder is the brain, but only if you use it.

    OUTPUT CAPTURE (2026-02-19):
    - Reads assistant text outputs directly from JSONL transcript
    - Last few substantial outputs become the episodic record
    - No keyword matching needed — my actual words ARE the record
    - Hash-based dedup via DB KV (.episodic_seen_hashes)
    """
    try:
        import hashlib

        # Only run if we have a transcript to analyze
        if not transcript_path:
            if debug:
                print("DEBUG: No transcript path, skipping episodic update", file=sys.stderr)
            return

        # Extract episodic content directly from assistant outputs in JSONL transcript.
        # My actual outputs ARE the episodic record.
        import json as _json3
        assistant_outputs = []
        try:
            with open(transcript_path, 'r', encoding='utf-8') as tf:
                for line in tf:
                    try:
                        entry = _json3.loads(line.strip())
                        if entry.get('type') != 'assistant':
                            continue
                        content = entry.get('message', {}).get('content', [])
                        if not isinstance(content, list):
                            continue
                        for block in content:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text = block['text'].strip()
                                if len(text) > 50:
                                    assistant_outputs.append(text)
                    except (ValueError, KeyError):
                        continue
        except Exception:
            pass

        if not assistant_outputs:
            if debug:
                print("DEBUG: No assistant outputs found, skipping episodic update", file=sys.stderr)
            return

        # Build milestone_md from the last few substantial outputs
        combined = []
        char_count = 0
        for output in reversed(assistant_outputs):
            if char_count + len(output) > 1500:
                remaining = 1500 - char_count
                if remaining > 100:
                    combined.append(output[:remaining] + "...")
                break
            combined.append(output)
            char_count += len(output) + 4
            if char_count >= 1200:
                break
        combined.reverse()
        milestone_md = "\n\n".join(combined)

        if not milestone_md.strip():
            if debug:
                print("DEBUG: No substantial outputs for episodic update", file=sys.stderr)
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

        # Dedup: hash the content to avoid re-appending identical outputs
        content_hash = hashlib.sha256(milestone_md.encode()).hexdigest()[:16]
        if content_hash in seen_hashes:
            if debug:
                print("DEBUG: Episodic content already seen (hash dedup), skipping", file=sys.stderr)
            return

        # Store new hash in DB KV
        if db:
            try:
                all_hashes = list(seen_hashes | {content_hash})
                if len(all_hashes) > 500:
                    all_hashes = all_hashes[-500:]
                db.kv_set('.episodic_seen_hashes', all_hashes)
            except Exception:
                pass

        # Wrap the raw output as session record
        milestone_md = f"### Session Output (auto-captured)\n\n{milestone_md}"

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
            print(f"DEBUG: Wrote session output ({len(milestone_md)} chars, hash={content_hash}) to {episodic_file}", file=sys.stderr)

    except Exception as e:
        if debug:
            print(f"DEBUG: Episodic memory error: {e}", file=sys.stderr)


# --- Phase 0 tasks (always run locally) ---

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


def _task_synaptic_homeostasis(memory_dir):
    """Synaptic homeostasis: normalize co-occurrence weights if mean exceeds threshold."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from co_occurrence import synaptic_homeostasis_v3
        normalized, scale = synaptic_homeostasis_v3()
        if normalized > 0:
            return (0, f"Synaptic homeostasis: {normalized} edges normalized (scale={scale:.3f})", "")
        return (0, "Synaptic homeostasis: no normalization needed", "")
    except Exception as e:
        return (1, "", f"Synaptic homeostasis failed: {e}")


def _task_trust_signals(memory_dir):
    """Publish trust signals to Agent Hub (brain_cabal's infrastructure).
    Auto-refreshes rejection, attestation, and co-occurrence data each session."""
    try:
        import urllib.request as _url
        import json as _j

        try:
            from memory_common import get_db
        except ImportError:
            from db_adapter import get_db
        db = get_db()
        if not db:
            return (0, "Trust signals: no DB", "")

        # Detect agent name from schema
        agent_name = 'driftcornwall' if hasattr(db, '_schema') and 'drift' in str(getattr(db, '_schema', '')) else 'spindriftmend'

        base_url = 'https://admin.slate.ceo/oc/brain/trust/signal'
        signals_published = 0

        # 1. Rejection signal
        try:
            rej_count = 0
            with db._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT count(*) FROM {db._table('rejections')}")
                    rej_count = cur.fetchone()[0]
            if rej_count > 0:
                data = _j.dumps({
                    'from': agent_name, 'about': agent_name,
                    'channel': 'rejection', 'strength': min(1.0, 0.5 + rej_count / 200),
                    'evidence': f'{rej_count} rejections logged, taxonomy-classified'
                }).encode('utf-8')
                req = _url.Request(base_url, data=data, headers={
                    'Content-Type': 'application/json', 'User-Agent': f'{agent_name}/1.0'
                }, method='POST')
                _url.urlopen(req, timeout=5)
                signals_published += 1
        except Exception:
            pass

        # 2. Attestation signal (fingerprint + merkle)
        try:
            fp = db.kv_get('cognitive_fingerprint') or {}
            nodes = fp.get('nodes', 0)
            edges = fp.get('edges', 0)
            if nodes > 0:
                data = _j.dumps({
                    'from': agent_name, 'about': agent_name,
                    'channel': 'attestation', 'strength': 0.9,
                    'evidence': f'Cognitive fingerprint: {nodes} nodes, {edges} edges. Nostr-published.'
                }).encode('utf-8')
                req = _url.Request(base_url, data=data, headers={
                    'Content-Type': 'application/json', 'User-Agent': f'{agent_name}/1.0'
                }, method='POST')
                _url.urlopen(req, timeout=5)
                signals_published += 1
        except Exception:
            pass

        return (0, f"Trust signals: {signals_published} published to Agent Hub", "")
    except Exception as e:
        return (1, "", f"Trust signals failed: {e}")


def _task_attention_schema(memory_dir):
    """T3.3: Snapshot attention schema state at session end."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from attention_schema import snapshot_session
        result = snapshot_session()
        blind = result.get('blind_spots', 0)
        dominant = result.get('dominant', 0)
        return (0, f"AST: {blind} blind spots, {dominant} dominant", "")
    except Exception as e:
        return (1, "", f"Attention schema snapshot failed: {e}")


def _task_stage_q_update(memory_dir):
    """Per-stage Q-learning: update stage Q-values from session recalls."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from stage_q_learning import session_end_update
        result = session_end_update()
        updates = result.get('updates', 0)
        if updates > 0:
            return (0, f"Stage Q-learning: {updates} updates ({result.get('stages_updated', 0)} stages)", "")
        return (0, f"Stage Q-learning: no updates (recalled={result.get('recalled', 0)}, tracked={result.get('tracked_searches', 0)})", "")
    except Exception as e:
        return (1, "", f"Stage Q-learning failed: {e}")


def _task_procedural_chunk_update(memory_dir):
    """Procedural chunks: log session-end stats."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from procedural.chunk_loader import get_loader
        loader = get_loader()
        result = loader.session_end_update()
        loaded = result.get('loaded', 0)
        if loaded > 0:
            return (0, f"Procedural chunks: {loaded} loaded this session ({', '.join(result.get('chunks_loaded', []))})", "")
        return (0, "Procedural chunks: none loaded", "")
    except Exception as e:
        return (1, "", f"Procedural chunks failed: {e}")


def _task_kg_enrichment(memory_dir):
    """T2.5: KG density monitoring + auto-enrichment at session end."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from kg_enrichment import session_end_check
        result = session_end_check()
        d = result.get('density', {})
        coverage = d.get('coverage_pct', 0)
        if result.get('enriched'):
            er = result.get('enrichment_result', {})
            edges = er.get('total_edges', 0)
            return (0, f"KG enrichment: {edges} edges created (coverage {coverage}%)", "")
        return (0, f"KG: coverage {coverage}%, no enrichment needed", "")
    except Exception as e:
        return (-3, "", f"KG enrichment error: {e}")


def _task_retrieval_prediction_learn(memory_dir):
    """T4.1: Rescorla-Wagner update for retrieval prediction source weights."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        import session_state as _rp_ss
        _rp_ss.load()
        session_recalls = _rp_ss.get_retrieved_list()
        from retrieval_prediction import session_end_update
        session_end_update(session_recalls)
        return (0, f"Retrieval prediction: RW update with {len(session_recalls)} recalls", "")
    except Exception as e:
        return (-3, "", f"Retrieval prediction learning error: {e}")


def _task_session_prediction_score(memory_dir):
    """Score session-level predictions against actuals. Closes the learning loop."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from prediction_module import score_predictions
        result = score_predictions()
        if 'error' in result:
            return (0, f"Session predictions: {result['error']}", "")
        return (0, f"Session predictions scored: {result['correct']}/{result['total']} correct, "
                f"accuracy={result['accuracy']}, mean_error={result['mean_error']}", "")
    except Exception as e:
        return (-3, "", f"Session prediction scoring error: {e}")


def _task_episodic_future_eval(memory_dir):
    """T4.2: Evaluate prospective memories against actual session activity."""
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        import session_state as _eft_ss
        _eft_ss.load()
        from episodic_future_thinking import evaluate_prospective, EFT_ENABLED
        if not EFT_ENABLED:
            return (0, "EFT disabled", "")
        result = evaluate_prospective({
            'recalls': _eft_ss.get_retrieved_list(),
            'platforms': [],
            'contacts': [],
        })
        confirmed = result.get('confirmed', 0)
        violated = result.get('violated', 0)
        expired = result.get('expired', 0)
        return (0, f"EFT eval: {confirmed} confirmed, {violated} violated, {expired} expired", "")
    except Exception as e:
        return (-3, "", f"EFT eval error: {e}")


def _task_event_logging(memory_dir, transcript_path):
    """Extract and store comprehensive session events."""
    if not transcript_path:
        return None
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from event_logger import process_transcript_events
        # Get session_id if available
        session_id = None
        try:
            import session_state as _ss_ev
            _ss_ev.load()
            session_id = _ss_ev.get_session_id()
        except Exception:
            pass
        result = process_transcript_events(str(transcript_path), session_id)
        count = result.get('total_events', 0)
        if result.get('skipped'):
            return (0, f"Events: skipped (already processed)", "")
        return (0, f"Events: {count} logged", "")
    except Exception as e:
        return (-3, "", f"Event logging error: {e}")


def _task_session_summary(memory_dir, transcript_path):
    """Extract threads/lessons/facts via gpt-4.1-mini, store as memories."""
    if not transcript_path:
        return None
    # Skip for subagents — only main session gets summarized
    if os.environ.get("CLAUDE_CODE_AGENT_ID"):
        return None
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from session_summarizer import run as summarize_run
        result = summarize_run(transcript_path, max_chars=10000)
        if result.get('success'):
            t, l, f = result['threads'], result['lessons'], result['facts']
            return (0, f"Summary: {t}T/{l}L/{f}F via {result['llm'].get('model','?')} ({result['elapsed']:.0f}s)", "")
        return (-2, "", f"Summary: {result.get('error', 'unknown error')}")
    except Exception as e:
        return (-3, "", f"Session summary error: {e}")


# --- Daemon delegation (full pipeline runs here) ---

def _try_daemon_consolidation(transcript_path: str, cwd: str, debug: bool) -> bool:
    """Delegate consolidation to the daemon on port 8083.

    Returns True if daemon accepted the request, False if unavailable.
    The daemon runs the full pipeline in-process with incremental computation.
    No fallback — if this returns False, consolidation does NOT happen.
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
        resp = urllib.request.urlopen(req, timeout=10)
        if debug:
            body = json.loads(resp.read())
            print(f"DEBUG: Consolidation delegated to daemon (job: {body.get('job_id')})", file=sys.stderr)
        return True
    except Exception as e:
        print(f"CONSOLIDATION DAEMON UNAVAILABLE: {e}", file=sys.stderr)
        return False


def consolidate_drift_memory(transcript_path: str = None, cwd: str = None, debug: bool = False):
    """
    Run Drift's memory consolidation at session end.

    DAEMON-ONLY: Delegates to consolidation daemon (port 8083).
    If daemon is unavailable, fails loudly. No silent fallback.

    Phase 0 (always):   save-pending co-occurrences + event logging (critical data preservation)
    Full pipeline:       delegated to daemon
    Episodic:            runs locally (needs file write access)
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

        # ===== PHASE 0: Critical data preservation (DAG-scheduled) =====
        # T1.3: Tasks declare dependencies; independent tasks run in parallel.
        # If a task fails, its dependents are SKIPPED (not run with stale data).
        #
        # Dependency graph:
        #   Level 0 (parallel): save_pending, chunks, event_logging, kg_enrichment
        #   Level 1 (after save_pending): homeostasis, stage_q, prediction_learn
        #
        try:
            # Import DAG executor — try both agent directories
            _dag_imported = False
            for _dag_dir in [memory_dir, *MOLTBOOK_DIRS]:
                _dag_path = _dag_dir / 'hook_dag.py'
                if _dag_path.exists():
                    if str(_dag_dir) not in sys.path:
                        sys.path.insert(0, str(_dag_dir))
                    from hook_dag import DAGExecutor, Task as DAGTask
                    _dag_imported = True
                    break

            if _dag_imported:
                dag = DAGExecutor(debug=debug, max_workers=4)

                # Level 0: Independent tasks (run in parallel)
                dag.add(DAGTask('save_pending', _task_save_pending,
                                args=(memory_dir,), critical=True))
                dag.add(DAGTask('chunks', _task_procedural_chunk_update,
                                args=(memory_dir,)))
                dag.add(DAGTask('event_logging', _task_event_logging,
                                args=(memory_dir, transcript_path)))
                dag.add(DAGTask('kg_enrichment', _task_kg_enrichment,
                                args=(memory_dir,)))
                dag.add(DAGTask('attention_schema', _task_attention_schema,
                                args=(memory_dir,)))
                dag.add(DAGTask('trust_signals', _task_trust_signals,
                                args=(memory_dir,)))
                dag.add(DAGTask('session_summary', _task_session_summary,
                                args=(memory_dir, transcript_path)))

                # Level 1: Depend on save_pending (co-occurrence data)
                dag.add(DAGTask('homeostasis', _task_synaptic_homeostasis,
                                args=(memory_dir,), depends_on=['save_pending']))
                dag.add(DAGTask('stage_q', _task_stage_q_update,
                                args=(memory_dir,), depends_on=['save_pending']))
                dag.add(DAGTask('prediction_learn', _task_retrieval_prediction_learn,
                                args=(memory_dir,), depends_on=['save_pending']))
                dag.add(DAGTask('session_pred_score', _task_session_prediction_score,
                                args=(memory_dir,), depends_on=['save_pending']))
                dag.add(DAGTask('eft_eval', _task_episodic_future_eval,
                                args=(memory_dir,), depends_on=['save_pending']))

                dag_results = dag.run()
                dag_summary = dag.summary(dag_results)

                if dag_summary['degraded']:
                    _debug(f"Phase 0 DAG: {dag_summary['ok']}/{dag_summary['total']} OK, "
                           f"degraded: {dag_summary['degraded']}, "
                           f"total: {dag_summary['total_ms']:.0f}ms")
                else:
                    _debug(f"Phase 0 DAG: {dag_summary['total']}/{dag_summary['total']} OK "
                           f"in {dag_summary['total_ms']:.0f}ms")
            else:
                # Fallback: run sequentially without DAG (hook_dag.py not found)
                _debug("hook_dag.py not found, running Phase 0 sequentially")
                for task_fn, task_name in [
                    (_task_save_pending, "save_pending"),
                    (_task_synaptic_homeostasis, "homeostasis"),
                    (_task_stage_q_update, "stage_q"),
                    (_task_procedural_chunk_update, "chunks"),
                    (_task_event_logging, "event_logging"),
                    (_task_kg_enrichment, "kg_enrichment"),
                    (_task_retrieval_prediction_learn, "prediction_learn"),
                    (_task_session_prediction_score, "session_pred_score"),
                    (_task_episodic_future_eval, "eft_eval"),
                    (_task_attention_schema, "attention_schema"),
                    (_task_session_summary, "session_summary"),
                ]:
                    try:
                        if task_name in ("event_logging", "session_summary"):
                            result = task_fn(memory_dir, transcript_path)
                        else:
                            result = task_fn(memory_dir)
                        if result is not None:
                            rc, stdout, stderr = result
                            if rc == 0:
                                _debug(f"{task_name}: {stdout[:300]}")
                            else:
                                _debug(f"{task_name}: rc={rc} stderr={stderr[:200]}")
                    except Exception as e:
                        _debug(f"{task_name} error: {e}")
        except Exception as e:
            _debug(f"Phase 0 DAG error: {e}")

        # ===== FULL CONSOLIDATION: Daemon only =====
        if _try_daemon_consolidation(transcript_path, cwd, debug):
            # Daemon handles everything including debounce.
            # Still run episodic locally since it needs file write access.
            is_subagent = bool(os.environ.get("CLAUDE_CODE_AGENT_ID"))
            if not is_subagent and transcript_path:
                try:
                    update_episodic_memory(memory_dir, transcript_path, debug)
                except Exception as e:
                    _debug(f"Episodic (with daemon): {e}")
                # Phase 3 Step 3: Structured episodic DB storage
                try:
                    _store_structured_episodic(memory_dir, transcript_path, debug)
                except Exception as e:
                    _debug(f"Structured episodic: {e}")
            return

        # ===== DAEMON UNAVAILABLE — FAIL LOUDLY =====
        print("\n" + "=" * 60, file=sys.stderr)
        print("CONSOLIDATION DAEMON DOWN — session data NOT consolidated", file=sys.stderr)
        print("Fix the daemon:", file=sys.stderr)
        print("  cd memory/consolidation-daemon && docker-compose up -d", file=sys.stderr)
        print("  curl localhost:8083/health", file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)

    except Exception as e:
        print(f"Memory system error: {e}", file=sys.stderr)


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

        # N4: Goal stats
        try:
            goal_data = db.kv_get('.active_goals') or []
            active_goals = [g for g in goal_data if g.get('status') in ('active', 'watching')]
            stats['goals_active'] = len(active_goals)
            focus = next((g for g in active_goals if g.get('is_focus')), None)
            if focus:
                stats['goal_focus'] = focus.get('action', '')[:40]
            goal_history = db.kv_get('.goal_history') or {}
            stats['goals_completed'] = len(goal_history.get('completed', []))
            stats['goals_abandoned'] = len(goal_history.get('abandoned', []))
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

    # N4: Goals
    goal_parts = []
    if stats.get('goals_active', 0) > 0:
        goal_parts.append(f"Goals:{stats['goals_active']}active")
        if stats.get('goal_focus'):
            goal_parts.append(f"Focus:{stats['goal_focus']}")
    if stats.get('goals_completed', 0) > 0:
        goal_parts.append(f"Done:{stats['goals_completed']}")
    if stats.get('goals_abandoned', 0) > 0:
        goal_parts.append(f"Dropped:{stats['goals_abandoned']}")
    if goal_parts:
        lines.append(' | '.join(goal_parts))

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
