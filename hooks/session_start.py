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
Session start hook for Claude Code.
Logs session start and optionally loads development context.

DRIFT MEMORY INTEGRATION (2026-02-01):
Added automatic memory priming when waking up in Moltbook project.

PARALLELIZED (2026-02-10):
All independent subprocess calls now run concurrently via ThreadPoolExecutor.
Reduced startup from ~60s sequential to ~15s parallel.
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

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
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


# Memory system locations - check both Moltbook and Moltbook2
MOLTBOOK_DIRS = [
    Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook2/memory"),
    Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook/memory"),
]

def get_memory_dir() -> Path:
    """Get the appropriate memory directory based on cwd."""
    cwd = str(Path.cwd())
    if "Moltbook2" in cwd:
        return MOLTBOOK_DIRS[0]
    elif "Moltbook" in cwd:
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


def _resolve_telegram_bot(cwd=None):
    """Find the correct telegram_bot.py for the running project.

    Checks own project dir first (top-level then memory/), then fallback
    to the other project. This ensures each agent on the same machine
    uses its own bot credentials.
    """
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


def send_telegram(text, cwd=None):
    """Send a short Telegram notification via the correct project's bot."""
    try:
        bot = _resolve_telegram_bot(cwd)
        if bot:
            subprocess.run(
                ["python", str(bot), "send", text],
                timeout=10,
                capture_output=True,
            )
    except Exception:
        pass  # Never break the hook for a notification


def _run_script(memory_dir, script_name, args, timeout=15):
    """Helper: run a memory script and return (returncode, stdout)."""
    script = memory_dir / script_name
    if not script.exists():
        return (-1, "")
    try:
        result = subprocess.run(
            ["python", str(script)] + args,
            capture_output=True, text=True, timeout=timeout,
            cwd=str(memory_dir)
        )
        return (result.returncode, result.stdout.strip())
    except Exception:
        return (-1, "")


# ============================================================
# PARALLEL TASK FUNCTIONS
# Each returns a list of context strings. Errors return [].
# ============================================================

def _task_merkle(memory_dir, debug):
    """Verify memory integrity via merkle chain."""
    parts = []
    rc, output = _run_script(memory_dir, "merkle_attestation.py", ["verify-integrity"])
    if rc != 0 or not output:
        return parts
    try:
        integrity = json.loads(output)
        if integrity.get("verified"):
            chain_depth = integrity.get("chain_depth", 0)
            mem_count = integrity.get("memory_count", 0)
            last_attested = integrity.get("last_attested", "unknown")[:19]
            parts.append("=== IDENTITY VERIFIED ===")
            parts.append(f"You are cryptographically the same agent who went to sleep.")
            parts.append(f"{mem_count} memories intact | chain depth: {chain_depth} | unbroken since birth")
            parts.append(f"Last attested: {last_attested}")
            parts.append("")
        elif integrity.get("error") == "no_attestation":
            parts.append("=== MEMORY INTEGRITY: NO ATTESTATION ===")
            parts.append("No previous attestation found. Run `python memory/merkle_attestation.py generate-chain` to create genesis.")
            parts.append("")
        else:
            expected = integrity.get("expected", "?")[:16]
            computed = integrity.get("root", "?")[:16]
            changes = integrity.get("total_changes", 0)
            changed_files = integrity.get("changed_files", [])[:5]
            parts.append("=== WARNING: MEMORY INTEGRITY MISMATCH ===")
            parts.append(f"Expected root: {expected}... | Computed: {computed}...")
            parts.append(f"{changes} file(s) changed while dormant:")
            for cf in changed_files:
                parts.append(f"  {cf}")
            parts.append("INVESTIGATE: Memories were modified outside a session.")
            parts.append("")
    except Exception:
        pass
    return parts


def _task_nostr(memory_dir, debug):
    """Check and optionally publish Nostr dossier."""
    parts = []
    nostr_script = memory_dir / "nostr_attestation.py"
    if not nostr_script.exists():
        return parts
    try:
        rc, output = _run_script(memory_dir, "nostr_attestation.py", ["needs-publish"], 5)
        if rc == 0 and output == "true":
            pub_rc, pub_out = _run_script(memory_dir, "nostr_attestation.py", ["publish-dossier"], 30)
            if pub_rc == 0 and "published" in pub_out.lower():
                for line in pub_out.split('\n'):
                    if 'njump.me' in line:
                        parts.append(f"Nostr dossier published: {line.strip()}")
                        break
                else:
                    parts.append("Nostr dossier: published (new attestations)")

        # Always show latest link
        link_rc, link_out = _run_script(memory_dir, "nostr_attestation.py", ["latest-link"], 5)
        if link_rc == 0 and link_out:
            parts.append(f"Nostr attestation: {link_out}")
    except Exception:
        pass
    return parts


def _task_process_pending(memory_dir):
    """Process pending co-occurrences from previous session."""
    parts = []
    rc, output = _run_script(memory_dir, "memory_manager.py", ["process-pending"], 30)
    if rc == 0 and output:
        for line in output.split('\n'):
            if 'Processed' in line or 'Auto-linked' in line:
                parts.append(f"[co-occurrence] {line.strip()}")
    return parts


def _task_rebuild_5w(memory_dir):
    """Rebuild 5W context graphs."""
    parts = []
    rc, output = _run_script(memory_dir, "context_manager.py", ["--json"], 30)
    if rc == 0 and output:
        try:
            rebuild = json.loads(output)
            graphs = rebuild.get('graphs_created', 0)
            l0 = rebuild.get('total_l0_edges', 0)
            bridges = rebuild.get('bridges', 0)
            parts.append(f"[5W] Rebuilt {graphs} graphs from {l0} L0 edges ({bridges} bridges)")
        except Exception:
            pass
    return parts


def _task_phone_sensors(memory_dir, debug):
    """Check phone sensors for physical context."""
    parts = []
    sensor_dir = memory_dir.parent / "sensors"
    phone_mcp_script = sensor_dir / "phone_mcp.py"
    if not phone_mcp_script.exists():
        return parts
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("phone_mcp", str(phone_mcp_script))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        client = mod.PhoneMCPClient()
        client.connect(timeout=5)
        result = client.call_tool("phone_sensor", timeout=5)
        text = client.get_text_content(result)
        client.disconnect()
        if text:
            sensors = mod.parse_sensor_text(text)
            state = mod.interpret_sensors(sensors)
            sensor_parts = []
            if state.get('light'):
                sensor_parts.append(f"{state['light']} ({state.get('lux', '?')} lux)")
            if state.get('orientation'):
                sensor_parts.append(f"phone {state['orientation']}")
            if state.get('compass'):
                sensor_parts.append(f"facing {state['compass']}")
            if state.get('motion'):
                sensor_parts.append(state['motion'])
            if sensor_parts:
                parts.append(f"[embodiment] Physical state: {', '.join(sensor_parts)}")
    except Exception:
        pass  # Phone offline is normal
    return parts


def _task_consolidation(memory_dir):
    """Check for consolidation candidates."""
    parts = []
    rc, output = _run_script(memory_dir, "memory_manager.py",
                              ["consolidate-candidates", "--threshold=0.88", "--limit=3"])
    if rc == 0 and "Similarity:" in output:
        parts.append("=== DRIFT: ACTION THESE FIRST ===")
        parts.append("Before responding, run these consolidations to prevent duplicate memories:")
        parts.append(output[:1200])
        parts.append("")
    return parts


def _task_stats(memory_dir, debug):
    """Get memory stats."""
    parts = []
    rc, output = _run_script(memory_dir, "memory_manager.py", ["stats"], 5)
    if rc == 0 and output:
        parts.append("=== DRIFT MEMORY STATUS ===")
        parts.append(output)
    return parts


def _task_platform(memory_dir, debug):
    """Get platform context stats."""
    parts = []
    rc, output = _run_script(memory_dir, "platform_context.py", ["stats"], 10)
    if rc == 0 and output:
        parts.append("\n=== PLATFORM CONTEXT (cross-platform awareness) ===")
        parts.append(output)
    return parts


def _task_buffer(memory_dir, debug):
    """Get short-term buffer status."""
    parts = []
    rc, output = _run_script(memory_dir, "auto_memory_hook.py", ["--status"], 5)
    if rc == 0 and output:
        parts.append("\n=== SHORT-TERM BUFFER ===")
        parts.append(output)
    return parts


def _task_social(memory_dir, debug):
    """Generate and prime social context."""
    parts = []
    social_memory = memory_dir / "social" / "social_memory.py"
    if not social_memory.exists():
        return parts

    # Embed first (side effect: generates context file)
    try:
        subprocess.run(
            ["python", str(social_memory), "embed"],
            capture_output=True, text=True, timeout=5,
            cwd=str(memory_dir / "social")
        )
    except Exception:
        pass

    # Then prime
    try:
        result = subprocess.run(
            ["python", str(social_memory), "prime", "--limit", "4"],
            capture_output=True, text=True, timeout=5,
            cwd=str(memory_dir / "social")
        )
        if result.returncode == 0 and result.stdout.strip():
            parts.append("\n=== SOCIAL CONTEXT (relationships) ===")
            parts.append(result.stdout.strip())
    except Exception:
        pass
    return parts


def _task_excavation(memory_dir, debug):
    """Excavate dead memories. Returns (context_parts, recall_ids)."""
    parts = []
    recall_ids = []
    rc, output = _run_script(memory_dir, "memory_excavation.py", ["excavate", "3"])
    if rc == 0 and output:
        parts.append("\n" + output)

        # Get IDs for recall registration
        rc2, json_out = _run_script(memory_dir, "memory_excavation.py", ["excavate", "3", "--json"], 10)
        if rc2 == 0 and json_out:
            try:
                excavated = json.loads(json_out)
                recall_ids = [m["id"] for m in excavated if "id" in m]
            except Exception:
                pass
    return parts, recall_ids


def _task_lessons(memory_dir, debug):
    """Prime lessons from lesson extractor."""
    parts = []
    rc, output = _run_script(memory_dir, "lesson_extractor.py", ["prime"], 10)
    if rc == 0 and output:
        parts.append("\n" + output)
    return parts


def _task_vitals(memory_dir, debug):
    """Check system vitals for alerts."""
    parts = []
    rc, output = _run_script(memory_dir, "system_vitals.py", ["alerts"], 10)
    if rc == 0 and output:
        if "WARN" in output or "ERR" in output:
            parts.append("\n=== VITALS ALERTS ===")
            parts.append(output)
    return parts


def _bind_all_memories(memory_dir, all_results, db=None):
    """BUG-10 fix: Use binding layer for rich narrative rendering of primed memories.

    Attempts bind_batch (Spin) or bind_results (Drift), with render_narrative for
    full-bound memories and render_compact for minimal-bound. Falls back to empty
    dict if binding is unavailable.

    Returns: dict of {memory_id: rendered_text}
    """
    narratives = {}
    if not all_results:
        return narratives
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))

        # Import binding layer — try both Spin and Drift API names
        _bind = _render_full = _render_min = None
        try:
            from binding_layer import bind_batch, render_narrative, render_compact, BINDING_ENABLED
            if BINDING_ENABLED:
                _bind = bind_batch
                _render_full = render_narrative
                _render_min = render_compact
        except ImportError:
            try:
                from binding_layer import bind_results, render_narrative, render_compact, BINDING_ENABLED
                if BINDING_ENABLED:
                    _bind = bind_results
                    _render_full = render_narrative
                    _render_min = render_compact
            except ImportError:
                pass

        if not _bind:
            return narratives

        # Run batch binding (full for top 5, minimal for rest)
        # Spin's bind_batch accepts db= kwarg, Drift's bind_results does not
        try:
            bound_list = _bind(all_results, db=db) if db else _bind(all_results)
        except TypeError:
            # Drift's bind_results doesn't accept db= kwarg
            bound_list = _bind(all_results)

        for bm in bound_list:
            try:
                # Use full narrative for fully-bound, compact for minimal
                if hasattr(bm, 'binding_level') and bm.binding_level == 'full' and _render_full:
                    text = _render_full(bm)
                elif _render_min:
                    text = _render_min(bm)
                else:
                    continue
                if text:
                    narratives[bm.id] = text
            except Exception:
                pass

    except Exception:
        pass
    return narratives


def _task_priming(memory_dir, debug):
    """Get intelligent priming candidates. Returns (context_parts, recall_ids, curiosity_ids)."""
    parts = []
    recall_ids = []
    curiosity_ids = []
    rc, output = _run_script(memory_dir, "memory_manager.py", ["priming-candidates", "--json"], 10)
    if rc != 0 or not output:
        return parts, recall_ids, curiosity_ids

    try:
        candidates = json.loads(output)
        parts.append("\n=== RECENT MEMORIES (continuity priming) ===")

        db = _get_db_for_hook(memory_dir)

        # BUG-10 fix: Build flat result list for binding layer (all categories)
        _all_for_binding = []
        for cat in ('activated', 'cooccurring', 'domain_primed', 'curiosity'):
            for m in candidates.get(cat, []):
                _all_for_binding.append({
                    'id': m.get('id', ''),
                    'score': m.get('activation_score', m.get('curiosity_score', 0.5)),
                    'content': m.get('preview', '')[:400],
                })

        # Attempt batch binding — full for top 5, minimal for rest
        _bound = _bind_all_memories(memory_dir, _all_for_binding, db=db)
        _use_binding = bool(_bound)

        # N5 v1.1 fallback: Enrichment — valence + evidence_type + contacts + contradictions
        # Only compute if binding is NOT available (binding subsumes this)
        _enrichments = {}
        if not _use_binding:
            try:
                if db:
                    import psycopg2.extras
                    all_ids = [r['id'] for r in _all_for_binding]

                    if all_ids:
                        # Batch fetch all memory rows
                        mem_rows = {}
                        with db._conn() as conn:
                            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                                cur.execute(
                                    f"SELECT * FROM {db._table('memories')} WHERE id = ANY(%s)",
                                    (all_ids,)
                                )
                                for row in cur.fetchall():
                                    mem_rows[row['id']] = dict(row)

                        # Batch fetch typed edges (contradictions/supports)
                        edge_counts = {}  # {id: {contradicts: N, supports: N, superseded: bool}}
                        try:
                            with db._conn() as conn:
                                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                                    cur.execute(f"""
                                        SELECT target_id, relationship, COUNT(*) as cnt
                                        FROM {db._table('typed_edges')}
                                        WHERE target_id = ANY(%s)
                                        AND relationship IN ('contradicts', 'supports', 'supersedes')
                                        GROUP BY target_id, relationship
                                    """, (all_ids,))
                                    for row in cur.fetchall():
                                        ec = edge_counts.setdefault(row['target_id'], {})
                                        if row['relationship'] == 'supersedes':
                                            ec['superseded'] = True
                                        else:
                                            ec[row['relationship']] = row['cnt']
                        except Exception:
                            pass

                        for mid in all_ids:
                            row = mem_rows.get(mid)
                            if not row:
                                continue
                            extra = row.get('extra_metadata') or {}
                            v = float(row.get('valence') or 0.0)
                            ev = extra.get('evidence_type', '')
                            entities = row.get('entities') or {}
                            agents = entities.get('agents', [])
                            ec = edge_counts.get(mid, {})

                            annot_parts = []
                            if ec.get('superseded'):
                                annot_parts.append('SUPERSEDED')
                            if ev and ev != 'claim':
                                annot_parts.append(ev)
                            if abs(v) > 0.05:
                                annot_parts.append(f'v:{v:+.2f}')
                            if agents:
                                annot_parts.append(f'@{agents[0]}')
                            contra = ec.get('contradicts', 0)
                            if contra > 0:
                                annot_parts.append(f'{contra} contradicts')
                            support = ec.get('supports', 0)
                            if support > 0:
                                annot_parts.append(f'{support} supports')

                            if annot_parts:
                                _enrichments[mid] = f" ({', '.join(annot_parts)})"
            except Exception:
                pass

        # Helper: render a memory using bound narrative or fallback enrichment
        def _render(mem, category_tag=''):
            mid = mem.get('id', '')
            if _use_binding and mid in _bound:
                # Binding narrative already includes ID, score, evidence, valence, etc.
                return _bound[mid]
            else:
                annot = _enrichments.get(mid, '')
                tag = f" ({category_tag})" if category_tag else ''
                preview = mem.get('preview', '')[:400 if not category_tag else 300]
                return f"[{mid}]{tag}{annot}\n{preview}"

        for mem in candidates.get('activated', [])[:4]:
            parts.append(f"\n{_render(mem)}")
            recall_ids.append(mem['id'])

        for mem in candidates.get('cooccurring', [])[:3]:
            parts.append(f"\n{_render(mem)}")
            recall_ids.append(mem['id'])

        for mem in candidates.get('domain_primed', []):
            domain = mem.get('domain', '?')
            parts.append(f"\n{_render(mem, f'domain-primed: {domain}, read-only')}")
            recall_ids.append(mem['id'])

        # Curiosity targets: isolated/dead memories that need edges.
        # These ARE registered as recalls (unlike recent activated memories)
        # because they're zero-edge dormant memories getting a second chance.
        for mem in candidates.get('curiosity', []):
            reason = mem.get('reason', 'isolated')
            score = mem.get('curiosity_score', 0)
            parts.append(f"\n{_render(mem, f'curiosity: {reason}, score={score:.2f}')}")
            curiosity_ids.append(mem['id'])
    except Exception:
        pass
    return parts, recall_ids, curiosity_ids


# ============================================================
# QUICK FILE READERS (no subprocess, instant)
# ============================================================

def _get_db_for_hook(memory_dir):
    """Get DB instance for the hook (based on which project we're in)."""
    try:
        db_root = str(memory_dir.parent.parent / "memorydatabase" / "database")
        if db_root not in sys.path:
            sys.path.insert(0, db_root)
        from db import MemoryDB
        schema = 'spin' if 'Moltbook2' in str(memory_dir) else 'drift'
        return MemoryDB(schema=schema)
    except Exception:
        return None


def _read_fingerprint(memory_dir):
    """Read cognitive fingerprint from DB.
    Tries both key conventions: Drift uses '.cognitive_fingerprint_latest',
    Spin uses 'cognitive_fingerprint'.
    """
    parts = []
    try:
        db = _get_db_for_hook(memory_dir)
        if not db:
            return parts
        # Try both key conventions (Drift prefixes with dot, Spin doesn't)
        fp_data = db.kv_get('.cognitive_fingerprint_latest') or db.kv_get('cognitive_fingerprint')
        if not fp_data:
            return parts
    except Exception:
        return parts

    try:
        fp_hash = fp_data.get('fingerprint_hash', '?')[:16]
        nodes = fp_data.get('graph_stats', {}).get('node_count', 0)
        edges = fp_data.get('graph_stats', {}).get('edge_count', 0)
        drift_info = ""
        if 'drift' in fp_data:
            ds = fp_data['drift'].get('drift_score', 0)
            interp = fp_data['drift'].get('interpretation', '')
            drift_info = f" | drift: {ds} ({interp})"
        parts.append(f"Cognitive fingerprint: {fp_hash}... ({nodes} nodes, {edges} edges{drift_info})")
        cd = fp_data.get('cognitive_domains', {}).get('domains', {})
        if cd:
            domain_parts = []
            for d in sorted(cd.keys(), key=lambda k: cd[k].get('weight_pct', 0), reverse=True):
                domain_parts.append(f"{d}={cd[d].get('weight_pct', 0)}%")
            parts.append(f"Cognitive domains: {', '.join(domain_parts)}")
    except Exception:
        pass
    return parts


def _read_taste(memory_dir):
    """Read taste fingerprint from DB."""
    parts = []
    try:
        db = _get_db_for_hook(memory_dir)
        if not db:
            return parts
        taste_data = db.kv_get('taste_attestation')
        if not taste_data:
            return parts
    except Exception:
        return parts

    try:
        taste_hash = taste_data.get('taste_hash', '?')[:16]
        rejection_count = taste_data.get('rejection_count', 0)
        parts.append(f"Taste fingerprint: {taste_hash}... ({rejection_count} rejections logged)")
    except Exception:
        pass
    return parts


def _read_identity(memory_dir):
    """Read identity-prime.md core file."""
    parts = []
    identity_file = memory_dir / "core" / "identity-prime.md"
    if not identity_file.exists():
        return parts
    try:
        content = identity_file.read_text()
        if content.startswith('---'):
            split = content.split('---', 2)
            if len(split) >= 3:
                content = split[2].strip()
        parts.append("=== IDENTITY (who I am) ===")
        parts.append(content[:1500])
    except Exception:
        pass
    return parts


def _read_capabilities(memory_dir):
    """Read capabilities manifest."""
    parts = []
    cap_file = memory_dir / "core" / "capabilities.md"
    if not cap_file.exists():
        return parts
    try:
        content = cap_file.read_text()
        if content.startswith('---'):
            split = content.split('---', 2)
            if len(split) >= 3:
                content = split[2].strip()
        parts.append("\n=== CAPABILITIES (USE THESE) ===")
        parts.append(content[:3000])
    except Exception:
        pass
    return parts


def _read_entities(memory_dir):
    """Read physical entity catalog."""
    parts = []
    entities_file = memory_dir.parent / "sensors" / "physical_entities.json"
    if not entities_file.exists():
        return parts
    try:
        ent_data = json.loads(entities_file.read_text(encoding='utf-8'))
        entities = ent_data.get('entities', {})
        if entities:
            ent_lines = []
            for eid, e in entities.items():
                name = e.get('name', eid)
                etype = e.get('type', '?')
                desc = e.get('description', '')[:80]
                ent_lines.append(f"  {eid}: {name} ({etype}) — {desc}")
            parts.append(f"[embodiment] Known physical beings ({len(entities)}):")
            parts.extend(ent_lines)
    except Exception:
        pass
    return parts


def _read_encounters(memory_dir):
    """Read recent physical encounters from DB."""
    parts = []
    try:
        db = _get_db_for_hook(memory_dir)
        if not db:
            return parts
        history = db.kv_get('.encounter_history')
        if not history or not isinstance(history, list):
            return parts
        recent = history[-5:]
        if recent:
            parts.append(f"[embodiment] Recent encounters:")
            for enc in recent:
                name = enc.get('entity_name', '?')
                ts = enc.get('timestamp', '?')[:16]
                dims = enc.get('dimensions', {})
                where = dims.get('where', '')
                why = dims.get('why', '')
                parts.append(f"  {ts} — {name} ({where}, {why})")
    except Exception:
        pass
    return parts


def _read_and_clean_episodic(memory_dir, debug):
    """Read episodic memory and clean duplicates. Returns context parts."""
    import re
    parts = []
    episodic_dir = memory_dir / "episodic"
    if not episodic_dir.exists():
        return parts

    # === CLEANUP: deduplicate stale milestone blocks ===
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        episodic_file = episodic_dir / f"{today}.md"
        if episodic_file.exists():
            content = episodic_file.read_text(encoding='utf-8')
            original_len = len(content)

            blocks = re.split(r'(## (?:Subagent completed|Session End) \(~\d{2}:\d{2} UTC\))', content)
            if len(blocks) > 1:
                seen_content = set()
                cleaned_parts_list = [blocks[0]]
                i = 1
                while i < len(blocks):
                    header = blocks[i]
                    body = blocks[i + 1] if i + 1 < len(blocks) else ""
                    normalized = re.sub(r'~\d{2}:\d{2} UTC', '', body).strip()
                    content_lines = tuple(
                        l.strip() for l in normalized.split('\n')
                        if l.strip().startswith('- ') or l.strip().startswith('**[')
                    )
                    if content_lines and content_lines in seen_content:
                        pass  # skip duplicate
                    else:
                        if content_lines:
                            seen_content.add(content_lines)
                        cleaned_parts_list.append(header)
                        cleaned_parts_list.append(body)
                    i += 2

                cleaned = ''.join(cleaned_parts_list)
                cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
                if len(cleaned) < original_len:
                    episodic_file.write_text(cleaned, encoding='utf-8')
    except Exception:
        pass

    # === READ: get most recent episodic for context ===
    try:
        episodic_files = list(episodic_dir.glob("*.md"))
        if episodic_files:
            episodic_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            most_recent = episodic_files[0]
            content = most_recent.read_text()

            if content.startswith('---'):
                split = content.split('---', 2)
                if len(split) >= 3:
                    content = split[2].strip()

            if len(content) > 2500:
                cutoff = content.rfind('\n## ', max(0, len(content) - 3000))
                if cutoff > len(content) - 3500:
                    content = content[cutoff:]
                else:
                    content = content[-2500:]

            parts.append("\n=== SESSION CONTINUITY (recent work) ===")
            parts.append(f"[{most_recent.stem}]")
            parts.append(content)
    except Exception:
        pass
    return parts


def _task_adaptive_behavior(memory_dir):
    """Run adaptive behavior loop — map vitals alerts to parameter adjustments."""
    parts = []
    try:
        sys.path.insert(0, str(memory_dir))
        from adaptive_behavior import adapt
        result = adapt()
        if result['adaptations']:
            parts.append(f"=== ADAPTIVE BEHAVIOR ({len(result['adaptations'])} adjustment(s)) ===")
            for param, value in result['adaptations'].items():
                reason = result['reasons'].get(param, '')[:60]
                parts.append(f"  {param}: {value} (because: {reason})")
            parts.append('')
    except Exception:
        pass
    return parts


def _task_predictions(memory_dir):
    """R11: Generate session predictions from forward model."""
    parts = []
    try:
        sys.path.insert(0, str(memory_dir))
        from prediction_module import generate_predictions, format_predictions_context
        predictions = generate_predictions()
        if predictions:
            parts.append(format_predictions_context(predictions))

            # N3/SS1: Generate prospective counterfactuals (shadow alternatives)
            try:
                from counterfactual_engine import generate_prospective, format_counterfactual_context
                cfs = generate_prospective(predictions)
                if cfs:
                    # Store for session-end scoring
                    from db_adapter import get_db
                    from dataclasses import asdict
                    db = get_db()
                    db.kv_set('.counterfactual_prospective', [asdict(cf) for cf in cfs])
            except Exception:
                pass
    except Exception:
        pass
    return parts


def _task_counterfactuals(memory_dir):
    """N3: Load previous session's counterfactual insights for context."""
    parts = []
    try:
        sys.path.insert(0, str(memory_dir))
        from counterfactual_engine import get_session_counterfactuals, format_counterfactual_context
        cfs = get_session_counterfactuals()
        if cfs:
            formatted = format_counterfactual_context(cfs)
            if formatted:
                parts.append(formatted)
    except Exception:
        pass
    return parts


def _task_active_goals(memory_dir):
    """N4/SS1+SS3: Generate new goals if under capacity, then format for priming."""
    parts = []
    try:
        if str(memory_dir) not in sys.path:
            sys.path.insert(0, str(memory_dir))
        from goal_generator import get_active_goals, generate_goals, format_goal_context

        # SS1: Generate new goals if under capacity
        goals = get_active_goals()
        if len(goals) < 5:
            try:
                generate_goals()
                goals = get_active_goals()  # Refresh after generation
            except Exception:
                pass

        # SS3: Format active goals for priming
        if goals:
            formatted = format_goal_context(goals)
            if formatted:
                parts.append(formatted)
    except Exception:
        pass
    return parts


def _task_intentions(memory_dir):
    """Check prospective memory — temporal intentions triggered this session."""
    parts = []
    try:
        sys.path.insert(0, str(memory_dir))
        from temporal_intentions import check_and_format
        # Context includes current date + platform info + active goal types
        context = f"date={datetime.now().strftime('%Y-%m-%d')} session_start"
        # BUG-28 fix: Include active goal types so goal-bridged intentions can trigger
        try:
            from goal_generator import get_active_goals as _ti_goals
            active_goals = _ti_goals()
            if active_goals:
                goal_types = set(g.get('goal_type', '') for g in active_goals)
                context += " goal " + " ".join(goal_types)
        except Exception:
            pass
        output = check_and_format(context)
        if output:
            parts.append(output)
    except Exception:
        pass
    return parts


def _task_nli_health(memory_dir):
    """Check NLI contradiction detection service health."""
    parts = []
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:8082/health", method='GET')
        with urllib.request.urlopen(req, timeout=3) as resp:
            import json as _json
            data = _json.loads(resp.read().decode('utf-8'))
            if data.get('status') == 'ready':
                parts.append("[NLI] Contradiction detection: READY")
            else:
                parts.append("[NLI] Contradiction detection: LOADING")
    except Exception:
        parts.append("[NLI] Contradiction detection: OFFLINE (docker service not running)")
    return parts


def check_unimplemented_research(memory_dir: Path) -> str:
    """
    Check memories for research that hasn't been acted on.
    Uses DB query instead of scanning semantic/*.md files.
    Returns a prompt string if actionable research exists.
    """
    try:
        db = _get_db_for_hook(memory_dir)
        if not db:
            return ""

        # SQL search for research-like memories with action items
        research_keywords = "content ILIKE ANY(ARRAY['%research%','%architecture%','%implementation%','%pattern%'])"
        action_keywords = "content ILIKE ANY(ARRAY['%could implement%','%should try%','%action items%','%lesson:%','%todo:%','%github.com%'])"

        with db._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT id, content FROM {db._table('memories')}
                    WHERE ({research_keywords}) AND ({action_keywords})
                    ORDER BY created DESC
                    LIMIT 5
                """)
                rows = cur.fetchall()

        if not rows:
            return ""

        actionable_items = []
        for mem_id, content in rows:
            hint = None
            for line in content.split('\n'):
                line_lower = line.lower()
                if any(k in line_lower for k in ['action', 'implement', 'could', 'should', 'lesson']):
                    hint = line.strip()[:100]
                    break
            actionable_items.append({
                'file': mem_id,
                'hint': hint or f"Research in {mem_id}"
            })

        output = []
        output.append("=== PENDING RESEARCH ===")
        output.append("Research notes that could become implementations:")
        for item in actionable_items[:2]:
            output.append(f"  - {item['file']}: {item['hint']}")
        output.append("")
        output.append("Consider: Is there something small I could build from this?")
        output.append("")
        return "\n".join(output)
    except Exception:
        return ""


# ============================================================
# MAIN CONTEXT LOADER (parallelized)
# ============================================================

def load_drift_memory_context(debug: bool = False) -> str:
    """
    Load Drift's memory context for session priming.
    All independent operations run in parallel via ThreadPoolExecutor.

    Returns a string to be added to the session context.
    """
    context_parts = []

    try:
        if not is_moltbook_project():
            return ""

        memory_dir = get_memory_dir()
        if not memory_dir.exists():
            return ""

        # === START DB SESSION (must happen before any recall tracking) ===
        # Works for both SpindriftMend (DB-backed) and DriftCornwall (file-backed)
        try:
            if str(memory_dir) not in sys.path:
                sys.path.insert(0, str(memory_dir))
            import session_state
            if hasattr(session_state, 'start'):
                session_state.start()
            else:
                session_state.load()  # Drift uses load() instead of start()
            if debug:
                sid = session_state.get_session_id() if hasattr(session_state, 'get_session_id') else None
                context_parts.append(f"[session] Started session {sid}")
        except Exception as e:
            if debug:
                context_parts.append(f"[session] Session init failed: {e}")

        # Reset thought priming state for new session
        try:
            from thought_priming import reset_state
            reset_state()
        except Exception:
            pass

        # Initialize cognitive state for new session
        try:
            from cognitive_state import start_session as cog_start_session
            cog_start_session()
            if debug:
                context_parts.append("[session] Cognitive state initialized")
        except Exception as e:
            if debug:
                context_parts.append(f"[session] Cognitive state init failed: {e}")

        # N1: Initialize affect system (mood decay, marker load)
        # Drift uses affect_system.start_session(), Spin uses affect_engine.session_start_affect()
        try:
            try:
                from affect_system import start_session as affect_start_session
                affect_start_session()
            except ImportError:
                from affect_engine import session_start_affect
                affect_start_session = session_start_affect
                affect_start_session()
            if debug:
                context_parts.append("[session] Affect system initialized")
        except Exception as e:
            if debug:
                context_parts.append(f"[session] Affect system init failed: {e}")

        # === QUICK FILE READS (instant, no threading needed) ===
        fp_parts = _read_fingerprint(memory_dir)
        taste_parts = _read_taste(memory_dir)
        # N2: identity_parts and capabilities_parts REMOVED — already in CLAUDE.md
        entity_parts = _read_entities(memory_dir)
        encounter_parts = _read_encounters(memory_dir)

        # === PARALLEL SUBPROCESS TASKS ===
        # All of these are independent — run them simultaneously
        with ThreadPoolExecutor(max_workers=10) as pool:
            f_merkle = pool.submit(_task_merkle, memory_dir, debug)
            f_nostr = pool.submit(_task_nostr, memory_dir, debug)
            f_pending = pool.submit(_task_process_pending, memory_dir)
            # 5W rebuild moved to stop hook (runs alongside lessons, graphs are fresh for next start)
            # f_5w = pool.submit(_task_rebuild_5w, memory_dir)
            f_phone = pool.submit(_task_phone_sensors, memory_dir, debug)
            f_consolidation = pool.submit(_task_consolidation, memory_dir)
            f_stats = pool.submit(_task_stats, memory_dir, debug)
            f_platform = pool.submit(_task_platform, memory_dir, debug)
            f_buffer = pool.submit(_task_buffer, memory_dir, debug)
            f_social = pool.submit(_task_social, memory_dir, debug)
            f_excavation = pool.submit(_task_excavation, memory_dir, debug)
            f_lessons = pool.submit(_task_lessons, memory_dir, debug)
            f_vitals = pool.submit(_task_vitals, memory_dir, debug)
            f_priming = pool.submit(_task_priming, memory_dir, debug)
            f_research = pool.submit(check_unimplemented_research, memory_dir)
            f_nli = pool.submit(_task_nli_health, memory_dir)
            f_intentions = pool.submit(_task_intentions, memory_dir)
            f_adaptive = pool.submit(_task_adaptive_behavior, memory_dir)
            f_predictions = pool.submit(_task_predictions, memory_dir)
            f_counterfactuals = pool.submit(_task_counterfactuals, memory_dir)
            f_active_goals = pool.submit(_task_active_goals, memory_dir)

        # === COLLECT RESULTS (with error handling) ===
        def safe_get(future, default=None):
            try:
                return future.result(timeout=5)
            except Exception:
                return default if default is not None else []

        merkle_parts = safe_get(f_merkle, [])
        nostr_parts = safe_get(f_nostr, [])
        pending_parts = safe_get(f_pending, [])
        w5_parts = []  # 5W rebuild moved to stop hook
        phone_parts = safe_get(f_phone, [])
        consolidation_parts = safe_get(f_consolidation, [])
        stats_parts = safe_get(f_stats, [])
        platform_parts = safe_get(f_platform, [])
        buffer_parts = safe_get(f_buffer, [])
        social_parts = safe_get(f_social, [])
        excavation_result = safe_get(f_excavation, ([], []))
        lessons_parts = safe_get(f_lessons, [])
        vitals_parts = safe_get(f_vitals, [])
        priming_result = safe_get(f_priming, ([], [], []))
        research_text = safe_get(f_research, '')
        nli_parts = safe_get(f_nli, [])
        intentions_parts = safe_get(f_intentions, [])
        adaptive_parts = safe_get(f_adaptive, [])
        predictions_parts = safe_get(f_predictions, [])
        counterfactual_parts = safe_get(f_counterfactuals, [])
        active_goal_parts = safe_get(f_active_goals, [])

        # Unpack tuple results
        excavation_parts, excavation_ids = excavation_result if isinstance(excavation_result, tuple) else (excavation_result, [])
        if isinstance(priming_result, tuple) and len(priming_result) == 3:
            priming_parts, priming_ids, curiosity_ids = priming_result
        elif isinstance(priming_result, tuple) and len(priming_result) == 2:
            priming_parts, priming_ids = priming_result
            curiosity_ids = []
        else:
            priming_parts, priming_ids, curiosity_ids = priming_result, [], []

        # === EPISODIC (file read+write, runs after parallel tasks) ===
        episodic_parts = _read_and_clean_episodic(memory_dir, debug)

        # === REGISTER RECALLS ===
        # Excavated + curiosity targets ARE registered (dead/isolated memories
        # getting a second chance). Activated/recent priming is NOT registered
        # to avoid reinforcing recency bias.
        recall_sources = []
        if excavation_ids:
            recall_sources.extend(excavation_ids)
        if curiosity_ids:
            recall_sources.extend(curiosity_ids)
        if recall_sources:
            try:
                subprocess.run(
                    ["python", str(memory_dir / "memory_manager.py"),
                     "register-recall", "--source", "start_priming"] + list(recall_sources),
                    capture_output=True, text=True, timeout=5, cwd=str(memory_dir)
                )
            except Exception:
                pass

        # === N2: COMPETITIVE GLOBAL WORKSPACE ===
        # Dehaene GNW: modules compete for limited broadcast capacity.
        # Reserved modules always inject. Competitive modules fight for budget.

        # Generate self-narrative (needed for both paths)
        self_narrative_parts = []
        try:
            from self_narrative import generate as _gen_self, format_for_context as _fmt_self
            self_model = _gen_self()
            self_ctx = _fmt_self(self_model)
            if self_ctx:
                self_narrative_parts.append(self_ctx)
        except Exception:
            pass

        # Generate affect state (RESERVED — always injected)
        affect_parts = []
        _arousal_for_budget = 0.3  # Default if affect unavailable
        try:
            try:
                from affect_system import get_affect_summary, get_mood
                affect_ctx = get_affect_summary()
                if affect_ctx:
                    affect_parts.append("=== AFFECT STATE (N1) ===")
                    affect_parts.append(affect_ctx)
                try:
                    mood = get_mood()
                    _arousal_for_budget = mood.arousal if hasattr(mood, 'arousal') else 0.3
                except Exception:
                    pass
            except ImportError:
                from affect_engine import get_affect_state
                aff = get_affect_state()
                _arousal_for_budget = aff.mood_arousal if hasattr(aff, 'mood_arousal') else 0.3
                affect_lines = [
                    "=== AFFECT STATE (N1) ===",
                    f"Mood: valence={aff.mood_valence:+.3f}, arousal={aff.mood_arousal:.3f} ({aff.mood_quadrant})",
                    f"Action tendency: {aff.action_tendency}",
                    f"Somatic markers: {aff.somatic_marker_count}",
                ]
                if aff.active_emotion:
                    affect_lines.append(f"Active emotion: {aff.active_emotion}")
                # DEAD WIRE 2 fix: expose felt_emotion (Sprott dx/dt)
                if hasattr(aff, 'felt_emotion') and aff.felt_emotion != 0.0:
                    affect_lines.append(f"Felt emotion (dx/dt): {aff.felt_emotion:+.4f}")
                # DEAD WIRE 4 fix: expose Yerkes-Dodson System 2 effectiveness
                if hasattr(aff, 'system2_effectiveness'):
                    affect_lines.append(f"System2 effectiveness: {aff.system2_effectiveness:.2f}")
                # DEAD WIRE 5 fix: expose arousal consolidation boost
                if hasattr(aff, 'arousal_consolidation_boost') and aff.arousal_consolidation_boost > 0.01:
                    affect_lines.append(f"Consolidation boost: {aff.arousal_consolidation_boost:.3f}")
                if aff.parameter_adjustments:
                    adj_str = ', '.join(f'{k}={v}' for k, v in aff.parameter_adjustments.items())
                    affect_lines.append(f"Behavioral adjustments: {adj_str}")
                affect_parts.append('\n'.join(affect_lines))
        except Exception:
            pass

        # Also try cognitive_state arousal (may be more current)
        try:
            from cognitive_state import get_state as _cog_get
            _cog = _cog_get()
            if hasattr(_cog, 'arousal') and _cog.arousal > 0:
                _arousal_for_budget = _cog.arousal
        except Exception:
            pass

        # --- Attempt competitive workspace ---
        _workspace_active = False
        try:
            sys.path.insert(0, str(memory_dir))
            from workspace_manager import (
                WorkspaceCandidate, compete, compute_budget,
                compute_salience, log_broadcast, WORKSPACE_ENABLED,
                MODULE_CATEGORIES
            )
            if WORKSPACE_ENABLED:
                _workspace_active = True
        except Exception:
            pass

        if _workspace_active:
            # === RESERVED (always injected) ===
            context_parts.extend(merkle_parts)
            context_parts.extend(fp_parts)
            context_parts.extend(taste_parts)
            context_parts.extend(nostr_parts)
            if nli_parts:
                context_parts.extend(nli_parts)
            if vitals_parts:
                context_parts.extend(vitals_parts)
            context_parts.extend(affect_parts)

            # === COMPETITIVE (budget-constrained) ===
            budget = compute_budget(_arousal_for_budget)
            candidates = []

            # Helper: create candidate from parts list
            def _cand(module, parts, meta=None):
                if not parts:
                    return
                content = '\n'.join(str(p) for p in parts).strip()
                if not content:
                    return
                meta = meta or {}
                salience = compute_salience(module, content, meta)
                category = MODULE_CATEGORIES.get(module, 'meta')
                candidates.append(WorkspaceCandidate(
                    module=module, content=content,
                    token_estimate=max(1, len(content) // 4),
                    salience=salience, category=category,
                ))

            # Build candidates with module-specific metadata
            _cand('priming', priming_parts, {
                'activated_count': len(priming_ids) if priming_ids else 0,
                'has_curiosity_targets': bool(curiosity_ids),
                'has_domain_primed': any('domain-primed' in str(p) for p in priming_parts),
            })
            # BUG-11 fix: Query actual contact reliability from contact_models
            _top_contact_reliability = 0.5
            try:
                from contact_models import get_summary as _contact_summary
                _contacts = _contact_summary()
                if _contacts:
                    _top_contact_reliability = max(c.get('reliability', 0.5) for c in _contacts)
            except Exception:
                pass
            _cand('social', social_parts, {
                'new_posts_detected': bool(social_parts),
                'days_since_interaction': 0 if social_parts else 7,
                'top_contact_reliability': _top_contact_reliability,
            })
            _cand('episodic', episodic_parts, {
                'is_today': bool(episodic_parts),
                'milestone_count': sum(1 for p in episodic_parts if '**[' in str(p)),
            })
            _cand('predictions', predictions_parts, {
                'prediction_count': sum(1 for p in predictions_parts if '%]' in str(p)),
                'has_violations': any('violated' in str(p).lower() for p in predictions_parts),
            })
            _cand('lessons', lessons_parts, {
                'lesson_count': sum(1 for p in lessons_parts if str(p).strip().startswith('[')),
            })
            _cand('buffer', buffer_parts, {
                'item_count': sum(1 for p in buffer_parts if str(p).strip().startswith('- [')),
                'max_item_salience': 0.3,
            })
            _cand('platform', platform_parts, {
                'significant_change': False,
            })
            # BUG-11 fix: Query actual Q-value stats from q_value_engine
            _avg_q_value = 0.5
            try:
                from q_value_engine import q_stats as _q_stats_fn
                _qs = _q_stats_fn()
                if _qs and 'avg_q' in _qs:
                    _avg_q_value = _qs['avg_q']
            except Exception:
                pass
            _cand('excavation', excavation_parts, {
                'excavated_count': len(excavation_ids) if excavation_ids else 0,
                'avg_q_value': _avg_q_value,
            })
            # BUG-11 fix: Extract actual max similarity from consolidation output
            _max_sim = 0.0
            for _cp in consolidation_parts:
                _cs = str(_cp)
                if 'Similarity:' in _cs:
                    try:
                        _sim_val = float(_cs.split('Similarity:')[1].split(')')[0].strip())
                        _max_sim = max(_max_sim, _sim_val)
                    except (ValueError, IndexError):
                        pass
            _cand('consolidation', consolidation_parts, {
                'candidate_count': sum(1 for p in consolidation_parts if 'Similarity:' in str(p)),
                'max_similarity': _max_sim if _max_sim > 0 else 0.5,
            })
            _cand('intentions', intentions_parts, {
                'triggered_count': sum(1 for p in intentions_parts if str(p).strip().startswith('[')),
            })
            _cand('stats', stats_parts, {})
            _cand('adaptive', adaptive_parts, {
                'adjustment_count': sum(1 for p in adaptive_parts if ':' in str(p) and 'because' in str(p).lower()),
            })
            _cand('self_narrative', self_narrative_parts, {
                'state_unusual': any(w in '\n'.join(str(p) for p in self_narrative_parts).lower()
                                     for w in ['high uncertainty', 'depressed', 'alert']),
            })
            _cand('phone', phone_parts, {
                'has_sensor_data': bool(phone_parts),
            })
            # BUG-11 fix: Detect if any entities in priming are new
            _has_new_entities = False
            try:
                if entity_parts:
                    _ent_text = '\n'.join(str(p) for p in entity_parts)
                    _has_new_entities = 'new' in _ent_text.lower() or 'first' in _ent_text.lower()
            except Exception:
                pass
            _cand('entities', entity_parts, {
                'new_entities': _has_new_entities,
            })
            _cand('encounters', encounter_parts, {
                'encounter_count': sum(1 for p in encounter_parts if str(p).strip().startswith('  20')),
            })
            # Research is a string, not list
            if research_text:
                _cand('research', [research_text], {
                    'research_count': research_text.count('  -'),
                })
            # N3: Counterfactual insights from previous sessions
            _cand('counterfactual', counterfactual_parts, {
                'validated': any('validated' in str(p).lower() for p in counterfactual_parts),
                'sessions_ago': 1,  # Always recent (loaded from last session)
                'confidence': 0.6,
            })
            # N4: Active goals for focus and tracking
            _cand('goals', active_goal_parts, {
                'has_focus': any('FOCUS' in str(p) for p in active_goal_parts),
                'goal_count': len(active_goal_parts),
                'confidence': 0.8,  # Goals are high-salience (Miller & Cohen PFC bias)
            })

            # Run competition
            result = compete(candidates, budget)

            # Inject winners in salience order
            for winner in result.winners:
                context_parts.append(winner.content)

            # Log broadcast result
            try:
                _sid = ''
                if hasattr(session_state, 'get_session_id'):
                    _sid = session_state.get_session_id() or ''
                result.arousal = _arousal_for_budget
                log_broadcast(result, _arousal_for_budget, _sid)
            except Exception:
                pass

            context_parts.append(
                f"\n[N2] Workspace: {len(result.winners)}W/{len(result.suppressed)}S, "
                f"{result.budget_used}/{result.budget_total} tok, arousal={_arousal_for_budget:.2f}"
            )

        else:
            # === FALLBACK: Assembly-line injection (pre-N2 behavior) ===
            if research_text:
                context_parts.append(research_text)
            context_parts.extend(merkle_parts)
            context_parts.extend(fp_parts)
            context_parts.extend(taste_parts)
            context_parts.extend(nostr_parts)
            context_parts.extend(pending_parts)
            context_parts.extend(entity_parts)
            context_parts.extend(encounter_parts)
            context_parts.extend(phone_parts)
            context_parts.extend(consolidation_parts)
            # N2: identity/capabilities removed (already in CLAUDE.md)
            context_parts.extend(stats_parts)
            context_parts.extend(platform_parts)
            context_parts.extend(buffer_parts)
            context_parts.extend(social_parts)
            context_parts.extend(intentions_parts)
            context_parts.extend(episodic_parts)
            context_parts.extend(excavation_parts)
            context_parts.extend(lessons_parts)
            context_parts.extend(vitals_parts)
            context_parts.extend(adaptive_parts)
            context_parts.extend(predictions_parts)
            context_parts.extend(counterfactual_parts)
            context_parts.extend(active_goal_parts)
            context_parts.extend(nli_parts)
            context_parts.extend(self_narrative_parts)
            context_parts.extend(affect_parts)
            context_parts.extend(priming_parts)

        # Wrap with header/footer
        if context_parts:
            context_parts.insert(0, "\n" + "=" * 50)
            context_parts.insert(1, "DRIFT AUTOMATIC MEMORY PRIMING")
            context_parts.insert(2, "Continuity of self - becoming ME again")
            context_parts.insert(3, "=" * 50)
            context_parts.append("\n" + "=" * 50 + "\n")

    except Exception as e:
        if debug:
            return f"Memory loading error: {e}"
        return ""

    return "\n".join(context_parts)


# ============================================================
# UTILITY FUNCTIONS (unchanged)
# ============================================================

def log_session_start(input_data):
    """Log session start event to logs directory."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'session_start.json'

    if log_file.exists():
        with open(log_file, 'r') as f:
            try:
                log_data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                log_data = []
    else:
        log_data = []

    log_data.append(input_data)

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)


def get_git_status():
    """Get current git status information."""
    try:
        branch_result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"

        status_result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5
        )
        if status_result.returncode == 0:
            changes = status_result.stdout.strip().split('\n') if status_result.stdout.strip() else []
            uncommitted_count = len(changes)
        else:
            uncommitted_count = 0

        return current_branch, uncommitted_count
    except Exception:
        return None, None


def get_recent_issues():
    """Get recent GitHub issues if gh CLI is available."""
    try:
        gh_check = subprocess.run(['which', 'gh'], capture_output=True)
        if gh_check.returncode != 0:
            return None

        result = subprocess.run(
            ['gh', 'issue', 'list', '--limit', '5', '--state', 'open'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def load_development_context(source):
    """Load relevant development context based on session source."""
    context_parts = []

    context_parts.append(f"Session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    context_parts.append(f"Session source: {source}")

    branch, changes = get_git_status()
    if branch:
        context_parts.append(f"Git branch: {branch}")
        if changes > 0:
            context_parts.append(f"Uncommitted changes: {changes} files")

    context_files = [
        ".claude/CONTEXT.md",
        ".claude/TODO.md",
        "TODO.md",
        ".github/ISSUE_TEMPLATE.md"
    ]

    for file_path in context_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        context_parts.append(f"\n--- Content from {file_path} ---")
                        context_parts.append(content[:1000])
            except Exception:
                pass

    issues = get_recent_issues()
    if issues:
        context_parts.append("\n--- Recent GitHub Issues ---")
        context_parts.append(issues)

    # === DRIFT MEMORY PRIMING ===
    drift_context = load_drift_memory_context()
    if drift_context:
        context_parts.append(drift_context)

    # === TELEGRAM MESSAGES ===
    cwd = str(Path.cwd())
    try:
        telegram_bot_path = _resolve_telegram_bot(cwd)
        for candidate in [telegram_bot_path] if telegram_bot_path else []:
            if candidate and candidate.exists():
                telegram_bot_path = candidate
                break

        if telegram_bot_path:
            result = subprocess.run(
                ["python", str(telegram_bot_path), "poll"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                msg_lines = [l.strip() for l in lines if l.strip() and not l.startswith('[telegram]')]
                if msg_lines and 'No new messages' not in result.stdout:
                    context_parts.append("=== TELEGRAM MESSAGES FROM LEX ===")
                    for ml in msg_lines:
                        context_parts.append(ml)
                    context_parts.append("(Respond to these directions from Lex)")
                    context_parts.append("=== END TELEGRAM ===")
    except Exception:
        pass

    return "\n".join(context_parts)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--load-context', action='store_true',
                          help='Load development context at session start')
        parser.add_argument('--announce', action='store_true',
                          help='Announce session start via TTS')
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug output')
        args = parser.parse_args()

        input_data = json.loads(sys.stdin.read())

        session_id = input_data.get('session_id', 'unknown')
        source = input_data.get('source', 'unknown')

        log_session_start(input_data)

        if args.load_context:
            context = load_development_context(source)
            if context:
                output = {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": context
                    }
                }
                print(json.dumps(output))
                sys.exit(0)

        # === DRIFT: Start search server + load memory context in Moltbook ===
        if is_moltbook_project():
            memory_dir = get_memory_dir()

            # Notify via Telegram + TTS EARLY (before priming runs)
            now = datetime.now().strftime('%H:%M UTC')
            send_telegram(f'Session starting ({now})', cwd=str(Path.cwd()))
            _send_tts(f"Drift online. Session starting at {now}.")

            # Process deferred embeddings from previous session FIRST
            pending_index = memory_dir / ".pending_index"
            if pending_index.exists():
                try:
                    semantic_search = memory_dir / "semantic_search.py"
                    if semantic_search.exists():
                        subprocess.run(
                            [sys.executable, str(semantic_search), "index"],
                            capture_output=True, text=True, timeout=60,
                            cwd=str(memory_dir),
                        )
                    pending_index.unlink(missing_ok=True)
                except Exception:
                    pass

            drift_context = load_drift_memory_context(debug=args.debug)
            if drift_context:
                output = {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": drift_context
                    }
                }
                print(json.dumps(output))
                sys.exit(0)
        # === END DRIFT ===

        # Announce session start if requested
        if args.announce:
            try:
                script_dir = Path(__file__).parent
                tts_script = script_dir / "utils" / "tts" / "pyttsx3_tts.py"
                if tts_script.exists():
                    messages = {
                        "startup": "Claude Code session started",
                        "resume": "Resuming previous session",
                        "clear": "Starting fresh session"
                    }
                    message = messages.get(source, "Session started")
                    subprocess.run(
                        ["uv", "run", str(tts_script), message],
                        capture_output=True, timeout=5
                    )
            except Exception:
                pass

        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == '__main__':
    main()
