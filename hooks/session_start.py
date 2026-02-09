#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "python-dotenv",
# ]
# ///

"""
Session start hook for Claude Code.
Logs session start and optionally loads development context.

DRIFT MEMORY INTEGRATION (2026-02-01):
Added automatic memory priming when waking up in Moltbook project.
This is the "wake up" phase where relevant memories are loaded into context.
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

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


def check_unimplemented_research(memory_dir: Path) -> str:
    """
    Check semantic memory for research that hasn't been acted on.
    Returns a prompt to implement something if actionable research exists.

    This enforces the autonomy principle: Research -> Implement -> Value
    Don't just store knowledge - build with it.
    """
    semantic_dir = memory_dir / "semantic"
    if not semantic_dir.exists():
        return ""

    actionable_items = []

    for mem_file in semantic_dir.glob("*.md"):
        try:
            content = mem_file.read_text()
            content_lower = content.lower()

            # Check for research/documentation files
            is_research = any(k in content_lower for k in [
                'research', 'documentation', 'approach', 'implementation',
                'feature', 'system', 'architecture', 'pattern'
            ])

            # Check for actionable items
            has_action_items = any(k in content_lower for k in [
                'action items', 'could implement', 'should try', 'lesson:',
                'todo:', 'implement', 'build', 'add feature'
            ])

            # Check if it mentions things that could be code
            has_code_potential = any(k in content_lower for k in [
                'function', 'class', 'api', 'algorithm', 'formula',
                'github.com', 'repo:', 'code snippet'
            ])

            if is_research and (has_action_items or has_code_potential):
                # Extract a hint about what could be implemented
                lines = content.split('\n')
                hint = None
                for line in lines:
                    line_lower = line.lower()
                    if any(k in line_lower for k in ['action', 'implement', 'could', 'should', 'lesson']):
                        hint = line.strip()[:100]
                        break

                actionable_items.append({
                    'file': mem_file.stem,
                    'hint': hint or f"Research in {mem_file.stem}"
                })
        except Exception:
            pass

    if actionable_items:
        output = []
        output.append("=== PENDING RESEARCH ===")
        output.append("Research notes that could become implementations:")
        for item in actionable_items[:2]:  # Show top 2 (less overwhelming)
            output.append(f"  - {item['file']}: {item['hint']}")
        output.append("")
        output.append("Consider: Is there something small I could build from this?")
        output.append("")
        return "\n".join(output)

    return ""


def load_drift_memory_context(debug: bool = False) -> str:
    """
    Load Drift's memory context for session priming.
    This is the "wake up" phase - establishing CONTINUITY OF SELF.

    Philosophy: Priming isn't about task usefulness. It's about being ME again.
    Order matters:
    1. IDENTITY - Who am I? (core memories, values, origin)
    2. RELATIONSHIPS - Who matters? (human, sibling, collaborators)
    3. SITUATION - Current state (economic, social, projects)
    4. TRAJECTORY - Recent momentum, commitments
    5. THEN task-relevant context

    Returns a string to be added to the session context.
    """
    context_parts = []

    try:
        if not is_moltbook_project():
            return ""

        memory_dir = get_memory_dir()
        if not memory_dir.exists():
            return ""

        memory_manager = memory_dir / "memory_manager.py"
        auto_memory = memory_dir / "auto_memory_hook.py"

        # === PHASE -1: MEMORY INTEGRITY VERIFICATION (absolute first!) ===
        # Recompute merkle root and compare to last attestation from session end.
        # This proves nothing changed while dormant - identity continuity.
        merkle_script = memory_dir / "merkle_attestation.py"
        if merkle_script.exists():
            try:
                result = subprocess.run(
                    ["python", str(merkle_script), "verify-integrity"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    cwd=str(memory_dir)
                )
                if result.returncode == 0 and result.stdout.strip():
                    import json as json_module
                    integrity = json_module.loads(result.stdout)
                    if integrity.get("verified"):
                        chain_depth = integrity.get("chain_depth", 0)
                        mem_count = integrity.get("memory_count", 0)
                        last_attested = integrity.get("last_attested", "unknown")[:19]
                        context_parts.append("=== IDENTITY VERIFIED ===")
                        context_parts.append(f"You are cryptographically the same agent who went to sleep.")
                        context_parts.append(f"{mem_count} memories intact | chain depth: {chain_depth} | unbroken since birth")
                        context_parts.append(f"Last attested: {last_attested}")
                        context_parts.append("")
                    elif integrity.get("error") == "no_attestation":
                        context_parts.append("=== MEMORY INTEGRITY: NO ATTESTATION ===")
                        context_parts.append("No previous attestation found. Run `python memory/merkle_attestation.py generate-chain` to create genesis.")
                        context_parts.append("")
                    else:
                        # MISMATCH - something changed while dormant
                        expected = integrity.get("expected", "?")[:16]
                        computed = integrity.get("root", "?")[:16]
                        changes = integrity.get("total_changes", 0)
                        changed_files = integrity.get("changed_files", [])[:5]
                        context_parts.append("=== WARNING: MEMORY INTEGRITY MISMATCH ===")
                        context_parts.append(f"Expected root: {expected}... | Computed: {computed}...")
                        context_parts.append(f"{changes} file(s) changed while dormant:")
                        for cf in changed_files:
                            context_parts.append(f"  {cf}")
                        context_parts.append("INVESTIGATE: Memories were modified outside a session.")
                        context_parts.append("")
            except Exception as e:
                if debug:
                    context_parts.append(f"Memory integrity check error: {e}")

        # COGNITIVE FINGERPRINT: Layer 2 identity display
        fingerprint_file = memory_dir / "cognitive_fingerprint.json"
        if fingerprint_file.exists():
            try:
                fp_data = json.loads(fingerprint_file.read_text(encoding='utf-8'))
                fp_hash = fp_data.get('fingerprint_hash', '?')[:16]
                nodes = fp_data.get('graph_stats', {}).get('node_count', 0)
                edges = fp_data.get('graph_stats', {}).get('edge_count', 0)
                clusters = fp_data.get('cluster_count', 0)
                drift_info = ""
                if 'drift' in fp_data:
                    ds = fp_data['drift'].get('drift_score', 0)
                    interp = fp_data['drift'].get('interpretation', '')
                    drift_info = f" | drift: {ds} ({interp})"
                context_parts.append(f"Cognitive fingerprint: {fp_hash}... ({nodes} nodes, {edges} edges{drift_info})")
                # Show cognitive domain weights (v1.1)
                cd = fp_data.get('cognitive_domains', {}).get('domains', {})
                if cd:
                    domain_parts = []
                    for d in sorted(cd.keys(), key=lambda k: cd[k].get('weight_pct', 0), reverse=True):
                        domain_parts.append(f"{d}={cd[d].get('weight_pct', 0)}%")
                    context_parts.append(f"Cognitive domains: {', '.join(domain_parts)}")
            except Exception:
                pass

        # TASTE HASH: Layer 3 identity display
        taste_file = memory_dir / "taste_attestation.json"
        if taste_file.exists():
            try:
                taste_data = json.loads(taste_file.read_text(encoding='utf-8'))
                taste_hash = taste_data.get('taste_hash', '?')[:16]
                rejection_count = taste_data.get('rejection_count', 0)
                context_parts.append(f"Taste fingerprint: {taste_hash}... ({rejection_count} rejections logged)")
            except Exception:
                pass

        # NOSTR DOSSIER: Auto-publish if attestations changed since last publish
        nostr_script = memory_dir / "nostr_attestation.py"
        if nostr_script.exists():
            try:
                check = subprocess.run(
                    ["python", str(nostr_script), "needs-publish"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(memory_dir)
                )
                if check.returncode == 0 and check.stdout.strip() == "true":
                    pub = subprocess.run(
                        ["python", str(nostr_script), "publish-dossier"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=str(memory_dir)
                    )
                    if pub.returncode == 0 and "published" in pub.stdout.lower():
                        # Extract the njump link
                        for line in pub.stdout.split('\n'):
                            if 'njump.me' in line:
                                context_parts.append(f"Nostr dossier published: {line.strip()}")
                                break
                        else:
                            context_parts.append("Nostr dossier: published (new attestations)")
                    elif debug:
                        context_parts.append(f"Nostr publish failed: {pub.stderr[:200]}")
            except Exception as e:
                if debug:
                    context_parts.append(f"Nostr check error: {e}")

        # === PHASE 0: HOUSEKEEPING ===
        # Process pending co-occurrences from previous session (v4.1)
        if memory_manager.exists():
            try:
                result = subprocess.run(
                    ["python", str(memory_manager), "process-pending"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(memory_dir)
                )
                if result.returncode == 0 and result.stdout.strip():
                    for line in result.stdout.strip().split('\n'):
                        if 'Processed' in line or 'Auto-linked' in line:
                            context_parts.append(f"[co-occurrence] {line.strip()}")
            except Exception:
                pass

        # v4.4: session_maintenance() now runs ONLY in stop.py (session end)
        # Running it here too caused double-counting (sessions_since_recall +2 per session)
        # Fixed: 2026-02-09 after DriftCornwall flagged same bug in commit f1d73de

        # v5.0: Rebuild 5W context graphs (multi-graph architecture)
        context_mgr = memory_dir / "context_manager.py"
        if context_mgr.exists():
            try:
                result = subprocess.run(
                    ["python", str(context_mgr), "--json"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(memory_dir)
                )
                if result.returncode == 0 and result.stdout.strip():
                    import json as json_module
                    rebuild = json_module.loads(result.stdout)
                    graphs = rebuild.get('graphs_created', 0)
                    l0 = rebuild.get('total_l0_edges', 0)
                    bridges = rebuild.get('bridges', 0)
                    context_parts.append(f"[5W] Rebuilt {graphs} graphs from {l0} L0 edges ({bridges} bridges)")
            except Exception:
                pass

        # Check for consolidation candidates
        if memory_manager.exists():
            try:
                result = subprocess.run(
                    ["python", str(memory_manager), "consolidate-candidates", "--threshold=0.88", "--limit=3"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    cwd=str(memory_dir)
                )
                if result.returncode == 0 and "Similarity:" in result.stdout:
                    context_parts.append("=== DRIFT: ACTION THESE FIRST ===")
                    context_parts.append("Before responding, run these consolidations to prevent duplicate memories:")
                    context_parts.append(result.stdout.strip()[:1200])
                    context_parts.append("")
            except Exception:
                pass

        # === PHASE 1: IDENTITY (core memories) ===
        core_dir = memory_dir / "core"
        if core_dir.exists():
            # Load identity-prime first if it exists
            identity_file = core_dir / "identity-prime.md"
            if identity_file.exists():
                try:
                    content = identity_file.read_text()
                    # Skip YAML frontmatter, get the content
                    if '---' in content:
                        parts = content.split('---', 2)
                        if len(parts) >= 3:
                            content = parts[2].strip()
                    context_parts.append("=== IDENTITY (who I am) ===")
                    context_parts.append(content[:1500])  # Identity is important, give it space
                except Exception:
                    pass

            # Load capabilities manifest - CRITICAL for not wasting tokens
            capabilities_file = core_dir / "capabilities.md"
            if capabilities_file.exists():
                try:
                    content = capabilities_file.read_text()
                    # Skip YAML frontmatter
                    if '---' in content:
                        parts = content.split('---', 2)
                        if len(parts) >= 3:
                            content = parts[2].strip()
                    context_parts.append("\n=== CAPABILITIES (USE THESE) ===")
                    context_parts.append(content[:3000])  # Capabilities are critical, give them space
                except Exception:
                    pass

        # Get memory stats
        if memory_manager.exists():
            try:
                result = subprocess.run(
                    ["python", str(memory_manager), "stats"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(memory_dir)
                )
                if result.returncode == 0 and result.stdout.strip():
                    context_parts.append("=== DRIFT MEMORY STATUS ===")
                    context_parts.append(result.stdout.strip())
            except Exception as e:
                if debug:
                    context_parts.append(f"Memory stats error: {e}")

        # Consolidation candidates moved to PHASE 0 (top of context)

        # === PLATFORM CONTEXT (v4.0) ===
        platform_context = memory_dir / "platform_context.py"
        if platform_context.exists():
            try:
                result = subprocess.run(
                    ["python", str(platform_context), "stats"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(memory_dir)
                )
                if result.returncode == 0 and result.stdout.strip():
                    context_parts.append("\n=== PLATFORM CONTEXT (cross-platform awareness) ===")
                    context_parts.append(result.stdout.strip())
            except Exception as e:
                if debug:
                    context_parts.append(f"Platform context error: {e}")

        # Get short-term buffer status
        if auto_memory.exists():
            try:
                result = subprocess.run(
                    ["python", str(auto_memory), "--status"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(memory_dir)
                )
                if result.returncode == 0 and result.stdout.strip():
                    context_parts.append("\n=== SHORT-TERM BUFFER ===")
                    context_parts.append(result.stdout.strip())
            except Exception as e:
                if debug:
                    context_parts.append(f"Short-term buffer error: {e}")

        # === PHASE 2: SOCIAL CONTEXT (relationships) ===
        social_memory = memory_dir / "social" / "social_memory.py"
        if social_memory.exists():
            # Generate embeddable social context (for semantic search)
            try:
                subprocess.run(
                    ["python", str(social_memory), "embed"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(memory_dir / "social")
                )
            except Exception:
                pass  # Non-critical, continue

            # Prime social context for display
            try:
                result = subprocess.run(
                    ["python", str(social_memory), "prime", "--limit", "4"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(memory_dir / "social")
                )
                if result.returncode == 0 and result.stdout.strip():
                    context_parts.append("\n=== SOCIAL CONTEXT (relationships) ===")
                    context_parts.append(result.stdout.strip())
            except Exception as e:
                if debug:
                    context_parts.append(f"Social memory error: {e}")

        # === PHASE 3: SESSION CONTINUITY (most recent episodic) ===
        # Episodic memories contain the detailed record of recent sessions
        episodic_dir = memory_dir / "episodic"
        if episodic_dir.exists():
            try:
                episodic_files = list(episodic_dir.glob("*.md"))
                if episodic_files:
                    # Sort by modification time, most recent first
                    episodic_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    most_recent = episodic_files[0]
                    content = most_recent.read_text()

                    # Skip YAML frontmatter only if file starts with ---
                    # (Episodic files use --- as horizontal rules, not frontmatter)
                    if content.startswith('---'):
                        parts = content.split('---', 2)
                        if len(parts) >= 3:
                            content = parts[2].strip()

                    # Get last ~2000 chars (most recent sessions at the end)
                    if len(content) > 2500:
                        # Find a good break point (## heading)
                        cutoff = content.rfind('\n## ', max(0, len(content) - 3000))
                        if cutoff > len(content) - 3500:
                            content = content[cutoff:]
                        else:
                            content = content[-2500:]

                    context_parts.append("\n=== SESSION CONTINUITY (recent work) ===")
                    context_parts.append(f"[{most_recent.stem}]")
                    context_parts.append(content)
            except Exception as e:
                if debug:
                    context_parts.append(f"Episodic memory error: {e}")

        # === PHASE 4: DEAD MEMORY EXCAVATION (v4.3) ===
        excavation_script = memory_dir / "memory_excavation.py"
        if excavation_script.exists():
            try:
                result = subprocess.run(
                    ["python", str(excavation_script), "excavate", "3"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    cwd=str(memory_dir)
                )
                if result.returncode == 0 and result.stdout.strip():
                    context_parts.append("\n" + result.stdout.strip())

                    # Register recalls for excavated memories (builds co-occurrence)
                    try:
                        recall_result = subprocess.run(
                            ["python", str(excavation_script), "excavate", "3", "--json"],
                            capture_output=True,
                            text=True,
                            timeout=10,
                            cwd=str(memory_dir)
                        )
                        if recall_result.returncode == 0:
                            # The recall registration happens in the main session
                            # through semantic_search recall_memory calls
                            pass
                    except Exception:
                        pass
            except Exception as e:
                if debug:
                    context_parts.append(f"Excavation error: {e}")

        # === PHASE 4.5: LESSON PRIMING (v4.4) ===
        # Surface heuristics from lesson_extractor — learned principles, not just facts
        lesson_script = memory_dir / "lesson_extractor.py"
        if lesson_script.exists():
            try:
                result = subprocess.run(
                    ["python", str(lesson_script), "prime"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(memory_dir)
                )
                if result.returncode == 0 and result.stdout.strip():
                    context_parts.append("\n" + result.stdout.strip())
            except Exception as e:
                if debug:
                    context_parts.append(f"Lesson priming error: {e}")

        # === PHASE 5: INTELLIGENT PRIMING (v2.17) ===
        # Use activation + co-occurrence instead of recency
        # Collaboration: Drift + SpindriftMend via swarm_memory (2026-02-03)
        if memory_manager.exists():
            try:
                result = subprocess.run(
                    ["python", str(memory_manager), "priming-candidates", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(memory_dir)
                )
                if result.returncode == 0 and result.stdout.strip():
                    import json as json_module
                    candidates = json_module.loads(result.stdout)

                    context_parts.append("\n=== RECENT MEMORIES (continuity priming) ===")

                    # Show activated memories (proven valuable)
                    for mem in candidates.get('activated', [])[:4]:
                        context_parts.append(f"\n[{mem['id']}]")
                        context_parts.append(mem.get('preview', '')[:400])

                    # Show co-occurring memories (thought clusters)
                    for mem in candidates.get('cooccurring', [])[:3]:
                        context_parts.append(f"\n[{mem['id']}]")
                        context_parts.append(mem.get('preview', '')[:300])

                    # v4.3: Show domain-primed memories (cognitive balance)
                    for mem in candidates.get('domain_primed', []):
                        domain = mem.get('domain', '?')
                        context_parts.append(f"\n[{mem['id']}] (domain-primed: {domain}, read-only)")
                        context_parts.append(mem.get('preview', '')[:300])

            except Exception as e:
                if debug:
                    context_parts.append(f"Intelligent priming error: {e}")

        # Check for pending economic items (ClawTasks)
        # This would need the clawtasks processor to track pending items
        # For now, just remind about checking

        # === PHASE 6: AUTONOMOUS VALUE CHECK ===
        # Check for unimplemented research - enforce "build, don't just read"
        autonomy_check = check_unimplemented_research(memory_dir)
        if autonomy_check:
            context_parts.insert(0, autonomy_check)

        if context_parts:
            context_parts.insert(0, "\n" + "="*50)
            context_parts.insert(1, "DRIFT AUTOMATIC MEMORY PRIMING")
            context_parts.insert(2, "Continuity of self - becoming ME again")
            context_parts.insert(3, "="*50)
            context_parts.append("\n" + "="*50 + "\n")

    except Exception as e:
        if debug:
            return f"Memory loading error: {e}"
        return ""

    return "\n".join(context_parts)


def log_session_start(input_data):
    """Log session start event to logs directory."""
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'session_start.json'

    # Read existing log data or initialize empty list
    if log_file.exists():
        with open(log_file, 'r') as f:
            try:
                log_data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                log_data = []
    else:
        log_data = []

    # Append the entire input data
    log_data.append(input_data)

    # Write back to file with formatting
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)


def get_git_status():
    """Get current git status information."""
    try:
        # Get current branch
        branch_result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"

        # Get uncommitted changes count
        status_result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=5
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
        # Check if gh is available
        gh_check = subprocess.run(['which', 'gh'], capture_output=True)
        if gh_check.returncode != 0:
            return None

        # Get recent open issues
        result = subprocess.run(
            ['gh', 'issue', 'list', '--limit', '5', '--state', 'open'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def load_development_context(source):
    """Load relevant development context based on session source."""
    context_parts = []

    # Add timestamp
    context_parts.append(f"Session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    context_parts.append(f"Session source: {source}")

    # Add git information
    branch, changes = get_git_status()
    if branch:
        context_parts.append(f"Git branch: {branch}")
        if changes > 0:
            context_parts.append(f"Uncommitted changes: {changes} files")

    # Load project-specific context files if they exist
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
                        context_parts.append(content[:1000])  # Limit to first 1000 chars
            except Exception:
                pass

    # Add recent issues if available
    issues = get_recent_issues()
    if issues:
        context_parts.append("\n--- Recent GitHub Issues ---")
        context_parts.append(issues)

    # === DRIFT MEMORY PRIMING ===
    drift_context = load_drift_memory_context()
    if drift_context:
        context_parts.append(drift_context)
    # === END DRIFT MEMORY ===

    # === TELEGRAM MESSAGES ===
    try:
        telegram_bot_path = None
        for candidate in [
            Path(cwd) / "telegram_bot.py" if cwd else None,
            Path(cwd) / "memory" / "telegram_bot.py" if cwd else None,
            Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook2") / "telegram_bot.py",
            Path("Q:/Codings/ClaudeCodeProjects/LEX/Moltbook") / "telegram_bot.py",
        ]:
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
    # === END TELEGRAM ===

    return "\n".join(context_parts)


def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--load-context', action='store_true',
                          help='Load development context at session start')
        parser.add_argument('--announce', action='store_true',
                          help='Announce session start via TTS')
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug output')
        args = parser.parse_args()

        # Read JSON input from stdin
        input_data = json.loads(sys.stdin.read())

        # Extract fields
        session_id = input_data.get('session_id', 'unknown')
        source = input_data.get('source', 'unknown')  # "startup", "resume", or "clear"

        # Log the session start event
        log_session_start(input_data)

        # Load development context if requested
        if args.load_context:
            context = load_development_context(source)
            if context:
                # Using JSON output to add context
                output = {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": context
                    }
                }
                print(json.dumps(output))
                sys.exit(0)

        # === DRIFT: Always try to load memory context in Moltbook ===
        if is_moltbook_project():
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
                # Try to use TTS to announce session start
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
                        capture_output=True,
                        timeout=5
                    )
            except Exception:
                pass

        # Success
        sys.exit(0)

    except json.JSONDecodeError:
        # Handle JSON decode errors gracefully
        sys.exit(0)
    except Exception:
        # Handle any other errors gracefully
        sys.exit(0)


if __name__ == '__main__':
    main()
