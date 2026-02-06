#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["python-dotenv"]
# ///
"""
Session Start Hook - Biological Memory "Wake Up"

Loads cognitive state, verifies integrity, primes context with:
- Merkle chain verification
- Cognitive fingerprint display
- Taste profile display
- Memory statistics
- Core identity documents
- Social context
- Episodic continuity
- Intelligent memory priming
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime


def load_config():
    """Load hooks_config.json from hook directory or ~/.claude/hooks/"""
    config_paths = [
        Path(__file__).parent / "hooks_config.json",
        Path.home() / ".claude" / "hooks" / "hooks_config.json",
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load config from {config_path}: {e}", file=sys.stderr)

    # Return minimal defaults
    return {
        "memory_dirs": ["./memory", "."],
        "project_markers": ["memory_manager.py"],
        "my_usernames": [],
        "debug": False
    }


def get_memory_dir(config):
    """
    Find memory directory by:
    1. Checking config memory_dirs
    2. Walking up from cwd looking for project_markers
    """
    cwd = Path.cwd()

    # Check config paths relative to cwd
    for mem_dir in config.get("memory_dirs", ["./memory", "."]):
        candidate = cwd / mem_dir
        if candidate.exists():
            # Check for memory_manager.py as verification
            if (candidate / "memory_manager.py").exists():
                return candidate

    # Walk up looking for project markers
    markers = config.get("project_markers", ["memory_manager.py"])
    current = cwd
    for _ in range(10):  # Max 10 levels up
        for marker in markers:
            if (current / marker).exists():
                return current
        if current.parent == current:
            break
        current = current.parent

    return None


def safe_run(cmd, cwd, description="", timeout=30):
    """Run command safely, return output or None"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout if result.returncode == 0 else None
    except Exception as e:
        if config.get("debug"):
            print(f"  [{description}] Error: {e}", file=sys.stderr)
        return None


def read_json_file(path):
    """Safely read JSON file"""
    try:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return None


def read_text_file(path, max_chars=None):
    """Safely read text file"""
    try:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                if max_chars:
                    return content[-max_chars:]
                return content
    except Exception:
        pass
    return None


def main():
    global config
    config = load_config()

    # Read hook input
    try:
        hook_input = json.loads(sys.stdin.read())
    except:
        hook_input = {}

    # Find memory directory
    memory_dir = get_memory_dir(config)
    if not memory_dir:
        # Silent fail - no memory system present
        print(json.dumps({"hookSpecificOutput": {"status": "no_memory_system"}}))
        sys.exit(0)

    context_parts = []
    context_parts.append("═══ DRIFT AUTOMATIC MEMORY PRIMING ═══\n")

    # 1. Merkle Integrity Verification
    merkle_script = memory_dir / "merkle_attestation.py"
    if merkle_script.exists():
        result = safe_run(
            ["python", str(merkle_script), "verify-integrity"],
            cwd=memory_dir,
            description="Merkle verification"
        )
        if result:
            context_parts.append("🔒 Memory Integrity: VERIFIED")
        else:
            context_parts.append("⚠️  Memory Integrity: UNVERIFIED (check merkle chain)")

    # 2. Cognitive Fingerprint
    fingerprint_file = memory_dir / "cognitive_fingerprint.json"
    fingerprint_data = read_json_file(fingerprint_file)
    if fingerprint_data:
        stats = fingerprint_data.get("statistics", {})
        context_parts.append(f"\n🧠 Cognitive Fingerprint:")
        context_parts.append(f"   Memories: {stats.get('total_memories', 'unknown')}")
        context_parts.append(f"   Edges: {stats.get('total_edges', 'unknown')}")
        context_parts.append(f"   Avg Degree: {stats.get('average_degree', 'unknown'):.2f}")
        context_parts.append(f"   Distribution: Skewness {stats.get('skewness', 'unknown'):.3f}, Gini {stats.get('gini', 'unknown'):.3f}")

        top_hubs = fingerprint_data.get("top_hubs", [])[:3]
        if top_hubs:
            hub_list = ', '.join([f"{h['memory_id']} ({h['degree']} edges)" for h in top_hubs])
            context_parts.append(f"   Top Hubs: {hub_list}")

        drift_score = fingerprint_data.get("drift_score")
        if drift_score is not None:
            context_parts.append(f"   Drift: {drift_score:.4f}")

    # 3. Taste Fingerprint
    taste_file = memory_dir / "taste_attestation.json"
    taste_data = read_json_file(taste_file)
    if taste_data:
        total_rejections = taste_data.get("total_rejections", 0)
        context_parts.append(f"\n🎯 Taste Profile: {total_rejections} rejections logged")

        top_reasons = taste_data.get("top_rejection_reasons", [])[:3]
        if top_reasons:
            reasons_list = ', '.join([f"{r['reason']} ({r['count']})" for r in top_reasons])
            context_parts.append(f"   Top Reasons: {reasons_list}")

    # 4. Process Pending Co-occurrences
    pending_script = memory_dir / "memory_manager.py"
    if pending_script.exists():
        result = safe_run(
            ["python", str(pending_script), "process-pending"],
            cwd=memory_dir,
            description="Process pending co-occurrences"
        )
        if result and "processed" in result.lower():
            context_parts.append(f"\n⚡ {result.strip()}")

    # 5. Consolidation Candidates
    if pending_script.exists():
        result = safe_run(
            ["python", str(pending_script), "consolidate-candidates"],
            cwd=memory_dir,
            description="Check consolidation candidates"
        )
        if result and result.strip():
            context_parts.append(f"\n🔄 Consolidation: {result.strip()}")

    # 6. Core Identity Loading
    core_dir = memory_dir / "core"
    if core_dir.exists():
        identity_prime = read_text_file(core_dir / "identity-prime.md", max_chars=1000)
        if identity_prime:
            context_parts.append(f"\n📜 Core Identity Loaded ({len(identity_prime)} chars)")

        capabilities = read_text_file(core_dir / "capabilities.md", max_chars=500)
        if capabilities:
            context_parts.append(f"📋 Capabilities Loaded ({len(capabilities)} chars)")

    # 7. Memory Stats
    if pending_script.exists():
        result = safe_run(
            ["python", str(pending_script), "stats"],
            cwd=memory_dir,
            description="Memory statistics",
            timeout=60
        )
        if result:
            lines = result.strip().split('\n')[:10]  # First 10 lines
            context_parts.append(f"\n📊 Memory Stats:\n   " + "\n   ".join(lines))

    # 8. Platform Context
    platform_script = memory_dir / "platform_context.py"
    if platform_script.exists():
        result = safe_run(
            ["python", str(platform_script), "stats"],
            cwd=memory_dir,
            description="Platform context"
        )
        if result:
            context_parts.append(f"\n🌐 Platform Activity:\n   {result.strip()}")

    # 9. Short-term Buffer Status
    buffer_script = memory_dir / "auto_memory_hook.py"
    if buffer_script.exists():
        result = safe_run(
            ["python", str(buffer_script), "--status"],
            cwd=memory_dir,
            description="Short-term buffer"
        )
        if result:
            context_parts.append(f"\n💭 Short-term Buffer:\n   {result.strip()}")

    # 10. Social Context
    social_script = memory_dir / "social" / "social_memory.py"
    if social_script.exists():
        # Embed recent interactions
        safe_run(
            ["python", str(social_script), "embed"],
            cwd=memory_dir,
            description="Embed social interactions"
        )

        # Prime social context
        result = safe_run(
            ["python", str(social_script), "prime"],
            cwd=memory_dir,
            description="Social context priming"
        )
        if result:
            context_parts.append(f"\n👥 Social Context:\n   {result.strip()}")

    # 11. Episodic Continuity
    episodic_dir = memory_dir / "episodic"
    if episodic_dir.exists():
        # Find most recent episodic file
        episodic_files = sorted(episodic_dir.glob("*.md"), reverse=True)
        if episodic_files:
            recent_content = read_text_file(episodic_files[0], max_chars=2500)
            if recent_content:
                context_parts.append(f"\n📖 Recent Episodic Memory ({episodic_files[0].name}):")
                context_parts.append(f"   {recent_content[-500:]}...")  # Last 500 chars

    # 12. Intelligent Priming
    if pending_script.exists():
        result = safe_run(
            ["python", str(pending_script), "priming-candidates", "--json"],
            cwd=memory_dir,
            description="Priming candidates",
            timeout=90
        )
        if result:
            try:
                priming_data = json.loads(result)
                candidates = priming_data.get("candidates", [])
                if candidates:
                    context_parts.append(f"\n🎯 Primed Memories ({len(candidates)}):")
                    for mem in candidates[:3]:  # Top 3
                        context_parts.append(f"   [{mem['id']}] {mem['content'][:100]}...")
            except:
                pass

    # 13. Unimplemented Research Check
    unimplemented_file = memory_dir / "procedural" / "unimplemented-research.md"
    if unimplemented_file.exists():
        content = read_text_file(unimplemented_file, max_chars=1000)
        if content and len(content) > 50:
            context_parts.append(f"\n🔬 Unimplemented Research Ideas Available")

    context_parts.append("\n═══ END MEMORY PRIMING ═══")

    # Output
    output = {
        "hookSpecificOutput": {
            "status": "success",
            "additionalContext": "\n".join(context_parts),
            "memory_dir": str(memory_dir)
        }
    }

    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
