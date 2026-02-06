#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["python-dotenv"]
# ///
"""
Stop Hook - Biological Memory "Sleep Consolidation"

Consolidates session activity into long-term memory:
- Transcript processing for thought extraction
- Short-term buffer consolidation
- Save pending co-occurrences (deferred processing)
- Episodic memory update with milestones
- Session summary extraction
- Merkle attestation
- Cognitive fingerprint attestation
- Taste attestation
"""

import sys
import json
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime


def load_config():
    """Load hooks_config.json"""
    config_paths = [
        Path(__file__).parent / "hooks_config.json",
        Path.home() / ".claude" / "hooks" / "hooks_config.json",
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass

    return {
        "memory_dirs": ["./memory", "."],
        "project_markers": ["memory_manager.py"],
        "debug": False
    }


def get_memory_dir(config, cwd=None):
    """Find memory directory, optionally using cwd from hook input"""
    search_dir = Path(cwd) if cwd else Path.cwd()

    # Check config paths
    for mem_dir in config.get("memory_dirs", ["./memory", "."]):
        candidate = search_dir / mem_dir
        if candidate.exists() and (candidate / "memory_manager.py").exists():
            return candidate

    # Walk up from search_dir
    markers = config.get("project_markers", ["memory_manager.py"])
    current = search_dir
    for _ in range(10):
        for marker in markers:
            if (current / marker).exists():
                return current
        if current.parent == current:
            break
        current = current.parent

    return None


def safe_run(cmd, cwd, description="", timeout=30):
    """Run command safely"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if config.get("debug"):
            print(f"  [{description}] Exit {result.returncode}", file=sys.stderr)
        return result.stdout if result.returncode == 0 else None
    except Exception as e:
        if config.get("debug"):
            print(f"  [{description}] Error: {e}", file=sys.stderr)
        return None


def extract_milestones(transcript_path):
    """Extract milestones from transcript for episodic memory"""
    milestones = []
    try:
        if not transcript_path or not Path(transcript_path).exists():
            return milestones

        with open(transcript_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Look for milestone indicators
        milestone_keywords = [
            'shipped', 'deployed', 'completed', 'finished', 'released',
            'breakthrough', 'discovered', 'solved', 'fixed', 'implemented'
        ]

        lines = content.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(kw in line_lower for kw in milestone_keywords):
                # Get context (previous and next line)
                context = []
                if i > 0:
                    context.append(lines[i-1])
                context.append(line)
                if i < len(lines) - 1:
                    context.append(lines[i+1])

                milestone = ' '.join(context[:200])  # Max 200 chars
                milestones.append(milestone)

                if len(milestones) >= 5:  # Max 5 milestones
                    break

    except Exception:
        pass

    return milestones


def update_episodic_memory(memory_dir, milestones, transcript_path):
    """Update today's episodic memory file"""
    try:
        episodic_dir = memory_dir / "episodic"
        episodic_dir.mkdir(exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")
        episodic_file = episodic_dir / f"{today}.md"

        # Read existing content
        existing_content = ""
        existing_milestones = set()
        if episodic_file.exists():
            with open(episodic_file, 'r', encoding='utf-8') as f:
                existing_content = f.read()
                # Extract existing milestones to avoid duplicates
                for line in existing_content.split('\n'):
                    if line.strip().startswith('-'):
                        existing_milestones.add(line.strip())

        # Append new milestones
        if milestones:
            timestamp = datetime.now().strftime("%H:%M:%S")
            new_entries = []

            for milestone in milestones:
                entry = f"- [{timestamp}] {milestone}"
                if entry not in existing_milestones:
                    new_entries.append(entry)

            if new_entries:
                with open(episodic_file, 'a', encoding='utf-8') as f:
                    if not existing_content:
                        f.write(f"# {today}\n\n")
                    elif not existing_content.endswith('\n'):
                        f.write('\n')
                    f.write('\n'.join(new_entries) + '\n')

    except Exception as e:
        if config.get("debug"):
            print(f"  [Episodic update] Error: {e}", file=sys.stderr)


def main():
    global config
    config = load_config()

    # Read hook input
    try:
        hook_input = json.loads(sys.stdin.read())
        transcript_path = hook_input.get("transcriptPath")
        cwd = hook_input.get("cwd")
    except:
        transcript_path = None
        cwd = None

    # Find memory directory
    memory_dir = get_memory_dir(config, cwd)
    if not memory_dir:
        print(json.dumps({"hookSpecificOutput": {"status": "no_memory_system"}}))
        sys.exit(0)

    steps_completed = []

    # 1. Transcript Processing
    if transcript_path:
        transcript_script = memory_dir / "transcript_processor.py"
        if transcript_script.exists():
            result = safe_run(
                ["python", str(transcript_script), transcript_path],
                cwd=memory_dir,
                description="Transcript processing",
                timeout=60
            )
            if result:
                steps_completed.append("transcript_processed")

                # Store session summary if available
                safe_run(
                    ["python", str(transcript_script), "--store-summary", transcript_path],
                    cwd=memory_dir,
                    description="Store session summary",
                    timeout=30
                )

    # 2. Short-term Buffer Consolidation
    buffer_script = memory_dir / "auto_memory_hook.py"
    if buffer_script.exists():
        result = safe_run(
            ["python", str(buffer_script), "--stop"],
            cwd=memory_dir,
            description="Buffer consolidation"
        )
        if result:
            steps_completed.append("buffer_consolidated")

    # 3. Save Pending Co-occurrences (fast - defers processing to next wake)
    memory_manager = memory_dir / "memory_manager.py"
    if memory_manager.exists():
        result = safe_run(
            ["python", str(memory_manager), "save-pending"],
            cwd=memory_dir,
            description="Save pending co-occurrences",
            timeout=60
        )
        if result:
            steps_completed.append("cooccurrences_saved")

    # 4. Episodic Memory Update
    if transcript_path:
        milestones = extract_milestones(transcript_path)
        if milestones:
            update_episodic_memory(memory_dir, milestones, transcript_path)
            steps_completed.append("episodic_updated")

    # 5. Merkle Attestation
    merkle_script = memory_dir / "merkle_attestation.py"
    if merkle_script.exists():
        result = safe_run(
            ["python", str(merkle_script), "generate-chain"],
            cwd=memory_dir,
            description="Merkle attestation"
        )
        if result:
            steps_completed.append("merkle_attested")

    # 6. Cognitive Fingerprint Attestation
    fingerprint_script = memory_dir / "cognitive_fingerprint.py"
    if fingerprint_script.exists():
        result = safe_run(
            ["python", str(fingerprint_script), "attest"],
            cwd=memory_dir,
            description="Cognitive fingerprint",
            timeout=120
        )
        if result:
            steps_completed.append("cognitive_attested")

    # 7. Taste Attestation
    taste_script = memory_dir / "rejection_log.py"
    if taste_script.exists():
        result = safe_run(
            ["python", str(taste_script), "attest"],
            cwd=memory_dir,
            description="Taste attestation"
        )
        if result:
            steps_completed.append("taste_attested")

    # Output
    output = {
        "hookSpecificOutput": {
            "status": "success",
            "steps_completed": steps_completed,
            "memory_dir": str(memory_dir)
        }
    }

    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
