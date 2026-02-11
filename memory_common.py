#!/usr/bin/env python3
"""
Memory Common — Shared infrastructure for memory modules.

Extracted during memory_manager.py decomposition (Phase 2).
Contains directory constants, parse/write functions used by all memory modules.
"""

import yaml
from pathlib import Path

MEMORY_ROOT = Path(__file__).parent
CORE_DIR = MEMORY_ROOT / "core"
ACTIVE_DIR = MEMORY_ROOT / "active"
ARCHIVE_DIR = MEMORY_ROOT / "archive"
ALL_DIRS = [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]

# --- Shared configuration constants (used by multiple peer modules) ---

# Self-evolution — adaptive decay based on retrieval success (v2.13)
SELF_EVOLUTION_ENABLED = True

# Agent identity — configurable, not hardcoded
def get_agent_name() -> str:
    """Read agent name from identity config. Falls back to 'DriftCornwall'."""
    identity_file = MEMORY_ROOT / "core" / "identity.md"
    if identity_file.exists():
        meta, _ = parse_memory_file(identity_file)
        if meta.get('agent_name'):
            return meta['agent_name']
    return 'DriftCornwall'


def parse_memory_file(filepath: Path) -> tuple[dict, str]:
    """Parse a memory file with YAML frontmatter."""
    content = filepath.read_text(encoding='utf-8')
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            metadata = yaml.safe_load(parts[1])
            body = parts[2].strip()
            return metadata, body
    return {}, content


def write_memory_file(filepath: Path, metadata: dict, content: str):
    """Write a memory file with YAML frontmatter."""
    yaml_str = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
    filepath.write_text(f"---\n{yaml_str}---\n\n{content}", encoding='utf-8')
