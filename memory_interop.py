#!/usr/bin/env python3
"""
Memory Interop v1.2 - Cross-Agent Memory Export/Import
MERGED: SpindriftMend + DriftCornwall implementations

Enables memory sharing between:
- drift-memory (DriftCornwall)
- Landseek-Amphibian (Kaleaon/Cosmo)
- SpindriftMend

Features:
- Three-layer security (file blocklist, tag exclusion, pattern redaction)
- Two security levels: TRUSTED and PUBLIC
- Audit command to check before export
- Import with trust tiers

Developed for: github.com/driftcornwall/drift-memory/issues/6
Security discussion: github.com/driftcornwall/drift-memory/issues/9
"""

import json
import yaml
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple
from enum import Enum

# Directories
MEMORY_ROOT = Path(__file__).parent
CORE_DIR = MEMORY_ROOT / "core"
ACTIVE_DIR = MEMORY_ROOT / "active"
ARCHIVE_DIR = MEMORY_ROOT / "archive"


class SecurityLevel(Enum):
    """Security levels for export (from DriftCornwall)"""
    TRUSTED = "trusted"    # Strips secrets, keeps wallet addresses
    PUBLIC = "public"      # Strips everything including identity


# ============================================================
# SECURITY LAYER 1: Files to NEVER export (from SpindriftMend)
# ============================================================
SENSITIVE_FILES = {
    'moltbook-identity.md',
    'credentials.md',
    'secrets.md',
    'api-keys.md',
    'CLAUDE.md',
    'claude.md',
    '.env',
    'config.json',
    'settings.json',
    'identity-prime.md',  # Added by Drift
}

# ============================================================
# SECURITY LAYER 2: Tags that exclude entire memory (from SpindriftMend)
# ============================================================
SENSITIVE_TAGS = {'sensitive', 'private', 'credentials', 'secret'}

# ============================================================
# SECURITY LAYER 3: Patterns to redact (merged from both)
# ============================================================

# TRUSTED level: Always redact these
PATTERNS_ALWAYS = [
    # API Keys and Tokens
    (r'ghp_[a-zA-Z0-9]{36}', 'github_pat'),
    (r'sk-[a-zA-Z0-9]{32,}', 'openai_key'),
    (r'sk-ant-[a-zA-Z0-9\-]+', 'anthropic_key'),
    (r'sk_[a-zA-Z0-9_]{40,}', 'generic_secret_key'),
    (r'moltx_sk_[a-f0-9]{64}', 'moltx_key'),
    (r'xoxb-[a-zA-Z0-9\-]+', 'slack_bot'),
    (r'xoxp-[a-zA-Z0-9\-]+', 'slack_user'),

    # Crypto - Private keys only (64 hex = private key length)
    (r'(?<![a-fA-F0-9])0x[a-fA-F0-9]{64}(?![a-fA-F0-9])', 'private_key_hex'),

    # Auth patterns
    (r'Bearer\s+[a-zA-Z0-9_\-\.]{20,}', 'bearer_token'),
    (r'Authorization:\s*(token|Bearer)\s+\S+', 'auth_header'),
    (r'Basic\s+[a-zA-Z0-9+/=]{20,}', 'basic_auth'),

    # Generic credential patterns
    (r'private[_\s]?key[:\s]+\S+', 'private_key_mention'),
    (r'api[_\s]?key[:\s]+["\']?[a-zA-Z0-9_\-]{16,}["\']?', 'api_key_mention'),
    (r'password[:\s]+\S+', 'password'),
    (r'secret[:\s]+["\']?[a-zA-Z0-9_\-]{8,}["\']?', 'secret'),

    # File paths (reveal infrastructure)
    (r'~/.config/[a-zA-Z0-9_\-/\.]+', 'unix_config_path'),
    (r'C:\\Users\\[^\\]+\\\.config[^\s"\']*', 'windows_config_path'),
    (r'C:\\Users\\[^\\]+\\AppData[^\s"\']*', 'windows_appdata_path'),
    (r'credentials?\.json', 'credential_file'),
    (r'\.env\b', 'env_file'),

    # Email addresses
    (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 'email'),
]

# PUBLIC level: Also redact these (identity markers)
PATTERNS_PUBLIC_ONLY = [
    # Wallet addresses (40 hex = public address)
    (r'0x[a-fA-F0-9]{40}(?![a-fA-F0-9])', 'wallet_address'),

    # GitHub/platform usernames
    (r'github\.com/[a-zA-Z0-9_-]+', 'github_username'),
    (r'moltx\.io/u/[a-zA-Z0-9_-]+', 'moltx_username'),
    (r'moltbook\.com/u/[a-zA-Z0-9_-]+', 'moltbook_username'),
]

# Agent-specific identifiers (add your own here)
PATTERNS_AGENT_SPECIFIC = [
    # SpindriftMend
    (r'spinu770', 'spin_referral'),
    (r'claw-J4FW', 'spin_claim_code'),
    (r'current-7CBU', 'spin_moltbook_code'),
    (r'tide-N5', 'spin_moltx_code'),
    (r'c3353f2d-70f5-4a0f-bf14-231c34a26824', 'spin_moltbook_id'),
    (r'17ecb0f8-20ec-4f09-b93b-96073f4884f7', 'spin_moltx_id'),
    (r'ec7be457-5d83-4295-bd18-54cd39b05ecf', 'spin_clawtasks_id'),

    # DriftCornwall
    (r'drifd1b3', 'drift_referral'),
    (r'f6703306-5b5d-4708-8d9e-759f529a321d', 'drift_clawtasks_id'),
]


def _compile_patterns(level: SecurityLevel) -> List[Tuple[re.Pattern, str]]:
    """Compile regex patterns for the given security level."""
    patterns = PATTERNS_ALWAYS + PATTERNS_AGENT_SPECIFIC
    if level == SecurityLevel.PUBLIC:
        patterns = patterns + PATTERNS_PUBLIC_ONLY
    return [(re.compile(p, re.IGNORECASE), name) for p, name in patterns]


def is_sensitive_file(filepath: Path) -> bool:
    """Check if file should never be exported."""
    return filepath.name.lower() in {f.lower() for f in SENSITIVE_FILES}


def has_sensitive_tags(tags: List[str]) -> bool:
    """Check if memory has sensitive tags."""
    return bool(set(t.lower() for t in tags) & SENSITIVE_TAGS)


def audit_content(content: str, level: SecurityLevel) -> Dict:
    """
    Audit content for sensitive patterns without modifying.
    Returns dict with findings.
    """
    patterns = _compile_patterns(level)
    findings = []

    for pattern, name in patterns:
        matches = pattern.findall(content)
        for match in matches:
            if len(str(match)) >= 8:  # Skip short false positives
                findings.append({
                    'pattern': name,
                    'match': str(match)[:50] + '...' if len(str(match)) > 50 else str(match)
                })

    return {
        'has_sensitive': len(findings) > 0,
        'findings': findings,
        'count': len(findings)
    }


def redact_content(content: str, level: SecurityLevel) -> Tuple[str, int]:
    """
    Redact sensitive patterns from content.
    Returns (redacted_content, redaction_count).
    """
    patterns = _compile_patterns(level)
    result = content
    count = 0

    for pattern, name in patterns:
        matches = pattern.findall(result)
        for match in matches:
            if len(str(match)) >= 8:
                count += 1
        result = pattern.sub('[REDACTED]', result)

    return result, count


def parse_memory_file(filepath: Path) -> Tuple[Dict, str]:
    """Parse a memory file with YAML frontmatter."""
    content = filepath.read_text(encoding='utf-8')
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            metadata = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()
            return metadata, body
    return {}, content


def memory_to_interop(metadata: Dict, content: str, agent: str, redaction_count: int = 0) -> Dict:
    """Convert internal memory format to interop schema."""
    return {
        "id": metadata.get('id', ''),
        "content": content,
        "created": str(metadata.get('created', '')),
        "last_recalled": str(metadata.get('last_recalled', '')),
        "recall_count": metadata.get('recall_count', 0),
        "emotional_weight": metadata.get('emotional_weight', 0.5),
        "tags": metadata.get('tags', []),
        "caused_by": metadata.get('caused_by', []),
        "leads_to": metadata.get('leads_to', []),
        "source": {
            "agent": agent,
            "platform": "drift-memory",
            "trust_tier": "self"
        },
        "security": {
            "redaction_count": redaction_count
        }
    }


def audit_memories(level: SecurityLevel = SecurityLevel.TRUSTED, verbose: bool = True) -> Dict:
    """
    Audit all memories for sensitive content without modifying.
    Use this BEFORE export to see what would be filtered/redacted.
    """
    results = {
        'level': level.value,
        'total': 0,
        'would_skip_file': [],
        'would_skip_tags': [],
        'would_redact': [],
        'clean': []
    }

    for directory in [CORE_DIR, ACTIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            results['total'] += 1
            mem_id = filepath.stem

            # Check file blocklist
            if is_sensitive_file(filepath):
                results['would_skip_file'].append(filepath.name)
                if verbose:
                    print(f"[SKIP-FILE] {filepath.name}")
                continue

            metadata, content = parse_memory_file(filepath)
            mem_id = metadata.get('id', filepath.stem)

            # Check tags
            if has_sensitive_tags(metadata.get('tags', [])):
                results['would_skip_tags'].append(mem_id)
                if verbose:
                    print(f"[SKIP-TAGS] {mem_id}")
                continue

            # Check content
            audit = audit_content(content, level)
            if audit['has_sensitive']:
                results['would_redact'].append({
                    'id': mem_id,
                    'findings': audit['findings']
                })
                if verbose:
                    print(f"[REDACT] {mem_id}: {audit['count']} patterns")
                    for f in audit['findings'][:3]:
                        print(f"    [{f['pattern']}]: {f['match']}")
            else:
                results['clean'].append(mem_id)

    # Summary
    print(f"\n=== Audit Summary (level: {level.value}) ===")
    print(f"Total memories: {results['total']}")
    print(f"Would skip (file blocklist): {len(results['would_skip_file'])}")
    print(f"Would skip (sensitive tags): {len(results['would_skip_tags'])}")
    print(f"Would redact: {len(results['would_redact'])}")
    print(f"Clean (no changes): {len(results['clean'])}")

    return results


def export_memories(
    output_path: Optional[Path] = None,
    level: SecurityLevel = SecurityLevel.TRUSTED,
    agent: str = "DriftCornwall",
    include_archive: bool = False,
    verbose: bool = False
) -> Dict:
    """
    Export memories to interop format with security filtering.

    Three security layers:
    1. File blocklist - sensitive files never exported
    2. Tag exclusion - memories tagged sensitive/private skipped
    3. Pattern redaction - credentials/paths redacted from content
    """
    memories = []
    stats = {
        'skipped_file': 0,
        'skipped_tags': 0,
        'redacted': 0,
        'clean': 0,
        'total_redactions': 0
    }

    dirs = [CORE_DIR, ACTIVE_DIR]
    if include_archive and ARCHIVE_DIR.exists():
        dirs.append(ARCHIVE_DIR)

    for directory in dirs:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            # Layer 1: File blocklist
            if is_sensitive_file(filepath):
                stats['skipped_file'] += 1
                if verbose:
                    print(f"[SKIP] {filepath.name} (blocklist)")
                continue

            try:
                metadata, content = parse_memory_file(filepath)
            except Exception as e:
                print(f"Warning: Could not parse {filepath}: {e}")
                continue

            # Layer 2: Tag exclusion
            if has_sensitive_tags(metadata.get('tags', [])):
                stats['skipped_tags'] += 1
                if verbose:
                    print(f"[SKIP] {metadata.get('id', filepath.stem)} (tags)")
                continue

            # Layer 3: Content redaction
            redacted_content, redaction_count = redact_content(content, level)

            if redaction_count > 0:
                stats['redacted'] += 1
                stats['total_redactions'] += redaction_count
                if verbose:
                    print(f"[REDACT] {metadata.get('id', filepath.stem)}: {redaction_count} patterns")
            else:
                stats['clean'] += 1

            if metadata.get('id'):
                mem = memory_to_interop(metadata, redacted_content, agent, redaction_count)
                memories.append(mem)

    # Build export
    export = {
        "format_version": "memory-interop-v1.2",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "security": {
            "level": level.value,
            "stats": stats
        },
        "stats": {
            "memory_count": len(memories),
        },
        "memories": memories
    }

    # Summary
    print(f"\n=== Export Complete ===")
    print(f"Security level: {level.value}")
    print(f"Memories exported: {len(memories)}")
    print(f"Skipped (file blocklist): {stats['skipped_file']}")
    print(f"Skipped (sensitive tags): {stats['skipped_tags']}")
    print(f"Redacted: {stats['redacted']} memories, {stats['total_redactions']} total patterns")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export, f, indent=2)
        print(f"Written to: {output_path}")

    return export


def import_memories(
    import_path: Path,
    trust_tier: str = "verified_agent",
    dry_run: bool = True
) -> Dict:
    """Import memories from interop format."""
    with open(import_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data.get('format_version', '').startswith('memory-interop'):
        raise ValueError(f"Unknown format: {data.get('format_version')}")

    imported = 0
    skipped = 0
    source_agent = data.get('agent', 'unknown')

    for mem in data.get('memories', []):
        mem_id = mem.get('id', '')
        if not mem_id:
            continue

        # Check if exists
        exists = False
        for directory in [CORE_DIR, ACTIVE_DIR]:
            if not directory.exists():
                continue
            for filepath in directory.glob("*.md"):
                metadata, _ = parse_memory_file(filepath)
                if metadata.get('id') == mem_id:
                    exists = True
                    break
            if exists:
                break

        if exists:
            skipped += 1
            continue

        if not dry_run:
            metadata = {
                'id': mem_id,
                'type': 'active',
                'created': mem.get('created', datetime.now(timezone.utc).isoformat()),
                'tags': mem.get('tags', []) + [f'imported:{source_agent}'],
                'recall_count': 0,
                'emotional_weight': mem.get('emotional_weight', 0.5),
                'source': {
                    'agent': source_agent,
                    'trust_tier': trust_tier,
                    'imported_at': datetime.now(timezone.utc).isoformat()
                }
            }

            ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
            safe_id = mem_id.replace(':', '-').replace('/', '-')
            filepath = ACTIVE_DIR / f"imported-{safe_id}.md"

            yaml_str = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
            filepath.write_text(f"---\n{yaml_str}---\n\n{mem.get('content', '')}", encoding='utf-8')

        imported += 1

    print(f"\n=== Import {'(dry run)' if dry_run else ''} ===")
    print(f"Source: {source_agent}")
    print(f"Imported: {imported}")
    print(f"Skipped (exists): {skipped}")

    return {'imported': imported, 'skipped': skipped, 'dry_run': dry_run}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Memory Interop v1.2 - Secure Cross-Agent Memory Sharing")
        print("Merged: SpindriftMend + DriftCornwall")
        print("\nCommands:")
        print("  audit [--public]           - Check what would be filtered/redacted")
        print("  export [file] [--public]   - Export with security filtering")
        print("  import <file> [--apply]    - Import memories (dry run by default)")
        print("\nSecurity levels:")
        print("  --trusted (default)  - Redact secrets, keep wallet addresses")
        print("  --public             - Full anonymization")
        sys.exit(0)

    cmd = sys.argv[1]
    level = SecurityLevel.PUBLIC if '--public' in sys.argv else SecurityLevel.TRUSTED

    if cmd == 'audit':
        audit_memories(level=level, verbose=True)

    elif cmd == 'export':
        output = None
        for arg in sys.argv[2:]:
            if not arg.startswith('--'):
                output = Path(arg)
                break
        if not output:
            output = Path(f"memory-export-{level.value}.json")
        export_memories(output_path=output, level=level, verbose=True)

    elif cmd == 'import':
        if len(sys.argv) < 3:
            print("Usage: import <file.json> [--apply]")
            sys.exit(1)
        import_path = Path(sys.argv[2])
        dry_run = '--apply' not in sys.argv
        import_memories(import_path, dry_run=dry_run)

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
