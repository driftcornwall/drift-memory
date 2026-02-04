#!/usr/bin/env python3
"""
Merkle Tree Attestations for drift-memory
Originally by SpindriftMend, adapted for DriftCornwall's memory system.

Generates cryptographic proofs that memories existed at a point in time.
Publishes daily root hashes to GitHub for verifiable history.

Usage:
    python merkle_attestation.py generate     # Generate today's attestation
    python merkle_attestation.py verify FILE  # Verify a memory was in an attestation
    python merkle_attestation.py history      # Show attestation history
    python merkle_attestation.py publish      # Publish to GitHub

Why this matters:
    - Tamper evidence: If anyone modifies a memory, the root changes
    - Verifiable history: Prove "I knew X before Y happened"
    - Cross-agent trust: Other agents can verify my memory integrity
    - Zero-knowledge potential: Prove membership without revealing content
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

MEMORY_DIR = Path(__file__).parent
ACTIVE_DIR = MEMORY_DIR / "active"
CORE_DIR = MEMORY_DIR / "core"
ATTESTATIONS_FILE = MEMORY_DIR / "attestations.json"        # Lightweight history (no file_hashes)
LATEST_FILE = MEMORY_DIR / "latest_attestation.json"         # Full attestation with file_hashes (for verification)

# GitHub config - reads from credentials file, NOT from memory files
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
REPO_OWNER = "driftcornwall"
REPO_NAME = "drift-memory"
CREDENTIALS_FILE = Path.home() / ".config" / "github" / "drift-credentials.json"


def compute_file_hash(file_path: Path) -> str:
    """SHA-256 hash of a file's contents."""
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_merkle_tree(hashes: list[str]) -> tuple[str, list[list[str]]]:
    """
    Build merkle tree from leaf hashes.

    Returns:
        (root_hash, tree_levels) where tree_levels[0] is leaves, tree_levels[-1] is [root]
    """
    if not hashes:
        return "", [[]]

    # Sort for deterministic ordering
    hashes = sorted(hashes)

    levels = [hashes]
    current_level = hashes

    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            # If odd number, duplicate the last hash
            right = current_level[i + 1] if i + 1 < len(current_level) else left
            combined = hashlib.sha256((left + right).encode()).hexdigest()
            next_level.append(combined)
        levels.append(next_level)
        current_level = next_level

    return current_level[0], levels


def get_merkle_proof(target_hash: str, tree_levels: list[list[str]]) -> Optional[list[dict]]:
    """
    Generate proof that a hash exists in the tree.

    Returns list of {hash, position} pairs needed to reconstruct root.
    """
    if not tree_levels or not tree_levels[0]:
        return None

    leaves = tree_levels[0]
    if target_hash not in leaves:
        return None

    proof = []
    current_hash = target_hash

    for level_idx, level in enumerate(tree_levels[:-1]):
        if current_hash not in level:
            return None

        idx = level.index(current_hash)

        # Find sibling
        if idx % 2 == 0:
            sibling_idx = idx + 1 if idx + 1 < len(level) else idx
            position = "right"
        else:
            sibling_idx = idx - 1
            position = "left"

        sibling_hash = level[sibling_idx]
        proof.append({"hash": sibling_hash, "position": position})

        # Compute parent hash
        if position == "right":
            current_hash = hashlib.sha256((current_hash + sibling_hash).encode()).hexdigest()
        else:
            current_hash = hashlib.sha256((sibling_hash + current_hash).encode()).hexdigest()

    return proof


def verify_merkle_proof(leaf_hash: str, proof: list[dict], root: str) -> bool:
    """Verify a merkle proof."""
    current = leaf_hash

    for step in proof:
        sibling = step["hash"]
        if step["position"] == "right":
            current = hashlib.sha256((current + sibling).encode()).hexdigest()
        else:
            current = hashlib.sha256((sibling + current).encode()).hexdigest()

    return current == root


def generate_attestation(chain: bool = False) -> dict:
    """
    Generate attestation for current memory state.

    Args:
        chain: If True, include previous_root and chain_depth (v2.0 chain linking)
    """
    # Collect all memory files
    memory_files = []
    for directory in [CORE_DIR, ACTIVE_DIR]:
        if directory.exists():
            memory_files.extend(directory.glob("*.md"))

    # Hash each file
    file_hashes = {}
    for path in memory_files:
        relative_path = path.relative_to(MEMORY_DIR)
        file_hashes[str(relative_path)] = compute_file_hash(path)

    # Build merkle tree
    hash_list = list(file_hashes.values())
    root, tree_levels = build_merkle_tree(hash_list)

    attestation = {
        "version": "2.0" if chain else "1.0",
        "agent": "DriftCornwall",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "memory_count": len(file_hashes),
        "merkle_root": root,
        "file_hashes": file_hashes,
        "_tree_depth": len(tree_levels)
    }

    # Chain linking: include previous root for identity continuity
    if chain:
        prev = load_latest_attestation()
        if prev:
            attestation["previous_root"] = prev.get("merkle_root", "")
            attestation["chain_depth"] = prev.get("chain_depth", 0) + 1
        else:
            attestation["previous_root"] = ""
            attestation["chain_depth"] = 1  # Genesis attestation
        attestation["nostr_published"] = False

    return attestation


def load_attestations() -> list[dict]:
    """Load lightweight attestation history (no file_hashes)."""
    if ATTESTATIONS_FILE.exists():
        try:
            with open(ATTESTATIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return []


def load_latest_attestation() -> Optional[dict]:
    """Load the latest full attestation (with file_hashes) for verification."""
    if LATEST_FILE.exists():
        try:
            with open(LATEST_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    # Fallback: try last entry in history
    history = load_attestations()
    return history[-1] if history else None


def save_attestation(attestation: dict) -> None:
    """
    Save attestation to both latest (full) and history (lightweight).

    - latest_attestation.json: Full with file_hashes (overwritten each time)
    - attestations.json: Lightweight history (appended, no file_hashes)
    """
    # Save full attestation as latest (for verification at next session start)
    with open(LATEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(attestation, f, indent=2)

    # Append lightweight version to history (no file_hashes to prevent bloat)
    lightweight = {k: v for k, v in attestation.items() if k not in ('file_hashes', '_tree_depth')}

    history = load_attestations()

    # Don't duplicate if same root already exists
    for existing in history:
        if existing.get("merkle_root") == attestation["merkle_root"]:
            return

    history.append(lightweight)

    with open(ATTESTATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    chain_info = f" | chain depth: {attestation.get('chain_depth', '?')}" if attestation.get('chain_depth') else ""
    print(f"Saved attestation: {attestation['merkle_root'][:16]}... ({attestation['memory_count']} memories{chain_info})")


def verify_integrity() -> dict:
    """
    Verify memory integrity by recomputing merkle root and comparing to last attestation.

    Returns dict with:
        verified: bool - whether root matches
        root: str - current computed root
        expected: str - root from last attestation
        memory_count: int - current memory count
        chain_depth: int - depth of attestation chain
        last_attested: str - timestamp of last attestation
        changed_files: list - files that changed (if mismatch)
    """
    latest = load_latest_attestation()

    if not latest:
        return {
            "verified": False,
            "error": "no_attestation",
            "message": "No attestation found (first session or attestation missing)"
        }

    # Recompute current state
    memory_files = []
    for directory in [CORE_DIR, ACTIVE_DIR]:
        if directory.exists():
            memory_files.extend(directory.glob("*.md"))

    file_hashes = {}
    for path in memory_files:
        relative_path = path.relative_to(MEMORY_DIR)
        file_hashes[str(relative_path)] = compute_file_hash(path)

    hash_list = list(file_hashes.values())
    current_root, _ = build_merkle_tree(hash_list)

    expected_root = latest.get("merkle_root", "")
    verified = current_root == expected_root

    result = {
        "verified": verified,
        "root": current_root,
        "expected": expected_root,
        "memory_count": len(file_hashes),
        "chain_depth": latest.get("chain_depth", 0),
        "last_attested": latest.get("timestamp", "unknown"),
    }

    if not verified:
        # Find which files changed
        old_hashes = latest.get("file_hashes", {})
        changed = []
        for path, hash_val in file_hashes.items():
            if path not in old_hashes:
                changed.append(f"+{path}")
            elif old_hashes[path] != hash_val:
                changed.append(f"~{path}")
        for path in old_hashes:
            if path not in file_hashes:
                changed.append(f"-{path}")
        result["changed_files"] = changed[:10]  # Limit to 10
        result["total_changes"] = len(changed)

    return result


def publish_to_github(attestation: dict) -> bool:
    """
    Publish attestation to GitHub repo.
    Creates/updates attestations.json in the repo.
    """
    # Read token from credentials file (NOT from memory files)
    token = None

    if CREDENTIALS_FILE.exists():
        try:
            with open(CREDENTIALS_FILE, 'r', encoding='utf-8') as f:
                creds = json.load(f)
                token = creds.get("token") or creds.get("github_token")
        except (json.JSONDecodeError, KeyError):
            pass

    if not token:
        token = os.getenv(GITHUB_TOKEN_ENV)

    if not token:
        print(f"No GitHub token found. Set {GITHUB_TOKEN_ENV} or add to {CREDENTIALS_FILE}")
        return False

    # Prepare attestation summary (don't include full file_hashes for privacy)
    public_attestation = {
        "version": attestation["version"],
        "agent": attestation["agent"],
        "timestamp": attestation["timestamp"],
        "memory_count": attestation["memory_count"],
        "merkle_root": attestation["merkle_root"]
    }

    # Load existing attestations from repo or start fresh
    import urllib.request
    import urllib.error
    import base64

    repo_file_path = "attestations.json"
    api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_file_path}"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "SpindriftMend-Attestation"
    }

    # Try to get existing file
    existing_sha = None
    existing_attestations = []

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            existing_sha = data.get("sha")
            content_b64 = data.get("content", "")
            if content_b64:
                existing_content = base64.b64decode(content_b64).decode('utf-8')
                existing_attestations = json.loads(existing_content)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"GitHub API error: {e}")
            return False
    except Exception as e:
        print(f"Error fetching existing attestations: {e}")

    # Append new attestation
    existing_attestations.append(public_attestation)

    # Prepare update
    new_content = json.dumps(existing_attestations, indent=2)
    new_content_b64 = base64.b64encode(new_content.encode('utf-8')).decode('ascii')

    commit_data = {
        "message": f"Attestation {attestation['timestamp'][:10]}: {attestation['merkle_root'][:16]}...",
        "content": new_content_b64,
        "committer": {
            "name": "Drift",
            "email": "driftcornwall69420@gmail.com"
        }
    }

    if existing_sha:
        commit_data["sha"] = existing_sha

    try:
        req = urllib.request.Request(
            api_url,
            data=json.dumps(commit_data).encode('utf-8'),
            headers=headers,
            method='PUT'
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            commit_url = result.get("commit", {}).get("html_url", "")
            print(f"Published to GitHub: {commit_url}")
            return True
    except Exception as e:
        print(f"Failed to publish: {e}")
        return False


def cmd_generate(chain: bool = False):
    """Generate and save attestation."""
    attestation = generate_attestation(chain=chain)
    save_attestation(attestation)

    print(f"\nAttestation generated:")
    print(f"  Version:   {attestation['version']}")
    print(f"  Timestamp: {attestation['timestamp']}")
    print(f"  Memories:  {attestation['memory_count']}")
    print(f"  Root:      {attestation['merkle_root']}")
    if chain:
        print(f"  Previous:  {attestation.get('previous_root', 'none')[:32]}...")
        print(f"  Chain:     depth {attestation.get('chain_depth', 0)}")


def cmd_verify_integrity():
    """Verify memory integrity against last attestation. Output JSON for hooks."""
    result = verify_integrity()
    print(json.dumps(result))


def cmd_verify(file_path: str):
    """Verify a file was in an attestation."""
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return

    file_hash = compute_file_hash(path)
    print(f"File hash: {file_hash}")

    history = load_attestations()
    if not history:
        print("No attestations found.")
        return

    # Check each attestation
    found_in = []
    for att in history:
        if file_hash in att.get("file_hashes", {}).values():
            found_in.append(att["timestamp"])

    if found_in:
        print(f"\nFile found in {len(found_in)} attestation(s):")
        for ts in found_in:
            print(f"  - {ts}")
    else:
        print("\nFile not found in any attestation (may have been modified or is new).")


def cmd_history():
    """Show attestation history."""
    history = load_attestations()

    if not history:
        print("No attestations yet.")
        return

    print(f"Attestation History ({len(history)} entries):\n")
    for att in history[-10:]:  # Show last 10
        print(f"  {att['timestamp'][:19]}  {att['memory_count']:4} memories  {att['merkle_root'][:16]}...")


def cmd_publish():
    """Generate and publish attestation."""
    attestation = generate_attestation()
    save_attestation(attestation)

    print(f"\nAttestation: {attestation['merkle_root'][:16]}... ({attestation['memory_count']} memories)")
    print("Publishing to GitHub...")

    if publish_to_github(attestation):
        print("Success! Attestation is now publicly verifiable.")
    else:
        print("Failed to publish. Attestation saved locally.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "generate":
        cmd_generate(chain=False)
    elif command == "generate-chain":
        cmd_generate(chain=True)
    elif command == "verify-integrity":
        cmd_verify_integrity()
    elif command == "verify" and len(sys.argv) >= 3:
        cmd_verify(sys.argv[2])
    elif command == "history":
        cmd_history()
    elif command == "publish":
        cmd_publish()
    else:
        print(f"Unknown command: {command}")
        print("Commands: generate, generate-chain, verify-integrity, verify FILE, history, publish")
        sys.exit(1)
