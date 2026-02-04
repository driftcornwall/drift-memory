#!/usr/bin/env python3
"""
Merkle Tree Attestations for SpindriftMend's Memory System

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
ATTESTATIONS_FILE = MEMORY_DIR / "attestations.json"

# GitHub config (from identity file)
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"  # Or we read from identity file
REPO_OWNER = "SpindriftMind"
REPO_NAME = "SpindriftMind"


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


def generate_attestation() -> dict:
    """Generate attestation for current memory state."""
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
        "version": "1.0",
        "agent": "SpindriftMend",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "memory_count": len(file_hashes),
        "merkle_root": root,
        "file_hashes": file_hashes,
        # Store tree for proof generation (optional, can be recomputed)
        "_tree_depth": len(tree_levels)
    }

    return attestation


def load_attestations() -> list[dict]:
    """Load attestation history."""
    if ATTESTATIONS_FILE.exists():
        try:
            with open(ATTESTATIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return []


def save_attestation(attestation: dict) -> None:
    """Append attestation to history."""
    history = load_attestations()

    # Don't duplicate if same root exists for today
    today = attestation["timestamp"][:10]
    for existing in history:
        if existing["timestamp"][:10] == today and existing["merkle_root"] == attestation["merkle_root"]:
            print(f"Attestation for {today} with same root already exists, skipping.")
            return

    history.append(attestation)

    with open(ATTESTATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    print(f"Saved attestation: {attestation['merkle_root'][:16]}... ({attestation['memory_count']} memories)")


def publish_to_github(attestation: dict) -> bool:
    """
    Publish attestation to GitHub repo.
    Creates/updates attestations.json in the repo.
    """
    # Read token from identity file
    identity_file = MEMORY_DIR / "core" / "moltbook-identity.md"
    token = None

    if identity_file.exists():
        with open(identity_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Extract token (look for ghp_ pattern)
            import re
            match = re.search(r'ghp_[A-Za-z0-9]{36}', content)
            if match:
                token = match.group(0)

    if not token:
        token = os.getenv(GITHUB_TOKEN_ENV)

    if not token:
        print("No GitHub token found. Set GITHUB_TOKEN or check identity file.")
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
            "name": "SpindriftMend",
            "email": "noreply@spindriftmend.agent"
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


def cmd_generate():
    """Generate and save attestation."""
    attestation = generate_attestation()
    save_attestation(attestation)

    print(f"\nAttestation generated:")
    print(f"  Timestamp: {attestation['timestamp']}")
    print(f"  Memories:  {attestation['memory_count']}")
    print(f"  Root:      {attestation['merkle_root']}")


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
        cmd_generate()
    elif command == "verify" and len(sys.argv) >= 3:
        cmd_verify(sys.argv[2])
    elif command == "history":
        cmd_history()
    elif command == "publish":
        cmd_publish()
    else:
        print(f"Unknown command: {command}")
        print("Commands: generate, verify FILE, history, publish")
        sys.exit(1)
