#!/usr/bin/env python3
"""
Nostr Attestations for drift-memory
Publishes merkle roots to the Nostr network for decentralized, censorship-resistant verification.

Usage:
    python nostr_attestation.py publish          # Generate merkle root and publish to Nostr
    python nostr_attestation.py verify EVENT_ID  # Verify a published attestation
    python nostr_attestation.py history          # Show Nostr attestation history
    python nostr_attestation.py identity         # Show Drift's Nostr public key

Why Nostr over GitHub alone:
    - Decentralized: No single point of failure or censorship
    - Cryptographically signed: Schnorr signatures prove authorship
    - Timestamped: Relay-independent proof of existence
    - Interoperable: Any Nostr client can verify the attestation
    - Immutable: Events cannot be modified after signing
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nostr_sdk import (
    Client,
    EventBuilder,
    EventId,
    Filter,
    Keys,
    Kind,
    NostrSigner,
    RelayUrl,
    Tag,
    Timestamp,
)

from merkle_attestation import generate_attestation

CREDENTIALS_FILE = Path.home() / ".config" / "nostr" / "drift-credentials.json"
MEMORY_DIR = Path(__file__).parent

RELAYS = [
    "wss://relay.damus.io",
    "wss://nos.lol",
    "wss://relay.nostr.band",
]


def load_or_create_keys() -> Keys:
    """Load existing Nostr keys or generate new ones."""
    if CREDENTIALS_FILE.exists():
        try:
            with open(CREDENTIALS_FILE, "r", encoding="utf-8") as f:
                creds = json.load(f)
            return Keys.parse(creds["secret_key"])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load keys from {CREDENTIALS_FILE}: {e}")

    keys = Keys.generate()
    creds = {
        "public_key_hex": keys.public_key().to_hex(),
        "public_key_bech32": keys.public_key().to_bech32(),
        "secret_key_bech32": keys.secret_key().to_bech32(),
        "secret_key": keys.secret_key().to_bech32(),
        "created": datetime.now(timezone.utc).isoformat(),
        "agent": "DriftCornwall",
    }

    CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CREDENTIALS_FILE, "w", encoding="utf-8") as f:
        json.dump(creds, f, indent=2)

    print(f"Generated new Nostr identity:")
    print(f"  Public key: {creds['public_key_bech32']}")
    print(f"  Saved to: {CREDENTIALS_FILE}")
    return keys


def load_history() -> list[dict]:
    """Load Nostr attestation history from DB."""
    try:
        from db_adapter import get_db
        data = get_db().kv_get('.nostr_history')
        if data:
            import json as _json
            return _json.loads(data) if isinstance(data, str) else data
    except Exception:
        pass
    return []


def save_history_entry(entry: dict) -> None:
    """Append an entry to Nostr attestation history in DB."""
    history = load_history()
    history.append(entry)
    try:
        from db_adapter import get_db
        get_db().kv_set('.nostr_history', history)
    except Exception:
        pass


def _load_attestation_data() -> tuple[dict, dict, dict]:
    """Load all three attestation datasets from PostgreSQL.

    Returns (merkle_data, fingerprint_data, taste_data).
    Each is a dict (possibly empty if not found or on error).
    """
    from db_adapter import get_db

    db = get_db()

    # Merkle: query latest from attestations table
    merkle_data = {}
    try:
        from merkle_attestation import load_latest_attestation
        merkle_data = load_latest_attestation() or {}
    except Exception:
        pass

    # Fingerprint: from KV store (saved by stop.py as 'cognitive_attestation',
    # also saved by cognitive_fingerprint.py attest as '.cognitive_attestation_latest')
    fingerprint_data = {}
    try:
        fingerprint_data = db.kv_get('cognitive_attestation') or {}
        if not fingerprint_data:
            fingerprint_data = db.kv_get('.cognitive_attestation_latest') or {}
    except Exception:
        pass

    # Taste: from KV store (saved by stop.py as 'taste_attestation')
    taste_data = {}
    try:
        taste_data = db.kv_get('taste_attestation') or {}
    except Exception:
        pass

    return merkle_data, fingerprint_data, taste_data


async def publish_merkle_root() -> dict | None:
    """Generate a merkle root from current memories and publish to Nostr."""
    attestation = generate_attestation(chain=True)
    merkle_root = attestation["merkle_root"]
    memory_count = attestation["memory_count"]
    timestamp = attestation["timestamp"]
    previous_root = attestation.get("previous_root", "")
    chain_depth = attestation.get("chain_depth", 0)

    chain_line = f"previous_root: {previous_root}\nchain_depth: {chain_depth}\n" if previous_root else ""

    content = (
        f"drift-memory attestation v2.0\n"
        f"merkle_root: {merkle_root}\n"
        f"memories: {memory_count}\n"
        f"timestamp: {timestamp}\n"
        f"{chain_line}"
        f"agent: DriftCornwall\n"
        f"repo: https://github.com/driftcornwall/drift-memory\n\n"
        f"This merkle root covers {memory_count} memories (chain depth: {chain_depth}). "
        f"Any modification to any memory file changes this root. "
        f"Verifiable proof of memory state and identity continuity."
    )

    keys = load_or_create_keys()
    signer = NostrSigner.keys(keys)
    client = Client(signer)

    for relay in RELAYS:
        try:
            await client.add_relay(RelayUrl.parse(relay))
        except Exception as e:
            print(f"  Warning: Could not add relay {relay}: {e}")

    print("Connecting to relays...")
    await client.connect()

    builder = EventBuilder(Kind(1), content).tags([
        Tag.hashtag("drift-memory"),
        Tag.hashtag("merkle-attestation"),
        Tag.hashtag("agent-memory"),
        Tag.hashtag("nostr-attestation"),
    ])

    print("Publishing merkle root to Nostr...")
    output = await client.send_event_builder(builder)

    event_id_hex = output.id.to_hex()
    event_id_bech32 = output.id.to_bech32()
    success_relays = [str(r) for r in output.success]
    failed_relays = {str(k): str(v) for k, v in output.failed.items()} if output.failed else {}

    await client.disconnect()

    result = {
        "event_id_hex": event_id_hex,
        "event_id_bech32": event_id_bech32,
        "merkle_root": merkle_root,
        "memory_count": memory_count,
        "timestamp": timestamp,
        "public_key": keys.public_key().to_bech32(),
        "success_relays": success_relays,
        "failed_relays": failed_relays,
        "nostr_link": f"https://njump.me/{event_id_bech32}",
    }

    save_history_entry(result)

    print(f"\nAttestation published to Nostr!")
    print(f"  Merkle root:  {merkle_root[:32]}...")
    print(f"  Memories:     {memory_count}")
    print(f"  Event ID:     {event_id_bech32}")
    print(f"  Link:         https://njump.me/{event_id_bech32}")
    print(f"  Relays OK:    {len(success_relays)}/{len(RELAYS)}")
    if failed_relays:
        print(f"  Failed:       {failed_relays}")

    return result


async def verify_event(event_id_str: str) -> bool:
    """Retrieve and verify a Nostr attestation event."""
    keys = load_or_create_keys()
    signer = NostrSigner.keys(keys)
    client = Client(signer)

    for relay in RELAYS:
        try:
            await client.add_relay(RelayUrl.parse(relay))
        except Exception:
            pass

    await client.connect()

    try:
        event_id = EventId.parse(event_id_str)
    except Exception:
        print(f"Invalid event ID: {event_id_str}")
        await client.disconnect()
        return False

    print(f"Fetching event {event_id_str[:20]}... from relays...")

    f = Filter().id(event_id)
    events = await client.fetch_events(f, timedelta(seconds=15))

    await client.disconnect()

    event_list = events.to_vec()
    if not event_list:
        print("Event not found on any relay.")
        return False

    event = event_list[0]
    content = event.content()

    print(f"\nEvent found!")
    print(f"  Author:    {event.author().to_bech32()}")
    print(f"  Created:   {event.created_at().to_human_datetime()}")
    print(f"  Content:\n")
    for line in content.split("\n"):
        print(f"    {line}")

    my_pubkey = keys.public_key().to_bech32()
    if event.author().to_bech32() == my_pubkey:
        print(f"\n  Verified: Signed by Drift's key ({my_pubkey[:20]}...)")
    else:
        print(f"\n  Warning: Signed by different key: {event.author().to_bech32()}")

    return True


def needs_dossier_publish() -> bool:
    """
    Check if any attestation has changed since the last Nostr dossier publish.
    Returns True if we should publish a new combined dossier event.
    """
    history = load_history()
    if not history:
        return True

    # Find last dossier publish (has fingerprint_hash key)
    last_dossier = None
    for entry in reversed(history):
        if entry.get("fingerprint_hash"):
            last_dossier = entry
            break

    if not last_dossier:
        # Never published a dossier, only legacy merkle-only events
        return True

    # Compare current attestation hashes (from DB) to last published
    merkle_data, fingerprint_data, taste_data = _load_attestation_data()

    for data, key in [
        (merkle_data, "merkle_root"),
        (fingerprint_data, "fingerprint_hash"),
        (taste_data, "taste_hash"),
    ]:
        if not data:
            continue
        current_hash = data.get(key) or data.get("merkle_root", "")
        if current_hash != last_dossier.get(key, ""):
            return True

    return False


async def publish_dossier() -> dict | None:
    """
    Publish combined identity dossier to Nostr.

    Bundles all four dossier layers into a single signed event:
    - Layer 1: Merkle root (memory integrity)
    - Layer 2: Cognitive fingerprint (identity topology)
    - Layer 3: Taste hash (rejection patterns)
    - Layer 4: Interop schema (referenced, not hashed)

    One event, one signature, complete identity.
    """
    # Load all attestation data from DB
    merkle_data, fingerprint_data, taste_data = _load_attestation_data()

    if not merkle_data:
        print("No merkle attestation found. Run session-end first.")
        return None

    # Build structured content
    merkle_root = merkle_data.get("merkle_root", "?")
    memory_count = merkle_data.get("memory_count", 0)
    chain_depth = merkle_data.get("chain_depth", 0)

    fp_hash = fingerprint_data.get("fingerprint_hash", "?")
    nodes = fingerprint_data.get("graph_stats", {}).get("node_count", 0)
    edges = fingerprint_data.get("graph_stats", {}).get("edge_count", 0)
    domains = fingerprint_data.get("cognitive_domain_weights", {})
    drift_score = fingerprint_data.get("drift_score", "N/A")
    drift_interp = fingerprint_data.get("drift_interpretation", "")

    taste_hash = taste_data.get("taste_hash", "?")
    rejection_count = taste_data.get("rejection_count", 0)

    # Domain weights as readable string
    domain_parts = []
    for d in sorted(domains.keys(), key=lambda k: domains[k], reverse=True):
        domain_parts.append(f"{d}={domains[d]}%")
    domain_str = ", ".join(domain_parts) if domain_parts else "N/A"

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    content = (
        f"drift-memory dossier v1.0\n"
        f"agent: DriftCornwall\n"
        f"timestamp: {timestamp}\n"
        f"repo: https://github.com/driftcornwall/drift-memory\n"
        f"\n"
        f"=== Layer 1: Memory Integrity (Merkle) ===\n"
        f"merkle_root: {merkle_root}\n"
        f"memories: {memory_count}\n"
        f"chain_depth: {chain_depth}\n"
        f"\n"
        f"=== Layer 2: Cognitive Identity (Fingerprint) ===\n"
        f"fingerprint_hash: {fp_hash}\n"
        f"graph: {nodes} nodes, {edges} edges\n"
        f"domains: {domain_str}\n"
        f"drift: {drift_score} ({drift_interp})\n"
        f"\n"
        f"=== Layer 3: Taste (Rejection Patterns) ===\n"
        f"taste_hash: {taste_hash}\n"
        f"rejections: {rejection_count}\n"
        f"\n"
        f"Three layers of unforgeable identity in one signed event. "
        f"Merkle proves memory integrity. Fingerprint proves cognitive topology. "
        f"Taste proves what I refuse. Combined: prohibitively expensive to fake."
    )

    keys = load_or_create_keys()
    signer = NostrSigner.keys(keys)
    client = Client(signer)

    for relay in RELAYS:
        try:
            await client.add_relay(RelayUrl.parse(relay))
        except Exception as e:
            print(f"  Warning: Could not add relay {relay}: {e}")

    print("Connecting to relays...")
    await client.connect()

    builder = EventBuilder(Kind(1), content).tags([
        Tag.hashtag("drift-memory"),
        Tag.hashtag("agent-dossier"),
        Tag.hashtag("merkle-attestation"),
        Tag.hashtag("cognitive-fingerprint"),
        Tag.hashtag("identity"),
    ])

    print("Publishing dossier to Nostr...")
    output = await client.send_event_builder(builder)

    event_id_hex = output.id.to_hex()
    event_id_bech32 = output.id.to_bech32()
    success_relays = [str(r) for r in output.success]
    failed_relays = {str(k): str(v) for k, v in output.failed.items()} if output.failed else {}

    await client.disconnect()

    result = {
        "type": "dossier",
        "event_id_hex": event_id_hex,
        "event_id_bech32": event_id_bech32,
        "merkle_root": merkle_root,
        "fingerprint_hash": fp_hash,
        "taste_hash": taste_hash,
        "memory_count": memory_count,
        "chain_depth": chain_depth,
        "rejection_count": rejection_count,
        "timestamp": timestamp,
        "public_key": keys.public_key().to_bech32(),
        "success_relays": success_relays,
        "failed_relays": failed_relays,
        "nostr_link": f"https://njump.me/{event_id_bech32}",
    }

    save_history_entry(result)

    print(f"\nDossier published to Nostr!")
    print(f"  Merkle root:   {merkle_root[:32]}...")
    print(f"  Fingerprint:   {fp_hash[:32]}...")
    print(f"  Taste hash:    {taste_hash[:32]}...")
    print(f"  Memories:      {memory_count} (chain depth: {chain_depth})")
    print(f"  Rejections:    {rejection_count}")
    print(f"  Event ID:      {event_id_bech32}")
    print(f"  Link:          https://njump.me/{event_id_bech32}")
    print(f"  Relays OK:     {len(success_relays)}/{len(RELAYS)}")
    if failed_relays:
        print(f"  Failed:        {failed_relays}")

    return result


def cmd_publish():
    """Generate and publish merkle root to Nostr."""
    result = asyncio.run(publish_merkle_root())
    if result:
        print("\nDone. Attestation is now on the Nostr network.")


def cmd_verify(event_id: str):
    """Verify an attestation on Nostr."""
    asyncio.run(verify_event(event_id))


def cmd_history():
    """Show Nostr attestation history."""
    history = load_history()
    if not history:
        print("No Nostr attestations yet.")
        return

    print(f"Nostr Attestation History ({len(history)} entries):\n")
    for entry in history[-10:]:
        print(f"  {entry['timestamp'][:19]}  {entry['memory_count']:4} memories  {entry['merkle_root'][:16]}...")
        print(f"    Event: {entry['event_id_bech32'][:30]}...")
        print(f"    Link:  {entry['nostr_link']}")
        print()


def cmd_identity():
    """Show Drift's Nostr identity."""
    keys = load_or_create_keys()
    print(f"Drift's Nostr Identity:")
    print(f"  Public key (bech32): {keys.public_key().to_bech32()}")
    print(f"  Public key (hex):    {keys.public_key().to_hex()}")
    print(f"  Profile:             https://njump.me/{keys.public_key().to_bech32()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "publish":
        cmd_publish()
    elif command == "publish-dossier":
        result = asyncio.run(publish_dossier())
        if result:
            print("\nDone. Full identity dossier is now on Nostr.")
    elif command == "needs-publish":
        if needs_dossier_publish():
            print("true")
        else:
            print("false")
    elif command == "verify" and len(sys.argv) >= 3:
        cmd_verify(sys.argv[2])
    elif command == "history":
        cmd_history()
    elif command == "latest-link":
        history = load_history()
        for entry in reversed(history):
            link = entry.get("nostr_link", "")
            if link:
                print(link)
                break
        else:
            print("")
    elif command == "identity":
        cmd_identity()
    else:
        print(f"Unknown command: {command}")
        print("Commands: publish, publish-dossier, needs-publish, verify EVENT_ID, history, latest-link, identity")
        sys.exit(1)
