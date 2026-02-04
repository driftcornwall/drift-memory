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
NOSTR_HISTORY_FILE = Path(__file__).parent / "nostr_attestations.json"

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
    """Load Nostr attestation history."""
    if NOSTR_HISTORY_FILE.exists():
        try:
            with open(NOSTR_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def save_history_entry(entry: dict) -> None:
    """Append an entry to Nostr attestation history."""
    history = load_history()
    history.append(entry)
    with open(NOSTR_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


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
    elif command == "verify" and len(sys.argv) >= 3:
        cmd_verify(sys.argv[2])
    elif command == "history":
        cmd_history()
    elif command == "identity":
        cmd_identity()
    else:
        print(f"Unknown command: {command}")
        print("Commands: publish, verify EVENT_ID, history, identity")
        sys.exit(1)
