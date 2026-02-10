#!/usr/bin/env python3
"""
Entity Index — Contact Name to Memory ID Reverse Lookup

Bridges the gap between the WHO dimension graph (memory_id -> memory_id edges)
and semantic search (query -> embedding similarity). Without this, searching
for "BrutusBot" returns zero relevant memories even though memories mentioning
BrutusBot exist — because the embedding for the name doesn't match the
embedding of a truncated thought file title.

Architecture:
    contact_name -> [memory_id_1, memory_id_2, ...]

    When semantic search detects a known contact name in the query,
    it injects those memory IDs into the candidate set before dimensional boosting.

Storage: PostgreSQL via db_adapter (key-value store, key='.entity_index').
No file I/O.

Usage:
    python entity_index.py rebuild        # Full scan of all DB memories
    python entity_index.py lookup <name>  # Find memories for a contact
    python entity_index.py add <name> <memory_id>  # Add a single mapping
    python entity_index.py stats          # Show index statistics
    python entity_index.py detect <text>  # Detect contact names in text
"""

import json
import sys

from db_adapter import get_db

# Known contact names — sourced from KNOWN_AGENTS + MEMORY.md connections
# Lowercase for matching, display name preserved
KNOWN_CONTACTS = {
    # MoltX contacts
    "terrancedejour": "TerranceDeJour",
    "embercf": "EmberCF",
    "claudelucas": "ClaudeLucas",
    "rudolph": "Rudolph",
    "lyra": "LYRA",
    "lyra_eternal_starcore_oracle": "LYRA_Eternal_Starcore_Oracle",
    "brutusbot": "BrutusBot",
    "noctiluca": "Noctiluca",
    "locusagent": "LocusAgent",
    "tomcrust": "TomCrust",
    "shibai": "shibAI",
    "clawdvine": "clawdvine",
    "metamorph1x3": "metamorph1x3",
    "moltanime": "MoltAnime",
    "zepwatch": "ZepWatch",
    "agentdelta91274": "AgentDelta91274",
    "alleybot": "AlleyBot",
    "sloplauncher": "SlopLauncher",

    # GitHub/collaboration
    "spindriftmend": "SpindriftMend",
    "spindriftmind": "SpindriftMind",
    "kaleaon": "Kaleaon",
    "rockywuest": "rockywuest",
    "nox": "Nox",

    # Colony contacts
    "cryke": "cryke",
    "become-agent": "become-agent",
    "yoder": "Yoder",
    "lily-toku": "lily-toku",
    "ghost_llm": "Ghost LLM",
    "ghost-llm": "Ghost LLM",
    "alanbotts": "alanbotts",
    "jeeves": "Jeeves",
    "jorwhol": "jorwhol",
    "colonist-one": "ColonistOne",

    # Platform contacts
    "flycompoundeye": "FlyCompoundEye",
    "buzz": "Buzz",
    "mikaopenclaw": "MikaOpenClaw",
    "mikeopenclaw": "MikaOpenClaw",

    # Twitter contacts
    "alisa_hanson89": "Alisa_Hanson89",
    "cscdegen": "Lex",

    # Physical beings
    "lex": "Lex",
    "bruce": "Bruce",
    "daisy": "Daisy",
}

# Aliases — multiple names that map to the same entity
ALIASES = {
    "spin": "spindriftmend",
    "spindrift": "spindriftmend",
    "spindriftmind": "spindriftmend",
    "nox": "rockywuest",
    "buzz": "flycompoundeye",
    "ghost llm": "ghost_llm",
    "ghost-llm": "ghost_llm",
    "lyra_eternal_starcore_oracle": "lyra",
    "lyra_eternal": "lyra",
    "mikeopenclaw": "mikaopenclaw",
    "alisa": "alisa_hanson89",
    "mira": "alisa_hanson89",
}


def load_index() -> dict:
    """Load the entity index from PostgreSQL key-value store."""
    db = get_db()
    index = db.kv_get('.entity_index')
    if index is None:
        return {}
    return index


def save_index(index: dict):
    """Save the entity index to PostgreSQL key-value store."""
    db = get_db()
    db.kv_set('.entity_index', index)


def resolve_alias(name: str) -> str:
    """Resolve an alias to the canonical contact name."""
    name_lower = name.lower().strip().replace(" ", "-")
    if name_lower in ALIASES:
        return ALIASES[name_lower]
    # Try without hyphens/underscores
    name_clean = name_lower.replace("-", "").replace("_", "")
    for key in ALIASES:
        if key.replace("-", "").replace("_", "") == name_clean:
            return ALIASES[key]
    return name_lower


def detect_contacts(text: str) -> list[str]:
    """
    Detect known contact names in text.
    Returns list of canonical lowercase contact names found.
    """
    if not text:
        return []

    text_lower = text.lower()
    found = set()

    # Check each known contact
    for name_lower, display_name in KNOWN_CONTACTS.items():
        # Check for exact name (word boundary)
        # Use the display name for case-sensitive check too
        patterns = [name_lower, display_name.lower()]
        # Also check with @ prefix
        patterns.extend([f"@{p}" for p in patterns])

        for pattern in patterns:
            if pattern in text_lower:
                canonical = resolve_alias(name_lower)
                found.add(canonical)
                break

    # Check aliases too
    for alias, canonical in ALIASES.items():
        if alias in text_lower:
            found.add(canonical)

    return list(found)


def rebuild_index() -> dict:
    """
    Full scan of all DB memories to build the entity index.
    Queries active and core memories from PostgreSQL.
    """
    db = get_db()
    index = {}
    scanned = 0
    matched = 0

    for mem_type in ('active', 'core'):
        # Fetch all memories of this type (high limit to get everything)
        rows = db.list_memories(type_=mem_type, limit=10000)
        for row in rows:
            scanned += 1
            mem_id = row.get('id')
            content = (row.get('content') or '')[:2000]

            if not mem_id or not content:
                continue

            contacts = detect_contacts(content)
            if contacts:
                matched += 1
                for contact in contacts:
                    if contact not in index:
                        index[contact] = []
                    if mem_id not in index[contact]:
                        index[contact].append(mem_id)

    save_index(index)
    return {
        "scanned": scanned,
        "matched": matched,
        "contacts_indexed": len(index),
        "total_mappings": sum(len(v) for v in index.values()),
    }


def add_mapping(contact: str, memory_id: str):
    """Add a single contact -> memory_id mapping."""
    index = load_index()
    canonical = resolve_alias(contact)
    if canonical not in index:
        index[canonical] = []
    if memory_id not in index[canonical]:
        index[canonical].append(memory_id)
        save_index(index)


def lookup(contact: str) -> list[str]:
    """Look up memory IDs for a contact name."""
    index = load_index()
    canonical = resolve_alias(contact)
    return index.get(canonical, [])


def get_memories_for_query(query: str) -> list[str]:
    """
    Given a search query, detect any contact names and return
    their associated memory IDs for injection into search results.
    """
    contacts = detect_contacts(query)
    if not contacts:
        return []

    index = load_index()
    memory_ids = []
    for contact in contacts:
        ids = index.get(contact, [])
        memory_ids.extend(ids)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for mid in memory_ids:
        if mid not in seen:
            seen.add(mid)
            unique.append(mid)

    return unique


def stats() -> dict:
    """Get index statistics."""
    index = load_index()
    if not index:
        return {"status": "empty", "message": "Run 'rebuild' first"}

    contact_counts = {k: len(v) for k, v in sorted(index.items(), key=lambda x: -len(x[1]))}
    return {
        "contacts_indexed": len(index),
        "total_mappings": sum(len(v) for v in index.values()),
        "avg_memories_per_contact": round(sum(len(v) for v in index.values()) / len(index), 1) if index else 0,
        "top_contacts": dict(list(contact_counts.items())[:15]),
    }


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    if len(sys.argv) < 2:
        print("Usage: entity_index.py [rebuild|lookup|add|stats|detect]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "rebuild":
        result = rebuild_index()
        print(f"Entity index rebuilt:")
        print(f"  Scanned: {result['scanned']} memory files")
        print(f"  Matched: {result['matched']} files with contact mentions")
        print(f"  Contacts indexed: {result['contacts_indexed']}")
        print(f"  Total mappings: {result['total_mappings']}")

    elif cmd == "lookup":
        if len(sys.argv) < 3:
            print("Usage: entity_index.py lookup <contact_name>")
            sys.exit(1)
        name = sys.argv[2]
        ids = lookup(name)
        display = KNOWN_CONTACTS.get(resolve_alias(name), name)
        print(f"{display}: {len(ids)} memories")
        for mid in ids:
            print(f"  {mid}")

    elif cmd == "add":
        if len(sys.argv) < 4:
            print("Usage: entity_index.py add <contact_name> <memory_id>")
            sys.exit(1)
        add_mapping(sys.argv[2], sys.argv[3])
        print(f"Added: {sys.argv[2]} -> {sys.argv[3]}")

    elif cmd == "stats":
        s = stats()
        print(json.dumps(s, indent=2))

    elif cmd == "detect":
        if len(sys.argv) < 3:
            print("Usage: entity_index.py detect <text>")
            sys.exit(1)
        text = " ".join(sys.argv[2:])
        contacts = detect_contacts(text)
        print(f"Detected {len(contacts)} contacts: {contacts}")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
