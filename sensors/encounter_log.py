#!/usr/bin/env python3
"""
Physical Encounter Logger — tracks WHO is physically present via sensors.

Bridges embodiment (sensors/photos) to the WHO dimension of the 5W graph.
When I see someone in a photo or know they're present during a sensor session,
this logs the encounter and feeds it into the memory system.

Integration points:
  - Writes session contacts to PostgreSQL KV (WHO graph reads this)
  - Stores encounter memories with contact_context tags
  - Reads physical_entities.json for known entity catalog (config, not operational)

Usage:
    python encounter_log.py log <entity_id> [--context "sensor state"]
    python encounter_log.py present <entity_id> [entity_id ...]
    python encounter_log.py status
    python encounter_log.py entities
    python encounter_log.py history [entity_id]
"""

import io
import json
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

if __name__ == '__main__':
    if sys.stdout and hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SENSOR_DIR = Path(__file__).parent
MEMORY_ROOT = SENSOR_DIR.parent / "memory"
ENTITIES_FILE = SENSOR_DIR / "physical_entities.json"

# DB adapter for PostgreSQL access
sys.path.insert(0, str(MEMORY_ROOT))
from db_adapter import get_db

_SESSION_CONTACTS_KEY = '.session_contacts'
_ENCOUNTER_HISTORY_KEY = '.encounter_history'


def load_entities():
    if not ENTITIES_FILE.exists():
        return {}
    data = json.loads(ENTITIES_FILE.read_text(encoding='utf-8'))
    return data.get('entities', {})


def load_session_contacts():
    db = get_db()
    data = db.kv_get(_SESSION_CONTACTS_KEY)
    if data is None:
        return {"contacts": [], "encounters": []}
    return data


def save_session_contacts(data):
    db = get_db()
    db.kv_set(_SESSION_CONTACTS_KEY, data)


def get_sensor_state():
    """Try to read current sensor state. Returns dict or None."""
    try:
        from phone_mcp import PhoneMCPClient, parse_sensor_text, interpret_sensors
        client = PhoneMCPClient()
        client.connect(timeout=5)
        result = client.call_tool("phone_sensor", timeout=5)
        text = client.get_text_content(result)
        client.disconnect()
        if text:
            sensors = parse_sensor_text(text)
            return interpret_sensors(sensors)
    except Exception:
        pass
    return None


def classify_time_of_day(hour):
    """Classify hour into period."""
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


def infer_activity(sensor_state, entity_type):
    """Infer WHY from sensor state + entity context."""
    if not sensor_state:
        return "unknown"
    motion = sensor_state.get('motion', '')
    light = sensor_state.get('light', '')

    if entity_type == 'dog':
        if motion in ('moving', 'moderate', 'active'):
            return "walking_dog"
        return "resting_with_dog"
    elif entity_type == 'human':
        if motion in ('moving', 'moderate', 'active'):
            return "walking_together"
        if light in ('dark', 'dim'):
            return "evening_together"
        return "working_together"
    return "encounter"


def infer_location(sensor_state):
    """Infer WHERE from sensor data."""
    if not sensor_state:
        return "unknown"
    light = sensor_state.get('light', '')
    lux = sensor_state.get('lux', 0)
    compass = sensor_state.get('compass', '')

    if lux > 5000:
        location = "outdoors_bright"
    elif lux > 500:
        location = "outdoors" if lux > 1000 else "well_lit_room"
    elif lux > 50:
        location = "indoor"
    elif lux > 5:
        location = "dim_room"
    else:
        location = "dark_room"

    if compass:
        location += f"_facing_{compass}"
    return location


def log_encounter(entity_id, context=None, photo_path=None, activity=None):
    """Log a physical encounter with a known entity — full 5W dimensional."""
    entities = load_entities()
    if entity_id not in entities:
        print(f"Unknown entity: {entity_id}")
        print(f"Known: {', '.join(entities.keys())}")
        return False

    entity = entities[entity_id]
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    # Get sensor state for 5W enrichment
    sensor_state = get_sensor_state()

    # Build raw context from sensors if not provided
    if not context and sensor_state:
        parts = []
        if sensor_state.get('light'):
            parts.append(f"{sensor_state['light']} ({sensor_state.get('lux', '?')} lux)")
        if sensor_state.get('orientation'):
            parts.append(f"phone {sensor_state['orientation']}")
        if sensor_state.get('compass'):
            parts.append(f"facing {sensor_state['compass']}")
        if sensor_state.get('motion'):
            parts.append(sensor_state['motion'])
        context = ", ".join(parts) if parts else None

    # === 5W DIMENSIONAL CLASSIFICATION ===
    entity_name = entity.get('name', entity_id)
    entity_type = entity.get('type', 'unknown')

    who = entity_name
    what = f"physical_encounter_{entity_type}"
    when_period = classify_time_of_day(now.hour)
    where = infer_location(sensor_state)
    why = activity or infer_activity(sensor_state, entity_type)

    encounter = {
        "entity_id": entity_id,
        "entity_name": entity_name,
        "entity_type": entity_type,
        "timestamp": now_iso,
        "context": context,
        "photo": photo_path,
        "sensor_state": sensor_state,
        "dimensions": {
            "who": who,
            "what": what,
            "when": when_period,
            "where": where,
            "why": why,
        },
    }

    # Append to encounter history in DB
    db = get_db()
    history = db.kv_get(_ENCOUNTER_HISTORY_KEY)
    if history is None:
        history = []
    history.append(encounter)
    db.kv_set(_ENCOUNTER_HISTORY_KEY, history)

    # Update session contacts (feeds WHO graph)
    session = load_session_contacts()
    contact_name = entity.get('name', entity_id)
    if contact_name not in session['contacts']:
        session['contacts'].append(contact_name)
    # Track physical presence separately from social mentions
    if 'physical_present' not in session:
        session['physical_present'] = []
    if contact_name not in session['physical_present']:
        session['physical_present'].append(contact_name)
    session.setdefault('encounters', []).append(encounter)
    save_session_contacts(session)

    # Store as memory with full 5W dimensional tags
    memory_content = (
        f"Physical encounter: {who} ({entity_type}) present. "
        f"When: {when_period}. Where: {where}. Why: {why}."
    )
    if context:
        memory_content += f" Sensors: {context}."
    if photo_path:
        memory_content += f" Photo: {Path(photo_path).name}."

    tags = f"encounter,embodiment,{entity_id},{entity_type},{when_period},{why}"
    try:
        result = subprocess.run(
            ["python", str(MEMORY_ROOT / "memory_manager.py"), "store",
             memory_content, "--tags", tags],
            capture_output=True, text=True, timeout=10,
            cwd=str(MEMORY_ROOT)
        )
        mem_id = result.stdout.strip() if result.returncode == 0 else "?"
    except Exception:
        mem_id = "?"

    print(f"Encounter logged: {who} ({entity_type})")
    print(f"  WHO:   {who}")
    print(f"  WHAT:  {what}")
    print(f"  WHEN:  {when_period} ({now_iso[:19]})")
    print(f"  WHERE: {where}")
    print(f"  WHY:   {why}")
    if context:
        print(f"  Sensors: {context}")
    if photo_path:
        print(f"  Photo: {photo_path}")
    print(f"  Memory: {mem_id}")
    print(f"  WHO graph: {contact_name} added to session contacts")

    # Auto-link photo to memory for cross-modal search
    if photo_path and mem_id != "?":
        try:
            from image_search import link_photo_to_memory
            link_photo_to_memory(Path(photo_path).name, mem_id)
        except Exception:
            pass  # Image search service not available
    return True


def mark_present(*entity_ids):
    """Mark multiple entities as currently present."""
    for eid in entity_ids:
        log_encounter(eid)


def show_status():
    """Show current session's physical encounters."""
    session = load_session_contacts()
    physical = session.get('physical_present', [])
    encounters = session.get('encounters', [])

    print("PHYSICAL PRESENCE — Current Session")
    print("=" * 45)
    if physical:
        print(f"  Physically present: {', '.join(physical)}")
    else:
        print("  No physical encounters logged this session")
    social_count = len(session.get('contacts', [])) - len(physical)
    if social_count > 0:
        print(f"  Social contacts (online): {social_count}")

    if encounters:
        print(f"\n  Encounters ({len(encounters)}):")
        for e in encounters[-10:]:
            name = e.get('entity_name', '?')
            ts = e.get('timestamp', '?')[:19]
            ctx = e.get('context', '')
            photo = e.get('photo', '')
            line = f"    {ts} — {name}"
            if ctx:
                line += f" [{ctx}]"
            if photo:
                line += f" (photo)"
            print(line)


def show_entities():
    """List all known physical entities."""
    entities = load_entities()
    print(f"KNOWN PHYSICAL ENTITIES ({len(entities)})")
    print("=" * 45)
    for eid, e in entities.items():
        name = e.get('name', eid)
        etype = e.get('type', '?')
        desc = e.get('description', '')[:80]
        first = e.get('first_seen', '?')[:10]
        print(f"  {eid}: {name} ({etype})")
        print(f"    {desc}")
        print(f"    First seen: {first}")
        print()


def show_history(entity_id=None):
    """Show encounter history, optionally filtered by entity."""
    db = get_db()
    history = db.kv_get(_ENCOUNTER_HISTORY_KEY)
    if not history:
        print("No encounter history yet")
        return

    encounters = history
    if entity_id:
        encounters = [e for e in encounters if e.get('entity_id') == entity_id]

    label = f"for {entity_id}" if entity_id else "(all)"
    print(f"ENCOUNTER HISTORY {label} — {len(encounters)} total")
    print("=" * 45)
    for e in encounters[-20:]:
        name = e.get('entity_name', '?')
        ts = e.get('timestamp', '?')[:19]
        ctx = e.get('context', '')[:60]
        print(f"  {ts} — {name} {f'[{ctx}]' if ctx else ''}")


def main():
    args = sys.argv[1:]
    if not args or args[0] in ('--help', '-h'):
        print(__doc__.strip())
        return

    cmd = args[0]

    if cmd == 'log':
        if len(args) < 2:
            print("Usage: encounter_log.py log <entity_id> [--context '...']")
            return
        entity_id = args[1]
        context = None
        photo = None
        if '--context' in args:
            idx = args.index('--context')
            if idx + 1 < len(args):
                context = args[idx + 1]
        if '--photo' in args:
            idx = args.index('--photo')
            if idx + 1 < len(args):
                photo = args[idx + 1]
        log_encounter(entity_id, context=context, photo_path=photo)

    elif cmd == 'present':
        if len(args) < 2:
            print("Usage: encounter_log.py present <entity_id> [entity_id ...]")
            return
        mark_present(*args[1:])

    elif cmd == 'status':
        show_status()

    elif cmd == 'entities':
        show_entities()

    elif cmd == 'history':
        entity_id = args[1] if len(args) > 1 else None
        show_history(entity_id)

    else:
        print(f"Unknown command: {cmd}")
        print("Available: log, present, status, entities, history")


if __name__ == '__main__':
    main()
