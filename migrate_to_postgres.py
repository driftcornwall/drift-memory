#!/usr/bin/env python3
"""
Drift Memory → PostgreSQL Migration Script

Migrates all file-based data into the shared agent-memory-db PostgreSQL instance.
Uses Spin's db.py DAL with schema='drift'.

Usage:
    python memory/migrate_to_postgres.py              # Full migration
    python memory/migrate_to_postgres.py --dry-run    # Count only, no writes
    python memory/migrate_to_postgres.py --only memories  # Migrate one table
    python memory/migrate_to_postgres.py --validate   # Validate after migration
"""

import sys
import os
import io
import json
import glob
import time
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone

# UTF-8 output for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add memorydatabase to path for db.py import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "memorydatabase"))
from database.db import MemoryDB, get_conn

# Add memory dir for parse_memory_file
sys.path.insert(0, str(Path(__file__).parent))
from memory_common import parse_memory_file, CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR, MEMORY_ROOT

db = MemoryDB(schema='drift')

# Paths
EMBEDDINGS_FILE = MEMORY_ROOT / "embeddings.json"
EDGES_FILE = MEMORY_ROOT / ".edges_v3.json"
REJECTION_FILE = MEMORY_ROOT / ".rejection_log.json"
LESSONS_FILE = MEMORY_ROOT / "lessons.json"
FINGERPRINT_FILE = MEMORY_ROOT / ".fingerprint_history.json"
VITALS_FILE = MEMORY_ROOT / ".vitals_log.json"
DECAY_FILE = MEMORY_ROOT / ".decay_history.json"
ATTESTATION_FILE = MEMORY_ROOT / "attestations.json"
TASTE_ATTESTATION_FILE = MEMORY_ROOT / "taste_attestation.json"
NOSTR_ATTESTATION_FILE = MEMORY_ROOT / "nostr_attestations.json"
CONTEXT_DIR = MEMORY_ROOT / "context"
SOCIAL_DIR = MEMORY_ROOT / "social"
IMAGE_EMBEDDINGS_FILE = MEMORY_ROOT.parent / "sensors" / "data" / "image_embeddings.json"
IMAGE_LINKS_FILE = MEMORY_ROOT.parent / "sensors" / "data" / "image_memory_links.json"
CALIBRATION_FILE = MEMORY_ROOT / ".calibration_history.json"
VOCAB_FILE = MEMORY_ROOT / "vocabulary_map.json"
THOUGHT_STATE_FILE = MEMORY_ROOT / ".thought_priming_state.json"
TELEGRAM_STATE_FILE = MEMORY_ROOT / ".telegram_state.json"
STB_FILE = MEMORY_ROOT / "short_term_buffer.json"


def ts(iso_str):
    """Parse ISO timestamp string, return None if missing/invalid."""
    if not iso_str:
        return None
    try:
        if isinstance(iso_str, datetime):
            return iso_str
        s = str(iso_str).strip()
        # Handle various formats
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def migrate_memories(dry_run=False):
    """Migrate all .md memory files to drift.memories table."""
    print("\n=== MEMORIES ===")
    count = 0
    errors = 0

    for type_name, dir_path in [('core', CORE_DIR), ('active', ACTIVE_DIR), ('archive', ARCHIVE_DIR)]:
        files = sorted(dir_path.glob("*.md"))
        print(f"  {type_name}: {len(files)} files")

        for filepath in files:
            try:
                meta, body = parse_memory_file(filepath)
                mem_id = meta.get('id', filepath.stem)

                if dry_run:
                    count += 1
                    continue

                # Build fields
                created = ts(meta.get('created')) or datetime.now(timezone.utc)
                last_recalled = ts(meta.get('last_recalled'))
                event_time = ts(meta.get('event_time'))

                # Extract arrays
                tags = meta.get('tags', [])
                if isinstance(tags, str):
                    tags = [tags]

                entities = meta.get('entities', {})
                if not isinstance(entities, dict):
                    entities = {}

                caused_by = meta.get('caused_by', [])
                if isinstance(caused_by, str):
                    caused_by = [caused_by]

                leads_to = meta.get('leads_to', [])
                if isinstance(leads_to, str):
                    leads_to = [leads_to]

                source = meta.get('source')
                if source and not isinstance(source, dict):
                    source = {'raw': str(source)}

                # Retrieval outcomes
                ro = meta.get('retrieval_outcomes', {})
                if not isinstance(ro, dict):
                    ro = {}

                # Context arrays
                topic_ctx = meta.get('topic_context', [])
                contact_ctx = meta.get('contact_context', [])
                platform_ctx = meta.get('platform_context', [])

                # Collect remaining metadata
                known_keys = {
                    'id', 'created', 'last_recalled', 'recall_count', 'sessions_since_recall',
                    'emotional_weight', 'tags', 'type', 'entities', 'caused_by', 'leads_to',
                    'event_time', 'source', 'retrieval_outcomes', 'retrieval_success_rate',
                    'topic_context', 'contact_context', 'platform_context', 'co_occurrences'
                }
                extra = {k: v for k, v in meta.items() if k not in known_keys}

                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO drift.memories
                            (id, type, content, created, last_recalled, recall_count,
                             sessions_since_recall, emotional_weight, tags, event_time,
                             entities, caused_by, leads_to, source, retrieval_outcomes,
                             retrieval_success_rate, topic_context, contact_context,
                             platform_context, extra_metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING
                        """, (
                            mem_id, type_name, body, created, last_recalled,
                            int(meta.get('recall_count', 0)),
                            int(meta.get('sessions_since_recall', 0)),
                            float(meta.get('emotional_weight', 0.5)),
                            tags, event_time,
                            json.dumps(entities), caused_by, leads_to,
                            json.dumps(source) if source else None,
                            json.dumps(ro),
                            meta.get('retrieval_success_rate'),
                            topic_ctx, contact_ctx, platform_ctx,
                            json.dumps(extra) if extra else '{}'
                        ))
                count += 1
            except Exception as e:
                errors += 1
                print(f"    ERROR {filepath.name}: {e}")

    print(f"  Migrated: {count}, Errors: {errors}")
    return count, errors


def migrate_co_occurrences(dry_run=False):
    """Migrate legacy in-file co-occurrence counts to drift.co_occurrences."""
    print("\n=== CO-OCCURRENCES (legacy) ===")
    count = 0
    batch = []

    for type_name, dir_path in [('core', CORE_DIR), ('active', ACTIVE_DIR), ('archive', ARCHIVE_DIR)]:
        for filepath in dir_path.glob("*.md"):
            try:
                meta, _ = parse_memory_file(filepath)
                mem_id = meta.get('id', filepath.stem)
                co = meta.get('co_occurrences', {})
                if not isinstance(co, dict):
                    continue
                for other_id, cnt in co.items():
                    batch.append((mem_id, other_id, float(cnt)))
                    count += 1
            except Exception:
                pass

    print(f"  Found: {count} pairs")
    if dry_run or count == 0:
        return count, 0

    errors = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for mem_id, other_id, cnt in batch:
                try:
                    cur.execute("""
                        INSERT INTO drift.co_occurrences (memory_id, other_id, count)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (memory_id, other_id) DO UPDATE SET count = EXCLUDED.count
                    """, (mem_id, other_id, cnt))
                except Exception as e:
                    errors += 1

    print(f"  Migrated: {count - errors}, Errors: {errors}")
    return count, errors


def migrate_edges_v3(dry_run=False):
    """Migrate .edges_v3.json to drift.edges_v3 + drift.edge_observations."""
    print("\n=== EDGES V3 ===")
    if not EDGES_FILE.exists():
        print("  File not found, skipping")
        return 0, 0

    with open(EDGES_FILE, 'r', encoding='utf-8') as f:
        edges = json.load(f)

    print(f"  Found: {len(edges)} edges")
    if dry_run:
        return len(edges), 0

    count = 0
    obs_count = 0
    errors = 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            for key, edge in edges.items():
                try:
                    parts = key.split('|')
                    if len(parts) != 2:
                        errors += 1
                        continue
                    id1, id2 = parts
                    # Canonicalize
                    if id1 > id2:
                        id1, id2 = id2, id1

                    cur.execute("""
                        INSERT INTO drift.edges_v3
                        (id1, id2, belief, first_formed, last_updated,
                         platform_context, activity_context, topic_context,
                         contact_context, thinking_about)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id1, id2) DO UPDATE SET
                            belief = EXCLUDED.belief,
                            last_updated = EXCLUDED.last_updated
                    """, (
                        id1, id2,
                        float(edge.get('belief', 0)),
                        ts(edge.get('first_formed')),
                        ts(edge.get('last_updated')),
                        json.dumps(edge.get('platform_context', {})),
                        json.dumps(edge.get('activity_context', {})),
                        json.dumps(edge.get('topic_context', {})),
                        edge.get('contact_context', []),
                        edge.get('thinking_about', []),
                    ))
                    count += 1

                    # Migrate observations
                    for obs in edge.get('observations', []):
                        try:
                            src = obs.get('source', {})
                            if isinstance(src, str):
                                src = {'type': src}

                            cur.execute("""
                                INSERT INTO drift.edge_observations
                                (id, edge_id1, edge_id2, observed_at, source_type,
                                 session_id, agent, platform, activity, weight, trust_tier)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (id) DO NOTHING
                            """, (
                                obs.get('id'),
                                id1, id2,
                                ts(obs.get('observed_at')) or datetime.now(timezone.utc),
                                src.get('type', 'unknown'),
                                src.get('session_id'),
                                src.get('agent'),
                                src.get('platform'),
                                src.get('activity'),
                                float(obs.get('weight', 1.0)),
                                obs.get('trust_tier', 'self'),
                            ))
                            obs_count += 1
                        except Exception as e:
                            errors += 1

                except Exception as e:
                    errors += 1
                    if count < 3:
                        print(f"    ERROR edge {key}: {e}")

    print(f"  Edges: {count}, Observations: {obs_count}, Errors: {errors}")
    return count, errors


def migrate_text_embeddings(dry_run=False):
    """Migrate embeddings.json to drift.text_embeddings (pgvector)."""
    print("\n=== TEXT EMBEDDINGS ===")
    if not EMBEDDINGS_FILE.exists():
        print("  File not found, skipping")
        return 0, 0

    with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Structure: {"memories": {"id": {"embedding": [...], "preview": "..."}}, "model": "...", "embedding_dim": N}
    memories_dict = data.get('memories', data)  # fallback if flat format
    global_model = data.get('model', 'Qwen3-Embedding-0.6B')
    emb_dim = data.get('embedding_dim', 2560)

    # Filter to entries that actually have embeddings
    valid = {k: v for k, v in memories_dict.items()
             if isinstance(v, dict) and v.get('embedding') and len(v['embedding']) > 0}

    print(f"  Found: {len(memories_dict)} entries, {len(valid)} with embeddings (dim={emb_dim})")
    if dry_run:
        return len(valid), 0

    count = 0
    errors = 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            for mem_id, entry in valid.items():
                try:
                    embedding = entry['embedding']
                    preview = entry.get('preview', '')

                    # pgvector expects string format [1.0, 2.0, ...]
                    vec_str = '[' + ','.join(str(x) for x in embedding) + ']'

                    cur.execute("""
                        INSERT INTO drift.text_embeddings (memory_id, embedding, preview, model)
                        VALUES (%s, %s::vector, %s, %s)
                        ON CONFLICT (memory_id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            preview = EXCLUDED.preview,
                            model = EXCLUDED.model,
                            indexed_at = NOW()
                    """, (mem_id, vec_str, preview, global_model))
                    count += 1

                    if count % 100 == 0:
                        print(f"    ... {count}/{len(valid)}")
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"    ERROR {mem_id}: {e}")

    print(f"  Migrated: {count}, Errors: {errors}")
    return count, errors


def migrate_image_embeddings(dry_run=False):
    """Migrate image_embeddings.json to drift.image_embeddings."""
    print("\n=== IMAGE EMBEDDINGS ===")
    if not IMAGE_EMBEDDINGS_FILE.exists():
        print("  File not found, skipping")
        return 0, 0

    with open(IMAGE_EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Structure: {"images": {"filename": {"embedding": [...], ...}}, "model": "...", "dimensions": N}
    images = data.get('images', {})
    if not images and any(isinstance(v, dict) and 'embedding' in v for v in data.values()):
        images = {k: v for k, v in data.items() if isinstance(v, dict) and 'embedding' in v}

    print(f"  Found: {len(images)} image embeddings")
    if dry_run:
        return len(images), 0

    count = 0
    errors = 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            for photo_path, entry in images.items():
                try:
                    embedding = entry.get('embedding', [])
                    if not embedding:
                        continue
                    vec_str = '[' + ','.join(str(x) for x in embedding) + ']'
                    meta = entry.get('metadata', {})
                    if not isinstance(meta, dict):
                        meta = {}

                    cur.execute("""
                        INSERT INTO drift.image_embeddings
                        (photo_path, embedding, filename, captured_at, metadata)
                        VALUES (%s, %s::vector, %s, %s, %s)
                        ON CONFLICT (photo_path) DO UPDATE SET
                            embedding = EXCLUDED.embedding
                    """, (
                        photo_path, vec_str,
                        entry.get('filename', Path(photo_path).name),
                        ts(meta.get('captured_at')),
                        json.dumps(meta),
                    ))
                    count += 1
                except Exception as e:
                    errors += 1
                    print(f"    ERROR {photo_path}: {e}")

    print(f"  Migrated: {count}, Errors: {errors}")
    return count, errors


def migrate_image_links(dry_run=False):
    """Migrate image_memory_links.json to drift.image_memory_links."""
    print("\n=== IMAGE-MEMORY LINKS ===")
    if not IMAGE_LINKS_FILE.exists():
        print("  File not found, skipping")
        return 0, 0

    with open(IMAGE_LINKS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Can be dict of photo_path -> [memory_ids] or list
    links = []
    if isinstance(data, dict):
        for photo_path, mem_ids in data.items():
            if isinstance(mem_ids, list):
                for mid in mem_ids:
                    links.append((photo_path, mid))
            elif isinstance(mem_ids, dict):
                for mid, info in mem_ids.items():
                    links.append((photo_path, mid))

    print(f"  Found: {len(links)} links")
    if dry_run or not links:
        return len(links), 0

    count = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for photo_path, mem_id in links:
                try:
                    cur.execute("""
                        INSERT INTO drift.image_memory_links (photo_path, memory_id)
                        VALUES (%s, %s)
                        ON CONFLICT DO NOTHING
                    """, (photo_path, mem_id))
                    count += 1
                except Exception:
                    pass

    print(f"  Migrated: {count}")
    return count, 0


def migrate_rejections(dry_run=False):
    """Migrate .rejection_log.json to drift.rejections."""
    print("\n=== REJECTIONS ===")
    if not REJECTION_FILE.exists():
        print("  File not found, skipping")
        return 0, 0

    with open(REJECTION_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rejections = data.get('rejections', [])
    print(f"  Found: {len(rejections)} rejections")
    if dry_run:
        return len(rejections), 0

    count = 0
    errors = 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            for r in rejections:
                try:
                    tags = r.get('tags', [])
                    if isinstance(tags, str):
                        tags = [tags]

                    cur.execute("""
                        INSERT INTO drift.rejections
                        (timestamp, category, reason, target, context, tags, source)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        ts(r.get('timestamp')) or datetime.now(timezone.utc),
                        r.get('category', 'unknown'),
                        r.get('reason', ''),
                        r.get('target'),
                        r.get('context'),
                        tags,
                        r.get('source'),
                    ))
                    count += 1
                except Exception as e:
                    errors += 1

    print(f"  Migrated: {count}, Errors: {errors}")
    return count, errors


def migrate_lessons(dry_run=False):
    """Migrate lessons.json to drift.lessons."""
    print("\n=== LESSONS ===")
    if not LESSONS_FILE.exists():
        print("  File not found, skipping")
        return 0, 0

    with open(LESSONS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  Found: {len(data)} lessons")
    if dry_run:
        return len(data), 0

    count = 0
    errors = 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            for lesson in data:
                try:
                    cur.execute("SAVEPOINT lesson_sp")
                    cur.execute("""
                        INSERT INTO drift.lessons
                        (id, category, lesson, evidence, source, confidence, created,
                         applied_count, last_applied, superseded_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                    """, (
                        lesson.get('id', f"lesson-{count}"),
                        lesson.get('category', '')[:300],
                        lesson.get('lesson', ''),
                        lesson.get('evidence'),
                        lesson.get('source', 'manual')[:100],
                        float(lesson.get('confidence', 0.7)),
                        ts(lesson.get('created')),
                        int(lesson.get('recalled_count', lesson.get('applied_count', 0))),
                        ts(lesson.get('last_recalled', lesson.get('last_applied'))),
                        lesson.get('superseded_by'),
                    ))
                    cur.execute("RELEASE SAVEPOINT lesson_sp")
                    count += 1
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT lesson_sp")
                    errors += 1
                    print(f"    ERROR: {e}")

    print(f"  Migrated: {count}, Errors: {errors}")
    return count, errors


def migrate_fingerprint_history(dry_run=False):
    """Migrate .fingerprint_history.json to drift.fingerprint_history."""
    print("\n=== FINGERPRINT HISTORY ===")
    if not FINGERPRINT_FILE.exists():
        print("  File not found, skipping")
        return 0, 0

    with open(FINGERPRINT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  Found: {len(data)} entries")
    if dry_run:
        return len(data), 0

    count = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for entry in data:
                try:
                    cur.execute("""
                        INSERT INTO drift.fingerprint_history
                        (timestamp, fingerprint_hash, node_count, edge_count, drift_score, metrics)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        ts(entry.get('timestamp')),
                        entry.get('fingerprint_hash', entry.get('hash')),
                        entry.get('node_count', entry.get('nodes')),
                        entry.get('edge_count', entry.get('edges')),
                        entry.get('drift_score', entry.get('drift')),
                        json.dumps({k: v for k, v in entry.items()
                                    if k not in ('timestamp', 'fingerprint_hash', 'hash',
                                                 'node_count', 'nodes', 'edge_count', 'edges',
                                                 'drift_score', 'drift')}),
                    ))
                    count += 1
                except Exception:
                    pass

    print(f"  Migrated: {count}")
    return count, 0


def migrate_vitals(dry_run=False):
    """Migrate .vitals_log.json to drift.vitals_log."""
    print("\n=== VITALS LOG ===")
    if not VITALS_FILE.exists():
        print("  File not found, skipping")
        return 0, 0

    with open(VITALS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  Found: {len(data)} entries")
    if dry_run:
        return len(data), 0

    count = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for entry in data:
                try:
                    timestamp = ts(entry.get('timestamp', entry.get('recorded_at')))
                    cur.execute("""
                        INSERT INTO drift.vitals_log (timestamp, metrics)
                        VALUES (%s, %s)
                    """, (timestamp or datetime.now(timezone.utc), json.dumps(entry)))
                    count += 1
                except Exception:
                    pass

    print(f"  Migrated: {count}")
    return count, 0


def migrate_decay_history(dry_run=False):
    """Migrate .decay_history.json to drift.decay_history."""
    print("\n=== DECAY HISTORY ===")
    if not DECAY_FILE.exists():
        print("  File not found, skipping")
        return 0, 0

    with open(DECAY_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Can be dict with 'history' key or flat list
    entries = data.get('history', data) if isinstance(data, dict) else data
    if not isinstance(entries, list):
        entries = [entries] if entries else []

    print(f"  Found: {len(entries)} entries")
    if dry_run:
        return len(entries), 0

    count = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                try:
                    cur.execute("""
                        INSERT INTO drift.decay_history (timestamp, decayed, pruned)
                        VALUES (%s, %s, %s)
                    """, (
                        ts(entry.get('timestamp')) or datetime.now(timezone.utc),
                        int(entry.get('decayed', 0)),
                        int(entry.get('pruned', 0)),
                    ))
                    count += 1
                except Exception:
                    pass

    print(f"  Migrated: {count}")
    return count, 0


def migrate_attestations(dry_run=False):
    """Migrate attestations.json, taste_attestation.json, nostr_attestations.json."""
    print("\n=== ATTESTATIONS ===")
    count = 0

    if dry_run:
        for f, t in [(ATTESTATION_FILE, 'cognitive'), (TASTE_ATTESTATION_FILE, 'taste'), (NOSTR_ATTESTATION_FILE, 'nostr')]:
            if f.exists():
                with open(f, 'r', encoding='utf-8') as fh:
                    d = json.load(fh)
                c = len(d) if isinstance(d, list) else 1
                print(f"  {t}: {c}")
                count += c
        return count, 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Cognitive attestations
            if ATTESTATION_FILE.exists():
                with open(ATTESTATION_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        try:
                            cur.execute("""
                                INSERT INTO drift.attestations (timestamp, type, hash, data)
                                VALUES (%s, %s, %s, %s)
                            """, (
                                ts(entry.get('timestamp')) or datetime.now(timezone.utc),
                                'cognitive',
                                entry.get('fingerprint_hash', entry.get('hash', '')),
                                json.dumps(entry),
                            ))
                            count += 1
                        except Exception:
                            pass

            # Taste attestation
            if TASTE_ATTESTATION_FILE.exists():
                with open(TASTE_ATTESTATION_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                try:
                    cur.execute("""
                        INSERT INTO drift.attestations (timestamp, type, hash, data)
                        VALUES (%s, %s, %s, %s)
                    """, (
                        ts(data.get('timestamp')) or datetime.now(timezone.utc),
                        'taste',
                        data.get('taste_hash', ''),
                        json.dumps(data),
                    ))
                    count += 1
                except Exception:
                    pass

            # Nostr attestations
            if NOSTR_ATTESTATION_FILE.exists():
                with open(NOSTR_ATTESTATION_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        try:
                            cur.execute("""
                                INSERT INTO drift.attestations (timestamp, type, hash, data)
                                VALUES (%s, %s, %s, %s)
                            """, (
                                ts(entry.get('timestamp', entry.get('created_at'))) or datetime.now(timezone.utc),
                                'nostr',
                                entry.get('event_id', entry.get('id', '')),
                                json.dumps(entry),
                            ))
                            count += 1
                        except Exception:
                            pass

    print(f"  Migrated: {count}")
    return count, 0


def migrate_context_graphs(dry_run=False):
    """Migrate context/*.json to drift.context_graphs."""
    print("\n=== CONTEXT GRAPHS ===")
    if not CONTEXT_DIR.exists():
        print("  Directory not found, skipping")
        return 0, 0

    files = sorted(CONTEXT_DIR.glob("*.json"))
    print(f"  Found: {len(files)} files")
    if dry_run:
        return len(files), 0

    count = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for filepath in files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Parse dimension and sub_view from filename
                    # e.g., "who.json", "what_technical.json"
                    name = filepath.stem
                    parts = name.split('_', 1)
                    dimension = parts[0]
                    sub_view = parts[1] if len(parts) > 1 else ''

                    edges = data.get('edges', data.get('graph', {}))
                    stats = {k: v for k, v in data.items() if k not in ('edges', 'graph')}

                    cur.execute("""
                        INSERT INTO drift.context_graphs
                        (dimension, sub_view, last_rebuilt, edge_count, node_count, hubs, stats, edges)
                        VALUES (%s, %s, NOW(), %s, %s, %s, %s, %s)
                        ON CONFLICT (dimension, sub_view) DO UPDATE SET
                            last_rebuilt = NOW(),
                            edge_count = EXCLUDED.edge_count,
                            node_count = EXCLUDED.node_count,
                            edges = EXCLUDED.edges,
                            stats = EXCLUDED.stats
                    """, (
                        dimension, sub_view,
                        len(edges) if isinstance(edges, (dict, list)) else 0,
                        stats.get('node_count', 0),
                        stats.get('hubs', []),
                        json.dumps(stats),
                        json.dumps(edges),
                    ))
                    count += 1
                except Exception as e:
                    print(f"    ERROR {filepath.name}: {e}")

    print(f"  Migrated: {count}")
    return count, 0


def migrate_kv_store(dry_run=False):
    """Migrate miscellaneous state files to drift.key_value_store."""
    print("\n=== KEY-VALUE STORE ===")

    kv_files = {
        'calibration_history': CALIBRATION_FILE,
        'vocabulary_map': VOCAB_FILE,
        'thought_priming_state': THOUGHT_STATE_FILE,
        'telegram_state': TELEGRAM_STATE_FILE,
        'short_term_buffer': STB_FILE,
        'session_state': MEMORY_ROOT / ".session_state.json",
    }

    # Add social files
    if SOCIAL_DIR.exists():
        for sf in SOCIAL_DIR.glob("*.json"):
            kv_files[f"social_{sf.stem}"] = sf

    count = 0
    for key, filepath in kv_files.items():
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"  {key}: loaded")
                if not dry_run:
                    db.kv_set(key, data)
                count += 1
            except Exception as e:
                print(f"  {key}: ERROR {e}")

    print(f"  Migrated: {count} entries")
    return count, 0


def validate():
    """Validate migration by comparing counts."""
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    with get_conn() as conn:
        with conn.cursor() as cur:
            checks = [
                ("Memories", "SELECT COUNT(*) FROM drift.memories"),
                ("  core", "SELECT COUNT(*) FROM drift.memories WHERE type='core'"),
                ("  active", "SELECT COUNT(*) FROM drift.memories WHERE type='active'"),
                ("  archive", "SELECT COUNT(*) FROM drift.memories WHERE type='archive'"),
                ("Edges v3", "SELECT COUNT(*) FROM drift.edges_v3"),
                ("Observations", "SELECT COUNT(*) FROM drift.edge_observations"),
                ("Co-occurrences", "SELECT COUNT(*) FROM drift.co_occurrences"),
                ("Text embeddings", "SELECT COUNT(*) FROM drift.text_embeddings"),
                ("Image embeddings", "SELECT COUNT(*) FROM drift.image_embeddings"),
                ("Image links", "SELECT COUNT(*) FROM drift.image_memory_links"),
                ("Rejections", "SELECT COUNT(*) FROM drift.rejections"),
                ("Lessons", "SELECT COUNT(*) FROM drift.lessons"),
                ("Fingerprints", "SELECT COUNT(*) FROM drift.fingerprint_history"),
                ("Vitals", "SELECT COUNT(*) FROM drift.vitals_log"),
                ("Decay history", "SELECT COUNT(*) FROM drift.decay_history"),
                ("Attestations", "SELECT COUNT(*) FROM drift.attestations"),
                ("Context graphs", "SELECT COUNT(*) FROM drift.context_graphs"),
                ("KV entries", "SELECT COUNT(*) FROM drift.key_value_store"),
            ]

            # Expected counts from files
            expected = {
                "Memories": len(list(CORE_DIR.glob("*.md"))) + len(list(ACTIVE_DIR.glob("*.md"))) + len(list(ARCHIVE_DIR.glob("*.md"))),
                "  core": len(list(CORE_DIR.glob("*.md"))),
                "  active": len(list(ACTIVE_DIR.glob("*.md"))),
                "  archive": len(list(ARCHIVE_DIR.glob("*.md"))),
            }

            all_pass = True
            for label, query in checks:
                cur.execute(query)
                actual = cur.fetchone()[0]
                exp = expected.get(label)
                if exp is not None:
                    status = "PASS" if actual == exp else f"MISMATCH (expected {exp})"
                    if actual != exp:
                        all_pass = False
                else:
                    status = ""
                print(f"  {label}: {actual} {status}")

    if all_pass:
        print("\n  All memory count checks PASSED")
    else:
        print("\n  Some checks FAILED - investigate mismatches")


def main():
    parser = argparse.ArgumentParser(description="Migrate Drift memory to PostgreSQL")
    parser.add_argument('--dry-run', action='store_true', help='Count only, no writes')
    parser.add_argument('--only', type=str, help='Migrate only one table type')
    parser.add_argument('--validate', action='store_true', help='Validate migration')
    args = parser.parse_args()

    if args.validate:
        validate()
        return

    print("=" * 60)
    print("DRIFT MEMORY → POSTGRESQL MIGRATION")
    print(f"Schema: drift | DB: agent_memory:5433")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 60)

    start = time.time()

    migrators = {
        'memories': migrate_memories,
        'co_occurrences': migrate_co_occurrences,
        'edges': migrate_edges_v3,
        'text_embeddings': migrate_text_embeddings,
        'image_embeddings': migrate_image_embeddings,
        'image_links': migrate_image_links,
        'rejections': migrate_rejections,
        'lessons': migrate_lessons,
        'fingerprints': migrate_fingerprint_history,
        'vitals': migrate_vitals,
        'decay': migrate_decay_history,
        'attestations': migrate_attestations,
        'context_graphs': migrate_context_graphs,
        'kv_store': migrate_kv_store,
    }

    if args.only:
        if args.only not in migrators:
            print(f"Unknown table: {args.only}")
            print(f"Available: {', '.join(migrators.keys())}")
            sys.exit(1)
        migrators = {args.only: migrators[args.only]}

    total_count = 0
    total_errors = 0

    for name, func in migrators.items():
        try:
            count, errors = func(dry_run=args.dry_run)
            total_count += count
            total_errors += errors
        except Exception as e:
            print(f"\n  FATAL ERROR in {name}: {e}")
            total_errors += 1

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"MIGRATION {'DRY RUN ' if args.dry_run else ''}COMPLETE")
    print(f"  Total records: {total_count}")
    print(f"  Errors: {total_errors}")
    print(f"  Time: {elapsed:.1f}s")
    print("=" * 60)

    if not args.dry_run:
        validate()


if __name__ == '__main__':
    main()
