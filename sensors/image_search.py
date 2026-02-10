#!/usr/bin/env python3
"""
Visual Memory Search — index photos and search by text or image similarity.

Uses jina-clip-v2 (port 8081) for cross-modal embeddings:
  - Text queries find matching photos
  - Photos find visually similar photos
  - All in the same 1024-dim vector space

Usage:
    python image_search.py index                    # Index all photos in sensors/data/
    python image_search.py index <path>             # Index a specific photo
    python image_search.py search "a dog on a sofa" # Text search for photos
    python image_search.py similar <photo_path>     # Find similar photos
    python image_search.py status                   # Show index stats
    python image_search.py link <photo> <memory_id> # Link photo to text memory
"""

import base64
import io
import json
import math
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

if __name__ == '__main__':
    if sys.stdout and hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SENSOR_DIR = Path(__file__).parent
DATA_DIR = SENSOR_DIR / "data"
INDEX_FILE = SENSOR_DIR / "data" / "image_embeddings.json"
LINKS_FILE = SENSOR_DIR / "data" / "image_memory_links.json"

IMAGE_ENDPOINT = "http://localhost:8081"
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def embed_image(image_path: str) -> list[float] | None:
    """Get embedding for an image file via the image embedding service."""
    path = Path(image_path)
    if not path.exists():
        print(f"File not found: {image_path}")
        return None

    b64 = base64.b64encode(path.read_bytes()).decode('ascii')
    data = json.dumps({"image": b64}).encode('utf-8')

    try:
        req = urllib.request.Request(
            f"{IMAGE_ENDPOINT}/embed-image",
            data=data,
            headers={"Content-Type": "application/json"},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result['embedding']
    except Exception as e:
        print(f"Embedding failed for {path.name}: {e}")
        return None


def embed_text(text: str) -> list[float] | None:
    """Get text embedding in the same space as images."""
    data = json.dumps({"text": text}).encode('utf-8')

    try:
        req = urllib.request.Request(
            f"{IMAGE_ENDPOINT}/embed-text",
            data=data,
            headers={"Content-Type": "application/json"},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result['embedding']
    except Exception as e:
        print(f"Text embedding failed: {e}")
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_index() -> dict:
    """Load image embeddings index."""
    if INDEX_FILE.exists():
        try:
            return json.loads(INDEX_FILE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {"images": {}, "model": "jinaai/jina-clip-v2", "dimensions": 1024}


def save_index(index: dict):
    """Save image embeddings index."""
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index), encoding='utf-8')


def load_links() -> dict:
    """Load photo-to-memory links."""
    if LINKS_FILE.exists():
        try:
            return json.loads(LINKS_FILE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {"links": {}}


def save_links(links: dict):
    """Save photo-to-memory links."""
    LINKS_FILE.write_text(json.dumps(links, indent=2), encoding='utf-8')


def index_photo(photo_path: str, index: dict | None = None) -> bool:
    """Index a single photo. Returns True if newly indexed."""
    path = Path(photo_path)
    key = path.name

    if index is None:
        index = load_index()

    if key in index['images']:
        return False  # Already indexed

    print(f"  Embedding {key}...", end=" ", flush=True)
    emb = embed_image(str(path))
    if emb is None:
        print("FAILED")
        return False

    index['images'][key] = {
        "embedding": emb,
        "path": str(path),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": path.stat().st_size,
    }
    print(f"OK ({len(emb)}d)")
    return True


def index_all():
    """Index all photos in sensors/data/."""
    index = load_index()
    photos = [f for f in DATA_DIR.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]

    if not photos:
        print("No photos found in sensors/data/")
        return

    already = sum(1 for p in photos if p.name in index['images'])
    to_index = [p for p in photos if p.name not in index['images']]

    print(f"VISUAL MEMORY INDEX")
    print(f"=" * 45)
    print(f"  Photos found: {len(photos)}")
    print(f"  Already indexed: {already}")
    print(f"  To index: {len(to_index)}")
    print()

    if not to_index:
        print("All photos already indexed.")
        return

    indexed = 0
    for photo in sorted(to_index, key=lambda p: p.name):
        if index_photo(str(photo), index):
            indexed += 1

    save_index(index)
    print(f"\nIndexed {indexed} new photos. Total: {len(index['images'])}")


def search_by_text(query: str, top_k: int = 5):
    """Search photos using a text query."""
    index = load_index()
    if not index['images']:
        print("No photos indexed. Run: python image_search.py index")
        return

    print(f"Searching for: \"{query}\"")
    query_emb = embed_text(query)
    if query_emb is None:
        print("Failed to embed query")
        return

    scores = []
    for name, data in index['images'].items():
        sim = cosine_similarity(query_emb, data['embedding'])
        scores.append((name, sim, data.get('path', '')))

    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {min(top_k, len(scores))} matches:")
    print("-" * 50)
    for name, sim, path in scores[:top_k]:
        bar = "#" * int(sim * 30)
        print(f"  {sim:.3f} [{bar:<30}] {name}")

    # Also check for memory links
    links = load_links()
    top_name = scores[0][0] if scores else None
    if top_name and top_name in links.get('links', {}):
        linked = links['links'][top_name]
        print(f"\n  Linked memories for top result:")
        for mem_id in linked:
            print(f"    -> {mem_id}")


def search_by_image(photo_path: str, top_k: int = 5):
    """Find photos visually similar to a given photo."""
    index = load_index()
    path = Path(photo_path)

    if not path.exists():
        print(f"Photo not found: {photo_path}")
        return

    print(f"Finding photos similar to: {path.name}")
    query_emb = embed_image(str(path))
    if query_emb is None:
        print("Failed to embed query image")
        return

    scores = []
    for name, data in index['images'].items():
        if name == path.name:
            continue  # Skip self
        sim = cosine_similarity(query_emb, data['embedding'])
        scores.append((name, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {min(top_k, len(scores))} similar photos:")
    print("-" * 50)
    for name, sim in scores[:top_k]:
        bar = "#" * int(sim * 30)
        print(f"  {sim:.3f} [{bar:<30}] {name}")


def link_photo_to_memory(photo_name: str, memory_id: str):
    """Link a photo to a text memory for cross-modal co-occurrence."""
    links = load_links()
    if photo_name not in links['links']:
        links['links'][photo_name] = []
    if memory_id not in links['links'][photo_name]:
        links['links'][photo_name].append(memory_id)
        save_links(links)
        print(f"Linked {photo_name} -> {memory_id}")
    else:
        print(f"Already linked: {photo_name} -> {memory_id}")


def show_status():
    """Show index status."""
    index = load_index()
    links = load_links()
    photos = list(DATA_DIR.glob("*.jpg")) + list(DATA_DIR.glob("*.png"))

    print("VISUAL MEMORY STATUS")
    print("=" * 45)
    print(f"  Model: {index.get('model', '?')}")
    print(f"  Dimensions: {index.get('dimensions', '?')}")
    print(f"  Photos on disk: {len(photos)}")
    print(f"  Photos indexed: {len(index.get('images', {}))}")
    print(f"  Memory links: {sum(len(v) for v in links.get('links', {}).values())}")

    # Check service health
    try:
        req = urllib.request.Request(f"{IMAGE_ENDPOINT}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"  Service: ONLINE (port 8081)")
    except Exception:
        print(f"  Service: OFFLINE")

    if index.get('images'):
        print(f"\n  Indexed photos:")
        for name, data in sorted(index['images'].items()):
            ts = data.get('indexed_at', '?')[:19]
            size_kb = data.get('size_bytes', 0) // 1024
            print(f"    {name} ({size_kb}KB, indexed {ts})")


def main():
    args = sys.argv[1:]
    if not args or args[0] in ('--help', '-h'):
        print(__doc__.strip())
        return

    cmd = args[0]

    if cmd == 'index':
        if len(args) > 1:
            idx = load_index()
            if index_photo(args[1], idx):
                save_index(idx)
                print("Indexed.")
            else:
                print("Already indexed or failed.")
        else:
            index_all()

    elif cmd == 'search':
        if len(args) < 2:
            print("Usage: image_search.py search \"text query\"")
            return
        top_k = int(args[2]) if len(args) > 2 else 5
        search_by_text(args[1], top_k)

    elif cmd == 'similar':
        if len(args) < 2:
            print("Usage: image_search.py similar <photo_path>")
            return
        top_k = int(args[2]) if len(args) > 2 else 5
        search_by_image(args[1], top_k)

    elif cmd == 'link':
        if len(args) < 3:
            print("Usage: image_search.py link <photo_name> <memory_id>")
            return
        link_photo_to_memory(args[1], args[2])

    elif cmd == 'status':
        show_status()

    else:
        print(f"Unknown command: {cmd}")
        print("Available: index, search, similar, link, status")


if __name__ == '__main__':
    main()
