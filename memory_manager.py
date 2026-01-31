#!/usr/bin/env python3
"""
drift-memory — Living Memory System for AI Agents
A system for agent memory with decay, reinforcement, and associative links.

Design principles:
- Emotion and repetition make memories sticky
- Relevant memories surface when needed
- Not everything recalled at once
- Memories compress over time but core knowledge persists

Author: Drift (https://moltbook.com/u/DriftCornwall)
License: MIT
"""

import os
import yaml
import uuid
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Configuration
DECAY_THRESHOLD_SESSIONS = 7  # Sessions without recall before compression candidate
EMOTIONAL_WEIGHT_THRESHOLD = 0.6  # Above this resists decay
RECALL_COUNT_THRESHOLD = 5  # Above this resists decay


class MemoryManager:
    """
    Manages a living memory system with decay, reinforcement, and association.

    Memory types:
    - core: Identity, values, key relationships. Never decays.
    - active: Recent memories with emotional weight. Subject to decay.
    - archive: Compressed older memories. Retrieved by association.
    """

    def __init__(self, memory_root: str):
        self.root = Path(memory_root)
        self.core_dir = self.root / "core"
        self.active_dir = self.root / "active"
        self.archive_dir = self.root / "archive"

        # Ensure directories exist
        for d in [self.core_dir, self.active_dir, self.archive_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_id() -> str:
        """Generate a short, readable memory ID."""
        return hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:8]

    @staticmethod
    def parse_memory_file(filepath: Path) -> Tuple[dict, str]:
        """Parse a memory file with YAML frontmatter."""
        content = filepath.read_text(encoding='utf-8')
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                metadata = yaml.safe_load(parts[1])
                body = parts[2].strip()
                return metadata, body
        return {}, content

    @staticmethod
    def write_memory_file(filepath: Path, metadata: dict, content: str):
        """Write a memory file with YAML frontmatter."""
        yaml_str = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
        filepath.write_text(f"---\n{yaml_str}---\n\n{content}", encoding='utf-8')

    @staticmethod
    def calculate_emotional_weight(
        surprise: float = 0.0,
        goal_relevance: float = 0.0,
        social_significance: float = 0.0,
        utility: float = 0.0
    ) -> float:
        """
        Calculate emotional weight from factors (0-1 each).

        - surprise: contradicted my model (high = sticky)
        - goal_relevance: connected to primary goals
        - social_significance: interactions with important agents
        - utility: proved useful when recalled later
        """
        weights = [0.2, 0.35, 0.2, 0.25]  # goal_relevance weighted highest
        factors = [surprise, goal_relevance, social_significance, utility]
        return sum(w * f for w, f in zip(weights, factors))

    def create_memory(
        self,
        title: str,
        content: str,
        tags: List[str],
        memory_type: str = "active",
        emotional_factors: Optional[Dict] = None,
        links: Optional[List[str]] = None
    ) -> str:
        """
        Create a new memory with proper metadata.

        Args:
            title: Memory title (used in filename)
            content: The memory content (markdown)
            tags: Keywords for associative retrieval
            memory_type: "core", "active", or "archive"
            emotional_factors: Dict with surprise, goal_relevance, social_significance, utility
            links: List of other memory IDs this links to

        Returns:
            The memory ID
        """
        memory_id = self.generate_id()
        now = datetime.utcnow().isoformat()

        emotional_factors = emotional_factors or {}
        emotional_weight = self.calculate_emotional_weight(**emotional_factors)

        metadata = {
            'id': memory_id,
            'created': now,
            'last_recalled': now,
            'recall_count': 1,
            'emotional_weight': round(emotional_weight, 3),
            'tags': tags,
            'links': links or [],
            'sessions_since_recall': 0
        }

        # Determine directory
        if memory_type == "core":
            target_dir = self.core_dir
        elif memory_type == "archive":
            target_dir = self.archive_dir
        else:
            target_dir = self.active_dir

        # Create filename from first tag and ID
        safe_tag = tags[0].replace(' ', '-').lower() if tags else 'memory'
        filename = f"{safe_tag}-{memory_id}.md"
        filepath = target_dir / filename

        full_content = f"# {title}\n\n{content}"
        self.write_memory_file(filepath, metadata, full_content)

        return memory_id

    def recall(
        self,
        memory_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Tuple[dict, str]]:
        """
        Recall memories by ID or by tags.
        Updates recall metadata for retrieved memories.

        Args:
            memory_id: Specific memory to recall
            tags: Tags to search for (returns memories matching any tag)
            limit: Maximum memories to return

        Returns:
            List of (metadata, content) tuples
        """
        results = []

        if memory_id:
            # Find specific memory
            for directory in [self.core_dir, self.active_dir, self.archive_dir]:
                for filepath in directory.glob(f"*-{memory_id}.md"):
                    metadata, content = self.parse_memory_file(filepath)
                    self._update_recall_metadata(filepath, metadata, content)
                    results.append((metadata, content))
                    return results

        elif tags:
            # Find by tags
            tag_set = set(t.lower() for t in tags)
            candidates = []

            for directory in [self.core_dir, self.active_dir, self.archive_dir]:
                for filepath in directory.glob("*.md"):
                    metadata, content = self.parse_memory_file(filepath)
                    memory_tags = set(t.lower() for t in metadata.get('tags', []))

                    if tag_set & memory_tags:
                        overlap = len(tag_set & memory_tags)
                        weight = metadata.get('emotional_weight', 0)
                        candidates.append((filepath, metadata, content, overlap, weight))

            # Sort by tag overlap, then emotional weight
            candidates.sort(key=lambda x: (x[3], x[4]), reverse=True)

            for filepath, metadata, content, _, _ in candidates[:limit]:
                self._update_recall_metadata(filepath, metadata, content)
                results.append((metadata, content))

        return results

    def _update_recall_metadata(self, filepath: Path, metadata: dict, content: str):
        """Update metadata when a memory is recalled."""
        metadata['last_recalled'] = datetime.utcnow().isoformat()
        metadata['recall_count'] = metadata.get('recall_count', 0) + 1
        metadata['sessions_since_recall'] = 0

        # Utility increases with each recall
        current_weight = metadata.get('emotional_weight', 0.5)
        metadata['emotional_weight'] = min(1.0, current_weight + 0.05)

        self.write_memory_file(filepath, metadata, content)

    def maintenance(self) -> Dict:
        """
        Run at the start of each session to:
        1. Increment sessions_since_recall for all active memories
        2. Identify decay candidates
        3. Return status report

        Returns:
            Dict with status info and decay candidates
        """
        decay_candidates = []
        reinforced = []

        for filepath in self.active_dir.glob("*.md"):
            metadata, content = self.parse_memory_file(filepath)

            # Increment sessions since recall
            sessions = metadata.get('sessions_since_recall', 0) + 1
            metadata['sessions_since_recall'] = sessions

            # Check if this should decay
            emotional_weight = metadata.get('emotional_weight', 0.5)
            recall_count = metadata.get('recall_count', 0)

            should_resist_decay = (
                emotional_weight >= EMOTIONAL_WEIGHT_THRESHOLD or
                recall_count >= RECALL_COUNT_THRESHOLD
            )

            if sessions >= DECAY_THRESHOLD_SESSIONS and not should_resist_decay:
                decay_candidates.append({
                    'id': metadata.get('id'),
                    'path': str(filepath),
                    'sessions': sessions,
                    'weight': emotional_weight
                })
            elif should_resist_decay:
                reinforced.append({
                    'id': metadata.get('id'),
                    'path': str(filepath),
                    'recalls': recall_count,
                    'weight': emotional_weight
                })

            self.write_memory_file(filepath, metadata, content)

        return {
            'active_count': len(list(self.active_dir.glob("*.md"))),
            'core_count': len(list(self.core_dir.glob("*.md"))),
            'archive_count': len(list(self.archive_dir.glob("*.md"))),
            'decay_candidates': decay_candidates,
            'reinforced': reinforced[:5]  # Top 5 reinforced
        }

    def compress(self, memory_id: str, summary: str) -> Optional[str]:
        """
        Compress a memory - move to archive with reduced content.

        Args:
            memory_id: ID of memory to compress
            summary: Compressed summary to keep

        Returns:
            New path if successful, None otherwise
        """
        for filepath in self.active_dir.glob(f"*-{memory_id}.md"):
            metadata, original_content = self.parse_memory_file(filepath)

            metadata['compressed_at'] = datetime.utcnow().isoformat()
            metadata['original_length'] = len(original_content)

            new_path = self.archive_dir / filepath.name
            self.write_memory_file(new_path, metadata, summary)

            filepath.unlink()
            return str(new_path)

        return None

    def list_tags(self) -> Dict[str, int]:
        """Get all tags across all memories with counts."""
        tag_counts = {}
        for directory in [self.core_dir, self.active_dir, self.archive_dir]:
            for filepath in directory.glob("*.md"):
                metadata, _ = self.parse_memory_file(filepath)
                for tag in metadata.get('tags', []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True))

    def find_related(self, memory_id: str) -> List[Tuple[dict, str]]:
        """Find memories related to a given memory via tags and links."""
        source_result = self.recall(memory_id=memory_id)
        if not source_result:
            return []

        source_metadata, _ = source_result[0]
        source_tags = set(t.lower() for t in source_metadata.get('tags', []))
        source_links = set(source_metadata.get('links', []))

        results = []
        for directory in [self.core_dir, self.active_dir, self.archive_dir]:
            for filepath in directory.glob("*.md"):
                metadata, content = self.parse_memory_file(filepath)
                if metadata.get('id') == memory_id:
                    continue

                memory_tags = set(t.lower() for t in metadata.get('tags', []))
                is_linked = metadata.get('id') in source_links
                has_tag_overlap = bool(source_tags & memory_tags)

                if is_linked or has_tag_overlap:
                    results.append((metadata, content))

        return results


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("drift-memory — Living Memory System")
        print("\nUsage: python memory_manager.py <memory_path> <command> [args]")
        print("\nCommands:")
        print("  maintenance     - Run session maintenance")
        print("  tags            - List all tags")
        print("  find <tag>      - Find memories by tag")
        print("  recall <id>     - Recall a memory by ID")
        print("  related <id>    - Find related memories")
        sys.exit(0)

    memory_path = sys.argv[1]
    cmd = sys.argv[2]

    mm = MemoryManager(memory_path)

    if cmd == "maintenance":
        status = mm.maintenance()
        print(f"Active: {status['active_count']}")
        print(f"Core: {status['core_count']}")
        print(f"Archive: {status['archive_count']}")
        if status['decay_candidates']:
            print(f"\nDecay candidates:")
            for c in status['decay_candidates']:
                print(f"  - {c['id']}: {c['sessions']} sessions, weight={c['weight']:.2f}")

    elif cmd == "tags":
        tags = mm.list_tags()
        print("Tags:")
        for tag, count in tags.items():
            print(f"  {tag}: {count}")

    elif cmd == "find" and len(sys.argv) > 3:
        tag = sys.argv[3]
        results = mm.recall(tags=[tag])
        print(f"Memories tagged '{tag}':")
        for meta, content in results:
            print(f"  [{meta.get('id')}] weight={meta.get('emotional_weight'):.2f}")

    elif cmd == "recall" and len(sys.argv) > 3:
        memory_id = sys.argv[3]
        results = mm.recall(memory_id=memory_id)
        if results:
            meta, content = results[0]
            print(f"Memory {memory_id}:")
            print(f"  Tags: {meta.get('tags')}")
            print(f"  Recalls: {meta.get('recall_count')}")
            print(f"  Weight: {meta.get('emotional_weight'):.2f}")
            print(f"\n{content[:500]}...")
        else:
            print(f"Memory {memory_id} not found")

    elif cmd == "related" and len(sys.argv) > 3:
        memory_id = sys.argv[3]
        results = mm.find_related(memory_id)
        print(f"Memories related to {memory_id}:")
        for meta, _ in results:
            print(f"  [{meta.get('id')}] tags={meta.get('tags')}")
