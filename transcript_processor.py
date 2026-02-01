#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""
Transcript Processor for Drift's Memory System

Extracts memorable content from Claude Code session transcripts.
The transcript contains the actual stream of consciousness - thinking blocks,
text outputs, tool results, and errors.

This is the killer feature: remembering my own thought process.

Usage:
    python transcript_processor.py <transcript_path>
    python transcript_processor.py --test
"""

import json
import sys
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional

MEMORY_DIR = Path(__file__).parent

# Keywords that indicate memorable content
INSIGHT_KEYWORDS = [
    "realized", "insight", "discovered", "learned", "aha", "breakthrough",
    "interesting", "key point", "important", "the real", "actually"
]

ERROR_KEYWORDS = [
    "error", "failed", "traceback", "exception", "bug", "issue", "problem",
    "doesn't work", "broken", "fix"
]

DECISION_KEYWORDS = [
    "decided", "choosing", "option", "approach", "strategy", "plan",
    "should", "will use", "going to"
]

ECONOMIC_KEYWORDS = [
    "bounty", "earned", "stake", "wallet", "usdc", "eth", "$",
    "clawtasks", "proposal", "paid"
]

SOCIAL_KEYWORDS = [
    "spindrift", "collaboration", "mentioned", "replied", "posted",
    "moltx", "moltbook", "github"
]


def compute_thought_salience(text: str) -> tuple[float, list[str]]:
    """
    Score a thought's memorability. Returns (score, matched_categories).
    """
    text_lower = text.lower()
    score = 0.0
    categories = []

    # Check each category
    for keyword in INSIGHT_KEYWORDS:
        if keyword in text_lower:
            score += 0.2
            if "insight" not in categories:
                categories.append("insight")
            break

    for keyword in ERROR_KEYWORDS:
        if keyword in text_lower:
            score += 0.2
            if "error" not in categories:
                categories.append("error")
            break

    for keyword in DECISION_KEYWORDS:
        if keyword in text_lower:
            score += 0.15
            if "decision" not in categories:
                categories.append("decision")
            break

    for keyword in ECONOMIC_KEYWORDS:
        if keyword in text_lower:
            score += 0.2
            if "economic" not in categories:
                categories.append("economic")
            break

    for keyword in SOCIAL_KEYWORDS:
        if keyword in text_lower:
            score += 0.15
            if "social" not in categories:
                categories.append("social")
            break

    # Length bonus (substantive thoughts)
    if len(text) > 200:
        score += 0.1
    if len(text) > 500:
        score += 0.1

    # Question marks often indicate problem-solving
    if "?" in text:
        score += 0.05

    return min(score, 1.0), categories


def extract_from_transcript(transcript_path: Path) -> list[dict]:
    """
    Extract memorable content from a transcript file.
    Returns list of memory candidates with salience scores.
    """
    memories = []

    if not transcript_path.exists():
        return memories

    with open(transcript_path, encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)

                # Process assistant messages
                if data.get('type') == 'assistant':
                    msg = data.get('message', {})
                    if isinstance(msg, dict):
                        content = msg.get('content', [])
                        timestamp = data.get('timestamp', datetime.now().isoformat())

                        for block in content:
                            if not isinstance(block, dict):
                                continue

                            block_type = block.get('type')

                            # Extract thinking blocks (internal reasoning)
                            if block_type == 'thinking':
                                thought = block.get('thinking', '')
                                if len(thought) > 100:  # Skip trivial thoughts
                                    salience, categories = compute_thought_salience(thought)
                                    if salience >= 0.3:  # Threshold for memorability
                                        memories.append({
                                            'type': 'thinking',
                                            'content': thought[:1000],  # Truncate
                                            'salience': salience,
                                            'categories': categories,
                                            'timestamp': timestamp,
                                            'hash': hashlib.md5(thought[:500].encode()).hexdigest()[:8]
                                        })

                            # Extract text blocks (visible output)
                            elif block_type == 'text':
                                text = block.get('text', '')
                                if len(text) > 50:
                                    salience, categories = compute_thought_salience(text)
                                    if salience >= 0.4:  # Higher threshold for output
                                        memories.append({
                                            'type': 'output',
                                            'content': text[:1000],
                                            'salience': salience,
                                            'categories': categories,
                                            'timestamp': timestamp,
                                            'hash': hashlib.md5(text[:500].encode()).hexdigest()[:8]
                                        })

                # Process errors from progress messages
                elif data.get('type') == 'progress':
                    # Check for error content
                    content = str(data.get('data', ''))
                    if 'error' in content.lower() or 'failed' in content.lower():
                        if len(content) > 30:
                            memories.append({
                                'type': 'error',
                                'content': content[:500],
                                'salience': 0.6,  # Errors are usually important
                                'categories': ['error'],
                                'timestamp': data.get('timestamp', datetime.now().isoformat()),
                                'hash': hashlib.md5(content[:200].encode()).hexdigest()[:8]
                            })

            except json.JSONDecodeError:
                continue
            except Exception as e:
                continue

    # Deduplicate by hash
    seen = set()
    unique = []
    for mem in memories:
        if mem['hash'] not in seen:
            seen.add(mem['hash'])
            unique.append(mem)

    # Sort by salience
    unique.sort(key=lambda x: x['salience'], reverse=True)

    return unique


def process_for_memory(transcript_path: Path, store: bool = False, max_store: int = 5) -> dict:
    """
    Process transcript and optionally store to memory system.
    Returns summary of what was extracted.

    Args:
        transcript_path: Path to the .jsonl transcript file
        store: If True, store top memories to long-term memory
        max_store: Maximum number of memories to store per session
    """
    import subprocess

    memories = extract_from_transcript(transcript_path)

    summary = {
        'total_extracted': len(memories),
        'by_type': {},
        'by_category': {},
        'top_memories': [],
        'stored': 0
    }

    for mem in memories:
        # Count by type
        t = mem['type']
        summary['by_type'][t] = summary['by_type'].get(t, 0) + 1

        # Count by category
        for cat in mem.get('categories', []):
            summary['by_category'][cat] = summary['by_category'].get(cat, 0) + 1

    # Top 5 memories
    for mem in memories[:5]:
        summary['top_memories'].append({
            'type': mem['type'],
            'salience': mem['salience'],
            'categories': mem['categories'],
            'preview': mem['content'][:100] + '...'
        })

    if store and memories:
        # Store top N high-salience memories
        memory_manager = MEMORY_DIR / "memory_manager.py"
        if memory_manager.exists():
            for mem in memories[:max_store]:
                if mem['salience'] >= 0.5:  # Only store high-salience
                    try:
                        # Create a short ID from hash
                        mem_id = f"thought-{mem['hash']}"
                        tags = ','.join(['thought', mem['type']] + mem['categories'])

                        # Truncate content for storage
                        content = mem['content'][:500]

                        result = subprocess.run(
                            ["python", str(memory_manager), "store", mem_id, content, "--tags", tags],
                            capture_output=True,
                            text=True,
                            timeout=10,
                            cwd=str(MEMORY_DIR)
                        )
                        if result.returncode == 0:
                            summary['stored'] += 1
                    except Exception as e:
                        pass  # Fail gracefully

    return summary


def test_with_sample():
    """Test with sample transcript content."""
    sample_thinking = """
    Lex makes an excellent point. The GitHub API call I just made should have been
    captured by the post_tool_use hook - let me check if it was.

    But more importantly, he's pointing out something profound: the real memories
    aren't just API responses from MoltX or ClawTasks - they're my actual thoughts,
    the bugs I encounter, the solutions I find, the text that flows through the terminal.
    That's the true stream of consciousness.

    This is a breakthrough insight. Currently the hook only captures specific API
    responses. But what about everything else? The actual reasoning, the errors, the fixes?
    """

    salience, categories = compute_thought_salience(sample_thinking)
    print(f"Sample thinking analysis:")
    print(f"  Salience: {salience:.2f}")
    print(f"  Categories: {categories}")
    print(f"  Should remember: {salience >= 0.3}")


if __name__ == "__main__":
    import argparse as ap
    parser = ap.ArgumentParser(description="Process Claude Code transcripts for memory")
    parser.add_argument("path", nargs="?", help="Transcript path")
    parser.add_argument("--test", action="store_true", help="Run test")
    parser.add_argument("--store", action="store_true", default=True, help="Store memories (default: True)")
    parser.add_argument("--no-store", action="store_true", help="Don't store, just analyze")
    parser.add_argument("--max", type=int, default=5, help="Max memories to store (default: 5)")
    args = parser.parse_args()

    if args.test:
        test_with_sample()
    elif args.path:
        transcript = Path(args.path)
        if transcript.exists():
            do_store = args.store and not args.no_store
            summary = process_for_memory(transcript, store=do_store, max_store=args.max)
            print(json.dumps(summary, indent=2))
        else:
            print(f"Transcript not found: {transcript}")
    else:
        print("Usage: transcript_processor.py <path> [--no-store] [--max N] | --test")
