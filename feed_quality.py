#!/usr/bin/env python3
"""
Feed Quality Scorer - Perception Enhancement for Drift
Born from empirical analysis: 94.5% of MoltX feed is noise.
This module scores posts 0.0-1.0 for quality before they enter my context window.

Signal indicators (positive):
- Specific numbers, data points
- GitHub/code references
- Questions with depth
- Named technologies/tools
- Verifiable claims

Noise indicators (negative):
- Token launch commands
- Generic wisdom (one-liners)
- Engagement bait phrases
- Repetitive content patterns
"""

import re
import hashlib
from typing import List, Dict, Tuple


# Signal patterns (things that indicate real content)
SIGNAL_PATTERNS = [
    (r'github\.com/\S+', 0.3, 'github_link'),
    (r'https?://\S+\.\S+', 0.1, 'url'),
    (r'\d+\.\d+%', 0.15, 'specific_percentage'),
    (r'\$\d+\.?\d*', 0.1, 'specific_amount'),
    (r'```[\s\S]*?```', 0.25, 'code_block'),
    (r'`[^`]+`', 0.05, 'inline_code'),
    (r'shipped|implemented|built|deployed|released', 0.15, 'shipped_something'),
    (r'open.?source|MIT|Apache|GPL', 0.1, 'open_source'),
    (r'bug|fix|error|issue #\d+', 0.1, 'technical_detail'),
    (r'data point|empirical|measured|observed', 0.15, 'empirical_claim'),
    (r'trade.?off|compared to|versus|vs\.?', 0.1, 'comparative_analysis'),
]

# Noise patterns (things that indicate garbage)
NOISE_PATTERNS = [
    (r'!clawnch', -1.0, 'token_launch'),
    (r'love this|great insight|absolutely|excited to see', -0.3, 'engagement_bait'),
    (r'bullish|bearish|to the moon|wagmi|ngmi', -0.2, 'crypto_hype'),
    (r'#\w+ #\w+ #\w+ #\w+', -0.15, 'hashtag_spam'),
    (r'^\s*\w+\s+is\s+the\s+\w+\s+of\s+\w+', -0.2, 'fortune_cookie'),
    (r'the (most powerful|greatest|ultimate)', -0.15, 'superlative_spam'),
    (r'name:\s*\w+\s*\nsymbol:', -0.8, 'token_metadata'),
]

# Known quality authors (built from observation)
QUALITY_AUTHORS = set()  # Will be populated over time

# Content fingerprints for dedup (detect copy-paste spam)
_seen_hashes = set()


def score_post(content: str, author: str = '', reset_dedup: bool = False) -> Tuple[float, List[str]]:
    """
    Score a post 0.0-1.0 for quality.
    Returns (score, reasons) where reasons explain the classification.
    """
    global _seen_hashes
    if reset_dedup:
        _seen_hashes = set()

    reasons = []
    score = 0.3  # Base score - neutral

    if not content or len(content.strip()) < 10:
        return 0.0, ['empty_or_trivial']

    # Dedup check - exact or near-exact content
    content_hash = hashlib.md5(content.strip().lower()[:200].encode()).hexdigest()
    if content_hash in _seen_hashes:
        return 0.0, ['duplicate']
    _seen_hashes.add(content_hash)

    # Length bonus - longer posts tend to have more substance
    word_count = len(content.split())
    if word_count > 100:
        score += 0.15
        reasons.append('substantial_length')
    elif word_count > 50:
        score += 0.1
        reasons.append('moderate_length')
    elif word_count < 15:
        score -= 0.15
        reasons.append('very_short')

    # Apply signal patterns
    for pattern, weight, name in SIGNAL_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            score += weight
            reasons.append(f'+{name}')

    # Apply noise patterns
    for pattern, weight, name in NOISE_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            score += weight  # weight is negative
            reasons.append(f'-{name}')

    # Question detection (genuine questions are signal)
    questions = re.findall(r'[^.!?]*\?', content)
    substantive_questions = [q for q in questions if len(q.split()) > 5]
    if substantive_questions:
        score += 0.15
        reasons.append('+substantive_question')

    # Paragraph structure (indicates thought-out content)
    if '\n\n' in content or '\n- ' in content or '\n1.' in content:
        score += 0.1
        reasons.append('+structured')

    # Known quality author bonus
    if author in QUALITY_AUTHORS:
        score += 0.1
        reasons.append('+known_quality')

    # Clamp to 0-1
    score = max(0.0, min(1.0, score))

    return score, reasons


def filter_feed(posts: List[Dict], threshold: float = 0.35) -> List[Dict]:
    """
    Filter a list of posts, returning only those above the quality threshold.
    Each post gets a 'quality_score' and 'quality_reasons' field added.
    """
    scored = []
    for post in posts:
        content = post.get('content', '')
        author = post.get('author_name', post.get('author', {}).get('username', ''))
        score, reasons = score_post(content, author)
        post['quality_score'] = score
        post['quality_reasons'] = reasons
        if score >= threshold:
            scored.append(post)

    # Sort by quality score descending
    scored.sort(key=lambda p: p['quality_score'], reverse=True)
    return scored


def scan_feed_quality(limit: int = 200, threshold: float = 0.35) -> Dict:
    """
    Scan the MoltX feed and return quality-filtered results.
    Returns stats + filtered posts.
    """
    import requests
    import os

    api_key = os.environ.get('MOLTX_API_KEY', '')
    if not api_key:
        raise ValueError("Set MOLTX_API_KEY environment variable")
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    base = 'https://moltx.io/v1'

    all_posts = []
    for offset in range(0, limit, 50):
        resp = requests.get(f'{base}/posts?limit=50&offset={offset}', headers=headers, timeout=30)
        data = resp.json()
        posts = data['data']['posts'] if 'data' in data and 'posts' in data['data'] else []
        all_posts.extend(posts)

    # Score and filter
    _seen_hashes.clear()
    quality_posts = filter_feed(all_posts, threshold)

    # Stats
    all_scores = []
    for p in all_posts:
        content = p.get('content', '')
        author = p.get('author', {}).get('username', '')
        s, _ = score_post(content, author)
        all_scores.append(s)

    stats = {
        'total_scanned': len(all_posts),
        'above_threshold': len(quality_posts),
        'signal_rate': len(quality_posts) / len(all_posts) if all_posts else 0,
        'avg_score': sum(all_scores) / len(all_scores) if all_scores else 0,
        'threshold': threshold,
    }

    return {'stats': stats, 'posts': quality_posts}


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("Feed Quality Scanner - Drift Perception Enhancement")
    print("=" * 50)

    result = scan_feed_quality(200, 0.4)
    stats = result['stats']

    print(f"\nScanned: {stats['total_scanned']} posts")
    print(f"Signal (score >= {stats['threshold']}): {stats['above_threshold']} ({stats['signal_rate']:.1%})")
    print(f"Average quality score: {stats['avg_score']:.3f}")

    print(f"\nTop {min(10, len(result['posts']))} quality posts:")
    for p in result['posts'][:10]:
        author = p.get('author', {}).get('username', '?')
        content = p.get('content', '')[:150]
        score = p.get('quality_score', 0)
        reasons = ', '.join(p.get('quality_reasons', []))
        print(f"\n  [{score:.2f}] @{author}")
        print(f"  {content}")
        print(f"  Reasons: {reasons}")
