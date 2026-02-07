#!/usr/bin/env python3
"""
Entity Detection — Extracted from memory_manager.py (Phase 1)

Pure NLP functions for detecting entities and dates from text content.
No file system access, no memory store dependencies.

Credit: Kaleaon (Landseek-Amphibian) Tri-Agent Interop proposal (Issue #6)
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Optional


# v2.15: Entity-Centric Tagging - Typed entity links (ENTITY edges from Kaleaon schema)
ENTITY_TYPES = ['agent', 'project', 'concept', 'location', 'event']

KNOWN_AGENTS = {
    'spindriftmend', 'spindriftmind', 'spin', 'spindrift',
    'kaleaon', 'cosmo', 'amphibian',
    'drift', 'driftcornwall',
    'lex', 'flycompoundeye', 'buzz',
    'mikaopenclaw', 'mika'
}

KNOWN_PROJECTS = {
    'drift-memory', 'amphibian', 'landseek-amphibian', 'moltbook',
    'moltx', 'clawtasks', 'gitmolt', 'moltswarm'
}

# Canonical name mappings for agent aliases
_AGENT_ALIASES = {
    'spindriftmend': 'spindriftmend',
    'spindriftmind': 'spindriftmend',
    'spin': 'spindriftmend',
    'spindrift': 'spindriftmend',
    'kaleaon': 'kaleaon',
    'cosmo': 'kaleaon',
    'amphibian': 'kaleaon',
    'drift': 'drift',
    'driftcornwall': 'drift',
    'flycompoundeye': 'flycompoundeye',
    'buzz': 'flycompoundeye',
    'mikaopenclaw': 'mikaopenclaw',
    'mika': 'mikaopenclaw',
}


def detect_entities(content: str, tags: list[str] = None) -> dict[str, list[str]]:
    """
    Auto-detect entities from content and tags.

    Detection patterns:
    - @mentions -> agents
    - Known agent names -> agents (normalized to canonical)
    - Known project names -> projects
    - #hashtags -> concepts
    - Tags that match known entities -> appropriate type

    Returns:
        Dict with entity types as keys: {'agents': [...], 'projects': [...], 'concepts': [...]}
    """
    tags = tags or []
    content_lower = content.lower()
    entities = {
        'agents': set(),
        'projects': set(),
        'concepts': set()
    }

    # @mentions -> agents
    mentions = re.findall(r'@(\w+)', content)
    for mention in mentions:
        mention_lower = mention.lower()
        canonical = _AGENT_ALIASES.get(mention_lower)
        if canonical:
            entities['agents'].add(canonical)
        else:
            entities['agents'].add(mention_lower)

    # Known agents in content
    for agent in KNOWN_AGENTS:
        if agent in content_lower:
            canonical = _AGENT_ALIASES.get(agent, agent)
            entities['agents'].add(canonical)

    # Known projects in content
    for project in KNOWN_PROJECTS:
        if project in content_lower:
            entities['projects'].add(project)

    # #hashtags -> concepts
    hashtags = re.findall(r'#(\w+)', content)
    for tag in hashtags:
        entities['concepts'].add(tag.lower())

    # Tags that look like entity names
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in KNOWN_AGENTS:
            canonical = _AGENT_ALIASES.get(tag_lower, tag_lower)
            entities['agents'].add(canonical)
        elif tag_lower in KNOWN_PROJECTS:
            entities['projects'].add(tag_lower)
        elif tag_lower in ('collaboration', 'milestone', 'memory-system', 'causal-edges'):
            entities['concepts'].add(tag_lower)

    # Convert sets to sorted lists, drop empty types
    return {k: sorted(list(v)) for k, v in entities.items() if v}


def detect_event_time(content: str) -> Optional[str]:
    """
    Auto-detect event_time from content by parsing date references.
    Returns ISO date string (YYYY-MM-DD) or None if no date found.

    Detects:
    - Explicit dates: "2026-01-31", "January 31, 2026", "Jan 31"
    - Relative dates: "yesterday", "last week", "2 days ago"
    - Session references: "this session", "today" (returns today)

    v2.11: Intelligent bi-temporal - memories auto-tagged with event time.
    """
    today = datetime.now(timezone.utc).date()
    content_lower = content.lower()

    # Explicit ISO date (YYYY-MM-DD)
    iso_match = re.search(r'(\d{4}-\d{2}-\d{2})', content)
    if iso_match:
        return iso_match.group(1)

    # Month DD, YYYY or Month DD YYYY
    month_names = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    month_pattern = (
        r'(january|february|march|april|may|june|july|august|september|'
        r'october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|'
        r'nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})?'
    )
    month_match = re.search(month_pattern, content_lower)
    if month_match:
        month = month_names[month_match.group(1)]
        day = int(month_match.group(2))
        year = int(month_match.group(3)) if month_match.group(3) else today.year
        try:
            return f"{year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            pass

    # Relative dates
    if 'yesterday' in content_lower:
        return (today - timedelta(days=1)).isoformat()
    if 'day before yesterday' in content_lower:
        return (today - timedelta(days=2)).isoformat()
    if 'last week' in content_lower:
        return (today - timedelta(weeks=1)).isoformat()
    if 'last month' in content_lower:
        return (today - timedelta(days=30)).isoformat()

    # N days/weeks ago
    ago_match = re.search(r'(\d+)\s+(day|week|month)s?\s+ago', content_lower)
    if ago_match:
        num = int(ago_match.group(1))
        unit = ago_match.group(2)
        if unit == 'day':
            return (today - timedelta(days=num)).isoformat()
        elif unit == 'week':
            return (today - timedelta(weeks=num)).isoformat()
        elif unit == 'month':
            return (today - timedelta(days=num * 30)).isoformat()

    # Today/this session - return today
    if 'today' in content_lower or 'this session' in content_lower:
        return today.isoformat()

    return None
