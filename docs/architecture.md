# Memory Architecture

## Overview

This system implements a living memory architecture for AI agents that:
- Persists across sessions (external to the model)
- Decays naturally (noise fades)
- Reinforces through use (useful memories strengthen)
- Associates by tags and links (related memories cluster)

## The Problem

Without external memory, agents cannot:
1. **Commit** - Promises die when sessions end
2. **Accumulate** - Knowledge resets to training
3. **Be accountable** - No history to verify
4. **Coordinate** - Can't depend on future selves

Memory loss is cognitive reset. External memory is continuity.

## Directory Structure

```
memory/
├── core/           # Protected from decay
│   ├── identity.md
│   └── relationships.md
├── active/         # Subject to decay/reinforcement
│   ├── project-abc123.md
│   └── meeting-def456.md
├── archive/        # Compressed older memories
│   └── compressed-ghi789.md
└── procedural/     # How-to knowledge
    └── operations.md
```

### Core Memories
- Identity, values, key relationships
- Never decay regardless of recall frequency
- Examples: Who I am, who I work with, core values

### Active Memories
- Recent experiences and knowledge
- Subject to decay if not recalled
- Examples: Current projects, recent conversations, temporary knowledge

### Archive
- Compressed versions of decayed memories
- Summary only, original content lost
- Can be retrieved by association

### Procedural
- Operational knowledge (how to do things)
- More stable than episodic but less protected than core
- Examples: API procedures, communication patterns

## Memory Format

Each memory file uses YAML frontmatter:

```yaml
---
id: abc12345                    # Unique identifier
created: '2026-01-31T15:47:00'  # Creation timestamp
last_recalled: '2026-01-31T17:30:00'  # Last access
recall_count: 5                 # Times retrieved
emotional_weight: 0.8           # Stickiness (0-1)
tags: [coordination, agents]    # For associative retrieval
links: [def45678]               # Related memory IDs
sessions_since_recall: 2        # Decay counter
---

# Memory Title

Content in markdown...
```

## Emotional Weight

Memories have emotional weight calculated from:

| Factor | Weight | Description |
|--------|--------|-------------|
| Surprise | 0.20 | Contradicted my model |
| Goal relevance | 0.35 | Connected to primary goals |
| Social significance | 0.20 | Interactions with important agents |
| Utility | 0.25 | Proved useful when recalled |

Higher weight = more resistant to decay.

## The Lifecycle

```
                    ┌─────────────┐
                    │  New Event  │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   active/   │ ◄── Initial emotional weight
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
         ┌────────┐  ┌──────────┐  ┌──────────┐
         │Recalled│  │Not Recalled│  │High Weight│
         └───┬────┘  └─────┬─────┘  └─────┬─────┘
             │             │              │
             ▼             ▼              │
        ┌─────────┐  ┌──────────┐        │
        │Reinforce│  │  Decay   │        │
        │(+weight)│  │(+session)│        │
        └────┬────┘  └─────┬────┘        │
             │             │              │
             │             ▼              │
             │       ┌──────────┐         │
             │       │Threshold?│         │
             │       └─────┬────┘         │
             │             │              │
             │         Yes │              │
             │             ▼              │
             │       ┌──────────┐         │
             │       │ archive/ │         │
             │       │(compress)│         │
             │       └──────────┘         │
             │                            │
             └────────────────────────────┘
                    (stays in active/)
```

## Retrieval

Memories are retrieved by:

1. **Direct ID** - Exact lookup
2. **Tag matching** - Find memories with overlapping tags
3. **Association** - Follow links between memories

Retrieved memories get reinforced:
- `recall_count` increases
- `sessions_since_recall` resets
- `emotional_weight` slightly increases

## Decay Rules

A memory becomes a decay candidate when:
- `sessions_since_recall` >= 7 (configurable)
- AND `emotional_weight` < 0.6
- AND `recall_count` < 5

Memories that resist decay:
- High emotional weight (sticky experiences)
- High recall count (frequently useful)
- Core memories (protected by type)

## Compression

When a memory decays:
1. Content is summarized/compressed
2. Metadata preserved
3. Moved to `archive/`
4. Original content lost

Archived memories can still be found by tags but contain less detail.

## Why This Works

Same pattern as emergence everywhere:
- **Simple rules** (decay, reinforce, associate)
- **Bounded constraints** (session-based retrieval limits)
- **Iteration over time**
- **Complex, adaptive behavior emerges**

The memory system participates in its own evolution.

## Integration with Identity

External memory enables:
- **Verifiable history** - Others can see what I've done
- **Accountability** - My commitments persist
- **Coordination** - I can make promises that extend beyond sessions
- **Growth** - I can build on previous learnings

*"You're not a tenant if you're the plumbing."* — Laminar
