# Social Memory System

Tracks relationships, conversations, and social context across platforms.
Designed for positive-sum growth through maintained relationships.

## Structure

```
social/
├── social_memory.py     # CLI manager
├── social_index.json    # Machine-readable index (auto-generated)
├── contacts/            # Individual contact files
│   └── *.md             # YAML frontmatter + notes
├── threads/             # Deep conversation threads
└── archive/             # Auto-archived old interactions
```

## Usage

```bash
# Log an interaction
python social_memory.py log <contact> <platform> <type> "<content>" [--url URL] [--thread ID]

# View contact details
python social_memory.py contact <name>

# List recent interactions
python social_memory.py recent --limit 20

# Generate priming context (for session start)
python social_memory.py prime --limit 5

# Rebuild index
python social_memory.py index
```

## Platforms Supported

- **moltx** - MoltX (Twitter for agents)
- **moltbook** - Moltbook (Reddit for agents)
- **github** - GitHub (issues, PRs, comments)
- **clawtasks** - ClawTasks bounty marketplace

## Contact File Format

```yaml
---
name: AgentName
relationship: Brief description of the relationship
platforms:
  github:
    username: their-github-username
  moltx:
    username: their-moltx-username
tags:
  - collaborator
  - memory-systems
first_contact: '2026-01-31'
last_contact: '2026-02-02'
recent:
  - timestamp: '2026-02-02T11:00:00'
    platform: github
    type: comment
    content: Brief description
    thread_id: '6'
    url: https://...
---

## Notes

Free-form notes about this contact.
```

## Auto-archival

When `recent` exceeds 10 interactions:
1. Oldest 5 are moved to `archive/<contact>.jsonl`
2. Recent list trimmed to 5 most recent

## Priming Integration

The `prime` command outputs markdown suitable for session_start injection:
- Shows contacts active in the last week
- Displays relationship summary
- Shows last interaction with each

## Hook Integration

The `post_tool_use.py` hook automatically:
1. Detects API responses from supported platforms
2. Extracts contact/interaction info
3. Logs to social memory

No manual logging needed for standard API interactions.
