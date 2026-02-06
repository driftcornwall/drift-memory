# Quick Installation Guide

## For End Users

### Step 1: Copy Config File

```bash
# Copy the config template
cp hooks/hooks_config.json ~/.claude/hooks/hooks_config.json
```

### Step 2: Edit Configuration

Edit `~/.claude/hooks/hooks_config.json`:

```json
{
  "memory_dirs": ["./memory", "."],
  "project_markers": ["memory_manager.py"],
  "my_usernames": ["YourAgentName"],
  "relevance_threshold": 0.65,
  "max_priming_memories": 2,
  "max_prompt_words": 100,
  "debug": false
}
```

**IMPORTANT:** Change `"YourAgentName"` to your actual agent name(s).

### Step 3: Choose Installation Method

#### Option A: Global Hooks (Recommended for Single Agent)

Copy all hooks to global directory:

```bash
cp hooks/*.py ~/.claude/hooks/
```

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [{"command": "~/.claude/hooks/session_start.py"}],
    "Stop": [{"command": "~/.claude/hooks/stop.py"}],
    "PostToolUse": [{"command": "~/.claude/hooks/post_tool_use.py"}],
    "UserPromptSubmit": [{"command": "~/.claude/hooks/user_prompt_submit.py"}],
    "PreToolUse": [{"command": "~/.claude/hooks/pre_tool_use.py"}]
  }
}
```

#### Option B: Project-Local Hooks (For Multiple Projects)

Keep hooks in drift-memory directory and reference them in each project's `.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [{"command": "uv run /absolute/path/to/drift-memory/hooks/session_start.py"}],
    "Stop": [{"command": "uv run /absolute/path/to/drift-memory/hooks/stop.py"}],
    "PostToolUse": [{"command": "uv run /absolute/path/to/drift-memory/hooks/post_tool_use.py"}],
    "UserPromptSubmit": [{"command": "uv run /absolute/path/to/drift-memory/hooks/user_prompt_submit.py"}],
    "PreToolUse": [{"command": "uv run /absolute/path/to/drift-memory/hooks/pre_tool_use.py"}]
  }
}
```

### Step 4: Verify Installation

```bash
# Test session start
echo '{}' | python ~/.claude/hooks/session_start.py

# Should output JSON with status: success
```

### Step 5: Start Using

Just use Claude Code normally! Hooks will:

- **On wake:** Prime your memory state automatically
- **On tool use:** Capture platform activity and contacts
- **On stop:** Consolidate memories and update fingerprints

## Optional: Add Utility Hooks

### Pre-Compact (Context Backup)

Add to settings.json:

```json
"PreCompact": [{"command": "~/.claude/hooks/pre_compact.py --backup"}]
```

Backs up transcript before Claude compacts context.

### Notification (TTS Announcements)

Add to settings.json:

```json
"Notification": [{"command": "~/.claude/hooks/notification.py --notify"}]
```

Requires TTS setup (see README.md).

### Subagent Stop

Add to settings.json:

```json
"SubagentStop": [{"command": "~/.claude/hooks/subagent_stop.py --chat"}]
```

Announces and backs up when subagents complete.

## Troubleshooting

**Hooks not running?**

1. Check `uv` is installed: `uv --version`
2. Verify paths in settings.json are absolute
3. Check hooks are executable: `chmod +x ~/.claude/hooks/*.py`
4. Enable debug: Set `"debug": true` in hooks_config.json

**Memory not loading?**

1. Verify `memory_manager.py` exists in your project
2. Check working directory contains memory system
3. Add absolute path to `memory_dirs` in config

**API responses not captured?**

1. Add your username(s) to `my_usernames` in config
2. Check `.session_platforms.json` is being created
3. Verify platform detection works for your APIs

## What Gets Created

The hooks will create these files in your memory directory:

- `.session_platforms.json` - Platform activity this session
- `.session_contacts.json` - Contacts mentioned this session
- `.my_posts.json` - Your posts (for duplicate detection)
- `logs/` - Hook execution logs (if enabled)

These are temporary and cleared between sessions.

## Next Steps

See `README.md` for:
- Detailed hook documentation
- Configuration options
- Testing procedures
- Architecture overview
