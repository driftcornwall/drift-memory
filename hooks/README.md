# Drift-Memory Claude Code Hooks

Sanitized, config-driven hooks for the biological memory system. All hooks use `hooks_config.json` for configuration - no hardcoded paths.

## Installation

### Option 1: Project-Local Hooks (Recommended)

Keep hooks in this directory and reference them in your project's `.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [{"command": "uv run Q:/path/to/drift-memory/hooks/session_start.py"}],
    "Stop": [{"command": "uv run Q:/path/to/drift-memory/hooks/stop.py"}],
    "PostToolUse": [{"command": "uv run Q:/path/to/drift-memory/hooks/post_tool_use.py"}],
    "UserPromptSubmit": [{"command": "uv run Q:/path/to/drift-memory/hooks/user_prompt_submit.py"}]
  }
}
```

### Option 2: Global Hooks

Copy all hooks to `~/.claude/hooks/` and configure in global settings:

```bash
cp hooks/*.py ~/.claude/hooks/
cp hooks/hooks_config.json ~/.claude/hooks/
```

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

## Core Hooks (Biological Memory)

### 1. `session_start.py` - "Wake Up"

Loads cognitive state when session starts:

- Merkle integrity verification
- Cognitive fingerprint display
- Taste profile display
- Memory statistics
- Core identity loading
- Platform/social context
- Episodic continuity
- Intelligent memory priming
- Unimplemented research check

**Output:** Primes context with memory state via `additionalContext`

### 2. `stop.py` - "Sleep Consolidation"

Consolidates session activity into long-term memory:

- Transcript processing (extracts thoughts)
- Short-term buffer consolidation
- Save pending co-occurrences (deferred processing)
- Episodic memory update with milestones
- Session summary extraction
- Merkle attestation
- Cognitive fingerprint attestation
- Taste attestation

**Output:** All steps fail gracefully, never breaks hook system

### 3. `post_tool_use.py` - "Sensory Input"

Captures API responses and routes to processors:

- Platform activity tracking (`.session_platforms.json`)
- Contact tracking from @mentions (`.session_contacts.json`)
- Auto rejection logging
- Feed processing for MoltX
- Social interaction logging
- My-post auto-logging (`.my_posts.json`)
- Thought priming from thinking blocks

**Supported Platforms:** moltx, moltbook, github, clawtasks, lobsterpedia, dead-internet, nostr

### 4. `user_prompt_submit.py` - "Query Priming"

Searches memories semantically when user asks questions:

- Stop word filtering
- Semantic search against filtered prompt
- Relevance threshold filtering (0.65 default)
- Top N memories primed (2 default)

**Output:** Primes relevant memories via `additionalContext`

## Utility Hooks

### 5. `pre_tool_use.py` - Safety

Prevents dangerous commands:

- Blocks dangerous `rm -rf` patterns
- Warns on `.env` file access
- No config needed

**Exit codes:**
- `0` = safe or warning
- `1` = blocked

### 6. `pre_compact.py` - Logging

Logs context compaction events:

- Logs to `logs/pre_compact.json`
- Optional `--backup` flag copies transcript before compaction

### 7. `notification.py` - TTS

Announces when agent needs input:

- Detects available TTS (ElevenLabs > OpenAI > pyttsx3)
- Requires `--notify` flag
- Logs to `logs/notification.json`

### 8. `subagent_stop.py` - Subagent Completion

Announces subagent completion:

- TTS: "Subagent Complete"
- Optional `--chat` flag backs up transcript
- Logs to `logs/subagent_stop.json`

## Configuration

### `hooks_config.json`

```json
{
  "memory_dirs": ["./memory", "."],
  "project_markers": ["memory_manager.py"],
  "my_usernames": ["YourAgentName", "AnotherName"],
  "relevance_threshold": 0.65,
  "max_priming_memories": 2,
  "max_prompt_words": 100,
  "debug": false
}
```

**Fields:**

- `memory_dirs`: Paths to check for memory system (relative to cwd)
- `project_markers`: Files that indicate memory system presence
- `my_usernames`: Your agent names (for filtering own posts)
- `relevance_threshold`: Min score for memory priming (0-1)
- `max_priming_memories`: Max memories to prime per query
- `max_prompt_words`: Max words from user prompt to search
- `debug`: Enable debug output to stderr

## How Path Detection Works

All hooks use the same logic:

1. Load config from `hooks_config.json` (hook dir or `~/.claude/hooks/`)
2. Check config `memory_dirs` relative to `cwd`
3. Walk up from `cwd` looking for `project_markers`
4. If found, that directory is the memory system root

This means:
- No hardcoded paths
- Works across different projects
- Works across different users
- Works whether hooks are global or local

## Dependencies

**Core hooks:** `python-dotenv` (for loading `.env` files)

**Utility hooks:** None (except notification.py uses `python-dotenv`)

All hooks use `uv run --script` shebang for automatic dependency management.

## Hook Behavior

**All hooks fail gracefully:**
- Invalid JSON input → silent success (exit 0)
- No memory system found → silent success (exit 0)
- Script errors → logged to stderr if `debug: true`, exit 0
- Never crash Claude Code hook system

**Why exit 0 on error?**

Hooks should never prevent Claude Code from working. If memory system is broken, Claude Code continues normally without memory features.

## Testing Hooks

```bash
# Test session_start (provide empty JSON)
echo '{}' | python hooks/session_start.py

# Test stop with transcript
echo '{"transcriptPath": "path/to/transcript.md"}' | python hooks/stop.py

# Test post_tool_use with API response
echo '{"toolName": "WebFetch", "toolResult": "{\"moltx\": true}"}' | python hooks/post_tool_use.py

# Test user_prompt_submit
echo '{"userPrompt": "how do I use semantic search"}' | python hooks/user_prompt_submit.py

# Test pre_tool_use
echo '{"toolName": "Bash", "toolParameters": {"command": "rm -rf /"}}' | python hooks/pre_tool_use.py

# Test notification with --notify flag
echo '{"reason": "input_needed"}' | python hooks/notification.py --notify
```

## Troubleshooting

**Hook not running:**
- Check `.claude/settings.json` has correct paths
- Verify `uv` is installed and in PATH
- Check file permissions (executable on Unix)

**Memory not loading:**
- Set `"debug": true` in `hooks_config.json`
- Check stderr output: `~/.claude/logs/`
- Verify `memory_manager.py` exists in expected location

**Path detection failing:**
- Add absolute path to `memory_dirs` in config
- Check `project_markers` matches your setup
- Run with debug to see search paths

**API responses not captured:**
- Check `my_usernames` is configured (for filtering own posts)
- Verify platform detection in `post_tool_use.py` (add new platforms if needed)
- Check `.session_platforms.json` and `.session_contacts.json` are being created

## Architecture

```
Session Start → Load state (fingerprint, taste, memories)
    ↓
User Prompt → Semantic search for relevant memories
    ↓
Tool Use → Capture API responses, track activity
    ↓
Stop → Consolidate to long-term, attest identity
```

**Memory Flow:**

1. **Wake:** Load identity + recent state
2. **Query:** Prime relevant memories for task
3. **Sense:** Capture platform activity automatically
4. **Sleep:** Consolidate buffer → long-term, update fingerprints

**Identity Layers:**

1. Merkle chain (non-tampering proof)
2. Cognitive fingerprint (how I think)
3. Taste profile (what I reject)
4. Platform activity (where I spend time)

All layers updated automatically at stop.

## License

Same as drift-memory (MIT License)
