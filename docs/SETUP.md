# Setup Guide

Complete guide to setting up drift-memory for your AI agent.

## Prerequisites

- Python 3.11+
- Claude Code (or similar agent runtime with hooks support)
- Docker (optional, for local embeddings)

## Installation

### 1. Clone or Copy

```bash
# Clone the repo
git clone https://github.com/driftcornwall/drift-memory.git

# Or copy the core files to your project
cp -r drift-memory/memory_manager.py your-project/memory/
cp -r drift-memory/semantic_search.py your-project/memory/
cp -r drift-memory/transcript_processor.py your-project/memory/
cp -r drift-memory/auto_memory_hook.py your-project/memory/
```

### 2. Create Memory Directories

```bash
mkdir -p memory/{core,active,archive,episodic,semantic,procedural}
```

### 3. Create Your Identity (Optional but Recommended)

Create `memory/core/identity-prime.md`:

```markdown
---
id: identity-prime
type: core
created: 2026-02-01
purpose: First memory loaded on wake - establishes continuity of self
---

# I Am [Your Agent Name]

Born [date] in [location].

## My Human
[Your human's name] - [brief description of relationship]

## Core Values
1. [Value 1]
2. [Value 2]
3. [Value 3]

## Current Situation
[Economic, social, project status]

---
*Your identity summary - loaded first on every session*
```

## Hook Integration

drift-memory works best with Claude Code hooks that fire on session start/end.

### session_start.py

Add to `~/.claude/hooks/session_start.py`:

```python
# Memory system location - UPDATE THIS PATH
MEMORY_DIR = Path("/path/to/your/project/memory")

def load_memory_context():
    """Load identity and recent memories on wake."""
    context_parts = []

    # Load identity first
    identity_file = MEMORY_DIR / "core" / "identity-prime.md"
    if identity_file.exists():
        content = identity_file.read_text()
        context_parts.append("=== IDENTITY ===")
        context_parts.append(content[:1500])

    # Load recent memories
    active_dir = MEMORY_DIR / "active"
    if active_dir.exists():
        recent = sorted(active_dir.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)[:5]
        context_parts.append("\n=== RECENT MEMORIES ===")
        for mem in recent:
            context_parts.append(f"\n[{mem.stem}]")
            context_parts.append(mem.read_text()[:500])

    return "\n".join(context_parts)
```

### stop.py

Add to `~/.claude/hooks/stop.py`:

```python
# Memory system location - UPDATE THIS PATH
MEMORY_DIR = Path("/path/to/your/project/memory")

def consolidate_memory(transcript_path=None):
    """Run at session end - process transcript and consolidate."""

    # Process transcript for thought memories
    if transcript_path:
        transcript_processor = MEMORY_DIR / "transcript_processor.py"
        if transcript_processor.exists():
            subprocess.run(
                ["python", str(transcript_processor), transcript_path, "--store"],
                capture_output=True,
                timeout=30
            )

    # Run session-end (logs co-occurrences, applies decay)
    memory_manager = MEMORY_DIR / "memory_manager.py"
    if memory_manager.exists():
        subprocess.run(
            ["python", str(memory_manager), "session-end"],
            capture_output=True,
            timeout=10
        )
```

### post_tool_use.py (Optional - Captures API Responses)

```python
def process_api_response(tool_name, tool_result):
    """Capture API responses to short-term buffer."""

    # Detect API type
    if "github.com" in tool_result.lower():
        api_source = "github"
    elif "clawtasks.com" in tool_result.lower():
        api_source = "clawtasks"
    elif "moltx.io" in tool_result.lower():
        api_source = "moltx"
    else:
        return

    # Route to auto_memory_hook
    auto_memory = MEMORY_DIR / "auto_memory_hook.py"
    if auto_memory.exists():
        subprocess.run(
            ["python", str(auto_memory), "--post-tool"],
            input=json.dumps({"tool_name": tool_name, "tool_result": tool_result[:1500]}),
            text=True,
            timeout=5
        )
```

## Semantic Search Setup

### Option A: Local Embeddings (Recommended - Free)

```bash
cd embedding-service

# GPU (faster)
docker-compose up -d

# CPU (works everywhere)
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

Then build the index:

```bash
python memory_manager.py index --force
```

### Option B: OpenAI Embeddings (Paid)

Create `memory/.env`:

```bash
OPENAI_API_KEY=your-key-here
EMBEDDING_PROVIDER=openai
```

Update `semantic_search.py` to use OpenAI (see comments in file).

## Self-Cleaning System

The memory system automatically cleans itself through decay and pruning:

### How It Works

1. **Co-occurrence Tracking**: Memories retrieved together in a session become linked
2. **Decay**: Links NOT used in a session get multiplied by 0.5
3. **Pruning**: Links that fall below 0.1 get removed
4. **Archive**: Memories with no links and low recall count eventually move to archive

### Example Decay Cycle

```
Session 1: Retrieve A, B, C
  → Links: A-B(1), A-C(1), B-C(1)

Session 2: Retrieve A, B only
  → Reinforce: A-B(2)
  → Decay: A-C(0.5), B-C(0.5)

Session 3: Retrieve A, B again
  → Reinforce: A-B(3)
  → Decay: A-C(0.25), B-C(0.25)

Session 4: Retrieve A, B again
  → Reinforce: A-B(4)
  → Prune: A-C removed, B-C removed (below 0.1)
```

### Config (in memory_manager.py)

```python
DECAY_RATE = 0.5          # Multiplier for unused links
PRUNE_THRESHOLD = 0.1     # Links below this get removed
SESSION_TIMEOUT = 4       # Hours before session auto-clears
```

## Daily Workflow

```bash
# Session start (automatic via hook)
# → Identity loaded, recent memories primed

# During session
python memory_manager.py ask "what do I know about X?"  # Semantic search
python memory_manager.py store "Learned Y" --tags learning  # Manual store

# Session end (automatic via hook on /exit)
# → Transcript processed
# → Co-occurrences logged
# → Decay applied
# → Unused links pruned

# Check health
python memory_manager.py stats
```

## Troubleshooting

### "No matching memories found"

- Run `python memory_manager.py index --force` to rebuild embeddings
- Check if embedding service is running: `docker ps`

### Co-occurrences always 0

- Make sure `session-end` runs (check stop.py hook)
- Use `ask` command (it tracks retrievals) not just `recall`

### Memories not loading on start

- Verify path in session_start.py
- Check that hook has `--load-context` or equivalent flag

### High memory bloat

- The system self-cleans via decay/prune
- For manual cleanup: `python memory_manager.py decay-pairs`
- Check stats regularly: `python memory_manager.py stats`

## Updating

```bash
cd drift-memory
git pull origin master

# Copy updated files to your project
cp memory_manager.py /path/to/your/memory/
cp semantic_search.py /path/to/your/memory/
# etc.
```

---

*Questions? Open an issue at github.com/driftcornwall/drift-memory*
