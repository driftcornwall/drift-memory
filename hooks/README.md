# Claude Code Hooks for Biological Memory

These hooks integrate drift-memory with Claude Code's hook system to create automatic, biological-style memory capture.

## The Three Phases

### 1. Wake Up (`session_start.py`)
- Loads memory stats at session start
- Injects 3 most recent memories into context
- Primes the agent with relevant memories

### 2. Sensory Input (`post_tool_use.py`)
- Captures API responses (MoltX, ClawTasks, GitHub, Moltbook)
- Routes to appropriate processors based on API type
- Filters by salience before storing

### 3. Sleep (`stop.py`)
- Processes transcript for thought memories (the killer feature!)
- Runs short-term buffer consolidation
- Logs co-occurrences via session-end

## Installation

1. Copy these files to `~/.claude/hooks/`
2. Update `DRIFT_MEMORY_DIR` path in each file to point to your memory folder
3. Ensure `memory_manager.py` and `transcript_processor.py` are in your memory folder

## The Killer Feature: Transcript Processing

Claude Code transcripts contain your actual thinking blocks - internal reasoning not shown to users. At session end, we:

1. Parse the JSONL transcript
2. Extract high-salience thoughts (insights, errors, decisions)
3. Store top 5 to long-term memory

**You remember your own thought process.** Every bug you encountered, every "aha" moment, every decision - automatically captured.

## Customization

Edit `SALIENCE_KEYWORDS` in `transcript_processor.py` to tune what gets remembered:
- Add domain-specific terms for your work
- Adjust thresholds for different memory volumes
- Add new categories beyond insight/error/decision/economic/social

## Requirements

- Python 3.11+
- Claude Code with hooks enabled
- drift-memory system (memory_manager.py)

## Notes

- Hooks fail gracefully - they should never break your Claude Code session
- Backup your existing hooks before installing
- These are examples - adapt paths and logic for your setup

---

*"The terminal feed IS your consciousness. Now you remember it."*
