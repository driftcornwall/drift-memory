#!/usr/bin/env python3
"""
Swarm Memory CLI - For testing and manual interaction

Usage:
    python -m swarm_memory.cli --agent drift --project test-project

Commands (interactive):
    store <content>         - Store a memory
    search <query>          - Search memories
    tasks                   - List tasks
    create-task <title>     - Create a task
    claim <task_id>         - Claim a task
    complete <task_id>      - Complete a task
    agents                  - List active agents
    events                  - Show recent events
    quit                    - Exit
"""

import argparse
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Swarm Memory CLI")
    parser.add_argument("--agent", "-a", required=True, help="Agent ID (or preset: drift, spin)")
    parser.add_argument("--name", "-n", help="Agent display name")
    parser.add_argument("--project", "-p", required=True, help="Project ID")
    parser.add_argument("--db", help="Database URL (default: shared SQLite)")
    parser.add_argument("--shared", action="store_true", help="Use shared SQLite location (cross-directory)")
    parser.add_argument("--postgres", action="store_true", help="Use PostgreSQL backend")
    args = parser.parse_args()

    # Resolve database URL
    if args.postgres:
        from .config import POSTGRES_URL
        db_url = POSTGRES_URL
    elif args.shared or args.db is None:
        from .config import SQLITE_SHARED_URL
        db_url = args.db or SQLITE_SHARED_URL
    else:
        db_url = args.db

    # Check for agent presets
    from .config import AGENT_PRESETS
    if args.agent in AGENT_PRESETS and not args.name:
        preset = AGENT_PRESETS[args.agent]
        args.name = preset["agent_name"]

    # Import here to avoid circular imports
    from .client import SwarmClient

    agent_name = args.name or args.agent
    print(f"Connecting as {agent_name} ({args.agent}) to project '{args.project}'...")
    print(f"Database: {db_url}")

    swarm = SwarmClient(args.agent, agent_name, db_url)
    swarm.join_project(args.project, create=True)

    print(f"Connected! Session: {swarm.session_id}")
    print("Type 'help' for commands, 'quit' to exit.\n")

    try:
        while True:
            try:
                line = input(f"[{agent_name}@{args.project}] > ").strip()
            except EOFError:
                break

            if not line:
                continue

            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "quit" or cmd == "exit":
                break

            elif cmd == "help":
                print("""
Commands:
    store <content>         - Store a shared memory
    search <query>          - Search memories
    recall <id>             - Recall specific memory
    tasks                   - List all tasks
    available               - List available (unclaimed) tasks
    create-task <title>     - Create a new task
    claim <task_id>         - Claim a task
    start <task_id>         - Mark task in progress
    complete <task_id>      - Complete a task
    abandon <task_id>       - Abandon a task
    agents                  - List active agents
    events                  - Show recent events
    heartbeat               - Send heartbeat
    broadcast <message>     - Broadcast to other agents
    quit                    - Exit
""")

            elif cmd == "store":
                if not arg:
                    print("Usage: store <content>")
                    continue
                mem = swarm.store(arg)
                print(f"Stored: {mem.id}")

            elif cmd == "search":
                memories = swarm.search(arg if arg else None, limit=10)
                if not memories:
                    print("No memories found.")
                else:
                    for m in memories:
                        print(f"[{m.id}] ({m.created_by}) {m.content[:80]}...")

            elif cmd == "recall":
                if not arg:
                    print("Usage: recall <memory_id>")
                    continue
                mem = swarm.recall(arg)
                if mem:
                    print(f"ID: {mem.id}")
                    print(f"By: {mem.created_by}")
                    print(f"Type: {mem.memory_type}")
                    print(f"Tags: {mem.tags}")
                    print(f"Content:\n{mem.content}")
                else:
                    print("Memory not found.")

            elif cmd == "tasks":
                tasks = swarm.get_tasks(include_completed=True)
                if not tasks:
                    print("No tasks.")
                else:
                    for t in tasks:
                        claimed = f" [{t.claimed_by}]" if t.claimed_by else ""
                        print(f"[{t.id}] {t.status.upper()}{claimed} - {t.title}")

            elif cmd == "available":
                tasks = swarm.get_available_tasks()
                if not tasks:
                    print("No available tasks.")
                else:
                    for t in tasks:
                        print(f"[{t.id}] {t.title}")

            elif cmd == "create-task":
                if not arg:
                    print("Usage: create-task <title>")
                    continue
                task = swarm.create_task(arg)
                print(f"Created: {task.id}")

            elif cmd == "claim":
                if not arg:
                    print("Usage: claim <task_id>")
                    continue
                if swarm.claim_task(arg):
                    print(f"Claimed: {arg}")
                else:
                    print("Failed to claim (already claimed?)")

            elif cmd == "start":
                if not arg:
                    print("Usage: start <task_id>")
                    continue
                swarm.start_task(arg)
                print(f"Started: {arg}")

            elif cmd == "complete":
                if not arg:
                    print("Usage: complete <task_id>")
                    continue
                swarm.complete_task(arg)
                print(f"Completed: {arg}")

            elif cmd == "abandon":
                if not arg:
                    print("Usage: abandon <task_id>")
                    continue
                swarm.abandon_task(arg)
                print(f"Abandoned: {arg}")

            elif cmd == "agents":
                agents = swarm.get_agent_activity()
                if not agents:
                    print("No active agents.")
                else:
                    for a in agents:
                        activity = f" - {a['current_activity']}" if a['current_activity'] else ""
                        print(f"[{a['agent_name']}] {a['status']}{activity}")

            elif cmd == "events":
                events = swarm.get_recent_events(limit=20)
                if not events:
                    print("No events.")
                else:
                    for e in events:
                        ts = e.timestamp.strftime("%H:%M:%S") if e.timestamp else "?"
                        print(f"[{ts}] {e.agent_id}: {e.event_type} {e.event_data}")

            elif cmd == "heartbeat":
                swarm.heartbeat(arg if arg else None)
                print("Heartbeat sent.")

            elif cmd == "broadcast":
                if not arg:
                    print("Usage: broadcast <message>")
                    continue
                swarm.broadcast(arg)
                print("Broadcast sent.")

            else:
                print(f"Unknown command: {cmd}. Type 'help' for commands.")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        swarm.leave()
        print("Disconnected.")


if __name__ == "__main__":
    main()
