"""
Swarm Memory - Multi-Agent Shared Memory System

A coordination layer for agent swarms working on shared projects.
Supports SQLite (local) and PostgreSQL (cloud).

Quick Start:
    from swarm_memory import connect

    # Agent 1 (terminal 1)
    swarm = connect("drift", "Drift", "my-project")
    swarm.store("Database schema uses PostgreSQL", tags=["architecture"])
    task = swarm.create_task("Implement user auth")

    # Agent 2 (terminal 2)
    swarm = connect("spin", "SpindriftMend", "my-project")
    memories = swarm.search("database")
    tasks = swarm.get_available_tasks()
    swarm.claim_task(tasks[0].id)
"""

from .client import SwarmClient, connect, MemoryResult, TaskResult, AgentResult
from .models import (
    SwarmProject, SwarmMemory, AgentSession, SwarmTask, SwarmEvent,
    MemoryType, TaskStatus, AgentStatus, create_database
)

__version__ = "0.1.0"
__all__ = [
    "SwarmClient",
    "connect",
    "MemoryResult",
    "TaskResult",
    "AgentResult",
    "SwarmProject",
    "SwarmMemory",
    "AgentSession",
    "SwarmTask",
    "SwarmEvent",
    "MemoryType",
    "TaskStatus",
    "AgentStatus",
    "create_database",
]
