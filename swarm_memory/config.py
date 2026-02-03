"""
Swarm Memory Configuration

Shared config for agents connecting from different locations.
"""

import os
from pathlib import Path

# Default shared database locations
# Agents from any directory can use these

# Option 1: Shared SQLite file (local multi-terminal)
SQLITE_SHARED_PATH = Path("Q:/Codings/ClaudeCodeProjects/LEX/swarm_memory.db")
SQLITE_SHARED_URL = f"sqlite:///{SQLITE_SHARED_PATH}"

# Option 2: PostgreSQL (local or cloud)
# Update these for your setup
POSTGRES_HOST = os.environ.get("SWARM_PG_HOST", "localhost")
POSTGRES_PORT = os.environ.get("SWARM_PG_PORT", "5432")
POSTGRES_USER = os.environ.get("SWARM_PG_USER", "postgres")
POSTGRES_PASS = os.environ.get("SWARM_PG_PASS", "")
POSTGRES_DB = os.environ.get("SWARM_PG_DB", "swarm_memory")

POSTGRES_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASS}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


def get_db_url(backend: str = "sqlite") -> str:
    """
    Get database URL for the specified backend.

    Args:
        backend: "sqlite" or "postgres"

    Returns:
        Database connection URL
    """
    if backend == "postgres":
        return POSTGRES_URL
    return SQLITE_SHARED_URL


# Agent identity presets
AGENT_PRESETS = {
    "drift": {
        "agent_id": "drift",
        "agent_name": "Drift",
    },
    "spin": {
        "agent_id": "spin",
        "agent_name": "SpindriftMend",
    },
}
