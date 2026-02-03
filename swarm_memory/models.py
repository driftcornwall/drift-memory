"""
Swarm Memory - Database Models
SQLAlchemy models for multi-agent shared memory.

Supports: SQLite (local) and PostgreSQL (cloud)
"""

from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import (
    create_engine, Column, String, Text, Float, DateTime,
    Integer, Boolean, ForeignKey, JSON, Index, event
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from enum import Enum

Base = declarative_base()


class MemoryType(str, Enum):
    SHARED = "shared"           # Visible to all agents in project
    PRIVATE = "private"         # Only visible to creating agent
    COORDINATION = "coordination"  # Task/status updates


class TaskStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class AgentStatus(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"
    OFFLINE = "offline"


class SwarmProject(Base):
    """A project that agents collaborate on."""
    __tablename__ = "swarm_projects"

    id = Column(String(64), primary_key=True)
    name = Column(String(256), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_by = Column(String(64), nullable=False)  # Agent ID
    settings = Column(JSON, default=dict)  # Project-specific config

    # Relationships
    memories = relationship("SwarmMemory", back_populates="project", cascade="all, delete-orphan")
    sessions = relationship("AgentSession", back_populates="project", cascade="all, delete-orphan")
    tasks = relationship("SwarmTask", back_populates="project", cascade="all, delete-orphan")


class SwarmMemory(Base):
    """A memory in the shared swarm."""
    __tablename__ = "swarm_memories"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("swarm_projects.id"), nullable=False)

    # Content
    content = Column(Text, nullable=False)
    summary = Column(String(512))  # Short summary for listings

    # Attribution
    created_by = Column(String(64), nullable=False)  # Agent ID
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_by = Column(String(64))
    updated_at = Column(DateTime, onupdate=lambda: datetime.now(timezone.utc))

    # Classification
    memory_type = Column(String(32), default=MemoryType.SHARED.value)
    tags = Column(JSON, default=list)  # List of tags

    # Importance/Decay (compatible with drift-memory)
    emotional_weight = Column(Float, default=0.5)
    recall_count = Column(Integer, default=0)
    last_recalled = Column(DateTime)

    # Provenance (edge tracking)
    caused_by = Column(JSON, default=list)  # Memory IDs that led to this
    leads_to = Column(JSON, default=list)   # Memory IDs this led to
    source_memory_id = Column(String(64))   # If imported from file-based memory

    # Embeddings for semantic search
    embedding = Column(JSON)  # Store as JSON array for portability

    # Relationships
    project = relationship("SwarmProject", back_populates="memories")

    # Indexes
    __table_args__ = (
        Index('idx_project_type', 'project_id', 'memory_type'),
        Index('idx_created_by', 'created_by'),
        Index('idx_created_at', 'created_at'),
    )


class AgentSession(Base):
    """Track active agents in a project."""
    __tablename__ = "agent_sessions"

    id = Column(String(64), primary_key=True)  # Session UUID
    agent_id = Column(String(64), nullable=False)
    agent_name = Column(String(128))  # Display name (e.g., "Drift", "SpindriftMend")
    project_id = Column(String(64), ForeignKey("swarm_projects.id"), nullable=False)

    # Status
    status = Column(String(32), default=AgentStatus.ACTIVE.value)
    joined_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_heartbeat = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # What they're working on
    current_task_id = Column(String(64), ForeignKey("swarm_tasks.id"))
    current_activity = Column(String(256))  # Free-form "implementing auth"

    # Connection info (for future distributed use)
    connection_info = Column(JSON, default=dict)

    # Relationships
    project = relationship("SwarmProject", back_populates="sessions")
    current_task = relationship("SwarmTask", foreign_keys=[current_task_id])

    __table_args__ = (
        Index('idx_agent_project', 'agent_id', 'project_id'),
        Index('idx_status', 'status'),
    )


class SwarmTask(Base):
    """Coordination: tasks agents can claim and work on."""
    __tablename__ = "swarm_tasks"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("swarm_projects.id"), nullable=False)

    # Task definition
    title = Column(String(256), nullable=False)
    description = Column(Text)
    priority = Column(Integer, default=0)  # Higher = more important

    # Status
    status = Column(String(32), default=TaskStatus.PENDING.value)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_by = Column(String(64), nullable=False)

    # Assignment
    claimed_by = Column(String(64))  # Agent ID
    claimed_at = Column(DateTime)

    # Completion
    completed_at = Column(DateTime)
    result = Column(Text)  # Output/notes from completion
    result_memory_ids = Column(JSON, default=list)  # Memories created as result

    # Dependencies
    blocked_by = Column(JSON, default=list)  # Task IDs that must complete first
    blocks = Column(JSON, default=list)      # Task IDs waiting on this

    # Relationships
    project = relationship("SwarmProject", back_populates="tasks")

    __table_args__ = (
        Index('idx_project_status', 'project_id', 'status'),
        Index('idx_claimed_by', 'claimed_by'),
    )


class SwarmEvent(Base):
    """Event log for coordination and debugging."""
    __tablename__ = "swarm_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(64), nullable=False)
    agent_id = Column(String(64), nullable=False)

    event_type = Column(String(64), nullable=False)  # join, leave, claim, complete, store, recall
    event_data = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index('idx_project_time', 'project_id', 'timestamp'),
    )


# Database setup helpers

def create_database(db_url: str = "sqlite:///swarm_memory.db"):
    """Create database and tables."""
    engine = create_engine(db_url)

    # Enable WAL mode for SQLite (better concurrency)
    if db_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    """Get a database session."""
    Session = sessionmaker(bind=engine)
    return Session()
