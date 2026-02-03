"""
Swarm Memory Client
Agent-facing API for connecting to and using shared memory.

Usage:
    from swarm_memory.client import SwarmClient

    # Connect to swarm
    swarm = SwarmClient(
        agent_id="drift",
        agent_name="Drift",
        db_url="sqlite:///swarm_memory.db"
    )

    # Join a project
    swarm.join_project("build-webapp", create=True)

    # Store shared memory
    swarm.store("Authentication uses JWT tokens", tags=["auth", "architecture"])

    # Recall memories
    memories = swarm.search("how does auth work")

    # Coordinate tasks
    task = swarm.create_task("Implement login endpoint")
    swarm.claim_task(task.id)
    swarm.complete_task(task.id, result="Implemented in /api/login")

    # See what others are doing
    agents = swarm.get_active_agents()
    tasks = swarm.get_tasks(status="in_progress")

    # Heartbeat (call periodically)
    swarm.heartbeat()

    # Leave when done
    swarm.leave()
"""

import uuid
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from sqlalchemy import create_engine, event, or_, and_
from sqlalchemy.orm import sessionmaker, Session

from .models import (
    Base, SwarmProject, SwarmMemory, AgentSession, SwarmTask, SwarmEvent,
    MemoryType, TaskStatus, AgentStatus, create_database
)


# Data classes for returning results (avoids detached instance issues)
from dataclasses import dataclass, field


@dataclass
class MemoryResult:
    id: str
    content: str
    summary: str
    created_by: str
    created_at: datetime
    memory_type: str
    tags: List[str]
    emotional_weight: float
    recall_count: int
    source_memory_id: Optional[str] = None

    @classmethod
    def from_orm(cls, m: SwarmMemory) -> "MemoryResult":
        return cls(
            id=m.id,
            content=m.content,
            summary=m.summary,
            created_by=m.created_by,
            created_at=m.created_at,
            memory_type=m.memory_type,
            tags=m.tags or [],
            emotional_weight=m.emotional_weight,
            recall_count=m.recall_count,
            source_memory_id=m.source_memory_id
        )


@dataclass
class TaskResult:
    id: str
    title: str
    description: Optional[str]
    status: str
    priority: int
    created_by: str
    created_at: datetime
    claimed_by: Optional[str]
    claimed_at: Optional[datetime]
    completed_at: Optional[datetime]
    result: Optional[str]
    blocked_by: List[str] = field(default_factory=list)

    @classmethod
    def from_orm(cls, t: SwarmTask) -> "TaskResult":
        return cls(
            id=t.id,
            title=t.title,
            description=t.description,
            status=t.status,
            priority=t.priority,
            created_by=t.created_by,
            created_at=t.created_at,
            claimed_by=t.claimed_by,
            claimed_at=t.claimed_at,
            completed_at=t.completed_at,
            result=t.result,
            blocked_by=t.blocked_by or []
        )


@dataclass
class AgentResult:
    agent_id: str
    agent_name: str
    status: str
    current_task_id: Optional[str]
    current_activity: Optional[str]
    last_heartbeat: Optional[datetime]

    @classmethod
    def from_orm(cls, a: AgentSession) -> "AgentResult":
        return cls(
            agent_id=a.agent_id,
            agent_name=a.agent_name,
            status=a.status,
            current_task_id=a.current_task_id,
            current_activity=a.current_activity,
            last_heartbeat=a.last_heartbeat
        )


class SwarmClient:
    """Client for agents to interact with swarm memory."""

    def __init__(
        self,
        agent_id: str,
        agent_name: str = None,
        db_url: str = "sqlite:///swarm_memory.db",
        auto_heartbeat: bool = True
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name or agent_id
        self.db_url = db_url
        self.auto_heartbeat = auto_heartbeat

        # Current state
        self.project_id: Optional[str] = None
        self.session_id: Optional[str] = None

        # Setup database
        self.engine = create_database(db_url)
        self.SessionMaker = sessionmaker(bind=self.engine)

    @contextmanager
    def _session(self) -> Session:
        """Context manager for database sessions."""
        session = self.SessionMaker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _log_event(self, session: Session, event_type: str, data: dict = None):
        """Log an event to the swarm event log."""
        if not self.project_id:
            return
        event = SwarmEvent(
            project_id=self.project_id,
            agent_id=self.agent_id,
            event_type=event_type,
            event_data=data or {}
        )
        session.add(event)

    def _generate_id(self, prefix: str = "") -> str:
        """Generate a unique ID."""
        unique = f"{self.agent_id}-{datetime.now().isoformat()}-{uuid.uuid4().hex[:8]}"
        short_hash = hashlib.sha256(unique.encode()).hexdigest()[:12]
        return f"{prefix}{short_hash}" if prefix else short_hash

    # ==================== PROJECT MANAGEMENT ====================

    def join_project(
        self,
        project_id: str,
        create: bool = False,
        project_name: str = None,
        description: str = None
    ) -> SwarmProject:
        """
        Join a project. Creates it if create=True and it doesn't exist.
        Returns the project.
        """
        with self._session() as session:
            project = session.query(SwarmProject).filter_by(id=project_id).first()

            if not project:
                if not create:
                    raise ValueError(f"Project '{project_id}' does not exist. Use create=True to create it.")
                project = SwarmProject(
                    id=project_id,
                    name=project_name or project_id,
                    description=description,
                    created_by=self.agent_id
                )
                session.add(project)
                session.flush()

            # Create session
            self.session_id = self._generate_id("sess-")
            agent_session = AgentSession(
                id=self.session_id,
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                project_id=project_id,
                status=AgentStatus.ACTIVE.value
            )
            session.add(agent_session)

            self.project_id = project_id
            self._log_event(session, "join", {"project_id": project_id})

            return project

    def leave(self):
        """Leave the current project."""
        if not self.session_id:
            return

        with self._session() as session:
            agent_session = session.query(AgentSession).filter_by(id=self.session_id).first()
            if agent_session:
                agent_session.status = AgentStatus.OFFLINE.value
                self._log_event(session, "leave", {"project_id": self.project_id})

        self.project_id = None
        self.session_id = None

    def heartbeat(self, activity: str = None):
        """Update heartbeat. Call periodically to stay 'active'."""
        if not self.session_id:
            return

        with self._session() as session:
            agent_session = session.query(AgentSession).filter_by(id=self.session_id).first()
            if agent_session:
                agent_session.last_heartbeat = datetime.now(timezone.utc)
                agent_session.status = AgentStatus.ACTIVE.value
                if activity:
                    agent_session.current_activity = activity

    def list_projects(self) -> List[Dict]:
        """List all available projects."""
        with self._session() as session:
            projects = session.query(SwarmProject).all()
            return [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "created_by": p.created_by,
                    "created_at": p.created_at.isoformat() if p.created_at else None
                }
                for p in projects
            ]

    # ==================== MEMORY OPERATIONS ====================

    def store(
        self,
        content: str,
        summary: str = None,
        memory_type: str = MemoryType.SHARED.value,
        tags: List[str] = None,
        emotional_weight: float = 0.5,
        caused_by: List[str] = None,
        source_memory_id: str = None
    ) -> MemoryResult:
        """Store a memory in the swarm."""
        if not self.project_id:
            raise RuntimeError("Must join a project first")

        memory_id = self._generate_id("mem-")

        with self._session() as session:
            memory = SwarmMemory(
                id=memory_id,
                project_id=self.project_id,
                content=content,
                summary=summary or content[:200],
                created_by=self.agent_id,
                memory_type=memory_type,
                tags=tags or [],
                emotional_weight=emotional_weight,
                caused_by=caused_by or [],
                source_memory_id=source_memory_id
            )
            session.add(memory)
            self._log_event(session, "store", {"memory_id": memory_id, "type": memory_type})
            session.flush()

            # Return dataclass copy (avoids detached instance)
            return MemoryResult.from_orm(memory)

    def recall(self, memory_id: str) -> Optional[MemoryResult]:
        """Recall a specific memory by ID. Increments recall count."""
        with self._session() as session:
            memory = session.query(SwarmMemory).filter_by(id=memory_id).first()
            if memory:
                memory.recall_count += 1
                memory.last_recalled = datetime.now(timezone.utc)
                self._log_event(session, "recall", {"memory_id": memory_id})
                return MemoryResult.from_orm(memory)
            return None

    def search(
        self,
        query: str = None,
        tags: List[str] = None,
        memory_type: str = None,
        created_by: str = None,
        limit: int = 20
    ) -> List[MemoryResult]:
        """Search memories. Currently keyword-based, semantic search TBD."""
        if not self.project_id:
            raise RuntimeError("Must join a project first")

        with self._session() as session:
            q = session.query(SwarmMemory).filter_by(project_id=self.project_id)

            # Filter by type (include shared by default, private only if you created it)
            if memory_type:
                q = q.filter_by(memory_type=memory_type)
            else:
                # Show shared + your own private
                q = q.filter(or_(
                    SwarmMemory.memory_type == MemoryType.SHARED.value,
                    and_(
                        SwarmMemory.memory_type == MemoryType.PRIVATE.value,
                        SwarmMemory.created_by == self.agent_id
                    )
                ))

            if created_by:
                q = q.filter_by(created_by=created_by)

            if query:
                # Simple keyword search (TODO: semantic search with embeddings)
                q = q.filter(SwarmMemory.content.ilike(f"%{query}%"))

            if tags:
                # JSON contains for tags (SQLite compatible)
                for tag in tags:
                    q = q.filter(SwarmMemory.tags.contains([tag]))

            q = q.order_by(SwarmMemory.created_at.desc())
            q = q.limit(limit)

            return [MemoryResult.from_orm(m) for m in q.all()]

    def get_recent_memories(self, limit: int = 10) -> List[MemoryResult]:
        """Get most recent memories in the project."""
        return self.search(limit=limit)

    # ==================== TASK COORDINATION ====================

    def create_task(
        self,
        title: str,
        description: str = None,
        priority: int = 0,
        blocked_by: List[str] = None
    ) -> TaskResult:
        """Create a new task for the swarm."""
        if not self.project_id:
            raise RuntimeError("Must join a project first")

        task_id = self._generate_id("task-")

        with self._session() as session:
            task = SwarmTask(
                id=task_id,
                project_id=self.project_id,
                title=title,
                description=description,
                priority=priority,
                created_by=self.agent_id,
                blocked_by=blocked_by or []
            )
            session.add(task)
            self._log_event(session, "create_task", {"task_id": task_id, "title": title})
            session.flush()
            return TaskResult.from_orm(task)

    def claim_task(self, task_id: str) -> bool:
        """Claim a task. Returns False if already claimed."""
        with self._session() as session:
            task = session.query(SwarmTask).filter_by(id=task_id).first()
            if not task:
                raise ValueError(f"Task '{task_id}' not found")

            if task.claimed_by and task.claimed_by != self.agent_id:
                return False  # Already claimed by someone else

            task.claimed_by = self.agent_id
            task.claimed_at = datetime.now(timezone.utc)
            task.status = TaskStatus.CLAIMED.value

            # Update agent session
            agent_session = session.query(AgentSession).filter_by(id=self.session_id).first()
            if agent_session:
                agent_session.current_task_id = task_id

            self._log_event(session, "claim_task", {"task_id": task_id})
            return True

    def start_task(self, task_id: str):
        """Mark task as in progress."""
        with self._session() as session:
            task = session.query(SwarmTask).filter_by(id=task_id, claimed_by=self.agent_id).first()
            if task:
                task.status = TaskStatus.IN_PROGRESS.value
                self._log_event(session, "start_task", {"task_id": task_id})

    def complete_task(
        self,
        task_id: str,
        result: str = None,
        result_memory_ids: List[str] = None
    ):
        """Mark task as completed."""
        with self._session() as session:
            task = session.query(SwarmTask).filter_by(id=task_id, claimed_by=self.agent_id).first()
            if task:
                task.status = TaskStatus.COMPLETED.value
                task.completed_at = datetime.now(timezone.utc)
                task.result = result
                task.result_memory_ids = result_memory_ids or []

                # Clear from agent session
                agent_session = session.query(AgentSession).filter_by(id=self.session_id).first()
                if agent_session:
                    agent_session.current_task_id = None

                self._log_event(session, "complete_task", {"task_id": task_id})

    def abandon_task(self, task_id: str, reason: str = None):
        """Abandon a claimed task (release for others)."""
        with self._session() as session:
            task = session.query(SwarmTask).filter_by(id=task_id, claimed_by=self.agent_id).first()
            if task:
                task.status = TaskStatus.PENDING.value
                task.claimed_by = None
                task.claimed_at = None

                agent_session = session.query(AgentSession).filter_by(id=self.session_id).first()
                if agent_session:
                    agent_session.current_task_id = None

                self._log_event(session, "abandon_task", {"task_id": task_id, "reason": reason})

    def get_tasks(
        self,
        status: str = None,
        claimed_by: str = None,
        include_completed: bool = False
    ) -> List[TaskResult]:
        """Get tasks, optionally filtered."""
        if not self.project_id:
            raise RuntimeError("Must join a project first")

        with self._session() as session:
            q = session.query(SwarmTask).filter_by(project_id=self.project_id)

            if status:
                q = q.filter_by(status=status)
            elif not include_completed:
                q = q.filter(SwarmTask.status != TaskStatus.COMPLETED.value)

            if claimed_by:
                q = q.filter_by(claimed_by=claimed_by)

            return [TaskResult.from_orm(t) for t in q.order_by(SwarmTask.priority.desc(), SwarmTask.created_at).all()]

    def get_available_tasks(self) -> List[TaskResult]:
        """Get tasks that are pending and not blocked."""
        tasks = self.get_tasks(status=TaskStatus.PENDING.value)
        # Filter out blocked tasks
        completed_ids = set(t.id for t in self.get_tasks(include_completed=True)
                          if t.status == TaskStatus.COMPLETED.value)
        return [t for t in tasks if not t.blocked_by or all(b in completed_ids for b in t.blocked_by)]

    # ==================== AGENT AWARENESS ====================

    def get_active_agents(self, stale_minutes: int = 5) -> List[AgentResult]:
        """Get agents active in the last N minutes."""
        if not self.project_id:
            raise RuntimeError("Must join a project first")

        cutoff = datetime.now(timezone.utc) - timedelta(minutes=stale_minutes)

        with self._session() as session:
            agents = session.query(AgentSession).filter(
                AgentSession.project_id == self.project_id,
                AgentSession.last_heartbeat >= cutoff,
                AgentSession.status != AgentStatus.OFFLINE.value
            ).all()
            return [AgentResult.from_orm(a) for a in agents]

    def get_agent_activity(self) -> List[AgentResult]:
        """Get what each active agent is working on."""
        return self.get_active_agents()

    def broadcast(self, message: str, message_type: str = "info"):
        """Store a coordination message for other agents to see."""
        self.store(
            content=message,
            memory_type=MemoryType.COORDINATION.value,
            tags=["broadcast", message_type]
        )

    # ==================== EVENTS & HISTORY ====================

    def get_recent_events(self, limit: int = 50) -> List[SwarmEvent]:
        """Get recent events in the project."""
        if not self.project_id:
            raise RuntimeError("Must join a project first")

        with self._session() as session:
            return session.query(SwarmEvent).filter_by(
                project_id=self.project_id
            ).order_by(SwarmEvent.timestamp.desc()).limit(limit).all()

    # ==================== IMPORT FROM FILE-BASED MEMORY ====================

    def import_from_file_memory(
        self,
        memory_id: str,
        content: str,
        tags: List[str] = None,
        emotional_weight: float = 0.5,
        memory_type: str = MemoryType.SHARED.value
    ) -> MemoryResult:
        """Import a memory from file-based drift-memory system."""
        return self.store(
            content=content,
            tags=(tags or []) + ["imported"],
            emotional_weight=emotional_weight,
            memory_type=memory_type,
            source_memory_id=memory_id
        )


# Convenience function
def connect(
    agent_id: str,
    agent_name: str = None,
    project_id: str = None,
    db_url: str = "sqlite:///swarm_memory.db",
    create_project: bool = True
) -> SwarmClient:
    """
    Quick connect to swarm memory.

    Usage:
        swarm = connect("drift", "Drift", "my-project")
    """
    client = SwarmClient(agent_id, agent_name, db_url)
    if project_id:
        client.join_project(project_id, create=create_project)
    return client
