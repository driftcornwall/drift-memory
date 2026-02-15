"""
Memory Consolidation Daemon — FastAPI server on port 8083.

Endpoints:
  POST /consolidate     — Queue consolidation (202 Accepted)
  POST /consolidate/now — Force immediate consolidation (200 with results)
  GET  /status          — Schema status + memory counts
  GET  /health          — Service health check
  GET  /info            — Version and module info
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, ConfigDict

from consolidation_engine import ConsolidationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("consolidation-daemon")

# Global engine instance
engine = ConsolidationEngine()

# Job tracking
jobs: dict[str, dict] = {}  # job_id -> {status, schema, started, result}
MAX_JOBS = 100  # Rotate old jobs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize engine on startup."""
    logger.info("Starting Memory Consolidation Daemon...")
    result = engine.init()
    for schema, info in result.items():
        if isinstance(info, dict) and info.get("modules_loaded"):
            logger.info(f"  [{schema}] {info['modules_loaded']} modules loaded, DB: {info.get('db_connected')}")
        else:
            logger.info(f"  [{schema}] {info}")
    logger.info("Daemon ready.")
    yield
    logger.info("Shutting down daemon.")


app = FastAPI(
    title="Memory Consolidation Daemon",
    version="1.0.0",
    lifespan=lifespan,
)


# ===== Request/Response models =====

class ConsolidateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    cwd: str = ""
    transcript_path: str = ""
    phases: list[str] = ["all"]
    schema_name: Optional[str] = None  # Auto-detect from cwd if not provided


class ConsolidateNowRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    schema_name: str
    transcript_path: str = ""
    phases: list[str] = ["all"]


class JobResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    job_id: str
    queued_at: str
    schema_name: str


# ===== Helper =====

def _detect_schema(cwd: str) -> str:
    """Auto-detect schema from working directory path."""
    if "Moltbook2" in cwd:
        return "spin"
    elif "Moltbook" in cwd:
        return "drift"
    return "drift"  # Default


def _run_consolidation(job_id: str, schema: str, transcript_path: str, phases: list[str], force: bool = False):
    """Run consolidation synchronously (called from background task or directly)."""
    jobs[job_id]["status"] = "running"
    jobs[job_id]["started_running"] = datetime.now(timezone.utc).isoformat()

    try:
        result = engine.consolidate(
            schema=schema,
            transcript_path=transcript_path,
            phases=phases,
            force=force,
        )
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result
        jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        logger.error(f"Job {job_id} failed: {e}")

    # Rotate old jobs
    if len(jobs) > MAX_JOBS:
        oldest = sorted(jobs.keys(), key=lambda k: jobs[k].get("queued_at", ""))
        for k in oldest[:len(jobs) - MAX_JOBS]:
            del jobs[k]


# ===== Endpoints =====

@app.post("/consolidate", response_model=JobResponse, status_code=202)
async def consolidate(req: ConsolidateRequest, background_tasks: BackgroundTasks):
    """Queue a consolidation job. Returns immediately with job ID."""
    schema = req.schema_name or _detect_schema(req.cwd)
    job_id = str(uuid.uuid4())[:8]

    jobs[job_id] = {
        "status": "queued",
        "schema": schema,
        "queued_at": datetime.now(timezone.utc).isoformat(),
        "cwd": req.cwd,
    }

    background_tasks.add_task(
        _run_consolidation,
        job_id=job_id,
        schema=schema,
        transcript_path=req.transcript_path,
        phases=req.phases,
    )

    return JobResponse(
        job_id=job_id,
        queued_at=jobs[job_id]["queued_at"],
        schema_name=schema,
    )


@app.post("/consolidate/now")
async def consolidate_now(req: ConsolidateNowRequest):
    """Force immediate consolidation and return results synchronously."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "queued",
        "schema": req.schema_name,
        "queued_at": datetime.now(timezone.utc).isoformat(),
    }

    # Run synchronously in thread pool to not block event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        _run_consolidation,
        job_id, req.schema_name, req.transcript_path, req.phases, True,
    )

    return jobs[job_id]


@app.get("/status")
async def status():
    """Get status for all schemas."""
    return engine.get_status()


@app.get("/health")
async def health():
    """Service health check."""
    return engine.health()


@app.get("/info")
async def info():
    """Service info."""
    health_data = engine.health()
    return {
        "version": "1.0.0",
        "service": "memory-consolidation-daemon",
        "port": 8083,
        "modules_loaded": sum(
            e.get("modules", 0) for e in health_data.get("engines", {}).values()
        ),
        "db_connected": all(
            e.get("db", False) for e in health_data.get("engines", {}).values()
        ),
        "incremental_merkle": True,
        "incremental_fingerprint": True,
        "schemas": health_data.get("schemas", []),
        "uptime_s": health_data.get("uptime_s", 0),
    }


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get status of a specific consolidation job."""
    if job_id not in jobs:
        return {"error": f"Job {job_id} not found"}
    return jobs[job_id]
