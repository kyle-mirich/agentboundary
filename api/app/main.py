from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import psycopg2.errors
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import Field

from .config import settings
from .database import _database_backend, _safe_database_target, get_connection, init_db
from .deep_agent import DeepAgentRunner
from .models import (
    ClassificationRequest,
    ClassificationResponse,
    ExampleInput,
    Label,
    LuckyPromptResponse,
    ProjectCreate,
    ProjectDetail,
    QuickStartRequest,
    QuickStartResponse,
    RunCreate,
    RunDetail,
    RunStatusResponse,
    RunStatus,
)
from .ml import classify_text
from .observability import log_event
from .repository import Repository
from .seed_generator import generate_lucky_description, generate_seeds


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    # Runs execute as in-process background tasks, so anything still marked
    # active after a restart can never finish. Fail it now instead of leaving
    # clients waiting on a stream that will never terminate.
    repository.reconcile_interrupted_runs()
    yield


app = FastAPI(title="Agent Boundary API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Session-Id"],
)

repository = Repository()
runner = DeepAgentRunner(repository)
started_at = time.monotonic()


class DatabaseWindowRateLimiter:
    """Fixed-window limiter backed by the database.

    Process-local state would multiply the effective limit by the number of
    workers and reset on every deploy, so hits are recorded in
    `rate_limit_hits` and shared by every instance pointed at the same
    database.
    """

    def __init__(self, name: str, message: str) -> None:
        self.name = name
        self.message = message
        self._last_pruned = 0.0
        self._prune_lock = threading.Lock()

    def clear(self) -> None:
        """Best-effort reset, used by tests and operational tooling.

        Tolerates a missing `rate_limit_hits` table so it stays safe to call
        before schema creation.
        """
        try:
            with get_connection() as connection:
                connection.execute(
                    "DELETE FROM rate_limit_hits WHERE bucket LIKE %s", (f"{self.name}:%",)
                )
        except (sqlite3.OperationalError, psycopg2.errors.UndefinedTable):
            pass
        self._last_pruned = 0.0

    def _prune_expired(self, now: float, window_seconds: int) -> None:
        cutoff = now - max(window_seconds, 1)
        with self._prune_lock:
            if time.time() - self._last_pruned < 60:
                return
            self._last_pruned = time.time()
        with get_connection() as connection:
            connection.execute("DELETE FROM rate_limit_hits WHERE hit_at < %s", (cutoff,))

    def check(self, key: str, *, limit: int, window_seconds: int) -> None:
        if limit <= 0:
            return
        now = time.time()
        cutoff = now - window_seconds
        bucket = f"{self.name}:{key}"
        with get_connection() as connection:
            connection.execute(
                "DELETE FROM rate_limit_hits WHERE bucket = %s AND hit_at < %s",
                (bucket, cutoff),
            )
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM rate_limit_hits WHERE bucket = %s",
                (bucket,),
            ).fetchone()
            if row["total"] >= limit:
                raise HTTPException(status_code=429, detail=self.message)
            connection.execute(
                "INSERT INTO rate_limit_hits (bucket, hit_at) VALUES (%s, %s)",
                (bucket, now),
            )
        self._prune_expired(now, window_seconds)


quick_start_rate_limiter = DatabaseWindowRateLimiter(
    "quick-start",
    "Quick-start limit reached for this browser session. Please wait before starting another run.",
)
run_rate_limiter = DatabaseWindowRateLimiter(
    "run",
    "Run limit reached for this browser session. Please wait before starting another run.",
)
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def session_fingerprint(session_id: str) -> str:
    """Short, stable, non-reversible tag so logs can correlate without storing the id."""
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]


def _normalize_scope_description(description: str) -> str:
    normalized = description.strip()
    lower = normalized.lower()
    if "in scope" in lower or "out of scope" in lower or "ambiguous" in lower:
        return normalized
    return (
        f"In scope: texts about {normalized}. "
        "Out of scope: texts primarily about anything else. "
        "Ambiguous: texts that mix in-scope topics with other topics, or are too unclear to classify confidently."
    )


def _run_quick_start_pipeline(
    *,
    project_id: str,
    run_id: str,
    description: str,
    required_rounds: int,
) -> None:
    repository.create_run_event(
        run_id,
        event_type="seed_generation_started",
        message="Generating labeled seed examples",
        payload={"description": description},
    )
    try:
        seeds = generate_seeds(description)
    except RuntimeError as exc:
        repository.update_run(
            run_id,
            status=RunStatus.FAILED,
            stop_reason=str(exc),
            summary="Seed generation failed",
        )
        repository.create_run_event(
            run_id,
            event_type="seed_generation_failed",
            message=str(exc),
            payload={"description": description},
        )
        return

    repository.add_examples(project_id, seeds)
    counts = {
        label.value: len([seed for seed in seeds if seed.label == label])
        for label in Label
    }
    repository.create_run_event(
        run_id,
        event_type="seed_generation_completed",
        message=f"Generated {len(seeds)} seed examples",
        payload={"counts": counts},
    )
    for label in Label:
        repository.create_run_event(
            run_id,
            event_type="seed_generation_label_count",
            message=f"Prepared {counts[label.value]} {label.value.replace('_', '-')} examples",
            payload={"label": label.value, "count": counts[label.value]},
        )

    runner.execute_run(project_id, run_id, required_rounds)


def _resolve_session_id(resolved: str | None) -> str:
    if not resolved:
        raise HTTPException(status_code=400, detail="Missing session id")
    if not SESSION_ID_PATTERN.fullmatch(resolved):
        raise HTTPException(status_code=400, detail="Invalid session id")
    return resolved


def get_session_id(
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> str:
    """Session id from the header only.

    Accepting it as a query parameter put the value (which is the sole
    credential scoping a project) into URLs, browser history, proxy logs and
    Referer headers on every endpoint.
    """
    return _resolve_session_id(x_session_id)


def get_stream_session_id(
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
    session_id: str | None = Query(default=None),
) -> str:
    """Session id for SSE, which cannot set request headers.

    `EventSource` has no way to send `X-Session-Id`, so the stream is the one
    endpoint that still accepts the id as a query parameter.
    """
    return _resolve_session_id(x_session_id or session_id)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


def _provider_health() -> dict[str, object]:
    models = sorted({settings.default_agent_model, settings.responses_generation_model})
    if not settings.openai_api_key:
        return {
            "status": "error",
            "configured": False,
            "models_checked": models,
            "error_type": "MissingCredential",
        }
    try:
        client = OpenAI(api_key=settings.openai_api_key, timeout=5.0, max_retries=0)
        for model in models:
            client.models.retrieve(model)
    except Exception as exc:
        log_event("provider_health_failed", error_type=type(exc).__name__)
        return {
            "status": "error",
            "configured": True,
            "models_checked": models,
            "error_type": type(exc).__name__,
        }
    return {
        "status": "ok",
        "configured": True,
        "models_checked": models,
    }


@app.get("/health/details")
def health_details() -> dict[str, object]:
    database = {
        "status": "ok",
        "backend": _database_backend(),
        "target": _safe_database_target(settings.database_url),
    }
    try:
        with get_connection() as connection:
            connection.execute("SELECT 1")
    except RuntimeError as exc:
        database["status"] = "error"
        database["error"] = str(exc)
    provider = _provider_health()
    return {
        "status": "ok" if database["status"] == "ok" and provider["status"] == "ok" else "degraded",
        "version": settings.app_version,
        "uptime_seconds": int(time.monotonic() - started_at),
        "database": database,
        "provider": provider,
        "model": {
            "classifier": settings.model_name,
            "agent": settings.default_agent_model,
            "generation": settings.responses_generation_model,
        },
    }


@app.post("/quick-start", response_model=QuickStartResponse)
def quick_start(
    payload: QuickStartRequest,
    background_tasks: BackgroundTasks,
    session_id: str = Depends(get_session_id),
) -> QuickStartResponse:
    quick_start_rate_limiter.check(
        session_id,
        limit=settings.quick_start_rate_limit,
        window_seconds=settings.quick_start_rate_window_seconds,
    )
    required_rounds = 3
    scope_description = _normalize_scope_description(payload.description)
    project = repository.create_project(
        ProjectCreate(
            name=payload.description[:60],
            support_domain_description=scope_description,
            max_rounds=required_rounds,
        ),
        session_id,
    )

    run = repository.create_run(project.id, str(settings.workspace_dir / "pending"))
    repository.update_run_workspace(run.id, str(settings.workspace_dir / run.id))
    repository.update_run(run.id, summary="Quick-start demo run")
    repository.create_run_event(
        run.id,
        event_type="run_queued",
        message="Run queued",
        payload={"project_id": project.id, "required_rounds": required_rounds},
    )
    log_event(
        "quick_start_queued",
        project_id=project.id,
        run_id=run.id,
        session=session_fingerprint(session_id),
        required_rounds=required_rounds,
    )
    background_tasks.add_task(
        _run_quick_start_pipeline,
        project_id=project.id,
        run_id=run.id,
        description=scope_description,
        required_rounds=required_rounds,
    )

    return QuickStartResponse(project_id=project.id, run_id=run.id)


@app.post("/quick-start/lucky", response_model=LuckyPromptResponse)
def quick_start_lucky() -> LuckyPromptResponse:
    return LuckyPromptResponse(description=generate_lucky_description())


@app.get("/projects")
def list_projects(session_id: str = Depends(get_session_id)):
    return repository.list_projects(session_id)


@app.post("/projects")
def create_project(payload: ProjectCreate, session_id: str = Depends(get_session_id)):
    return repository.create_project(payload, session_id)


@app.get("/projects/{project_id}")
def get_project(project_id: str, session_id: str = Depends(get_session_id)):
    try:
        project = repository.get_project(project_id, session_id=session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    holdout_counts = repository.get_holdout_counts(project_id)
    return ProjectDetail(
        project=project,
        examples=repository.list_examples(project_id),
        runs=repository.list_runs(project_id),
        holdout_counts=holdout_counts,
        holdout_ready=bool(holdout_counts),
    )


@app.post("/projects/{project_id}/examples")
def add_examples(
    project_id: str,
    payload: Annotated[list[ExampleInput], Field(max_length=settings.max_examples_per_request)],
    session_id: str = Depends(get_session_id),
):
    try:
        repository.get_project(project_id, session_id=session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return repository.add_examples(project_id, payload)


@app.post("/projects/{project_id}/runs")
def create_run(
    project_id: str,
    payload: RunCreate,
    background_tasks: BackgroundTasks,
    session_id: str = Depends(get_session_id),
):
    try:
        project = repository.get_project(project_id, session_id=session_id)
        repository.ensure_seed_minimums(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Every run costs OpenAI orchestration plus a full training cycle, so it is
    # bounded the same way quick-start is. Without this, the project page's
    # "Start run" button was an unmetered spend path. Checked after validation
    # so requests that cannot succeed do not consume the session's quota.
    run_rate_limiter.check(
        session_id,
        limit=settings.run_rate_limit,
        window_seconds=settings.run_rate_window_seconds,
    )
    if repository.has_active_run(project_id):
        raise HTTPException(
            status_code=409,
            detail="This project already has a run in progress. Wait for it to finish before starting another.",
        )
    run = repository.create_run(project_id, str(settings.workspace_dir / "pending"))
    # Always give the run its own workspace. Sharing a directory across runs
    # lets one run read another run's plan, review and summary artifacts.
    repository.update_run_workspace(run.id, str(settings.workspace_dir / run.id))
    run = repository.update_run(run.id, summary=f"Queued for sandbox profile {project.sandbox_profile}")
    repository.create_run_event(
        run.id,
        event_type="run_queued",
        message="Run queued",
        payload={"project_id": project_id, "max_rounds_override": payload.max_rounds_override},
    )
    log_event(
        "run_queued",
        project_id=project_id,
        run_id=run.id,
        session=session_fingerprint(session_id),
        max_rounds_override=payload.max_rounds_override,
    )
    background_tasks.add_task(runner.execute_run, project_id, run.id, payload.max_rounds_override)
    return run


@app.get("/projects/{project_id}/runs")
def list_runs(project_id: str, session_id: str = Depends(get_session_id)):
    repository.get_project(project_id, session_id=session_id)
    return repository.list_runs(project_id)


@app.get("/runs/{run_id}")
def get_run(run_id: str, session_id: str = Depends(get_session_id)):
    try:
        run = repository.get_run(run_id, session_id=session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    plan_path = Path(run.workspace_root) / "plan.md"
    review_path = Path(run.workspace_root) / "reviews"
    final_summary_path = Path(run.workspace_root) / "reports" / "final-summary.md"
    latest_review = ""
    if review_path.exists():
        reviews = sorted(review_path.glob("*.md"))
        if reviews:
            latest_review = reviews[-1].read_text()
    rounds = repository.list_rounds(run_id)
    holdout_summary = {}
    for round_record in reversed(rounds):
        if round_record.holdout_metrics:
            holdout_summary = round_record.holdout_metrics
            break
    return RunDetail(
        **run.model_dump(),
        rounds=rounds,
        plan_markdown=plan_path.read_text() if plan_path.exists() else "",
        review_markdown=latest_review,
        final_summary_markdown=final_summary_path.read_text() if final_summary_path.exists() else "",
        holdout_summary=holdout_summary,
        events=repository.list_run_events(run_id, session_id=session_id),
    )


@app.get("/runs/{run_id}/status", response_model=RunStatusResponse)
def get_run_status(run_id: str, session_id: str = Depends(get_session_id)):
    try:
        run = repository.get_run(run_id, session_id=session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    events = repository.list_run_events(run_id, session_id=session_id)
    return RunStatusResponse(
        id=run.id,
        project_id=run.project_id,
        status=run.status,
        summary=run.summary,
        stop_reason=run.stop_reason,
        best_round_id=run.best_round_id,
        best_macro_f1=run.best_macro_f1,
        event_count=len(events),
        latest_event=events[-1] if events else None,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


@app.get("/runs/{run_id}/events")
def get_run_events(run_id: str, session_id: str = Depends(get_session_id)):
    try:
        repository.get_run(run_id, session_id=session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return repository.list_run_events(run_id, session_id=session_id)


@app.get("/runs/{run_id}/events/stream")
async def stream_run_events(
    run_id: str,
    request: Request,
    session_id: str = Depends(get_stream_session_id),
):
    try:
        repository.get_run(run_id, session_id=session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def event_generator():
        last_id = 0
        deadline = time.monotonic() + settings.run_stream_max_seconds
        while True:
            # Without an explicit disconnect check this coroutine outlives its
            # client: `asyncio.sleep` never touches the socket, so nothing
            # notices the peer is gone and the loop polls the database forever.
            if await request.is_disconnected():
                return

            events = repository.list_run_events(run_id, after_id=last_id, session_id=session_id)
            for event in events:
                last_id = event.id
                yield f"event: run_event\ndata: {json.dumps(event.model_dump(mode='json'))}\n\n"

            run = repository.get_run(run_id, session_id=session_id)
            terminal = run.status in {RunStatus.COMPLETED, RunStatus.FAILED}
            if terminal:
                yield f"event: run_done\ndata: {json.dumps(run.model_dump(mode='json'))}\n\n"
                return

            if time.monotonic() >= deadline:
                yield (
                    "event: stream_closed\n"
                    f"data: {json.dumps({'reason': 'stream_timeout', 'run_id': run_id})}\n\n"
                )
                return

            await asyncio.sleep(settings.run_stream_poll_interval)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/projects/{project_id}/classify", response_model=ClassificationResponse)
def classify(project_id: str, payload: ClassificationRequest, session_id: str = Depends(get_session_id)):
    try:
        project = repository.get_project(project_id, session_id=session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not project.promoted_run_id:
        raise HTTPException(status_code=400, detail="No promoted run exists for this project yet")
    run = repository.get_run(project.promoted_run_id, session_id=session_id)
    if not run.best_round_id:
        raise HTTPException(status_code=400, detail="Promoted run has no best round")
    round_record = repository.get_round(run.best_round_id)
    if not round_record.checkpoint_path:
        raise HTTPException(status_code=400, detail="Best round has no checkpoint path")
    result = classify_text(payload.text, Path(round_record.checkpoint_path))
    log_event(
        "classification_completed",
        project_id=project_id,
        run_id=run.id,
        session=session_fingerprint(session_id),
        label=result.label.value,
        confidence=result.confidence,
    )
    return result


@app.post("/projects/{project_id}/promote/{run_id}")
def promote_run(project_id: str, run_id: str, session_id: str = Depends(get_session_id)):
    try:
        repository.get_project(project_id, session_id=session_id)
        run = repository.get_run(run_id, session_id=session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if run.project_id != project_id:
        raise HTTPException(status_code=400, detail="Run does not belong to this project")
    return repository.promote_run(project_id, run_id)
