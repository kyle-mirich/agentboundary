from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import settings
from .database import get_connection, init_db, utc_now
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
    RunStatus,
)
from .ml import classify_text
from .repository import Repository
from .seed_generator import generate_lucky_description, generate_seeds


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="Agent Boundary API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repository = Repository()
runner = DeepAgentRunner(repository)


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


def get_session_id(
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
    session_id: str | None = Query(default=None),
) -> str:
    resolved = x_session_id or session_id
    if not resolved:
        raise HTTPException(status_code=400, detail="Missing session id")
    return resolved

@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/quick-start", response_model=QuickStartResponse)
def quick_start(
    payload: QuickStartRequest,
    background_tasks: BackgroundTasks,
    session_id: str = Depends(get_session_id),
) -> QuickStartResponse:
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

    workspace_root = str(settings.workspace_dir / "pending")
    run = repository.create_run(project.id, workspace_root)
    run_workspace = str(settings.workspace_dir / run.id)
    repository.update_run(run.id, summary="Quick-start demo run")
    if project.sandbox_profile == "isolated_fs":
        with get_connection() as connection:
            connection.execute(
                "UPDATE runs SET workspace_root = %s, updated_at = %s WHERE id = %s",
                (run_workspace, utc_now(), run.id),
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
def add_examples(project_id: str, payload: list[ExampleInput], session_id: str = Depends(get_session_id)):
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
    workspace_root = str(settings.workspace_dir / "pending")
    run = repository.create_run(project_id, workspace_root)
    run_workspace = str(settings.workspace_dir / run.id)
    run = repository.update_run(run.id, summary=f"Queued for sandbox profile {project.sandbox_profile}")
    # runs created with isolated_fs always map to a deterministic local workspace path
    if project.sandbox_profile == "isolated_fs":
        with get_connection() as connection:
            connection.execute(
                "UPDATE runs SET workspace_root = %s, updated_at = %s WHERE id = %s",
                (run_workspace, utc_now(), run.id),
            )
    background_tasks.add_task(runner.execute_run, project_id, run.id, payload.max_rounds_override)
    return repository.get_run(run.id)


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


@app.get("/runs/{run_id}/events")
def get_run_events(run_id: str, session_id: str = Depends(get_session_id)):
    try:
        repository.get_run(run_id, session_id=session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return repository.list_run_events(run_id, session_id=session_id)


@app.get("/runs/{run_id}/events/stream")
async def stream_run_events(run_id: str, session_id: str = Depends(get_session_id)):
    try:
        repository.get_run(run_id, session_id=session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def event_generator():
        last_id = 0
        while True:
            events = repository.list_run_events(run_id, after_id=last_id, session_id=session_id)
            for event in events:
                last_id = event.id
                yield f"event: run_event\ndata: {json.dumps(event.model_dump(mode='json'))}\n\n"
            run = repository.get_run(run_id, session_id=session_id)
            if run.status in {RunStatus.COMPLETED, RunStatus.FAILED}:
                yield f"event: run_done\ndata: {json.dumps(run.model_dump(mode='json'))}\n\n"
                break
            await asyncio.sleep(1)

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
    return classify_text(payload.text, Path(round_record.checkpoint_path))


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
