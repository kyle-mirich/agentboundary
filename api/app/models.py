from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .config import settings


class Label(str, Enum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS = "ambiguous"


class ExampleSource(str, Enum):
    HUMAN_SEED = "human_seed"
    SYNTHETIC_EXPAND = "synthetic_expand"
    SYNTHETIC_HARD_NEGATIVE = "synthetic_hard_negative"
    SYNTHETIC_AMBIGUOUS = "synthetic_ambiguous"
    SYNTHETIC_HOLDOUT = "synthetic_holdout"


class Split(str, Enum):
    TRAIN = "train"
    EVAL = "eval"
    HOLDOUT = "holdout"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ProjectCreate(BaseModel):
    name: str
    support_domain_description: str
    allowed_topics: list[str] = Field(default_factory=list)
    disallowed_topics: list[str] = Field(default_factory=list)
    routing_notes: str = ""
    agent_model: str = Field(default_factory=lambda: settings.default_agent_model)
    max_rounds: int = 3
    target_macro_f1: float = 0.9
    target_out_of_scope_precision: float = 0.95
    sandbox_profile: str = "isolated_fs"


class ProjectRecord(ProjectCreate):
    id: str
    created_at: datetime
    updated_at: datetime
    promoted_run_id: str | None = None


class ExampleInput(BaseModel):
    text: str
    label: Label
    source: ExampleSource = ExampleSource.HUMAN_SEED
    approved: bool = True


class ExampleRecord(ExampleInput):
    id: str
    project_id: str
    split: Split | None = None
    created_at: datetime


class RunCreate(BaseModel):
    max_rounds_override: int | None = None


class RunRecord(BaseModel):
    id: str
    project_id: str
    status: RunStatus
    stop_reason: str | None = None
    best_round_id: str | None = None
    best_macro_f1: float | None = None
    summary: str | None = None
    workspace_root: str
    created_at: datetime
    updated_at: datetime


class RoundRecord(BaseModel):
    id: str
    run_id: str
    round_index: int
    status: str
    candidate_file: str | None = None
    dataset_summary_file: str | None = None
    review_file: str | None = None
    evaluation_file: str | None = None
    holdout_file: str | None = None
    holdout_evaluation_file: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    holdout_metrics: dict[str, Any] = Field(default_factory=dict)
    checkpoint_path: str | None = None
    note: str | None = None
    created_at: datetime
    updated_at: datetime


class ProjectSummary(ProjectRecord):
    seed_counts: dict[str, int] = Field(default_factory=dict)
    holdout_counts: dict[str, int] = Field(default_factory=dict)


class RunDetail(RunRecord):
    rounds: list[RoundRecord] = Field(default_factory=list)
    plan_markdown: str = ""
    review_markdown: str = ""
    final_summary_markdown: str = ""
    holdout_summary: dict[str, Any] = Field(default_factory=dict)
    events: list["RunEventRecord"] = Field(default_factory=list)


class ProjectDetail(BaseModel):
    project: ProjectRecord
    examples: list[ExampleRecord] = Field(default_factory=list)
    runs: list[RunRecord] = Field(default_factory=list)
    holdout_counts: dict[str, int] = Field(default_factory=dict)
    holdout_ready: bool = False


class ClassificationRequest(BaseModel):
    text: str


class ClassificationResponse(BaseModel):
    label: Label
    confidence: float
    probabilities: dict[str, float]
    explanation: str


class RunEventRecord(BaseModel):
    id: int
    run_id: str
    event_type: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class QuickStartRequest(BaseModel):
    description: str


class QuickStartResponse(BaseModel):
    project_id: str
    run_id: str


class LuckyPromptResponse(BaseModel):
    description: str
