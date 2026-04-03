from __future__ import annotations

import json
import sqlite3
import uuid
from collections import Counter
from datetime import datetime
from typing import Iterable

import psycopg2.errors

from .config import settings
from .database import decode_json, encode_json, get_connection, utc_now
from .models import (
    ExampleInput,
    ExampleRecord,
    ExampleSource,
    Label,
    ProjectCreate,
    ProjectRecord,
    ProjectSummary,
    RoundRecord,
    RunEventRecord,
    RunRecord,
    RunStatus,
    Split,
)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


class Repository:
    def create_project(self, payload: ProjectCreate, session_id: str) -> ProjectRecord:
        project_id = str(uuid.uuid4())
        now = utc_now()
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO projects (
                    id, session_id, name, support_domain_description, allowed_topics_json,
                    disallowed_topics_json, routing_notes, agent_model, max_rounds,
                    target_macro_f1, target_out_of_scope_precision, sandbox_profile,
                    promoted_run_id, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, %s, %s)
                """,
                (
                    project_id,
                    session_id,
                    payload.name,
                    payload.support_domain_description,
                    encode_json(payload.allowed_topics),
                    encode_json(payload.disallowed_topics),
                    payload.routing_notes,
                    payload.agent_model,
                    payload.max_rounds,
                    payload.target_macro_f1,
                    payload.target_out_of_scope_precision,
                    payload.sandbox_profile,
                    now,
                    now,
                ),
            )
        return self.get_project(project_id, session_id=session_id)

    def list_projects(self, session_id: str) -> list[ProjectSummary]:
        with get_connection() as connection:
            rows = connection.execute(
                "SELECT * FROM projects WHERE session_id = %s ORDER BY created_at DESC",
                (session_id,),
            ).fetchall()
            counts = connection.execute(
                """
                SELECT e.project_id, e.label, e.split, COUNT(*) AS total
                FROM examples e
                JOIN projects p ON p.id = e.project_id
                WHERE p.session_id = %s
                GROUP BY e.project_id, e.label, e.split
                """,
                (session_id,),
            ).fetchall()
        counter_map: dict[str, dict[str, int]] = {}
        holdout_map: dict[str, dict[str, int]] = {}
        for row in counts:
            target = holdout_map if row["split"] == Split.HOLDOUT.value else counter_map
            target.setdefault(row["project_id"], {})[row["label"]] = row["total"]
        return [
            ProjectSummary(
                **self._project_row_to_dict(row),
                seed_counts=counter_map.get(row["id"], {}),
                holdout_counts=holdout_map.get(row["id"], {}),
            )
            for row in rows
        ]

    def get_project(self, project_id: str, *, session_id: str | None = None) -> ProjectRecord:
        with get_connection() as connection:
            if session_id is None:
                row = connection.execute("SELECT * FROM projects WHERE id = %s", (project_id,)).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM projects WHERE id = %s AND session_id = %s",
                    (project_id, session_id),
                ).fetchone()
        if row is None:
            raise KeyError(f"Project {project_id} not found")
        return ProjectRecord(**self._project_row_to_dict(row))

    def add_examples(
        self,
        project_id: str,
        payloads: Iterable[ExampleInput],
        *,
        split: Split | None = None,
    ) -> list[ExampleRecord]:
        now = utc_now()
        example_rows: list[ExampleRecord] = []
        with get_connection() as connection:
            for payload in payloads:
                example_id = str(uuid.uuid4())
                connection.execute(
                    """
                    INSERT INTO examples (
                        id, project_id, text, label, source, approved, split, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        example_id,
                        project_id,
                        payload.text.strip(),
                        payload.label.value,
                        payload.source.value,
                        payload.approved,
                        split.value if split else None,
                        now,
                    ),
                )
                example_rows.append(
                    ExampleRecord(
                        id=example_id,
                        project_id=project_id,
                        text=payload.text.strip(),
                        label=payload.label,
                        source=payload.source,
                        approved=payload.approved,
                        split=split,
                        created_at=_dt(now),
                    )
                )
        return example_rows

    def list_examples(self, project_id: str) -> list[ExampleRecord]:
        with get_connection() as connection:
            rows = connection.execute(
                "SELECT * FROM examples WHERE project_id = %s ORDER BY created_at ASC",
                (project_id,),
            ).fetchall()
        return [self._example_row_to_model(row) for row in rows]

    def get_seed_counts(self, project_id: str) -> Counter[str]:
        examples = self.list_examples(project_id)
        return Counter(
            example.label.value
            for example in examples
            if example.source == ExampleSource.HUMAN_SEED
        )

    def ensure_seed_minimums(self, project_id: str) -> None:
        counts = self.get_seed_counts(project_id)
        required = {
            Label.IN_SCOPE.value: 5,
            Label.OUT_OF_SCOPE.value: 5,
            Label.AMBIGUOUS.value: 3,
        }
        missing = {
            label: minimum - counts.get(label, 0)
            for label, minimum in required.items()
            if counts.get(label, 0) < minimum
        }
        if missing:
            raise ValueError(f"Seed minimums not met: {missing}")

    def assign_locked_eval_split(self, project_id: str) -> None:
        with get_connection() as connection:
            eval_exists = connection.execute(
                "SELECT COUNT(*) AS total FROM examples WHERE project_id = %s AND split = %s",
                (project_id, Split.EVAL.value),
            ).fetchone()["total"]
            if eval_exists:
                connection.execute(
                    "UPDATE examples SET split = %s WHERE project_id = %s AND split IS NULL",
                    (Split.TRAIN.value, project_id),
                )
                return

            rows = connection.execute(
                """
                SELECT * FROM examples
                WHERE project_id = %s AND source = %s AND approved = TRUE
                ORDER BY created_at ASC
                """,
                (project_id, "human_seed"),
            ).fetchall()
            per_label: dict[str, list[str]] = {}
            for row in rows:
                per_label.setdefault(row["label"], []).append(row["id"])
            eval_ids = set()
            for ids in per_label.values():
                if not ids:
                    continue
                holdout_count = max(1, int(len(ids) * settings.eval_holdout_ratio))
                eval_ids.update(ids[:holdout_count])

            if eval_ids:
                connection.executemany(
                    "UPDATE examples SET split = %s WHERE id = %s",
                    [(Split.EVAL.value, example_id) for example_id in eval_ids],
                )
            connection.execute(
                "UPDATE examples SET split = %s WHERE project_id = %s AND split IS NULL",
                (Split.TRAIN.value, project_id),
            )

    def create_run(self, project_id: str, workspace_root: str) -> RunRecord:
        run_id = str(uuid.uuid4())
        now = utc_now()
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    id, project_id, status, stop_reason, best_round_id, best_macro_f1,
                    summary, workspace_root, created_at, updated_at
                ) VALUES (%s, %s, %s, NULL, NULL, NULL, NULL, %s, %s, %s)
                """,
                (run_id, project_id, RunStatus.QUEUED.value, workspace_root, now, now),
            )
        return self.get_run(run_id)

    def update_run(
        self,
        run_id: str,
        *,
        status: RunStatus | None = None,
        stop_reason: str | None = None,
        best_round_id: str | None = None,
        best_macro_f1: float | None = None,
        summary: str | None = None,
    ) -> RunRecord:
        run = self.get_run(run_id)
        status_value = status.value if status else run.status.value
        with get_connection() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = %s, stop_reason = %s, best_round_id = %s, best_macro_f1 = %s, summary = %s, updated_at = %s
                WHERE id = %s
                """,
                (
                    status_value,
                    stop_reason if stop_reason is not None else run.stop_reason,
                    best_round_id if best_round_id is not None else run.best_round_id,
                    best_macro_f1 if best_macro_f1 is not None else run.best_macro_f1,
                    summary if summary is not None else run.summary,
                    utc_now(),
                    run_id,
                ),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str, *, session_id: str | None = None) -> RunRecord:
        with get_connection() as connection:
            if session_id is None:
                row = connection.execute("SELECT * FROM runs WHERE id = %s", (run_id,)).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT r.*
                    FROM runs r
                    JOIN projects p ON p.id = r.project_id
                    WHERE r.id = %s AND p.session_id = %s
                    """,
                    (run_id, session_id),
                ).fetchone()
        if row is None:
            raise KeyError(f"Run {run_id} not found")
        return RunRecord(
            id=row["id"],
            project_id=row["project_id"],
            status=RunStatus(row["status"]),
            stop_reason=row["stop_reason"],
            best_round_id=row["best_round_id"],
            best_macro_f1=row["best_macro_f1"],
            summary=row["summary"],
            workspace_root=row["workspace_root"],
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
        )

    def create_run_event(
        self,
        run_id: str,
        *,
        event_type: str,
        message: str,
        payload: dict | None = None,
    ) -> RunEventRecord:
        now = utc_now()
        with get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO run_events (run_id, event_type, message, payload_json, created_at)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (run_id, event_type, message, encode_json(payload or {}), now),
            )
            event_id = cursor.lastrowid
        return self.get_run_event(event_id)

    def get_run_event(self, event_id: int) -> RunEventRecord:
        with get_connection() as connection:
            row = connection.execute("SELECT * FROM run_events WHERE id = %s", (event_id,)).fetchone()
        if row is None:
            raise KeyError(f"Run event {event_id} not found")
        return RunEventRecord(
            id=row["id"],
            run_id=row["run_id"],
            event_type=row["event_type"],
            message=row["message"],
            payload=decode_json(row["payload_json"]),
            created_at=_dt(row["created_at"]),
        )

    def list_run_events(
        self,
        run_id: str,
        *,
        after_id: int | None = None,
        session_id: str | None = None,
    ) -> list[RunEventRecord]:
        with get_connection() as connection:
            params: list[object] = [run_id]
            base_sql = """
                SELECT re.*
                FROM run_events re
                JOIN runs r ON r.id = re.run_id
                JOIN projects p ON p.id = r.project_id
                WHERE re.run_id = %s
            """
            if session_id is not None:
                base_sql += " AND p.session_id = %s"
                params.append(session_id)
            if after_id is None:
                rows = connection.execute(
                    base_sql + " ORDER BY re.id ASC",
                    tuple(params),
                ).fetchall()
            else:
                rows = connection.execute(
                    base_sql + " AND re.id > %s ORDER BY re.id ASC",
                    tuple([*params, after_id]),
                ).fetchall()
        return [
            RunEventRecord(
                id=row["id"],
                run_id=row["run_id"],
                event_type=row["event_type"],
                message=row["message"],
                payload=decode_json(row["payload_json"]),
                created_at=_dt(row["created_at"]),
            )
            for row in rows
        ]

    def list_runs(self, project_id: str) -> list[RunRecord]:
        with get_connection() as connection:
            rows = connection.execute(
                "SELECT * FROM runs WHERE project_id = %s ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        return [self.get_run(row["id"]) for row in rows]

    def create_round(self, run_id: str, round_index: int, candidate_file: str) -> RoundRecord:
        round_id = str(uuid.uuid4())
        now = utc_now()
        try:
            with get_connection() as connection:
                connection.execute(
                    """
                    INSERT INTO rounds (
                        id, run_id, round_index, status, candidate_file, dataset_summary_file,
                        review_file, evaluation_file, metrics_json, checkpoint_path, note,
                        created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, NULL, NULL, NULL, %s, NULL, NULL, %s, %s)
                    """,
                    (round_id, run_id, round_index, "queued", candidate_file, encode_json({}), now, now),
                )
        except (psycopg2.errors.UniqueViolation, sqlite3.IntegrityError):
            existing = self.get_round_by_index(run_id, round_index)
            if existing is None:
                raise
            return existing
        return self.get_round(round_id)

    def update_round(
        self,
        round_id: str,
        *,
        status: str | None = None,
        dataset_summary_file: str | None = None,
        review_file: str | None = None,
        evaluation_file: str | None = None,
        holdout_file: str | None = None,
        holdout_evaluation_file: str | None = None,
        metrics: dict | None = None,
        holdout_metrics: dict | None = None,
        checkpoint_path: str | None = None,
        note: str | None = None,
    ) -> RoundRecord:
        current = self.get_round(round_id)
        with get_connection() as connection:
            connection.execute(
                """
                UPDATE rounds
                SET status = %s, dataset_summary_file = %s, review_file = %s, evaluation_file = %s,
                    holdout_file = %s, holdout_evaluation_file = %s, metrics_json = %s, holdout_metrics_json = %s,
                    checkpoint_path = %s, note = %s, updated_at = %s
                WHERE id = %s
                """,
                (
                    status or current.status,
                    dataset_summary_file if dataset_summary_file is not None else current.dataset_summary_file,
                    review_file if review_file is not None else current.review_file,
                    evaluation_file if evaluation_file is not None else current.evaluation_file,
                    holdout_file if holdout_file is not None else current.holdout_file,
                    holdout_evaluation_file if holdout_evaluation_file is not None else current.holdout_evaluation_file,
                    encode_json(metrics if metrics is not None else current.metrics),
                    encode_json(holdout_metrics if holdout_metrics is not None else current.holdout_metrics),
                    checkpoint_path if checkpoint_path is not None else current.checkpoint_path,
                    note if note is not None else current.note,
                    utc_now(),
                    round_id,
                ),
            )
        return self.get_round(round_id)

    def get_round(self, round_id: str) -> RoundRecord:
        with get_connection() as connection:
            row = connection.execute("SELECT * FROM rounds WHERE id = %s", (round_id,)).fetchone()
        if row is None:
            raise KeyError(f"Round {round_id} not found")
        return RoundRecord(
            id=row["id"],
            run_id=row["run_id"],
            round_index=row["round_index"],
            status=row["status"],
            candidate_file=row["candidate_file"],
            dataset_summary_file=row["dataset_summary_file"],
            review_file=row["review_file"],
            evaluation_file=row["evaluation_file"],
            holdout_file=row["holdout_file"],
            holdout_evaluation_file=row["holdout_evaluation_file"],
            metrics=decode_json(row["metrics_json"]),
            holdout_metrics=decode_json(row["holdout_metrics_json"]),
            checkpoint_path=row["checkpoint_path"],
            note=row["note"],
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
        )

    def list_rounds(self, run_id: str) -> list[RoundRecord]:
        with get_connection() as connection:
            rows = connection.execute(
                "SELECT * FROM rounds WHERE run_id = %s ORDER BY round_index ASC",
                (run_id,),
            ).fetchall()
        return [self.get_round(row["id"]) for row in rows]

    def get_round_by_index(self, run_id: str, round_index: int) -> RoundRecord | None:
        with get_connection() as connection:
            row = connection.execute(
                "SELECT * FROM rounds WHERE run_id = %s AND round_index = %s",
                (run_id, round_index),
            ).fetchone()
        if row is None:
            return None
        return self.get_round(row["id"])

    def get_examples_for_split(self, project_id: str, split: Split) -> list[ExampleRecord]:
        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM examples
                WHERE project_id = %s AND split = %s AND approved = TRUE
                ORDER BY created_at ASC
                """,
                (project_id, split.value),
            ).fetchall()
        return [self._example_row_to_model(row) for row in rows]

    def get_holdout_counts(self, project_id: str) -> dict[str, int]:
        holdout_examples = self.get_examples_for_split(project_id, Split.HOLDOUT)
        counts: dict[str, int] = {}
        for example in holdout_examples:
            counts[example.label.value] = counts.get(example.label.value, 0) + 1
        return counts

    def promote_run(self, project_id: str, run_id: str) -> ProjectRecord:
        with get_connection() as connection:
            connection.execute(
                "UPDATE projects SET promoted_run_id = %s, updated_at = %s WHERE id = %s",
                (run_id, utc_now(), project_id),
            )
        return self.get_project(project_id)

    def _project_row_to_dict(self, row) -> dict:
        return {
            "id": row["id"],
            "name": row["name"],
            "support_domain_description": row["support_domain_description"],
            "allowed_topics": decode_json(row["allowed_topics_json"]),
            "disallowed_topics": decode_json(row["disallowed_topics_json"]),
            "routing_notes": row["routing_notes"],
            "agent_model": row["agent_model"],
            "max_rounds": row["max_rounds"],
            "target_macro_f1": row["target_macro_f1"],
            "target_out_of_scope_precision": row["target_out_of_scope_precision"],
            "sandbox_profile": row["sandbox_profile"],
            "promoted_run_id": row["promoted_run_id"],
            "created_at": _dt(row["created_at"]),
            "updated_at": _dt(row["updated_at"]),
        }

    def _example_row_to_model(self, row) -> ExampleRecord:
        return ExampleRecord(
            id=row["id"],
            project_id=row["project_id"],
            text=row["text"],
            label=Label(row["label"]),
            source=row["source"],
            approved=bool(row["approved"]),
            split=Split(row["split"]) if row["split"] else None,
            created_at=_dt(row["created_at"]),
        )
