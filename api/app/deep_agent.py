from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from openai import OpenAI
from pydantic import BaseModel, Field

from .config import settings
from .ml import classify_text, evaluate_model, train_model
from .models import ExampleInput, ExampleSource, Label, RunStatus, Split
from .repository import Repository


try:
    from langchain_runloop import RunloopSandbox
    from runloop_api_client import RunloopSDK
except ImportError:  # pragma: no cover - optional dependency
    RunloopSandbox = None
    RunloopSDK = None


class WorkspaceIO:
    def read_text(self, relative_path: str) -> str:
        raise NotImplementedError

    def write_text(self, relative_path: str, content: str) -> None:
        raise NotImplementedError

    def exists(self, relative_path: str) -> bool:
        raise NotImplementedError

    def ensure(self) -> None:
        raise NotImplementedError


class LocalWorkspaceIO(WorkspaceIO):
    def __init__(self, root: Path) -> None:
        self.root = root

    def resolve(self, relative_path: str) -> Path:
        relative = relative_path.removeprefix("/workspace/").lstrip("/")
        return self.root / relative

    def read_text(self, relative_path: str) -> str:
        return self.resolve(relative_path).read_text()

    def write_text(self, relative_path: str, content: str) -> None:
        target = self.resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def exists(self, relative_path: str) -> bool:
        return self.resolve(relative_path).exists()

    def ensure(self) -> None:
        for folder in ("rounds", "datasets", "reviews", "reports"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)


class RunloopWorkspaceIO(WorkspaceIO):
    def __init__(self, backend) -> None:
        self.backend = backend

    def ensure(self) -> None:
        self.backend.execute("mkdir -p /workspace/rounds /workspace/datasets /workspace/reviews /workspace/reports")

    def read_text(self, relative_path: str) -> str:
        downloaded = self.backend.download_files([relative_path])[0]
        return downloaded.content.decode("utf-8")

    def write_text(self, relative_path: str, content: str) -> None:
        self.backend.upload_files([(relative_path, content.encode("utf-8"))])

    def exists(self, relative_path: str) -> bool:
        try:
            self.backend.download_files([relative_path])
        except Exception:
            return False
        return True


@dataclass
class AgentContext:
    repository: Repository
    project_id: str
    run_id: str
    workspace: WorkspaceIO
    artifacts_root: Path


class GeneratedExample(BaseModel):
    text: str = Field(min_length=8)
    label: Label
    source: ExampleSource


class GeneratedExampleBatch(BaseModel):
    examples: list[GeneratedExample] = Field(default_factory=list)


class GeneratedTextBatch(BaseModel):
    examples: list[str] = Field(default_factory=list)


def _round_paths(round_index: int) -> dict[str, str]:
    return {
        "candidate": f"/workspace/datasets/round-{round_index:02d}-candidates.jsonl",
        "dataset_summary": f"/workspace/rounds/round-{round_index:02d}-dataset-summary.json",
        "evaluation": f"/workspace/rounds/round-{round_index:02d}-evaluation.json",
        "holdout": f"/workspace/rounds/round-{round_index:02d}-holdout.jsonl",
        "holdout_evaluation": f"/workspace/rounds/round-{round_index:02d}-holdout-evaluation.json",
        "review": f"/workspace/reviews/round-{round_index:02d}-review.md",
    }


FINAL_SUMMARY_PATH = "/workspace/reports/final-summary.md"


def _build_backend(project_id: str, run_id: str, sandbox_profile: str):
    if sandbox_profile == "runloop":
        if RunloopSandbox is None or RunloopSDK is None:
            raise RuntimeError("Runloop support is not installed. Install the optional runloop dependency.")
        if not settings.runloop_api_key:
            raise RuntimeError("APP_RUNLOOP_API_KEY is required when sandbox_profile=runloop")
        client = RunloopSDK(bearer_token=settings.runloop_api_key)
        devbox = client.devbox.create()
        runloop_backend = RunloopSandbox(devbox=devbox)
        workspace = RunloopWorkspaceIO(runloop_backend)
        workspace.ensure()
        memories_root = settings.memory_dir / project_id
        memories_root.mkdir(parents=True, exist_ok=True)
        backend = lambda runtime: CompositeBackend(
            default=runloop_backend,
            routes={"/memories/": FilesystemBackend(root_dir=str(memories_root), virtual_mode=True)},
        )
        return backend, workspace

    workspace_root = settings.workspace_dir / run_id
    workspace = LocalWorkspaceIO(workspace_root)
    workspace.ensure()
    memories_root = settings.memory_dir / project_id
    memories_root.mkdir(parents=True, exist_ok=True)
    backend = lambda runtime: CompositeBackend(
        default=StateBackend(runtime),
        routes={
            "/workspace/": FilesystemBackend(root_dir=str(workspace_root), virtual_mode=True),
            "/memories/": FilesystemBackend(root_dir=str(memories_root), virtual_mode=True),
        },
    )
    return backend, workspace


def _make_tools(context: AgentContext) -> list:
    repository = context.repository
    project = repository.get_project(context.project_id)
    openai_client = OpenAI(api_key=settings.openai_api_key)
    targeted_example_count = max(4, settings.generated_examples_per_label // 3)

    def emit(event_type: str, message: str, payload: dict[str, Any] | None = None) -> None:
        repository.create_run_event(
            context.run_id,
            event_type=event_type,
            message=message,
            payload=payload or {},
        )

    def ensure_round(round_index: int, candidate_file: str):
        existing = repository.get_round_by_index(context.run_id, round_index)
        if existing is not None:
            return existing
        return repository.create_round(context.run_id, round_index, candidate_file)

    def summarize_round(round_record) -> dict[str, Any]:
        return {
            "round_index": round_record.round_index,
            "status": round_record.status,
            "macro_f1": round_record.metrics.get("macro_f1"),
            "out_of_scope_precision": round_record.metrics.get("out_of_scope_precision"),
            "holdout_macro_f1": round_record.holdout_metrics.get("macro_f1"),
            "holdout_out_of_scope_precision": round_record.holdout_metrics.get("out_of_scope_precision"),
            "note": round_record.note,
        }

    def resolve_candidate_file(round_index: int, requested_candidate_file: str | None) -> tuple[Any, str]:
        canonical_candidate = _round_paths(round_index)["candidate"]
        round_record = ensure_round(round_index, canonical_candidate)
        candidates = [
            path
            for path in [requested_candidate_file, round_record.candidate_file, canonical_candidate]
            if path
        ]
        seen_paths: set[str] = set()
        for path in candidates:
            if path in seen_paths:
                continue
            seen_paths.add(path)
            if context.workspace.exists(path):
                return round_record, path

        if round_index == 1:
            context.workspace.write_text(canonical_candidate, "")
            return round_record, canonical_candidate

        raise RuntimeError(
            f"Candidate file not found for round {round_index}. "
            "Call generate_candidates first and pass its returned candidate_file to run_round."
        )

    def parse_structured_response(prompt: str) -> GeneratedTextBatch:
        response = openai_client.responses.parse(
            model=settings.responses_generation_model,
            input=[
                {
                    "role": "system",
                    "content": "Generate only structured example data for a generic in-scope classifier.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            text_format=GeneratedTextBatch,
        )
        for output in response.output:
            if output.type != "message":
                continue
            for item in output.content:
                if item.type == "refusal":
                    raise RuntimeError(item.refusal)
                if item.parsed:
                    return item.parsed
        raise RuntimeError("Could not parse structured example response")

    def create_examples(
        topic: str,
        n: int,
        on_topic: bool,
        generation_phase: str = "candidate",
    ) -> list[GeneratedExample]:
        label = Label.IN_SCOPE if on_topic else Label.OUT_OF_SCOPE
        source = ExampleSource.SYNTHETIC_EXPAND if on_topic else ExampleSource.SYNTHETIC_HARD_NEGATIVE
        message_scope = "in-scope" if on_topic else "out-of-scope"
        emit(
            "generation_in_progress",
            f"Generating {n} {message_scope} {generation_phase} examples",
            {"topic": topic, "count": n, "label": label.value, "phase": generation_phase},
        )
        parsed = parse_structured_response(
            (
                f"Create {n} distinct examples for the topic '{topic}'. "
                f"These examples should be {'clearly in-scope texts about that topic' if on_topic else 'clearly out-of-scope texts about other topics'}. "
                "Return plain user messages only."
            )
        )
        return [
            GeneratedExample(text=item.strip(), label=label, source=source)
            for item in parsed.examples
            if item.strip()
        ]

    def create_ambiguous_examples(
        topic: str,
        n: int,
        generation_phase: str = "candidate",
    ) -> list[GeneratedExample]:
        emit(
            "generation_in_progress",
            f"Generating {n} ambiguous {generation_phase} examples",
            {"topic": topic, "count": n, "label": Label.AMBIGUOUS.value, "phase": generation_phase},
        )
        parsed = parse_structured_response(
            (
                f"Create {n} ambiguous examples for the topic '{topic}'. "
                "Each example must mix an in-scope topic with an out-of-scope topic, or remain genuinely unclear. "
                "Return plain user messages only."
            )
        )
        return [
            GeneratedExample(text=item.strip(), label=Label.AMBIGUOUS, source=ExampleSource.SYNTHETIC_AMBIGUOUS)
            for item in parsed.examples
            if item.strip()
        ]

    def create_holdout_examples(topic: str, n: int, label: Label) -> list[GeneratedExample]:
        if label == Label.AMBIGUOUS:
            return create_ambiguous_examples(topic, n, generation_phase="holdout")
        return create_examples(topic, n, on_topic=label == Label.IN_SCOPE, generation_phase="holdout")

    def write_holdout_file(round_index: int, holdout_examples: list[ExampleInput]) -> str:
        holdout_path = _round_paths(round_index)["holdout"]
        lines = "\n".join(
            json.dumps(
                {
                    "text": example.text.strip(),
                    "label": example.label.value,
                    "source": example.source.value,
                }
            )
            for example in holdout_examples
        )
        context.workspace.write_text(holdout_path, lines)
        return holdout_path

    @tool(parse_docstring=True)
    def generate_candidates(round_index: int, focus_note: str = "") -> str:
        """Optionally generate targeted candidate examples and write them to the workspace.

        Args:
            round_index: The current round index.
            focus_note: Concrete boundary case or failure mode to target. Leave blank to keep the initial dataset fixed.
        """
        candidate_path = _round_paths(round_index)["candidate"]
        emit(
            "round_started",
            f"Round {round_index} started",
            {"round_index": round_index, "focus_note": focus_note},
        )
        emit(
            "candidate_generation_started",
            "Preparing candidate augmentation plan",
            {"round_index": round_index, "focus_note": focus_note},
        )
        if not focus_note.strip():
            context.workspace.write_text(candidate_path, "")
            ensure_round(round_index, candidate_path)
            summary = {
                "round_index": round_index,
                "candidate_file": candidate_path,
                "generated_count": 0,
                "focus_note": "",
                "mode": "fixed_seed_baseline",
            }
            context.workspace.write_text(
                f"/workspace/rounds/round-{round_index:02d}-generation.json",
                json.dumps(summary, indent=2),
            )
            emit(
                "candidate_generation_skipped",
                "Using the initial labeled seed set for round 1",
                summary,
            )
            return json.dumps(summary)

        existing_examples = repository.list_examples(context.project_id)
        seen_texts = {example.text.strip().lower() for example in existing_examples}
        topic = ", ".join(project.allowed_topics) or project.support_domain_description
        off_topic = ", ".join(project.disallowed_topics) or "unrelated general questions"
        example_count = targeted_example_count
        generated = (
            create_examples(f"{topic}. Target this failure mode: {focus_note}", example_count, True)
            + create_examples(f"{off_topic}. Target this failure mode: {focus_note}", example_count, False)
            + create_ambiguous_examples(f"{topic}. Target this failure mode: {focus_note}", example_count)
        )
        filtered_examples: list[dict[str, str]] = []
        for example in generated:
            normalized = example.text.strip().lower()
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)
            filtered_examples.append(
                {
                    "text": example.text.strip(),
                    "label": example.label.value,
                    "source": example.source.value,
                }
            )
        lines = "\n".join(json.dumps(item) for item in filtered_examples)
        context.workspace.write_text(candidate_path, lines)
        ensure_round(round_index, candidate_path)
        summary = {
            "round_index": round_index,
            "candidate_file": candidate_path,
            "generated_count": len(filtered_examples),
            "focus_note": focus_note,
            "mode": "targeted_augmentation",
        }
        context.workspace.write_text(
            f"/workspace/rounds/round-{round_index:02d}-generation.json",
            json.dumps(summary, indent=2),
        )
        emit("candidates_generated", "Synthetic candidates generated", summary)
        return json.dumps(summary)

    @tool(parse_docstring=True)
    def create_holdout(round_index: int, count_per_label: int = 8, source_split: str = "train") -> str:
        """Create a final holdout dataset similar to the current project split.

        Args:
            round_index: The current round index.
            count_per_label: Number of holdout examples to generate per label.
            source_split: Which split to mirror when generating the holdout dataset.
        """
        if source_split not in {"train", "eval"}:
            raise RuntimeError("source_split must be 'train' or 'eval'")
        emit(
            "holdout_generation_started",
            "Preparing final holdout set",
            {"round_index": round_index, "count_per_label": count_per_label, "source_split": source_split},
        )
        existing_holdout = repository.get_examples_for_split(context.project_id, Split.HOLDOUT)
        if existing_holdout:
            holdout_path = write_holdout_file(
                round_index,
                [
                    ExampleInput(
                        text=example.text,
                        label=example.label,
                        source=example.source,
                        approved=example.approved,
                    )
                    for example in existing_holdout
                ],
            )
            round_record = ensure_round(round_index, _round_paths(round_index)["candidate"])
            label_counts = {
                label.value: sum(1 for example in existing_holdout if example.label == label)
                for label in Label
            }
            repository.update_round(
                round_record.id,
                holdout_file=holdout_path,
                note=f"Reused persisted holdout set with {len(existing_holdout)} examples",
            )
            emit("holdout_reused", "Reused persisted holdout set", {"round_index": round_index, "count": len(existing_holdout)})
            return json.dumps(
                {
                    "round_index": round_index,
                    "holdout_file": holdout_path,
                    "count_per_label": count_per_label,
                    "generated_count": len(existing_holdout),
                    "source_split": source_split,
                    "reused": True,
                    "label_counts": label_counts,
                }
            )
        split_examples = repository.get_examples_for_split(context.project_id, Split(source_split))
        by_label = {
            label: [example.text for example in split_examples if example.label == label]
            for label in Label
        }
        topic = ", ".join(project.allowed_topics) or project.support_domain_description
        off_topic = ", ".join(project.disallowed_topics) or "unrelated general questions"
        holdout_payloads: list[ExampleInput] = []
        seen_texts = {example.text.strip().lower() for example in repository.list_examples(context.project_id)}
        for label in Label:
            label_topic = topic if label != Label.OUT_OF_SCOPE else off_topic
            focus_prompt = (
                f"Use these {source_split} examples as style guidance and create a fresh holdout set. "
                f"Avoid paraphrasing too closely: {json.dumps(by_label[label][:8])}"
            )
            generated = create_holdout_examples(f"{label_topic}. {focus_prompt}", count_per_label, label)
            for example in generated:
                normalized = example.text.strip().lower()
                if normalized in seen_texts:
                    continue
                seen_texts.add(normalized)
                holdout_payloads.append(
                    ExampleInput(
                        text=example.text.strip(),
                        label=example.label,
                        source=ExampleSource.SYNTHETIC_HOLDOUT,
                        approved=True,
                    )
                )
        stored_holdout = repository.add_examples(context.project_id, holdout_payloads, split=Split.HOLDOUT)
        holdout_path = write_holdout_file(round_index, holdout_payloads)
        round_record = ensure_round(round_index, _round_paths(round_index)["candidate"])
        repository.update_round(
            round_record.id,
            holdout_file=holdout_path,
            note=f"Created persisted holdout set with {len(stored_holdout)} examples",
        )
        emit("holdout_created", "Created persisted holdout set", {"round_index": round_index, "count": len(stored_holdout)})
        payload = {
            "round_index": round_index,
            "holdout_file": holdout_path,
            "count_per_label": count_per_label,
            "generated_count": len(stored_holdout),
            "source_split": source_split,
            "reused": False,
            "label_counts": {
                label.value: sum(1 for example in stored_holdout if example.label == label)
                for label in Label
            },
        }
        return json.dumps(payload)

    def prepare_dataset_impl(round_index: int, candidate_file: str) -> dict[str, Any]:
        round_record, candidate_file = resolve_candidate_file(round_index, candidate_file)
        emit(
            "dataset_prep_started",
            "Preparing dataset from generated candidates",
            {"round_index": round_index, "candidate_file": candidate_file},
        )
        raw_text = context.workspace.read_text(candidate_file)
        lines = [line for line in raw_text.splitlines() if line.strip()]
        existing_texts = {example.text.strip().lower() for example in repository.list_examples(context.project_id)}
        accepted: list[ExampleInput] = []
        seen = set(existing_texts)

        for line in lines:
            payload = json.loads(line)
            text = payload["text"].strip()
            normalized = text.lower()
            label = payload["label"]
            source = payload.get("source", ExampleSource.SYNTHETIC_EXPAND.value)
            if label not in {item.value for item in Label}:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            accepted.append(
                ExampleInput(
                    text=text,
                    label=Label(label),
                    source=ExampleSource(source),
                    approved=True,
                )
            )

        if accepted:
            repository.add_examples(context.project_id, accepted)
        repository.assign_locked_eval_split(context.project_id)
        train_examples = repository.get_examples_for_split(context.project_id, Split.TRAIN)
        eval_examples = repository.get_examples_for_split(context.project_id, Split.EVAL)
        summary = {
            "round_index": round_index,
            "accepted_examples": len(accepted),
            "train_count": len(train_examples),
            "eval_count": len(eval_examples),
            "label_counts": {
                label.value: sum(1 for example in train_examples if example.label == label)
                for label in Label
            },
        }
        summary_path = _round_paths(round_index)["dataset_summary"]
        context.workspace.write_text(summary_path, json.dumps(summary, indent=2))
        repository.update_round(
            round_record.id,
            status="dataset_prepared",
            dataset_summary_file=summary_path,
        )
        emit("dataset_prepared", "Dataset prepared", summary)
        return {"dataset_summary_file": summary_path, **summary}

    def train_classifier_impl(round_index: int) -> dict[str, Any]:
        round_record = ensure_round(round_index, _round_paths(round_index)["candidate"])
        emit(
            "training_started",
            "Starting DistilBERT fine-tuning",
            {"round_index": round_index, "model_name": settings.model_name},
        )
        train_examples = repository.get_examples_for_split(context.project_id, Split.TRAIN)
        checkpoint_dir = context.artifacts_root / context.run_id / f"round-{round_index:02d}"
        result = train_model(train_examples, checkpoint_dir)
        payload = {
            "round_index": round_index,
            "checkpoint_path": str(result.checkpoint_path),
            "training_loss": round(result.training_loss, 4),
            "train_count": result.train_count,
        }
        context.workspace.write_text(f"/workspace/rounds/round-{round_index:02d}-train.json", json.dumps(payload, indent=2))
        repository.update_round(
            round_record.id,
            status="trained",
            checkpoint_path=str(result.checkpoint_path),
            note=f"Training loss {payload['training_loss']}",
        )
        emit("training_complete", "Fine-tuning complete", payload)
        return payload

    def evaluate_classifier_impl(round_index: int) -> dict[str, Any]:
        round_record = ensure_round(round_index, _round_paths(round_index)["candidate"])
        emit(
            "evaluation_started",
            "Evaluating checkpoint on locked eval split",
            {"round_index": round_index},
        )
        eval_examples = repository.get_examples_for_split(context.project_id, Split.EVAL)
        checkpoint_dir = context.artifacts_root / context.run_id / f"round-{round_index:02d}"
        metrics = evaluate_model(eval_examples, checkpoint_dir)
        metrics["round_index"] = round_index
        evaluation_path = _round_paths(round_index)["evaluation"]
        context.workspace.write_text(evaluation_path, json.dumps(metrics, indent=2))
        repository.update_round(
            round_record.id,
            status="evaluated",
            evaluation_file=evaluation_path,
            metrics=metrics,
        )
        emit("evaluation_complete", "Evaluation complete", metrics)
        return {"evaluation_file": evaluation_path, **metrics}

    @tool(parse_docstring=True)
    def evaluate_holdout(round_index: int, holdout_file: str) -> str:
        """Evaluate the current checkpoint on a generated holdout dataset.

        Args:
            round_index: The current round index.
            holdout_file: A /workspace path to the generated holdout JSONL file.
        """
        emit(
            "holdout_evaluation_started",
            "Evaluating checkpoint on persisted holdout set",
            {"round_index": round_index, "holdout_file": holdout_file},
        )
        holdout_examples = repository.get_examples_for_split(context.project_id, Split.HOLDOUT)
        if not holdout_examples:
            raise RuntimeError("No persisted holdout set exists for this project")
        if not context.workspace.exists(holdout_file):
            holdout_file = write_holdout_file(
                round_index,
                [
                    ExampleInput(
                        text=example.text,
                        label=example.label,
                        source=example.source,
                        approved=example.approved,
                    )
                    for example in holdout_examples
                ],
            )
        checkpoint_dir = context.artifacts_root / context.run_id / f"round-{round_index:02d}"
        metrics = evaluate_model(
            [
                type("EvalExample", (), {
                    "text": example.text,
                    "label": example.label,
                })()
                for example in holdout_examples
            ],
            checkpoint_dir,
        )
        metrics["round_index"] = round_index
        metrics["holdout_count"] = len(holdout_examples)
        holdout_eval_path = _round_paths(round_index)["holdout_evaluation"]
        context.workspace.write_text(holdout_eval_path, json.dumps(metrics, indent=2))
        round_record = ensure_round(round_index, _round_paths(round_index)["candidate"])
        repository.update_round(
            round_record.id,
            holdout_evaluation_file=holdout_eval_path,
            holdout_metrics=metrics,
        )
        emit("holdout_evaluated", "Holdout evaluation complete", metrics)
        return json.dumps({"holdout_evaluation_file": holdout_eval_path, **metrics})

    @tool(parse_docstring=True)
    def run_round(round_index: int, candidate_file: str) -> str:
        """Run one deterministic classifier round in order.

        Args:
            round_index: The current round index.
            candidate_file: A /workspace path to the candidate JSONL file produced by DatasetGenerator.
        """
        _, candidate_file = resolve_candidate_file(round_index, candidate_file)
        dataset_summary = prepare_dataset_impl(round_index, candidate_file)
        training_summary = train_classifier_impl(round_index)
        evaluation_summary = evaluate_classifier_impl(round_index)
        payload = {
            "round_index": round_index,
            "candidate_file": candidate_file,
            "dataset_summary": dataset_summary,
            "training": training_summary,
            "evaluation": evaluation_summary,
        }
        emit("round_complete", f"Round {round_index} complete", payload)
        return json.dumps(payload)

    @tool(parse_docstring=True)
    def record_review(round_index: int, review_file: str, note: str = "") -> str:
        """Attach a review artifact to the current round.

        Args:
            round_index: The current round index.
            review_file: A /workspace Markdown path written by the reviewer subagent.
            note: A short summary of the reviewer recommendation.
        """
        emit(
            "review_started",
            "Reviewer is writing final assessment",
            {"round_index": round_index},
        )
        round_record = ensure_round(round_index, _round_paths(round_index)["candidate"])
        if not context.workspace.exists(review_file):
            raise RuntimeError(f"Review file not found: {review_file}")
        repository.update_round(
            round_record.id,
            status="reviewed",
            review_file=review_file,
            note=note or f"Review recorded for round {round_index}",
        )
        emit("review_recorded", "Review captured", {"round_index": round_index, "note": note})
        return json.dumps({"round_index": round_index, "review_file": review_file, "note": note})

    @tool(parse_docstring=True)
    def write_final_summary(summary_markdown: str, selected_round_index: int, note: str = "") -> str:
        """Write the final run summary document.

        Args:
            summary_markdown: Full Markdown summary of the run, including what happened and why the selected round won.
            selected_round_index: The round chosen as the final winner.
            note: Short one-line reason for the final selection.
        """
        context.workspace.write_text(FINAL_SUMMARY_PATH, summary_markdown.strip() + "\n")
        payload = {
            "summary_file": FINAL_SUMMARY_PATH,
            "selected_round_index": selected_round_index,
            "note": note,
        }
        emit("final_summary_recorded", "Final run summary written", payload)
        return json.dumps(payload)

    @tool(parse_docstring=True)
    def promote_checkpoint(round_index: int, note: str = "") -> str:
        """Promote the current checkpoint as the best run checkpoint.

        Args:
            round_index: The current round index.
            note: A short summary of why the checkpoint is being promoted.
        """
        rounds = repository.list_rounds(context.run_id)
        selected = next(item for item in rounds if item.round_index == round_index)
        repository.update_run(
            context.run_id,
            best_round_id=selected.id,
            summary=note or f"Promoted round {round_index}",
        )
        repository.promote_run(context.project_id, context.run_id)
        emit("checkpoint_promoted", "Best checkpoint promoted", {"round_index": round_index, "note": note})
        return json.dumps({"promoted_round_id": selected.id, "round_index": round_index, "note": note})

    @tool(parse_docstring=True)
    def classify_message(text: str, round_index: int | None = None) -> str:
        """Classify a support message with the best available checkpoint.

        Args:
            text: The input text to classify.
            round_index: Optional round index override.
        """
        run = repository.get_run(context.run_id)
        if round_index is None:
            rounds = repository.list_rounds(context.run_id)
            if not rounds:
                raise RuntimeError("No checkpoints available yet")
            round_index = rounds[-1].round_index
        checkpoint_dir = context.artifacts_root / context.run_id / f"round-{round_index:02d}"
        prediction = classify_text(text, checkpoint_dir)
        return prediction.model_dump_json()

    return [
        generate_candidates,
        create_holdout,
        run_round,
        evaluate_holdout,
        record_review,
        write_final_summary,
        promote_checkpoint,
        classify_message,
    ]


def _build_subagents() -> list[dict[str, Any]]:
    return [
        {
            "name": "ExperimentReviewer",
            "description": "Review evaluation results, analyze misclassifications, and write recommendations into /workspace/reviews.",
            "system_prompt": (
                "You review classifier evaluation reports. Read round evaluation files, identify systematic errors, "
                "and write concise Markdown recommendations to the requested /workspace review path. Recommend stopping "
                "when metrics meet target or when there is no concrete next dataset intervention."
            ),
            "tools": [],
        },
    ]


class DeepAgentRunner:
    def __init__(self, repository: Repository) -> None:
        self.repository = repository

    def _emit(self, run_id: str, event_type: str, message: str, payload: dict[str, Any] | None = None) -> None:
        self.repository.create_run_event(
            run_id,
            event_type=event_type,
            message=message,
            payload=payload or {},
        )

    def _build_round_comparison(self, run_id: str) -> list[dict[str, Any]]:
        return [self._summarize_round_for_selection(round_record) for round_record in self.repository.list_rounds(run_id)]

    def _summarize_round_for_selection(self, round_record) -> dict[str, Any]:
        return {
            "round_index": round_record.round_index,
            "status": round_record.status,
            "macro_f1": round_record.metrics.get("macro_f1"),
            "out_of_scope_precision": round_record.metrics.get("out_of_scope_precision"),
            "holdout_macro_f1": round_record.holdout_metrics.get("macro_f1"),
            "holdout_out_of_scope_precision": round_record.holdout_metrics.get("out_of_scope_precision"),
            "note": round_record.note,
            "review_file": round_record.review_file,
        }

    def _ensure_final_summary(self, run_record, best_round, rounds: int) -> str:
        summary_path = Path(run_record.workspace_root) / "reports" / "final-summary.md"
        if summary_path.exists():
            return summary_path.read_text()

        comparisons = self._build_round_comparison(run_record.id)
        comparison_lines = "\n".join(
            (
                f"- Round {item['round_index']}: "
                f"macro_f1={item['macro_f1']}, "
                f"oos_precision={item['out_of_scope_precision']}, "
                f"holdout_macro_f1={item['holdout_macro_f1']}, "
                f"status={item['status']}"
            )
            for item in comparisons
        )
        summary = (
            "# Final Run Summary\n\n"
            "## Outcome\n\n"
            f"- Selected round: {best_round.round_index}\n"
            f"- Completed round budget: {rounds}\n"
            f"- Best macro F1: {best_round.metrics.get('macro_f1')}\n"
            f"- Best out-of-scope precision: {best_round.metrics.get('out_of_scope_precision')}\n"
            f"- Holdout macro F1: {best_round.holdout_metrics.get('macro_f1')}\n\n"
            "## What Happened\n\n"
            f"{comparison_lines}\n\n"
            "## Why This Round Won\n\n"
            f"Round {best_round.round_index} was selected because it provided the strongest final metric profile "
            "available to the runner after all rounds completed. This fallback summary was generated by the backend "
            "because the agent did not write a custom final summary document.\n"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary)
        self._emit(
            run_record.id,
            "final_summary_recorded",
            "Final run summary written",
            {
                "summary_file": FINAL_SUMMARY_PATH,
                "selected_round_index": best_round.round_index,
                "note": "Fallback backend summary generated",
            },
        )
        return summary

    def execute_run(self, project_id: str, run_id: str, max_rounds: int | None = None) -> None:
        try:
            project = self.repository.get_project(project_id)
            self.repository.ensure_seed_minimums(project_id)
            self.repository.assign_locked_eval_split(project_id)
            self.repository.update_run(run_id, status=RunStatus.RUNNING, summary="Run started")
            self._emit(run_id, "run_started", "Run started", {"project_id": project_id})
            resolved_openai_api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not resolved_openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required to run Deep Agents orchestration")

            backend_factory, workspace = _build_backend(project_id, run_id, project.sandbox_profile)
            context = AgentContext(
                repository=self.repository,
                project_id=project_id,
                run_id=run_id,
                workspace=workspace,
                artifacts_root=settings.artifacts_dir,
            )
            tools = _make_tools(context)
            agent = create_deep_agent(
                model=ChatOpenAI(model=project.agent_model, api_key=resolved_openai_api_key),
                tools=tools,
                subagents=_build_subagents(),
                backend=backend_factory,
                checkpointer=InMemorySaver(),
                system_prompt=(
                    "You orchestrate bounded classifier experiments. Use the planning tool, filesystem, and subagents carefully. "
                    "Keep intermediate artifacts in /workspace and keep context concise."
                ),
            )
            rounds = max_rounds or project.max_rounds
            plan_text = (
                f"# Run Plan\n\n"
                f"- Project: {project.name}\n"
                f"- Objective: Classify texts as in_scope, out_of_scope, or ambiguous.\n"
                f"- Round budget: {rounds}\n"
                f"- Macro F1 target: {project.target_macro_f1}\n"
                f"- Out-of-scope precision target: {project.target_out_of_scope_precision}\n"
            )
            workspace.write_text("/workspace/plan.md", plan_text)
            self._emit(run_id, "plan_ready", "Strategy planned", {"round_budget": rounds})

            prompt = f"""
You are the orchestrator for a generic in-scope classifier experiment.

Project:
- name: {project.name}
- support_domain_description: {project.support_domain_description}
- allowed_topics: {project.allowed_topics}
- disallowed_topics: {project.disallowed_topics}
- routing_notes: {project.routing_notes}

Requirements:
1. Treat the initial human seed set as the default training dataset. Do not invent bulk augmentation unless there is a concrete failure mode to address.
2. For round 1, call generate_candidates with an empty focus_note so the fixed baseline dataset is preserved, then call run_round exactly once. Do not call any lower-level train/evaluate steps directly.
3. For later rounds, only call generate_candidates when the reviewer identifies a specific weakness or boundary issue worth targeting. If there is no concrete intervention, stop rather than generating more generic data.
4. After run_round, call create_holdout using the current dataset characteristics and a reasonable holdout size, then call evaluate_holdout on that file.
5. Ask ExperimentReviewer to read both the standard evaluation file and the holdout evaluation file, write a review to /workspace/reviews/round-XX-review.md, then call record_review.
6. Update your todo list and keep plan.md current.
7. Complete every planned round. Do not stop early, even if a target is met or the first two rounds look strong.
8. Only after all {rounds} rounds are complete, compare the finished rounds, choose the single strongest checkpoint, write a full Markdown summary to /workspace/reports/final-summary.md using write_final_summary, and then promote that checkpoint.
9. The final summary must include: a round-by-round timeline, key metrics for each round, the main intervention or focus note per round when available, and a direct explanation of why the selected round beat the other completed rounds.
10. The final decision must explain why the selected round beats the other completed rounds.

Use round numbers starting at 1 and keep summaries concise.
"""

            config = {"configurable": {"thread_id": run_id}}
            for _ in agent.stream({"messages": [{"role": "user", "content": prompt}]}, config, stream_mode="updates"):
                pass
            run_record = self.repository.get_run(run_id)
            best_round = None
            if run_record.best_round_id:
                best_round = next(
                    (
                        round_record
                        for round_record in self.repository.list_rounds(run_id)
                        if round_record.id == run_record.best_round_id
                    ),
                    None,
                )
            if best_round is None:
                best_round = self._pick_best_round(run_id)
            if best_round:
                round_comparison = self._build_round_comparison(run_id)
                self._emit(
                    run_id,
                    "run_reviewed",
                    f"All {rounds} rounds complete. Selecting the strongest checkpoint.",
                    {
                        "best_round_index": best_round.round_index,
                        "macro_f1": best_round.metrics.get("macro_f1"),
                        "holdout_macro_f1": best_round.holdout_metrics.get("macro_f1"),
                        "round_budget": rounds,
                        "round_comparison": round_comparison,
                    },
                )
                self._ensure_final_summary(run_record, best_round, rounds)
                self.repository.update_run(
                    run_id,
                    status=RunStatus.COMPLETED,
                    stop_reason="Completed by Deep Agent",
                    best_round_id=best_round.id,
                    best_macro_f1=best_round.metrics.get("macro_f1"),
                    summary=f"Best round {best_round.round_index}",
                )
                self.repository.promote_run(project_id, run_id)
                self._emit(
                    run_id,
                    "run_completed",
                    f"Round {best_round.round_index} selected after {rounds} rounds and promoted to production",
                    {
                        "best_round_index": best_round.round_index,
                        "best_macro_f1": best_round.metrics.get("macro_f1"),
                        "holdout_macro_f1": best_round.holdout_metrics.get("macro_f1"),
                        "summary_file": FINAL_SUMMARY_PATH,
                        "round_comparison": round_comparison,
                        "round_budget": rounds,
                    },
                )
            else:
                self.repository.update_run(
                    run_id,
                    status=RunStatus.COMPLETED,
                    stop_reason="No completed rounds were produced",
                    summary="Run completed without a promotable checkpoint",
                )
                self._emit(run_id, "run_completed", "Run completed without a promotable checkpoint")
        except Exception as exc:
            self.repository.update_run(
                run_id,
                status=RunStatus.FAILED,
                stop_reason=str(exc),
                summary="Run failed",
            )
            self._emit(run_id, "run_failed", "Run failed", {"error": str(exc)})
            raise

    def _pick_best_round(self, run_id: str):
        rounds = [item for item in self.repository.list_rounds(run_id) if item.metrics]
        if not rounds:
            return None
        return max(
            rounds,
            key=lambda item: (
                item.metrics.get("macro_f1", 0.0),
                item.metrics.get("out_of_scope_precision", 0.0),
                item.metrics.get("per_class", {}).get("ambiguous", {}).get("precision", 0.0),
            ),
        )
