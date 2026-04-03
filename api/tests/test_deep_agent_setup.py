import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app import config as config_module
from app.database import get_connection, utc_now
from app.deep_agent import AgentContext, DeepAgentRunner, LocalWorkspaceIO, _make_tools
from app.models import ExampleInput, Label, ProjectCreate
from app.repository import Repository


class StubAgent:
    def __init__(self, on_stream=None) -> None:
        self.stream_calls: list[tuple[dict, dict, str]] = []
        self.on_stream = on_stream

    def stream(self, payload: dict, config: dict, stream_mode: str = "updates"):
        self.stream_calls.append((payload, config, stream_mode))
        if self.on_stream is not None:
            self.on_stream()
        yield {"event": "noop"}


def set_run_workspace(run_id: str) -> None:
    workspace_root = str(config_module.settings.workspace_dir / run_id)
    with get_connection() as connection:
        connection.execute(
            "UPDATE runs SET workspace_root = %s, updated_at = %s WHERE id = %s",
            (workspace_root, utc_now(), run_id),
        )


@pytest.mark.no_db
def test_project_create_uses_configured_default_agent_model():
    with patch.object(config_module.settings, "default_agent_model", "gpt-test-agent"):
        project = ProjectCreate(
            name="Support",
            support_domain_description="Classify support requests",
        )

    assert project.agent_model == "gpt-test-agent"


def test_runner_uses_run_id_thread_id_and_persistent_memory_route():
    repository = Repository()
    project = repository.create_project(
        ProjectCreate(
            name="Support",
            support_domain_description="Billing and account support",
            allowed_topics=["billing"],
            disallowed_topics=["astronomy"],
            routing_notes="",
            sandbox_profile="isolated_fs",
        ),
        session_id="test-session",
    )
    repository.add_examples(
        project.id,
        [ExampleInput(text=f"billing {index}", label=Label.IN_SCOPE) for index in range(5)]
        + [ExampleInput(text=f"astronomy {index}", label=Label.OUT_OF_SCOPE) for index in range(5)]
        + [ExampleInput(text=f"mixed {index}", label=Label.AMBIGUOUS) for index in range(3)],
    )
    run = repository.create_run(project.id, str(config_module.settings.workspace_dir / "pending"))
    set_run_workspace(run.id)
    runner = DeepAgentRunner(repository)
    stub_agent = StubAgent()

    with (
        patch.object(config_module.settings, "openai_api_key", "test-key"),
        patch("app.deep_agent.create_deep_agent", return_value=stub_agent) as create_agent,
    ):
        runner.execute_run(project.id, run.id, max_rounds=1)

    kwargs = create_agent.call_args.kwargs
    backend = kwargs["backend"](SimpleNamespace())

    assert stub_agent.stream_calls
    assert stub_agent.stream_calls[0][1]["configurable"]["thread_id"] == run.id
    assert "/workspace/" in backend.routes
    assert "/memories/" in backend.routes


def test_runner_preserves_model_selected_best_round():
    repository = Repository()
    project = repository.create_project(
        ProjectCreate(
            name="Support",
            support_domain_description="Billing and account support",
            sandbox_profile="isolated_fs",
        ),
        session_id="test-session",
    )
    repository.add_examples(
        project.id,
        [ExampleInput(text=f"billing {index}", label=Label.IN_SCOPE) for index in range(5)]
        + [ExampleInput(text=f"astronomy {index}", label=Label.OUT_OF_SCOPE) for index in range(5)]
        + [ExampleInput(text=f"mixed ask {index}", label=Label.AMBIGUOUS) for index in range(3)],
    )
    run = repository.create_run(project.id, str(config_module.settings.workspace_dir / "pending"))
    set_run_workspace(run.id)
    runner = DeepAgentRunner(repository)

    def mark_round_three_as_selected() -> None:
        round_one = repository.create_round(run.id, 1, "/workspace/datasets/round-01-candidates.jsonl")
        round_three = repository.create_round(run.id, 3, "/workspace/datasets/round-03-candidates.jsonl")
        repository.update_round(
            round_one.id,
            status="completed",
            metrics={"macro_f1": 1.0, "out_of_scope_precision": 1.0, "per_class": {}},
            holdout_metrics={"macro_f1": 0.46},
        )
        repository.update_round(
            round_three.id,
            status="completed",
            metrics={"macro_f1": 0.8, "out_of_scope_precision": 0.8, "per_class": {}},
            holdout_metrics={"macro_f1": 0.82},
        )
        repository.update_run(run.id, best_round_id=round_three.id, summary="Promoted round 3")
        repository.promote_run(project.id, run.id)

    stub_agent = StubAgent(on_stream=mark_round_three_as_selected)

    with (
        patch.object(config_module.settings, "openai_api_key", "test-key"),
        patch("app.deep_agent.create_deep_agent", return_value=stub_agent),
        patch.object(DeepAgentRunner, "_pick_best_round", return_value=None),
    ):
        runner.execute_run(project.id, run.id, max_rounds=3)

    completed_run = repository.get_run(run.id)
    selected_round = repository.get_round(completed_run.best_round_id)

    assert completed_run.best_round_id == selected_round.id
    assert selected_round.round_index == 3


def test_runner_writes_final_summary_when_agent_does_not():
    repository = Repository()
    project = repository.create_project(
        ProjectCreate(
            name="Support",
            support_domain_description="Billing and account support",
            sandbox_profile="isolated_fs",
        ),
        session_id="test-session",
    )
    repository.add_examples(
        project.id,
        [ExampleInput(text=f"billing {index}", label=Label.IN_SCOPE) for index in range(5)]
        + [ExampleInput(text=f"astronomy {index}", label=Label.OUT_OF_SCOPE) for index in range(5)]
        + [ExampleInput(text=f"mixed ask {index}", label=Label.AMBIGUOUS) for index in range(3)],
    )
    run = repository.create_run(project.id, str(config_module.settings.workspace_dir / "pending"))
    set_run_workspace(run.id)
    runner = DeepAgentRunner(repository)

    def mark_completed_rounds_without_summary() -> None:
        round_one = repository.create_round(run.id, 1, "/workspace/datasets/round-01-candidates.jsonl")
        repository.update_round(
            round_one.id,
            status="completed",
            metrics={"macro_f1": 0.91, "out_of_scope_precision": 0.94, "per_class": {}},
            holdout_metrics={"macro_f1": 0.89, "out_of_scope_precision": 0.92},
            note="Best balance across eval and holdout",
        )

    stub_agent = StubAgent(on_stream=mark_completed_rounds_without_summary)

    with (
        patch.object(config_module.settings, "openai_api_key", "test-key"),
        patch("app.deep_agent.create_deep_agent", return_value=stub_agent),
    ):
        runner.execute_run(project.id, run.id, max_rounds=1)

    run_record = repository.get_run(run.id)
    summary_path = config_module.settings.workspace_dir / run.id / "reports" / "final-summary.md"
    events = repository.list_run_events(run.id)

    assert run_record.status.value == "completed"
    assert summary_path.exists()
    assert "Final Run Summary" in summary_path.read_text()
    assert any(event.event_type == "final_summary_recorded" for event in events)


def test_generate_candidates_keeps_fixed_seed_baseline_without_focus_note():
    repository = Repository()
    project = repository.create_project(
        ProjectCreate(
            name="Support",
            support_domain_description="Billing and account support",
            allowed_topics=["billing"],
            disallowed_topics=["astronomy"],
            routing_notes="",
            sandbox_profile="isolated_fs",
        ),
        session_id="test-session",
    )
    repository.add_examples(
        project.id,
        [ExampleInput(text=f"billing {index}", label=Label.IN_SCOPE) for index in range(5)]
        + [ExampleInput(text=f"astronomy {index}", label=Label.OUT_OF_SCOPE) for index in range(5)]
        + [ExampleInput(text=f"mixed {index}", label=Label.AMBIGUOUS) for index in range(3)],
    )
    run = repository.create_run(project.id, str(config_module.settings.workspace_dir / "pending"))
    workspace_root = config_module.settings.workspace_dir / run.id
    workspace = LocalWorkspaceIO(workspace_root)
    workspace.ensure()
    context = AgentContext(
        repository=repository,
        project_id=project.id,
        run_id=run.id,
        workspace=workspace,
        artifacts_root=config_module.settings.artifacts_dir,
    )

    with patch.object(config_module.settings, "openai_api_key", "test-key"):
        tools = _make_tools(context)

    generate_candidates = next(tool for tool in tools if tool.name == "generate_candidates")
    raw_result = generate_candidates.invoke({"round_index": 1, "focus_note": ""})
    result = json.loads(raw_result)
    candidate_path = workspace_root / "datasets" / "round-01-candidates.jsonl"

    assert result["generated_count"] == 0
    assert result["mode"] == "fixed_seed_baseline"
    assert candidate_path.exists()
    assert candidate_path.read_text() == ""


def test_run_round_uses_recorded_candidate_file_when_agent_passes_wrong_path():
    repository = Repository()
    project = repository.create_project(
        ProjectCreate(
            name="Support",
            support_domain_description="Billing and account support",
            allowed_topics=["billing"],
            disallowed_topics=["astronomy"],
            routing_notes="",
            sandbox_profile="isolated_fs",
        ),
        session_id="test-session",
    )
    repository.add_examples(
        project.id,
        [ExampleInput(text=f"billing {index}", label=Label.IN_SCOPE) for index in range(5)]
        + [ExampleInput(text=f"astronomy {index}", label=Label.OUT_OF_SCOPE) for index in range(5)]
        + [ExampleInput(text=f"mixed {index}", label=Label.AMBIGUOUS) for index in range(3)],
    )
    repository.assign_locked_eval_split(project.id)
    run = repository.create_run(project.id, str(config_module.settings.workspace_dir / "pending"))
    workspace_root = config_module.settings.workspace_dir / run.id
    workspace = LocalWorkspaceIO(workspace_root)
    workspace.ensure()
    context = AgentContext(
        repository=repository,
        project_id=project.id,
        run_id=run.id,
        workspace=workspace,
        artifacts_root=config_module.settings.artifacts_dir,
    )

    with patch.object(config_module.settings, "openai_api_key", "test-key"):
        tools = _make_tools(context)

    generate_candidates = next(tool for tool in tools if tool.name == "generate_candidates")
    run_round = next(tool for tool in tools if tool.name == "run_round")

    generate_candidates.invoke({"round_index": 1, "focus_note": ""})

    with patch("app.deep_agent.train_model") as train_model_mock, patch("app.deep_agent.evaluate_model") as evaluate_model_mock:
        train_model_mock.return_value = SimpleNamespace(
            checkpoint_path=config_module.settings.artifacts_dir / run.id / "round-01",
            training_loss=0.12,
            train_count=10,
        )
        evaluate_model_mock.return_value = {
            "macro_f1": 0.81,
            "out_of_scope_precision": 0.9,
            "per_class": {},
            "confusion_matrix": [],
            "misclassified": [],
            "eval_count": 3,
        }
        raw_result = run_round.invoke(
            {
                "round_index": 1,
                "candidate_file": "/workspace/candidates_round_1.jsonl",
            }
        )

    result = json.loads(raw_result)
    assert result["dataset_summary"]["accepted_examples"] == 0
    assert result["candidate_file"] == "/workspace/datasets/round-01-candidates.jsonl"


def test_run_round_creates_round_one_baseline_when_agent_skips_generate_candidates():
    repository = Repository()
    project = repository.create_project(
        ProjectCreate(
            name="Support",
            support_domain_description="Billing and account support",
            allowed_topics=["billing"],
            disallowed_topics=["astronomy"],
            routing_notes="",
            sandbox_profile="isolated_fs",
        ),
        session_id="test-session",
    )
    repository.add_examples(
        project.id,
        [ExampleInput(text=f"billing {index}", label=Label.IN_SCOPE) for index in range(5)]
        + [ExampleInput(text=f"astronomy {index}", label=Label.OUT_OF_SCOPE) for index in range(5)]
        + [ExampleInput(text=f"mixed {index}", label=Label.AMBIGUOUS) for index in range(3)],
    )
    repository.assign_locked_eval_split(project.id)
    run = repository.create_run(project.id, str(config_module.settings.workspace_dir / "pending"))
    workspace_root = config_module.settings.workspace_dir / run.id
    workspace = LocalWorkspaceIO(workspace_root)
    workspace.ensure()
    context = AgentContext(
        repository=repository,
        project_id=project.id,
        run_id=run.id,
        workspace=workspace,
        artifacts_root=config_module.settings.artifacts_dir,
    )

    with patch.object(config_module.settings, "openai_api_key", "test-key"):
        tools = _make_tools(context)

    run_round = next(tool for tool in tools if tool.name == "run_round")

    with patch("app.deep_agent.train_model") as train_model_mock, patch("app.deep_agent.evaluate_model") as evaluate_model_mock:
        train_model_mock.return_value = SimpleNamespace(
            checkpoint_path=config_module.settings.artifacts_dir / run.id / "round-01",
            training_loss=0.12,
            train_count=10,
        )
        evaluate_model_mock.return_value = {
            "macro_f1": 0.81,
            "out_of_scope_precision": 0.9,
            "per_class": {},
            "confusion_matrix": [],
            "misclassified": [],
            "eval_count": 3,
        }
        raw_result = run_round.invoke(
            {
                "round_index": 1,
                "candidate_file": "/workspace/round-1-candidates.jsonl",
            }
        )

    result = json.loads(raw_result)
    baseline_path = workspace_root / "datasets" / "round-01-candidates.jsonl"

    assert baseline_path.exists()
    assert baseline_path.read_text() == ""
    assert result["dataset_summary"]["accepted_examples"] == 0
    assert result["candidate_file"] == "/workspace/datasets/round-01-candidates.jsonl"
