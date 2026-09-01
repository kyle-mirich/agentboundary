from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import config as config_module
from app.main import app, repository, run_rate_limiter
from app.models import ExampleInput, Label, RunStatus

SESSION_HEADERS = {"X-Session-Id": "test-session"}

SEED_PAYLOAD = (
    [{"text": f"billing issue {index}", "label": "in_scope"} for index in range(5)]
    + [{"text": f"astronomy {index}", "label": "out_of_scope"} for index in range(5)]
    + [{"text": f"mixed ask {index}", "label": "ambiguous"} for index in range(3)]
)


def _create_project(client: TestClient, name: str = "Billing Gate") -> str:
    response = client.post(
        "/projects",
        headers=SESSION_HEADERS,
        json={
            "name": name,
            "support_domain_description": "Allowed / in scope: SaaS billing questions.",
        },
    )
    assert response.status_code == 200
    project_id = response.json()["id"]
    assert client.post(
        f"/projects/{project_id}/examples", headers=SESSION_HEADERS, json=SEED_PAYLOAD
    ).status_code == 200
    return project_id


def test_run_endpoint_is_rate_limited_per_session():
    client = TestClient(app)
    project_id = _create_project(client)
    original_limit = config_module.settings.run_rate_limit
    config_module.settings.run_rate_limit = 1
    run_rate_limiter.clear()
    try:
        with patch("app.main.runner.execute_run"):
            assert client.post(
                f"/projects/{project_id}/runs", headers=SESSION_HEADERS, json={}
            ).status_code == 200
            second = client.post(f"/projects/{project_id}/runs", headers=SESSION_HEADERS, json={})
        assert second.status_code == 429
    finally:
        config_module.settings.run_rate_limit = original_limit
        run_rate_limiter.clear()


def test_run_endpoint_rejects_a_second_concurrent_run():
    client = TestClient(app)
    project_id = _create_project(client)
    with patch("app.main.runner.execute_run"):
        assert client.post(
            f"/projects/{project_id}/runs", headers=SESSION_HEADERS, json={}
        ).status_code == 200
        # The first run is still queued, so a second one must be refused rather
        # than started alongside it.
        second = client.post(f"/projects/{project_id}/runs", headers=SESSION_HEADERS, json={})
    assert second.status_code == 409


def test_completed_run_does_not_block_a_new_run():
    client = TestClient(app)
    project_id = _create_project(client)
    with patch("app.main.runner.execute_run"):
        first = client.post(f"/projects/{project_id}/runs", headers=SESSION_HEADERS, json={})
        run_id = first.json()["id"]
        repository.update_run(run_id, status=RunStatus.COMPLETED)
        second = client.post(f"/projects/{project_id}/runs", headers=SESSION_HEADERS, json={})
    assert second.status_code == 200


def test_each_run_gets_its_own_workspace():
    client = TestClient(app)
    project_id = _create_project(client)
    with patch("app.main.runner.execute_run"):
        first = client.post(f"/projects/{project_id}/runs", headers=SESSION_HEADERS, json={}).json()
        repository.update_run(first["id"], status=RunStatus.COMPLETED)
        second = client.post(f"/projects/{project_id}/runs", headers=SESSION_HEADERS, json={}).json()

    first_root = repository.get_run(first["id"]).workspace_root
    second_root = repository.get_run(second["id"]).workspace_root
    assert first_root != second_root
    # The placeholder "pending" directory must never survive: every run that
    # shared it could read another run's plan, review and summary artifacts.
    assert "pending" not in first_root
    assert "pending" not in second_root


def test_reconcile_interrupted_runs_fails_runs_without_progress():
    client = TestClient(app)
    project_id = _create_project(client)
    run = repository.create_run(project_id, str(config_module.settings.workspace_dir / "pending"))
    repository.update_run(run.id, status=RunStatus.RUNNING)

    reclaimed = repository.reconcile_interrupted_runs(max_age_seconds=0)

    assert reclaimed == [run.id]
    assert repository.get_run(run.id).status == RunStatus.FAILED


def test_reconcile_interrupted_runs_leaves_recent_runs_alone():
    client = TestClient(app)
    project_id = _create_project(client)
    run = repository.create_run(project_id, str(config_module.settings.workspace_dir / "pending"))
    repository.update_run(run.id, status=RunStatus.RUNNING)

    reclaimed = repository.reconcile_interrupted_runs(max_age_seconds=3600)

    assert reclaimed == []
    assert repository.get_run(run.id).status == RunStatus.RUNNING


def test_stream_closes_instead_of_polling_forever():
    client = TestClient(app)
    project_id = _create_project(client)
    with patch("app.main.runner.execute_run"):
        run_id = client.post(
            f"/projects/{project_id}/runs", headers=SESSION_HEADERS, json={}
        ).json()["id"]
    original_max = config_module.settings.run_stream_max_seconds
    config_module.settings.run_stream_max_seconds = 0
    try:
        with client.stream(
            "GET", f"/runs/{run_id}/events/stream", headers=SESSION_HEADERS
        ) as response:
            body = "".join(response.iter_text())
    finally:
        config_module.settings.run_stream_max_seconds = original_max

    # A run that never reaches a terminal state must not keep the stream open.
    assert "event: stream_closed" in body
    assert "event: run_done" not in body


def test_add_examples_rejects_oversized_batches():
    client = TestClient(app)
    project_id = _create_project(client)
    oversized = [
        {"text": f"example {index}", "label": "in_scope"}
        for index in range(config_module.settings.max_examples_per_request + 1)
    ]

    response = client.post(
        f"/projects/{project_id}/examples", headers=SESSION_HEADERS, json=oversized
    )

    assert response.status_code == 422


def test_add_examples_accepts_a_full_batch():
    client = TestClient(app)
    project_id = _create_project(client)
    batch = [
        {"text": f"example {index}", "label": "in_scope"}
        for index in range(config_module.settings.max_examples_per_request)
    ]

    response = client.post(
        f"/projects/{project_id}/examples", headers=SESSION_HEADERS, json=batch
    )

    assert response.status_code == 200


def test_session_id_is_rejected_as_a_query_parameter():
    client = TestClient(app)

    response = client.get("/projects?session_id=test-session")

    # Accepting the session id in a URL leaks the value that scopes a project
    # into history, proxy logs and Referer headers.
    assert response.status_code == 400


def test_session_id_header_still_works():
    client = TestClient(app)
    assert client.get("/projects", headers=SESSION_HEADERS).status_code == 200


def test_seed_generation_uses_the_generation_model():
    from app.seed_generator import generate_seeds

    captured: dict[str, object] = {}
    seeds = [
        {"text": f"text {index} example", "label": label}
        for label in ("in_scope", "out_of_scope", "ambiguous")
        for index in range(config_module.settings.seed_examples_per_label)
    ]

    with (
        patch.object(config_module.settings, "openai_api_key", "test-key"),
        patch("app.seed_generator.OpenAI") as mock_cls,
    ):
        create = mock_cls.return_value.chat.completions.create

        def capture(**kwargs):
            captured.update(kwargs)
            response = type("R", (), {})()
            message = type("M", (), {"content": __import__("json").dumps(seeds)})()
            response.choices = [type("C", (), {"message": message})()]
            return response

        create.side_effect = capture
        result = generate_seeds("classify customer support tickets")

    assert captured["model"] == config_module.settings.responses_generation_model
    assert len(result) == config_module.settings.seed_examples_per_label * 3


def test_seed_generation_truncates_oversized_responses():
    from app.seed_generator import generate_seeds

    per_label = config_module.settings.seed_examples_per_label
    oversized = [
        {"text": f"text {index} example", "label": label}
        for label in ("in_scope", "out_of_scope", "ambiguous")
        for index in range(per_label + 20)
    ]

    with (
        patch.object(config_module.settings, "openai_api_key", "test-key"),
        patch("app.seed_generator.OpenAI") as mock_cls,
    ):
        response = type("R", (), {})()
        message = type("M", (), {"content": __import__("json").dumps(oversized)})()
        response.choices = [type("C", (), {"message": message})()]
        mock_cls.return_value.chat.completions.create.return_value = response
        result = generate_seeds("classify customer support tickets")

    assert len(result) == per_label * 3


def test_seed_generation_retries_when_a_label_is_short():
    from app.seed_generator import generate_seeds

    per_label = config_module.settings.seed_examples_per_label

    def payload(count):
        return [
            {"text": f"text {index} example", "label": label}
            for label in ("in_scope", "out_of_scope", "ambiguous")
            for index in range(count)
        ]

    with (
        patch.object(config_module.settings, "openai_api_key", "test-key"),
        patch("app.seed_generator.OpenAI") as mock_cls,
    ):
        responses = []
        for count in (1, per_label):
            response = type("R", (), {})()
            message = type("M", (), {"content": __import__("json").dumps(payload(count))})()
            response.choices = [type("C", (), {"message": message})()]
            responses.append(response)
        mock_cls.return_value.chat.completions.create.side_effect = responses
        result = generate_seeds("classify customer support tickets")

    assert len(result) == per_label * 3
    assert mock_cls.return_value.chat.completions.create.call_count == 2


def test_seed_generation_fails_when_labels_stay_short():
    from app.seed_generator import generate_seeds

    short = [{"text": "only one example", "label": "in_scope"}]

    with (
        patch.object(config_module.settings, "openai_api_key", "test-key"),
        patch("app.seed_generator.OpenAI") as mock_cls,
    ):
        response = type("R", (), {})()
        message = type("M", (), {"content": __import__("json").dumps(short)})()
        response.choices = [type("C", (), {"message": message})()]
        mock_cls.return_value.chat.completions.create.return_value = response
        with pytest.raises(RuntimeError, match="Seed generation failed"):
            generate_seeds("classify customer support tickets")
