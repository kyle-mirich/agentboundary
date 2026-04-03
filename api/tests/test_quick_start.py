import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.models import ExampleInput, ExampleSource, Label

SESSION_HEADERS = {"X-Session-Id": "test-session"}

MOCK_SEEDS = (
    [ExampleInput(text=f"billing {i}", label=Label.IN_SCOPE, source=ExampleSource.HUMAN_SEED) for i in range(30)]
    + [ExampleInput(text=f"off topic {i}", label=Label.OUT_OF_SCOPE, source=ExampleSource.HUMAN_SEED) for i in range(30)]
    + [ExampleInput(text=f"mixed {i}", label=Label.AMBIGUOUS, source=ExampleSource.HUMAN_SEED) for i in range(30)]
)


def test_quick_start_returns_project_and_run_ids():
    client = TestClient(app)
    with (
        patch("app.main.generate_seeds", return_value=MOCK_SEEDS),
        patch("app.main.runner.execute_run"),
    ):
        response = client.post(
            "/quick-start",
            headers=SESSION_HEADERS,
            json={"description": "classify customer support tickets"},
        )
    assert response.status_code == 200
    body = response.json()
    assert "project_id" in body
    assert "run_id" in body
    assert isinstance(body["project_id"], str)
    assert isinstance(body["run_id"], str)


def test_quick_start_creates_project_with_description():
    client = TestClient(app)
    description = "route SaaS billing support messages"
    with (
        patch("app.main.generate_seeds", return_value=MOCK_SEEDS),
        patch("app.main.runner.execute_run"),
    ):
        response = client.post(
            "/quick-start",
            headers=SESSION_HEADERS,
            json={"description": description},
        )
    project_id = response.json()["project_id"]

    detail = client.get(f"/projects/{project_id}", headers=SESSION_HEADERS)
    assert detail.status_code == 200
    assert description in detail.json()["project"]["support_domain_description"]
    assert detail.json()["project"]["max_rounds"] == 3


def test_quick_start_enforces_three_round_run_budget():
    client = TestClient(app)
    with (
        patch("app.main.generate_seeds", return_value=MOCK_SEEDS),
        patch("app.main.runner.execute_run") as execute_run,
    ):
        response = client.post(
            "/quick-start",
            headers=SESSION_HEADERS,
            json={"description": "classify customer support tickets"},
        )

    assert response.status_code == 200
    execute_run.assert_called_once()
    _, _, max_rounds = execute_run.call_args.args
    assert max_rounds == 3


def test_quick_start_logs_real_seed_generation_events():
    client = TestClient(app)
    with (
        patch("app.main.generate_seeds", return_value=MOCK_SEEDS),
        patch("app.main.runner.execute_run"),
    ):
        response = client.post(
            "/quick-start",
            headers=SESSION_HEADERS,
            json={"description": "classify customer support tickets"},
        )

    run_id = response.json()["run_id"]
    events = client.get(f"/runs/{run_id}/events", headers=SESSION_HEADERS).json()
    event_types = [event["event_type"] for event in events]

    assert "seed_generation_started" in event_types
    assert "seed_generation_completed" in event_types
    assert "seed_generation_label_count" in event_types


def test_quick_start_adds_90_examples():
    client = TestClient(app)
    with (
        patch("app.main.generate_seeds", return_value=MOCK_SEEDS),
        patch("app.main.runner.execute_run"),
    ):
        response = client.post(
            "/quick-start",
            headers=SESSION_HEADERS,
            json={"description": "classify support tickets"},
        )
    project_id = response.json()["project_id"]

    detail = client.get(f"/projects/{project_id}", headers=SESSION_HEADERS)
    examples = detail.json()["examples"]
    assert len(examples) == 90


def test_quick_start_seed_failure_marks_run_failed():
    client = TestClient(app)
    with patch("app.main.generate_seeds", side_effect=RuntimeError("Seed generation failed")):
        response = client.post(
            "/quick-start",
            headers=SESSION_HEADERS,
            json={"description": "classify tickets"},
        )
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    run_detail = client.get(f"/runs/{run_id}", headers=SESSION_HEADERS)
    assert run_detail.status_code == 200
    assert run_detail.json()["status"] == "failed"
    assert run_detail.json()["stop_reason"] == "Seed generation failed"


def test_quick_start_requires_session_id():
    client = TestClient(app)
    response = client.post("/quick-start", json={"description": "test"})
    assert response.status_code == 400


@pytest.mark.no_db
def test_quick_start_lucky_returns_description():
    client = TestClient(app)
    with patch("app.main.generate_lucky_description", return_value="Route SaaS support messages by billing relevance."):
        response = client.post("/quick-start/lucky")
    assert response.status_code == 200
    assert response.json() == {"description": "Route SaaS support messages by billing relevance."}
