from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app, repository
from app.models import ClassificationResponse, RunStatus

SESSION_HEADERS = {"X-Session-Id": "test-session"}


def test_healthcheck_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_project_run_stream_and_classify_endpoints(tmp_path):
    client = TestClient(app)

    create_project_response = client.post(
        "/projects",
        headers=SESSION_HEADERS,
        json={
            "name": "Billing Gate",
            "support_domain_description": "Allowed / in scope: SaaS billing questions.",
        },
    )
    assert create_project_response.status_code == 200
    project_id = create_project_response.json()["id"]

    add_examples_response = client.post(
        f"/projects/{project_id}/examples",
        headers=SESSION_HEADERS,
        json=[
            {"text": f"billing issue {index}", "label": "in_scope"} for index in range(5)
        ]
        + [{"text": f"astronomy {index}", "label": "out_of_scope"} for index in range(5)]
        + [{"text": f"mixed ask {index}", "label": "ambiguous"} for index in range(3)],
    )
    assert add_examples_response.status_code == 200

    list_projects_response = client.get("/projects", headers=SESSION_HEADERS)
    assert list_projects_response.status_code == 200
    assert any(item["id"] == project_id for item in list_projects_response.json())

    get_project_response = client.get(f"/projects/{project_id}", headers=SESSION_HEADERS)
    assert get_project_response.status_code == 200
    assert get_project_response.json()["project"]["id"] == project_id

    with patch("app.main.runner.execute_run"):
        create_run_response = client.post(
            f"/projects/{project_id}/runs",
            headers=SESSION_HEADERS,
            json={},
        )
    assert create_run_response.status_code == 200
    run_id = create_run_response.json()["id"]

    list_runs_response = client.get(f"/projects/{project_id}/runs", headers=SESSION_HEADERS)
    assert list_runs_response.status_code == 200
    assert any(item["id"] == run_id for item in list_runs_response.json())

    get_run_response = client.get(f"/runs/{run_id}", headers=SESSION_HEADERS)
    assert get_run_response.status_code == 200
    assert get_run_response.json()["id"] == run_id

    get_events_response = client.get(f"/runs/{run_id}/events", headers=SESSION_HEADERS)
    assert get_events_response.status_code == 200
    assert get_events_response.json() == []

    repository.create_run_event(
        run_id,
        event_type="round_started",
        message="Round 1 started",
        payload={"round_index": 1},
    )
    repository.update_run(run_id, status=RunStatus.COMPLETED)

    with client.stream("GET", f"/runs/{run_id}/events/stream", headers=SESSION_HEADERS) as response:
        stream_body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: run_event" in stream_body
    assert "event: run_done" in stream_body
    assert "Round 1 started" in stream_body

    promote_response = client.post(
        f"/projects/{project_id}/promote/{run_id}",
        headers=SESSION_HEADERS,
    )
    assert promote_response.status_code == 200
    assert promote_response.json()["promoted_run_id"] == run_id

    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_text("stub checkpoint")
    round_record = repository.create_round(run_id, 1, "candidate.py")
    repository.update_round(round_record.id, checkpoint_path=str(checkpoint_path))
    repository.update_run(run_id, best_round_id=round_record.id, status=RunStatus.COMPLETED)

    with patch(
        "app.main.classify_text",
        return_value=ClassificationResponse(
            label="in_scope",
            confidence=0.97,
            probabilities={"in_scope": 0.97, "out_of_scope": 0.02, "ambiguous": 0.01},
            explanation="The request is clearly about billing support.",
        ),
    ) as classify_text:
        classify_response = client.post(
            f"/projects/{project_id}/classify",
            headers=SESSION_HEADERS,
            json={"text": "Can I get a refund for my plan?"},
        )

    assert classify_response.status_code == 200
    assert classify_response.json()["label"] == "in_scope"
    classify_text.assert_called_once()
