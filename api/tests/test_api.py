from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

SESSION_HEADERS = {"X-Session-Id": "test-session"}


def test_project_creation_and_listing():
    client = TestClient(app)
    create_response = client.post(
        "/projects",
        headers=SESSION_HEADERS,
        json={
            "name": "Acme Support",
            "support_domain_description": "Support for SaaS billing and account issues.",
            "allowed_topics": ["billing", "refunds", "accounts"],
            "disallowed_topics": ["coding help", "astronomy"],
            "routing_notes": "Keep it limited to customer support.",
            "max_rounds": 2,
            "target_macro_f1": 0.8,
            "target_out_of_scope_precision": 0.9,
            "sandbox_profile": "isolated_fs",
        },
    )
    assert create_response.status_code == 200
    project_id = create_response.json()["id"]

    examples_response = client.post(
        f"/projects/{project_id}/examples",
        headers=SESSION_HEADERS,
        json=[
            {"text": f"billing issue {index}", "label": "in_scope"} for index in range(5)
        ]
        + [{"text": f"astronomy {index}", "label": "out_of_scope"} for index in range(5)]
        + [{"text": f"mixed ask {index}", "label": "ambiguous"} for index in range(3)],
    )
    assert examples_response.status_code == 200

    list_response = client.get("/projects", headers=SESSION_HEADERS)
    assert list_response.status_code == 200
    assert any(item["id"] == project_id for item in list_response.json())


def test_promote_run_rejects_run_from_another_project():
    client = TestClient(app)
    project_a = client.post(
        "/projects",
        headers=SESSION_HEADERS,
        json={
            "name": "Project A",
            "support_domain_description": "Billing support",
        },
    ).json()
    project_b = client.post(
        "/projects",
        headers=SESSION_HEADERS,
        json={
            "name": "Project B",
            "support_domain_description": "Account support",
        },
    ).json()

    client.post(
        f"/projects/{project_b['id']}/examples",
        headers=SESSION_HEADERS,
        json=[
            {"text": f"account issue {index}", "label": "in_scope"} for index in range(5)
        ]
        + [{"text": f"gardening {index}", "label": "out_of_scope"} for index in range(5)]
        + [{"text": f"mixed account ask {index}", "label": "ambiguous"} for index in range(3)],
    )
    with patch("app.main.runner.execute_run"):
        run_response = client.post(
            f"/projects/{project_b['id']}/runs",
            headers=SESSION_HEADERS,
            json={},
        )
    run_id = run_response.json()["id"]

    promote_response = client.post(
        f"/projects/{project_a['id']}/promote/{run_id}",
        headers=SESSION_HEADERS,
    )

    assert promote_response.status_code == 400
    assert promote_response.json()["detail"] == "Run does not belong to this project"
