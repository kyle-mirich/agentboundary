# Interview Demo UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dashboard UI with a single-flow Apple-minimal experience: one prompt → GPT-5 generates seeds → live training → test your classifier.

**Architecture:** A new `POST /quick-start` backend endpoint handles the full setup (seed generation via GPT-5, project creation, run start) in one call. The frontend is a single-page state machine with three phases (`prompt → training → testing`) connected by CSS crossfade transitions. The SSE stream already exists and is reused as-is to drive the training checklist in real time.

**Tech Stack:** FastAPI, psycopg2/Supabase, OpenAI Python SDK (already installed), Next.js 14, TypeScript, CSS transitions (no animation library)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `api/app/models.py` | Modify | Add `QuickStartRequest`, `QuickStartResponse` |
| `api/app/seed_generator.py` | Create | GPT-5 seed generation logic |
| `api/app/main.py` | Modify | Add `POST /quick-start` endpoint |
| `api/tests/test_quick_start.py` | Create | Tests for quick-start endpoint |
| `web/lib/api.ts` | Modify | Add `quickStart()` and `promoteRun()` |
| `web/app/globals.css` | Modify | Add demo-flow CSS classes |
| `web/app/page.tsx` | Rewrite | 3-screen state machine component |

---

## Task 1: Add QuickStart models to `models.py`

**Files:**
- Modify: `api/app/models.py`

- [ ] **Step 1: Add the two new models at the bottom of `api/app/models.py`**

```python
class QuickStartRequest(BaseModel):
    description: str


class QuickStartResponse(BaseModel):
    project_id: str
    run_id: str
```

- [ ] **Step 2: Commit**

```bash
git add api/app/models.py
git commit -m "feat: add QuickStartRequest and QuickStartResponse models"
```

---

## Task 2: Create `seed_generator.py`

**Files:**
- Create: `api/app/seed_generator.py`
- Create: `api/tests/test_seed_generator.py`

- [ ] **Step 1: Write the failing test**

Create `api/tests/test_seed_generator.py`:

```python
import json
from unittest.mock import MagicMock, patch

import pytest

from app.models import ExampleSource, Label
from app.seed_generator import generate_seeds


def _make_openai_response(data: list[dict]) -> MagicMock:
    msg = MagicMock()
    msg.content = json.dumps(data)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


VALID_SEEDS = (
    [{"text": f"billing issue {i}", "label": "in_scope"} for i in range(30)]
    + [{"text": f"off topic {i}", "label": "out_of_scope"} for i in range(30)]
    + [{"text": f"mixed {i}", "label": "ambiguous"} for i in range(30)]
)


def test_generate_seeds_returns_90_examples():
    with patch("app.seed_generator.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = (
            _make_openai_response(VALID_SEEDS)
        )
        results = generate_seeds("classify customer support tickets")

    assert len(results) == 90


def test_generate_seeds_label_distribution():
    with patch("app.seed_generator.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = (
            _make_openai_response(VALID_SEEDS)
        )
        results = generate_seeds("classify customer support tickets")

    labels = {r.label for r in results}
    assert labels == {Label.IN_SCOPE, Label.OUT_OF_SCOPE, Label.AMBIGUOUS}


def test_generate_seeds_source_is_human_seed():
    with patch("app.seed_generator.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = (
            _make_openai_response(VALID_SEEDS)
        )
        results = generate_seeds("any domain")

    assert all(r.source == ExampleSource.HUMAN_SEED for r in results)


def test_generate_seeds_retries_on_bad_json():
    bad_response = _make_openai_response([])
    bad_response.choices[0].message.content = "not json"
    good_response = _make_openai_response(VALID_SEEDS)

    with patch("app.seed_generator.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.side_effect = [
            bad_response,
            good_response,
        ]
        results = generate_seeds("any domain")

    assert len(results) == 90


def test_generate_seeds_raises_after_two_failures():
    bad = MagicMock()
    bad.choices[0].message.content = "not json"

    with patch("app.seed_generator.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = bad
        with pytest.raises(RuntimeError, match="Seed generation failed"):
            generate_seeds("any domain")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/kyle/Dev/projects/agentboundary/api
uv run pytest tests/test_seed_generator.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `seed_generator` doesn't exist yet.

- [ ] **Step 3: Create `api/app/seed_generator.py`**

```python
from __future__ import annotations

import json

from openai import OpenAI

from .config import settings
from .models import ExampleInput, ExampleSource, Label

_SEED_PROMPT = """\
You are generating training data for a text classifier.

Domain: {description}

Generate exactly 90 labeled examples in JSON format:
- 30 labeled "in_scope": messages clearly within the domain
- 30 labeled "out_of_scope": messages clearly outside the domain
- 30 labeled "ambiguous": messages that mix in-scope and out-of-scope intent

Return a JSON array only, no explanation:
[{{"text": "...", "label": "in_scope"}}, ...]

Make examples realistic, varied in length, and challenging enough to require a real classifier.\
"""


def generate_seeds(description: str) -> list[ExampleInput]:
    """Call GPT-5 to produce 90 labeled seed examples for the given domain description.

    Retries once on parse failure. Raises RuntimeError if both attempts fail.
    """
    client = OpenAI(api_key=settings.openai_api_key)

    def _attempt() -> list[ExampleInput]:
        response = client.chat.completions.create(
            model=settings.default_agent_model,
            messages=[
                {
                    "role": "user",
                    "content": _SEED_PROMPT.format(description=description),
                }
            ],
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)
        examples = [
            ExampleInput(
                text=item["text"],
                label=Label(item["label"]),
                source=ExampleSource.HUMAN_SEED,
            )
            for item in data
        ]
        labels_present = {e.label for e in examples}
        required = {Label.IN_SCOPE, Label.OUT_OF_SCOPE, Label.AMBIGUOUS}
        if not required.issubset(labels_present):
            raise ValueError(f"Missing labels: {required - labels_present}")
        return examples

    try:
        return _attempt()
    except Exception:
        try:
            return _attempt()
        except Exception as exc:
            raise RuntimeError("Seed generation failed") from exc
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /Users/kyle/Dev/projects/agentboundary/api
uv run pytest tests/test_seed_generator.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add api/app/seed_generator.py api/tests/test_seed_generator.py
git commit -m "feat: add GPT-5 seed generator with retry logic"
```

---

## Task 3: Add `POST /quick-start` endpoint

**Files:**
- Modify: `api/app/main.py`
- Create: `api/tests/test_quick_start.py`

- [ ] **Step 1: Write the failing test**

Create `api/tests/test_quick_start.py`:

```python
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


def test_quick_start_seed_failure_returns_500():
    client = TestClient(app)
    with patch("app.main.generate_seeds", side_effect=RuntimeError("Seed generation failed")):
        response = client.post(
            "/quick-start",
            headers=SESSION_HEADERS,
            json={"description": "classify tickets"},
        )
    assert response.status_code == 500
    assert "Seed generation failed" in response.json()["detail"]


def test_quick_start_requires_session_id():
    client = TestClient(app)
    response = client.post("/quick-start", json={"description": "test"})
    assert response.status_code == 400
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /Users/kyle/Dev/projects/agentboundary/api
uv run pytest tests/test_quick_start.py -v
```

Expected: all fail with 404 or `ImportError` — endpoint doesn't exist yet.

- [ ] **Step 3: Add the endpoint to `api/app/main.py`**

Add the import for `generate_seeds` at the top of `main.py`, after the existing imports:

```python
from .seed_generator import generate_seeds
```

Add the import for `QuickStartRequest` and `QuickStartResponse` to the existing models import block:

```python
from .models import (
    ClassificationRequest,
    ClassificationResponse,
    ExampleInput,
    ProjectCreate,
    ProjectDetail,
    QuickStartRequest,
    QuickStartResponse,
    RunCreate,
    RunDetail,
    RunStatus,
)
```

Add the endpoint after the `healthcheck` route:

```python
@app.post("/quick-start", response_model=QuickStartResponse)
def quick_start(
    payload: QuickStartRequest,
    background_tasks: BackgroundTasks,
    session_id: str = Depends(get_session_id),
) -> QuickStartResponse:
    try:
        seeds = generate_seeds(payload.description)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    project = repository.create_project(
        ProjectCreate(
            name=payload.description[:60],
            support_domain_description=payload.description,
        ),
        session_id,
    )
    repository.add_examples(project.id, seeds)

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
    background_tasks.add_task(runner.execute_run, project.id, run.id, None)

    return QuickStartResponse(project_id=project.id, run_id=run.id)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /Users/kyle/Dev/projects/agentboundary/api
uv run pytest tests/test_quick_start.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Run the full test suite to verify no regressions**

```bash
cd /Users/kyle/Dev/projects/agentboundary/api
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add api/app/main.py api/tests/test_quick_start.py
git commit -m "feat: add POST /quick-start endpoint"
```

---

## Task 4: Add `quickStart` and `promoteRun` to `web/lib/api.ts`

**Files:**
- Modify: `web/lib/api.ts`

- [ ] **Step 1: Add the two new functions to the `api` object in `web/lib/api.ts`**

Add `QuickStartResponse` type above the `api` export:

```typescript
export type QuickStartResponse = {
  project_id: string;
  run_id: string;
};
```

Add two entries to the existing `api` object (after `classify`):

```typescript
  quickStart: (description: string) =>
    request<QuickStartResponse>("/quick-start", {
      method: "POST",
      body: JSON.stringify({ description }),
    }),
  promoteRun: (projectId: string, runId: string) =>
    request<void>(`/projects/${projectId}/promote/${runId}`, { method: "POST" }),
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd /Users/kyle/Dev/projects/agentboundary/web
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add web/lib/api.ts
git commit -m "feat: add quickStart and promoteRun to api client"
```

---

## Task 5: Add demo-flow CSS to `globals.css`

**Files:**
- Modify: `web/app/globals.css`

- [ ] **Step 1: Append the following CSS block to the end of `web/app/globals.css`**

```css
/* ─── Demo flow ─── */

.demo-shell {
  min-height: 100vh;
  background: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #111;
}

/* Screen wrapper — all three screens live here, toggled by opacity */
.demo-screen {
  width: 100%;
  max-width: 480px;
  padding: 0 24px;
  transition: opacity 400ms ease-in-out;
}
.demo-screen[aria-hidden="true"] {
  opacity: 0;
  pointer-events: none;
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}

/* Screen 1 — Prompt */
.demo-eyebrow {
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #9ca3af;
  margin-bottom: 24px;
  text-align: center;
}
.demo-heading {
  font-size: clamp(22px, 4vw, 30px);
  font-weight: 700;
  color: #111;
  line-height: 1.25;
  text-align: center;
  margin-bottom: 28px;
}
.demo-textarea {
  width: 100%;
  border: 1.5px solid #d1d5db;
  border-radius: 12px;
  padding: 14px 16px;
  font-size: 15px;
  color: #111;
  resize: none;
  outline: none;
  background: #fff;
  transition: border-color 150ms;
  box-sizing: border-box;
  font-family: inherit;
}
.demo-textarea:focus {
  border-color: #111;
}
.demo-btn-primary {
  width: 100%;
  margin-top: 12px;
  padding: 14px;
  background: #111;
  color: #fff;
  border: none;
  border-radius: 12px;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 150ms;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}
.demo-btn-primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Screen 2 — Training */
.demo-subdesc {
  font-size: 12px;
  color: #9ca3af;
  margin-bottom: 28px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  text-align: center;
}
.demo-checklist {
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin-bottom: 24px;
}
.demo-check-item {
  display: flex;
  align-items: center;
  gap: 14px;
}
.demo-check-icon {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 200ms, border-color 200ms;
}
.demo-check-icon.pending {
  border: 1.5px solid #d1d5db;
}
.demo-check-icon.active {
  border: 1.5px solid #111;
}
.demo-check-icon.done {
  background: #111;
}
.demo-check-label {
  font-size: 14px;
  transition: color 200ms;
}
.demo-check-label.pending { color: #9ca3af; }
.demo-check-label.active  { color: #111; font-weight: 500; }
.demo-check-label.done    { color: #374151; }

/* Spinner inside active check icon */
.demo-spin {
  width: 10px;
  height: 10px;
  border: 2px solid #111;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}

/* Log box */
.demo-log {
  background: #f9fafb;
  border-radius: 10px;
  padding: 12px 14px;
  font-size: 11px;
  font-family: "SF Mono", "Fira Mono", monospace;
  color: #6b7280;
  line-height: 1.7;
  max-height: 140px;
  overflow-y: auto;
}

/* Screen 2 inline error */
.demo-error {
  margin-top: 16px;
  padding: 12px 14px;
  background: #fef2f2;
  border-radius: 10px;
  font-size: 13px;
  color: #dc2626;
}
.demo-retry-btn {
  margin-top: 10px;
  padding: 8px 16px;
  background: #111;
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 13px;
  cursor: pointer;
}

/* Screen 3 — Test */
.demo-live-heading {
  font-size: clamp(26px, 5vw, 36px);
  font-weight: 700;
  color: #111;
  text-align: center;
  margin-bottom: 6px;
}
.demo-live-sub {
  font-size: 13px;
  color: #9ca3af;
  text-align: center;
  margin-bottom: 32px;
}
.demo-test-row {
  display: flex;
  gap: 8px;
}
.demo-test-input {
  flex: 1;
  border: 1.5px solid #d1d5db;
  border-radius: 10px;
  padding: 12px 14px;
  font-size: 14px;
  color: #111;
  outline: none;
  font-family: inherit;
  transition: border-color 150ms;
}
.demo-test-input:focus { border-color: #111; }
.demo-test-enter {
  padding: 12px 16px;
  background: #111;
  color: #fff;
  border: none;
  border-radius: 10px;
  font-size: 14px;
  cursor: pointer;
  transition: opacity 150ms;
}
.demo-test-enter:disabled { opacity: 0.4; cursor: not-allowed; }

/* Result feed */
.demo-results {
  margin-top: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.demo-result-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  background: #f9fafb;
  border-radius: 8px;
  animation: fadeInUp 150ms ease-out;
}
.demo-result-text {
  font-size: 13px;
  color: #374151;
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.demo-result-conf {
  font-size: 11px;
  color: #9ca3af;
  flex-shrink: 0;
}

/* Label badges */
.demo-badge {
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  flex-shrink: 0;
}
.demo-badge.in-scope     { background: #111; color: #fff; }
.demo-badge.out-of-scope { background: #fef2f2; color: #dc2626; border: 1px solid #fca5a5; }
.demo-badge.ambiguous    { background: #fffbeb; color: #d97706; border: 1px solid #fcd34d; }

/* "View training details" link */
.demo-details-link {
  position: fixed;
  bottom: 24px;
  right: 28px;
  font-size: 12px;
  color: #9ca3af;
  cursor: pointer;
  text-decoration: none;
  transition: color 150ms;
}
.demo-details-link:hover { color: #374151; }

/* Drawer */
.demo-drawer-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.15);
  z-index: 40;
}
.demo-drawer {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  height: 50vh;
  background: #fff;
  border-radius: 16px 16px 0 0;
  box-shadow: 0 -4px 24px rgba(0,0,0,0.08);
  z-index: 50;
  display: flex;
  flex-direction: column;
  transform: translateY(100%);
  transition: transform 300ms ease-out;
}
.demo-drawer.open {
  transform: translateY(0);
}
.demo-drawer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px 12px;
  border-bottom: 1px solid #f3f4f6;
  flex-shrink: 0;
}
.demo-drawer-title {
  font-size: 14px;
  font-weight: 600;
  color: #111;
}
.demo-drawer-close {
  font-size: 20px;
  color: #9ca3af;
  background: none;
  border: none;
  cursor: pointer;
  line-height: 1;
}
.demo-drawer-body {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px;
}
.demo-drawer-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  margin-bottom: 16px;
}
.demo-drawer-table th {
  text-align: left;
  color: #9ca3af;
  padding: 4px 8px 8px 0;
  font-weight: 500;
}
.demo-drawer-table td {
  padding: 4px 8px 4px 0;
  color: #374151;
}
.demo-drawer-log {
  font-size: 11px;
  font-family: "SF Mono", "Fira Mono", monospace;
  color: #6b7280;
  line-height: 1.8;
  white-space: pre-wrap;
}
```

- [ ] **Step 2: Commit**

```bash
git add web/app/globals.css
git commit -m "feat: add demo-flow CSS classes"
```

---

## Task 6: Rewrite `web/app/page.tsx` — Screen 1 (Prompt)

**Files:**
- Rewrite: `web/app/page.tsx`

Start with just Screen 1 rendering and the submit handler that calls `quickStart`. Screens 2 and 3 will be stubbed.

- [ ] **Step 1: Verify the existing page renders (baseline)**

```bash
cd /Users/kyle/Dev/projects/agentboundary/web
npm run dev
```

Open http://localhost:3000 — confirm the current UI loads without errors.

- [ ] **Step 2: Replace `web/app/page.tsx` with the Screen 1 skeleton**

```tsx
"use client";

import { KeyboardEvent, useEffect, useRef, useState } from "react";
import { api, API_BASE_URL, ClassificationResponse, RunEvent, QuickStartResponse, getClientSessionId } from "../lib/api";

// ─── Types ────────────────────────────────────────────────────────────────────

type Phase = "prompt" | "training" | "testing";

type ChecklistStatus = "pending" | "active" | "done";

interface ChecklistItem {
  id: string;
  label: string;
  status: ChecklistStatus;
}

interface TestResult {
  text: string;
  label: string;
  confidence: number;
}

interface RoundSummary {
  index: number;
  trainF1: number | null;
  holdoutF1: number | null;
}

const INITIAL_CHECKLIST: ChecklistItem[] = [
  { id: "seeds",   label: "Generating seeds with GPT-5",  status: "active" },
  { id: "round1",  label: "Training Round 1",              status: "pending" },
  { id: "round2",  label: "Training Round 2",              status: "pending" },
  { id: "eval",    label: "Evaluating & finishing",        status: "pending" },
];

function labelClass(label: string): string {
  if (label === "in_scope")     return "in-scope";
  if (label === "out_of_scope") return "out-of-scope";
  return "ambiguous";
}

function labelDisplay(label: string): string {
  if (label === "in_scope")     return "In Scope";
  if (label === "out_of_scope") return "Out of Scope";
  return "Ambiguous";
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function HomePage() {
  const [phase, setPhase]               = useState<Phase>("prompt");
  const [description, setDescription]   = useState("");
  const [submitting, setSubmitting]     = useState(false);
  const [error, setError]               = useState("");

  // Training state
  const [projectId, setProjectId]       = useState("");
  const [runId, setRunId]               = useState("");
  const [checklist, setChecklist]       = useState<ChecklistItem[]>(INITIAL_CHECKLIST);
  const [logLines, setLogLines]         = useState<string[]>([]);
  const [runFailed, setRunFailed]       = useState(false);
  const logRef                          = useRef<HTMLDivElement>(null);

  // Testing state
  const [bestF1, setBestF1]             = useState<number | null>(null);
  const [exampleCount, setExampleCount] = useState(90);
  const [testInput, setTestInput]       = useState("");
  const [testResults, setTestResults]   = useState<TestResult[]>([]);
  const [classifying, setClassifying]   = useState(false);
  const [drawerOpen, setDrawerOpen]     = useState(false);
  const [rounds, setRounds]             = useState<RoundSummary[]>([]);
  const [startedAt, setStartedAt]       = useState<Date | null>(null);
  const [duration, setDuration]         = useState("");

  // ── Escape closes drawer ──────────────────────────────────────────────────
  useEffect(() => {
    function onKey(e: globalThis.KeyboardEvent) {
      if (e.key === "Escape") setDrawerOpen(false);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ── Auto-scroll log ───────────────────────────────────────────────────────
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logLines]);

  // ── Submit handler ────────────────────────────────────────────────────────
  async function handleSubmit() {
    if (!description.trim() || submitting) return;
    setSubmitting(true);
    setError("");
    try {
      const result = await api.quickStart(description.trim());
      setProjectId(result.project_id);
      setRunId(result.run_id);
      setStartedAt(new Date());
      setPhase("training");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong. Try again.");
      setSubmitting(false);
    }
  }

  // ── SSE connection (runs when phase becomes "training") ───────────────────
  useEffect(() => {
    if (phase !== "training" || !runId) return;
    const sessionId = getClientSessionId();
    const url = `${API_BASE_URL}/runs/${runId}/events/stream?session_id=${encodeURIComponent(sessionId)}`;
    const es = new EventSource(url);

    function updateChecklist(updater: (items: ChecklistItem[]) => ChecklistItem[]) {
      setChecklist((prev) => updater(prev));
    }

    function markDone(id: string) {
      updateChecklist((items) =>
        items.map((item) => (item.id === id ? { ...item, status: "done" } : item))
      );
    }

    function markActive(id: string) {
      updateChecklist((items) =>
        items.map((item) =>
          item.id === id
            ? { ...item, status: "active" }
            : item.status === "pending"
            ? item
            : item
        )
      );
    }

    es.addEventListener("run_event", (e: MessageEvent) => {
      const event = JSON.parse(e.data) as RunEvent;
      if (event.message) setLogLines((prev) => [...prev, event.message]);

      const { event_type, payload } = event;
      const roundIdx = (payload as Record<string, number>)?.round_index;

      if (event_type === "run_started") {
        markDone("seeds");
        markActive("round1");
      }
      if (event_type === "round_complete" && roundIdx === 1) {
        markDone("round1");
        markActive("round2");
        const f1 = (payload as Record<string, number>)?.macro_f1 ?? null;
        setRounds((prev) => [...prev, { index: 1, trainF1: f1, holdoutF1: null }]);
      }
      if (event_type === "round_complete" && roundIdx === 2) {
        markDone("round2");
        markActive("eval");
        const f1 = (payload as Record<string, number>)?.macro_f1 ?? null;
        setRounds((prev) => [...prev, { index: 2, trainF1: f1, holdoutF1: null }]);
      }
      if (event_type === "review_started") {
        markActive("eval");
      }
    });

    es.addEventListener("run_done", (e: MessageEvent) => {
      es.close();
      const run = JSON.parse(e.data) as { status: string; best_macro_f1: number | null };
      if (run.status === "failed") {
        setRunFailed(true);
        return;
      }
      markDone("eval");
      setBestF1(run.best_macro_f1);
      if (startedAt) {
        const secs = Math.round((Date.now() - startedAt.getTime()) / 1000);
        const m = Math.floor(secs / 60);
        const s = secs % 60;
        setDuration(`${m}m ${s}s`);
      }
      // Promote run so classify works, then reveal testing screen
      api.promoteRun(projectId, runId).catch(() => {}).finally(() => {
        setTimeout(() => setPhase("testing"), 800);
      });
    });

    es.onerror = () => {
      es.close();
      setRunFailed(true);
    };

    return () => es.close();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, runId]);

  // ── Classify handler ──────────────────────────────────────────────────────
  async function handleClassify() {
    if (!testInput.trim() || classifying) return;
    const text = testInput.trim();
    setClassifying(true);
    try {
      const result = await api.classify(projectId, text);
      setTestResults((prev) =>
        [{ text, label: result.label, confidence: result.confidence }, ...prev].slice(0, 5)
      );
      setTestInput("");
    } catch {
      // silent — keep input so user can retry
    } finally {
      setClassifying(false);
    }
  }

  function handleTestKey(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") handleClassify();
  }

  function handlePromptKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="demo-shell">

      {/* ── Screen 1: Prompt ── */}
      <div
        className="demo-screen"
        aria-hidden={phase !== "prompt" ? "true" : undefined}
        style={{ position: "relative" }}
      >
        <p className="demo-eyebrow">Classifier Studio</p>
        <h1 className="demo-heading">What kind of classifier<br />do you want to build?</h1>
        <textarea
          className="demo-textarea"
          rows={3}
          autoFocus
          placeholder="e.g. classify customer support tickets for a SaaS billing product"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          onKeyDown={handlePromptKey}
          disabled={submitting}
        />
        {error && <p style={{ color: "#dc2626", fontSize: 13, marginTop: 8 }}>{error}</p>}
        <button
          className="demo-btn-primary"
          onClick={handleSubmit}
          disabled={submitting || !description.trim()}
        >
          {submitting ? (
            <>
              <span className="demo-spin" style={{ border: "2px solid #fff", borderTopColor: "transparent" }} />
              Setting up…
            </>
          ) : (
            "Build it →"
          )}
        </button>
      </div>

      {/* ── Screen 2: Training ── */}
      <div
        className="demo-screen"
        aria-hidden={phase !== "training" ? "true" : undefined}
        style={{ position: phase === "training" ? "relative" : undefined }}
      >
        <p className="demo-subdesc">{description.slice(0, 60)}{description.length > 60 ? "…" : ""}</p>

        <div className="demo-checklist">
          {checklist.map((item) => (
            <div key={item.id} className="demo-check-item">
              <div className={`demo-check-icon ${item.status}`}>
                {item.status === "done" && (
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                    <path d="M2 6l3 3 5-5" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                )}
                {item.status === "active" && <span className="demo-spin" />}
              </div>
              <span className={`demo-check-label ${item.status}`}>{item.label}</span>
            </div>
          ))}
        </div>

        <div className="demo-log" ref={logRef}>
          {logLines.map((line, i) => (
            <div key={i}>{line}</div>
          ))}
          {logLines.length === 0 && <span style={{ color: "#d1d5db" }}>Starting…</span>}
        </div>

        {runFailed && (
          <div className="demo-error">
            Training failed.
            <br />
            <button
              className="demo-retry-btn"
              onClick={() => {
                setRunFailed(false);
                setChecklist(INITIAL_CHECKLIST);
                setLogLines([]);
                setPhase("prompt");
                setSubmitting(false);
              }}
            >
              Try again
            </button>
          </div>
        )}
      </div>

      {/* ── Screen 3: Test ── */}
      <div
        className="demo-screen"
        aria-hidden={phase !== "testing" ? "true" : undefined}
        style={{ position: phase === "testing" ? "relative" : undefined }}
      >
        <h1 className="demo-live-heading">Your classifier is live.</h1>
        <p className="demo-live-sub">
          {bestF1 != null ? `F1 ${bestF1.toFixed(3)} · ` : ""}
          {exampleCount} examples
          {duration ? ` · ${duration}` : ""}
        </p>

        <div className="demo-test-row">
          <input
            className="demo-test-input"
            type="text"
            placeholder="Type anything to test it…"
            value={testInput}
            onChange={(e) => setTestInput(e.target.value)}
            onKeyDown={handleTestKey}
            autoFocus
          />
          <button
            className="demo-test-enter"
            onClick={handleClassify}
            disabled={classifying || !testInput.trim()}
          >
            ↵
          </button>
        </div>

        <div className="demo-results">
          {testResults.map((r, i) => (
            <div key={i} className="demo-result-item">
              <span className={`demo-badge ${labelClass(r.label)}`}>
                {labelDisplay(r.label)}
              </span>
              <span className="demo-result-text">{r.text}</span>
              <span className="demo-result-conf">{(r.confidence * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── View training details link (Screen 3 only) ── */}
      {phase === "testing" && (
        <button
          className="demo-details-link"
          onClick={() => setDrawerOpen(true)}
        >
          View training details ↗
        </button>
      )}

      {/* ── Details Drawer ── */}
      {phase === "testing" && drawerOpen && (
        <>
          <div className="demo-drawer-backdrop" onClick={() => setDrawerOpen(false)} />
          <div className={`demo-drawer ${drawerOpen ? "open" : ""}`}>
            <div className="demo-drawer-header">
              <span className="demo-drawer-title">Training Details</span>
              <button className="demo-drawer-close" onClick={() => setDrawerOpen(false)}>×</button>
            </div>
            <div className="demo-drawer-body">
              {rounds.length > 0 && (
                <table className="demo-drawer-table">
                  <thead>
                    <tr>
                      <th>Round</th>
                      <th>Train F1</th>
                      <th>Holdout F1</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rounds.map((r) => (
                      <tr key={r.index}>
                        <td>Round {r.index}</td>
                        <td>{r.trainF1 != null ? r.trainF1.toFixed(3) : "—"}</td>
                        <td>{r.holdoutF1 != null ? r.holdoutF1.toFixed(3) : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              <p style={{ fontSize: 12, color: "#9ca3af", marginBottom: 8 }}>
                {exampleCount} examples generated · Best F1: {bestF1 != null ? bestF1.toFixed(3) : "—"}
              </p>
              <div className="demo-drawer-log">
                {logLines.join("\n")}
              </div>
            </div>
          </div>
        </>
      )}

    </div>
  );
}
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd /Users/kyle/Dev/projects/agentboundary/web
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 4: Start dev server and verify Screen 1 renders**

```bash
cd /Users/kyle/Dev/projects/agentboundary/web
npm run dev
```

Open http://localhost:3000. Verify:
- Pure white background, no dark UI
- "CLASSIFIER STUDIO" label, large heading, textarea, "Build it →" button
- No other UI elements visible

- [ ] **Step 5: Commit**

```bash
git add web/app/page.tsx
git commit -m "feat: rewrite homepage as 3-screen demo flow"
```

---

## Task 7: End-to-end smoke test

- [ ] **Step 1: Start the API locally**

```bash
cd /Users/kyle/Dev/projects/agentboundary/api
uv run uvicorn app.main:app --reload --port 8000
```

- [ ] **Step 2: Start the web dev server**

```bash
cd /Users/kyle/Dev/projects/agentboundary/web
npm run dev
```

- [ ] **Step 3: Manual flow test**

1. Open http://localhost:3000
2. Type a description: "Classify customer support messages for a SaaS billing product"
3. Press Enter or click "Build it →"
4. Verify: transitions to Screen 2 with checklist + log
5. Wait for training to complete (checklist checks off in sequence)
6. Verify: crossfades to Screen 3 with "Your classifier is live."
7. Type "I was charged twice this month" → press Enter
8. Verify: "IN SCOPE" badge appears with confidence %
9. Type "What's the meaning of life" → Enter
10. Verify: "OUT OF SCOPE" badge appears
11. Click "View training details ↗"
12. Verify: drawer slides up from bottom with F1 table and log
13. Press Escape — verify drawer closes

- [ ] **Step 4: Commit all remaining changes and push**

```bash
cd /Users/kyle/Dev/projects/agentboundary
git add -A
git status   # verify only expected files are staged
git commit -m "feat: complete interview demo UX — prompt, training, and test screens"
```
