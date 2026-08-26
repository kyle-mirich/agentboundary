# Agent Boundary Case Study

## Problem

Chatbots and agents need a boundary before they answer. A customer-support assistant should handle account, billing, or product questions, but it should reject recipes, legal advice, coding help, and other out-of-scope requests before the main agent sees them.

Agent Boundary demonstrates that guardrail as a product workflow: describe the scope, generate labeled examples, run an agent-guided training loop, promote the best checkpoint, and test the classifier live.

## What The Demo Shows

1. A reviewer enters a plain-English scope.
2. The API creates a durable project and run record.
3. Seed generation creates a balanced labeled dataset.
4. Deep Agents plans and executes a bounded three-round experiment.
5. The deterministic training layer prepares data, trains, evaluates, and records metrics.
6. The UI streams persisted run events as an agent-style terminal.
7. The best checkpoint is promoted for live classification.

## Production Topology

```text
Vercel frontend
  -> Railway FastAPI backend
     -> Supabase Postgres
     -> OpenAI / Deep Agents orchestration
     -> PyTorch / Transformers classifier
```

The frontend is static/Next.js-friendly. The backend owns Python dependencies, long-running training work, and SSE progress streaming. Supabase stores projects, examples, runs, rounds, and event history.

## Engineering Choices

- The agent chooses experiment strategy, but training and evaluation are deterministic backend tools.
- SQLite is the local default, so reviewers can run the repo without provisioning services.
- PostgreSQL is used in production for shared state and durable event replay.
- The public demo is unauthenticated, but browser sessions are isolated with `X-Session-Id`.
- Quick-start runs are rate-limited per session to protect the OpenAI-backed workflow.
- `/health/details` exposes safe operational diagnostics without leaking secrets.

## Current Tradeoffs

- Runs still execute inside the API service. The durable run table and status endpoint make the job model explicit, but a separate worker would be the next reliability upgrade.
- Filesystem artifacts are local to the backend container. For longer retention, checkpoints and reports should move to object storage.
- The public session model is intentionally lightweight. A multi-user product would add auth and stricter tenant boundaries.

## What I Would Improve Next

- Move run execution to a worker process with queue-backed retries.
- Store checkpoints and final reports in object storage.
- Add OpenTelemetry traces around seed generation, training, evaluation, and classification.
- Add deployment automation with explicit preview/prod approvals.
- Add abuse controls by IP or account once the project has real users.
