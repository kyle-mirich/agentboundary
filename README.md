# Agent Boundary

Live demo: [agentboundry.vercel.app](https://agentboundry.vercel.app)

Agent Boundary is an open-source full-stack project for building an in-scope text classifier end to end. A Next.js frontend walks a user through defining what counts as in scope, generating labeled seeds, launching a Deep Agents-driven experiment run, reviewing training rounds, and testing the promoted classifier live.

The project is intentionally opinionated:

- `web/` delivers a polished interactive frontend in Next.js 15 + React 19.
- `api/` owns project state, seed generation, orchestration, training, evaluation, and promotion.
- Local development works out of the box with SQLite by default, while production can point at PostgreSQL.

## Why This Repo Is Worth Reviewing

- Deep Agents is used for bounded experiment planning and review, not as a vague chat wrapper.
- The backend couples deterministic PyTorch training with agent-driven dataset iteration.
- The UI is designed as a polished product experience instead of an internal admin surface.
- The production shape is explicit: Vercel frontend, Railway FastAPI service, Supabase Postgres, OpenAI-powered seed generation, and persisted run events.
- The repo includes CI, reproducible local commands, production diagnostics, rate limits, and supporting architecture and deployment docs.

## Architecture At A Glance

```text
Vercel / Next.js
  -> FastAPI on Railway
     -> Supabase Postgres for projects, runs, rounds, examples, events
     -> OpenAI + Deep Agents for experiment planning and review
     -> PyTorch / Transformers classifier checkpoints
```

The browser owns a local session id and sends it as `X-Session-Id`, so reviewers can try the public demo without account setup while still keeping projects scoped per browser.

## Tech Stack

### Frontend

- Next.js 15
- React 19
- TypeScript
- CSS modules + global design system styles

### Backend

- FastAPI
- Deep Agents / LangChain / LangGraph
- PyTorch + Transformers
- SQLite for zero-config local development
- PostgreSQL for shared and production deployments

## Repository Layout

```text
.
├── api/                    FastAPI service, agent runtime, training loop, tests
├── web/                    Next.js frontend
├── docs/                   Architecture, flows, database, deployment notes
├── .github/workflows/      CI checks for backend and frontend
├── .env.example            Local environment template
└── Makefile                Common install, run, and verification commands
```

## Quick Start

### Prerequisites

- Python 3.11–3.13
- `uv`
- Node.js 22+
- npm
- An OpenAI API key for Deep Agents orchestration and LLM-generated seeds

### 1. Install Dependencies

```bash
make install
```

### 2. Configure Environment

```bash
cp .env.example .env
```

The local default is intentionally simple:

- If `DATABASE_URL` is unset, the API uses `api/data/app.db`.
- If `DATABASE_URL` is set, the API connects to that PostgreSQL database instead.

### 3. Run the API

```bash
make dev-api
```

The API starts on `http://127.0.0.1:8000`.

### 4. Run the Web App

In a second terminal:

```bash
make dev-web
```

The frontend starts on `http://localhost:3000`.

## Verification

Run the same checks used for GitHub CI:

```bash
make test
```

That executes:

- `cd api && uv run pytest`
- `cd web && npm run build`

Useful focused checks:

```bash
cd api && uv run pytest tests/test_quick_start.py tests/test_endpoint_coverage.py
cd web && npm run typecheck
cd web && npm audit --omit=dev
```

## Environment Variables

### Required

- `OPENAI_API_KEY`: used by the Deep Agents runtime and seed generator

### Optional

- `NEXT_PUBLIC_API_BASE_URL`: frontend API base URL, defaults to `http://127.0.0.1:8000`
- `DATABASE_URL`: PostgreSQL connection string for shared/prod deployments
- `APP_CORS_ORIGINS`: comma-separated allowed origins
- `APP_AGENT_MODEL`: default orchestration model, defaults to `gpt-5.6-luna`
- `APP_RESPONSES_GENERATION_MODEL`: model for structured example generation, defaults to `gpt-5.6-luna`
- `APP_MODEL_NAME`: classifier model identifier, defaults to `distilbert-base-uncased`
- `APP_QUICK_START_RATE_LIMIT`: quick-start runs allowed per browser session per window, defaults to `3`
- `APP_QUICK_START_RATE_WINDOW_SECONDS`: rate-limit window, defaults to `3600`
- `APP_RUN_RATE_LIMIT`: runs allowed per browser session per window for `POST /projects/{project_id}/runs`, defaults to `6`
- `APP_RUN_RATE_WINDOW_SECONDS`: run rate-limit window, defaults to `3600`
- `APP_RUN_STREAM_MAX_SECONDS`: how long an event stream stays open before it closes itself, defaults to `1800`
- `APP_RUN_STREAM_POLL_INTERVAL`: seconds between stream database polls, defaults to `1.0`
- `APP_ORPHANED_RUN_TIMEOUT_SECONDS`: a run with no progress for this long is marked failed on startup, defaults to `3600`
- `APP_MODEL_CACHE_SIZE`: number of classifier checkpoints kept resident, defaults to `1`
- `APP_RANDOM_SEED`: seed for training and eval-split selection, defaults to `42`
- `APP_SEED_EXAMPLES_PER_LABEL`: seed examples generated per label, defaults to `30` (90 total)
- `APP_MAX_EXAMPLES_PER_REQUEST`: maximum examples accepted in one `POST /projects/{project_id}/examples` call, defaults to `500`
- `APP_DATABASE_POOL_MIN_SIZE` / `APP_DATABASE_POOL_MAX_SIZE`: PostgreSQL pool bounds, default `1` and `10`
- `APP_VERSION`: release identifier returned by `/health/details`, defaults to `local`
- `APP_ARTIFACTS_DIR`: checkpoint directory, defaults to `data/artifacts` so one mounted data volume persists all runtime files
- `RUNLOOP_API_KEY`: required only when using the optional `runloop` sandbox profile

## Rate Limits And Concurrency

Runs are expensive: each one orchestrates an agent, generates examples, and
fine-tunes a classifier. Two endpoints start them and both are bounded:

- `POST /quick-start` — limited by `APP_QUICK_START_RATE_LIMIT` per browser session.
- `POST /projects/{project_id}/runs` — limited by `APP_RUN_RATE_LIMIT` per browser session.

Limits are recorded in the database, so they hold across workers and restarts.
A project can also only have one active run at a time: starting a second run
while one is `queued` or `running` returns `409`.

## Production Diagnostics

The API exposes two health surfaces:

- `GET /health`: minimal uptime check, returns `{"status":"ok"}`.
- `GET /health/details`: safe operational readiness with database connectivity, provider credential/model validation, sanitized targets, app version, and uptime. It does not return secrets.

## Deployment Overview

The frontend and backend are designed to deploy separately:

- `web/`: deploy to Vercel
- `api/`: deploy to Railway or another container platform using `api/Dockerfile`

The main production requirements are:

- a persistent PostgreSQL database
- an `OPENAI_API_KEY`
- a frontend `NEXT_PUBLIC_API_BASE_URL` pointing at the deployed API
- `APP_CORS_ORIGINS` including the deployed frontend origin
- `APP_MODEL_NAME=distilbert-base-uncased` unless the backend training path is changed to another compatible classifier

Detailed deployment notes live in `docs/deployment.md`.

## Documentation

- `docs/architecture.md`
- `docs/database.md`
- `docs/flows.md`
- `docs/deployment.md`
- `docs/case-study.md`

## Notes for Reviewers

- The local backend now bootstraps its own schema, so reviewers do not need a pre-existing database to run tests.
- Seed generation is enforced, not just requested: the prompt asks for `APP_SEED_EXAMPLES_PER_LABEL` examples per label (30 by default, 90 total), oversized responses are truncated, and a short response retries before failing loudly.
- Runs are persisted as database records with event streams, status endpoints, and replayable trace details.
- The repository is intentionally scoped as an open-source project, so the focus is clarity, end-to-end flow, and code quality over enterprise-level feature breadth.
