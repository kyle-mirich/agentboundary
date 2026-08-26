# Architecture

## Overview

Agent Boundary is split into a presentation app and an orchestration/training API.

```text
Next.js UI on Vercel
   |
   | HTTP + SSE
   v
FastAPI API on Railway
   |
   +--> repository layer
   |      |
   |      +--> SQLite (local default)
   |      +--> Supabase PostgreSQL (production)
   |
   +--> DeepAgentRunner
   |      |
   |      +--> Deep Agents / LangGraph runtime
   |      +--> local workspace or Runloop sandbox
   |
   +--> PyTorch + Transformers
          |
          +--> training checkpoints
          +--> eval + holdout metrics
```

## Frontend Responsibilities

The frontend in `web/` is an open-source interface with two core views:

- the landing experience for quickly defining a classifier brief
- the project workspace for seed management, live run monitoring, and classifier testing

It is responsible for:

- creating a stable browser session id
- calling the API with that session id for per-user isolation
- rendering live run progress via server-sent events
- presenting project metrics, seeds, runs, and classifications in a polished UI

## Backend Responsibilities

The backend in `api/` handles the application state and all machine-learning work:

- creates and stores projects, examples, runs, rounds, and run events
- generates quick-start seed examples from an LLM or a deterministic fallback
- launches the Deep Agents experiment loop
- trains and evaluates the classifier for each round
- promotes the best run for live classification
- enforces lightweight public-demo guardrails, including session-scoped quick-start limits
- exposes safe production diagnostics through `/health` and `/health/details`

## Agent Runtime

`DeepAgentRunner` bridges the orchestration layer and the deterministic ML layer.

The agent is bounded by explicit tools that:

- generate candidates
- prepare datasets
- run training/evaluation rounds
- write reviews and artifacts

That design keeps the agent focused on experiment strategy while the actual model training remains deterministic and inspectable.

## Persistence Model

For local development and CI:

- the API defaults to SQLite at `api/data/app.db`
- schema creation happens automatically on startup and in tests

For production:

- set `DATABASE_URL` to Supabase Postgres or another PostgreSQL instance
- the same repository layer and schema bootstrap logic are reused
- `run_events` persists the terminal feed, which lets the frontend replay progress after refreshes or stream reconnects

Artifacts such as checkpoints, workspace files, and memory directories are written to local filesystem paths under `api/data/` and `api/artifacts/`.

## Runtime Flow

1. The browser creates a local session id and starts a quick-start run.
2. FastAPI creates a project, run row, and initial `run_queued` event.
3. Seed generation writes human-seed examples into the database.
4. Deep Agents plans the experiment, calls bounded tools, and writes run artifacts.
5. The deterministic training layer prepares datasets, trains checkpoints, evaluates metrics, and stores round records.
6. The frontend subscribes to server-sent events and renders an agent-style terminal from persisted events.
7. The best round is promoted and the classification endpoint serves the completed project.

## Deployment Shape

The repo is best deployed as two services:

- frontend on Vercel
- backend on Railway or another Docker-capable platform
- production database on Supabase Postgres

This keeps the user-facing application fast while allowing the Python backend to own training dependencies cleanly.

## Production Guardrails

- `APP_CORS_ORIGINS` restricts browser origins.
- `APP_QUICK_START_RATE_LIMIT` caps expensive public demo runs per browser session.
- Pydantic request models cap prompt and classification input length.
- `/health/details` reports operational status without exposing secrets.
- Queue and classification actions emit structured JSON application logs for Railway log search.
- `APP_MODEL_NAME` must stay compatible with the classifier training path; production currently uses `distilbert-base-uncased`.
