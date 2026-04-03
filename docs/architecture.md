# Architecture

## Overview

Agent Boundary is split into a presentation app and an orchestration/training API.

```text
Next.js UI
   |
   | HTTP + SSE
   v
FastAPI API
   |
   +--> repository layer
   |      |
   |      +--> SQLite (local default)
   |      +--> PostgreSQL (shared / production)
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

- set `DATABASE_URL` to a PostgreSQL instance
- the same repository layer and schema bootstrap logic are reused

Artifacts such as checkpoints, workspace files, and memory directories are written to local filesystem paths under `api/data/` and `api/artifacts/`.

## Deployment Shape

The repo is best deployed as two services:

- frontend on Vercel
- backend on Railway or another Docker-capable platform

This keeps the user-facing application fast while allowing the Python backend to own training dependencies cleanly.
