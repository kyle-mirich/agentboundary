# Agent Boundary

Agent Boundary is an open-source full-stack project for building an in-scope text classifier end to end. A Next.js frontend walks a user through defining what counts as in scope, generating labeled seeds, launching a Deep Agents-driven experiment run, reviewing training rounds, and testing the promoted classifier live.

The project is intentionally opinionated:

- `web/` delivers a polished interactive frontend in Next.js 15 + React 19.
- `api/` owns project state, seed generation, orchestration, training, evaluation, and promotion.
- Local development works out of the box with SQLite by default, while production can point at PostgreSQL.

## Why This Repo Is Worth Reviewing

- Deep Agents is used for bounded experiment planning and review, not as a vague chat wrapper.
- The backend couples deterministic PyTorch training with agent-driven dataset iteration.
- The UI is designed as a polished product experience instead of an internal admin surface.
- The repo now includes CI, reproducible local commands, and supporting architecture and deployment docs.

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

- Python 3.11+
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

## Environment Variables

### Required

- `OPENAI_API_KEY`: used by the Deep Agents runtime and seed generator

### Optional

- `NEXT_PUBLIC_API_BASE_URL`: frontend API base URL, defaults to `http://127.0.0.1:8000`
- `DATABASE_URL`: PostgreSQL connection string for shared/prod deployments
- `APP_CORS_ORIGINS`: comma-separated allowed origins
- `APP_AGENT_MODEL`: default orchestration model, defaults to `gpt-5.4-mini`
- `APP_RESPONSES_GENERATION_MODEL`: model for structured example generation, defaults to `gpt-5.4-mini`
- `RUNLOOP_API_KEY`: required only when using the optional `runloop` sandbox profile

## Deployment Overview

The frontend and backend are designed to deploy separately:

- `web/`: deploy to Vercel
- `api/`: deploy to Railway or another container platform using `api/Dockerfile`

The main production requirements are:

- a persistent PostgreSQL database
- an `OPENAI_API_KEY`
- a frontend `NEXT_PUBLIC_API_BASE_URL` pointing at the deployed API
- `APP_CORS_ORIGINS` including the deployed frontend origin

Detailed deployment notes live in `docs/deployment.md`.

## Documentation

- `docs/architecture.md`
- `docs/database.md`
- `docs/flows.md`
- `docs/deployment.md`

## Notes for Reviewers

- The local backend now bootstraps its own schema, so reviewers do not need a pre-existing database to run tests.
- The classifier seed generation path is aligned across product copy, tests, and implementation at 90 examples per quick-start run.
- The repository is intentionally scoped as an open-source project, so the focus is clarity, end-to-end flow, and code quality over enterprise-level feature breadth.
