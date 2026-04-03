# Database

## Storage Strategy

Agent Boundary supports two persistence modes:

- SQLite by default for local development, tests, and portfolio review
- PostgreSQL for shared and production deployments

If `DATABASE_URL` is unset, the API creates and uses `api/data/app.db`.

If `DATABASE_URL` is set, the API connects to that PostgreSQL database instead.

## Core Tables

### `projects`

Stores the classifier configuration and promoted run state.

Key fields:

- `id`
- `session_id`
- `name`
- `support_domain_description`
- `allowed_topics_json`
- `disallowed_topics_json`
- `routing_notes`
- `agent_model`
- `max_rounds`
- `target_macro_f1`
- `target_out_of_scope_precision`
- `sandbox_profile`
- `promoted_run_id`

### `examples`

Stores labeled seed data and generated holdout examples.

Key fields:

- `id`
- `project_id`
- `text`
- `label`
- `source`
- `approved`
- `split`

### `runs`

Tracks each experiment run for a project.

Key fields:

- `id`
- `project_id`
- `status`
- `stop_reason`
- `best_round_id`
- `best_macro_f1`
- `summary`
- `workspace_root`

### `rounds`

Captures each train/evaluate iteration within a run.

Key fields:

- `id`
- `run_id`
- `round_index`
- `candidate_file`
- `dataset_summary_file`
- `evaluation_file`
- `holdout_file`
- `holdout_evaluation_file`
- `metrics_json`
- `holdout_metrics_json`
- `checkpoint_path`

### `run_events`

Stores the event stream rendered live in the UI.

Key fields:

- `id`
- `run_id`
- `event_type`
- `message`
- `payload_json`
- `created_at`

## Session Isolation

Projects are scoped by `session_id`, which is generated in the browser and sent through the `X-Session-Id` header. This keeps the local review flow simple without adding a full authentication system.

## Local Review Workflow

For a reviewer cloning the repo, no database setup is required:

1. Leave `DATABASE_URL` unset.
2. Start the API.
3. The schema is created automatically.
4. Run tests with `cd api && uv run pytest`.

## Production Notes

For production PostgreSQL:

- use a persistent managed Postgres instance
- include the deployed frontend origin in `APP_CORS_ORIGINS`
- back up the database if you intend to preserve projects or artifacts

For Supabase pooler connections on port `6543`, the username must include the project ref, for example `postgres.<project-ref>`.
