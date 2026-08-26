# Deployment

## Recommended Topology

Deploy the repo as two services:

- frontend: Vercel
- backend: Railway or another Docker-capable Python host
- database: Supabase Postgres or another managed PostgreSQL provider

That split matches the current repository shape and keeps the operational model simple.

## Frontend Deployment

The frontend lives in `web/`.

### Required environment variable

- `NEXT_PUBLIC_API_BASE_URL=https://your-api-domain`

### Vercel settings

- Framework preset: Next.js
- Root directory: `web`
- Build command: `npm run build`
- Output: default Next.js output

## Backend Deployment

The backend lives in `api/` and already includes:

- `api/Dockerfile`
- `api/railway.json`

Attach a persistent volume at `/app/data`. Workspaces, agent memories, and promoted classifier
checkpoints all live below that directory so they survive restarts and deploys.

### Required environment variables

- `OPENAI_API_KEY`
- `DATABASE_URL` for production PostgreSQL
- `APP_CORS_ORIGINS=https://your-frontend-domain`
- `APP_MODEL_NAME=distilbert-base-uncased`

### Optional environment variables

- `APP_AGENT_MODEL`
- `APP_RESPONSES_GENERATION_MODEL`
- `APP_QUICK_START_RATE_LIMIT`
- `APP_QUICK_START_RATE_WINDOW_SECONDS`
- `APP_VERSION`
- `RUNLOOP_API_KEY`

### Health checks

- `GET /health` is the platform health check.
- `GET /health/details` validates database connectivity and configured OpenAI model access, then returns safe diagnostics without credentials.
- Run queueing and classification emit JSON application logs with event names such as `quick_start_queued`, `run_queued`, and `classification_completed`.

Do not expose raw `DATABASE_URL`, OpenAI keys, or provider credentials in diagnostics.

## Production Database Recommendation

Use PostgreSQL in production even though local development defaults to SQLite.

Why:

- concurrent writes are more predictable
- production backups are easier to manage
- the repo is already designed around PostgreSQL as the shared deployment target

If using Supabase:

- keep the project unpaused before starting or restarting Railway
- use the direct Postgres URL or the pooler URL provided by Supabase
- for pooler connections on port `6543`, the username must include the project ref, for example `postgres.<project-ref>`

## Deployment Checklist

1. Deploy the backend first.
2. Confirm `GET /health` returns `{"status": "ok"}`.
3. Confirm `GET /health/details` reports `status: "ok"`, `provider.status: "ok"`, and the expected model names.
4. Configure frontend `NEXT_PUBLIC_API_BASE_URL` to the deployed backend URL.
5. Add the frontend origin to `APP_CORS_ORIGINS`.
6. Build and deploy the frontend.
7. Run through a quick-start workflow in production.
8. Test the promoted classifier from the completed run screen.

## GitHub Readiness

The repo now includes GitHub Actions CI at `.github/workflows/ci.yml`, which validates:

- backend tests
- frontend production build

The current workflow is a merge gate, not an automatic deploy pipeline. If deploy automation is added later, the likely secrets are:

- `RAILWAY_TOKEN`
- `VERCEL_TOKEN`

Keep deploy automation separate from the existing test/build checks until rollback, preview environment behavior, and production approvals are explicit.
