# Deployment

## Recommended Topology

Deploy the repo as two services:

- frontend: Vercel
- backend: Railway or another Docker-capable Python host

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

### Required environment variables

- `OPENAI_API_KEY`
- `DATABASE_URL` for production PostgreSQL
- `APP_CORS_ORIGINS=https://your-frontend-domain`

### Optional environment variables

- `APP_AGENT_MODEL`
- `APP_RESPONSES_GENERATION_MODEL`
- `RUNLOOP_API_KEY`

## Production Database Recommendation

Use PostgreSQL in production even though local development defaults to SQLite.

Why:

- concurrent writes are more predictable
- production backups are easier to manage
- the repo is already designed around PostgreSQL as the shared deployment target

## Deployment Checklist

1. Deploy the backend first.
2. Confirm `GET /health` returns `{"status": "ok"}`.
3. Configure frontend `NEXT_PUBLIC_API_BASE_URL` to the deployed backend URL.
4. Add the frontend origin to `APP_CORS_ORIGINS`.
5. Build and deploy the frontend.
6. Run through a quick-start workflow in production.

## GitHub Readiness

The repo now includes GitHub Actions CI at `.github/workflows/ci.yml`, which validates:

- backend tests
- frontend production build

On pushes to `main`, the same workflow can also deploy:

- `api/` to Railway
- `web/` to Vercel

Required GitHub secrets:

- `RAILWAY_TOKEN`
- `VERCEL_TOKEN`

That should be treated as the minimum merge gate before publishing future changes.
