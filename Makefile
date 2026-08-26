.PHONY: install install-api install-web dev-api dev-web test backend-test frontend-check frontend-build

install: install-api install-web

install-api:
	cd api && uv sync --extra dev

install-web:
	cd web && npm install

dev-api:
	cd api && uv run uvicorn app.main:app --reload

dev-web:
	cd web && npm run dev

test: backend-test frontend-check

backend-test:
	cd api && uv run python -m pytest

frontend-check:
	cd web && npm run typecheck
	cd web && npm audit --omit=dev
	cd web && npm run build

frontend-build:
	cd web && npm run build
