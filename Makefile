.PHONY: install install-api install-web dev-api dev-web test backend-test frontend-build

install: install-api install-web

install-api:
	cd api && uv sync --extra dev

install-web:
	cd web && npm install

dev-api:
	cd api && uv run uvicorn app.main:app --reload

dev-web:
	cd web && npm run dev

test: backend-test frontend-build

backend-test:
	cd api && uv run pytest

frontend-build:
	cd web && npm run build
