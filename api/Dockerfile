FROM python:3.12-slim

WORKDIR /app
ENV PORT=8000

RUN pip install --no-cache-dir uv

# Install CPU-only torch first (~600MB vs ~2GB for CUDA build)
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch

# Install remaining dependencies
COPY pyproject.toml ./
RUN uv pip install --system \
    "fastapi>=0.115.0" \
    "uvicorn>=0.34.0" \
    "transformers>=4.48.0" \
    "scikit-learn>=1.5.0" \
    "openai>=2.29.0" \
    "psycopg2-binary>=2.9.0" \
    "langchain-openai>=0.3.0" \
    "pydantic-settings>=2.6.0" \
    "deepagents>=0.0.10"

COPY app/ ./app/

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
