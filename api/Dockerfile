FROM python:3.12-slim

WORKDIR /app
ENV PORT=8000 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/tmp/hf-cache

RUN pip install --no-cache-dir uv

# Install CPU-only torch first (~600MB vs ~2GB for CUDA build)
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project --no-hashes --format requirements.txt --prune torch --output-file requirements.txt \
    && uv pip install --system -r requirements.txt

COPY app/ ./app/
RUN uv pip install --system --no-deps .

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
