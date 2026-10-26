# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    TOKENIZERS_PARALLELISM=false \
    CHROMA_DB_PATH=/data/chroma_db \
    ANONYMIZED_TELEMETRY=False
# OCR engines: Tesseract (English) and Poppler (pdftoppm for pdf2image).
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng poppler-utils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
# CPU-only PyTorch (no CUDA, ARM64 and x86_64 wheels).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
COPY app/ ./app/
RUN mkdir -p /data/chroma_db /models
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Test image: runtime + test dependencies, tests and the retrieval benchmark.
FROM runtime AS test
COPY requirements-dev.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements-dev.txt
COPY tests/ ./tests/
COPY benchmarks/ ./benchmarks/
CMD ["python", "-m", "pytest", "-q", "tests"]
