# Multi-stage build for scf-surrogate API
# Stage 1 (builder): install all dependencies
# Stage 2 (runtime): copy only what is needed to serve

# ------- Stage 1: builder -------
FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml .
COPY src/ src/
RUN pip install --upgrade pip && pip install --no-cache-dir -e ".[serve]"

# ------- Stage 2: runtime -------
FROM python:3.11-slim AS runtime

# Non-root user - never run production containers as root
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ src/
COPY app/ app/
COPY configs/ configs/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
