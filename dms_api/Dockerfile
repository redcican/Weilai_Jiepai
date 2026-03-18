# DMS API Gateway Dockerfile
# Multi-stage build for optimized production image

# ===== Build Stage =====
FROM python:3.12-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ===== Production Stage =====
FROM python:3.12-slim as production

# Labels
LABEL maintainer="DMS API Team"
LABEL version="2.0.0"
LABEL description="DMS API Gateway - FastAPI"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    DMS_HOST=0.0.0.0 \
    DMS_PORT=8000

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copy application code
COPY --chown=appuser:appgroup app/ ./app/

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health/live', timeout=5).raise_for_status()"

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# ===== Development Stage =====
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    httpx

# Switch back to non-root user
USER appuser

# Enable reload for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
