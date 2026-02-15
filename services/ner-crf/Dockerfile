# ==============================================================================
# MedAI CRF NER Service - Dockerfile
# ==============================================================================
# scikit-learn + sklearn-crfsuite based CRF NER service.
#
# Image size: medium (scikit-learn)
# Startup time: < 2s (local model load)

FROM python:3.12.10-slim

WORKDIR /app

# Build deps for python-crfsuite wheels (fallback to source build when needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY services/ner-crf/requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/ner-crf/app ./app

# Copy model files
COPY services/ner-crf/models ./models

# Shared schemas are mounted via docker-compose volumes
# See docker-compose.*.yml: ./shared:/app/shared:ro

# Expose service port
EXPOSE 8004

# Health check (readiness probe)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8004/readyz').read()" || exit 1

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]

