# ==============================================================================
# MedAI BiLSTM NER Service - Dockerfile
# ==============================================================================
# TensorFlow-based image for BiLSTM NER service (BiLSTM-CRF).
# Model files are included in the image for fast startup.
#
# Image size: ~1.2GB (TensorFlow + model files)
# Startup time: 5-10s (model loading from disk)

FROM python:3.11-slim

# Install system dependencies for TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY services/ner-bilstm/requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/ner-bilstm/app ./app

# Copy CRF wrapper (required for loading CRF models)
COPY services/ner-bilstm/mwrapper.py ./

# Copy model files (LSTM model weights and vocabularies)
COPY services/ner-bilstm/models ./models

# Shared schemas are mounted via docker-compose volumes
# See docker-compose.micro.dev.yml: ./shared:/app/shared:ro

# Expose service port
EXPOSE 8002

# Health check (readiness probe)
# Note: Uses /readyz because model needs to load first
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/readyz').read()" || exit 1

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
