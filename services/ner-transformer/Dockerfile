# ==============================================================================
# MedAI Transformer NER Service - Dockerfile
# ==============================================================================
# PyTorch-based image for Transformer NER service (BETO/RoBERTa).
# Downloads models from Hugging Face Hub on first startup.
#
# Image size: ~1.8GB (PyTorch + Transformers)
# Startup time: 5-30s (model download + loading, cached after first run)

FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY services/ner-transformer/requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/ner-transformer/app ./app

# Shared schemas are mounted via docker-compose volumes
# See docker-compose.micro.dev.yml: ./shared:/app/shared:ro

# Create cache directory for Hugging Face models
RUN mkdir -p /root/.cache/huggingface

# Expose service port
EXPOSE 8001

# Health check (readiness probe)
# Note: Uses /readyz because model needs to load first
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/readyz').read()" || exit 1

# Run FastAPI with Uvicorn
# Increased timeout for model loading
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--timeout-keep-alive", "120"]
