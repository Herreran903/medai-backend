# ==============================================================================
# MedAI LLM NER Service - Dockerfile
# ==============================================================================
# Lightweight image for LLM-based entity extraction service.
# Uses external APIs (Claude/GPT) - no local model loading required.
#
# Image size: ~100MB (minimal dependencies)
# Startup time: <1 second (API key validation only)

FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY services/ner-llm/requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/ner-llm/app ./app

# Shared schemas are mounted via docker-compose volumes
# See docker-compose.micro.dev.yml: ./shared:/app/shared:ro

# Expose service port
EXPOSE 8003

# Health check (liveness probe)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/health').read()" || exit 1

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
