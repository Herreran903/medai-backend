# Production Dockerfile for Render deployment
FROM python:3.11-slim

# Install system dependencies required for ML libraries + TLS (Atlas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# (Opcional, pero a veces ayuda a que Python use estos certs explícitamente)
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app

# EXPOSE es solo informativo; Render usa PORT, así que deja un puerto fijo
EXPOSE 8000

# Run uvicorn with PORT from environment variable
# No --reload flag for production
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
