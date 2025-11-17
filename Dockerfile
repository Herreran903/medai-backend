# Production Dockerfile for Render deployment
FROM python:3.11-slim

# Install system dependencies required for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app

# Expose port (Render will set PORT env variable)
EXPOSE ${PORT:-8000}

# Run uvicorn with PORT from environment variable
# No --reload flag for production
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
