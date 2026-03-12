# MedAI Backend

Clinical Named Entity Recognition (NER) API for extracting medical entities from mechanical ventilation clinical notes.

[![Documentation](https://img.shields.io/badge/docs-Docusaurus-blue)](https://herreran903.github.io/docs-medai/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

MedAI Backend is a production-grade REST API service that provides clinical Named Entity Recognition (NER) capabilities for Spanish medical text, specifically optimized for mechanical ventilation clinical notes. The service extracts structured clinical entities such as ventilation parameters, vital signs, diagnoses, and laboratory values from unstructured clinical text.

### Key Features

- **Multiple NER Models**: Choose from CRF, BiLSTM, BiLSTM-CRF, Transformer (RoBERTa), or LLM-based (GPT) extraction
- **Microservices Architecture**: Independent services for each model with isolated dependencies
- **Value Normalization**: Regex-based normalization of extracted values into `code` (e.g., `FiO2 60%` -> `60`)
- **Batch Processing**: Process multiple clinical notes in a single request
- **Document Support**: Accept PDF, DOCX, and plain text files
- **Persistent Storage**: MongoDB-based storage with content deduplication
- **OpenAPI Documentation**: Auto-generated API documentation with Swagger/ReDoc

## Architecture (Microservices)

The backend uses a microservices architecture where each NER model runs in an independent container, providing better isolation, faster startup times, and independent scaling capabilities.

```
API Gateway (8000)
  -> Transformer service (8001, RoBERTa)
  -> BiLSTM service (8002, no CRF)
  -> LLM service (8003, GPT-only)
  -> CRF service (8004, sklearn-crfsuite)
  -> BiLSTM-CRF service (8005, Viterbi)

MongoDB (27017) stores extracted notes and metadata.
```

### Services

| Service | Port | Description | Image Size | Startup Time |
|---------|------|-------------|------------|--------------|
| **Gateway** | 8000 | API REST, routing, MongoDB | ~200MB | <2s |
| **Transformer** | 8001 | RoBERTa NER (RoBERTa-only) | ~1.8GB | 5-30s |
| **BiLSTM** | 8002 | BiLSTM NER (no CRF) | ~1.0GB | 5-10s |
| **BiLSTM-CRF** | 8005 | BiLSTM-CRF NER | ~1.2GB | 5-10s |
| **LLM** | 8003 | GPT NER (GPT-only) | ~100MB | <1s |
| **CRF** | 8004 | sklearn-crfsuite CRF NER | ~400MB | <2s |
| **MongoDB** | 27017 | Database | - | <5s |

### Microservices Benefits

- **Lightweight Gateway**: 94% reduction in size (3.5GB -> 200MB)
- **Fast Startup**: Gateway ready in <2s vs 10-15s for monolith
- **Isolated Dependencies**: PyTorch, TensorFlow, and LLM SDKs in separate containers
- **No Version Conflicts**: Each model can use different library versions
- **Granular Scaling**: Scale only the Transformer service if needed
- **Selective Deployment**: Update one service without affecting others

## Quick Start

### Development

```bash
# Start all microservices
docker compose -f docker-compose.dev.yml up -d --build

# Verify services are ready (wait 30-60s for models to load)
curl http://localhost:8000/health
```

### Production

```bash
# Configure environment variables first
cp .env.prod .env.prod.local
# Edit .env.prod.local with your MongoDB URI and API keys

# Start services
docker compose --env-file .env.prod.local -f docker-compose.prod.yml up --build
```

### Testing

```bash
# Basic health check
curl http://localhost:8000/health

# Minimal extraction test (text)
curl -X POST "http://localhost:8000/extract" \
  -F "text=Paciente con FiO2 60%, PEEP 8 cmH2O" \
  -F "model=transformer" \
  -F "episode_id=EP-001" \
  -F "note_date=2024-01-15T10:30:00"
```

## API Endpoints

### Extraction

#### `POST /extract`
Extract entities from a single clinical note.

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "text=Paciente con FiO2 60%, PEEP 8 cmH2O" \
  -F "model=transformer" \
  -F "episode_id=EP-001" \
  -F "note_date=2024-01-15T10:30:00"
```

**Request Parameters:**
- `text` or `file`: Clinical note text or file (PDF/DOCX/TXT)
- `model`: Model type (`lstm`, `lstm_crf`, `crf`, `transformer`, or `llm`)
- `model_variant`: Model variant (optional)
  - For `transformer`: `roberta` (fixed)
  - For `llm`: `gpt` (fixed)
- `episode_id`: Episode identifier (required; API returns 400 if missing)
- `note_date`: Clinical note date (ISO 8601, required; API returns 400 if missing)
- Results are always persisted to MongoDB
- `expand`: Include full result in response (default: `false`)

#### `POST /extract-batch`
Process multiple files in a single request.

```bash
curl -X POST "http://localhost:8000/extract-batch" \
  -F "files=@nota_001.pdf" \
  -F "files=@nota_002.pdf" \
  -F "model=transformer" \
  -F 'notes_meta=[{"filename":"nota_001.pdf","episode_id":"EP-001","note_date":"2024-01-15"},{"filename":"nota_002.pdf","episode_id":"EP-001","note_date":"2024-01-16"}]'
```

`notes_meta` is required. Each uploaded file must have matching `episode_id` and `note_date` metadata.

### Retrieval

#### `GET /notes/{note_id}`
Retrieve a stored extraction result.

```bash
curl "http://localhost:8000/notes/550e8400-e29b-41d4-a716-446655440000"
```

### Health

#### `GET /health`
Health check endpoint for container orchestration.

```bash
curl "http://localhost:8000/health"
```

## Supported Entity Types

MedAI recognizes clinical entities specific to mechanical ventilation notes:

### Ventilation Configuration
| Entity | Description | Example |
|--------|-------------|---------|
| `MODO` | Ventilator mode | AC/VC, PC, SIMV, PSV |
| `FIO2` | Fraction of inspired oxygen | 60% |
| `PEEP` | Positive end-expiratory pressure | 8 cmH2O |
| `FR` | Respiratory rate | 14/20 rpm |
| `VT` | Tidal volume | 420 mL |
| `FLUJO` | Flow rate | 40 L/min |
| `I_E` | Inspiratory/Expiratory ratio | 1:2 |
| `SENS` | Sensitivity | -2 cmH2O |

### Vital Signs
| Entity | Description | Example |
|--------|-------------|---------|
| `TEMP` | Body temperature | 38.5 C |
| `PA` | Blood pressure | 120/80 mmHg |
| `PAS` | Systolic blood pressure | 120 mmHg |
| `PAD` | Diastolic blood pressure | 80 mmHg |
| `PAM` | Mean arterial pressure | 93 mmHg |
| `FC` | Heart rate | 92 lpm |
| `SAO2` | Oxygen saturation | 95% |
| `GLICEMIA` | Blood glucose | 110 mg/dL |

### Arterial Blood Gases
| Entity | Description | Example |
|--------|-------------|---------|
| `PH` | Arterial pH | 7.35 |
| `PACO2` | Partial pressure of CO2 | 45 mmHg |
| `PAO2` | Partial pressure of O2 | 80 mmHg |
| `HCO3` | Bicarbonate | 24 mEq/L |
| `BE` | Base excess | -2 mEq/L |
| `PAFI` | PaO2/FiO2 ratio | 250 |

### Ventilation Response
| Entity | Description | Example |
|--------|-------------|---------|
| `PP` | Plateau pressure | 25 cmH2O |
| `PMES` | Meseta pressure | 23 cmH2O |
| `PM` | Mean pressure | 15 cmH2O |

### Anthropometry
| Entity | Description | Example |
|--------|-------------|---------|
| `EDAD` | Age | 65 anos |
| `PESO` | Weight | 75 kg |
| `TALLA` | Height | 1.70 m |

### Clinical
| Entity | Description | Example |
|--------|-------------|---------|
| `DX` | Diagnosis | Neumonia adquirida en comunidad |
| `POSTURA` | Patient position | Prono |

## Configuration

Configuration is managed through environment variables. Create a `.env` file based on `.env.dev` or `.env.prod`:

### Core Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_NAME` | Application name | MedAI Backend |
| `ENVIRONMENT` | Deployment environment | dev |
| `HOST` | Server bind address | 0.0.0.0 |
| `PORT` | Server port | 8000 |
| `LOG_LEVEL` | Logging level | info |

### Database

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URI` | MongoDB connection string | mongodb://mongo:27017 |
| `MONGODB_DB` | Database name | medai |

### NER Service URLs (for microservices)

| Variable | Description | Default |
|----------|-------------|---------|
| `NER_TRANSFORMER_URL` | Transformer service URL | http://ner-transformer:8001 |
| `NER_LSTM_URL` | BiLSTM service URL | http://ner-bilstm:8002 |
| `NER_LSTM_CRF_URL` | BiLSTM-CRF service URL | http://ner-bilstm-crf:8005 |
| `NER_LLM_URL` | LLM service URL | http://ner-llm:8003 |
| `NER_CRF_URL` | CRF service URL | http://ner-crf:8004 |

### NER Request Policy

| Variable | Description | Default |
|----------|-------------|---------|
| `NER_REQUEST_TIMEOUT` | Timeout for NER service HTTP requests (seconds) | 120.0 |
| `NER_RETRY_ATTEMPTS` | Max retry attempts for transient NER network/timeout errors | 3 |

### Transformer (RoBERTa) windowing

The Transformer microservice processes long notes using sliding windows (stride) instead of truncating to 512 tokens.

| Variable | Description | Default |
|----------|-------------|---------|
| `TRANSFORMER_MAX_LEN` | Token window length (subword tokens) | 512 |
| `TRANSFORMER_STRIDE` | Sliding-window overlap between windows (tokens) | 128 |
| `TRANSFORMER_WINDOW_BATCH_SIZE` | Windows processed per forward pass | 8 |

### API Keys (optional)

| Variable | Description | Required For |
|----------|-------------|--------------|
| `OPENAI_API_KEY` | OpenAI API key for GPT | LLM model with GPT (configured in ner-llm service) |

## Model Selection

### BiLSTM (model=`lstm`)
Fast inference with moderate accuracy (no CRF layer). Best for high-throughput scenarios.

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "model=lstm" \
  -F "text=Paciente con FiO2 60%" \
  -F "episode_id=EP-001" \
  -F "note_date=2024-01-15T10:00:00"
```

### BiLSTM-CRF (model=`lstm_crf`)
BiLSTM with CRF decoding (Viterbi). Useful to compare against BiLSTM-only.

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "model=lstm_crf" \
  -F "text=Paciente con FiO2 60%" \
  -F "episode_id=EP-001" \
  -F "note_date=2024-01-15T10:00:00"
```

### CRF (model=`crf`)
Lightweight classic CRF (sklearn-crfsuite). Useful as a fast baseline model.

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "model=crf" \
  -F "text=Paciente con FiO2 60%" \
  -F "episode_id=EP-001" \
  -F "note_date=2024-01-15T10:00:00"
```

### Transformer (Recommended)
Best accuracy for clinical NER (RoBERTa).

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "model=transformer" \
  -F "text=Paciente con FiO2 60%" \
  -F "episode_id=EP-001" \
  -F "note_date=2024-01-15T10:00:00"
```

### LLM
Highest flexibility with structured outputs (GPT-only). Requires API key.

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "model=llm" \
  -F "text=Paciente con FiO2 60%" \
  -F "episode_id=EP-001" \
  -F "note_date=2024-01-15T10:00:00"
```

## Testing

No automated test scripts are included in this repository at this time.

Use these manual checks to validate a running stack:

```bash
# Health check
curl http://localhost:8000/health

# Minimal extraction
curl -X POST "http://localhost:8000/extract" \
  -F "text=Paciente con FiO2 60%, PEEP 8 cmH2O" \
  -F "model=transformer" \
  -F "episode_id=EP-001" \
  -F "note_date=2024-01-15T10:30:00"
```

```bash
# Model matrix smoke test (gateway -> all microservices)
for model in transformer lstm lstm_crf crf llm; do
  variant=""
  if [ "$model" = "transformer" ]; then variant="-F model_variant=roberta"; fi
  if [ "$model" = "llm" ]; then variant="-F model_variant=gpt"; fi
  curl -X POST "http://localhost:8000/extract" \
    -F "text=Paciente en modo AC, FiO2 45%, PEEP 8 cmH2O, FC 92 lpm" \
    -F "model=$model" $variant \
    -F "episode_id=EP-SMOKE-$model" \
    -F "note_date=2024-01-15T10:30:00"
done
```
- All models (CRF, BiLSTM, BiLSTM-CRF, Transformer RoBERTa, LLM GPT)
- Document processing (PDF/DOCX/TXT)
- Result retrieval
- Deduplication

## Deployment

### Development (Local MongoDB)

```bash
docker compose -f docker-compose.dev.yml up --build
```

This configuration:
- Uses local MongoDB container
- Publishes only Gateway (`8000`) and MongoDB (`27017`) to host
- Keeps NER services on the internal Docker network (`medai-network`)

### Production (MongoDB Atlas)

```bash
# Configure environment variables first
cp .env.prod .env.prod.local
# Edit .env.prod.local with your MongoDB URI and API keys
# Requires local Cloudflare credentials in ~/.cloudflared/config.yml
# and an existing named tunnel: medai-backend.
docker compose --env-file .env.prod.local -f docker-compose.prod.yml up --build
```

### Secure Exposure with Cloudflare Tunnel

```bash
# Cloudflare Tunnel runs as the `cloudflared` service in prod compose
# (gateway remains the only exposed backend endpoint)
docker compose -f docker-compose.prod.yml logs -f cloudflared

# Verify external access
curl https://medai.your-domain.com/health
```

## Monitoring and Troubleshooting

### View Logs

```bash
# All services
docker compose -f docker-compose.dev.yml logs -f

# Specific service
docker compose -f docker-compose.dev.yml logs -f gateway
docker compose -f docker-compose.dev.yml logs -f ner-transformer
docker compose -f docker-compose.dev.yml logs -f ner-bilstm
docker compose -f docker-compose.dev.yml logs -f ner-bilstm-crf
docker compose -f docker-compose.dev.yml logs -f ner-crf
docker compose -f docker-compose.dev.yml logs -f ner-llm
```

### Health Checks

```bash
# Gateway (liveness)
curl http://localhost:8000/health

# NER services are internal (not published to host ports in compose).
# Check readiness from inside the gateway container:
docker compose -f docker-compose.dev.yml exec gateway \
  python -c "import urllib.request; print(urllib.request.urlopen('http://ner-transformer:8001/readyz').status)"
docker compose -f docker-compose.dev.yml exec gateway \
  python -c "import urllib.request; print(urllib.request.urlopen('http://ner-bilstm:8002/readyz').status)"
docker compose -f docker-compose.dev.yml exec gateway \
  python -c "import urllib.request; print(urllib.request.urlopen('http://ner-bilstm-crf:8005/readyz').status)"
docker compose -f docker-compose.dev.yml exec gateway \
  python -c "import urllib.request; print(urllib.request.urlopen('http://ner-crf:8004/readyz').status)"
docker compose -f docker-compose.dev.yml exec gateway \
  python -c "import urllib.request; print(urllib.request.urlopen('http://ner-llm:8003/readyz').status)"
```

### Common Issues

#### Transformer takes long to start
**Cause**: First download of model from Hugging Face Hub (~500MB)

**Solution**: Wait 30-60 seconds on first run. Model is cached in volume `transformer_cache` for subsequent runs.

#### BiLSTM fails to start
**Cause**: Model files not copied correctly

**Solution**:
```bash
# Verify files exist
ls services/ner-bilstm/models/model/

# Rebuild service
docker compose -f docker-compose.dev.yml build ner-bilstm
```

#### LLM returns 503
**Cause**: API keys not configured

**Solution**:
```bash
# Configure in .env.dev
OPENAI_API_KEY=sk-...

# Restart service
docker compose -f docker-compose.dev.yml restart ner-llm
```

#### Gateway cannot connect to services
**Cause**: Services in different network or not started

**Solution**:
```bash
# Verify all services in same network
docker network inspect medai-microservices-dev

# Check service status
docker compose -f docker-compose.dev.yml ps

# Restart all
docker compose -f docker-compose.dev.yml down
docker compose -f docker-compose.dev.yml up
```

## Development

### Project Structure

```
medai-backend/
|-- app/                       # API Gateway application
|   |-- config.py              # Application configuration
|   |-- deps.py                # FastAPI dependency injection
|   |-- indexes.py             # MongoDB index definitions
|   |-- main.py                # FastAPI application entry point
|   |-- schemas.py             # Pydantic request/response models
|   |-- repositories/          # Data access layer (Repository pattern)
|   |   |-- __init__.py
|   |   `-- episode_repository.py  # Episode/note data access
|   |-- routers/
|   |   `-- extract.py         # Extraction API endpoints
|   `-- services/
|       |-- extraction_service.py   # Business logic layer (Service pattern)
|       |-- ner_client.py      # NER microservices client
|       |-- pipeline.py        # Extraction orchestration
|       |-- registry.py        # Model registry
|       |-- text_utils.py      # Document text extraction
|       `-- utils.py           # Shared utilities
|-- services/                  # NER microservices
|   |-- ner-transformer/       # Transformer service (RoBERTa only)
|   |-- ner-bilstm/            # BiLSTM service (no CRF)
|   |-- ner-bilstm-crf/        # BiLSTM-CRF service
|   |-- ner-crf/               # CRF service (sklearn-crfsuite)
|   `-- ner-llm/               # LLM service (GPT only)
|-- shared/
|   `-- schemas.py             # Shared Pydantic models
|-- scripts/
|   `-- export_openapi.py      # OpenAPI schema export (multi-service)
|-- docker-compose.dev.yml     # Development configuration
|-- docker-compose.prod.yml    # Production configuration
|-- Dockerfile.gateway         # Gateway container image
|-- requirements-gateway.txt   # Gateway dependencies
`-- README.md                  # This file
```

### Running Tests

```bash
# No automated test suite is included in this repository.
# Use the quick curl-based checks above to validate a running stack.
```

### Code Formatting

```bash
py -m isort --profile black app/ services/ shared/ scripts/
py -m black app/ services/ shared/ scripts/
```

### Type Checking

```bash
mypy app/
```

## OpenAPI Documentation

The API documentation is automatically generated and available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Exporting OpenAPI Schemas

Generate OpenAPI schemas for **all services** (gateway + microservices):

```bash
python scripts/export_openapi.py
```

**CI/Docs Tip:** set `DOCS_BUILD=1` to skip model loading and heavy dependencies.
With this flag, `requirements.docs.txt` is sufficient to export schemas.

This creates:
- `openapi/gateway.json` - Gateway API (public endpoints)
- `openapi/ner-transformer.json` - Transformer service
- `openapi/ner-bilstm.json` - BiLSTM service
- `openapi/ner-bilstm-crf.json` - BiLSTM-CRF service
- `openapi/ner-crf.json` - CRF service
- `openapi/ner-llm.json` - LLM service
- `openapi.json` - Gateway schema (backward compatibility)

The schemas can be used for:
- **Documentation**: Docusaurus integration
- **Client Generation**: TypeScript, Python, etc.
- **API Testing**: Contract validation

## Frontend Integration

MedAI Backend is designed to work with the MedAI Frontend application:

- **Frontend Repository**: [medai-frontend](https://github.com/herreran903/medai-frontend)
- **Live Application**: [MedAI Frontend](https://medai-frontend-seven.vercel.app)

The frontend consumes the extraction API and provides a user interface for:
- Clinical note upload and processing
- Entity visualization and editing
- Episode management
- Export functionality

## Documentation

Comprehensive documentation is available at:

**[MedAI Documentation](https://herreran903.github.io/docs-medai/)**

The documentation includes:
- Getting started guides
- API reference
- Architecture overview
- Deployment guides

## License

This project is part of a university thesis project at Universidad del Valle.

## Contributing

Contributions are welcome. Please ensure:

1. Code follows existing style conventions
2. Manual smoke checks pass (`/health` and at least one `POST /extract`)
3. Documentation is updated for new features
4. Commit messages are descriptive

## Support

For issues and feature requests, please use the GitHub issue tracker.

---

**MedAI Backend** - Clinical NER for Mechanical Ventilation Notes
