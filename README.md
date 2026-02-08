# MedAI Backend

Clinical Named Entity Recognition (NER) API for extracting medical entities from mechanical ventilation clinical notes.

[![Documentation](https://img.shields.io/badge/docs-Docusaurus-blue)](https://herreran903.github.io/docs-medai/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

MedAI Backend is a production-grade REST API service that provides clinical Named Entity Recognition (NER) capabilities for Spanish medical text, specifically optimized for mechanical ventilation clinical notes. The service extracts structured clinical entities such as ventilation parameters, vital signs, diagnoses, and laboratory values from unstructured clinical text.

### Key Features

- **Multiple NER Models**: Choose from BiLSTM, Transformer (RoBERTa), or LLM-based (GPT) extraction
- **Microservices Architecture**: Independent services for each model with isolated dependencies
- **Entity Normalization**: UMLS-based normalization to SNOMED-CT/ICD-10 (module available; currently disabled in gateway)
- **Batch Processing**: Process multiple clinical notes in a single request
- **Document Support**: Accept PDF, DOCX, and plain text files
- **Persistent Storage**: MongoDB-based storage with content deduplication
- **OpenAPI Documentation**: Auto-generated API documentation with Swagger/ReDoc

## Architecture (Microservices)

The backend uses a microservices architecture where each NER model runs in an independent container, providing better isolation, faster startup times, and independent scaling capabilities.

```
                    ┌─────────────────────┐
                    │   API Gateway       │
                    │   Port: 8000        │
                    │   Size: ~200MB      │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐      ┌───────────────┐     ┌──────────────┐
│ Transformer   │      │   BiLSTM     │     │     LLM      │
│ Port: 8001    │      │  Port: 8002   │     │  Port: 8003  │
│ Size: ~1.8GB  │      │  Size: ~1.2GB │     │  Size: ~100MB│
│  RoBERTa      │      │  BiLSTM-CRF   │     │    GPT       │
└───────────────┘      └───────────────┘     └──────────────┘
```

### Services

| Service | Port | Description | Image Size | Startup Time |
|---------|------|-------------|------------|--------------|
| **Gateway** | 8000 | API REST, routing, MongoDB | ~200MB | <2s |
| **Transformer** | 8001 | RoBERTa NER (fixed) | ~1.8GB | 5-30s |
| **BiLSTM** | 8002 | BiLSTM-CRF NER | ~1.2GB | 5-10s |
| **LLM** | 8003 | GPT NER (fixed) | ~100MB | <1s |
| **MongoDB** | 27017 | Database | - | <5s |

### Microservices Benefits

- **Lightweight Gateway**: 94% reduction in size (3.5GB → 200MB)
- **Fast Startup**: Gateway ready in <2s vs 10-15s for monolith
- **Isolated Dependencies**: PyTorch, TensorFlow, and LLM SDKs in separate containers
- **No Version Conflicts**: Each model can use different library versions
- **Granular Scaling**: Scale only the Transformer service if needed
- **Selective Deployment**: Update one service without affecting others

## Quick Start

### Development

```bash
# Start all microservices
docker-compose -f docker-compose.dev.yml up --build

# Verify services are ready (wait 30-60s for models to load)
curl http://localhost:8000/health
```

### Production

```bash
# Configure environment variables first
cp .env.prod .env.prod.local
# Edit .env.prod.local with your MongoDB URI and API keys

# Start services
docker-compose -f docker-compose.prod.yml up --build
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
- `model`: Model type (`lstm`, `transformer`, or `llm`)
- `model_variant`: Model variant (optional)
  - For `transformer`: `roberta` (fixed for experiments)
  - For `llm`: `gpt` (fixed for experiments)
- `episode_id`: Episode identifier (required; API returns 400 if missing)
- `note_date`: Clinical note date (ISO 8601, required; API returns 400 if missing)
- `save`: Save result (default: `true`)
- `expand`: Include full result in response (default: `false`)
- `normalize`: Enable UMLS normalization (currently ignored by gateway; default: `false`)
- `systems_csv`: Comma-separated target coding systems (relevant only if normalization were enabled)
- `restrict_types_csv`: Comma-separated entity types to normalize (relevant only if normalization were enabled)

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
| `TEMP` | Body temperature | 38.5°C |
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
| `EDAD` | Age | 65 años |
| `PESO` | Weight | 75 kg |
| `TALLA` | Height | 1.70 m |

### Clinical
| Entity | Description | Example |
|--------|-------------|---------|
| `DX` | Diagnosis | Neumonía adquirida en comunidad |
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
| `SAVE_RESULTS` | Enable result persistence | true |

### NER Service URLs (for microservices)

| Variable | Description | Default |
|----------|-------------|---------|
| `NER_TRANSFORMER_URL` | Transformer service URL | http://ner-transformer:8001 |
| `NER_LSTM_URL` | BiLSTM service URL | http://ner-bilstm:8002 |
| `NER_LLM_URL` | LLM service URL | http://ner-llm:8003 |

### Model Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `TRANSFORMER_BETO_MODEL_ID` | NOT USED - BETO disabled (legacy) | NicolasUnivalle/beto-vm-ner-full |
| `TRANSFORMER_ROBERTA_MODEL_ID` | RoBERTa model Hugging Face ID (ACTIVE) | NicolasUnivalle/roberta-vm-ner-full |

### API Keys (optional)

| Variable | Description | Required For |
|----------|-------------|--------------|
| `UMLS_APIKEY` | UMLS API key for normalization | Entity normalization (module; gateway disabled) |
| `ANTHROPIC_API_KEY` | NOT USED - Claude disabled (legacy) | Legacy (Claude) |
| `OPENAI_API_KEY` | OpenAI API key for GPT | LLM model with GPT |

## Model Selection

### BiLSTM (model=`lstm`)
Fast inference with moderate accuracy. Best for high-throughput scenarios.

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "model=lstm" \
  -F "text=Paciente con FiO2 60%" \
  -F "episode_id=EP-001" \
  -F "note_date=2024-01-15T10:00:00"
```

### Transformer (Recommended)
Best accuracy for clinical NER. Fixed to RoBERTa for experiments.

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "model=transformer" \
  -F "text=Paciente con FiO2 60%" \
  -F "episode_id=EP-001" \
  -F "note_date=2024-01-15T10:00:00"
```

### LLM
Highest flexibility with structured outputs. Fixed to GPT for experiments. Requires API key.

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "model=llm" \
  -F "text=Paciente con FiO2 60%" \
  -F "episode_id=EP-001" \
  -F "note_date=2024-01-15T10:00:00"
```

## Entity Normalization (Currently Disabled in Gateway)

The normalization module can link diagnosis entities to SNOMED-CT and ICD-10 codes, but
the gateway currently forces `normalize=false` (requests are accepted and ignored).

If/when normalization is re-enabled, use:

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "model=transformer" \
  -F "normalize=true" \
  -F "systems_csv=SNOMEDCT_US,ICD10CM" \
  -F "text=Diagnóstico: neumonía adquirida en comunidad" \
  -F "episode_id=EP-001" \
  -F "note_date=2024-01-15"
```

**Requirements (when enabled):**
- Set `UMLS_APIKEY` environment variable
- Register for UMLS API access at https://uts.nlm.nih.gov/

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
- All models (BiLSTM, Transformer RoBERTa, LLM GPT)
- Document processing (PDF/DOCX/TXT)
- Result retrieval
- Deduplication

## Deployment

### Development (Local MongoDB)

```bash
docker-compose -f docker-compose.dev.yml up --build
```

This configuration:
- Uses local MongoDB container
- Enables hot reload for development
- Exposes all service ports for debugging

### Production (MongoDB Atlas)

```bash
# 1. Configure environment
cp .env.prod .env.prod.local
# Edit MONGODB_URI, ANTHROPIC_API_KEY, OPENAI_API_KEY

# 2. Start services
docker-compose -f docker-compose.prod.yml up --build
```

### Production (Local MongoDB)

```bash
docker-compose -f docker-compose.prod.yml \
               -f docker-compose.prod.localdb.yml up --build
```

### Secure Exposure with Cloudflare Tunnel

```bash
# The tunnel only exposes the gateway (port 8000)
# NER services remain internal
cloudflared tunnel run medai-backend

# Verify external access
curl https://medai.your-domain.com/health
```

## Monitoring and Troubleshooting

### View Logs

```bash
# All services
docker-compose -f docker-compose.dev.yml logs -f

# Specific service
docker-compose -f docker-compose.dev.yml logs -f gateway
docker-compose -f docker-compose.dev.yml logs -f ner-transformer
docker-compose -f docker-compose.dev.yml logs -f ner-bilstm
docker-compose -f docker-compose.dev.yml logs -f ner-llm
```

### Health Checks

```bash
# Gateway (liveness)
curl http://localhost:8000/health

# NER services
curl http://localhost:8001/health   # Transformer (liveness)
curl http://localhost:8002/health   # BiLSTM (liveness)
curl http://localhost:8003/health   # LLM (liveness)

# NER services (readiness - checks if model is loaded)
curl http://localhost:8001/readyz  # Transformer
curl http://localhost:8002/readyz  # BiLSTM
curl http://localhost:8003/readyz  # LLM
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
docker-compose -f docker-compose.dev.yml build ner-bilstm
```

#### LLM returns 503
**Cause**: API keys not configured

**Solution**:
```bash
# Configure in .env.dev
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Restart service
docker-compose -f docker-compose.dev.yml restart ner-llm
```

#### Gateway cannot connect to services
**Cause**: Services in different network or not started

**Solution**:
```bash
# Verify all services in same network
docker network inspect medai-microservices-dev

# Check service status
docker-compose -f docker-compose.dev.yml ps

# Restart all
docker-compose -f docker-compose.dev.yml down
docker-compose -f docker-compose.dev.yml up
```

## Development

### Project Structure

```
medai-backend/
├── app/                       # API Gateway application
│   ├── config.py              # Application configuration
│   ├── deps.py                # FastAPI dependency injection
│   ├── indexes.py             # MongoDB index definitions
│   ├── main.py                # FastAPI application entry point
│   ├── schemas.py             # Pydantic request/response models
│   ├── repositories/          # Data access layer (Repository pattern)
│   │   ├── __init__.py
│   │   └── episode_repository.py  # Episode/note data access
│   ├── routers/
│   │   └── extract.py         # Extraction API endpoints
│   └── services/
│       ├── extraction_service.py   # Business logic layer (Service pattern)
│       ├── ner_client.py      # NER microservices client
│       ├── normalizer.py      # UMLS entity normalization
│       ├── pipeline.py        # Extraction orchestration
│       ├── registry.py        # Model registry
│       ├── text_utils.py      # Document text extraction
│       └── utils.py           # Shared utilities
├── services/                  # NER microservices
│   ├── ner-transformer/       # Transformer service (RoBERTa only)
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── app/
│   │       ├── main.py        # FastAPI app
│   │       ├── config.py
│   │       └── extractor.py   # NER extraction logic
│   ├── ner-bilstm/            # BiLSTM service (BiLSTM-CRF)
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── app/
│   │   └── models/            # BiLSTM model files
│   └── ner-llm/               # LLM service (GPT only)
│       ├── Dockerfile
│       ├── requirements.txt
│       └── app/
│           ├── main.py
│           ├── config.py
│           └── extractor.py
├── shared/                    # Shared code between services
│   ├── schemas.py             # Shared Pydantic models
│   └── utils.py               # Shared utilities
├── scripts/
│   └── export_openapi.py      # OpenAPI schema export (multi-service)
├── docker-compose.dev.yml     # Development configuration
├── docker-compose.prod.yml    # Production configuration
├── Dockerfile.gateway         # Gateway container image
├── requirements.txt           # All gateway dependencies
├── requirements-gateway.txt   # Minimal gateway dependencies
└── README.md                  # This file
```

### Running Tests

```bash
# No automated test suite is included in this repository.
# Use the quick curl-based checks above to validate a running stack.
```

### Code Formatting

```bash
black app/ scripts/
isort app/ scripts/
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

📘 **[MedAI Documentation](https://herreran903.github.io/docs-medai/)**

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
2. All tests pass
3. Documentation is updated for new features
4. Commit messages are descriptive

## Support

For issues and feature requests, please use the GitHub issue tracker.

---

**MedAI Backend** - Clinical NER for Mechanical Ventilation Notes
