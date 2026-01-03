# MedAI Backend

Clinical Named Entity Recognition (NER) API for extracting medical entities from mechanical ventilation clinical notes.

[![Documentation](https://img.shields.io/badge/docs-Docusaurus-blue)](https://herreran903.github.io/docs-medai/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

MedAI Backend is a production-grade REST API service that provides clinical Named Entity Recognition (NER) capabilities for Spanish medical text, specifically optimized for mechanical ventilation clinical notes. The service extracts structured clinical entities such as ventilation parameters, vital signs, diagnoses, and laboratory values from unstructured clinical text.

### Key Features

- **Multiple NER Models**: Choose from LSTM, Transformer (BETO/RoBERTa), or LLM-based extraction
- **Entity Normalization**: Optional UMLS-based normalization to SNOMED-CT and ICD-10 codes
- **Batch Processing**: Process multiple clinical notes in a single request
- **Document Support**: Accept PDF, DOCX, and plain text files
- **Persistent Storage**: MongoDB-based storage with content deduplication
- **OpenAPI Documentation**: Auto-generated API documentation with Swagger/ReDoc

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MedAI Backend                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   FastAPI   │───▶│   Pipeline  │───▶│   Models    │         │
│  │   Router    │    │   Service   │    │  Registry   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                  │                  │                 │
│         │                  │           ┌──────┴──────┐          │
│         │                  │           │             │          │
│         ▼                  ▼           ▼             ▼          │
│  ┌─────────────┐    ┌─────────────┐  ┌─────┐  ┌───────────┐    │
│  │   Storage   │    │ Normalizer  │  │LSTM │  │Transformer│    │
│  │  (MongoDB)  │    │   (UMLS)    │  └─────┘  └───────────┘    │
│  └─────────────┘    └─────────────┘           ┌───────────┐    │
│                                               │    LLM    │    │
│                                               └───────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Description |
|-----------|-------------|
| **FastAPI Router** | REST API endpoints for extraction and retrieval |
| **Pipeline Service** | Orchestrates model selection, extraction, and normalization |
| **Model Registry** | Centralized registry of available NER models |
| **LSTM Extractor** | BiLSTM-CRF model for fast inference |
| **Transformer Extractor** | Fine-tuned BETO/RoBERTa for high accuracy |
| **LLM Extractor** | Claude/GPT-based extraction with structured outputs |
| **Normalizer** | UMLS-based entity normalization to standard codes |
| **Storage** | MongoDB persistence with deduplication |

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

### Vital Signs
| Entity | Description | Example |
|--------|-------------|---------|
| `TEMP` | Body temperature | 38.5°C |
| `PA` | Blood pressure | 120/80 mmHg |
| `FC` | Heart rate | 92 lpm |
| `SAO2` | Oxygen saturation | 95% |

### Arterial Blood Gases
| Entity | Description | Example |
|--------|-------------|---------|
| `PH` | Arterial pH | 7.35 |
| `PACO2` | Partial pressure of CO2 | 45 mmHg |
| `PAO2` | Partial pressure of O2 | 80 mmHg |
| `PAFI` | PaO2/FiO2 ratio | 250 |

### Clinical
| Entity | Description | Example |
|--------|-------------|---------|
| `DX` | Diagnosis | Neumonía adquirida en comunidad |

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

#### `POST /extract-batch`
Process multiple files in a single request.

```bash
curl -X POST "http://localhost:8000/extract-batch" \
  -F "files=@nota_001.pdf" \
  -F "files=@nota_002.pdf" \
  -F "model=transformer" \
  -F 'notes_meta=[{"filename":"nota_001.pdf","episode_id":"EP-001","note_date":"2024-01-15"}]'
```

### Retrieval

#### `GET /notes/{note_id}`
Retrieve a stored extraction result.

```bash
curl "http://localhost:8000/notes/550e8400-e29b-41d4-a716-446655440000"
```

### Health

#### `GET /healthz`
Health check endpoint for container orchestration.

```bash
curl "http://localhost:8000/healthz"
```

## Installation

### Prerequisites

- Python 3.10+
- MongoDB 4.4+
- Docker (optional, for containerized deployment)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/herreran903/medai-backend.git
   cd medai-backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.dev .env
   # Edit .env with your configuration
   ```

5. **Start MongoDB**
   ```bash
   docker run -d -p 27017:27017 --name medai-mongo mongo:6
   ```

6. **Run the application**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Docker Deployment

#### Development
```bash
docker-compose -f docker-compose.dev.yml up --build
```

#### Production
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Configuration

Configuration is managed through environment variables. Create a `.env` file based on `.env.dev`:

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_NAME` | Application name | MedAI Backend |
| `ENVIRONMENT` | Deployment environment | dev |
| `HOST` | Server bind address | 0.0.0.0 |
| `PORT` | Server port | 8000 |
| `LOG_LEVEL` | Logging level | info |
| `MONGODB_URI` | MongoDB connection string | mongodb://mongo:27017 |
| `MONGODB_DB` | Database name | medai |
| `SAVE_RESULTS` | Enable result persistence | true |
| `UMLS_APIKEY` | UMLS API key for normalization | (optional) |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude | (optional) |
| `OPENAI_API_KEY` | OpenAI API key for GPT | (optional) |
| `TRANSFORMER_BETO_MODEL_ID` | BETO model Hugging Face ID | NicolasUnivalle/beto-vm-ner-full |
| `TRANSFORMER_ROBERTA_MODEL_ID` | RoBERTa model Hugging Face ID | NicolasUnivalle/roberta-vm-ner-full |

## Model Selection

### LSTM
Fast inference with moderate accuracy. Best for high-throughput scenarios.

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "model=lstm" \
  -F "text=..." \
  -F "episode_id=..." \
  -F "note_date=..."
```

### Transformer (Recommended)
Best accuracy for clinical NER. Supports BETO and RoBERTa variants.

```bash
# BETO (default)
curl -X POST "http://localhost:8000/extract" \
  -F "model=transformer" \
  -F "model_variant=beto" \
  -F "text=..." \
  -F "episode_id=..." \
  -F "note_date=..."

# RoBERTa
curl -X POST "http://localhost:8000/extract" \
  -F "model=transformer" \
  -F "model_variant=roberta" \
  -F "text=..." \
  -F "episode_id=..." \
  -F "note_date=..."
```

### LLM
Highest flexibility with structured outputs. Requires API keys.

```bash
# Claude (default)
curl -X POST "http://localhost:8000/extract" \
  -F "model=llm" \
  -F "model_variant=claude" \
  -F "text=..." \
  -F "episode_id=..." \
  -F "note_date=..."

# GPT
curl -X POST "http://localhost:8000/extract" \
  -F "model=llm" \
  -F "model_variant=gpt" \
  -F "text=..." \
  -F "episode_id=..." \
  -F "note_date=..."
```

## Entity Normalization

Enable UMLS-based normalization to link diagnosis entities to SNOMED-CT and ICD-10 codes:

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "model=transformer" \
  -F "normalize=true" \
  -F "systems_csv=SNOMEDCT_US,ICD10CM" \
  -F "text=Diagnóstico: neumonía adquirida en comunidad" \
  -F "episode_id=EP-001" \
  -F "note_date=2024-01-15"
```

**Requirements:**
- Set `UMLS_APIKEY` environment variable
- Register for UMLS API access at https://uts.nlm.nih.gov/

## OpenAPI Documentation

The API documentation is automatically generated and available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Exporting OpenAPI Schema

Generate the OpenAPI schema for documentation or client generation:

```bash
python scripts/export_openapi.py
```

This creates `openapi.json` in the repository root.

## Project Structure

```
medai-backend/
├── app/
│   ├── config.py           # Application configuration
│   ├── deps.py             # FastAPI dependencies
│   ├── indexes.py          # MongoDB index definitions
│   ├── main.py             # FastAPI application entry point
│   ├── schemas.py          # Pydantic request/response models
│   ├── models/
│   │   ├── llm.py          # LLM-based extractor (Claude/GPT)
│   │   ├── lstm.py         # BiLSTM-CRF extractor
│   │   └── transformer.py  # Transformer extractor (BETO/RoBERTa)
│   ├── routers/
│   │   └── extract.py      # Extraction API endpoints
│   └── services/
│       ├── normalizer.py   # UMLS entity normalization
│       ├── pipeline.py     # Extraction orchestration
│       ├── registry.py     # Model registry
│       ├── semantic_sim.py # Semantic similarity computation
│       ├── store.py        # MongoDB storage
│       ├── text_utils.py   # Document text extraction
│       └── translator.py   # Spanish-English translation
├── scripts/
│   └── export_openapi.py   # OpenAPI schema export
├── docker-compose.dev.yml  # Development Docker configuration
├── docker-compose.prod.yml # Production Docker configuration
├── Dockerfile              # Production container image
├── Dockerfile.dev          # Development container image
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

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

## Development

### Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
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
