#!/usr/bin/env python3
"""
OpenAPI Schema Export Script for MedAI Microservices.

This script exports OpenAPI specifications from all MedAI services:
- Gateway API (public API)
- NER Transformer microservice
- NER LSTM microservice
- NER LLM microservice

Purpose:
    The exported OpenAPI schemas are used by:

    - **Docusaurus**: Generates API documentation for the MedAI docs site
    - **Client Generators**: Creates typed API clients for frontend/service integration
    - **API Testing**: Validates API contracts in CI/CD pipelines

Output:
    Generates multiple OpenAPI JSON files in ``openapi/`` directory:

    - ``gateway.json``: Public API (POST /extract, GET /notes, etc.)
    - ``ner-transformer.json``: Transformer NER service (POST /predict)
    - ``ner-bilstm.json``: BiLSTM NER service (POST /predict)
    - ``ner-llm.json``: LLM NER service (POST /predict)

    Also creates ``openapi.json`` in root for backward compatibility.

Usage:
    From the repository root::

        python scripts/export_openapi.py

    Or with explicit Python path::

        PYTHONPATH=. python scripts/export_openapi.py

Environment:
    Sets ``DOCS_BUILD=1`` environment variable to signal documentation
    build mode. This prevents expensive model loading during schema generation.

Integration:
    This script is typically invoked by:

    - GitHub Actions workflow for documentation updates
    - Local development for API documentation preview
    - CI/CD pipelines for API contract validation

Example:
    >>> # Generate all OpenAPI schemas
    >>> python scripts/export_openapi.py
    >>>
    >>> # Verify gateway schema
    >>> cat openapi/gateway.json | jq '.info.title'
    "MedAI Backend"

See Also:
    - :mod:`app.main` for Gateway API definition
    - :mod:`services.ner-transformer.app.main` for Transformer service
    - :mod:`services.ner-bilstm.app.main` for BiLSTM service
    - :mod:`services.ner-llm.app.main` for LLM service
"""

import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict


@contextmanager
def _prepend_sys_path(path: Path):
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(path))
        except ValueError:
            pass


def _clear_app_modules() -> None:
    for name in list(sys.modules.keys()):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]


def _load_app(
    service_root: Path,
    app_module: str,
    app_attr: str = "app",
):
    _clear_app_modules()
    with _prepend_sys_path(service_root):
        module = importlib.import_module(app_module)
    return getattr(module, app_attr)


def export_service_schema(
    repo_root: Path,
    service_name: str,
    app_module: str,
    service_root: Path,
    app_attr: str = "app",
) -> Path:
    """
    Export OpenAPI schema for a single service.

    Args:
        repo_root: Repository root directory
        service_name: Service identifier (e.g., "gateway", "ner-transformer")
        app_module: Python module path (e.g., "app.main")
        service_root: Filesystem root for the service (added to sys.path for import)
        app_attr: FastAPI app attribute name (default: "app")

    Returns:
        Path: Output file path where schema was written

    Example:
        >>> export_service_schema(
        ...     repo_root=Path("/app"),
        ...     service_name="gateway",
        ...     app_module="app.main",
        ...     app_attr="app"
        ... )
        Path("/app/openapi/gateway.json")
    """
    # Dynamically import the FastAPI app with a service-specific sys.path
    app = _load_app(service_root=service_root, app_module=app_module, app_attr=app_attr)

    # Extract OpenAPI schema
    schema: Dict[str, Any] = app.openapi()

    # Ensure output directory exists
    output_dir = repo_root / "openapi"
    output_dir.mkdir(exist_ok=True)

    # Write schema to file
    output_path = output_dir / f"{service_name}.json"
    output_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    return output_path


def main() -> int:
    """
    Export OpenAPI schemas from all MedAI services.

    This function:

    1. Configures Python path for module imports
    2. Sets documentation build environment flag
    3. Exports schemas for all services:
       - Gateway API (app/main.py)
       - NER Transformer (services/ner-transformer/app/main.py)
       - NER BiLSTM (services/ner-bilstm/app/main.py)
       - NER LLM (services/ner-llm/app/main.py)
    4. Creates backward-compatible openapi.json (gateway schema)

    Returns:
        int: Exit code (0 for success).

    Output Files:
        Creates:
        - ``openapi/gateway.json``: Gateway API schema
        - ``openapi/ner-transformer.json``: Transformer service schema
        - ``openapi/ner-bilstm.json``: BiLSTM service schema
        - ``openapi/ner-llm.json``: LLM service schema
        - ``openapi.json``: Gateway schema (backward compatibility)

    Note:
        The ``DOCS_BUILD`` environment variable is set to signal that
        services are being loaded for documentation purposes only.
        This prevents expensive model loading.
    """
    # Resolve repository root directory
    repo_root = Path(__file__).resolve().parents[1]

    # Add repository root to Python path for imports
    sys.path.insert(0, str(repo_root))

    # Set documentation build flag to skip model loading
    os.environ["DOCS_BUILD"] = "1"

    print("Exporting OpenAPI schemas for all MedAI services...\n")

    # Define services to export
    services = [
        {
            "name": "gateway",
            "module": "app.main",
            "root": repo_root,
            "description": "Gateway API (public endpoints)",
        },
        {
            "name": "ner-transformer",
            "module": "app.main",
            "root": repo_root / "services" / "ner-transformer",
            "description": "Transformer NER Service",
        },
        {
            "name": "ner-bilstm",
            "module": "app.main",
            "root": repo_root / "services" / "ner-bilstm",
            "description": "LSTM NER Service",
        },
        {
            "name": "ner-llm",
            "module": "app.main",
            "root": repo_root / "services" / "ner-llm",
            "description": "LLM NER Service",
        },
    ]

    # Export each service schema
    exported_paths = []
    for service in services:
        try:
            output_path = export_service_schema(
                repo_root=repo_root,
                service_name=service["name"],
                app_module=service["module"],
                service_root=service["root"],
            )
            exported_paths.append(output_path)
            print(f"✓ {service['description']:30s} → {output_path.relative_to(repo_root)}")

        except Exception as e:
            print(f"✗ {service['description']:30s} → ERROR: {e}")
            # Continue with other services even if one fails
            continue

    # Create backward-compatible openapi.json (gateway schema)
    gateway_schema_path = repo_root / "openapi" / "gateway.json"
    if gateway_schema_path.exists():
        legacy_path = repo_root / "openapi.json"
        legacy_path.write_text(gateway_schema_path.read_text(), encoding="utf-8")
        print(f"\n✓ Legacy openapi.json created (gateway schema copy)")

    print(f"\n✓ Exported {len(exported_paths)}/{len(services)} schemas successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
