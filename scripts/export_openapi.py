#!/usr/bin/env python3
"""
OpenAPI Schema Export Script.

This script exports the OpenAPI specification from the MedAI FastAPI application
to a JSON file for documentation generation and API client code generation.

Purpose:
    The exported OpenAPI schema is used by:

    - **Docusaurus**: Generates API documentation for the MedAI docs site
    - **Client Generators**: Creates typed API clients for frontend integration
    - **API Testing**: Validates API contracts in CI/CD pipelines

Output:
    Generates ``openapi.json`` in the repository root directory containing
    the complete OpenAPI 3.0 specification with:

    - All endpoint definitions
    - Request/response schemas
    - Parameter descriptions
    - Error responses

Usage:
    From the repository root::

        python scripts/export_openapi.py

    Or with explicit Python path::

        PYTHONPATH=. python scripts/export_openapi.py

Environment:
    Sets ``DOCS_BUILD=1`` environment variable to signal documentation
    build mode. This can be used by the application to adjust behavior
    during schema generation (e.g., skip model loading).

Integration:
    This script is typically invoked by:

    - GitHub Actions workflow for documentation updates
    - Local development for API documentation preview
    - CI/CD pipelines for API contract validation

Example:
    >>> # Generate OpenAPI schema
    >>> python scripts/export_openapi.py
    >>>
    >>> # Verify output
    >>> cat openapi.json | jq '.info.title'
    "MedAI Backend"

See Also:
    - :mod:`app.main` for FastAPI application definition
    - :mod:`app.routers.extract` for endpoint documentation
    - :mod:`app.schemas` for Pydantic model definitions
"""

import json
import os
import sys
from pathlib import Path


def main() -> int:
    """
    Export OpenAPI schema from FastAPI application.

    This function:

    1. Configures Python path for module imports
    2. Sets documentation build environment flag
    3. Imports the FastAPI application
    4. Extracts the OpenAPI schema
    5. Writes schema to ``openapi.json``

    Returns:
        int: Exit code (0 for success).

    Output File:
        Creates ``openapi.json`` in the repository root with the complete
        OpenAPI 3.0 specification.

    Note:
        The ``DOCS_BUILD`` environment variable is set to signal that
        the application is being loaded for documentation purposes.
        This can be used to skip expensive initialization (e.g., model loading).
    """
    # Resolve repository root directory
    repo_root = Path(__file__).resolve().parents[1]

    # Add repository root to Python path for imports
    sys.path.insert(0, str(repo_root))

    # Set documentation build flag
    os.environ["DOCS_BUILD"] = "1"

    # Import FastAPI application
    from app.main import app

    # Extract OpenAPI schema
    schema = app.openapi()

    # Write schema to file
    output_path = repo_root / "openapi.json"
    output_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print(f"OpenAPI schema exported to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
