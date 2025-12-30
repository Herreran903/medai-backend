#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    os.environ["DOCS_BUILD"] = "1"

    from app.main import app

    schema = app.openapi()
    output_path = repo_root / "openapi.json"
    output_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
