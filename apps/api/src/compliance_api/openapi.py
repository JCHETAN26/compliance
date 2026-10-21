"""Write the OpenAPI schema to a file (input for the web app's generated TS types).

``python -m compliance_api.openapi apps/web/src/lib/api/openapi.json``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import SecretStr

from compliance_api.main import create_app
from compliance_core.config import AppEnv, Settings


def render() -> str:
    # Schema generation needs no live resources; the lifespan is never started.
    app = create_app(Settings(app_env=AppEnv.TEST, jwt_secret=SecretStr("x" * 32)))
    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m compliance_api.openapi OUTPUT_PATH")
    Path(sys.argv[1]).write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
