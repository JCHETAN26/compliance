"""``python -m compliance_seed.fetch``: download and verify the pinned ContractNLI archive
into ``$ARTIFACT_CACHE_DIR/datasets`` (license: see ml/data/LICENSES.md)."""

from __future__ import annotations

import sys

from compliance_core.config import get_settings
from compliance_core.downloads import DownloadError
from compliance_seed.job import ensure_dataset


def main() -> int:
    try:
        path = ensure_dataset(get_settings(), force_download=True)
    except DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"ContractNLI archive verified at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
