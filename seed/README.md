# seed/ — reproducible demo workspace

`compliance_seed` builds the flagship demo data and loads it through the same
validation and registration path as uploads:

- demo user **demo@example.com / demo1234** (the credentials the handoff mandates for
  local demos);
- workspace **"Demo - Vendor confidentiality review"**;
- `vendor_register.xlsx`, a synthetic register of 11 vendors (name, data access level,
  contract file name, renewal date). A title row sits above the header to exercise
  header detection. A second "Sources" sheet records each contract's provenance.
- 10 contracts, all NDAs from the ContractNLI **test split**:
  - 7 original PDFs, re-saved with provenance metadata;
  - 2 DOCX files typeset from the dataset text;
  - 1 **image-only scanned PDF** with no text layer, so ingestion must OCR it.
- 1 vendor (Northwind Analytics, fictitious) whose listed contract is deliberately
  missing, for the later cross-document "missing contract" check.

The mapping lives in `src/compliance_seed/manifest.yaml`. Vendor access levels and
renewal dates are **synthetic**; the vendor names are taken from the contracts and imply
no real relationship.

## Running

- **Automatic:** on a clean `docker compose up` (no demo workspace yet), the worker
  queues a one-off seed job. Later restarts never re-seed, so a deleted demo document
  stays deleted and a changed password stays changed.
- **Explicit:** `make seed`, which runs `python -m compliance_seed --wait 900` in the
  worker. It is idempotent:
  - an existing demo user keeps its password;
  - the workspace is reused;
  - files already present (same SHA-256 in the workspace) are skipped and missing ones
    added;
  - it waits until every seeded document is READY or FAILED, and exits non-zero on
    failure or timeout.
- **Production:** seeding is refused when `APP_ENV=production`.

Generation is deterministic (fixed PDF IDs and metadata dates, fixed zip timestamps, a
seeded scan-noise RNG), so reruns produce byte-identical files and SHA-256 dedupe works.

## Data and license

The ContractNLI archive (CC BY 4.0, Koreeda & Manning 2021; see
`ml/data/LICENSES.md`) is downloaded once, verified against a pinned SHA-256, and cached
in the git-ignored artifact cache. Generated files go to a temporary directory and
into document storage; nothing derived from the dataset is committed.
