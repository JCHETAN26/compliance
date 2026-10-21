# Independent build supervision

The user assigned all implementation to Claude Code and management, review, and
verification to Codex. The authoritative requirements are in `BUILD_HANDOFF.md`.

## Phase acceptance ledger

| Phase | Implementation | Independently verified acceptance |
|---|---|---|
| 0 — Skeleton | Complete | Accepted: local gates and hosted CI passed |
| 1 — Ingestion and viewer | Complete locally | Local gates accepted; hosted CI pending |
| 2 — Retrieval | Not started | Pending |
| 3 — Trained model | Not started | Pending |
| 4 — Agents and review UI | Not started | Pending |
| 5 — MCP | Not started | Pending |
| 6 — Evaluation and polish | Not started | Pending |

No phase is accepted solely on the implementer's report. Checks must be run and
results recorded. Mock, synthetic, and smoke results do not establish performance
on official datasets, real-provider behavior, or Claude Desktop interoperability.

## Initial environment observations

- Docker daemon and Compose are available outside the execution sandbox.
- Host Python is 3.10.12; the application requires Python 3.11+ in containers.
- Node 22.14.0 and npm 10.9.2 are available.
- `nvidia-smi` fails to communicate with the NVIDIA driver; GPU training is not
  currently verified as available.
- Initial free disk: approximately 7.4 GB. Monitor build and model storage needs;
  do not delete unrelated files, images, or volumes.
- Local Git identity: `JCHETAN26`, account-specific GitHub noreply email.
- Remote: private `JCHETAN26/compliance` repository.
- No AI coauthor trailers. Do not publish credentials or raw training datasets.

## User-confirmed choices

- Full GPU training: Colab when the training phase is ready.
- Real application LLM provider and zero-shot baseline: Anthropic API.
- Claude Code Max remains the implementation tool; it is not the application's
  Anthropic API credential or a substitute for real-provider validation.

## Independent Phase 0 checks — first backend pass

Commands run by Codex after Claude reported the backend ready:

- `env -u PYTHONPATH .venv/bin/python -m pytest -q`: **80 passed in 5.97s**.
  Run outside the sandbox against the isolated project's Postgres/Redis test stack.
- `env -u PYTHONPATH .venv/bin/ruff check .`: passed.
- `.venv/bin/ruff format --check .`: 55 files already formatted.
- `env -u PYTHONPATH .venv/bin/mypy packages apps/api apps/worker apps/model-service apps/mcp-server`:
  passed, 52 source files.

These checks do not yet accept Phase 0: the complete container build, web checks,
browser sign-up/workspace flow, and hosted CI remain pending.

## Independent Phase 0 checks — first frontend pass

- `npm run lint && npm run typecheck && npm test` in `apps/web`: passed,
  **53 tests in 8 files** (1.83s).
- An earlier typecheck during new test creation failed on the injected fetch
  function's TypeScript signature. Claude corrected the signature; the combined
  check above then passed. No implementation edits were made by Codex.

## Independent Phase 0 checks — running containers and browser

- `docker compose ps --format json`: all seven services running and healthy.
- App host port is 3300 (`WEB_PORT=3300`); the normal default remains 3000.
- Stable browser walkthrough command in `PHASE0_REVIEW.md`: passed with no console
  or page errors. Covers signup, keyboard submission, workspace CRUD, membership,
  backend error display, signout/login, and server/client session expiry.
- Workspace-detail screenshot visually inspected by Codex.
- First independent browser attempt overlapped with test edits and timed out;
  fixed-copy rerun succeeded, as recorded in the review file.

## Reviewed configuration installation

Claude authored `.pre-commit-config.yaml` but its noninteractive file tool requested
an interactive permission for that sensitive filename. Codex reviewed the proposed
content (local uv/npm quality hooks), copied the exact unmodified Claude-authored
artifact from the session log into the repository, and validated it with:

`PRE_COMMIT_HOME=/tmp/compliance-supervision/pre-commit .venv/bin/pre-commit validate-config`

Passed. A first validation attempt used the tool's default cache location, which
is read-only in the sandbox; the /tmp cache override resolved that environment
failure. No product implementation was authored by Codex.

## Independent Phase 0 final local gate

- `make ci` (log: `/tmp/compliance-supervision/phase0-final-checks.log`): **exit 0**.
  Includes Ruff lint and format, strict mypy (including scripts and conftest), ESLint,
  TypeScript, OpenAPI schema/type drift checks, 80 pytest tests, 53 Vitest tests,
  and the Next.js production build.
- `make smoke WEB_PORT=3300`: **exit 0, all 10 checks passed**, including
  browser-facing proxy auth, CSRF, workspace/membership controls, token revocation,
  and API/model/MCP health behavior.
- Redis recovery independently confirmed: all seven containers healthy, and
  `curl -fsS http://127.0.0.1:3300/healthz` returned web/API status `ok`.

Local Phase 0 acceptance is complete. Hosted CI is still required before starting
Phase 1 under the handoff's sequential gates.

## Phase 0 commit and hosted CI

- Commit `e6b709e944208a5a411fd3e6127054dd0b1da3ae` pushed to `main` on the private
  `JCHETAN26/compliance` repository. Author confirmed as `JCHETAN26` with the
  account-specific noreply email; no coauthor trailer.
- Hosted CI run: https://github.com/JCHETAN26/compliance/actions/runs/39678264159
  Python and web jobs passed; Docker build/smoke job pending at time of this entry.

## Scoped cache cleanup

Validated the 80 proposed BuildKit record IDs against fresh daemon metadata:
all private, reclaimable, and created during this project's Phase 0 build window.
Pruned using individual `--filter id=<verified-id>` arguments only. Free disk rose
from roughly 2.2 GB to 4.0 GB. 30 records were removed; other selected records remain
because of cache dependencies. All 48 older cache records were independently
confirmed preserved. No images, containers, volumes, source, or other-project
artifacts were removed. All seven application services remain healthy.

## Phase 0 accepted

Hosted run `39678264159` completed successfully: Python, web, and Docker build +
Compose smoke-test jobs all passed for commit `e6b709e`. Phase 1 is authorized to
begin. Full project acceptance remains pending Phases 1–6.

## Independent dataset-source check before Phase 1 sample data

Checked the publishers' pages on 2026-10-21 before using dataset content:

- ContractNLI: https://stanfordnlp.github.io/contract-nli/ — publisher identifies
  CC BY 4.0, 607 NDAs, 17 hypotheses, official document spans and annotations.
- CUAD: https://www.atticusprojectai.org/legal/ — publisher lists CUAD under
  CC BY 4.0. CUAD usage is a later-phase requirement.

The implementer must still record provenance, attribution and the distributed
license terms in `ml/data/LICENSES.md` before downloads/use. Keep raw data out of Git.

## Phase 1 local acceptance

See `PHASE1_REVIEW.md` for independent checks and corrections. Full final `make ci`
passed with 197 Python tests and 73 frontend tests, plus lint, strict typing, schema
drift checks and production build. Production browser and fresh-volume automatic
seed smoke both passed. The temporary acceptance project was removed; the main app
and data are retained. Hosted CI and final commit hygiene remain pending before
Phase 2 can begin. No full-project completion is claimed.
