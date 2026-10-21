# Developer entry points. Python runs through uv (Python 3.11 from .python-version);
# the web app uses npm inside apps/web.
SHELL := /bin/bash
.DEFAULT_GOAL := help

# Some hosts export PYTHONPATH for unrelated toolchains (e.g. ROS); it must not leak
# third-party pytest plugins or modules into this project's interpreter.
unexport PYTHONPATH

UV ?= uv
NPM ?= npm
WEB := apps/web
COMPOSE ?= docker compose
TEST_COMPOSE := $(COMPOSE) -f docker-compose.test.yml
WEB_PORT ?= 3000
API_PORT ?= 8000
MODEL_SERVICE_PORT ?= 8010
MCP_PORT ?= 8020

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install Python (uv) and web (npm) dependencies from lockfiles
	$(UV) sync --frozen
	$(NPM) --prefix $(WEB) ci --no-audit --no-fund

# ---------------------------------------------------------------- quality gates
.PHONY: lint
lint: ## ruff (lint + format check) and eslint
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(NPM) --prefix $(WEB) run lint

.PHONY: format
format: ## Auto-format Python code
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

.PHONY: typecheck
typecheck: ## mypy --strict and tsc --noEmit
	$(UV) run mypy packages apps conftest.py scripts
	$(NPM) --prefix $(WEB) run typecheck

.PHONY: test-infra-up
test-infra-up: ## Start throwaway Postgres+Redis for integration tests (ports 55432/56379)
	$(TEST_COMPOSE) up -d --wait

.PHONY: test-infra-down
test-infra-down: ## Stop and remove the integration-test containers
	$(TEST_COMPOSE) down -v

.PHONY: test-python
test-python: ## pytest (needs `make test-infra-up` or TEST_DATABASE_URL/TEST_REDIS_URL)
	$(UV) run pytest

.PHONY: test-web
test-web: ## Vitest unit/component tests
	$(NPM) --prefix $(WEB) test

.PHONY: test
test: test-python test-web ## All tests

.PHONY: openapi
openapi: ## Regenerate the OpenAPI schema and the web app's TypeScript types
	$(UV) run python -m compliance_api.openapi $(WEB)/src/lib/api/openapi.json
	$(NPM) --prefix $(WEB) run generate:api

.PHONY: openapi-check
openapi-check: ## Fail if the committed schema or TS types are stale
	@tmp=$$(mktemp -d) && trap 'rm -rf "$$tmp"' EXIT && \
	$(UV) run python -m compliance_api.openapi "$$tmp/openapi.json" && \
	diff -u $(WEB)/src/lib/api/openapi.json "$$tmp/openapi.json" && \
	(cd $(WEB) && npx openapi-typescript "$$tmp/openapi.json" -o "$$tmp/schema.d.ts" >/dev/null) && \
	diff -u $(WEB)/src/lib/api/schema.d.ts "$$tmp/schema.d.ts" && \
	echo "OpenAPI schema and TypeScript types are up to date"

.PHONY: build-web
build-web: ## Production Next.js build
	$(NPM) --prefix $(WEB) run build

.PHONY: ci
ci: lint typecheck openapi-check test build-web ## Everything the CI python+web jobs run

# ---------------------------------------------------------------- docker stack
.PHONY: build
build: ## Build all service images
	$(COMPOSE) build

.PHONY: up
up: ## Start the full stack and wait until every service is healthy
	$(COMPOSE) up -d --build --wait

.PHONY: down
down: ## Stop the stack (keeps volumes)
	$(COMPOSE) down

.PHONY: ps
ps: ## Show service health
	$(COMPOSE) ps

.PHONY: smoke
smoke: ## Phase 0 end-to-end smoke test against the running stack
	$(UV) run python scripts/smoke_phase0.py \
		--web http://127.0.0.1:$(WEB_PORT) --api http://127.0.0.1:$(API_PORT) \
		--model-service http://127.0.0.1:$(MODEL_SERVICE_PORT) --mcp http://127.0.0.1:$(MCP_PORT)

.PHONY: fetch-artifacts
fetch-artifacts: ## Download + verify the pinned embedding model and ContractNLI into ./.cache
	ARTIFACT_CACHE_DIR=.cache $(UV) run python -m compliance_ingestion.embeddings fetch
	ARTIFACT_CACHE_DIR=.cache $(UV) run python -m compliance_seed.fetch

.PHONY: seed
seed: ## Seed demo user/workspace/register/NDAs in the running stack and wait for ingestion
	$(COMPOSE) exec worker python -m compliance_seed --wait 900

.PHONY: smoke-phase1
smoke-phase1: ## Phase 1 end-to-end smoke test (upload -> ingest -> viewer data) against the stack
	$(UV) run python scripts/smoke_phase1.py --web http://127.0.0.1:$(WEB_PORT)

.PHONY: migrate
migrate: ## Apply database migrations (uses DATABASE_URL)
	$(UV) run python -m compliance_core.db.migrate upgrade head

# ---------------------------------------------------------------- later phases
NOT_IMPLEMENTED = @echo "make $@: not implemented yet ($(1) of docs/BUILD_HANDOFF.md)" >&2; exit 2

.PHONY: train eval-model export-model eval-retrieval eval-all
eval-retrieval: ## [Phase 2] Retrieval eval (not implemented)
	$(call NOT_IMPLEMENTED,Phase 2)
train: ## [Phase 3] Train the rule-check model (not implemented)
	$(call NOT_IMPLEMENTED,Phase 3)
eval-model: ## [Phase 3] Model evaluation (not implemented)
	$(call NOT_IMPLEMENTED,Phase 3)
export-model: ## [Phase 3] ONNX export + int8 quantization (not implemented)
	$(call NOT_IMPLEMENTED,Phase 3)
eval-all: ## [Phase 6] Full eval suite (not implemented)
	$(call NOT_IMPLEMENTED,Phase 6)
