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
	$(UV) run mypy packages apps conftest.py scripts evals ml
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
	$(UV) run python -m pytest -q ml/tests

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
fetch-artifacts: ## Download + verify the pinned embedding + reranker models and ContractNLI into ./.cache
	ARTIFACT_CACHE_DIR=.cache $(UV) run python -m compliance_ingestion.embeddings fetch
	ARTIFACT_CACHE_DIR=.cache $(UV) run python -m compliance_model_service.rerank fetch
	ARTIFACT_CACHE_DIR=.cache $(UV) run python -m compliance_seed.fetch

.PHONY: seed
seed: ## Seed demo user/workspace/register/NDAs in the running stack and wait for ingestion
	$(COMPOSE) exec worker python -m compliance_seed --wait 900

.PHONY: smoke-phase1
smoke-phase1: ## Phase 1 end-to-end smoke test (upload -> ingest -> viewer data) against the stack
	$(UV) run python scripts/smoke_phase1.py --web http://127.0.0.1:$(WEB_PORT)

.PHONY: smoke-phase2
smoke-phase2: ## Phase 2 end-to-end smoke test (hybrid search, exact spans, isolation) against the stack
	$(UV) run python scripts/smoke_phase2.py --web http://127.0.0.1:$(WEB_PORT) \
		--model-service http://127.0.0.1:$(MODEL_SERVICE_PORT)

.PHONY: smoke-phase3
smoke-phase3: ## Phase 3 smoke (rule-check status/UI banner source, /v1/check contract) against the stack
	$(UV) run python scripts/smoke_phase3.py --web http://127.0.0.1:$(WEB_PORT) \
		--model-service http://127.0.0.1:$(MODEL_SERVICE_PORT)

.PHONY: smoke-phase4
smoke-phase4: ## Phase 4 smoke (synthetic flagship review through the web origin; mock LLM by default)
	$(UV) run python scripts/smoke_phase4.py --web http://127.0.0.1:$(WEB_PORT)

.PHONY: migrate
migrate: ## Apply database migrations (uses DATABASE_URL)
	$(UV) run python -m compliance_core.db.migrate upgrade head

# ---------------------------------------------------------------- later phases
NOT_IMPLEMENTED = @echo "make $@: not implemented yet ($(1) of docs/BUILD_HANDOFF.md)" >&2; exit 2

# Retrieval eval (Phase 2). Needs `make test-infra-up` (uses an isolated database
# compliance_eval on the throwaway test Postgres; override with EVAL_DATABASE_URL) and
# `make fetch-artifacts`. Writes evals/results/ and evals/retrieval/frozen_config.json.
.PHONY: eval-retrieval eval-retrieval-dev eval-retrieval-test eval-hnsw-sweep
eval-retrieval: ## ContractNLI retrieval eval: tune on DEV, freeze, measure TEST, HNSW sweep
	$(UV) run python -m evals.retrieval.run all
	$(UV) run python -m evals.retrieval.sweep
eval-retrieval-dev: ## DEV tuning only (rewrites evals/retrieval/frozen_config.json)
	$(UV) run python -m evals.retrieval.run dev
eval-retrieval-test: ## TEST measurement with the frozen configuration only
	$(UV) run python -m evals.retrieval.run test
eval-hnsw-sweep: ## HNSW m / ef_construction / ef_search sweep (ANN recall vs latency)
	$(UV) run python -m evals.retrieval.sweep

# ---------------------------------------------------------------- rule-check model (Phase 3)
# ML tool packages (onnx, onnxscript, sentencepiece, anthropic) run as a pinned uv
# overlay so that uv.lock and the shared Docker dependency layer stay unchanged.
ML := $(UV) run --frozen --with-requirements ml/requirements-tools.txt python
CONFIG ?= ml/train/configs/deberta_v3_base.yaml
RUN_DIR ?= .cache/ml/runs/$(basename $(notdir $(CONFIG)))
ABLATION_RUN_DIR ?=
UNSEEN_RUN_DIR ?=
ARTIFACT_DIR ?= .cache/ml/artifacts/$(notdir $(RUN_DIR))
HOST_LABEL ?= $(shell hostname -s)
SPLIT ?= dev
LLM_ARGS ?=
EVAL_ARGS ?=
EXPORT_ARGS ?=

.PHONY: ml-data ml-data-smoke baselines baseline-llm train train-smoke eval-model export-model \
	bench-model eval-onnx ml-smoke ml-test install-model import-colab-bundle eval-all
ml-data: ## ContractNLI -> segments, gold mapping, BGE candidate ranking (.cache/ml/data/full; ~20 min CPU)
	$(UV) run python -m ml.data.prepare
ml-data-smoke: ## Small data subset for the CPU smoke (.cache/ml/data/smoke)
	$(UV) run python -m ml.data.prepare --subset smoke
baselines: ## Majority + TF-IDF/logistic-regression baselines on the prepared full data
	$(UV) run python -m ml.baselines.classic
baseline-llm: ## Anthropic zero-shot baseline: SPLIT=dev|test LLM_ARGS="--dry-run" (needs ANTHROPIC_API_KEY)
	$(ML) -m ml.baselines.llm --split $(SPLIT) $(LLM_ARGS)
train: ## Train: CONFIG=ml/train/configs/<x>.yaml RUN_DIR=<dir> (production configs need CUDA: Colab)
	$(ML) -m ml.train.train --config $(CONFIG) --run-dir $(RUN_DIR)
train-smoke: ## CPU smoke training (tiny random backbone, data subset; ~1 min; NOT a model)
	$(ML) -m ml.train.train --config ml/train/configs/smoke.yaml --run-dir .cache/ml/runs/smoke
eval-model: ## Evaluate RUN_DIR (+ABLATION_RUN_DIR, UNSEEN_RUN_DIR) vs baselines -> ml/eval/results
	$(UV) run python -m ml.eval.evaluate --run-dir $(RUN_DIR) \
		$(if $(ABLATION_RUN_DIR),--ablation-run-dir $(ABLATION_RUN_DIR)) \
		$(if $(UNSEEN_RUN_DIR),--unseen-run-dir $(UNSEEN_RUN_DIR)) $(EVAL_ARGS)
export-model: ## ONNX fp32 + int8 export with parity checks: RUN_DIR -> ARTIFACT_DIR (EXPORT_ARGS=--refresh-manifest)
	$(ML) -m ml.export.export --run-dir $(RUN_DIR) --params ml/eval/results/inference_params.json \
		--model-eval ml/eval/results/model_eval.json --out $(ARTIFACT_DIR) $(EXPORT_ARGS)
bench-model: ## CPU latency per (rule, document) pair: torch fp32 / ONNX fp32 / ONNX int8 on this host
	$(ML) -m ml.export.bench --artifact-dir $(ARTIFACT_DIR) --run-dir $(RUN_DIR) --label $(HOST_LABEL) \
		--out ml/eval/results/latency_$(HOST_LABEL).json
eval-onnx: ## Accuracy of the exported ONNX variants on full TEST (quantization effect)
	$(ML) -m ml.export.eval_onnx --artifact-dir $(ARTIFACT_DIR) --run-dir $(RUN_DIR) --split test \
		--out ml/eval/results/quantization_eval.json
ml-test: ## ML unit tests (no tool overlay needed)
	$(UV) run python -m pytest -q ml/tests
ml-smoke: ## Bounded CPU smoke: data subset -> train -> baselines -> eval -> ONNX -> service (~2-3 min)
	$(ML) -m pytest -q -p no:cacheprovider ml/smoke/tests
install-model: ## Verify ARTIFACT_DIR and make it the model service's current rule-check artifact
	$(UV) run python -m ml.colab.bundle install-artifact --artifact-dir $(ARTIFACT_DIR)
import-colab-bundle: ## Verify + install a Colab results bundle: BUNDLE=<path to .tar>
	$(UV) run python -m ml.colab.bundle import --bundle $(BUNDLE)
eval-all: ## [Phase 6] Full eval suite (not implemented)
	$(call NOT_IMPLEMENTED,Phase 6)
