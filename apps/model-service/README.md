# model-service

Local CPU inference for the other services.

| Endpoint | Behaviour |
|---|---|
| `POST /v1/embed` | Query/passage embeddings with the pinned ingestion encoder (`BAAI/bge-small-en-v1.5`), Phase 1–2 |
| `POST /v1/rerank` | Optional cross-encoder, only with `RERANK_ENABLED=true` (otherwise 409) |
| `POST /v1/check` | Rule-check model (Phase 3): `{hypothesis, spans:[{span_id, text}]}` -> `{label, status, confidence, probabilities, evidence:[{span_id, score}], considered, escalate, truncated_inputs, model_version, model_kind, variant}` |
| `GET /health` | `embedding=ready`, `rerank=ready|disabled`, `rule_check=ready|unavailable`, `rule_check_mode=model|untrained_fallback` |
| `GET /metadata` | Model ids/revisions/SHA-256; rule-check artifact version, kind, served parameters, metrics summary, provenance and live micro-batching statistics |

## Rule check

- **Artifact.** A directory with `manifest.json`, `tokenizer.json` and `model.int8.onnx` (optionally
  `model.fp32.onnx`), default `<ARTIFACT_CACHE_DIR>/models/rule-check/current` (in Docker
  `/cache/models/rule-check/current`; `RULE_CHECK_ARTIFACT_DIR` overrides). The manifest is
  validated (label order, finite parameters, safe file names, SHA-256 and size of every file)
  before ONNX Runtime loads anything. It is produced by `make export-model` and installed by
  `make install-model` or `make import-colab-bundle` (see `ml/README.md`).
- **Untrained fallback.** Without a verified artifact the service still starts (embeddings keep
  working), `/health` and `/metadata` say `untrained_fallback` with the reason, `/v1/check` answers
  **503 `rule_check_unavailable`** and never a verdict, and the web app shows a banner (via
  `GET /api/v1/models/status`, which exposes no paths).
- **Smoke artifacts** (CI: tiny random backbone) are refused unless `RULE_CHECK_ALLOW_SMOKE=true`
  (tests only); responses then carry `model_kind: "smoke"` and the banner says so.
- **Inference** (shared with `ml/eval`, so served == evaluated): every span is scored by the evidence
  head; the top-k (`inference.top_k`, DEV-selected) in request order form the label input; label
  logits / DEV-fitted temperature -> softmax. `evidence` lists spans whose score reaches the
  DEV-selected threshold; `escalate` is `confidence < tau` when a DEV-selected tau is in the artifact.
- **Dynamic micro-batching.** Rows (token-id sequences) from concurrent requests are collected until
  `RULE_CHECK_MAX_BATCH_ROWS` rows or `RULE_CHECK_MAX_WAIT_MS` after the first row, run once (sorted by
  length, sub-batched by `RULE_CHECK_MAX_TOKENS_PER_FORWARD`), and scattered back. Outputs are checked
  for shape and finiteness before anyone receives them (502 otherwise).
- **Bounded admission (503 `overloaded`).** At most `RULE_CHECK_MAX_PENDING_REQUESTS` requests with
  tokenization queued or running (a slot is freed only when that work really ends) and at most
  `RULE_CHECK_MAX_QUEUE_ROWS` rows retained by the batcher (cancelled requests keep their rows counted
  until the batcher releases them). Requests over `RULE_CHECK_MAX_REQUEST_CHARS` total characters get
  413; blank hypothesis/span text and duplicate span ids get 422.
- **Timeouts.** The service has no server-side request deadline. Callers set one
  (`MODEL_SERVICE_TIMEOUT_SECONDS` for the shared client); a disconnected client cancels its request,
  and the bounds above still hold for work that cannot be interrupted.
- **int8 note.** Dynamic int8 quantization computes activation scales per input tensor, so a row's
  int8 output can differ slightly with the rows batched alongside it (measured at export as
  `batch_sensitivity`; fp32 does not have this property).

```bash
uv run uvicorn --factory compliance_model_service.main:create_app --port 8010
```
