# model-service

**Phase 0 status: not implemented.** No rule-check model has been trained, loaded or
served. That work (training, ONNX int8 export, dynamic batching, the "untrained
fallback" mode) is Phase 3 of `docs/BUILD_HANDOFF.md`.

What exists now, so the compose topology and the API contract are in place:

| Endpoint | Behaviour |
|---|---|
| `GET /health` | `{"status": "ok", "mode": "not_implemented", "model_loaded": false}` |
| `GET /metadata` | `implemented: false`, `planned_phase: 3`, no model version, no metrics |
| `POST /v1/check` | Validates `{hypothesis, spans[]}`, then returns **501** `not_implemented`. It never returns a verdict. |

```bash
uv run uvicorn --factory compliance_model_service.main:create_app --port 8010
```
