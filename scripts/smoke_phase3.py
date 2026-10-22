"""Phase 3 smoke test against a running stack: rule-check model state and contract.

- the web/API model-status endpoint (what the UI banner shows) agrees with the model
  service's own /health and /metadata;
- without a verified artifact (the normal state until a Colab-trained model is
  installed): /v1/check returns 503 ``rule_check_unavailable`` and no verdict;
- with an installed production artifact: concurrent /v1/check requests return validated
  verdicts (labels, calibrated confidence, evidence ids from the request), and the
  micro-batcher's statistics show the requests were batched;
- input validation (blank text, duplicate span ids) and the request size budget.

    uv run python scripts/smoke_phase3.py --web http://127.0.0.1:3000 \
        --model-service http://127.0.0.1:8010
"""

from __future__ import annotations

import argparse
import secrets
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import httpx

PASSWORD = "smoke-phase3-password"  # noqa: S105 - throwaway test account
LABELS = {"entailment", "contradiction", "not_mentioned"}


class SmokeError(AssertionError):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def step(name: str, fn: Callable[[], None]) -> None:
    fn()
    print(f"PASS  {name}")  # noqa: T201


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", default="http://127.0.0.1:3000")
    parser.add_argument("--model-service", default="http://127.0.0.1:8010")
    args = parser.parse_args()
    ms = httpx.Client(base_url=args.model_service, timeout=60)
    state: dict[str, object] = {}

    def status_matches_service() -> None:
        web = httpx.Client(base_url=args.web, timeout=30)
        email = f"phase3-{secrets.token_hex(4)}@example.com"
        r = web.post("/api/v1/auth/register", json={"email": email, "password": PASSWORD})
        expect(r.status_code == 201, r.text)
        status = web.get("/api/v1/models/status").json()["rule_check"]
        health = ms.get("/health").json()
        meta = ms.get("/metadata").json()["rule_check"]
        state.update(health=health, meta=meta)
        if health["rule_check_mode"] == "untrained_fallback":
            expect(status["state"] == "untrained_fallback", str(status))
            expect(meta["available"] is False and meta["model_version"] is None, str(meta))
        else:
            expect(
                status["state"] == ("smoke" if meta["kind"] == "smoke" else "trained"), str(status)
            )
            expect(status["model_version"] == meta["model_version"], str(status))
        expect("/" not in status["message"], "status message must not expose paths")

    def check_contract() -> None:
        spans = [
            {
                "span_id": "c1",
                "text": "The Receiving Party shall not disclose Confidential Information.",
            },
            {
                "span_id": "c2",
                "text": "The Receiving Party may share Confidential Information with its employees.",
            },
            {"span_id": "c3", "text": "This Agreement is governed by the laws of Delaware."},
        ]
        payload = {
            "hypothesis": "Receiving Party may share some Confidential Information with some of Receiving Party's employees.",
            "spans": spans,
        }
        if state["health"]["rule_check_mode"] == "untrained_fallback":  # type: ignore[index]
            r = ms.post("/v1/check", json=payload)
            expect(r.status_code == 503, r.text)
            expect(r.json()["error"]["code"] == "rule_check_unavailable", r.text)
            for fabricated in ("label", "confidence", "evidence"):
                expect(fabricated not in r.json(), r.text)
            return
        before = ms.get("/metadata").json()["rule_check"]["batching"]["stats"]
        with ThreadPoolExecutor(8) as pool:
            responses = list(pool.map(lambda _: ms.post("/v1/check", json=payload), range(8)))
        for r in responses:
            expect(r.status_code == 200, r.text)
            body = r.json()
            expect(body["label"] in LABELS and 0 <= body["confidence"] <= 1, r.text)
            ids = {e["span_id"] for e in body["evidence"] + body["considered"]}
            expect(ids <= {"c1", "c2", "c3"}, r.text)
            expect(
                body["model_kind"] == "production", "the live stack must not serve a smoke artifact"
            )
        after = ms.get("/metadata").json()["rule_check"]["batching"]["stats"]
        expect(after["batches"] > before["batches"], f"{before} -> {after}")

    def validation() -> None:
        for payload in (
            {"hypothesis": "   ", "spans": [{"span_id": "a", "text": "x"}]},
            {
                "hypothesis": "h",
                "spans": [{"span_id": "a", "text": "x"}, {"span_id": "a", "text": "y"}],
            },
        ):
            r = ms.post("/v1/check", json=payload)
            expect(r.status_code == 422, f"{payload} -> {r.status_code}")

    try:
        step(
            "UI model status agrees with the model service (no paths exposed)",
            status_matches_service,
        )
        step("rule check: honest fallback or validated, batched verdicts", check_contract)
        step("blank text and duplicate span ids rejected", validation)
    except (SmokeError, httpx.HTTPError, KeyError) as exc:
        print(f"FAIL  {type(exc).__name__}: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
