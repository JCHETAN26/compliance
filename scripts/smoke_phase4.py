"""Phase 4 end-to-end smoke test against a running stack (through the web origin).

Uploads the synthetic ten-vendor flagship corpus (vendor register, text and Word NDAs,
an image-only scanned NDA, a prompt-injection NDA; one listed contract deliberately
missing) into a fresh workspace, starts a Vendor Confidentiality Review, follows the
run to the end and checks the Phase 4 acceptance invariants:

- every finding is verified with exact, currently valid citations, or is needs_review;
- nothing unverified is compliant; the report states only verified/human findings;
- the missing contract is flagged; the scanned PDF was OCR'd and checked;
- the event cursor is strictly increasing, the SSE stream replays after Last-Event-ID;
- citations resolve (members only) and their document file is served.

    uv run python scripts/smoke_phase4.py --web http://127.0.0.1:3000 [--timeout 1800]

With the default mock LLM this makes no external requests. With a real provider
configured on the server it spends real money: do not run it against such a stack
without approval.
"""

from __future__ import annotations

import argparse
import secrets
import sys
import time
from collections.abc import Callable
from typing import Any

import httpx

from compliance_seed.synthetic import build

PASSWORD = "smoke-phase4-password"  # noqa: S105 - throwaway test account
TERMINAL = {"completed", "failed", "cancelled", "paused", "awaiting_review"}


class SmokeError(AssertionError):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def step(name: str, fn: Callable[[], None]) -> None:
    fn()
    print(f"PASS  {name}")  # noqa: T201


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--web", default="http://127.0.0.1:3000")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds to wait for the run")
    parser.add_argument("--budget-usd", type=float, default=None, help="hard USD cap for the run")
    parser.add_argument(
        "--budget-tokens", type=int, default=None, help="hard token cap for the run"
    )
    parser.add_argument("--max-concurrency", type=int, default=None)
    args = parser.parse_args()
    client = httpx.Client(base_url=args.web, timeout=120)
    state: dict[str, Any] = {}

    def signup() -> None:
        email = f"smoke4-{secrets.token_hex(4)}@example.com"
        r = client.post("/api/v1/auth/register", json={"email": email, "password": PASSWORD})
        expect(r.status_code == 201, f"register: {r.status_code} {r.text}")
        client.headers["X-CSRF-Token"] = r.json()["csrf_token"]
        ws = client.post("/api/v1/workspaces", json={"name": "Phase 4 smoke"})
        expect(ws.status_code == 201, ws.text)
        state["ws"] = ws.json()["id"]

    def upload() -> None:
        files = build()
        r = client.post(
            f"/api/v1/workspaces/{state['ws']}/documents",
            files=[
                ("files", (name, data, "application/octet-stream")) for name, data in files.items()
            ],
        )
        expect(r.status_code == 200, f"upload: {r.status_code} {r.text}")
        outcomes = {x["outcome"] for x in r.json()["results"]}
        expect(outcomes == {"created"}, f"upload outcomes {outcomes}")
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            docs = client.get(f"/api/v1/workspaces/{state['ws']}/documents").json()
            if all(d["status"] in ("ready", "failed") for d in docs) and len(docs) == len(files):
                expect(all(d["status"] == "ready" for d in docs), f"ingestion failed: {docs}")
                scanned = next(d for d in docs if "scanned" in d["filename"])
                expect(scanned["ocr_page_count"] >= 1, "the scanned NDA was not OCR'd")
                state["docs"] = {d["filename"]: d["id"] for d in docs}
                return
            time.sleep(3)
        raise SmokeError("documents did not finish processing")

    def run_review() -> None:
        packs = client.get("/api/v1/rule-packs", params={"workspace_id": state["ws"]}).json()
        pack = next(p for p in packs if p["key"] == "vendor_confidentiality_review")
        body: dict[str, Any] = {"rule_pack_id": pack["id"]}
        for key, value in (
            ("budget_usd", args.budget_usd),
            ("budget_tokens", args.budget_tokens),
            ("max_concurrency", args.max_concurrency),
        ):
            if value is not None:
                body[key] = value
        r = client.post(f"/api/v1/workspaces/{state['ws']}/reviews", json=body)
        expect(r.status_code == 202, f"start: {r.status_code} {r.text}")
        run = r.json()
        state["run"] = run["id"]
        print(f"      run {run['id']} provider={run['llm_provider']} mock={run['llm_is_mock']}")  # noqa: T201
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            run = client.get(f"/api/v1/reviews/{state['run']}").json()
            if run["status"] in TERMINAL:
                break
            time.sleep(3)
        expect(
            run["status"] == "completed", f"run ended {run['status']}: {run.get('error_message')}"
        )
        state["run_read"] = run
        budget = run["budget"]
        print(  # noqa: T201
            f"      usage: {budget['tokens_used']} tokens, ${budget['usd_used']:.6f} "
            f"({budget['usage_source']}); {budget['calls']} calls, {budget['calls_unknown']} unknown"
        )

    def findings() -> None:
        body = client.get(f"/api/v1/reviews/{state['run']}/findings", params={"limit": 1000}).json()
        items = body["items"]
        tasks = client.get(f"/api/v1/reviews/{state['run']}/tasks").json()["tasks"]
        expect(len(items) == len(tasks) and items, f"{len(items)} findings for {len(tasks)} tasks")
        for f in items:
            if f["status"] != "needs_review":
                expect(f["verified"] or f["decided_by"] == "human", f"unverified {f['task_key']}")
                expect(bool(f["citations"]), f"no citations for {f['task_key']}")
                expect(
                    all(c["exact_match"] and c["current"] == "valid" for c in f["citations"]),
                    f"invalid citation in {f['task_key']}",
                )
            if f["compliance"] == "compliant":
                expect(f["verified"], f"compliant but unverified {f['task_key']}")
        missing = [
            f
            for f in items
            if f["rule_key"] == "high_access_contract_on_file"
            and f["subject_label"] == "Umbrella GmbH"
        ]
        expect(
            len(missing) == 1 and missing[0]["status"] == "violated", "missing contract not flagged"
        )
        scanned = state["docs"]["initech_nda_scanned.pdf"]
        expect(any(f["document_id"] == scanned for f in items), "scanned NDA was not checked")
        state["citation"] = next(c for f in items for c in f["citations"] if c["document_id"])
        counts: dict[str, int] = {}
        for f in items:
            counts[f["status"]] = counts.get(f["status"], 0) + 1
        print(f"      {len(items)} findings: {counts}")  # noqa: T201

    def report() -> None:
        body = client.get(f"/api/v1/reviews/{state['run']}/report").json()
        by_id = {f["id"]: f for f in body["findings"]}
        for statement in body["summary"]:
            if "need human review" in statement["text"]:
                continue
            expect(
                all(
                    by_id[i]["verified"] or by_id[i]["decided_by"] == "human"
                    for i in statement["finding_ids"]
                ),
                f"unverified statement: {statement['text']}",
            )
        md = client.get(f"/api/v1/reviews/{state['run']}/report", params={"format": "md"})
        expect(md.status_code == 200 and "Umbrella GmbH" in md.text, "markdown export")

    def citations() -> None:
        cid = state["citation"]["id"]
        detail = client.get(f"/api/v1/citations/{cid}").json()
        expect(detail["current"] == "valid", str(detail))
        file = client.get(f"/api/v1/documents/{detail['document_id']}/file")
        expect(file.status_code == 200, "cited document file")
        other = httpx.Client(base_url=args.web, timeout=60)
        r = other.post(
            "/api/v1/auth/register",
            json={
                "email": f"smoke4-other-{secrets.token_hex(4)}@example.com",
                "password": PASSWORD,
            },
        )
        expect(r.status_code == 201, r.text)
        expect(other.get(f"/api/v1/citations/{cid}").status_code == 404, "citation leaked")
        expect(other.get(f"/api/v1/reviews/{state['run']}").status_code == 404, "run leaked")

    def events() -> None:
        history = client.get(
            f"/api/v1/reviews/{state['run']}/events/history", params={"limit": 1000}
        ).json()["items"]
        seqs = [e["seq"] for e in history]
        expect(seqs == sorted(set(seqs)) and len(seqs) > 10, "event cursor")
        agents = {e["agent"] for e in history}
        for agent in (
            "planner",
            "extraction",
            "retrieval",
            "rule_check",
            "cross_document",
            "citation_verifier",
            "report_writer",
        ):
            expect(agent in agents, f"no events from {agent}")
        cursor = seqs[len(seqs) // 2]
        with client.stream(
            "GET", f"/api/v1/reviews/{state['run']}/events", headers={"Last-Event-ID": str(cursor)}
        ) as stream:
            text = "".join(stream.iter_text())
        ids = [int(line[4:]) for line in text.splitlines() if line.startswith("id: ")]
        expect(
            bool(ids) and ids[0] > cursor and ids == sorted(ids), "SSE replay after Last-Event-ID"
        )
        expect("event: end" in text, "SSE stream did not end for a finished run")

    try:
        step("sign up and create a workspace", signup)
        step("upload the synthetic flagship corpus and wait for ingestion (OCR included)", upload)
        step("run the Vendor Confidentiality Review to completion", run_review)
        step("findings: verified exact citations or needs_review", findings)
        step("report states only verified or human findings; Markdown export", report)
        step("citation detail, document file, cross-user isolation", citations)
        step("agent events: cursor, all seven agents, SSE replay", events)
    except (SmokeError, httpx.HTTPError, StopIteration, KeyError) as exc:
        print(f"FAIL  {type(exc).__name__}: {exc}")  # noqa: T201
        return 1
    print("Phase 4 smoke: all checks passed")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
