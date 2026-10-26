"""Phase 4 API: rule packs, review settings, review runs, findings, overrides, citations,
reports and events (paginated JSON and the real SSE route).

Authorisation is checked on every new route: non-members get 404 (no existence
disclosure, no cross-workspace citation leaks), viewers can read but not start, control,
override or import, and cookie-authenticated writes need the CSRF header."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import httpx
import pytest
import yaml
from sqlalchemy import select, update

from compliance_core.db.models import Citation, Finding, ReviewRun, RulePack
from compliance_core.services.review_events import append_event
from compliance_core.services.rule_packs import sync_builtin_packs

pytestmark = pytest.mark.integration
CSV = b"Vendor name,Data access level,Contract file name,Renewal date\nAcme,High,acme.pdf,2027-01-01\n"


async def _setup(clients: Any, api_app: Any, ingest_ready: Any) -> dict[str, Any]:
    owner = await clients.signed_up("owner@example.com")
    me = (await owner.get("/api/v1/auth/me")).json()
    ws = (await owner.post("/api/v1/workspaces", json={"name": "Reviews"})).json()["id"]
    async with api_app.state.ctx.session_factory() as session:
        await sync_builtin_packs(session)
    doc = await ingest_ready(uuid.UUID(ws), uuid.UUID(me["id"]), "register.csv", CSV, "csv")
    packs = (await owner.get("/api/v1/rule-packs", params={"workspace_id": ws})).json()
    vendor = next(p for p in packs if p["key"] == "vendor_confidentiality_review")
    return {"owner": owner, "ws": ws, "doc": str(doc), "pack": vendor["id"], "user": me["id"]}


async def _member(
    owner: httpx.AsyncClient, ws: str, clients: Any, email: str, role: str
) -> httpx.AsyncClient:
    client: httpx.AsyncClient = await clients.signed_up(email)
    response = await owner.post(
        f"/api/v1/workspaces/{ws}/members", json={"email": email, "role": role}
    )
    assert response.status_code == 201, response.text
    return client


async def _start(client: httpx.AsyncClient, ws: str, pack: str, **extra: Any) -> httpx.Response:
    return await client.post(
        f"/api/v1/workspaces/{ws}/reviews", json={"rule_pack_id": pack, **extra}
    )


async def _finding_with_citation(api_app: Any, run_id: str, doc: str) -> tuple[str, str]:
    """Insert a minimal verified finding + citation for authorisation tests."""
    async with api_app.state.ctx.session_factory() as session, session.begin():
        finding = Finding(
            run_id=uuid.UUID(run_id),
            task_key="pd|x|y",
            kind="per_document",
            rule_key="third_party_sharing_restricted",
            rule_title="Third-party sharing restricted",
            severity="high",
            document_id=uuid.UUID(doc),
            subject_key="doc:y",
            subject_label="register.csv",
            status="needs_review",
            machine_status="needs_review",
            compliance="needs_review",
            decided_by="system",
            rationale="test",
        )
        session.add(finding)
        await session.flush()
        citation = Citation(
            finding_id=finding.id,
            position=0,
            role="evidence",
            document_id=uuid.UUID(doc),
            document_filename="register.csv",
            quote="Vendor name: Acme",
            char_start=0,
            char_end=17,
            bbox_precision="none",
            bbox_json=[],
        )
        session.add(citation)
        await session.flush()
        return str(finding.id), str(citation.id)


async def test_start_review_queues_and_enqueues(
    clients: Any, api_app: Any, ingest_ready: Any
) -> None:
    s = await _setup(clients, api_app, ingest_ready)
    response = await _start(s["owner"], s["ws"], s["pack"])
    assert response.status_code == 202, response.text
    run = response.json()
    assert run["status"] == "queued"
    assert run["llm_provider"] == "mock"
    assert run["llm_is_mock"] is True
    assert run["document_count"] == 1
    assert run["budget"]["usd_used"] == 0
    listed = (await s["owner"].get(f"/api/v1/workspaces/{s['ws']}/reviews")).json()
    assert [r["id"] for r in listed] == [run["id"]]
    redis = api_app.state.ctx.redis
    keys = [k.decode() async for k in redis.scan_iter("arq:job:review:*")]
    assert any(uuid.UUID(run["id"]).hex in k for k in keys)


async def test_start_review_validation(clients: Any, api_app: Any, ingest_ready: Any) -> None:
    s = await _setup(clients, api_app, ingest_ready)
    owner, ws, pack = s["owner"], s["ws"], s["pack"]
    assert (await _start(owner, ws, str(uuid.uuid4()))).status_code == 404
    assert (await _start(owner, ws, pack, document_ids=[str(uuid.uuid4())])).status_code == 404
    unconfigured = await _start(owner, ws, pack, llm_provider="anthropic")
    assert unconfigured.status_code == 409
    assert "not configured" in unconfigured.json()["error"]["message"]
    other = await clients.signed_up("other@example.com")
    other_ws = (await other.post("/api/v1/workspaces", json={"name": "Other"})).json()["id"]
    imported = await other.post(
        "/api/v1/rule-packs/import",
        json={
            "workspace_id": other_ws,
            "yaml": (await owner.get(f"/api/v1/rule-packs/{pack}/export")).text.replace(
                "key: vendor_confidentiality_review", "key: copied_pack", 1
            ),
        },
    )
    assert imported.status_code == 201, imported.text
    # Another workspace's pack and documents are "not found".
    assert (await _start(owner, ws, imported.json()["id"])).status_code == 404
    assert (await _start(other, other_ws, pack, document_ids=[s["doc"]])).status_code == 404


async def test_roles_csrf_and_non_members(clients: Any, api_app: Any, ingest_ready: Any) -> None:
    s = await _setup(clients, api_app, ingest_ready)
    owner, ws, pack = s["owner"], s["ws"], s["pack"]
    run = (await _start(owner, ws, pack)).json()
    rid = run["id"]
    finding_id, citation_id = await _finding_with_citation(api_app, rid, s["doc"])
    async with api_app.state.ctx.session_factory() as session, session.begin():
        await session.execute(
            update(ReviewRun).where(ReviewRun.id == uuid.UUID(rid)).values(status="completed")
        )

    viewer = await _member(owner, ws, clients, "viewer@example.com", "viewer")
    for path in (
        f"/api/v1/reviews/{rid}",
        f"/api/v1/reviews/{rid}/tasks",
        f"/api/v1/reviews/{rid}/findings",
        f"/api/v1/reviews/{rid}/report",
        f"/api/v1/reviews/{rid}/events/history",
        f"/api/v1/findings/{finding_id}",
        f"/api/v1/citations/{citation_id}",
    ):
        assert (await viewer.get(path)).status_code == 200, path
    assert (await _start(viewer, ws, pack)).status_code == 403
    assert (
        await viewer.post(f"/api/v1/reviews/{rid}/control", json={"action": "pause"})
    ).status_code == 403
    assert (
        await viewer.post(
            f"/api/v1/findings/{finding_id}/override", json={"status": "violated", "comment": "no"}
        )
    ).status_code == 403
    assert (
        await viewer.post("/api/v1/rule-packs/import", json={"workspace_id": ws, "yaml": "x: 1"})
    ).status_code == 403
    assert (
        await viewer.put(
            f"/api/v1/workspaces/{ws}/review-settings",
            json={
                "llm_provider": "mock",
                "budget_tokens": 5000,
                "budget_usd": 1,
                "pause_on_needs_review": False,
                "max_concurrency": 2,
            },
        )
    ).status_code == 403

    outsider = await clients.signed_up("outsider@example.com")
    for method, path, body in (
        ("GET", f"/api/v1/reviews/{rid}", None),
        ("GET", f"/api/v1/reviews/{rid}/tasks", None),
        ("GET", f"/api/v1/reviews/{rid}/findings", None),
        ("GET", f"/api/v1/reviews/{rid}/report", None),
        ("GET", f"/api/v1/reviews/{rid}/events", None),
        ("GET", f"/api/v1/reviews/{rid}/events/history", None),
        ("GET", f"/api/v1/findings/{finding_id}", None),
        ("GET", f"/api/v1/citations/{citation_id}", None),
        ("POST", f"/api/v1/findings/{finding_id}/override", {"status": "violated", "comment": "x"}),
        ("POST", f"/api/v1/reviews/{rid}/control", {"action": "cancel"}),
        ("POST", f"/api/v1/reviews/{rid}/resume", None),
        ("GET", f"/api/v1/workspaces/{ws}/reviews", None),
        ("GET", f"/api/v1/workspaces/{ws}/review-settings", None),
        ("GET", f"/api/v1/rule-packs?workspace_id={ws}", None),
    ):
        response = await outsider.request(method, path, json=body)
        assert response.status_code == 404, (method, path, response.status_code)

    no_csrf = clients.new()
    no_csrf.cookies = owner.cookies
    csrf_cases: tuple[tuple[str, str, dict[str, object]], ...] = (
        ("POST", f"/api/v1/workspaces/{ws}/reviews", {"rule_pack_id": pack}),
        ("POST", f"/api/v1/reviews/{rid}/control", {"action": "cancel"}),
        ("POST", f"/api/v1/findings/{finding_id}/override", {"status": "violated", "comment": "x"}),
        ("POST", "/api/v1/rule-packs/import", {"workspace_id": ws, "yaml": "x: 1"}),
        (
            "PUT",
            f"/api/v1/workspaces/{ws}/review-settings",
            {
                "llm_provider": "mock",
                "budget_tokens": 5000,
                "budget_usd": 1,
                "pause_on_needs_review": False,
                "max_concurrency": 2,
            },
        ),
    )
    for method, path, payload in csrf_cases:
        response = await no_csrf.request(method, path, json=payload)
        assert response.status_code == 403, (method, path)
        assert "CSRF" in response.json()["error"]["message"]
    anonymous = clients.new()
    assert (await anonymous.get(f"/api/v1/reviews/{rid}/events")).status_code == 401


async def test_citations_never_leak_across_workspaces(
    clients: Any, api_app: Any, ingest_ready: Any
) -> None:
    s = await _setup(clients, api_app, ingest_ready)
    run = (await _start(s["owner"], s["ws"], s["pack"])).json()
    other = await clients.signed_up("other@example.com")
    other_ws = (await other.post("/api/v1/workspaces", json={"name": "Other"})).json()["id"]
    other_me = (await other.get("/api/v1/auth/me")).json()
    other_doc = await ingest_ready(
        uuid.UUID(other_ws),
        uuid.UUID(other_me["id"]),
        "theirs.csv",
        CSV + b"Beta,Low,b.pdf,2027-01-01\n",
        "csv",
    )
    # A citation in the owner's run that (wrongly) points at the other workspace's document
    # must not be served to anyone through the owner's run.
    _, citation_id = await _finding_with_citation(api_app, run["id"], str(other_doc))
    assert (await s["owner"].get(f"/api/v1/citations/{citation_id}")).status_code == 404
    assert (await other.get(f"/api/v1/citations/{citation_id}")).status_code == 404


async def test_control_transitions(clients: Any, api_app: Any, ingest_ready: Any) -> None:
    s = await _setup(clients, api_app, ingest_ready)
    owner = s["owner"]
    rid = (await _start(owner, s["ws"], s["pack"])).json()["id"]
    paused = await owner.post(f"/api/v1/reviews/{rid}/control", json={"action": "pause"})
    assert paused.json()["status"] == "paused"
    resumed = await owner.post(f"/api/v1/reviews/{rid}/resume")
    assert resumed.json()["status"] == "queued"
    async with api_app.state.ctx.session_factory() as session:
        generation = await session.scalar(
            select(ReviewRun.job_generation).where(ReviewRun.id == uuid.UUID(rid))
        )
    assert generation == 2
    async with api_app.state.ctx.session_factory() as session, session.begin():
        await session.execute(
            update(ReviewRun).where(ReviewRun.id == uuid.UUID(rid)).values(status="running")
        )
    running_cancel = await owner.post(f"/api/v1/reviews/{rid}/control", json={"action": "cancel"})
    assert running_cancel.json()["status"] == "running"
    assert running_cancel.json()["control"] == "cancel"
    async with api_app.state.ctx.session_factory() as session, session.begin():
        await session.execute(
            update(ReviewRun)
            .where(ReviewRun.id == uuid.UUID(rid))
            .values(status="completed", control=None)
        )
    assert (
        await owner.post(f"/api/v1/reviews/{rid}/control", json={"action": "cancel"})
    ).status_code == 409
    assert (await owner.post(f"/api/v1/reviews/{rid}/resume")).status_code == 409


async def test_events_pagination_and_real_sse_stream(
    clients: Any, api_app: Any, ingest_ready: Any
) -> None:
    s = await _setup(clients, api_app, ingest_ready)
    owner = s["owner"]
    rid = (await _start(owner, s["ws"], s["pack"])).json()["id"]
    sessions = api_app.state.ctx.session_factory
    async with sessions() as session, session.begin():
        for i in range(5):
            await append_event(session, uuid.UUID(rid), "planner", "result", {"i": i})
    page = (
        await owner.get(f"/api/v1/reviews/{rid}/events/history", params={"after": 0, "limit": 3})
    ).json()
    assert [e["seq"] for e in page["items"]] == [1, 2, 3]
    assert page["has_more"]
    page2 = (
        await owner.get(
            f"/api/v1/reviews/{rid}/events/history", params={"after": page["next_after"]}
        )
    ).json()
    assert [e["seq"] for e in page2["items"]] == [4, 5, 6]

    async def finish_later() -> None:
        await asyncio.sleep(1.5)
        async with sessions() as session, session.begin():
            await append_event(session, uuid.UUID(rid), "report_writer", "result", {"live": True})
            await session.execute(
                update(ReviewRun).where(ReviewRun.id == uuid.UUID(rid)).values(status="completed")
            )

    task = asyncio.create_task(finish_later())
    response = await owner.get(f"/api/v1/reviews/{rid}/events", headers={"Last-Event-ID": "4"})
    await task
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "no-transform" in response.headers["cache-control"]
    ids = [int(line[4:]) for line in response.text.splitlines() if line.startswith("id: ")]
    assert ids == [5, 6, 7]  # replay strictly after the cursor, then the live event
    assert '"live":true' in response.text
    assert response.text.rstrip().endswith('"status":"completed"}')


async def test_rule_pack_api(clients: Any, api_app: Any, ingest_ready: Any) -> None:
    s = await _setup(clients, api_app, ingest_ready)
    owner, ws, pack = s["owner"], s["ws"], s["pack"]
    detail = (await owner.get(f"/api/v1/rule-packs/{pack}")).json()
    assert detail["builtin"] is True
    assert detail["editable"] is False
    assert detail["model_supported_count"] == 3
    assert detail["register"]["columns"]["vendor"]
    exported = await owner.get(f"/api/v1/rule-packs/{pack}/export")
    assert exported.headers["content-type"].startswith("application/yaml")
    doc = yaml.safe_load(exported.text)
    assert (await owner.put(f"/api/v1/rule-packs/{pack}", json=doc)).status_code == 403
    doc["key"] = "my_vendor_pack"
    custom = next(r for r in doc["rules"] if r["key"] == "third_party_sharing_restricted")
    custom["model_supported"] = True
    rejected = await owner.post(
        "/api/v1/rule-packs/import", json={"workspace_id": ws, "yaml": yaml.safe_dump(doc)}
    )
    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "invalid_rule_pack"
    assert any("model_supported" in d["msg"] for d in rejected.json()["error"]["details"])
    custom["model_supported"] = False
    created = await owner.post(
        "/api/v1/rule-packs/import", json={"workspace_id": ws, "yaml": yaml.safe_dump(doc)}
    )
    assert created.status_code == 201, created.text
    pid = created.json()["id"]
    assert created.json()["editable"] is True
    assert (
        await owner.post(
            "/api/v1/rule-packs/import", json={"workspace_id": ws, "yaml": yaml.safe_dump(doc)}
        )
    ).status_code == 409
    doc["version"] = "1.1.0"
    doc["rules"][0]["severity"] = "low"
    updated = await owner.put(f"/api/v1/rule-packs/{pid}", json=doc)
    assert updated.status_code == 200, updated.text
    assert updated.json()["version"] == "1.1.0"
    assert updated.json()["rules"][0]["severity"] == "low"
    doc["key"] = "renamed"
    assert (await owner.put(f"/api/v1/rule-packs/{pid}", json=doc)).status_code == 409
    bad = await owner.post(
        "/api/v1/rule-packs/import", json={"workspace_id": ws, "yaml": "rules: [unclosed"}
    )
    assert bad.status_code == 422
    async with api_app.state.ctx.session_factory() as session:
        assert (
            await session.scalar(select(RulePack.source).where(RulePack.id == uuid.UUID(pid)))
            == "import"
        )


async def test_settings_api(clients: Any, api_app: Any, ingest_ready: Any) -> None:
    s = await _setup(clients, api_app, ingest_ready)
    owner, ws = s["owner"], s["ws"]
    llm = (await owner.get("/api/v1/settings/llm")).json()
    assert llm["deployment_provider"] == "mock"
    assert llm["mock_only"] is True
    assert [p["provider"] for p in llm["providers"]] == ["mock"]
    assert "key" not in str(llm).lower().replace("credential", "")
    current = (await owner.get(f"/api/v1/workspaces/{ws}/review-settings")).json()
    assert current["is_default"] is True
    body = {
        "llm_provider": "mock",
        "llm_model": "",
        "tau": 0.5,
        "budget_tokens": 50_000,
        "budget_usd": 0.25,
        "pause_on_needs_review": True,
        "max_concurrency": 2,
    }
    saved = await owner.put(f"/api/v1/workspaces/{ws}/review-settings", json=body)
    assert saved.status_code == 200, saved.text
    assert saved.json()["is_default"] is False
    assert saved.json()["budget_usd"] == 0.25
    assert (
        await owner.put(
            f"/api/v1/workspaces/{ws}/review-settings", json={**body, "llm_provider": "openai"}
        )
    ).status_code == 409
    run = (await _start(owner, ws, s["pack"])).json()
    assert run["budget"]["tokens_cap"] == 50_000
    assert run["pause_on_needs_review"] is True
    assert run["tau"] == 0.5
