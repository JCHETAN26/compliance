"""Phase 0 end-to-end smoke test against a running `docker compose` stack.

Drives the same HTTP paths the browser uses: everything auth/workspace related goes
through the web origin (Next.js proxy -> FastAPI), so cookies, CSRF and the proxy are
exercised together. Exits non-zero on the first failed check.

    uv run python scripts/smoke_phase0.py --web http://127.0.0.1:3000
"""

from __future__ import annotations

import argparse
import secrets
import sys
from collections.abc import Callable

import httpx

PASSWORD = "smoke-test-password"  # noqa: S105 - throwaway test account


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
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--model-service", default="http://127.0.0.1:8010")
    parser.add_argument("--mcp", default="http://127.0.0.1:8020")
    args = parser.parse_args()

    suffix = secrets.token_hex(4)
    owner_email = f"smoke-owner-{suffix}@example.com"
    viewer_email = f"smoke-viewer-{suffix}@example.com"
    owner = httpx.Client(base_url=args.web, timeout=15, follow_redirects=False)
    viewer = httpx.Client(base_url=args.web, timeout=15, follow_redirects=False)
    state: dict[str, str] = {}

    def web_health() -> None:
        r = owner.get("/healthz")
        expect(r.status_code == 200 and r.json()["status"] == "ok", f"/healthz: {r.text}")

    def unauthenticated_redirect() -> None:
        r = owner.get("/workspaces")
        expect(r.status_code == 307, f"expected redirect, got {r.status_code}")
        expect(r.headers["location"].endswith("/login?next=%2Fworkspaces"), r.headers["location"])
        r = owner.get("/login")
        expect(r.status_code == 200 and "Sign in" in r.text, "login page did not render")

    def signup() -> None:
        r = owner.post("/api/v1/auth/register", json={"email": owner_email, "password": PASSWORD})
        expect(r.status_code == 201, f"register: {r.status_code} {r.text}")
        cookies = r.headers.get_list("set-cookie")
        expect(any(c.startswith("cw_access=") and "HttpOnly" in c for c in cookies), str(cookies))
        state["csrf"] = r.json()["csrf_token"]
        me = owner.get("/api/v1/auth/me")
        expect(me.status_code == 200 and me.json()["email"] == owner_email, me.text)

    def authenticated_page() -> None:
        r = owner.get("/workspaces")
        expect(r.status_code == 200, f"/workspaces page: {r.status_code}")
        expect("Workspaces" in r.text and owner_email in r.text, "app shell missing user/heading")

    def csrf_enforced() -> None:
        r = owner.post("/api/v1/workspaces", json={"name": "No CSRF"})
        expect(r.status_code == 403, f"missing CSRF accepted: {r.status_code}")

    def create_workspace() -> None:
        r = owner.post(
            "/api/v1/workspaces",
            json={"name": f"Smoke workspace {suffix}"},
            headers={"X-CSRF-Token": state["csrf"]},
        )
        expect(r.status_code == 201, f"create: {r.status_code} {r.text}")
        state["workspace_id"] = r.json()["id"]
        listed = owner.get("/api/v1/workspaces").json()
        expect(any(w["id"] == state["workspace_id"] for w in listed), "workspace not listed")
        page = owner.get(f"/workspaces/{state['workspace_id']}")
        expect(page.status_code == 200, f"workspace page: {page.status_code}")

    def membership() -> None:
        r = viewer.post("/api/v1/auth/register", json={"email": viewer_email, "password": PASSWORD})
        expect(r.status_code == 201, r.text)
        viewer_csrf = r.json()["csrf_token"]
        ws = f"/api/v1/workspaces/{state['workspace_id']}"
        expect(viewer.get(ws).status_code == 404, "non-member could read workspace")
        added = owner.post(
            f"{ws}/members",
            json={"email": viewer_email, "role": "viewer"},
            headers={"X-CSRF-Token": state["csrf"]},
        )
        expect(added.status_code == 201, added.text)
        expect(viewer.get(ws).json()["role"] == "viewer", "viewer role not visible")
        denied = viewer.patch(ws, json={"name": "x"}, headers={"X-CSRF-Token": viewer_csrf})
        expect(denied.status_code == 403, f"viewer rename: {denied.status_code}")

    def rename_and_delete() -> None:
        ws = f"/api/v1/workspaces/{state['workspace_id']}"
        headers = {"X-CSRF-Token": state["csrf"]}
        r = owner.patch(ws, json={"name": "Renamed by smoke"}, headers=headers)
        expect(r.status_code == 200 and r.json()["name"] == "Renamed by smoke", r.text)
        r = owner.delete(ws, headers=headers)
        expect(r.status_code == 204, r.text)
        expect(owner.get(ws).status_code == 404, "deleted workspace still readable")

    def logout() -> None:
        token = owner.cookies.get("cw_access")
        r = owner.post("/api/v1/auth/logout", headers={"X-CSRF-Token": state["csrf"]})
        expect(r.status_code == 204, r.text)
        expect(owner.get("/api/v1/auth/me").status_code == 401, "session survived logout")
        replay = httpx.Client(base_url=args.web, timeout=15, cookies={"cw_access": token or ""})
        expect(replay.get("/api/v1/auth/me").status_code == 401, "revoked token accepted")

    def backend_health() -> None:
        r = httpx.get(f"{args.api}/api/v1/health/ready", timeout=10)
        expect(r.status_code == 200 and r.json()["status"] == "ok", r.text)
        mode = httpx.get(f"{args.model_service}/health", timeout=10).json()["rule_check_mode"]
        r = httpx.post(
            f"{args.model_service}/v1/check",
            json={"hypothesis": "h", "spans": [{"span_id": "s", "text": "t"}]},
            timeout=30,
        )
        if mode == "untrained_fallback":  # no verified artifact: honest refusal, no verdict
            expect(r.status_code == 503, f"fallback /v1/check must be 503, got {r.status_code}")
            expect(r.json()["error"]["code"] == "rule_check_unavailable", r.text)
        else:  # a verified production artifact is installed
            expect(r.status_code == 200 and r.json()["model_kind"] == "production", r.text)
        r = httpx.get(f"{args.mcp}/health", timeout=10)
        expect(r.status_code == 200 and len(r.json()["implemented_tools"]) == 9, r.text)
        # The MCP endpoint itself demands a per-user API token (Phase 5).
        r = httpx.post(f"{args.mcp}/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        expect(r.status_code == 401 and r.headers["www-authenticate"].startswith("Bearer"), r.text)

    try:
        step("web /healthz reaches API", web_health)
        step("unauthenticated page redirects to /login", unauthenticated_redirect)
        step("sign up via web proxy sets httpOnly session", signup)
        step("server-verified app shell renders for session", authenticated_page)
        step("state change without CSRF token rejected", csrf_enforced)
        step("create + list workspace; detail page renders", create_workspace)
        step("membership: non-member 404, viewer cannot rename", membership)
        step("owner rename + delete", rename_and_delete)
        step("logout revokes session server-side", logout)
        step("api ready; model-service rule check honest; mcp health + token auth", backend_health)
    except (SmokeError, httpx.HTTPError, KeyError) as exc:
        print(f"FAIL  {type(exc).__name__}: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    print("Phase 0 smoke test passed")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
