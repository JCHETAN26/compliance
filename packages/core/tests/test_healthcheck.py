from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from compliance_core.healthcheck import check, main

ROUTES: dict[str, tuple[int, bytes]] = {
    "/ok": (200, json.dumps({"status": "ok"}).encode()),
    "/degraded": (200, json.dumps({"status": "error"}).encode()),
    "/unavailable": (503, json.dumps({"status": "error"}).encode()),
    "/html": (200, b"<html></html>"),
    "/list": (200, b"[1, 2]"),
}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        status, body = ROUTES.get(self.path, (404, b"{}"))
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_healthy_endpoint_passes(base_url: str) -> None:
    assert check(f"{base_url}/ok", [("status", "ok")], 2.0) == (True, "healthy")
    assert main([f"{base_url}/ok", "--expect", "status=ok"]) == 0


@pytest.mark.parametrize(
    ("path", "fragment"),
    [
        ("/degraded", "status='error'"),
        ("/unavailable", "HTTP 503"),
        ("/html", "not JSON"),
        ("/list", "not a JSON object"),
        ("/missing", "HTTP 404"),
    ],
)
def test_unhealthy_responses_fail(base_url: str, path: str, fragment: str) -> None:
    ok, message = check(f"{base_url}{path}", [("status", "ok")], 2.0)
    assert not ok
    assert fragment in message
    assert main([f"{base_url}{path}", "--expect", "status=ok"]) == 1


def test_connection_refused_fails() -> None:
    ok, message = check("http://127.0.0.1:9/health", [], 1.0)
    assert not ok
    assert "request failed" in message


def test_non_http_scheme_refused() -> None:
    ok, message = check("file:///etc/passwd", [], 1.0)
    assert not ok
    assert "non-HTTP" in message
