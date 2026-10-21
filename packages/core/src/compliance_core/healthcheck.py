"""Container healthcheck client (slim Python images ship without curl).

Usage: ``python -m compliance_core.healthcheck URL [--expect KEY=VALUE ...]``

Exits 0 only when the endpoint answers HTTP 200 with a JSON object whose fields match
every ``--expect`` pair, so a process that is up but degraded is reported unhealthy.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def check(url: str, expectations: list[tuple[str, str]], timeout: float) -> tuple[bool, str]:
    if not url.startswith(("http://", "https://")):
        return False, f"refusing non-HTTP url {url!r}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - scheme checked above
            status = response.status
            body = response.read(65536)
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.read(2048).decode(errors='replace')}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, f"request failed: {exc}"
    if status != 200:
        return False, f"HTTP {status}"
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return False, "response is not JSON"
    if not isinstance(payload, dict):
        return False, "response is not a JSON object"
    for key, expected in expectations:
        actual = payload.get(key)
        if str(actual).lower() != expected.lower():
            return False, f"{key}={actual!r}, expected {expected!r}"
    return True, "healthy"


def _pair(raw: str) -> tuple[str, str]:
    key, sep, value = raw.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {raw!r}")
    return key, value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--expect", type=_pair, action="append", default=[])
    parser.add_argument("--timeout", type=float, default=4.0)
    args = parser.parse_args(argv)
    ok, message = check(args.url, args.expect, args.timeout)
    print(message, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
