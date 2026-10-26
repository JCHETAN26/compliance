"""Wheel packaging: every file a workspace wheel force-includes must reach the Docker build.

The service images build the workspace wheels from the Docker build context (filtered by
``.dockerignore``). A force-included path that the context excludes makes hatchling fail
the image build, so the built-in rule packs could never ship.
"""

from __future__ import annotations

import posixpath
import re
import tomllib
from pathlib import Path

from compliance_core.rule_packs import builtin_pack_files

ROOT = Path(__file__).resolve().parents[3]


def _pattern_regex(pattern: str) -> re.Pattern[str]:
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile(out)


def _dockerignore() -> list[tuple[bool, re.Pattern[str]]]:
    rules = []
    for raw in (ROOT / ".dockerignore").read_text("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        pattern = posixpath.normpath(line.removeprefix("!").strip()).lstrip("/")
        rules.append((negated, _pattern_regex(pattern)))
    return rules


def _excluded(rel: str, rules: list[tuple[bool, re.Pattern[str]]]) -> bool:
    """Docker semantics: the last matching rule wins; a rule matching a parent applies."""
    parts = rel.split("/")
    candidates = ["/".join(parts[: n + 1]) for n in range(len(parts))]
    excluded = False
    for negated, regex in rules:
        if any(regex.fullmatch(c) for c in candidates):
            excluded = not negated
    return excluded


def _force_includes() -> list[tuple[Path, str]]:
    root = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    found = []
    for member in root["tool"]["uv"]["workspace"]["members"]:
        project = tomllib.loads((ROOT / member / "pyproject.toml").read_text("utf-8"))
        targets = project.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {})
        for source, dest in targets.get("wheel", {}).get("force-include", {}).items():
            found.append(((ROOT / member / source).resolve(), dest))
    return found


def test_dockerignore_matcher_follows_docker_rules() -> None:
    rules = _dockerignore()
    assert _excluded("docs/PHASE4_REVIEW.md", rules)
    assert _excluded("packages/core/tests/test_packaging.py", rules)
    assert _excluded("packages/core/src/compliance_core/__pycache__/x.pyc", rules)
    assert not _excluded("packages/core/src/compliance_core/rule_packs.py", rules)
    assert not _excluded("uv.lock", rules)


def test_force_included_wheel_files_are_in_the_docker_build_context() -> None:
    rules = _dockerignore()
    includes = _force_includes()
    assert includes, "no force-include found; update this test with the packaging"
    for source, dest in includes:
        assert source.exists(), f"{source} (force-included as {dest}) is missing"
        files = [source] if source.is_file() else [p for p in source.rglob("*") if p.is_file()]
        assert files, f"{source} is empty"
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            assert not _excluded(rel, rules), f".dockerignore excludes force-included {rel}"


def test_wheel_ships_the_packs_the_checkout_loads() -> None:
    includes = {dest: source for source, dest in _force_includes()}
    # builtin_pack_dir() looks for this package-data directory in an installed wheel.
    source = includes["compliance_core/builtin_rule_packs"]
    shipped = sorted(p.name for p in source.glob("*.yaml"))
    assert shipped == sorted(p.name for p in builtin_pack_files())
    assert {"nda_standard.yaml", "vendor_confidentiality_review.yaml"} <= set(shipped)
