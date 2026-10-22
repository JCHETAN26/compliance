"""The Colab notebook is a clean, syntactically valid thin wrapper over existing scripts."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from ml.common.provenance import REPO

NOTEBOOK = REPO / "ml" / "notebooks" / "colab_train.ipynb"


def _cells() -> list[dict[str, object]]:
    return list(json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"])


def test_notebook_has_no_outputs_and_valid_code() -> None:
    for cell in _cells():
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            ast.parse("".join(cell["source"]))  # type: ignore[arg-type]


def test_notebook_only_calls_existing_repository_modules() -> None:
    source = "\n".join("".join(c["source"]) for c in _cells())  # type: ignore[arg-type]
    modules = set(re.findall(r'run\(\s*"([a-z_.]+)"', source))
    assert {"ml.data.prepare", "ml.train.train", "ml.eval.evaluate", "ml.export.export"} <= modules
    assert re.search(r'"ml.colab.bundle",\s*"pack"', source)
    for module in modules:
        assert find_spec(module) is not None, module
    for config in re.findall(r"ml/train/configs/[a-z0-9_]+\.yaml", source):
        assert (REPO / config).is_file(), config


def test_notebook_never_embeds_or_prints_secrets() -> None:
    source = "\n".join("".join(c["source"]) for c in _cells())  # type: ignore[arg-type]
    assert not re.search(r"(ghp_|github_pat_|sk-ant-)[A-Za-z0-9_]+", source)
    assert "@github.com" not in source  # no credentials in clone URLs
    for line in source.splitlines():
        if "print(" in line:
            assert "TOKEN" not in line
            assert "API_KEY" not in line
    assert 'userdata.get("GITHUB_TOKEN")' in source  # Colab secret, never a literal
    assert "GIT_ASKPASS" in source


def _notebook_source_guard() -> Any:
    """The notebook's own guard (GENERATED_RESULTS + implementation_changes), executed as is."""
    cell = next(c for c in _cells() if "def implementation_changes" in "".join(c["source"]))  # type: ignore[arg-type]
    tree = ast.parse("".join(cell["source"]))  # type: ignore[arg-type]
    keep: list[ast.stmt] = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "implementation_changes")
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "GENERATED_RESULTS" for t in node.targets)
        )
    ]
    assert len(keep) == 2
    namespace: dict[str, Any] = {"subprocess": subprocess}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)  # noqa: S102
    return namespace["implementation_changes"]


def test_notebook_source_guard_allows_only_generated_results(tmp_path: Path) -> None:
    implementation_changes = _notebook_source_guard()
    repo = tmp_path / "repo"
    git = [
        "git",
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
        "-c",
        "commit.gpgsign=false",
        "-C",
        str(repo),
    ]
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)  # noqa: S603, S607
    (repo / "ml/eval/results").mkdir(parents=True)
    (repo / "ml/train").mkdir(parents=True)
    (repo / "ml/eval/results/data_report.json").write_text("{}\n")
    (repo / "ml/train/train.py").write_text("x = 1\n")
    subprocess.run([*git, "add", "."], check=True)  # noqa: S603
    subprocess.run([*git, "commit", "--quiet", "-m", "init"], check=True)  # noqa: S603
    assert implementation_changes(repo) == ""

    # What a same-runtime rerun produces: the tracked report rewritten, new result files.
    (repo / "ml/eval/results/data_report.json").write_text('{"regenerated": true}\n')
    (repo / "ml/eval/results/model_eval.json").write_text("{}\n")
    (repo / "ml/eval/results/plots").mkdir()
    (repo / "ml/eval/results/plots/confusion.png").write_bytes(b"png")
    assert implementation_changes(repo) == ""

    # Implementation changes are still refused: a modified tracked source file ...
    (repo / "ml/train/train.py").write_text("x = 2\n")
    assert "ml/train/train.py" in implementation_changes(repo)
    (repo / "ml/train/train.py").write_text("x = 1\n")
    assert implementation_changes(repo) == ""
    # ... an untracked implementation file, including one that merely shares the prefix ...
    for name in ("ml/train/extra.py", "ml/eval/results_extra.py"):
        (repo / name).write_text("y = 1\n")
        assert name in implementation_changes(repo)
        (repo / name).unlink()
    # ... and a deleted tracked file.
    (repo / "ml/train/train.py").unlink()
    assert "ml/train/train.py" in implementation_changes(repo)
