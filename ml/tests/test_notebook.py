"""The Colab notebook is a clean, syntactically valid thin wrapper over existing scripts."""

from __future__ import annotations

import ast
import json
import re
from importlib.util import find_spec

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
