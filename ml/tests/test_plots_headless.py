"""Evaluation plots work under a notebook's inherited, unloadable matplotlib backend.

Kaggle (and Colab) kernels export MPLBACKEND=module://matplotlib_inline.backend_inline; every
pipeline step is a child process in the project environment, which has no matplotlib-inline,
so the inherited value made `import matplotlib` raise inside `ml.eval.evaluate`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ml.common.provenance import REPO

INLINE = "module://matplotlib_inline.backend_inline"


def _child(code: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["MPLBACKEND"] = INLINE
    return subprocess.run(  # noqa: S603 - fixed interpreter and code
        [sys.executable, "-c", code],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def test_plots_and_evaluate_import_with_an_inherited_inline_backend(tmp_path: Path) -> None:
    # Precondition: the project environment really cannot load the inherited backend.
    control = _child(
        "import importlib.util, sys\n"
        "assert importlib.util.find_spec('matplotlib_inline') is None\n"
        "import matplotlib\n"
    )
    assert control.returncode != 0, "the inherited backend no longer fails; test is moot"
    assert "ValueError" in control.stderr
    assert "backend" in control.stderr

    out = tmp_path / "confusion.png"
    child = _child(
        "import sys\n"
        "from pathlib import Path\n"
        "import ml.eval.evaluate  # the module whose import failed on Kaggle\n"
        "from ml.eval import plots\n"
        "import matplotlib\n"
        "print('backend', matplotlib.get_backend())\n"
        f"plots.confusion(Path({str(out)!r}), [[5, 1, 0], [2, 7, 1], [0, 1, 9]], 'test')\n"
    )
    assert child.returncode == 0, child.stderr
    assert "backend agg" in child.stdout.lower()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert out.stat().st_size > 1000
