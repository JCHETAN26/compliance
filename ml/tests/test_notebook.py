"""The Colab notebook is a clean, syntactically valid thin wrapper over existing scripts."""

from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest

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


def _source() -> str:
    return "\n".join("".join(c["source"]) for c in _cells())  # type: ignore[arg-type]


def _helpers_source() -> str:
    cell = next(c for c in _cells() if "def stream(" in "".join(c["source"]))  # type: ignore[arg-type]
    return "".join(cell["source"])  # type: ignore[arg-type]


def _helpers(tmp_path: Path, *, location: str = "local") -> dict[str, Any]:
    """The notebook's helper cell, executed as is with a temporary RUNS / DRIVE_ROOT."""
    drive = tmp_path / "drive" / "compliance-runs"
    runs = drive if location == "drive" else tmp_path / "local-runs"
    drive.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    namespace: dict[str, Any] = {
        "RUNS": runs,
        "DRIVE_ROOT": drive,
        "RUN_LOCATION": location,
        "PY": sys.executable,
        "EXPECTED_COMMIT": "0123456789ab" + "0" * 28,
    }
    exec(compile(_helpers_source(), str(NOTEBOOK), "exec"), namespace)  # noqa: S102
    return namespace


def test_helper_cell_only_defines() -> None:
    """Running the helper cell has no side effects (it is safe to rerun at any point)."""
    for node in ast.parse(_helpers_source()).body:
        assert isinstance(node, ast.Import | ast.ImportFrom | ast.FunctionDef | ast.Assign), node


def test_layout_defaults_to_local_runs_and_never_deletes_on_drive() -> None:
    source = _source()
    assert 'RUN_LOCATION = "local"' in source
    assert 'pathlib.Path("/content/compliance-runs")' in source
    assert 'BUNDLE = DRIVE_ROOT / "compliance.bundle"' in source
    assert 'DATA_DRIVE = DRIVE_ROOT / "data-full"' in source
    assert "RUN_LLM = False" in source
    assert 'EXPECTED_COMMIT = ""' in source  # filled in only in the supervisor's copy
    for name in ("deberta_v3_base", "modernbert_base", "deberta_v3_base_unseen4"):
        assert f'train("{name}", "ml/train/configs/{name}.yaml")' in source
    for line in source.splitlines():
        if re.search(r"rmtree|\.unlink\(|os\.remove|\.rename\(|\.replace\(", line):
            assert "DRIVE" not in line, line  # nothing on Drive is ever removed or moved


def test_stream_shows_and_logs_stdout_and_stderr_without_environment(
    tmp_path: Path, capsys: Any
) -> None:
    ns = _helpers(tmp_path)
    script = "import os, sys; print('out-line'); print('err-line', file=sys.stderr); print('secret-len', len(os.environ['X_SECRET']))"
    log_path = ns["stream"]([sys.executable, "-c", script], "child", {"X_SECRET": "s3cr3t"})
    shown = capsys.readouterr().out
    logged = Path(log_path).read_text()
    assert log_path.parent == tmp_path / "local-runs" / "logs"
    for text in (shown, logged):
        assert "out-line" in text
        assert "err-line" in text
        assert "secret-len 6" in text
        assert "s3cr3t" not in text
        assert "X_SECRET=" not in text
    assert str(log_path) in shown


def test_run_propagates_failure_with_its_output(tmp_path: Path, capsys: Any) -> None:
    ns = _helpers(tmp_path)
    (tmp_path / "failing_child.py").write_text(
        "import sys\nprint('boom', file=sys.stderr)\nsys.exit(3)\n"
    )
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(subprocess.CalledProcessError) as err:
            ns["run"]("failing_child")
    finally:
        os.chdir(cwd)
    assert err.value.returncode == 3
    shown = capsys.readouterr().out
    assert "boom" in shown
    assert "FAILED: failing_child exited with 3" in shown


_HARNESS = """
import sys
from pathlib import Path
ns = {"RUNS": Path(sys.argv[2]), "DRIVE_ROOT": Path(sys.argv[3]), "RUN_LOCATION": "local", "PY": sys.executable}
exec(Path(sys.argv[1]).read_text(), ns)
ns["STOP_GRACE_S"] = (float(sys.argv[4]), float(sys.argv[4]))
try:
    ns["run"]("slow_child")
except KeyboardInterrupt:
    print("HARNESS-INTERRUPTED", flush=True)
"""

_CHILD = """
import os, signal, sys, time
from pathlib import Path
if os.environ.get("STUBBORN"):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path("child.pid").write_text(str(os.getpid()))
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("CHILD-GOT-SIGINT", flush=True)
    sys.exit(130)
"""


@pytest.mark.parametrize("stubborn", [False, True])
def test_interrupt_stops_and_reaps_the_child(tmp_path: Path, stubborn: bool) -> None:
    (tmp_path / "helpers.py").write_text(_helpers_source())
    (tmp_path / "harness.py").write_text(_HARNESS)
    (tmp_path / "slow_child.py").write_text(_CHILD)
    (tmp_path / "runs").mkdir()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if stubborn:
        env["STUBBORN"] = "1"
    harness = subprocess.Popen(
        [sys.executable, "harness.py", "helpers.py", "runs", "drive", "0.5"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    pid_file = tmp_path / "child.pid"
    deadline = time.monotonic() + 30
    while not pid_file.exists() or not pid_file.read_text():
        assert time.monotonic() < deadline, "child did not start"
        time.sleep(0.05)
    child = int(pid_file.read_text())
    harness.send_signal(signal.SIGINT)  # the Colab stop button interrupts the kernel only
    out, _ = harness.communicate(timeout=60)
    assert harness.returncode == 0, out
    assert "stopping slow_child" in out
    assert "after KeyboardInterrupt" in out
    assert "HARNESS-INTERRUPTED" in out
    assert ("CHILD-GOT-SIGINT" in out) is (not stubborn)  # the child's own output is not lost
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)  # stopped and reaped: no orphan keeps writing


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_backup_to_drive_copies_verified_into_new_folders(tmp_path: Path, capsys: Any) -> None:
    ns = _helpers(tmp_path)
    run_dir = ns["RUNS"] / "deberta_v3_base"
    (run_dir / "checkpoints" / "best").mkdir(parents=True)
    (run_dir / "checkpoints" / "last").mkdir()
    (run_dir / "checkpoints" / "best" / "model.safetensors").write_bytes(b"w" * 1000)
    (run_dir / "checkpoints" / "last" / "trainer.pt").write_bytes(b"t" * 1000)
    (run_dir / "run.json").write_text("{}")
    existing = ns["DRIVE_ROOT"] / "deberta_v3_base" / "keep.txt"  # e.g. the failed run's folder
    existing.parent.mkdir()
    existing.write_text("untouched")

    first = ns["backup_to_drive"](
        run_dir, "deberta_v3_base", exclude=("checkpoints/last",), reserve_gib=0
    )
    second = ns["backup_to_drive"](
        run_dir, "deberta_v3_base", exclude=("checkpoints/last",), reserve_gib=0
    )
    assert first != second
    assert first.parent == second.parent == ns["DRIVE_ROOT"] / "backups"
    expected = {k: v for k, v in _tree(run_dir).items() if not k.startswith("checkpoints/last/")}
    for dest in (first, second):
        copied = _tree(dest)
        sums = copied.pop("SHA256SUMS").decode()
        assert copied == expected
        assert len(sums.splitlines()) == len(expected)
    assert existing.read_text() == "untouched"

    single = ns["backup_to_drive"](run_dir / "run.json", "run-record", reserve_gib=0)
    assert (single / "run.json").read_text() == "{}"

    before = _tree(ns["DRIVE_ROOT"])
    assert ns["backup_to_drive"](run_dir, "too-big", reserve_gib=10**9) is None
    assert "SKIPPED" in capsys.readouterr().out
    assert _tree(ns["DRIVE_ROOT"]) == before  # nothing copied, nothing removed

    (run_dir / "link").symlink_to(run_dir / "run.json")
    with pytest.raises(RuntimeError, match="not a regular file"):
        ns["backup_to_drive"](run_dir, "with-link", reserve_gib=0)
    assert _tree(ns["DRIVE_ROOT"]) == before  # refused before copying anything


def test_backup_is_skipped_when_runs_are_on_drive(tmp_path: Path) -> None:
    ns = _helpers(tmp_path, location="drive")
    (ns["RUNS"] / "x").write_text("x")
    assert ns["backup_to_drive"](ns["RUNS"] / "x", "x", reserve_gib=0) is None
    assert not (ns["DRIVE_ROOT"] / "backups").exists()


def test_require_space_refuses_without_deleting(tmp_path: Path) -> None:
    ns = _helpers(tmp_path)
    ns["require_space"](tmp_path, 0, "ok")
    with pytest.raises(RuntimeError, match="nothing is deleted automatically"):
        ns["require_space"](tmp_path, 10**9, "training")


class _BrokenFromThreads:
    """sys.stdout whose writes fail outside the main thread, i.e. in stream()'s output relay."""

    def __init__(self, real: Any) -> None:
        self.real = real

    def write(self, text: str) -> int:
        if threading.current_thread() is not threading.main_thread():
            raise OSError("relay broken")
        return int(self.real.write(text))

    def flush(self) -> None:
        self.real.flush()


_CHATTY = """
import os, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
while True:
    print("x" * 1000)
"""


def _chatty(ns: dict[str, Any], tmp_path: Path) -> tuple[Any, Path]:
    """stream() on a child that writes forever (it blocks as soon as nobody drains its pipe)."""
    pid_file = tmp_path / "chatty.pid"
    with pytest.raises(RuntimeError, match="showing/logging the output of chatty failed") as err:
        ns["stream"]([sys.executable, "-c", _CHATTY, pid_file], "chatty")
    return err.value, pid_file


def test_output_relay_failure_stops_the_child_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ns = _helpers(tmp_path)
    ns["STOP_GRACE_S"] = (2.0, 2.0)
    monkeypatch.setattr(sys, "stdout", _BrokenFromThreads(sys.stdout))
    started = time.monotonic()
    error, pid_file = _chatty(ns, tmp_path)
    assert time.monotonic() - started < 30  # neither the child nor the reader blocks the cell
    assert isinstance(error.__cause__, OSError)
    assert "log: " in str(error)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)  # stopped and reaped


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_log_failure_stops_the_child_and_raises(tmp_path: Path) -> None:
    ns = _helpers(tmp_path)
    ns["STOP_GRACE_S"] = (2.0, 2.0)
    logs = ns["RUNS"] / "logs"
    logs.mkdir()
    logs.chmod(0o500)  # the log cannot be created
    try:
        error, pid_file = _chatty(ns, tmp_path)
    finally:
        logs.chmod(0o700)
    assert isinstance(error.__cause__, PermissionError)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def _source_cell() -> dict[str, Any]:
    """The source cell's definitions (no clone or checkout at /content), executed as is."""
    cell = next(c for c in _cells() if "def checkout_source" in "".join(c["source"]))  # type: ignore[arg-type]
    tree = ast.parse("".join(cell["source"]))  # type: ignore[arg-type]
    keep: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.Import | ast.FunctionDef)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id in ("GENERATED_RESULTS", "GITHUB_REPO")
                for t in node.targets
            )
        )
    ]
    namespace: dict[str, Any] = {
        "subprocess": subprocess,
        "pathlib": pathlib,
        "shutil": shutil,
        "os": os,
    }
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)  # noqa: S102
    return namespace


_GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]


def _git_out(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603
        [*_GIT, "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(origin: Path, train_py: str) -> str:
    (origin / "ml/train/train.py").write_text(train_py)
    _git_out(origin, "add", ".")
    _git_out(origin, "commit", "--quiet", "-m", train_py)
    return _git_out(origin, "rev-parse", "HEAD")


def test_source_cell_updates_an_existing_checkout_only_from_a_verified_bundle(
    tmp_path: Path,
) -> None:
    checkout = _source_cell()["checkout_source"]
    origin = tmp_path / "origin"
    (origin / "ml/eval/results").mkdir(parents=True)
    (origin / "ml/train").mkdir(parents=True)
    _git_out(tmp_path, "init", "--quiet", "-b", "main", str(origin))
    (origin / "ml/eval/results/data_report.json").write_text("{}\n")
    bundle = tmp_path / "drive" / "compliance.bundle"
    bundle.parent.mkdir()

    def publish() -> None:  # what the supervisor uploads: a complete bundle replacing the old one
        bundle.unlink(missing_ok=True)
        _git_out(origin, "bundle", "create", str(bundle), "HEAD", "main")

    a = _commit(origin, "x = 1\n")
    publish()
    repo = tmp_path / "content" / "compliance"
    assert checkout(repo, a, bundle, False) == a  # fresh runtime: verified clone

    # Same runtime, new commit in a replaced bundle; the pipeline has rewritten a generated file.
    (repo / "ml/eval/results/data_report.json").write_text('{"regenerated": true}\n')
    b = _commit(origin, "x = 2\n")
    publish()
    assert checkout(repo, b, bundle, False) == b
    assert (repo / "ml/train/train.py").read_text() == "x = 2\n"
    assert (repo / "ml/eval/results/data_report.json").read_text() == '{"regenerated": true}\n'
    assert _git_out(repo, "rev-parse", "refs/heads/main") == a  # fetched into FETCH_HEAD only

    # An implementation change is refused before the checkout; nothing is overwritten.
    c = _commit(origin, "x = 3\n")
    publish()
    (repo / "ml/train/train.py").write_text("local edit\n")
    with pytest.raises(RuntimeError, match="implementation changes"):
        checkout(repo, c, bundle, False)
    assert _git_out(repo, "rev-parse", "HEAD") == b
    assert (repo / "ml/train/train.py").read_text() == "local edit\n"
    (repo / "ml/train/train.py").write_text("x = 2\n")

    # Commits that are not available, a corrupt bundle, Option B on an existing checkout.
    with pytest.raises(RuntimeError, match="is not in"):
        checkout(repo, "f" * 40, bundle, False)
    with pytest.raises(RuntimeError, match="Disconnect and delete runtime"):
        checkout(repo, "e" * 40, bundle, True)  # no network access attempted
    d = _commit(origin, "x = 4\n")
    bundle.write_bytes(b"not a bundle")
    with pytest.raises(RuntimeError, match=r"git bundle verify .* failed"):
        checkout(repo, d, bundle, False)
    assert _git_out(repo, "rev-parse", "HEAD") == b

    # A commit already present needs no bundle; the result passes the exact-source guard.
    assert checkout(repo, c, bundle, False) == c
    assert _source_cell()["implementation_changes"](repo) == ""
