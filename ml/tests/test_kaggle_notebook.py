"""The Kaggle notebook is a thin, safe wrapper over the same scripts as the Colab notebook.

Notebook cells are executed as they are (definitions only, with temporary paths); the runtime
module is the file the notebook loads from the verified checkout. No Kaggle service is contacted.
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest

from ml.common.provenance import REPO
from ml.notebooks import kaggle_runtime as rt

NOTEBOOK = REPO / "ml" / "notebooks" / "kaggle_train.ipynb"
RUNTIME = REPO / "ml" / "notebooks" / "kaggle_runtime.py"
METADATA = REPO / "ml" / "notebooks" / "kaggle-kernel-metadata.template.json"
FINGERPRINT = "f" * 64
DATA_VERSION = "620358b51ef96fac" + "0" * 48


def _cells() -> list[dict[str, Any]]:
    return list(json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"])


def _code() -> list[str]:
    return ["".join(c["source"]) for c in _cells() if c["cell_type"] == "code"]


def _source() -> str:
    return "\n".join(_code())


def _cell(marker: str) -> str:
    return next(c for c in _code() if marker in c)


def _definitions(marker: str, namespace: dict[str, Any]) -> dict[str, Any]:
    """The imports, functions and constant assignments of one cell, executed as is."""
    tree = ast.parse(_cell(marker))
    keep: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.Import | ast.ImportFrom | ast.FunctionDef)
        or (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant))
    ]
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)  # noqa: S102
    return namespace


def _layout_cell() -> dict[str, Any]:
    return _definitions("def check_layout", {})


def _source_cell() -> dict[str, Any]:
    return _definitions("def checkout_source", _layout_cell())


# ----------------------------------------------------------------------------- structure


def test_notebook_has_no_outputs_and_valid_code() -> None:
    cells = _cells()
    assert len({c["id"] for c in cells}) == len(cells)
    for cell in cells:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            ast.parse("".join(cell["source"]))


def test_no_colab_drive_or_github_token_path() -> None:
    source = _source()
    for forbidden in ("google.colab", "drive.mount", "/content", "userdata", "GITHUB_TOKEN"):
        assert forbidden not in source, forbidden


def test_settings_defaults() -> None:
    settings = _code()[0]
    namespace: dict[str, Any] = {}
    exec(settings, namespace)  # noqa: S102 - constants only
    assert namespace["EXPECTED_COMMIT"] == ""  # filled in only in the supervisor's ready copy
    assert namespace["SOURCE_BUNDLE"] is None
    assert namespace["IMPORT_PREVIOUS_OUTPUT"] is None
    assert namespace["RUN_LLM"] is False
    assert namespace["OUTPUT_LIMIT_GB"] == 20
    assert set(namespace["TRAIN_RUNS"]) == set(rt.RUN_NAMES)
    assert all(isinstance(n, ast.Assign) for n in ast.parse(settings).body)


def test_only_existing_modules_configs_and_the_full_pipeline_in_order() -> None:
    source = _source()
    modules = set(re.findall(r'run\(\s*"([a-z_.]+)"', source))
    for module in modules:
        assert find_spec(module) is not None, module
    for config in re.findall(r"ml/train/configs/[a-z0-9_]+\.yaml", source):
        assert (REPO / config).is_file(), config
    steps = [
        '"ml.data.prepare"',
        '"ml.baselines.classic"',
        'train("deberta_v3_base", "ml/train/configs/deberta_v3_base.yaml")',
        'train("modernbert_base", "ml/train/configs/modernbert_base.yaml")',
        'train("deberta_v3_base_unseen4", "ml/train/configs/deberta_v3_base_unseen4.yaml")',
        '"ml.eval.evaluate"',
        '"ml.export.export"',
        '"ml.export.bench"',
        '"ml.export.eval_onnx"',
        '"ml.baselines.llm"',
        'evaluate("--bench", "ml/eval/results/latency_kaggle_cpu.json")',
        '"--refresh-manifest"',
        '"ml.colab.bundle",\n        "pack"',
    ]
    positions = [source.find(step) for step in steps]
    assert -1 not in positions, [s for s, p in zip(steps, positions, strict=True) if p == -1]
    assert positions == sorted(positions)
    assert source.count('"--split",\n        "test"') == 1  # quantized TEST accuracy
    for name in rt.RUN_NAMES:
        assert f'"{name}"' in source


def test_environment_is_isolated_locked_and_cuda_checked() -> None:
    env_cell = _cell('stream([UV, "sync", "--frozen"]')
    source = _source()
    assert '"--target",\n            str(UV_TOOL)' in env_cell  # uv beside, not in, Kaggle's env
    assert '"uv==0.8.22"' in env_cell
    assert '"torch==2.14.1+cu126"' in env_cell
    assert "https://download.pytorch.org/whl/cu126" in env_cell
    assert 'PY = str(REPO / ".venv/bin/python")' in env_cell
    assert "uv run" not in source
    assert '"make"' not in source
    assert 'info["torch"] == "2.14.1+cu126"' in env_cell
    assert 'info["python"] == [3, 11]' in env_cell
    # every pip/uv install names its target: never the kernel's own environment
    for call in re.findall(r"\[[^\[\]]*\"install\"[^\[\]]*\]", source, flags=re.S):
        assert '"--target"' in call or '"--python"' in call, call
    # caches in scratch; one GPU
    for var in ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "PIP_CACHE_DIR", "HF_HOME"):
        assert f'("{var}",' in source
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in source
    assert source.count("assert_cuda()") >= 5  # before data prep, each training run and export


def test_secret_is_read_only_in_run_llm_branches_and_never_shown() -> None:
    source = _source()
    assert not re.search(r"(ghp_|github_pat_|sk-ant-)[A-Za-z0-9_]+", source)
    # Only passed to the runtime helpers: an early check and the late read for the child.
    assert source.count("UserSecretsClient().get_secret)") == 2
    assert "get_secret(" not in source
    assert source.find("rt.require_secret(") < source.find('"ml.data.prepare"')
    for line in source.splitlines():
        if "print(" in line or "log(" in line:
            assert "API_KEY" not in line
            assert "key_env" not in line
    for code in _code():
        for node in ast.parse(code).body:
            if "UserSecretsClient" in ast.unparse(node):
                assert isinstance(node, ast.If), ast.unparse(node)
                assert ast.unparse(node.test) == "RUN_LLM"
    llm = ast.parse(_cell('rt.secret_env("ANTHROPIC_API_KEY"'))
    finals = [f for n in ast.walk(llm) if isinstance(n, ast.Try) for f in n.finalbody]
    assert "del key_env" in [ast.unparse(f) for f in finals]
    # The runtime never logs a child's environment.
    assert "environ" not in "".join(
        line for line in RUNTIME.read_text().splitlines() if "log(" in line
    )


def test_outputs_go_to_working_and_nothing_is_removed_there() -> None:
    source = _source()
    assert 'RUNS = WORKING / "compliance-runs"' in source
    assert 'REPO = SCRATCH / "compliance"' in source
    assert "BUNDLE = source_bundle(INPUT, SOURCE_BUNDLE)" in source
    assert 'DATA_SAVED = RUNS / "data-full"' in source
    for line in source.splitlines():
        if re.search(r"rmtree|\.unlink\(|os\.remove|\.rename\(|\.replace\(", line):
            assert "RUNS" not in line
            assert "WORKING" not in line
            assert "INPUT" not in line
    for line in RUNTIME.read_text().splitlines():
        if "rmtree" in line:  # only scratch copies and the empty bundle-check repository
            assert "dest" in line, line


def test_kernel_metadata_template_is_private_gpu_with_internet() -> None:
    meta = json.loads(METADATA.read_text())
    assert meta["is_private"] == "true"
    assert meta["enable_gpu"] == "true"
    assert meta["enable_internet"] == "true"
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["kernel_type"] == "notebook"
    assert meta["id"].startswith("KAGGLE_USERNAME/")  # no account filled in
    assert meta["dataset_sources"] == ["KAGGLE_USERNAME/compliance-source"]


# ----------------------------------------------------------------------------- layout and source


def test_layout_keeps_inputs_outputs_and_scratch_apart(tmp_path: Path) -> None:
    ns = _layout_cell()
    check = ns["check_layout"]
    inp, work = Path("/kaggle/input"), Path("/kaggle/working")
    check(inp, work, work / "compliance-runs", Path("/kaggle/temp/s"), Path("/kaggle/temp/s/c"))
    check(inp, work, work / "compliance-runs", Path("/tmp/s"), Path("/tmp/s/c"))  # noqa: S108
    bad = [
        (work.parent / "runs", Path("/tmp/s"), Path("/tmp/s/c")),  # noqa: S108 - RUNS not saved
        (work / "runs", work / "scratch", work / "scratch/c"),  # venv would be saved
        (work / "runs", Path("/tmp/s"), work / "c"),  # noqa: S108 - checkout in the output
        (work / "runs", inp / "s", inp / "s/c"),  # writes to the inputs
        (work, Path("/tmp/s"), Path("/tmp/s/c")),  # noqa: S108 - RUNS is the working root
    ]
    for runs, scratch, repo in bad:
        with pytest.raises(RuntimeError, match="unsafe layout"):
            check(inp, work, runs, scratch, repo)
    first, second = tmp_path / "a", tmp_path / "b"
    second.mkdir()
    assert ns["pick_scratch"]([first, second]) == second  # a missing /kaggle/temp falls back
    first.mkdir()
    assert ns["pick_scratch"]([first, second]) == first


def test_bundle_discovery_needs_exactly_one_input_bundle(tmp_path: Path) -> None:
    ns = _source_cell()
    inp = tmp_path / "input"
    (inp / "other-dataset").mkdir(parents=True)
    (inp / "other-dataset" / "data.csv").write_text("x")
    with pytest.raises(RuntimeError, match=r"exactly one \*\.bundle .* found 0"):
        ns["source_bundle"](inp)
    one = inp / "compliance-source" / "compliance.bundle"
    one.parent.mkdir()
    one.write_bytes(b"b")
    assert ns["source_bundle"](inp) == one
    two = inp / "old-source" / "nested" / "compliance.bundle"
    two.parent.mkdir(parents=True)
    two.write_bytes(b"b")
    with pytest.raises(RuntimeError, match="found 2") as err:
        ns["source_bundle"](inp)
    assert str(one) in str(err.value)
    assert str(two) in str(err.value)
    assert ns["source_bundle"](inp, str(two)) == two  # an explicit choice resolves it
    outside = tmp_path / "working" / "compliance.bundle"
    outside.parent.mkdir()
    outside.write_bytes(b"b")
    for wrong in (outside, inp / "missing.bundle", inp / ".." / "working" / "compliance.bundle"):
        with pytest.raises(RuntimeError, match="is not a file under"):
            ns["source_bundle"](inp, str(wrong))
    deep = inp / "d1" / "d2" / "d3" / "d4" / "d5" / "deep.bundle"
    deep.parent.mkdir(parents=True)
    deep.write_bytes(b"b")
    assert deep not in ns["find_bundles"](inp)  # bounded search
    linked = tmp_path / "mounted"
    linked.mkdir()
    (linked / "x.bundle").write_bytes(b"b")
    (inp / "linked").symlink_to(linked)
    assert inp / "linked" / "x.bundle" in ns["find_bundles"](inp)


_GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603
        [*_GIT, "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(origin: Path, train_py: str) -> str:
    (origin / "ml/train/train.py").write_text(train_py)
    _git(origin, "add", ".")
    _git(origin, "commit", "--quiet", "-m", train_py)
    return _git(origin, "rev-parse", "HEAD")


def test_source_checkout_is_exact_clean_and_bundle_verified(tmp_path: Path) -> None:
    checkout = _source_cell()["checkout_source"]
    changes = _source_cell()["implementation_changes"]
    origin = tmp_path / "origin"
    (origin / "ml/eval/results").mkdir(parents=True)
    (origin / "ml/train").mkdir(parents=True)
    _git(tmp_path, "init", "--quiet", "-b", "main", str(origin))
    (origin / "ml/eval/results/data_report.json").write_text("{}\n")
    bundle = tmp_path / "input" / "compliance-source" / "compliance.bundle"
    bundle.parent.mkdir(parents=True)

    def publish() -> None:  # a new dataset version replacing the bundle
        bundle.unlink(missing_ok=True)
        _git(origin, "bundle", "create", str(bundle), "HEAD", "main")

    a = _commit(origin, "x = 1\n")
    b = _commit(origin, "x = 2\n")
    publish()
    repo = tmp_path / "scratch" / "compliance"
    assert checkout(repo, a, bundle) == a  # fresh session: an older commit, exactly
    assert (repo / "ml/train/train.py").read_text() == "x = 1\n"
    assert changes(repo) == ""
    assert not list((tmp_path / "scratch").glob("bundle-check-*"))

    # Same session: generated results rewritten, then a new commit from a replaced bundle.
    (repo / "ml/eval/results/data_report.json").write_text('{"regenerated": true}\n')
    (repo / "ml/eval/results/model_eval.json").write_text("{}\n")
    assert checkout(repo, b, bundle) == b
    c = _commit(origin, "x = 3\n")
    publish()
    assert checkout(repo, c, bundle) == c
    assert _git(repo, "rev-parse", "refs/heads/main") == b  # FETCH_HEAD only

    # An implementation change is refused before any checkout; nothing is overwritten.
    d = _commit(origin, "x = 4\n")
    publish()
    (repo / "ml/train/train.py").write_text("local edit\n")
    with pytest.raises(RuntimeError, match="implementation changes"):
        checkout(repo, d, bundle)
    assert (repo / "ml/train/train.py").read_text() == "local edit\n"
    assert _git(repo, "rev-parse", "HEAD") == c
    (repo / "ml/train/train.py").write_text("x = 3\n")
    (repo / "ml/train/extra.py").write_text("y = 1\n")  # untracked implementation file
    with pytest.raises(RuntimeError, match="implementation changes"):
        checkout(repo, d, bundle)
    (repo / "ml/train/extra.py").unlink()

    with pytest.raises(RuntimeError, match="is not in"):
        checkout(repo, "f" * 40, bundle)
    bundle.write_bytes(b"not a bundle")
    with pytest.raises(RuntimeError, match=r"git bundle verify .* failed"):
        checkout(repo, d, bundle)
    with pytest.raises(RuntimeError, match=r"git bundle verify .* failed"):
        checkout(tmp_path / "scratch" / "fresh", a, bundle)
    assert not (tmp_path / "scratch" / "fresh").exists()
    assert _git(repo, "rev-parse", "HEAD") == c


def _runtime_cell(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    ns: dict[str, Any] = {
        "REPO": REPO,
        "RUNS": tmp_path / "runs",
        "SCRATCH": tmp_path,
        "OUTPUT_LIMIT_GB": 20,
        "TRAIN_RUNS": ("deberta_v3_base",),
        "SCRATCH_MIN_GIB": 0,
        "RUN_LLM": False,
        **extra,
    }
    ns["RUNS"].mkdir(exist_ok=True)
    exec(compile(_cell("spec_from_file_location"), str(NOTEBOOK), "exec"), ns)  # noqa: S102
    return ns


def test_runtime_cell_loads_the_checkout_module_and_skips_unselected_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ns = _runtime_cell(tmp_path)
    assert Path(ns["rt"].__file__) == RUNTIME
    assert ns["OUTPUT_LIMIT"] == 20 * 10**9

    def no_gpu() -> None:
        raise AssertionError("an unselected run must not start")

    ns["assert_cuda"] = no_gpu
    ns["train"]("modernbert_base", "ml/train/configs/modernbert_base.yaml")
    assert "not in TRAIN_RUNS" in capsys.readouterr().out
    with pytest.raises(AssertionError):
        _runtime_cell(tmp_path, TRAIN_RUNS=("deberta_v3_base", "bogus"))


SECRET = "sk-ant-api03-test-value"  # a fake value; never a real key


def _fake_kaggle_secrets(monkeypatch: pytest.MonkeyPatch, value: object) -> list[str]:
    """kaggle_secrets with a UserSecretsClient whose get_secret returns (or raises) value."""
    calls: list[str] = []

    class UserSecretsClient:
        def get_secret(self, name: str) -> object:
            calls.append(name)
            if isinstance(value, BaseException):
                raise value
            return value

    module = types.ModuleType("kaggle_secrets")
    module.UserSecretsClient = UserSecretsClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kaggle_secrets", module)
    return calls


def test_early_secret_check_only_when_enabled_and_without_leaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _fake_kaggle_secrets(monkeypatch, SECRET)
    _runtime_cell(tmp_path)  # RUN_LLM = False: Kaggle Secrets is never asked
    assert calls == []
    capsys.readouterr()

    failures: list[object] = [
        None,
        "",
        "  \n",
        42,
        PermissionError(f"denied: x-api-key {SECRET}"),  # an error text must not be relayed
    ]
    for bad in failures:
        _fake_kaggle_secrets(monkeypatch, bad)
        with pytest.raises(RuntimeError, match=r"Add-ons -> Secrets") as err:
            _runtime_cell(tmp_path, RUN_LLM=True)
        assert err.value.__cause__ is None
        assert err.value.__context__ is None
        assert SECRET not in str(err.value)
    assert "not available (PermissionError)" in str(err.value)
    assert "saved output" not in capsys.readouterr().out  # stopped before the preflight

    calls = _fake_kaggle_secrets(monkeypatch, SECRET)
    ns = _runtime_cell(tmp_path, RUN_LLM=True)
    assert calls == ["ANTHROPIC_API_KEY"]
    shown = capsys.readouterr().out
    assert "ANTHROPIC_API_KEY is configured (value not shown" in shown
    assert SECRET not in shown
    assert all(value != SECRET for value in ns.values())  # nothing kept in the notebook


def test_secret_env_returns_the_value_only_for_the_child(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert rt.secret_env("ANTHROPIC_API_KEY", lambda name: SECRET) == {"ANTHROPIC_API_KEY": SECRET}
    with pytest.raises(RuntimeError, match="is empty or not text"):
        rt.secret_env("ANTHROPIC_API_KEY", lambda name: "")
    assert SECRET not in capsys.readouterr().out


# ----------------------------------------------------------------------------- commands


def test_stream_shows_and_logs_output_without_environment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = (
        "import os, sys; print('out-line'); print('err-line', file=sys.stderr); "
        "print('secret-len', len(os.environ['X_SECRET']))"
    )
    log_path = rt.stream(
        [sys.executable, "-c", script], "child", tmp_path / "logs", {"X_SECRET": "s3cr3t"}
    )
    shown = capsys.readouterr().out
    logged = log_path.read_text()
    assert log_path.parent == tmp_path / "logs"
    for text in (shown, logged):
        assert "out-line" in text
        assert "err-line" in text
        assert "secret-len 6" in text
        assert "s3cr3t" not in text
    (tmp_path / "failing_child.py").write_text("import sys\nprint('boom')\nsys.exit(3)\n")
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(subprocess.CalledProcessError) as err:
            rt.run_module(sys.executable, tmp_path / "logs", "failing_child")
    finally:
        os.chdir(cwd)
    assert err.value.returncode == 3
    shown = capsys.readouterr().out
    assert "boom" in shown
    assert "FAILED: failing_child exited with 3" in shown


def _kill_group(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


def _gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)


def _bounded_harness(
    tmp_path: Path, script: str, args: list[str], env: dict[str, str], pid_file: Path
) -> tuple[subprocess.Popen[str], Any]:
    """Start script in its own session; returns (process, cleanup). cleanup() always kills
    whatever is left of the harness and of the child named in pid_file (each in its own
    process group), so a failing or regressed test never leaves orphans behind."""
    (tmp_path / "harness.py").write_text(script)
    harness = subprocess.Popen(  # noqa: S603
        [sys.executable, "harness.py", *args],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    def cleanup() -> None:
        if harness.poll() is None:
            _kill_group(harness.pid)
        with contextlib.suppress(subprocess.TimeoutExpired, ValueError):
            harness.communicate(timeout=10)
        if pid_file.is_file() and pid_file.read_text().strip():
            child = int(pid_file.read_text())
            if not _gone(child, 0):
                _kill_group(child)  # stream() starts every child in its own session

    return harness, cleanup


def _wait_for_pid(pid_file: Path, harness: subprocess.Popen[str]) -> int:
    deadline = time.monotonic() + 30
    while not pid_file.is_file() or not pid_file.read_text().strip():
        assert harness.poll() is None, "harness exited before the child started"
        assert time.monotonic() < deadline, "child did not start"
        time.sleep(0.05)
    return int(pid_file.read_text())


_HARNESS = """
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("kaggle_runtime", sys.argv[1])
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)
rt.STOP_GRACE_S = (float(sys.argv[3]), float(sys.argv[3]))
try:
    rt.run_module(sys.executable, Path(sys.argv[2]), "slow_child")
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
    (tmp_path / "slow_child.py").write_text(_CHILD)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if stubborn:
        env["STUBBORN"] = "1"
    pid_file = tmp_path / "child.pid"
    harness, cleanup = _bounded_harness(
        tmp_path, _HARNESS, [str(RUNTIME), "logs", "0.5"], env, pid_file
    )
    try:
        child = _wait_for_pid(pid_file, harness)
        harness.send_signal(signal.SIGINT)  # the stop button interrupts the kernel only
        out, _ = harness.communicate(timeout=60)
        assert harness.returncode == 0, out
        assert "stopping slow_child" in out
        assert "HARNESS-INTERRUPTED" in out
        assert ("CHILD-GOT-SIGINT" in out) is (not stubborn)
        assert _gone(child), "the child survived the interrupt"  # no orphan keeps writing
    finally:
        cleanup()


_RELAY_HARNESS = """
import importlib.util, sys, threading
from pathlib import Path
spec = importlib.util.spec_from_file_location("kaggle_runtime", sys.argv[1])
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)
rt.STOP_GRACE_S = (2.0, 2.0)


class BrokenFromThreads:  # writes fail outside the main thread, i.e. in stream()'s relay
    def __init__(self, real):
        self.real = real

    def write(self, text):
        if threading.current_thread() is not threading.main_thread():
            raise OSError("relay broken")
        return self.real.write(text)

    def flush(self):
        self.real.flush()


sys.stdout = BrokenFromThreads(sys.stdout)
chatty = (
    "import os, pathlib, sys\\npathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\\n"
    "while True:\\n    print('x' * 1000)\\n"
)
try:
    rt.stream([sys.executable, "-c", chatty, sys.argv[2]], "chatty", Path(sys.argv[3]))
except RuntimeError as exc:
    print("RAISED", exc, "CAUSE", type(exc.__cause__).__name__, flush=True)
"""


def test_output_relay_failure_stops_the_child(tmp_path: Path) -> None:
    pid_file = tmp_path / "chatty.pid"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    harness, cleanup = _bounded_harness(
        tmp_path, _RELAY_HARNESS, [str(RUNTIME), str(pid_file), "logs"], env, pid_file
    )
    try:
        out, _ = harness.communicate(timeout=60)  # bounded even if the cleanup regresses
        assert harness.returncode == 0, out
        assert "RAISED showing/logging the output of chatty failed" in out
        assert "CAUSE OSError" in out
        assert _gone(int(pid_file.read_text())), "the child survived the relay failure"
    finally:
        cleanup()


# ----------------------------------------------------------------------------- copies and space


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _write(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data)
    else:
        path.write_bytes(data)
    return path


def test_copy_verified_never_overwrites_and_refuses_links(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write(src / "a.txt", "a")
    _write(src / "sub" / "b.bin", b"b" * 100)
    _write(src / "skip" / "c", "c")
    assert rt.copy_verified(src, tmp_path / "out", exclude=("skip",)) == 101
    assert _tree(tmp_path / "out") == {"a.txt": b"a", "sub/b.bin": b"b" * 100}
    with pytest.raises(FileExistsError):
        rt.copy_verified(src, tmp_path / "out")
    assert _tree(tmp_path / "out") == {"a.txt": b"a", "sub/b.bin": b"b" * 100}
    (src / "link").symlink_to(src / "a.txt")
    with pytest.raises(RuntimeError, match="not a regular file"):
        rt.copy_verified(src, tmp_path / "out2")
    assert not (tmp_path / "out2").exists()
    assert not list(tmp_path.glob(".staging-*"))  # refused before anything was staged

    scratch = tmp_path / "scratch"
    _write(scratch / "repo" / "data" / "old.txt", "old")
    rt.replace_scratch_copy(tmp_path / "out", scratch / "repo" / "data", scratch)
    assert _tree(scratch / "repo" / "data") == _tree(tmp_path / "out")
    for outside in (tmp_path / "out", scratch):
        with pytest.raises(RuntimeError, match="not inside scratch"):
            rt.replace_scratch_copy(src, outside, scratch)
    assert (tmp_path / "out" / "a.txt").read_text() == "a"


_WRITER = """
import sys, time
from pathlib import Path
path, go = Path(sys.argv[1]), Path(sys.argv[2])
path.parent.mkdir(parents=True, exist_ok=True)  # as ml.baselines.llm does
with path.open("a", encoding="utf-8") as handle:
    handle.write("attempt-1\\n")
    handle.flush()  # ml.baselines.llm flushes after every recorded attempt
    deadline = time.monotonic() + 30
    while not go.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    handle.write("attempt-2\\n")
"""


def test_llm_cache_link_puts_each_flushed_attempt_into_runs(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    link = scratch / "compliance" / ".cache" / "ml" / "llm"
    canonical = tmp_path / "working" / "compliance-runs" / "llm-cache"
    rt.link_cache(link, canonical, scratch)
    rt.link_cache(link, canonical, scratch)  # idempotent
    assert link.is_symlink()
    assert canonical.is_dir()
    assert not canonical.is_symlink()
    saved = canonical / "v" / "model-low-dev.jsonl"
    go = tmp_path / "go"
    child = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", _WRITER, str(link / "v" / "model-low-dev.jsonl"), str(go)],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not (saved.is_file() and saved.read_text() == "attempt-1\n"):
            assert child.poll() is None, "the writer exited early"
            assert time.monotonic() < deadline, "the attempt never reached RUNS"
            time.sleep(0.05)
        assert child.poll() is None  # observed in the saved output while the child still runs
        go.write_text("go")
        assert child.wait(timeout=30) == 0
    finally:
        if child.poll() is None:
            _kill_group(child.pid)
            child.wait(timeout=10)
    assert saved.read_text() == "attempt-1\nattempt-2\n"


def test_link_cache_merges_an_earlier_scratch_copy_without_losing_attempts(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    link = scratch / "repo" / ".cache" / "ml" / "llm"
    canonical = tmp_path / "runs" / "llm-cache"
    _write(canonical / "v" / "a.jsonl", "1\n2\n")  # canonical already holds more
    _write(canonical / "v" / "b.jsonl", "x\n")
    _write(link / "v" / "a.jsonl", "1\n")  # a stale prefix: canonical is kept
    _write(link / "v" / "b.jsonl", "x\ny\n")  # extends canonical
    _write(link / "v" / "c.jsonl", "c\n")  # only in the scratch copy
    old_copy = _tree(link)
    rt.link_cache(link, canonical, scratch)
    assert _tree(canonical) == {"v/a.jsonl": b"1\n2\n", "v/b.jsonl": b"x\ny\n", "v/c.jsonl": b"c\n"}
    assert link.is_symlink()
    (aside,) = link.parent.glob("llm.pre-link-*")
    assert _tree(aside) == old_copy  # moved aside within scratch, not deleted


def test_link_cache_refuses_divergence_foreign_links_and_paths_outside_scratch(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    link = scratch / "repo" / "llm"
    canonical = tmp_path / "runs" / "llm-cache"
    _write(canonical / "z.jsonl", "1\n")
    _write(link / "c.jsonl", "new\n")
    _write(link / "z.jsonl", "other\n")
    with pytest.raises(RuntimeError, match="diverges"):
        rt.link_cache(link, canonical, scratch)
    assert _tree(canonical) == {"z.jsonl": b"1\n"}  # checked before anything was written
    assert not link.is_symlink()
    assert _tree(link) == {"c.jsonl": b"new\n", "z.jsonl": b"other\n"}

    foreign = scratch / "repo" / "llm2"
    foreign.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(RuntimeError, match="points to"):
        rt.link_cache(foreign, canonical, scratch)
    assert os.readlink(foreign) == str(tmp_path / "elsewhere")

    linked_canonical = tmp_path / "runs" / "linked"
    linked_canonical.symlink_to(tmp_path / "input")
    with pytest.raises(RuntimeError, match="must be a real folder"):
        rt.link_cache(scratch / "repo" / "llm3", linked_canonical, scratch)
    with pytest.raises(RuntimeError, match="not inside scratch"):
        rt.link_cache(tmp_path / "working" / "llm", canonical, scratch)
    escape = scratch / "escape"
    escape.symlink_to(tmp_path / "working", target_is_directory=True)
    with pytest.raises(RuntimeError, match="not inside scratch"):
        rt.link_cache(escape / "llm", canonical, scratch)


def _run_dir(
    root: Path,
    name: str,
    *,
    status: str = "running",
    fingerprint: str = FINGERPRINT,
    data_version: str = DATA_VERSION,
    checkpoint: str | None = "last",
    complete: bool = True,
    weights: int = 10,
) -> Path:
    run = root / name
    _write(
        run / "run.json",
        json.dumps(
            {
                "status": status,
                "training_code_fingerprint": fingerprint,
                "data_version": data_version,
            }
        ),
    )
    _write(run / "checkpoints" / "best" / "model.safetensors", b"w" * weights)
    if checkpoint:
        _write(run / "checkpoints" / checkpoint / "trainer.pt", b"t" * weights)
        if complete:
            _write(run / "checkpoints" / checkpoint / "COMPLETE", "ok\n")
    return run


def test_run_status_mirrors_the_training_resume_rules(tmp_path: Path) -> None:
    assert rt.run_status(tmp_path / "none") == "absent"
    assert rt.run_status(_run_dir(tmp_path, "done", status="complete")) == "complete"
    assert rt.run_status(_run_dir(tmp_path, "r1")) == "resumable"
    assert rt.run_status(_run_dir(tmp_path, "r2", checkpoint=".last-staging")) == "resumable"
    assert (
        rt.run_status(_run_dir(tmp_path, "b", checkpoint=".last-staging", complete=False))
        == "blocked"
    )
    assert rt.run_status(_run_dir(tmp_path, "w", checkpoint=None)) == "restart"  # weights only


def test_training_room_counts_saved_output_and_free_space_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    _write(runs / "data-full" / "x", b"d" * 1000)
    monkeypatch.setattr(rt, "FINAL_BYTES", {**rt.FINAL_BYTES, "modernbert_base": 5000})
    monkeypatch.setattr(rt, "TRAIN_FREE_GIB", 0)
    rt.require_training_room(runs, "modernbert_base", limit_bytes=6000)
    with pytest.raises(RuntimeError, match="above the saved-output limit"):
        rt.require_training_room(runs, "modernbert_base", limit_bytes=5999)
    # A resumed run already counts with what it holds: only the rest is added.
    _run_dir(runs, "modernbert_base", weights=2000)  # ~4 kB already present
    rt.require_training_room(runs, "modernbert_base", limit_bytes=6050)
    # A complete run is not checked at all; the filesystem check is separate.
    _write(runs / "big", b"x" * 10_000)
    with pytest.raises(RuntimeError, match="above the saved-output limit"):
        rt.require_training_room(runs, "modernbert_base", limit_bytes=6050)
    _run_dir(runs, "deberta_v3_base", status="complete")
    rt.require_training_room(runs, "deberta_v3_base", limit_bytes=1)
    monkeypatch.setattr(rt, "TRAIN_FREE_GIB", 10**9)
    with pytest.raises(RuntimeError, match="nothing is deleted automatically"):
        rt.require_training_room(runs, "deberta_v3_base_unseen4")
    assert (runs / "big").exists()


def test_output_report_separates_budget_and_filesystem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    _run_dir(runs, "deberta_v3_base", status="complete")
    _write(runs / "logs" / "a.log", "x")
    report = rt.output_report(runs, limit_bytes=10**6, scratch=tmp_path)
    used, files = rt.tree_size(runs)
    assert report["output_bytes"] == used > 0
    assert report["output_files"] == files == 5
    assert report["output_remaining_bytes"] == 10**6 - used
    assert 0 < report["working_fs_free_bytes"] <= shutil.disk_usage(runs).total
    assert report["scratch_shares_working_fs"] is True
    assert report["entries"]["deberta_v3_base"]["run_status"] == "complete"
    assert report["entries"]["logs"]["run_status"] is None
    shown = capsys.readouterr().out
    assert "saved output" in shown
    assert "actually free" in shown


# ----------------------------------------------------------------------------- import


def _previous_output(root: Path) -> Path:
    prev = root / "compliance-runs"
    _write(prev / "data-full" / "manifest.json", json.dumps({"data_version": DATA_VERSION}))
    _write(
        prev / "baselines" / "classic_baselines.json", json.dumps({"data_version": DATA_VERSION})
    )
    _write(prev / "llm-cache" / "v" / "dev.jsonl", "1\n")
    _write(prev / "logs" / "old.log", "never imported")
    _write(prev / "results-bundle-x.tar", b"never imported")
    _run_dir(prev, "deberta_v3_base", status="complete")
    _run_dir(prev, "modernbert_base")  # unfinished with a complete checkpoint
    _run_dir(prev, "deberta_v3_base_unseen4", checkpoint=None)  # weights only
    return root


def test_import_copies_only_verified_reusable_parts_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rt, "free_bytes", lambda path: 10**12)  # independent of the host disk
    source = _previous_output(tmp_path / "input" / "previous-version")
    before = _tree(source)
    runs = tmp_path / "working" / "compliance-runs"
    _write(runs / "baselines" / "mine.json", "kept")  # an existing entry is never replaced
    outcome = rt.import_previous_output(source, runs, training_fingerprint=FINGERPRINT)
    assert _tree(source) == before  # the input is only read
    assert outcome["data-full"].startswith("imported")
    assert outcome["llm-cache"].startswith("imported")
    assert outcome["baselines"].startswith("kept")
    assert _tree(runs / "baselines") == {"mine.json": b"kept"}
    assert outcome["deberta_v3_base"].startswith("imported: complete run")
    assert not (runs / "deberta_v3_base" / "checkpoints" / "last").exists()  # not needed
    assert (runs / "deberta_v3_base" / "checkpoints" / "best" / "model.safetensors").is_file()
    assert rt.run_status(runs / "deberta_v3_base") == "complete"
    assert outcome["modernbert_base"].startswith("imported: unfinished run")
    assert rt.run_status(runs / "modernbert_base") == "resumable"
    assert _tree(runs / "modernbert_base") == _tree(source / "compliance-runs" / "modernbert_base")
    assert "NOT resumable" in outcome["deberta_v3_base_unseen4"]
    assert not (runs / "deberta_v3_base_unseen4").exists()  # starts fresh, never warm-started
    assert not (runs / "logs").exists()
    assert not list(runs.glob("*.tar"))
    assert not list(runs.glob(".staging-*"))

    # Importing again changes nothing: everything present is kept.
    after = _tree(runs)
    again = rt.import_previous_output(source, runs, training_fingerprint=FINGERPRINT)
    assert _tree(runs) == after
    assert all(v.startswith(("kept", "skipped")) for v in again.values())


def test_import_refuses_other_code_other_data_and_over_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rt, "free_bytes", lambda path: 0)
    with pytest.raises(RuntimeError, match="Nothing was copied"):
        rt.import_previous_output(
            _previous_output(tmp_path / "low"), tmp_path / "runs0", training_fingerprint=FINGERPRINT
        )
    assert not any(p.is_file() for p in (tmp_path / "runs0").rglob("*"))
    monkeypatch.setattr(rt, "free_bytes", lambda path: 10**12)
    source = _previous_output(tmp_path / "prev")
    runs = tmp_path / "runs"
    other = rt.import_previous_output(source, runs, training_fingerprint="0" * 64)
    assert "different training code" in other["deberta_v3_base"]
    assert "different training code" in other["modernbert_base"]
    assert not (runs / "deberta_v3_base").exists()

    runs2 = tmp_path / "runs2"
    _write(runs2 / "data-full" / "manifest.json", json.dumps({"data_version": "1" * 64}))
    mismatch = rt.import_previous_output(source, runs2, training_fingerprint=FINGERPRINT)
    assert "data version differs" in mismatch["deberta_v3_base"]
    assert "data version differs" in mismatch["baselines"]
    assert mismatch["data-full"].startswith("kept")

    runs3 = tmp_path / "runs3"
    with pytest.raises(RuntimeError, match="above the saved-output limit"):
        rt.import_previous_output(source, runs3, training_fingerprint=FINGERPRINT, limit_bytes=10)
    assert not any(p.is_file() for p in runs3.rglob("*"))  # checked before copying anything

    with pytest.raises(RuntimeError, match="overlaps RUNS"):
        rt.import_previous_output(
            source, source / "compliance-runs" / "x", training_fingerprint=FINGERPRINT
        )
    (source / "compliance-runs" / "modernbert_base" / "link").symlink_to(tempfile.gettempdir())
    with pytest.raises(RuntimeError, match="not a regular file"):
        rt.import_previous_output(source, tmp_path / "runs4", training_fingerprint=FINGERPRINT)
    assert not any(p.is_file() for p in (tmp_path / "runs4").rglob("*"))


def test_runtime_module_is_standard_library_only() -> None:
    tree = ast.parse(RUNTIME.read_text())
    roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert roots <= set(sys.stdlib_module_names) | {"__future__"}, roots
    assert pathlib.Path(rt.__file__) == RUNTIME
