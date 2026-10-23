"""Runtime helpers for ``kaggle_train.ipynb`` (training the rule-check model on a Kaggle GPU).

Standard library only, and compatible with Python 3.10+: the notebook loads this file from the
verified source checkout into Kaggle's own kernel, whose interpreter is not the project's
(the project's Python 3.11 environment is a separate ``.venv`` that runs every pipeline step as
a child process). Nothing here trains, evaluates or selects anything; it runs the repository's
scripts, shows and logs their output, budgets the saved output and copies files with
verification. See ``ml/notebooks/KAGGLE.md``.

Layout on Kaggle (enforced by the notebook):

- ``/kaggle/input/...``: read-only inputs (the source bundle dataset, optionally a previous
  version's saved output). Never written.
- ``RUNS = /kaggle/working/compliance-runs``: everything needed to resume or report (runs,
  prepared data, baselines, logs, exports, results). ``/kaggle/working`` is what a saved
  version keeps, up to Kaggle's output limit (20 GB per the Kaggle Notebooks documentation).
- scratch (``/kaggle/temp`` if present, otherwise ``/tmp``): source checkout, ``.venv``, uv/pip
  and model caches. Not saved.

Nothing here deletes anything outside scratch; existing outputs are never overwritten.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

GIB = 1024**3
GB = 1000**3
# Kaggle saves up to 20 GB of /kaggle/working; read as decimal GB (the stricter reading).
OUTPUT_LIMIT_BYTES = 20 * GB
# After an interrupt: seconds to wait after SIGINT, then after SIGTERM, before SIGKILL.
STOP_GRACE_S = (30.0, 10.0)
RUN_NAMES = ("deberta_v3_base", "modernbert_base", "deberta_v3_base_unseen4")
# Free space a training run needs on the working filesystem while it runs: checkpoints/last
# plus its staging copy while promoting (2 x 2.2 GB), best, reserve (as on Colab).
TRAIN_FREE_GIB = 10
# Expected size each step leaves in RUNS (ml/notebooks/README.md space plan: best is measured,
# the rest are estimates from parameter counts); the results bundle is bounded by
# ml.colab.bundle.MAX_BUNDLE_BYTES.
FINAL_BYTES = {
    "deberta_v3_base": int(3.0 * GB),
    "modernbert_base": int(2.4 * GB),
    "deberta_v3_base_unseen4": int(3.0 * GB),
    "export": int(1.0 * GB),
    "bundle": int(1.2 * GB),
}
# What a previous version's output may contribute to RUNS (never logs, exports or bundles).
IMPORTABLE = ("data-full", "baselines", "llm-cache", *RUN_NAMES)
CHECKPOINT_DIRS = ("last", ".last-staging", ".last-old")  # ml.train.train's names
COMPLETE = "COMPLETE"


def log(message: str) -> None:
    print(message, flush=True)  # noqa: T201 - the notebook's output


def _say(message: str) -> None:
    """log(), never raising: used while stopping a child, when the output itself may be broken."""
    with contextlib.suppress(Exception):
        log(message)


def stamp() -> str:
    # timezone.utc rather than datetime.UTC: Kaggle's kernel Python may predate 3.11.
    now = datetime.datetime.now(datetime.timezone.utc)  # noqa: UP017
    return now.strftime("%Y%m%dT%H%M%S%fZ")


# ----------------------------------------------------------------------------- commands


def stop_child(proc: subprocess.Popen[str]) -> None:
    """SIGINT, then SIGTERM, then SIGKILL to the child's process group until it has exited."""
    steps = (
        (signal.SIGINT, STOP_GRACE_S[0]),
        (signal.SIGTERM, STOP_GRACE_S[1]),
        (signal.SIGKILL, None),
    )
    for sig, timeout in steps:
        if proc.poll() is not None:
            break
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            continue  # a repeated interrupt escalates instead of abandoning the child
    proc.wait()
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)  # anything the child left behind in its group


def stream(
    cmd: Sequence[object], name: str, log_dir: Path, env: Mapping[str, str] | None = None
) -> Path:
    """Run cmd with its stdout and stderr shown live and appended to a new log in log_dir.

    Raises CalledProcessError on a non-zero exit. On an interrupt (the stop button), or if showing
    or logging the output fails (RuntimeError, chained), the child is stopped with stop_child and
    waited for before the error propagates, so no orphan keeps writing. Only the command line and
    the child's output are shown or logged, never the environment (secrets reach a child only
    through env).
    """
    argv = [str(c) for c in cmd]
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stamp()}-{name}.log"
    log("$ " + " ".join(argv))
    log(f"  log: {log_path}")
    proc = subprocess.Popen(  # noqa: S603 - the notebook's own fixed commands
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})},
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,  # an interrupt reaches the child only through stop_child
    )
    relay_error: list[BaseException] = []

    def pump() -> None:
        assert proc.stdout is not None  # noqa: S101 - stdout=PIPE above
        try:
            with log_path.open("a", encoding="utf-8") as fh:
                for line in proc.stdout:
                    print(line, end="", flush=True)  # noqa: T201
                    fh.write(line)
                    fh.flush()
        except Exception as exc:  # reported by the main thread
            relay_error.append(exc)
            with contextlib.suppress(Exception):
                for _ in proc.stdout:  # keep draining: the child must never block on a full pipe
                    pass

    def relay_failed() -> RuntimeError:
        return RuntimeError(f"showing/logging the output of {name} failed; log: {log_path}")

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        while True:
            if relay_error:
                raise relay_failed() from relay_error[0]
            try:
                code = proc.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException as exc:
        _say(f"\nstopping {name} (pid {proc.pid}) after {type(exc).__name__} ...")
        stop_child(proc)
        reader.join(timeout=30)
        _say(f"{name} stopped (exit {proc.returncode}); log: {log_path}")
        raise
    reader.join(timeout=30)
    if code == 0 and relay_error:
        raise relay_failed() from relay_error[0]
    if code != 0:
        log(f"FAILED: {name} exited with {code}; output in {log_path}")
        raise subprocess.CalledProcessError(code, argv)
    return log_path


def run_module(py: Path | str, log_dir: Path, module: str, *args: str, **env: str) -> Path:
    """A repository module with the project interpreter (see stream); never `uv run`/`make`."""
    return stream([py, "-m", module, *args], module, log_dir, {"PYTHONPATH": "", **env})


# ----------------------------------------------------------------------------- space


def tree_size(path: Path) -> tuple[int, int]:
    """(bytes, files) under path, without following symlinks; (0, 0) if it does not exist."""
    path = Path(path)
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size, 1
    if not path.is_dir():
        return 0, 0
    total = files = 0
    for root, dirs, names in os.walk(path):  # does not descend into symlinked directories
        for entry in (*names, *(d for d in dirs if Path(root, d).is_symlink())):
            total += Path(root, entry).lstat().st_size
            files += 1
    return total, files


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def require_free(path: Path, gib: float, what: str) -> None:
    """Stop (deleting nothing) unless path's filesystem has at least gib GiB free."""
    free = free_bytes(path) / GIB
    if free < gib:
        raise RuntimeError(
            f"{what}: {free:.1f} GiB free at {path}, {gib} GiB needed; free space deliberately "
            "(nothing is deleted automatically)"
        )
    log(f"{what}: {free:.1f} GiB free at {path} ({gib} GiB needed)")


def require_output_room(
    runs: Path, add_bytes: int, what: str, limit_bytes: int = OUTPUT_LIMIT_BYTES
) -> None:
    """Stop (deleting nothing) unless RUNS plus add_bytes stays within the saved-output limit."""
    used = tree_size(runs)[0]
    if used + add_bytes > limit_bytes:
        raise RuntimeError(
            f"{what}: RUNS would hold {(used + add_bytes) / GB:.2f} GB "
            f"({used / GB:.2f} GB now + {add_bytes / GB:.2f} GB), above the saved-output limit "
            f"of {limit_bytes / GB:.1f} GB. Nothing was deleted; see KAGGLE.md (space plan)."
        )


def run_status(run_dir: Path) -> str:
    """What ml.train.train will do with run_dir.

    absent | complete | resumable (a COMPLETE checkpoint exists) | restart (a run record but no
    checkpoint folders: training starts again from step 0) | blocked (only incomplete checkpoint
    folders: training refuses with CheckpointError).
    """
    run_dir = Path(run_dir)
    record = run_dir / "run.json"
    if not record.is_file():
        return "absent" if not run_dir.exists() else "restart"
    if json.loads(record.read_text(encoding="utf-8")).get("status") == "complete":
        return "complete"
    base = run_dir / "checkpoints"
    if any((base / d / COMPLETE).is_file() for d in CHECKPOINT_DIRS):
        return "resumable"
    if any((base / d).exists() for d in CHECKPOINT_DIRS):
        return "blocked"
    return "restart"


def require_training_room(runs: Path, name: str, limit_bytes: int = OUTPUT_LIMIT_BYTES) -> None:
    """Free working-filesystem space and saved-output budget for one training run.

    A resumed run already counts with its current size. Nothing is checked for a complete run
    (training returns immediately).
    """
    run_dir = Path(runs) / name
    if run_status(run_dir) == "complete":
        log(f"{name}: already complete")
        return
    require_free(runs, TRAIN_FREE_GIB, f"training {name}")
    current = tree_size(run_dir)[0]
    require_output_room(runs, max(0, FINAL_BYTES[name] - current), f"training {name}", limit_bytes)


def output_report(
    runs: Path, limit_bytes: int = OUTPUT_LIMIT_BYTES, scratch: Path | None = None
) -> dict[str, Any]:
    """Print and return the saved-output size (RUNS) against the limit, separately from the
    actual free space of the working filesystem (and of scratch)."""
    runs = Path(runs)
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted(runs.iterdir()) if runs.is_dir() else []:
        size, files = tree_size(path)
        status = run_status(path) if path.name in RUN_NAMES else None
        entries[path.name] = {"bytes": size, "files": files, "run_status": status}
    used, files = tree_size(runs)
    report: dict[str, Any] = {
        "runs": str(runs),
        "output_bytes": used,
        "output_files": files,
        "output_limit_bytes": limit_bytes,
        "output_remaining_bytes": limit_bytes - used,
        "working_fs_free_bytes": free_bytes(runs if runs.exists() else runs.parent),
        "entries": entries,
    }
    log(
        f"saved output (RUNS {runs}): {used / GB:.2f} GB in {files} files of a "
        f"{limit_bytes / GB:.1f} GB limit ({(limit_bytes - used) / GB:.2f} GB left)"
    )
    log(f"working filesystem: {report['working_fs_free_bytes'] / GIB:.1f} GiB actually free")
    if scratch is not None:
        report["scratch_fs_free_bytes"] = free_bytes(scratch)
        same = os.stat(scratch).st_dev == os.stat(runs if runs.exists() else runs.parent).st_dev
        report["scratch_shares_working_fs"] = same
        log(
            f"scratch {scratch}: {report['scratch_fs_free_bytes'] / GIB:.1f} GiB free"
            + (" (same filesystem as /kaggle/working)" if same else "")
        )
    for name, entry in entries.items():
        extra = f" [{entry['run_status']}]" if entry["run_status"] else ""
        log(f"  {name}: {entry['bytes'] / GB:.3f} GB, {entry['files']} files{extra}")
    return report


# ----------------------------------------------------------------------------- copies


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded(rel: PurePosixPath, exclude: Iterable[str]) -> bool:
    return any(rel == PurePosixPath(e) or PurePosixPath(e) in rel.parents for e in exclude)


def regular_files(src: Path, exclude: Iterable[str] = ()) -> list[PurePosixPath]:
    """Relative paths of the files under the directory src; links/special files are refused."""
    src = Path(src)
    if src.is_symlink() or not src.is_dir():
        raise RuntimeError(f"cannot copy {src}: not a directory (or a symlink)")
    exclude = tuple(exclude)
    files = []
    for path in sorted(src.rglob("*")):
        rel = PurePosixPath(path.relative_to(src).as_posix())
        if _excluded(rel, exclude):
            continue
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise RuntimeError(f"cannot copy {path}: not a regular file or directory")
        if path.is_file():
            files.append(rel)
    return files


def copy_verified(src: Path, dest: Path, exclude: Iterable[str] = ()) -> int:
    """Copy the directory src to the NEW path dest, every file SHA-256 verified; returns bytes.

    Refuses an existing dest (nothing is overwritten). The copy is assembled in a staging
    sibling and renamed into place only when complete and verified; a failed copy leaves only
    that clearly named staging folder.
    """
    src, dest = Path(src), Path(dest)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(f"{dest} already exists; it is never overwritten")
    files = regular_files(src, exclude)
    staging = dest.parent / f".staging-{dest.name}-{stamp()}"
    staging.mkdir(parents=True)
    total = 0
    for rel in files:
        target = staging / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, target)
        if sha256(target) != sha256(src / rel):
            raise RuntimeError(f"copy verification failed for {src / rel}; partial copy {staging}")
        total += target.stat().st_size
    staging.rename(dest)
    return total


def replace_scratch_copy(src: Path, dest: Path, scratch: Path) -> None:
    """Replace dest (which must lie inside scratch) by a verified copy of src."""
    dest = Path(dest)
    scratch_root = Path(scratch).resolve()
    if dest.resolve() == scratch_root or scratch_root not in dest.resolve().parents:
        raise RuntimeError(f"{dest} is not inside scratch {scratch_root}; refusing to replace it")
    shutil.rmtree(dest, ignore_errors=True)  # the scratch copy only
    dest.parent.mkdir(parents=True, exist_ok=True)
    copy_verified(src, dest)


def merge_append_only(src: Path, dest: Path) -> int:
    """Merge the append-only files of src into dest (the LLM response cache); returns writes.

    Per file: missing in dest -> copied; src extends dest -> dest extended; dest already
    extends src -> dest kept; anything else diverges and is refused. Every file is checked
    before anything is written; nothing is deleted.
    """
    src, dest = Path(src), Path(dest)
    updates: list[tuple[Path, bytes]] = []
    for rel in regular_files(src):
        new = (src / rel).read_bytes()
        target = dest / rel
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise RuntimeError(f"{target} is not a regular file")
        old = target.read_bytes() if target.exists() else None
        if old is None or (new != old and new.startswith(old)):
            updates.append((target, new))
        elif not old.startswith(new):
            raise RuntimeError(f"{target} diverges from {src / rel}; refusing to merge")
    for target, data in updates:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / f".{target.name}.{stamp()}.partial"
        tmp.write_bytes(data)
        os.replace(tmp, target)
    return len(updates)


def link_cache(link: Path, canonical: Path, scratch: Path) -> None:
    """Make link (a cache path inside scratch) a symlink to canonical (a real folder in RUNS).

    The pipeline's own cache path then writes straight into the saved output: every attempt a
    child flushes is in RUNS at once, not only after a later copy. An existing real folder at
    link (an earlier scratch copy) is first merged into canonical (merge_append_only) and then
    moved aside within scratch, never deleted. A symlink at link pointing anywhere else, a link
    outside scratch and a canonical path that is itself a symlink or a file are refused.
    """
    link, canonical = Path(link), Path(canonical).absolute()
    scratch_root = Path(scratch).resolve()
    parent = link.parent.resolve()  # where the link really lives
    if parent != scratch_root and scratch_root not in parent.parents:
        raise RuntimeError(f"{link} is not inside scratch {scratch_root}; refusing to link it")
    if canonical.is_symlink() or (canonical.exists() and not canonical.is_dir()):
        raise RuntimeError(f"{canonical} must be a real folder (not a symlink or a file)")
    if link.is_symlink():
        if Path(os.readlink(link)) != canonical:
            raise RuntimeError(f"{link} points to {os.readlink(link)}, not {canonical}; refusing")
        canonical.mkdir(parents=True, exist_ok=True)
        return
    if link.exists() and not link.is_dir():
        raise RuntimeError(f"{link} is not a folder; refusing to replace it")
    canonical.mkdir(parents=True, exist_ok=True)
    if link.exists():
        merged = merge_append_only(link, canonical)
        aside = link.parent / f"{link.name}.pre-link-{stamp()}"
        link.rename(aside)
        log(f"merged {merged} file(s) of {link} into {canonical}; old copy kept at {aside}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(canonical, target_is_directory=True)
    log(f"{link} -> {canonical} (written directly to the saved output)")


# ----------------------------------------------------------------------------- secrets


def _secret_help(name: str, problem: str) -> str:
    return (
        f"Kaggle secret {name} {problem}. In the notebook editor: Add-ons -> Secrets, add {name} "
        "and enable it for this notebook (the Kaggle CLI cannot create or attach secrets); or "
        "set RUN_LLM = False."
    )


def secret_env(name: str, get_secret: Callable[[str], object]) -> dict[str, str]:
    """{name: value} for a child process's environment, read with get_secret (Kaggle Secrets).

    A missing, empty or non-text secret is refused with setup instructions. The value, its
    length and the lookup error's text (which could carry request details) never appear in a
    message or an exception chain; only the error's type is named.
    """
    value: object = None
    failure = None
    try:
        value = get_secret(name)
    except Exception as exc:
        failure = type(exc).__name__
    if failure is None and isinstance(value, str) and value.strip():
        return {name: value}
    value = None
    raise RuntimeError(
        _secret_help(name, f"is not available ({failure})" if failure else "is empty or not text")
    )


def require_secret(name: str, get_secret: Callable[[str], object]) -> None:
    """Early check that the secret is configured; the value is dropped right away."""
    env = secret_env(name, get_secret)
    env.clear()
    log(f"Kaggle secret {name} is configured (value not shown; read again when needed)")


# ----------------------------------------------------------------------------- import


def _data_version(data_dir: Path) -> str | None:
    manifest = Path(data_dir) / "manifest.json"
    if not manifest.is_file():
        return None
    value = json.loads(manifest.read_text(encoding="utf-8")).get("data_version")
    return str(value) if value else None


def _run_import(
    src: Path, fingerprint: str, data_version: str | None
) -> tuple[bool, str, tuple[str, ...]]:
    """(import?, reason, excluded paths) for one previous run directory."""
    record_path = src / "run.json"
    if not record_path.is_file():
        return False, "no run.json", ()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("training_code_fingerprint") != fingerprint:
        return False, "trained by different training code (fingerprint differs)", ()
    if data_version is None or record.get("data_version") != data_version:
        return False, "data version differs from the prepared data (or no prepared data)", ()
    if record.get("status") == "complete":
        if not (src / "checkpoints" / "best" / "model.safetensors").is_file():
            return False, "complete, but checkpoints/best/model.safetensors is missing", ()
        # A complete run is never resumed: its resume checkpoint is not needed (2.2 GB).
        excluded = tuple(f"checkpoints/{d}" for d in CHECKPOINT_DIRS)
        return True, "complete run (without its resume checkpoint)", excluded
    if any((src / "checkpoints" / d / COMPLETE).is_file() for d in CHECKPOINT_DIRS):
        return True, "unfinished run with a complete resume checkpoint", ()
    return (
        False,
        "unfinished and NOT resumable (no complete checkpoint, e.g. weights only); "
        "this run starts fresh in RUNS",
        (),
    )


def _baselines_import(src: Path, data_version: str | None) -> tuple[bool, str]:
    report = src / "classic_baselines.json"
    if not report.is_file():
        return False, "no classic_baselines.json"
    if json.loads(report.read_text(encoding="utf-8")).get("data_version") != data_version:
        return False, "data version differs from the prepared data (or no prepared data)"
    return True, "classic baselines"


def import_previous_output(
    source: Path,
    runs: Path,
    *,
    training_fingerprint: str,
    limit_bytes: int = OUTPUT_LIMIT_BYTES,
) -> dict[str, str]:
    """Copy reusable parts of a previous version's output (read-only input) into RUNS.

    source is a previous ``compliance-runs`` folder (or a folder containing one). Only IMPORTABLE
    entries are considered; an entry already in RUNS is kept and never overwritten. Runs must
    have the current training-code fingerprint and the data version of the prepared data in use;
    an unfinished run is imported only with a complete resume checkpoint (weights alone are never
    resumed or warm-started). Everything is checked and the space budgeted before anything is
    copied; each copy is SHA-256 verified. The source is only read. Returns name -> outcome.
    """
    source, runs = Path(source), Path(runs)
    if (source / "compliance-runs").is_dir():
        source = source / "compliance-runs"
    if not source.is_dir():
        raise RuntimeError(f"previous output {source} is not a directory")
    src_r, runs_r = source.resolve(), runs.resolve()
    if src_r == runs_r or src_r in runs_r.parents or runs_r in src_r.parents:
        raise RuntimeError(f"previous output {source} overlaps RUNS {runs}; refusing")
    runs.mkdir(parents=True, exist_ok=True)

    data_version = _data_version(runs / "data-full")
    if data_version is None and not (runs / "data-full").exists():
        data_version = _data_version(source / "data-full")
    outcome: dict[str, str] = {}
    plan: list[tuple[str, tuple[str, ...]]] = []
    for name in IMPORTABLE:
        src = source / name
        if not (src.exists() or src.is_symlink()):
            continue
        if (runs / name).exists() or (runs / name).is_symlink():
            outcome[name] = "kept the existing entry in RUNS (not overwritten)"
            continue
        if src.is_symlink() or not src.is_dir():
            outcome[name] = "skipped: not a directory"
            continue
        excluded: tuple[str, ...] = ()
        if name == "data-full":
            ok, why = (_data_version(src) is not None), "prepared data"
        elif name == "baselines":
            ok, why = _baselines_import(src, data_version)
        elif name == "llm-cache":
            ok, why = True, "LLM response cache"
        else:
            ok, why, excluded = _run_import(src, training_fingerprint, data_version)
        if not ok:
            outcome[name] = f"skipped: {why}"
            continue
        regular_files(src, excluded)  # refuses links/special files before anything is copied
        plan.append((name, excluded))
        outcome[name] = f"imported: {why}"

    need = 0
    for name, excluded in plan:
        need += sum(
            (source / name / rel).stat().st_size for rel in regular_files(source / name, excluded)
        )
    if plan:
        if free_bytes(runs) < need + GIB:
            raise RuntimeError(
                f"importing needs {need / GIB:.2f} GiB + 1 GiB reserve; "
                f"{free_bytes(runs) / GIB:.2f} GiB free. Nothing was copied."
            )
        require_output_room(runs, need, "import of the previous output", limit_bytes)
    for name, excluded in plan:
        copy_verified(source / name, runs / name, excluded)
    for name, result in outcome.items():
        log(f"  {name}: {result}")
    if not outcome:
        log(f"  nothing importable in {source}")
    return outcome
