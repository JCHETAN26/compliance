"""Results bundle between Colab and this repository (verifiable, no secrets, no weights).

On Colab (after training/evaluation/export)::

    python -m ml.colab.bundle pack --out /content/drive/MyDrive/compliance-runs/bundle.tar \
        --data .cache/ml/data/full --run deberta_v3_base=<dir> --run modernbert_base=<dir> \
        --run deberta_v3_base_unseen4=<dir> --baselines .cache/ml/baselines/<version> \
        --results ml/eval/results --artifact <artifact dir>

Locally (after downloading the .tar from Drive)::

    make import-colab-bundle BUNDLE=~/Downloads/bundle.tar

The bundle holds: the processed dataset directory the runs used, each run's record,
config, logs, tokenizer and DEV/TEST predictions (no checkpoints: training weights stay
on Drive), the baseline predictions, the evaluation results/plots, and the int8 ONNX
artifact (the fp32 export stays on Drive). ``BUNDLE_MANIFEST.json`` lists the SHA-256 and
size of every file and the Colab checkout's git commit/dirty state.

Import refuses unsafe archive members (absolute paths, ``..``, links, devices), extracts
into a staging directory on the same filesystem, verifies every hash, then installs by
renames (no duplicate copies). A local dataset with a different ``data_version`` is kept
aside, never overwritten. The artifact is installed under ``.cache/models/rule-check/``
and ``current`` is pointed at it only after ``load_artifact`` verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compliance_model_service.rule_check.artifact import load_artifact
from ml.common import provenance
from ml.common.io import CACHE, RESULTS, read_json, write_json

MANIFEST = "BUNDLE_MANIFEST.json"
RUN_FILES = (
    "run.json",
    "config.yaml",
    "dev_history.json",
    "train_log.jsonl",
    "tokenizer.json",
    "predictions_dev.jsonl.gz",
    "predictions_test.jsonl.gz",
)
ARTIFACT_FILES = ("manifest.json", "tokenizer.json", "model.int8.onnx", "export_report.json")
MODELS = provenance.REPO / ".cache" / "models" / "rule-check"


def log(message: str) -> None:
    print(message, flush=True)  # noqa: T201


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------------------- pack


def pack(args: argparse.Namespace) -> int:
    members: dict[str, Path] = {}

    def add_dir(prefix: str, root: Path, names: tuple[str, ...] | None = None) -> None:
        for path in sorted(root.iterdir()):
            if path.is_file() and (names is None or path.name in names):
                members[f"{prefix}/{path.name}"] = path

    add_dir("data", args.data)
    for spec in args.run:
        name, _, directory = spec.partition("=")
        run_dir = Path(directory)
        if read_json(run_dir / "run.json").get("status") != "complete":
            raise SystemExit(f"{run_dir} is not a complete run")
        add_dir(f"runs/{name}", run_dir, RUN_FILES)
    if args.baselines:
        add_dir("baselines", args.baselines)
    add_dir("results", args.results)
    if args.artifact:
        add_dir("artifact", args.artifact, ARTIFACT_FILES)
    manifest: dict[str, Any] = {
        "format": 1,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": provenance.git_state(),
        "code_fingerprint": provenance.code_fingerprint(),
        "files": {
            name: {"sha256": _sha(path), "bytes": path.stat().st_size}
            for name, path in members.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".partial")
    with tarfile.open(tmp, "w") as tar:
        data = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo(MANIFEST)
        info.size = len(data)
        info.mtime = 0
        import io

        tar.addfile(info, io.BytesIO(data))
        for name, path in members.items():
            tar.add(path, arcname=name, recursive=False)
    os.replace(tmp, args.out)
    total = sum(f["bytes"] for f in manifest["files"].values())
    log(f"wrote {args.out}: {len(members)} files, {total / 1e6:.1f} MB")
    return 0


# ----------------------------------------------------------------------------- import

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DATA_NAMES = frozenset({"full", "smoke"})
RESULT_SUFFIXES = frozenset({".json", ".png"})
MAX_MEMBERS = 2_000
MAX_BUNDLE_BYTES = 1_200_000_000  # int8 DeBERTa-base artifact + data + predictions, generously
FREE_SPACE_MARGIN = 300_000_000


class BundleError(SystemExit):
    pass


def _check_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = tar.getmembers()
    if len(members) > MAX_MEMBERS:
        raise BundleError(f"bundle has {len(members)} members; the limit is {MAX_MEMBERS}")
    total = 0
    for member in members:
        name = member.name
        if not (member.isfile() or member.isdir()):
            raise BundleError(f"refusing non-regular archive member {name!r}")
        if name.startswith("/") or ".." in Path(name).parts:
            raise BundleError(f"refusing unsafe archive path {name!r}")
        total += member.size
    if total > MAX_BUNDLE_BYTES:
        raise BundleError(f"bundle expands to {total} bytes; the limit is {MAX_BUNDLE_BYTES}")
    return members


def _safe(name: str, what: str) -> str:
    if not SAFE_NAME.fullmatch(name):
        raise BundleError(f"unsafe {what} {name!r} in the bundle")
    return name


RESERVED_VERSIONS = frozenset({"current"})


def _artifact_name(version: str) -> str:
    _safe(version, "model version")
    if version in RESERVED_VERSIONS or version.startswith(".current"):
        raise BundleError(f"model version {version!r} is reserved")
    return version


def _inside(base: Path, path: Path) -> Path:
    if not path.resolve().is_relative_to(base.resolve()):
        raise BundleError(f"{path} would leave {base}")
    return path


def _backup_name(path: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    candidate = path.with_name(f"{path.name}.backup-{stamp}-{os.getpid()}")
    n = 0
    while candidate.exists():
        n += 1
        candidate = path.with_name(f"{path.name}.backup-{stamp}-{os.getpid()}-{n}")
    return candidate


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + _sha(path).encode())
    return digest.hexdigest()


def install_artifact(source: Path, *, move: bool) -> Path:
    """Verify an artifact and make it ``.cache/models/rule-check/current``.

    The target name comes from the verified manifest (a safe file-name pattern) and must
    stay inside the models directory. An existing different artifact of the same version
    is kept as a backup, never deleted.
    """
    artifact = load_artifact(source, variant="int8", allow_smoke=True)
    manifest = artifact.manifest
    target = MODELS / _artifact_name(manifest.model_version)
    src, dst = source.resolve(), target.resolve()
    if src != dst and (dst.is_relative_to(src) or src.is_relative_to(dst)):
        raise BundleError(f"refusing to install {source} into its own ancestor/descendant {target}")
    MODELS.mkdir(parents=True, exist_ok=True)
    _inside(MODELS, target)
    if target.resolve() != source.resolve():
        if target.exists() and _tree_digest(target) == _tree_digest(source):
            log(f"{target} already holds this artifact")
            if move:
                shutil.rmtree(source)  # the staged duplicate only
        else:
            if target.exists():
                target.rename(_backup_name(target))
            if move:
                source.rename(target)
            else:
                shutil.copytree(source, target)
    target.chmod(0o755)
    for path in target.iterdir():
        path.chmod(0o644)  # read by the model-service container's own uid
    load_artifact(target, variant="int8", allow_smoke=True)
    link = MODELS / "current"
    tmp_link = MODELS / f".current-new-{os.getpid()}"
    tmp_link.unlink(missing_ok=True)
    tmp_link.symlink_to(target.name)  # relative: also valid inside the container mount
    os.replace(tmp_link, link)
    note = (
        " (SMOKE artifact: the service refuses it unless RULE_CHECK_ALLOW_SMOKE=true)"
        if manifest.kind == "smoke"
        else ""
    )
    log(f"installed {manifest.model_version} ({manifest.kind}) as {link}{note}")
    return target


def _preflight(staging: Path) -> dict[str, Any]:
    """Validate every staged component and destination; nothing is changed here."""
    from ml.data.store import verify_files

    data_manifest = read_json(staging / "data" / "manifest.json")
    name = data_manifest.get("name")
    version = str(data_manifest.get("data_version", ""))
    if name not in DATA_NAMES or not HEX64.fullmatch(version):
        raise BundleError(f"invalid dataset identity {name!r} / {version!r}")
    verify_files(staging / "data", data_manifest)
    plan: dict[str, Any] = {
        "data_name": name,
        "data_version": version,
        "runs": [],
        "baselines": [],
        "results": [],
    }
    plan["data_target"] = _inside(CACHE, CACHE / "data" / name)
    plan["staged_data"] = staging / "data"
    for run in sorted((staging / "runs").iterdir()) if (staging / "runs").is_dir() else []:
        _safe(run.name, "run name")
        record = read_json(run / "run.json")
        if record.get("status") != "complete" or record.get("data_version") != version:
            raise BundleError(f"run {run.name} is incomplete or used another data version")
        if {p.name for p in run.iterdir()} - set(RUN_FILES):
            raise BundleError(f"run {run.name} has unexpected files")
        plan["runs"].append((run, _inside(CACHE, CACHE / "runs" / run.name)))
    if (staging / "baselines").is_dir():
        target = _inside(CACHE, CACHE / "baselines" / version[:16])
        for path in sorted((staging / "baselines").iterdir()):
            _safe(path.name, "baseline file")
            if path.name.endswith(".meta.json") and read_json(path).get("data_version") != version:
                raise BundleError(f"{path.name} used another data version")
            plan["baselines"].append((path, target / path.name))
    for path in sorted((staging / "results").iterdir()):
        _safe(path.name, "result file")
        if path.suffix not in RESULT_SUFFIXES:
            raise BundleError(f"unexpected result file {path.name}")
        plan["results"].append((path, _inside(RESULTS, RESULTS / path.name)))
    if (staging / "artifact").is_dir():
        artifact = load_artifact(staging / "artifact", variant="int8", allow_smoke=True)
        _artifact_name(artifact.manifest.model_version)
        if artifact.manifest.provenance.get("data_version") != version:
            raise BundleError("the artifact was trained on another data version")
        plan["artifact"] = staging / "artifact"
    return plan


def _apply(plan: dict[str, Any], replaced: Path) -> dict[str, Any]:
    installed: dict[str, Any] = {}
    target: Path = plan["data_target"]
    staged_data: Path = plan["staged_data"]
    if (
        target.exists()
        and read_json(target / "manifest.json").get("data_version") == plan["data_version"]
    ):
        installed["data"] = {
            "dir": str(target),
            "data_version": plan["data_version"],
            "reused": True,
        }
    else:
        if target.exists():
            aside = _backup_name(target)
            target.rename(aside)
            log(f"local data with another data_version kept at {aside}")
        target.parent.mkdir(parents=True, exist_ok=True)
        staged_data.rename(target)
        installed["data"] = {
            "dir": str(target),
            "data_version": plan["data_version"],
            "reused": False,
        }
    installed["runs"] = {}
    for run, dest in plan["runs"]:
        if dest.exists():
            dest.rename(_backup_name(dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        run.rename(dest)
        installed["runs"][run.name] = str(dest)
    conflicts = []
    for path, dest in plan["baselines"]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and _sha(dest) != _sha(path):
            conflicts.append(path.name)  # keep the local file, report it
            continue
        path.rename(dest)
    if plan["baselines"]:
        installed["baselines"] = {"kept_local_on_conflict": conflicts}
    RESULTS.mkdir(parents=True, exist_ok=True)
    changed = []
    for path, dest in plan["results"]:
        if dest.exists() and _sha(dest) != _sha(path):
            replaced.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, replaced / dest.name)  # previous version kept outside git
        if not dest.exists() or _sha(dest) != _sha(path):
            changed.append(path.name)
        path.rename(dest)
    installed["results_written"] = changed
    installed["replaced_results_backup"] = str(replaced) if replaced.exists() else None
    if "artifact" in plan:
        installed["artifact"] = str(install_artifact(plan["artifact"], move=True))
    return installed


def import_bundle(args: argparse.Namespace) -> int:
    bundle: Path = args.bundle
    imports = CACHE / "imports"
    imports.mkdir(parents=True, exist_ok=True)
    staging = imports / f".staging-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        with tarfile.open(bundle, "r:") as tar:
            members = _check_members(tar)
            needed = sum(m.size for m in members) + FREE_SPACE_MARGIN
            free = shutil.disk_usage(imports).free
            if free < needed:
                raise BundleError(f"not enough disk: {free} bytes free, {needed} needed")
            tar.extractall(staging, members=members, filter="data")
        manifest = read_json(staging / MANIFEST)
        on_disk = {p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file()} - {
            MANIFEST
        }
        if on_disk != set(manifest["files"]):
            raise BundleError("bundle contents do not match its manifest")
        for name, entry in manifest["files"].items():
            path = staging / name
            if path.stat().st_size != entry["bytes"] or _sha(path) != entry["sha256"]:
                raise BundleError(f"{name} fails its SHA-256/size check")
        log(f"verified {len(on_disk)} files from {bundle.name} (source {manifest['source']})")
        plan = _preflight(staging)  # everything validated before anything is changed
        installed = _apply(plan, imports / f"replaced-{datetime.now(UTC):%Y%m%dT%H%M%S}")
        record = {
            "bundle": str(bundle),
            "bundle_sha256": _sha(bundle),
            "imported_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "bundle_manifest": manifest,
            "installed": installed,
            "provenance": provenance.collect(sys.argv, seed=None),
        }
        write_json(RESULTS / "colab_import.json", record)
        log(
            f"import complete; record in {RESULTS / 'colab_import.json'}. The downloaded "
            f"bundle ({bundle.stat().st_size / 1e6:.0f} MB) can now be deleted to save space."
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Colab results bundle: pack / import / install-artifact"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", action="append", default=[], help="name=run_dir (repeatable)")
    p.add_argument("--baselines", type=Path, default=None)
    p.add_argument("--results", type=Path, default=RESULTS)
    p.add_argument("--artifact", type=Path, default=None)
    i = sub.add_parser("import")
    i.add_argument("--bundle", type=Path, required=True)
    a = sub.add_parser("install-artifact")
    a.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "pack":
        return pack(args)
    if args.command == "import":
        return import_bundle(args)
    install_artifact(args.artifact_dir, move=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
