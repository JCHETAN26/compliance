"""Re-serialize an existing processed dataset with the deterministic writer (no re-embedding).

    uv run --frozen python -m ml.data.reserialize --data-dir .cache/ml/data/full --verify-ranking dev

Why: early processed files were written by a gzip writer that embedded the random
temporary file name in the gzip header, so equal content did not give equal file
hashes. This tool keeps the expensive dense rankings and every row unchanged:

1. verifies the old files against the old manifest;
2. decodes every file and rewrites it with ``ml.common.io.write_jsonl_gz`` (fixed
   header) into a staging directory, and checks the decoded rows are identical;
3. independently rebuilds documents/segments/pairs from the pinned archive with the
   current code and checks they equal the stored rows (all splits);
4. optionally re-embeds one split (``--verify-ranking``) with the pinned encoder and
   checks the stored ranking is reproduced exactly;
5. writes a new manifest (new file hashes and ``data_version``) that keeps the original
   manifest, including its provenance, and records this conversion; the old directory
   is kept as ``<name>.pre-reserialize-<old version>``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from compliance_core.config import Settings
from compliance_ingestion.embeddings import ensure_model, spec_from_settings
from compliance_seed.job import ensure_dataset
from ml.common import provenance
from ml.common.io import RESULTS, read_json, read_jsonl_gz, write_json, write_jsonl_gz
from ml.common.provenance import file_sha256
from ml.data.contractnli import SPLITS, build_split
from ml.data.prepare import DeviceEmbedder, rank_segments
from ml.data.store import document_to_row, pair_to_row, verify_files


def log(message: str) -> None:
    print(message, flush=True)  # noqa: T201


def _normalise(rows: list[Any]) -> list[Any]:
    import json

    return [json.loads(json.dumps(r)) for r in rows]  # tuples -> lists, as stored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--verify-ranking", choices=SPLITS, default=None)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args(argv)
    root: Path = args.data_dir
    old = read_json(root / "manifest.json")
    verify_files(root, old)
    log(f"old files verified (data_version {old['data_version'][:16]})")
    staging = root.with_name(root.name + ".reserialize-staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    files: dict[str, str] = {}
    for name in old["files"]:
        if name.endswith(".jsonl.gz"):
            rows = list(read_jsonl_gz(root / name))
            write_jsonl_gz(staging / name, rows)
            if list(read_jsonl_gz(staging / name)) != rows:
                raise SystemExit(f"{name}: rows changed on rewrite")
        else:
            write_json(staging / name, read_json(root / name))
        files[name] = file_sha256(staging / name)
    log("rewrote every file; decoded rows identical")

    settings = Settings(artifact_cache_dir=provenance.REPO / ".cache", artifact_auto_download=False)
    archive = ensure_dataset(settings)
    rebuilt = {}
    for split in SPLITS:
        stored_docs = list(read_jsonl_gz(root / f"documents_{split}.jsonl.gz"))
        ids = [d["doc_id"] for d in stored_docs]
        sd = build_split(archive, split, ids if old.get("subset") else None)
        if _normalise([document_to_row(d) for d in sd.documents.values()]) != stored_docs:
            raise SystemExit(f"{split}: stored documents differ from a fresh rebuild")
        if _normalise([pair_to_row(p) for p in sd.pairs]) != list(
            read_jsonl_gz(root / f"pairs_{split}.jsonl.gz")
        ):
            raise SystemExit(f"{split}: stored pairs differ from a fresh rebuild")
        rebuilt[split] = sd
    log("documents/segments/pairs equal a fresh rebuild from the pinned archive (all splits)")

    ranking_check: dict[str, Any] = {"performed": False}
    if args.verify_ranking:
        split = args.verify_ranking
        encoder = ensure_model(
            settings.model_cache_dir, spec_from_settings(settings), allow_download=False
        )
        embedder = DeviceEmbedder(
            encoder, device=old["encoder"].get("device", "cpu"), threads=args.threads, batch_size=64
        )
        fresh = rank_segments(
            {split: rebuilt[split]}, embedder, settings.embedding_query_instruction
        )[split]
        stored = {
            r["pair_id"]: (r["segments"], r["scores"])
            for r in read_jsonl_gz(root / f"retrieval_{split}.jsonl.gz")
        }
        same_order = sum(fresh[pid][0] == stored[pid][0] for pid in stored)
        max_score_diff = max(
            abs(a - b) for pid in stored for a, b in zip(fresh[pid][1], stored[pid][1], strict=True)
        )
        ranking_check = {
            "performed": True,
            "split": split,
            "pairs": len(stored),
            "identical_rankings": same_order,
            "max_abs_score_difference": max_score_diff,
        }
        log(f"re-embedded {split}: {ranking_check}")
        if same_order != len(stored):
            raise SystemExit("stored ranking is not reproduced; refusing to re-label it")

    new_version = provenance.config_hash(files)
    manifest = {
        **{k: v for k, v in old.items() if k not in ("files", "data_version", "provenance")},
        "data_version": new_version,
        "files": files,
        "provenance": old["provenance"],
        "original_preparation_note": (
            "provenance.code_fingerprint above was computed when the original preparation "
            "finished; other files changed while it ran (the gzip writer among them). The "
            "row content is verified by this re-serialization instead."
        ),
        "reserialization": {
            "reason": "gzip headers embedded random temporary file names (non-deterministic hashes)",
            "old_data_version": old["data_version"],
            "old_files": old["files"],
            "rows_identical": True,
            "documents_and_pairs_equal_fresh_rebuild": True,
            "ranking_reverification": ranking_check,
            "provenance": provenance.collect(sys.argv, seed=None),
        },
    }
    write_json(staging / "manifest.json", manifest)
    backup = root.with_name(f"{root.name}.pre-reserialize-{old['data_version'][:16]}")
    shutil.rmtree(backup, ignore_errors=True)
    root.rename(backup)
    staging.rename(root)
    log(f"new data_version {new_version[:16]}; old directory kept at {backup}")
    report = RESULTS / "data_report.json"
    if old.get("name") == "full" and report.is_file():
        data = read_json(report)
        if data.get("data_dir_version") == old["data_version"]:
            data["data_dir_version"] = new_version
            data["reserialized_from_data_version"] = old["data_version"]
            write_json(report, data)
            log(f"updated {report} to the new data_version (content unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
