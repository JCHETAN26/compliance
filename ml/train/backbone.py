"""Pinned backbone + tokenizer fetching, verification and construction.

Files are downloaded at the pinned revision into ``.cache/ml/backbones/`` and every
pinned file's SHA-256 is checked (on download and on every use). The tokenizer is
materialised as a single ``tokenizer.json`` that the ``tokenizers`` library loads; for
DeBERTa-v3 that is a conversion of ``spm.model`` by ``transformers`` (needs the
``sentencepiece`` tool package), after which the converted ids are checked against the
``transformers`` tokenizer on sample text.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from compliance_core.downloads import sha256_file
from ml.common.io import CACHE
from ml.train.config import BackboneSpec, PinnedFile

PARITY_SAMPLES = (
    (
        "Receiving Party may share some Confidential Information with some third-parties.",
        "The Recipient shall not disclose any “Confidential Information” to any third party, "
        "except its attorneys, accountants and advisors (each, a “Representative”).",
    ),
    (
        "Some obligations of Agreement may survive termination of Agreement.",
        "Sections 3, 4.2(a) and 7 shall survive for five (5) years after termination.  § 9.",
    ),
)


class BackboneError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackboneFiles:
    root: Path
    tokenizer_json: Path


def _cache_dir(repo_id: str, revision: str, cache: Path) -> Path:
    return cache / "backbones" / f"{repo_id.replace('/', '--')}--{revision[:12]}"


def _verify(path: Path, pinned: PinnedFile) -> None:
    if not path.is_file():
        raise BackboneError(f"missing pinned file {path}")
    if pinned.bytes is not None and path.stat().st_size != pinned.bytes:
        raise BackboneError(f"{path} has {path.stat().st_size} bytes, pinned {pinned.bytes}")
    actual = sha256_file(path)
    if actual != pinned.sha256:
        raise BackboneError(f"{path} SHA-256 {actual} != pinned {pinned.sha256}")


def _pinned_files(spec: BackboneSpec, *, need_weights: bool) -> list[PinnedFile]:
    files = list(spec.tokenizer.files)
    weights = spec.weights if spec.kind == "pretrained" and need_weights else None
    for extra in (spec.config_file, weights):
        if extra is not None and all(f.name != extra.name for f in files):
            files.append(extra)
    return files


def fetch_backbone(
    spec: BackboneSpec,
    *,
    cache: Path = CACHE,
    allow_download: bool = True,
    need_weights: bool = True,
) -> BackboneFiles:
    """Verified local files; ``need_weights=False`` (export) skips the pretrained weights."""
    if spec.tokenizer.repo_id != spec.repo_id or spec.tokenizer.revision != spec.revision:
        raise BackboneError("tokenizer and backbone must come from the same pinned revision")
    root = _cache_dir(spec.repo_id, spec.revision, cache)
    files = _pinned_files(spec, need_weights=need_weights)
    missing = [f for f in files if not (root / f.name).is_file()]
    if missing:
        if not allow_download:
            raise BackboneError(f"{spec.repo_id}@{spec.revision} not cached in {root}")
        from compliance_core.downloads import download_verified

        # Plain HTTPS with the pinned SHA-256 and a size cap (independent of any
        # HF_HUB_OFFLINE flag that other components set process-wide).
        for pinned in missing:
            download_verified(
                f"https://huggingface.co/{spec.repo_id}/resolve/{spec.revision}/{pinned.name}",
                root / pinned.name,
                sha256=pinned.sha256,
                max_bytes=pinned.bytes or 2_000_000_000,
                timeout=120.0,
            )
    for pinned in files:
        _verify(root / pinned.name, pinned)
    return BackboneFiles(root=root, tokenizer_json=_tokenizer_json(spec, root))


TOKENIZER_META = "rulecheck-tokenizer.meta.json"


def _converter_identity() -> dict[str, str]:
    import tokenizers
    import transformers

    return {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__}


def _expected_meta(spec: BackboneSpec) -> dict[str, Any]:
    return {
        "source": spec.tokenizer.source,
        "repo_id": spec.tokenizer.repo_id,
        "revision": spec.tokenizer.revision,
        "source_files": {f.name: f.sha256 for f in spec.tokenizer.files},
        "converter": _converter_identity() if spec.tokenizer.source == "spm" else None,
    }


def _cached_tokenizer_ok(spec: BackboneSpec, target: Path, meta_path: Path) -> bool:
    if not (target.is_file() and meta_path.is_file()):
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    derived = meta.pop("derived_sha256", None)
    return bool(meta == _expected_meta(spec) and derived == sha256_file(target))


def _tokenizer_json(spec: BackboneSpec, root: Path) -> Path:
    """The verified derived ``tokenizer.json`` used by training, export and serving.

    A cached derivative is reused only if its sidecar records the same pinned source
    files, the same converter versions and its own current SHA-256; otherwise it is
    regenerated (and, for spm conversion, parity-checked) before being promoted.
    """
    target = root / "rulecheck-tokenizer.json"
    meta_path = root / TOKENIZER_META
    if _cached_tokenizer_ok(spec, target, meta_path):
        return target
    staging = root / ".tokenizer-staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    if spec.tokenizer.source == "tokenizer_json":
        shutil.copyfile(root / "tokenizer.json", staging / "tokenizer.json")
    else:
        from transformers import AutoTokenizer

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        hf = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
        if not getattr(hf, "is_fast", False):
            raise BackboneError("expected a fast (tokenizers-backed) tokenizer after conversion")
        hf.save_pretrained(str(staging))
        check_tokenizer_parity(hf, staging / "tokenizer.json")
    meta = {**_expected_meta(spec), "derived_sha256": sha256_file(staging / "tokenizer.json")}
    os.replace(staging / "tokenizer.json", target)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    shutil.rmtree(staging, ignore_errors=True)
    return target


def check_tokenizer_parity(hf_tokenizer: Any, tokenizer_json: Path) -> None:
    """The converted tokenizer.json must reproduce the transformers tokenizer exactly."""
    from tokenizers import Tokenizer

    fast = Tokenizer.from_file(str(tokenizer_json))
    for first, second in PARITY_SAMPLES:
        expected = hf_tokenizer(first, second)["input_ids"]
        actual = fast.encode(first, second).ids
        if list(expected) != list(actual):
            raise BackboneError(
                "converted tokenizer.json disagrees with the transformers tokenizer"
            )


def build_architecture(spec: BackboneSpec, files: BackboneFiles) -> Any:
    """The backbone architecture from its pinned config only (weights loaded afterwards)."""
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(str(files.root), local_files_only=True)
    for key, value in (spec.tiny_overrides or {}).items():
        setattr(config, key, value)
    return AutoModel.from_config(config)  # type: ignore[no-untyped-call]


def build_backbone(spec: BackboneSpec, files: BackboneFiles, *, seed: int) -> Any:
    """The backbone ``transformers`` model: pinned pretrained weights or seeded random tiny."""
    import torch
    from transformers import AutoConfig, AutoModel

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if spec.kind == "pretrained":
        return AutoModel.from_pretrained(str(files.root), local_files_only=True)
    config = AutoConfig.from_pretrained(str(files.root), local_files_only=True)
    for key, value in (spec.tiny_overrides or {}).items():
        setattr(config, key, value)
    torch.manual_seed(seed)
    return AutoModel.from_config(config)  # type: ignore[no-untyped-call]


def sentencepiece_fidelity(
    files: BackboneFiles, texts: list[str], normalization: Literal["NFKC"] | None
) -> dict[str, Any]:
    """How often the served tokenization differs from the original ``spm.model`` ids.

    Measured on TRAIN text only and recorded in the run; it is a fidelity statistic of
    the tokenizer conversion, not something any choice is tuned on.
    """
    import unicodedata

    import sentencepiece as spm  # type: ignore[import-not-found,unused-ignore]  # tool overlay
    from tokenizers import Tokenizer

    reference = spm.SentencePieceProcessor(model_file=str(files.root / "spm.model"))
    fast = Tokenizer.from_file(str(files.tokenizer_json))
    differ = 0
    for text in texts:
        prepared = unicodedata.normalize(normalization, text) if normalization else text
        differ += reference.encode(text) != fast.encode(prepared, add_special_tokens=False).ids
    return {
        "texts": len(texts),
        "differing_texts": differ,
        "rate": differ / len(texts) if texts else 0.0,
        "normalization": normalization,
        "note": "remaining differences: byte-fallback pieces for private-use glyphs",
    }
