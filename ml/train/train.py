"""Train the rule-check model; select on DEV only; predict TEST after selection is frozen.

    uv run python -m ml.train.train --config ml/train/configs/deberta_v3_base.yaml \
        --run-dir /content/drive/MyDrive/compliance-runs/deberta_v3_base

Run directory (resumable; re-running the same command continues an interrupted run)::

    config.yaml, run.json          config copy; provenance, selection and status
    train_log.jsonl                loss / lr every ``LOG_EVERY`` optimiser steps
    dev_history.json               DEV metrics per epoch and per candidate k
    checkpoints/last/              model + optimiser + scheduler + scaler + RNG + position
    checkpoints/best/              weights of the DEV-selected epoch
    predictions_dev.jsonl.gz       DEV predictions of the selected checkpoint (all k)
    predictions_test.jsonl.gz      TEST predictions (selected checkpoint and k only)
    tokenizer.json                 the exact tokenizer used

Selection is DEV-only: after every epoch the full inference pipeline (evidence head's own
top-k, no gold) runs on DEV; the (epoch, k) with the best DEV macro-F1 over the seen
hypotheses' pairs is kept (ties: earlier epoch, smaller k). Only TRAIN and DEV are loaded
for training and selection; TEST is loaded after training, once the selected weights and
k are frozen. Class weights for the label loss come from the label counts of the TRAIN
pairs actually used (seen hypotheses only). Production runs refuse subset data.

Optimiser steps: with fp16 a GradScaler may skip an update after an overflow; a skipped
update neither advances the LR schedule nor counts as a step (``skipped_updates``).

Checkpoints: ``checkpoints/last`` is written to a staging directory, marked ``COMPLETE``
last, then promoted by renames. If a session dies mid-promotion, the newest complete
checkpoint (``last`` > staging > ``.last-old``) is recovered; an incomplete layout is an
error, never a silent restart. A resume must match the run's data version, derived
tokenizer hash and training-code fingerprint; otherwise use a new run directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from compliance_model_service.rule_check.encoding import InputEncoder
from compliance_model_service.rule_check.labels import LABELS
from ml.common import provenance
from ml.common.io import read_json, write_json
from ml.data.contractnli import Pair, label_counts
from ml.data.prepare import CANDIDATE_DEPTH, inverse_frequency_weights
from ml.data.store import Dataset, load_dataset
from ml.eval.infer import TokenCache, TorchForward, predict_pairs, save_predictions
from ml.eval.metrics import macro_f1
from ml.train.backbone import (
    BackboneFiles,
    build_backbone,
    fetch_backbone,
    sentencepiece_fidelity,
)
from ml.train.config import TrainConfig, load_config
from ml.train.examples import EvidenceExample, LabelExample, build_epoch
from ml.train.model import RuleCheckModel

LOG_EVERY = 50


def log(message: str) -> None:
    print(message, flush=True)  # noqa: T201


class ConfigMismatchError(RuntimeError):
    pass


class ResumeIdentityError(RuntimeError):
    pass


class CheckpointError(RuntimeError):
    pass


class _StopForTest(Exception):  # noqa: N818 - test hook: simulated interruption
    pass


COMPLETE = "COMPLETE"


@dataclass
class Batch:
    kind: str  # "evidence" | "label"
    rows: list[list[int]]
    targets: list[float] | list[int]


def make_batches(
    evidence: list[EvidenceExample],
    labels: list[LabelExample],
    tokens: TokenCache,
    encoder: InputEncoder,
    cfg: TrainConfig,
    epoch: int,
) -> list[Batch]:
    rng = random.Random(f"{cfg.seed}:batches:{epoch}")  # noqa: S311 - reproducible order
    ev = evidence[:]
    lab = labels[:]
    rng.shuffle(ev)
    rng.shuffle(lab)
    batches: list[Batch] = []
    size = cfg.training.evidence_batch_size
    for i in range(0, len(ev), size):
        chunk = ev[i : i + size]
        rows = [
            encoder.evidence_input(
                tokens.hypotheses[e.key], tokens.segments("train", e.doc_id)[e.segment]
            ).ids
            for e in chunk
        ]
        batches.append(Batch("evidence", rows, [e.target for e in chunk]))
    size = cfg.training.label_batch_size
    for i in range(0, len(lab), size):
        chunk_l = lab[i : i + size]
        rows = [
            encoder.label_input(
                tokens.hypotheses[e.key],
                [tokens.segments("train", e.doc_id)[s] for s in e.segments],
            ).ids
            for e in chunk_l
        ]
        batches.append(Batch("label", rows, [e.target for e in chunk_l]))
    rng.shuffle(batches)
    return batches


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _param_groups(model: RuleCheckModel, cfg: TrainConfig) -> list[dict[str, Any]]:
    head_ids = {id(p) for p in model.head_parameters()}
    groups: dict[tuple[bool, bool], list[Any]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = id(param) in head_ids
        no_decay = param.ndim < 2 or "norm" in name.lower() or name.endswith(".bias")
        groups.setdefault((is_head, no_decay), []).append(param)
    t = cfg.training
    return [
        {
            "params": params,
            "lr": t.head_lr if is_head else t.lr,
            "weight_decay": 0.0 if no_decay else t.weight_decay,
        }
        for (is_head, no_decay), params in sorted(groups.items())
    ]


def _save_weights(model: RuleCheckModel, path: Path) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    state = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, str(tmp))
    os.replace(tmp, path)


def _load_weights(model: RuleCheckModel, path: Path) -> None:
    from safetensors.torch import load_file

    missing, unexpected = model.load_state_dict(load_file(str(path)), strict=True)
    if missing or unexpected:  # pragma: no cover - strict=True raises first
        raise RuntimeError(f"checkpoint mismatch: {missing} {unexpected}")


def _save_last(
    run_dir: Path,
    model: RuleCheckModel,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    state: dict[str, Any],
) -> None:
    import torch

    final = run_dir / "checkpoints" / "last"
    staging = run_dir / "checkpoints" / ".last-staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    _save_weights(model, staging / "model.safetensors")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        staging / "trainer.pt",
    )
    write_json(staging / "state.json", state)
    (staging / COMPLETE).write_text("ok\n")  # written last: the checkpoint is whole
    old = run_dir / "checkpoints" / ".last-old"
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        final.rename(old)
    staging.rename(final)
    shutil.rmtree(old, ignore_errors=True)


def recover_last(run_dir: Path) -> Path | None:
    """The newest complete ``last`` checkpoint, repairing an interrupted promotion."""
    base = run_dir / "checkpoints"
    final, staging, old = base / "last", base / ".last-staging", base / ".last-old"
    if (final / COMPLETE).is_file():
        chosen = final
    elif (staging / COMPLETE).is_file():
        chosen = staging
    elif (old / COMPLETE).is_file():
        chosen = old
    else:
        leftovers = [d.name for d in (final, staging, old) if d.exists()]
        if leftovers:
            raise CheckpointError(
                f"{base} has only incomplete checkpoints {leftovers}; refusing to restart silently"
            )
        return None
    if chosen != final:
        shutil.rmtree(final, ignore_errors=True)
        chosen.rename(final)
        log(f"recovered checkpoint from {chosen.name} after an interrupted promotion")
    for leftover in (staging, old):
        shutil.rmtree(leftover, ignore_errors=True)
    return final


def optimizer_update(
    model: Any, optimizer: Any, scheduler: Any, scaler: Any, max_grad_norm: float
) -> bool:
    """One (possibly skipped) optimiser update; True if the parameters were updated.

    GradScaler skips ``optimizer.step`` when it finds inf/NaN gradients and then lowers
    its scale, so a lower scale after ``update()`` identifies a skipped update.
    """
    import torch

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    before = scaler.get_scale() if scaler.is_enabled() else None
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    applied = before is None or scaler.get_scale() >= before
    if applied:
        scheduler.step()
    return bool(applied)


def _dev_scores(predictions: list[Any], selection_ids: set[str], ks: list[int]) -> dict[int, float]:
    chosen = [p for p in predictions if p.pair_id in selection_ids]
    gold = [p.gold for p in chosen]
    return {
        k: macro_f1(gold, [LABELS[int(np.argmax(p.label_logits[k]))] for p in chosen]) for k in ks
    }


def run(
    cfg: TrainConfig, run_dir: Path, config_path: Path, *, _stop_after_step: int | None = None
) -> dict[str, Any]:
    import torch
    from transformers import get_linear_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.requires_cuda and device != "cuda":
        raise SystemExit(
            f"config {cfg.run_name} requires a CUDA GPU (run it on Colab, see "
            "ml/notebooks/README.md); use ml/train/configs/smoke.yaml for the CPU smoke"
        )
    if device == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    run_dir.mkdir(parents=True, exist_ok=True)
    config_dump = cfg.model_dump(mode="json")
    config_hash = provenance.config_hash(config_dump)
    previous: dict[str, Any] | None = None
    if (run_dir / "run.json").exists():
        previous = read_json(run_dir / "run.json")
        if previous.get("config_hash") != config_hash:
            raise ConfigMismatchError(
                f"{run_dir} belongs to a run with a different config; use a new run directory"
            )
        if previous.get("status") == "complete":
            log(f"{run_dir} is already complete; nothing to do")
            return previous
    shutil.copyfile(config_path, run_dir / "config.yaml")
    _seed_everything(cfg.seed)

    data_dir = provenance.REPO / cfg.data.data_dir
    dataset: Dataset = load_dataset(
        data_dir, splits=("train", "dev")
    )  # TEST loaded after selection
    if cfg.kind == "production" and (
        dataset.manifest.get("name") != "full" or dataset.manifest.get("subset")
    ):
        raise SystemExit(
            f"production runs need the full official data, not {dataset.manifest.get('name')!r}"
        )
    held_out = set(cfg.data.held_out_hypotheses)
    unknown = held_out - set(dataset.hypotheses)
    if unknown:
        raise ValueError(f"unknown held-out hypotheses {sorted(unknown)}")
    train_pairs: list[Pair] = [p for p in dataset.pairs["train"] if p.key not in held_out]
    dev_pairs = dataset.pairs["dev"]
    selection_ids = {p.pair_id for p in dev_pairs if p.key not in held_out}
    counts = label_counts(train_pairs)
    weights = (
        inverse_frequency_weights(counts)
        if cfg.training.class_weighting == "inverse_frequency"
        else dict.fromkeys(LABELS, 1.0)
    )
    log(f"TRAIN pairs used {len(train_pairs)} {counts}; label loss weights {weights}")

    files: BackboneFiles = fetch_backbone(cfg.backbone)
    fidelity = None
    if cfg.backbone.tokenizer.source == "spm":
        texts = [
            d.segment_text(i)
            for d in dataset.documents["train"].values()
            for i in range(len(d.segments))
        ]
        fidelity = sentencepiece_fidelity(files, texts, cfg.encoder.normalization)
        log(f"tokenizer vs original SentencePiece on TRAIN segments: {fidelity}")
    tokenizer_sha = provenance.file_sha256(files.tokenizer_json)
    code_sha = provenance.training_code_fingerprint()
    if previous is not None:
        identity = {
            "data_version": (previous.get("data_version"), dataset.data_version),
            "tokenizer_sha256": (previous.get("tokenizer_sha256"), tokenizer_sha),
            "training_code_fingerprint": (previous.get("training_code_fingerprint"), code_sha),
        }
        changed = {k: v for k, v in identity.items() if v[0] != v[1]}
        if changed:
            raise ResumeIdentityError(
                f"cannot resume {run_dir}: {sorted(changed)} differ from the original run; "
                "use a new run directory"
            )
    shutil.copyfile(files.tokenizer_json, run_dir / "tokenizer.json")  # identity checked above
    encoder = InputEncoder.from_file(run_dir / "tokenizer.json", cfg.encoder.to_config())
    tokens = TokenCache(dataset, encoder)
    model = RuleCheckModel(
        build_backbone(cfg.backbone, files, seed=cfg.seed), dropout=cfg.training.dropout
    )
    model.to(device)

    label_index = {label: i for i, label in enumerate(LABELS)}
    ev0, lab0 = build_epoch(
        dataset, train_pairs, cfg.sampling, seed=cfg.seed, epoch=0, label_index=label_index
    )
    batches_per_epoch = math.ceil(len(ev0) / cfg.training.evidence_batch_size) + math.ceil(
        len(lab0) / cfg.training.label_batch_size
    )
    steps_per_epoch = math.ceil(batches_per_epoch / cfg.training.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.training.epochs
    if cfg.training.max_steps is not None:
        total_steps = min(total_steps, cfg.training.max_steps)
    optimizer = torch.optim.AdamW(_param_groups(model, cfg))
    scheduler = get_linear_schedule_with_warmup(  # type: ignore[no-untyped-call]
        optimizer, int(cfg.training.warmup_ratio * total_steps), total_steps
    )
    use_fp16 = cfg.training.fp16 and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    class_weight = torch.tensor(
        [weights[label] for label in LABELS], dtype=torch.float32, device=device
    )
    ce = torch.nn.CrossEntropyLoss(weight=class_weight)
    bce = torch.nn.BCEWithLogitsLoss()

    state: dict[str, Any] = {
        "epoch": 0,
        "batch": 0,
        "step": 0,
        "best": None,
        "history": [],
        "elapsed_s": 0.0,
        "skipped_updates": 0,
    }
    last = recover_last(run_dir)
    if last is not None:
        state = read_json(last / "state.json")
        _load_weights(model, last / "model.safetensors")
        trainer = torch.load(last / "trainer.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(trainer["optimizer"])
        scheduler.load_state_dict(trainer["scheduler"])
        scaler.load_state_dict(trainer["scaler"])
        torch.set_rng_state(trainer["torch_rng"])
        if trainer["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(trainer["cuda_rng"])
        log(f"resuming at epoch {state['epoch']} batch {state['batch']} step {state['step']}")

    run_record: dict[str, Any] = {
        "run_name": cfg.run_name,
        "kind": cfg.kind,
        "status": "running",
        "config_hash": config_hash,
        "data_version": dataset.data_version,
        "held_out_hypotheses": sorted(held_out),
        "train_label_counts": counts,
        "label_loss_weights": weights,
        "examples_per_epoch": {"evidence": len(ev0), "label": len(lab0)},
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "device": device,
        "backbone_files": str(files.root),
        "tokenizer_fidelity_vs_sentencepiece": fidelity,
        "tokenizer_sha256": tokenizer_sha,
        "training_code_fingerprint": code_sha,
        # Identity of the session that started the run; never overwritten by a resume.
        "provenance": previous["provenance"]
        if previous
        else provenance.collect(sys.argv, seed=cfg.seed, config=config_dump),
        # Every session (start and resumes), oldest first.
        "sessions": [
            *(previous.get("sessions", []) if previous else []),
            {
                **provenance.collect(sys.argv, seed=None),
                "resumed_from": dict(state) if last is not None else None,
            },
        ],
    }
    write_json(run_dir / "run.json", run_record)
    log_file = (run_dir / "train_log.jsonl").open("a", encoding="utf-8")
    started = time.perf_counter() - float(state["elapsed_s"])

    def train_epoch(epoch: int) -> bool:
        """Returns False when max_steps stopped training early."""
        evidence, labels = build_epoch(
            dataset, train_pairs, cfg.sampling, seed=cfg.seed, epoch=epoch, label_index=label_index
        )
        batches = make_batches(evidence, labels, tokens, encoder, cfg, epoch)
        model.train()
        accum = cfg.training.grad_accum_steps
        running: dict[str, list[float]] = {"evidence": [], "label": []}
        for b_idx in range(int(state["batch"]), len(batches)):
            batch = batches[b_idx]
            ids, mask = encoder.pad(batch.rows)
            ids_t = torch.from_numpy(ids).to(device)
            mask_t = torch.from_numpy(mask).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                ev_logits, lab_logits = model(ids_t, mask_t)
            if batch.kind == "evidence":
                target = torch.tensor(batch.targets, dtype=torch.float32, device=device)
                loss = cfg.training.evidence_loss_weight * bce(ev_logits.float(), target)
            else:
                target_l = torch.tensor(batch.targets, dtype=torch.long, device=device)
                loss = cfg.training.label_loss_weight * ce(lab_logits.float(), target_l)
            running[batch.kind].append(float(loss.detach()))
            scaler.scale(loss / accum).backward()
            state["batch"] = b_idx + 1
            if (b_idx + 1) % accum == 0 or b_idx + 1 == len(batches):
                if not optimizer_update(
                    model, optimizer, scheduler, scaler, cfg.training.max_grad_norm
                ):
                    state["skipped_updates"] = int(state.get("skipped_updates", 0)) + 1
                    continue  # no parameter update: no step, no LR schedule advance
                state["step"] += 1
                if state["step"] % LOG_EVERY == 0 or state["step"] == 1:
                    entry = {
                        "step": state["step"],
                        "epoch": epoch,
                        "batch": b_idx + 1,
                        "of": len(batches),
                        "lr": scheduler.get_last_lr()[0],
                        "evidence_loss": float(np.mean(running["evidence"]))
                        if running["evidence"]
                        else None,
                        "label_loss": float(np.mean(running["label"]))
                        if running["label"]
                        else None,
                        "elapsed_s": round(time.perf_counter() - started, 1),
                    }
                    log_file.write(json.dumps(entry) + "\n")
                    log_file.flush()
                    log(f"  step {entry['step']}/{total_steps} {entry}")
                    running = {"evidence": [], "label": []}
                if state["step"] % cfg.training.checkpoint_every_steps == 0:
                    state["elapsed_s"] = time.perf_counter() - started
                    _save_last(run_dir, model, optimizer, scheduler, scaler, state)
                    if _stop_after_step is not None and state["step"] >= _stop_after_step:
                        raise _StopForTest
                if state["step"] >= total_steps:
                    return False
        return True

    ks = cfg.selection.top_k_candidates

    def candidates_of(data: Dataset) -> Any:
        return lambda pair: data.retrieved(pair, CANDIDATE_DEPTH)

    while state["epoch"] < cfg.training.epochs:
        epoch = int(state["epoch"])
        log(f"epoch {epoch + 1}/{cfg.training.epochs}")
        completed = train_epoch(epoch)
        log("  DEV evaluation (full pipeline, evidence head's own top-k) ...")
        forward = TorchForward(
            model,
            encoder,
            device=device,
            fp16=use_fp16,
            max_tokens=cfg.inference.max_tokens_per_forward,
        )
        dev_pred = predict_pairs(
            dev_pairs,
            tokens,
            forward,
            top_k_values=ks,
            chunk_rows=cfg.inference.chunk_rows,
            progress=log,
            candidates=candidates_of(dataset),
        )
        scores = _dev_scores(dev_pred, selection_ids, ks)
        state["history"].append(
            {
                "epoch": epoch + 1,
                "step": state["step"],
                "dev_macro_f1": {str(k): v for k, v in scores.items()},
            }
        )
        log(f"  DEV macro-F1 by k (seen hypotheses): {scores}")
        best_k = min(ks, key=lambda k: (-scores[k], k))
        best = state["best"]
        if best is None or scores[best_k] > best["dev_macro_f1"]:
            state["best"] = {
                "epoch": epoch + 1,
                "k": best_k,
                "dev_macro_f1": scores[best_k],
                "step": state["step"],
            }
            _save_weights(model, run_dir / "checkpoints" / "best" / "model.safetensors")
            save_predictions(run_dir / "predictions_dev.jsonl.gz", dev_pred)
            log(f"  new best: {state['best']}")
        write_json(run_dir / "dev_history.json", state["history"])
        state["epoch"] = epoch + 1
        state["batch"] = 0
        state["elapsed_s"] = time.perf_counter() - started
        _save_last(run_dir, model, optimizer, scheduler, scaler, state)
        if not completed:
            log(f"  max_steps={cfg.training.max_steps} reached; stopping")
            break
    log_file.close()

    best = state["best"]
    assert best is not None  # at least one epoch ran  # noqa: S101
    _load_weights(model, run_dir / "checkpoints" / "best" / "model.safetensors")
    log(
        f"TEST prediction with the DEV-selected checkpoint (epoch {best['epoch']}, k={best['k']}) ..."
    )
    test_data = load_dataset(data_dir, splits=("test",))  # first TEST read: selection is frozen
    if test_data.data_version != dataset.data_version:
        raise ResumeIdentityError("the dataset changed during the run")
    forward = TorchForward(
        model,
        encoder,
        device=device,
        fp16=use_fp16,
        max_tokens=cfg.inference.max_tokens_per_forward,
    )
    test_pred = predict_pairs(
        test_data.pairs["test"],
        TokenCache(test_data, encoder),
        forward,
        top_k_values=[best["k"]],
        chunk_rows=cfg.inference.chunk_rows,
        progress=log,
        candidates=candidates_of(test_data),
    )
    save_predictions(run_dir / "predictions_test.jsonl.gz", test_pred)
    run_record.update(
        {
            "status": "complete",
            "selection": best,
            "dev_history": state["history"],
            "train_seconds": round(float(state["elapsed_s"]), 1),
            "optimizer_steps": state["step"],
            "skipped_updates": int(state.get("skipped_updates", 0)),
            "test_predicted_with": {"epoch": best["epoch"], "k": best["k"]},
            "predictions": {
                "dev": provenance.file_sha256(run_dir / "predictions_dev.jsonl.gz"),
                "test": provenance.file_sha256(run_dir / "predictions_test.jsonl.gz"),
            },
            "best_weights_sha256": provenance.file_sha256(
                run_dir / "checkpoints" / "best" / "model.safetensors"
            ),
        }
    )
    write_json(run_dir / "run.json", run_record)
    log(f"run complete: {run_dir}")
    return run_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the rule-check model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg, args.run_dir, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
