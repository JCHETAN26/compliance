"""Evaluate trained runs against the baselines; fit calibration and thresholds on DEV.

    uv run python -m ml.eval.evaluate --run-dir <deberta run> \
        [--ablation-run-dir <modernbert run>] [--unseen-run-dir <unseen-4 run>] \
        [--baselines-dir .cache/ml/baselines/<data_version>] [--bench <latency json>] \
        [--out ml/eval/results]

For each model run (predictions written by ``ml.train.train``):

1. k is the run's DEV-selected top-k; TEST predictions exist for that k only.
2. On DEV (seen hypotheses only) the label temperature T is fitted (NLL) and the
   evidence threshold is chosen (segment F1). TEST is never used for any choice.
3. TEST is scored with exactly the same code as every baseline (``ml.eval.scoring``).

The serving backbone is the run with the higher DEV selection score (primary vs
ablation). The cascade uses that model's calibrated confidence and, if complete real
LLM predictions exist on DEV and TEST, the LLM's answers and measured costs.

Output files (``--out``): ``model_eval.json`` (+ ``inference_params.json`` read by the
exporter), ``baselines_comparison.json``, ``ablation.json``, ``generalization_unseen.json``,
``cascade.json`` and PNG figures. Smoke or subset inputs are refused for
``ml/eval/results`` (the default) and are labelled SMOKE wherever they are written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from compliance_model_service.rule_check.encoding import sigmoid, softmax
from compliance_model_service.rule_check.labels import LABELS
from ml.common import provenance
from ml.common.io import CACHE, RESULTS, read_json, write_json
from ml.data.contractnli import Pair
from ml.data.store import Dataset, load_dataset
from ml.eval import cascade, metrics, plots, scoring
from ml.eval.calibration import fit_temperature, select_threshold
from ml.eval.infer import PairPrediction, load_predictions
from ml.eval.systems import SystemPrediction, has_system, load_system

SMOKE_NOTE = "SMOKE: tiny random backbone and/or data subset. Pipeline check only; NOT a result."
BASELINE_SYSTEMS = ("majority_global", "majority_per_hypothesis", "tfidf_logreg", "llm_zero_shot")
CANDIDATE_SCOPE = {
    "majority_global / majority_per_hypothesis": "no document input",
    "tfidf_logreg": "label: the BGE top-8 candidate segments; evidence: every segment of the document",
    "llm_zero_shot": "label and evidence: the same BGE top-8 candidate segments (cannot cite others)",
    "rule_check_model": "every segment scored by the learned evidence head; label reads its own top-k (the intended pipeline)",
    "rule_check_model_top8_candidates": "same weights; label top-k restricted to the BGE top-8 candidates (isolates head quality)",
    "evidence_denominators": "end-to-end: all annotated evidence of the document for every system",
    "note": "Candidate scopes differ between rule_check_model and the input-bearing baselines, so a full-pipeline gain is not attributable to the classifier alone; compare the top8 variant for that.",
}


class EvaluationInputError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, flush=True)  # noqa: T201


def _fit_variant(
    dataset: Dataset,
    preds: dict[str, dict[str, PairPrediction]],
    k: int,
    held_out: set[str],
    *,
    candidates_only: bool,
) -> dict[str, Any]:
    """Fit T and the evidence threshold on DEV (seen hypotheses), convert DEV and TEST.

    ``candidates_only``: the fixed BGE top-8 candidate variant. Its label logits come from
    the top-k chosen among the candidates; evidence scores outside the candidates are 0.
    """
    gold_of = {p.pair_id: p for p in dataset.pairs["dev"]}

    def label_logits(pred: PairPrediction) -> list[float]:
        return pred.candidate_label_logits[k] if candidates_only else pred.label_logits[k]

    def evidence(pred: PairPrediction) -> list[float]:
        scores = [float(x) for x in sigmoid(np.array(pred.evidence_logits))]
        if candidates_only:
            allowed = set(pred.candidates or [])
            scores = [s if i in allowed else 0.0 for i, s in enumerate(scores)]
        return scores

    dev_sel = [preds["dev"][p.pair_id] for p in dataset.pairs["dev"] if p.key not in held_out]
    temperature = fit_temperature(
        np.array([label_logits(p) for p in dev_sel]), [LABELS.index(p.gold) for p in dev_sel]
    )
    dev_ec = [p for p in dev_sel if p.gold != "not_mentioned"]
    threshold = select_threshold(
        [evidence(p) for p in dev_ec], [set(gold_of[p.pair_id].gold_segments) for p in dev_ec]
    )
    t, theta = temperature["temperature"], threshold["threshold"]

    def convert(pred: PairPrediction, temp: float) -> SystemPrediction:
        probs = softmax(np.array(label_logits(pred)), temp)
        scores = evidence(pred)
        return SystemPrediction(
            pred.pair_id,
            LABELS[int(np.argmax(probs))],
            probs=[float(x) for x in probs],
            evidence_scores=scores,
            evidence_set=[i for i, s in enumerate(scores) if s >= theta and s > 0.0],
        )

    return {
        "temperature": temperature,
        "evidence_threshold": threshold,
        "systems": {
            split: [convert(preds[split][p.pair_id], t) for p in dataset.pairs[split]]
            for split in ("dev", "test")
        },
        "uncalibrated": {
            split: [convert(preds[split][p.pair_id], 1.0) for p in dataset.pairs[split]]
            for split in ("dev", "test")
        },
    }


def model_system(run_dir: Path, dataset: Dataset) -> dict[str, Any]:
    run = read_json(run_dir / "run.json")
    if run.get("status") != "complete":
        raise EvaluationInputError(f"{run_dir} is not a complete training run")
    if run["data_version"] != dataset.data_version:
        raise EvaluationInputError(
            f"{run_dir} was trained on data {run['data_version'][:16]}, not {dataset.data_version[:16]}"
        )
    k = int(run["selection"]["k"])
    held_out = set(run.get("held_out_hypotheses", []))
    preds = {
        "dev": {p.pair_id: p for p in load_predictions(run_dir / "predictions_dev.jsonl.gz")},
        "test": {p.pair_id: p for p in load_predictions(run_dir / "predictions_test.jsonl.gz")},
    }
    for split in ("dev", "test"):
        if set(preds[split]) != {p.pair_id for p in dataset.pairs[split]}:
            raise EvaluationInputError(
                f"{run_dir} {split} predictions do not cover the official pairs"
            )
    main = _fit_variant(dataset, preds, k, held_out, candidates_only=False)
    has_candidates = all(
        k in p.candidate_label_logits for split in preds.values() for p in split.values()
    )
    variant = (
        _fit_variant(dataset, preds, k, held_out, candidates_only=True) if has_candidates else None
    )
    truncation = {
        split: {
            "evidence_inputs_truncated": sum(
                p.truncated_evidence_inputs for p in preds[split].values()
            ),
            "label_inputs_truncated": sum(
                bool(p.truncated_label_inputs.get(k)) for p in preds[split].values()
            ),
        }
        for split in ("dev", "test")
    }
    return {
        "run": run,
        "run_dir": str(run_dir),
        "k": k,
        "held_out": held_out,
        **main,
        "top8_candidates_variant": variant,
        "truncation": truncation,
    }


def calibration_report(dataset: Dataset, model: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"temperature": model["temperature"]}
    for split in ("dev", "test"):
        gold = [p.label for p in dataset.pairs[split]]
        for name, rows in (
            ("before", model["uncalibrated"][split]),
            ("after", model["systems"][split]),
        ):
            conf = [max(r.probs or []) for r in rows]
            correct = [g == r.label for g, r in zip(gold, rows, strict=True)]
            out[f"{split}_{name}"] = metrics.expected_calibration_error(conf, correct)
    return out


def subset(
    pairs: list[Pair], rows: list[SystemPrediction], keys: set[str], inside: bool
) -> tuple[list[Pair], list[SystemPrediction]]:
    by_id = {r.pair_id: r for r in rows}
    chosen = [p for p in pairs if (p.key in keys) == inside]
    return chosen, [by_id[p.pair_id] for p in chosen]


def load_baselines(
    baselines_dir: Path | None, dataset: Dataset
) -> tuple[dict[str, dict[str, list[SystemPrediction]]], dict[str, Any]]:
    systems: dict[str, dict[str, list[SystemPrediction]]] = {}
    notes: dict[str, Any] = {}
    if baselines_dir is None:
        return systems, {"note": "no baselines directory given"}
    for name in (*BASELINE_SYSTEMS, "llm_zero_shot_partial"):
        if not all(has_system(baselines_dir, name, s) for s in ("dev", "test")):
            notes[name] = "not available (missing DEV or TEST predictions)"
            continue
        per_split: dict[str, list[SystemPrediction]] = {}
        for split in ("dev", "test"):
            rows, meta = load_system(baselines_dir, name, split)
            if meta["data_version"] != dataset.data_version:
                raise EvaluationInputError(f"{name} {split} used data {meta['data_version'][:16]}")
            if meta.get("partial"):
                notes[name] = "partial sanity run only; not scored as a baseline"
                break
            per_split[split] = rows
        else:
            meta = {
                k: v
                for k, v in load_system(baselines_dir, name, "test")[1].items()
                if k != "provenance"
            }
            if name == "llm_zero_shot":
                usability = cascade.llm_usability(per_split)
                if not usability["usable"]:
                    # Rejected requests are not answers: not a baseline, not cascade input.
                    notes[name] = {
                        "excluded": "unusable: too few answered pairs (see usability); not "
                        "scored as a baseline and not used for the cascade",
                        "usability": usability,
                        "meta": meta,
                    }
                    continue
                notes[name] = {"meta": meta, "usability": usability}
            else:
                notes[name] = {"meta": meta}
            systems[name] = per_split
    return systems, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate rule-check runs and baselines")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ablation-run-dir", type=Path, default=None)
    parser.add_argument("--unseen-run-dir", type=Path, default=None)
    parser.add_argument("--baselines-dir", type=Path, default=None)
    parser.add_argument(
        "--bench", type=Path, default=None, help="latency JSON from ml.export.bench"
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args(argv)

    primary_cfg = read_json(args.run_dir / "run.json")["provenance"]["config"]
    data_dir = args.data_dir or provenance.REPO / primary_cfg["data"]["data_dir"]
    dataset = load_dataset(data_dir)
    runs = {
        "primary": args.run_dir,
        "ablation": args.ablation_run_dir,
        "unseen": args.unseen_run_dir,
    }
    kinds = {name: read_json(d / "run.json")["kind"] for name, d in runs.items() if d is not None}
    smoke = dataset.manifest["name"] != "full" or any(k != "production" for k in kinds.values())
    if smoke and args.out.resolve() == RESULTS.resolve():
        raise EvaluationInputError(
            "smoke/subset inputs cannot be written to ml/eval/results; pass --out"
        )
    baselines_dir = args.baselines_dir or (CACHE / "baselines" / dataset.data_version[:16])
    baselines, baseline_notes = load_baselines(
        baselines_dir if baselines_dir.is_dir() else None, dataset
    )
    common = {
        "scope": SMOKE_NOTE
        if smoke
        else "Official ContractNLI splits; all pairs; every choice fitted on DEV only, no TEST feedback",
        "data_version": dataset.data_version,
        "provenance": provenance.collect(sys.argv, seed=scoring.BOOTSTRAP_SEED),
    }
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    log("model runs: fitting temperature / evidence threshold on DEV ...")
    models = {
        name: model_system(d, dataset)
        for name, d in runs.items()
        if d is not None and name != "unseen"
    }
    unseen = model_system(args.unseen_run_dir, dataset) if args.unseen_run_dir else None
    dev_scores = {name: float(m["run"]["selection"]["dev_macro_f1"]) for name, m in models.items()}
    chosen_name = max(dev_scores, key=lambda n: (dev_scores[n], n == "primary"))
    chosen = models[chosen_name]
    log(f"serving backbone (DEV selection): {chosen_name} {dev_scores}")

    test_pairs, dev_pairs = dataset.pairs["test"], dataset.pairs["dev"]
    model_test = scoring.score(dataset, test_pairs, chosen["systems"]["test"])
    model_dev = scoring.score(dataset, dev_pairs, chosen["systems"]["dev"], bootstrap=False)
    calibration = calibration_report(dataset, chosen)
    inference = {
        "run": chosen_name,
        "run_dir": chosen["run_dir"],
        "backbone": chosen["run"]["provenance"]["config"]["backbone"]["repo_id"],
        "top_k": chosen["k"],
        "temperature": chosen["temperature"]["temperature"],
        "evidence_threshold": chosen["evidence_threshold"]["threshold"],
    }

    # ---------------------------------------------------------------- baselines
    table: dict[str, Any] = {}
    for name, splits in baselines.items():
        table[name] = {
            "test": scoring.score(dataset, test_pairs, splits["test"]),
            "dev": scoring.score(dataset, dev_pairs, splits["dev"], bootstrap=False),
        }
    table["rule_check_model"] = {"test": model_test, "dev": model_dev}
    variant = chosen["top8_candidates_variant"]
    if variant is not None:
        table["rule_check_model_top8_candidates"] = {
            "test": scoring.score(dataset, test_pairs, variant["systems"]["test"]),
            "dev": scoring.score(dataset, dev_pairs, variant["systems"]["dev"], bootstrap=False),
            "temperature": variant["temperature"],
            "evidence_threshold": variant["evidence_threshold"],
        }
    comparisons: dict[str, Any] = {}
    for other in ("tfidf_logreg", "llm_zero_shot", "majority_per_hypothesis", "majority_global"):
        if other in baselines:
            comparisons[f"model_minus_{other}"] = scoring.paired_difference(
                test_pairs, chosen["systems"]["test"], baselines[other]["test"]
            )
    criterion: dict[str, Any] = {
        "criterion": "model TEST macro-F1 > TF-IDF baseline TEST macro-F1 (handoff Phase 3)"
    }
    if "tfidf_logreg" in table:
        m, b = model_test["label"]["macro_f1"], table["tfidf_logreg"]["test"]["label"]["macro_f1"]
        criterion.update(
            model=m,
            tfidf=b,
            met=m > b,
            difference_ci95=comparisons["model_minus_tfidf_logreg"]["ci95"],
        )
    else:
        criterion["met"] = None
        criterion["note"] = "TF-IDF baseline predictions not available"

    # ---------------------------------------------------------------- cascade
    llm = baselines.get("llm_zero_shot")
    bench = read_json(args.bench) if args.bench else None
    model_latency = None
    if bench is not None:
        model_latency = bench.get("per_pair_latency_s", {}).get("onnx_int8", {}).get("p50")
    cascade_out: dict[str, Any] = {}
    curves: dict[str, list[dict[str, Any]]] = {}
    for split in ("dev", "test"):
        rows = chosen["systems"][split]
        pairs = dataset.pairs[split]
        curves[split] = cascade.curve(
            [p.label for p in pairs],
            [r.label or "" for r in rows],
            [max(r.probs or [0.0]) for r in rows],
            llm_label=[r.label for r in llm[split]] if llm else None,
            llm_cost=[r.cost_usd or 0.0 for r in llm[split]] if llm else None,
            llm_latency=[r.latency_s or 0.0 for r in llm[split]] if llm else None,
            model_latency_s=model_latency,
            documents=len({p.doc_id for p in pairs}),
        )
    llm_dev_acc = table["llm_zero_shot"]["dev"]["label"]["accuracy"] if llm else None
    tau = cascade.select_tau(curves["dev"], llm_dev_acc)
    inference["tau"] = tau["tau"]
    inference["tau_provisional"] = tau["provisional"]
    # The exporter serves a tau only with this explicit evidence of a usable LLM baseline.
    inference["llm_baseline_usable"] = llm is not None
    llm_note = baseline_notes.get("llm_zero_shot") or "not available"
    if llm:
        llm_status = "usable"
    elif isinstance(llm_note, dict):
        llm_status = llm_note.get("excluded", "not available")
    else:
        llm_status = llm_note
    cascade_out = {
        "llm_baseline": llm_status,
        "tau_selection": tau,
        "test_at_tau": cascade.at_tau(curves["test"], tau["tau"]),
        "dev_at_tau": cascade.at_tau(curves["dev"], tau["tau"]),
        "llm_only_test": (
            {
                "accuracy": table["llm_zero_shot"]["test"]["label"]["accuracy"],
                "macro_f1": table["llm_zero_shot"]["test"]["label"]["macro_f1"],
                "cost": table["llm_zero_shot"]["test"].get("cost"),
                "latency_s": table["llm_zero_shot"]["test"].get("latency_s"),
            }
            if llm
            else "pending: no complete real Anthropic predictions"
        ),
        "model_latency_p50_s_per_pair": model_latency,
        "curves": curves,
    }

    # ---------------------------------------------------------------- unseen hypotheses
    generalization: dict[str, Any] = {"status": "pending: unseen-hypothesis run not provided"}
    if unseen is not None:
        held = unseen["held_out"]
        u_pairs, u_rows = subset(test_pairs, unseen["systems"]["test"], held, inside=True)
        s_pairs, s_rows = subset(test_pairs, unseen["systems"]["test"], held, inside=False)
        p_pairs, p_rows = subset(test_pairs, chosen["systems"]["test"], held, inside=True)
        generalization = {
            "held_out_hypotheses": sorted(held),
            "selection_note": "held-out keys drawn with random.Random(20261021).sample before training",
            "unseen_run_on_held_out_test": scoring.score(dataset, u_pairs, u_rows),
            "unseen_run_on_seen_test": scoring.score(dataset, s_pairs, s_rows),
            "primary_run_on_same_held_out_test (trained with them)": scoring.score(
                dataset, p_pairs, p_rows
            ),
            "unseen_run": {
                "dir": unseen["run_dir"],
                "k": unseen["k"],
                "temperature": unseen["temperature"],
            },
        }

    # ---------------------------------------------------------------- write
    model_eval = {
        **common,
        "serving_choice": {
            "chosen": chosen_name,
            "dev_selection_macro_f1": dev_scores,
            "rule": "higher DEV macro-F1 (seen hypotheses)",
        },
        "run": {
            k: chosen["run"][k]
            for k in (
                "run_name",
                "kind",
                "config_hash",
                "selection",
                "train_seconds",
                "optimizer_steps",
                "label_loss_weights",
                "train_label_counts",
            )
            if k in chosen["run"]
        },
        "inference": inference,
        "evidence_threshold": chosen["evidence_threshold"],
        "calibration": calibration,
        "truncation": chosen["truncation"],
        "test": model_test,
        "dev": model_dev,
    }
    write_json(out / "model_eval.json", model_eval)
    write_json(
        out / "inference_params.json",
        {**inference, "data_version": dataset.data_version, "scope": common["scope"]},
    )
    if variant is not None and "tfidf_logreg" in baselines:
        comparisons["model_top8_candidates_minus_tfidf_logreg"] = scoring.paired_difference(
            test_pairs, variant["systems"]["test"], baselines["tfidf_logreg"]["test"]
        )
    write_json(
        out / "baselines_comparison.json",
        {
            **common,
            "candidate_scope": CANDIDATE_SCOPE,
            "handoff_criterion": criterion,
            "systems": table,
            "paired_differences": comparisons,
            "baseline_notes": baseline_notes,
        },
    )
    write_json(out / "cascade.json", {**common, **cascade_out})
    write_json(out / "generalization_unseen.json", {**common, **generalization})
    if "ablation" in models:
        abl = models["ablation"]
        write_json(
            out / "ablation.json",
            {
                **common,
                "dev_selection_macro_f1": dev_scores,
                "chosen_for_serving": chosen_name,
                "runs": {
                    name: {
                        "backbone": m["run"]["provenance"]["config"]["backbone"]["repo_id"],
                        "k": m["k"],
                        "temperature": m["temperature"],
                        "test": scoring.score(dataset, test_pairs, m["systems"]["test"]),
                    }
                    for name, m in models.items()
                },
                "note": "Backbone choice uses DEV only; TEST numbers of both are reported for transparency.",
            },
        )
        del abl
    plots.confusion(
        out / "confusion_test.png",
        model_test["label"]["confusion"]["rows_gold_cols_pred"],
        f"Rule-check model, TEST ({chosen_name})",
    )
    plots.reliability(
        out / "reliability_test.png",
        calibration["test_before"],
        calibration["test_after"],
        f"TEST reliability (T={inference['temperature']:.2f} fitted on DEV)",
    )
    plots.coverage_accuracy(
        out / "coverage_accuracy.png",
        {"DEV": curves["dev"], "TEST": curves["test"]},
        tau["tau"],
        f"Cascade coverage vs accuracy (tau={tau['tau']}{' provisional' if tau['provisional'] else ''})",
    )
    bars = [
        (
            name,
            s["test"]["label"]["macro_f1"],
            s["test"]["label"]["macro_f1_ci95"]["low"],
            s["test"]["label"]["macro_f1_ci95"]["high"],
        )
        for name, s in table.items()
    ]
    plots.systems_bar(
        out / "systems_macro_f1_test.png",
        bars,
        "ContractNLI TEST label macro-F1" + (" (SMOKE)" if smoke else ""),
    )
    log(f"wrote evaluation to {out}; handoff criterion: {criterion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
