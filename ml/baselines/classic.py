"""Majority-class and TF-IDF + logistic-regression baselines (fit on TRAIN, chosen on DEV).

    uv run python -m ml.baselines.classic [--data-dir .cache/ml/data/full]

Systems written to ``<out>/`` in the common prediction format (all official DEV and
TEST pairs):

- ``majority_global``: the most frequent TRAIN label for every pair (probabilities =
  TRAIN label frequencies). Required by the handoff.
- ``majority_per_hypothesis``: the most frequent TRAIN label of the pair's hypothesis
  (a stronger, informative prior baseline).
- ``tfidf_logreg``:
  - label: multinomial logistic regression over TF-IDF features of the pair's
    candidate text, i.e. the top-``CANDIDATE_DEPTH`` (8) dense-ranked segments, the same
    non-gold candidates the LLM baseline reads. Features are word 1-2 grams, both shared
    and conjoined with the hypothesis key (so weights can be hypothesis-specific).
  - evidence: a logistic regression over TF-IDF features of a single segment (shared +
    hypothesis-conjoined) trained on every TRAIN (pair, segment); every segment of a
    DEV/TEST pair is scored. The set prediction uses a DEV-selected threshold.
  - C, class weighting and the evidence C are selected by DEV macro-F1 / DEV mAP; the
    final models are fit on TRAIN only (as the neural model). TEST is predicted only
    with the frozen DEV choices; TEST results are never used to change them. The label C
    grid was extended upward (to 30 and 100) after an earlier run selected its upper
    edge C=10 on DEV; that extension was DEV-driven.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from compliance_model_service.rule_check.labels import LABELS
from ml.common import provenance
from ml.common.io import CACHE, write_json
from ml.data.contractnli import Pair
from ml.data.prepare import CANDIDATE_DEPTH
from ml.data.store import Dataset, load_dataset
from ml.eval import metrics
from ml.eval.calibration import select_threshold
from ml.eval.systems import SystemPrediction, save_system

SEED = 20261021
# Extended upward after the first DEV selection hit the grid edge (C=10); DEV-only.
LABEL_C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
CLASS_WEIGHT_GRID: tuple[str | None, ...] = (None, "balanced")
EVIDENCE_C_GRID = (0.3, 1.0, 3.0)
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def log(message: str) -> None:
    print(message, flush=True)  # noqa: T201


def analyzer(document: str) -> list[str]:
    """``"<key>\\t<text>"`` -> shared and hypothesis-conjoined word 1-2 grams."""
    key, _, text = document.partition("\t")
    words = _WORD.findall(text.lower())
    grams = words + [f"{a} {b}" for a, b in itertools.pairwise(words)]
    return [*grams, *(f"{key}|{g}" for g in grams)]


def candidate_text(dataset: Dataset, pair: Pair) -> str:
    doc = dataset.document(pair)
    segs = sorted(dataset.retrieved(pair, CANDIDATE_DEPTH))
    return pair.key + "\t" + " ".join(doc.segment_text(i) for i in segs)


def segment_rows(dataset: Dataset, pairs: Sequence[Pair]) -> Iterator[tuple[Pair, int, str]]:
    for pair in pairs:
        doc = dataset.document(pair)
        for i in range(len(doc.segments)):
            yield pair, i, pair.key + "\t" + doc.segment_text(i)


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer=analyzer, min_df=2, sublinear_tf=True, dtype=np.float32)


def majority(dataset: Dataset, out: Path, meta_base: dict[str, Any]) -> dict[str, Any]:
    train = dataset.pairs["train"]
    counts = Counter(p.label for p in train)
    total = sum(counts.values())
    freq = [counts[label] / total for label in LABELS]
    top = max(LABELS, key=lambda label: (counts[label], -LABELS.index(label)))
    per_key: dict[str, Counter[str]] = {}
    for p in train:
        per_key.setdefault(p.key, Counter())[p.label] += 1
    key_top = {
        k: max(LABELS, key=lambda label: (c[label], -LABELS.index(label)))
        for k, c in per_key.items()
    }
    key_freq = {k: [c[label] / sum(c.values()) for label in LABELS] for k, c in per_key.items()}
    for split in ("dev", "test"):
        pairs = dataset.pairs[split]
        save_system(
            out,
            "majority_global",
            split,
            [SystemPrediction(p.pair_id, top, probs=freq) for p in pairs],
            {
                **meta_base,
                "system": "majority_global",
                "split": split,
                "train_label": top,
                "train_frequencies": dict(zip(LABELS, freq, strict=True)),
            },
        )
        save_system(
            out,
            "majority_per_hypothesis",
            split,
            [SystemPrediction(p.pair_id, key_top[p.key], probs=key_freq[p.key]) for p in pairs],
            {
                **meta_base,
                "system": "majority_per_hypothesis",
                "split": split,
                "train_labels": key_top,
            },
        )
    return {"global_label": top, "train_counts": dict(counts), "per_hypothesis": key_top}


def tfidf(dataset: Dataset, out: Path, meta_base: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    train, dev, test = dataset.pairs["train"], dataset.pairs["dev"], dataset.pairs["test"]
    label_of = {label: i for i, label in enumerate(LABELS)}

    # ---- label model over the shared top-8 candidate text
    vec = _vectorizer()
    x_train = vec.fit_transform([candidate_text(dataset, p) for p in train])
    x_dev = vec.transform([candidate_text(dataset, p) for p in dev])
    y_train = np.array([label_of[p.label] for p in train])
    grid: list[dict[str, Any]] = []
    best: tuple[float, float, str | None] | None = None
    for c in LABEL_C_GRID:
        for cw in CLASS_WEIGHT_GRID:
            model = LogisticRegression(C=c, class_weight=cw, max_iter=4000, random_state=SEED)
            model.fit(x_train, y_train)
            pred = [LABELS[i] for i in model.predict(x_dev)]
            f1 = metrics.macro_f1([p.label for p in dev], pred)
            grid.append({"C": c, "class_weight": cw, "dev_macro_f1": f1})
            if best is None or f1 > best[0]:
                best = (f1, c, cw)
    assert best is not None  # noqa: S101
    _, c_best, cw_best = best
    label_model = LogisticRegression(
        C=c_best, class_weight=cw_best, max_iter=4000, random_state=SEED
    )
    label_model.fit(x_train, y_train)
    log(f"  label model: DEV-selected C={c_best} class_weight={cw_best}")

    # ---- evidence model over single segments
    ev_vec = _vectorizer()
    train_rows = list(segment_rows(dataset, train))
    x_ev = ev_vec.fit_transform([t for _, _, t in train_rows])
    y_ev = np.array([int(i in p.gold_segments) for p, i, _ in train_rows])
    dev_rows = list(segment_rows(dataset, dev))
    x_ev_dev = ev_vec.transform([t for _, _, t in dev_rows])
    ev_grid: list[dict[str, Any]] = []
    best_ev: tuple[float, float] | None = None
    dev_ec = [p for p in dev if p.label != "not_mentioned"]

    def score_rows(
        model: LogisticRegression, rows: list[tuple[Pair, int, str]], x: Any
    ) -> dict[str, list[float]]:
        probs = model.predict_proba(x)[:, 1]
        by_pair: dict[str, list[float]] = {}
        for (p, _, _), s in zip(rows, probs, strict=True):
            by_pair.setdefault(p.pair_id, []).append(float(s))
        return by_pair

    for c in EVIDENCE_C_GRID:
        model = LogisticRegression(C=c, max_iter=4000, random_state=SEED)
        model.fit(x_ev, y_ev)
        scores = score_rows(model, dev_rows, x_ev_dev)
        m = metrics.mean_average_precision(
            [scores[p.pair_id] for p in dev_ec], [set(p.gold_segments) for p in dev_ec]
        )
        ev_grid.append({"C": c, "dev_map": m})
        if best_ev is None or m > best_ev[0]:
            best_ev = (m, c)
    assert best_ev is not None  # noqa: S101
    ev_model = LogisticRegression(C=best_ev[1], max_iter=4000, random_state=SEED)
    ev_model.fit(x_ev, y_ev)
    dev_scores = score_rows(ev_model, dev_rows, x_ev_dev)
    threshold = select_threshold(
        [dev_scores[p.pair_id] for p in dev_ec], [set(p.gold_segments) for p in dev_ec]
    )
    log(f"  evidence model: DEV-selected C={best_ev[1]}, threshold {threshold}")

    selection = {
        "label": {"C": c_best, "class_weight": cw_best, "grid": grid},
        "evidence": {"C": best_ev[1], "grid": ev_grid, "threshold": threshold},
        "features": "TF-IDF word 1-2 grams (min_df=2, sublinear tf), shared + hypothesis-conjoined",
        "label_input": f"top-{CANDIDATE_DEPTH} dense-ranked candidate segments (no gold)",
        "fit_on": "TRAIN only; selection on DEV only; TEST predicted with frozen DEV choices, no TEST feedback",
    }
    for split, pairs in (("dev", dev), ("test", test)):
        if split == "dev":
            x_lab, ev_scores = x_dev, dev_scores
        else:
            x_lab = vec.transform([candidate_text(dataset, p) for p in pairs])
            rows = list(segment_rows(dataset, pairs))
            ev_scores = score_rows(ev_model, rows, ev_vec.transform([t for _, _, t in rows]))
        probs = label_model.predict_proba(x_lab)
        order = list(label_model.classes_)
        preds = []
        for p, row in zip(pairs, probs, strict=True):
            full = [float(row[order.index(i)]) if i in order else 0.0 for i in range(len(LABELS))]
            seg_scores = ev_scores[p.pair_id]
            preds.append(
                SystemPrediction(
                    p.pair_id,
                    LABELS[int(np.argmax(full))],
                    probs=full,
                    evidence_scores=[round(s, 6) for s in seg_scores],
                    evidence_set=[
                        i for i, s in enumerate(seg_scores) if s >= threshold["threshold"]
                    ],
                )
            )
        save_system(
            out,
            "tfidf_logreg",
            split,
            preds,
            {**meta_base, "system": "tfidf_logreg", "split": split, "selection": selection},
        )
    selection["seconds"] = round(time.perf_counter() - started, 1)
    return selection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Majority and TF-IDF + logistic regression baselines"
    )
    parser.add_argument("--data-dir", type=Path, default=CACHE / "data" / "full")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    dataset = load_dataset(args.data_dir)
    out: Path = args.out or CACHE / "baselines" / dataset.data_version[:16]
    meta_base = {
        "data_version": dataset.data_version,
        "data_name": dataset.manifest["name"],
        "candidate_depth": CANDIDATE_DEPTH,
        "provenance": provenance.collect(sys.argv, seed=SEED),
    }
    log(f"majority baselines -> {out}")
    summary: dict[str, Any] = {"majority": majority(dataset, out, meta_base)}
    log("TF-IDF + logistic regression (TRAIN fit, DEV selection) ...")
    summary["tfidf_logreg"] = tfidf(dataset, out, meta_base)
    write_json(out / "classic_baselines.json", {**meta_base, **summary})
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
