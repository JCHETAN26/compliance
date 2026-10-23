"""PNG figures for the model evaluation (matplotlib, headless)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Figures are only ever written to files, so the backend is always the headless Agg, chosen
# BEFORE matplotlib is imported: an inherited MPLBACKEND that the project environment cannot
# load (Jupyter/Kaggle/Colab set module://matplotlib_inline.backend_inline) makes
# `import matplotlib` itself raise.
os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")  # also if matplotlib was already imported with another backend
import matplotlib.pyplot as plt

from compliance_model_service.rule_check.labels import LABELS


def confusion(path: Path, matrix: list[list[int]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=20)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    peak = max(max(row) for row in matrix) or 1
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > peak / 2 else "black",
            )
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def reliability(path: Path, before: dict[str, Any], after: dict[str, Any], title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot(
        [0, 1], [0, 1], color="#999999", linewidth=1, linestyle="--", label="perfect calibration"
    )
    for table, name, color in (
        (before, "before (T=1)", "#d95f02"),
        (after, "after temperature scaling", "#1b9e77"),
    ):
        xs = [b["mean_confidence"] for b in table["reliability"] if b["count"]]
        ys = [b["accuracy"] for b in table["reliability"] if b["count"]]
        ax.plot(xs, ys, marker="o", color=color, label=f"{name}: ECE {table['ece']:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence (predicted label probability)")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def coverage_accuracy(
    path: Path, curves: dict[str, list[dict[str, Any]]], tau: float, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for name, rows in curves.items():
        xs = [r["coverage"] for r in rows if r["model_accuracy_on_covered"] is not None]
        ys = [
            r["model_accuracy_on_covered"]
            for r in rows
            if r["model_accuracy_on_covered"] is not None
        ]
        ax.plot(xs, ys, label=f"{name}: model accuracy on covered pairs")
        if rows and "cascade_accuracy" in rows[0]:
            ax.plot(
                [r["coverage"] for r in rows],
                [r["cascade_accuracy"] for r in rows],
                linestyle="--",
                label=f"{name}: cascade accuracy (model + LLM)",
            )
        chosen = next((r for r in rows if abs(r["tau"] - tau) < 1e-9), None)
        if chosen is not None and chosen["model_accuracy_on_covered"] is not None:
            ax.scatter(
                [chosen["coverage"]], [chosen["model_accuracy_on_covered"]], color="black", zorder=3
            )
    ax.set_xlabel("coverage (fraction decided by the model)")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1.02)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def systems_bar(
    path: Path, rows: list[tuple[str, float, float | None, float | None]], title: str
) -> None:
    """rows: (system, macro-F1, ci low, ci high)."""
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(rows) + 1.2))
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    ax.barh(names, values, color="#4c78a8")
    for i, (_, v, lo, hi) in enumerate(rows):
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [i, i], color="black", linewidth=1.2)
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("TEST macro-F1 (95% document-bootstrap CI)")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
