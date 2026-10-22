"""Static PNG figures for the retrieval eval (the JSON results are the table view).

Palette: the validated reference categorical order (blue, orange, aqua, yellow; adjacent
CVD dE >= 9.1, first three all-pairs dE >= 9.2), a neutral gray for the random
baseline, and a one-hue blue ramp for the heatmap. Light surface only (static images).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BASELINE = "#b5b3ad"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(
    width: float, height: float, ncols: int = 1, *, sharey: bool = False
) -> tuple[Any, Any]:
    fig, axes = plt.subplots(
        1, ncols, figsize=(width, height), facecolor=SURFACE, squeeze=False, sharey=sharey
    )
    return fig, axes[0]


def method_bars(methods: dict[str, dict[str, float]], title: str, path: Path) -> None:
    """Grouped bars: metrics on x, one bar per method (random baseline in gray)."""
    metrics = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10"]
    names = list(methods)
    fig, (ax,) = _figure(9.5, 4.8)
    _style(ax)
    width = 0.8 / len(names)
    series = iter(SERIES)
    for i, name in enumerate(names):
        color = BASELINE if name.startswith("random") else next(series)
        xs = [m + (i - (len(names) - 1) / 2) * width for m in range(len(metrics))]
        ax.bar(
            xs,
            [methods[name][m] for m in metrics],
            width=width - 0.02,  # 2px-ish surface gap between adjacent bars
            color=color,
            label=name,
            edgecolor=SURFACE,
            linewidth=0.6,
        )
    ax.set_xticks(range(len(metrics)), [m.replace("@", " @") for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("score (mean over pairs)", color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(
        frameon=False,
        fontsize=8.5,
        ncols=len(names),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        labelcolor=INK_2,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def tuning_heatmap(
    grid: dict[tuple[float, int], float],
    selected: tuple[float, int],
    title: str,
    path: Path,
) -> None:
    """recall@3 over keyword weight (rows) x RRF k (columns); selected cell outlined."""
    weights = sorted({w for w, _ in grid})
    ks = sorted({k for _, k in grid})
    values = [[grid[(w, k)] for k in ks] for w in weights]
    flat = [v for row in values for v in row]
    lo, hi = min(flat), max(flat)
    fig, (ax,) = _figure(7.5, 4.6)
    ax.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    image = ax.imshow(values, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
    for r, row in enumerate(values):
        for c, value in enumerate(row):
            dark = (value - lo) / ((hi - lo) or 1) > 0.55
            ax.text(
                c,
                r,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#ffffff" if dark else INK,
            )
    sr, sc = weights.index(selected[0]), ks.index(selected[1])
    ax.add_patch(Rectangle((sc - 0.5, sr - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=2))
    ax.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax.set_yticks(range(len(weights)), [f"{w:g}" for w in weights])
    ax.set_xlabel("RRF constant k", color=INK_2, fontsize=9)
    ax.set_ylabel("keyword weight (vector weight = 1)", color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.04)
    bar.ax.tick_params(colors=INK_2, labelsize=8)
    bar.outline.set_visible(False)
    bar.set_label("DEV evidence recall@3", color=INK_2, fontsize=9)
    ax.set_title(textwrap.fill(title, 70), color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def sweep_curves(rows: list[dict[str, Any]], path: Path) -> None:
    """Small multiples by m; one line per ef_construction; points labelled by ef_search."""
    ms = sorted({r["m"] for r in rows})
    efcs = sorted({r["ef_construction"] for r in rows})
    fig, axes = _figure(4.0 * len(ms), 4.2, ncols=len(ms), sharey=True)
    all_lat = [r["latency_ms"]["p50"] for r in rows]
    for ax, m in zip(axes, ms, strict=True):
        _style(ax)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        for color, efc in zip(SERIES, efcs, strict=False):
            pts = sorted(
                (r for r in rows if r["m"] == m and r["ef_construction"] == efc),
                key=lambda r: r["ef_search"],
            )
            xs = [p["latency_ms"]["p50"] for p in pts]
            ys = [p["ann_recall@10"] for p in pts]
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2,
                marker="o",
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
                label=f"ef_construction {efc}",
            )
            if efc == efcs[-1]:
                for p, x, y in zip(pts, xs, ys, strict=True):
                    ax.annotate(
                        str(p["ef_search"]),
                        (x, y),
                        textcoords="offset points",
                        xytext=(4, -10),
                        fontsize=7,
                        color=MUTED,
                    )
        ax.set_title(f"m = {m}", color=INK, fontsize=10, loc="left")
        ax.set_xlim(0, max(all_lat) * 1.1)
        ax.set_xlabel("p50 query latency (ms)", color=INK_2, fontsize=9)
    axes[0].set_ylabel("ANN recall@10 vs exact neighbours", color=INK_2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK_2)
    fig.suptitle("HNSW sweep (labels: ef_search)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
