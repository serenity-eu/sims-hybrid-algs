#!/usr/bin/env python3
"""Generate SVG plots from hybrid two-phase benchmark JSONL data.

Usage:
    uv run --with matplotlib python3 scripts/generate_hybrid_benchmark_plots.py \
        --jsonl-file results/hybrid_benchmark_sml.jsonl \
        --output-dir docs/hybrid_benchmark_plots

Produces:
    1. hv_by_algo_ratio_grouped.svg        – Grouped bar: avg HV per algo×ratio, grouped by size tier
    2. hv_heatmap.svg                      – Heatmap: HV per (algo×ratio × instance)
    3. hv_seed_effect.svg                  – Line chart: HV vs ratio showing seed benefit, per algo
    4. hv_seed_effect_by_tier.svg          – Line chart: seed benefit broken down by size tier
    5. front_size_grouped.svg              – Grouped bar: avg front size per algo×ratio by tier
    6. front_size_ratio_concurrent_exh.svg – Bar: concurrent/exhaustive front size ratio by tier
    7. rank_heatmap.svg                    – Heatmap: avg rank per algo×ratio across instances
    8. hv_by_algo_across_ratios.svg        – Bar chart: per-algo avg HV (all ratios pooled)
    9. time_adherence.svg                  – Scatter: wall time vs PLS budget for 0:100 ratio
   10. hv_boxplot.svg                      – Box plot: HV distribution per algo (all ratios)
   11. hv_gain_from_seeds.svg              – Grouped bar: HV(50:50) − HV(0:100) per algo per tier
   12. concurrent_advantage_scaling.svg    – Line: concurrent vs exhaustive HV gap by tier & ratio
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import numpy as np

# ── Constants & Styling ──────────────────────────────────────────────────────

ALGO_ORDER = ["exhaustive", "concurrent", "probabilistic"]
ALGO_LABELS = {
    "exhaustive": "Exhaustive PLS",
    "concurrent": "Concurrent PLS",
    "probabilistic": "Probabilistic PLS",
    "seeds_only": "Seeds Only",
}
ALGO_SHORT = {
    "exhaustive": "Exhaustive",
    "concurrent": "Concurrent",
    "probabilistic": "Probabilistic",
    "seeds_only": "Seeds Only",
}
ALGO_COLORS = {
    "exhaustive": "#2196F3",
    "concurrent": "#4CAF50",
    "probabilistic": "#FF9800",
    "seeds_only": "#9E9E9E",
}
ALGO_MARKERS = {
    "exhaustive": "o",
    "concurrent": "s",
    "probabilistic": "D",
    "seeds_only": "x",
}

RATIO_ORDER = ["100:0", "50:50", "25:75", "0:100"]
RATIO_PLS = ["50:50", "25:75", "0:100"]  # ratios where PLS actually runs
RATIO_COLORS = {
    "100:0": "#9E9E9E",
    "50:50": "#2196F3",
    "25:75": "#FF9800",
    "0:100": "#E91E63",
}
RATIO_HATCHES = {
    "50:50": "",
    "25:75": "//",
    "0:100": "xx",
}

TIER_ORDER = ["small", "medium", "large"]
TIER_LABELS = {"small": "Small (30)", "medium": "Medium (50)", "large": "Large (100)"}
TIER_SIZES = {"small": 30, "medium": 50, "large": 100}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)

# ── Data loading ─────────────────────────────────────────────────────────────


def load_jsonl(path: str) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def group_by(records, key_fn):
    groups = defaultdict(list)
    for r in records:
        groups[key_fn(r)].append(r)
    return groups


def avg(vals):
    return sum(vals) / len(vals) if vals else 0.0


def extract_city(instance_name: str) -> str:
    parts = instance_name.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else instance_name


# ── Plot 1: Grouped bar — avg HV per algo×ratio, grouped by size tier ───────


def plot_hv_by_algo_ratio_grouped(data: list[dict], out_dir: str):
    """For each size tier, show bars for each (algo, ratio) combo."""
    # Filter to PLS ratios only
    pls_data = [r for r in data if r["algorithm_kind"] != "seeds_only"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax_idx, tier in enumerate(TIER_ORDER):
        ax = axes[ax_idx]
        tier_data = [r for r in pls_data if r["size_tier"] == tier]

        combos = [(a, rat) for a in ALGO_ORDER for rat in RATIO_PLS]
        n_combos = len(combos)
        x = np.arange(len(ALGO_ORDER))
        n_ratios = len(RATIO_PLS)
        bar_width = 0.75 / n_ratios

        for ri, ratio in enumerate(RATIO_PLS):
            means = []
            stds = []
            for algo in ALGO_ORDER:
                vals = [
                    r["hypervolume"]
                    for r in tier_data
                    if r["algorithm_kind"] == algo and r["ratio_label"] == ratio
                ]
                means.append(avg(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0)

            offset = (ri - n_ratios / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset,
                means,
                bar_width * 0.9,
                label=ratio if ax_idx == 0 else None,
                color=RATIO_COLORS[ratio],
                hatch=RATIO_HATCHES.get(ratio, ""),
                edgecolor="white",
                linewidth=0.5,
                yerr=stds,
                capsize=2,
                error_kw={"linewidth": 0.8},
            )
            for bar, val in zip(bars, means):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        rotation=45,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([ALGO_SHORT[a] for a in ALGO_ORDER], fontsize=10)
        ax.set_title(TIER_LABELS[tier], fontweight="bold")
        ax.set_ylim(bottom=0.75, top=1.0)
        ax.grid(axis="y", alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel("Normalised Hypervolume")

    axes[0].legend(
        title="Exact:PLS Ratio", fontsize=9, title_fontsize=9, loc="lower left"
    )

    fig.suptitle(
        "Average Hypervolume by Algorithm, Ratio, and Instance Size",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(out_dir, "hv_by_algo_ratio_grouped.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 2: Heatmap — HV per (algo×ratio × instance) ────────────────────────


def plot_hv_heatmap(data: list[dict], out_dir: str):
    # Rows: algo×ratio combos (including seeds_only)
    row_keys = [("seeds_only", "100:0")]
    for algo in ALGO_ORDER:
        for ratio in RATIO_PLS:
            row_keys.append((algo, ratio))

    row_labels = []
    for algo, ratio in row_keys:
        if algo == "seeds_only":
            row_labels.append("Seeds Only (100:0)")
        else:
            row_labels.append(f"{ALGO_SHORT[algo]} ({ratio})")

    # Columns: instances sorted by tier then name
    instances = sorted(
        set(r["instance"] for r in data),
        key=lambda n: (
            TIER_ORDER.index(next(r["size_tier"] for r in data if r["instance"] == n)),
            n,
        ),
    )

    # Build matrix
    matrix = np.zeros((len(row_keys), len(instances)))
    for ci, inst in enumerate(instances):
        for ri, (algo, ratio) in enumerate(row_keys):
            matches = [
                r
                for r in data
                if r["instance"] == inst
                and r["algorithm_kind"] == algo
                and r["ratio_label"] == ratio
            ]
            if matches:
                matrix[ri, ci] = matches[0]["hypervolume"]

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=1.0)

    ax.set_xticks(np.arange(len(instances)))
    inst_labels = [n.replace("_", "\n", 1).rsplit("_", 1) for n in instances]
    ax.set_xticklabels(
        [f"{extract_city(n)}\n{n.rsplit('_', 1)[1]}" for n in instances],
        fontsize=7,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    # Annotate cells
    for ri in range(len(row_keys)):
        for ci in range(len(instances)):
            val = matrix[ri, ci]
            color = "white" if val < 0.55 else "black"
            ax.text(
                ci,
                ri,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color=color,
                fontweight="bold",
            )

    # Tier separators
    sizes = [
        sum(
            1
            for n in instances
            if next(r["size_tier"] for r in data if r["instance"] == n) == t
        )
        for t in TIER_ORDER
    ]
    cumsum = 0
    for i, s in enumerate(sizes[:-1]):
        cumsum += s
        ax.axvline(cumsum - 0.5, color="white", linewidth=2)

    # Algo separators on y-axis
    ax.axhline(0.5, color="white", linewidth=2)  # after seeds_only
    ax.axhline(3.5, color="white", linewidth=1.5)
    ax.axhline(6.5, color="white", linewidth=1.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Normalised Hypervolume", fontsize=10)

    ax.set_title(
        "Hypervolume Heatmap: Algorithm × Ratio × Instance",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    path = os.path.join(out_dir, "hv_heatmap.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 3: Line chart — HV vs ratio (seed effect) per algo ─────────────────


def plot_hv_seed_effect(data: list[dict], out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    ratios_x = {"50:50": 0, "25:75": 1, "0:100": 2}

    for algo in ALGO_ORDER:
        means = []
        stds = []
        for ratio in RATIO_PLS:
            vals = [
                r["hypervolume"]
                for r in data
                if r["algorithm_kind"] == algo and r["ratio_label"] == ratio
            ]
            means.append(avg(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0)

        x = [ratios_x[r] for r in RATIO_PLS]
        ax.errorbar(
            x,
            means,
            yerr=stds,
            label=ALGO_LABELS[algo],
            color=ALGO_COLORS[algo],
            marker=ALGO_MARKERS[algo],
            markersize=10,
            linewidth=2.5,
            capsize=5,
            capthick=1.5,
        )

    # Add seeds_only reference line
    seeds_hv = avg(
        [r["hypervolume"] for r in data if r["algorithm_kind"] == "seeds_only"]
    )
    ax.axhline(
        seeds_hv,
        color=ALGO_COLORS["seeds_only"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Seeds Only (HV={seeds_hv:.3f})",
    )

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(
        ["50:50\n(half exact, half PLS)", "25:75\n(quarter exact)", "0:100\n(PLS only)"]
    )
    ax.set_ylabel("Normalised Hypervolume (avg across all instances)")
    ax.set_title(
        "Effect of Exact-Phase Seeds on PLS Quality", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "hv_seed_effect.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 4: Seed effect broken down by tier ──────────────────────────────────


def plot_hv_seed_effect_by_tier(data: list[dict], out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)

    for ax_idx, tier in enumerate(TIER_ORDER):
        ax = axes[ax_idx]
        tier_data = [r for r in data if r["size_tier"] == tier]

        for algo in ALGO_ORDER:
            means = []
            for ratio in RATIO_PLS:
                vals = [
                    r["hypervolume"]
                    for r in tier_data
                    if r["algorithm_kind"] == algo and r["ratio_label"] == ratio
                ]
                means.append(avg(vals))

            x = [0, 1, 2]
            ax.plot(
                x,
                means,
                label=ALGO_LABELS[algo] if ax_idx == 0 else None,
                color=ALGO_COLORS[algo],
                marker=ALGO_MARKERS[algo],
                markersize=8,
                linewidth=2,
            )

        seeds_hv = avg(
            [r["hypervolume"] for r in tier_data if r["algorithm_kind"] == "seeds_only"]
        )
        ax.axhline(
            seeds_hv,
            color=ALGO_COLORS["seeds_only"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
        )
        ax.text(
            2.05,
            seeds_hv,
            f"seeds\n{seeds_hv:.3f}",
            fontsize=7,
            va="center",
            color=ALGO_COLORS["seeds_only"],
        )

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["50:50", "25:75", "0:100"], fontsize=10)
        ax.set_xlabel("Exact:PLS Ratio")
        ax.set_title(TIER_LABELS[tier], fontweight="bold")
        ax.grid(alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel("Normalised Hypervolume")

    axes[0].legend(fontsize=9, loc="best")
    fig.suptitle("Seed Effect by Instance Size Tier", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    path = os.path.join(out_dir, "hv_seed_effect_by_tier.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 5: Front size grouped bar ───────────────────────────────────────────


def plot_front_size_grouped(data: list[dict], out_dir: str):
    pls_data = [r for r in data if r["algorithm_kind"] != "seeds_only"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax_idx, tier in enumerate(TIER_ORDER):
        ax = axes[ax_idx]
        tier_data = [r for r in pls_data if r["size_tier"] == tier]

        x = np.arange(len(ALGO_ORDER))
        n_ratios = len(RATIO_PLS)
        bar_width = 0.75 / n_ratios

        for ri, ratio in enumerate(RATIO_PLS):
            means = []
            for algo in ALGO_ORDER:
                vals = [
                    r["archive_size"]
                    for r in tier_data
                    if r["algorithm_kind"] == algo and r["ratio_label"] == ratio
                ]
                means.append(avg(vals))

            offset = (ri - n_ratios / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset,
                means,
                bar_width * 0.9,
                label=ratio if ax_idx == 0 else None,
                color=RATIO_COLORS[ratio],
                hatch=RATIO_HATCHES.get(ratio, ""),
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, val in zip(bars, means):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(means) * 0.01,
                        f"{int(val)}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        rotation=45,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([ALGO_SHORT[a] for a in ALGO_ORDER], fontsize=10)
        ax.set_title(TIER_LABELS[tier], fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel("Avg Pareto Front Size")

    axes[0].legend(
        title="Exact:PLS Ratio", fontsize=9, title_fontsize=9, loc="upper left"
    )
    fig.suptitle(
        "Pareto Front Size by Algorithm, Ratio, and Instance Size",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(out_dir, "front_size_grouped.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 6: Concurrent / Exhaustive front-size ratio ─────────────────────────


def plot_front_size_ratio(data: list[dict], out_dir: str):
    pls_data = [r for r in data if r["algorithm_kind"] in ("exhaustive", "concurrent")]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(TIER_ORDER))
    n_ratios = len(RATIO_PLS)
    bar_width = 0.7 / n_ratios

    for ri, ratio in enumerate(RATIO_PLS):
        ratios_val = []
        for tier in TIER_ORDER:
            instances = sorted(
                set(r["instance"] for r in pls_data if r["size_tier"] == tier)
            )
            inst_ratios = []
            for inst in instances:
                conc = [
                    r["archive_size"]
                    for r in pls_data
                    if r["instance"] == inst
                    and r["algorithm_kind"] == "concurrent"
                    and r["ratio_label"] == ratio
                ]
                exh = [
                    r["archive_size"]
                    for r in pls_data
                    if r["instance"] == inst
                    and r["algorithm_kind"] == "exhaustive"
                    and r["ratio_label"] == ratio
                ]
                if conc and exh and exh[0] > 0:
                    inst_ratios.append(conc[0] / exh[0])
            ratios_val.append(avg(inst_ratios) if inst_ratios else 1.0)

        offset = (ri - n_ratios / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            ratios_val,
            bar_width * 0.9,
            label=ratio,
            color=RATIO_COLORS[ratio],
            hatch=RATIO_HATCHES.get(ratio, ""),
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, ratios_val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}×",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in TIER_ORDER])
    ax.set_ylabel("Front Size Ratio (Concurrent / Exhaustive)")
    ax.set_title(
        "Concurrent PLS Front Size Advantage Over Exhaustive PLS",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(title="Exact:PLS Ratio", fontsize=10, title_fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0.9)

    fig.tight_layout()
    path = os.path.join(out_dir, "front_size_ratio_concurrent_exh.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 7: Rank heatmap ────────────────────────────────────────────────────


def plot_rank_heatmap(data: list[dict], out_dir: str):
    combos = [(a, r) for a in ALGO_ORDER for r in RATIO_PLS]
    combo_labels = [f"{ALGO_SHORT[a]} ({r})" for a, r in combos]

    instances = sorted(set(r["instance"] for r in data))

    # For each instance, rank the 9 combos by HV (descending)
    rank_matrix = np.zeros((len(combos), len(TIER_ORDER)))
    rank_counts = defaultdict(lambda: defaultdict(list))

    for inst in instances:
        tier = next(r["size_tier"] for r in data if r["instance"] == inst)
        inst_scores = []
        for algo, ratio in combos:
            matches = [
                r
                for r in data
                if r["instance"] == inst
                and r["algorithm_kind"] == algo
                and r["ratio_label"] == ratio
            ]
            hv = matches[0]["hypervolume"] if matches else 0
            inst_scores.append(hv)

        # Rank (1 = best)
        sorted_indices = np.argsort(inst_scores)[::-1]
        ranks = np.zeros(len(combos))
        for rank_pos, idx in enumerate(sorted_indices):
            ranks[idx] = rank_pos + 1

        for ci, (algo, ratio) in enumerate(combos):
            rank_counts[(algo, ratio)][tier].append(ranks[ci])

    # Build avg-rank matrix: rows=combos, cols=tiers
    matrix = np.zeros((len(combos), len(TIER_ORDER)))
    for ri, (algo, ratio) in enumerate(combos):
        for ti, tier in enumerate(TIER_ORDER):
            vals = rank_counts[(algo, ratio)].get(tier, [])
            matrix[ri, ti] = avg(vals) if vals else 5.0

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=9)

    ax.set_xticks(np.arange(len(TIER_ORDER)))
    ax.set_xticklabels([TIER_LABELS[t] for t in TIER_ORDER], fontsize=11)
    ax.set_yticks(np.arange(len(combo_labels)))
    ax.set_yticklabels(combo_labels, fontsize=9)

    for ri in range(len(combos)):
        for ti in range(len(TIER_ORDER)):
            val = matrix[ri, ti]
            color = "white" if val > 6 else "black"
            ax.text(
                ti,
                ri,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=10,
                color=color,
                fontweight="bold",
            )

    # Algo group separators
    ax.axhline(2.5, color="white", linewidth=2)
    ax.axhline(5.5, color="white", linewidth=2)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Average Rank (lower is better)", fontsize=10)

    ax.set_title(
        "Average HV-Based Rank by Size Tier\n(1 = best among 9 algo×ratio combos)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    path = os.path.join(out_dir, "rank_heatmap.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 8: Per-algo avg HV (pooled across ratios) ──────────────────────────


def plot_hv_by_algo(data: list[dict], out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))

    algos = ALGO_ORDER + ["seeds_only"]
    means = []
    stds = []
    colors = []
    labels = []

    for algo in algos:
        vals = [r["hypervolume"] for r in data if r["algorithm_kind"] == algo]
        means.append(avg(vals))
        stds.append(np.std(vals) if len(vals) > 1 else 0)
        colors.append(ALGO_COLORS[algo])
        labels.append(ALGO_LABELS[algo])

    x = np.arange(len(algos))
    bars = ax.bar(
        x,
        means,
        0.6,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        yerr=stds,
        capsize=5,
        error_kw={"linewidth": 1.2},
    )

    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Normalised Hypervolume")
    ax.set_title(
        "Average Hypervolume by Algorithm (All Ratios Pooled)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0.4)

    fig.tight_layout()
    path = os.path.join(out_dir, "hv_by_algo_across_ratios.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 9: Time adherence (0:100 ratio) ────────────────────────────────────


def plot_time_adherence(data: list[dict], out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    budgets = {"small": 10000, "medium": 20000, "large": 50000}

    for algo in ALGO_ORDER:
        algo_data = [
            r
            for r in data
            if r["algorithm_kind"] == algo and r["ratio_label"] == "0:100"
        ]
        xs = []
        ys = []
        for r in algo_data:
            budget = budgets.get(r["size_tier"], 50000)
            xs.append(budget / 1000)
            ys.append(r["wall_time_ms"] / 1000)

        ax.scatter(
            xs,
            ys,
            label=ALGO_LABELS[algo],
            color=ALGO_COLORS[algo],
            marker=ALGO_MARKERS[algo],
            s=80,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
        )

    # Perfect adherence line
    max_budget = 55
    ax.plot(
        [0, max_budget],
        [0, max_budget],
        "k--",
        linewidth=1,
        alpha=0.5,
        label="Perfect adherence",
    )

    ax.set_xlabel("PLS Budget (seconds)")
    ax.set_ylabel("Actual Wall Time (seconds)")
    ax.set_title(
        "Time Budget Adherence (0:100 Ratio — Full PLS Budget)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    path = os.path.join(out_dir, "time_adherence.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 10: HV distribution box plot ───────────────────────────────────────


def plot_hv_boxplot(data: list[dict], out_dir: str):
    fig, ax = plt.subplots(figsize=(12, 6))

    box_data = []
    box_labels = []
    box_colors = []

    for algo in ALGO_ORDER:
        for ratio in RATIO_PLS:
            vals = [
                r["hypervolume"]
                for r in data
                if r["algorithm_kind"] == algo and r["ratio_label"] == ratio
            ]
            box_data.append(vals)
            box_labels.append(f"{ALGO_SHORT[algo]}\n({ratio})")
            box_colors.append(ALGO_COLORS[algo])

    # Add seeds_only
    seeds_vals = [r["hypervolume"] for r in data if r["algorithm_kind"] == "seeds_only"]
    box_data.append(seeds_vals)
    box_labels.append("Seeds\nOnly")
    box_colors.append(ALGO_COLORS["seeds_only"])

    bp = ax.boxplot(
        box_data,
        patch_artist=True,
        notch=True,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 1},
        flierprops={"markersize": 4},
    )

    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(box_labels, fontsize=8)
    ax.set_ylabel("Normalised Hypervolume")
    ax.set_title(
        "Hypervolume Distribution Across Instances", fontsize=13, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)

    # Add algo group labels
    for i, algo in enumerate(ALGO_ORDER):
        mid = i * 3 + 2  # middle of the 3 boxes for this algo
        ax.text(
            mid,
            ax.get_ylim()[0] - 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            ALGO_LABELS[algo],
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=ALGO_COLORS[algo],
        )

    fig.tight_layout()
    path = os.path.join(out_dir, "hv_boxplot.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 11: HV gain from seeds (50:50 minus 0:100) per algo per tier ───────


def plot_hv_gain_from_seeds(data: list[dict], out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(TIER_ORDER))
    n_algos = len(ALGO_ORDER)
    bar_width = 0.7 / n_algos

    for ai, algo in enumerate(ALGO_ORDER):
        gains = []
        errs = []
        for tier in TIER_ORDER:
            tier_data = [r for r in data if r["size_tier"] == tier]
            instances = sorted(set(r["instance"] for r in tier_data))
            inst_gains = []
            for inst in instances:
                hv_50 = next(
                    (
                        r["hypervolume"]
                        for r in tier_data
                        if r["instance"] == inst
                        and r["algorithm_kind"] == algo
                        and r["ratio_label"] == "50:50"
                    ),
                    None,
                )
                hv_0 = next(
                    (
                        r["hypervolume"]
                        for r in tier_data
                        if r["instance"] == inst
                        and r["algorithm_kind"] == algo
                        and r["ratio_label"] == "0:100"
                    ),
                    None,
                )
                if hv_50 is not None and hv_0 is not None:
                    inst_gains.append(hv_50 - hv_0)
            gains.append(avg(inst_gains) if inst_gains else 0)
            errs.append(np.std(inst_gains) if len(inst_gains) > 1 else 0)

        offset = (ai - n_algos / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            gains,
            bar_width * 0.9,
            label=ALGO_LABELS[algo],
            color=ALGO_COLORS[algo],
            edgecolor="white",
            linewidth=0.5,
            yerr=errs,
            capsize=3,
            error_kw={"linewidth": 0.8},
        )
        for bar, val in zip(bars, gains):
            sign = "+" if val >= 0 else ""
            y_pos = bar.get_height() + 0.001 if val >= 0 else bar.get_height() - 0.003
            va = "bottom" if val >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{sign}{val:.4f}",
                ha="center",
                va=va,
                fontsize=7.5,
                fontweight="bold",
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in TIER_ORDER])
    ax.set_ylabel("HV(50:50) − HV(0:100)")
    ax.set_title(
        "Hypervolume Gain from Exact-Phase Seeds\n(50:50 vs 0:100, positive = seeds help)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "hv_gain_from_seeds.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 12: Concurrent advantage scaling ───────────────────────────────────


def plot_concurrent_advantage(data: list[dict], out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: HV improvement % (concurrent over exhaustive)
    ax = axes[0]
    for ratio in RATIO_PLS:
        improvements = []
        for tier in TIER_ORDER:
            tier_data = [r for r in data if r["size_tier"] == tier]
            instances = sorted(set(r["instance"] for r in tier_data))
            inst_imps = []
            for inst in instances:
                hv_c = next(
                    (
                        r["hypervolume"]
                        for r in tier_data
                        if r["instance"] == inst
                        and r["algorithm_kind"] == "concurrent"
                        and r["ratio_label"] == ratio
                    ),
                    None,
                )
                hv_e = next(
                    (
                        r["hypervolume"]
                        for r in tier_data
                        if r["instance"] == inst
                        and r["algorithm_kind"] == "exhaustive"
                        and r["ratio_label"] == ratio
                    ),
                    None,
                )
                if hv_c and hv_e and hv_e > 0:
                    inst_imps.append((hv_c - hv_e) / hv_e * 100)
            improvements.append(avg(inst_imps) if inst_imps else 0)

        ax.plot(
            [0, 1, 2],
            improvements,
            label=ratio,
            color=RATIO_COLORS[ratio],
            marker="o",
            linewidth=2.5,
            markersize=8,
        )
        for xi, val in enumerate(improvements):
            ax.annotate(
                f"{val:+.2f}%",
                (xi, val),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=8,
            )

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([TIER_LABELS[t] for t in TIER_ORDER])
    ax.set_ylabel("HV Improvement (%)")
    ax.set_title("Concurrent vs Exhaustive:\nHV Improvement", fontweight="bold")
    ax.legend(title="Exact:PLS Ratio", fontsize=9, title_fontsize=9)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    # Right: Front size ratio
    ax = axes[1]
    for ratio in RATIO_PLS:
        size_ratios = []
        for tier in TIER_ORDER:
            tier_data = [r for r in data if r["size_tier"] == tier]
            instances = sorted(set(r["instance"] for r in tier_data))
            inst_rats = []
            for inst in instances:
                fs_c = next(
                    (
                        r["archive_size"]
                        for r in tier_data
                        if r["instance"] == inst
                        and r["algorithm_kind"] == "concurrent"
                        and r["ratio_label"] == ratio
                    ),
                    None,
                )
                fs_e = next(
                    (
                        r["archive_size"]
                        for r in tier_data
                        if r["instance"] == inst
                        and r["algorithm_kind"] == "exhaustive"
                        and r["ratio_label"] == ratio
                    ),
                    None,
                )
                if fs_c and fs_e and fs_e > 0:
                    inst_rats.append(fs_c / fs_e)
            size_ratios.append(avg(inst_rats) if inst_rats else 1.0)

        ax.plot(
            [0, 1, 2],
            size_ratios,
            label=ratio,
            color=RATIO_COLORS[ratio],
            marker="s",
            linewidth=2.5,
            markersize=8,
        )
        for xi, val in enumerate(size_ratios):
            ax.annotate(
                f"{val:.2f}×",
                (xi, val),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=8,
            )

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([TIER_LABELS[t] for t in TIER_ORDER])
    ax.set_ylabel("Front Size Ratio (Concurrent / Exhaustive)")
    ax.set_title("Concurrent vs Exhaustive:\nFront Size Multiplier", fontweight="bold")
    ax.legend(title="Exact:PLS Ratio", fontsize=9, title_fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Concurrent PLS Advantage Scaling with Instance Size",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    path = os.path.join(out_dir, "concurrent_advantage_scaling.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 13: Per-instance faceted bar plot (5 cities × 3 tiers) ─────────────


def plot_per_instance_bars(data: list[dict], out_dir: str):
    """Faceted small-multiple bar plot: one subplot per instance arranged as
    3 rows (size tiers) × 5 columns (cities).  Each subplot shows bars for
    the 9 algo×ratio combos plus a seeds-only reference line."""

    cities_order = [
        "lagos_nigeria",
        "mexico_city",
        "paris",
        "rio_de_janeiro",
        "tokyo_bay",
    ]
    city_labels = {
        "lagos_nigeria": "Lagos",
        "mexico_city": "Mexico City",
        "paris": "Paris",
        "rio_de_janeiro": "Rio de Janeiro",
        "tokyo_bay": "Tokyo Bay",
    }

    n_rows = len(TIER_ORDER)
    n_cols = len(cities_order)

    # Combos shown as bars (seeds_only is a reference line, not a bar)
    combos = [(a, r) for a in ALGO_ORDER for r in RATIO_PLS]
    combo_short = []
    for a, r in combos:
        combo_short.append(f"{ALGO_SHORT[a][0]}{r.split(':')[0]}")  # e.g. E50, C25, P0

    # Assign a colour per combo: algo colour with ratio-based lightness
    import matplotlib.colors as mcolors

    def lighten(hex_color, amount):
        rgb = mcolors.to_rgb(hex_color)
        return tuple(min(1.0, c + (1.0 - c) * amount) for c in rgb)

    ratio_lightness = {"50:50": 0.0, "25:75": 0.25, "0:100": 0.50}
    combo_colors = []
    combo_edge = []
    for a, r in combos:
        base = ALGO_COLORS[a]
        combo_colors.append(lighten(base, ratio_lightness[r]))
        combo_edge.append(base)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(24, 13), sharey="row", squeeze=False
    )

    # Global y-range per row for consistent comparison within a tier
    for row_idx, tier in enumerate(TIER_ORDER):
        for col_idx, city in enumerate(cities_order):
            ax = axes[row_idx][col_idx]
            inst_name = f"{city}_{TIER_SIZES[tier]}"

            # Collect HV for each combo
            hvs = []
            for a, r in combos:
                matches = [
                    rec
                    for rec in data
                    if rec["instance"] == inst_name
                    and rec["algorithm_kind"] == a
                    and rec["ratio_label"] == r
                ]
                hvs.append(matches[0]["hypervolume"] if matches else 0)

            # Seeds-only HV for reference line
            seeds_match = [
                rec
                for rec in data
                if rec["instance"] == inst_name
                and rec["algorithm_kind"] == "seeds_only"
            ]
            seeds_hv = seeds_match[0]["hypervolume"] if seeds_match else 0

            x = np.arange(len(combos))
            bars = ax.bar(
                x,
                hvs,
                0.8,
                color=combo_colors,
                edgecolor=combo_edge,
                linewidth=0.7,
            )

            # Reference line for seeds-only
            ax.axhline(
                seeds_hv,
                color=ALGO_COLORS["seeds_only"],
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
            )

            # Find best combo and annotate
            if hvs:
                best_idx = int(np.argmax(hvs))
                bars[best_idx].set_edgecolor("black")
                bars[best_idx].set_linewidth(2.0)

            # Value labels on bars
            for bar, val in zip(bars, hvs):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=5,
                        rotation=90,
                    )

            # Axis formatting
            ax.set_xticks(x)
            if row_idx == n_rows - 1:
                ax.set_xticklabels(combo_short, fontsize=5.5, rotation=90)
            else:
                ax.set_xticklabels([])

            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)

            # Titles: city on top row, tier on left column
            if row_idx == 0:
                ax.set_title(city_labels[city], fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(
                    f"{TIER_LABELS[tier]}\nHV",
                    fontsize=9,
                    fontweight="bold",
                )

            # Set y-limits to zoom in on interesting range
            if hvs:
                ymin = max(0, min(v for v in hvs if v > 0) - 0.06)
                ymax = max(hvs) + 0.04
                # Include seeds line in range
                ymin = min(ymin, seeds_hv - 0.03) if seeds_hv > 0 else ymin
                ax.set_ylim(bottom=max(0, ymin), top=min(1.02, ymax))

    # Build a legend
    legend_handles = []
    for a in ALGO_ORDER:
        for r in RATIO_PLS:
            c = lighten(ALGO_COLORS[a], ratio_lightness[r])
            ec = ALGO_COLORS[a]
            label = f"{ALGO_SHORT[a]} ({r})"
            legend_handles.append(
                mpatches.Patch(facecolor=c, edgecolor=ec, linewidth=0.8, label=label)
            )
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            color=ALGO_COLORS["seeds_only"],
            linestyle="--",
            linewidth=1.5,
            label="Seeds Only (100:0)",
        )
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        fontsize=8.5,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "Hypervolume per Instance: Algorithm × Ratio\n"
        "(black border = best configuration; dashed line = seeds-only baseline)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.95])

    path = os.path.join(out_dir, "hv_per_instance_bars.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 14: Per-instance front-size faceted bar plot ────────────────────────


def plot_per_instance_front_size_bars(data: list[dict], out_dir: str):
    """Same faceted layout as plot 13 but showing Pareto front size instead of HV."""

    cities_order = [
        "lagos_nigeria",
        "mexico_city",
        "paris",
        "rio_de_janeiro",
        "tokyo_bay",
    ]
    city_labels = {
        "lagos_nigeria": "Lagos",
        "mexico_city": "Mexico City",
        "paris": "Paris",
        "rio_de_janeiro": "Rio de Janeiro",
        "tokyo_bay": "Tokyo Bay",
    }

    n_rows = len(TIER_ORDER)
    n_cols = len(cities_order)

    combos = [(a, r) for a in ALGO_ORDER for r in RATIO_PLS]
    combo_short = []
    for a, r in combos:
        combo_short.append(f"{ALGO_SHORT[a][0]}{r.split(':')[0]}")

    import matplotlib.colors as mcolors

    def lighten(hex_color, amount):
        rgb = mcolors.to_rgb(hex_color)
        return tuple(min(1.0, c + (1.0 - c) * amount) for c in rgb)

    ratio_lightness = {"50:50": 0.0, "25:75": 0.25, "0:100": 0.50}
    combo_colors = []
    combo_edge = []
    for a, r in combos:
        base = ALGO_COLORS[a]
        combo_colors.append(lighten(base, ratio_lightness[r]))
        combo_edge.append(base)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(24, 13), sharey="row", squeeze=False
    )

    for row_idx, tier in enumerate(TIER_ORDER):
        for col_idx, city in enumerate(cities_order):
            ax = axes[row_idx][col_idx]
            inst_name = f"{city}_{TIER_SIZES[tier]}"

            fronts = []
            for a, r in combos:
                matches = [
                    rec
                    for rec in data
                    if rec["instance"] == inst_name
                    and rec["algorithm_kind"] == a
                    and rec["ratio_label"] == r
                ]
                fronts.append(matches[0]["archive_size"] if matches else 0)

            seeds_match = [
                rec
                for rec in data
                if rec["instance"] == inst_name
                and rec["algorithm_kind"] == "seeds_only"
            ]
            seeds_front = seeds_match[0]["archive_size"] if seeds_match else 0

            x = np.arange(len(combos))
            bars = ax.bar(
                x,
                fronts,
                0.8,
                color=combo_colors,
                edgecolor=combo_edge,
                linewidth=0.7,
            )

            ax.axhline(
                seeds_front,
                color=ALGO_COLORS["seeds_only"],
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
            )

            if fronts:
                best_idx = int(np.argmax(fronts))
                bars[best_idx].set_edgecolor("black")
                bars[best_idx].set_linewidth(2.0)

            for bar, val in zip(bars, fronts):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(fronts) * 0.01,
                        f"{int(val)}",
                        ha="center",
                        va="bottom",
                        fontsize=5,
                        rotation=90,
                    )

            ax.set_xticks(x)
            if row_idx == n_rows - 1:
                ax.set_xticklabels(combo_short, fontsize=5.5, rotation=90)
            else:
                ax.set_xticklabels([])

            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)

            if row_idx == 0:
                ax.set_title(city_labels[city], fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(
                    f"{TIER_LABELS[tier]}\nFront Size",
                    fontsize=9,
                    fontweight="bold",
                )

    legend_handles = []
    for a in ALGO_ORDER:
        for r in RATIO_PLS:
            c = lighten(ALGO_COLORS[a], ratio_lightness[r])
            ec = ALGO_COLORS[a]
            label = f"{ALGO_SHORT[a]} ({r})"
            legend_handles.append(
                mpatches.Patch(facecolor=c, edgecolor=ec, linewidth=0.8, label=label)
            )
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            color=ALGO_COLORS["seeds_only"],
            linestyle="--",
            linewidth=1.5,
            label="Seeds Only (100:0)",
        )
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        fontsize=8.5,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "Pareto Front Size per Instance: Algorithm × Ratio\n"
        "(black border = largest front; dashed line = seeds-only baseline)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.95])

    path = os.path.join(out_dir, "front_size_per_instance_bars.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate hybrid benchmark plots from JSONL data."
    )
    parser.add_argument(
        "--jsonl-file", required=True, help="Path to JSONL checkpoint file"
    )
    parser.add_argument(
        "--output-dir",
        default="docs/hybrid_benchmark_plots",
        help="Directory for output SVG files",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.jsonl_file}...")
    data = load_jsonl(args.jsonl_file)
    print(f"  Loaded {len(data)} records")
    print(f"  Instances: {len(set(r['instance'] for r in data))}")
    print(f"  Algorithms: {sorted(set(r['algorithm_kind'] for r in data))}")
    print(f"  Ratios: {sorted(set(r['ratio_label'] for r in data))}")
    print()

    print("Generating plots...")
    plot_hv_by_algo_ratio_grouped(data, args.output_dir)
    plot_hv_heatmap(data, args.output_dir)
    plot_hv_seed_effect(data, args.output_dir)
    plot_hv_seed_effect_by_tier(data, args.output_dir)
    plot_front_size_grouped(data, args.output_dir)
    plot_front_size_ratio(data, args.output_dir)
    plot_rank_heatmap(data, args.output_dir)
    plot_hv_by_algo(data, args.output_dir)
    plot_time_adherence(data, args.output_dir)
    plot_hv_boxplot(data, args.output_dir)
    plot_hv_gain_from_seeds(data, args.output_dir)
    plot_concurrent_advantage(data, args.output_dir)
    plot_per_instance_bars(data, args.output_dir)
    plot_per_instance_front_size_bars(data, args.output_dir)

    print(f"\nDone! {14} plots written to {args.output_dir}/")


if __name__ == "__main__":
    main()
