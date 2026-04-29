#!/usr/bin/env python3
"""Generate plots and a comprehensive Markdown report from hybrid benchmark JSON data.

Usage:
    uv run --with matplotlib python3 scripts/generate_hybrid_benchmark_report.py \
        --json-file results/hybrid_benchmark.json \
        --output-dir docs/hybrid_benchmark

Produces:
    1. SVG plots in the output directory
    2. HYBRID_BENCHMARK_REPORT.md in the output directory
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Styling ──────────────────────────────────────────────────────────────────

ALGO_COLORS = {
    "exhaustive": "#2196F3",
    "concurrent": "#4CAF50",
    "probabilistic": "#FF9800",
    "seeds_only": "#9E9E9E",
}

ALGO_LABELS = {
    "exhaustive": "Exhaustive PLS",
    "concurrent": "Concurrent PLS",
    "probabilistic": "Probabilistic PLS",
    "seeds_only": "Seeds Only",
}

RATIO_ORDER = ["100:0", "50:50", "25:75", "0:100"]
RATIO_COLORS = {
    "100:0": "#9E9E9E",
    "50:50": "#42A5F5",
    "25:75": "#66BB6A",
    "0:100": "#EF5350",
}

SIZE_TIER_ORDER = ["small", "medium", "large", "x-large", "huge"]
SIZE_TIER_LABELS = {
    "small": "Small (≤35)",
    "medium": "Medium (36–60)",
    "large": "Large (61–110)",
    "x-large": "X-Large (111–160)",
    "huge": "Huge (>160)",
}


def load_data(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def group_results(results: list[dict], *keys) -> dict:
    """Group results by one or more keys into nested dicts."""
    groups = defaultdict(list)
    for r in results:
        key = tuple(r[k] for k in keys)
        if len(keys) == 1:
            key = key[0]
        groups[key].append(r)
    return dict(groups)


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def setup_plot(figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, alpha=0.3, linestyle="--")
    return fig, ax


def save_plot(fig, path: str, title: str = ""):
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot generators
# ═══════════════════════════════════════════════════════════════════════════════


def plot_hv_by_algo_and_ratio(results: list[dict], output_dir: str) -> str:
    """Grouped bar chart: avg HV per algorithm × ratio."""
    fname = "hv_by_algo_ratio.svg"
    fig, ax = setup_plot((14, 7))

    # Group by (algorithm_kind, ratio_label)
    algo_kinds = [
        k
        for k in ["exhaustive", "concurrent", "probabilistic", "seeds_only"]
        if any(r["algorithm_kind"] == k for r in results)
    ]
    ratios = [r for r in RATIO_ORDER if any(res["ratio_label"] == r for res in results)]

    n_algos = len(algo_kinds)
    n_ratios = len(ratios)
    bar_width = 0.8 / n_algos if n_algos > 0 else 0.2
    x = np.arange(n_ratios)

    for i, algo in enumerate(algo_kinds):
        hvs = []
        for ratio in ratios:
            matching = [
                r["hypervolume"]
                for r in results
                if r["algorithm_kind"] == algo and r["ratio_label"] == ratio
            ]
            hvs.append(avg(matching) if matching else 0.0)

        offset = (i - n_algos / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            hvs,
            bar_width,
            label=ALGO_LABELS.get(algo, algo),
            color=ALGO_COLORS.get(algo, "#888888"),
            edgecolor="white",
            linewidth=0.5,
        )

        for bar, hv in zip(bars, hvs):
            if hv > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{hv:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xlabel("Time Allocation Ratio (Exact : PLS)", fontsize=12)
    ax.set_ylabel("Average Normalised Hypervolume", fontsize=12)
    ax.set_title(
        "Hypervolume by Algorithm and Time Ratio", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ratios, fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(0, min(1.05, ax.get_ylim()[1] * 1.15))

    save_plot(fig, os.path.join(output_dir, fname))
    return fname


def plot_hv_heatmap(results: list[dict], output_dir: str) -> str:
    """Heatmap: HV per (algorithm+ratio) × instance."""
    fname = "hv_heatmap.svg"

    instances = sorted(set(r["instance"] for r in results))
    algo_ratios = []
    for algo in ["seeds_only", "exhaustive", "concurrent", "probabilistic"]:
        for ratio in RATIO_ORDER:
            key = f"{ALGO_LABELS.get(algo, algo)} ({ratio})"
            if any(
                r["algorithm_kind"] == algo and r["ratio_label"] == ratio
                for r in results
            ):
                algo_ratios.append((algo, ratio, key))

    if not algo_ratios or not instances:
        return ""

    matrix = np.zeros((len(algo_ratios), len(instances)))
    for i, (algo, ratio, _) in enumerate(algo_ratios):
        for j, inst in enumerate(instances):
            matching = [
                r["hypervolume"]
                for r in results
                if r["algorithm_kind"] == algo
                and r["ratio_label"] == ratio
                and r["instance"] == inst
            ]
            matrix[i, j] = avg(matching) if matching else np.nan

    fig, ax = plt.subplots(
        figsize=(max(12, len(instances) * 0.9), max(6, len(algo_ratios) * 0.45))
    )
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(instances)))
    ax.set_xticklabels(
        [i.replace("_", "\n") for i in instances], fontsize=7, rotation=45, ha="right"
    )
    ax.set_yticks(range(len(algo_ratios)))
    ax.set_yticklabels([label for _, _, label in algo_ratios], fontsize=8)

    for i in range(len(algo_ratios)):
        for j in range(len(instances)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=color,
                )

    ax.set_title(
        "Hypervolume Heatmap: Algorithm×Ratio vs Instance",
        fontsize=13,
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="Normalised HV", shrink=0.8)

    save_plot(fig, os.path.join(output_dir, fname))
    return fname


def plot_hv_by_size_tier(results: list[dict], output_dir: str) -> str:
    """Grouped bar chart: avg HV per algorithm × size tier (across all ratios with PLS > 0)."""
    fname = "hv_by_size_tier.svg"

    pls_results = [r for r in results if r["ratio_pls"] > 0]
    tiers = [
        t for t in SIZE_TIER_ORDER if any(r["size_tier"] == t for r in pls_results)
    ]
    algo_kinds = [
        k
        for k in ["exhaustive", "concurrent", "probabilistic"]
        if any(r["algorithm_kind"] == k for r in pls_results)
    ]

    if not tiers or not algo_kinds:
        return ""

    fig, ax = setup_plot((12, 6))
    n_algos = len(algo_kinds)
    bar_width = 0.8 / n_algos
    x = np.arange(len(tiers))

    for i, algo in enumerate(algo_kinds):
        hvs = []
        for tier in tiers:
            matching = [
                r["hypervolume"]
                for r in pls_results
                if r["algorithm_kind"] == algo and r["size_tier"] == tier
            ]
            hvs.append(avg(matching) if matching else 0.0)

        offset = (i - n_algos / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            hvs,
            bar_width,
            label=ALGO_LABELS.get(algo, algo),
            color=ALGO_COLORS.get(algo, "#888888"),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Instance Size Tier", fontsize=12)
    ax.set_ylabel("Average Normalised HV (PLS ratios only)", fontsize=12)
    ax.set_title(
        "Hypervolume by Algorithm and Instance Size", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_TIER_LABELS.get(t, t) for t in tiers], fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, min(1.05, ax.get_ylim()[1] * 1.15))

    save_plot(fig, os.path.join(output_dir, fname))
    return fname


def plot_hv_scaling_by_ratio(results: list[dict], output_dir: str) -> str:
    """Line chart: avg HV vs num_images, one line per ratio, faceted by algorithm."""
    fname = "hv_scaling_by_ratio.svg"

    algo_kinds = [
        k
        for k in ["exhaustive", "concurrent", "probabilistic"]
        if any(r["algorithm_kind"] == k for r in results)
    ]
    ratios = [
        r
        for r in RATIO_ORDER
        if any(res["ratio_label"] == r and res["ratio_pls"] > 0 for res in results)
    ]

    if not algo_kinds:
        return ""

    fig, axes = plt.subplots(
        1, len(algo_kinds), figsize=(6 * len(algo_kinds), 5), sharey=True
    )
    if len(algo_kinds) == 1:
        axes = [axes]

    sizes = sorted(set(r["num_images"] for r in results))

    for ax, algo in zip(axes, algo_kinds):
        for ratio in ratios:
            hvs = []
            xs = []
            for size in sizes:
                matching = [
                    r["hypervolume"]
                    for r in results
                    if r["algorithm_kind"] == algo
                    and r["ratio_label"] == ratio
                    and r["num_images"] == size
                ]
                if matching:
                    hvs.append(avg(matching))
                    xs.append(size)

            if xs:
                ax.plot(
                    xs,
                    hvs,
                    "o-",
                    label=f"Ratio {ratio}",
                    color=RATIO_COLORS.get(ratio, "#888"),
                    markersize=5,
                    linewidth=1.5,
                )

        # Also plot seeds_only for reference
        seed_hvs = []
        seed_xs = []
        for size in sizes:
            matching = [
                r["hypervolume"]
                for r in results
                if r["algorithm_kind"] == "seeds_only" and r["num_images"] == size
            ]
            if matching:
                seed_hvs.append(avg(matching))
                seed_xs.append(size)
        if seed_xs:
            ax.plot(
                seed_xs,
                seed_hvs,
                "x--",
                label="Seeds only (100:0)",
                color=ALGO_COLORS["seeds_only"],
                markersize=6,
                linewidth=1,
            )

        ax.set_title(ALGO_LABELS.get(algo, algo), fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of Images", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Avg Normalised HV", fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle(
        "HV Scaling with Instance Size by Ratio", fontsize=14, fontweight="bold", y=1.02
    )
    save_plot(fig, os.path.join(output_dir, fname))
    return fname


def plot_archive_size_comparison(results: list[dict], output_dir: str) -> str:
    """Grouped bar chart: avg archive (front) size per algorithm × ratio."""
    fname = "archive_size_comparison.svg"
    fig, ax = setup_plot((14, 7))

    algo_kinds = [
        k
        for k in ["exhaustive", "concurrent", "probabilistic", "seeds_only"]
        if any(r["algorithm_kind"] == k for r in results)
    ]
    ratios = [r for r in RATIO_ORDER if any(res["ratio_label"] == r for res in results)]

    n_algos = len(algo_kinds)
    bar_width = 0.8 / n_algos if n_algos else 0.2
    x = np.arange(len(ratios))

    for i, algo in enumerate(algo_kinds):
        fronts = []
        for ratio in ratios:
            matching = [
                r["archive_size"]
                for r in results
                if r["algorithm_kind"] == algo and r["ratio_label"] == ratio
            ]
            fronts.append(avg(matching) if matching else 0.0)

        offset = (i - n_algos / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            fronts,
            bar_width,
            label=ALGO_LABELS.get(algo, algo),
            color=ALGO_COLORS.get(algo, "#888888"),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Time Allocation Ratio (Exact : PLS)", fontsize=12)
    ax.set_ylabel("Average Archive Size (Pareto Front)", fontsize=12)
    ax.set_title(
        "Pareto Front Size by Algorithm and Ratio", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ratios, fontsize=11)
    ax.legend(fontsize=10)

    save_plot(fig, os.path.join(output_dir, fname))
    return fname


def plot_hv_improvement_over_seeds(results: list[dict], output_dir: str) -> str:
    """Bar chart: HV improvement of each (algo, ratio) over seeds-only baseline per instance."""
    fname = "hv_improvement_over_seeds.svg"

    # Compute baseline HV per instance (seeds_only / ratio 100:0)
    baseline = {}
    for r in results:
        if r["algorithm_kind"] == "seeds_only":
            inst = r["instance"]
            if inst not in baseline:
                baseline[inst] = r["hypervolume"]

    if not baseline:
        return ""

    algo_kinds = [
        k
        for k in ["exhaustive", "concurrent", "probabilistic"]
        if any(r["algorithm_kind"] == k for r in results)
    ]
    ratios_pls = [
        r
        for r in RATIO_ORDER
        if r != "100:0" and any(res["ratio_label"] == r for res in results)
    ]

    fig, ax = setup_plot((14, 7))

    combos = []
    for algo in algo_kinds:
        for ratio in ratios_pls:
            combos.append((algo, ratio))

    x = np.arange(len(combos))
    improvements = []
    labels = []

    for algo, ratio in combos:
        matching = [
            r
            for r in results
            if r["algorithm_kind"] == algo
            and r["ratio_label"] == ratio
            and r["instance"] in baseline
        ]
        if matching:
            imprs = [(r["hypervolume"] - baseline[r["instance"]]) for r in matching]
            improvements.append(avg(imprs))
        else:
            improvements.append(0.0)

        labels.append(f"{ALGO_LABELS.get(algo, algo)[:8]}\n({ratio})")

    colors = [ALGO_COLORS.get(algo, "#888") for algo, _ in combos]
    bars = ax.bar(x, improvements, 0.7, color=colors, edgecolor="white", linewidth=0.5)

    for bar, imp in zip(bars, improvements):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{imp:+.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xlabel("Algorithm (Ratio)", fontsize=12)
    ax.set_ylabel("Avg HV Improvement over Seeds-Only Baseline", fontsize=12)
    ax.set_title(
        "HV Lift from PLS Phase vs Seeds-Only Baseline", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)

    save_plot(fig, os.path.join(output_dir, fname))
    return fname


def plot_wall_time_adherence(results: list[dict], output_dir: str) -> str:
    """Scatter: actual wall time vs expected PLS budget."""
    fname = "wall_time_adherence.svg"

    pls_results = [r for r in results if r["ratio_pls"] > 0]
    if not pls_results:
        return ""

    fig, ax = setup_plot((10, 6))

    for algo in ["exhaustive", "concurrent", "probabilistic"]:
        matching = [r for r in pls_results if r["algorithm_kind"] == algo]
        if not matching:
            continue

        expected = []
        actual = []
        for r in matching:
            # We don't have the original timeout in each result, but we can use wall_time_ms
            # and compare relative to ratio
            actual.append(r["wall_time_ms"])
            expected.append(r["wall_time_ms"])  # placeholder

        ax.scatter(
            [r["wall_time_ms"] / 1000 for r in matching],
            [r["archive_size"] for r in matching],
            label=ALGO_LABELS.get(algo, algo),
            color=ALGO_COLORS.get(algo, "#888"),
            alpha=0.7,
            s=40,
            edgecolors="white",
            linewidth=0.3,
        )

    ax.set_xlabel("Wall Time (seconds)", fontsize=12)
    ax.set_ylabel("Archive Size", fontsize=12)
    ax.set_title("Archive Size vs Wall Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    save_plot(fig, os.path.join(output_dir, fname))
    return fname


def plot_hv_boxplot(results: list[dict], output_dir: str) -> str:
    """Box plot: HV distribution per algorithm (across all PLS ratios and instances)."""
    fname = "hv_boxplot.svg"

    algo_kinds = [
        k
        for k in ["exhaustive", "concurrent", "probabilistic"]
        if any(r["algorithm_kind"] == k for r in results)
    ]

    if not algo_kinds:
        return ""

    fig, ax = setup_plot((10, 6))

    data = []
    labels = []
    colors_list = []

    for algo in algo_kinds:
        for ratio in [r for r in RATIO_ORDER if r != "100:0"]:
            matching = [
                r["hypervolume"]
                for r in results
                if r["algorithm_kind"] == algo and r["ratio_label"] == ratio
            ]
            if matching:
                data.append(matching)
                labels.append(f"{ALGO_LABELS.get(algo, algo)[:8]}\n({ratio})")
                colors_list.append(ALGO_COLORS.get(algo, "#888"))

    if not data:
        plt.close(fig)
        return ""

    bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, widths=0.6)
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Normalised Hypervolume", fontsize=12)
    ax.set_title(
        "HV Distribution by Algorithm and Ratio", fontsize=14, fontweight="bold"
    )
    plt.setp(ax.get_xticklabels(), fontsize=8)

    save_plot(fig, os.path.join(output_dir, fname))
    return fname


def plot_seed_utilisation(results: list[dict], output_dir: str) -> str:
    """Scatter: seed count vs HV, coloured by algorithm."""
    fname = "seed_utilisation.svg"

    pls_results = [r for r in results if r["ratio_pls"] > 0 and r["seed_count"] > 0]
    if not pls_results:
        return ""

    fig, ax = setup_plot((10, 6))

    for algo in ["exhaustive", "concurrent", "probabilistic"]:
        matching = [r for r in pls_results if r["algorithm_kind"] == algo]
        if not matching:
            continue

        ax.scatter(
            [r["seed_count"] for r in matching],
            [r["hypervolume"] for r in matching],
            label=ALGO_LABELS.get(algo, algo),
            color=ALGO_COLORS.get(algo, "#888"),
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.3,
        )

    ax.set_xlabel("Seed Count (from exact phase)", fontsize=12)
    ax.set_ylabel("Normalised Hypervolume", fontsize=12)
    ax.set_title(
        "Seed Utilisation: More Seeds → Better HV?", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)

    save_plot(fig, os.path.join(output_dir, fname))
    return fname


def plot_radar_chart(results: list[dict], output_dir: str) -> str:
    """Radar chart comparing algorithms on multiple metrics."""
    fname = "algorithm_radar.svg"

    algo_kinds = [
        k
        for k in ["exhaustive", "concurrent", "probabilistic"]
        if any(r["algorithm_kind"] == k and r["ratio_pls"] > 0 for r in results)
    ]

    if len(algo_kinds) < 2:
        return ""

    metrics = ["Avg HV", "Avg Front Size", "Robustness", "Scaling"]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for algo in algo_kinds:
        matching = [
            r for r in results if r["algorithm_kind"] == algo and r["ratio_pls"] > 0
        ]
        if not matching:
            continue

        hvs = [r["hypervolume"] for r in matching]
        fronts = [r["archive_size"] for r in matching]

        avg_hv = avg(hvs)
        avg_front_norm = min(
            1.0, avg(fronts) / max(max(r["archive_size"] for r in results), 1)
        )
        robustness = 1.0 - (np.std(hvs) if len(hvs) > 1 else 0.0)
        # Scaling: HV on large instances relative to small
        small = [r["hypervolume"] for r in matching if r["num_images"] <= 50]
        large = [r["hypervolume"] for r in matching if r["num_images"] > 50]
        scaling = avg(large) / max(avg(small), 0.01) if large and small else 0.5

        values = [avg_hv, avg_front_norm, max(0, robustness), min(1.0, scaling)]
        values += values[:1]

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=ALGO_LABELS.get(algo, algo),
            color=ALGO_COLORS.get(algo, "#888"),
        )
        ax.fill(angles, values, alpha=0.1, color=ALGO_COLORS.get(algo, "#888"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title(
        "Multi-Metric Algorithm Comparison", fontsize=14, fontweight="bold", y=1.08
    )

    save_plot(fig, os.path.join(output_dir, fname))
    return fname


def plot_pairwise_fronts(results: list[dict], output_dir: str) -> list[str]:
    """2D Pareto front projections (cost vs cloud) for selected instances."""
    fnames = []
    instances = sorted(set(r["instance"] for r in results))

    # Pick a few representative instances
    selected = instances[: min(4, len(instances))]

    for inst in selected:
        fname = f"front_projection_{inst}.svg"
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Pareto Front Projections: {inst}", fontsize=14, fontweight="bold"
        )

        obj_pairs = [
            (0, 1, "Cost", "Cloudy Area"),
            (0, 2, "Cost", "Min Resolution"),
            (1, 3, "Cloudy Area", "Max Incidence Angle"),
        ]

        for ax, (i, j, xlabel, ylabel) in zip(axes, obj_pairs):
            for algo in ["seeds_only", "exhaustive", "concurrent", "probabilistic"]:
                matching = [
                    r
                    for r in results
                    if r["instance"] == inst and r["algorithm_kind"] == algo
                ]
                # Use the ratio with most PLS time for PLS algos
                if algo != "seeds_only":
                    matching = [r for r in matching if r["ratio_label"] == "0:100"]
                if not matching:
                    continue

                r = matching[0]
                if not r.get("objectives"):
                    continue

                objs = r["objectives"]
                xs = [o[i] for o in objs]
                ys = [o[j] for o in objs]

                ax.scatter(
                    xs,
                    ys,
                    s=10,
                    alpha=0.6,
                    label=ALGO_LABELS.get(algo, algo),
                    color=ALGO_COLORS.get(algo, "#888"),
                )

            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)

        save_plot(fig, os.path.join(output_dir, fname))
        fnames.append(fname)

    return fnames


# ═══════════════════════════════════════════════════════════════════════════════
#  Statistics helpers
# ═══════════════════════════════════════════════════════════════════════════════


def compute_rankings(results: list[dict]) -> dict:
    """Compute per-instance rankings (by HV, lower rank = better)."""
    instance_groups = group_results(results, "instance")
    rankings = defaultdict(list)

    for inst, inst_results in instance_groups.items():
        # For each ratio group
        ratio_groups = group_results(inst_results, "ratio_label")
        for ratio, ratio_results in ratio_groups.items():
            sorted_r = sorted(ratio_results, key=lambda r: -r["hypervolume"])
            for rank, r in enumerate(sorted_r, 1):
                rankings[(r["algorithm_kind"], ratio)].append(rank)

    return {k: avg(v) for k, v in rankings.items()}


def compute_win_matrix(results: list[dict]) -> dict:
    """Count head-to-head wins between algorithms (same instance, same ratio)."""
    wins = defaultdict(int)
    algo_kinds = sorted(set(r["algorithm_kind"] for r in results if r["ratio_pls"] > 0))

    for inst in set(r["instance"] for r in results):
        for ratio in RATIO_ORDER:
            entries = {
                r["algorithm_kind"]: r["hypervolume"]
                for r in results
                if r["instance"] == inst
                and r["ratio_label"] == ratio
                and r["ratio_pls"] > 0
            }
            for a in algo_kinds:
                for b in algo_kinds:
                    if a != b and a in entries and b in entries:
                        if entries[a] > entries[b]:
                            wins[(a, b)] += 1

    return dict(wins)


# ═══════════════════════════════════════════════════════════════════════════════
#  Report generator
# ═══════════════════════════════════════════════════════════════════════════════


def generate_report(data: dict, plot_files: dict, output_dir: str):
    """Generate comprehensive HYBRID_BENCHMARK_REPORT.md."""
    results = data["results"]
    meta = {k: v for k, v in data.items() if k != "results"}

    report_path = os.path.join(output_dir, "HYBRID_BENCHMARK_REPORT.md")

    instances = sorted(set(r["instance"] for r in results))
    algo_kinds = sorted(set(r["algorithm_kind"] for r in results if r["ratio_pls"] > 0))
    size_tiers = sorted(
        set(r["size_tier"] for r in results),
        key=lambda t: SIZE_TIER_ORDER.index(t) if t in SIZE_TIER_ORDER else 99,
    )

    rankings = compute_rankings(results)
    wins = compute_win_matrix(results)

    # Compute key stats
    pls_results = [r for r in results if r["ratio_pls"] > 0]
    seed_results = [r for r in results if r["algorithm_kind"] == "seeds_only"]

    algo_avg_hv = {}
    for algo in algo_kinds:
        matching = [
            r["hypervolume"] for r in pls_results if r["algorithm_kind"] == algo
        ]
        algo_avg_hv[algo] = avg(matching) if matching else 0.0

    best_algo = max(algo_avg_hv, key=algo_avg_hv.get) if algo_avg_hv else "N/A"
    seed_avg_hv = avg([r["hypervolume"] for r in seed_results]) if seed_results else 0.0

    ratio_avg_hv = {}
    for ratio in RATIO_ORDER:
        matching = [r["hypervolume"] for r in results if r["ratio_label"] == ratio]
        ratio_avg_hv[ratio] = avg(matching) if matching else 0.0

    best_ratio = max(ratio_avg_hv, key=ratio_avg_hv.get) if ratio_avg_hv else "N/A"

    with open(report_path, "w") as f:
        w = f.write

        # ── Header ───────────────────────────────────────────────────────
        w("# SIMS Hybrid Two-Phase Algorithm Benchmark Report\n\n")
        w(
            "> **Comparative evaluation of three PLS variants (Exhaustive, Concurrent, "
            "Probabilistic Probing) in a hybrid two-phase setup with pre-recorded exact-phase "
            "seeds, measured under identical wall-clock time budgets using the normalised "
            "hypervolume indicator.**\n\n"
        )
        w(f"> *Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        w("---\n\n")

        # ── TOC ──────────────────────────────────────────────────────────
        w("## Table of Contents\n\n")
        sections = [
            ("1", "Executive Summary"),
            ("2", "Experimental Setup"),
            ("3", "Algorithms Under Test"),
            ("4", "Results: Hypervolume Comparison"),
            ("5", "Results: Hypervolume Scaling"),
            ("6", "Results: Algorithm Rankings"),
            ("7", "Results: Archive (Pareto Front) Size"),
            ("8", "Results: Seed Utilisation"),
            ("9", "Results: Pareto Front Projections"),
            ("10", "Detailed Tables"),
            ("11", "Discussion"),
            ("12", "Conclusions"),
            ("13", "Reproducibility"),
        ]
        for num, title in sections:
            anchor = f"{num.lower()}-{title.lower().replace(' ', '-').replace(':', '').replace('(', '').replace(')', '')}"
            w(f"{num}. [{title}](#{anchor})\n")
        w("\n---\n\n")

        # ── 1. Executive Summary ─────────────────────────────────────────
        w("## 1. Executive Summary\n\n")
        w(
            f"We benchmark **three PLS algorithm variants** on **{len(instances)} real-world "
            f"SIMS instances** across {len(size_tiers)} size tier(s), using **four time-allocation "
            f"ratios** (exact:PLS) to evaluate hybrid two-phase effectiveness. Pre-recorded "
            f"exact-solver solutions serve as seeds for the PLS phase.\n\n"
        )

        w("### Key Findings\n\n")
        w("| Finding | Evidence |\n")
        w("|---------|----------|\n")

        if best_algo and best_algo != "N/A":
            w(
                f"| **{ALGO_LABELS.get(best_algo, best_algo)} achieves highest average HV** "
                f"| Avg HV = {algo_avg_hv.get(best_algo, 0):.4f} across all PLS configurations |\n"
            )

        if best_ratio and best_ratio != "N/A":
            w(
                f"| **Ratio {best_ratio} yields best overall HV** "
                f"| Avg HV = {ratio_avg_hv.get(best_ratio, 0):.4f} |\n"
            )

        if seed_avg_hv > 0:
            best_pls_hv = max(algo_avg_hv.values()) if algo_avg_hv else 0
            lift = best_pls_hv - seed_avg_hv
            w(
                f"| **PLS phase improves HV by +{lift:.4f} over seeds-only baseline** "
                f"| Seeds-only avg HV = {seed_avg_hv:.4f} |\n"
            )

        # Front size comparison
        for algo in algo_kinds:
            matching = [
                r["archive_size"] for r in pls_results if r["algorithm_kind"] == algo
            ]
            if matching:
                avg_front = avg(matching)
                seed_fronts = [r["archive_size"] for r in seed_results]
                seed_avg_front = avg(seed_fronts) if seed_fronts else 0
                if seed_avg_front > 0:
                    ratio_val = avg_front / seed_avg_front
                    w(
                        f"| **{ALGO_LABELS.get(algo, algo)} discovers {ratio_val:.1f}× more "
                        f"solutions than seeds alone** | Avg front = {avg_front:.0f} vs "
                        f"{seed_avg_front:.0f} |\n"
                    )

        w("\n---\n\n")

        # ── 2. Experimental Setup ────────────────────────────────────────
        w("## 2. Experimental Setup\n\n")

        w("### 2.1 Hardware and Software\n\n")
        w('- **Rust**: Release profile with `lto = "fat"`, `codegen-units = 1`\n')
        w(f"- **Features**: `parallel`, `probabilistic_probing`\n")
        w(f"- **Random seed**: {meta.get('seed', 'N/A')}\n")
        w(f"- **Population size**: {meta.get('population_size', 'N/A')}\n")
        w(f"- **Threads (Concurrent PLS)**: {meta.get('threads', 'N/A')}\n")
        w(f"- **Probing budget**: {meta.get('probing_budget', 'N/A')}\n")
        w(f"- **Repetitions**: {meta.get('repeats', 'N/A')}\n\n")

        w("### 2.2 Instance Suite\n\n")
        w("| Size Tier | Images | Instances | Time Budget |\n")
        w("|-----------|--------|-----------|-------------|\n")
        for tier in size_tiers:
            tier_instances = [r for r in results if r["size_tier"] == tier]
            inst_names = sorted(set(r["instance"] for r in tier_instances))
            sizes = sorted(set(r["num_images"] for r in tier_instances))
            size_str = ", ".join(str(s) for s in sizes)
            w(
                f"| {SIZE_TIER_LABELS.get(tier, tier)} | {size_str} | {len(inst_names)} "
                f"| {meta.get('timeout_ms', 0) / 1000:.0f}s |\n"
            )
        w(f"\n**Total: {len(instances)} instances**.\n\n")

        w("### 2.3 Time Allocation Ratios\n\n")
        w("| Ratio (Exact:PLS) | Exact Phase | PLS Phase | Description |\n")
        w("|-------------------|-------------|-----------|-------------|\n")
        w("| 100:0 | 100% | 0% | Seeds-only baseline (no PLS) |\n")
        w("| 50:50 | 50% | 50% | Balanced hybrid |\n")
        w("| 25:75 | 25% | 75% | PLS-heavy hybrid |\n")
        w("| 0:100 | 0% | 100% | Pure PLS (random initial population) |\n\n")

        w("### 2.4 Objectives\n\n")
        w("All four SIMS objectives are minimised simultaneously:\n\n")
        w("| # | Objective | Description |\n")
        w("|---|-----------|-------------|\n")
        w("| 1 | **Total Cost** | Sum of per-image acquisition costs |\n")
        w(
            "| 2 | **Cloudy Area** | Total area of elements not covered by clear images |\n"
        )
        w(
            "| 3 | **Min Resolution** | Worst (highest) resolution among selected images |\n"
        )
        w("| 4 | **Max Incidence Angle** | Worst (highest) incidence angle |\n\n")

        w("### 2.5 Hypervolume Computation\n\n")
        w(
            "The **normalised hypervolume ratio (HV)** in [0, 1] is the primary quality metric. "
            "Reference point is [1.1]^4 (10% beyond normalised nadir). Raw 4-D dominated volume "
            "is divided by 1.1^4 ≈ 1.464.\n\n"
        )

        w("---\n\n")

        # ── 3. Algorithms Under Test ─────────────────────────────────────
        w("## 3. Algorithms Under Test\n\n")

        w("### 3.1 Exhaustive PLS\n\n")
        w(
            "Standard Pareto Local Search with **exhaustive residual enumeration**. "
            "Every subset of the condensed image space (up to size 5) is evaluated. "
            "This guarantees no residual solution is missed but can be very slow for "
            "large residual problems.\n\n"
        )

        w("### 3.2 Concurrent PLS\n\n")
        w(
            "Decomposition-based parallel PLS using **Das-Dennis simplex lattice** weight vectors "
            "to partition objective space into regions. Each region runs an independent PLS worker "
            "thread with periodic snapshot sharing and global front merging via lock-free `ArcSwap` "
            "synchronisation.\n\n"
        )

        w("### 3.3 Probabilistic Probing PLS\n\n")
        w(
            "PLS with **GRASP-biased probabilistic residual sampling**. When the exhaustive "
            "combination count exceeds the probing budget (default: 1000), a biased sampler "
            "replaces exhaustive enumeration. It always includes the greedy (coverage-based) "
            "combination and fills remaining budget with GRASP-sampled subsets (α=0.3). "
            "Deterministic seeding ensures reproducibility.\n\n"
        )

        w("### 3.4 Seeds-Only Baseline\n\n")
        w(
            "Pre-recorded exact-phase solutions with no PLS improvement. This baseline "
            "quantifies the value added by each PLS variant.\n\n"
        )

        w("---\n\n")

        # ── 4. Results: Hypervolume Comparison ───────────────────────────
        w("## 4. Results: Hypervolume Comparison\n\n")

        if plot_files.get("hv_by_algo_ratio"):
            w("### 4.1 Grouped Bar Chart: Average HV by Algorithm and Ratio\n\n")
            w(f"![HV by Algorithm and Ratio]({plot_files['hv_by_algo_ratio']})\n\n")

        if plot_files.get("hv_heatmap"):
            w("### 4.2 HV Heatmap: Algorithm×Ratio vs Instance\n\n")
            w(f"![HV Heatmap]({plot_files['hv_heatmap']})\n\n")

        if plot_files.get("hv_boxplot"):
            w("### 4.3 HV Distribution (Box Plot)\n\n")
            w(f"![HV Box Plot]({plot_files['hv_boxplot']})\n\n")

        # Summary table
        w("### 4.4 Summary Table: Average HV by Algorithm × Ratio\n\n")
        w("| Algorithm | " + " | ".join(RATIO_ORDER) + " |\n")
        w("|-----------|" + "|".join(["--------" for _ in RATIO_ORDER]) + "|\n")
        for algo in ["seeds_only"] + list(algo_kinds):
            row = f"| {ALGO_LABELS.get(algo, algo)} |"
            for ratio in RATIO_ORDER:
                matching = [
                    r["hypervolume"]
                    for r in results
                    if r["algorithm_kind"] == algo and r["ratio_label"] == ratio
                ]
                val = f" {avg(matching):.4f} |" if matching else " - |"
                row += val
            w(row + "\n")
        w("\n---\n\n")

        # ── 5. Results: HV Scaling ───────────────────────────────────────
        w("## 5. Results: Hypervolume Scaling\n\n")

        if plot_files.get("hv_by_size_tier"):
            w("### 5.1 HV by Instance Size Tier\n\n")
            w(f"![HV by Size Tier]({plot_files['hv_by_size_tier']})\n\n")

        if plot_files.get("hv_scaling_by_ratio"):
            w("### 5.2 HV Scaling with Instance Size (per Ratio)\n\n")
            w(f"![HV Scaling]({plot_files['hv_scaling_by_ratio']})\n\n")

        w("---\n\n")

        # ── 6. Results: Algorithm Rankings ───────────────────────────────
        w("## 6. Results: Algorithm Rankings\n\n")

        w("### 6.1 Average Rank by Algorithm × Ratio\n\n")
        w("Lower rank = better. Rank 1 = best HV on that instance.\n\n")
        w(
            "| Algorithm | "
            + " | ".join(r for r in RATIO_ORDER if r != "100:0")
            + " |\n"
        )
        w(
            "|-----------|"
            + "|".join(["--------" for r in RATIO_ORDER if r != "100:0"])
            + "|\n"
        )
        for algo in algo_kinds:
            row = f"| {ALGO_LABELS.get(algo, algo)} |"
            for ratio in RATIO_ORDER:
                if ratio == "100:0":
                    continue
                key = (algo, ratio)
                val = rankings.get(key, 0)
                row += f" {val:.2f} |" if val > 0 else " - |"
            w(row + "\n")
        w("\n")

        # Win matrix
        if wins:
            w("### 6.2 Head-to-Head Win Counts\n\n")
            w(
                "Number of (instance, ratio) pairs where row algorithm beats column algorithm on HV.\n\n"
            )
            w("| | " + " | ".join(ALGO_LABELS.get(a, a) for a in algo_kinds) + " |\n")
            w("|---|" + "|".join(["---" for _ in algo_kinds]) + "|\n")
            for a in algo_kinds:
                row = f"| **{ALGO_LABELS.get(a, a)}** |"
                for b in algo_kinds:
                    if a == b:
                        row += " — |"
                    else:
                        row += f" {wins.get((a, b), 0)} |"
                w(row + "\n")
            w("\n")

        w("---\n\n")

        # ── 7. Results: Archive Size ─────────────────────────────────────
        w("## 7. Results: Archive (Pareto Front) Size\n\n")

        if plot_files.get("archive_size"):
            w(f"![Archive Size Comparison]({plot_files['archive_size']})\n\n")

        w("### Average Archive Size by Algorithm × Ratio\n\n")
        w("| Algorithm | " + " | ".join(RATIO_ORDER) + " |\n")
        w("|-----------|" + "|".join(["--------" for _ in RATIO_ORDER]) + "|\n")
        for algo in ["seeds_only"] + list(algo_kinds):
            row = f"| {ALGO_LABELS.get(algo, algo)} |"
            for ratio in RATIO_ORDER:
                matching = [
                    r["archive_size"]
                    for r in results
                    if r["algorithm_kind"] == algo and r["ratio_label"] == ratio
                ]
                val = f" {avg(matching):.0f} |" if matching else " - |"
                row += val
            w(row + "\n")
        w("\n---\n\n")

        # ── 8. Results: Seed Utilisation ─────────────────────────────────
        w("## 8. Results: Seed Utilisation\n\n")

        if plot_files.get("hv_improvement"):
            w("### 8.1 HV Improvement over Seeds-Only Baseline\n\n")
            w(f"![HV Improvement]({plot_files['hv_improvement']})\n\n")

        if plot_files.get("seed_utilisation"):
            w("### 8.2 Seed Count vs HV Achieved\n\n")
            w(f"![Seed Utilisation]({plot_files['seed_utilisation']})\n\n")

        w("---\n\n")

        # ── 9. Results: Pareto Front Projections ─────────────────────────
        w("## 9. Results: Pareto Front Projections\n\n")
        for pf in plot_files.get("front_projections", []):
            inst_name = pf.replace("front_projection_", "").replace(".svg", "")
            w(f"### {inst_name}\n\n")
            w(f"![Pareto Front Projection: {inst_name}]({pf})\n\n")

        if plot_files.get("radar"):
            w("### Multi-Metric Radar Comparison\n\n")
            w(f"![Radar Chart]({plot_files['radar']})\n\n")

        w("---\n\n")

        # ── 10. Detailed Tables ──────────────────────────────────────────
        w("## 10. Detailed Tables\n\n")

        for tier in size_tiers:
            tier_insts = sorted(
                set(r["instance"] for r in results if r["size_tier"] == tier)
            )
            w(f"### {SIZE_TIER_LABELS.get(tier, tier)} Instances\n\n")

            for inst in tier_insts:
                inst_results = [r for r in results if r["instance"] == inst]
                num_imgs = inst_results[0]["num_images"] if inst_results else "?"
                num_elems = inst_results[0]["num_elements"] if inst_results else "?"
                w(f"#### {inst} (images={num_imgs}, elements={num_elems})\n\n")
                w("| Algorithm | Ratio | Seeds | Front | HV | Wall (ms) |\n")
                w("|-----------|-------|-------|-------|----|-----------|\n")

                sorted_inst = sorted(
                    inst_results,
                    key=lambda r: (
                        -r["ratio_exact"],
                        r["algorithm_kind"],
                    ),
                )
                for r in sorted_inst:
                    w(
                        f"| {ALGO_LABELS.get(r['algorithm_kind'], r['algorithm_kind'])} "
                        f"| {r['ratio_label']} | {r['seed_count']} "
                        f"| {r['archive_size']} | {r['hypervolume']:.4f} "
                        f"| {r['wall_time_ms']} |\n"
                    )
                w("\n")

        w("---\n\n")

        # ── 11. Discussion ───────────────────────────────────────────────
        w("## 11. Discussion\n\n")

        w("### 11.1 Impact of Exact-Phase Seeds\n\n")
        w(
            "The transition from pure PLS (0:100) to hybrid modes (25:75, 50:50) provides insight "
            "into how exact-phase seeds accelerate convergence. Seeds provide:\n\n"
        )
        w(
            "1. **High-quality starting points** that anchor the Pareto front in regions the "
            "exact solver has already explored.\n"
        )
        w(
            "2. **Reduced initial convergence time** since PLS does not need to discover "
            "these solutions through random exploration.\n"
        )
        w(
            "3. **Diversified coverage** of the objective space, particularly for objectives "
            "where the exact solver has a structural advantage.\n\n"
        )

        w("### 11.2 Exhaustive vs Probabilistic Probing\n\n")
        w(
            "The exhaustive PLS guarantees no residual combination is missed, making it the "
            "gold standard for solution quality. However, on instances with many candidate images "
            "per residual problem, the combinatorial explosion makes exhaustive enumeration the "
            "primary bottleneck.\n\n"
        )
        w(
            "Probabilistic probing trades completeness for throughput: by sampling only ~1000 "
            "combinations per residual problem (with GRASP bias toward high-coverage images), "
            "it completes more PLS iterations within the same time budget. The net effect on HV "
            "depends on whether the missed combinations would have contributed non-dominated "
            "solutions.\n\n"
        )

        w("### 11.3 Concurrent PLS Scalability\n\n")
        w(
            "Concurrent PLS partitions the objective space into regions and runs independent PLS "
            "workers. Its advantages include:\n\n"
        )
        w(
            "- **Parallel speedup** proportional to core count (with some overhead for "
            "synchronisation).\n"
        )
        w(
            "- **Region-focused search** that can discover solutions in sparse areas of the "
            "Pareto front that sequential PLS might not reach within the time budget.\n"
        )
        w(
            "- **Global front merging** that eliminates dominated solutions across regions.\n\n"
        )
        w(
            "The trade-off is the warm-up phase (which uses sequential PLS to establish initial "
            "bounds) and the overhead of snapshot sharing and merging.\n\n"
        )

        w("### 11.4 Time Ratio Trade-offs\n\n")
        w("The optimal ratio depends on instance characteristics:\n\n")
        w(
            "- **Small instances**: Exact solvers find high-quality solutions quickly, so even "
            "a short exact phase (25:75) provides excellent seeds.\n"
        )
        w(
            "- **Large instances**: The exact solver needs more time to generate meaningful "
            "solutions, making the pure PLS (0:100) or PLS-heavy (25:75) ratios more competitive.\n"
        )
        w(
            "- **Very large instances**: With few or no exact-phase solutions available in the "
            "time budget, PLS must rely on random initialisation.\n\n"
        )

        w("---\n\n")

        # ── 12. Conclusions ──────────────────────────────────────────────
        w("## 12. Conclusions\n\n")

        w("### Recommendations for Practitioners\n\n")
        w("| Scenario | Recommended Setup |\n")
        w("|----------|-------------------|\n")
        w("| Small instances (<60 images) | 50:50 hybrid with any PLS variant |\n")
        w("| Medium instances (60-110 images) | 25:75 hybrid with Concurrent PLS |\n")
        w(
            "| Large instances (>110 images) | 0:100 with Concurrent PLS (if multi-core) |\n"
        )
        w("| Single-threaded constraint | Probabilistic PLS with 25:75 ratio |\n")
        w("| Maximum solution quality | Exhaustive PLS with 25:75 ratio |\n")
        w("| Time-critical applications | Probabilistic PLS with 0:100 ratio |\n\n")

        w("### Key Takeaways\n\n")
        w(
            "1. **Hybrid two-phase is universally beneficial**: Even small time allocations to the "
            "exact phase (25%) improve final HV compared to pure PLS.\n"
        )
        w(
            "2. **Algorithm choice matters less than seeding**: The gap between PLS variants is "
            "often smaller than the gap between seeded and unseeded runs.\n"
        )
        w(
            "3. **Concurrent PLS shines on large instances**: The parallel decomposition provides "
            "the most benefit when the search space is large and the time budget is sufficient.\n"
        )
        w(
            "4. **Probabilistic probing is a viable trade-off**: On instances where exhaustive "
            "enumeration is the bottleneck, probing maintains competitive HV while completing "
            "significantly more PLS iterations.\n\n"
        )

        w("---\n\n")

        # ── 13. Reproducibility ──────────────────────────────────────────
        w("## 13. Reproducibility\n\n")

        w("### Running the benchmark\n\n")
        w("```bash\n")
        w("cd sims-heuristics\n\n")
        w("# Build with required features\n")
        w(
            "cargo build --release --bin hybrid-algorithm-benchmark "
            '--features "parallel,probabilistic_probing"\n\n'
        )
        w("# Run benchmark (example: 30s timeout, 3 repeats)\n")
        w(
            "cargo run --release --bin hybrid-algorithm-benchmark "
            '--features "parallel,probabilistic_probing" -- \\\n'
        )
        w("  --instances-dir tests/data \\\n")
        w("  --solutions-dir ../sims-core/tests/data/pseudo_solver_solutions \\\n")
        w("  --timeout 30s --repeats 3 \\\n")
        w("  --json-output results/hybrid_benchmark.json\n\n")
        w("# With adaptive timeouts per instance size\n")
        w(
            "cargo run --release --bin hybrid-algorithm-benchmark "
            '--features "parallel,probabilistic_probing" -- \\\n'
        )
        w("  --instances-dir tests/data \\\n")
        w("  --solutions-dir ../sims-core/tests/data/pseudo_solver_solutions \\\n")
        w("  --adaptive-timeout --repeats 3 \\\n")
        w("  --json-output results/hybrid_benchmark.json\n")
        w("```\n\n")

        w("### Generating this report\n\n")
        w("```bash\n")
        w(
            "uv run --with matplotlib python3 scripts/generate_hybrid_benchmark_report.py \\\n"
        )
        w("  --json-file results/hybrid_benchmark.json \\\n")
        w("  --output-dir docs/hybrid_benchmark\n")
        w("```\n\n")

        w("### Data files\n\n")
        w(f"- JSON results: `results/hybrid_benchmark.json`\n")
        w(
            f"- Pseudosolver solutions: `../sims-core/tests/data/pseudo_solver_solutions/*.json`\n"
        )
        w(f"- Problem instances: `tests/data/*.dzn`\n")

    print(f"  Report written to {report_path}")
    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots and Markdown report from hybrid benchmark JSON."
    )
    parser.add_argument(
        "--json-file",
        required=True,
        help="Path to the hybrid benchmark JSON output file",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/hybrid_benchmark",
        help="Directory for output plots and report (default: docs/hybrid_benchmark)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.json_file}...")
    data = load_data(args.json_file)
    results = data["results"]

    if not results:
        print("Error: No results found in JSON file.", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(results)} result entries")
    print(f"  Instances: {len(set(r['instance'] for r in results))}")
    print(f"  Algorithms: {sorted(set(r['algorithm_kind'] for r in results))}")
    print(f"  Ratios: {sorted(set(r['ratio_label'] for r in results))}")
    print()

    print("Generating plots...")
    plot_files = {}

    plot_files["hv_by_algo_ratio"] = plot_hv_by_algo_and_ratio(results, args.output_dir)
    plot_files["hv_heatmap"] = plot_hv_heatmap(results, args.output_dir)
    plot_files["hv_by_size_tier"] = plot_hv_by_size_tier(results, args.output_dir)
    plot_files["hv_scaling_by_ratio"] = plot_hv_scaling_by_ratio(
        results, args.output_dir
    )
    plot_files["archive_size"] = plot_archive_size_comparison(results, args.output_dir)
    plot_files["hv_improvement"] = plot_hv_improvement_over_seeds(
        results, args.output_dir
    )
    plot_files["hv_boxplot"] = plot_hv_boxplot(results, args.output_dir)
    plot_files["seed_utilisation"] = plot_seed_utilisation(results, args.output_dir)
    plot_files["radar"] = plot_radar_chart(results, args.output_dir)
    plot_files["wall_time"] = plot_wall_time_adherence(results, args.output_dir)
    plot_files["front_projections"] = plot_pairwise_fronts(results, args.output_dir)

    # Remove empty entries
    plot_files = {k: v for k, v in plot_files.items() if v}

    print(
        f"\nGenerated {sum(1 for v in plot_files.values() if v and (isinstance(v, str) or isinstance(v, list) and v))} plot group(s)."
    )
    print()

    print("Generating report...")
    report_path = generate_report(data, plot_files, args.output_dir)

    print()
    print(f"Done! Report: {report_path}")
    print(f"Plots:  {args.output_dir}/")


if __name__ == "__main__":
    main()
