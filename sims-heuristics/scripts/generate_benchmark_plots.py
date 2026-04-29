#!/usr/bin/env python3
"""Generate SVG plots from algorithm benchmark JSON data.

Usage:
    uv run --with matplotlib python3 scripts/generate_benchmark_plots.py \
        --json-files /tmp/bench_30.json /tmp/bench_50.json /tmp/bench_100.json /tmp/bench_145.json \
        --output-dir docs/benchmark_plots

Produces:
    1. hv_by_algorithm_grouped.svg     – Grouped bar chart: avg HV per algorithm across size tiers
    2. hv_heatmap.svg                  – Heatmap: HV per (algorithm × instance)
    3. rank_by_size.svg                – Line chart: avg rank vs instance size
    4. archive_size_by_algorithm.svg   – Grouped bar chart: avg archive size per algorithm
    5. wall_time_accuracy.svg          – Scatter: actual wall time vs budget
    6. hv_scaling.svg                  – Line chart: HV vs instance size per algorithm
    7. pairwise_2d_fronts_*.svg        – 2-D projections (cost vs cloud) for selected instances
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Styling ──────────────────────────────────────────────────────────────────

ALGO_ORDER = [
    "PLS",
    "NSGA-II (custom)",
    "MOEA/D (custom)",
    "moors NSGA-II",
    "moors SPEA-2",
    "optirustic NSGA-II",
    "optirustic NSGA-III",
]

ALGO_SHORT = {
    "PLS": "PLS",
    "NSGA-II (custom)": "NSGA-II",
    "MOEA/D (custom)": "MOEA/D",
    "moors NSGA-II": "moors\nNSGA-II",
    "moors SPEA-2": "moors\nSPEA-2",
    "optirustic NSGA-II": "optirustic\nNSGA-II",
    "optirustic NSGA-III": "optirustic\nNSGA-III",
}

ALGO_COLORS = {
    "PLS": "#2196F3",
    "NSGA-II (custom)": "#4CAF50",
    "MOEA/D (custom)": "#FF9800",
    "moors NSGA-II": "#9C27B0",
    "moors SPEA-2": "#E91E63",
    "optirustic NSGA-II": "#607D8B",
    "optirustic NSGA-III": "#795548",
}

ALGO_MARKERS = {
    "PLS": "o",
    "NSGA-II (custom)": "s",
    "MOEA/D (custom)": "D",
    "moors NSGA-II": "^",
    "moors SPEA-2": "v",
    "optirustic NSGA-II": "P",
    "optirustic NSGA-III": "X",
}

CATEGORY_LABELS = {
    "custom": ["PLS", "NSGA-II (custom)", "MOEA/D (custom)"],
    "moors": ["moors NSGA-II", "moors SPEA-2"],
    "optirustic": ["optirustic NSGA-II", "optirustic NSGA-III"],
}

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


def load_benchmarks(json_paths: list[str]) -> list[dict]:
    """Load and merge benchmark JSON files, returning flat list of instance records."""
    instances = []
    for p in json_paths:
        with open(p) as f:
            data = json.load(f)
        timeout_ms = data["timeout_ms"]
        for inst in data["instances"]:
            inst["timeout_ms"] = timeout_ms
            instances.append(inst)
    return instances


def base_name(name: str) -> str:
    idx = name.find("(")
    if idx > 0 and not name.startswith("NSGA") and not name.startswith("MOEA"):
        return name[:idx].strip()
    return name


def normalise_algo_name(raw: str) -> str:
    """Map raw algorithm name (possibly with iter/gen count) to canonical name."""
    for canon in ALGO_ORDER:
        if raw == canon:
            return canon
        if raw.startswith(canon):
            return canon
    # Fallback: strip parenthesised suffix for external solvers
    if "moors NSGA-II" in raw:
        return "moors NSGA-II"
    if "moors SPEA-2" in raw:
        return "moors SPEA-2"
    if "optirustic NSGA-II" in raw and "III" not in raw:
        return "optirustic NSGA-II"
    if "optirustic NSGA-III" in raw:
        return "optirustic NSGA-III"
    return raw


def extract_size(name: str) -> int:
    """Extract instance size (number after last underscore) from instance name."""
    parts = name.rsplit("_", 1)
    try:
        return int(parts[-1])
    except ValueError:
        return 0


def extract_city(name: str) -> str:
    parts = name.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else name


# ── Plot 1: Grouped bar — avg HV by algorithm, grouped by size tier ─────────


def plot_hv_grouped_bar(instances: list[dict], out_dir: str):
    # Group by size tier
    size_tiers = sorted({extract_size(inst["name"]) for inst in instances})

    # algo -> size -> list of HV values
    data = defaultdict(lambda: defaultdict(list))
    for inst in instances:
        size = extract_size(inst["name"])
        for algo_rec in inst["algorithms"]:
            algo = normalise_algo_name(algo_rec["name"])
            data[algo][size].append(algo_rec["hypervolume"])

    algos = [a for a in ALGO_ORDER if a in data]
    n_algos = len(algos)
    n_tiers = len(size_tiers)

    fig, ax = plt.subplots(figsize=(max(10, n_tiers * 2.5), 6))

    bar_width = 0.8 / n_algos
    x = np.arange(n_tiers)

    for i, algo in enumerate(algos):
        means = []
        for s in size_tiers:
            vals = data[algo].get(s, [])
            means.append(np.mean(vals) if vals else 0)
        offset = (i - n_algos / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            means,
            bar_width * 0.9,
            label=algo,
            color=ALGO_COLORS.get(algo, "#888"),
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels on top
        for bar, val in zip(bars, means):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} images" for s in size_tiers])
    ax.set_ylabel("Normalised Hypervolume (higher is better)")
    ax.set_title("Average Hypervolume by Algorithm and Instance Size")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(out_dir, "hv_by_algorithm_grouped.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 2: Heatmap — HV per (algorithm × instance) ─────────────────────────


def plot_hv_heatmap(instances: list[dict], out_dir: str):
    # Sort instances by size then city
    sorted_inst = sorted(instances, key=lambda i: (extract_size(i["name"]), i["name"]))
    inst_names = [i["name"] for i in sorted_inst]

    algos = []
    for inst in sorted_inst:
        for ar in inst["algorithms"]:
            a = normalise_algo_name(ar["name"])
            if a not in algos:
                algos.append(a)
    # Reorder to canonical
    algos = [a for a in ALGO_ORDER if a in algos]

    matrix = np.zeros((len(algos), len(inst_names)))
    for j, inst in enumerate(sorted_inst):
        for ar in inst["algorithms"]:
            a = normalise_algo_name(ar["name"])
            if a in algos:
                i = algos.index(a)
                matrix[i, j] = ar["hypervolume"]

    fig, ax = plt.subplots(figsize=(max(10, len(inst_names) * 0.9), 5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(len(inst_names)))
    ax.set_xticklabels(inst_names, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos, fontsize=9)

    # Annotate cells
    for i in range(len(algos)):
        for j in range(len(inst_names)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.65 else "black"
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=color
            )

    fig.colorbar(im, ax=ax, label="Normalised HV", shrink=0.8)
    ax.set_title("Hypervolume Heatmap (Algorithm × Instance)")

    path = os.path.join(out_dir, "hv_heatmap.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 3: Line chart — average rank vs instance size ───────────────────────


def plot_rank_by_size(instances: list[dict], out_dir: str):
    size_tiers = sorted({extract_size(inst["name"]) for inst in instances})

    # For each instance, compute HV-based ranks
    algo_ranks = defaultdict(lambda: defaultdict(list))  # algo -> size -> [ranks]
    for inst in instances:
        size = extract_size(inst["name"])
        hvs = []
        for ar in inst["algorithms"]:
            a = normalise_algo_name(ar["name"])
            hvs.append((a, ar["hypervolume"]))
        # Rank descending by HV
        hvs_sorted = sorted(hvs, key=lambda x: -x[1])
        for rank, (a, _) in enumerate(hvs_sorted, 1):
            algo_ranks[a][size].append(rank)

    algos = [a for a in ALGO_ORDER if a in algo_ranks]

    fig, ax = plt.subplots(figsize=(9, 6))

    for algo in algos:
        means = []
        sizes_present = []
        for s in size_tiers:
            vals = algo_ranks[algo].get(s, [])
            if vals:
                means.append(np.mean(vals))
                sizes_present.append(s)
        ax.plot(
            sizes_present,
            means,
            marker=ALGO_MARKERS.get(algo, "o"),
            color=ALGO_COLORS.get(algo, "#888"),
            label=algo,
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Instance Size (number of images)")
    ax.set_ylabel("Average Rank (lower is better)")
    ax.set_title("Algorithm Ranking by Instance Size")
    ax.legend(fontsize=8, loc="center right")
    ax.set_xticks(size_tiers)
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    path = os.path.join(out_dir, "rank_by_size.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 4: Archive size bar chart ───────────────────────────────────────────


def plot_archive_size(instances: list[dict], out_dir: str):
    size_tiers = sorted({extract_size(inst["name"]) for inst in instances})
    data = defaultdict(lambda: defaultdict(list))
    for inst in instances:
        size = extract_size(inst["name"])
        for ar in inst["algorithms"]:
            a = normalise_algo_name(ar["name"])
            data[a][size].append(ar["archive_size"])

    algos = [a for a in ALGO_ORDER if a in data]
    n_algos = len(algos)
    n_tiers = len(size_tiers)

    fig, ax = plt.subplots(figsize=(max(10, n_tiers * 2.5), 6))
    bar_width = 0.8 / n_algos
    x = np.arange(n_tiers)

    for i, algo in enumerate(algos):
        means = []
        for s in size_tiers:
            vals = data[algo].get(s, [])
            means.append(np.mean(vals) if vals else 0)
        offset = (i - n_algos / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            means,
            bar_width * 0.9,
            label=algo,
            color=ALGO_COLORS.get(algo, "#888"),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} images" for s in size_tiers])
    ax.set_ylabel("Average Archive (Pareto Front) Size")
    ax.set_title("Non-Dominated Solutions Found by Algorithm and Instance Size")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(out_dir, "archive_size_by_algorithm.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 5: Wall time accuracy (actual vs budget) ───────────────────────────


def plot_wall_time_accuracy(instances: list[dict], out_dir: str):
    fig, ax = plt.subplots(figsize=(9, 6))

    for inst in instances:
        budget = inst["timeout_ms"]
        for ar in inst["algorithms"]:
            algo = normalise_algo_name(ar["name"])
            wall = ar["wall_time_ms"]
            ratio = wall / budget if budget > 0 else 1.0
            ax.scatter(
                budget / 1000,
                ratio,
                color=ALGO_COLORS.get(algo, "#888"),
                marker=ALGO_MARKERS.get(algo, "o"),
                s=50,
                alpha=0.7,
                label=algo,
            )

    # Remove duplicate legends
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            unique_handles.append(h)
            unique_labels.append(l)
    ax.legend(unique_handles, unique_labels, fontsize=8, loc="upper left")

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Budget")
    ax.set_xlabel("Time Budget (seconds)")
    ax.set_ylabel("Wall Time / Budget Ratio")
    ax.set_title("Time Budget Adherence (1.0 = perfect, >1 = overshoot)")
    ax.grid(alpha=0.3)

    path = os.path.join(out_dir, "wall_time_accuracy.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 6: HV scaling lines ────────────────────────────────────────────────


def plot_hv_scaling(instances: list[dict], out_dir: str):
    size_tiers = sorted({extract_size(inst["name"]) for inst in instances})
    data = defaultdict(lambda: defaultdict(list))
    for inst in instances:
        size = extract_size(inst["name"])
        for ar in inst["algorithms"]:
            a = normalise_algo_name(ar["name"])
            data[a][size].append(ar["hypervolume"])

    algos = [a for a in ALGO_ORDER if a in data]

    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in algos:
        sizes_present = []
        means = []
        stds = []
        for s in size_tiers:
            vals = data[algo].get(s, [])
            if vals:
                sizes_present.append(s)
                means.append(np.mean(vals))
                stds.append(np.std(vals))

        means = np.array(means)
        stds = np.array(stds)
        ax.plot(
            sizes_present,
            means,
            marker=ALGO_MARKERS.get(algo, "o"),
            color=ALGO_COLORS.get(algo, "#888"),
            label=algo,
            linewidth=2,
            markersize=8,
        )
        if len(sizes_present) > 1:
            ax.fill_between(
                sizes_present,
                means - stds,
                means + stds,
                color=ALGO_COLORS.get(algo, "#888"),
                alpha=0.1,
            )

    ax.set_xlabel("Instance Size (number of images)")
    ax.set_ylabel("Normalised Hypervolume")
    ax.set_title("Hypervolume Scaling with Instance Size")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xticks(size_tiers)
    ax.grid(alpha=0.3)

    path = os.path.join(out_dir, "hv_scaling.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 7: 2-D Pareto front projections (Cost vs Cloud) ────────────────────


def plot_2d_fronts(instances: list[dict], out_dir: str):
    """For each instance, produce a scatter of objective[0] vs objective[1]."""
    # Pick up to 4 representative instances (one per size tier)
    size_tiers = sorted({extract_size(inst["name"]) for inst in instances})
    selected = []
    for s in size_tiers:
        candidates = [i for i in instances if extract_size(i["name"]) == s]
        if candidates:
            # Pick the one with the most total solutions across algorithms
            best = max(
                candidates,
                key=lambda i: sum(len(a["objectives"]) for a in i["algorithms"]),
            )
            selected.append(best)

    if not selected:
        return

    n = len(selected)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.5 * rows), squeeze=False)

    for idx, inst in enumerate(selected):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        for ar in inst["algorithms"]:
            algo = normalise_algo_name(ar["name"])
            objs = ar["objectives"]
            if not objs:
                continue
            xs = [o[0] for o in objs]  # TotalCost
            ys = [o[1] for o in objs]  # CloudyArea
            ax.scatter(
                xs,
                ys,
                s=12,
                alpha=0.5,
                color=ALGO_COLORS.get(algo, "#888"),
                marker=ALGO_MARKERS.get(algo, "o"),
                label=algo,
            )
        ax.set_xlabel("Total Cost")
        ax.set_ylabel("Cloudy Area")
        title = inst["name"].replace("_", " ").title()
        ax.set_title(f"{title} ({inst['num_images']} images)")
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7, markerscale=1.5, loc="upper right")

    # Hide unused subplots
    for idx in range(len(selected), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        "Pareto Front Projections: Total Cost vs Cloudy Area", fontsize=14, y=1.01
    )
    fig.tight_layout()

    path = os.path.join(out_dir, "pareto_front_2d_projections.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 8: Box plot of HV distributions per algorithm ───────────────────────


def plot_hv_boxplot(instances: list[dict], out_dir: str):
    data = defaultdict(list)
    for inst in instances:
        for ar in inst["algorithms"]:
            a = normalise_algo_name(ar["name"])
            data[a].append(ar["hypervolume"])

    algos = [a for a in ALGO_ORDER if a in data]
    values = [data[a] for a in algos]

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(
        values,
        labels=[ALGO_SHORT.get(a, a) for a in algos],
        patch_artist=True,
        notch=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="black", markersize=5),
    )

    for patch, algo in zip(bp["boxes"], algos):
        patch.set_facecolor(ALGO_COLORS.get(algo, "#ccc"))
        patch.set_alpha(0.7)

    ax.set_ylabel("Normalised Hypervolume")
    ax.set_title("Hypervolume Distribution Across All Instances")
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(out_dir, "hv_boxplot.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 9: Category comparison (custom vs moors vs optirustic) ──────────────


def plot_category_comparison(instances: list[dict], out_dir: str):
    """Bar chart comparing HV of custom, moors, and optirustic algorithm families."""
    size_tiers = sorted({extract_size(inst["name"]) for inst in instances})

    cat_data = defaultdict(lambda: defaultdict(list))  # category -> size -> [HV]
    for inst in instances:
        size = extract_size(inst["name"])
        for ar in inst["algorithms"]:
            algo = normalise_algo_name(ar["name"])
            for cat, members in CATEGORY_LABELS.items():
                if algo in members:
                    cat_data[cat][size].append(ar["hypervolume"])
                    break

    cats = ["custom", "moors", "optirustic"]
    cat_colors = {"custom": "#2E7D32", "moors": "#7B1FA2", "optirustic": "#546E7A"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    n_cats = len(cats)
    bar_width = 0.8 / n_cats
    x = np.arange(len(size_tiers))

    for i, cat in enumerate(cats):
        means = []
        for s in size_tiers:
            vals = cat_data[cat].get(s, [])
            means.append(np.mean(vals) if vals else 0)
        offset = (i - n_cats / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            means,
            bar_width * 0.9,
            label=cat.title(),
            color=cat_colors[cat],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )
        for bar, val in zip(bars, means):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} images" for s in size_tiers])
    ax.set_ylabel("Average Normalised HV")
    ax.set_title("Custom vs External Library Solvers (Avg HV by Family)")
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(out_dir, "category_comparison.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark SVG plots")
    parser.add_argument(
        "--json-files",
        nargs="+",
        required=True,
        help="Benchmark JSON files (from algorithm-benchmark --json-output)",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/benchmark_plots",
        help="Directory for SVG output files",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {len(args.json_files)} JSON file(s)...")
    instances = load_benchmarks(args.json_files)
    print(f"Loaded {len(instances)} instance(s).\n")
    print("Generating plots:")

    plot_hv_grouped_bar(instances, out_dir)
    plot_hv_heatmap(instances, out_dir)
    plot_rank_by_size(instances, out_dir)
    plot_archive_size(instances, out_dir)
    plot_wall_time_accuracy(instances, out_dir)
    plot_hv_scaling(instances, out_dir)
    plot_2d_fronts(instances, out_dir)
    plot_hv_boxplot(instances, out_dir)
    plot_category_comparison(instances, out_dir)

    print(f"\nDone. {9} plots written to {out_dir}/")


if __name__ == "__main__":
    main()
