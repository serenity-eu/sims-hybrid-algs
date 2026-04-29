#!/usr/bin/env python3
"""
Flamegraph analyzer for PLS profiling.

Accepts either:
  - A .folded file (inferno folded-stacks format, preferred)
  - A .svg flamegraph file (legacy, less accurate)

Usage:
    python3 analyze_flamegraph.py flamegraph.folded
    python3 analyze_flamegraph.py flamegraph.svg

To produce a .folded file alongside the SVG, use cargo flamegraph with:
    cargo flamegraph --post-process 'tee flamegraph_optN.folded' --output flamegraph_optN.svg ...
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


# ---------------------------------------------------------------------------
# Folded-stacks parser (preferred)
# ---------------------------------------------------------------------------

def parse_folded(filepath: str):
    """
    Parse inferno folded-stacks file.
    Returns list of (frames: list[str], count: int).
    """
    stacks = []
    with open(filepath) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if not line:
                continue
            sep = line.rfind(' ')
            if sep == -1:
                continue
            frames_part = line[:sep]
            try:
                count = int(line[sep + 1:])
            except ValueError:
                continue
            frames = frames_part.split(';')
            stacks.append((frames, count))
    return stacks


def analyze_folded(stacks):
    """
    Compute self-time and inclusive-time from folded stacks.
    Returns (self_counts, incl_counts, total) all keyed by function name.
    """
    self_counts: dict[str, int] = defaultdict(int)
    incl_counts: dict[str, int] = defaultdict(int)
    total = 0

    for frames, count in stacks:
        total += count
        if frames:
            self_counts[frames[-1]] += count
        seen = set()
        for f in frames:
            if f not in seen:
                incl_counts[f] += count
                seen.add(f)

    return self_counts, incl_counts, total


# ---------------------------------------------------------------------------
# Phase budget from folded stacks
# ---------------------------------------------------------------------------

# Each phase is identified by the innermost (leaf-closest) frame whose name
# contains one of the given substrings. Phases are tried in order; first
# match (walking from leaf toward root) wins for each stack.
PHASES = [
    ("Tracker::peek_addition",    ["peek_addition_packed_small", "peek_addition_two_level",
                                   "peek_addition_general", "peek_addition_delta"]),
    ("Tracker::peek_removal",     ["peek_removal_packed_small", "peek_removal_two_level",
                                   "peek_removal_general", "peek_removal_delta"]),
    ("Tracker::track_addition",   ["track_addition_packed_small", "track_addition_two_level",
                                   "track_addition_general", "track_image_addition"]),
    ("Tracker::track_removal",    ["track_removal_packed_small", "track_removal_two_level",
                                   "track_removal_general", "track_image_removal"]),
    ("Tracker::initialize_from",  ["initialize_from"]),
    ("Candidate selection",       ["best_unselected_images"]),
    ("Residual::solve",           ["residual_problem", "ResidualProblem"]),
    ("Compute objectives",        ["compute_residual_objectives"]),
    ("ParetoFront::try_insert",   ["try_insert", "nd_tree", "ndtree"]),
    ("Objectives eq/dominance",   ["spec_eq", "partialeq", "dominates"]),
]

def phase_budget(stacks, total):
    """
    For each stack classify it by searching frames from leaf toward root
    for the first phase keyword match (i.e. the innermost hot phase).
    Returns dict phase_name -> (count, pct).
    """
    budget: dict[str, int] = defaultdict(int)
    unclassified = 0

    for frames, count in stacks:
        matched = None
        for frame in reversed(frames):
            frame_lower = frame.lower()
            for phase_name, keywords in PHASES:
                if any(kw.lower() in frame_lower for kw in keywords):
                    matched = phase_name
                    break
            if matched:
                break
        if matched:
            budget[matched] += count
        else:
            unclassified += count

    result = {k: (v, v / total * 100) for k, v in budget.items()}
    if unclassified:
        result["(unclassified)"] = (unclassified, unclassified / total * 100)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def short_name(full: str) -> str:
    parts = re.split(r'::', full)
    meaningful = [p for p in parts if p and not p.startswith('<') and
                  p not in ('core', 'std', 'alloc', 'pls')]
    if meaningful:
        return '::'.join(meaningful[-2:])[:60]
    return full[:60]


NOISE_PATTERNS = [
    'criterion', 'rayon_core', '__ieee754', '__GI___', '__math_',
    'std::panic', 'futex', 'syscall', 'pthread', 'libm',
    '__libc', 'signal_handler', 'perf_event',
]

def is_noise(name: str) -> bool:
    nl = name.lower()
    return any(p.lower() in nl for p in NOISE_PATTERNS)


# ---------------------------------------------------------------------------
# Folded report
# ---------------------------------------------------------------------------

def print_folded_report(stacks):
    self_counts, incl_counts, total = analyze_folded(stacks)
    if total == 0:
        print("No samples found.", file=sys.stderr)
        return

    print(f"TOTAL SAMPLES: {total:,}\n")

    # Phase budget
    budget = phase_budget(stacks, total)
    print("PHASE BUDGET (innermost hot-frame classification)")
    print(f"  {'Phase':<35s}  {'%':>6}  {'samples':>10}")
    print(f"  {'-'*35}  {'-'*6}  {'-'*10}")
    for phase, (count, pct) in sorted(budget.items(), key=lambda x: -x[1][0]):
        print(f"  {phase:<35s}  {pct:6.1f}%  {count:>10,}")
    print()

    # Top self-time hotspots
    print("TOP 30 HOTSPOTS (self-time, aggregated across all call sites)")
    print(f"  {'self%':>6}  {'Function'}")
    print(f"  {'-'*6}  {'-'*60}")
    filtered = [(n, c) for n, c in self_counts.items() if not is_noise(n) and c > 0]
    for name, cnt in sorted(filtered, key=lambda x: -x[1])[:30]:
        print(f"  {cnt/total*100:6.2f}%  {short_name(name)}")
    print()

    # Key function inclusive times
    KEY_FUNCTIONS = [
        ("BitsetNeighborhoodIter",    ["bitsetneighborhooditer", "neighborhood_iter"]),
        ("best_unselected_images",    ["best_unselected_images"]),
        ("ResidualProblem::solve*",   ["residualproblem", "residual_problem"]),
        ("compute_residual_obj",      ["compute_residual_objectives"]),
        ("initialize_from",           ["initialize_from"]),
        ("peek_addition (all)",       ["peek_addition"]),
        ("peek_removal (all)",        ["peek_removal"]),
        ("track_addition (all)",      ["track_addition", "track_image_addition"]),
        ("track_removal (all)",       ["track_removal", "track_image_removal"]),
        ("ParetoFront / nd_tree",     ["nd_tree", "ndtree", "try_insert", "pareto_front"]),
        ("objectives eq/cmp",         ["spec_eq", "spec_array", "partialeq"]),
        ("FixedBitSet ops",           ["fixedbitset"]),
    ]
    print("KEY FUNCTION INCLUSIVE TIMES")
    print(f"  {'Function':<35s}  {'incl%':>6}")
    print(f"  {'-'*35}  {'-'*6}")
    for label, keywords in KEY_FUNCTIONS:
        total_incl = sum(c for n, c in incl_counts.items()
                         if any(kw.lower() in n.lower() for kw in keywords))
        if total_incl > 0:
            print(f"  {label:<35s}  {total_incl/total*100:6.1f}%")
    print()

    # Objective self-time
    OBJ_PATTERNS = {
        'MinResolution':     ['minresolution', 'min_resolution'],
        'CloudyArea':        ['cloudyarea', 'cloudy_area'],
        'TotalCost':         ['totalcost', 'total_cost'],
        'MaxIncidenceAngle': ['maxincidenceangle', 'max_incidence'],
    }
    print("OBJECTIVE SELF-TIME")
    for obj, patterns in OBJ_PATTERNS.items():
        cnt = sum(c for n, c in self_counts.items()
                  if any(p in n.lower() for p in patterns)
                  and 'objectivestate' not in n.lower())
        if cnt:
            print(f"  {obj:<22s}  {cnt/total*100:5.1f}%")
    print()


# ---------------------------------------------------------------------------
# Legacy SVG report
# ---------------------------------------------------------------------------

def parse_flamegraph_svg(filepath: str):
    tree = ET.parse(filepath)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    frames = []
    frames_by_y = defaultdict(list)

    for elem in root.iter():
        if not elem.tag.endswith('g'):
            continue
        title_elem = elem.find('svg:title', ns)
        rect_elem = elem.find('svg:rect', ns)
        if title_elem is None or rect_elem is None:
            continue
        title_text = title_elem.text or ""
        raw_width = rect_elem.get('{http://github.com/jonhoo/inferno}w')
        width = float(raw_width) if raw_width else float(rect_elem.get('width', '0').rstrip('%'))
        raw_x = rect_elem.get('{http://github.com/jonhoo/inferno}x')
        x = float(raw_x) if raw_x else float(rect_elem.get('x', '0').rstrip('%'))
        y = float(rect_elem.get('y', '0'))
        m = re.match(r'^(.*?) \((\d+(?:,\d+)*)(?: samples?)?, ([\d\.]+)%\)$', title_text)
        func_name = m.group(1) if m else title_text
        frame = {'name': func_name, 'width': width, 'x': x, 'y': y, 'children_width': 0.0}
        frames.append(frame)
        frames_by_y[y].append(frame)

    if not frames:
        return [], 0.0

    true_root = max(frames, key=lambda f: f['width'])
    total_width = true_root['width']
    y_levels = sorted(frames_by_y.keys())
    root_at_bottom = (true_root['y'] == y_levels[-1])
    ordered_levels = sorted(y_levels, reverse=root_at_bottom)

    for i in range(1, len(ordered_levels)):
        parent_y = ordered_levels[i - 1]
        child_y = ordered_levels[i]
        parents = sorted([(p['x'], p['x'] + p['width'], p) for p in frames_by_y[parent_y]])
        for child in frames_by_y[child_y]:
            mid_x = child['x'] + child['width'] / 2.0
            for px_start, px_end, parent in parents:
                if px_start <= mid_x <= px_end:
                    parent['children_width'] += child['width']
                    break

    for frame in frames:
        frame['self_time'] = max(0.0, frame['width'] - frame['children_width'])

    return frames, total_width


def print_svg_report(filepath: str):
    frames, total_width = parse_flamegraph_svg(filepath)
    if not frames:
        print("No frames found", file=sys.stderr)
        sys.exit(1)

    agg: dict[str, float] = defaultdict(float)
    for f in frames:
        if not is_noise(f['name']) and f['self_time'] > 0:
            agg[f['name']] += f['self_time']

    print("NOTE: SVG mode is approximate. Use .folded for accurate hierarchy analysis.\n")
    print("TOP 30 HOTSPOTS (self-time, aggregated)")
    print(f"  {'self%':>6}  {'Function'}")
    for name, agg_self in sorted(agg.items(), key=lambda x: -x[1])[:30]:
        print(f"  {agg_self/total_width*100:6.2f}%  {short_name(name)}")
    print()

    OBJ_PATTERNS = {
        'MinResolution':     ['minresolution', 'min_resolution'],
        'CloudyArea':        ['cloudyarea', 'cloudy_area'],
        'TotalCost':         ['totalcost', 'total_cost'],
        'MaxIncidenceAngle': ['maxincidenceangle', 'max_incidence'],
    }
    print("OBJECTIVE SELF-TIME")
    for obj, patterns in OBJ_PATTERNS.items():
        total_self = sum(s for n, s in agg.items()
                        if any(p in n.lower() for p in patterns)
                        and 'objectivestate' not in n.lower())
        if total_self:
            print(f"  {obj:<22s}  {total_self/total_width*100:5.1f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Analyze PLS flamegraph profile (.folded preferred, .svg legacy)',
        epilog="Produce .folded: cargo flamegraph --post-process 'tee out.folded' --output out.svg ..."
    )
    parser.add_argument('profile_file', help='.folded or .svg file')
    args = parser.parse_args()

    if args.profile_file.endswith('.folded'):
        stacks = parse_folded(args.profile_file)
        if not stacks:
            print("No stacks found", file=sys.stderr)
            sys.exit(1)
        print_folded_report(stacks)
    else:
        print_svg_report(args.profile_file)


if __name__ == "__main__":
    main()
