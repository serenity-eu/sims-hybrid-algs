#!/usr/bin/env python3
"""
Flamegraph analyzer for AltTracker benchmark profiling.
Outputs only key performance percentages without advice.
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def parse_flamegraph_svg(filepath: str):
    """Parse flamegraph SVG and return frames with self-time percentages."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg', 'fg': 'http://github.com/jonhoo/inferno'}

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

        match = re.match(r'^(.*?) \((\d+(?:,\d+)*)(?: samples?)?, ([\d\.]+)%\)$', title_text)
        if match:
            func_name = match.group(1)
            percent = float(match.group(3))
        else:
            func_name = title_text
            percent = 0.0

        frame = {'name': func_name, 'percent': percent, 'width': width, 'x': x, 'y': y, 'children_width': 0.0}
        frames.append(frame)
        frames_by_y[y].append(frame)

    if not frames:
        return [], 0.0

    true_root = max(frames, key=lambda f: f['width'])
    total_width = true_root['width']

    # Build parent-child for self-time calculation
    y_levels = sorted(frames_by_y.keys())
    root_at_bottom = (true_root['y'] == y_levels[-1])
    ordered_levels = sorted(y_levels, reverse=root_at_bottom)

    for i in range(1, len(ordered_levels)):
        parent_y = ordered_levels[i - 1]
        child_y = ordered_levels[i]
        parents = [(p['x'], p['x'] + p['width'], p) for p in frames_by_y[parent_y]]
        parents.sort()

        for child in frames_by_y[child_y]:
            mid_x = child['x'] + child['width'] / 2.0
            for px_start, px_end, parent in parents:
                if px_start <= mid_x <= px_end:
                    parent['children_width'] += child['width']
                    break

    for frame in frames:
        frame['self_time'] = max(0.0, frame['width'] - frame['children_width'])

    return frames, total_width


def extract_metrics(frames, total_width):
    """Extract key percentages for optimization analysis."""
    if total_width <= 0:
        return {}

    NOISE = ['criterion', 'rayon_core', '__ieee754', '__GI___', '__math_', 'std::panic', 
             'futex', 'syscall', 'pthread', '<f64>::recip', '<f64>::ln', 'libm']

    def is_noise(name):
        name_lower = name.lower()
        return any(p.lower() in name_lower for p in NOISE)

    def self_pct(frame):
        return (frame['self_time'] / total_width) * 100.0

    def incl_pct(frame):
        return (frame['width'] / total_width) * 100.0

    # Objective breakdown (self-time)
    OBJECTIVES = {
        'TotalCost': ['totalcost', 'total_cost'],
        'CloudyArea': ['cloudyarea', 'cloudy_area'],
        'MinResolution': ['minresolution', 'min_resolution'],
        'MaxIncidenceAngle': ['maxincidenceangle', 'max_incidence', 'incidenceangle'],
    }

    objective_pct = {}
    for obj, patterns in OBJECTIVES.items():
        matching = [f for f in frames if any(p in f['name'].lower() for p in patterns) and 'objectivestate' not in f['name'].lower()]
        if matching:
            objective_pct[obj] = round(sum(f['self_time'] for f in matching) / total_width * 100, 1)

    # Method breakdown (self-time)
    METHODS = ['track_image_removal', 'track_image_addition', 'peek_removal_delta', 'peek_addition_delta']
    method_pct = {}
    for method in METHODS:
        matching = [f for f in frames if method in f['name'].lower()]
        if matching:
            method_pct[method] = round(sum(f['self_time'] for f in matching) / total_width * 100, 1)

    # Cross-tabulation: Objective × Method
    matrix = {}
    for obj, patterns in OBJECTIVES.items():
        for method in METHODS:
            matching = [f for f in frames if method in f['name'].lower() and any(p in f['name'].lower() for p in patterns)]
            if matching:
                key = f"{obj}::{method.replace('track_', '').replace('peek_', 'pk_')}"
                pct = sum(f['self_time'] for f in matching) / total_width * 100
                if pct >= 0.5:
                    matrix[key] = round(pct, 1)

    # Top hotspots (self-time) - individual functions
    tracker_frames = [f for f in frames if 
                      ('tracker' in f['name'].lower() or 'resolution' in f['name'].lower() or 
                       'cloudy' in f['name'].lower() or 'cost' in f['name'].lower() or 
                       'incidence' in f['name'].lower())
                      and not is_noise(f['name']) and f['self_time'] > 0]

    hotspots = []
    for f in sorted(tracker_frames, key=lambda x: x['self_time'], reverse=True)[:12]:
        # Extract short name
        state_match = re.search(r'(Alt)?([A-Z][A-Za-z]+)State', f['name'])
        method_match = re.search(r'>::(\w+)::', f['name'])
        if state_match:
            short = f"{state_match.group(1) or ''}{state_match.group(2)}::{method_match.group(1) if method_match else '?'}"
        elif 'position' in f['name'].lower():
            short = 'iter::position'
        elif 'replay_trace' in f['name']:
            short = 'replay_trace'
        else:
            parts = [p for p in f['name'].split('::') if p and not p.startswith('<')]
            short = '::'.join(parts[-2:])[:35] if len(parts) >= 2 else f['name'][:35]
        hotspots.append({'fn': short, 'self': round(self_pct(f), 1), 'incl': round(incl_pct(f), 1)})

    # Pareto front operations (self-time)
    PARETO_PATTERNS = {
        'nd_tree':      ['ndtree', 'nd_tree', 'ndnode', 'nd_node'],
        'try_insert':   ['try_insert', 'pareto_front::insert', 'paretofront::insert'],
        'dominates':    ['dominates', 'domination', 'is_dominated'],
        'vec_front':    ['vecsolutionset', 'vec_solution_set', 'vecparetofront'],
        'archive':      ['approximated_pareto_set', 'archive'],
    }
    pareto_pct = {}
    for label, patterns in PARETO_PATTERNS.items():
        matching = [f for f in frames if any(p in f['name'].lower() for p in patterns)]
        total = sum(f['self_time'] for f in matching)
        if total > 0:
            pareto_pct[label] = round(total / total_width * 100, 2)

    # Top-20 hotspots by self-time regardless of category
    all_hotspots = []
    for f in sorted(frames, key=lambda x: x['self_time'], reverse=True)[:20]:
        if is_noise(f['name']) or f['self_time'] <= 0:
            continue
        parts = [p for p in f['name'].split('::') if p and not p.startswith('<')]
        short = '::'.join(parts[-2:])[:50] if len(parts) >= 2 else f['name'][:50]
        all_hotspots.append({'fn': short, 'self': round(self_pct(f), 2), 'incl': round(incl_pct(f), 2)})

    return {
        'objective_pct': dict(sorted(objective_pct.items(), key=lambda x: -x[1])),
        'method_pct': dict(sorted(method_pct.items(), key=lambda x: -x[1])),
        'matrix': dict(sorted(matrix.items(), key=lambda x: -x[1])),
        'hotspots': hotspots,
        'pareto_pct': dict(sorted(pareto_pct.items(), key=lambda x: -x[1])),
        'all_hotspots': all_hotspots,
    }


def print_report(metrics, output_json=False):
    """Print key percentages."""
    if output_json:
        print(json.dumps(metrics, indent=2))
        return

    print("OBJECTIVE %")
    for obj, pct in metrics.get('objective_pct', {}).items():
        print(f"  {obj:20s} {pct:5.1f}%")

    print("\nMETHOD %")
    for method, pct in metrics.get('method_pct', {}).items():
        print(f"  {method:25s} {pct:5.1f}%")

    print("\nHOTSPOT MATRIX (Objective::Method)")
    for key, pct in list(metrics.get('matrix', {}).items())[:10]:
        print(f"  {key:35s} {pct:5.1f}%")

    print("\nPARETO FRONT %")
    total_pareto = sum(metrics.get('pareto_pct', {}).values())
    for label, pct in metrics.get('pareto_pct', {}).items():
        print(f"  {label:25s} {pct:5.2f}%")
    print(f"  {'TOTAL':25s} {total_pareto:5.2f}%")

    print("\nTOP 20 HOTSPOTS (self%)")
    for h in metrics.get('all_hotspots', []):
        print(f"  {h['self']:5.2f}%  {h['fn']}")

    print("\nTRACKER HOTSPOTS (self%)")
    for h in metrics.get('hotspots', [])[:8]:
        print(f"  {h['self']:5.1f}%  {h['fn']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('svg_file')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    frames, total_width = parse_flamegraph_svg(args.svg_file)
    if not frames:
        print("No frames found", file=sys.stderr)
        sys.exit(1)

    metrics = extract_metrics(frames, total_width)
    print_report(metrics, args.json)


if __name__ == "__main__":
    main()

