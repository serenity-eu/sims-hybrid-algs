#!/usr/bin/env python3
"""
Flamegraph analyzer specialized for AltTracker/StandardTracker benchmarks.

Extracts key metrics for optimization:
- Per-tracker-method time breakdown
- Per-objective time distribution
- Specific hotspot functions within tracker code

Usage:
    python analyze_tracker_flamegraph.py <flamegraph.svg> [--json]
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Frame:
    name: str
    samples: int
    percent: float
    width: float
    x: float
    y: float
    self_time: float = 0.0


def parse_flamegraph_svg(filepath: str) -> Tuple[List[Frame], float]:
    """Parse flamegraph SVG and return frames with total width."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg', 'fg': 'http://github.com/jonhoo/inferno'}

    frames = []
    frames_by_y: Dict[float, List[Frame]] = defaultdict(list)

    for elem in root.iter():
        if not elem.tag.endswith('g'):
            continue

        title_elem = elem.find('svg:title', ns)
        rect_elem = elem.find('svg:rect', ns)

        if title_elem is None or rect_elem is None:
            continue

        title_text = title_elem.text or ""
        
        # Parse width
        raw_width = rect_elem.get('{http://github.com/jonhoo/inferno}w')
        if raw_width:
            width = float(raw_width)
        else:
            w_str = rect_elem.get('width', '0')
            width = float(w_str.rstrip('%')) if w_str.endswith('%') else float(w_str)

        # Parse x
        raw_x = rect_elem.get('{http://github.com/jonhoo/inferno}x')
        if raw_x:
            x = float(raw_x)
        else:
            x_str = rect_elem.get('x', '0')
            x = float(x_str.rstrip('%')) if x_str.endswith('%') else float(x_str)

        y = float(rect_elem.get('y', '0'))

        # Parse title: "function_name (samples, %)" or "function_name (samples samples, %)"
        match = re.match(r'^(.*?) \((\d+(?:,\d+)*)(?: samples?)?, ([\d\.]+)%\)$', title_text)
        if match:
            func_name = match.group(1)
            samples = int(match.group(2).replace(',', ''))
            percent = float(match.group(3))
        else:
            func_name = title_text
            samples = 0
            percent = 0.0

        frame = Frame(
            name=func_name,
            samples=samples,
            percent=percent,
            width=width,
            x=x,
            y=y
        )
        frames.append(frame)
        frames_by_y[y].append(frame)

    if not frames:
        return [], 0.0

    # Find root (widest frame)
    true_root = max(frames, key=lambda f: f.width)
    total_width = true_root.width

    # Build parent-child relationships to calculate self-time
    y_levels = sorted(frames_by_y.keys())
    root_at_bottom = (true_root.y == y_levels[-1])
    ordered_levels = sorted(y_levels, reverse=root_at_bottom)

    # Link children to parents
    for i in range(1, len(ordered_levels)):
        parent_y = ordered_levels[i - 1]
        child_y = ordered_levels[i]
        parents = frames_by_y[parent_y]
        children = frames_by_y[child_y]

        # Build index for fast lookup
        parent_by_x: List[Tuple[float, float, Frame]] = [(p.x, p.x + p.width, p) for p in parents]
        parent_by_x.sort()

        for child in children:
            mid_x = child.x + child.width / 2.0
            for px_start, px_end, parent in parent_by_x:
                if px_start <= mid_x <= px_end:
                    # Child found for parent
                    if not hasattr(parent, '_children_width'):
                        parent._children_width = 0.0
                    parent._children_width += child.width
                    break

    # Calculate self-time
    for frame in frames:
        children_width = getattr(frame, '_children_width', 0.0)
        frame.self_time = max(0.0, frame.width - children_width)

    return frames, total_width


def extract_tracker_metrics(frames: List[Frame], total_width: float) -> Dict:
    """Extract key metrics specifically for tracker benchmark analysis."""
    if total_width <= 0:
        return {}

    # Filter out benchmark/criterion overhead and focus on tracker code
    NOISE_PATTERNS = [
        'criterion', 'rayon_core', '__ieee754', '__GI___', '__math_',
        'std::panic', 'std::panicking', '__clone', 'start_thread',
        'futex', 'syscall', '__x86_indirect_thunk', 'pthread',
        '<f64>::recip', '<f64>::ln', '<f64>::exp', 'libm',
    ]

    def is_noise(name: str) -> bool:
        name_lower = name.lower()
        return any(p.lower() in name_lower for p in NOISE_PATTERNS)

    def self_pct(frame: Frame) -> float:
        return (frame.self_time / total_width) * 100.0 if total_width > 0 else 0.0

    def inclusive_pct(frame: Frame) -> float:
        return (frame.width / total_width) * 100.0 if total_width > 0 else 0.0

    # 1. Identify tracker implementation type
    alt_tracker_frames = [f for f in frames if 'alternative_trackers' in f.name.lower() or 'alttracker' in f.name.lower()]
    std_tracker_frames = [f for f in frames if 'standard_trackers' in f.name.lower() or 'standardtracker' in f.name.lower()]

    tracker_type = "unknown"
    if alt_tracker_frames and not std_tracker_frames:
        tracker_type = "AltTracker"
    elif std_tracker_frames and not alt_tracker_frames:
        tracker_type = "StandardTracker"
    elif alt_tracker_frames and std_tracker_frames:
        tracker_type = "mixed"

    # 2. Method-level breakdown
    # Use self-time to avoid double-counting when methods call other methods
    TRACKER_METHODS = [
        'peek_removal_delta',
        'peek_addition_delta',
        'track_image_removal',
        'track_image_addition',
        'initialize_from',
    ]

    method_times: Dict[str, float] = {}
    for method in TRACKER_METHODS:
        # Match frames where the method name appears directly (not in child calls)
        matching = [f for f in frames 
                    if method in f.name.lower() 
                    and ('objectivetracker' in f.name.lower() or 'trackercollection' in f.name.lower())]
        if matching:
            # Use self-time to avoid double-counting
            total_method_time = sum(f.self_time for f in matching)
            method_times[method] = (total_method_time / total_width) * 100.0

    # 3. Objective-specific breakdown
    OBJECTIVES = [
        ('TotalCost', ['totalcost', 'total_cost']),
        ('CloudyArea', ['cloudyarea', 'cloudy_area']),
        ('MinResolution', ['minresolution', 'min_resolution']),
        ('MaxIncidenceAngle', ['maxincidenceangle', 'max_incidence', 'incidenceangle']),
    ]

    objective_times: Dict[str, float] = {}
    for obj_name, patterns in OBJECTIVES:
        matching = [f for f in frames 
                    if any(p in f.name.lower() for p in patterns)
                    and 'objectivestate' not in f.name.lower()]  # Skip type definitions
        if matching:
            # Use self-time to avoid counting overlapping child calls
            total_obj_self_time = sum(f.self_time for f in matching)
            objective_times[obj_name] = (total_obj_self_time / total_width) * 100.0

    # 4. Data structure operations (more precise matching)
    DATASTRUCTURES = [
        ('FixedBitSet', ['fixedbitset::']),
        ('slice iter', ['slice::iter', '<[u', '<[i']),
        ('Vec ops', ['vec::', 'rawvec']),
        ('Arc deref', ['arc::']),
        ('unchecked access', ['get_unchecked']),
        ('branch/cmp', ['core::cmp', 'ord::cmp']),
    ]

    ds_times: Dict[str, float] = {}
    for ds_name, patterns in DATASTRUCTURES:
        matching = [f for f in frames 
                    if any(p in f.name.lower() for p in patterns)
                    and not is_noise(f.name)]
        if matching:
            total_ds_time = sum(f.self_time for f in matching)
            ds_times[ds_name] = (total_ds_time / total_width) * 100.0

    # 5. Top hotspots (excluding noise)
    tracker_related = [f for f in frames 
                       if ('tracker' in f.name.lower() or 
                           'objective' in f.name.lower() or
                           'resolution' in f.name.lower() or
                           'cloudy' in f.name.lower() or
                           'cost' in f.name.lower() or
                           'incidence' in f.name.lower())
                       and not is_noise(f.name)
                       and f.self_time > 0]

    top_hotspots = sorted(tracker_related, key=lambda f: f.self_time, reverse=True)[:15]
    hotspots = [(f.name, self_pct(f), inclusive_pct(f)) for f in top_hotspots]

    # 6. Memory/allocation related
    alloc_frames = [f for f in frames 
                    if any(p in f.name.lower() for p in ['alloc', 'dealloc', 'drop', 'clone', 'to_vec', 'from_iter'])
                    and not is_noise(f.name)]
    alloc_time = sum(f.self_time for f in alloc_frames) / total_width * 100.0 if alloc_frames else 0.0

    # 7. Compute total tracker time (excluding noise)
    tracker_frames = [f for f in frames 
                      if ('tracker' in f.name.lower() or 
                          'pls::objective_tracker' in f.name.lower())
                      and not is_noise(f.name)]
    total_tracker_self = sum(f.self_time for f in tracker_frames) / total_width * 100.0

    # Find the replay_trace function to get benchmark payload time
    replay_frames = [f for f in frames if 'replay_trace' in f.name.lower()]
    replay_inclusive = max((f.width for f in replay_frames), default=0.0) / total_width * 100.0

    # 8. Operation counts breakdown by objective and method
    op_breakdown: Dict[str, Dict[str, float]] = {}
    for obj_name, patterns in OBJECTIVES:
        op_breakdown[obj_name] = {}
        for method in TRACKER_METHODS:
            matching = [f for f in frames 
                        if method in f.name.lower() 
                        and any(p in f.name.lower() for p in patterns)]
            if matching:
                total = sum(f.self_time for f in matching)
                op_breakdown[obj_name][method] = (total / total_width) * 100.0

    return {
        'tracker_type': tracker_type,
        'total_frames': len(frames),
        'total_tracker_self_pct': round(total_tracker_self, 2),
        'replay_trace_inclusive_pct': round(replay_inclusive, 2),
        'method_times_pct': {k: round(v, 2) for k, v in sorted(method_times.items(), key=lambda x: -x[1])},
        'objective_times_pct': {k: round(v, 2) for k, v in sorted(objective_times.items(), key=lambda x: -x[1])},
        'datastructure_times_pct': {k: round(v, 2) for k, v in sorted(ds_times.items(), key=lambda x: -x[1])},
        'allocation_time_pct': round(alloc_time, 2),
        'operation_breakdown': {obj: {m: round(v, 2) for m, v in methods.items() if v > 0.05} 
                                for obj, methods in op_breakdown.items() if methods},
        'top_hotspots': [(name[:200], round(self_p, 2), round(incl_p, 2)) for name, self_p, incl_p in hotspots if self_p > 0.1],
    }


def print_report(metrics: Dict, output_json: bool = False):
    """Print the analysis report."""
    if output_json:
        print(json.dumps(metrics, indent=2))
        return

    print("=" * 70)
    print(f"TRACKER PROFILE: {metrics.get('tracker_type', 'unknown')}")
    print("=" * 70)
    
    print(f"\nTracker Self-Time: {metrics.get('total_tracker_self_pct', 0):.1f}%")

    # Summary table
    print("\n" + "─" * 70)
    print("OBJECTIVE TIME DISTRIBUTION")
    print("─" * 70)
    total_obj = sum(metrics.get('objective_times_pct', {}).values())
    for obj, pct in metrics.get('objective_times_pct', {}).items():
        rel_pct = (pct / total_obj * 100) if total_obj > 0 else 0
        bar_len = int(rel_pct / 5)
        bar = "█" * bar_len + "░" * max(0, 20 - bar_len)
        print(f"  {obj:20s} {pct:5.1f}% ({rel_pct:4.1f}% rel) {bar}")

    # Method breakdown  
    print("\n" + "─" * 70)
    print("METHOD TIME DISTRIBUTION")
    print("─" * 70)
    for method, pct in metrics.get('method_times_pct', {}).items():
        if pct > 0.5:
            bar_len = int(pct / 3)
            bar = "█" * bar_len + "░" * max(0, 20 - bar_len)
            print(f"  {method:25s} {pct:5.1f}% {bar}")

    # Cross-tabulation: Objective × Method
    op_breakdown = metrics.get('operation_breakdown', {})
    if op_breakdown:
        print("\n" + "─" * 70)
        print("HOTSPOT MATRIX (Objective × Method, self-time %)")
        print("─" * 70)
        
        # Header
        methods = sorted(set(m for obj_methods in op_breakdown.values() for m in obj_methods))
        header = "                    " + "".join(f"{m[:10]:>12s}" for m in methods)
        print(header)
        
        for obj, obj_methods in op_breakdown.items():
            row = f"  {obj:18s}"
            for m in methods:
                val = obj_methods.get(m, 0)
                if val > 0.05:
                    row += f"{val:11.1f}%"
                else:
                    row += "           -"
            print(row)

    # Top hotspots
    print("\n" + "─" * 70)
    print("TOP FUNCTION HOTSPOTS")
    print("─" * 70)
    hotspots = metrics.get('top_hotspots', [])[:10]
    for name, self_pct, _incl_pct in hotspots:
        # Extract State type and method from Rust mangled name
        # Pattern: <...::AltMinResolutionState as ...>::track_image_removal::<...>
        state_match = re.search(r'(Alt)?([A-Z][A-Za-z]+)State', name)
        method_match = re.search(r'>::(\w+)::', name)  # Pattern: >::method_name::
        
        if state_match:
            prefix = state_match.group(1) or ""
            state = state_match.group(2)
            method = method_match.group(1) if method_match else "?"
            short = f"{prefix}{state}::{method}"
        elif 'Tracker' in name:
            tracker_match = re.search(r'(Alt|Standard)Tracker(?:Array)?', name)
            tracker = tracker_match.group(0) if tracker_match else "Tracker"
            method = method_match.group(1) if method_match else "?"
            short = f"{tracker}::{method}"
        elif 'replay_trace' in name:
            short = "replay_trace"
        elif 'position' in name.lower():
            short = "Iterator::position"
        else:
            # Fallback: last two path components
            parts = [p for p in name.split('::') if p and not p.startswith('<')]
            short = '::'.join(parts[-2:])[:40] if len(parts) >= 2 else name[:40]
        
        print(f"  {self_pct:5.1f}%  {short}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Analyze tracker benchmark flamegraph')
    parser.add_argument('svg_file', help='Path to flamegraph SVG file')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    try:
        frames, total_width = parse_flamegraph_svg(args.svg_file)
    except Exception as e:
        print(f"Error parsing SVG: {e}", file=sys.stderr)
        sys.exit(1)

    if not frames:
        print("No frames found in flamegraph", file=sys.stderr)
        sys.exit(1)

    metrics = extract_tracker_metrics(frames, total_width)
    print_report(metrics, args.json)


if __name__ == "__main__":
    main()
