#!/usr/bin/env python3
"""
Parse hypervolume data from result.json files and generate summary2.json.

This script processes result.json files from hybrid algorithm test runs,
validates the solutions to extract the Pareto front, computes hypervolume
from the solution objectives, and stores final hypervolumes in summary2.json.

Output format:
{
    "lagos_nigeria_30": {
        "hypervolumes": [0.85, 0.87, 0.89, 0.91]  # Final HV for each ratio
    },
    "paris_50": {
        "hypervolumes": [...]
    },
    ...
}
"""

import json
import sys
import tarfile
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    print("Warning: markdown not available, HTML export will not be available", file=sys.stderr)

# Import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    np = None
    plt = None
    print("Warning: matplotlib not available, plots will not be generated", file=sys.stderr)

# Import pymoo for hypervolume computation (C-optimized, handles dominated points)
try:
    from pymoo.indicators.hv import HV as PymooHV
except ImportError:
    print("Error: pymoo not found. Install with: pip install pymoo", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Result file cache -- avoids re-reading large JSON files multiple times
# ---------------------------------------------------------------------------
_result_cache: Dict[str, dict] = {}


def load_result_cached(result_path: Path) -> dict:
    """Load and cache a result.json file.  Subsequent calls return the cached copy."""
    key = str(result_path)
    if key not in _result_cache:
        with open(result_path, 'r') as f:
            _result_cache[key] = json.load(f)
    return _result_cache[key]


def clear_result_cache() -> None:
    """Free memory by clearing the result cache."""
    _result_cache.clear()


def extract_objectives_from_solution(solution: dict, objectives_list: List[str]) -> List[int]:
    """
    Extract objective values from a solution based on the objectives list.
    
    Args:
        solution: Solution dictionary from result.json
        objectives_list: List of objective names (e.g., ["min_cost", "cloud_coverage", ...])
        
    Returns:
        List of objective values in the order specified by objectives_list
    """
    objective_values = []
    
    for obj_name in objectives_list:
        if obj_name == "min_cost":
            objective_values.append(solution["cost"])
        elif obj_name == "cloud_coverage":
            objective_values.append(solution["cloudy_area"])
        elif obj_name == "min_max_incidence_angle":
            objective_values.append(solution["max_incidence_angle"])
        elif obj_name == "min_resolution":
            objective_values.append(solution["min_resolutions_sum"])
        else:
            raise ValueError(f"Unknown objective: {obj_name}")
    
    return objective_values


def dominates(sol_a: List[int], sol_b: List[int]) -> bool:
    """
    Check if solution A dominates solution B (for minimization).
    
    A dominates B if A is <= B in all objectives and < B in at least one.
    """
    if len(sol_a) != len(sol_b):
        return False
    
    at_least_one_better = False
    for a, b in zip(sol_a, sol_b):
        if a > b:  # A is worse in this objective
            return False
        if a < b:  # A is better in this objective
            at_least_one_better = True
    
    return at_least_one_better


def extract_pareto_front(solutions: List[dict], objectives_list: List[str]) -> List[List[int]]:
    """
    Extract the Pareto front from a list of solutions.
    
    Uses numpy vectorization for efficient O(n * m * k) filtering where
    n = number of solutions, m = non-dominated solutions, k = objectives.
    Falls back to pure Python if numpy is not available.
    
    Args:
        solutions: List of solution dictionaries
        objectives_list: List of objective names
        
    Returns:
        List of objective vectors for Pareto-optimal solutions
    """
    # Extract objective values for all solutions
    all_objectives = [
        extract_objectives_from_solution(sol, objectives_list)
        for sol in solutions
    ]
    
    if not all_objectives:
        return []
    
    if np is not None:
        return _extract_pareto_front_numpy(all_objectives)
    else:
        return _extract_pareto_front_pure(all_objectives)


def _extract_pareto_front_numpy(all_objectives: List[List[int]]) -> List[List[int]]:
    """
    Numpy-vectorized Pareto front extraction (minimization).
    
    Uses fully vectorized pairwise dominance checking.
    For n points with k objectives, runs in O(n^2 * k) but with
    numpy broadcast operations (no Python inner loop).
    """
    arr = np.array(all_objectives, dtype=np.int64)
    n = arr.shape[0]
    
    if n == 0:
        return []
    if n == 1:
        return [arr[0].tolist()]
    
    # De-duplicate first to reduce n
    unique_arr = np.unique(arr, axis=0)
    n_unique = unique_arr.shape[0]
    
    if n_unique == 1:
        return [unique_arr[0].tolist()]
    
    # Vectorized non-dominated filtering:
    # Point i is dominated if there exists any j where
    # all(arr[j] <= arr[i]) and any(arr[j] < arr[i])
    is_dominated = np.zeros(n_unique, dtype=bool)
    
    # Process in chunks to limit memory (n^2 can be large)
    chunk_size = min(n_unique, 2048)
    for i_start in range(0, n_unique, chunk_size):
        i_end = min(i_start + chunk_size, n_unique)
        # points_i shape: (chunk, 1, k), unique_arr shape: (1, n, k)
        points_i = unique_arr[i_start:i_end, np.newaxis, :]  # (chunk, 1, k)
        points_all = unique_arr[np.newaxis, :, :]             # (1, n, k)
        
        # le[c, j, k] = unique_arr[j, k] <= points_i[c, k]
        le = points_all <= points_i   # (chunk, n, k)
        lt = points_all < points_i    # (chunk, n, k)
        
        # j dominates i iff all objectives <= and at least one <
        all_le = le.all(axis=2)   # (chunk, n)
        any_lt = lt.any(axis=2)   # (chunk, n)
        dominates_i = all_le & any_lt  # (chunk, n)
        
        # Exclude self-domination (diagonal)
        for offset, idx in enumerate(range(i_start, i_end)):
            dominates_i[offset, idx] = False
        
        is_dominated[i_start:i_end] = dominates_i.any(axis=1)
    
    return unique_arr[~is_dominated].tolist()


def _extract_pareto_front_pure(all_objectives: List[List[int]]) -> List[List[int]]:
    """Pure Python fallback for Pareto front extraction."""
    # De-duplicate
    seen = set()
    unique = []
    for obj in all_objectives:
        key = tuple(obj)
        if key not in seen:
            seen.add(key)
            unique.append(obj)
    
    # Find Pareto front
    pareto_front = []
    for i, obj_i in enumerate(unique):
        is_dominated = False
        for j, obj_j in enumerate(unique):
            if i != j and dominates(obj_j, obj_i):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_front.append(obj_i)
    
    return pareto_front


def compute_objective_bounds(pareto_front: List[List[int]]) -> List[List[int]]:
    """
    Compute objective bounds from the Pareto front.
    
    Args:
        pareto_front: List of objective vectors
        
    Returns:
        List of [min, max] bounds for each objective
    """
    if not pareto_front:
        return []
    
    num_objectives = len(pareto_front[0])
    bounds = []
    
    for obj_idx in range(num_objectives):
        values = [sol[obj_idx] for sol in pareto_front]
        min_val = min(values)
        max_val = max(values)
        bounds.append([min_val, max_val])
    
    return bounds


def compute_reference_point(objective_bounds: List[List[int]], margin: float = 0.1) -> List[int]:
    """
    Compute a reference point from objective bounds.
    
    The reference point should be dominated by all Pareto front solutions.
    We add a margin to the maximum values.
    
    Args:
        objective_bounds: List of [min, max] for each objective
        margin: Fraction to add to max values (default 10%)
        
    Returns:
        Reference point as list of integers
    """
    reference = []
    for bounds in objective_bounds:
        max_val = bounds[1]
        # Add margin and round up
        ref_val = int(max_val * (1 + margin)) + 1
        reference.append(ref_val)
    
    return reference


def load_bounds_from_trace(trace_path: Path) -> Optional[Tuple[List[List[int]], List[int]]]:
    """
    Load objective bounds and reference point from a trace.tar.gz file.
    
    Args:
        trace_path: Path to trace.tar.gz file
        
    Returns:
        Tuple of (objective_bounds, reference_point) or None if extraction fails
    """
    try:
        with tarfile.open(trace_path, 'r:gz') as tar:
            # Extract metadata.json from the archive
            for member in tar.getmembers():
                if member.name.endswith('metadata.json'):
                    f = tar.extractfile(member)
                    if f is None:
                        return None
                    
                    metadata = json.load(f)
                    objective_bounds = metadata.get('objective_bounds')
                    reference_point = metadata.get('reference_point')
                    
                    if objective_bounds and reference_point:
                        return (objective_bounds, reference_point)
                    
                    return None
        
        return None
        
    except Exception as e:
        print(f"Error loading bounds from {trace_path}: {e}", file=sys.stderr)
        return None


def compute_hypervolume_from_result(result_path: Path) -> Optional[float]:
    """
    Compute hypervolume from a result.json file using bounds from trace.tar.gz.
    
    Args:
        result_path: Path to result.json file
        
    Returns:
        Hypervolume value as float, or None if computation fails
    """
    try:
        # First, try to load bounds from the corresponding trace.tar.gz
        trace_path = result_path.parent / "trace.tar.gz"
        objective_bounds = None
        reference_point = None
        
        if trace_path.exists():
            bounds_and_ref = load_bounds_from_trace(trace_path)
            if bounds_and_ref:
                objective_bounds, reference_point = bounds_and_ref
        
        # Load result.json (cached)
        data = load_result_cached(result_path)
        
        # Extract metadata
        objectives_list = data.get('objectives', [])
        solutions = data.get('solutions', [])
        
        if not objectives_list:
            print(f"Warning: No objectives found in {result_path}", file=sys.stderr)
            return None
        
        if not solutions:
            print(f"Warning: No solutions found in {result_path}", file=sys.stderr)
            return None
        
        # Extract objective values for all solutions
        all_objectives = [
            extract_objectives_from_solution(sol, objectives_list)
            for sol in solutions
        ]
        
        if not all_objectives:
            print(f"Warning: Empty objectives for {result_path}", file=sys.stderr)
            return None
        
        # If we couldn't load bounds from trace, compute them from all solutions
        if objective_bounds is None or reference_point is None:
            objective_bounds = compute_objective_bounds(all_objectives)
            reference_point = compute_reference_point(objective_bounds)
        
        # Solutions are already a Pareto front (non-dominated)
        if not all_objectives:
            print(f"Warning: Empty Pareto front for {result_path}", file=sys.stderr)
            return None
        
        # Normalize objectives to [0, 1] range using bounds and compute HV with pymoo
        print("Normalizing objectives and computing hypervolume with pymoo...", file=sys.stderr)
        bounds_arr = np.array(objective_bounds, dtype=np.float64)  # shape (k, 2)
        ref_arr = np.array(reference_point, dtype=np.float64)  # shape (k,)
        points_arr = np.array(all_objectives, dtype=np.float64)  # shape (nd, k)
        
        ranges = ref_arr - bounds_arr[:, 0]
        ranges[ranges == 0] = 1.0
        
        scaled_points = (points_arr - bounds_arr[:, 0]) / ranges
        scaled_ref = np.ones(len(reference_point))  # ref maps to 1.0

        print(f"Computing hypervolume with reference point: {reference_point} and bounds: {objective_bounds}", file=sys.stderr)
        hv = float(PymooHV(ref_point=scaled_ref)(scaled_points))
        
        return hv
        
    except Exception as e:
        print(f"Error processing {result_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def compute_hypervolume_from_result_with_bounds(
    result_path: Path,
    objective_bounds: List[List[int]],
    reference_point: List[int]
) -> Optional[float]:
    """
    Compute hypervolume from a result.json file using provided bounds and reference point.
    
    Args:
        result_path: Path to result.json file
        objective_bounds: Global objective bounds for the instance
        reference_point: Global reference point for the instance
        
    Returns:
        Hypervolume value as float, or None if computation fails
    """
    import time
    try:
        t0 = time.monotonic()
        # Load result.json (cached)
        data = load_result_cached(result_path)
        t_load = time.monotonic()
        
        # Extract metadata
        objectives_list = data.get('objectives', [])
        solutions = data.get('solutions', [])
        
        if not objectives_list:
            print(f"Warning: No objectives found in {result_path}", file=sys.stderr)
            return None
        
        if not solutions:
            print(f"Warning: No solutions found in {result_path}", file=sys.stderr)
            return None
        
        # Extract objective values for all solutions
        all_objectives = [
            extract_objectives_from_solution(sol, objectives_list)
            for sol in solutions
        ]
        t_extract = time.monotonic()
        
        # Solutions are already a Pareto front (non-dominated)
        if not all_objectives:
            print(f"Warning: Empty Pareto front for {result_path}", file=sys.stderr, flush=True)
            return None
        
        # Normalize objectives to [0, 1] range using bounds and compute HV with pymoo
        bounds_arr = np.array(objective_bounds, dtype=np.float64)
        ref_arr = np.array(reference_point, dtype=np.float64)
        points_arr = np.array(all_objectives, dtype=np.float64)
        
        ranges = ref_arr - bounds_arr[:, 0]
        ranges[ranges == 0] = 1.0
        
        scaled_points = (points_arr - bounds_arr[:, 0]) / ranges
        scaled_ref = np.ones(len(reference_point))
        
        hv = float(PymooHV(ref_point=scaled_ref)(scaled_points))
        t_hv = time.monotonic()
        
        print(f"    [{len(all_objectives)} pts] load={t_load-t0:.2f}s extract={t_extract-t_load:.2f}s hv={t_hv-t_extract:.2f}s total={t_hv-t0:.2f}s", file=sys.stderr, flush=True)
        
        return float(hv)
        
    except Exception as e:
        print(f"Error processing {result_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def extract_phase_counts(result_path: Path) -> Tuple[int, int]:
    """
    Extract solution counts by phase from a result.json file.
    
    Args:
        result_path: Path to result.json file
        
    Returns:
        Tuple of (exact_count, heuristic_count)
    """
    try:
        data = load_result_cached(result_path)
        
        solutions = data.get('solutions', [])
        exact_count = sum(1 for sol in solutions if sol.get('phase') == 'exact')
        heuristic_count = sum(1 for sol in solutions if sol.get('phase') == 'heuristic')
        
        return (exact_count, heuristic_count)
        
    except Exception as e:
        print(f"Error extracting phase counts from {result_path}: {e}", file=sys.stderr)
        return (0, 0)


def extract_instance_name(path: Path) -> Optional[str]:
    """
    Extract instance name from path structure.
    
    Expected structure:
    .../lagos_nigeria_30/0_100/iter0/result.json
    OR
    .../solve_two_phase_4d_small/lagos_nigeria_30/result.json
    OR
    .../solve_two_phase_4d_small_nd-tree_sequential/lagos_nigeria_30/100_0/iter0/result.json
    OR
    .../solve_two_phase_4d_small_nd-tree_concurrent/lagos_nigeria_30/100_0/iter0/result.json
    
    Returns: "lagos_nigeria_30"
    """
    if path.name == "result.json":
        # Try new structure: result.json -> iter0 -> 0_100 -> lagos_nigeria_30
        if path.parent.name.startswith('iter'):
            # Go up two more levels to get instance name
            return path.parent.parent.parent.name
        # Try old structure: result.json -> lagos_nigeria_30
        elif path.parent.name and not path.parent.name.startswith('solve_'):
            return path.parent.name
    return None


def extract_ratio_from_algorithm_dir(dir_name: str) -> Optional[Tuple[int, int]]:
    """
    Extract ratio from algorithm directory name.
    
    Example: "solve_two_phase_4d_small_0_100" -> (0, 100)
    Example: "solve_two_phase_4d_medium_50_50" -> (50, 50)
    
    Returns: Tuple (phase1_ratio, phase2_ratio) or None
    """
    parts = dir_name.split('_')
    
    # Look for pattern like [..., "ratio1", "ratio2"]
    if len(parts) >= 2:
        try:
            ratio2 = int(parts[-1])
            ratio1 = int(parts[-2])
            if ratio1 + ratio2 == 100 and 0 <= ratio1 <= 100 and 0 <= ratio2 <= 100:
                return (ratio1, ratio2)
        except ValueError:
            pass
    
    return None


def extract_ratio_from_path(path: Path) -> Optional[Tuple[int, int]]:
    """
    Extract ratio from path structure.
    
    Expected structure:
    .../lagos_nigeria_30/0_100/iter0/result.json -> (0, 100)
    OR
    .../solve_two_phase_4d_small_0_100/lagos_nigeria_30/result.json -> (0, 100)
    
    Returns: Tuple (phase1_ratio, phase2_ratio) or None
    """
    # Try new structure: look in grandparent of iter directory
    if path.parent.name.startswith('iter'):
        ratio_dir = path.parent.parent.name
        parts = ratio_dir.split('_')
        if len(parts) == 2:
            try:
                ratio1 = int(parts[0])
                ratio2 = int(parts[1])
                if ratio1 + ratio2 == 100 and 0 <= ratio1 <= 100 and 0 <= ratio2 <= 100:
                    return (ratio1, ratio2)
            except ValueError:
                pass
    
    # Try old structure: look in algorithm directory name
    for parent in path.parents:
        ratio = extract_ratio_from_algorithm_dir(parent.name)
        if ratio:
            return ratio
    
    return None


def export_to_html(md_file: Path, output_file: Optional[Path] = None, embed_imgs: bool = True) -> Optional[Path]:
    """
    Convert Markdown report to HTML with embedded images and styling.
    
    Args:
        md_file: Path to markdown file
        output_file: Optional output path (defaults to md_file with .html extension)
        embed_imgs: Whether to embed images as base64 data URIs
        
    Returns:
        Path to generated HTML file, or None if failed
    """
    if not MARKDOWN_AVAILABLE:
        print("Warning: markdown module not available, skipping HTML export", file=sys.stderr)
        return None
    
    if not md_file.exists():
        print(f"Warning: Markdown file {md_file} not found", file=sys.stderr)
        return None
    
    if output_file is None:
        output_file = md_file.with_suffix('.html')
    
    # Read markdown
    with open(md_file, 'r') as f:
        md_content = f.read()
    
    # Convert to HTML
    html_body = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'codehilite', 'nl2br', 'sane_lists']
    )
    
    # Embed images if requested
    if embed_imgs:
        plots_dir = md_file.parent / "plots"
        if plots_dir.exists():
            for img_file in plots_dir.glob("*.png"):
                with open(img_file, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                
                # Replace relative path with data URI
                img_path = f"plots/{img_file.name}"
                data_uri = f"data:image/png;base64,{img_data}"
                html_body = html_body.replace(f'src="{img_path}"', f'src="{data_uri}"')
    
    # Create full HTML with styling
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hypervolume Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #555;
            margin-top: 25px;
        }}
        h4 {{
            color: #666;
            margin-top: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        tr:nth-child(even) {{
            background-color: #fafafa;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}
        ul, ol {{
            padding-left: 30px;
        }}
        li {{
            margin: 8px 0;
        }}
        ul ul, ol ul {{
            padding-left: 40px;
            margin-top: 5px;
        }}
        li > ul, li > ol {{
            margin-top: 5px;
            margin-bottom: 5px;
        }}
        .summary-stats {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_body}
    </div>
</body>
</html>
"""
    
    # Write output
    with open(output_file, 'w') as f:
        f.write(html_template)
    
    print(f"HTML report generated: {output_file}", file=sys.stderr)
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB", file=sys.stderr)
    
    return output_file


def compute_unified_bounds(directories: List[Path]) -> Tuple[Dict, Dict]:
    """
    Compute unified global bounds across multiple directories.
    
    Args:
        directories: List of artifact directories to merge bounds from
        
    Returns:
        Tuple of (unified_bounds_map, unified_ref_map)
    """
    instance_global_bounds = defaultdict(lambda: {"bounds_list": [], "reference_list": [], "pareto_fronts": []})
    
    for artifacts_dir in directories:
        result_files = list(artifacts_dir.rglob("result.json"))
        
        for result_path in result_files:
            instance_name = extract_instance_name(result_path)
            if instance_name is None:
                continue
            
            # Try to load bounds from trace.tar.gz if it exists
            trace_path = result_path.parent / "trace.tar.gz"
            if trace_path.exists():
                bounds_and_ref = load_bounds_from_trace(trace_path)
                if bounds_and_ref:
                    bounds, ref = bounds_and_ref
                    instance_global_bounds[instance_name]["bounds_list"].append(bounds)
                    instance_global_bounds[instance_name]["reference_list"].append(ref)
            
            # Fallback: compute bounds directly from all solutions (no Pareto filtering needed)
            try:
                data = load_result_cached(result_path)
                objectives_list = data.get('objectives', [])
                solutions = data.get('solutions', [])
                if objectives_list and solutions:
                    # Extract objective values for all solutions to get min/max bounds
                    all_objectives = [
                        extract_objectives_from_solution(sol, objectives_list)
                        for sol in solutions
                    ]
                    if all_objectives:
                        # Compute bounds directly - no need for expensive Pareto filtering
                        num_obj = len(all_objectives[0])
                        bounds = []
                        for obj_idx in range(num_obj):
                            values = [sol[obj_idx] for sol in all_objectives]
                            bounds.append([min(values), max(values)])
                        instance_global_bounds[instance_name]["bounds_list"].append(bounds)
            except Exception as e:
                print(f"Warning: Could not load bounds from {result_path}: {e}", file=sys.stderr)
    
    # Compute unified global bounds for each instance
    unified_bounds_map = {}
    unified_ref_map = {}
    
    for instance_name, data in instance_global_bounds.items():
        # If we have bounds from trace files, use those
        if data["bounds_list"]:
            num_objectives = len(data["bounds_list"][0])
            global_bounds = []
            
            for obj_idx in range(num_objectives):
                all_mins = [bounds[obj_idx][0] for bounds in data["bounds_list"]]
                all_maxs = [bounds[obj_idx][1] for bounds in data["bounds_list"]]
                global_min = min(all_mins)
                global_max = max(all_maxs)
                global_bounds.append([global_min, global_max])
            
            global_ref = [b[1] + 1 for b in global_bounds]
            unified_bounds_map[instance_name] = global_bounds
            unified_ref_map[instance_name] = global_ref
        
        if instance_name in unified_bounds_map:
            print(f"  {instance_name}: unified_bounds={unified_bounds_map[instance_name]}, ref={unified_ref_map[instance_name]}", file=sys.stderr)
    
    return unified_bounds_map, unified_ref_map


def recompute_with_unified_bounds(artifacts_dir: Path, unified_bounds_map: Dict, unified_ref_map: Dict) -> Dict:
    """
    Recompute hypervolumes using unified global bounds.
    
    Args:
        artifacts_dir: Directory containing result files
        unified_bounds_map: Unified bounds for all instances
        unified_ref_map: Unified reference points for all instances
        
    Returns:
        Dictionary mapping instance names to their hypervolume data
    """
    results = defaultdict(lambda: {"ratio_data": {}, "hypervolumes": [], "hypervolumes_stderr": [], "ratios": [], "phase_counts": []})
    
    # Find all result.json files
    result_files = list(artifacts_dir.rglob("result.json"))
    
    if not result_files:
        print(f"Warning: No result.json files found in {artifacts_dir}", file=sys.stderr)
        return {}
    
    print(f"Found {len(result_files)} result files to process", file=sys.stderr, flush=True)
    
    for file_idx, result_path in enumerate(sorted(result_files), 1):
        # Extract instance name
        instance_name = extract_instance_name(result_path)
        if instance_name is None:
            print(f"Warning: Could not extract instance name from {result_path}", file=sys.stderr, flush=True)
            continue
        
        # Extract ratio from path
        ratio = extract_ratio_from_path(result_path)
        if ratio is None:
            print(f"Warning: Could not extract ratio from {result_path}", file=sys.stderr, flush=True)
            continue
        
        # Get unified bounds for this instance
        if instance_name not in unified_bounds_map:
            print(f"Warning: No unified bounds for {instance_name}", file=sys.stderr, flush=True)
            continue
        
        objective_bounds = unified_bounds_map[instance_name]
        reference_point = unified_ref_map[instance_name]
        
        print(f"  [{file_idx}/{len(result_files)}] Processing {instance_name} ratio={ratio}...", file=sys.stderr, flush=True)
        
        # Compute hypervolume from result.json with unified bounds
        hv = compute_hypervolume_from_result_with_bounds(
            result_path, objective_bounds, reference_point
        )
        if hv is None:
            continue
        
        # Extract phase counts
        exact_count, heuristic_count = extract_phase_counts(result_path)
        
        # Create unique key for (instance, ratio) pair
        ratio_key = f"{ratio[0]}_{ratio[1]}"
        if "ratio_data" not in results[instance_name]:
            results[instance_name]["ratio_data"] = {}
        
        if ratio_key not in results[instance_name]["ratio_data"]:
            results[instance_name]["ratio_data"][ratio_key] = {
                "hypervolumes": [],
                "phase_counts": [],
                "ratio": ratio
            }
        
        results[instance_name]["ratio_data"][ratio_key]["hypervolumes"].append(hv)
        results[instance_name]["ratio_data"][ratio_key]["phase_counts"].append((exact_count, heuristic_count))
        
        print(f"  {instance_name} [{ratio[0]}/{ratio[1]}]: HV = {hv:.6f}", file=sys.stderr)
    
    # Compute averages and standard errors for each (instance, ratio) pair
    import numpy as np
    
    for instance_name in results:
        ratio_data = results[instance_name]["ratio_data"]
        
        # Initialize as empty lists (clearing any defaults)
        hvs_list = []
        stderr_list = []
        ratios_list = []
        phase_counts_list = []
        
        # Sort by ratio to ensure consistent ordering: 0/100, 20/80, 50/50, 100/0
        sorted_ratios = sorted(ratio_data.items(), key=lambda x: x[1]["ratio"])
        
        for ratio_key, data in sorted_ratios:
            hvs = np.array(data["hypervolumes"])
            mean_hv = np.mean(hvs)
            stderr_hv = np.std(hvs, ddof=1) / np.sqrt(len(hvs)) if len(hvs) > 1 else 0.0
            
            hvs_list.append(mean_hv)
            stderr_list.append(stderr_hv)
            ratios_list.append(data["ratio"])
            
            # Compute average phase counts
            if data["phase_counts"]:
                avg_exact = np.mean([pc[0] for pc in data["phase_counts"]])
                avg_heuristic = np.mean([pc[1] for pc in data["phase_counts"]])
                phase_counts_list.append((avg_exact, avg_heuristic))
            else:
                phase_counts_list.append((0, 0))
        
        # Replace with computed lists
        results[instance_name]["hypervolumes"] = hvs_list
        results[instance_name]["hypervolumes_stderr"] = stderr_list
        results[instance_name]["ratios"] = ratios_list
        results[instance_name]["phase_counts"] = phase_counts_list
    
    return dict(results)


def parse_test_artifacts(artifacts_dir: Path) -> Tuple[Dict[str, Dict[str, List[float]]], Dict, Dict]:
    """
    Parse all result.json files in the test artifacts directory.
    
    Args:
        artifacts_dir: Root directory containing algorithm subdirectories
        
    Returns:
        Tuple of (results, global_bounds_map, global_ref_map) where:
        - results: Dictionary mapping instance names to their hypervolume data
        - global_bounds_map: Global bounds for each instance
        - global_ref_map: Global reference points for each instance
    """
    results = defaultdict(lambda: {"ratio_data": {}, "hypervolumes": [], "hypervolumes_stderr": [], "ratios": [], "phase_counts": []})
    
    # Find all result.json files
    result_files = list(artifacts_dir.rglob("result.json"))
    
    if not result_files:
        print(f"Warning: No result.json files found in {artifacts_dir}", file=sys.stderr)
        return {}, {}, {}
    
    print(f"Found {len(result_files)} result files to process", file=sys.stderr)
    
    # First pass: collect all bounds for each instance across all ratios
    print("\nCollecting global bounds for each instance...", file=sys.stderr)
    instance_global_bounds = defaultdict(lambda: {"bounds_list": [], "reference_list": []})
    
    for result_path in result_files:
        instance_name = extract_instance_name(result_path)
        if instance_name is None:
            continue
        
        # Try to load bounds from trace.tar.gz if it exists
        trace_path = result_path.parent / "trace.tar.gz"
        if trace_path.exists():
            bounds_and_ref = load_bounds_from_trace(trace_path)
            if bounds_and_ref:
                bounds, ref = bounds_and_ref
                instance_global_bounds[instance_name]["bounds_list"].append(bounds)
                instance_global_bounds[instance_name]["reference_list"].append(ref)
        
        # Fallback: compute bounds directly from all solutions (no Pareto filtering needed)
        try:
            data = load_result_cached(result_path)
            objectives_list = data.get('objectives', [])
            solutions = data.get('solutions', [])
            if objectives_list and solutions:
                all_objectives = [
                    extract_objectives_from_solution(sol, objectives_list)
                    for sol in solutions
                ]
                if all_objectives:
                    num_obj = len(all_objectives[0])
                    bounds = []
                    for obj_idx in range(num_obj):
                        values = [sol[obj_idx] for sol in all_objectives]
                        bounds.append([min(values), max(values)])
                    instance_global_bounds[instance_name]["bounds_list"].append(bounds)
        except Exception as e:
            print(f"Warning: Could not load bounds from {result_path}: {e}", file=sys.stderr)
    
    # Compute global bounds for each instance
    global_bounds_map = {}
    global_ref_map = {}
    
    for instance_name, data in instance_global_bounds.items():
        if data["bounds_list"]:
            num_objectives = len(data["bounds_list"][0])
            global_bounds = []
            
            for obj_idx in range(num_objectives):
                all_mins = [bounds[obj_idx][0] for bounds in data["bounds_list"]]
                all_maxs = [bounds[obj_idx][1] for bounds in data["bounds_list"]]
                global_min = min(all_mins)
                global_max = max(all_maxs)
                global_bounds.append([global_min, global_max])
            
            global_ref = [b[1] + 1 for b in global_bounds]
            global_bounds_map[instance_name] = global_bounds
            global_ref_map[instance_name] = global_ref
        
        if instance_name in global_bounds_map:
            print(f"  {instance_name}: bounds={global_bounds_map[instance_name]}, ref={global_ref_map[instance_name]}", file=sys.stderr)
    
    # Second pass: compute hypervolumes using global bounds
    print("\nComputing hypervolumes with global bounds...", file=sys.stderr)
    
    for result_path in sorted(result_files):
        # Extract instance name
        instance_name = extract_instance_name(result_path)
        if instance_name is None:
            print(f"Warning: Could not extract instance name from {result_path}", file=sys.stderr)
            continue
        
        # Extract ratio from path
        ratio = extract_ratio_from_path(result_path)
        if ratio is None:
            print(f"Warning: Could not extract ratio from {result_path}", file=sys.stderr)
            continue
        
        # Get global bounds for this instance
        if instance_name not in global_bounds_map:
            print(f"Warning: No global bounds for {instance_name}", file=sys.stderr)
            continue
        
        objective_bounds = global_bounds_map[instance_name]
        reference_point = global_ref_map[instance_name]
        
        # Compute hypervolume from result.json with global bounds
        hv = compute_hypervolume_from_result_with_bounds(
            result_path, objective_bounds, reference_point
        )
        if hv is None:
            continue
        
        # Extract phase counts
        exact_count, heuristic_count = extract_phase_counts(result_path)
        
        # Create unique key for (instance, ratio) pair
        ratio_key = f"{ratio[0]}_{ratio[1]}"
        if "ratio_data" not in results[instance_name]:
            results[instance_name]["ratio_data"] = {}
        
        if ratio_key not in results[instance_name]["ratio_data"]:
            results[instance_name]["ratio_data"][ratio_key] = {
                "hypervolumes": [],
                "phase_counts": [],
                "ratio": ratio
            }
        
        results[instance_name]["ratio_data"][ratio_key]["hypervolumes"].append(hv)
        results[instance_name]["ratio_data"][ratio_key]["phase_counts"].append((exact_count, heuristic_count))
        
        print(f"  {instance_name} [{ratio[0]}/{ratio[1]}]: HV = {hv:.6f}", file=sys.stderr)
    
    # Compute averages and standard errors for each (instance, ratio) pair
    import numpy as np
    
    for instance_name in results:
        ratio_data = results[instance_name]["ratio_data"]
        
        # Initialize as empty lists (clearing any defaults)
        hvs_list = []
        stderr_list = []
        ratios_list = []
        phase_counts_list = []
        
        # Sort by ratio to ensure consistent ordering: 0/100, 20/80, 50/50, 100/0
        sorted_ratios = sorted(ratio_data.items(), key=lambda x: x[1]["ratio"])
        
        for ratio_key, data in sorted_ratios:
            hvs = np.array(data["hypervolumes"])
            mean_hv = np.mean(hvs)
            stderr_hv = np.std(hvs, ddof=1) / np.sqrt(len(hvs)) if len(hvs) > 1 else 0.0
            
            # Average phase counts as well
            exact_counts = [pc[0] for pc in data["phase_counts"]]
            heur_counts = [pc[1] for pc in data["phase_counts"]]
            mean_exact = int(np.mean(exact_counts))
            mean_heur = int(np.mean(heur_counts))
            
            hvs_list.append(mean_hv)
            stderr_list.append(stderr_hv)
            ratios_list.append(data["ratio"])
            phase_counts_list.append((mean_exact, mean_heur))
        
        # Replace with computed values
        results[instance_name]["hypervolumes"] = hvs_list
        results[instance_name]["hypervolumes_stderr"] = stderr_list
        results[instance_name]["ratios"] = ratios_list
        results[instance_name]["phase_counts"] = phase_counts_list
    
    return dict(results), global_bounds_map, global_ref_map


def create_summary(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Create the summary output format preserving the averages and standard errors.
    
    Args:
        results: Full results with hypervolumes already averaged per ratio
        
    Returns:
        Dictionary with hypervolumes, stderr, and ratios
    """
    summary = {}
    for instance_name, data in results.items():
        summary[instance_name] = {
            "hypervolumes": data["hypervolumes"],
            "hypervolumes_stderr": data.get("hypervolumes_stderr", []),
            "ratios": data["ratios"]
        }
    return summary


def analyze_zero_hypervolumes(artifacts_dir: Path, results: Dict[str, Dict[str, List]]) -> List[Dict]:
    """
    Analyze why certain hypervolumes are zero.
    
    Args:
        artifacts_dir: Root directory containing algorithm subdirectories
        results: Full results with hypervolumes and ratios
        
    Returns:
        List of analysis records for zero hypervolumes
    """
    zero_analyses = []
    
    for instance_name, data in results.items():
        hypervolumes = data["hypervolumes"]
        ratios = data["ratios"]
        
        for i, hv in enumerate(hypervolumes):
            if hv == 0.0:
                ratio = ratios[i]
                # Find the corresponding result.json file
                ratio_str = f"{ratio[0]}_{ratio[1]}"
                result_files = list(artifacts_dir.rglob(f"*{ratio_str}/{instance_name}/result.json"))
                
                if result_files:
                    result_path = result_files[0]
                    
                    try:
                        result_data = load_result_cached(result_path)
                        
                        num_solutions = len(result_data.get('solutions', []))
                        objectives_list = result_data.get('objectives', [])
                        
                        # Extract Pareto front
                        solutions = result_data.get('solutions', [])
                        if solutions:
                            pareto_front = extract_pareto_front(solutions, objectives_list)
                            
                            analysis = {
                                'instance': instance_name,
                                'ratio': f"{ratio[0]}/{ratio[1]}",
                                'total_solutions': num_solutions,
                                'pareto_size': len(pareto_front),
                                'reason': ''
                            }
                            
                            if len(pareto_front) == 0:
                                analysis['reason'] = 'Empty Pareto front'
                            elif len(pareto_front) == 1:
                                # Check if single solution is at boundary of global bounds
                                sol_obj = pareto_front[0]
                                
                                # Get global bounds for this instance
                                trace_files = list(artifacts_dir.rglob(f"{instance_name}/trace.tar.gz"))
                                if trace_files:
                                    # Load global bounds
                                    all_bounds = []
                                    for trace_path in trace_files:
                                        bounds_and_ref = load_bounds_from_trace(trace_path)
                                        if bounds_and_ref:
                                            all_bounds.append(bounds_and_ref[0])
                                    
                                    if all_bounds:
                                        num_objectives = len(all_bounds[0])
                                        global_bounds = []
                                        
                                        for obj_idx in range(num_objectives):
                                            all_mins = [bounds[obj_idx][0] for bounds in all_bounds]
                                            all_maxs = [bounds[obj_idx][1] for bounds in all_bounds]
                                            global_min = min(all_mins)
                                            global_max = max(all_maxs)
                                            global_bounds.append([global_min, global_max])
                                        
                                        # Check how many objectives are at boundaries
                                        at_max_boundary = sum(1 for i, val in enumerate(sol_obj) 
                                                             if val >= global_bounds[i][1])
                                        at_min_boundary = sum(1 for i, val in enumerate(sol_obj) 
                                                             if val <= global_bounds[i][0])
                                        beyond_bounds = sum(1 for i, val in enumerate(sol_obj) 
                                                           if val > global_bounds[i][1])
                                        
                                        analysis['reason'] = (
                                            f'Single solution at boundary: {at_max_boundary} objectives at max bound, '
                                            f'{at_min_boundary} at min bound, {beyond_bounds} beyond bounds'
                                        )
                                        analysis['solution_objectives'] = sol_obj
                                        analysis['global_bounds'] = global_bounds
                                else:
                                    analysis['reason'] = 'Single solution (bounds not found)'
                            else:
                                # Multiple solutions but still zero HV - likely all dominated or at boundary
                                analysis['reason'] = f'{len(pareto_front)} solutions in Pareto front, likely at global bounds boundary'
                            
                            zero_analyses.append(analysis)
                            
                    except Exception as e:
                        zero_analyses.append({
                            'instance': instance_name,
                            'ratio': f"{ratio[0]}/{ratio[1]}",
                            'total_solutions': 0,
                            'pareto_size': 0,
                            'reason': f'Error analyzing: {str(e)}'
                        })
    
    return zero_analyses


def compute_phase_hypervolumes(
    result_path: Path,
    objective_bounds: List[List[int]],
    reference_point: List[int]
) -> Tuple[Optional[float], Optional[float], int, int]:
    """
    Compute hypervolumes for phase 1 (exact) and cumulative (exact + heuristic).
    
    Args:
        result_path: Path to result.json file
        objective_bounds: Global objective bounds for the instance
        reference_point: Global reference point for the instance
        
    Returns:
        Tuple of (phase1_hv, total_hv, exact_count, heuristic_count)
    """
    try:
        data = load_result_cached(result_path)
        
        objectives_list = data.get('objectives', [])
        solutions = data.get('solutions', [])
        
        if not objectives_list or not solutions:
            return (None, None, 0, 0)
        
        # Separate solutions by phase
        exact_solutions = [sol for sol in solutions if sol.get('phase') == 'exact']
        heuristic_solutions = [sol for sol in solutions if sol.get('phase') == 'heuristic']
        
        exact_count = len(exact_solutions)
        heuristic_count = len(heuristic_solutions)
        
        # Pre-compute normalization parameters (shared)
        bounds_arr = np.array(objective_bounds, dtype=np.float64)
        ref_arr = np.array(reference_point, dtype=np.float64)
        ranges = ref_arr - bounds_arr[:, 0]
        ranges[ranges == 0] = 1.0
        scaled_ref = np.ones(len(reference_point))
        
        # Compute phase 1 hypervolume (exact only)
        # pymoo HV internally filters dominated points, no need for extract_pareto_front
        phase1_hv = None
        if exact_solutions:
            phase1_objs = [
                extract_objectives_from_solution(sol, objectives_list)
                for sol in exact_solutions
            ]
            if phase1_objs:
                pts = np.array(phase1_objs, dtype=np.float64)
                scaled = (pts - bounds_arr[:, 0]) / ranges
                phase1_hv = float(PymooHV(ref_point=scaled_ref)(scaled))
        
        # Compute total hypervolume (all solutions -- already a Pareto front)
        total_objs = [
            extract_objectives_from_solution(sol, objectives_list)
            for sol in solutions
        ]
        total_hv = None
        if total_objs:
            pts = np.array(total_objs, dtype=np.float64)
            scaled = (pts - bounds_arr[:, 0]) / ranges
            total_hv = float(PymooHV(ref_point=scaled_ref)(scaled))
        
        return (float(phase1_hv) if phase1_hv is not None else None,
                float(total_hv) if total_hv is not None else None,
                exact_count, heuristic_count)
        
    except Exception as e:
        print(f"Error computing phase hypervolumes for {result_path}: {e}", file=sys.stderr)
        return (None, None, 0, 0)


def generate_plots(results: Dict, output_dir: Path, artifacts_dir: Path, 
                   global_bounds_map: Dict, global_ref_map: Dict) -> Dict[str, str]:
    """
    Generate composed bar plots for each instance showing phase-wise hypervolumes with error bars.
    
    Args:
        results: Full results dictionary with hypervolumes, ratios, and phase_counts
        output_dir: Directory to save plot images
        artifacts_dir: Root directory containing algorithm subdirectories
        global_bounds_map: Global bounds for each instance
        global_ref_map: Global reference points for each instance
        
    Returns:
        Dictionary mapping instance names to relative plot paths
    """
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib not available, skipping plot generation", file=sys.stderr)
        return {}
    
    # Create plots subdirectory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    plot_paths = {}
    ratio_labels = ['100/0\n(Pure Exact)', '50/50\n(Hybrid)', '20/80\n(Mostly Heur.)', '0/100\n(Pure Heur.)']
    all_ratios = [(100, 0), (50, 50), (20, 80), (0, 100)]  # All possible ratios in order
    
    # Sapphire and Ruby colors
    sapphire_color = '#0F52BA'  # Sapphire blue for exact solver
    ruby_color = '#E0115F'      # Ruby red for PLS solver
    
    for instance_name, data in results.items():
        ratios = data["ratios"]
        
        # Get global bounds for this instance
        if instance_name not in global_bounds_map:
            continue
        
        objective_bounds = global_bounds_map[instance_name]
        reference_point = global_ref_map[instance_name]
        
        # Build a map from ratio to index in the data (convert lists to tuples for lookup)
        ratio_to_idx = {tuple(ratio): i for i, ratio in enumerate(ratios)}
        
        # Compute phase-wise hypervolumes for all possible ratios
        phase_data = []
        for ratio in all_ratios:
            # Check if this ratio exists in the data
            if ratio not in ratio_to_idx:
                # Ratio not available - mark as N/A
                phase_data.append(None)
                continue
            
            ratio_str = f"{ratio[0]}_{ratio[1]}"
            
            # Find all result files for this instance and ratio (across iterations)
            result_files = list(artifacts_dir.rglob(f"*/{instance_name}/{ratio_str}/iter*/result.json"))
            
            if result_files:
                # Compute phase HVs for each iteration
                phase1_hvs = []
                total_hvs = []
                exact_counts = []
                heur_counts = []
                
                for result_file in result_files:
                    phase1_hv, total_hv, exact_cnt, heur_cnt = compute_phase_hypervolumes(
                        result_file, objective_bounds, reference_point
                    )
                    if total_hv is not None:
                        # phase1_hv can be None for pure heuristic (0/100) ratios
                        phase1_hvs.append(phase1_hv if phase1_hv is not None else 0.0)
                        total_hvs.append(total_hv)
                        exact_counts.append(exact_cnt)
                        heur_counts.append(heur_cnt)
                
                # Compute means and stderr
                if phase1_hvs and total_hvs:
                    mean_phase1 = np.mean(phase1_hvs)
                    mean_total = np.mean(total_hvs)
                    stderr_phase1 = np.std(phase1_hvs, ddof=1) / np.sqrt(len(phase1_hvs)) if len(phase1_hvs) > 1 else 0.0
                    stderr_total = np.std(total_hvs, ddof=1) / np.sqrt(len(total_hvs)) if len(total_hvs) > 1 else 0.0
                    mean_exact = int(np.mean(exact_counts))
                    mean_heur = int(np.mean(heur_counts))
                    
                    phase_data.append({
                        'phase1_hv': mean_phase1,
                        'phase1_stderr': stderr_phase1,
                        'total_hv': mean_total,
                        'total_stderr': stderr_total,
                        'exact_count': mean_exact,
                        'heuristic_count': mean_heur
                    })
                else:
                    phase_data.append({
                        'phase1_hv': 0.0,
                        'phase1_stderr': 0.0,
                        'total_hv': 0.0,
                        'total_stderr': 0.0,
                        'exact_count': 0,
                        'heuristic_count': 0
                    })
            else:
                phase_data.append({
                    'phase1_hv': 0.0,
                    'phase1_stderr': 0.0,
                    'total_hv': 0.0,
                    'total_stderr': 0.0,
                    'exact_count': 0,
                    'heuristic_count': 0
                })
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for stacked bars with error bars
        x_pos = np.arange(len(ratio_labels))
        phase1_hvs = []
        phase2_hvs = []
        phase1_stderrs = []
        total_stderrs = []
        exact_counts = []
        heuristic_counts = []
        total_hvs = []
        
        for d in phase_data:
            if d is None:
                # N/A data - use -0.05 as a sentinel value to display N/A
                phase1_hvs.append(0.0)
                phase2_hvs.append(0.0)
                phase1_stderrs.append(0.0)
                total_stderrs.append(0.0)
                exact_counts.append(0)
                heuristic_counts.append(0)
                total_hvs.append(-0.05)  # Sentinel for N/A
            else:
                phase1_hvs.append(d['phase1_hv'])
                phase2_hvs.append(d['total_hv'] - d['phase1_hv'])
                phase1_stderrs.append(d['phase1_stderr'])
                total_stderrs.append(d['total_stderr'])
                exact_counts.append(d['exact_count'])
                heuristic_counts.append(d['heuristic_count'])
                total_hvs.append(d['total_hv'])
        
        # Create stacked bars
        bar_width = 0.6
        bars1 = ax.bar(x_pos, phase1_hvs, bar_width, label='Exact Phase', 
                      color=sapphire_color, alpha=0.9, edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x_pos, phase2_hvs, bar_width, bottom=phase1_hvs, 
                      label='PLS Phase', color=ruby_color, alpha=0.9, 
                      edgecolor='black', linewidth=1.2)
        
        # Add error bars for total hypervolume (only where data exists)
        for i, (x, y, err) in enumerate(zip(x_pos, total_hvs, total_stderrs)):
            if y >= 0:  # Only add error bars for valid data (not N/A)
                ax.errorbar(x, y, yerr=err, fmt='none', 
                           ecolor='black', elinewidth=2, capsize=5, capthick=2, zorder=10)
        
        # Customize plot
        ax.set_xlabel('Algorithm Configuration (Exact/Heuristic Ratio)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Hypervolume', fontsize=16, fontweight='bold')
        ax.set_title(f'Hypervolume Performance by Phase: {instance_name}', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ratio_labels, fontsize=13)
        ax.set_ylim(0, 1.19)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=13)
        
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=13)
        
        # Add value labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            total_hv = total_hvs[i]
            phase1_hv = phase1_hvs[i]
            phase2_hv = phase2_hvs[i]
            exact_cnt = exact_counts[i]
            heur_cnt = heuristic_counts[i]
            total_sols = exact_cnt + heur_cnt
            
            if total_hv < 0:
                # N/A data
                ax.text(bar1.get_x() + bar1.get_width()/2., 0.1,
                       'N/A',
                       ha='center', va='bottom', fontsize=14, fontweight='bold', 
                       color='gray', style='italic')
            elif total_hv > 0:
                # Label at top of bar with total HV and solution count
                ax.text(bar2.get_x() + bar2.get_width()/2., total_hv + 0.02,
                       f'{total_hv:.3f}\n({total_sols} sols.)',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                # Label on bottom part (exact phase) if it has meaningful height
                if phase1_hv > 0.05:  # Only show label if bar is tall enough
                    ax.text(bar1.get_x() + bar1.get_width()/2., phase1_hv / 2.,
                           f'{phase1_hv:.3f}\n({exact_cnt})',
                           ha='center', va='center', fontsize=11, fontweight='bold', 
                           color='white')
                elif phase1_hv == 0 and exact_cnt > 0:
                    # Show label just above baseline if phase1 HV is 0 but there are exact solutions
                    ax.text(bar1.get_x() + bar1.get_width()/2., 0.01,
                           f'0.000\n({exact_cnt})',
                           ha='center', va='bottom', fontsize=11, fontweight='bold', 
                           color='white')
                
                # Label on top part (PLS phase) if it has meaningful height
                if phase2_hv > 0.05:  # Only show label if bar is tall enough
                    ax.text(bar2.get_x() + bar2.get_width()/2., phase1_hv + phase2_hv / 2.,
                           f'{phase2_hv:.3f}\n({heur_cnt})',
                           ha='center', va='center', fontsize=11, fontweight='bold',
                           color='white')
            else:
                ax.text(bar1.get_x() + bar1.get_width()/2., 0.05,
                       'ZERO\n(0 sols.)',
                       ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')
        
        # Save plot
        plot_filename = f"{instance_name}_hypervolume.png"
        plot_path = plots_dir / plot_filename
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Store relative path
        plot_paths[instance_name] = f"plots/{plot_filename}"
    
    return plot_paths


def generate_markdown_report(
    output_path: Path,
    summary: Dict,
    results: Dict,
    zero_analyses: List[Dict],
    plot_paths: Optional[Dict[str, str]] = None,
    unique_counts: Optional[Dict[str, List[int]]] = None,
    correlation_plots: Optional[Dict[str, str]] = None,
    correlation_analyses: Optional[Dict[str, str]] = None
):
    """
    Generate a detailed Markdown report explaining zero hypervolumes.
    
    Args:
        output_path: Path to save the Markdown report
        summary: Summary dictionary with hypervolumes
        results: Full results dictionary with hypervolumes, ratios, and phase_counts
        zero_analyses: List of analysis records for zero hypervolumes
        plot_paths: Optional dictionary mapping instance names to plot image paths
        unique_counts: Optional dictionary mapping instance names to unique objective value counts
        correlation_plots: Optional dictionary mapping instance names to correlation plot paths
    """
    if plot_paths is None:
        plot_paths = {}
    
    with open(output_path, 'w') as f:
        f.write("# Hypervolume Analysis Report\n\n")
        f.write(f"Generated: {Path.cwd()}\n\n")
        
        # Summary statistics
        total_count = sum(len(summary[inst]["hypervolumes"]) for inst in summary)
        zero_count = len(zero_analyses)
        
        f.write("## Summary\n\n")
        f.write(f"- **Total hypervolume measurements**: {total_count}\n")
        f.write(f"- **Zero hypervolumes**: {zero_count} ({100*zero_count/total_count:.1f}%)\n")
        f.write(f"- **Non-zero hypervolumes**: {total_count - zero_count} ({100*(total_count-zero_count)/total_count:.1f}%)\n\n")
        
        # Zero hypervolume analysis
        f.write("## Zero Hypervolume Cases\n\n")
        f.write("This section explains why certain algorithm configurations resulted in zero hypervolume.\n\n")
        
        if not zero_analyses:
            f.write("No zero hypervolumes found.\n\n")
        else:
            # Group by instance
            by_instance = defaultdict(list)
            for analysis in zero_analyses:
                by_instance[analysis['instance']].append(analysis)
            
            for instance in sorted(by_instance.keys()):
                f.write(f"### {instance}\n\n")
                
                for analysis in sorted(by_instance[instance], key=lambda x: x['ratio']):
                    f.write(f"#### Ratio: {analysis['ratio']}\n\n")
                    f.write(f"- **Total solutions found**: {analysis['total_solutions']}\n")
                    f.write(f"- **Pareto front size**: {analysis['pareto_size']}\n")
                    f.write(f"- **Reason**: {analysis['reason']}\n")
                    
                    if 'solution_objectives' in analysis and 'global_bounds' in analysis:
                        f.write(f"\n**Solution objectives**: `{analysis['solution_objectives']}`\n\n")
                        f.write("**Global bounds**:\n")
                        for i, bounds in enumerate(analysis['global_bounds']):
                            obj_val = analysis['solution_objectives'][i]
                            f.write(f"- Objective {i}: [{bounds[0]}, {bounds[1]}] (solution value: {obj_val})\n")
                        f.write("\n")
                    
                    f.write("\n---\n\n")
        
        # Full results table
        f.write("## Full Results Table\n\n")
        f.write("| Instance | 100/0 | 50/50 | 20/80 | 0/100 |\n")
        f.write("|----------|-------|-------|-------|-------|\n")
        
        def instance_sort_key(instance_name: str) -> tuple:
            parts = instance_name.rsplit('_', 1)
            if len(parts) == 2:
                city_name = parts[0]
                try:
                    size = int(parts[1])
                    return (size, city_name)
                except ValueError:
                    pass
            return (0, instance_name)
        
        for instance_name in sorted(summary.keys(), key=instance_sort_key):
            hvs = summary[instance_name]["hypervolumes"]
            stderrs = summary[instance_name].get("hypervolumes_stderr", [])
            ratios = summary[instance_name].get("ratios", [])
            
            # Create a mapping from ratio to (hv, stderr)
            ratio_map = {}
            for i, ratio in enumerate(ratios):
                ratio_key = f"{ratio[0]}/{ratio[1]}"
                hv_val = hvs[i] if i < len(hvs) else 0.0
                stderr_val = stderrs[i] if i < len(stderrs) else 0.0
                ratio_map[ratio_key] = (hv_val, stderr_val)
            
            # Format each column, using N/A if ratio not present
            cols = []
            for ratio_key in ["100/0", "50/50", "20/80", "0/100"]:
                if ratio_key in ratio_map:
                    hv_val, stderr_val = ratio_map[ratio_key]
                    if stderr_val > 0:
                        cols.append(f"{hv_val:.6f} ± {stderr_val:.6f}")
                    else:
                        cols.append(f"{hv_val:.6f}")
                else:
                    cols.append("N/A")
            
            f.write(f"| {instance_name} | {cols[0]} | {cols[1]} | {cols[2]} | {cols[3]} |\n")
        
        f.write("\n")
        
        # Add solution counts table (Exact/Heuristic)
        f.write("## Solution Counts by Phase\n\n")
        f.write("Shows the number of solutions from each phase (Exact/Heuristic) for each algorithm configuration.\n\n")
        f.write("| Instance | 100/0 | 50/50 | 20/80 | 0/100 |\n")
        f.write("|----------|-------|-------|-------|-------|\n")
        
        for instance_name in sorted(results.keys(), key=instance_sort_key):
            if instance_name in results:
                phase_counts = results[instance_name].get("phase_counts", [])
                ratios = results[instance_name].get("ratios", [])
                
                # Create mapping from ratio to phase_count
                phase_map = {}
                for i, ratio in enumerate(ratios):
                    ratio_key = f"{ratio[0]}/{ratio[1]}"
                    if i < len(phase_counts):
                        exact_cnt, heur_cnt = phase_counts[i]
                        phase_map[ratio_key] = f"{exact_cnt}/{heur_cnt}"
                
                # Format each column
                cols = []
                for ratio_key in ["100/0", "50/50", "20/80", "0/100"]:
                    if ratio_key in phase_map:
                        cols.append(phase_map[ratio_key])
                    else:
                        cols.append("N/A")
                
                f.write(f"| {instance_name} | {cols[0]} | {cols[1]} | {cols[2]} | {cols[3]} |\n")
        
        f.write("\n")
        
        # Add unique objective values table if available
        if unique_counts:
            f.write("## Unique Objective Values per Instance\n\n")
            f.write("Shows the number of unique values for each of the four objectives across all solutions.\n\n")
            f.write("| Instance | Obj 1 (Cost) | Obj 2 (Cloud) | Obj 3 (Angle) | Obj 4 (Resolution) |\n")
            f.write("|----------|--------------|---------------|---------------|--------------------|\n")
            
            for instance_name in sorted(summary.keys(), key=instance_sort_key):
                if instance_name in unique_counts:
                    counts = unique_counts[instance_name]
                    if len(counts) >= 4:
                        f.write(f"| {instance_name} | {counts[0]} | {counts[1]} | {counts[2]} | {counts[3]} |\n")
                    else:
                        f.write(f"| {instance_name} | N/A | N/A | N/A | N/A |\n")
                else:
                    f.write(f"| {instance_name} | N/A | N/A | N/A | N/A |\n")
            
            f.write("\n")
        
        # Add plots section if available
        if plot_paths:
            f.write("## Hypervolume Visualizations\n\n")
            f.write("Bar plots showing hypervolume performance across different algorithm configurations for each instance.\n\n")
            
            for instance_name in sorted(summary.keys(), key=instance_sort_key):
                if instance_name in plot_paths:
                    f.write(f"### {instance_name}\n\n")
                    f.write(f"![{instance_name} Hypervolume]({plot_paths[instance_name]})\n\n")
        
        # Add correlation plots section if available
        if correlation_plots:
            f.write("\n## Objective Correlation Analysis\n\n")
            
            # Add methodology description
            f.write("### Methodology\n\n")
            f.write("This analysis examines the pairwise correlations between all four objectives across all solutions "
                   "generated by different algorithm configurations (100/0, 50/50, 20/80, 0/100 ratios) for each instance.\n\n")
            f.write("**Analysis approach:**\n\n")
            f.write("- **Data aggregation**: All solutions from all algorithm configurations and iterations are pooled together for each instance\n")
            f.write("- **Correlation metric**: Pearson correlation coefficient (ρ) is computed for each pair of objectives\n")
            f.write("- **Visualization**: 4×4 correlation matrices show:\n")
            f.write("    - **Diagonal**: Histograms showing the distribution of values for each objective\n")
            f.write("    - **Off-diagonal**: Scatter plots showing the relationship between objective pairs, with correlation coefficient displayed\n")
            f.write("- **Color coding**: Background colors indicate correlation strength:\n")
            f.write("    - **Red shades**: Positive correlation (stronger red = stronger positive correlation, ρ > 0.4)\n")
            f.write("    - **Blue shades**: Negative correlation (stronger blue = stronger negative correlation, ρ < -0.4)\n")
            f.write("    - **White**: Weak or no correlation (|ρ| ≤ 0.4)\n\n")
            f.write("**Interpretation:**\n\n")
            f.write("- **Strong positive correlation (ρ > 0.7)**: Objectives tend to improve or worsen together (conflicting objectives)\n")
            f.write("- **Strong negative correlation (ρ < -0.7)**: Improving one objective tends to worsen the other (trade-off relationship)\n")
            f.write("- **Weak correlation (|ρ| ≤ 0.4)**: Objectives are relatively independent\n\n")
            f.write("### Correlation Matrices\n\n")
            
            for instance_name in sorted(summary.keys(), key=instance_sort_key):
                if instance_name in correlation_plots:
                    f.write(f"### {instance_name}\n\n")
                    f.write(f"![{instance_name} Correlation Matrix]({correlation_plots[instance_name]})\n\n")
                    
                    # Add automated analysis if available
                    if correlation_analyses and instance_name in correlation_analyses:
                        f.write(f"{correlation_analyses[instance_name]}\n\n")
        
        # Interpretation guide
        f.write("## Interpretation Guide\n\n")
        f.write("### Why Zero Hypervolumes Occur\n\n")
        f.write("1. **Single Solution at Boundary**: When an algorithm finds only one solution that happens to be at the maximum boundary of the global objective bounds, it contributes zero hypervolume. This typically happens when:\n")
        f.write("   - The exact solver (100/0) times out on difficult instances\n")
        f.write("   - The algorithm configuration is unsuitable for the problem\n\n")
        f.write("2. **All Solutions Dominated**: In rare cases, all solutions found might be dominated by solutions from other configurations, resulting in an empty Pareto front.\n\n")
        f.write("3. **Solutions at Global Bounds**: When normalized using global bounds across all ratios, solutions at the extreme boundaries contribute zero volume.\n\n")
        f.write("### Global Bounds Methodology\n\n")
        f.write("Hypervolumes are computed using **global objective bounds** extracted from all algorithm runs for each instance. This ensures:\n")
        f.write("- Fair comparison across different ratio configurations\n")
        f.write("- Consistent normalization to [0,1] range\n")
        f.write("- Proper handling of cases where different ratios explore different regions of the objective space\n\n")
        f.write("A zero hypervolume is **legitimate** and indicates poor performance for that specific configuration on that instance.\n\n")


def generate_comparison_plots(results1: Dict, results2: Dict, output_dir: Path,
                              artifacts_dir1: Path, artifacts_dir2: Path,
                              bounds_map1: Dict, bounds_map2: Dict,
                              ref_map1: Dict, ref_map2: Dict,
                              label1: str, label2: str) -> Dict[str, str]:
    """
    Generate comparison bar plots showing stacked phase hypervolumes from two different runs side by side.
    
    Args:
        results1: Results from first directory
        results2: Results from second directory
        output_dir: Directory to save plots
        artifacts_dir1: First artifacts directory for finding result files
        artifacts_dir2: Second artifacts directory for finding result files
        bounds_map1: Global bounds for first dataset
        bounds_map2: Global bounds for second dataset
        ref_map1: Reference points for first dataset
        ref_map2: Reference points for second dataset
        label1: Label for first dataset
        label2: Label for second dataset
        
    Returns:
        Dictionary mapping instance names to plot paths
    """
    if not PLOTTING_AVAILABLE:
        return {}
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    plot_paths = {}
    ratio_labels = ['100/0\n(Pure Exact)', '50/50\n(Hybrid)', '20/80\n(Mostly Heur.)', '0/100\n(Pure Heur.)']
    all_ratios = [(100, 0), (50, 50), (20, 80), (0, 100)]
    
    # Sapphire and Ruby colors
    sapphire_color = '#0F52BA'  # Sapphire blue for exact solver
    ruby_color = '#E0115F'      # Ruby red for PLS solver
    
    # Get all common instances
    common_instances = set(results1.keys()) & set(results2.keys())
    
    for instance_name in sorted(common_instances):
        data1 = results1[instance_name]
        data2 = results2[instance_name]
        
        if instance_name not in bounds_map1 or instance_name not in bounds_map2:
            continue
        
        objective_bounds1 = bounds_map1[instance_name]
        reference_point1 = ref_map1[instance_name]
        objective_bounds2 = bounds_map2[instance_name]
        reference_point2 = ref_map2[instance_name]
        
        # Build ratio maps
        ratio_map1 = {tuple(r): i for i, r in enumerate(data1["ratios"])}
        ratio_map2 = {tuple(r): i for i, r in enumerate(data2["ratios"])}
        
        # Compute phase data for both datasets
        phase_data1 = []
        phase_data2 = []
        
        for ratio in all_ratios:
            # Process dataset 1
            if ratio not in ratio_map1:
                phase_data1.append(None)
            else:
                ratio_str = f"{ratio[0]}_{ratio[1]}"
                result_files1 = list(artifacts_dir1.rglob(f"{instance_name}/{ratio_str}/iter*/result.json"))
                
                if result_files1:
                    phase1_hvs, total_hvs, exact_counts, heur_counts = [], [], [], []
                    for result_file in result_files1:
                        phase1_hv, total_hv, exact_cnt, heur_cnt = compute_phase_hypervolumes(
                            result_file, objective_bounds1, reference_point1
                        )
                        if total_hv is not None:
                            phase1_hvs.append(phase1_hv if phase1_hv is not None else 0.0)
                            total_hvs.append(total_hv)
                            exact_counts.append(exact_cnt)
                            heur_counts.append(heur_cnt)
                    
                    if phase1_hvs and total_hvs:
                        phase_data1.append({
                            'phase1_hv': np.mean(phase1_hvs),
                            'phase1_stderr': np.std(phase1_hvs, ddof=1) / np.sqrt(len(phase1_hvs)) if len(phase1_hvs) > 1 else 0.0,
                            'total_hv': np.mean(total_hvs),
                            'total_stderr': np.std(total_hvs, ddof=1) / np.sqrt(len(total_hvs)) if len(total_hvs) > 1 else 0.0,
                            'exact_count': int(np.mean(exact_counts)),
                            'heuristic_count': int(np.mean(heur_counts))
                        })
                    else:
                        phase_data1.append(None)
                else:
                    phase_data1.append(None)
            
            # Process dataset 2
            if ratio not in ratio_map2:
                phase_data2.append(None)
            else:
                ratio_str = f"{ratio[0]}_{ratio[1]}"
                result_files2 = list(artifacts_dir2.rglob(f"{instance_name}/{ratio_str}/iter*/result.json"))
                
                if result_files2:
                    phase1_hvs, total_hvs, exact_counts, heur_counts = [], [], [], []
                    for result_file in result_files2:
                        phase1_hv, total_hv, exact_cnt, heur_cnt = compute_phase_hypervolumes(
                            result_file, objective_bounds2, reference_point2
                        )
                        if total_hv is not None:
                            phase1_hvs.append(phase1_hv if phase1_hv is not None else 0.0)
                            total_hvs.append(total_hv)
                            exact_counts.append(exact_cnt)
                            heur_counts.append(heur_cnt)
                    
                    if phase1_hvs and total_hvs:
                        phase_data2.append({
                            'phase1_hv': np.mean(phase1_hvs),
                            'phase1_stderr': np.std(phase1_hvs, ddof=1) / np.sqrt(len(phase1_hvs)) if len(phase1_hvs) > 1 else 0.0,
                            'total_hv': np.mean(total_hvs),
                            'total_stderr': np.std(total_hvs, ddof=1) / np.sqrt(len(total_hvs)) if len(total_hvs) > 1 else 0.0,
                            'exact_count': int(np.mean(exact_counts)),
                            'heuristic_count': int(np.mean(heur_counts))
                        })
                    else:
                        phase_data2.append(None)
                else:
                    phase_data2.append(None)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x_pos = np.arange(len(ratio_labels))
        bar_width = 0.35
        
        # Plot dataset 1 (left bars)
        for i, d in enumerate(phase_data1):
            if d is not None:
                x = x_pos[i] - bar_width/2
                phase1_hv = d['phase1_hv']
                phase2_hv = d['total_hv'] - d['phase1_hv']
                total_hv = d['total_hv']
                
                # Stacked bars
                bar1 = ax.bar(x, phase1_hv, bar_width, color=sapphire_color, 
                            alpha=0.9, edgecolor='black', linewidth=1.2)
                bar2 = ax.bar(x, phase2_hv, bar_width, bottom=phase1_hv, 
                            color=ruby_color, alpha=0.9, edgecolor='black', linewidth=1.2)
                
                # Error bar
                ax.errorbar(x, total_hv, yerr=d['total_stderr'], fmt='none',
                          ecolor='black', elinewidth=2, capsize=5, capthick=2, zorder=10)
                
                # Labels
                total_sols = d['exact_count'] + d['heuristic_count']
                ax.text(x, total_hv + 0.02, f'{total_hv:.3f}\n({total_sols})',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot dataset 2 (right bars)
        for i, d in enumerate(phase_data2):
            if d is not None:
                x = x_pos[i] + bar_width/2
                phase1_hv = d['phase1_hv']
                phase2_hv = d['total_hv'] - d['phase1_hv']
                total_hv = d['total_hv']
                
                # Stacked bars
                bar1 = ax.bar(x, phase1_hv, bar_width, color=sapphire_color,
                            alpha=0.9, edgecolor='black', linewidth=1.2)
                bar2 = ax.bar(x, phase2_hv, bar_width, bottom=phase1_hv,
                            color=ruby_color, alpha=0.9, edgecolor='black', linewidth=1.2)
                
                # Error bar
                ax.errorbar(x, total_hv, yerr=d['total_stderr'], fmt='none',
                          ecolor='black', elinewidth=2, capsize=5, capthick=2, zorder=10)
                
                # Labels
                total_sols = d['exact_count'] + d['heuristic_count']
                ax.text(x, total_hv + 0.02, f'{total_hv:.3f}\n({total_sols})',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Customize plot
        ax.set_xlabel('Algorithm Configuration (Exact/Heuristic Ratio)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Hypervolume', fontsize=14, fontweight='bold')
        ax.set_title(f'Hypervolume Comparison: {instance_name}\n{label1} (left) vs {label2} (right)', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ratio_labels, fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=sapphire_color, edgecolor='black', label='Exact Phase'),
            Patch(facecolor=ruby_color, edgecolor='black', label='PLS Phase'),
            Patch(facecolor='white', edgecolor='black', label=f'{label1} (left)'),
            Patch(facecolor='gray', edgecolor='black', alpha=0.3, label=f'{label2} (right)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        
        plot_path = plots_dir / f"{instance_name}_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        plot_paths[instance_name] = f"plots/{instance_name}_comparison.png"
    
    return plot_paths


def generate_comparison_plots_merged(results1: Dict, results2: Dict, output_dir: Path,
                                     artifacts_dirs1: List[Path], artifacts_dirs2: List[Path],
                                     bounds_map1: Dict, bounds_map2: Dict,
                                     ref_map1: Dict, ref_map2: Dict,
                                     label1: str, label2: str) -> Dict[str, str]:
    """
    Generate comparison plots for merged results from multiple directories.
    
    Args:
        results1: Merged results from first set of directories
        results2: Merged results from second set of directories
        output_dir: Directory to save plots
        artifacts_dirs1: List of directories to search for first dataset
        artifacts_dirs2: List of directories to search for second dataset
        bounds_map1: Global bounds for first dataset
        bounds_map2: Global bounds for second dataset
        ref_map1: Reference points for first dataset
        ref_map2: Reference points for second dataset
        label1: Label for first dataset
        label2: Label for second dataset
        
    Returns:
        Dictionary mapping instance names to plot paths
    """
    if not PLOTTING_AVAILABLE:
        return {}
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    plot_paths = {}
    ratio_labels = ['100/0\n(Pure Exact)', '50/50\n(Hybrid)', '20/80\n(Mostly Heur.)', '0/100\n(Pure Heur.)']
    all_ratios = [(100, 0), (50, 50), (20, 80), (0, 100)]
    
    # Sapphire and Ruby colors
    sapphire_color = '#0F52BA'
    ruby_color = '#E0115F'
    
    # Get all common instances
    common_instances = set(results1.keys()) & set(results2.keys())
    
    for instance_name in sorted(common_instances):
        data1 = results1[instance_name]
        data2 = results2[instance_name]
        
        if instance_name not in bounds_map1 or instance_name not in bounds_map2:
            continue
        
        objective_bounds1 = bounds_map1[instance_name]
        reference_point1 = ref_map1[instance_name]
        objective_bounds2 = bounds_map2[instance_name]
        reference_point2 = ref_map2[instance_name]
        
        # Build ratio maps
        ratio_map1 = {tuple(r): i for i, r in enumerate(data1["ratios"])}
        ratio_map2 = {tuple(r): i for i, r in enumerate(data2["ratios"])}
        
        # Compute phase data for both datasets by searching across all directories
        phase_data1 = []
        phase_data2 = []
        
        for ratio in all_ratios:
            # Process dataset 1 - search across all dirs
            if ratio not in ratio_map1:
                phase_data1.append(None)
            else:
                ratio_str = f"{ratio[0]}_{ratio[1]}"
                result_files1 = []
                for artifacts_dir in artifacts_dirs1:
                    result_files1.extend(list(artifacts_dir.rglob(f"{instance_name}/{ratio_str}/iter*/result.json")))
                
                if result_files1:
                    phase1_hvs, total_hvs, exact_counts, heur_counts = [], [], [], []
                    for result_file in result_files1:
                        phase1_hv, total_hv, exact_cnt, heur_cnt = compute_phase_hypervolumes(
                            result_file, objective_bounds1, reference_point1
                        )
                        if total_hv is not None:
                            phase1_hvs.append(phase1_hv if phase1_hv is not None else 0.0)
                            total_hvs.append(total_hv)
                            exact_counts.append(exact_cnt)
                            heur_counts.append(heur_cnt)
                    
                    if phase1_hvs and total_hvs:
                        phase_data1.append({
                            'phase1_hv': np.mean(phase1_hvs),
                            'phase1_stderr': np.std(phase1_hvs, ddof=1) / np.sqrt(len(phase1_hvs)) if len(phase1_hvs) > 1 else 0.0,
                            'total_hv': np.mean(total_hvs),
                            'total_stderr': np.std(total_hvs, ddof=1) / np.sqrt(len(total_hvs)) if len(total_hvs) > 1 else 0.0,
                            'exact_count': int(np.mean(exact_counts)),
                            'heuristic_count': int(np.mean(heur_counts))
                        })
                    else:
                        phase_data1.append(None)
                else:
                    phase_data1.append(None)
            
            # Process dataset 2 - search across all dirs
            if ratio not in ratio_map2:
                phase_data2.append(None)
            else:
                ratio_str = f"{ratio[0]}_{ratio[1]}"
                result_files2 = []
                for artifacts_dir in artifacts_dirs2:
                    result_files2.extend(list(artifacts_dir.rglob(f"{instance_name}/{ratio_str}/iter*/result.json")))
                
                if result_files2:
                    phase1_hvs, total_hvs, exact_counts, heur_counts = [], [], [], []
                    for result_file in result_files2:
                        phase1_hv, total_hv, exact_cnt, heur_cnt = compute_phase_hypervolumes(
                            result_file, objective_bounds2, reference_point2
                        )
                        if total_hv is not None:
                            phase1_hvs.append(phase1_hv if phase1_hv is not None else 0.0)
                            total_hvs.append(total_hv)
                            exact_counts.append(exact_cnt)
                            heur_counts.append(heur_cnt)
                    
                    if phase1_hvs and total_hvs:
                        phase_data2.append({
                            'phase1_hv': np.mean(phase1_hvs),
                            'phase1_stderr': np.std(phase1_hvs, ddof=1) / np.sqrt(len(phase1_hvs)) if len(phase1_hvs) > 1 else 0.0,
                            'total_hv': np.mean(total_hvs),
                            'total_stderr': np.std(total_hvs, ddof=1) / np.sqrt(len(total_hvs)) if len(total_hvs) > 1 else 0.0,
                            'exact_count': int(np.mean(exact_counts)),
                            'heuristic_count': int(np.mean(heur_counts))
                        })
                    else:
                        phase_data2.append(None)
                else:
                    phase_data2.append(None)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x_pos = np.arange(len(ratio_labels))
        bar_width = 0.35
        
        # Plot dataset 1 (left bars)
        for i, d in enumerate(phase_data1):
            if d is not None:
                x = x_pos[i] - bar_width/2
                phase1_hv = d['phase1_hv']
                phase2_hv = d['total_hv'] - d['phase1_hv']
                total_hv = d['total_hv']
                
                # Stacked bars
                ax.bar(x, phase1_hv, bar_width, color=sapphire_color,
                      alpha=0.9, edgecolor='black', linewidth=1.2)
                ax.bar(x, phase2_hv, bar_width, bottom=phase1_hv,
                      color=ruby_color, alpha=0.9, edgecolor='black', linewidth=1.2)
                
                # Error bar
                ax.errorbar(x, total_hv, yerr=d['total_stderr'], fmt='none',
                          ecolor='black', elinewidth=2, capsize=5, capthick=2, zorder=10)
                
                # Labels
                total_sols = d['exact_count'] + d['heuristic_count']
                ax.text(x, total_hv + 0.02, f'{total_hv:.3f}\n({total_sols})',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot dataset 2 (right bars)
        for i, d in enumerate(phase_data2):
            if d is not None:
                x = x_pos[i] + bar_width/2
                phase1_hv = d['phase1_hv']
                phase2_hv = d['total_hv'] - d['phase1_hv']
                total_hv = d['total_hv']
                
                # Stacked bars
                ax.bar(x, phase1_hv, bar_width, color=sapphire_color,
                      alpha=0.9, edgecolor='black', linewidth=1.2)
                ax.bar(x, phase2_hv, bar_width, bottom=phase1_hv,
                      color=ruby_color, alpha=0.9, edgecolor='black', linewidth=1.2)
                
                # Error bar
                ax.errorbar(x, total_hv, yerr=d['total_stderr'], fmt='none',
                          ecolor='black', elinewidth=2, capsize=5, capthick=2, zorder=10)
                
                # Labels
                total_sols = d['exact_count'] + d['heuristic_count']
                ax.text(x, total_hv + 0.02, f'{total_hv:.3f}\n({total_sols})',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Customize plot
        ax.set_xlabel('Algorithm Configuration (Exact/Heuristic Ratio)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Hypervolume', fontsize=14, fontweight='bold')
        ax.set_title(f'Hypervolume Comparison: {instance_name}\n{label1} (left) vs {label2} (right)',
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ratio_labels, fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=sapphire_color, edgecolor='black', label='Exact Phase'),
            Patch(facecolor=ruby_color, edgecolor='black', label='PLS Phase'),
            Patch(facecolor='white', edgecolor='black', label=f'{label1} (left)'),
            Patch(facecolor='gray', edgecolor='black', alpha=0.3, label=f'{label2} (right)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        
        plot_path = plots_dir / f"{instance_name}_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        plot_paths[instance_name] = f"plots/{instance_name}_comparison.png"
    
    return plot_paths


def compute_unique_objective_values(artifacts_dirs: List[Path], unified_bounds_map: Dict) -> Dict[str, List[int]]:
    """
    Compute the number of unique values for each objective across all instances.
    
    Args:
        artifacts_dirs: List of directories to search for result files
        unified_bounds_map: Unified bounds map containing instance names
        
    Returns:
        Dictionary mapping instance names to list of unique value counts per objective
    """
    instance_unique_counts = {}
    
    for instance_name in unified_bounds_map.keys():
        # Collect all objective values across all result files for this instance
        objective_values_by_idx = defaultdict(set)
        
        for artifacts_dir in artifacts_dirs:
            result_files = list(artifacts_dir.rglob(f"{instance_name}/*/iter*/result.json"))
            
            for result_file in result_files:
                try:
                    data = load_result_cached(result_file)
                    
                    objectives_list = data.get('objectives', [])
                    solutions = data.get('solutions', [])
                    
                    if not objectives_list or not solutions:
                        continue
                    
                    # Extract objective values from all solutions
                    for sol in solutions:
                        obj_values = extract_objectives_from_solution(sol, objectives_list)
                        for obj_idx, val in enumerate(obj_values):
                            objective_values_by_idx[obj_idx].add(val)
                            
                except Exception as e:
                    print(f"Warning: Could not process {result_file}: {e}", file=sys.stderr)
                    continue
        
        # Count unique values for each objective
        unique_counts = []
        num_objectives = len(objective_values_by_idx)
        for obj_idx in range(num_objectives):
            unique_counts.append(len(objective_values_by_idx[obj_idx]))
        
        if unique_counts:
            instance_unique_counts[instance_name] = unique_counts
    
    return instance_unique_counts


def generate_correlation_plots(artifacts_dirs: List[Path], output_dir: Path, unified_bounds_map: Dict) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Generate correlation plots showing relationships between objectives.
    
    Args:
        artifacts_dirs: List of directories to search for result files
        output_dir: Directory to save plot images
        unified_bounds_map: Unified bounds map containing instance names
        
    Returns:
        Tuple of (plot_paths_dict, analysis_dict) where:
        - plot_paths_dict: Dictionary mapping instance names to plot paths
        - analysis_dict: Dictionary mapping instance names to correlation analysis text
    """
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib not available, skipping correlation plots", file=sys.stderr)
        return {}
    
    # Create plots subdirectory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    correlation_plot_paths = {}
    correlation_analyses = {}
    
    # Objective names for labeling
    obj_names = ['Cost', 'Cloud Coverage', 'Angle', 'Resolution']
    
    for instance_name in unified_bounds_map.keys():
        # Collect all objective values across all result files for this instance
        all_objectives = []
        
        for artifacts_dir in artifacts_dirs:
            result_files = list(artifacts_dir.rglob(f"{instance_name}/*/iter*/result.json"))
            
            for result_file in result_files:
                try:
                    data = load_result_cached(result_file)
                    
                    objectives_list = data.get('objectives', [])
                    solutions = data.get('solutions', [])
                    
                    if not objectives_list or not solutions:
                        continue
                    
                    # Extract objective values from all solutions
                    for sol in solutions:
                        obj_values = extract_objectives_from_solution(sol, objectives_list)
                        all_objectives.append(obj_values)
                        
                except Exception as e:
                    print(f"Warning: Could not process {result_file}: {e}", file=sys.stderr)
                    continue
        
        if len(all_objectives) < 10:  # Skip if too few data points
            continue
        
        # Convert to numpy array for easier manipulation
        obj_array = np.array(all_objectives)
        
        # Compute full correlation matrix for analysis
        corr_matrix = np.corrcoef(obj_array.T)
        
        # Generate analysis text
        analysis_lines = []
        analysis_lines.append(f"**Key findings for {instance_name}:**\n")
        
        # Identify strong correlations
        strong_positive = []
        strong_negative = []
        moderate_positive = []
        moderate_negative = []
        
        for i in range(4):
            for j in range(i+1, 4):
                corr = corr_matrix[i, j]
                pair = f"{obj_names[i]} vs {obj_names[j]}"
                
                if corr > 0.7:
                    strong_positive.append((pair, corr))
                elif corr < -0.7:
                    strong_negative.append((pair, corr))
                elif corr > 0.4:
                    moderate_positive.append((pair, corr))
                elif corr < -0.4:
                    moderate_negative.append((pair, corr))
        
        # Write findings with nested lists
        if strong_positive:
            analysis_lines.append("- **Strong positive correlations** (conflicting objectives):")
            for pair, corr in sorted(strong_positive, key=lambda x: -x[1]):
                analysis_lines.append(f"    - {pair}: ρ = {corr:.3f} — These objectives tend to worsen together")
        
        if strong_negative:
            analysis_lines.append("- **Strong negative correlations** (trade-offs):")
            for pair, corr in sorted(strong_negative, key=lambda x: x[1]):
                analysis_lines.append(f"    - {pair}: ρ = {corr:.3f} — Improving one objective degrades the other")
        
        if moderate_positive:
            analysis_lines.append("- **Moderate positive correlations:**")
            for pair, corr in sorted(moderate_positive, key=lambda x: -x[1]):
                analysis_lines.append(f"    - {pair}: ρ = {corr:.3f}")
        
        if moderate_negative:
            analysis_lines.append("- **Moderate negative correlations:**")
            for pair, corr in sorted(moderate_negative, key=lambda x: x[1]):
                analysis_lines.append(f"    - {pair}: ρ = {corr:.3f}")
        
        # Add interpretation
        if not strong_positive and not strong_negative:
            analysis_lines.append("- **Overall**: The objectives show relatively weak correlations, suggesting they are largely independent")
        elif len(strong_positive) > len(strong_negative):
            analysis_lines.append("- **Overall**: Objectives are predominantly conflicting, making this a challenging multi-objective optimization problem")
        elif len(strong_negative) > len(strong_positive):
            analysis_lines.append("- **Overall**: Strong trade-offs exist between objectives, requiring careful balance in solution selection")
        else:
            analysis_lines.append("- **Overall**: Mix of conflicting and trade-off relationships indicates complex objective space structure")
        
        correlation_analyses[instance_name] = "\n".join(analysis_lines)
        
        # Create 4x4 correlation plot with extra space for legend
        fig, axes = plt.subplots(4, 4, figsize=(16, 18))
        fig.suptitle(f'Objective Correlation Matrix: {instance_name}', fontsize=20, fontweight='bold', y=0.98)
        
        for i in range(4):
            for j in range(4):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histogram of the objective
                    ax.hist(obj_array[:, i], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
                    ax.set_ylabel('Frequency', fontsize=10)
                    ax.set_xlabel(obj_names[i], fontsize=11, fontweight='bold')
                    ax.set_title(f'{obj_names[i]} Distribution', fontsize=10, fontweight='bold', pad=5)
                else:
                    # Off-diagonal: scatter plot
                    ax.scatter(obj_array[:, j], obj_array[:, i], alpha=0.3, s=10, color='steelblue')
                    
                    # Compute and display correlation coefficient
                    correlation = np.corrcoef(obj_array[:, j], obj_array[:, i])[0, 1]
                    
                    # Color-code the background based on correlation strength
                    if abs(correlation) > 0.7:
                        bg_color = '#ffcccc' if correlation > 0 else '#ccccff'
                    elif abs(correlation) > 0.4:
                        bg_color = '#ffe6e6' if correlation > 0 else '#e6e6ff'
                    else:
                        bg_color = 'white'
                    ax.set_facecolor(bg_color)
                    
                    # Add correlation text
                    ax.text(0.05, 0.95, f'ρ = {correlation:.3f}',
                           transform=ax.transAxes, fontsize=10, fontweight='bold',
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    if i == 3:
                        ax.set_xlabel(obj_names[j], fontsize=11, fontweight='bold')
                    if j == 0:
                        ax.set_ylabel(obj_names[i], fontsize=11, fontweight='bold')
                
                # Formatting
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.tick_params(labelsize=9)
                
                # Hide labels for non-edge plots
                if i < 3:
                    ax.set_xticklabels([])
                if j > 0:
                    ax.set_yticklabels([])
        
        # Add legend explaining the objectives at the bottom with better spacing
        legend_text = (
            'Objectives: Obj 1 (Cost) = Total acquisition cost | '
            'Obj 2 (Cloud) = Cloud coverage area | '
            'Obj 3 (Angle) = Max incidence angle | '
            'Obj 4 (Resolution) = Sum of resolutions\n'
            'All objectives are minimized. Color coding: Red background = strong positive correlation (ρ > 0.7), '
            'Blue background = strong negative correlation (ρ < -0.7)'
        )
        fig.text(0.5, 0.01, legend_text, ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8),
                wrap=True)
        
        plt.tight_layout(rect=[0, 0.06, 1, 0.97])
        
        # Save plot
        plot_filename = f"{instance_name}_correlation.png"
        plot_path = plots_dir / plot_filename
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        correlation_plot_paths[instance_name] = f"plots/{plot_filename}"
    
    return correlation_plot_paths, correlation_analyses


def auto_detect_archives(base_dir: Path) -> Dict[str, List[Path]]:
    """
    Automatically detect and group archives by their suffix pattern.
    
    Args:
        base_dir: Base directory containing multiple archive subdirectories
        
    Returns:
        Dictionary mapping suffix patterns to lists of matching directories
    """
    archives_by_suffix = defaultdict(list)
    
    # Scan all subdirectories
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        # Extract suffix pattern (e.g., 'sequential', 'concurrent')
        name = subdir.name
        
        # Skip comparison directories
        if name.startswith('comparison_'):
            continue
        
        # Common patterns to look for
        if '_concurrent' in name:
            archives_by_suffix['concurrent'].append(subdir)
        elif '_sequential' in name:
            archives_by_suffix['sequential'].append(subdir)
    
    return dict(archives_by_suffix)


def run_comparison_mode(dir1: Path, dir2: Path, labels: Optional[List[str]] = None):
    """
    Run comparison mode to analyze and compare results from two directories.
    
    Args:
        dir1: First artifacts directory
        dir2: Second artifacts directory
        labels: Optional labels for the two directories
    """
    label1 = labels[0] if labels else dir1.name
    label2 = labels[1] if labels else dir2.name
    
    print(f"Comparison mode: {label1} vs {label2}", file=sys.stderr)
    print(f"Directory 1: {dir1}", file=sys.stderr)
    print(f"Directory 2: {dir2}", file=sys.stderr)
    
    # First, collect bounds from both directories to compute unified global bounds
    print(f"\nCollecting bounds from both directories...", file=sys.stderr)
    unified_bounds, unified_ref = compute_unified_bounds([dir1, dir2])
    
    # Parse both directories with unified bounds
    print(f"\nParsing {label1} with unified bounds...", file=sys.stderr)
    results1 = recompute_with_unified_bounds(dir1, unified_bounds, unified_ref)
    
    print(f"Parsing {label2} with unified bounds...", file=sys.stderr)
    results2 = recompute_with_unified_bounds(dir2, unified_bounds, unified_ref)
    
    if not results1 or not results2:
        print("Error: No valid results found in one or both directories", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory for comparison
    output_dir = dir1.parent / f"comparison_{dir1.name}_vs_{dir2.name}"
    output_dir.mkdir(exist_ok=True)
    
    # Generate comparison plots
    print("\nGenerating comparison plots...", file=sys.stderr)
    plot_paths = generate_comparison_plots(results1, results2, output_dir, 
                                          dir1, dir2,
                                          unified_bounds, unified_bounds,
                                          unified_ref, unified_ref,
                                          label1, label2)
    print(f"Generated {len(plot_paths)} comparison plots", file=sys.stderr)
    
    # Compute unique objective values
    print("\nComputing unique objective values...", file=sys.stderr)
    unique_counts = compute_unique_objective_values([dir1, dir2], unified_bounds)
    
    # Generate correlation plots
    print("\nGenerating correlation plots...", file=sys.stderr)
    correlation_plots, correlation_analyses = generate_correlation_plots([dir1, dir2], output_dir, unified_bounds)
    print(f"Generated {len(correlation_plots)} correlation plots", file=sys.stderr)
    
    # Generate comparison report
    report_path = output_dir / "comparison_report.md"
    generate_comparison_report(report_path, results1, results2, label1, label2, plot_paths, unique_counts, correlation_plots, correlation_analyses)
    print(f"\nComparison report saved to: {report_path}", file=sys.stderr)
    
    # Export to HTML
    html_path = export_to_html(report_path)
    if html_path:
        print(f"HTML report exported to: {html_path}", file=sys.stderr)
    
    # Print summary table
    print("\n" + "="*120, file=sys.stderr)
    print(f"HYPERVOLUME COMPARISON: {label1} vs {label2}", file=sys.stderr)
    print("="*120, file=sys.stderr)
    
    common_instances = sorted(set(results1.keys()) & set(results2.keys()))
    
    for instance in common_instances:
        print(f"\n{instance}:", file=sys.stderr)
        print(f"  {'Ratio':<10} {label1:<15} {label2:<15} {'Difference':<15}", file=sys.stderr)
        print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15}", file=sys.stderr)
        
        data1 = results1[instance]
        data2 = results2[instance]
        
        ratio_map1 = {tuple(r): i for i, r in enumerate(data1["ratios"])}
        ratio_map2 = {tuple(r): i for i, r in enumerate(data2["ratios"])}
        
        for ratio in [(100, 0), (50, 50), (20, 80), (0, 100)]:
            ratio_str = f"{ratio[0]}/{ratio[1]}"
            hv1 = data1["hypervolumes"][ratio_map1[ratio]] if ratio in ratio_map1 else None
            hv2 = data2["hypervolumes"][ratio_map2[ratio]] if ratio in ratio_map2 else None
            
            if hv1 is not None and hv2 is not None:
                diff = hv2 - hv1
                diff_str = f"{diff:+.6f}" if diff != 0 else "0.000000"
                print(f"  {ratio_str:<10} {hv1:<15.6f} {hv2:<15.6f} {diff_str:<15}", file=sys.stderr)
            elif hv1 is not None:
                print(f"  {ratio_str:<10} {hv1:<15.6f} {'N/A':<15} {'N/A':<15}", file=sys.stderr)
            elif hv2 is not None:
                print(f"  {ratio_str:<10} {'N/A':<15} {hv2:<15.6f} {'N/A':<15}", file=sys.stderr)
    
    print("="*120, file=sys.stderr)


def generate_comparison_report(report_path: Path, results1: Dict, results2: Dict,
                               label1: str, label2: str, plot_paths: Dict[str, str],
                               unique_counts: Optional[Dict[str, List[int]]] = None,
                               correlation_plots: Optional[Dict[str, str]] = None,
                               correlation_analyses: Optional[Dict[str, str]] = None):
    """Generate markdown comparison report."""
    with open(report_path, 'w') as f:
        f.write(f"# Hypervolume Comparison Report\n\n")
        f.write(f"**{label1}** vs **{label2}**\n\n")
        
        common_instances = sorted(set(results1.keys()) & set(results2.keys()))
        
        f.write(f"## Summary\n\n")
        f.write(f"- **Common instances**: {len(common_instances)}\n")
        f.write(f"- **Total comparisons**: {sum(min(len(results1[i]['hypervolumes']), len(results2[i]['hypervolumes'])) for i in common_instances)}\n\n")
        
        # Add unique objective values table if provided
        if unique_counts:
            f.write("## Unique Objective Values per Instance\n\n")
            f.write("Shows the number of unique values for each of the four objectives across all solutions.\n\n")
            f.write("| Instance | Obj 1 (Cost) | Obj 2 (Cloud) | Obj 3 (Angle) | Obj 4 (Resolution) |\n")
            f.write("|----------|--------------|---------------|---------------|--------------------|\n")
            
            def instance_sort_key(instance_name: str) -> tuple:
                parts = instance_name.rsplit('_', 1)
                if len(parts) == 2:
                    try:
                        return (int(parts[1]), parts[0])
                    except ValueError:
                        pass
                return (0, instance_name)
            
            for instance in sorted(common_instances, key=instance_sort_key):
                if instance in unique_counts:
                    counts = unique_counts[instance]
                    if len(counts) >= 4:
                        f.write(f"| {instance} | {counts[0]} | {counts[1]} | {counts[2]} | {counts[3]} |\n")
                    else:
                        f.write(f"| {instance} | N/A | N/A | N/A | N/A |\n")
                else:
                    f.write(f"| {instance} | N/A | N/A | N/A | N/A |\n")
            
            f.write("\n")
        
        f.write("## Comparison Table\n\n")
        f.write(f"| Instance | Ratio | {label1} | {label2} | Difference | Winner |\n")
        f.write("|----------|-------|----------|----------|------------|--------|\n")
        
        def instance_sort_key(instance_name: str) -> tuple:
            parts = instance_name.rsplit('_', 1)
            if len(parts) == 2:
                try:
                    return (int(parts[1]), parts[0])
                except ValueError:
                    pass
            return (0, instance_name)
        
        for instance in sorted(common_instances, key=instance_sort_key):
            data1 = results1[instance]
            data2 = results2[instance]
            
            ratio_map1 = {tuple(r): i for i, r in enumerate(data1["ratios"])}
            ratio_map2 = {tuple(r): i for i, r in enumerate(data2["ratios"])}
            
            for ratio in [(100, 0), (50, 50), (20, 80), (0, 100)]:
                ratio_str = f"{ratio[0]}/{ratio[1]}"
                hv1 = data1["hypervolumes"][ratio_map1[ratio]] if ratio in ratio_map1 else None
                hv2 = data2["hypervolumes"][ratio_map2[ratio]] if ratio in ratio_map2 else None
                
                if hv1 is not None and hv2 is not None:
                    diff = hv2 - hv1
                    winner = label2 if diff > 0.001 else (label1 if diff < -0.001 else "Tie")
                    f.write(f"| {instance} | {ratio_str} | {hv1:.6f} | {hv2:.6f} | {diff:+.6f} | {winner} |\n")
                elif hv1 is not None:
                    f.write(f"| {instance} | {ratio_str} | {hv1:.6f} | N/A | N/A | {label1} |\n")
                elif hv2 is not None:
                    f.write(f"| {instance} | {ratio_str} | N/A | {hv2:.6f} | N/A | {label2} |\n")
        
        f.write("\n## Solution Counts by Phase\n\n")
        f.write(f"Shows the number of solutions from each phase (Exact/Heuristic) for each algorithm configuration.\n\n")
        f.write(f"### {label1}\n\n")
        f.write("| Instance | 100/0 | 50/50 | 20/80 | 0/100 |\n")
        f.write("|----------|-------|-------|-------|-------|\n")
        
        for instance in sorted(common_instances, key=instance_sort_key):
            if instance in results1:
                phase_counts = results1[instance].get("phase_counts", [])
                ratios = results1[instance].get("ratios", [])
                
                # Create mapping from ratio to phase_count
                phase_map = {}
                for i, ratio in enumerate(ratios):
                    ratio_key = f"{ratio[0]}/{ratio[1]}"
                    if i < len(phase_counts):
                        exact_cnt, heur_cnt = phase_counts[i]
                        phase_map[ratio_key] = f"{int(exact_cnt)}/{int(heur_cnt)}"
                
                # Format each column
                cols = []
                for ratio_key in ["100/0", "50/50", "20/80", "0/100"]:
                    if ratio_key in phase_map:
                        cols.append(phase_map[ratio_key])
                    else:
                        cols.append("N/A")
                
                f.write(f"| {instance} | {cols[0]} | {cols[1]} | {cols[2]} | {cols[3]} |\n")
        
        f.write(f"\n### {label2}\n\n")
        f.write("| Instance | 100/0 | 50/50 | 20/80 | 0/100 |\n")
        f.write("|----------|-------|-------|-------|-------|\n")
        
        for instance in sorted(common_instances, key=instance_sort_key):
            if instance in results2:
                phase_counts = results2[instance].get("phase_counts", [])
                ratios = results2[instance].get("ratios", [])
                
                # Create mapping from ratio to phase_count
                phase_map = {}
                for i, ratio in enumerate(ratios):
                    ratio_key = f"{ratio[0]}/{ratio[1]}"
                    if i < len(phase_counts):
                        exact_cnt, heur_cnt = phase_counts[i]
                        phase_map[ratio_key] = f"{int(exact_cnt)}/{int(heur_cnt)}"
                
                # Format each column
                cols = []
                for ratio_key in ["100/0", "50/50", "20/80", "0/100"]:
                    if ratio_key in phase_map:
                        cols.append(phase_map[ratio_key])
                    else:
                        cols.append("N/A")
                
                f.write(f"| {instance} | {cols[0]} | {cols[1]} | {cols[2]} | {cols[3]} |\n")
        
        f.write("\n## Visualizations\n\n")
        
        for instance in sorted(common_instances, key=instance_sort_key):
            if instance in plot_paths:
                f.write(f"### {instance}\n\n")
                f.write(f"![{instance} Comparison]({plot_paths[instance]})\n\n")
        
        # Add correlation plots section if available
        if correlation_plots:
            f.write("\n## Objective Correlation Analysis\n\n")
            
            # Add methodology description
            f.write("### Methodology\n\n")
            f.write("This analysis examines the pairwise correlations between all four objectives across all solutions "
                   "generated by different algorithm configurations (100/0, 50/50, 20/80, 0/100 ratios) for each instance.\n\n")
            f.write("**Analysis approach:**\n")
            f.write("- **Data aggregation**: All solutions from all algorithm configurations and iterations are pooled together for each instance\n")
            f.write("- **Correlation metric**: Pearson correlation coefficient (ρ) is computed for each pair of objectives\n")
            f.write("- **Visualization**: 4×4 correlation matrices show:\n")
            f.write("  - **Diagonal**: Histograms showing the distribution of values for each objective\n")
            f.write("  - **Off-diagonal**: Scatter plots showing the relationship between objective pairs, with correlation coefficient displayed\n")
            f.write("- **Color coding**: Background colors indicate correlation strength:\n")
            f.write("  - **Red shades**: Positive correlation (stronger red = stronger positive correlation, ρ > 0.4)\n")
            f.write("  - **Blue shades**: Negative correlation (stronger blue = stronger negative correlation, ρ < -0.4)\n")
            f.write("  - **White**: Weak or no correlation (|ρ| ≤ 0.4)\n\n")
            f.write("**Interpretation:**\n")
            f.write("- **Strong positive correlation (ρ > 0.7)**: Objectives tend to improve or worsen together (conflicting objectives)\n")
            f.write("- **Strong negative correlation (ρ < -0.7)**: Improving one objective tends to worsen the other (trade-off relationship)\n")
            f.write("- **Weak correlation (|ρ| ≤ 0.4)**: Objectives are relatively independent\n\n")
            f.write("### Correlation Matrices\n\n")
            
            for instance in sorted(common_instances, key=instance_sort_key):
                if instance in correlation_plots:
                    f.write(f"### {instance}\n\n")
                    f.write(f"![{instance} Correlation Matrix]({correlation_plots[instance]})\n\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Parse hypervolume data from test artifacts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single directory analysis
  python parse_hypervolumes_from_results.py test_artifacts/hybrid_4d_wednesday_20251022_202800
  
  # Auto-detect and compare all archives in directory (compares sequential vs concurrent automatically)
  python parse_hypervolumes_from_results.py test_artifacts/medium_instances_20260222_120000 --compare
  
  # Manual comparison mode (compare two specific subdirectories)
  python parse_hypervolumes_from_results.py test_artifacts/medium_instances --compare solve_nd-tree_sequential solve_nd-tree_concurrent
  python parse_hypervolumes_from_results.py test_artifacts/medium_instances --compare solve_nd-tree_sequential solve_nd-tree_concurrent --labels "Sequential" "Concurrent"
        '''
    )
    
    parser.add_argument('artifacts_dir', help='Base directory containing archives')
    parser.add_argument('--compare', nargs='*', metavar='ARCHIVE',
                       help='Enable comparison mode. If no archives specified, auto-detect sequential vs concurrent. Otherwise specify two archive names.')
    parser.add_argument('--labels', nargs=2, metavar=('LABEL1', 'LABEL2'),
                       help='Labels for the two directories in comparison mode')
    parser.add_argument('--output', help='Output JSON file (default: artifacts_dir/summary.json)')
    
    args = parser.parse_args()
    
    artifacts_dir = Path(args.artifacts_dir)
    
    if not artifacts_dir.exists():
        print(f"Error: Directory {artifacts_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not artifacts_dir.is_dir():
        print(f"Error: {artifacts_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    
    # Check if comparison mode
    if args.compare is not None:
        if len(args.compare) == 0:
            # Auto-detect mode
            print("Auto-detecting archives for comparison...", file=sys.stderr)
            archives_by_suffix = auto_detect_archives(artifacts_dir)
            
            if 'sequential' not in archives_by_suffix or 'concurrent' not in archives_by_suffix:
                print(f"Error: Could not find both sequential and concurrent archives in {artifacts_dir}", file=sys.stderr)
                print(f"Found archives: {list(archives_by_suffix.keys())}", file=sys.stderr)
                sys.exit(1)
            
            # Merge all sequential directories and all concurrent directories
            ndtree_dirs = archives_by_suffix['sequential']
            vector_dirs = archives_by_suffix['concurrent']
            
            print(f"Found {len(ndtree_dirs)} sequential archive(s): {[d.name for d in ndtree_dirs]}", file=sys.stderr)
            print(f"Found {len(vector_dirs)} concurrent archive(s): {[d.name for d in vector_dirs]}", file=sys.stderr)
            
            # Use custom labels or default
            labels = args.labels if args.labels else ['Sequential', 'Concurrent']
            
            # Merge all directories of each type for comprehensive comparison
            if len(ndtree_dirs) > 1 or len(vector_dirs) > 1:
                print(f"Merging {len(ndtree_dirs)} sequential and {len(vector_dirs)} concurrent archives for comprehensive comparison", file=sys.stderr)
                
                # Collect unified bounds from all directories
                all_dirs = ndtree_dirs + vector_dirs
                unified_bounds, unified_ref = compute_unified_bounds(all_dirs)
                
                # Merge results from all sequential directories
                print(f"\nMerging results from all Sequential archives...", file=sys.stderr)
                merged_results1 = {}
                for nd_dir in ndtree_dirs:
                    results = recompute_with_unified_bounds(nd_dir, unified_bounds, unified_ref)
                    for instance, data in results.items():
                        if instance not in merged_results1:
                            merged_results1[instance] = data
                        else:
                            # Merge ratio data
                            for ratio in data["ratios"]:
                                if ratio not in merged_results1[instance]["ratios"]:
                                    idx = data["ratios"].index(ratio)
                                    merged_results1[instance]["ratios"].append(ratio)
                                    merged_results1[instance]["hypervolumes"].append(data["hypervolumes"][idx])
                                    merged_results1[instance]["hypervolumes_stderr"].append(data["hypervolumes_stderr"][idx])
                                    merged_results1[instance]["phase_counts"].append(data["phase_counts"][idx])
                
                # Merge results from all concurrent directories
                print(f"Merging results from all Concurrent archives...", file=sys.stderr)
                merged_results2 = {}
                for vec_dir in vector_dirs:
                    results = recompute_with_unified_bounds(vec_dir, unified_bounds, unified_ref)
                    for instance, data in results.items():
                        if instance not in merged_results2:
                            merged_results2[instance] = data
                        else:
                            # Merge ratio data
                            for ratio in data["ratios"]:
                                if ratio not in merged_results2[instance]["ratios"]:
                                    idx = data["ratios"].index(ratio)
                                    merged_results2[instance]["ratios"].append(ratio)
                                    merged_results2[instance]["hypervolumes"].append(data["hypervolumes"][idx])
                                    merged_results2[instance]["hypervolumes_stderr"].append(data["hypervolumes_stderr"][idx])
                                    merged_results2[instance]["phase_counts"].append(data["phase_counts"][idx])
                
                if not merged_results1 or not merged_results2:
                    print("Error: No valid results found after merging", file=sys.stderr)
                    sys.exit(1)
                
                # Create output directory for merged comparison
                output_dir = artifacts_dir / f"comparison_merged_sequential_vs_concurrent"
                output_dir.mkdir(exist_ok=True)
                
                common_instances = sorted(set(merged_results1.keys()) & set(merged_results2.keys()))
                print(f"Found {len(common_instances)} common instances across all archives", file=sys.stderr)
                
                # Generate comparison plots by searching across all directories
                print("\nGenerating comparison plots for merged data...", file=sys.stderr)
                plot_paths = generate_comparison_plots_merged(
                    merged_results1, merged_results2, output_dir,
                    ndtree_dirs, vector_dirs,
                    unified_bounds, unified_bounds,
                    unified_ref, unified_ref,
                    labels[0], labels[1]
                )
                print(f"Generated {len(plot_paths)} comparison plots", file=sys.stderr)
                
                # Compute unique objective values
                print("\nComputing unique objective values...", file=sys.stderr)
                unique_counts = compute_unique_objective_values(ndtree_dirs + vector_dirs, unified_bounds)
                
                # Generate correlation plots
                print("\nGenerating correlation plots...", file=sys.stderr)
                correlation_plots = generate_correlation_plots(ndtree_dirs + vector_dirs, output_dir, unified_bounds)
                print(f"Generated {len(correlation_plots)} correlation plots", file=sys.stderr)
                
                # Generate report with plots
                report_path = output_dir / "comparison_report.md"
                generate_comparison_report(report_path, merged_results1, merged_results2, labels[0], labels[1], plot_paths, unique_counts, correlation_plots)
                print(f"\nComparison report saved to: {report_path}", file=sys.stderr)
                
                # Export to HTML
                html_path = export_to_html(report_path)
                if html_path:
                    print(f"HTML report exported to: {html_path}", file=sys.stderr)
                
                # Print summary
                print("\n" + "="*120, file=sys.stderr)
                print(f"MERGED HYPERVOLUME COMPARISON: {labels[0]} vs {labels[1]}", file=sys.stderr)
                print("="*120, file=sys.stderr)
                
                for instance in common_instances:
                    print(f"\n{instance}:", file=sys.stderr)
                    print(f"  {'Ratio':<10} {labels[0]:<15} {labels[1]:<15} {'Difference':<15}", file=sys.stderr)
                    print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15}", file=sys.stderr)
                    
                    data1 = merged_results1[instance]
                    data2 = merged_results2[instance]
                    
                    ratio_map1 = {tuple(r): i for i, r in enumerate(data1["ratios"])}
                    ratio_map2 = {tuple(r): i for i, r in enumerate(data2["ratios"])}
                    
                    for ratio in [(100, 0), (50, 50), (20, 80), (0, 100)]:
                        ratio_str = f"{ratio[0]}/{ratio[1]}"
                        hv1 = data1["hypervolumes"][ratio_map1[ratio]] if ratio in ratio_map1 else None
                        hv2 = data2["hypervolumes"][ratio_map2[ratio]] if ratio in ratio_map2 else None
                        
                        if hv1 is not None and hv2 is not None:
                            diff = hv2 - hv1
                            diff_str = f"{diff:+.6f}" if diff != 0 else "0.000000"
                            print(f"  {ratio_str:<10} {hv1:<15.6f} {hv2:<15.6f} {diff_str:<15}", file=sys.stderr)
                        elif hv1 is not None:
                            print(f"  {ratio_str:<10} {hv1:<15.6f} {'N/A':<15} {'N/A':<15}", file=sys.stderr)
                        elif hv2 is not None:
                            print(f"  {ratio_str:<10} {'N/A':<15} {hv2:<15.6f} {'N/A':<15}", file=sys.stderr)
                
                print("="*120, file=sys.stderr)
            else:
                # Single directory of each type
                run_comparison_mode(ndtree_dirs[0], vector_dirs[0], labels)
            clear_result_cache()
            return
            
        elif len(args.compare) == 2:
            # Manual mode with two specified archives
            sub1, sub2 = args.compare
            dir1 = artifacts_dir / sub1
            dir2 = artifacts_dir / sub2
            
            if not dir1.exists():
                print(f"Error: Directory {dir1} does not exist", file=sys.stderr)
                sys.exit(1)
            if not dir1.is_dir():
                print(f"Error: {dir1} is not a directory", file=sys.stderr)
                sys.exit(1)
                
            if not dir2.exists():
                print(f"Error: Directory {dir2} does not exist", file=sys.stderr)
                sys.exit(1)
            if not dir2.is_dir():
                print(f"Error: {dir2} is not a directory", file=sys.stderr)
                sys.exit(1)
            
            # Run comparison mode
            run_comparison_mode(dir1, dir2, args.labels)
            clear_result_cache()
            return
        else:
            print("Error: --compare requires either 0 arguments (auto-detect) or 2 arguments (manual)", file=sys.stderr)
            sys.exit(1)
    
    # Single directory mode
    output_file = Path(args.output) if args.output else artifacts_dir / "summary.json"
    
    if not artifacts_dir.exists():
        print(f"Error: Directory {artifacts_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not artifacts_dir.is_dir():
        print(f"Error: {artifacts_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    
    print(f"Parsing test artifacts from: {artifacts_dir}", file=sys.stderr)
    
    # Parse all result files
    results, global_bounds_map, global_ref_map = parse_test_artifacts(artifacts_dir)
    
    if not results:
        print("Error: No valid results found", file=sys.stderr)
        sys.exit(1)
    
    # Create summary output
    summary = create_summary(results)
    
    # Write output
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    
    print(f"\nSummary written to: {output_file}", file=sys.stderr)
    print(f"Processed {len(summary)} instances", file=sys.stderr)
    
    # Print results in table format
    print("\n" + "="*100, file=sys.stderr)
    print("HYPERVOLUME RESULTS", file=sys.stderr)
    print("="*100, file=sys.stderr)
    print(f"{'Instance':<25} {'100/0':<12} {'50/50':<12} {'20/80':<12} {'0/100':<12}", file=sys.stderr)
    print("-"*100, file=sys.stderr)
    
    # Sort instances by size first, then by name
    def instance_sort_key(instance_name: str) -> tuple:
        """Extract (size, city_name) for sorting."""
        parts = instance_name.rsplit('_', 1)
        if len(parts) == 2:
            city_name = parts[0]
            try:
                size = int(parts[1])
                return (size, city_name)
            except ValueError:
                pass
        return (0, instance_name)
    
    for instance_name in sorted(summary.keys(), key=instance_sort_key):
        hvs = summary[instance_name]["hypervolumes"]
        if len(hvs) == 4:
            # Reorder: original order is [0/100, 20/80, 50/50, 100/0]
            # Display order should be [100/0, 50/50, 20/80, 0/100]
            print(f"{instance_name:<25} {hvs[3]:<12.6f} {hvs[2]:<12.6f} {hvs[1]:<12.6f} {hvs[0]:<12.6f}", file=sys.stderr)
        else:
            print(f"{instance_name:<25} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}", file=sys.stderr)
    
    print("="*100, file=sys.stderr)
    
    # Print summary statistics
    zero_count = sum(1 for inst in summary for hv in summary[inst]["hypervolumes"] if hv == 0.0)
    total_count = sum(len(summary[inst]["hypervolumes"]) for inst in summary)
    print(f"\nTotal hypervolume values: {total_count}", file=sys.stderr)
    print(f"Zero hypervolumes: {zero_count} ({100*zero_count/total_count:.1f}%)", file=sys.stderr)
    print(f"Non-zero hypervolumes: {total_count - zero_count} ({100*(total_count-zero_count)/total_count:.1f}%)", file=sys.stderr)
    
    # Generate plots
    print("\nGenerating plots...", file=sys.stderr)
    plot_paths = generate_plots(results, output_file.parent, artifacts_dir,
                                global_bounds_map, global_ref_map)
    if plot_paths:
        print(f"Generated {len(plot_paths)} plots", file=sys.stderr)
    
    # Compute unique objective values
    print("\nComputing unique objective values...", file=sys.stderr)
    unique_counts = compute_unique_objective_values([artifacts_dir], global_bounds_map)
    
    # Generate correlation plots
    print("\nGenerating correlation plots...", file=sys.stderr)
    correlation_plots, correlation_analyses = generate_correlation_plots([artifacts_dir], output_file.parent, global_bounds_map)
    print(f"Generated {len(correlation_plots)} correlation plots", file=sys.stderr)
    
    # Analyze zero hypervolumes and generate report
    print("\nAnalyzing zero hypervolumes...", file=sys.stderr)
    zero_analyses = analyze_zero_hypervolumes(artifacts_dir, results)
    
    # Generate markdown report
    report_path = output_file.parent / f"{output_file.stem}_report.md"
    generate_markdown_report(report_path, summary, results, zero_analyses, plot_paths, unique_counts, correlation_plots, correlation_analyses)
    print(f"Detailed Markdown report saved to: {report_path}", file=sys.stderr)
    
    # Export to HTML
    html_path = export_to_html(report_path)
    if html_path:
        print(f"HTML report exported to: {html_path}", file=sys.stderr)
    
    # Free cached result data
    clear_result_cache()


if __name__ == "__main__":
    main()
