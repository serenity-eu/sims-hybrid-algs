#!/usr/bin/env python3
"""
Parse all results.json files from test artifacts and compute objective bounds per instance.

Output format:
{
    "objectives": ["min_cost", "cloud_coverage", "min_max_incidence_angle"],
    "bounds": {
        "lagos_nigeria_30": [[min_cost_min, min_cost_max], [cloud_min, cloud_max], [angle_min, angle_max]],
        "paris_50": [[...], [...], [...]],
        ...
    }
}
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def extract_instance_name_size(path: Path) -> str | None:
    """
    Extract instance name and size from path like:
    .../solve_two_phase_3d_large_50_50/lagos_nigeria_100/result.json
    Returns: "lagos_nigeria_100"
    """
    # The instance name should be the parent directory of result.json
    if path.name == "result.json" and path.parent.name:
        return path.parent.name
    return None


def parse_result_json(result_path: Path) -> Tuple[List[str], List[List[float]]] | None:
    """
    Parse a single result.json file and extract objectives and their values.
    
    Returns:
        Tuple of (objectives_list, solutions_objectives) where solutions_objectives
        is a list of objective vectors, one per solution.
        Returns None if parsing fails.
    """
    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        
        # Extract objectives list
        objectives = data.get('objectives', [])
        if not objectives:
            print(f"Warning: No objectives found in {result_path}", file=sys.stderr)
            return None
        
        # Collect all objective values from solutions
        all_objectives = []
        
        # Map objective names to solution fields
        field_map = {
            'min_cost': 'cost',
            'cloud_coverage': 'cloudy_area',
            'min_max_incidence_angle': 'max_incidence_angle',
            'min_resolution': 'min_resolutions_sum'
        }
        
        # Get solutions from top-level 'solutions' field
        solutions = data.get('solutions', [])
        if solutions:
            for solution in solutions:
                obj_values = []
                for obj_name in objectives:
                    field = field_map.get(obj_name, obj_name)
                    value = solution.get(field)
                    if value is not None and value != -1:  # Skip -1 values (not computed)
                        obj_values.append(int(value))
                
                if len(obj_values) == len(objectives):
                    all_objectives.append(obj_values)
        
        if not all_objectives:
            print(f"Warning: No solutions found in {result_path}", file=sys.stderr)
            return None
        
        return (objectives, all_objectives)
    
    except Exception as e:
        print(f"Error parsing {result_path}: {e}", file=sys.stderr)
        return None


def compute_bounds(solutions_objectives: List[List[float]]) -> List[Tuple[float, float]]:
    """
    Compute min/max bounds for each objective across all solutions.
    
    Args:
        solutions_objectives: List of objective vectors
    
    Returns:
        List of (min, max) tuples, one per objective
    """
    if not solutions_objectives:
        return []
    
    num_objectives = len(solutions_objectives[0])
    bounds = []
    
    for obj_idx in range(num_objectives):
        values = [sol[obj_idx] for sol in solutions_objectives]
        bounds.append((min(values), max(values)))
    
    return bounds


def main():
    # Path to test artifacts directory
    artifacts_dir = Path("/home/vhlushchenko/sims-hybrid-algs/sims-core/test_artifacts/long_hybrid_friday_20251003_085019")
    
    if not artifacts_dir.exists():
        print(f"Error: Directory not found: {artifacts_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Find all result.json files
    result_files = list(artifacts_dir.rglob("result.json"))
    print(f"Found {len(result_files)} result.json files", file=sys.stderr)
    
    if not result_files:
        print("No result.json files found!", file=sys.stderr)
        sys.exit(1)
    
    # Aggregate data per instance
    instance_data: Dict[str, List[List[float]]] = defaultdict(list)
    objectives_list = None
    
    for result_path in result_files:
        instance_name = extract_instance_name_size(result_path)
        if not instance_name:
            print(f"Warning: Could not extract instance name from {result_path}", file=sys.stderr)
            continue
        
        parsed = parse_result_json(result_path)
        if not parsed:
            continue
        
        objectives, solutions_objectives = parsed
        
        # Store objectives list (should be same for all files)
        if objectives_list is None:
            objectives_list = objectives
        elif objectives_list != objectives:
            print(f"Warning: Objectives mismatch in {result_path}: {objectives} vs {objectives_list}", file=sys.stderr)
        
        # Accumulate solutions for this instance
        instance_data[instance_name].extend(solutions_objectives)
        print(f"Processed {result_path}: {len(solutions_objectives)} solutions for {instance_name}", file=sys.stderr)
    
    if not instance_data:
        print("Error: No valid data parsed!", file=sys.stderr)
        sys.exit(1)
    
    if objectives_list is None:
        print("Error: No objectives found!", file=sys.stderr)
        sys.exit(1)
    
    # Compute bounds for each instance
    bounds_dict = {}
    for instance_name, solutions_objectives in sorted(instance_data.items()):
        bounds = compute_bounds(solutions_objectives)
        # Convert to list of lists for JSON serialization
        bounds_dict[instance_name] = [[min_val, max_val] for min_val, max_val in bounds]
        print(f"Instance {instance_name}: {len(solutions_objectives)} total solutions, bounds: {bounds_dict[instance_name]}", file=sys.stderr)
    
    # Create output structure
    output = {
        "objectives": objectives_list,
        "bounds": bounds_dict
    }
    
    # Write to output file
    output_path = artifacts_dir / "objective_bounds.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nOutput written to: {output_path}", file=sys.stderr)
    print(f"Processed {len(bounds_dict)} instances", file=sys.stderr)
    
    # Also print to stdout for convenience
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
