#!/usr/bin/env python3
"""Analyze MILP solution times from test artifacts.

Calculates the average time required to find the next solution.
"""

import json
import statistics
from pathlib import Path
from typing import Dict


def analyze_solution_times(artifacts_dir: Path) -> Dict[str, any]:
    """Analyze solution times from MILP test artifacts.
    
    Args:
        artifacts_dir: Path to the artifacts directory
        
    Returns:
        Dictionary with analysis results
    """
    all_times = []
    test_results = []
    
    # Walk through all result.json files
    for result_file in artifacts_dir.rglob("result.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            test_name = result_file.parent.parent.name
            instance_name = result_file.parent.name
            
            solutions = data.get("solutions", [])
            execution_time = data.get("execution_time", 0)
            
            if not solutions:
                continue
            
            # Extract timestamps
            timestamps = []
            for solution in solutions:
                ts = solution.get("timestamp_s")
                if ts is not None:
                    timestamps.append(ts)
            
            if len(timestamps) < 2:
                continue
            
            # Calculate time differences between consecutive solutions
            time_diffs = []
            for i in range(1, len(timestamps)):
                diff = timestamps[i] - timestamps[i-1]
                time_diffs.append(diff)
                all_times.append(diff)
            
            avg_time = sum(time_diffs) / len(time_diffs) if time_diffs else 0
            std_dev = statistics.stdev(time_diffs) if len(time_diffs) > 1 else 0
            
            test_results.append({
                "test_name": test_name,
                "instance_name": instance_name,
                "num_solutions": len(solutions),
                "last_solution_time": timestamps[-1] if timestamps else 0,
                "execution_time": execution_time,
                "avg_time_per_solution": avg_time,
                "std_dev": std_dev,
                "time_diffs": time_diffs
            })
            
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
            continue
    
    # Calculate overall statistics
    overall_avg = sum(all_times) / len(all_times) if all_times else 0
    overall_min = min(all_times) if all_times else 0
    overall_max = max(all_times) if all_times else 0
    
    return {
        "test_results": test_results,
        "overall_stats": {
            "average_time": overall_avg,
            "min_time": overall_min,
            "max_time": overall_max,
            "total_transitions": len(all_times)
        }
    }


def print_analysis(results: Dict[str, any]):
    """Print analysis results in a readable format."""
    print("=" * 80)
    print("MILP Solution Time Analysis")
    print("=" * 80)
    print()
    
    # Overall statistics
    stats = results["overall_stats"]
    print("Overall Statistics:")
    print(f"  Total solution transitions: {stats['total_transitions']}")
    print(f"  Average time to next solution: {stats['average_time']:.2f} seconds")
    print(f"  Min time to next solution: {stats['min_time']:.2f} seconds")
    print(f"  Max time to next solution: {stats['max_time']:.2f} seconds")
    print()
    
    # Per-test results in table format
    print("Per-Instance Results:")
    print("-" * 125)
    print(f"{'Instance':<30} {'Solutions':>10} {'Last Sol (s)':>13} {'Exec Time (s)':>14} {'Avg (s)':>10} {'StdDev (s)':>12} {'Est 20 Sol (s)':>15}")
    print("-" * 125)
    
    # Sort by size first, then by instance name
    def sort_key(x):
        instance = x["instance_name"]
        # Extract city name and size
        parts = instance.rsplit('_', 1)
        city_name = parts[0] if len(parts) > 1 else instance
        size = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return (size, city_name)
    
    test_results = sorted(results["test_results"], key=sort_key)
    
    # Group results by size (145 -> 150)
    size_groups = {}
    for result in test_results:
        instance = result["instance_name"]
        parts = instance.rsplit('_', 1)
        size = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        # Group 145 with 150
        if size == 145:
            size = 150
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append(result)
    
    for result in test_results:
        est_20_sols = result['avg_time_per_solution'] * 20
        print(f"{result['instance_name']:<30} "
              f"{result['num_solutions']:>10} "
              f"{result['last_solution_time']:>13.2f} "
              f"{result['execution_time']:>14.2f} "
              f"{result['avg_time_per_solution']:>10.2f} "
              f"{result['std_dev']:>12.2f} "
              f"{est_20_sols:>15.2f}")
    
    print("-" * 125)
    print()
    
    # Calculate and print average time for 20 solutions per size group
    print("=" * 80)
    print("Average Time for 20 Solutions by Instance Size")
    print("=" * 80)
    print()
    print(f"{'Size':<10} {'Num Instances':>15} {'Avg Time for 20 Sol (s)':>25}")
    print("-" * 80)
    
    for size in sorted(size_groups.keys()):
        instances = size_groups[size]
        avg_times = [result['avg_time_per_solution'] * 20 for result in instances]
        overall_avg = sum(avg_times) / len(avg_times)
        print(f"{size:<10} {len(instances):>15} {overall_avg:>25.2f}")
    
    print("-" * 80)
    print()
    print("=" * 80)


def main():
    artifacts_dir = Path("/home/vhlushchenko/sims-hybrid-algs/sims-core/test_artifacts/milp_friday_20251024_212404")
    
    if not artifacts_dir.exists():
        print(f"Error: Artifacts directory not found: {artifacts_dir}")
        return
    
    print(f"Analyzing artifacts in: {artifacts_dir}")
    print()
    
    results = analyze_solution_times(artifacts_dir)
    print_analysis(results)


if __name__ == "__main__":
    main()
