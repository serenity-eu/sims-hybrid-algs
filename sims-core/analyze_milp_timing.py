#!/usr/bin/env python3
"""
Analyze MILP solution timing from test artifacts.

Calculates the average time required to find the next solution across all instances
in a MILP test run.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class InstanceStats:
    """Statistics for a single instance."""
    instance_name: str
    test_type: str
    num_solutions: int
    total_time: float
    avg_time_per_solution: float
    time_to_first_solution: float
    avg_time_between_solutions: float
    solution_times: List[float]


def analyze_instance(result_file: Path) -> InstanceStats:
    """Analyze a single instance result file."""
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    instance_name = data['instance_name']
    test_type = data['test_type']
    solutions = data.get('solutions', [])
    total_time = data.get('execution_time', 0)
    
    if not solutions:
        return InstanceStats(
            instance_name=instance_name,
            test_type=test_type,
            num_solutions=0,
            total_time=total_time,
            avg_time_per_solution=0,
            time_to_first_solution=0,
            avg_time_between_solutions=0,
            solution_times=[]
        )
    
    # Extract timestamps
    solution_times = [sol['timestamp_s'] for sol in solutions]
    
    # Calculate time to first solution
    time_to_first = solution_times[0]
    
    # Calculate time between consecutive solutions
    time_deltas = []
    for i in range(1, len(solution_times)):
        delta = solution_times[i] - solution_times[i-1]
        time_deltas.append(delta)
    
    avg_time_between = sum(time_deltas) / len(time_deltas) if time_deltas else 0
    avg_time_per_solution = total_time / len(solutions) if solutions else 0
    
    return InstanceStats(
        instance_name=instance_name,
        test_type=test_type,
        num_solutions=len(solutions),
        total_time=total_time,
        avg_time_per_solution=avg_time_per_solution,
        time_to_first_solution=time_to_first,
        avg_time_between_solutions=avg_time_between,
        solution_times=solution_times
    )


def find_all_result_files(base_path: Path) -> List[Path]:
    """Find all result.json files in the directory tree."""
    return list(base_path.rglob('result.json'))


def main(artifacts_dir: str):
    """Main analysis function."""
    base_path = Path(artifacts_dir)
    
    if not base_path.exists():
        print(f"Error: Directory {artifacts_dir} does not exist")
        sys.exit(1)
    
    # Find all result files
    result_files = find_all_result_files(base_path)
    
    if not result_files:
        print(f"No result.json files found in {artifacts_dir}")
        sys.exit(1)
    
    print(f"Found {len(result_files)} instance results")
    print("=" * 80)
    print()
    
    # Analyze each instance
    all_stats = []
    for result_file in sorted(result_files):
        stats = analyze_instance(result_file)
        all_stats.append(stats)
    
    # Group by test type
    stats_by_type = {}
    for stats in all_stats:
        if stats.test_type not in stats_by_type:
            stats_by_type[stats.test_type] = []
        stats_by_type[stats.test_type].append(stats)
    
    # Print detailed statistics for each instance
    for test_type in sorted(stats_by_type.keys()):
        print(f"\n{'='*80}")
        print(f"Test Type: {test_type}")
        print(f"{'='*80}\n")
        
        type_stats = stats_by_type[test_type]
        
        for stats in sorted(type_stats, key=lambda x: x.instance_name):
            print(f"Instance: {stats.instance_name}")
            print(f"  Solutions found: {stats.num_solutions}")
            print(f"  Total execution time: {stats.total_time:.2f}s")
            
            if stats.num_solutions > 0:
                print(f"  Time to first solution: {stats.time_to_first_solution:.4f}s")
                print(f"  Avg time per solution (total/count): {stats.avg_time_per_solution:.2f}s")
                
                if stats.num_solutions > 1:
                    print(f"  Avg time between solutions: {stats.avg_time_between_solutions:.4f}s")
                    
                    # Show distribution of time deltas
                    deltas = []
                    for i in range(1, len(stats.solution_times)):
                        deltas.append(stats.solution_times[i] - stats.solution_times[i-1])
                    
                    if deltas:
                        print(f"    Min time between solutions: {min(deltas):.4f}s")
                        print(f"    Max time between solutions: {max(deltas):.4f}s")
                        print(f"    Median time between solutions: {sorted(deltas)[len(deltas)//2]:.4f}s")
            print()
    
    # Overall statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}\n")
    
    total_solutions = sum(s.num_solutions for s in all_stats)
    total_execution_time = sum(s.total_time for s in all_stats)
    instances_with_solutions = [s for s in all_stats if s.num_solutions > 0]
    
    print(f"Total instances analyzed: {len(all_stats)}")
    print(f"Instances with solutions: {len(instances_with_solutions)}")
    print(f"Total solutions found: {total_solutions}")
    print(f"Total execution time: {total_execution_time:.2f}s ({total_execution_time/3600:.2f} hours)")
    
    if instances_with_solutions:
        # Average time to first solution
        avg_first_time = sum(s.time_to_first_solution for s in instances_with_solutions) / len(instances_with_solutions)
        print(f"\nAverage time to first solution: {avg_first_time:.4f}s")
        
        # Average time between solutions (only for instances with 2+ solutions)
        instances_with_multiple = [s for s in all_stats if s.num_solutions > 1]
        if instances_with_multiple:
            avg_between_time = sum(s.avg_time_between_solutions for s in instances_with_multiple) / len(instances_with_multiple)
            print(f"Average time between solutions (across {len(instances_with_multiple)} instances): {avg_between_time:.4f}s")
            
            # Collect all time deltas across all instances
            all_deltas = []
            for stats in instances_with_multiple:
                for i in range(1, len(stats.solution_times)):
                    delta = stats.solution_times[i] - stats.solution_times[i-1]
                    all_deltas.append(delta)
            
            if all_deltas:
                print(f"\nGlobal statistics for time between consecutive solutions:")
                print(f"  Total transitions: {len(all_deltas)}")
                print(f"  Mean: {sum(all_deltas)/len(all_deltas):.4f}s")
                print(f"  Min: {min(all_deltas):.4f}s")
                print(f"  Max: {max(all_deltas):.4f}s")
                print(f"  Median: {sorted(all_deltas)[len(all_deltas)//2]:.4f}s")
                
                # Percentiles
                sorted_deltas = sorted(all_deltas)
                p25 = sorted_deltas[len(sorted_deltas)//4]
                p75 = sorted_deltas[3*len(sorted_deltas)//4]
                p90 = sorted_deltas[9*len(sorted_deltas)//10]
                p95 = sorted_deltas[95*len(sorted_deltas)//100]
                p99 = sorted_deltas[99*len(sorted_deltas)//100]
                
                print(f"  25th percentile: {p25:.4f}s")
                print(f"  75th percentile: {p75:.4f}s")
                print(f"  90th percentile: {p90:.4f}s")
                print(f"  95th percentile: {p95:.4f}s")
                print(f"  99th percentile: {p99:.4f}s")
        
        # Group by test type for summary
        print(f"\n{'='*80}")
        print("SUMMARY BY TEST TYPE")
        print(f"{'='*80}\n")
        
        for test_type in sorted(stats_by_type.keys()):
            type_stats = stats_by_type[test_type]
            type_with_solutions = [s for s in type_stats if s.num_solutions > 0]
            type_with_multiple = [s for s in type_stats if s.num_solutions > 1]
            
            print(f"{test_type}:")
            print(f"  Instances: {len(type_stats)}")
            print(f"  Total solutions: {sum(s.num_solutions for s in type_stats)}")
            
            if type_with_solutions:
                avg_first = sum(s.time_to_first_solution for s in type_with_solutions) / len(type_with_solutions)
                print(f"  Avg time to first solution: {avg_first:.4f}s")
                
                if type_with_multiple:
                    avg_between = sum(s.avg_time_between_solutions for s in type_with_multiple) / len(type_with_multiple)
                    print(f"  Avg time between solutions: {avg_between:.4f}s")
            print()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <artifacts_directory>")
        print(f"\nExample:")
        print(f"  {sys.argv[0]} /path/to/test_artifacts/milp_friday_20251024_212404")
        sys.exit(1)
    
    main(sys.argv[1])
