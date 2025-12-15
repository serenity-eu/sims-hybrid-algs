#!/usr/bin/env python3
"""
Parse MILP solver results and generate summary with hypervolume analysis.

This script processes result.json files from MILP solver test runs,
validates the solutions to extract the Pareto front, computes hypervolume
from the solution objectives, and generates plots.

Expected directory structure:
    test_artifacts/
        solve_milp_4d_small/
            lagos_nigeria_30/
                result.json
            mexico_city_30/
                result.json
        solve_milp_4d_medium/
            lagos_nigeria_100/
                result.json
        solve_milp_4d_large/
            lagos_nigeria_145/
                result.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available, plots will not be generated", file=sys.stderr)

# Import sims_problem for hypervolume computation
try:
    from sims_problem import compute_hypervolume
except ImportError:
    print("Error: sims_problem module not found. Make sure it's installed.", file=sys.stderr)
    sys.exit(1)


def parse_test_artifacts(artifacts_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """
    Parse test artifacts directory and collect results.
    
    Returns:
        Tuple of (results_dict, global_bounds_map, global_ref_map)
        - results_dict: {instance_name: {test_type: result_data}}
        - global_bounds_map: {instance_name: [[min, max], ...]} for each objective
        - global_ref_map: {instance_name: [ref1, ref2, ...]} reference points
    """
    results = defaultdict(dict)
    
    # Find all result.json files
    result_files = list(artifacts_dir.rglob("result.json"))
    
    if not result_files:
        print(f"No result.json files found in {artifacts_dir}")
        return {}, {}, {}
    
    print(f"Found {len(result_files)} result files to process")
    
    # Parse each result file
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            instance_name = data.get('instance_name')
            test_type = data.get('test_type', 'unknown')
            
            if not instance_name:
                print(f"Warning: No instance_name in {result_file}")
                continue
            
            # Store the result
            results[instance_name][test_type] = data
            
        except Exception as e:
            print(f"Error parsing {result_file}: {e}")
            continue
    
    # Compute global bounds across all results for each instance
    global_bounds_map = {}
    global_ref_map = {}
    
    print("\nCollecting global bounds for each instance...")
    for instance_name, test_results in results.items():
        # Collect all solutions from all test types for this instance
        all_solutions = []
        for test_type, data in test_results.items():
            solutions = data.get('solutions', [])
            all_solutions.extend(solutions)
        
        if not all_solutions:
            continue
        
        # Extract objectives from solution fields (cost, cloudy_area, min_resolutions_sum, max_incidence_angle)
        # These correspond to: min_cost, cloud_coverage, min_resolution, min_max_incidence_angle
        objective_fields = ['cost', 'cloudy_area', 'min_resolutions_sum', 'max_incidence_angle']
        num_objectives = len(objective_fields)
        
        # Initialize bounds
        bounds = [[float('inf'), float('-inf')] for _ in range(num_objectives)]
        
        # Update bounds
        for sol in all_solutions:
            for i, field in enumerate(objective_fields):
                if field in sol:
                    val = sol[field]
                    bounds[i][0] = min(bounds[i][0], val)
                    bounds[i][1] = max(bounds[i][1], val)
        
        # Skip if no valid bounds
        if any(bounds[i][0] == float('inf') for i in range(num_objectives)):
            continue
        
        # Compute reference point (max + 1 for each objective, assuming minimization)
        ref_point = [bounds[i][1] + 1 for i in range(num_objectives)]
        
        global_bounds_map[instance_name] = bounds
        global_ref_map[instance_name] = ref_point
        
        print(f"  {instance_name}: bounds={bounds}, ref={ref_point}")
    
    return results, global_bounds_map, global_ref_map


def compute_hypervolume_for_result(result_data: Dict, ref_point: list, bounds: list) -> float:
    """
    Compute the hypervolume for a single result, given a reference point and bounds.
    
    Args:
        result_data: Result dictionary with 'solutions' key
        ref_point: Reference point for hypervolume computation
        bounds: Bounds for normalization [[min1, max1], [min2, max2], ...]
        
    Returns:
        The computed hypervolume (0 if no valid solutions)
    """
    solutions = result_data.get('solutions', [])
    if not solutions:
        return 0.0
    
    # Extract objectives from solution fields
    # MILP solutions have: cost, cloudy_area, min_resolutions_sum, max_incidence_angle
    objective_fields = ['cost', 'cloudy_area', 'min_resolutions_sum', 'max_incidence_angle']
    objectives_list = []
    
    for sol in solutions:
        # Build objectives array from individual fields
        objs = []
        valid = True
        for field in objective_fields:
            if field in sol:
                objs.append(sol[field])
            else:
                valid = False
                break
        
        if valid and objs:
            objectives_list.append(objs)
    
    if not objectives_list:
        return 0.0
    
    # Compute hypervolume using sims_problem.compute_hypervolume
    try:
        hv = compute_hypervolume(objectives_list, bounds, ref_point, normalized=True)
        return hv
    except Exception as e:
        print(f"Warning: Failed to compute hypervolume: {e}")
        return 0.0


def generate_summary(results: Dict, global_bounds_map: Dict, global_ref_map: Dict, 
                    output_dir: Path) -> Dict:
    """Generate summary with hypervolume computation."""
    summary = {}
    
    print("\nComputing hypervolumes with global bounds...")
    
    for instance_name in sorted(results.keys()):
        test_results = results[instance_name]
        ref_point = global_ref_map.get(instance_name)
        bounds = global_bounds_map.get(instance_name)
        
        if not ref_point or not bounds:
            continue
        
        hypervolumes = {}
        solution_counts = {}
        
        for test_type, data in test_results.items():
            solutions = data.get('solutions', [])
            hv = compute_hypervolume_for_result(data, ref_point, bounds)
            hypervolumes[test_type] = hv
            solution_counts[test_type] = len(solutions)
            
            print(f"  {instance_name} [{test_type}]: HV = {hv:.6f}, Solutions = {len(solutions)}")
        
        summary[instance_name] = {
            'hypervolumes': hypervolumes,
            'solution_counts': solution_counts,
            'reference_point': ref_point,
            'bounds': global_bounds_map.get(instance_name)
        }
    
    # Write summary to JSON
    summary_file = output_dir / "milp_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary written to: {summary_file}")
    print(f"Processed {len(summary)} instances")
    
    return summary


def generate_plots(summary: Dict, output_dir: Path):
    """Generate plots for MILP results."""
    if not PLOTTING_AVAILABLE:
        print("Plotting not available, skipping plot generation")
        return
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print("\nGenerating plots...")
    
    # Custom sort: first by city name, then by size
    def sort_key(instance_name):
        # Extract city name and size
        parts = instance_name.rsplit('_', 1)
        if len(parts) == 2:
            city_name = parts[0]
            try:
                size = int(parts[1])
            except ValueError:
                size = 0
            return (city_name, size)
        return (instance_name, 0)
    
    # Group instances by size category
    small_instances = []
    medium_instances = []
    large_instances = []
    
    for instance_name in sorted(summary.keys(), key=sort_key):
        if any(x in instance_name for x in ['_30', '_50']):
            small_instances.append(instance_name)
        elif '_100' in instance_name:
            medium_instances.append(instance_name)
        else:
            large_instances.append(instance_name)
    
    # Generate plot for each size category
    for category, instances in [
        ('small', small_instances),
        ('medium', medium_instances),
        ('large', large_instances)
    ]:
        if not instances:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        hvs = []
        counts = []
        labels = []
        
        for instance_name in instances:
            data = summary[instance_name]
            # Get the first (and typically only) test type's data
            test_type = list(data['hypervolumes'].keys())[0]
            hv = data['hypervolumes'][test_type]
            count = data['solution_counts'][test_type]
            
            hvs.append(hv)
            counts.append(count)
            labels.append(instance_name.replace('.dzn', '').replace('_', ' ').title())
        
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, hvs, color='#0F52BA', alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add labels on bars
        for i, (bar, hv, count) in enumerate(zip(bars, hvs, counts)):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, height,
                       f'{hv:.3f}\n({count} sols)',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Instance', fontsize=14, fontweight='bold')
        ax.set_ylabel('Hypervolume', fontsize=14, fontweight='bold')
        ax.set_title(f'MILP Solver Performance: {category.upper()} Instances', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 1.19)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        plt.tight_layout()
        
        plot_file = plots_dir / f"milp_{category}_instances.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Generated plot: {plot_file}")
    
    print(f"Generated {len([f for f in plots_dir.glob('milp_*_instances.png')])} bar plots")


def generate_timeline_plots(summary: Dict, results: Dict, output_dir: Path):
    """Generate solution count over time plots for each category."""
    if not PLOTTING_AVAILABLE:
        print("Plotting not available, skipping timeline plot generation")
        return
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print("\nGenerating timeline plots...")
    
    # Custom sort: first by city name, then by size
    def sort_key(instance_name):
        parts = instance_name.rsplit('_', 1)
        if len(parts) == 2:
            city_name = parts[0]
            try:
                size = int(parts[1])
            except ValueError:
                size = 0
            return (city_name, size)
        return (instance_name, 0)
    
    # Group instances by size category
    categories = {
        'small': [],
        'medium': [],
        'large': []
    }
    
    for instance_name in sorted(summary.keys(), key=sort_key):
        if any(x in instance_name for x in ['_30', '_50']):
            categories['small'].append(instance_name)
        elif '_100' in instance_name:
            categories['medium'].append(instance_name)
        else:
            categories['large'].append(instance_name)
    
    # Color palette for different instances
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Generate timeline plot for each size category
    for category, instances in categories.items():
        if not instances:
            continue
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for idx, instance_name in enumerate(instances):
            test_results = results.get(instance_name, {})
            for test_type, data in test_results.items():
                solutions = data.get('solutions', [])
                if not solutions:
                    continue
                
                # Extract timestamps and sort by time
                timestamps = []
                for sol in solutions:
                    ts = sol.get('timestamp_s', 0)
                    timestamps.append(ts)
                
                timestamps.sort()
                
                # Create cumulative count
                solution_counts = list(range(1, len(timestamps) + 1))
                
                # Plot the series
                color = colors[idx % len(colors)]
                label = instance_name.replace('_', ' ').title()
                ax.plot(timestamps, solution_counts, marker='o', markersize=5,
                       linewidth=2, alpha=0.7, color=color, label=label)
        
        ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative Solution Count', fontsize=14, fontweight='bold')
        ax.set_title(f'Solution Discovery Over Time: {category.upper()} Instances', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        # Set x-axis to start at 0
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        plot_file = plots_dir / f"milp_{category}_timeline.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Generated timeline plot: {plot_file}")
    
    print(f"Generated {len([f for f in plots_dir.glob('milp_*_timeline.png')])} timeline plots")



def generate_report(summary: Dict, output_dir: Path):
    """Generate markdown report."""
    report_file = output_dir / "milp_report.md"
    
    # Custom sort: first by city name, then by size
    def sort_key(instance_name):
        parts = instance_name.rsplit('_', 1)
        if len(parts) == 2:
            city_name = parts[0]
            try:
                size = int(parts[1])
            except ValueError:
                size = 0
            return (city_name, size)
        return (instance_name, 0)
    
    with open(report_file, 'w') as f:
        f.write("# MILP Solver Results Report\n\n")
        f.write("## Summary Statistics\n\n")
        
        total_instances = len(summary)
        total_solutions = sum(
            sum(data['solution_counts'].values()) 
            for data in summary.values()
        )
        
        f.write(f"- **Total Instances**: {total_instances}\n")
        f.write(f"- **Total Solutions**: {total_solutions}\n\n")
        
        f.write("## Hypervolume Results\n\n")
        f.write("| Instance | Test Type | Hypervolume | Solutions |\n")
        f.write("|----------|-----------|-------------|----------|\n")
        
        for instance_name in sorted(summary.keys(), key=sort_key):
            data = summary[instance_name]
            for test_type in sorted(data['hypervolumes'].keys()):
                hv = data['hypervolumes'][test_type]
                count = data['solution_counts'][test_type]
                f.write(f"| {instance_name} | {test_type} | {hv:.6f} | {count} |\n")
        
        f.write("\n## Instance Categories\n\n")
        
        for category, pattern, plot_name in [
            ('Small (30-50 nodes)', ['_30', '_50'], 'milp_small_instances.png'),
            ('Medium (100 nodes)', ['_100'], 'milp_medium_instances.png'),
            ('Large (145+ nodes)', ['_145', '_150', '_200'], 'milp_large_instances.png')
        ]:
            f.write(f"### {category}\n\n")
            instances = [name for name in sorted(summary.keys()) 
                        if any(p in name for p in pattern)]
            
            if instances:
                avg_hv = sum(
                    list(summary[name]['hypervolumes'].values())[0]
                    for name in instances
                ) / len(instances)
                avg_sols = sum(
                    list(summary[name]['solution_counts'].values())[0]
                    for name in instances
                ) / len(instances)
                
                f.write(f"- **Count**: {len(instances)}\n")
                f.write(f"- **Average Hypervolume**: {avg_hv:.6f}\n")
                f.write(f"- **Average Solutions**: {avg_sols:.1f}\n\n")
                
                # Add plot image
                plot_path = f"plots/{plot_name}"
                if (output_dir / plot_path).exists():
                    f.write(f"![{category}]({plot_path})\n\n")
    
    print(f"Report written to: {report_file}")


def generate_report_with_timestamps(summary: Dict, results: Dict, output_dir: Path):
    """Generate markdown report with timestamps for large instances."""
    report_file = output_dir / "milp_report.md"
    
    # Custom sort: first by city name, then by size
    def sort_key(instance_name):
        parts = instance_name.rsplit('_', 1)
        if len(parts) == 2:
            city_name = parts[0]
            try:
                size = int(parts[1])
            except ValueError:
                size = 0
            return (city_name, size)
        return (instance_name, 0)
    
    with open(report_file, 'w') as f:
        f.write("# MILP Solver Results Report\n\n")
        f.write("## Summary Statistics\n\n")
        
        total_instances = len(summary)
        total_solutions = sum(
            sum(data['solution_counts'].values()) 
            for data in summary.values()
        )
        
        f.write(f"- **Total Instances**: {total_instances}\n")
        f.write(f"- **Total Solutions**: {total_solutions}\n\n")
        
        f.write("## Hypervolume Results\n\n")
        f.write("| Instance | Test Type | Hypervolume | Solutions |\n")
        f.write("|----------|-----------|-------------|----------|\n")
        
        for instance_name in sorted(summary.keys(), key=sort_key):
            data = summary[instance_name]
            for test_type in sorted(data['hypervolumes'].keys()):
                hv = data['hypervolumes'][test_type]
                count = data['solution_counts'][test_type]
                f.write(f"| {instance_name} | {test_type} | {hv:.6f} | {count} |\n")
        
        f.write("\n## Instance Categories\n\n")
        
        for category, pattern, plot_name, timeline_plot in [
            ('Small (30-50 nodes)', ['_30', '_50'], 'milp_small_instances.png', 'milp_small_timeline.png'),
            ('Medium (100 nodes)', ['_100'], 'milp_medium_instances.png', 'milp_medium_timeline.png'),
            ('Large (145+ nodes)', ['_145', '_150', '_200'], 'milp_large_instances.png', 'milp_large_timeline.png')
        ]:
            f.write(f"### {category}\n\n")
            instances = [name for name in sorted(summary.keys(), key=sort_key) 
                        if any(p in name for p in pattern)]
            
            if instances:
                avg_hv = sum(
                    list(summary[name]['hypervolumes'].values())[0]
                    for name in instances
                ) / len(instances)
                avg_sols = sum(
                    list(summary[name]['solution_counts'].values())[0]
                    for name in instances
                ) / len(instances)
                
                f.write(f"- **Count**: {len(instances)}\n")
                f.write(f"- **Average Hypervolume**: {avg_hv:.6f}\n")
                f.write(f"- **Average Solutions**: {avg_sols:.1f}\n\n")
                
                # Add hypervolume bar plot
                plot_path = f"plots/{plot_name}"
                if (output_dir / plot_path).exists():
                    f.write(f"![{category}]({plot_path})\n\n")
                
                # Add timeline plot
                timeline_path = f"plots/{timeline_plot}"
                if (output_dir / timeline_path).exists():
                    f.write(f"**Solution Discovery Timeline:**\n\n")
                    f.write(f"![{category} Timeline]({timeline_path})\n\n")
                
                # Add timestamps section for large instances
                if 'Large' in category:
                    f.write("#### Solution Timestamps\n\n")
                    for instance_name in instances:
                        test_results = results.get(instance_name, {})
                        for test_type, data in test_results.items():
                            solutions = data.get('solutions', [])
                            if solutions:
                                f.write(f"**{instance_name}** ({len(solutions)} solutions):\n")
                                for sol in solutions:
                                    timestamp = sol.get('timestamp_s', 0)
                                    index = sol.get('index', '?')
                                    f.write(f"  - Solution {index}: {timestamp:.2f}s\n")
                                f.write("\n")
    
    print(f"Report written to: {report_file}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_milp_results.py <test_artifacts_dir>")
        sys.exit(1)
    
    artifacts_dir = Path(sys.argv[1])
    
    if not artifacts_dir.exists():
        print(f"Error: Directory {artifacts_dir} does not exist")
        sys.exit(1)
    
    print(f"Parsing MILP test artifacts from: {artifacts_dir}")
    
    # Parse results
    results, global_bounds_map, global_ref_map = parse_test_artifacts(artifacts_dir)
    
    if not results:
        print("Error: No valid results found")
        sys.exit(1)
    
    # Generate summary
    summary = generate_summary(results, global_bounds_map, global_ref_map, artifacts_dir)
    
    # Generate plots
    generate_plots(summary, artifacts_dir)
    
    # Generate timeline plots
    generate_timeline_plots(summary, results, artifacts_dir)
    
    # Generate report with timestamps
    generate_report_with_timestamps(summary, results, artifacts_dir)
    
    print("\nMILP results analysis complete!")


if __name__ == "__main__":
    main()
