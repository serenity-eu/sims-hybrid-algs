#!/usr/bin/env python3
"""
Deep Analysis Script for SIMS Problem Instances.

This script iterates through all .dzn instances in tests/data, loads them,
and computes detailed statistical and correlation metrics specific to the SIMS problem domain.
"""

import argparse
from pathlib import Path
import statistics
import math
from typing import List, Dict, Any, Tuple
from rich.console import Console
from rich.table import Table
from rich.progress import track
import sims_problem

def calculate_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient between two lists."""
    if len(x) != len(y):
        raise ValueError("Lists must be of same length")
    if len(x) < 2:
        return 0.0
    
    # Check for constant values (stdev would be 0)
    if len(set(x)) == 1 or len(set(y)) == 1:
        return 0.0

    try:
        # Use statistics.correlation if available (Python 3.10+)
        # Or manual calculation for robustness
        n = len(x)
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        sum_sq_diff_x = sum((xi - mean_x) ** 2 for xi in x)
        sum_sq_diff_y = sum((yi - mean_y) ** 2 for yi in y)
        
        denominator = math.sqrt(sum_sq_diff_x * sum_sq_diff_y)
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator
    except Exception:
        return 0.0

def analyze_instance(file_path: Path) -> Dict[str, Any]:
    """Load and analyze a single SIMS instance."""
    try:
        problem = sims_problem.SimsDiscreteProblem.from_dzn(str(file_path))
    except Exception as e:
        return {"error": str(e)}

    # Basic Dimensions
    num_images = problem.num_images
    universe = problem.universe

    # -- Set Cover Statistics --
    
    # 1. Image Coverage Size (How many elements does each image cover?)
    image_coverage_sizes = [len(img_set) for img_set in problem.images]
    avg_img_coverage = statistics.mean(image_coverage_sizes)
    
    # 2. Element Coverage (How many images cover a specific element?)
    # Initialize count for each element
    element_coverage_counts = [0] * universe
    for img_set in problem.images:
        for elem_idx in img_set:
            if elem_idx < universe:
                element_coverage_counts[elem_idx] += 1
            
    # Filter out potential out-of-bounds just in case, though shouldn't happen in valid instance
    min_element_cov = min(element_coverage_counts) if element_coverage_counts else 0
    max_element_cov = max(element_coverage_counts) if element_coverage_counts else 0
    avg_element_cov = statistics.mean(element_coverage_counts) if element_coverage_counts else 0
    
    density = sum(image_coverage_sizes) / (num_images * universe) if (num_images * universe) > 0 else 0

    # -- Objective Statistics (Per Image) --
    costs = problem.costs
    resolutions = problem.resolution
    incidences = problem.incidence_angle
    
    # Cloudiness calculation per image
    # cloudiness = (number of cloudy pixels in image) / (total pixels in image)
    # Note: problem.clouds[i] is the set of cloudy elements for image i.
    # problem.images[i] is the set of total elements covered by image i.
    cloud_ratios = []
    for i in range(num_images):
        total_pixels = len(problem.images[i])
        if total_pixels > 0:
            cloudy_pixels = len(problem.clouds[i])
            cloud_ratios.append(cloudy_pixels / total_pixels)
        else:
            cloud_ratios.append(0.0)
    
    avg_cloudiness = statistics.mean(cloud_ratios)

    # -- Correlations --
    # Correlate Cost with other factors
    corr_cost_size = calculate_correlation(costs, image_coverage_sizes)
    corr_cost_res = calculate_correlation(costs, resolutions)
    corr_cost_cloud = calculate_correlation(costs, cloud_ratios)
    
    # Correlate Resolution with others
    corr_res_incidence = calculate_correlation(resolutions, incidences)
    
    return {
        "name": file_path.stem,
        "universe": universe,
        "num_images": num_images,
        "density": density,
        "avg_img_coverage": avg_img_coverage,
        "min_elem_cov": min_element_cov,
        "max_elem_cov": max_element_cov,
        "avg_elem_cov": avg_element_cov,
        "avg_cost": statistics.mean(costs),
        "avg_res": statistics.mean(resolutions),
        "avg_cloudiness": avg_cloudiness,
        "corr_cost_size": corr_cost_size,
        "corr_cost_res": corr_cost_res,
        "corr_cost_cloud": corr_cost_cloud,
        "corr_res_incidence": corr_res_incidence
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze SIMS problem instances.")
    parser.add_argument("--data-dir", type=str, default="tests/data", help="Directory containing .dzn files")
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: {data_path} does not exist.")
        return

    files = sorted(list(data_path. glob("*.dzn")), key=lambda p: p.name)
    if not files:
        print("No .dzn files found.")
        return

    console = Console()
    
    results = []
    with console.status("[bold green]Analyzing instances...") as status:
        for file_path in files:
            # Skip lagos_nigeria_30 if it's causing issues or just process all
            res = analyze_instance(file_path)
            if "error" not in res:
                results.append(res)
            else:
                console.print(f"[red]Error analyzing {file_path.name}: {res['error']}[/red]")

    # Create Summary Table
    table = Table(title="SIMS Instance Analysis", show_lines=True)
    table.add_column("Instance", style="cyan", no_wrap=True)
    table.add_column("Univ / Img", justify="center")
    table.add_column("Density", justify="right")
    table.add_column("Elem Cov\n(Min/Avg/Max)", justify="center")
    table.add_column("Cloud%", justify="right")
    table.add_column("Corr: Cost\nvs Size", justify="right")
    table.add_column("Corr: Cost\nvs Res", justify="right")
    table.add_column("Corr: Res\nvs Inc", justify="right")

    for r in results:
        table.add_row(
            r["name"],
            f"{r['universe']} / {r['num_images']}",
            f"{r['density']:.3f}",
            f"{r['min_elem_cov']} / {r['avg_elem_cov']:.1f} / {r['max_elem_cov']}",
            f"{r['avg_cloudiness']*100:.1f}%",
            f"{r['corr_cost_size']:.2f}",
            f"{r['corr_cost_res']:.2f}",
            f"{r['corr_res_incidence']:.2f}"
        )

    console.print(table)
    
    # Identify outliers or interesting patterns
    console.print("\n[bold]Key Observations:[/bold]")
    
    # Verify density assumption (often sparse?)
    avg_density = statistics.mean([r['density'] for r in results])
    console.print(f"Average Problem Density: {avg_density:.4f}")
    
    # Check max coverage vs 16 (for SmallVec/ArrayVec optimization)
    above_16 = [r['name'] for r in results if r['max_elem_cov'] > 16]
    if above_16:
        console.print(f"[yellow]Instances exceeding 16 images per element (ArrayVec risk):[/yellow] {', '.join(above_16)}")
    else:
        console.print("[green]All instances have max element coverage <= 16 (Safe for ArrayVec<16>)[/green]")

if __name__ == "__main__":
    main()
