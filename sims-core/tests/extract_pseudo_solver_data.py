#!/usr/bin/env python3
"""
Extract solution data from test artifacts (100_0 ratio) to create pseudo-solver data.
This script aggregates solutions from all iterations and instances to create a dataset
that can be used by a pseudo-solver for faster testing.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_solutions_from_artifact(artifact_dir: Path) -> dict:
    """Extract solutions from a single test artifact directory."""
    result_file = artifact_dir / "result.json"
    
    if not result_file.exists():
        logger.warning(f"No result.json found in {artifact_dir}")
        return {}
    
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    return data


def aggregate_solutions_by_instance(artifacts_base_dir: Path) -> dict[str, dict]:
    """
    Aggregate all solutions from 100_0 ratio artifacts, grouped by instance.
    
    Returns a dict with structure:
    {
        "instance_name": {
            "test_type": "2d/3d/4d",
            "objectives": ["min_cost", "cloud_coverage", ...],
            "solutions": [list of all solutions with timestamps]
        }
    }
    """
    aggregated: dict[str, dict] = {}
    
    # Find all 100_0 ratio directories
    for ratio_dir in artifacts_base_dir.rglob("100_0"):
        # Get instance name from parent directory
        instance_name = ratio_dir.parent.name
        
        # Process all iterations in this ratio directory
        for iter_dir in ratio_dir.iterdir():
            if not iter_dir.is_dir() or not iter_dir.name.startswith("iter"):
                continue
            
            data = extract_solutions_from_artifact(iter_dir)
            
            if not data or "solutions" not in data:
                continue
            
            # Initialize instance data if not present
            if instance_name not in aggregated:
                aggregated[instance_name] = {
                    "solutions": [],
                    "test_type": None,
                    "objectives": None
                }
            
            # Set metadata if not already set
            if aggregated[instance_name]["test_type"] is None:
                aggregated[instance_name]["test_type"] = data.get("test_type", "unknown")
            if aggregated[instance_name]["objectives"] is None:
                aggregated[instance_name]["objectives"] = data.get("objectives", [])
            
            # Add solutions from this iteration
            aggregated[instance_name]["solutions"].extend(data["solutions"])
    
    # Sort solutions by timestamp for each instance
    for instance_name in aggregated:
        aggregated[instance_name]["solutions"].sort(
            key=lambda s: s.get("timestamp_s", 0)
        )
    
    return aggregated


def save_aggregated_data(aggregated_data: dict, output_dir: Path):
    """Save aggregated solution data to JSON files, one per instance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for instance_name, data in aggregated_data.items():
        output_file = output_dir / f"{instance_name}.json"
        
        # Create output structure
        output = {
            "instance_name": instance_name,
            "test_type": data["test_type"],
            "objectives": data["objectives"],
            "num_solutions": len(data["solutions"]),
            "solutions": data["solutions"]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved {len(data['solutions'])} solutions for {instance_name} to {output_file}")


def main():
    """Main extraction process."""
    # Find the test artifacts directory
    script_dir = Path(__file__).parent
    artifacts_dir = script_dir.parent / "test_artifacts"
    
    if not artifacts_dir.exists():
        logger.error(f"Test artifacts directory not found: {artifacts_dir}")
        logger.info(f"Script directory: {script_dir}")
        logger.info(f"Looking for artifacts in: {artifacts_dir}")
        return
    
    # Output directory for pseudo-solver data
    output_dir = script_dir / "data" / "pseudo_solver_solutions"
    
    logger.info(f"Searching for 100_0 ratio artifacts in {artifacts_dir}")
    
    # Aggregate all solutions
    aggregated_data = aggregate_solutions_by_instance(artifacts_dir)
    
    if not aggregated_data:
        logger.error("No solution data found in artifacts")
        return
    
    logger.info(f"Found solution data for {len(aggregated_data)} instances")
    
    # Save the aggregated data
    save_aggregated_data(aggregated_data, output_dir)
    
    logger.info(f"Extraction complete. Data saved to {output_dir}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    for instance_name, data in sorted(aggregated_data.items()):
        print(f"{instance_name}: {len(data['solutions'])} solutions ({data['test_type']})")


if __name__ == "__main__":
    main()