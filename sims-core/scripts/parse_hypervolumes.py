#!/usr/bin/env python3
"""
Parse hypervolume data from test artifacts and generate summary.json.

This script processes trace.tar.gz archives from hybrid algorithm test runs,
extracting the final hypervolume value for each instance and ratio combination.

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
import struct
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


def extract_final_hypervolume(trace_path: Path) -> Optional[float]:
    """
    Extract the final (last) hypervolume value from a trace.tar.gz archive.
    
    According to TRACE_SPECIFICATION.md:
    - hypervolume.bin contains f64 values in little-endian format
    - Each value is 8 bytes
    - Values represent cumulative hypervolume progression
    - We need the last value
    
    Args:
        trace_path: Path to trace.tar.gz file
        
    Returns:
        Final hypervolume value as float, or None if extraction fails
    """
    try:
        with tarfile.open(trace_path, 'r:gz') as tar:
            # Extract hypervolume.bin from the archive
            hypervolume_member = None
            for member in tar.getmembers():
                if member.name.endswith('hypervolume.bin'):
                    hypervolume_member = member
                    break
            
            if hypervolume_member is None:
                print(f"Warning: No hypervolume.bin found in {trace_path}", file=sys.stderr)
                return None
            
            # Read the binary data
            f = tar.extractfile(hypervolume_member)
            if f is None:
                print(f"Warning: Could not extract hypervolume.bin from {trace_path}", file=sys.stderr)
                return None
            
            data = f.read()
            
            # Each hypervolume value is 8 bytes (f64)
            if len(data) == 0:
                print(f"Warning: Empty hypervolume.bin in {trace_path}", file=sys.stderr)
                return None
            
            if len(data) % 8 != 0:
                print(f"Warning: Invalid hypervolume.bin size in {trace_path} (not multiple of 8)", file=sys.stderr)
                return None
            
            # Read the last f64 value (little-endian)
            last_hv_bytes = data[-8:]
            final_hypervolume = struct.unpack('<d', last_hv_bytes)[0]
            
            return final_hypervolume
            
    except Exception as e:
        print(f"Error processing {trace_path}: {e}", file=sys.stderr)
        return None


def extract_instance_name(path: Path) -> Optional[str]:
    """
    Extract instance name from path structure.
    
    Expected structure:
    .../solve_two_phase_4d_small_0_100/lagos_nigeria_30/trace.tar.gz
    
    Returns: "lagos_nigeria_30"
    """
    if path.name == "trace.tar.gz" and path.parent.name:
        return path.parent.name
    return None


def extract_ratio_from_algorithm_dir(dir_name: str) -> Optional[tuple]:
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


def parse_test_artifacts(artifacts_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    """
    Parse all trace.tar.gz files in the test artifacts directory.
    
    Args:
        artifacts_dir: Root directory containing algorithm subdirectories
        
    Returns:
        Dictionary mapping instance names to their hypervolume data:
        {
            "lagos_nigeria_30": {
                "hypervolumes": [hv1, hv2, hv3, hv4],
                "ratios": [(0, 100), (20, 80), (50, 50), (100, 0)]
            }
        }
    """
    results = defaultdict(lambda: {"hypervolumes": [], "ratios": []})
    
    # Find all trace.tar.gz files
    trace_files = list(artifacts_dir.rglob("trace.tar.gz"))
    
    if not trace_files:
        print(f"Warning: No trace.tar.gz files found in {artifacts_dir}", file=sys.stderr)
        return {}
    
    print(f"Found {len(trace_files)} trace files to process", file=sys.stderr)
    
    for trace_path in sorted(trace_files):
        # Extract instance name
        instance_name = extract_instance_name(trace_path)
        if instance_name is None:
            print(f"Warning: Could not extract instance name from {trace_path}", file=sys.stderr)
            continue
        
        # Extract ratio from algorithm directory (parent of instance directory)
        algorithm_dir = trace_path.parent.parent.name
        ratio = extract_ratio_from_algorithm_dir(algorithm_dir)
        if ratio is None:
            print(f"Warning: Could not extract ratio from {algorithm_dir}", file=sys.stderr)
            continue
        
        # Extract final hypervolume
        final_hv = extract_final_hypervolume(trace_path)
        if final_hv is None:
            continue
        
        # Store the result
        results[instance_name]["hypervolumes"].append(final_hv)
        results[instance_name]["ratios"].append(ratio)
        
        print(f"  {instance_name} [{ratio[0]}/{ratio[1]}]: HV = {final_hv:.6f}", file=sys.stderr)
    
    # Sort hypervolumes by ratio for consistent ordering
    for instance_name in results:
        # Create list of (ratio, hypervolume) pairs, sort by ratio, then extract hypervolumes
        paired = list(zip(results[instance_name]["ratios"], results[instance_name]["hypervolumes"]))
        paired.sort(key=lambda x: x[0])  # Sort by ratio tuple
        
        results[instance_name]["hypervolumes"] = [hv for _, hv in paired]
        results[instance_name]["ratios"] = [ratio for ratio, _ in paired]
    
    return dict(results)


def create_summary(results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    """
    Create the summary output format (just hypervolumes, no ratios).
    
    Args:
        results: Full results with hypervolumes and ratios
        
    Returns:
        Simplified dictionary with just hypervolumes per instance
    """
    summary = {}
    for instance_name, data in results.items():
        summary[instance_name] = {
            "hypervolumes": data["hypervolumes"]
        }
    return summary


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python parse_hypervolumes.py <artifacts_directory> [output_file]")
        print("Example: python parse_hypervolumes.py test_artifacts/hybrid_4d_wednesday_20251022_202800")
        sys.exit(1)
    
    artifacts_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else artifacts_dir / "summary.json"
    
    if not artifacts_dir.exists():
        print(f"Error: Directory {artifacts_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not artifacts_dir.is_dir():
        print(f"Error: {artifacts_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    
    print(f"Parsing test artifacts from: {artifacts_dir}", file=sys.stderr)
    
    # Parse all trace files
    results = parse_test_artifacts(artifacts_dir)
    
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
    
    # Print summary statistics
    print("\nSummary Statistics:", file=sys.stderr)
    for instance_name in sorted(summary.keys()):
        num_ratios = len(summary[instance_name]["hypervolumes"])
        print(f"  {instance_name}: {num_ratios} ratios", file=sys.stderr)


if __name__ == "__main__":
    main()
