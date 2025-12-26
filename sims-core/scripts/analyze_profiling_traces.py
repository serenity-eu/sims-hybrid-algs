#!/usr/bin/env python3
"""
Analyze Chrome profiling traces from PLS runs to compare performance between different Pareto archive implementations.

This script parses profiling_trace.json files and provides insights into:
- Total execution time
- Time spent in different functions/operations
- Why one implementation might find fewer solutions
- Performance bottlenecks
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import argparse


def load_trace(trace_path: Path) -> List[Dict[str, Any]]:
    """Load a Chrome trace JSON file."""
    with open(trace_path, 'r') as f:
        return json.load(f)


def analyze_trace_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze trace events and compute statistics."""
    stats = {
        'total_duration_us': 0,
        'function_durations': defaultdict(float),
        'function_counts': defaultdict(int),
        'phase_durations': defaultdict(float),
        'event_count': len(events),
    }
    
    # Track begin/end pairs for duration events
    stack = {}
    
    for event in events:
        if 'name' not in event:
            continue
            
        name = event['name']
        ph = event.get('ph', '')
        ts = event.get('ts', 0)  # timestamp in microseconds
        dur = event.get('dur', 0)  # duration in microseconds
        
        # Handle complete events (X phase)
        if ph == 'X' and dur > 0:
            stats['function_durations'][name] += dur
            stats['function_counts'][name] += 1
            
        # Handle begin/end pairs (B/E phase)
        elif ph == 'B':
            # Begin event
            tid = event.get('tid', 0)
            key = (tid, name)
            stack[key] = ts
        elif ph == 'E':
            # End event
            tid = event.get('tid', 0)
            key = (tid, name)
            if key in stack:
                begin_ts = stack.pop(key)
                duration = ts - begin_ts
                stats['function_durations'][name] += duration
                stats['function_counts'][name] += 1
    
    # Calculate total duration from first to last event
    if events:
        timestamps = [e.get('ts', 0) for e in events if 'ts' in e]
        if timestamps:
            stats['total_duration_us'] = max(timestamps) - min(timestamps)
    
    return stats


def format_duration(microseconds: float) -> str:
    """Format duration in human-readable form."""
    if microseconds < 1000:
        return f"{microseconds:.2f} µs"
    elif microseconds < 1_000_000:
        return f"{microseconds / 1000:.2f} ms"
    else:
        return f"{microseconds / 1_000_000:.2f} s"


def compare_traces(trace1_path: Path, trace2_path: Path, label1: str = "Trace 1", label2: str = "Trace 2"):
    """Compare two profiling traces and print analysis."""
    
    print(f"\n{'='*80}")
    print(f"Profiling Trace Comparison: {label1} vs {label2}")
    print(f"{'='*80}\n")
    
    # Load traces
    print(f"Loading {label1}: {trace1_path}")
    events1 = load_trace(trace1_path)
    print(f"Loading {label2}: {trace2_path}")
    events2 = load_trace(trace2_path)
    
    # Analyze both traces
    stats1 = analyze_trace_events(events1)
    stats2 = analyze_trace_events(events2)
    
    print(f"\n{'-'*80}")
    print("Overall Statistics")
    print(f"{'-'*80}")
    print(f"{'Metric':<40} {label1:>18} {label2:>18}")
    print(f"{'-'*80}")
    print(f"{'Total Events':<40} {stats1['event_count']:>18,} {stats2['event_count']:>18,}")
    print(f"{'Total Duration':<40} {format_duration(stats1['total_duration_us']):>18} {format_duration(stats2['total_duration_us']):>18}")
    print(f"{'Unique Functions':<40} {len(stats1['function_durations']):>18,} {len(stats2['function_durations']):>18,}")
    
    # Compare function durations
    print(f"\n{'-'*80}")
    print("Top Functions by Total Duration")
    print(f"{'-'*80}")
    
    # Get top functions from both traces
    all_functions = set(stats1['function_durations'].keys()) | set(stats2['function_durations'].keys())
    
    # Sort by maximum duration across both traces
    sorted_functions = sorted(
        all_functions,
        key=lambda f: max(stats1['function_durations'].get(f, 0), stats2['function_durations'].get(f, 0)),
        reverse=True
    )[:20]  # Top 20 functions
    
    print(f"{'Function':<40} {label1 + ' (ms)':>18} {label2 + ' (ms)':>18} {'Diff (ms)':>18} {'% Diff':>10}")
    print(f"{'-'*80}")
    
    for func in sorted_functions:
        dur1 = stats1['function_durations'].get(func, 0) / 1000  # convert to ms
        dur2 = stats2['function_durations'].get(func, 0) / 1000
        diff = dur2 - dur1
        pct_diff = (diff / dur1 * 100) if dur1 > 0 else 0
        
        # Truncate function name if too long
        display_name = func if len(func) <= 39 else func[:36] + "..."
        
        print(f"{display_name:<40} {dur1:>18.2f} {dur2:>18.2f} {diff:>18.2f} {pct_diff:>9.1f}%")
    
    # Compare function call counts
    print(f"\n{'-'*80}")
    print("Top Functions by Call Count")
    print(f"{'-'*80}")
    
    sorted_by_count = sorted(
        all_functions,
        key=lambda f: max(stats1['function_counts'].get(f, 0), stats2['function_counts'].get(f, 0)),
        reverse=True
    )[:20]
    
    print(f"{'Function':<40} {label1 + ' (calls)':>18} {label2 + ' (calls)':>18} {'Diff':>18} {'% Diff':>10}")
    print(f"{'-'*80}")
    
    for func in sorted_by_count:
        count1 = stats1['function_counts'].get(func, 0)
        count2 = stats2['function_counts'].get(func, 0)
        diff = count2 - count1
        pct_diff = (diff / count1 * 100) if count1 > 0 else 0
        
        display_name = func if len(func) <= 39 else func[:36] + "..."
        
        print(f"{display_name:<40} {count1:>18,} {count2:>18,} {diff:>18,} {pct_diff:>9.1f}%")
    
    # Look for PLS-specific operations
    print(f"\n{'-'*80}")
    print("PLS-Specific Operations")
    print(f"{'-'*80}")
    
    pls_keywords = ['insert', 'dominated', 'dominates', 'pareto', 'neighbor', 'explore', 'iteration']
    pls_functions = [f for f in all_functions if any(kw in f.lower() for kw in pls_keywords)]
    
    if pls_functions:
        print(f"{'Function':<40} {label1 + ' (ms)':>18} {label2 + ' (ms)':>18} {'Diff (ms)':>18}")
        print(f"{'-'*80}")
        
        for func in sorted(pls_functions, key=lambda f: stats2['function_durations'].get(f, 0), reverse=True)[:15]:
            dur1 = stats1['function_durations'].get(func, 0) / 1000
            dur2 = stats2['function_durations'].get(func, 0) / 1000
            diff = dur2 - dur1
            
            display_name = func if len(func) <= 39 else func[:36] + "..."
            print(f"{display_name:<40} {dur1:>18.2f} {dur2:>18.2f} {diff:>18.2f}")
    
    # Summary insights
    print(f"\n{'-'*80}")
    print("Key Insights")
    print(f"{'-'*80}")
    
    # Compare results
    result1_path = trace1_path.parent / "result.json"
    result2_path = trace2_path.parent / "result.json"
    
    if result1_path.exists() and result2_path.exists():
        with open(result1_path) as f:
            result1 = json.load(f)
        with open(result2_path) as f:
            result2 = json.load(f)
        
        sol_count1 = result1.get('num_solutions', 0)
        sol_count2 = result2.get('num_solutions', 0)
        
        print(f"\n• Solutions found: {label1} = {sol_count1}, {label2} = {sol_count2}")
        
        if sol_count2 < sol_count1:
            print(f"• {label2} found {sol_count1 - sol_count2} fewer solutions ({(sol_count1 - sol_count2) / sol_count1 * 100:.1f}% less)")
        elif sol_count2 > sol_count1:
            print(f"• {label2} found {sol_count2 - sol_count1} more solutions ({(sol_count2 - sol_count1) / sol_count1 * 100:.1f}% more)")
        
        exec_time1 = result1.get('execution_time', 0)
        exec_time2 = result2.get('execution_time', 0)
        print(f"• Execution time: {label1} = {exec_time1:.2f}s, {label2} = {exec_time2:.2f}s")
    
    # Find functions where trace2 is significantly slower
    print(f"\n• Functions where {label2} is significantly slower:")
    slowest = []
    for func in all_functions:
        dur1 = stats1['function_durations'].get(func, 0)
        dur2 = stats2['function_durations'].get(func, 0)
        if dur1 > 0 and dur2 > dur1 * 1.5:  # At least 50% slower
            slowest.append((func, dur2 - dur1, (dur2 - dur1) / dur1 * 100))
    
    slowest.sort(key=lambda x: x[1], reverse=True)
    for func, diff, pct in slowest[:5]:
        print(f"  - {func}: {format_duration(diff)} slower ({pct:.1f}% increase)")


def main():
    parser = argparse.ArgumentParser(description="Analyze and compare Chrome profiling traces")
    parser.add_argument("trace1", type=Path, help="Path to first profiling trace JSON")
    parser.add_argument("trace2", type=Path, help="Path to second profiling trace JSON")
    parser.add_argument("--label1", default="Trace 1", help="Label for first trace")
    parser.add_argument("--label2", default="Trace 2", help="Label for second trace")
    
    args = parser.parse_args()
    
    if not args.trace1.exists():
        print(f"Error: {args.trace1} does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not args.trace2.exists():
        print(f"Error: {args.trace2} does not exist", file=sys.stderr)
        sys.exit(1)
    
    compare_traces(args.trace1, args.trace2, args.label1, args.label2)


if __name__ == "__main__":
    main()
