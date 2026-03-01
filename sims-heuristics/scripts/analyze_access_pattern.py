import re
import statistics
import sys
from pathlib import Path

def parse_dzn_sets(content, var_name):
    pattern = rf"{var_name}\s*=\s*\[(.*?)\];"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return []
    
    array_content = match.group(1)
    sets_str = re.split(r"\}\s*,\s*\{", array_content)
    
    parsed_sets = []
    for s in sets_str:
        s = s.replace("{", "").replace("}", "").strip()
        if not s:
            parsed_sets.append([])
            continue
        nums = [int(x) for x in re.split(r",\s*", s) if x]
        parsed_sets.append(sorted(nums))
    return parsed_sets

def analyze_gaps(sets, name=""):
    all_gaps = []
    total_elements = 0
    total_jumps = 0
    total_elements_in_sets = sum(len(s) for s in sets)
    
    for s in sets:
        if len(s) < 2:
            continue
        for i in range(len(s) - 1):
            gap = abs(s[i+1] - s[i])
            all_gaps.append(gap)
            total_jumps += 1
            total_elements += 1
            
    if not all_gaps:
        print(f"[{name}] No data")
        return

    avg_gap = statistics.mean(all_gaps)
    median_gap = statistics.median(all_gaps)
    max_gap = max(all_gaps)
    
    print(f"[{name}] Analysis of {len(sets)} sets:")
    print(f"  Total Transitions: {total_jumps}")
    print(f"  Average Gap:       {avg_gap:.2f}")
    print(f"  Median Gap:        {median_gap}")
    print(f"  Max Gap:           {max_gap}")
    
    cache_line_hits = sum(1 for g in all_gaps if g <= 16)
    print(f"  Cache Line Hits:   {cache_line_hits} ({cache_line_hits/total_jumps*100:.1f}%)")

    # Run Length Analysis
    elements_in_runs = 0
    run_threshold = 4
    stride_2_elements = 0
    
    for s in sets:
        if not s: continue
        
        # Consec runs
        current_run = 1
        for i in range(len(s) - 1):
            if s[i+1] == s[i] + 1:
                current_run += 1
            else:
                if current_run >= run_threshold:
                    elements_in_runs += current_run
                current_run = 1
        if current_run >= run_threshold:
            elements_in_runs += current_run

        # Stride 2 runs (1, 3, 5)
        curr_stride_2 = 1
        for i in range(len(s) - 1):
            if s[i+1] == s[i] + 2:
                curr_stride_2 += 1
            else:
                if curr_stride_2 >= run_threshold:
                    stride_2_elements += curr_stride_2
                curr_stride_2 = 1
        if curr_stride_2 >= run_threshold:
            stride_2_elements += curr_stride_2

    print(f"  Strict Runs (>=4): {elements_in_runs} ({elements_in_runs/total_elements_in_sets*100:.1f}%)")
    print(f"  Stride-2 Runs(>=4):{stride_2_elements} ({stride_2_elements/total_elements_in_sets*100:.1f}%)")
    print("-" * 20)

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <file.dzn>")
        sys.exit(1)
        
    path = Path(sys.argv[1])
    content = path.read_text()
    
    images = parse_dzn_sets(content, "images")
    clouds = parse_dzn_sets(content, "clouds")
    
    analyze_gaps(images, "Images (Coverage)")
    analyze_gaps(clouds, "Clouds (Exclusion)")

if __name__ == "__main__":
    main()
