# Optimization Trace Archive Format Specification

**Version**: 1.0  
**Date**: October 2025  
**Format**: `trace.tar.gz`

## Overview

The optimization trace archive format stores detailed timeline data from optimization algorithms in a compressed TAR-GZ archive. This format captures the complete evolution of the optimization process, including solution discovery, domination relationships, and hypervolume progression.

## File Structure

The archive contains the following files:

```
trace.tar.gz
├── objectives.bin      # Solution objectives (binary)
├── dominated.bin       # Domination relationships (binary)
├── timestamp.bin       # Discovery timestamps (binary)
├── hypervolume.bin     # Hypervolume progression (binary)
└── metadata.json       # Archive metadata (JSON)
```

## Container Format

- **Compression**: GNU gzip (RFC 1952)
- **Archive**: POSIX.1-1988 TAR format with GNU extensions
- **File Order**: Files may appear in any order within the archive

## Binary Data Encoding

All binary data uses **little-endian (LE) byte order** unless otherwise specified.

### 1. objectives.bin

**Purpose**: Raw objective values for each solution in discovery order.

**Format**:
- **Data Type**: `u64` (8 bytes per objective value)
- **Byte Order**: Little-endian
- **Structure**: Concatenated objective values for all solutions

**Layout**:
```
[Solution 0 Obj 0][Solution 0 Obj 1]...[Solution 0 Obj N-1]
[Solution 1 Obj 0][Solution 1 Obj 1]...[Solution 1 Obj N-1]
...
[Solution M-1 Obj 0][Solution M-1 Obj 1]...[Solution M-1 Obj N-1]
```

**Size Calculation**:
```
file_size = num_solutions × num_objectives × 8 bytes
```

**Example** (2 solutions, 3 objectives):
```
Solution 0: [1500, 250, 450] → bytes [0-23]
Solution 1: [1200, 300, 400] → bytes [24-47]
Total: 48 bytes
```

### 2. dominated.bin

**Purpose**: Domination relationships between solutions.

**Format**:
- **Data Type**: `u32` (4 bytes per domination index)
- **Byte Order**: Little-endian
- **Structure**: One domination index per solution

**Values**:
- `u32::MAX` (4,294,967,295): Solution is never dominated
- `0 to num_solutions-1`: Index of the first solution that dominates this solution

**Layout**:
```
[Solution 0 Dominated By][Solution 1 Dominated By]...[Solution M-1 Dominated By]
```

**Size Calculation**:
```
file_size = num_solutions × 4 bytes
```

**Domination Logic**:
- Solution A dominates Solution B if A is better or equal in all objectives and strictly better in at least one
- The domination index points to the **first** (earliest discovered) solution that dominates the current solution

### 3. timestamp.bin

**Purpose**: Discovery time for each solution.

**Format**:
- **Data Type**: `u32` (4 bytes per timestamp)
- **Byte Order**: Little-endian
- **Unit**: Microseconds since optimization start
- **Structure**: One timestamp per solution

**Layout**:
```
[Solution 0 Timestamp][Solution 1 Timestamp]...[Solution M-1 Timestamp]
```

**Size Calculation**:
```
file_size = num_solutions × 4 bytes
```

**Notes**:
- Timestamps are truncated from `u64` to `u32` (max ~4,295 seconds ≈ 71 minutes)
- Solutions should be sorted by timestamp in discovery order

### 4. hypervolume.bin

**Purpose**: Cumulative hypervolume progression over time.

**Format**:
- **Data Type**: `f64` (8 bytes per hypervolume value)
- **Byte Order**: Little-endian (IEEE 754 double precision)
- **Range**: [0.0, 1.0] (normalized hypervolume)
- **Structure**: One hypervolume value per solution

**Layout**:
```
[HV at Solution 0][HV at Solution 1]...[HV at Solution M-1]
```

**Size Calculation**:
```
file_size = num_solutions × 8 bytes
```

**Computation Details**:
- Hypervolume is computed using scaled objectives in [0,1] range
- Each value represents the hypervolume of the Pareto front **up to that point in time**
- Values are constant between hypervolume change points
- Reference point is determined from objective bounds: `(max_bound - min_bound) / (max_bound - min_bound) = 1.0` for each objective

**Change Point Detection**:
Hypervolume changes only when a non-dominated solution is discovered:
- Solution `i` causes hypervolume change if `dominated[i] > i` or `dominated[i] == u32::MAX`

### 5. metadata.json

**Purpose**: Archive metadata and configuration.

**Format**: UTF-8 encoded JSON

**Schema**:
```json
{
  "objectives": ["string", ...],           // Objective names (required)
  "total_duration": integer,               // Total optimization time in μs (required)
  "algorithm": "string",                   // Algorithm name (required)
  "solution_count": integer,               // Number of solutions (required)
  "ratio": [integer, integer] | null,     // Phase ratio for hybrid algorithms (optional)
  "objective_bounds": [[min, max], ...],  // Min/max bounds per objective (required)
  "reference_point": [integer, ...],      // Reference point for hypervolume (required)
  "second_phase_start_index": integer | null  // Index where second phase starts (optional)
}
```

**Field Descriptions**:

- **objectives**: Array of objective names corresponding to columns in `objectives.bin`
  - Example: `["min_cost", "cloud_coverage", "min_max_incidence_angle"]`
  - Order must match the objective order in binary data

- **total_duration**: Total optimization runtime in microseconds
  - Type: 64-bit unsigned integer
  - For merged traces: sum of all phase durations
  - Includes all optimization phases

- **algorithm**: Algorithm identifier
  - Single-phase: `"pls"`, `"gpba-a"`, `"aneja-nair"`
  - Multi-phase: `"hybrid"` (for merged traces), `"two-phase-X-Y"` (where X+Y=100)

- **solution_count**: Total number of solutions in the archive
  - Must match the number of entries in binary files
  - For merged traces: sum of all phase solution counts

- **ratio**: Phase allocation for hybrid algorithms (**Added during merge**)
  - Format: `[phase1_percent, phase2_percent]`
  - Example: `[20, 80]` means 20% exact solver, 80% heuristic
  - `null` for single-phase algorithms
  - **Merge behavior**: Extracted from combined algorithm name or defaults to `[50, 50]`

- **objective_bounds**: Min/max bounds for each objective
  - Format: `[[min_0, max_0], [min_1, max_1], ...]`
  - Used for hypervolume normalization
  - Order matches objectives array
  - **Merge behavior**: Provided as parameters to merge operation

- **reference_point**: Hypervolume reference point
  - Values should be ≥ maximum possible objective values
  - Used for hypervolume computation
  - **Merge behavior**: Provided as parameters to merge operation

- **second_phase_start_index**: Solution index where second phase begins (**Added during merge**)
  - `null` for single-phase algorithms
  - For merged traces: equals the solution count of the first phase
  - Used to distinguish phases in analysis and visualization
  - **Merge behavior**: Always set to `first_trace.solution_count`

## Trace Merging Operations

When merging two trace archives using `merge_traces()`, several transformations occur:

### Metadata Transformations
1. **Algorithm name**: Changed to `"hybrid"` regardless of input algorithm names
2. **Total duration**: Sum of both trace durations
3. **Solution count**: Sum of both trace solution counts
4. **Ratio**: Extracted from `combined_algorithm` parameter (e.g., `"two-phase-20-80"` → `[20, 80]`)
5. **Second phase start index**: Set to the solution count of the first trace

### Binary Data Transformations
1. **Objectives**: Direct concatenation of both traces
2. **Dominated**: Direct concatenation of both traces  
3. **Timestamps**: Second trace timestamps offset by first trace's total duration
4. **Hypervolume**: Direct concatenation of both traces

### Timestamp Offset Calculation
```
merged_timestamp[i] = {
  first_trace.timestamp[i]                           if i < first_trace.solution_count
  second_trace.timestamp[j] + first_trace.duration   if i >= first_trace.solution_count
}
where j = i - first_trace.solution_count
```

### Merged Trace Identification
A merged trace can be identified by:
- `algorithm == "hybrid"`
- `ratio != null`
- `second_phase_start_index != null`
- Solution count equals sum of constituent phases

## Data Validation

### Data Integrity
- Hypervolume values must be in range [0.0, 1.0]
- Timestamps must be monotonically non-decreasing
- Domination indices must be < `num_solutions` or equal to `u32::MAX`
- Solution count must match file sizes
- For merged traces: `second_phase_start_index` must be ≤ `solution_count`

## Merged Trace Specific Validations

### Phase Boundary Validation
```python
def validate_merged_trace(metadata, timestamps):
    if metadata.get('second_phase_start_index') is not None:
        phase_boundary = metadata['second_phase_start_index']
        
        # Validate phase boundary is within bounds
        assert 0 <= phase_boundary <= metadata['solution_count']
        
        # Validate timestamp continuity at phase boundary
        if phase_boundary < len(timestamps) - 1:
            phase1_last = timestamps[phase_boundary - 1]
            phase2_first = timestamps[phase_boundary]
            assert phase2_first >= phase1_last, "Phase 2 must start after phase 1"
```

### Ratio Validation
```python
def validate_ratio(metadata):
    ratio = metadata.get('ratio')
    if ratio is not None:
        assert len(ratio) == 2, "Ratio must have exactly 2 elements"
        assert sum(ratio) == 100, "Ratio elements must sum to 100"
        assert all(0 <= x <= 100 for x in ratio), "Ratio values must be 0-100"
```

## Version Compatibility

This specification describes version 1.0 of the trace format. Future versions will maintain backward compatibility for the core binary structure while potentially adding new optional files or metadata fields.

**Identifying Version**:
- Version 1.0: Contains exactly 5 files (objectives.bin, dominated.bin, timestamp.bin, hypervolume.bin, metadata.json)
- No explicit version field in metadata (implied v1.0)

**Identifying Merged Traces**:
- `algorithm == "hybrid"`
- `ratio` field is present and not null
- `second_phase_start_index` field is present and not null

## Implementation Notes

1. **Memory Efficiency**: Binary files can be memory-mapped for large archives
2. **Streaming**: Files can be processed sequentially without loading entire archive
3. **Validation**: Always validate file sizes against metadata before processing
4. **Endianness**: All parsers must handle little-endian byte order correctly
5. **Error Handling**: Malformed archives should fail gracefully with descriptive errors
6. **Merge Operations**: Preserve original trace metadata when possible, only transform necessary fields

## Reference Implementation

The canonical implementation is available in Rust at:
- Repository: `sims-hybrid-algs`
- File: `sims-problem/src/trace.rs`
- Functions: `create_optimization_trace_archive()`, `merge_traces()`, `extract_trace_data()`