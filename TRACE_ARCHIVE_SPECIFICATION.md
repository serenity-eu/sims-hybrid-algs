# SIMS Optimization Trace Archive Specification

## Overview

The SIMS optimization trace archive is a compressed binary format (`.tar.gz`) that contains detailed information about the optimization process, including all explored solutions, their objectives, domination relationships, and timing information. This format is designed for efficient storage and analysis of multi-objective optimization traces.

## Archive Format

The trace archive is a **gzip-compressed tar archive** containing four files:

```
trace.tar.gz
├── objectives.bin      # Binary objectives data
├── dominated.bin       # Binary domination relationships  
├── timestamp.bin       # Binary timing data
└── metadata.json       # JSON metadata
```

## File Specifications

### 1. objectives.bin

**Format**: Binary file containing objective values for all explored solutions.

- **Data Type**: `u64` (8 bytes per objective value)
- **Byte Order**: Little Endian (LE)
- **Structure**: Sequential objectives for each solution

```
Solution 1: [obj1_u64_le] [obj2_u64_le] [obj3_u64_le] ...
Solution 2: [obj1_u64_le] [obj2_u64_le] [obj3_u64_le] ...
...
```

**Size Calculation**: `file_size = num_solutions × num_objectives × 8 bytes`

### 2. dominated.bin

**Format**: Binary file containing domination relationships between solutions.

- **Data Type**: `u32` (4 bytes per solution)
- **Byte Order**: Little Endian (LE)
- **Values**: 
  - Index of the first solution that dominates this solution
  - `0xFFFFFFFF` (u32::MAX) if the solution is not dominated

```
[dominator_index_u32_le] [dominator_index_u32_le] [u32_max_le] ...
```

**Size Calculation**: `file_size = num_solutions × 4 bytes`

### 3. timestamp.bin

**Format**: Binary file containing timestamps for when each solution was found.

- **Data Type**: `u32` (4 bytes per solution)
- **Byte Order**: Little Endian (LE)
- **Units**: Microseconds since optimization start
- **Range**: 0 to ~4,294 seconds (truncated from u64 to u32)

```
[timestamp_us_u32_le] [timestamp_us_u32_le] [timestamp_us_u32_le] ...
```

**Size Calculation**: `file_size = num_solutions × 4 bytes`

### 4. metadata.json

**Format**: JSON file containing human-readable metadata about the trace.

```json
{
  "objectives": ["min_cost", "cloud_coverage", "max_incidence_angle"],
  "total_duration": 2000000,
  "algorithm": "PLS-2D",
  "solution_count": 1500
}
```

**Fields**:
- `objectives`: Array of objective names (strings)
- `total_duration`: Total optimization time in microseconds (u64)
- `algorithm`: Name of the optimization algorithm used (string)
- `solution_count`: Number of solutions in the trace (usize)

## Data Ordering

All arrays (objectives, dominated, timestamp) are ordered by solution index:
- Index 0 corresponds to the first solution found
- Solutions are sorted by timestamp (chronological order)
- All binary files have the same number of entries

## Usage Examples

### Python Example

```python
import sims_problem

# Create solutions (example)
solutions = [
    sims_problem.Solution.create([0, 2, 5], 1500, 250, 100000, 45, 800),
    sims_problem.Solution.create([1, 3, 7], 1200, 300, 500000, 40, 900),
]

# Create trace archive
archive_bytes = sims_problem.create_optimization_trace_archive(
    solutions, 
    ["min_cost", "cloud_coverage", "max_incidence_angle"], 
    2000000, 
    "pls"
)

# Save to file
with open("trace.tar.gz", "wb") as f:
    f.write(archive_bytes)
```

### Reading the Archive

```python
import tarfile
import gzip
import json
import struct

# Extract archive
with gzip.open("trace.tar.gz", "rb") as gz_file:
    with tarfile.open(fileobj=gz_file, mode="r") as tar:
        # Read metadata
        metadata_data = tar.extractfile("metadata.json").read()
        metadata = json.loads(metadata_data.decode('utf-8'))
        
        num_objectives = len(metadata["objectives"])
        num_solutions = metadata["solution_count"]
        
        # Read objectives
        objectives_data = tar.extractfile("objectives.bin").read()
        objectives = []
        for i in range(num_solutions):
            solution_objectives = []
            for j in range(num_objectives):
                offset = (i * num_objectives + j) * 8
                value = struct.unpack("<Q", objectives_data[offset:offset+8])[0]
                solution_objectives.append(value)
            objectives.append(solution_objectives)
        
        # Read domination data
        dominated_data = tar.extractfile("dominated.bin").read()
        domination_indices = []
        for i in range(num_solutions):
            offset = i * 4
            value = struct.unpack("<I", dominated_data[offset:offset+4])[0]
            domination_indices.append(None if value == 0xFFFFFFFF else value)
        
        # Read timestamps
        timestamp_data = tar.extractfile("timestamp.bin").read()
        timestamps = []
        for i in range(num_solutions):
            offset = i * 4
            value = struct.unpack("<I", timestamp_data[offset:offset+4])[0]
            timestamps.append(value)
```

## Algorithm Support

The trace format supports multiple optimization algorithms:

- **PLS-2D**: Pareto Local Search for 2D problems
- **PLS-3D**: Pareto Local Search for 3D problems  
- **Hybrid**: Hybrid algorithm combining different approaches
- **MILP**: Mixed Integer Linear Programming approaches

## Storage Efficiency

The binary format provides efficient storage:
- **Objectives**: 8 bytes × num_objectives × num_solutions
- **Domination**: 4 bytes × num_solutions
- **Timestamps**: 4 bytes × num_solutions
- **Metadata**: ~100-500 bytes (JSON)
- **Compression**: Additional ~30-70% size reduction via gzip

## Limitations

1. **Timestamp Precision**: Timestamps are truncated to u32, limiting to ~4,294 seconds
2. **Objective Values**: Limited to u64 range (may require scaling for very large values)
3. **Solution Ordering**: Solutions must be chronologically ordered by timestamp
4. **Memory Requirements**: Full trace stored in memory during creation

## Version History

- **v1.0**: Initial implementation with 4-file archive format