# TrackedNdTree Documentation

## Overview

`TrackedNdTree` is an extension of `NDTree` that adds solution index tracking and domination relationship reporting. It's designed for use cases where you need to maintain a history of which solutions dominated others, such as trace generation for multi-objective optimization algorithms.

## Key Features

- **Sequential Index Assignment**: Each accepted solution receives a unique, sequential index (0, 1, 2, ...)
- **Domination Reporting**: Returns a list of dominated solution indices on each insertion
- **Configurable Filtering**: Choose whether to include dominated-at-discovery solutions
- **Complete Tracking**: Maintains mappings between indices and solutions
- **Efficient Lookups**: O(1) lookup of solutions by index or index by objectives

## Filtering Modes

### No-Filtering Mode (`filter_dominated = false`)

**Behavior:**
- **All solutions receive an index**, even if dominated at discovery time
- Dominated solutions are tracked in index maps but removed from the tree
- Useful for complete temporal trace generation

**Use Cases:**
- Generating complete optimization traces with all explored solutions
- Analyzing algorithm behavior including dominated intermediate solutions
- Creating `trace.tar.gz` files with `dominated.bin` information

**Example:**
```rust
use nd_tree::TrackedNdTree;
use nd_tree::nd_tree::Solution;

let mut tree: TrackedNdTree<Solution<2>, 8, 2, 4> =
    TrackedNdTree::new_without_filtering();

// S0: Non-dominated
let r0 = tree.insert(Solution::new([10, 20]));
assert_eq!(r0.assigned_index, Some(0));  // Gets index 0

// S1: Dominates S0
let r1 = tree.insert(Solution::new([5, 15]));
assert_eq!(r1.assigned_index, Some(1));  // Gets index 1
assert_eq!(r1.dominated_indices, vec![0]); // Dominated S0

// S2: Dominated by S1 (but still gets an index!)
let r2 = tree.insert(Solution::new([12, 25]));
assert_eq!(r2.assigned_index, Some(2));  // Gets index 2
assert!(r2.was_dominated_at_discovery);

// All 3 solutions tracked
assert_eq!(tree.num_tracked_solutions(), 3);
// But only 1 non-dominated solution in tree
assert_eq!(tree.num_tree_solutions(), 1);
```

### Filtering Mode (`filter_dominated = true`)

**Behavior:**
- **Only non-dominated-at-discovery solutions receive indices**
- Dominated solutions are rejected immediately (no index assigned)
- Useful for maintaining a pure Pareto front

**Use Cases:**
- Maintaining only the Pareto-optimal set
- Memory-efficient tracking when dominated solutions aren't needed
- Filtering traces to include only meaningful progress

**Example:**
```rust
use nd_tree::TrackedNdTree;
use nd_tree::nd_tree::Solution;

let mut tree: TrackedNdTree<Solution<2>, 8, 2, 4> =
    TrackedNdTree::new_with_filtering();

// S0: Non-dominated
let r0 = tree.insert(Solution::new([10, 20]));
assert_eq!(r0.assigned_index, Some(0));  // Gets index 0

// S1: Dominates S0
let r1 = tree.insert(Solution::new([5, 15]));
assert_eq!(r1.assigned_index, Some(1));  // Gets index 1
assert_eq!(r1.dominated_indices, vec![0]); // Dominated S0

// S2: Dominated by S1 (rejected, no index!)
let r2 = tree.insert(Solution::new([12, 25]));
assert!(!r2.inserted);                   // Not inserted
assert_eq!(r2.assigned_index, None);     // No index assigned

// Only 1 solution tracked (S0 was removed when dominated)
assert_eq!(tree.num_tracked_solutions(), 1);
assert_eq!(tree.num_tree_solutions(), 1);
```

## API Reference

### Construction

```rust
// Create with default config (filtering enabled)
let tree = TrackedNdTree::default();

// Create with custom config
let config = TrackedNdTreeConfig { filter_dominated: false };
let tree = TrackedNdTree::new(config);

// Convenience constructors
let tree_filtering = TrackedNdTree::new_with_filtering();
let tree_no_filtering = TrackedNdTree::new_without_filtering();
```

### InsertionResult

Returned by `insert()`, contains:

```rust
pub struct InsertionResult {
    /// Whether the solution was inserted into the tree
    pub inserted: bool,
    
    /// The index assigned to this solution (None if rejected)
    pub assigned_index: Option<u32>,
    
    /// Indices of solutions this solution dominated
    pub dominated_indices: Vec<u32>,
    
    /// Whether this solution was dominated at discovery
    pub was_dominated_at_discovery: bool,
}
```

### Core Methods

#### `insert(&mut self, solution: T) -> InsertionResult`

Insert a solution and receive detailed insertion information.

**Returns:**
- `InsertionResult` with index assignment and domination information

**Example:**
```rust
let result = tree.insert(solution);
if result.inserted {
    let idx = result.assigned_index.unwrap();
    println!("Solution {} dominated {} others", 
             idx, result.dominated_indices.len());
}
```

#### `get_solution(&self, index: u32) -> Option<&T>`

Retrieve a solution by its index.

**Returns:**
- `Some(&T)` if the index exists and solution is still tracked
- `None` if index invalid or solution was removed (filtering mode)

**Example:**
```rust
if let Some(sol) = tree.get_solution(5) {
    println!("Solution 5: {:?}", sol.objectives());
}
```

#### `get_index(&self, objectives: &[u64; D]) -> Option<u32>`

Find the index of a solution by its objectives.

**Returns:**
- `Some(index)` if a solution with those objectives was tracked
- `None` if no such solution exists

**Example:**
```rust
let objectives = [10, 20];
if let Some(idx) = tree.get_index(&objectives) {
    println!("Solution {:?} has index {}", objectives, idx);
}
```

#### `get_all_tracked(&self) -> Vec<(u32, &T)>`

Get all tracked solutions with their indices, sorted by index.

**Returns:**
- Vector of `(index, solution)` pairs in ascending index order

**Example:**
```rust
for (idx, sol) in tree.get_all_tracked() {
    println!("Solution {}: {:?}", idx, sol.objectives());
}
```

### Query Methods

```rust
// Number of solutions tracked
tree.num_tracked_solutions()  // Includes dominated in no-filtering mode

// Number of solutions in tree
tree.num_tree_solutions()     // Always non-dominated only

// Next index to be assigned
tree.next_index()

// Check if empty
tree.is_empty()

// Iterate over non-dominated solutions
for solution in tree.iter() {
    // Process solution
}

// Get configuration
let config = tree.config();
```

### Utility Methods

```rust
// Clear all data and reset
tree.clear();
```

## Domination Tracking Examples

### Example 1: Building a Trace with Domination History

```rust
use nd_tree::{TrackedNdTree, TrackedNdTreeConfig};
use nd_tree::nd_tree::Solution;

let mut tree: TrackedNdTree<Solution<4>, 8, 4, 4> =
    TrackedNdTree::new_without_filtering();

let mut dominated_by: Vec<Option<u32>> = Vec::new();

// Track discovery order
let solutions = vec![
    [100, 200, 300, 400],
    [90, 180, 290, 380],   // Dominates previous
    [110, 210, 320, 410],  // Dominated by second
];

for sol_obj in solutions {
    let result = tree.insert(Solution::new(sol_obj));
    
    if let Some(idx) = result.assigned_index {
        // Mark which solutions this one dominated
        for &dominated_idx in &result.dominated_indices {
            if (dominated_idx as usize) < dominated_by.len() {
                dominated_by[dominated_idx as usize] = Some(idx);
            }
        }
        
        // Add entry for this solution
        if result.was_dominated_at_discovery {
            // Find who dominated it
            for existing_idx in 0..idx {
                if let Some(existing) = tree.get_solution(existing_idx) {
                    if Solution::new(sol_obj).is_dominated_by(existing.objectives()) {
                        dominated_by.push(Some(existing_idx));
                        break;
                    }
                }
            }
        } else {
            dominated_by.push(None);
        }
    }
}

// dominated_by[i] = Some(j) means solution i is dominated by solution j
// dominated_by[i] = None means solution i is non-dominated
```

### Example 2: Real-time Pareto Front Updates

```rust
use nd_tree::{TrackedNdTree, InsertionResult};
use nd_tree::nd_tree::Solution;

let mut tree: TrackedNdTree<Solution<2>, 8, 2, 4> =
    TrackedNdTree::new_with_filtering();

fn process_new_solution(tree: &mut TrackedNdTree<Solution<2>, 8, 2, 4>, 
                       sol: Solution<2>) {
    let result = tree.insert(sol);
    
    if result.inserted {
        let idx = result.assigned_index.unwrap();
        println!("✓ Solution {} added to Pareto front", idx);
        
        if !result.dominated_indices.is_empty() {
            println!("  Removed {} dominated solutions: {:?}", 
                     result.dominated_indices.len(),
                     result.dominated_indices);
        }
    } else {
        println!("✗ Solution rejected (dominated at discovery)");
    }
}
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `insert()` | O(N·D) | N = solutions in tree, D = dimensions |
| `get_solution()` | O(1) | HashMap lookup |
| `get_index()` | O(1) | HashMap lookup |
| `get_all_tracked()` | O(N log N) | Sorting by index |
| `is_dominated_by_any()` | O(N·D) | Checks all tree solutions |
| `find_dominated_by()` | O(N·D) | Checks all tree solutions |

### Space Complexity

| Mode | Complexity | Storage |
|------|------------|---------|
| Filtering | O(P) | P = Pareto front size |
| No-Filtering | O(N) | N = total solutions inserted |

**Note:** No-filtering mode stores all solutions in the index maps, even dominated ones.

## Comparison: TrackedNdTree vs NDTree

| Feature | NDTree | TrackedNdTree |
|---------|--------|---------------|
| Solution storage | ✓ | ✓ |
| Dominance filtering | ✓ | ✓ |
| Index assignment | ✗ | ✓ |
| Domination reporting | ✗ | ✓ |
| Solution lookup by index | ✗ | ✓ |
| Index lookup by objectives | ✗ | ✓ |
| Dominated solution tracking | ✗ | ✓ (no-filtering mode) |
| Memory overhead | None | 2 HashMaps + counters |

**When to use NDTree:**
- Pure Pareto front maintenance
- No need for historical information
- Minimal memory overhead required

**When to use TrackedNdTree:**
- Generating optimization traces
- Need domination history
- Want to query solutions by index
- Need to track dominated solutions

## Common Patterns

### Pattern 1: Trace Generation

```rust
// Setup for trace generation
let mut tree = TrackedNdTree::new_without_filtering();
let mut timestamps = Vec::new();
let mut objectives = Vec::new();

// During optimization
for solution in discovered_solutions {
    let result = tree.insert(solution.clone());
    
    if let Some(idx) = result.assigned_index {
        timestamps.push(current_time_us());
        objectives.push(solution.objectives().to_vec());
    }
}

// Export to trace format
let all_tracked = tree.get_all_tracked();
// Write objectives.bin, dominated.bin, timestamp.bin, etc.
```

### Pattern 2: Progress Monitoring

```rust
let mut tree = TrackedNdTree::new_with_filtering();
let mut progress_log = Vec::new();

for (iteration, solution) in algorithm_run() {
    let result = tree.insert(solution);
    
    if result.inserted {
        progress_log.push((
            iteration,
            result.assigned_index.unwrap(),
            tree.num_tree_solutions(),
            result.dominated_indices.len(),
        ));
    }
}

// Analyze progress
for (iter, idx, front_size, removed) in progress_log {
    println!("Iter {}: Solution {} | Front size: {} | Removed: {}",
             iter, idx, front_size, removed);
}
```

### Pattern 3: Solution Ancestry Tracking

```rust
let mut tree = TrackedNdTree::new_without_filtering();
let mut domination_graph: Vec<Vec<u32>> = Vec::new();

for solution in solutions {
    let result = tree.insert(solution);
    
    if let Some(idx) = result.assigned_index {
        // Create adjacency list entry for this solution
        domination_graph.push(result.dominated_indices.clone());
    }
}

// domination_graph[i] = list of solutions that solution i dominated
```

## Testing

The implementation includes comprehensive tests covering:

- Sequential index assignment
- Domination reporting in both modes
- Filtering vs non-filtering behavior
- Solution retrieval by index
- Index retrieval by objectives
- Multiple simultaneous dominations
- Clearing and resetting

Run tests with:
```bash
cargo test --lib tracked_nd_tree
```

## Integration with SIMS Project

`TrackedNdTree` is designed to integrate seamlessly with the SIMS hybrid algorithms project:

1. **Trace Generation**: Replace manual domination tracking with `TrackedNdTree` in no-filtering mode
2. **PLS Algorithms**: Use in filtering mode for efficient Pareto front maintenance
3. **Benchmarking**: Compare dominated solution counts across algorithm variants
4. **Analysis**: Export domination graphs for algorithm behavior analysis

## Future Enhancements

Potential improvements:

- [ ] Batch insertion API for better performance
- [ ] Domination graph export utilities
- [ ] Snapshot/restore functionality
- [ ] Statistics collection (avg dominated per insertion, etc.)
- [ ] Integration with trace serialization formats

## License

Part of the SIMS Hybrid Algorithms project.