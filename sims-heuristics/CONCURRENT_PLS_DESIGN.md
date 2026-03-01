# Concurrent Pareto Local Search -- Design Document

> Design for parallelizing the PLS implementation in `sims-heuristics` using objective-space decomposition with weight vectors, independent per-region archives, non-blocking snapshot-based synchronization, and lazy tombstones.

## Table of Contents

1. [Design Decisions Summary](#1-design-decisions-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Objective-Space Decomposition](#3-objective-space-decomposition)
4. [Per-Region Worker Architecture](#4-per-region-worker-architecture)
5. [Archive Management & Tombstones](#5-archive-management--tombstones)
6. [Non-Blocking Synchronization](#6-non-blocking-synchronization)
7. [Duplicate Detection](#7-duplicate-detection)
8. [Boundary Handling](#8-boundary-handling)
9. [Initial Population Distribution](#9-initial-population-distribution)
10. [Tracker Allocation](#10-tracker-allocation)
11. [Tracing & Diagnostics](#11-tracing--diagnostics)
12. [Data Structures & Type Signatures](#12-data-structures--type-signatures)
13. [Algorithm Pseudocode](#13-algorithm-pseudocode)
14. [Implementation Plan](#14-implementation-plan)
15. [Risk Analysis](#15-risk-analysis)
16. [Scalarized-Auxiliary PLS (SA-PLS): Deep Region Search](#16-scalarized-auxiliary-pls-sa-pls-deep-region-search)

---

## 1. Design Decisions Summary

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| D1 | Parallelism model | Objective-space decomposition | Each thread owns a region of objective space. Reduces archive contention to zero within regions. |
| D2 | Archive concurrency | Independent per-region archives + periodic merge/redistribute | Each region has its own `NdTreeSolutionSet`. No shared mutable archive state between threads. Simplest correctness model. |
| D3 | Tombstone strategy | Thread-local `HashSet` + periodic rebuild | Mark dominated solutions by adding their fingerprint to a local `HashSet`. Rebuild ND-Tree when tombstone ratio exceeds threshold (30-50%). Minimizes structural churn without atomics. |
| D4 | Exploration semantics | Relaxed (tolerate some duplicate exploration) | Per-thread dedup; no global synchronization on explored set. Matches literature recommendations for parallel PLS. |
| D5 | Decomposition method | Weight-vector based (Das-Dennis scalarization) | Well-studied (MOEA/D, NSGA-III). Controllable region count. No empty regions in 4D. Natural search direction focus. |
| D6 | Worker-to-region mapping | Fixed thread-per-region | Each `std::thread::scope` thread owns one region for the full run. Simplest thread model, allows borrowing problem data. |
| D7 | Synchronization model | Non-blocking background merger with `Arc` snapshots | Workers atomically publish snapshots; background merger reads all, computes global front, publishes back. No thread ever blocks on another. |
| D8 | Duplicate detection | Per-thread `HashMap`, final merge only | Each thread maintains its own `ExploredSolutionsData`. No cross-thread contention on dedup; cross-region dedup is resolved during final archive merge. |
| D9 | Tracker allocation | One tracker per thread, reused within region | `ProvenSafeTrackerArray` is ~KBs. One per thread is negligible memory. Reused across all neighborhood explorations in the region. |
| D10 | Region count | Configurable, matched to thread count | User specifies thread count; Das-Dennis generates matching weight vectors. Default: number of CPU cores. |
| D11 | Concurrency primitives | `std::thread::scope` + `crossbeam` | `std::thread::scope` for outer region threads (allows borrowing). `crossbeam` for channels and epoch-based utilities. Rayon can be added inside regions later for inner parallelism. |
| D12 | Boundary handling | Soft: solutions near boundaries assigned to multiple regions | Solutions within distance threshold of multiple weight vectors are inserted into all matching regions. Improves coverage at moderate duplication cost. |
| D13 | Tracing | Per-region traces + global merger trace | Each region writes its own binary trace. Merger writes global trace. Merged offline for analysis. |
| D14 | Region-constrained search | Scalarized-Auxiliary PLS (SA-PLS): decouple archive and auxiliary acceptance criteria | The **archive** gate remains global Pareto non-dominance (unchanged). The **auxiliary** gate is replaced with a scalarized improvement check: neighbor enters the next-iteration seed population if and only if $g^{atch}(n \mid \mathbf{w}, \mathbf{z}^*) < g^{atch}(\text{parent} \mid \mathbf{w}, \mathbf{z}^*)$. This makes the population evolve in the direction of the weight vector regardless of global Pareto geometry, enabling genuinely deep regional search. |
| D15 | Scalarized population ordering | Population explored in ascending $g^{atch}$ order per step | Before exploring a step's population, sort solutions by their Tchebycheff scalarized score under the region's weight vector. Ensures that when time runs out mid-step, the most regionally relevant solutions have already been explored. O(P log P) per step with typical P = 10–200. |

---

## 2. Architecture Overview

```
                        +------------------+
                        |   Coordinator    |
                        |  (main thread)   |
                        +--------+---------+
                                 |
                    spawns N+1 threads
                 /       |       |       \
        +-------+  +-------+  +-------+  +-----------+
        |Region | |Region | |Region |  | Background |
        |  W_0  | |  W_1  | |  ...  |  |  Merger    |
        +-------+  +-------+  +-------+  +-----------+
            |          |          |             |
         owns:      owns:      owns:        reads:
         - local     - local   - local      - all snapshot
           archive     archive   archive      slots
         - local     - local   - local     writes:
           dedup       dedup     dedup      - global front
           HashMap     HashMap   HashMap      snapshot
         - tracker   - tracker - tracker
         - weight    - weight  - weight
           vector      vector    vector

        Snapshot slots:
        [slot_0..slot_N-1]: Arc<Vec<ObjectiveSnapshot<T, D>>>
        [global_front]: Arc<GlobalFrontSnapshot<T, D>>
             \       |       /                  |
              read by merger              read by workers
```

### Thread Topology

- **N region worker threads** (`std::thread::scope`): Each runs PLS on its assigned region.
- **1 background merger thread**: Periodically reads all region snapshots, computes global Pareto front, publishes it back.
- **Main thread (coordinator)**: Spawns all threads, waits for completion, collects final results.

### Zero-Blocking Guarantee

No thread ever calls `lock()`, `wait()`, or blocks on another thread's progress. All sharing is through:
- Atomic `Arc` pointer swaps (snapshot publication)
- Atomic reads of `Arc` pointers (snapshot consumption)
- `crossbeam::channel` for termination signals only

---

## 3. Objective-Space Decomposition

### Das-Dennis Weight Vector Generation

For $D = 4$ objectives and parameter $H$, the Das-Dennis method generates $\binom{H + D - 1}{D - 1}$ uniformly distributed weight vectors on the unit simplex.

| $H$ | Number of weight vectors | Good for |
|-----|--------------------------|----------|
| 1   | 4                        | 4-core machines, debugging |
| 2   | 10                       | 8-12 core machines |
| 3   | 20                       | 16-24 core machines |
| 4   | 35                       | 32+ core machines |

**Algorithm**: Generate all non-negative integer solutions to $w_1 + w_2 + w_3 + w_4 = H$, then normalize each to sum to 1.0.

**Auto-selecting H from thread count**: When `das_dennis_h` is `None`, find the smallest $H$ such that $\binom{H + D - 1}{D - 1} \geq \text{num\_threads}$. For $D = 4$: $H=1 \to 4$, $H=2 \to 10$, $H=3 \to 20$, $H=4 \to 35$. The Das-Dennis grid may have more weight vectors than threads; extra workers are omitted starting from the most interior weight vectors.

### Solution-to-Region Assignment

A solution $s$ with objectives $\mathbf{f}(s) = (f_1, f_2, f_3, f_4)$ is assigned to the region whose weight vector $\mathbf{w}$ minimizes the **augmented Tchebycheff scalarization**:

$$g^{atch}(s \mid \mathbf{w}, \mathbf{z}^*) = \max_{j=1..D} \left\{ w_j \cdot \frac{\max(0, f_j(s) - z_j^*)}{\Delta_j} \right\} + \rho \sum_{j=1}^{D} w_j \cdot \frac{\max(0, f_j(s) - z_j^*)}{\Delta_j}$$

where:
- $\mathbf{z}^* = (z_1^*, \ldots, z_D^*)$ is the ideal point (best known value per objective)
- $\Delta_j = \max_j - \min_j$ is the objective range used for normalization
- $\rho$ is a small augmentation coefficient (default: $10^{-3}$, see Appendix B for rationale)

**Why augmented Tchebycheff**: Plain weighted sum cannot find solutions in non-convex regions of the Pareto front. Tchebycheff decomposition covers the entire front uniformly.

### Region Data Structure

```rust
struct Region<const D: usize> {
    /// Index of this region (0..N)
    index: usize,
    /// Weight vector for this region (sums to 1.0)
    weight_vector: [f64; D],
    /// Normalization bounds: (min, max) per objective.
    /// These are problem-level bounds (e.g., minimum/maximum possible cost, cloud coverage),
    /// obtained from `problem.objective_bounds()` before distributing the initial population.
    /// They are the same for all regions and do not change during the run.
    objective_bounds: [(f64, f64); D],
}
```

---

## 4. Per-Region Worker Architecture

Each region worker is essentially a **complete sequential PLS instance** with two added operations:

1. **Publish**: Periodically swap the current archive snapshot into the shared slot.
2. **Prune**: Periodically read the global front snapshot and tombstone locally dominated solutions.

### Worker State

> **Implementation note**: In the code, all PLS state (archive, population,
> explored solutions, tracker, neighborhood structure) is encapsulated in a single
> `pls: ParetoLocalSearch<T, P, D>` field, accessed through methods like
> `self.pls.archive()` and `self.pls.explored_solutions_data()`. The flat fields
> shown below are for conceptual clarity.

```rust
struct RegionWorker<'a, T, S, P, const D: usize> {
    region: Region<D>,
    
    // --- PLS state (same as sequential) ---
    problem: &'a P,
    population: S,                           // unexplored solutions in this region
    local_archive: S,                        // NdTreeSolutionSet for this region
    explored_solutions: ExploredSolutionsData<D>,
    spare_tracker: T::Trackers,
    neighborhood_structure: u32,
    neighborhood_size_range: RangeInclusive<u32>,
    ideal_point: [f64; D],
    is_deterministic: bool,
    /// Fingerprints of solutions in local_archive dominated by the global front.
    tombstones: HashSet<u64>,
    
    // --- Concurrency state ---
    /// All regions in the decomposition; needed for belongs_to_region() during solution adoption.
    all_regions: &'a [Region<D>],
    /// Slot where this worker publishes its archive snapshot
    snapshot_slot: Arc<ArcSwap<Vec<ObjectiveSnapshot<T, D>>>>,
    /// Slot where background merger publishes the global front (as an NDTree for fast pruning)
    global_front_slot: Arc<ArcSwap<GlobalFrontSnapshot<T, D>>>,
    /// Dropped when this worker exits, signaling the merger that this region is done.
    /// The merger holds the Receiver end and detects Disconnected when all workers exit.
    _done_guard: crossbeam::channel::Sender<()>,
    
    // --- Configuration ---
    /// How often to publish/prune (every K PLS steps)
    sync_interval: usize,
    /// Tombstone rebuild threshold (fraction of tombstones)
    tombstone_rebuild_threshold: f64,
    
    // --- Trace ---
    region_trace: RegionTraceWriter,
}
```

### Worker Main Loop

```
fn run(max_iterations, max_duration):
    timer = Timer::start(max_duration)
    
    for iteration in 1..=max_iterations:
        step_status = step(iteration, timer)
        
        // Periodic snapshot publish (non-blocking)
        if iteration % sync_interval == 0:
            publish_snapshot()
            prune_from_global_front()       // tombstone locally dominated solutions
            adopt_from_global_front()       // insert new cross-region solutions into local archive
        
        // Check termination
        if step_status == AllExplored || timer.is_expired():
            break
    
    // Final publish
    publish_snapshot()
    return local_archive
```

The `step()` method is identical to `ParetoLocalSearch::step()` but operates on the region's local archive and population.

---

## 5. Archive Management & Tombstones

### Tombstone Mechanism

Instead of wrapping solutions in atomic flags, each worker maintains a thread-local `HashSet` of tombstoned solution fingerprints. Since only the worker thread mutates its local archive, no atomics are needed.

```rust
struct RegionWorker<...> {
    // ...
    /// Fingerprints of solutions in local_archive that are dominated by the global front
    tombstones: HashSet<u64>,
}
```

**Tombstoning** (marking dominated solutions):
- When the global front indicates a local solution is dominated, insert its fingerprint into `tombstones`.
- Tombstoned solutions are skipped during iteration but remain in the tree structure.
- ND-Tree node bounding boxes are NOT updated on tombstone (would require tree restructuring).

**Rebuild trigger**: When `tombstones.len() / total_count > rebuild_threshold`:

```
fn rebuild_archive(archive, tombstones):
    live_solutions = archive.iter().filter(|s| !tombstones.contains(&hash(s))).collect()
    new_archive = NdTreeSolutionSet::from_iter(live_solutions)
    replace archive with new_archive
    tombstones.clear()
```

Rebuild happens **within the worker thread** (no cross-thread coordination needed since each thread owns its archive).

### Archive Operations Summary

| Operation | Frequency | Blocking? |
|-----------|-----------|-----------|
| `try_insert(neighbor)` | Every neighbor evaluation | No (thread-local) |
| `tombstone(solution)` | During prune-from-global | No (HashSet insert) |
| `rebuild()` | When threshold exceeded | No (thread-local) |
| `publish_snapshot()` | Every K iterations | No (Arc swap) |

---

## 6. Non-Blocking Synchronization

### Snapshot Publication

Each worker publishes its archive as a lightweight snapshot. To allow other workers to adopt globally non-dominated solutions, the snapshot must include the solution representation (e.g., `selected_images`).

```rust
/// Compact snapshot of a solution for cross-region sharing
#[derive(Clone)]
pub struct ObjectiveSnapshot<T, const D: usize> {
    pub objectives: [u64; D],
    pub fingerprint: u64,
    /// The actual solution representation (e.g., FixedBitSet)
    pub solution: T,
}
```

**Publish operation** (worker side):

```rust
fn publish_snapshot(&self) {
    let snapshot: Vec<ObjectiveSnapshot<T, D>> = self.local_archive
        .iter()
        .filter(|s| !self.tombstones.contains(&hash(s)))
        .map(|s| ObjectiveSnapshot {
            objectives: s.objectives(),
            fingerprint: hash(s),
            solution: s.clone(), // BitsetEncodedSolution clone is cheap
        })
        .collect();
    
    // Atomic pointer swap -- O(1), non-blocking
    self.snapshot_slot.store(Arc::new(snapshot));
}
```

Uses `arc-swap` crate for lock-free `Arc` pointer swaps.

### Background Merger Thread

```
// done_rx is disconnected once all worker _done_guard Senders are dropped.
fn merger_loop(snapshot_slots, global_front_slot, done_rx, merge_interval):
    loop:
        sleep(merge_interval)  // e.g., 50-200ms

        // Read all region snapshots (non-blocking Arc loads)
        all_snapshots = []
        for slot in snapshot_slots:
            snapshot = slot.load()
            all_snapshots.extend(snapshot.iter().cloned())

        // Dedup by fingerprint: a solution published by multiple regions is kept once
        by_fingerprint = HashMap from all_snapshots keyed by .fingerprint (last-writer wins)

        // Compute global Pareto front via NdTree.
        // from_iter uses try_insert internally (non-dominated insertion),
        // so globally dominated solutions are dropped from the tree.
        global_front = NdTreeSolutionSet::from_iter(by_fingerprint.into_values())
        ideal_point = compute_ideal_from_front(&global_front)

        // Publish global front snapshot (atomic swap)
        global_front_slot.store(Arc::new(GlobalFrontSnapshot {
            front: global_front,
            ideal_point,
        }))

        // Check if all workers are done: all Senders dropped → Disconnected
        if done_rx.try_recv() == Err(Disconnected):
            break

    // Final merge
    compute_and_publish_final_front()
```

### Prune-from-Global Operation (Worker Side)

```rust
fn prune_from_global_front(&mut self) {
    let global_snapshot = self.global_front_slot.load();
    let global_front = &global_snapshot.front;
    // Note: ideal point is read fresh from the snapshot each time adopt_from_global_front
    // runs; no persistent local ideal is maintained until Phase 5 (SA-PLS).
    
    for solution in self.local_archive.iter() {
        let fp = hash(solution);
        if self.tombstones.contains(&fp) {
            continue;
        }
        // O(|front|) dominance check: iterate global ND-Tree members.
        // A tree-based dominance query could reduce this to O(log |front|)
        // by pruning subtrees whose bounding boxes cannot dominate the candidate;
        // this is a future optimization.
        if is_dominated_by_front(solution.objectives(), global_front) {
            self.tombstones.insert(fp);
        }
    }
    
    // Check rebuild threshold
    if self.tombstones.len() as f64 / self.local_archive.len() as f64
        > self.tombstone_rebuild_threshold
    {
        self.rebuild_archive();
    }
}
```

### Timing & Frequency

The background merger runs on a configurable interval. Recommended defaults:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `merge_interval` | 100ms | How often the merger checks snapshots |
| `worker_sync_interval` | 5 PLS steps | How often workers publish + prune |
| `tombstone_rebuild_threshold` | 0.4 (40%) | Fraction of tombstones triggering rebuild |

---

## 7. Duplicate Detection

### Per-Thread HashMap

Each worker maintains its own `ExploredSolutionsData<D>`:

```rust
pub struct ExploredSolutionsData<const D: usize> {
    pub solutions: HashMap<u64, SolutionFingerprint<D>>,
    // ...
}
```

- **No cross-thread sharing** of explored sets during execution.
- A solution explored by worker W_0 may be re-explored by worker W_1 -- this is acceptable under relaxed semantics (D4).
- The background merger does NOT merge explored sets (would require synchronization).

### Dedup at Final Merge

When the coordinator collects final results from all regions, it performs a single dedup pass:

```
final_front = NdTreeSolutionSet::new()
for region in all_regions:
    for solution in region.local_archive:
        if !region.tombstones.contains(&hash(solution)):
            final_front.try_insert(solution)
```

This naturally eliminates duplicates and dominated solutions across regions.

---

## 8. Boundary Handling

### Soft Assignment

Solutions near region boundaries are assigned to **all nearby regions** to avoid missing promising neighbors:

```rust
fn assign_to_regions(solution: &T, regions: &[Region<D>], threshold: f64) -> Vec<usize> {
    let mut scores: Vec<(usize, f64)> = regions.iter()
        .enumerate()
        .map(|(i, r)| (i, tchebycheff_score(solution, r)))
        .collect();
    
    scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    let best_score = scores[0].1;
    
    // Edge case: if best score is ~0, all regions are equidistant; assign to all
    if best_score.abs() < f64::EPSILON {
        return (0..regions.len()).collect();
    }
    
    // Assign to all regions within threshold of best (inclusive)
    scores.iter()
        .take_while(|(_, score)| (score - best_score) / best_score <= threshold)
        .map(|(idx, _)| *idx)
        .collect()
}
```

**`threshold` parameter**: controls how aggressively solutions are shared across regions.
- `0.0` = hard assignment (each solution to exactly one region)
- `0.1` = soft (solutions within 10% of best score go to multiple regions)
- Recommended default: `0.05` (5%)

### `belongs_to_region` Predicate

`belongs_to_region` is used during solution adoption to avoid inserting cross-region solutions
into the wrong worker's archive. A solution `g` belongs to region `i` if region `i`'s weight
vector yields the minimum Tchebycheff score for `g` among all regions (ties included):

```rust
fn belongs_to_region<T, const D: usize>(
    g: &ObjectiveSnapshot<T, D>,
    region: &Region<D>,
    all_regions: &[Region<D>],
    ideal: &[f64; D],
) -> bool {
    let my_score = tchebycheff_score(&g.objectives, &region.weight_vector, ideal,
                                     &region.objective_bounds, RHO);
    // Epsilon relaxation: prevents floating-point ties from incorrectly rejecting
    // a solution as not belonging to its own region.
    all_regions.iter().all(|r| {
        tchebycheff_score(&g.objectives, &r.weight_vector, ideal,
                          &r.objective_bounds, RHO) >= my_score - f64::EPSILON
    })
}
```

This means each globally non-dominated solution is adopted by exactly one region
(or shared across tied regions at boundaries).

### Boundary Solutions in Practice

With 4 SIMS objectives, boundary duplication is expected to be modest:
- Most solutions are clearly closest to one weight vector
- Only solutions in "transition zones" between adjacent weight vectors get duplicated
- Duplication provides redundancy that improves coverage quality

---

## 9. Initial Population Distribution

### Strategy

1. For each solution in the initial population, compute its region assignment(s) using soft boundary assignment.
2. Insert each solution into the corresponding region worker's initial population.
3. For regions that receive zero solutions, seed them by cycling through the initial population (round-robin). Note: `random_feasible()` does not currently exist in the codebase; the implementation reuses existing feasible solutions from the initial population. If needed in the future, a random feasible generator can be added as a trait method.

```rust
fn distribute_initial_population(
    initial_pop: &[T],
    regions: &[Region<D>],
    boundary_threshold: f64,
) -> Vec<Vec<T>> {
    let mut region_populations: Vec<Vec<T>> = vec![vec![]; regions.len()];
    
    for solution in initial_pop {
        let assigned = assign_to_regions(solution, regions, boundary_threshold);
        for region_idx in assigned {
            region_populations[region_idx].push(solution.clone());
        }
    }
    
    // Seed empty regions by cycling through initial population
    for (i, pop) in region_populations.iter_mut().enumerate() {
        if pop.is_empty() {
            let seed = &initial_pop[i % initial_pop.len()];
            pop.push(seed.clone());
        }
    }
    
    region_populations
}
```

### Ideal Point Initialization

The ideal point $\mathbf{z}^*$ (needed for Tchebycheff scalarization) is initialized from the initial population:

$$z_j^* = \min_{s \in \text{initial\_pop}} f_j(s) \quad \text{for each objective } j$$

Updated whenever a new best value is discovered by any worker (via global front snapshot).

---

## 10. Tracker Allocation

### One Tracker Per Thread

Each `RegionWorker` owns exactly one `ProvenSafeTrackerArray<D>` instance:

```rust
struct RegionWorker<...> {
    spare_tracker: T::Trackers,
    // ...
}
```

The tracker is reused across all neighborhood explorations within the region, exactly as in the sequential PLS where `spare_tracker` is a field of `ParetoLocalSearch`.

**Memory cost**: Each `ProvenSafeTrackerArray<D>` is ~4-16KB depending on problem size (CSR arrays, interval structures, packed counts). With N=20 threads, total tracker memory is ~80-320KB -- negligible.

**No change** from the sequential implementation's tracker usage pattern: initialize once, reuse for every `neighborhood_iter()` call.

---

## 11. Tracing & Diagnostics

> **Status**: Binary trace files are not yet implemented. Per-region stats are collected
> in `RegionStats` and logged via `tracing::info!` in the orchestrator. Binary trace files
> following the format below are deferred to Phase 4.

### Per-Region Trace Files

Each region writes its own binary trace following the existing `TRACE_SPECIFICATION.md` format:

```
trace_region_{region_index}.tar.gz
    objectives.bin     -- archive objectives over time
    dominated.bin      -- dominated solution counts
    timestamp.bin      -- wall-clock timestamps
    hypervolume.bin    -- hypervolume indicator
```

### Global Merger Trace

The background merger writes a global trace representing the merged Pareto front:

```
trace_global.tar.gz
    objectives.bin     -- global front objectives at each merge
    timestamp.bin      -- merge timestamps
    hypervolume.bin    -- global hypervolume
    region_sizes.bin   -- per-region archive sizes at each merge
```

### Diagnostic Metrics

Each worker periodically logs (via `tracing`):

| Metric | Description |
|--------|-------------|
| `region_archive_size` | Number of live solutions in local archive |
| `tombstone_count` | Number of tombstoned solutions |
| `tombstone_ratio` | `tombstone_count / total` |
| `neighbors_explored` | Total neighbors evaluated |
| `duplicates_skipped` | Neighbors rejected by dedup |
| `global_prune_count` | Solutions tombstoned from global front info |
| `rebuild_count` | Number of archive rebuilds |

---

## 12. Data Structures & Type Signatures

### New Types

```rust
/// Weight vector decomposition of objective space.
/// Implementation note: this type is not materialized as a struct in the code;
/// `build_regions()` returns `Vec<Region<D>>` directly and the boundary threshold
/// is stored in `ConcurrentPLSConfig`.
pub struct WeightVectorDecomposition<const D: usize> {
    regions: Vec<Region<D>>,
    boundary_threshold: f64,
}

/// Compact objective snapshot for cross-region sharing.
/// Implements `ImageSet<D>` (delegating to inner solution), `HasObjectives<D>`,
/// `MoSolution<D>`, `PartialEq` (by fingerprint+objectives), `Debug`, and `Clone`,
/// so it can be stored directly in `NdTreeSolutionSet`.
#[derive(Clone)]
pub struct ObjectiveSnapshot<T, const D: usize> {
    pub objectives: [u64; D],
    pub fingerprint: u64,
    pub solution: T,
}

pub struct GlobalFrontSnapshot<T, const D: usize>
where T: Clone + ImageSet<D>
{
    pub front: NdTreeSolutionSet<ObjectiveSnapshot<T, D>, D>,
    pub ideal_point: [f64; D],
}

/// Configuration for concurrent PLS
pub struct ConcurrentPLSConfig {
    pub num_threads: usize,
    pub sync_interval_steps: usize,
    pub merge_interval: Duration,
    pub tombstone_rebuild_threshold: f64,
    pub boundary_threshold: f64,
    pub das_dennis_h: Option<usize>,  // if None, find smallest H s.t. C(H+D-1, D-1) >= num_threads
    pub max_iterations: usize,
    pub max_duration: Duration,
    pub neighborhood_size_range: RangeInclusive<u32>,
    pub is_deterministic: bool,
}

/// Builder for concurrent PLS. Construct with `ConcurrentPLS::new(problem, config)`,
/// then call `.solve(initial_pop)` to spawn all threads and return the final Pareto front.
/// Implementation note: `S` is hardcoded to `NdTreeSolutionSet` internally;
/// the type parameter is omitted to keep the API simpler.
pub struct ConcurrentPLS<'prob, T, P, const D: usize> {
    config: ConcurrentPLSConfig,
    problem: &'prob P,
    _phantom: std::marker::PhantomData<T>,
}

/// Result from a single region worker.
/// `explored_solutions` is intentionally omitted: per-thread dedup data is not useful
/// after the run; final dedup is via NDTree `try_insert` during the global merge.
pub struct RegionResult<T, const D: usize> {
    pub region_index: usize,
    pub weight_vector: [f64; D],
    pub archive: NdTreeSolutionSet<T, D>,
    pub tombstones: HashSet<u64>,
    pub stats: RegionStats,
}

/// Per-region statistics (matches implementation in worker.rs)
pub struct RegionStats {
    pub iterations_completed: usize,
    pub snapshots_published: usize,
    pub tombstones_added: usize,
    pub archive_rebuilds: usize,
    pub solutions_adopted: usize,
    pub wall_time: Duration,
    pub initial_pop_size: usize,
    pub final_archive_size: usize,
    pub out_of_region_count: usize,
    // Phase 5 additions:
    pub scalarized_exhausted_at_iteration: Option<usize>,
}
```

### Dependencies to Add

```toml
[dependencies]
# Existing deps unchanged...
crossbeam-channel = { version = "0.5", optional = true }  # Done-signal channels
arc-swap = { version = "1.7", optional = true }             # Lock-free Arc pointer swaps
num_cpus = { version = "1.16", optional = true }            # Auto-detect thread count
```

### Feature Gate

```toml
[features]
default = ["bitmaps", "plotting", "bounds_check"]
parallel = ["crossbeam-channel", "arc-swap", "num_cpus", "bitmaps"]  # NOT in default
```

---

## 13. Algorithm Pseudocode

### Top-Level Entry Point

```
fn solve_concurrent_pls(problem, initial_pop, config) -> ParetoFront:
    // 1. Generate weight vectors and distribute initial population
    decomposition = das_dennis(config.num_threads, D=4)
    region_pops = distribute_initial_population(initial_pop, decomposition, problem)

    // 2. Create shared snapshot slots
    snapshot_slots = [ArcSwap::new(Arc::new(vec![])); config.num_threads]
    global_front_slot = ArcSwap::new(Arc::new(GlobalFrontSnapshot {
        front: NdTreeSolutionSet::new(),
        ideal_point: compute_ideal(initial_pop),
    }))

    region_results = []
    std::thread::scope(|scope| {
        // 3. Create done-channel: workers hold Sender clones; merger holds Receiver.
        //    When all workers finish and their Senders are dropped, merger sees Disconnected.
        let (done_tx, done_rx) = crossbeam::channel::bounded::<()>(0)

        // 3b. Spawn merger thread
        merger_handle = scope.spawn(|| {
            merger_loop(snapshot_slots, global_front_slot, done_rx, config.merge_interval)
        })

        // 4. Spawn region worker threads, each with a done_tx clone
        worker_handles = []
        for i in 0..config.num_threads:
            let worker_done_tx = done_tx.clone()
            worker_handles.push(scope.spawn(|| {
                region_worker_loop(
                    region = decomposition.regions[i],
                    all_regions = &decomposition.regions,  // needed for belongs_to_region
                    initial_pop = region_pops[i],
                    problem,
                    snapshot_slot = snapshot_slots[i],
                    global_front_slot,
                    _done_guard = worker_done_tx,  // dropped on worker exit
                    config,
                )
            }))

        // 5. Collect worker results
        for handle in worker_handles:
            region_results.push(handle.join())

        // 6. Drop the coordinator's own done_tx so done_rx disconnects and merger exits
        drop(done_tx)
        merger_handle.join()
    })

    // 7. Final global merge and dedup
    final_front = NdTreeSolutionSet::new()
    for result in region_results:
        for solution in result.archive:
            if !result.tombstones.contains(&hash(solution)):
                final_front.try_insert(solution)

    return final_front
```

### Region Worker Loop (Detail)

```
fn region_worker_loop(region, all_regions, initial_pop, problem, snapshot_slot, global_front_slot, _done_guard, config):
    // Initialize local PLS state
    local_archive = NdTreeSolutionSet::from_iter(initial_pop)
    population = local_archive.clone()
    explored = ExploredSolutionsData::new()
    tracker = ProvenSafeTrackerArray::new(problem)
    k = config.neighborhood_size_range.start()
    timer = Timer::start(config.max_duration)
    tombstones = HashSet::new()
    stats = RegionStats::default()
    
    for iteration in 1..=config.max_iterations:
        // --- Standard PLS step ---
        auxiliary = NdTreeSolutionSet::new()
        
        shuffle(population)
        for solution in population:
            for neighbor in solution.neighborhood_iter(tracker, k, problem, timer):
                if explored.is_registered(neighbor): continue
                explored.register(neighbor)
                
                if neighbor.dominated_by(solution): continue
                
                if local_archive.try_insert(neighbor):
                    auxiliary.try_insert(neighbor)
                    
            if timer.is_expired(): break
        
        // --- PLS step determination ---
        if auxiliary.is_empty():
            if can_increase_k(k, config):
                k += 1
                population = eligible_solutions(local_archive, explored, k)
            else:
                break  // all explored
        else:
            population = auxiliary
            k = config.neighborhood_size_range.start()
        
        // --- Periodic sync (non-blocking) ---
        if iteration % config.sync_interval_steps == 0:
            // Publish local archive snapshot
            snapshot = local_archive.iter()
                .filter(|s| !tombstones.contains(&hash(s)))
                .map(to_objective_snapshot)
                .collect()
            snapshot_slot.store(Arc::new(snapshot))
            
            // Read global front and prune locally
            global_snapshot = global_front_slot.load()
            global_front = &global_snapshot.front
            
            for solution in local_archive.iter():
                fp = hash(solution)
                if tombstones.contains(&fp): continue
                
                if is_dominated_by_front(solution.objectives(), global_front):
                    tombstones.insert(fp)
            
            // Rebuild if needed
            if tombstones.len() as f64 / local_archive.len() as f64 > config.tombstone_rebuild_threshold:
                local_archive.rebuild(tombstones)
            
            // Adopt globally non-dominated solutions that belong to this region.
            // g.solution is T (EncodedSolution) which provides its own objectives;
            // belongs_to_region checks that this region's weight vector gives the
            // minimum Tchebycheff score for g among all regions (see §8).
            for g in global_front.iter():
                if belongs_to_region(&g, &region, all_regions, &global_snapshot.ideal_point) and not explored.is_registered(&g.solution):
                    local_archive.try_insert(g.solution.clone())
                    population.try_insert(g.solution.clone())
        
        if timer.is_expired(): break
    
    // Final publish
    publish_final_snapshot()
    
    return RegionResult {
        archive: local_archive,
        explored_solutions: explored,
        tombstones,
        stats,
    }
```

---

## 14. Implementation Plan

### Phase 1: Foundation (no parallelism yet) -- COMPLETE

Implemented in `concurrent_pls/decomposition.rs` and `concurrent_pls/snapshot.rs`.

1. **Weight vector generation**: Das-Dennis simplex lattice for D=4. (`das_dennis_weight_vectors`, `auto_select_h`)
2. **Tchebycheff scalarization**: Solution-to-region assignment with soft boundaries. (`tchebycheff_score`, `assign_to_regions`)
3. **Thread-local tombstone set**: Fingerprint-based tombstone tracking and rebuild trigger logic. (in `RegionWorker`)
4. **Archive rebuild**: ND-Tree rebuild from live solutions. (`ParetoLocalSearch::rebuild_without_tombstones`)
5. **`ObjectiveSnapshot<T, D>`**: Snapshot including objectives + fingerprint + solution payload + NDTree trait impls. (in `snapshot.rs`)
6. **Unit tests**: Decomposition, assignment, tombstone mechanics verified.

### Phase 2: Single-Threaded Concurrent Architecture -- COMPLETE

Implemented in `concurrent_pls/worker.rs` and `concurrent_pls/config.rs`.

7. **`RegionWorker`**: Wraps `ParetoLocalSearch` with sync/prune/adopt operations.
8. **`ConcurrentPLSConfig`**: Configuration struct with sensible defaults.
9. **Snapshot publish/consume**: Implemented with `ArcSwap`.
10. **Background merger logic**: Merges snapshots into a global `NdTreeSolutionSet`-based front.
11. **Integration test**: Concurrent PLS runs correctly on single and multi-thread configurations.

### Phase 3: Multi-Threaded Execution -- COMPLETE

Implemented in `concurrent_pls/orchestrator.rs`.

12. **Thread spawning**: `std::thread::scope` for N workers + 1 merger.
13. **`ConcurrentPLS` orchestrator**: Top-level spawn/join/merge with detailed per-region diagnostics.
14. **Termination coordination**: Timer-based + done channels via `crossbeam_channel`.
15. **Prune-from-global**: Workers read global front and tombstone dominated local solutions.
16. **Stress testing**: Validated on size-100 instances (5 cities, 10 threads, 120s). Concurrent PLS finds +36% more non-dominated solutions than sequential.

### Phase 4: Integration & Optimization

17. **PyO3 bindings**: Expose `solve_with_concurrent_pls()` in `sims-problem`.
18. **Benchmark suite**: Compare concurrent vs sequential on standard instances.
19. **Tuning**: Optimize sync intervals, tombstone thresholds, boundary thresholds.
20. **Trace analysis**: Verify per-region + global trace files.

### Phase 5: Scalarized-Auxiliary PLS (SA-PLS)

21. **Add `step_scalarized` to `ParetoLocalSearch`**: New method that (a) sorts the population by $g^{atch}(\cdot, \mathbf{w}, \mathbf{z}^*)$ before iterating, and (b) gates the auxiliary by scalarized improvement over the parent instead of archive Pareto-dominance. The archive (`approximated_pareto_set`) insertion path is unchanged. See §16 for full specification.
22. **Add `local_ideal` field to `RegionWorker`**: Initialized from the initial population's component-wise objective minimum, updated per accepted neighbor (O(D)) and on every sync event (component-wise min with global ideal).
23. **Add `RegionSearchMode` enum to `ConcurrentPLSConfig`**: `Unconstrained` (Phase 3 behaviour, **default** for backward compatibility), `ScalarizedAuxiliary` (SA-PLS, D14), `ScalarizedAuxiliaryWithFallback` (SA-PLS + standard PLS fallback after regional exhaustion).
24. **Update adoption gate**: When adopting solutions from the global front, accept only if the candidate also improves `g_atch` relative to the worker's current best scalarized solution. Prevents regional pollution from cross-worker solutions.
25. **Benchmark SA-PLS vs Phase 3**: Run `--region-search-mode scalarized-auxiliary` and `--region-search-mode scalarized-auxiliary-with-fallback` on size-100 instances (120s, 10 threads). Compare: in-region%, final front size, per-region archive size distribution, and time-to-regional-convergence.
26. **Tune `scalarized_rho`**: Benchmark $\rho \in \{10^{-4}, 10^{-3}, 10^{-2}\}$. The rho parameter controls how much the augmentation term breaks ties -- too small allows cycling, too large distorts the true Tchebycheff landscape. The default $10^{-3}$ targets integer-objective granularity after normalization.

---

## 15. Risk Analysis

### R1: Load Imbalance

**Risk**: Some regions may have many more solutions than others, causing idle threads.

**Mitigation**: Soft boundary assignment spreads solutions. Monitor per-region sizes via diagnostics. Future work: dynamic region rebalancing or work stealing.

### R2: Stale Global Front

**Risk**: Workers may explore solutions that the global front has already superseded.

**Mitigation**: This is expected under relaxed semantics. More frequent sync reduces staleness. The 100ms merger interval keeps the global front reasonably fresh.

### R3: Tombstone Overhead

**Risk**: High tombstone ratios degrade ND-Tree query performance (scanning dead nodes).

**Mitigation**: Configurable rebuild threshold (default 40%). Rebuild is thread-local and fast (just re-insert live solutions into fresh tree).

### R4: Memory Overhead from Snapshots

**Risk**: Snapshot `Vec<ObjectiveSnapshot>` copies consume memory.

**Mitigation**: Snapshot size now includes solution payload (`T`), so memory scales with representation size. Use configurable publish cadence and optionally `Arc<T>` payloads to cap copy overhead when archive size is large.

### R5: Determinism

**Risk**: Concurrent PLS is inherently non-deterministic (thread scheduling varies).

**Mitigation**: Accept non-determinism for concurrent mode. Keep sequential PLS available with `is_deterministic` flag. Concurrent mode does not claim reproducibility.

### R6: Boundary Solution Quality

**Risk**: Soft boundary assignment may cause important "boundary" solutions to be inadequately explored.

**Mitigation**: Boundary threshold parameter (default 5%) is tunable. Solutions in boundary zones get explored by multiple regions, providing redundancy.

### R7: Regional Exhaustion under SA-PLS

**Risk**: In SA-PLS, a worker's scalarized auxiliary empties out when no neighbor of any population member improves $g^{atch}(\cdot, \mathbf{w}, \mathbf{z}^*)$ relative to its parent. For regions with very flat Tchebycheff landscapes (many equally-scalarized solutions), this may happen quickly, leaving the CPU idle for most of the time budget.

**Mitigation**: `ScalarizedAuxiliaryWithFallback` mode: once scalarized search is regionally exhausted (k reaches maximum with empty auxiliary), the worker switches to standard Pareto-dominance PLS from its current seed. The starting point at this switch is far more specialized than a random restart (it is a local optimum in the scalarized sense), so the fallback phase still provides value in filling the global archive. Regional exhaustion frequency is tracked in `RegionStats` (`scalarized_exhausted_at_iteration`).

### R8: Stale Ideal Point Distorts Tchebycheff Scoring

**Risk**: The ideal point $\mathbf{z}^*$ shifts as new solutions are found globally. A stale local ideal causes SA-PLS to compute incorrect $g^{atch}$ values: neighbors that genuinely improve the scalarized score may be rejected, or vice versa.

**Mitigation**: Workers update $\mathbf{z}^*_j = \min(\mathbf{z}^*_j, f_j(n))$ for every accepted neighbor, at O(D) cost with no synchronization. On each sync event, additionally take the component-wise minimum with the global ideal from the shared snapshot. This bounds staleness to at most `sync_interval_steps` iterations of local-only updates, while keeping the within-step ideal fully current.

### R9: SA-PLS Auxiliary Unbounded Growth

**Risk**: In SA-PLS, each step iterates over all population members and generates up to $N_k$ neighbors per member. Multiple parents may produce overlapping neighbors that all pass the scalarized improvement test, causing the auxiliary to grow to O(P * Nk) per step. This is orders of magnitude larger than the typical Pareto archive and wastes memory and sorting time.

**Mitigation**: Apply a dominance-filtering post-pass on auxiliary candidates at the end of each step (insert all into a temporary NDTree, retain only non-dominated). Alternatively, cap the auxiliary at top-K by scalarized score. Track auxiliary size in `RegionStats` to detect pathological growth.

---

## 16. Scalarized-Auxiliary PLS (SA-PLS): Deep Region Search

### 16.1 Empirical Baseline and Diagnosis

Benchmarking Phase 3 (10 threads, 120s, size-100 instances) exposed the root cause of low specialisation. The weight vector controls only the **initial population seed**; once search begins, every region runs pure Pareto-dominance PLS, which has no memory of the weight vector at all.

| Region | Focus objective | Avg in-region% |
|--------|----------------|---------------:|
| R5 | Cloud (0/1/0/0) | 91.8% |
| R8 | Cost+Cloud (0.5/0.5/0/0) | 23.5% |
| R9 | Cost (1/0/0/0) | 7.3% |
| R0, R1, R3, R6 | Angle variants | 0–12% |
| R2, R4, R7 | Resolution variants | 0–1% |

Phase 3 gains +36% over sequential at 120s through **parallel restart diversity**, not decomposition. The gain shrinks as the time budget grows, because sequential PLS eventually covers the same Pareto front.

### 16.2 Why AuxiliaryFilter Does Not Fix This

The obvious first attempt is to gate the auxiliary population by `belongs_to_region`. This gates only the **seed** for the next iteration, but the search criterion itself is unchanged: workers still accept neighbors into the auxiliary only when they pass `archive.try_insert(n)`, which is global Pareto non-dominance. The acceptance chain in `process_neighbor_streaming` is:

```
neighbor n (explored from parent p):
  1. dedup check                              → register or skip
  2. dominated by p?                          → discard
  3. archive.try_insert(n)                    → Pareto non-dominated globally
  4. IF step 3 passes: auxiliary.try_insert(n) → gated by second Pareto check
```

Steps 3 and 4 both use global Pareto dominance. Adding a `belongs_to_region` gate at step 4 is cosmetic: it just discards out-of-region improvements from the seed pool *after* they were already found by doing unconstrained global Pareto search. The worker still explores the entire neighbourhood unconditionally and still follows the global Pareto geometry. The weight vector plays no role in the search whatsoever.

### 16.3 The Core Fix: Decouple Archive and Auxiliary Criteria

The archive and the auxiliary have different jobs:

- **Archive**: record every globally Pareto-non-dominated solution ever found. This is a quality metric for the final result.
- **Auxiliary** (next-iteration seed): drive the search **toward the region's weight vector direction**. This is a steering mechanism.

These two jobs require different acceptance criteria. The current code couples them: auxiliary ⊆ archive insertions ⊆ globally non-dominated. The fix is to make them independent:

```
neighbor n explored from parent p, region has weight vector w:

  Gate 1 — Archive (unchanged):
    archive.try_insert(n)           ← Pareto non-dominated w.r.t. global archive
    contributes to final result regardless of region

  Gate 2 — Auxiliary (new, independent of Gate 1):
    g_atch(n, w, z*) < g_atch(p, w, z*)  ← scalarized improvement over PARENT
    → auxiliary.try_insert(n)
    drives the next iteration in the region's direction
```

The two gates run independently. A neighbor can:

| Gate 1 (archive) | Gate 2 (auxiliary) | Interpretation |
|:-:|:-:|---|
| pass | pass | Pareto-improving **and** regionally relevant — best case |
| pass | fail | Globally improving but belongs to another region — archive it, don't chase it |
| fail | pass | Dominated globally but best in this region's direction — **chase it but don't archive it** — this is the new capability |
| fail | fail | Neither Pareto-improving nor regionally relevant — discard |

The "fail/pass" row is the key insight absent from Phase 3. A solution dominated by a member of the global archive may still be the best locally-reachable solution **in the direction of the weight vector**. Following it is exactly what allows the worker to descend the Tchebycheff landscape in its assigned slice, even when the global Pareto front has already surpassed it in other directions.

### 16.4 Scalarized-Auxiliary PLS (SA-PLS) Algorithm

Each region worker runs SA-PLS with its weight vector $\mathbf{w}$ and a dynamically maintained local ideal point $\mathbf{z}^*$.

**Acceptance criterion for auxiliary population:**

$$\text{accept\_aux}(n, p) \iff g^{atch}(n \mid \mathbf{w}, \mathbf{z}^*) < g^{atch}(p \mid \mathbf{w}, \mathbf{z}^*)$$

where the augmented Tchebycheff scalarization is (same as section 3):

$$g^{atch}(s \mid \mathbf{w}, \mathbf{z}^*) = \max_{j=1}^{D}\left[w_j \cdot \frac{\max(0,\, f_j(s) - z^*_j)}{\Delta_j}\right] + \rho \sum_{j=1}^{D} w_j \cdot \frac{\max(0,\, f_j(s) - z^*_j)}{\Delta_j}$$

with $\Delta_j$ the per-objective range and $\rho = 10^{-3}$ ensuring strict improvement direction uniqueness (see Appendix B for the rationale behind this value).

**Key properties:**
1. The acceptance criterion compares neighbor against its **parent** (the solution it was generated from), not against the archive. This is the MOEA/D decomposition acceptance rule.
2. The auxiliary can contain globally dominated solutions. This is intentional: they may be optimal under $\mathbf{w}$.
3. The archive comparison is removed from the auxiliary gate entirely. Workers chase their weight vector direction regardless of what other workers have found.

**Full per-neighbor processing:**

```
fn process_neighbor(n, p, w, z*, archive, auxiliary, explored):

    // Deduplication (unchanged)
    if explored.contains(n): return
    explored.insert(n)

    // Gate 1: global archive (unchanged, always run)
    archive.try_insert(n)           // Pareto non-dominance vs global archive

    // Gate 2: scalarized auxiliary (new, independent)
    if g_atch(n, w, z*) < g_atch(p, w, z*):
        aux_candidates.push(n)

    // Gate 3: dynamic ideal update (new)
    for j in 0..D:
        z*[j] = min(z*[j], n.f[j])
```

### 16.5 Population Management: Scalarized Priority Queue

Standard PLS uses an unordered `NdTreeSolutionSet` (Pareto front) as the population. In SA-PLS the population is a **Pareto front sorted by scalarized score** (a priority queue by $g^{atch}$ ascending):

- Exploration order: best scalarized solutions first.
- When time runs out mid-iteration, the most regionally relevant solutions have already been explored.
- The auxiliary is also ordered: better scalarized solutions propagate first into the next iteration.

The `NdTreeSolutionSet` supports iteration but not priority ordering. Two options:

**Option A (incremental)**: After `population.into_iter()`, sort the resulting `Vec<T>` by $g^{atch}(\cdot, \mathbf{w}, \mathbf{z}^*)$ before the exploration loop. One sort per step; $O(P \log P)$ with small $P$ (typical population size 10–200). This requires only changing two lines in `explore_population_neighborhoods`: the shuffle-or-sort line.

**Option B (full rewrite)**: Replace `NdTreeSolutionSet<population>` with a `BinaryHeap<ScoreOf<T>>` for the population only (archive stays as NdTree). More invasive but eliminates dominated solutions from the queue automatically.

Option A is preferred for Phase 5 because it reuses the existing PLS infrastructure. The sort replaces the shuffle.

### 16.6 Auxiliary Population as Scalarized Best Set

The current auxiliary uses `try_insert`, which enforces Pareto non-dominance. For SA-PLS, the **entry criterion** must be scalarized improvement over the parent:

$$g^{atch}(n \mid \mathbf{w}, \mathbf{z}^*) < g^{atch}(p \mid \mathbf{w}, \mathbf{z}^*)$$

To keep the implementation compatible with existing structures, use a two-stage approach:

1. Collect scalarized-accepted neighbors into `aux_candidates: Vec<T>` (independent of archive insertion).
2. Build next-step population from `aux_candidates`, sorted by scalarized score ascending.

This avoids introducing non-existent APIs (`insert_always`, `try_insert_or_replace_if_scalarized_better`) while preserving the key SA-PLS property: auxiliary acceptance is independent from global archive dominance.

> **Note (Unconstrained mode):** In `Unconstrained` mode, the existing double Pareto gate is preserved: a neighbor is first tested via `archive.try_insert(n)` (Gate 1), and then -- if archive-accepted -- also inserted into the auxiliary via `auxiliary.try_insert(n)` (Gate 2). The SA-PLS modes replace only Gate 2 with the scalarized criterion.

### 16.7 Termination and k-Increase

Phase 3 termination:
- Auxiliary empty after a step → try k+1
- k at maximum → return (fully explored neighbourhood)

SA-PLS termination:
- Scalarized auxiliary empty → worker has descended to a local optimum **in its weight direction** → try k+1 (wider neighbourhood flips may reveal escapes)
- k at maximum → worker's regional search is complete; continue archive updates from new scalarized-improved solutions if time remains (optional)

The relevant question: should a worker with exhausted scalarized search contribute to the global archive by running standard PLS steps? Yes -- the worker already holds a strongly specialised starting point. Once scalarized search is exhausted, switching to standard PLS uses that starting point to fill the global archive. This is the same as Phase 3 but **starting from a much better regional seed**.

On fallback, the PLS instance's population is re-seeded from the archive via the standard k-increase mechanism (`add_eligible_pareto_solutions`), just as in sequential PLS when the auxiliary empties.

```rust
enum SAPlsStatus {
    NewPopulation,
    LocalOptimumInRegion,
    ScalarizedExhausted,
}
```

### 16.8 Ideal Point Dynamics

The Tchebycheff score is sensitive to $\mathbf{z}^*$. Each worker maintains a **local ideal** updated per-neighbor (Gate 3 above). On every sync event, the worker takes the component-wise minimum of its local ideal and the global ideal from the shared snapshot:

```rust
fn update_ideal_from_global(&mut self) {
    let snapshot = self.global_front_slot.load();
    for j in 0..D {
        self.local_ideal[j] = self.local_ideal[j].min(snapshot.ideal[j]);
    }
}
```

When $\mathbf{z}^*$ improves mid-run, previously computed $g^{atch}$ scores become stale. This is acceptable: the population and auxiliary are regenerated each step, so scores are recomputed fresh at the start of each iteration using the current $\mathbf{z}^*$.

> **Mid-step semantics**: The ideal point $\mathbf{z}^*$ is updated continuously within a step (Gate 3 fires for every accepted neighbor). The population exploration order is set once per step (sorted by scalarized score at step start) and is **not** re-sorted mid-step. This means later population members see a more up-to-date ideal than earlier ones, which is acceptable given the incremental nature of improvements within a single step.

### 16.9 Changes to Existing Structures

**`ParetoLocalSearch::process_neighbor_streaming`** — add a fourth parameter:

```rust
fn process_neighbor_streaming(
    ...,
    // NEW: auxiliary acceptance mode
    aux_mode: AuxiliaryAcceptanceMode<'_, T>,
) {
    ...
    let archive_inserted = archive.try_insert(n);   // Gate 1: unchanged

    // Gate 2: auxiliary — independent from archive in scalarized mode
    let accept_aux = match aux_mode {
        AuxiliaryAcceptanceMode::Unconstrained => {
            // Preserve existing double Pareto gate: only insert into
            // auxiliary if also archive-accepted (backward compat).
            if archive_inserted { auxiliary.try_insert(n.clone()); }
            archive_inserted
        },
        AuxiliaryAcceptanceMode::Scalarized(ref f) => f(n, current_solution),
    };
    if accept_aux {
        aux_candidates.push(n.clone());
    }

    // Gate 3: ideal update
    if let Some(ideal) = local_ideal.as_mut() {
        for j in 0..D { ideal[j] = ideal[j].min(n.objectives()[j]); }
    }
}
```

**`ParetoLocalSearch::step`** — unchanged; gains a `step_scalarized` variant:

```rust
pub(crate) fn step_scalarized(
    &mut self,
    iteration: usize,
    timer: &Timer,
    weight: &[f64; D],
    local_ideal: &mut [f64; D],
    obj_ranges: &[f64; D],
    rho: f64,
) -> SAPlsStatus {
    // Sort population by g_atch before exploration
    // Apply scalarized auxiliary acceptance independent from archive insertion
    ...
}
```

**`RegionWorker`** — holds `local_ideal: [f64; D]` and calls `pls.step_scalarized(...)` in the main loop.

**`ConcurrentPLSConfig`** — adds:

```rust
pub region_search_mode: RegionSearchMode,
pub scalarized_rho: f64,     // augmentation coefficient; default 1e-3
```

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RegionSearchMode {
    /// Phase 3 behaviour: unconstrained Pareto search, parallel restarts.
    #[default]
    Unconstrained,
    /// SA-PLS: auxiliary criterion = scalarized improvement over parent.
    /// Archive remains global Pareto non-dominated.
    ScalarizedAuxiliary,
    /// SA-PLS with fallback: when scalarized auxiliary is exhausted, finish
    /// remaining time with standard PLS from the current regional seed.
    ScalarizedAuxiliaryWithFallback,
}
```

### 16.10 Full SA-PLS Worker Loop Pseudocode

```
fn region_worker_sapls(region, all_regions, initial_pop, problem,
                        snapshot_slot, global_front_slot, config):

    // Initialize local ideal from initial population
    local_ideal = component_wise_min(initial_pop.objectives())

    // Sync with global before starting
    local_ideal = min(local_ideal, global_front_slot.load().ideal)

    pls = ParetoLocalSearch::new(problem, initial_pop, ...)

    exhausted_scalarized = false

    for iteration in 1..=max_iterations:
        if config.region_search_mode == Unconstrained:
            status = pls.step(iteration, timer)     // Phase 3

        elif not exhausted_scalarized:
            // SA-PLS step: scalarized aux criterion, sorted population
            status = pls.step_scalarized(
                iteration, timer,
                region.weight_vector, local_ideal,
                region.objective_bounds.ranges(), config.scalarized_rho
            )
            if status == ScalarizedExhausted:
                if config.region_search_mode == ScalarizedAuxiliaryWithFallback:
                    exhausted_scalarized = true    // switch to standard PLS
                else:
                    break

        else:
            // Fallback: standard PLS from regionally-seeded starting point
            status = pls.step(iteration, timer)
            if status == AllNeighborhoodStructuresExplored:
                break

        // Periodic sync: publish local snapshot, update local ideal from global
        if iteration % config.sync_interval_steps == 0:
            publish_snapshot(pls.archive())
            local_ideal = min(local_ideal, global_front_slot.load().ideal)
            prune_tombstones_from_global()
            adopt_from_global_if_scalarized_improving()

        if timer.is_expired():
            break

    publish_final_snapshot(pls.archive())
    return RegionResult { archive: pls.archive(), stats, weight_vector: region.weight_vector }
```

### 16.11 Expected Impact

The core expected property is that workers with `ScalarizedAuxiliary` mode will:

1. **Stay in their region**: the population evolves toward the weight vector's direction because every iteration seed passed the scalarized improvement test.
2. **Find deep local optima in the region**: the worker descends the Tchebycheff landscape level by level, finding solutions that standard PLS would only reach by chance.
3. **Still contribute globally**: every Pareto-improving neighbor goes into the global archive regardless of the regional scalarized test.

The cross-worker adoption mechanism from Phase 3 remains but its role changes: a worker adopts a solution from the global front only if it also improves the worker's scalarized objective. This prevents workers from being polluted by solutions found by their neighbours.

**Testable hypotheses (not guarantees) vs Phase 3:**

| Metric | Phase 3 (Unconstrained) | SA-PLS (ScalarizedAuxiliary) |
|--------|------------------------|------------------------------|
| Avg in-region% | 18% baseline observed | higher than baseline |
| Cross-worker adoption quality | mostly near-zero useful adoption | higher precision via scalarized gate |
| Time to regional extremes | often late / unstable | earlier convergence in assigned direction |
| Final front size (fixed time) | +36% vs sequential (observed at 120s) | non-negative or positive delta to be verified statistically |

### 16.12 Evaluation Protocol (Scientific Rigor)

To validate SA-PLS soundly, run a preregistered, paired experimental protocol:

1. **Instances**: all size-100 benchmark instances used in Phase 3 plus at least one larger set (size-300) for stress behavior.
2. **Algorithms**: `Unconstrained`, `ScalarizedAuxiliary`, `ScalarizedAuxiliaryWithFallback`.
3. **Seeds**: fixed seed set $S$ (e.g., 30 seeds) used identically across all algorithms.
4. **Budgets**: fixed wall-clock budgets (e.g., 60s, 120s, 300s).
5. **Primary endpoints**:
    - global final front size,
    - hypervolume,
    - per-region in-region%,
    - time-to-first-regional-extreme.
6. **Secondary endpoints**:
    - adoption precision (fraction of adopted solutions that remain live),
    - regional exhaustion iteration,
    - load-balance spread (max/min iterations and archive sizes).
7. **Statistics**:
    - paired comparisons per instance-seed pair,
    - report median delta, IQR, and 95% bootstrap CI,
    - Wilcoxon signed-rank test for non-normal metrics.
8. **Ablations**:
    - $\rho \in \{10^{-4}, 10^{-3}, 10^{-2}\}$,
    - with/without scalarized adoption gate,
    - with/without fallback mode.

Success criterion: SA-PLS is accepted if it improves at least one primary endpoint without statistically significant degradation in others at the same budget.

---


## Appendix A: Das-Dennis Weight Vector Generation

```rust
/// Generate Das-Dennis weight vectors for D objectives with parameter H.
/// Returns C(H+D-1, D-1) uniformly distributed weight vectors on the unit simplex.
fn das_dennis_weight_vectors<const D: usize>(h: usize) -> Vec<[f64; D]> {
    let mut vectors = Vec::new();
    let mut current = [0usize; D];
    generate_recursive::<D>(&mut vectors, &mut current, h, 0, h);
    vectors
}

fn generate_recursive<const D: usize>(
    vectors: &mut Vec<[f64; D]>,
    current: &mut [usize; D],
    h: usize,
    depth: usize,
    remaining: usize,
) {
    if depth == D - 1 {
        current[depth] = remaining;
        let weight: [f64; D] = std::array::from_fn(|i| current[i] as f64 / h as f64);
        vectors.push(weight);
        return;
    }
    for i in 0..=remaining {
        current[depth] = i;
        generate_recursive::<D>(vectors, current, h, depth + 1, remaining - i);
    }
}
```

## Appendix B: Augmented Tchebycheff Scalarization

```rust
/// Compute augmented Tchebycheff scalarization value.
/// Lower is better (solution is closer to ideal in this weight direction).
fn tchebycheff_score<const D: usize>(
    objectives: &[u64; D],
    weight: &[f64; D],
    ideal: &[f64; D],
    bounds: &[(f64, f64); D],  // (min, max) per objective
    rho: f64,                   // augmentation coefficient (default: 1e-3)
) -> f64 {
    let mut max_term = f64::NEG_INFINITY;
    let mut sum_term = 0.0;
    
    for j in 0..D {
        let (min_j, max_j) = bounds[j];
        let range = (max_j - min_j).max(1.0);  // avoid division by zero
        // Normalize using objective bounds (min_j/max_j), not the ideal point.
        // The ideal point z*_j is used separately to compute distance from ideal.
        let dist_from_ideal = (objectives[j] as f64 - ideal[j]).max(0.0);
        let normalized = dist_from_ideal / range;
        let weighted = weight[j] * normalized;
        max_term = max_term.max(weighted);
        sum_term += weighted;
    }
    
    max_term + rho * sum_term
}
```

## Appendix C: References

1. **ND-Tree**: Jaszkiewicz, A. "ND-Tree-based update: A Fast Algorithm for the Dynamic Non-Dominance Problem." [arXiv:1603.04798](https://arxiv.org/abs/1603.04798)
2. **Parallel PLS**: Drugan, M. & Thierens, D. "Parallel Pareto Local Search Revisited." GECCO 2018. [PDF](https://www.cmap.polytechnique.fr/~nikolaus.hansen/proceedings/2018/GECCO/proceedings/proceedings_files/pap465s3-file1.pdf)
3. **Reverse Strategy**: Li, M. et al. "Reverse Strategy for Non-dominated Archiving." [PDF](https://hisaolab-sustech.github.io/pdf/Reverse%20Strategy%20for%20Non-dominated.pdf)
4. **MOEA/D Decomposition**: Zhang, Q. & Li, H. "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition." IEEE TEVC, 2007.
5. **Das-Dennis**: Das, I. & Dennis, J.E. "Normal-Boundary Intersection: A New Method for Generating the Pareto Surface." SIAM J. Optim., 1998.
