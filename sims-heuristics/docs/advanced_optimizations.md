# Advanced Optimization Strategies for SIMS Objective Trackers

## Current Status
- **Baseline (StandardTracker)**: ~290ms
- **Target (5x Speedup)**: ~58ms
- **Current Best (SimdTracker)**: ~77ms (3.77x speedup)
- **Gap**: ~19ms (need ~25% further reduction)

## Bottleneck Analysis
The current `SimdTracker` implementation has optimized the instruction stream significantly (branchless, loop unrolling, packed counters), but the remaining gap to the 5× target likely requires changing the *shape* of the updates.

In tracker workloads, the dominant costs are usually some combination of:
- **Metadata bandwidth** (reading element IDs / indices)
- **Loop overhead** (one iteration per element)
- **Update dependency chains** (load → modify → store on the same cache line)

Crucially, the “random CSR access” assumption is not always correct for SIMS instances; many incidence lists are already sorted and strongly run-length-compressible.

## Proposed Optimization Strategies

### 1. Data Layout & compression

#### A. Bit-Sliced Counters (Vertical Counters)
Instead of storing counts as an array of integers (Horizontal), store them as bit-planes (Vertical).
- **Concept**:
  - `Plane0`: Bit 0 of all counts
  - `Plane1`: Bit 1 of all counts
  - `Plane2`: Bit 2 of all counts
- **Pros**:
  - If most counts are small (0, 1, 2), we might only access `Plane0` and `Plane1`.
  - Massive SIMD parallelism: Updates can handle 256 or 512 counters at once using bitwise logic.
- **Cons**:
  - Random access updates are expensive (need to update multiple bit-words). This is better for dense updates, but our updates are sparse/scattered.
  - **Verdict**: Likely **NOT** suitable for random access updates, but powerful for scanning/recalculating from scratch.

#### B. Blocked/Cluster Layout
Group elements often accessed together into the same cache lines.
- **Concept**: Reorder element IDs so that elements covered by the same images are close to each other.
- **Algorithm**: Use hypergraph partitioning or simple clustering (e.g., sort elements by the image that covers them most).
- **Pros**: Increases spatial locality. If Image A covers elements {1, 100, 2000}, remapping them to {1, 2, 3} makes the update linear access.
- **Verdict**: **High Potential**. Pre-processing the instance to re-index elements could drastically reduce cache misses.

### 2. Algorithmic Optimizations

#### A. Lazy Updates / Delta Tracking
Don't update everything immediately.
- **Concept**: If an image is removed, we only need to know if it *was* the critical cover (count=1 -> 0).
- **Implementation**: Maintain a "Dirty" bitset. Batch updates or only compute exact values when queried.
- **Verdict**: Complex for `track_` operations which need to return exact deltas immediately.

#### B. 4-Level Quantization (for MinResolution)
Since MinResolution heavily relies on `c0` (low-res count) and `c1` (high-res count), and we mostly care about 0 vs 1 vs >1:
- We only need to track states: {0, 1, 2+}.
- This fits in 2 bits per counter.
- We can pack 16 counters into a `u32`.
- **Verdict**: High complexity to handle overflow (what happens at 3?), but reduces memory footprint by 8x.

### 3. SIMD & Instruction Level Optimizations

#### A. Software Pipelining / Prefetching (Revisited)
We tried `prefetch`, but maybe incorrectly.
- **Strategy**: Prefetch `k` iterations ahead.
- **Loop Structure**:
  1. Load indices for batch `i`
  2. Prefetch addresses for batch `i`
  3. Process data for batch `i-1`
- **Verdict**: Hard to tune, but classic solution for latency hiding.

#### B. Gather/Scatter Intrinsics (AVX-512)
If the hardware supports AVX-512 (not universally available, but good to know):
- `vpgatherdd`: Load 16 integers from scattered addresses.
- `vpscatterdd`: Store 16 integers to scattered addresses.
- **Verdict**: Requires specific hardware. `portable_simd` might not expose scatter yet.

### 4. Hybrid Structure (Sparse + Dense)

#### A. Sparse Storage for High Counts
Most elements might be covered by very few images, or very many.
- If an element is covered by 100 images, its count never drops to critical levels (0 or 1) during a single local search step.
- **Optimization**: Mark "Safe" elements that have high counts. Skip updating them entirely until a "Refresh" phase.
- **Risk**: Correctness. We need to know exactly when it drops to 1. But if we know `count >= safety_margin`, we can decrement a `safety_counter` instead of the main logic.

### 5. Instance Remapping & Data Structure Transformation

#### Analysis of Current Data (Lagos 30)
We analyzed the memory access patterns of the `lagos_nigeria_30` instance and found:
- **95.5%** of element transitions are within a single cache line (gap <= 16).
- **89.5%** of elements belong to strictly consecutive runs of length >= 4 (e.g., `{10, 11, 12, 13...}`).
- **Median Gap** is 1.0.

Additional interval-compression stats for `lagos_nigeria_30` (30 sets):
- `E[i]` (“images”): average size **112.3**, average **15.5 intervals**, average interval length **7.23** (≈ **7.23×** compression)
- `clouds[i]`: average size **50.6**, average **26.4 intervals**, average interval length **1.92** (≈ **1.92×** compression)

#### Implications
This means the random access problem is actually a **Sequential Access** problem in disguise.
The current tracker iterates over these runs one by one (`for id in elements`), treating them as random indices. This prevents the CPU from efficiently vectorizing the load-add-store cycle.

#### Proposal: Run-Length Encoding (RLE) / Interval Tracking
Instead of storing `image_elements` as `Vec<u16>`, we should convert them (internally in the tracker) to a list of **Intervals**:
```rust
struct Interval {
    start: u16,
    len: u16,
}
// Image -> Vec<Interval>
```
For the `lagos` dataset, this would effectively compress the incidence list by ~10x (since avg run length is likely high).

**Benefits:**
1. **Explicit SIMD**: We can process an entire interval using SIMD instructions.
   - `start..start+len`: Load 8 integers, Add 8 integers, Store 8 integers.
2. **Reduced Metadata Bandwidth**: We fetch far fewer indices from memory.
3. **Loop Overhead**: One loop over intervals vs one loop over every single element.

**Proof of Concept Plan:**
1. Create `IntervalTracker`.
2. In `new()`, scan the `SimsDiscreteProblem` and build `Vec<Vec<Interval>>`.
3. Implement `track_addition` to iterate intervals.
4. Implement simple `for i in range` loops (compiler autovectorization is excellent on ranges) or use `step_by` manual unrolling.

#### Legacy: Instance Remapping
(Previously proposed) Reordering elements to create clusters is **not necessary** for this dataset, as it is already highly clustered. The RLE approach exploits the existing structure more directly.

### 6. SOTA: Range-Update Trees (Segment Tree with 0/1/2+ Histograms)

If you can represent each per-image update set as **a small number of intervals**, you can go one step beyond “vectorized loops” and switch to **range updates**.

#### A. CloudyArea as a range-update problem

For `CloudyArea`, the standard tracker maintains `count_clear[e]` and cares only about `count_clear[e] == 0` and `== 1` transitions.

That makes it a perfect fit for a lazy-propagation segment tree that stores, per node:
- `n0, n1, n2p`: how many leaves (elements) have count 0, 1, or ≥2
- `sum_area0`: sum of `area[e]` for elements with count 0
- (optionally) `sum_area1`: sum of `area[e]` for elements with count 1 (helps compute deltas quickly)

When applying `+1` to a fully-covered node:
- `0 → 1`, `1 → 2+`, `2+ → 2+`

When applying `-1`:
- `1 → 0`, `2+ → 1`, `0 → 0` (shouldn’t happen under valid traces)

This update is *purely a permutation of buckets* and can be done in O(1) per node, with a small lazy tag.
Then a whole image update becomes `O(#intervals · log U)`.

This approach is widely used in “range add + count zeros/ones” problems (competitive programming), but here it becomes practical because SIMS incidence lists are often interval-heavy.

#### B. MinResolutionSum in SIMS is (often) a 2-counter cross-product

In SIMS instances, `res[i]` is frequently binary (e.g., 30 vs 50), which means per-element contribution is:
- `0` if uncovered
- `50` if covered only by 50-res images
- `30` if covered by any 30-res image

This can be tracked with two saturating counters per element (`c50`, `c30`), but in a range-update tree you typically store the **cross-product state distribution**.

One practical representation is a 3×3 bucket table per node:
- `c50 ∈ {0,1,2+}`
- `c30 ∈ {0,1,2+}`

Range update on `c30` becomes a shift along the `c30` dimension; range update on `c50` shifts along `c50`.
From the 3×3 distribution you can derive the counts of elements contributing 0/30/50 and compute deltas without touching every element.

### 7. SOTA: Saturating Counters + SIMD for interval bodies

Even without a tree, switching from “exact counts” to “0/1/2+ saturating counts” is often a win:
- Store counts as `u8` (clamp at 2)
- For each interval, load a SIMD vector (e.g., 16×u8 or 32×u8), compare to 0/1, update, and accumulate a delta count

For `CloudyArea`, you additionally need weights (`area[e]`), which complicates pure SIMD accumulation. Two common ways around that:
1) Maintain `sum_area0` in a separate structure (tree or block sums), or
2) Use block-wise precomputed prefix sums of `area` and accept scalar fixups on the few “boundary” elements.

### 8. SOTA: Compressed index formats (beyond plain intervals)

Intervals are the best case. When sets are sorted but not very run-length-compressible, a common SOTA approach is:
- Store a `base` per block (e.g., every 32 IDs)
- Store `u8` or `u16` deltas within that block
- Optionally decode in SIMD

This can reduce index bandwidth and keep your update loop compute-bound rather than memory-bound.

### 9. SOTA: Multi-objective fusion and “hot/cold” splits

When multiple objective trackers iterate over similar sets, consider:
- **Fusing loops** (touch `packed_counts`, `areas`, and any other arrays once)
- **Hot/cold splitting**: keep the frequently-touched per-element state in a compact, tightly packed struct-of-arrays; move rarely-used fields elsewhere

This can matter as much as any single micro-optimization once you’re chasing sub-80ms numbers.

### 10. SOTA: Succinct monotone set encodings (Elias–Fano)

When sets are sorted but not strongly run-length-compressible, a high-end alternative to intervals is **succinct monotone sequence encoding**.

The classic choice is **Elias–Fano**, widely used in search engines and graph analytics:
- Very compact representation of increasing integer lists
- Fast sequential iteration
- Great cache behavior when you iterate many sets back-to-back

In practice, a simpler and often equally effective stepping stone is “base + small deltas per block”, optionally decoded in SIMD.

### 11. SOTA: Roaring/Concise bitmaps (when sets are large and you need set ops)

If you ever decide to compute `covered_clear` via bitmap unions (instead of per-element counters), use a compressed bitmap format rather than a dense bitset:
- **Roaring bitmaps** shine when the universe is large and the set density varies.

This is not a drop-in win for the current counter-based incremental approach, but it can become relevant if you restructure the tracker to batch `Peek*` evaluations (many unions against the same baseline state).

### 12. SOTA: Cache-blocked CSR / BCSR

If your sets are neither interval-friendly nor compressible by small deltas, you can still win by changing the *physical layout*:
- Store indices in fixed-size blocks (e.g., 32/64) aligned to cache lines
- Optionally store per-block metadata (min/max, base) for fast bounds checks

This reduces branchiness and helps the CPU’s hardware prefetchers.
