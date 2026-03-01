# Tracker Wars: SIMS Objective Trackers (Codeforces-style task)

This document describes a performance challenge based on the SIMS (Satellite Image Mosaic Selection) objective trackers in this repository.

You are given a *set cover instance* (elements = ground “parts”; images = subsets of parts) with 4 objectives. Your task is to implement an **incremental tracker** that can:

1) report the objective **delta** for hypothetical operations (`peek_*`),
2) update its internal state for real operations (`track_*`),
3) keep the tracked objective values correct after a long sequence of operations.

The fastest correct implementation wins.

---

## Problem statement

There are:

- `M` images, indexed `0..M-1`
- `U` universe elements (parts), indexed `0..U-1`

For each image `i`:

- `E[i]` is the set of universe elements covered by image `i`.
- `clouds[i]` is the subset of elements of `E[i]` that are cloudy *in that image*.
- `clear[i] = E[i] \ clouds[i]` are the elements covered **clearly** by image `i`.

Each element `e` has area `area[e]`.
Each image `i` has:

- cost `cost[i]`
- resolution `res[i]`
- incidence angle `ang[i]`

A solution is a set `S ⊆ {0..M-1}` of selected images.

All objectives in this repo are treated as **minimization** objectives.

---

## Objectives (exact semantics)

The objective vector is:

- `F_cost(S)`
- `F_cloudy(S)`
- `F_minres(S)`
- `F_maxang(S)`

### 1) Total Cost

\[
F_{cost}(S) = \sum_{i \in S} cost[i]
\]

### 2) Cloudy Area

An element is **clear** if at least one selected image covers it clearly:

\[
covered\_clear(S) = \bigcup_{i \in S} clear[i]
\]

Then cloudy area is the sum of areas of elements **not** in `covered_clear(S)`:

\[
F_{cloudy}(S) = \sum_{e \notin covered\_clear(S)} area[e]
\]

This matches the implementation in `ObjectiveState::CloudyArea::calculate_value()` and the incremental `CloudyAreaState` tracker.

### 3) Minimum Resolution Sum

For each element `e`, look at all selected images that cover `e` (cloudiness does **not** matter here; only membership in `E[i]`).

Define:

\[
minRes(e, S) = \min_{i \in S,\ e \in E[i]} res[i]
\]

If `e` is not covered by any selected image, its contribution is `0`.

\[
F_{minres}(S) = \sum_{e=0}^{U-1} \begin{cases}
0 & \text{if } \nexists i \in S: e \in E[i]\\
minRes(e, S) & \text{otherwise}
\end{cases}
\]

### 4) Maximum Incidence Angle

\[
F_{maxang}(S) = \max_{i \in S} ang[i]
\]

If `S` is empty, `F_maxang(S) = 0`.

---

## Operations and deltas

You will process `Q` operations starting from an **empty** selection `S = ∅`.

Each operation is one of:

- `TrackAdd i`: update state as if image `i` is added to `S`.
- `TrackRem i`: update state as if image `i` is removed from `S`.
- `PeekAdd i`: compute the objective delta **if** we added `i` (do not modify state).
- `PeekRem i`: compute the objective delta **if** we removed `i` (do not modify state).

The delta is always defined as:

\[
\Delta(S, op) = F(S_{after}) - F(S_{before})
\]

Deltas are a 4-vector in the same objective order: `(Δcost, Δcloudy, Δminres, Δmaxang)`.

### Validity assumptions

For the competitive task, assume the operation sequence is **consistent**:

- `TrackAdd i` is only called when `i ∉ S`
- `TrackRem i` is only called when `i ∈ S`

In this repo’s benchmark trace, consistency is enforced by explicit `Reset` events (see “repo benchmark mode”): a `Reset` resets the selection and tracker state to empty, and between resets the `TrackAdd`/`TrackRem` sequence is consistent.

---

## Input / output format (Codeforces-style)

### Option A (standalone, text I/O)

Input:

1. `M U`
2. For `i=0..M-1`: list `E[i]` (elements covered by image i)
3. For `i=0..M-1`: list `clouds[i]` (cloudy subset)
4. Arrays `cost[0..M-1]`, `res[0..M-1]`, `ang[0..M-1]`
5. Array `area[0..U-1]`
6. `Q`
7. `Q` lines: `type i` where `type ∈ {0,1,2,3}` for `{TrackAdd, TrackRem, PeekAdd, PeekRem}`

Output:

- For each of the `Q` operations, print one line with 4 signed integers:
  `Δcost Δcloudy Δminres Δmaxang`
- After all operations, print one final line with the **absolute** objective values:
  `F_cost F_cloudy F_minres F_maxang`

### Option B (repo benchmark mode)

This repository already includes:

- A MiniZinc `.dzn` instance file, e.g. [sims-heuristics/data/lagos_nigeria_30.dzn](../data/lagos_nigeria_30.dzn)
- A binary u16 trace file, e.g. [sims-heuristics/benches/data/debug_lagos_30_tracker_calls.u16](../benches/data/debug_lagos_30_tracker_calls.u16)

In benchmark mode, the “input” is fixed to those files:

- Instance: `lagos_nigeria_30.dzn` (MiniZinc 1-based indices, normalized to 0-based when parsed)
- Operations: u16 records: `record = (op << 12) | image_index`, little-endian
  - `op=0 TrackAdd`
  - `op=1 TrackRem`
  - `op=2 PeekAdd`
  - `op=3 PeekRem`
  - `op=4 Reset` (reset selection + trackers to empty)
  - `image_index` is 0-based, stored in the lower 12 bits

For `Reset`, `image_index` is ignored (written as 0).

The Criterion benchmark replays this trace and measures throughput in “events per second”.

---

## Why naïve recomputation is too slow (what you’re optimizing)

A correct but slow approach is:

- maintain the selected set `S`
- for every operation, recompute every objective from scratch

That is typically:

- `TotalCost`: `O(|S|)` (or `O(1)` if tracked)
- `CloudyArea`: `O(U/word)` union of bitsets or `O(Σ|clear[i]|)` over selected images
- `MinResolutionSum`: for each element, scan selected images that cover it → expensive
- `MaxIncidenceAngle`: `O(|S|)` scan

With millions of operations (`Q ~ 10^6 .. 10^7`), the above becomes the bottleneck.

---

## Tracker design challenges (per objective)

### TotalCost

Trivial: track `current_cost` and update by `±cost[i]`.

### CloudyArea (the tricky one)

You need `F_cloudy(S)` where an element is cloudy iff **no selected image covers it clearly**.

Key difficulty: when toggling one image, many elements may change from cloudy→clear or clear→cloudy.

A standard incremental structure is:

- `count_clear[e] = number of selected images that cover element e clearly`
- `F_cloudy(S) = Σ area[e] for which count_clear[e] == 0`

Then for `TrackAdd i`:

- for each `e ∈ clear[i]`: if `count_clear[e]==0` then `F_cloudy -= area[e]`; then `count_clear[e]++`

For `TrackRem i`:

- for each `e ∈ clear[i]`: if `count_clear[e]==1` then `F_cloudy += area[e]`; then `count_clear[e]--`

`Peek*` must compute the same delta without mutating.

### MinResolutionSum

You need, for each element `e`, the minimum `res[i]` over selected images that cover it.

The hard case is removal: if you remove the (unique) minimum-resolution image for some element, the element’s min becomes the *next* minimum among remaining selected images covering that element.

Typical approaches:

- Maintain per-element multiset of resolutions (small in SIMS: number of images ≤ ~200)
- Cache `min[e]` and `count_of_min[e]`
- On removal:
  - if removed resolution is not the min → no change
  - else if `count_of_min[e] > 1` → no change to min, just decrement count
  - else scan the multiset to find the next min

This is exactly the performance hotspot in the tracker code: it wants `O(1)` for most cases and only occasionally `O(k)` scan.

### MaxIncidenceAngle

You need the maximum of `ang[i]` among selected images.

Hard case is removal of the unique maximum.

Typical approaches:

- Maintain a sorted vector of selected angles (small-N), or
- Maintain a multiset / heap + deletion bookkeeping

With small `M` (≤200) a sorted `Vec<u64>` with binary search insertion/removal is often fastest in practice.

---

## Implementation notes for this repo

- Trackers live in [sims-heuristics/src/trackers.rs](../src/trackers.rs)
- Objectives and their exact “from scratch” definitions live in [sims-heuristics/src/objectives.rs](../src/objectives.rs)
- The benchmark that replays the recorded trace lives in [sims-heuristics/benches/tracker_replay_lagos_30.rs](../benches/tracker_replay_lagos_30.rs)

Recommended workflow:

- Start with correctness (cross-check tracker deltas against `ObjectiveState::calculate_value` on random small sequences)
- Then optimize allocations, cache locality, and branch patterns
- Then benchmark using the replay bench

---

## What to submit (challenge tasks)

1. Implement a tracker that supports all 4 objectives and all 4 operations.
2. `peek_*` must be side-effect free and match the delta of the corresponding `track_*` on the same state.
3. The final tracked objective values must match the definitions above.
4. Make it fast.

Optional “extra credit” tasks:

- Add a second tracker implementation (different data layout) and compare on the replay bench.
- Add a validator that checks `peek` vs `track` consistency on random sequences.

