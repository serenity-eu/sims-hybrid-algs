# Pareto Local Search (PLS) design in `sims-heuristics`

This document describes the current design of the Pareto Local Search implementation in the `sims-heuristics` crate, with a focus on algorithmic structure, heuristic choices, and how the neighborhood is explored.

Primary entrypoints:
- PLS loop: [sims-heuristics/src/pareto_local_search.rs](sims-heuristics/src/pareto_local_search.rs#L53)
- Neighborhood for the main solution type (`BitsetEncodedSolution`): [sims-heuristics/src/solution_impl/bitset_encoded_solution.rs](sims-heuristics/src/solution_impl/bitset_encoded_solution.rs#L1571)
- Main runnable example used for debugging: [sims-heuristics/src/bin/debug_lagos_30.rs](sims-heuristics/src/bin/debug_lagos_30.rs)

## 1) What PLS is solving

The SIMS problem is modeled as a *set cover*:
- Universe elements are ground “tiles/parts” that must all be covered.
- Each satellite image covers a subset of elements.
- Feasibility constraint: selected images must cover all elements.

PLS optimizes multiple objectives (typically 4):
- Total cost
- Cloudy area (area that remains cloudy across chosen images)
- Minimum resolution sum
- Maximum incidence angle

Objectives are evaluated as minimization objectives (lower is better).

The abstract problem interface used by PLS is [SetCoverProblem](sims-heuristics/src/problem.rs#L11), and the fast bitset-backed implementation used by `debug_lagos_30` is [ProblemBitset](sims-heuristics/src/problem_bitset.rs#L13).

## 2) High-level architecture

The PLS implementation is intentionally generic over:
- `P`: problem type implementing `SetCoverProblem<D>`
- `T`: solution type implementing `EncodedSolution` + `ProbabilisticProbingNeighborhood`
- `S`: Pareto archive/population type implementing the `pareto::ParetoFront` interface

In practice (including `debug_lagos_30`), the common instantiation is:
- `P = ProblemBitset<4>`
- `T = BitsetEncodedSolution<ProblemBitset<4>, 4>`
- `S = NdTreeSolutionSet<BitsetEncodedSolution<...>, 4>`

### Key components

**PLS orchestrator**
- `struct ParetoLocalSearch` maintains:
  - `population`: current “frontier” to expand
  - `approximated_pareto_set`: global non-dominated archive
  - `neigborhood_structure`: current neighborhood “size” $k$
  - `explored_solutions`: global duplicate tracking + “local optimality by k” tracking
  - `spare_tracker`: reusable objective trackers to reduce allocations

See [ParetoLocalSearch struct](sims-heuristics/src/pareto_local_search.rs#L53).

**Pareto archive implementation**
- `NdTreeSolutionSet` wraps an ND-tree for dominance management.
- `try_insert()` updates the ND-tree and (in debug builds) asserts “no dominated pairs exist”.

See [NdTreeSolutionSet](sims-heuristics/src/solution_set_impl/ndtree_solution_set.rs#L10).

**Solution type + neighborhood generator**
- `BitsetEncodedSolution` stores the selected images as a `FixedBitSet`.
- It exposes neighborhood generation through `neighborhood_iter()`.

Traits:
- `ImageSet` (selected/unselected iteration) [sims-heuristics/src/solution.rs](sims-heuristics/src/solution.rs#L13)
- `SIMSModifiable` includes `neighborhood_iter()` and `is_valid()` [sims-heuristics/src/solution.rs](sims-heuristics/src/solution.rs#L54)

**Incremental objective tracking**
- Trackers maintain objective-specific state to allow fast “delta” evaluation on add/remove.
- `CloudyArea` deltas are *not* computed by the generic objective delta function; they are computed via trackers.

See [trackers](sims-heuristics/src/trackers.rs#L1) and the explicit guard in [objective delta](sims-heuristics/src/objectives.rs#L198).

## 3) Entering PLS: `ParetoLocalSearch::run()`

The main loop is [ParetoLocalSearch::run](sims-heuristics/src/pareto_local_search.rs#L577).

### Initialization

`ParetoLocalSearch::new()`:
- Creates a new `population` by inserting the provided initial solutions through `try_insert()` (dominance filtering).
- Registers each initial solution in `ExploredSolutionsData` with iteration = 0.
- Clones population into `approximated_pareto_set`.

See [constructor](sims-heuristics/src/pareto_local_search.rs#L88).

### Iteration/termination

At each iteration:
1. `step(i)` explores the neighborhood of all solutions currently in `population` using the current neighborhood size $k$.
2. Depending on whether a new “auxiliary population” is found, either:
   - restart exploration from the smallest neighborhood size, or
   - increase neighborhood size $k$ and re-seed the population from eligible archive solutions.
3. Stop when:
   - all neighborhood sizes have been explored (no new solutions and $k$ reached max), or
   - time limit expires, or
   - max iterations reached.

## 4) Step workflow (population expansion)

A single `step()` is implemented in [sims-heuristics/src/pareto_local_search.rs](sims-heuristics/src/pareto_local_search.rs#L155).

Conceptually:

1. Validate population archive invariants (`population.validate()`).
2. Move (`mem::replace`) the population out of `self.population` so we can build a fresh one.
3. For each solution in the old population:
   - generate its neighborhood solutions
   - filter/insert them into:
     - the global Pareto archive (`approximated_pareto_set`)
     - an auxiliary population (`auxiliary_population`) for next iteration
   - mark the current solution as explored up to neighborhood size $k$.
4. Decide next step:
   - if auxiliary population non-empty ⇒ replace population and reset $k$ to the minimum
   - else ⇒ increase $k$ and add eligible archive solutions

### Duplicate suppression and “explored up to k”

Two distinct mechanisms are used:

1) **Global duplicate suppression**
- Before evaluating a neighbor, PLS checks `explored_solutions.is_registered(neighbor)`.
- If registered, it’s counted as a duplicate and skipped.
- Otherwise it is registered (with iteration and timestamp).

See [process_neighbor](sims-heuristics/src/pareto_local_search.rs#L257).

2) **Local optimality tracking by neighborhood size**
- After finishing a solution’s neighborhood exploration at size $k$, PLS records that fact:
  `update_explored_neighborhood_size(solution, k)`.
- When increasing $k$, PLS re-adds archive solutions that have been explored only up to smaller neighborhood sizes.

See [add_eligible_pareto_solutions](sims-heuristics/src/pareto_local_search.rs#L524).

## 5) Neighborhood exploration: what a “move” means

PLS requests neighbors via `solution.neighborhood_iter(trackers, k, problem, timer, deterministic)`.

For `BitsetEncodedSolution`, this call is backed by [neighborhood_iter_impl](sims-heuristics/src/solution_impl/bitset_encoded_solution.rs#L1571), which implements a *remove-then-repair* neighborhood based on a **Residual Problem**.

### 5.1 Neighborhood parameter: `k`

`k` (called `neigborhood_structure` in PLS) controls how many images are removed simultaneously.

- When $k=1$:
  - candidates are all selected images that are “replaceable” (`is_replaceable`).
  - The iterator lazily yields one candidate removal at a time.

- When $k>1$:
  - candidates are drawn from a *small* shortlist of “worst” selected images.
  - `worst_selected_images()` uses a scalarized heuristic key over normalized objective deltas (with trackers).
  - The iterator explores combinations of size `k` over that shortlist.

See candidate generation in [neighborhood_iter_impl](sims-heuristics/src/solution_impl/bitset_encoded_solution.rs#L1589).

### 5.2 The “Residual Problem” concept

After removing `k` images, the solution may become infeasible (uncovered elements). Instead of exploring arbitrary add/remove sequences, the neighborhood is defined by:

1. Remove a chosen set of images (the “removal candidates”).
2. Compute uncovered elements.
3. Build a small **decision set** of images to consider for (re-)adding.
4. Enumerate small subsets of that decision set (subset sizes up to 5) that restore feasibility.
5. Each feasible subset corresponds to a neighbor.

This is implemented by:
- `BitsetEncodedSolution::create_residual_problem(...)` [sims-heuristics/src/solution_impl/bitset_encoded_solution.rs](sims-heuristics/src/solution_impl/bitset_encoded_solution.rs#L542)
- `ResidualProblem` [sims-heuristics/src/residual_problem.rs](sims-heuristics/src/residual_problem.rs#L14)

#### Decision set construction (heuristic)

`create_residual_problem` builds the decision set (called `image_index_map` inside `ResidualProblem`) as:
- `decision_set = removal_candidates ∪ best_addition_candidates` (stable order; set-union)

Where `best_addition_candidates` is a top-N shortlist (N≈9) of unselected images that:
- cover at least one uncovered element, and
- look promising under a scalarized weighted sum of **scaled objective deltas**.

See:
- [create_residual_problem](sims-heuristics/src/solution_impl/bitset_encoded_solution.rs#L542)
- [best_unselected_images_with_trackers](sims-heuristics/src/solution_impl/bitset_encoded_solution.rs#L604)
- [ResidualProblem::new](sims-heuristics/src/residual_problem.rs#L31)

This union is a deliberate tractability heuristic:
- It keeps the residual candidate pool small (≈ `k + 9`, minus dedup), while still allowing both “re-add removed images” and “add promising new images” repair paths.

### 5.3 Residual solving and neighbor generation

Once a `ResidualProblem` exists:
- `BitsetNeighborhoodIter` keeps it as state.
- It repeatedly calls `residual_problem.solve_next()` to get the next feasible subset.

See:
- iterator loop: [BitsetNeighborhoodIter::next](sims-heuristics/src/solution_impl/bitset_encoded_solution.rs#L1635)
- feasibility enumeration: [ResidualProblem::solve_next](sims-heuristics/src/residual_problem.rs#L281)

`solve_next()`:
- enumerates subsets of the decision set of sizes 0..=5
- checks set cover in the **condensed** residual universe using bitset unions
- returns the first subset that covers all residual elements

Note: `solve_next()` does *not* enforce Pareto-optimality in residual space; it just yields feasible repairs. Pareto filtering happens at the PLS layer when inserting neighbors into the global archive.

### 5.4 Merging residual solutions back into a full neighbor

The residual solution uses **condensed indices**, so it must be merged back into the original solution domain.

This is done by `MergeableWithResidual::merge_residual_solution` implemented for `BitsetEncodedSolution`.

See [merge_residual_solution](sims-heuristics/src/solution_impl/bitset_encoded_solution.rs#L903).

## 6) How objective evaluation is made fast

### 6.1 Incremental trackers

The neighborhood logic is add/remove heavy, so objective evaluation is incremental:
- Each objective has a tracker that can:
  - peek deltas without mutation
  - apply changes and update its internal state

See:
- tracker interface: [ObjectiveTracker](sims-heuristics/src/trackers.rs#L16)
- standard tracker enum: [StandardTracker](sims-heuristics/src/trackers.rs#L82)

`BitsetEncodedSolution::add_image` and `remove_image` update objectives using tracker-produced deltas and [apply_delta](sims-heuristics/src/objectives.rs#L365).

### 6.2 Bitsets for set operations

Bitsets are used in multiple places:
- solution selection (`FixedBitSet` of selected images)
- problem images (`FixedBitSet` of elements per image)
- residual problem (condensed image coverage)

This allows fast union/intersection operations (critical for set-cover checks in residual solving).

## 7) Dominance filtering and archive management

PLS uses two layers of dominance filtering:

1) **Cheap dominance against the current solution**
- If a neighbor is dominated by the current solution’s objectives, it is immediately discarded.

See [evaluate_neighbor](sims-heuristics/src/pareto_local_search.rs#L335).

2) **Global archive dominance filtering**
- If not immediately discarded, neighbor insertion into `approximated_pareto_set` is attempted.
- The archive is responsible for rejecting dominated points and potentially removing dominated existing ones.

See:
- [try_add_to_pareto_set](sims-heuristics/src/pareto_local_search.rs#L372)
- ND-tree insertion: [NdTreeSolutionSet::try_insert](sims-heuristics/src/solution_set_impl/ndtree_solution_set.rs#L42)

The auxiliary population only collects neighbors that survive dominance checks within that auxiliary set.

## 8) Determinism knobs

The code supports deterministic runs to make debugging reproducible:
- In `debug_lagos_30`, initial solutions are generated with fixed seeds, and `is_deterministic = true`.

See [debug_lagos_30.rs](sims-heuristics/src/bin/debug_lagos_30.rs#L1).

Determinism influences:
- random weights used for scalarization in candidate scoring (deterministic ⇒ equal weights)
- random choice in some probing strategies (where used)

## 9) Probabilistic probing neighborhood (implemented, not the default PLS path)

There is a `ProbabilisticProbingNeighborhood` trait intended for sampling neighbors in objective space.
- PLS currently calls `neighborhood_iter()` in the main loop.
- `BitsetEncodedSolution` *also* implements probabilistic probing as an alternative generator.

See:
- trait: [probabilistic_probing_neighborhood.rs](sims-heuristics/src/probabilistic_probing_neighborhood.rs)
- impl: [bitset_encoded_solution.rs](sims-heuristics/src/solution_impl/bitset_encoded_solution.rs#L962)

This design allows experimenting with “sampled neighborhoods” without changing the PLS orchestration.

## 10) Walkthrough: `debug_lagos_30.rs`

The debug binary is a minimal reproducible driver:

1. Loads `lagos_nigeria_30.dzn` into `ProblemBitset<4>` with a fixed objective ordering.
2. Builds an initial population by generating random feasible solutions with deterministic seeds.
3. Runs PLS with:
   - neighborhood range `1..=5`
   - deterministic mode enabled
   - 60s timeout

See [debug_lagos_30.rs](sims-heuristics/src/bin/debug_lagos_30.rs#L1).

## 11) Practical notes / gotchas

- PLS currently collects the iterator’s neighbors into a `Vec` per solution before evaluation (see `neighbors: Vec<_> = ...collect()` in [explore_population_neighborhoods](sims-heuristics/src/pareto_local_search.rs#L179)). This is simple but can increase memory pressure compared to streaming evaluation.
- Residual enumeration is capped to subset size ≤ 5 ([ResidualProblem](sims-heuristics/src/residual_problem.rs#L69)), which is a major tractability heuristic.
- Validity checking is explicit and aggressive in debug builds (`is_valid()` is called for each neighbor in [process_neighbor](sims-heuristics/src/pareto_local_search.rs#L257)).

---

If you want, I can also add an ASCII sequence diagram of the exact call flow from `run()` → `step()` → `neighborhood_iter_impl()` → `ResidualProblem::solve_next()` → merge → Pareto insertion.
