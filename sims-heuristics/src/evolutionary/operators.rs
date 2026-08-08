//! Genetic Operators for Set-Cover Evolutionary Algorithms
//!
//! This module provides crossover, mutation, and repair operators specifically designed
//! for the SIMS problem (a constrained set-cover problem). All operators maintain or
//! restore feasibility: every solution must cover all elements in the universe.
//!
//! ## Operators
//!
//! ### Crossover
//! - **Uniform crossover**: Each image is independently inherited from either parent,
//!   followed by greedy repair to restore coverage.
//! - **Coverage-biased crossover**: Preferentially keeps images that uniquely cover
//!   elements in their respective parent, producing leaner offspring.
//!
//! ### Mutation
//! - **Swap mutation**: Removes a randomly chosen (non-critical) image and greedily
//!   adds another to restore coverage.
//! - **Shift mutation**: Replaces an image with the best alternative covering similar
//!   elements, guided by objective deltas.
//! - **Add-then-prune mutation**: Adds a random image and then removes any images
//!   made redundant by the addition.
//!
//! ### Repair
//! - **Greedy repair**: Adds cheapest-first images until all elements are covered.
//! - **Redundancy removal**: Iteratively removes images whose elements are all
//!   covered by other selected images (most expensive first).

use fixedbitset::FixedBitSet;
use rand::Rng;
use rand::RngCore;
use rand::rngs::SmallRng;
use rand::seq::IteratorRandom;

use crate::objective_tracker::{ProvenSafeTrackerArray, TrackerCollection};
use crate::problem::SetCoverProblem;
use crate::solution::ImageSet;
use crate::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;

// ---------------------------------------------------------------------------
// Repair operators
// ---------------------------------------------------------------------------

/// Greedy repair: adds images until all elements are covered.
///
/// Images are selected by choosing a random uncovered element, then picking the
/// image that covers the most *currently uncovered* elements among those that
/// cover the chosen element. Ties are broken by lower image index (deterministic)
/// or randomly.
///
/// This is the most critical operator: every crossover and most mutations call it.
/// When the problem exposes `image_bitset()`, bulk `FixedBitSet` word ops replace
/// element-by-element iteration (O(N/64) vs O(N) for coverage updates and gain
/// computation).
pub fn greedy_repair<P, const D: usize>(
    selected: &mut FixedBitSet,
    problem: &P,
    trackers: &mut ProvenSafeTrackerArray<D>,
) where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    // Ensure the bitset can hold all image indices (it may have been
    // created via `collect()` which sizes to max_value+1, not num_images).
    selected.grow(problem.num_images());

    let universe_size = problem.num_elements();
    let num_images = problem.num_images();

    // Build covered-elements bitset from current selection.
    // Fast path: union_with is a SIMD word-level OR, O(universe_size/64).
    let mut covered = FixedBitSet::with_capacity(universe_size);
    for img in selected.ones() {
        if let Some(bs) = problem.image_bitset(img) {
            covered.union_with(bs);
        } else {
            for elem in problem.image_elements(img) {
                covered.insert(elem);
            }
        }
    }

    // Add images until every element is covered, choosing each addition with the
    // same objective-aware ranking PLS uses: `best_unselected_images_with_trackers`
    // prefers the cheapest scaled cost+cloud delta among images that cover a
    // currently-uncovered element (equal weights, deterministic — no per-repair
    // RNG, so runs stay seed-reproducible). `trackers` is kept in sync as we add.
    while covered.count_ones(..) < universe_size {
        // `covered` has capacity == universe_size, so its zero bits are exactly
        // the uncovered element indices.
        let uncovered: Vec<usize> = covered.zeroes().collect();

        let chosen = BitsetEncodedSolution::<P, D>::best_unselected_images_with_trackers(
            selected, trackers, &uncovered, problem, true,
        )
        .and_then(|imgs| imgs.into_iter().next())
        .or_else(|| {
            // Fallback: any unselected image covering some uncovered element.
            uncovered.iter().find_map(|&e| {
                problem
                    .element_images(e)
                    .find(|&c| c < num_images && !selected.contains(c))
            })
        });

        let Some(img) = chosen else {
            break; // remaining elements are uncoverable (malformed instance)
        };

        selected.insert(img);
        trackers.track_image_addition(img, problem);
        if let Some(bs) = problem.image_bitset(img) {
            covered.union_with(bs);
        } else {
            for elem in problem.image_elements(img) {
                covered.insert(elem);
            }
        }
    }
}

/// Remove redundant images, worst-objective-first.
///
/// An image is *redundant* if every element it covers is also covered by at least
/// one other selected image. Removing redundant images improves cost-related
/// objectives without violating feasibility.
///
/// Redundant images are dropped in the order given by PLS's objective-aware
/// `worst_selected_images_with_trackers` ranking (least scaled cost+cloud value
/// per covered element first), so the least useful images go first. `trackers`
/// is kept in sync with each removal.
///
/// Fast path (when `image_bitset` is available): the redundancy check uses a
/// `doubly_covered` `FixedBitSet` and `is_subset`, replacing the per-element count
/// loop with a single O(universe/64) bitset scan.
pub fn remove_redundant_images<P, const D: usize>(
    selected: &mut FixedBitSet,
    problem: &P,
    trackers: &mut ProvenSafeTrackerArray<D>,
) where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    selected.grow(problem.num_images());

    let universe_size = problem.num_elements();

    // Build per-element coverage counts and a doubly_covered bitset.
    // doubly_covered[e] = true iff element e is covered by ≥2 selected images.
    // We use coverage_count for correct incremental updates when images are removed,
    // and doubly_covered for the fast is_subset redundancy check.
    let mut coverage_count = vec![0u32; universe_size];
    let mut doubly_covered = FixedBitSet::with_capacity(universe_size);

    for img in selected.ones() {
        if let Some(bs) = problem.image_bitset(img) {
            for elem in bs.ones() {
                coverage_count[elem] += 1;
                if coverage_count[elem] >= 2 {
                    doubly_covered.insert(elem);
                }
            }
        } else {
            for elem in problem.image_elements(img) {
                coverage_count[elem] += 1;
                if coverage_count[elem] >= 2 {
                    doubly_covered.insert(elem);
                }
            }
        }
    }

    // Worst-first objective order over all currently selected images (equal weights,
    // deterministic). Computed once; coverage_count is updated as removals happen so
    // later redundancy checks see the shrinking selection.
    let num_selected = selected.count_ones(..);
    let candidates = BitsetEncodedSolution::<P, D>::worst_selected_images_with_trackers(
        selected,
        trackers,
        problem,
        true,
        num_selected,
    );

    for img in candidates {
        if !selected.contains(img) {
            continue;
        }

        // Fast redundancy check: image is redundant iff its coverage ⊆ doubly_covered.
        let is_redundant = problem.image_bitset(img).map_or_else(
            || {
                problem
                    .image_elements(img)
                    .all(|elem| coverage_count[elem] >= 2)
            },
            |bs| bs.is_subset(&doubly_covered),
        );

        if is_redundant {
            selected.toggle(img);
            trackers.track_image_removal(img, problem);
            // Update coverage_count and doubly_covered for the removed image.
            if let Some(bs) = problem.image_bitset(img) {
                for elem in bs.ones() {
                    coverage_count[elem] -= 1;
                    if coverage_count[elem] < 2 {
                        doubly_covered.set(elem, false);
                    }
                }
            } else {
                for elem in problem.image_elements(img) {
                    coverage_count[elem] -= 1;
                    if coverage_count[elem] < 2 {
                        doubly_covered.set(elem, false);
                    }
                }
            }
        }
    }
}

/// Full repair pipeline: objective-aware greedy cover then worst-first redundancy
/// removal.
///
/// Builds one tracker synced to `selected` and threads it through both
/// phases, so they reuse PLS's exact image-ranking primitives
/// (`best_unselected_images_with_trackers` / `worst_selected_images_with_trackers`).
///
/// Repair is deterministic given the selection (equal objective weights, no RNG),
/// so the `_rng` argument is retained only for signature compatibility with the
/// many crossover/mutation call sites.
/// Returns the final objective values of the repaired selection (from the synced
/// tracker), so callers can build the solution without a second
/// `recalculate_objectives` pass.
pub fn repair_and_prune<P, const D: usize>(
    selected: &mut FixedBitSet,
    problem: &P,
    _rng: &mut SmallRng,
) -> [u64; D]
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    let mut trackers = BitsetEncodedSolution::<P, D>::synced_trackers(selected, problem);
    greedy_repair::<P, D>(selected, problem, &mut trackers);
    remove_redundant_images::<P, D>(selected, problem, &mut trackers);
    trackers.values()
}

// ---------------------------------------------------------------------------
// Crossover operators
// ---------------------------------------------------------------------------

/// Uniform crossover: each image is independently inherited from parent A or B
/// with equal probability (0.5). The offspring is then repaired and pruned.
pub fn uniform_crossover<P, const D: usize>(
    parent_a: &BitsetEncodedSolution<P, D>,
    parent_b: &BitsetEncodedSolution<P, D>,
    problem: &P,
    rng: &mut SmallRng,
) -> BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    let num_images = problem.num_images();
    let a = &parent_a.selected_images;
    let b = &parent_b.selected_images;

    // Random mask: one RNG word per block (vs one `random_bool` per differing
    // image in the scalar version). `next_u64` avoids any distribution bound.
    let num_blocks = a.as_slice().len();
    let mask = FixedBitSet::with_capacity_and_blocks(
        num_images,
        (0..num_blocks).map(|_| rng.next_u64() as usize),
    );

    // Vectorised uniform crossover via FixedBitSet's (SIMD-accelerated) bitwise
    // operators — four binary ops:
    //   child = (a & b) | ((a ^ b) & mask)
    // Bits where the parents agree are inherited directly; each differing bit is
    // taken from the random mask (50/50), matching the per-image semantics.
    // Both parents have their unused trailing bits cleared, so `(a ^ b) & mask`
    // is 0 in the tail regardless of the mask — the child stays valid even
    // though `mask`'s final block may hold random high bits.
    let mut child_selected = &(a & b) | &(&(a ^ b) & &mask);

    let objectives = repair_and_prune::<P, D>(&mut child_selected, problem, rng);

    BitsetEncodedSolution::from_selected_ones_with_objectives(
        child_selected.ones(),
        objectives,
        problem,
    )
}

/// Coverage-biased crossover.
///
/// Builds offspring by first taking the *intersection*
/// of both parents (images selected in both), then greedily adding images from
/// the *symmetric difference* (images in exactly one parent) to cover the remaining
/// elements, preferring images that cover more uncovered elements.
///
/// This tends to produce offspring that are smaller (fewer images) than either parent.
pub fn coverage_biased_crossover<P, const D: usize>(
    parent_a: &BitsetEncodedSolution<P, D>,
    parent_b: &BitsetEncodedSolution<P, D>,
    problem: &P,
    rng: &mut SmallRng,
) -> BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    let num_images = problem.num_images();
    let universe_size = problem.num_elements();

    // Start with intersection (images in both parents)
    let mut child_selected = FixedBitSet::with_capacity(num_images);
    for img in 0..num_images {
        if parent_a.is_image_selected(img) && parent_b.is_image_selected(img) {
            child_selected.insert(img);
        }
    }

    // Build covered elements from intersection
    let mut covered = FixedBitSet::with_capacity(universe_size);
    for img in child_selected.ones() {
        for elem in problem.image_elements(img) {
            covered.insert(elem);
        }
    }

    // Collect symmetric difference candidates (in exactly one parent)
    let mut sym_diff: Vec<usize> = (0..num_images)
        .filter(|&img| parent_a.is_image_selected(img) ^ parent_b.is_image_selected(img))
        .collect();

    // Greedily add from symmetric difference by max uncovered coverage
    while covered.count_ones(..) < universe_size && !sym_diff.is_empty() {
        // Score each candidate
        let mut best_idx = 0;
        let mut best_gain = 0usize;
        for (i, &img) in sym_diff.iter().enumerate() {
            let gain = problem
                .image_elements(img)
                .filter(|&e| !covered.contains(e))
                .count();
            if gain > best_gain {
                best_gain = gain;
                best_idx = i;
            }
        }

        if best_gain == 0 {
            break; // remaining candidates don't help
        }

        let img = sym_diff.swap_remove(best_idx);
        child_selected.insert(img);
        for elem in problem.image_elements(img) {
            covered.insert(elem);
        }
    }

    // Repair anything still uncovered (could happen if parents together don't cover)
    let objectives = repair_and_prune::<P, D>(&mut child_selected, problem, rng);

    BitsetEncodedSolution::from_selected_ones_with_objectives(
        child_selected.ones(),
        objectives,
        problem,
    )
}

// ---------------------------------------------------------------------------
// Mutation operators
// ---------------------------------------------------------------------------

/// Swap mutation: removes one randomly chosen non-critical image and then repairs.
///
/// A *non-critical* image is one whose removal doesn't leave any element uncovered
/// (i.e., it is redundant), OR one that can be replaced by other images covering
/// the same elements. We attempt removal regardless and let repair fix any breakage.
///
/// `mutation_rate` is the probability of performing the mutation at all.
pub fn swap_mutation<P, const D: usize>(
    solution: &BitsetEncodedSolution<P, D>,
    problem: &P,
    rng: &mut SmallRng,
    mutation_rate: f64,
) -> BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    if !rng.random_bool(mutation_rate) {
        return solution.clone();
    }

    let mut child_selected = solution.selected_images.clone();
    child_selected.grow(problem.num_images());
    let selected_count = child_selected.count_ones(..);
    if selected_count <= 1 {
        return solution.clone();
    }

    // Pick a random selected image to remove
    if let Some(img_to_remove) = child_selected.ones().choose(rng) {
        child_selected.toggle(img_to_remove);
    }

    let objectives = repair_and_prune::<P, D>(&mut child_selected, problem, rng);

    BitsetEncodedSolution::from_selected_ones_with_objectives(
        child_selected.ones(),
        objectives,
        problem,
    )
}

/// Multi-swap mutation: removes `k` random images and re-repairs.
/// More disruptive than single swap; good for escaping local optima.
pub fn multi_swap_mutation<P, const D: usize>(
    solution: &BitsetEncodedSolution<P, D>,
    problem: &P,
    rng: &mut SmallRng,
    mutation_rate: f64,
    max_removals: usize,
) -> BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    if !rng.random_bool(mutation_rate) {
        return solution.clone();
    }

    let mut child_selected = solution.selected_images.clone();
    child_selected.grow(problem.num_images());
    let selected_count = child_selected.count_ones(..);
    if selected_count <= 1 {
        return solution.clone();
    }

    let k = rng.random_range(1..=max_removals.min(selected_count - 1));

    // Remove k random selected images
    let to_remove: Vec<usize> = child_selected.ones().choose_multiple(rng, k);
    for img in to_remove {
        child_selected.toggle(img);
    }

    let objectives = repair_and_prune::<P, D>(&mut child_selected, problem, rng);

    BitsetEncodedSolution::from_selected_ones_with_objectives(
        child_selected.ones(),
        objectives,
        problem,
    )
}

/// Add-then-prune mutation: adds a random unselected image, then removes any
/// images made redundant. This can discover solutions with the same coverage
/// but different objective trade-offs.
pub fn add_then_prune_mutation<P, const D: usize>(
    solution: &BitsetEncodedSolution<P, D>,
    problem: &P,
    rng: &mut SmallRng,
    mutation_rate: f64,
) -> BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    if !rng.random_bool(mutation_rate) {
        return solution.clone();
    }

    let mut child_selected = solution.selected_images.clone();
    child_selected.grow(problem.num_images());

    // Pick a random unselected image to add
    if let Some(img_to_add) = child_selected.zeroes().choose(rng) {
        child_selected.insert(img_to_add);
    }

    let mut trackers = BitsetEncodedSolution::<P, D>::synced_trackers(&child_selected, problem);
    remove_redundant_images::<P, D>(&mut child_selected, problem, &mut trackers);
    let objectives = trackers.values();

    BitsetEncodedSolution::from_selected_ones_with_objectives(
        child_selected.ones(),
        objectives,
        problem,
    )
}

/// Coverage-guided shift mutation.
///
/// Removes a randomly chosen selected image and
/// replaces it with the unselected image that covers the most elements that would
/// become exposed (uncovered) by the removal, while also covering the most
/// additional *already-covered* elements (to maximize overlap and enable further
/// pruning). This is more directed than plain swap mutation because it considers
/// coverage structure when choosing the replacement.
///
/// After replacement, redundancy removal may eliminate other images that became
/// redundant due to the new image's coverage, effectively "shifting" the solution
/// in objective space.
///
/// Fast path: `exposed_elements` is a `FixedBitSet`, and candidate scoring uses
/// `intersection_count` + `count_ones` instead of linear `Vec::contains` search.
pub fn shift_mutation<P, const D: usize>(
    solution: &BitsetEncodedSolution<P, D>,
    problem: &P,
    rng: &mut SmallRng,
    mutation_rate: f64,
) -> BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    if !rng.random_bool(mutation_rate) {
        return solution.clone();
    }

    let mut child_selected = solution.selected_images.clone();
    child_selected.grow(problem.num_images());
    let selected_count = child_selected.count_ones(..);
    if selected_count <= 1 {
        return solution.clone();
    }

    // Pick a random selected image to remove
    let Some(img_to_remove) = child_selected.ones().choose(rng) else {
        return solution.clone();
    };

    // Collect elements that would become exposed (uncovered) if we remove this image.
    // exposed_elements is a FixedBitSet for O(1) lookup in the scoring loop.
    let mut coverage_count = vec![0u32; problem.num_elements()];
    for img in child_selected.ones() {
        if let Some(bs) = problem.image_bitset(img) {
            for elem in bs.ones() {
                coverage_count[elem] += 1;
            }
        } else {
            for elem in problem.image_elements(img) {
                coverage_count[elem] += 1;
            }
        }
    }

    let mut exposed_elements = FixedBitSet::with_capacity(problem.num_elements());
    if let Some(bs) = problem.image_bitset(img_to_remove) {
        for elem in bs.ones() {
            if coverage_count[elem] == 1 {
                exposed_elements.insert(elem);
            }
        }
    } else {
        for elem in problem.image_elements(img_to_remove) {
            if coverage_count[elem] == 1 {
                exposed_elements.insert(elem);
            }
        }
    }

    child_selected.toggle(img_to_remove);

    // Find the best replacement among unselected images:
    // 1. Primary: maximize coverage of exposed elements (restore feasibility)
    // 2. Secondary: maximize total element coverage (create more redundancy for pruning)
    // Fast path: use intersection_count + count_ones instead of per-element loops.
    let mut best_candidate: Option<usize> = None;
    let mut best_exposed_coverage = 0usize;
    let mut best_total_coverage = 0usize;

    for candidate in child_selected.zeroes() {
        if candidate == img_to_remove {
            continue;
        }

        let (exposed_coverage, total_coverage) = problem.image_bitset(candidate).map_or_else(
            || {
                let mut exp = 0usize;
                let mut tot = 0usize;
                for elem in problem.image_elements(candidate) {
                    tot += 1;
                    if exposed_elements.contains(elem) {
                        exp += 1;
                    }
                }
                (exp, tot)
            },
            |bs| (bs.intersection_count(&exposed_elements), bs.count_ones(..)),
        );

        if exposed_coverage > best_exposed_coverage
            || (exposed_coverage == best_exposed_coverage && total_coverage > best_total_coverage)
        {
            best_exposed_coverage = exposed_coverage;
            best_total_coverage = total_coverage;
            best_candidate = Some(candidate);
        }
    }

    if let Some(replacement) = best_candidate {
        child_selected.insert(replacement);
    }

    // Repair and prune to ensure feasibility and leanness
    let objectives = repair_and_prune::<P, D>(&mut child_selected, problem, rng);

    BitsetEncodedSolution::from_selected_ones_with_objectives(
        child_selected.ones(),
        objectives,
        problem,
    )
}

/// Ensures at least one mutation is applied. If `mutated` is identical to `original`
/// (same selected images), forces a swap mutation with rate 1.0.
pub fn ensure_mutated<P, const D: usize>(
    original: &BitsetEncodedSolution<P, D>,
    mutated: BitsetEncodedSolution<P, D>,
    problem: &P,
    rng: &mut SmallRng,
) -> BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    if original.selected_images == mutated.selected_images {
        // No mutation took effect — force a swap
        swap_mutation(&mutated, problem, rng, 1.0)
    } else {
        mutated
    }
}

/// Bit-flip mutation: each image independently flips its selection status with
/// probability `per_bit_rate`. The solution is then repaired and pruned.
///
/// This is the most standard GA mutation but adapted for set-cover feasibility.
pub fn bitflip_mutation<P, const D: usize>(
    solution: &BitsetEncodedSolution<P, D>,
    problem: &P,
    rng: &mut SmallRng,
    per_bit_rate: f64,
) -> BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    let num_images = problem.num_images();

    // Nothing can flip — return the original untouched (keeps cached objectives).
    if per_bit_rate <= 0.0 || num_images == 0 {
        return solution.clone();
    }

    let mut child_selected = solution.selected_images.clone();
    child_selected.grow(num_images);

    let any_flip = if per_bit_rate >= 1.0 {
        for img in 0..num_images {
            child_selected.toggle(img);
        }
        num_images > 0
    } else {
        // Geometric skip-sampling. Each image flips independently with probability
        // `per_bit_rate`, so the number of images kept before the next flip is
        // Geometric(p): for u ~ Uniform(0,1], gap = floor(ln(u) / ln(1-p)). We jump
        // straight to each flipped image, using one RNG draw per *flip* (≈ N·p in
        // expectation) instead of one Bernoulli draw per image.
        let log_keep = (1.0 - per_bit_rate).ln(); // < 0 for 0 < p < 1
        let mut pos: usize = 0;
        let mut flipped = false;
        loop {
            // u ∈ (0, 1] avoids ln(0) = -inf (random::<f64>() yields [0, 1)).
            let u = 1.0 - rng.random::<f64>();
            let gap = (u.ln() / log_keep).floor() as usize;
            pos = pos.saturating_add(gap);
            if pos >= num_images {
                break;
            }
            child_selected.toggle(pos);
            flipped = true;
            pos += 1;
        }
        flipped
    };

    if !any_flip {
        return solution.clone();
    }

    let objectives = repair_and_prune::<P, D>(&mut child_selected, problem, rng);

    BitsetEncodedSolution::from_selected_ones_with_objectives(
        child_selected.ones(),
        objectives,
        problem,
    )
}

// ---------------------------------------------------------------------------
// Population initialization
// ---------------------------------------------------------------------------

/// Generate an initial population of `size` random feasible solutions.
pub fn random_population<P, const D: usize>(
    problem: &P,
    size: usize,
    rng: &mut SmallRng,
) -> Vec<BitsetEncodedSolution<P, D>>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    (0..size)
        .map(|_| {
            let seed: u64 = rng.random();
            BitsetEncodedSolution::random_with_seed(problem, seed)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Dominance utilities
// ---------------------------------------------------------------------------

/// Compute the non-dominated rank (front index) for each solution in the population.
///
/// Returns a `Vec<usize>` of the same length as `population`, where entry `i` is the
/// front index of solution `i` (0 = first Pareto front, 1 = second, …).
///
/// Uses the standard fast non-dominated sorting from Deb et al. (2002).
#[must_use]
pub fn fast_non_dominated_sort<P, const D: usize>(
    population: &[BitsetEncodedSolution<P, D>],
) -> Vec<Vec<usize>>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    use pareto::{HasObjectives, MoSolution};

    let n = population.len();
    // domination_count[i] = number of solutions that dominate solution i
    let mut domination_count = vec![0usize; n];
    // dominated_set[i] = set of solution indices that solution i dominates
    let mut dominated_set: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        for j in (i + 1)..n {
            if population[i].dominates(population[j].objectives()) {
                dominated_set[i].push(j);
                domination_count[j] += 1;
            } else if population[j].dominates(population[i].objectives()) {
                dominated_set[j].push(i);
                domination_count[i] += 1;
            }
        }
    }

    let mut fronts: Vec<Vec<usize>> = Vec::new();

    // First front: solutions with domination_count == 0
    let mut current_front: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();

    while !current_front.is_empty() {
        let mut next_front: Vec<usize> = Vec::new();
        for &i in &current_front {
            for &j in &dominated_set[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        fronts.push(std::mem::take(&mut current_front));
        current_front = next_front;
    }

    fronts
}

/// Compute crowding distance for a set of solution indices within one front.
///
/// Returns a `Vec<f64>` of the same length as `front`, giving the crowding distance
/// for each solution in the front. Boundary solutions receive `f64::INFINITY`.
#[must_use]
pub fn crowding_distance<P, const D: usize>(
    population: &[BitsetEncodedSolution<P, D>],
    front: &[usize],
) -> Vec<f64>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    use pareto::HasObjectives;

    let n = front.len();
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }

    let mut distances = vec![0.0f64; n];

    for obj_idx in 0..D {
        // Sort front indices by this objective
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| {
            let oa = population[front[a]].objectives()[obj_idx];
            let ob = population[front[b]].objectives()[obj_idx];
            oa.cmp(&ob)
        });

        // Boundary solutions get infinity
        distances[sorted_indices[0]] = f64::INFINITY;
        distances[sorted_indices[n - 1]] = f64::INFINITY;

        let f_min = population[front[sorted_indices[0]]].objectives()[obj_idx] as f64;
        let f_max = population[front[sorted_indices[n - 1]]].objectives()[obj_idx] as f64;
        let range = f_max - f_min;

        if range < f64::EPSILON {
            continue; // all solutions have the same value for this objective
        }

        for i in 1..(n - 1) {
            let prev_val = population[front[sorted_indices[i - 1]]].objectives()[obj_idx] as f64;
            let next_val = population[front[sorted_indices[i + 1]]].objectives()[obj_idx] as f64;
            distances[sorted_indices[i]] += (next_val - prev_val) / range;
        }
    }

    distances
}

/// Compute the Tchebycheff scalarized value for a solution given a weight vector
/// and an ideal point (component-wise minimum across population).
///
/// `tchebycheff(x, w, z*) = max_i { w_i * |f_i(x) - z*_i| }`
///
/// Lower is better.
#[must_use]
pub fn tchebycheff_value<const D: usize>(
    objectives: &[u64; D],
    weight: &[f64; D],
    ideal_point: &[f64; D],
) -> f64 {
    let mut max_val = f64::NEG_INFINITY;
    for i in 0..D {
        // For minimization: f_i - z*_i >= 0
        let diff = (objectives[i] as f64) - ideal_point[i];
        // Use max(w_i, epsilon) to avoid division-by-zero-like issues in flat regions
        let w = if weight[i] < 1e-6 { 1e-6 } else { weight[i] };
        let val = w * diff.abs();
        if val > max_val {
            max_val = val;
        }
    }
    max_val
}

/// Compute the weighted-sum scalarized value for a solution.
///
/// `ws(x, w) = sum_i { w_i * f_i(x) }`
///
/// Lower is better (all objectives are minimization).
#[must_use]
pub fn weighted_sum_value<const D: usize>(objectives: &[u64; D], weight: &[f64; D]) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..D {
        sum += weight[i] * (objectives[i] as f64);
    }
    sum
}

/// Generate `n` uniformly distributed weight vectors for D objectives using the
/// simplex-lattice design (Das & Dennis, 1998).
///
/// For D=2 this produces weights along the line w1+w2=1.
/// For D>=3 it produces points on the (D-1)-simplex with `n` divisions per axis.
///
/// The number of generated vectors equals C(n + D - 1, D - 1).
#[must_use]
pub fn generate_weight_vectors<const D: usize>(num_divisions: usize) -> Vec<[f64; D]> {
    let mut weights: Vec<[f64; D]> = Vec::new();
    let mut current = [0.0f64; D];
    generate_weights_recursive::<D>(&mut weights, &mut current, 0, num_divisions, num_divisions);
    weights
}

fn generate_weights_recursive<const D: usize>(
    weights: &mut Vec<[f64; D]>,
    current: &mut [f64; D],
    depth: usize,
    remaining: usize,
    total: usize,
) {
    if depth == D - 1 {
        current[depth] = remaining as f64 / total as f64;
        weights.push(*current);
        return;
    }

    for i in 0..=remaining {
        current[depth] = i as f64 / total as f64;
        generate_weights_recursive::<D>(weights, current, depth + 1, remaining - i, total);
    }
}

/// Compute the ideal point (component-wise minimum) from a population.
#[must_use]
pub fn compute_ideal_point<P, const D: usize>(
    population: &[BitsetEncodedSolution<P, D>],
) -> [f64; D]
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    use pareto::HasObjectives;

    let mut ideal = [f64::INFINITY; D];
    for sol in population {
        for (ideal_i, &obj) in ideal.iter_mut().zip(sol.objectives().iter()) {
            let val = obj as f64;
            if val < *ideal_i {
                *ideal_i = val;
            }
        }
    }
    ideal
}

/// Binary tournament selection: picks two random individuals and returns the index
/// of the one with better (lower) rank, breaking ties by higher crowding distance.
pub fn binary_tournament(
    ranks: &[usize],
    crowding: &[f64],
    rng: &mut SmallRng,
    pop_size: usize,
) -> usize {
    let a = rng.random_range(0..pop_size);
    let b = rng.random_range(0..pop_size);

    if ranks[a] < ranks[b] {
        a
    } else if ranks[b] < ranks[a] {
        b
    } else if crowding[a] > crowding[b] {
        a
    } else if crowding[b] > crowding[a] {
        b
    } else if rng.random_bool(0.5) {
        a
    } else {
        b
    }
}

/// Compute per-solution rank array from the front structure returned by
/// `fast_non_dominated_sort`.
#[must_use]
pub fn ranks_from_fronts(fronts: &[Vec<usize>], pop_size: usize) -> Vec<usize> {
    let mut ranks = vec![0usize; pop_size];
    for (rank, front) in fronts.iter().enumerate() {
        for &idx in front {
            ranks[idx] = rank;
        }
    }
    ranks
}

/// Compute per-solution crowding distance array from the front structure.
#[must_use]
pub fn crowding_from_fronts<P, const D: usize>(
    population: &[BitsetEncodedSolution<P, D>],
    fronts: &[Vec<usize>],
) -> Vec<f64>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    let pop_size = population.len();
    let mut distances = vec![0.0f64; pop_size];
    for front in fronts {
        let front_distances = crowding_distance(population, front);
        for (local_idx, &global_idx) in front.iter().enumerate() {
            distances[global_idx] = front_distances[local_idx];
        }
    }
    distances
}

/// Euclidean distance between two weight vectors (used for MOEA/D neighbourhood).
#[must_use]
pub fn weight_distance<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..D {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum.sqrt()
}

/// For each weight vector, compute the indices of its `T` closest neighbours.
///
/// # Panics
///
/// Panics if any pairwise weight distance is NaN (weights are expected finite).
#[must_use]
pub fn compute_neighbourhoods<const D: usize>(
    weights: &[[f64; D]],
    neighbourhood_size: usize,
) -> Vec<Vec<usize>> {
    let n = weights.len();
    let t = neighbourhood_size.min(n);
    weights
        .iter()
        .map(|wi| {
            let mut dists: Vec<(usize, f64)> = weights
                .iter()
                .enumerate()
                .map(|(j, wj)| (j, weight_distance(wi, wj)))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.into_iter().take(t).map(|(idx, _)| idx).collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_generation_2d() {
        let weights = generate_weight_vectors::<2>(10);
        assert_eq!(weights.len(), 11); // C(10+1, 1) = 11
        for w in &weights {
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "weights must sum to 1.0");
        }
        // First and last should be boundary weights
        assert!((weights[0][0] - 0.0).abs() < 1e-9);
        assert!((weights[0][1] - 1.0).abs() < 1e-9);
        assert!((weights[10][0] - 1.0).abs() < 1e-9);
        assert!((weights[10][1] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_weight_generation_3d() {
        let weights = generate_weight_vectors::<3>(5);
        // C(5+2, 2) = 21
        assert_eq!(weights.len(), 21);
        for w in &weights {
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "weights must sum to 1.0");
        }
    }

    #[test]
    fn test_tchebycheff() {
        let objectives = [100u64, 200u64];
        let weight = [0.5, 0.5];
        let ideal = [50.0, 100.0];
        let val = tchebycheff_value(&objectives, &weight, &ideal);
        // max(0.5 * |100-50|, 0.5 * |200-100|) = max(25, 50) = 50
        assert!((val - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_sum() {
        let objectives = [100u64, 200u64];
        let weight = [0.3, 0.7];
        let val = weighted_sum_value(&objectives, &weight);
        // 0.3*100 + 0.7*200 = 30 + 140 = 170
        assert!((val - 170.0).abs() < 1e-9);
    }

    #[test]
    fn test_neighbourhood_computation() {
        let weights = generate_weight_vectors::<2>(4);
        let neighbourhoods = compute_neighbourhoods(&weights, 3);
        assert_eq!(neighbourhoods.len(), weights.len());
        for nb in &neighbourhoods {
            assert_eq!(nb.len(), 3);
        }
        // Each weight's closest neighbour should be itself (distance 0)
        for (i, nb) in neighbourhoods.iter().enumerate() {
            assert_eq!(nb[0], i, "closest neighbour should be self");
        }
    }

    #[test]
    fn test_ranks_from_fronts() {
        let fronts = vec![vec![0, 2], vec![1, 3], vec![4]];
        let ranks = ranks_from_fronts(&fronts, 5);
        assert_eq!(ranks, vec![0, 1, 0, 1, 2]);
    }
}
