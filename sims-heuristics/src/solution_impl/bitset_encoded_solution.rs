// Bitset-based solution implementation
// This module is only compiled when the "bitmaps" feature is enabled.

use fixedbitset::FixedBitSet;
#[cfg(feature = "bitmaps")]
use pareto::Objectives;
use pareto::{HasObjectives, MoSolution, Random};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::{Rng, seq::IteratorRandom};
use std::{collections::BinaryHeap, fmt::Debug, hash::Hash, time::Duration};

use crate::objective_tracker::{ObjectiveTracker, TrackerCollection};
use crate::objective_tracker_impl::proven_safe_trackers::ProvenSafeTrackerArray;
use crate::problem::{ComparableImage, ImageObjectiveDeltas, ScaledObjectiveDeltas};
use crate::residual_problem::ResidualProblem;
use crate::residual_solution::ResidualSolution;
use crate::pls_config::PlsOptimizations;
use crate::solution::{ImageSet, MergeableWithResidual, SIMSCore, SIMSModifiable, SIMSSolution};
use crate::solution_set_impl::NdTreeSolutionSet;
use crate::timer::Timer;
use nd_tree::nd_tree::NDTreeSolutionIntoIterator;
use crate::{SetCoverProblem, objectives};

use itertools::Itertools;

/// A temporary state representing a solution with specific images removed.
/// Used as an intermediate step in creating a `ResidualProblem`.
/// Uses `FixedBitSet` for efficient storage and set operations.
pub struct UndercoveredSolution<const D: usize> {
    pub partial_selected_images: FixedBitSet,
    pub removed_images: FixedBitSet,
    pub uncovered_elements: FixedBitSet,
    pub partial_trackers: ProvenSafeTrackerArray<D>,
}

/// Lightweight view for `ImageSet` operations - only used for tracker `peek_delta` calls
struct ImageSetView<'a> {
    selected_images: &'a FixedBitSet,
}

impl<const D: usize> crate::solution::ImageSet<D> for ImageSetView<'_> {
    fn selected_images(&self) -> impl Iterator<Item = usize> {
        self.selected_images.ones()
    }

    fn unselected_images(&self) -> impl Iterator<Item = usize> {
        self.selected_images.zeroes()
    }

    fn is_image_selected(&self, image_index: usize) -> bool {
        self.selected_images[image_index]
    }

    fn num_selected_images(&self) -> usize {
        self.selected_images.count_ones(..)
    }

    fn set_image(&mut self, _image_index: usize, _selected: bool) {
        panic!("ImageSetView is read-only")
    }
}

#[cfg(feature = "bitmaps")]
#[derive(Clone)]
pub struct BitsetEncodedSolution<P, const D: usize>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    pub selected_images: FixedBitSet,
    pub objectives: Objectives<D>,
    pub timestamp: Duration,
    _phantom: std::marker::PhantomData<P>,
}

// Iterator types for BitsetEncodedSolution - leverage FixedBitSet's built-in iterators
pub type BitsetSelectedImagesIter<'a> = fixedbitset::Ones<'a>;
pub type BitsetUnselectedImagesIter<'a> = fixedbitset::Zeroes<'a>;

impl<P, const D: usize> HasObjectives<D> for BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    fn objectives(&self) -> &pareto::Objectives<D> {
        &self.objectives
    }
}

impl<P, const D: usize> MoSolution<D> for BitsetEncodedSolution<P, D> where
    P: SetCoverProblem<D> + Clone + Send + Sync
{
}

// Implement ImageSet<D> trait
impl<P, const D: usize> ImageSet<D> for BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    fn selected_images(&self) -> impl Iterator<Item = usize> {
        self.selected_images.ones()
    }

    fn unselected_images(&self) -> impl Iterator<Item = usize> {
        self.selected_images.zeroes()
    }

    fn is_image_selected(&self, image_index: usize) -> bool {
        self.selected_images[image_index]
    }

    fn num_selected_images(&self) -> usize {
        self.selected_images.count_ones(..)
    }

    fn set_image(&mut self, image_index: usize, selected: bool) {
        self.selected_images.set(image_index, selected);
    }
}

// Implement SIMSCore trait
impl<P, const D: usize> SIMSCore<P, D> for BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    fn to_debug_solution(&self) -> SIMSSolution {
        SIMSSolution {
            selected_images: self.selected_images.ones().collect(),
        }
    }

    fn objectives_mut(&mut self) -> &mut pareto::Objectives<D> {
        &mut self.objectives
    }

    fn greedy_initial_solutions(problem: &P) -> Vec<Self> {
        use tracing::debug;

        // Generate weight vectors for diverse initial solutions:
        //   D   single-objective vectors  (weight 1.0 on one objective)
        // + D*(D-1)/2 pairwise vectors    (weight 0.5 on each of two objectives)
        // + 1   balanced vector            (equal weight on all objectives)
        // For D=4 this gives 4 + 6 + 1 = 11 solutions instead of 5.
        let num_pairs = D * (D.saturating_sub(1)) / 2;
        let mut weight_vectors: Vec<[f64; D]> = Vec::with_capacity(D + num_pairs + 1);

        // Single-objective weight vectors
        for obj_idx in 0..D {
            let mut weights = [0.0f64; D];
            weights[obj_idx] = 1.0;
            weight_vectors.push(weights);
        }

        // Pairwise weight vectors: for each pair (i, j), weight 0.5 each
        for i in 0..D {
            for j in (i + 1)..D {
                let mut weights = [0.0f64; D];
                weights[i] = 0.5;
                weights[j] = 0.5;
                weight_vectors.push(weights);
            }
        }

        // Balanced weight vector
        let balanced = [1.0 / D as f64; D];
        weight_vectors.push(balanced);

        let mut solutions = Vec::with_capacity(weight_vectors.len());
        for (i, weights) in weight_vectors.iter().enumerate() {
            let solution = Self::greedy_construction(problem, *weights);
            debug!(
                "Greedy solution {i} (weights={weights:?}): objectives={:?}, images={}",
                solution.objectives,
                solution.num_selected_images()
            );
            solutions.push(solution);
        }
        solutions
    }
}

// Implement Random trait from pareto
impl<P, const D: usize> Random for BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    fn random() -> Self {
        panic!("BitsetEncodedSolution::random() needs a Problem parameter")
    }

    fn random_with_seed(_seed: u64) -> Self {
        panic!("BitsetEncodedSolution::random_with_seed() needs a Problem parameter")
    }
}

// Implement SIMSModifiable trait
impl<P, const D: usize> SIMSModifiable<P, D> for BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    type Trackers = ProvenSafeTrackerArray<D>;

    fn add_image(&mut self, image_index: usize, problem: &P, trackers: &mut Self::Trackers) {
        log::debug!("ADD_IMAGE: Starting addition of image {image_index}");
        log::debug!("  Current objectives: {:?}", self.objectives);

        // Debug: Validate inputs
        debug_assert!(
            image_index < problem.num_images(),
            "add_image: invalid image index {} (max: {})",
            image_index,
            problem.num_images()
        );
        debug_assert!(
            !self.selected_images[image_index],
            "add_image: trying to add already selected image {image_index}"
        );

        // Use trackers for delta calculation and state update
        let deltas = trackers.track_image_addition(image_index, problem);
        // Debug: Validate tracker deltas are reasonable
        debug_assert!(
            deltas.iter().all(|&d| d.abs() < (u64::MAX / 4) as i64),
            "add_image: unreasonable tracker delta for image {image_index}: {deltas:?}"
        );

        objectives::apply_delta(&mut self.objectives, &deltas);

        self.selected_images.set(image_index, true);

        log::debug!("ADD_IMAGE: Completed addition of image {image_index}");
        log::debug!("  Final objectives: {:?}", self.objectives);
    }

    fn remove_image(&mut self, image_index: usize, problem: &P, trackers: &mut Self::Trackers) {
        debug_assert!(
            self.selected_images[image_index],
            "remove_image: trying to remove unselected image {image_index}"
        );

        debug_assert!(
            self.are_objectives_valid(problem),
            "Objectives are invalid before removing image"
        );

        log::debug!("REMOVE_IMAGE: Starting removal of image {image_index}");
        log::debug!("  Current objectives: {:?}", self.objectives);
        log::debug!(
            "  Selected images before removal: {:?}",
            self.selected_images.ones().collect::<Vec<_>>()
        );
        log::debug!("  Image {image_index} IS selected: {}", self.selected_images[image_index]);

        // Use trackers for delta calculation and state update
        let deltas = trackers.track_image_removal(image_index, problem);
        log::debug!("  Deltas from tracker: {deltas:?}");
        objectives::apply_delta(&mut self.objectives, &deltas);

        self.selected_images.set(image_index, false);
        log::debug!("REMOVE_IMAGE: Completed removal of image {image_index}");
        log::debug!("  Final objectives: {:?}", self.objectives);
        log::debug!(
            "  Selected images after removal: {:?}",
            self.selected_images.ones().collect::<Vec<_>>()
        );

        debug_assert!(
            self.are_objectives_valid(problem),
            "Objectives are invalid after removing image"
        );
    }

    fn scaled_image_objective_deltas(
        &self,
        images: &[usize],
        problem: &P,
        trackers: &Self::Trackers,
    ) -> Vec<ScaledObjectiveDeltas<D>> {
        self.scaled_image_objective_deltas_impl(images.iter().copied(), problem, trackers)
    }

    fn find_best_image_to_add(&self, problem: &P, trackers: &Self::Trackers) -> Option<usize> {
        let unselected: Vec<usize> = self.selected_images.zeroes().collect();
        if unselected.is_empty() {
            return None;
        }

        // Greedy add - best unselected image according to some heuristic
        let scaled_objective_deltas =
            self.scaled_image_objective_deltas_impl(unselected.iter().copied(), problem, trackers);

        let min_index = (0..scaled_objective_deltas.len()).min_by(|&i, &j| {
            // Use first component of scaled objectives
            scaled_objective_deltas[i].scaled_deltas[0]
                .partial_cmp(&scaled_objective_deltas[j].scaled_deltas[0])
                .unwrap()
        })?;

        Some(scaled_objective_deltas[min_index].image_index)
    }

    fn find_best_image_to_remove(&self, problem: &P, trackers: &Self::Trackers) -> Option<usize> {
        let selected: Vec<usize> = self.selected_images.ones().collect();
        if selected.is_empty() {
            return None;
        }

        // Greedy remove - worst selected image according to some heuristic
        let scaled_objective_deltas =
            self.scaled_image_objective_deltas_impl(selected.iter().copied(), problem, trackers);

        let max_index = (0..scaled_objective_deltas.len()).max_by(|&i, &j| {
            // Use first component of scaled objectives
            scaled_objective_deltas[i].scaled_deltas[0]
                .partial_cmp(&scaled_objective_deltas[j].scaled_deltas[0])
                .unwrap()
        })?;

        Some(scaled_objective_deltas[max_index].image_index)
    }

    fn neighborhood(
        &self,
        k: u32,
        problem: &P,
        timer: &Timer,
        is_deterministic: bool,
        trackers: &mut Self::Trackers,
    ) -> Vec<Self> {
        use tracing::debug_span;

        let candidates_span = debug_span!("find_removal_candidates", k = k);
        let removal_candidates_lists: Vec<Vec<usize>> = {
            let _guard = candidates_span.enter();
            if k == 1 {
                let is_replaceable_span = debug_span!("check_is_replaceable");
                let _replaceable_guard = is_replaceable_span.enter();
                self.selected_images()
                    .filter_map(|selected_image| {
                        if self.is_replaceable(selected_image, problem) {
                            Some(vec![selected_image])
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                let worst_images = self.worst_selected_images(problem, is_deterministic, trackers);

                let combinations_span = debug_span!("generate_combinations", k = k);
                let _comb_guard = combinations_span.enter();
                worst_images.into_iter().combinations(k as usize).collect()
            }
        };

        let mut residual_solutions: Vec<Self> = Vec::new();

        let solve_span = debug_span!(
            "solve_residual_problems",
            num_problems = removal_candidates_lists.len()
        );
        let _solve_guard = solve_span.enter();

        for (idx, removal_candidates) in removal_candidates_lists.into_iter().enumerate() {
            let problem_span = debug_span!("solve_residual_problem", problem_index = idx);
            let _problem_guard = problem_span.enter();

            if let Some(mut residual_problem) = self.create_residual_problem(
                removal_candidates,
                problem,
                is_deterministic,
                trackers,
            ) {
                // Pass the modified trackers (with removals applied) to solve the residual problem
                let neighborhood_iter = residual_problem
                    .solve::<NdTreeSolutionSet<ResidualSolution<D>, D>>(
                        problem,
                        timer,
                        trackers.clone(),
                    );

                let extend_span = debug_span!("extend_residual_solutions");
                let _extend_guard = extend_span.enter();
                residual_solutions.extend(neighborhood_iter.into_iter());
            }
            // iteration_trackers is dropped here, original trackers remain unmodified
            if timer.is_expired() {
                break;
            }
        }

        return residual_solutions;
    }

    fn neighborhood_iter<'a>(
        &'a self,
        trackers: &'a mut Self::Trackers,
        k: u32,
        problem: &'a P,
        timer: &'a crate::timer::Timer,
        is_deterministic: bool,
        optimizations: &'a PlsOptimizations,
    ) -> Box<dyn Iterator<Item = Self> + 'a>
    where
        Self: 'a,
    {
        Box::new(self.neighborhood_iter_impl(trackers, k, problem, timer, is_deterministic, optimizations))
    }

    fn is_valid(&self, problem: &P) -> bool {
        if !problem.is_set_cover(self) {
            // Find which elements are not covered
            let selected_images = self.selected_images.ones().collect::<Vec<_>>();
            let mut covered_elements = FixedBitSet::with_capacity(problem.num_elements());
            for img_idx in &selected_images {
                for elem in problem.image_elements(*img_idx) {
                    covered_elements.set(elem, true);
                }
            }
            let uncovered: Vec<usize> = (0..problem.num_elements())
                .filter(|&e| !covered_elements[e])
                .collect();

            eprintln!(
                "Solution is NOT a set cover! Selected images: {:?}, Uncovered elements ({} total): {:?}",
                selected_images,
                uncovered.len(),
                if uncovered.len() <= 20 {
                    format!("{uncovered:?}")
                } else {
                    format!("{:?}... ({} more)", &uncovered[..20], uncovered.len() - 20)
                }
            );
            return false;
        }

        return self.are_objectives_valid(problem);
    }
}

// Implement EncodedSolution trait
impl<P, const D: usize> crate::solution::EncodedSolution<P, D> for BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    fn timestamp(&self) -> Duration {
        self.timestamp
    }
}

impl<P, const D: usize> BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    /// Creates a `BitsetEncodedSolution` from a list of selected image indices.
    #[must_use]
    pub fn from_selected_images(selected_images_vec: &[usize], problem: &P) -> Self {
        let objectives = [0u64; D];

        let mut solution = Self {
            selected_images: selected_images_vec.iter().copied().collect::<FixedBitSet>(),
            objectives,
            timestamp: Duration::new(0, 0),
            _phantom: std::marker::PhantomData,
        };

        solution.recalculate_objectives(problem);

        solution
    }

    /// Generate a random feasible solution (choose element randomly, then choose image randomly from those that contain the element iff it is not already covered by another image)
    ///
    /// # Panics
    ///
    /// Panics if there is no image covering an uncovered element (i.e., `.unwrap()` fails).
    #[must_use]
    pub fn random_with_seed(problem: &P, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut selected_images_bitset = FixedBitSet::with_capacity(problem.num_images());
        let mut covered_elements = vec![false; problem.num_elements()];
        let mut num_covered_elements = 0;

        while num_covered_elements < problem.num_elements() {
            let element_index = rng.random_range(0..problem.num_elements());
            if covered_elements[element_index] {
                continue;
            }

            // Choose random image that covers the element
            let covering_images: Vec<usize> = problem.element_images(element_index).collect();
            let image_index = covering_images
                .iter()
                .filter(|&&img| !selected_images_bitset[img])
                .choose(&mut rng)
                .unwrap();
            selected_images_bitset.set(*image_index, true);

            // Mark all elements of the image as covered
            for part in problem.image_elements(*image_index) {
                if !covered_elements[part] {
                    covered_elements[part] = true;
                    num_covered_elements += 1;
                }
            }
        }

        // Convert bitset to vec and use from_selected_images to properly initialize trackers
        let selected_images_vec: Vec<usize> = selected_images_bitset.ones().collect();
        Self::from_selected_images(&selected_images_vec, problem)
    }

    /// Generate a random feasible solution
    #[must_use]
    pub fn random(problem: &P) -> Self {
        Self::random_with_seed(problem, rand::random())
    }

    /// Construct a solution using a greedy set cover heuristic that scores
    /// unselected images based on weighted scalarization of per-image metrics.
    /// Different weight vectors target different regions of the objective space.
    ///
    /// The `weights` parameter controls which objective to prioritize.
    /// All objectives are treated as minimization targets, so images with
    /// lower weighted scores are preferred.
    #[must_use]
    pub fn greedy_construction(problem: &P, weights: [f64; D]) -> Self {
        use crate::objectives::ObjectiveState;

        let num_images = problem.num_images();
        let num_elements = problem.num_elements();

        // Pre-compute per-image objective proxy values for scoring.
        // Each value represents the "cost" an image contributes towards
        // the corresponding objective (lower is better).
        let image_values: Vec<[f64; D]> = (0..num_images)
            .map(|img| {
                let mut values = [0.0f64; D];
                for obj_idx in 0..D {
                    values[obj_idx] = match problem.objective(obj_idx) {
                        ObjectiveState::TotalCost { costs, .. } => costs[img] as f64,
                        ObjectiveState::CloudyArea { clear_images, .. } => {
                            // Proxy: count of elements this image covers but does NOT
                            // clear. Fewer cloudy elements → lower score → preferred.
                            let total_elems = problem.image_elements(img).count();
                            let clear_elems = clear_images[img].count_ones(..);
                            (total_elems.saturating_sub(clear_elems)) as f64
                        }
                        ObjectiveState::MinResolution { resolutions, .. } => {
                            resolutions[img] as f64
                        }
                        ObjectiveState::MaxIncidenceAngle {
                            incidence_angles, ..
                        } => incidence_angles[img] as f64,
                    };
                }
                values
            })
            .collect();

        let mut selected = FixedBitSet::with_capacity(num_images);
        let mut covered = FixedBitSet::with_capacity(num_elements);
        let mut num_covered: usize = 0;

        while num_covered < num_elements {
            let mut best_image: Option<usize> = None;
            let mut best_score = f64::MAX;

            for img in 0..num_images {
                if selected[img] {
                    continue;
                }

                // Count newly covered elements
                let newly_covered: usize = problem
                    .image_elements(img)
                    .filter(|&elem| !covered[elem])
                    .count();

                if newly_covered == 0 {
                    continue;
                }

                // Compute weighted score per newly covered element
                let weighted_value: f64 = (0..D)
                    .map(|obj_idx| weights[obj_idx] * image_values[img][obj_idx])
                    .sum();

                let score = weighted_value / newly_covered as f64;

                if score < best_score {
                    best_score = score;
                    best_image = Some(img);
                }
            }

            if let Some(img) = best_image {
                selected.set(img, true);
                for elem in problem.image_elements(img) {
                    if !covered[elem] {
                        covered.set(elem, true);
                        num_covered += 1;
                    }
                }
            } else {
                // No image can cover any more elements — should not happen
                // with a valid problem instance, but break to avoid infinite loop.
                #[cfg(feature = "bounds_check")]
                debug_assert!(
                    num_covered >= num_elements,
                    "greedy_construction: stuck with {} uncovered elements",
                    num_elements - num_covered
                );
                break;
            }
        }

        let selected_vec: Vec<usize> = selected.ones().collect();
        Self::from_selected_images(&selected_vec, problem)
    }

    /// Scalarizing function using weighted sum, for solution quality comparison
    #[must_use]
    pub fn scalarizing_fn(&self, weights: &[f32; D], _max_values: pareto::Objectives<D>) -> f32 {
        return objectives::weighted_sum(&self.objectives, weights);
    }

    /// Check if objective values are correct
    #[must_use]
    pub fn are_objectives_valid(&self, problem: &P) -> bool {
        for i in 0..D {
            let expected_value = self.calculate_objective(i, problem);
            if self.objectives[i] != expected_value {
                eprintln!(
                    "Objective {} is invalid. Expected {}, got {}",
                    i, expected_value, self.objectives[i]
                );
                eprintln!(
                    "  Selected images: {:?}",
                    self.selected_images.ones().collect::<Vec<_>>()
                );
                return false;
            }
        }

        true
    }

    /// Returns iterator over indices of selected images of solution
    #[must_use]
    pub fn selected_images(&self) -> BitsetSelectedImagesIter<'_> {
        self.selected_images.ones()
    }

    /// Returns iterator over indices of unselected images of solution
    #[must_use]
    pub fn unselected_images(&self) -> BitsetUnselectedImagesIter<'_> {
        self.selected_images.zeroes()
    }


    /// Check whether image at index i can be replaced by another image(s)
    #[must_use]
    pub fn is_replaceable(&self, i: usize, problem: &P) -> bool {
        // For each part of the image, check if there is another image that covers the part
        self.unselected_images()
            .any(|image_index| problem.overlap(i, image_index) > 0)
    }

    /// Generate random weights for objectives
    #[must_use]
    pub fn generate_weights(&self) -> [f32; D] {
        return objectives::generate_weights::<D>();
    }

    /// Transition to an `UndercoveredSolution` state by removing specific images.
    /// This is the first step in creating a `ResidualProblem`.
    #[must_use]
    #[tracing::instrument(skip(self, problem), level = "debug")]
    pub fn build_undercovered_solution(
        &self,
        images_to_remove: &[usize],
        problem: &P,
    ) -> UndercoveredSolution<D> {
        // 1. Create bitset of removed images
        let removed_images: FixedBitSet = images_to_remove.iter().copied().collect();

        // 2. Create partial solution using bitwise difference (A \ B)
        let mut partial_selected_images = self.selected_images.clone();
        partial_selected_images.difference_with(&removed_images);

        let uncovered_elements_indices = problem.uncovered_elements(partial_selected_images.ones());
        let uncovered_elements: FixedBitSet = uncovered_elements_indices.collect();

        UndercoveredSolution {
            partial_selected_images,
            removed_images,
            uncovered_elements,
            partial_trackers: ProvenSafeTrackerArray::new(problem),
        }
    }

    /// Create residual problem, composed of removed images, candidates to be added, and images covering the rest of the uncovered elements.
    #[must_use]
    #[tracing::instrument(skip(self, problem, trackers), level = "debug")]
    pub fn create_residual_problem(
        &self,
        mut removal_candidates_indices: Vec<usize>,
        problem: &P,
        is_deterministic: bool,
        trackers: &mut ProvenSafeTrackerArray<D>,
    ) -> Option<ResidualProblem<Self, P, D>> {
        let mut active_solution = self.clone();

        for &img_idx in &removal_candidates_indices {
            active_solution.remove_image(img_idx, problem, trackers);
        }

        let uncovered_elements_indices: Vec<usize> = problem
            .uncovered_elements(active_solution.selected_images())
            .collect();

        // If there are no uncovered elements after removal, no residual problem needed
        if uncovered_elements_indices.is_empty() {
            return None;
        }

        // Step 3: Find candidates using the modified trackers
        let mut best_addition_candidates = Self::best_unselected_images_with_trackers(
            &active_solution.selected_images,
            trackers,
            &uncovered_elements_indices,
            problem,
            is_deterministic,
        )?;

        // Keep ordering stable for deterministic behavior.
        removal_candidates_indices.sort_unstable();
        best_addition_candidates.sort_unstable();

        let residual_problem = ResidualProblem::new(
            active_solution,
            &removal_candidates_indices,
            &best_addition_candidates,
            uncovered_elements_indices,
            problem,
        );

        Some(residual_problem)
    }

    /// Get indices of the best replacement image(s) which is not selected yet, returns None when image cannot be replaced
    /// Static helper that accepts trackers and `selected_images` directly
    #[must_use]
    pub fn best_unselected_images_with_trackers(
        partial_selected_images: &FixedBitSet,
        partial_trackers: &ProvenSafeTrackerArray<D>,
        uncovered_elements: &[usize],
        problem: &P,
        is_deterministic: bool,
    ) -> Option<Vec<usize>> {
        let weights: [f32; D] = if is_deterministic {
            // For deterministic mode, use equal weights
            let equal_weight = 1.0 / D as f32;
            [equal_weight; D]
        } else {
            objectives::generate_weights::<D>()
        };

        // Use global objective bounds for normalization.
        // Computed once here (outside the image loop) since it only depends on the problem,
        // not on which images are unselected.
        let normalization_ranges: Vec<f32> = problem.objective_bounds().as_ref().map_or_else(
            || {
                problem
                    .max_objectives()
                    .iter()
                    .map(|&max_val| if max_val > 0 { max_val as f32 } else { 1.0 })
                    .collect()
            },
            |bounds| {
                bounds
                    .iter()
                    .map(|bound| {
                        let range = bound[1] as f32 - bound[0] as f32;
                        if range > 0.0 { range } else { 1.0 }
                    })
                    .collect()
            },
        );

        let image_set_view = ImageSetView { selected_images: partial_selected_images };

        // Compute peek_addition_delta only for images that cover at least one uncovered element.
        //
        // When uncovered_elements is non-empty, build the candidate set via the inverse index
        // (element → images) instead of iterating all unselected images and doing a per-image
        // element scan:
        //   Old: O(n_unselected × avg_elements_per_image) ≈ O(80 × 200) = 16 000 ops
        //   New: O(n_uncovered × avg_images_per_element) ≈ O(15 × 10)  = 150  ops
        let mut comparable_unselected_images: Vec<ComparableImage> = if uncovered_elements
            .is_empty()
        {
            // No coverage filter — all unselected images are candidates.
            partial_selected_images
                .zeroes()
                .map(|image_index| {
                    let raw_deltas = partial_trackers.peek_addition_delta(
                        image_index,
                        problem,
                        &image_set_view,
                    );
                    let mut scaled_deltas = [0.0f32; D];
                    for i in 0..D {
                        scaled_deltas[i] = raw_deltas[i].abs() as f32 / normalization_ranges[i];
                    }
                    let key = objectives::weighted_sum_f32(&scaled_deltas, &weights);
                    ComparableImage {
                        index: image_index,
                        key: (1_000_000.0 * key) as usize,
                    }
                })
                .collect()
        } else {
            // Pre-filter: use the inverse (element→images) index to build a bitset of
            // unselected images that cover at least one uncovered element.
            let n_images = partial_selected_images.len();
            let mut candidate_unselected = FixedBitSet::with_capacity(n_images);
            for &elem in uncovered_elements {
                for img in problem.element_images(elem) {
                    // Bounds-check: partial_selected_images may have capacity < total images
                    // (when created from a selected-images vec, FixedBitSet capacity = max_idx+1).
                    // Images at indices >= n_images are implicitly unselected but excluded from
                    // zeroes() in the original code — preserve that behavior here.
                    if img < n_images && !partial_selected_images.contains(img) {
                        candidate_unselected.insert(img);
                    }
                }
            }
            candidate_unselected
                .ones()
                .map(|image_index| {
                    let raw_deltas = partial_trackers.peek_addition_delta(
                        image_index,
                        problem,
                        &image_set_view,
                    );
                    let mut scaled_deltas = [0.0f32; D];
                    for i in 0..D {
                        scaled_deltas[i] = raw_deltas[i].abs() as f32 / normalization_ranges[i];
                    }
                    let key = objectives::weighted_sum_f32(&scaled_deltas, &weights);
                    ComparableImage {
                        index: image_index,
                        key: (1_000_000.0 * key) as usize,
                    }
                })
                .collect()
        };

        if comparable_unselected_images.is_empty() {
            return None;
        }

        let best_unselected_images = if comparable_unselected_images.len() > 9 {
            comparable_unselected_images
                .select_nth_unstable_by_key(9, |ci| ci.key)
                .0
                .iter()
                .map(|ci| ci.index)
                .collect::<Vec<usize>>()
        } else {
            comparable_unselected_images
                .into_iter()
                .map(|ci| ci.index)
                .collect::<Vec<usize>>()
        };
        Some(best_unselected_images)
    }

    /// Get indices of the best replacement image(s) which is not selected yet, returns None when image cannot be replaced
    #[must_use]
    pub fn best_unselected_images(
        &self,
        uncovered_elements: &[usize],
        problem: &P,
        is_deterministic: bool,
        trackers: &ProvenSafeTrackerArray<D>,
    ) -> Option<Vec<usize>> {
        Self::best_unselected_images_with_trackers(
            &self.selected_images,
            trackers,
            uncovered_elements,
            problem,
            is_deterministic,
        )
    }

    #[tracing::instrument(skip(self, problem, trackers), level = "debug")]
    pub fn worst_selected_images(
        &self,
        problem: &P,
        is_deterministic: bool,
        trackers: &ProvenSafeTrackerArray<D>,
    ) -> Vec<usize> {
        self.worst_selected_images_n(problem, is_deterministic, trackers, 9)
    }

    /// Return up to `limit` selected images ranked by "worst first" heuristic.
    ///
    /// Images that contribute least value per covered element are returned first.
    /// This is the generalised version of [`worst_selected_images`] (which defaults
    /// to `limit = 9`).
    pub fn worst_selected_images_n(
        &self,
        problem: &P,
        is_deterministic: bool,
        trackers: &ProvenSafeTrackerArray<D>,
        limit: usize,
    ) -> Vec<usize> {
        let weights: [f32; D] = if is_deterministic {
            // For deterministic mode, use equal weights
            let equal_weight = 1.0 / D as f32;
            [equal_weight; D]
        } else {
            self.generate_weights()
        };

        let comparable_selected_images: BinaryHeap<ComparableImage> = self
            .scaled_image_objective_deltas_impl(self.selected_images(), problem, trackers)
            .into_iter()
            .map(|scaled_image_deltas| {
                let covered_elements_count = problem
                    .image_elements(scaled_image_deltas.image_index)
                    .count();

                let comparision_heur_key =
                    objectives::weighted_sum_f32(&scaled_image_deltas.scaled_deltas, &weights)
                        / covered_elements_count as f32;
                return ComparableImage {
                    index: scaled_image_deltas.image_index,
                    key: (100_000.0 * comparision_heur_key) as usize,
                };
            })
            .collect();

        let worst_selected_images = comparable_selected_images
            .iter()
            .take(limit)
            .map(|comparable_image| comparable_image.index)
            .collect::<Vec<usize>>();
        return worst_selected_images;
    }

    #[must_use]
    pub fn as_sims_solution(&self) -> SIMSSolution {
        SIMSSolution {
            selected_images: self.selected_images.ones().collect(),
        }
    }

    fn scaled_image_objective_deltas_impl<I: Iterator<Item = usize>>(
        &self,
        images: I,
        problem: &P,
        trackers: &ProvenSafeTrackerArray<D>,
    ) -> Vec<ScaledObjectiveDeltas<D>> {
        let raw_comparable_images: Vec<ImageObjectiveDeltas<D>> = images
            .map(|image_index| {
                // Use trackers for delta calculation
                let is_removing = self.is_image_selected(image_index);
                let mut deltas = [0i64; D];
                for (obj_index, delta) in deltas.iter_mut().enumerate().take(D) {
                    *delta = if is_removing {
                        trackers
                            .get(obj_index)
                            .peek_removal_delta(image_index, problem, self)
                    } else {
                        trackers
                            .get(obj_index)
                            .peek_addition_delta(image_index, problem, self)
                    };
                }
                ImageObjectiveDeltas {
                    image_index,
                    deltas,
                }
            })
            .collect();

        // Use global objective bounds for normalization if available
        // Otherwise fall back to max_objectives as before
        let normalization_ranges: Vec<f32> = problem.objective_bounds().as_ref().map_or_else(
            || {
                // Fallback: use max_objectives (nadir point approximation)
                problem
                    .max_objectives()
                    .iter()
                    .map(|&max_val| if max_val > 0 { max_val as f32 } else { 1.0 })
                    .collect()
            },
            |bounds| {
                bounds
                    .iter()
                    .map(|bound| {
                        let range = bound[1] as f32 - bound[0] as f32;
                        if range > 0.0 { range } else { 1.0 }
                    })
                    .collect()
            },
        );

        raw_comparable_images
            .iter()
            .map(|objective_deltas| {
                let mut scaled_deltas = [0.0f32; D];
                let raw_deltas = objective_deltas.deltas;

                for i in 0..D {
                    // Scale using absolute value divided by global range
                    // This ensures consistent scaling across all batches
                    scaled_deltas[i] = raw_deltas[i].abs() as f32 / normalization_ranges[i];
                }

                ScaledObjectiveDeltas {
                    image_index: objective_deltas.image_index,
                    raw_deltas,
                    scaled_deltas,
                }
            })
            .collect()
    }
}

impl<P> PartialOrd for BitsetEncodedSolution<P, 2>
where
    P: SetCoverProblem<2> + Clone + Send + Sync,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<P> Ord for BitsetEncodedSolution<P, 2>
where
    P: SetCoverProblem<2> + Clone + Send + Sync,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.objectives.first().cmp(&other.objectives.first())
    }
}

impl<P, const D: usize> PartialEq for BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    fn eq(&self, other: &Self) -> bool {
        self.selected_images == other.selected_images
    }
}

impl<P, const D: usize> Eq for BitsetEncodedSolution<P, D> where
    P: SetCoverProblem<D> + Clone + Send + Sync
{
}

impl<P, const D: usize> Hash for BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.selected_images.hash(state);
    }
}

#[expect(
    clippy::missing_fields_in_debug,
    reason = "Custom Debug impl only shows relevant fields for readability"
)]
impl<P, const D: usize> Debug for BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let selected_images = self.selected_images().collect::<Vec<usize>>();
        f.debug_struct("BitsetEncodedSolution")
            .field("objectives", &self.objectives)
            .field("images_count", &selected_images.len())
            .field("selected_images", &selected_images)
            .finish()
    }
}

impl<P, const D: usize> MergeableWithResidual<P, D> for BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    fn merge_residual_solution(
        &mut self,
        residual_solution: &ResidualSolution<D>,
        residual_problem: &ResidualProblem<Self, P, D>,
        problem: &P,
        trackers: &mut Self::Trackers,
    ) {
        log::debug!(
            "MERGE: ResidualSolution has condensed indices: {:?}",
            residual_solution.selected_images
        );
        log::debug!(
            "MERGE: image_index_map: {:?}",
            residual_problem.image_map_condensed_to_original
        );

        // Reconstruct the partial solution by removing images that were removed in residual problem
        let num_removal_candidates = residual_problem
            .condensed_original_removed_images
            .count_ones(..);
        for i in 0..num_removal_candidates {
            let original_img_idx = residual_problem.image_map_condensed_to_original[i];
            debug_assert!(
                original_img_idx < problem.num_images(),
                "merge_residual_solution: invalid image index {original_img_idx} from map"
            );
            if self.is_image_selected(original_img_idx) {
                self.selected_images.set(original_img_idx, false);
            }
        }

        log::debug!(
            "MERGE: Partial solution after removals has {} images",
            self.num_selected_images()
        );

        // Map condensed indices (0..residual_problem.image_index_map.len()) to original indices
        let original_indices: Vec<usize> = residual_solution
            .selected_images
            .iter()
            .map(|&condensed_idx| {
                let original_idx = residual_problem.image_map_condensed_to_original[condensed_idx];
                log::debug!("MERGE: mapping condensed {condensed_idx} -> original {original_idx}");
                original_idx
            })
            .collect();

        log::debug!("MERGE: Adding original indices to solution: {original_indices:?}");

        // Now add only the new images from the residual solution
        for &image_index in &original_indices {
            self.add_image(image_index, problem, trackers);
        }

        // Restore tracker by removing images that were added above
        for &image_index in &original_indices {
            trackers.track_image_removal(image_index, problem);
        }

        // Inherit timestamp from ResidualSolution
        self.timestamp = residual_solution.timestamp;
    }
}

impl<P, const D: usize> BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    /// Like [`MergeableWithResidual::merge_residual_solution`] but does NOT restore
    /// tracker state after the merge. The caller is responsible for restoring the
    /// trackers (e.g. via [`ProvenSafeTrackerArray::snapshot_mutable_state_into`]).
    ///
    /// This avoids O(k) individual `track_image_removal` calls per merged neighbor,
    /// replacing them with a single bulk checkpoint restore in the iterator.
    pub fn merge_residual_no_tracker_restore(
        &mut self,
        residual_solution: &ResidualSolution<D>,
        residual_problem: &ResidualProblem<Self, P, D>,
        problem: &P,
        trackers: &mut <Self as SIMSModifiable<P, D>>::Trackers,
    ) {
        // Steps 1-3 are identical to merge_residual_solution.

        // Step 1: Remove original removed images from the solution (bitset only, no tracker update)
        let num_removal_candidates = residual_problem
            .condensed_original_removed_images
            .count_ones(..);
        for i in 0..num_removal_candidates {
            let original_img_idx = residual_problem.image_map_condensed_to_original[i];
            debug_assert!(
                original_img_idx < problem.num_images(),
                "merge_residual_no_tracker_restore: invalid image index {original_img_idx} from map"
            );
            if self.is_image_selected(original_img_idx) {
                self.selected_images.set(original_img_idx, false);
            }
        }

        // Step 2: Map condensed indices to original indices
        let original_indices: Vec<usize> = residual_solution
            .selected_images
            .iter()
            .map(|&condensed_idx| residual_problem.image_map_condensed_to_original[condensed_idx])
            .collect();

        // Step 3: Add residual solution images (updates objectives AND trackers)
        for &image_index in &original_indices {
            self.add_image(image_index, problem, trackers);
        }

        // Step 4 is OMITTED: caller restores trackers via checkpoint

        self.timestamp = residual_solution.timestamp;
    }
}

impl<P, const D: usize> BitsetEncodedSolution<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    pub fn neighborhood_iter_impl<'a>(
        &'a self,
        trackers: &'a mut ProvenSafeTrackerArray<D>,
        k: u32,
        problem: &'a P,
        timer: &'a Timer,
        is_deterministic: bool,
        optimizations: &'a PlsOptimizations,
    ) -> BitsetNeighborhoodIter<'a, P, D> {
        use tracing::debug_span;

        // Initialize trackers from current solution
        trackers.initialize_from(self, problem);

        let candidates_span = debug_span!("find_removal_candidates", k = k);
        let _guard = candidates_span.enter();

        let removal_candidates_iter: Box<dyn Iterator<Item = Vec<usize>> + 'a> = if k == 1 {
            if optimizations.use_ranked_candidates {
                // Ranked iteration for k=1: use worst_selected_images-style scoring
                // to prioritise the most promising removal candidates and limit their
                // count, avoiding exhaustive exploration on large instances.
                let ranked = self.worst_selected_images_n(
                    problem,
                    is_deterministic,
                    trackers,
                    optimizations.max_k1_candidates,
                );
                Box::new(
                    ranked
                        .into_iter()
                        .filter(|&img| self.is_replaceable(img, problem))
                        .map(|img| vec![img]),
                )
            } else {
                // Exhaustive iteration for k=1: try every replaceable selected image
                let all_selected: Vec<usize> = self.selected_images().collect();
                Box::new(
                    all_selected
                        .into_iter()
                        .filter(|&img| self.is_replaceable(img, problem))
                        .map(|img| vec![img]),
                )
            }
        } else {
            // Lazy iteration for k>1 using combinations
            // For k > 1, we still compute worst_images upfront as it depends on sorting
            let worst_images = self.worst_selected_images(problem, is_deterministic, trackers);
            // But we use the iterator from combinations() instead of collecting
            Box::new(worst_images.into_iter().combinations(k as usize))
        };

        // Save S_base state before any removal candidates are processed.
        // Used to bulk-restore when transitioning between removal combinations.
        let base_checkpoint = trackers.clone();

        // Pre-allocate checkpoint tracker (same shape as trackers, reused across
        // removal combinations to avoid per-combination allocation).
        let checkpoint = trackers.clone();

        BitsetNeighborhoodIter {
            original_solution: self.clone(),
            problem,
            timer,
            trackers,
            removal_candidates_iter,
            current_residual_problem: None,
            current_removal_candidates: Vec::new(),
            filtered_residual_iter: None,
            is_deterministic,
            base_checkpoint,
            checkpoint,
            use_checkpoint: optimizations.use_checkpoint,
            neighborhood_budget: optimizations.neighborhood_budget,
            neighbors_yielded: 0,
        }
    }
}

// Iterator for BitsetEncodedSolution neighborhood
pub struct BitsetNeighborhoodIter<'a, P, const D: usize>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    original_solution: BitsetEncodedSolution<P, D>,
    problem: &'a P,
    timer: &'a Timer,
    trackers: &'a mut ProvenSafeTrackerArray<D>,
    removal_candidates_iter: Box<dyn Iterator<Item = Vec<usize>> + 'a>,

    // State for current residual problem - now we can cache it directly without lifetime issues
    current_residual_problem: Option<ResidualProblem<BitsetEncodedSolution<P, D>, P, D>>,
    current_removal_candidates: Vec<usize>,

    // Consuming iterator over Pareto-filtered residual solutions for the current residual problem.
    // When a new residual problem is created, all its valid residual solutions are
    // enumerated (with merged CloudyArea objectives), filtered through an NdTree Pareto front,
    // and drained lazily via this iterator.
    filtered_residual_iter: Option<NDTreeSolutionIntoIterator<ResidualSolution<D>, 32, D, 4>>,

    is_deterministic: bool,

    /// Checkpoint of the S_base tracker state (before any removals).
    /// Used to bulk-restore when transitioning between removal combinations.
    base_checkpoint: ProvenSafeTrackerArray<D>,

    /// Pre-allocated checkpoint of the tracker state saved after image removals
    /// (the "S_removed" state). Used to bulk-restore trackers after each merge
    /// instead of undoing individual `track_image_removal` calls.
    checkpoint: ProvenSafeTrackerArray<D>,

    /// Whether to use bulk checkpoint/restore for tracker state after each merge.
    /// When false, falls back to per-image undo operations.
    use_checkpoint: bool,

    /// Optional cap on the total number of neighbors yielded per solution.
    /// When exhausted the iterator terminates early, focusing exploration time
    /// on other solutions in the population.
    neighborhood_budget: Option<usize>,

    /// Counter of neighbors yielded so far (compared against `neighborhood_budget`).
    neighbors_yielded: usize,
}

impl<P, const D: usize> Iterator for BitsetNeighborhoodIter<'_, P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    type Item = BitsetEncodedSolution<P, D>;

    fn next(&mut self) -> Option<Self::Item> {
        use pareto::ParetoFront;

        // Outer loop to avoid recursion
        loop {
            // Check timer first
            if self.timer.is_expired() {
                return None;
            }

            // Check neighborhood budget
            if let Some(budget) = self.neighborhood_budget {
                if self.neighbors_yielded >= budget {
                    return None;
                }
            }

            // 1. If we have a non-exhausted iterator of non-dominated residual solutions, yield from it
            if let Some(residual_solution) = self.filtered_residual_iter.as_mut().and_then(|it| it.next()) {
                let residual_problem = self
                    .current_residual_problem
                    .as_ref()
                    .expect("residual problem must exist while iterator is active");
                let mut neighbor = residual_problem.unmodified_solution.clone();

                if self.use_checkpoint {
                    // Use the no-restore variant: add images to compute objectives
                    // but skip per-image tracker undo. Bulk-restore from checkpoint below.
                    neighbor.merge_residual_no_tracker_restore(
                        &residual_solution,
                        residual_problem,
                        self.problem,
                        self.trackers,
                    );

                    // Bulk-restore trackers to the S_removed state via checkpoint memcpy,
                    // replacing O(k) individual track_image_removal calls.
                    self.checkpoint.snapshot_mutable_state_into(self.trackers);
                } else {
                    // Per-image undo path: merge then undo each added image individually
                    neighbor.merge_residual_solution(
                        &residual_solution,
                        residual_problem,
                        self.problem,
                        self.trackers,
                    );
                }

                self.neighbors_yielded += 1;
                return Some(neighbor);
            }

            // 2. Iterator exhausted. If we had a residual problem, we've drained it -- clean up.
            if self.current_residual_problem.is_some() {
                self.current_residual_problem = None;
                self.filtered_residual_iter = None;
                // Restore trackers to original solution state (S_base).
                if self.use_checkpoint {
                    self.base_checkpoint.snapshot_mutable_state_into(self.trackers);
                } else {
                    for &removal_candidate in &self.current_removal_candidates {
                        self.trackers
                            .track_image_addition(removal_candidate, self.problem);
                    }
                }
            }

            // 3. Get next removal candidate and create a new residual problem
            loop {
                let removal_candidates = self.removal_candidates_iter.next()?;

                // Create residual problem (this will modify trackers — applies removals)
                if let Some(mut residual_problem) =
                    self.original_solution.create_residual_problem(
                        removal_candidates.clone(),
                        self.problem,
                        self.is_deterministic,
                        self.trackers,
                    )
                {
                    self.current_removal_candidates = removal_candidates;

                    // Save checkpoint of the S_removed tracker state. All residual
                    // solutions for this removal combination will restore to this
                    // state after merge via snapshot_mutable_state_into.
                    self.trackers.snapshot_mutable_state_into(&mut self.checkpoint);

                    // 4. Enumerate ALL valid residual solutions, fix non-additive objectives,
                    //    and filter through Pareto front.
                    //
                    //    - CloudyArea: overwrite with merged value (sound)
                    //    - MinResolution: neutralize to 0 (unsound in residual space,
                    //      so we disable it for filtering; the global archive still
                    //      filters on full merged objectives)
                    let min_res_obj_idx = self.problem.objective_types().iter().position(
                        |t| matches!(t, crate::objectives::ObjectiveType::MinResolution),
                    );

                    let mut nd_set: NdTreeSolutionSet<ResidualSolution<D>, D> =
                        NdTreeSolutionSet::default();
                    while let Some(mut residual_solution) =
                        residual_problem.solve_next(self.problem, self.timer)
                    {
                        // Overwrite CloudyArea objective with merged value
                        if let Some(merged_cloudy) =
                            residual_problem.compute_merged_cloudy_area(&residual_solution)
                        {
                            let obj_idx = residual_problem
                                .cloudy_area_data
                                .as_ref()
                                .unwrap()
                                .objective_index;
                            residual_solution.objectives[obj_idx] = merged_cloudy;
                        }
                        // Neutralize MinResolution so it never causes dominance
                        if let Some(idx) = min_res_obj_idx {
                            residual_solution.objectives[idx] = 0;
                        }
                        nd_set.try_insert(&residual_solution);
                    }

                    // Store the consuming iterator -- no Vec allocation
                    self.filtered_residual_iter = Some(nd_set.into_iter());
                    self.current_residual_problem = Some(residual_problem);
                    break;
                }

                // Restore trackers to original solution state (S_base)
                if self.use_checkpoint {
                    self.base_checkpoint.snapshot_mutable_state_into(self.trackers);
                } else {
                    for &removal_candidate in &removal_candidates {
                        self.trackers
                            .track_image_addition(removal_candidate, self.problem);
                    }
                }
            }
            // Loop back to yield from the freshly-filled iterator
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objectives::ObjectiveType;
    use crate::problem_bitset::ProblemBitset;
    use crate::solution::SIMSModifiable;
    use std::path::PathBuf;

    const NUM_OBJECTIVES: usize = 4;
    const OBJECTIVE_TYPES: [ObjectiveType; NUM_OBJECTIVES] = [
        ObjectiveType::TotalCost,
        ObjectiveType::CloudyArea,
        ObjectiveType::MinResolution,
        ObjectiveType::MaxIncidenceAngle,
    ];

    fn make_test_problem() -> ProblemBitset<NUM_OBJECTIVES> {
        ProblemBitset::from_minizinc_datafile(
            "tests/data/lagos_nigeria_30.dzn",
            OBJECTIVE_TYPES,
        )
        .expect("failed to load tests/data/lagos_nigeria_30.dzn")
    }

    #[test]
    fn test_greedy_initial_solutions_are_valid() {
        let problem = make_test_problem();

        let solutions = BitsetEncodedSolution::<ProblemBitset<NUM_OBJECTIVES>, NUM_OBJECTIVES>::greedy_initial_solutions(&problem);

        assert!(
            !solutions.is_empty(),
            "greedy_initial_solutions should produce at least one solution"
        );

        for (i, solution) in solutions.iter().enumerate() {
            assert!(
                problem.is_set_cover(solution),
                "Greedy solution {i} is not a valid set cover: objectives={:?}",
                solution.objectives()
            );
            assert!(
                solution.is_valid(&problem),
                "Greedy solution {i} failed full validity check: objectives={:?}",
                solution.objectives()
            );
            assert!(
                solution.are_objectives_valid(&problem),
                "Greedy solution {i} has inconsistent objective values: objectives={:?}",
                solution.objectives()
            );
        }
    }

    #[test]
    fn test_greedy_initial_solutions_do_not_dominate_pseudo_solver_solutions() {
        let problem = make_test_problem();

        let greedy_solutions =
            BitsetEncodedSolution::<ProblemBitset<NUM_OBJECTIVES>, NUM_OBJECTIVES>::greedy_initial_solutions(&problem);

        let pseudo_path = PathBuf::from("../sims-core/tests/data/pseudo_solver_solutions/lagos_nigeria_30.json");
        let pseudo_json = std::fs::read_to_string(&pseudo_path)
            .expect("failed to read ../sims-core/tests/data/pseudo_solver_solutions/lagos_nigeria_30.json");
        let pseudo_data: serde_json::Value =
            serde_json::from_str(&pseudo_json).expect("failed to parse pseudo solver json");
        let pseudo_solutions = pseudo_data["solutions"]
            .as_array()
            .expect("pseudo solver json missing 'solutions' array");

        let pseudo_objectives: Vec<[u64; NUM_OBJECTIVES]> = pseudo_solutions
            .iter()
            .map(|sol| {
                [
                    sol["cost"].as_u64().expect("pseudo solution missing cost"),
                    sol["cloudy_area"]
                        .as_u64()
                        .expect("pseudo solution missing cloudy_area"),
                    sol["min_resolutions_sum"]
                        .as_u64()
                        .expect("pseudo solution missing min_resolutions_sum"),
                    sol["max_incidence_angle"]
                        .as_u64()
                        .expect("pseudo solution missing max_incidence_angle"),
                ]
            })
            .collect();

        for (g_idx, greedy) in greedy_solutions.iter().enumerate() {
            for (p_idx, pseudo_obj) in pseudo_objectives.iter().enumerate() {
                let dominates = greedy.objectives()[0] <= pseudo_obj[0]
                    && greedy.objectives()[1] <= pseudo_obj[1]
                    && greedy.objectives()[2] <= pseudo_obj[2]
                    && greedy.objectives()[3] <= pseudo_obj[3]
                    && (greedy.objectives()[0] < pseudo_obj[0]
                        || greedy.objectives()[1] < pseudo_obj[1]
                        || greedy.objectives()[2] < pseudo_obj[2]
                        || greedy.objectives()[3] < pseudo_obj[3]);

                assert!(
                    !dominates,
                    "Greedy solution {g_idx} with objectives {:?} dominates pseudo-solver solution {p_idx} with objectives {:?}",
                    greedy.objectives(),
                    pseudo_obj,
                );
            }
        }
    }

}
