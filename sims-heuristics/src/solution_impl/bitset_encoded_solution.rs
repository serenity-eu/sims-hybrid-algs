// Bitset-based solution implementation
// This module is only compiled when the "bitmaps" feature is enabled.

use fixedbitset::FixedBitSet;
#[cfg(feature = "bitmaps")]
use pareto::Objectives;
use pareto::{HasObjectives, MoSolution, Random};
use rand::SeedableRng;
use rand::{Rng, seq::IteratorRandom};
use std::{collections::BinaryHeap, fmt::Debug, hash::Hash, time::Duration};

use crate::objectives;
use crate::probabilistic_probing_neighborhood::{
    ObjectiveBasedSelector, ProbabilisticProbingNeighborhood, ProbingConfig,
};
use crate::problem::{ComparableImage, ImageObjectiveDeltas, Problem, ScaledObjectiveDeltas};
use crate::residual_problem::ResidualProblem;
use crate::residual_solution::ResidualSolution;
use crate::solution::{ImageSet, MergeableWithResidual, SIMSCore, SIMSModifiable, SIMSSolution};
use crate::solution_set_impl::NdTreeSolutionSet;
use crate::timer::Timer;
use crate::trackers::{ObjectiveTracker, StandardTracker, StandardTrackerArray, TrackerCollection};
use crate::util::IntersectionIterator;
use itertools::Itertools;

/// A temporary state representing a solution with specific images removed.
/// Used as an intermediate step in creating a `ResidualProblem`.
/// Uses `FixedBitSet` for efficient storage and set operations.
pub struct UndercoveredSolution<const D: usize> {
    pub partial_selected_images: FixedBitSet,
    pub removed_images: FixedBitSet,
    pub uncovered_elements: FixedBitSet,
    pub partial_trackers: StandardTrackerArray<D>,
}

/// Lightweight view for `ImageSet` operations - only used for tracker `peek_delta` calls
struct ImageSetView<'a> {
    selected_images: &'a FixedBitSet,
}

impl<const D: usize> crate::solution::ImageSet<D> for ImageSetView<'_> {
    fn selected_images(&self) -> Vec<usize> {
        self.selected_images.ones().collect()
    }

    fn unselected_images(&self) -> Vec<usize> {
        self.selected_images.zeroes().collect()
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
pub struct BitsetEncodedSolution<const D: usize> {
    pub selected_images: FixedBitSet,
    pub objectives: Objectives<D>,
    pub timestamp: Duration,
    pub trackers: StandardTrackerArray<D>,
}

// Iterator types for BitsetEncodedSolution - leverage FixedBitSet's built-in iterators
pub type BitsetSelectedImagesIter<'a> = fixedbitset::Ones<'a>;
pub type BitsetUnselectedImagesIter<'a> = fixedbitset::Zeroes<'a>;

impl<const D: usize> HasObjectives<D> for BitsetEncodedSolution<D> {
    fn objectives(&self) -> &pareto::Objectives<D> {
        &self.objectives
    }
}

impl<const D: usize> MoSolution<D> for BitsetEncodedSolution<D> {}

// Implement ImageSet<D> trait
impl<const D: usize> ImageSet<D> for BitsetEncodedSolution<D> {
    fn selected_images(&self) -> Vec<usize> {
        self.selected_images.ones().collect()
    }

    fn unselected_images(&self) -> Vec<usize> {
        self.selected_images.zeroes().collect()
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
impl<const D: usize> SIMSCore<D> for BitsetEncodedSolution<D> {
    fn to_debug_solution(&self) -> SIMSSolution {
        SIMSSolution {
            selected_images: self.selected_images.ones().collect(),
        }
    }

    fn objectives_mut(&mut self) -> &mut pareto::Objectives<D> {
        &mut self.objectives
    }
}

// Implement Random trait from pareto
impl<const D: usize> Random for BitsetEncodedSolution<D> {
    fn random() -> Self {
        panic!("BitsetEncodedSolution::random() needs a Problem parameter")
    }

    fn random_with_seed(_seed: u64) -> Self {
        panic!("BitsetEncodedSolution::random_with_seed() needs a Problem parameter")
    }
}

// Implement SIMSModifiable trait
impl<const D: usize> SIMSModifiable<D> for BitsetEncodedSolution<D> {
    type Trackers = StandardTrackerArray<D>;

    fn trackers(&self) -> &Self::Trackers {
        &self.trackers
    }

    fn trackers_mut(&mut self) -> &mut Self::Trackers {
        &mut self.trackers
    }

    fn add_image(&mut self, image_index: usize, problem: &Problem<Self, D>) {
        self.add_image(image_index, problem);
    }

    fn remove_image(&mut self, image_index: usize, problem: &Problem<Self, D>) {
        self.remove_image(image_index, problem);
    }

    fn scaled_image_objective_deltas(
        &self,
        images: &[usize],
        problem: &Problem<Self, D>,
    ) -> Vec<ScaledObjectiveDeltas<D>> {
        self.scaled_image_objective_deltas_impl(images.iter().copied(), problem)
    }

    fn find_best_image_to_add(&self, problem: &Problem<Self, D>) -> Option<usize> {
        let unselected: Vec<usize> = self.selected_images.zeroes().collect();
        if unselected.is_empty() {
            return None;
        }

        // Greedy add - best unselected image according to some heuristic
        let scaled_objective_deltas =
            self.scaled_image_objective_deltas_impl(unselected.iter().copied(), problem);

        let min_index = (0..scaled_objective_deltas.len()).min_by(|&i, &j| {
            // Use first component of scaled objectives
            scaled_objective_deltas[i].scaled_deltas[0]
                .partial_cmp(&scaled_objective_deltas[j].scaled_deltas[0])
                .unwrap()
        })?;

        Some(scaled_objective_deltas[min_index].image_index)
    }

    fn find_best_image_to_remove(&self, problem: &Problem<Self, D>) -> Option<usize> {
        let selected: Vec<usize> = self.selected_images.ones().collect();
        if selected.is_empty() {
            return None;
        }

        // Greedy remove - worst selected image according to some heuristic
        let scaled_objective_deltas =
            self.scaled_image_objective_deltas_impl(selected.iter().copied(), problem);

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
        problem: &Problem<Self, D>,
        timer: &Timer,
        is_deterministic: bool,
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
                let worst_images_span = debug_span!("compute_worst_images");
                let worst_images = {
                    let _worst_guard = worst_images_span.enter();
                    self.worst_selected_images(problem, is_deterministic)
                };
                
                let combinations_span = debug_span!("generate_combinations", k = k);
                let _comb_guard = combinations_span.enter();
                worst_images
                    .into_iter()
                    .combinations(k as usize)
                    .collect()
            }
        };

        let mut residual_solutions: Vec<Self> = Vec::new();

        let solve_span = debug_span!("solve_residual_problems", num_problems = removal_candidates_lists.len());
        let _solve_guard = solve_span.enter();

        for (idx, removal_candidates) in removal_candidates_lists.into_iter().enumerate() {
            let problem_span = debug_span!("residual_problem", problem_index = idx);
            let _problem_guard = problem_span.enter();
            
            let create_span = debug_span!("create_residual_problem");
            let residual_problem_opt = {
                let _create_guard = create_span.enter();
                self.create_residual_problem(removal_candidates, problem, is_deterministic)
            };
            
            if let Some(mut residual_problem) = residual_problem_opt {
                let solve_residual_span = debug_span!("solve_residual");
                let neighborhood_iter = {
                    let _solve_residual_guard = solve_residual_span.enter();
                    residual_problem.solve::<NdTreeSolutionSet<ResidualSolution<D>, D>>(timer)
                };
                
                let extend_span = debug_span!("extend_solutions");
                let _extend_guard = extend_span.enter();
                residual_solutions.extend(neighborhood_iter.into_iter());
            }
            if timer.is_expired() {
                break;
            }
        }

        return residual_solutions;
    }

    fn is_valid(&self, problem: &Problem<Self, D>) -> bool {
        let mut covered_elements = vec![false; problem.universe.len()];
        self.selected_images().for_each(|image_index| {
            problem.images[image_index].parts.iter().for_each(|&part| {
                covered_elements[part] = true;
            });
        });

        let all_elements_covered = covered_elements.iter().all(|&covered| covered);
        if !all_elements_covered {
            log::error!("Not all elements are covered");
            return false;
        }

        // Note: clear_parts_counts and element_coverage validation removed - 
        // these are now maintained automatically by trackers

        return self.are_objectives_valid(problem);
    }
}

// Implement EncodedSolution trait
impl<const D: usize> crate::solution::EncodedSolution<D> for BitsetEncodedSolution<D> {
    fn timestamp(&self) -> Duration {
        self.timestamp
    }
}

impl<const D: usize> BitsetEncodedSolution<D> {
    /// Creates a `BitsetEncodedSolution` from a list of selected image indices.
    #[must_use]
    pub fn from_selected_images(selected_images_vec: &[usize], problem: &Problem<Self, D>) -> Self {
        // Initialize with empty solution - no images selected
        let mut solution = Self {
            selected_images: FixedBitSet::with_capacity(problem.images.len()),
            objectives: [0; D], // Will be recalculated below
            timestamp: Duration::new(0, 0), // Initial solutions have timestamp 0
            trackers: StandardTrackerArray::new(problem),
        };

        // Calculate correct objectives for empty solution (no images selected)
        // This is crucial for CloudyArea which should start at total universe area
        solution.recalculate_objectives(problem);

        // Now add the specified images
        for &image_index in selected_images_vec {
            solution.add_image(image_index, problem);
        }
        solution
    }

    /// Generate a random feasible solution (choose element randomly, then choose image randomly from those that contain the element iff it is not already covered by another image)
    ///
    /// # Panics
    ///
    /// Panics if there is no image covering an uncovered element (i.e., `.unwrap()` fails).
    #[must_use]
    pub fn random_with_seed(problem: &Problem<Self, D>, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut selected_images = FixedBitSet::with_capacity(problem.images.len());
        let mut covered_elements = vec![false; problem.universe.len()];
        let mut num_covered_elements = 0;

        while num_covered_elements < problem.universe.len() {
            let element_index = rng.random_range(0..problem.universe.len());
            if covered_elements[element_index] {
                continue;
            }

            // Choose random image that covers the element
            let image_index = problem.universe[element_index]
                .images
                .iter()
                .filter(|&image_index| !selected_images[*image_index])
                .choose(&mut rng)
                .unwrap();
            selected_images.set(*image_index, true);

            // Mark all elements of the image as covered
            problem.images[*image_index].parts.iter().for_each(|&part| {
                if !covered_elements[part] {
                    covered_elements[part] = true;
                    num_covered_elements += 1;
                }
            });
        }

        let mut sims_solution = Self {
            selected_images,
            objectives: [0; D],
            timestamp: Duration::new(0, 0),
            trackers: StandardTrackerArray::new(problem),
        };
        sims_solution.compute_objectives(problem);
        sims_solution
    }

    /// Generate a random feasible solution
    #[must_use]
    pub fn random(problem: &Problem<Self, D>) -> Self {
        Self::random_with_seed(problem, rand::random())
    }

    /// Compute the objectives of the solution
    fn compute_objectives(&mut self, problem: &Problem<Self, D>) {
        // Use the new generic objective calculation system
        self.recalculate_objectives(problem);
    }

    /// Compute area covered by clouds
    #[must_use]
    pub fn cloudy_area(&self, problem: &Problem<Self, D>) -> u64 {
        let mut clear_elements = vec![false; problem.universe.len()];
        self.selected_images().for_each(|image_index| {
            problem.images[image_index]
                .clear_parts
                .iter()
                .for_each(|&clear_part| {
                    clear_elements[clear_part] = true;
                });
        });
        let cloudy_area = clear_elements
            .iter()
            .enumerate()
            .filter_map(|(element_index, &is_clear)| {
                if is_clear {
                    None
                } else {
                    Some(problem.universe[element_index].area)
                }
            })
            .sum();
        return cloudy_area;
    }

    /// Compute total cost
    #[must_use]
    pub fn total_cost(&self, problem: &Problem<Self, D>) -> u64 {
        self.selected_images()
            .map(|image_index| problem.images[image_index].cost())
            .sum()
    }

    /// Scalarizing function using weighted sum, for solution quality comparison
    #[must_use]
    pub fn scalarizing_fn(&self, weights: &[f32; D], _max_values: pareto::Objectives<D>) -> f32 {
        return objectives::weighted_sum(&self.objectives, weights);
    }

    /// Check if objective values are correct
    #[must_use]
    pub fn are_objectives_valid(&self, problem: &Problem<Self, D>) -> bool {
        let first_objective_is_valid = self.objectives[0] == self.total_cost(problem);
        if !first_objective_is_valid {
            log::trace!(
                "First objective is invalid. Expected {}, got {}",
                self.total_cost(problem),
                self.objectives[0]
            );
            return false;
        }
        let second_objective_is_valid = self.objectives[1] == self.cloudy_area(problem);
        if !second_objective_is_valid {
            log::trace!(
                "Second objective is invalid. Expected {}, got {}",
                self.cloudy_area(problem),
                self.objectives[1]
            );
            return false;
        }
        return true;
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

    /// Remove image at index i
    ///
    /// # Panics
    ///
    /// Panics if `D` is not equal to 2, as only `D = 2` is currently supported.
    pub fn remove_image(&mut self, i: usize, problem: &Problem<Self, D>) {
        debug_assert!(
            self.are_objectives_valid(problem),
            "Objectives are invalid before removing image"
        );

        log::debug!("REMOVE_IMAGE: Starting removal of image {i}");
        log::debug!("  Current objectives: {:?}", self.objectives);

        // Use trackers for delta calculation
        let mut deltas = [0i64; D];
        for (obj_index, delta) in deltas.iter_mut().enumerate().take(D) {
            *delta = self.trackers().get(obj_index).peek_delta(i, true, problem, self);
        }
        
        objectives::apply_delta(&mut self.objectives, &deltas);
        
        // Apply tracker updates
        for obj_index in 0..D {
            self.trackers_mut().get_mut(obj_index).apply(i, true, problem);
        }

        self.selected_images.set(i, false);
        
        log::debug!("REMOVE_IMAGE: Completed removal of image {i}");
        log::debug!("  Final objectives: {:?}", self.objectives);
        
        debug_assert!(
            self.are_objectives_valid(problem),
            "Objectives are invalid after removing image"
        );
    }

    /// Add image at index i
    pub fn add_image(&mut self, i: usize, problem: &Problem<Self, D>) {
        log::debug!("ADD_IMAGE: Starting addition of image {i}");
        log::debug!("  Current objectives: {:?}", self.objectives);
        
        // Use trackers for delta calculation
        let mut deltas = [0i64; D];
        for (obj_index, delta) in deltas.iter_mut().enumerate().take(D) {
            *delta = self.trackers().get(obj_index).peek_delta(i, false, problem, self);
        }
        
        objectives::apply_delta(&mut self.objectives, &deltas);
        
        // Apply tracker updates
        for obj_index in 0..D {
            self.trackers_mut().get_mut(obj_index).apply(i, false, problem);
        }

        self.selected_images.set(i, true);
        
        log::debug!("ADD_IMAGE: Completed addition of image {i}");
        log::debug!("  Final objectives: {:?}", self.objectives);
    }

    /// Check whether image at index i can be replaced by another image(s)
    #[must_use]
    pub fn is_replaceable(&self, i: usize, problem: &Problem<Self, D>) -> bool {
        // For each part of the image, check if there is another image that covers the part
        self.unselected_images()
            .any(|image_index| problem.overlap_matrix[i][image_index] > 0)
    }

    /// Generate random weights for objectives
    #[must_use]
    pub fn generate_weights(&self) -> [f32; D] {
        return objectives::generate_weights::<D>();
    }

    /// Transition to an `UndercoveredSolution` state by removing specific images.
    /// This is the first step in creating a `ResidualProblem`.
    #[must_use]
    pub fn build_undercovered_solution(
        &self,
        images_to_remove: &[usize],
        problem: &Problem<Self, D>,
    ) -> UndercoveredSolution<D> {
        // 1. Create bitset of removed images
        let mut removed_images = FixedBitSet::with_capacity(self.selected_images.len());
        for &img in images_to_remove {
            removed_images.insert(img);
        }

        // 2. Create partial solution using bitwise difference (A \ B)
        // This is much faster than iterating and unsetting individual bits
        let mut partial_selected_images = self.selected_images.clone();
        partial_selected_images.difference_with(&removed_images);

        // 3. Clone trackers and apply removals
        let mut partial_trackers = self.trackers.clone();
        for &img_idx in images_to_remove {
            for obj_index in 0..D {
                partial_trackers.get_mut(obj_index).apply(img_idx, true, problem);
            }
        }

        let mut uncovered_elements = FixedBitSet::with_capacity(problem.universe.len());

        // Efficiently find uncovered elements using the CloudyArea tracker from the partial trackers
        let cloudy_tracker_idx = problem.objective_types.iter()
            .position(|obj| matches!(obj, crate::objectives::ObjectiveType::CloudyArea));

        if let Some(idx) = cloudy_tracker_idx {
            if let StandardTracker::CloudyArea(state) = partial_trackers.get(idx) {
                // Check which elements are uncovered (count == 0)
                // O(U) where U is universe size
                for (elem_idx, &count) in state.counts.iter().enumerate() {
                    if count == 0 {
                        uncovered_elements.insert(elem_idx);
                    }
                }
            } else {
                // Fallback if tracker state is unexpected
                Self::calculate_uncovered_fallback_for_partial(&partial_selected_images, problem, &mut uncovered_elements);
            }
        } else {
            // Fallback if CloudyArea objective is not present
            Self::calculate_uncovered_fallback_for_partial(&partial_selected_images, problem, &mut uncovered_elements);
        }

        UndercoveredSolution {
            partial_selected_images,
            removed_images,
            uncovered_elements,
            partial_trackers,
        }
    }

    /// Helper to calculate uncovered elements when tracker is not available
    fn calculate_uncovered_fallback_for_partial(partial_selected: &FixedBitSet, problem: &Problem<Self, D>, uncovered: &mut FixedBitSet) {
        for element_index in 0..problem.universe.len() {
             let is_covered = partial_selected.ones().any(|img_idx| {
                 problem.images[img_idx].parts.contains(&element_index)
             });
             if !is_covered {
                 uncovered.insert(element_index);
             }
        }
    }

    /// Create residual problem, composed of removed images, candidates to be added, and images covering the rest of the uncovered elements.
    #[must_use]
    pub fn create_residual_problem<'a>(
        &'a self,
        mut removal_candidates_indices: Vec<usize>,
        problem: &'a Problem<Self, D>,
        is_deterministic: bool,
    ) -> Option<ResidualProblem<'a, Self, D>> {
        // Step 1: Transition to Undercovered State (includes trackers)
        let undercovered_solution = self.build_undercovered_solution(&removal_candidates_indices, problem);

        // Step 2: Extract data from the intermediate state
        let uncovered_elements_indices: Vec<usize> = undercovered_solution.uncovered_elements.ones().collect();
        
        // Get clear parts counts for uncovered elements from CloudyArea tracker
        let original_clear_parts_counts: Vec<usize> = problem.objective_types.iter()
            .position(|obj| matches!(obj, crate::objectives::ObjectiveType::CloudyArea))
            .map_or_else(
                || vec![0; uncovered_elements_indices.len()], 
                |cloudy_area_tracker| {
                    if let StandardTracker::CloudyArea(state) = undercovered_solution.partial_trackers.get(cloudy_area_tracker) {
                        uncovered_elements_indices
                            .iter()
                            .map(|&element| state.counts[element] as usize)
                            .collect()
                    } else {
                        vec![0; uncovered_elements_indices.len()]
                    }
                }
            );

        // Step 3: Find candidates to fix the undercovered state using the static helper
        let mut best_addition_candidates = Self::best_unselected_images_with_trackers(
            &undercovered_solution.partial_selected_images,
            &undercovered_solution.partial_trackers,
            &uncovered_elements_indices, 
            problem, 
            is_deterministic
        )?;

        removal_candidates_indices.sort_unstable();
        best_addition_candidates.sort_unstable();

        Some(ResidualProblem::new(
            self.clone(), // unmodified_solution
            undercovered_solution.partial_trackers,
            &removal_candidates_indices,
            &best_addition_candidates,
            uncovered_elements_indices,
            original_clear_parts_counts,
            problem,
        ))
    }

    /// Get indices of the best replacement image(s) which is not selected yet, returns None when image cannot be replaced
    /// Static helper that accepts trackers and `selected_images` directly
    #[must_use]
    pub fn best_unselected_images_with_trackers(
        partial_selected_images: &FixedBitSet,
        partial_trackers: &StandardTrackerArray<D>,
        uncovered_elements: &[usize],
        problem: &Problem<Self, D>,
        is_deterministic: bool,
    ) -> Option<Vec<usize>> {
        let unselected_images: Vec<usize> = partial_selected_images.zeroes().collect();

        // If there is no unselected images, return
        if unselected_images.is_empty() {
            return None;
        }

        let weights: [f32; D] = if is_deterministic {
            // For deterministic mode, use equal weights
            let equal_weight = 1.0 / D as f32;
            [equal_weight; D]
        } else {
            objectives::generate_weights::<D>()
        };

        // Calculate scaled deltas for unselected images
        let unselected_images_scaled_deltas: Vec<ScaledObjectiveDeltas<D>> = {
            let raw_comparable_images: Vec<ImageObjectiveDeltas<D>> = unselected_images.iter()
                .map(|&image_index| {
                    let mut deltas = [0i64; D];
                    for (obj_index, delta) in deltas.iter_mut().enumerate().take(D) {
                        // Use trackers to peek at delta (adding image, so is_removing = false)
                        *delta = partial_trackers.get(obj_index).peek_delta(
                            image_index, 
                            false, 
                            problem, 
                            &ImageSetView { selected_images: partial_selected_images }
                        );
                    }
                    ImageObjectiveDeltas {
                        image_index,
                        deltas,
                    }
                })
                .collect();

            // Use global objective bounds for normalization
            let normalization_ranges: Vec<f32> = problem.objective_bounds().as_ref().map_or_else(
                || {
                    problem.max_objectives().iter().map(|&max_val| {
                        if max_val > 0 { max_val as f32 } else { 1.0 }
                    }).collect()
                },
                |bounds| {
                    bounds.iter().map(|bound| {
                        let range = bound[1] as f32 - bound[0] as f32;
                        if range > 0.0 { range } else { 1.0 }
                    }).collect()
                }
            );

            raw_comparable_images
                .iter()
                .map(|objective_deltas| {
                    let mut scaled_deltas = [0.0f32; D];
                    let raw_deltas = objective_deltas.deltas;

                    for i in 0..D {
                        scaled_deltas[i] = raw_deltas[i].abs() as f32 / normalization_ranges[i];
                    }

                    ScaledObjectiveDeltas {
                        image_index: objective_deltas.image_index,
                        raw_deltas,
                        scaled_deltas,
                    }
                })
                .collect()
        };

        let mut comparable_unselected_images = unselected_images_scaled_deltas
            .into_iter()
            .filter_map(|scaled_objective_deltas| {
                let image = &problem.images[scaled_objective_deltas.image_index];
                // If there are no uncovered_elements, added images can stil bring value by adding clear parts
                if !uncovered_elements.is_empty() {
                    let covered_elements_count = image
                        .parts
                        .iter()
                        .intersection(uncovered_elements.iter())
                        .count();

                    // If image does not cover any uncovered elements, skip it
                    if covered_elements_count == 0 {
                        return None;
                    }
                }

                let comparision_heur_key =
                    objectives::weighted_sum_f32(&scaled_objective_deltas.scaled_deltas, &weights);

                Some(ComparableImage {
                    index: scaled_objective_deltas.image_index,
                    key: (1_000_000.0 * comparision_heur_key) as usize,
                })
            })
            .collect::<Vec<ComparableImage>>();

        if comparable_unselected_images.is_empty() {
            return None;
        }
        let best_unselected_images = if comparable_unselected_images.len() > 9 {
            comparable_unselected_images
                .select_nth_unstable_by_key(9, |comparable_image| comparable_image.key)
                .0
                .iter()
                .map(|comparable_image| comparable_image.index)
                .collect::<Vec<usize>>()
        } else {
            comparable_unselected_images
                .into_iter()
                .map(|comparable_image| comparable_image.index)
                .collect::<Vec<usize>>()
        };
        Some(best_unselected_images)
    }

    /// Get indices of the best replacement image(s) which is not selected yet, returns None when image cannot be replaced
    #[must_use]
    pub fn best_unselected_images(
        &self,
        uncovered_elements: &[usize],
        problem: &Problem<Self, D>,
        is_deterministic: bool,
    ) -> Option<Vec<usize>> {
        Self::best_unselected_images_with_trackers(
            &self.selected_images,
            &self.trackers,
            uncovered_elements,
            problem,
            is_deterministic,
        )
    }

    #[must_use]
    pub fn worst_selected_images(
        &self,
        problem: &Problem<Self, D>,
        is_deterministic: bool,
    ) -> Vec<usize> {
        let weights: [f32; D] = if is_deterministic {
            // For deterministic mode, use equal weights
            let equal_weight = 1.0 / D as f32;
            [equal_weight; D]
        } else {
            self.generate_weights()
        };

        let selected_images: Vec<usize> = self.selected_images().collect();
        let selected_images_scaled_deltas: Vec<ScaledObjectiveDeltas<D>> =
            self.scaled_image_objective_deltas(&selected_images, problem);

        let comparable_selected_images: BinaryHeap<ComparableImage> = selected_images_scaled_deltas
            .into_iter()
            .map(|scaled_image_deltas| {
                let image = &problem.images[scaled_image_deltas.image_index];
                let covered_elements_count = image.parts.len();

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
            .take(9)
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
        problem: &Problem<Self, D>,
    ) -> Vec<ScaledObjectiveDeltas<D>> {
        let raw_comparable_images: Vec<ImageObjectiveDeltas<D>> = images
            .map(|image_index| {
                // Use the new generic objective delta calculation system
                let mut deltas = [0i64; D];
                for (obj_index, delta) in deltas.iter_mut().enumerate().take(D) {
                    *delta = self.calculate_objective_delta(obj_index, image_index, problem);
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
                problem.max_objectives().iter().map(|&max_val| {
                    if max_val > 0 { max_val as f32 } else { 1.0 }
                }).collect()
            },
            |bounds| {
                bounds.iter().map(|bound| {
                    let range = bound[1] as f32 - bound[0] as f32;
                    if range > 0.0 { range } else { 1.0 }
                }).collect()
            }
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

impl PartialOrd for BitsetEncodedSolution<2> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BitsetEncodedSolution<2> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.objectives.first().cmp(&other.objectives.first())
    }
}

impl<const D: usize> PartialEq for BitsetEncodedSolution<D> {
    fn eq(&self, other: &Self) -> bool {
        self.selected_images == other.selected_images
    }
}

impl<const D: usize> Eq for BitsetEncodedSolution<D> {}

impl<const D: usize> Hash for BitsetEncodedSolution<D> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.selected_images.hash(state);
    }
}

#[expect(
    clippy::missing_fields_in_debug,
    reason = "Custom Debug impl only shows relevant fields for readability"
)]
impl<const D: usize> Debug for BitsetEncodedSolution<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let selected_images = self.selected_images().collect::<Vec<usize>>();
        f.debug_struct("BitsetEncodedSolution")
            .field("objectives", &self.objectives)
            .field("images_count", &selected_images.len())
            .field("selected_images", &selected_images)
            .finish()
    }
}

impl<const D: usize> MergeableWithResidual<D> for BitsetEncodedSolution<D> {
    fn merge_residual_solution(
        &mut self,
        residual_solution: &ResidualSolution<D>,
        residual_problem: &ResidualProblem<'_, Self, D>,
        problem: &Problem<Self, D>,
    ) {
        residual_solution
            .selected_images
            .iter()
            .map(|&condensed_idx| residual_problem.image_index_map[condensed_idx])
            .for_each(|image_index| {
                self.add_image(image_index, problem);
            });

        // Inherit timestamp from ResidualSolution
        self.timestamp = residual_solution.timestamp;
    }
}

// Implementation of ProbabilisticProbingNeighborhood trait for BitsetEncodedSolution
impl<const D: usize> ProbabilisticProbingNeighborhood<D> for BitsetEncodedSolution<D> {
    #[allow(clippy::too_many_lines)]
    fn probabilistic_probing_neighborhood(
        &self,
        k: u32,
        problem: &Problem<Self, D>,
        timer: &Timer,
        is_deterministic: bool,
        probing_probability: f64,
        max_probes: usize,
        objective_weights: Option<&[f64; D]>,
    ) -> Vec<Self> {
        // Create probing configuration
        let config = ProbingConfig {
            probing_probability,
            max_probes,
            objective_weights: objective_weights.copied(),
            temperature: 1.0,
            improvement_threshold: 0.01,
        };

        // Initialize random number generator
        let mut rng = if is_deterministic {
            rand::rngs::StdRng::seed_from_u64(42)
        } else {
            rand::rngs::StdRng::seed_from_u64(rand::random())
        };

        let mut neighborhood_solutions = Vec::new();
        let mut probes_attempted = 0;

        // Generate candidate moves based on k-opt strategy
        let removal_candidates = if k == 1 {
            // Single image removal - select probabilistically from selected images
            self.selected_images()
                .filter(|&selected_image| {
                    // Only consider images that can be safely removed
                    self.is_replaceable(selected_image, problem)
                        && selected_image < problem.images.len()
                })
                .collect::<Vec<_>>()
        } else {
            // Multi-image removal - use worst selected images as candidates
            self.worst_selected_images(problem, is_deterministic)
                .into_iter()
                .filter(|&image_idx| image_idx < problem.images.len())
                .collect()
        };

        // Early return if no valid candidates
        if removal_candidates.is_empty() {
            log::debug!("No valid removal candidates found for probabilistic probing");
            return Vec::new();
        }

        // Calculate objective deltas for candidates to guide probabilistic selection
        let candidate_deltas: Vec<(usize, ScaledObjectiveDeltas<D>)> = removal_candidates
            .iter()
            .map(|&image_index| {
                let deltas =
                    self.scaled_image_objective_deltas_impl(std::iter::once(image_index), problem);
                (image_index, deltas.into_iter().next().unwrap())
            })
            .collect();

        // Create objective-based selector for probabilistic probing
        let selector = ObjectiveBasedSelector::new(candidate_deltas, config.clone());

        while probes_attempted < max_probes && !timer.is_expired() {
            // Probabilistically select images to remove based on objective improvements
            let images_to_remove = if k == 1 {
                selector.select_candidates(&mut rng, 1)
            } else {
                // For k > 1, select combination of images
                let selected_indices = selector.select_candidates(&mut rng, k as usize);
                selected_indices
                    .into_iter()
                    .combinations(k as usize)
                    .next()
                    .unwrap_or_default()
            };

            if images_to_remove.is_empty() {
                log::trace!("No images selected for removal in probe {probes_attempted}");
                probes_attempted += 1;
                continue;
            }

            log::trace!(
                "Probe {}: removing {} images: {:?}",
                probes_attempted,
                images_to_remove.len(),
                images_to_remove
            );

            // Apply probabilistic acceptance based on objective space improvements
            let should_explore = if is_deterministic {
                probes_attempted < max_probes / 2 // Explore first half deterministically
            } else {
                let random_value = rand::random::<f64>();
                random_value < probing_probability
            };

            if should_explore {
                log::trace!(
                    "Probe {}: exploring with {} images to remove",
                    probes_attempted,
                    images_to_remove.len()
                );

                // Generate new solutions directly by modifying current solution
                let generated_solutions =
                    self.generate_neighbor_solutions(&images_to_remove, problem, &config, &mut rng);

                log::trace!(
                    "Probe {}: generated {} candidate solutions",
                    probes_attempted,
                    generated_solutions.len()
                );

                // Strict validation and filtering of generated solutions
                let mut valid_count = 0;
                let mut invalid_count = 0;

                for solution in generated_solutions {
                    // Early check: verify all elements are covered (quick check)
                    let uncovered_count = self.find_uncovered_elements(&solution, problem).len();
                    if uncovered_count > 0 {
                        log::trace!("Rejecting solution with {uncovered_count} uncovered elements");
                        invalid_count += 1;
                        continue;
                    }

                    // Full validation and acceptance check
                    if solution.is_valid(problem)
                        && self.should_accept_solution(&solution, &config, problem)
                    {
                        neighborhood_solutions.push(solution);
                        valid_count += 1;
                    } else {
                        log::trace!("Solution failed validation or acceptance criteria");
                        invalid_count += 1;
                    }
                }

                log::trace!(
                    "Probe {probes_attempted}: {valid_count} valid, {invalid_count} invalid solutions"
                );
            }

            probes_attempted += 1;

            // Early termination if enough good solutions found
            if neighborhood_solutions.len() >= max_probes / 2 {
                break;
            }
        }

        // Apply additional filtering based on diversity in objective space
        // Note: All solutions in neighborhood_solutions are already validated via is_valid()
        let initial_solution_count = neighborhood_solutions.len();
        let filtered_solutions =
            self.filter_diverse_solutions(neighborhood_solutions, problem, &config);

        log::debug!(
            "Probabilistic probing: {} valid solutions from {} probes ({:.1}% success rate), {} filtered for diversity",
            initial_solution_count,
            probes_attempted,
            (initial_solution_count as f64 / probes_attempted as f64) * 100.0,
            filtered_solutions.len()
        );

        // Log warning if no valid solutions were generated
        if filtered_solutions.is_empty() && probes_attempted > 0 {
            log::warn!(
                "Probabilistic probing failed to generate any valid solutions after {probes_attempted} probes"
            );
        }

        filtered_solutions
    }
}

// Helper methods for probabilistic probing implementation
impl<const D: usize> BitsetEncodedSolution<D> {
    /// Generate neighbor solutions directly by removing specified images and adding new ones
    fn generate_neighbor_solutions(
        &self,
        images_to_remove: &[usize],
        problem: &Problem<Self, D>,
        config: &ProbingConfig<D>,
        rng: &mut rand::rngs::StdRng,
    ) -> Vec<Self> {
        log::trace!(
            "Generating solutions by removing {} images: {:?}",
            images_to_remove.len(),
            images_to_remove
        );

        let mut generated_solutions = Vec::new();

        // Create a base solution by removing specified images
        let mut base_solution = self.clone();
        for &image_idx in images_to_remove {
            base_solution.remove_image(image_idx, problem);
        }

        // Find uncovered elements after removal
        let uncovered_after_removal = self.find_uncovered_elements(&base_solution, problem);
        log::trace!(
            "After removing {} images, {} elements are uncovered",
            images_to_remove.len(),
            uncovered_after_removal.len()
        );

        // Generate multiple variants by ensuring complete coverage
        let max_variants = config.max_probes.min(10); // Reduced variants for debugging

        for variant_idx in 0..max_variants {
            let mut candidate = base_solution.clone();

            // Find uncovered elements for this variant
            let uncovered_elements = self.find_uncovered_elements(&candidate, problem);

            // If no elements are uncovered, the base solution is complete
            if uncovered_elements.is_empty() {
                log::trace!("Variant {variant_idx}: base solution already complete");
                if candidate.is_valid(problem)
                    && candidate.selected_images().collect::<Vec<_>>()
                        != self.selected_images().collect::<Vec<_>>()
                {
                    generated_solutions.push(candidate);
                    log::trace!("Variant {variant_idx}: added complete base solution");
                }
                continue;
            }

            log::trace!(
                "Variant {}: {} elements need coverage",
                variant_idx,
                uncovered_elements.len()
            );

            // ENSURE complete coverage by selecting images to cover ALL uncovered elements
            let covering_images =
                self.ensure_complete_coverage(&uncovered_elements, problem, config, rng);

            log::trace!(
                "Variant {}: coverage algorithm selected {} images: {:?}",
                variant_idx,
                covering_images.len(),
                covering_images
            );

            // Add selected images to create complete solution
            for &image_idx in &covering_images {
                if !candidate.is_image_selected(image_idx) {
                    candidate.add_image(image_idx, problem);
                }
            }

            // Verify solution is complete and valid before adding
            let final_uncovered = self.find_uncovered_elements(&candidate, problem);
            log::trace!(
                "Variant {}: after adding {} images, {} elements remain uncovered",
                variant_idx,
                covering_images.len(),
                final_uncovered.len()
            );

            if final_uncovered.is_empty()
                && candidate.is_valid(problem)
                && candidate.selected_images().collect::<Vec<_>>()
                    != self.selected_images().collect::<Vec<_>>()
            {
                generated_solutions.push(candidate);
                log::trace!("Variant {variant_idx}: successfully generated valid solution");
            } else {
                // Log detailed failure information
                log::trace!(
                    "Variant {}: FAILED - uncovered: {}, valid: {}, different: {}",
                    variant_idx,
                    final_uncovered.len(),
                    candidate.is_valid(problem),
                    candidate.selected_images().collect::<Vec<_>>()
                        != self.selected_images().collect::<Vec<_>>()
                );

                if !final_uncovered.is_empty() {
                    log::trace!(
                        "Variant {variant_idx}: remaining uncovered elements: {:?}",
                        if final_uncovered.len() <= 10 {
                            format!("{final_uncovered:?}")
                        } else {
                            format!("{:?}...", &final_uncovered[..10])
                        }
                    );
                }
            }
        }

        log::trace!(
            "Generated {} complete solutions from {} variants",
            generated_solutions.len(),
            max_variants
        );
        generated_solutions
    }

    /// Find elements that are not covered by the current solution
    #[allow(clippy::unused_self)]
    fn find_uncovered_elements(&self, solution: &Self, problem: &Problem<Self, D>) -> Vec<usize> {
        let mut covered = vec![false; problem.universe.len()];

        // Mark covered elements
        for image_idx in solution.selected_images() {
            for &element in &problem.images[image_idx].parts {
                if element < covered.len() {
                    covered[element] = true;
                }
            }
        }

        // Return uncovered elements
        covered
            .iter()
            .enumerate()
            .filter_map(|(idx, &is_covered)| if is_covered { None } else { Some(idx) })
            .collect()
    }

    /// Ensure complete coverage of all uncovered elements using a greedy + probabilistic approach
    #[allow(clippy::too_many_lines)]
    fn ensure_complete_coverage(
        &self,
        uncovered_elements: &[usize],
        problem: &Problem<Self, D>,
        config: &ProbingConfig<D>,
        rng: &mut rand::rngs::StdRng,
    ) -> Vec<usize> {
        if uncovered_elements.is_empty() {
            return Vec::new();
        }

        log::trace!(
            "Starting coverage for {} uncovered elements: {:?}",
            uncovered_elements.len(),
            if uncovered_elements.len() <= 10 {
                format!("{uncovered_elements:?}")
            } else {
                format!("{:?}...", &uncovered_elements[..10])
            }
        );

        let mut selected_images = Vec::new();
        let mut remaining_uncovered = uncovered_elements.to_vec();
        let mut iteration_count = 0;
        let max_iterations = uncovered_elements.len() * 3; // Increased max iterations

        while !remaining_uncovered.is_empty() && iteration_count < max_iterations {
            iteration_count += 1;

            // Find images that can cover remaining uncovered elements
            let mut covering_candidates = Vec::new();

            for (img_idx, image) in problem.images.iter().enumerate() {
                // Skip already selected images
                if selected_images.contains(&img_idx) {
                    continue;
                }

                let covered_elements: Vec<usize> = image
                    .parts
                    .iter()
                    .filter(|&&part| remaining_uncovered.contains(&part))
                    .copied()
                    .collect();

                let coverage_count = covered_elements.len();

                if coverage_count > 0 {
                    // Prioritize coverage count heavily
                    let mut score = (coverage_count as f64) * 1000.0; // Base score heavily weighted on coverage

                    // Add small objective preference (secondary priority)
                    if let Some(weights) = &config.objective_weights {
                        let obj_deltas = self
                            .scaled_image_objective_deltas_impl(std::iter::once(img_idx), problem);
                        if let Some(delta) = obj_deltas.first() {
                            for (i, &delta_val) in delta.scaled_deltas.iter().enumerate() {
                                // Small objective influence compared to coverage
                                score += weights[i]
                                    * (-f64::from(delta_val) / config.temperature.max(1.0)).exp();
                            }
                        }
                    }

                    covering_candidates.push((img_idx, score, coverage_count, covered_elements));
                }
            }

            if covering_candidates.is_empty() {
                log::error!(
                    "No covering images found for {} remaining uncovered elements: {:?}",
                    remaining_uncovered.len(),
                    if remaining_uncovered.len() <= 20 {
                        format!("{remaining_uncovered:?}")
                    } else {
                        format!("{:?}...", &remaining_uncovered[..20])
                    }
                );
                break; // This should not happen in valid problems
            }

            // Sort by score (higher is better) - coverage count dominates
            covering_candidates
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Use more conservative selection for reliability
            let selected_idx =
                if remaining_uncovered.len() > 10 || iteration_count > max_iterations / 2 {
                    // For many uncovered elements or late iterations, use pure greedy
                    0
                } else if rand::random::<f64>() < config.probing_probability * 0.5 {
                    // Reduced probability for more reliable coverage
                    let selection_pool_size = covering_candidates.len().min(2); // Very small pool
                    rng.random_range(0..selection_pool_size)
                } else {
                    // Greedy selection (best coverage)
                    0
                };

            let (selected, _, coverage_count, covered_elements) =
                &covering_candidates[selected_idx];

            selected_images.push(*selected);

            // Remove covered elements from remaining uncovered
            let initial_uncovered_count = remaining_uncovered.len();
            remaining_uncovered.retain(|&elem| !covered_elements.contains(&elem));

            // Verify progress
            let progress = initial_uncovered_count - remaining_uncovered.len();
            if progress == 0 {
                log::warn!(
                    "No progress in iteration {iteration_count}, image {selected} should cover {coverage_count} elements but didn't"
                );
                break; // Prevent infinite loop
            }

            log::trace!(
                "Iteration {}: selected image {} covers {} elements, {} elements remain",
                iteration_count,
                selected,
                progress,
                remaining_uncovered.len()
            );
        }

        if remaining_uncovered.is_empty() {
            log::trace!(
                "Coverage successful: {} images selected after {} iterations",
                selected_images.len(),
                iteration_count
            );
        } else {
            log::error!(
                "COVERAGE FAILURE: {} elements remain uncovered after {} iterations. Selected {} images: {:?}",
                remaining_uncovered.len(),
                iteration_count,
                selected_images.len(),
                selected_images
            );
            log::error!(
                "Uncovered elements: {:?}",
                if remaining_uncovered.len() <= 20 {
                    format!("{remaining_uncovered:?}")
                } else {
                    format!("{:?}...", &remaining_uncovered[..20])
                }
            );
        }

        selected_images
    }
    /// Determine if a solution should be accepted based on objective improvements
    fn should_accept_solution(
        &self,
        candidate: &Self,
        config: &ProbingConfig<D>,
        problem: &Problem<Self, D>,
    ) -> bool {
        // Calculate objective improvements
        let mut improvement_score = 0.0;
        let weights = config.objective_weights.as_ref();

        for i in 0..D {
            let current_obj = self.calculate_objective(i, problem) as f64;
            let candidate_obj = candidate.calculate_objective(i, problem) as f64;
            let improvement = (current_obj - candidate_obj) / current_obj.max(1.0);

            let weight = weights.map_or(1.0, |w| w[i]);
            improvement_score += weight * improvement;
        }

        improvement_score >= config.improvement_threshold
    }

    /// Filter solutions to maintain diversity in objective space
    #[allow(clippy::unused_self)]
    fn filter_diverse_solutions(
        &self,
        mut solutions: Vec<Self>,
        problem: &Problem<Self, D>,
        config: &ProbingConfig<D>,
    ) -> Vec<Self> {
        if solutions.len() <= 1 {
            return solutions;
        }

        // Sort by overall objective quality
        solutions.sort_by(|a, b| {
            let score_a = Self::calculate_solution_score(a, problem, config);
            let score_b = Self::calculate_solution_score(b, problem, config);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep top solutions while maintaining diversity
        let mut filtered = Vec::new();
        let diversity_threshold = 0.1; // Minimum objective space distance

        for solution in solutions {
            if filtered.is_empty()
                || filtered.iter().all(|existing| {
                    Self::objective_distance(&solution, existing, problem) >= diversity_threshold
                })
            {
                filtered.push(solution);

                // Limit number of solutions to prevent explosion
                if filtered.len() >= config.max_probes / 4 {
                    break;
                }
            }
        }

        filtered
    }

    /// Calculate overall quality score for a solution
    fn calculate_solution_score(
        solution: &Self,
        problem: &Problem<Self, D>,
        config: &ProbingConfig<D>,
    ) -> f64 {
        let mut score = 0.0;
        let weights = config.objective_weights.as_ref();

        for i in 0..D {
            let obj_value = solution.calculate_objective(i, problem) as f64;
            let weight = weights.map_or(1.0, |w| w[i]);
            score += weight * (1.0 / (1.0 + obj_value)); // Minimize objectives
        }

        score
    }

    /// Calculate distance between two solutions in objective space
    fn objective_distance(solution1: &Self, solution2: &Self, problem: &Problem<Self, D>) -> f64 {
        let mut distance = 0.0;

        for i in 0..D {
            let obj1 = solution1.calculate_objective(i, problem) as f64;
            let obj2 = solution2.calculate_objective(i, problem) as f64;
            let max_obj = problem.max_objectives()[i] as f64;

            // Normalized distance
            let normalized_diff = (obj1 - obj2).abs() / max_obj.max(1.0);
            distance += normalized_diff * normalized_diff;
        }

        distance.sqrt()
    }
}
