use itertools::Itertools;
use log::trace;
use pareto::{HasObjectives, MoSolution, Random};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::{Rng, seq::IteratorRandom};
use std::{collections::BinaryHeap, fmt::Debug, hash::Hash, time::Duration, vec};

use crate::objective_tracker::{SimdTrackerArray, TrackerCollection};
use crate::objectives;
use crate::problem::{ComparableImage, ImageObjectiveDeltas, ScaledObjectiveDeltas};
use crate::residual_problem::ResidualProblem;
use crate::residual_solution::ResidualSolution;
use crate::solution::{ImageSet, MergeableWithResidual, SIMSCore, SIMSModifiable, SIMSSolution};
use crate::solution_set_impl::NdTreeSolutionSet;

#[derive(Clone)]
pub struct VecEncodedSolution<P, const D: usize>
where
    P: crate::problem::SetCoverProblem<D> + Clone + Send + Sync,
{
    pub selected_images: Vec<bool>,
    pub objectives: pareto::Objectives<D>,
    pub timestamp: Duration,
    _phantom: std::marker::PhantomData<P>,
}

// Iterator types for VecEncodedSolution
pub struct SelectedImagesIter<'a> {
    images: &'a Vec<bool>,
    index: usize,
}

impl SelectedImagesIter<'_> {
    #[must_use]
    pub const fn new(images: &Vec<bool>) -> SelectedImagesIter<'_> {
        SelectedImagesIter { images, index: 0 }
    }
}

impl Iterator for SelectedImagesIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.images.len() {
            if self.images[self.index] {
                let result = Some(self.index);
                self.index += 1;
                return result;
            }
            self.index += 1;
        }
        None
    }
}

pub struct UnselectedImagesIter<'a> {
    images: &'a Vec<bool>,
    index: usize,
}

impl UnselectedImagesIter<'_> {
    #[must_use]
    pub const fn new(images: &Vec<bool>) -> UnselectedImagesIter<'_> {
        UnselectedImagesIter { images, index: 0 }
    }
}

impl Iterator for UnselectedImagesIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.images.len() {
            if !self.images[self.index] {
                let result = Some(self.index);
                self.index += 1;
                return result;
            }
            self.index += 1;
        }
        None
    }
}

impl<P, const D: usize> HasObjectives<D> for VecEncodedSolution<P, D>
where
    P: crate::problem::SetCoverProblem<D> + Clone + Send + Sync,
{
    fn objectives(&self) -> &pareto::Objectives<D> {
        &self.objectives
    }
}

impl<P, const D: usize> MoSolution<D> for VecEncodedSolution<P, D> where
    P: crate::problem::SetCoverProblem<D> + Clone + Send + Sync
{
}

// Implement ImageSet<D> trait
impl<P, const D: usize> ImageSet<D> for VecEncodedSolution<P, D>
where
    P: crate::problem::SetCoverProblem<D> + Clone + Send + Sync,
{
    fn selected_images(&self) -> impl Iterator<Item = usize> {
        SelectedImagesIter::new(&self.selected_images)
    }

    fn unselected_images(&self) -> impl Iterator<Item = usize> {
        UnselectedImagesIter::new(&self.selected_images)
    }

    fn is_image_selected(&self, image_index: usize) -> bool {
        self.selected_images[image_index]
    }

    fn num_selected_images(&self) -> usize {
        self.selected_images.iter().filter(|&&x| x).count()
    }

    fn set_image(&mut self, image_index: usize, selected: bool) {
        self.selected_images[image_index] = selected;
    }
}

// Implement SIMSCore trait
impl<P, const D: usize> SIMSCore<P, D> for VecEncodedSolution<P, D>
where
    P: crate::problem::SetCoverProblem<D> + Clone + Send + Sync,
{
    fn to_debug_solution(&self) -> SIMSSolution {
        self.as_sims_solution()
    }

    fn objectives_mut(&mut self) -> &mut pareto::Objectives<D> {
        &mut self.objectives
    }
}

// Implement Random trait from pareto
impl<P, const D: usize> Random for VecEncodedSolution<P, D>
where
    P: crate::problem::SetCoverProblem<D> + Clone + Send + Sync,
{
    fn random() -> Self {
        panic!("VecEncodedSolution::random() needs a Problem parameter")
    }

    fn random_with_seed(_seed: u64) -> Self {
        panic!("VecEncodedSolution::random_with_seed() needs a Problem parameter")
    }
}

// Add utility methods for random generation that work with Problem
impl<P, const D: usize> VecEncodedSolution<P, D>
where
    P: crate::problem::SetCoverProblem<D> + Clone + Send + Sync,
{
    #[must_use]
    pub fn from_selected_images(selected_images_vec: &[usize], problem: &P) -> Self {
        // Initialize with empty solution - no images selected
        let mut solution = Self {
            selected_images: vec![false; problem.num_images()],
            objectives: [0; D],             // Will be recalculated below
            timestamp: Duration::new(0, 0), // Initial solutions have timestamp 0
            _phantom: std::marker::PhantomData,
        };

        // Calculate correct objectives for empty solution (no images selected)
        // This is crucial for CloudyArea which should start at total universe area
        solution.recalculate_objectives(problem);

        // Now add the specified images
        let mut trackers = SimdTrackerArray::new(problem);
        for &image_index in selected_images_vec {
            solution.add_image(image_index, problem, &mut trackers);
        }
        solution
    }

    /// Generate a random solution with problem parameter
    #[must_use]
    pub fn random_with_problem(problem: &P) -> Self {
        Self::random(problem)
    }

    /// Generate a random solution with seed and problem parameter
    #[must_use]
    pub fn random_with_problem_and_seed(problem: &P, seed: u64) -> Self {
        Self::random_with_seed(problem, seed)
    }
}

// Implement SIMSModifiable trait
impl<P, const D: usize> SIMSModifiable<P, D> for VecEncodedSolution<P, D>
where
    P: crate::problem::SetCoverProblem<D> + Clone + Send + Sync,
{
    type Trackers = SimdTrackerArray<D>;

    fn add_image(&mut self, image_index: usize, problem: &P, trackers: &mut Self::Trackers) {
        self.add_image(image_index, problem, trackers);
    }

    fn remove_image(&mut self, image_index: usize, problem: &P, trackers: &mut Self::Trackers) {
        self.remove_image(image_index, problem, trackers);
    }

    fn scaled_image_objective_deltas(
        &self,
        images: &[usize],
        problem: &P,
        trackers: &SimdTrackerArray<D>,
    ) -> Vec<ScaledObjectiveDeltas<D>> {
        self.scaled_image_objective_deltas(images.iter().copied(), problem, trackers)
    }

    fn find_best_image_to_add(
        &self,
        problem: &P,
        trackers: &SimdTrackerArray<D>,
    ) -> Option<usize> {
        let unselected_iter = UnselectedImagesIter::new(&self.selected_images);
        let unselected: Vec<usize> = unselected_iter.collect();
        if unselected.is_empty() {
            return None;
        }

        // Greedy add - best unselected image according to some heuristic
        let scaled_objective_deltas =
            self.scaled_image_objective_deltas(unselected.iter().copied(), problem, trackers);

        let min_index = (0..scaled_objective_deltas.len()).min_by(|&i, &j| {
            // Use first component of scaled objectives
            scaled_objective_deltas[i].scaled_deltas[0]
                .partial_cmp(&scaled_objective_deltas[j].scaled_deltas[0])
                .unwrap()
        })?;

        Some(scaled_objective_deltas[min_index].image_index)
    }

    fn find_best_image_to_remove(
        &self,
        problem: &P,
        trackers: &SimdTrackerArray<D>,
    ) -> Option<usize> {
        let selected_iter = SelectedImagesIter::new(&self.selected_images);
        let selected: Vec<usize> = selected_iter.collect();
        if selected.is_empty() {
            return None;
        }

        // Greedy remove - worst selected image according to some heuristic
        let scaled_objective_deltas =
            self.scaled_image_objective_deltas(selected.iter().copied(), problem, trackers);

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
        timer: &crate::timer::Timer,
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
                let worst_images_span = debug_span!("compute_worst_images");
                let worst_images = {
                    let _worst_guard = worst_images_span.enter();
                    self.worst_selected_images(problem, is_deterministic)
                };

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
            let problem_span = debug_span!("residual_problem", problem_index = idx);
            let _problem_guard = problem_span.enter();

            let create_span = debug_span!("create_residual_problem");
            let residual_problem_opt = {
                let _create_guard = create_span.enter();
                self.create_residual_problem(
                    removal_candidates,
                    problem,
                    is_deterministic,
                    trackers,
                )
            };

            if let Some(mut residual_problem) = residual_problem_opt {
                let solve_residual_span = debug_span!("solve_residual");
                let residual_trackers = trackers.clone();
                let neighborhood_iter = {
                    let _solve_residual_guard = solve_residual_span.enter();
                    residual_problem.solve::<NdTreeSolutionSet<ResidualSolution<D>, D>>(
                        problem,
                        timer,
                        residual_trackers,
                    )
                };

                let extend_span = debug_span!("extend_solutions");
                let _extend_guard = extend_span.enter();
                residual_solutions.extend(neighborhood_iter);
            }
            if timer.is_expired() {
                break;
            }
        }

        return residual_solutions;
    }

    fn is_valid(&self, problem: &P) -> bool {
        let mut covered_elements = vec![false; problem.num_elements()];
        self.selected_images().for_each(|image_index| {
            problem.image_elements(image_index).for_each(|part| {
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
impl<P, const D: usize> crate::solution::EncodedSolution<P, D> for VecEncodedSolution<P, D>
where
    P: crate::problem::SetCoverProblem<D> + Clone + Send + Sync,
{
    fn timestamp(&self) -> Duration {
        self.timestamp
    }
}

impl<P, const D: usize> VecEncodedSolution<P, D>
where
    P: crate::problem::SetCoverProblem<D> + Clone + Send + Sync,
{
    /// Generate a random feasible solution (choose element randomly, then choose image randomly from those that contain the element iff it is not already covered by another image)
    ///
    /// # Panics
    ///
    /// Panics if there is no image that covers an uncovered element (i.e., `.choose(&mut rng).unwrap()` fails).
    #[must_use]
    pub fn random_with_seed(problem: &P, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut selected_images = vec![false; problem.num_images()];
        let mut covered_elements = vec![false; problem.num_elements()];
        let mut num_covered_elements = 0;

        while num_covered_elements < problem.num_elements() {
            let element_index = rng.random_range(0..problem.num_elements());
            if covered_elements[element_index] {
                continue;
            }

            // Choose random image that covers the element
            let image_index = problem
                .element_images(element_index)
                .filter(|&image_index| !selected_images[image_index])
                .choose(&mut rng)
                .unwrap();
            selected_images[image_index] = true;

            // Mark all elements of the image as covered
            problem.image_elements(image_index).for_each(|part| {
                if !covered_elements[part] {
                    covered_elements[part] = true;
                    num_covered_elements += 1;
                }
            });
        }

        let mut sims_solution = Self {
            selected_images,
            objectives: [0; D],
            timestamp: Duration::new(0, 0), // Initial solutions have timestamp 0
            _phantom: std::marker::PhantomData,
        };
        sims_solution.compute_objectives(problem);
        sims_solution
    }

    /// Generate a random feasible solution
    #[must_use]
    pub fn random(problem: &P) -> Self {
        Self::random_with_seed(problem, rand::random())
    }

    /// Compute the objectives of the solution
    fn compute_objectives(&mut self, problem: &P) {
        // Use the new generic objective calculation system
        self.recalculate_objectives(problem);
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
                trace!(
                    "Objective {} is invalid. Expected {}, got {}",
                    i, expected_value, self.objectives[i]
                );
                return false;
            }
        }

        true
    }

    /// Returns iterator over indices of selected images of solution
    #[must_use]
    pub const fn selected_images(&self) -> SelectedImagesIter<'_> {
        SelectedImagesIter::new(&self.selected_images)
    }

    /// Returns iterator over indices of unselected images of solution
    const fn unselected_images(&self) -> UnselectedImagesIter<'_> {
        UnselectedImagesIter::new(&self.selected_images)
    }

    /// Remove image at index i
    pub fn remove_image(&mut self, i: usize, problem: &P, trackers: &mut SimdTrackerArray<D>) {
        debug_assert!(
            self.are_objectives_valid(problem),
            "Objectives are invalid before removing image"
        );

        // Use trackers for delta calculation
        let deltas = trackers.peek_removal_delta(i, problem, self);
        objectives::apply_delta(&mut self.objectives, &deltas);

        // Apply tracker updates
        trackers.track_image_removal(i, problem);

        self.selected_images[i] = false;
        debug_assert!(
            self.are_objectives_valid(problem),
            "Objectives are invalid after removing image"
        );
    }

    /// Add image at index i
    pub fn add_image(&mut self, i: usize, problem: &P, trackers: &mut SimdTrackerArray<D>) {
        // Use trackers for delta calculation and state update
        let deltas = trackers.track_image_addition(i, problem);
        objectives::apply_delta(&mut self.objectives, &deltas);

        self.selected_images[i] = true;
    }

    /// Check whether image at index i can be replaced by another image(s)
    #[must_use]
    pub fn is_replaceable(&self, i: usize, problem: &P) -> bool {
        // For each part of the image, check if there is another image that covers the part
        self.unselected_images()
            .any(|image_index| problem.overlap(i, image_index) > 0)
    }

    /// Generate random weights for objectives (generic version)
    #[must_use]
    pub fn generate_weights(&self) -> [f32; D] {
        return objectives::generate_weights::<D>();
    }

    /// Create residual problem, composed of removed images, candidates to be added, and images covering the rest of the uncovered elements.
    #[must_use]
    pub fn create_residual_problem(
        &self,
        mut removal_candidates_indices: Vec<usize>,
        problem: &P,
        is_deterministic: bool,
        trackers: &mut SimdTrackerArray<D>,
    ) -> Option<ResidualProblem<Self, P, D>> {
        // Apply removals to trackers
        for &removed_image_index in &removal_candidates_indices {
            trackers.track_image_removal(removed_image_index, problem);
        }

        // Create partial selected_images state
        let mut partial_selected_images = self.selected_images.clone();
        for &removed_image_index in &removal_candidates_indices {
            partial_selected_images[removed_image_index] = false;
        }

        // Get list of uncovered elements by checking which elements have zero coverage
        let uncovered_elements_indices = (0..problem.num_elements())
            .filter(|&element_index| {
                // Check if any selected image in partial state covers this element
                !partial_selected_images
                    .iter()
                    .enumerate()
                    .any(|(img_idx, &selected)| {
                        selected && problem.image_contains_element(img_idx, element_index)
                    })
            })
            .collect::<Vec<usize>>();

        // Find best image(s) to replace removed image(s)
        let mut best_addition_candidates =
            self.best_unselected_images(&uncovered_elements_indices, problem, is_deterministic)?;

        // Keep ordering stable for deterministic behavior.
        removal_candidates_indices.sort_unstable();
        best_addition_candidates.sort_unstable();

        let residual_problem = ResidualProblem::new(
            self.clone(), // Pass owned value
            &removal_candidates_indices,
            &best_addition_candidates,
            uncovered_elements_indices,
            problem,
        );

        Some(residual_problem)
    }

    // Get scaled objective deltas for list of given images
    fn scaled_image_objective_deltas<I: Iterator<Item = usize>>(
        &self,
        images: I,
        problem: &P,
        _trackers: &SimdTrackerArray<D>,
    ) -> Vec<ScaledObjectiveDeltas<D>> {
        let raw_comparable_images: Vec<ImageObjectiveDeltas<D>> = images
            .map(|image_index| {
                // Use the new generic objective delta calculation system
                let mut deltas = [0i64; D];
                for (obj_index, delta) in deltas.iter_mut().enumerate() {
                    *delta = self.calculate_objective_delta(obj_index, image_index, problem);
                }
                ImageObjectiveDeltas {
                    image_index,
                    deltas,
                }
            })
            .collect();

        // Calculate min/max for each objective dimension
        let mut min_deltas = [i64::MAX; D];
        let mut max_deltas = [i64::MIN; D];

        for image_deltas in &raw_comparable_images {
            for (i, &delta) in image_deltas.deltas.iter().enumerate() {
                min_deltas[i] = min_deltas[i].min(delta);
                max_deltas[i] = max_deltas[i].max(delta);
            }
        }

        // Calculate ranges for scaling
        let mut ranges = [1i64; D]; // Default to 1 to avoid division by zero
        for i in 0..D {
            let range = max_deltas[i] - min_deltas[i];
            if range > 0 {
                ranges[i] = range;
            }
        }

        raw_comparable_images
            .iter()
            .map(|objective_deltas| {
                let mut scaled_deltas = [0.0f32; D];
                let raw_deltas = objective_deltas.deltas;

                for i in 0..D {
                    scaled_deltas[i] = (raw_deltas[i] - min_deltas[i]) as f32 / ranges[i] as f32;
                }

                ScaledObjectiveDeltas {
                    image_index: objective_deltas.image_index,
                    raw_deltas,
                    scaled_deltas,
                }
            })
            .collect()
    }

    /// Get indices of the best replacement image(s) which is not selected yet, returns None when image cannot be replaced
    #[must_use]
    pub fn best_unselected_images(
        &self,
        uncovered_elements: &[usize],
        problem: &P,
        is_deterministic: bool,
    ) -> Option<Vec<usize>> {
        // If there is no unselected images, return
        if self.unselected_images().count() == 0 {
            return None;
        }

        let weights: [f32; D] = if is_deterministic {
            // For deterministic mode, use equal weights
            let equal_weight = 1.0 / D as f32;
            [equal_weight; D]
        } else {
            self.generate_weights()
        };

        let temp_trackers = SimdTrackerArray::new(problem);
        let unselected_images_scaled_deltas: Vec<ScaledObjectiveDeltas<D>> =
            self.scaled_image_objective_deltas(self.unselected_images(), problem, &temp_trackers);

        let mut comparable_unselected_images = unselected_images_scaled_deltas
            .into_iter()
            .filter_map(|scaled_objective_deltas| {
                let image_index = scaled_objective_deltas.image_index;
                // If there are no uncovered_elements, added images can stil bring value by adding clear parts
                if !uncovered_elements.is_empty() {
                    let covered_elements_count = problem
                        .image_elements(image_index)
                        .filter(|&element| uncovered_elements.contains(&element))
                        .count();

                    // If image does not cover any uncovered elements, skip it
                    if covered_elements_count == 0 {
                        return None;
                    }
                }

                let comparision_heur_key =
                    objectives::weighted_sum_f32(&scaled_objective_deltas.scaled_deltas, &weights);
                // / denominator;

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
        return Some(best_unselected_images);
    }

    #[must_use]
    pub fn worst_selected_images(&self, problem: &P, is_deterministic: bool) -> Vec<usize> {
        let weights: [f32; D] = if is_deterministic {
            // For deterministic mode, use equal weights
            let equal_weight = 1.0 / D as f32;
            [equal_weight; D]
        } else {
            self.generate_weights()
        };
        let temp_trackers = SimdTrackerArray::new(problem);
        let selected_images_scaled_deltas: Vec<ScaledObjectiveDeltas<D>> =
            self.scaled_image_objective_deltas(self.selected_images(), problem, &temp_trackers);

        let comparable_selected_images: BinaryHeap<ComparableImage> = selected_images_scaled_deltas
            .into_iter()
            .map(|scaled_image_deltas| {
                let image_index = scaled_image_deltas.image_index;
                let covered_elements_count = problem.image_elements(image_index).count();

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
        let selected_images = self.selected_images().collect::<Vec<usize>>();
        SIMSSolution { selected_images }
    }
}

impl<P, const D: usize> PartialEq for VecEncodedSolution<P, D>
where
    P: crate::problem::SetCoverProblem<D> + Clone + Send + Sync,
{
    fn eq(&self, other: &Self) -> bool {
        self.selected_images == other.selected_images
    }
}

impl<P, const D: usize> Eq for VecEncodedSolution<P, D> where
    P: crate::problem::SetCoverProblem<D> + Clone + Send + Sync
{
}

impl<P, const D: usize> Hash for VecEncodedSolution<P, D>
where
    P: crate::problem::SetCoverProblem<D> + Clone + Send + Sync,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.selected_images.hash(state);
    }
}

// Implement Debug for SIMSEncodedSolution by converting it to SIMSSolution
#[expect(
    clippy::missing_fields_in_debug,
    reason = "Custom Debug impl only shows relevant fields for readability"
)]
impl<P, const D: usize> Debug for VecEncodedSolution<P, D>
where
    P: crate::problem::SetCoverProblem<D> + Clone + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let selected_images = &self
            .selected_images
            .iter()
            .enumerate()
            .filter_map(|(index, &selected)| if selected { Some(index) } else { None })
            .collect::<Vec<usize>>();
        f.debug_struct("SIMSEncodedSolution")
            .field("objectives", &self.objectives)
            .field("images_count", &selected_images.len())
            .field("selected_images", selected_images)
            .finish()
    }
}

impl<P, const D: usize> MergeableWithResidual<P, D> for VecEncodedSolution<P, D>
where
    P: crate::problem::SetCoverProblem<D> + Clone + Send + Sync,
{
    fn merge_residual_solution(
        &mut self,
        residual_solution: &ResidualSolution<D>,
        residual_problem: &ResidualProblem<Self, P, D>,
        problem: &P,
        partial_trackers: &mut Self::Trackers,
    ) {
        residual_solution
            .selected_images
            .iter()
            .map(|&condensed_idx| residual_problem.image_map_condensed_to_original[condensed_idx])
            .for_each(|image_index| {
                self.add_image(image_index, problem, partial_trackers);
            });

        // Inherit timestamp from ResidualSolution
        self.timestamp = residual_solution.timestamp;
    }
}
