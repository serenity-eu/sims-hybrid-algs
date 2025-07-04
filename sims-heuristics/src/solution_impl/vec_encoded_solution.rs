use itertools::Itertools;
use log::{error, trace};
use pareto::{HasObjectives, MoSolution, Random};
use rand::SeedableRng;
use rand::{Rng, seq::IteratorRandom};
use std::{collections::BinaryHeap, fmt::Debug, hash::Hash, vec};

use crate::objectives::{self, SolutionEvaluator};
use crate::problem::{ComparableImage, ImageObjectiveDeltas, Problem, ScaledObjectiveDeltas};
use crate::residual_problem::ResidualProblem;
use crate::residual_solution::ResidualSolution;
use crate::solution::{
    ImageSet, SIMSConstructible, SIMSCore, SIMSModifiable, SIMSSolution, VecEncodedSolution,
};
use crate::timer::Timer;
use crate::util::IntersectionIterator;

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

impl<const D: usize> HasObjectives<D> for VecEncodedSolution<D> {
    fn objectives(&self) -> &pareto::Objectives<D> {
        &self.objectives
    }
}

impl<const D: usize> MoSolution<D> for VecEncodedSolution<D> {}

// Implement ImageSet trait
impl<const D: usize> ImageSet for VecEncodedSolution<D> {
    fn selected_images(&self) -> Vec<usize> {
        SelectedImagesIter::new(&self.selected_images).collect()
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
impl<const D: usize> SIMSCore<D> for VecEncodedSolution<D> {
    fn to_debug_solution(&self) -> SIMSSolution {
        self.as_sims_solution()
    }

    fn objectives_mut(&mut self) -> &mut pareto::Objectives<D> {
        &mut self.objectives
    }
}

// Implement Random trait from pareto
impl<const D: usize> Random for VecEncodedSolution<D> {
    fn random() -> Self {
        panic!("VecEncodedSolution::random() needs a Problem parameter")
    }

    fn random_with_seed(_seed: u64) -> Self {
        panic!("VecEncodedSolution::random_with_seed() needs a Problem parameter")
    }
}

// Implement SIMSConstructible trait
impl<const D: usize> SIMSConstructible<D> for VecEncodedSolution<D> {
    fn from_selected_images(selected_images_vec: &[usize], problem: &Problem<D>) -> Self {
        Self::from_selected_images(selected_images_vec, problem)
    }

    fn random_with_problem(problem: &Problem<D>) -> Self {
        Self::random(problem)
    }

    fn random_with_problem_and_seed(problem: &Problem<D>, seed: u64) -> Self {
        Self::random_with_seed(problem, seed)
    }
}

// Add utility methods for random generation that work with Problem
impl<const D: usize> VecEncodedSolution<D> {
    /// Generate a random solution with problem parameter
    #[must_use]
    pub fn random_with_problem(problem: &Problem<D>) -> Self {
        Self::random(problem)
    }

    /// Generate a random solution with seed and problem parameter
    #[must_use]
    pub fn random_with_problem_and_seed(problem: &Problem<D>, seed: u64) -> Self {
        Self::random_with_seed(problem, seed)
    }
}

// Implement SIMSModifiable trait
impl<const D: usize> SIMSModifiable<D> for VecEncodedSolution<D> {
    fn unselected_images(&self) -> Vec<usize> {
        UnselectedImagesIter::new(&self.selected_images).collect()
    }

    fn clear_parts_counts(&self) -> &[usize] {
        &self.clear_parts_counts
    }

    fn element_coverage(&self) -> &[usize] {
        &self.element_coverage
    }

    fn add_image(&mut self, image_index: usize, problem: &Problem<D>) {
        self.add_image(image_index, problem);
    }

    fn remove_image(&mut self, image_index: usize, problem: &Problem<D>) {
        self.remove_image(image_index, problem);
    }

    fn scaled_image_objective_deltas(
        &self,
        images: &[usize],
        problem: &Problem<D>,
    ) -> Vec<ScaledObjectiveDeltas<D>> {
        self.scaled_image_objective_deltas(images.iter().copied(), problem)
    }

    fn find_best_image_to_add(&self, problem: &Problem<D>) -> Option<usize> {
        let unselected_iter = UnselectedImagesIter::new(&self.selected_images);
        let unselected: Vec<usize> = unselected_iter.collect();
        if unselected.is_empty() {
            return None;
        }

        // Greedy add - best unselected image according to some heuristic
        let scaled_objective_deltas =
            self.scaled_image_objective_deltas(unselected.iter().copied(), problem);

        let min_index = (0..scaled_objective_deltas.len()).min_by(|&i, &j| {
            // Use first component of scaled objectives
            scaled_objective_deltas[i].scaled_deltas[0]
                .partial_cmp(&scaled_objective_deltas[j].scaled_deltas[0])
                .unwrap()
        })?;

        Some(scaled_objective_deltas[min_index].image_index)
    }

    fn find_best_image_to_remove(&self, problem: &Problem<D>) -> Option<usize> {
        let selected_iter = SelectedImagesIter::new(&self.selected_images);
        let selected: Vec<usize> = selected_iter.collect();
        if selected.is_empty() {
            return None;
        }

        // Greedy remove - worst selected image according to some heuristic
        let scaled_objective_deltas =
            self.scaled_image_objective_deltas(selected.iter().copied(), problem);

        let max_index = (0..scaled_objective_deltas.len()).max_by(|&i, &j| {
            // Use first component of scaled objectives
            scaled_objective_deltas[i].scaled_deltas[0]
                .partial_cmp(&scaled_objective_deltas[j].scaled_deltas[0])
                .unwrap()
        })?;

        Some(scaled_objective_deltas[max_index].image_index)
    }

    fn get_neighborhood(&self, problem: &Problem<D>) -> Vec<Self> {
        // Create a timer for the neighborhood method
        let timer = Timer::start(std::time::Duration::from_secs(60)); // 1 minute default

        // Use the existing neighborhood method with default parameters
        let k = 1; // Default value for local search
        let is_deterministic = true;

        self.neighborhood(k, problem, &timer, is_deterministic)
    }
}

impl<const D: usize> VecEncodedSolution<D> {
    /// Creates an `EncodedSolution` from a list of selected image indices.
    ///
    /// # Panics
    ///
    /// Panics if `D` is not equal to 2, as only `D = 2` is currently supported.
    #[must_use]
    pub fn from_selected_images(selected_images_vec: &[usize], problem: &Problem<D>) -> Self {
        let mut objectives = [0; D];
        assert!(D == 2, "EncodedSolution only supports D = 2 for now");
        objectives[1] = problem.total_area();
        let mut solution = Self {
            selected_images: vec![false; problem.images.len()],
            objectives,
            clear_parts_counts: vec![0; problem.universe.len()],
            element_coverage: vec![0; problem.universe.len()],
        };

        for &image_index in selected_images_vec {
            solution.add_image(image_index, problem);
        }
        solution
    }

    /// Generate a random feasible solution (choose element randomly, then choose image randomly from those that contain the element iff it is not already covered by another image)
    ///
    /// # Panics
    ///
    /// Panics if there is no image that covers an uncovered element (i.e., `.choose(&mut rng).unwrap()` fails).
    #[must_use]
    pub fn random_with_seed(problem: &Problem<D>, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut selected_images = vec![false; problem.images.len()];
        let mut covered_elements = vec![false; problem.universe.len()];
        let mut num_covered_elements = 0;
        let mut clear_parts_counts = vec![0; problem.universe.len()];
        let mut part_coverage_counts = vec![0; problem.universe.len()];

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
            selected_images[*image_index] = true;

            // Mark all elements of the image as covered
            problem.images[*image_index].parts.iter().for_each(|&part| {
                if !covered_elements[part] {
                    covered_elements[part] = true;
                    num_covered_elements += 1;
                }
                part_coverage_counts[part] += 1;
            });

            // Mark cloudy parts
            problem.images[*image_index]
                .clear_parts
                .iter()
                .for_each(|&clear_part| {
                    clear_parts_counts[clear_part] += 1;
                });
        }

        let mut sims_solution = Self {
            selected_images,
            objectives: [0; D],
            clear_parts_counts,
            element_coverage: part_coverage_counts,
        };
        sims_solution.compute_objectives(problem);
        sims_solution
    }

    /// Generate a random feasible solution
    #[must_use]
    pub fn random(problem: &Problem<D>) -> Self {
        Self::random_with_seed(problem, rand::random())
    }

    /// Compute the objectives of the solution
    fn compute_objectives(&mut self, problem: &Problem<D>) {
        // Use the new generic objective calculation system
        self.recalculate_objectives(problem);
    }

    /// Compute area covered by clouds
    #[must_use]
    pub fn cloudy_area(&self, problem: &Problem<D>) -> u64 {
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
    pub fn total_cost(&self, problem: &Problem<D>) -> u64 {
        self.selected_images()
            .map(|image_index| problem.images[image_index].cost())
            .sum()
    }

    /// Scalarizing function using weighted sum, for solution quality comparison
    #[must_use]
    pub fn scalarizing_fn(&self, weights: &[f32; D], _max_values: pareto::Objectives<D>) -> f32 {
        return objectives::weighted_sum(&self.objectives, weights);
    }

    /// Check if solution is valid
    #[must_use]
    pub fn is_valid(&self, problem: &Problem<D>) -> bool {
        let mut covered_elements = vec![false; problem.universe.len()];
        self.selected_images().for_each(|image_index| {
            problem.images[image_index].parts.iter().for_each(|&part| {
                covered_elements[part] = true;
            });
        });

        let all_elements_covered = covered_elements.iter().all(|&covered| covered);
        if !all_elements_covered {
            error!("Not all elements are covered");
            return false;
        }

        let clear_parts_counts_valid =
            self.clear_parts_counts
                .iter()
                .enumerate()
                .all(|(index, &count)| {
                    count
                        == self
                            .selected_images()
                            .filter(|&image_index| {
                                problem.images[image_index].clear_parts.contains(&index)
                            })
                            .count()
                });

        if !clear_parts_counts_valid {
            error!("Clear parts counts are invalid");
            return false;
        }

        let element_coverage_valid =
            self.element_coverage
                .iter()
                .enumerate()
                .all(|(index, &count)| {
                    count
                        == self
                            .selected_images()
                            .filter(|&image_index| {
                                problem.images[image_index].parts.contains(&index)
                            })
                            .count()
                });

        if !element_coverage_valid {
            error!("Element coverage is invalid");
            return false;
        }

        return self.are_objectives_valid(problem);
    }

    /// Check if objective values are correct
    #[must_use]
    pub fn are_objectives_valid(&self, problem: &Problem<D>) -> bool {
        let first_objective_is_valid = self.objectives[0] == self.total_cost(problem);
        if !first_objective_is_valid {
            trace!(
                "First objective is invalid. Expected {}, got {}",
                self.total_cost(problem),
                self.objectives[0]
            );
            return false;
        }
        let second_objective_is_valid = self.objectives[1] == self.cloudy_area(problem);
        if !second_objective_is_valid {
            trace!(
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
    pub const fn selected_images(&self) -> SelectedImagesIter<'_> {
        SelectedImagesIter::new(&self.selected_images)
    }

    /// Returns iterator over indices of unselected images of solution
    const fn unselected_images(&self) -> UnselectedImagesIter<'_> {
        UnselectedImagesIter::new(&self.selected_images)
    }

    /// Remove image at index i
    pub fn remove_image(&mut self, i: usize, problem: &Problem<D>) {
        debug_assert!(
            self.are_objectives_valid(problem),
            "Objectives are invalid before removing image"
        );

        // Use the new generic objective delta calculation
        let mut deltas = [0i64; D];
        for (obj_index, delta) in deltas.iter_mut().enumerate().take(D) {
            *delta = self.calculate_objective_delta(obj_index, i, problem);
        }
        objectives::apply_delta(&mut self.objectives, &deltas);

        problem.images[i].parts.iter().for_each(|&part| {
            self.element_coverage[part] -= 1;
        });
        problem.images[i]
            .clear_parts
            .iter()
            .for_each(|&clear_part| {
                debug_assert!(self.clear_parts_counts[clear_part] > 0);
                self.clear_parts_counts[clear_part] -= 1;
            });
        self.selected_images[i] = false;
        debug_assert!(
            self.are_objectives_valid(problem),
            "Objectives are invalid after removing image"
        );
    }

    /// Add image at index i
    pub fn add_image(&mut self, i: usize, problem: &Problem<D>) {
        // Use the new generic objective delta calculation
        let mut deltas = [0i64; D];
        for (obj_index, delta) in deltas.iter_mut().enumerate().take(D) {
            *delta = self.calculate_objective_delta(obj_index, i, problem);
        }
        objectives::apply_delta(&mut self.objectives, &deltas);

        problem.images[i].parts.iter().for_each(|&part| {
            self.element_coverage[part] += 1;
        });
        problem.images[i]
            .clear_parts
            .iter()
            .for_each(|&clear_part| {
                self.clear_parts_counts[clear_part] += 1;
            });
        self.selected_images[i] = true;
    }

    /// Check whether image at index i can be replaced by another image(s)
    #[must_use]
    pub fn is_replaceable(&self, i: usize, problem: &Problem<D>) -> bool {
        // For each part of the image, check if there is another image that covers the part
        self.unselected_images()
            .any(|image_index| problem.overlap_matrix[i][image_index] > 0)
    }

    /// Generate random weights for objectives (generic version)
    #[must_use]
    pub fn generate_weights(&self) -> [f32; D] {
        return objectives::generate_weights::<D>();
    }

    /// Create residual problem, composed of removed images, candidates to be added, and images covering the rest of the uncovered elements.
    #[must_use]
    pub fn create_residual_problem<'a>(
        &'a self,
        mut removal_candidates_indices: Vec<usize>,
        problem: &'a Problem<D>,
        is_deterministic: bool,
    ) -> Option<ResidualProblem<'a, Self, D>> {
        let mut unmodified_solution = self.clone();

        // Remove images
        for &removed_image_index in &removal_candidates_indices {
            unmodified_solution.remove_image(removed_image_index, problem);
        }

        // Get list of uncovered elements
        let uncovered_elements_indices = unmodified_solution
            .element_coverage
            .iter()
            .enumerate()
            .filter_map(|(element_index, &part_coverage_count)| {
                if part_coverage_count == 0 {
                    Some(element_index)
                } else {
                    None
                }
            })
            .collect::<Vec<usize>>();

        // Get clear parts counts for uncovered elements
        let original_clear_parts_counts = uncovered_elements_indices
            .iter()
            .map(|&element| unmodified_solution.clear_parts_counts[element])
            .collect();

        // Find best image(s) to replace removed image(s)
        let mut best_addition_candidates =
            self.best_unselected_images(&uncovered_elements_indices, problem, is_deterministic)?;

        removal_candidates_indices.sort_unstable();
        best_addition_candidates.sort_unstable();

        return Some(ResidualProblem::new(
            unmodified_solution,
            removal_candidates_indices,
            best_addition_candidates,
            uncovered_elements_indices,
            original_clear_parts_counts,
            problem,
        ));
    }

    // Get scaled objective deltas for list of given images
    fn scaled_image_objective_deltas<I: Iterator<Item = usize>>(
        &self,
        images: I,
        problem: &Problem<D>,
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
        problem: &Problem<D>,
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

        let unselected_images_scaled_deltas: Vec<ScaledObjectiveDeltas<D>> =
            self.scaled_image_objective_deltas(self.unselected_images(), problem);

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
    pub fn worst_selected_images(
        &self,
        problem: &Problem<D>,
        is_deterministic: bool,
    ) -> Vec<usize> {
        let weights: [f32; D] = if is_deterministic {
            // For deterministic mode, use equal weights
            let equal_weight = 1.0 / D as f32;
            [equal_weight; D]
        } else {
            self.generate_weights()
        };
        let selected_images_scaled_deltas: Vec<ScaledObjectiveDeltas<D>> =
            self.scaled_image_objective_deltas(self.selected_images(), problem);

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
        let selected_images = self.selected_images().collect::<Vec<usize>>();
        SIMSSolution { selected_images }
    }

    pub fn merge_residual_solution(
        &mut self,
        residual_solution: &ResidualSolution<D>,
        residual_problem: &ResidualProblem<Self, D>,
        problem: &Problem<D>,
    ) {
        residual_solution
            .selected_images
            .iter()
            .map(|&image_index| residual_problem.all_images[image_index].index)
            .for_each(|image_index| {
                self.add_image(image_index, problem);
            });
    }

    /// Explore neighborhood of size k
    #[must_use]
    pub fn neighborhood(
        &self,
        k: u32,
        problem: &Problem<D>,
        timer: &Timer,
        is_deterministic: bool,
    ) -> Vec<Self> {
        let removal_candidates_lists: Vec<Vec<usize>> = if k == 1 {
            self.selected_images()
                .filter_map(|selected_image| {
                    if self.is_replaceable(selected_image, problem) {
                        return Some(vec![selected_image]);
                    }
                    return None;
                })
                .collect()
        } else {
            self.worst_selected_images(problem, is_deterministic)
                .into_iter()
                .combinations(k as usize)
                .collect()
        };

        let mut residual_solutions: Vec<Self> = Vec::new();

        for removal_candidates in removal_candidates_lists {
            if let Some(mut residual_problem) =
                self.create_residual_problem(removal_candidates, problem, is_deterministic)
            {
                let neighborhood_iter = residual_problem.solve();
                residual_solutions.extend(neighborhood_iter);
            }
            if timer.is_expired() {
                break;
            }
        }

        return residual_solutions;
    }
}

/// Implement non-dominance relation for solutions (where a < b iff a dominates b)
/// Note: We minimize both objectives, so the smaller objective the better
#[expect(
    clippy::non_canonical_partial_ord_impl,
    reason = "Compare only first objective"
)]
impl<const D: usize> PartialOrd for VecEncodedSolution<D> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.objectives[0].partial_cmp(&other.objectives[0])
    }
}

impl<const D: usize> PartialEq for VecEncodedSolution<D> {
    fn eq(&self, other: &Self) -> bool {
        self.selected_images == other.selected_images
    }
}

/// Implement ordering for solutions (based on first objective)
impl<const D: usize> Ord for VecEncodedSolution<D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.objectives[0].cmp(&other.objectives[0])
    }
}

impl<const D: usize> Hash for VecEncodedSolution<D> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.selected_images.hash(state);
    }
}

// Implement Debug for SIMSEncodedSolution by converting it to SIMSSolution
#[expect(
    clippy::missing_fields_in_debug,
    reason = "Custom Debug impl only shows relevant fields for readability"
)]
impl<const D: usize> Debug for VecEncodedSolution<D> {
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

impl<const D: usize> crate::solution::ResidualSolutionCapable<D> for VecEncodedSolution<D> {
    fn merge_residual_solution(
        &mut self,
        residual_solution: &crate::residual_solution::ResidualSolution<D>,
        residual_problem: &crate::residual_problem::ResidualProblem<'_, Self, D>,
        problem: &crate::problem::Problem<D>,
    ) {
        self.merge_residual_solution(residual_solution, residual_problem, problem);
    }
}

// Implement SolutionEvaluator trait for VecEncodedSolution
impl<const D: usize> SolutionEvaluator<D> for VecEncodedSolution<D> {
    fn clear_parts_counts(&self) -> &[usize] {
        &self.clear_parts_counts
    }

    fn element_coverage(&self) -> &[usize] {
        &self.element_coverage
    }
}
