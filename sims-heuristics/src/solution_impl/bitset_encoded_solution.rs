// Bitset-based solution implementation
// This module is only compiled when the "bitmaps" feature is enabled.

use fixedbitset::FixedBitSet;
use pareto::{HasObjectives, MoSolution, Random};
use rand::SeedableRng;
use rand::{Rng, seq::IteratorRandom};
use std::{collections::BinaryHeap, fmt::Debug, hash::Hash};

use crate::objectives::{self, SolutionEvaluator};
use crate::probabilistic_probing_neighborhood::{
    ObjectiveBasedSelector, ProbabilisticProbingNeighborhood, ProbingConfig,
};
use crate::problem::{ComparableImage, ImageObjectiveDeltas, Problem, ScaledObjectiveDeltas};
use crate::residual_problem::ResidualProblem;
use crate::residual_solution::ResidualSolution;
use crate::solution::{ImageSet, MergeableWithResidual, SIMSCore, SIMSModifiable, SIMSSolution};
use crate::solution_set_impl::NdTreeSolutionSet;
use crate::timer::Timer;
use crate::util::IntersectionIterator;
use itertools::Itertools;

#[cfg(feature = "bitmaps")]
#[derive(Clone, Eq)]
pub struct BitsetEncodedSolution<const D: usize> {
    pub selected_images: FixedBitSet,
    pub objectives: pareto::Objectives<D>,
    pub clear_parts_counts: Vec<usize>,
    pub element_coverage: Vec<usize>,
}

// Iterator types for BitsetEncodedSolution - leverage FixedBitSet's built-in iterators
pub type BitsetSelectedImagesIter<'a> = fixedbitset::Ones<'a>;
pub type BitsetUnselectedImagesIter<'a> = fixedbitset::Zeroes<'a>;

// Implement SolutionEvaluator trait for BitsetEncodedSolution
impl<const D: usize> SolutionEvaluator<D> for BitsetEncodedSolution<D> {
    fn clear_parts_counts(&self) -> &[usize] {
        &self.clear_parts_counts
    }

    fn element_coverage(&self) -> &[usize] {
        &self.element_coverage
    }
}

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

    fn clear_parts_counts(&self) -> &[usize] {
        &self.clear_parts_counts
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
    fn clear_parts_counts(&self) -> &[usize] {
        &self.clear_parts_counts
    }

    fn element_coverage(&self) -> &[usize] {
        &self.element_coverage
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

    fn get_neighborhood(&self, problem: &Problem<Self, D>) -> Vec<Self> {
        // Create a timer for the neighborhood method
        let timer = Timer::start(std::time::Duration::from_secs(60)); // 1 minute default

        // Use the probabilistic probing neighborhood method with default parameters
        let k = 1; // Default value for local search
        let is_deterministic = true;

        self.neighborhood(k, problem, &timer, is_deterministic)
    }

    fn neighborhood(
        &self,
        k: u32,
        problem: &Problem<Self, D>,
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
                let neighborhood_iter =
                    residual_problem.solve::<NdTreeSolutionSet<ResidualSolution<D>, D>>();
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
            log::error!("Clear parts counts are invalid");
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
            log::error!("Element coverage is invalid");
            return false;
        }

        return self.are_objectives_valid(problem);
    }
}

impl<const D: usize> BitsetEncodedSolution<D> {
    /// Creates a `BitsetEncodedSolution` from a list of selected image indices.
    #[must_use]
    pub fn from_selected_images(selected_images_vec: &[usize], problem: &Problem<Self, D>) -> Self {
        let mut solution = Self {
            selected_images: FixedBitSet::with_capacity(problem.images.len()),
            objectives: [0; D],
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
    /// Panics if there is no image covering an uncovered element (i.e., `.unwrap()` fails).
    #[must_use]
    pub fn random_with_seed(problem: &Problem<Self, D>, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut selected_images = FixedBitSet::with_capacity(problem.images.len());
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
            selected_images.set(*image_index, true);

            // Mark all elements of the image as covered
            problem.images[*image_index].parts.iter().for_each(|&part| {
                if !covered_elements[part] {
                    covered_elements[part] = true;
                    num_covered_elements += 1;
                }
                part_coverage_counts[part] += 1;
            });
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
        self.selected_images.set(i, false);
        debug_assert!(
            self.are_objectives_valid(problem),
            "Objectives are invalid after removing image"
        );
    }

    /// Add image at index i
    pub fn add_image(&mut self, i: usize, problem: &Problem<Self, D>) {
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
        self.selected_images.set(i, true);
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

    /// Create residual problem, composed of removed images, candidates to be added, and images covering the rest of the uncovered elements.
    #[must_use]
    pub fn create_residual_problem<'a>(
        &'a self,
        mut removal_candidates_indices: Vec<usize>,
        problem: &'a Problem<Self, D>,
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

        Some(ResidualProblem::new(
            unmodified_solution,
            removal_candidates_indices,
            best_addition_candidates,
            uncovered_elements_indices,
            original_clear_parts_counts,
            problem,
        ))
    }

    /// Get indices of the best replacement image(s) which is not selected yet, returns None when image cannot be replaced
    #[must_use]
    pub fn best_unselected_images(
        &self,
        uncovered_elements: &[usize],
        problem: &Problem<Self, D>,
        is_deterministic: bool,
    ) -> Option<Vec<usize>> {
        let unselected_images: Vec<usize> = self.unselected_images().collect();

        // If there is no unselected images, return
        if unselected_images.is_empty() {
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
            self.scaled_image_objective_deltas(&unselected_images, problem);

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
        return Some(best_unselected_images);
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
            .map(|&image_index| residual_problem.all_images[image_index].index)
            .for_each(|image_index| {
                self.add_image(image_index, problem);
            });
    }
}

// Implementation of ProbabilisticProbingNeighborhood trait for BitsetEncodedSolution
impl<const D: usize> ProbabilisticProbingNeighborhood<D> for BitsetEncodedSolution<D> {
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
                log::trace!("No images selected for removal in probe {}", probes_attempted);
                probes_attempted += 1;
                continue;
            }

            log::trace!("Probe {}: removing {} images: {:?}", probes_attempted, images_to_remove.len(), images_to_remove);

            // Apply probabilistic acceptance based on objective space improvements
            let should_explore = if is_deterministic {
                probes_attempted < max_probes / 2 // Explore first half deterministically
            } else {
                let random_value = rand::random::<f64>();
                random_value < probing_probability
            };

            if should_explore {
                log::trace!("Probe {}: exploring with {} images to remove", probes_attempted, images_to_remove.len());
                
                // Generate new solutions directly by modifying current solution
                let generated_solutions =
                    self.generate_neighbor_solutions(&images_to_remove, problem, &config, &mut rng);

                log::trace!("Probe {}: generated {} candidate solutions", probes_attempted, generated_solutions.len());

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
                
                log::trace!("Probe {}: {} valid, {} invalid solutions", probes_attempted, valid_count, invalid_count);
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
            log::warn!("Probabilistic probing failed to generate any valid solutions after {probes_attempted} probes");
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
        log::trace!("Generating solutions by removing {} images: {:?}", images_to_remove.len(), images_to_remove);
        
        let mut generated_solutions = Vec::new();

        // Create a base solution by removing specified images
        let mut base_solution = self.clone();
        for &image_idx in images_to_remove {
            base_solution.remove_image(image_idx, problem);
        }

        // Find uncovered elements after removal
        let uncovered_after_removal = self.find_uncovered_elements(&base_solution, problem);
        log::trace!("After removing {} images, {} elements are uncovered", images_to_remove.len(), uncovered_after_removal.len());

        // Generate multiple variants by ensuring complete coverage
        let max_variants = config.max_probes.min(10); // Reduced variants for debugging

        for variant_idx in 0..max_variants {
            let mut candidate = base_solution.clone();

            // Find uncovered elements for this variant
            let uncovered_elements = self.find_uncovered_elements(&candidate, problem);

            // If no elements are uncovered, the base solution is complete
            if uncovered_elements.is_empty() {
                log::trace!("Variant {}: base solution already complete", variant_idx);
                if candidate.is_valid(problem)
                    && candidate.selected_images().collect::<Vec<_>>()
                        != self.selected_images().collect::<Vec<_>>()
                {
                    generated_solutions.push(candidate);
                    log::trace!("Variant {}: added complete base solution", variant_idx);
                }
                continue;
            }

            log::trace!("Variant {}: {} elements need coverage", variant_idx, uncovered_elements.len());

            // ENSURE complete coverage by selecting images to cover ALL uncovered elements
            let covering_images =
                self.ensure_complete_coverage(&uncovered_elements, problem, config, rng);

            log::trace!("Variant {}: coverage algorithm selected {} images: {:?}", 
                       variant_idx, covering_images.len(), covering_images);

            // Add selected images to create complete solution
            for &image_idx in &covering_images {
                if !candidate.is_image_selected(image_idx) {
                    candidate.add_image(image_idx, problem);
                }
            }

            // Verify solution is complete and valid before adding
            let final_uncovered = self.find_uncovered_elements(&candidate, problem);
            log::trace!("Variant {}: after adding {} images, {} elements remain uncovered", 
                       variant_idx, covering_images.len(), final_uncovered.len());
            
            if final_uncovered.is_empty() && candidate.is_valid(problem)
                && candidate.selected_images().collect::<Vec<_>>()
                    != self.selected_images().collect::<Vec<_>>()
            {
                generated_solutions.push(candidate);
                log::trace!("Variant {}: successfully generated valid solution", variant_idx);
            } else {
                // Log detailed failure information
                log::trace!("Variant {}: FAILED - uncovered: {}, valid: {}, different: {}", 
                           variant_idx, 
                           final_uncovered.len(),
                           candidate.is_valid(problem),
                           candidate.selected_images().collect::<Vec<_>>() != self.selected_images().collect::<Vec<_>>());
                           
                if !final_uncovered.is_empty() {
                    log::trace!("Variant {}: remaining uncovered elements: {:?}", 
                               variant_idx, 
                               if final_uncovered.len() <= 10 { format!("{:?}", final_uncovered) } else { format!("{:?}...", &final_uncovered[..10]) });
                }
            }
        }

        log::trace!("Generated {} complete solutions from {} variants", generated_solutions.len(), max_variants);
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
        
        log::trace!("Starting coverage for {} uncovered elements: {:?}", uncovered_elements.len(), 
                   if uncovered_elements.len() <= 10 { format!("{:?}", uncovered_elements) } else { format!("{:?}...", &uncovered_elements[..10]) });
        
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
                                score += weights[i] * (-f64::from(delta_val) / config.temperature.max(1.0)).exp();
                            }
                        }
                    }

                    covering_candidates.push((img_idx, score, coverage_count, covered_elements));
                }
            }

            if covering_candidates.is_empty() {
                log::error!("No covering images found for {} remaining uncovered elements: {:?}", 
                           remaining_uncovered.len(), 
                           if remaining_uncovered.len() <= 20 { format!("{:?}", remaining_uncovered) } else { format!("{:?}...", &remaining_uncovered[..20]) });
                break; // This should not happen in valid problems
            }

            // Sort by score (higher is better) - coverage count dominates
            covering_candidates
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Use more conservative selection for reliability
            let selected_idx = if remaining_uncovered.len() > 10 || iteration_count > max_iterations / 2 {
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

            let (selected, _, coverage_count, covered_elements) = &covering_candidates[selected_idx];

            selected_images.push(*selected);

            // Remove covered elements from remaining uncovered
            let initial_uncovered_count = remaining_uncovered.len();
            remaining_uncovered.retain(|&elem| !covered_elements.contains(&elem));
            
            // Verify progress
            let progress = initial_uncovered_count - remaining_uncovered.len();
            if progress == 0 {
                log::warn!("No progress in iteration {}, image {} should cover {} elements but didn't", 
                          iteration_count, selected, coverage_count);
                break; // Prevent infinite loop
            }
            
            log::trace!("Iteration {}: selected image {} covers {} elements, {} elements remain", 
                       iteration_count, selected, progress, remaining_uncovered.len());
        }

        if !remaining_uncovered.is_empty() {
            log::error!("COVERAGE FAILURE: {} elements remain uncovered after {} iterations. Selected {} images: {:?}", 
                       remaining_uncovered.len(), iteration_count, selected_images.len(), selected_images);
            log::error!("Uncovered elements: {:?}", if remaining_uncovered.len() <= 20 { format!("{:?}", remaining_uncovered) } else { format!("{:?}...", &remaining_uncovered[..20]) });
        } else {
            log::trace!("Coverage successful: {} images selected after {} iterations", selected_images.len(), iteration_count);
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
            let max_obj = problem.max_objectives[i] as f64;

            // Normalized distance
            let normalized_diff = (obj1 - obj2).abs() / max_obj.max(1.0);
            distance += normalized_diff * normalized_diff;
        }

        distance.sqrt()
    }
}
