use itertools::Itertools;
use itertools::MinMaxResult::{MinMax, NoElements, OneElement};
use log::{error, trace};
use pareto::{HasObjectives, MoSolution};
use rand::SeedableRng;
use rand::{seq::IteratorRandom, Rng};
use std::{collections::BinaryHeap, fmt::Debug, hash::Hash, vec};

use crate::objectives;
use crate::problem::{ComparableImage, ImageObjectiveDeltas, Problem, ScaledObjectiveDeltas};
use crate::residual_problem::ResidualProblem;
use crate::residual_solution::ResidualSolution;
use crate::timer::Timer;
use crate::util::IntersectionIterator;

/// Trait that combines pareto functionality with SIMS-specific solution methods
pub trait SIMSSolutionTrait<const D: usize>: HasObjectives<D> + MoSolution<D> {
    /// Generate a random feasible solution
    fn random(problem: &Problem<D>) -> Self;

    /// Generate a random feasible solution with a specific seed
    fn random_with_seed(problem: &Problem<D>, seed: u64) -> Self;

    /// Check if this solution is dominated by another
    fn is_dominated(&self, other: &Self) -> bool;

    /// Check if this solution is weakly dominated by another
    fn is_weakly_dominated(&self, other: &Self) -> bool;

    /// Get the objectives as a tuple (for compatibility)
    fn objectives_tuple(&self) -> pareto::Objectives<D>;
}

pub struct SIMSSolution {
    selected_images: Vec<usize>,
}

impl Debug for SIMSSolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SIMSSolution")
            .field("num_images", &self.selected_images.len())
            .field("selected_images", &self.selected_images)
            // .field("images_per_element", &self.images_per_element)
            .finish()
    }
}

#[derive(Clone, Eq)]
pub struct EncodedSolution<const D: usize> {
    selected_images: Vec<bool>,
    pub objectives: pareto::Objectives<D>,
    pub clear_parts_counts: Vec<usize>,
    pub element_coverage: Vec<usize>,
}

pub struct SelectedImagesIter<'a> {
    images: &'a Vec<bool>,
    index: usize,
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

impl<const D: usize> HasObjectives<D> for EncodedSolution<D> {
    fn objectives(&self) -> &pareto::Objectives<D> {
        &self.objectives
    }
}

impl<const D: usize> MoSolution<D> for EncodedSolution<D> {}

impl<const D: usize> SIMSSolutionTrait<D> for EncodedSolution<D> {
    /// Generate a random feasible solution (choose element randomly, then choose image randomly from those that contain the element iff it is not already covered by another image)
    fn random_with_seed(problem: &Problem<D>, seed: u64) -> Self {
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

    fn random(problem: &Problem<D>) -> Self {
        return Self::random_with_seed(problem, rand::rng().random());
    }

    fn is_dominated(&self, other: &Self) -> bool {
        // Solution is dominated by other solution iff it is greater or equal in all objectives, with at least one objective being strictly greater
        let dominance_relation = self.objectives.partial_cmp(&other.objectives);
        return dominance_relation == Some(std::cmp::Ordering::Greater);
    }

    fn is_weakly_dominated(&self, other: &Self) -> bool {
        // Solution is weakly dominated by other solution iff it is greater or equal in all objectives
        let dominance_relation = self.objectives.partial_cmp(&other.objectives);
        return (dominance_relation == Some(std::cmp::Ordering::Greater))
            || (dominance_relation == Some(std::cmp::Ordering::Equal));
    }

    fn objectives_tuple(&self) -> pareto::Objectives<D> {
        self.objectives
    }
}

impl<const D: usize> EncodedSolution<D> {
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

    /// Compute the objectives of the solution
    fn compute_objectives(&mut self, problem: &Problem<D>) {
        let mut objectives = [0; D];
        if D >= 1 {
            objectives[0] = self.total_cost(problem);
        }
        if D >= 2 {
            objectives[1] = self.cloudy_area(problem);
        }
        self.objectives = objectives;
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
        SelectedImagesIter {
            images: &self.selected_images,
            index: 0,
        }
    }

    /// Returns iterator over indices of unselected images of solution
    const fn unselected_images(&self) -> UnselectedImagesIter<'_> {
        UnselectedImagesIter {
            images: &self.selected_images,
            index: 0,
        }
    }

    /// Remove image at index i
    pub fn remove_image(&mut self, i: usize, problem: &Problem<D>) {
        debug_assert!(
            self.are_objectives_valid(problem),
            "Objectives are invalid before removing image"
        );
        let cost_delta = self.cost_delta(i, problem);
        let cloudy_area_delta = self.cloudy_area_delta(i, problem);
        let mut deltas = [0; D];
        if D >= 1 {
            deltas[0] = cost_delta;
        }
        if D >= 2 {
            deltas[1] = cloudy_area_delta;
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
        let cost_delta = self.cost_delta(i, problem);
        let cloudy_area_delta = self.cloudy_area_delta(i, problem);
        objectives::apply_delta_2d(&mut self.objectives, (cost_delta, cloudy_area_delta));
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

    /// Generate random weights for objectives
    #[must_use]
    pub fn generate_weights(&self) -> (f32, f32) {
        return objectives::generate_weights_2d();
    }

    /// Create residual problem, composed of removed images, candidates to be added, and images covering the rest of the uncovered elements.
    #[must_use]
    pub fn create_residual_problem<'a>(
        &'a self,
        mut removal_candidates_indices: Vec<usize>,
        problem: &'a Problem<D>,
        is_deterministic: bool,
    ) -> Option<ResidualProblem<'a, D>> {
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
    ) -> Vec<ScaledObjectiveDeltas> {
        let raw_comparable_images: Vec<ImageObjectiveDeltas> = images
            .map(|image_index| ImageObjectiveDeltas {
                image_index,
                deltas: (
                    self.cost_delta(image_index, problem),
                    self.cloudy_area_delta(image_index, problem),
                ),
            })
            .collect();

        let cost_min_max = raw_comparable_images
            .iter()
            .minmax_by_key(|&image| image.deltas.0);
        let (min_cost, max_cost) = match cost_min_max {
            MinMax(min_cost_image, max_cost_image) => {
                (min_cost_image.deltas.0, max_cost_image.deltas.0)
            }
            OneElement(min_max_cost_image) => {
                (min_max_cost_image.deltas.0, min_max_cost_image.deltas.0)
            }
            NoElements => panic!("No elements in cost_min_max"),
        };

        let cloudy_area_min_max = raw_comparable_images
            .iter()
            .minmax_by_key(|&image| image.deltas.1);
        let (min_cloudy_area, max_cloudy_area) = match cloudy_area_min_max {
            MinMax(min_cloudy_area_image, max_cloudy_area_image) => (
                min_cloudy_area_image.deltas.1,
                max_cloudy_area_image.deltas.1,
            ),
            OneElement(min_max_cloudy_area_image) => (
                min_max_cloudy_area_image.deltas.1,
                min_max_cloudy_area_image.deltas.1,
            ),
            NoElements => panic!("No elements in cloudy_area_min_max"),
        };
        let cost_range = max_cost - min_cost;
        let cloudy_area_range = max_cloudy_area - min_cloudy_area;

        
        raw_comparable_images
            .iter()
            .map(|objective_deltas| {
                let scaled_obj0 = (objective_deltas.deltas.0 - min_cost) as f32 / cost_range as f32;
                let scaled_obj1 =
                    (objective_deltas.deltas.1 - min_cloudy_area) as f32 / cloudy_area_range as f32;
                ScaledObjectiveDeltas {
                    image_index: objective_deltas.image_index,
                    scaled_objectives: (scaled_obj0, scaled_obj1),
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
        if self.unselected_images().images.is_empty() {
            return None;
        }

        let weights: (f32, f32) = if is_deterministic {
            (0.5, 0.5)
        } else {
            self.generate_weights()
        };

        let unselected_images_scaled_deltas: Vec<ScaledObjectiveDeltas> =
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

                let comparision_heur_key = scaled_objective_deltas.scaled_objectives.0 * weights.0
                    + scaled_objective_deltas.scaled_objectives.1 * weights.1;
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
        let weights: (f32, f32) = if is_deterministic {
            (0.5, 0.5)
        } else {
            self.generate_weights()
        };
        let selected_images_scaled_deltas: Vec<ScaledObjectiveDeltas> =
            self.scaled_image_objective_deltas(self.selected_images(), problem);

        let comparable_selected_images: BinaryHeap<ComparableImage> = selected_images_scaled_deltas
            .into_iter()
            .map(|scaled_image_deltas| {
                let image = &problem.images[scaled_image_deltas.image_index];
                let covered_elements_count = image.parts.len();

                let comparision_heur_key = (scaled_image_deltas.scaled_objectives.0 * weights.0
                    + scaled_image_deltas.scaled_objectives.1 * weights.1)
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
        let selected_images = self
            .selected_images
            .iter()
            .enumerate()
            .filter_map(|(index, &selected)| if selected { Some(index) } else { None })
            .collect::<Vec<usize>>();
        SIMSSolution { selected_images }
    }

    pub fn merge_residual_solution(
        &mut self,
        residual_solution: &ResidualSolution<D>,
        residual_problem: &ResidualProblem<D>,
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

    fn cloudy_area_delta(&self, image_index: usize, problem: &Problem<D>) -> i64 {
        let mut cloudy_area_delta: i64 = 0;
        if self.selected_images[image_index] {
            problem.images[image_index]
                .clear_parts
                .iter()
                .for_each(|&clear_part| {
                    // If this is the last image with clear part covering the element, add element area to delta
                    if self.clear_parts_counts[clear_part] == 1 {
                        cloudy_area_delta += problem.universe[clear_part].area as i64;
                    }
                });
        } else {
            problem.images[image_index]
                .clear_parts
                .iter()
                .for_each(|&clear_part| {
                    // If this is the first image with clear part covering the element, subtract element area from delta
                    if self.clear_parts_counts[clear_part] == 0 {
                        cloudy_area_delta -= problem.universe[clear_part].area as i64;
                    }
                });
        }
        return cloudy_area_delta;
    }

    fn cost_delta(&self, image_index: usize, problem: &Problem<D>) -> i64 {
        if self.selected_images[image_index] {
            return -(problem.images[image_index].cost() as i64);
        }
        return problem.images[image_index].cost() as i64;
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
impl<const D: usize> PartialOrd for EncodedSolution<D> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.objectives[0].partial_cmp(&other.objectives[0])
    }
}

impl<const D: usize> PartialEq for EncodedSolution<D> {
    fn eq(&self, other: &Self) -> bool {
        self.selected_images == other.selected_images
    }
}

/// Implement ordering for solutions (based on first objective)
impl<const D: usize> Ord for EncodedSolution<D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.objectives[0].cmp(&other.objectives[0])
    }
}

impl<const D: usize> Hash for EncodedSolution<D> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.selected_images.hash(state);
    }
}

// Implement Debug for SIMSEncodedSolution by converting it to SIMSSolution
#[expect(
    clippy::missing_fields_in_debug,
    reason = "Custom Debug impl only shows relevant fields for readability"
)]
impl<const D: usize> Debug for EncodedSolution<D> {
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
