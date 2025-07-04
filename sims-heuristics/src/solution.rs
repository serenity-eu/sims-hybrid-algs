use pareto::{HasObjectives, MoSolution, Random};
use std::fmt::Debug;
use std::hash::Hash;

#[cfg(feature = "bitmaps")]
use fixedbitset::FixedBitSet;

use crate::{
    objectives::SolutionEvaluator,
    problem::{Problem, ScaledObjectiveDeltas},
};

// Re-export solution implementations
pub use crate::solution_impl::*;

/// Trait for image set operations - applicable to all solution types
pub trait ImageSet {
    /// Get vector of selected image indices
    fn selected_images(&self) -> Vec<usize>;

    /// Check if an image is selected
    fn is_image_selected(&self, image_index: usize) -> bool;

    /// Get the number of selected images
    fn num_selected_images(&self) -> usize;

    /// Set an image's selection state
    fn set_image(&mut self, image_index: usize, selected: bool);
}

/// Core trait for basic SIMS solution operations that all solution types must support
pub trait SIMSCore<const D: usize>:
    HasObjectives<D>
    + MoSolution<D>
    + ImageSet
    + SolutionEvaluator<D>
    + Clone
    + Eq
    + PartialEq
    + PartialOrd
    + Ord
    + Hash
    + Debug
    + Send
    + Sync
{
    /// Convert to debug representation
    fn to_debug_solution(&self) -> SIMSSolution;

    /// Get mutable reference to objectives (for internal use)
    fn objectives_mut(&mut self) -> &mut pareto::Objectives<D>;

    /// Calculate value for a specific objective using the new generic system
    fn calculate_objective(&self, obj_index: usize, problem: &Problem<D>) -> u64 {
        problem.get_objective_definition(obj_index).map_or_else(
            || match obj_index {
                0 => self
                    .selected_images()
                    .iter()
                    .map(|&i| problem.images[i].cost())
                    .sum(),
                1 => {
                    let mut clear_elements = vec![false; problem.universe.len()];
                    for &image_index in &self.selected_images() {
                        for &clear_part in &problem.images[image_index].clear_parts {
                            clear_elements[clear_part] = true;
                        }
                    }
                    clear_elements
                        .iter()
                        .enumerate()
                        .filter_map(|(element_index, &is_clear)| {
                            if is_clear {
                                None
                            } else {
                                Some(problem.universe[element_index].area)
                            }
                        })
                        .sum()
                }
                _ => panic!("Legacy system only supports objectives 0 and 1"),
            },
            |obj_def| obj_def.calculate_value(self, problem),
        )
    }

    /// Calculate delta for a specific objective when adding/removing an image
    fn calculate_objective_delta(
        &self,
        obj_index: usize,
        image_index: usize,
        problem: &Problem<D>,
    ) -> i64 {
        problem.get_objective_definition(obj_index).map_or_else(
            || {
                // Fallback to legacy calculation for backward compatibility
                let is_selected = self.is_image_selected(image_index);
                match obj_index {
                    0 => {
                        // Cost delta
                        if is_selected {
                            -(problem.images[image_index].cost() as i64)
                        } else {
                            problem.images[image_index].cost() as i64
                        }
                    }
                    1 => {
                        // Cloudy area delta
                        let mut cloudy_area_delta: i64 = 0;
                        if is_selected {
                            for &clear_part in &problem.images[image_index].clear_parts {
                                if self.clear_parts_counts()[clear_part] == 1 {
                                    cloudy_area_delta += problem.universe[clear_part].area as i64;
                                }
                            }
                        } else {
                            for &clear_part in &problem.images[image_index].clear_parts {
                                if self.clear_parts_counts()[clear_part] == 0 {
                                    cloudy_area_delta -= problem.universe[clear_part].area as i64;
                                }
                            }
                        }
                        cloudy_area_delta
                    }
                    _ => panic!("Legacy system only supports objectives 0 and 1"),
                }
            },
            |obj_def| {
                let is_selected = self.is_image_selected(image_index);
                obj_def.calculate_delta(image_index, is_selected, self, problem)
            },
        )
    }

    /// Recalculate all objectives from scratch using the new generic system
    fn recalculate_objectives(&mut self, problem: &Problem<D>) {
        for i in 0..D {
            self.objectives_mut()[i] = self.calculate_objective(i, problem);
        }
    }
}

/// Trait for constructible solutions (not applicable to `ResidualSolution`)
pub trait SIMSConstructible<const D: usize>: SIMSCore<D> + Random {
    /// Create solution from selected image indices
    fn from_selected_images(selected_images: &[usize], problem: &Problem<D>) -> Self;

    /// Generate a random feasible solution
    fn random_with_problem(problem: &Problem<D>) -> Self;

    /// Generate a random feasible solution with a specific seed
    fn random_with_problem_and_seed(problem: &Problem<D>, seed: u64) -> Self;
}

/// Trait for solutions that support modification operations (not applicable to `ResidualSolution`)
pub trait SIMSModifiable<const D: usize>: SIMSCore<D> {
    /// Get vector of unselected image indices
    fn unselected_images(&self) -> Vec<usize>;

    /// Get the clear parts counts
    fn clear_parts_counts(&self) -> &[usize];

    /// Get the element coverage
    fn element_coverage(&self) -> &[usize];

    /// Add an image to the solution
    fn add_image(&mut self, image_index: usize, problem: &Problem<D>);

    /// Remove an image from the solution
    fn remove_image(&mut self, image_index: usize, problem: &Problem<D>);

    /// Get scaled objective deltas for given images
    fn scaled_image_objective_deltas(
        &self,
        images: &[usize],
        problem: &Problem<D>,
    ) -> Vec<ScaledObjectiveDeltas<D>>;

    /// Find the best image to add (for greedy algorithms)
    fn find_best_image_to_add(&self, problem: &Problem<D>) -> Option<usize>;

    /// Find the best image to remove (for local search)
    fn find_best_image_to_remove(&self, problem: &Problem<D>) -> Option<usize>;

    /// Get neighborhood solutions for local search
    fn get_neighborhood(&self, problem: &Problem<D>) -> Vec<Self>;
}

/// Combined trait for full encoded solutions (`VecEncodedSolution` and `BitsetEncodedSolution`)
pub trait SIMSSolutionTrait<const D: usize>:
    SIMSCore<D> + SIMSConstructible<D> + SIMSModifiable<D>
{
}

/// Automatic implementation for types that implement all required traits
impl<T, const D: usize> SIMSSolutionTrait<D> for T where
    T: SIMSCore<D> + SIMSConstructible<D> + SIMSModifiable<D>
{
}

/// Trait for solutions that can work with residual problems
pub trait ResidualSolutionCapable<const D: usize>: SIMSSolutionTrait<D> {
    /// Merge a residual solution into this solution
    fn merge_residual_solution(
        &mut self,
        residual_solution: &crate::residual_solution::ResidualSolution<D>,
        residual_problem: &crate::residual_problem::ResidualProblem<'_, Self, D>,
        problem: &Problem<D>,
    );
}

pub struct SIMSSolution {
    pub selected_images: Vec<usize>,
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
pub struct VecEncodedSolution<const D: usize> {
    pub selected_images: Vec<bool>,
    pub objectives: pareto::Objectives<D>,
    pub clear_parts_counts: Vec<usize>,
    pub element_coverage: Vec<usize>,
}

#[cfg(feature = "bitmaps")]
#[derive(Clone, Eq)]
pub struct BitsetEncodedSolution<const D: usize> {
    pub selected_images: FixedBitSet,
    pub objectives: pareto::Objectives<D>,
    pub clear_parts_counts: Vec<usize>,
    pub element_coverage: Vec<usize>,
}

/// Type alias for the default encoded solution based on the selected feature
#[cfg(feature = "bitmaps")]
pub type EncodedSolution<const D: usize> = BitsetEncodedSolution<D>;

#[cfg(not(feature = "bitmaps"))]
pub type EncodedSolution<const D: usize> = VecEncodedSolution<D>;
