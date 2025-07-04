use pareto::{HasObjectives, MoSolution};
use std::fmt::Debug;
use std::hash::Hash;

#[cfg(feature = "bitmaps")]
use fixedbitset::FixedBitSet;

use crate::problem::{Problem, ScaledObjectiveDeltas};

// Re-export solution implementations
pub use crate::solution_impl::*;

/// Core trait for SIMS solutions that provides all necessary operations
/// This trait makes the codebase agnostic to the specific solution implementation
pub trait SIMSSolutionTrait<const D: usize>:
    HasObjectives<D>
    + MoSolution<D>
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
    /// Generate a random feasible solution
    fn random(problem: &Problem<D>) -> Self;

    /// Generate a random feasible solution with a specific seed
    fn random_with_seed(problem: &Problem<D>, seed: u64) -> Self;

    /// Create solution from selected image indices
    fn from_selected_images(selected_images: &[usize], problem: &Problem<D>) -> Self;

    /// Check if this solution is dominated by another
    fn is_dominated(&self, other: &Self) -> bool;

    /// Check if this solution is weakly dominated by another
    fn is_weakly_dominated(&self, other: &Self) -> bool;

    /// Get the objectives as a tuple (for compatibility)
    fn objectives_tuple(&self) -> pareto::Objectives<D>;

    /// Get vector of selected image indices
    fn selected_images(&self) -> Vec<usize>;

    /// Get vector of unselected image indices
    fn unselected_images(&self) -> Vec<usize>;

    /// Check if an image is selected
    fn is_image_selected(&self, image_index: usize) -> bool;

    /// Get the number of selected images
    fn num_selected_images(&self) -> usize;

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
    ) -> Vec<ScaledObjectiveDeltas>;

    /// Find the best image to add (for greedy algorithms)
    fn find_best_image_to_add(&self, problem: &Problem<D>) -> Option<usize>;

    /// Find the best image to remove (for local search)
    fn find_best_image_to_remove(&self, problem: &Problem<D>) -> Option<usize>;

    /// Get neighborhood solutions for local search
    fn get_neighborhood(&self, problem: &Problem<D>) -> Vec<Self>;

    /// Convert to debug representation (replaces `SIMSSolution` conversion)
    fn to_debug_solution(&self) -> SIMSSolution;
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
