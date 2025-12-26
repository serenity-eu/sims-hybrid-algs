use pareto::{HasObjectives, MoSolution};
use std::fmt::Debug;
use std::hash::Hash;

use crate::{
    problem::{Problem, ScaledObjectiveDeltas},
    residual_problem::ResidualProblem,
    residual_solution::ResidualSolution,
    trackers::TrackerCollection,
};

// Re-export solution implementations
pub use crate::solution_impl::*;

/// Trait for image set operations - applicable to all solution types
pub trait ImageSet<const D: usize>: Sized {
    /// Get vector of selected image indices
    fn selected_images(&self) -> Vec<usize>;

    /// Get vector of unselected image indices
    fn unselected_images(&self) -> Vec<usize>;

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
    + ImageSet<D>
    + Clone
    + Eq
    + PartialEq
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
    fn calculate_objective(&self, obj_index: usize, problem: &Problem<Self, D>) -> u64 {
        problem.objective(obj_index).calculate_value(self, problem)
    }

    /// Calculate delta for a specific objective when adding/removing an image
    fn calculate_objective_delta(
        &self,
        obj_index: usize,
        image_index: usize,
        problem: &Problem<Self, D>,
    ) -> i64 {
        let is_selected = self.is_image_selected(image_index);
        problem
            .objective(obj_index)
            .calculate_delta(image_index, is_selected, self, problem)
    }

    /// Recalculate all objectives from scratch using the new generic system
    fn recalculate_objectives(&mut self, problem: &Problem<Self, D>) {
        for i in 0..D {
            self.objectives_mut()[i] = self.calculate_objective(i, problem);
        }
    }
}

/// Trait for solutions that support modification operations (not applicable to `ResidualSolution`)
pub trait SIMSModifiable<const D: usize>: SIMSCore<D> {
    /// The type of tracker collection this solution uses.
    type Trackers: TrackerCollection<D>;

    /// Access the trackers
    fn trackers(&self) -> &Self::Trackers;

    /// Mutable access to trackers
    fn trackers_mut(&mut self) -> &mut Self::Trackers;

    /// Add an image to the solution
    fn add_image(&mut self, image_index: usize, problem: &Problem<Self, D>);

    /// Remove an image from the solution
    fn remove_image(&mut self, image_index: usize, problem: &Problem<Self, D>);

    /// Get scaled objective deltas for given images
    fn scaled_image_objective_deltas(
        &self,
        images: &[usize],
        problem: &Problem<Self, D>,
    ) -> Vec<ScaledObjectiveDeltas<D>>;

    /// Find the best image to add (for greedy algorithms)
    fn find_best_image_to_add(&self, problem: &Problem<Self, D>) -> Option<usize>;

    /// Find the best image to remove (for local search)
    fn find_best_image_to_remove(&self, problem: &Problem<Self, D>) -> Option<usize>;

    /// Generate neighborhood solutions with specific parameters for Pareto Local Search
    fn neighborhood(
        &self,
        k: u32,
        problem: &Problem<Self, D>,
        timer: &crate::timer::Timer,
        is_deterministic: bool,
    ) -> Vec<Self>;

    /// Validate that the solution covers all elements
    fn is_valid(&self, problem: &Problem<Self, D>) -> bool;
}

/// Combined trait for full encoded solutions (`VecEncodedSolution` and `BitsetEncodedSolution`)
pub trait EncodedSolution<const D: usize>: SIMSCore<D> + SIMSModifiable<D> {
    /// Get the timestamp when this solution was created/found
    fn timestamp(&self) -> std::time::Duration;
}

/// Trait for solutions that can work with residual problems
pub trait MergeableWithResidual<const D: usize>: EncodedSolution<D> {
    /// Merge a residual solution into this solution
    fn merge_residual_solution(
        &mut self,
        residual_solution: &ResidualSolution<D>,
        residual_problem: &ResidualProblem<'_, Self, D>,
        problem: &Problem<Self, D>,
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
