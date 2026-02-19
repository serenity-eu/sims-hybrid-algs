//! Objective Trackers for Incremental Evaluation
//!
//! This module provides tracker implementations for maintaining objective-specific state
//! to enable efficient delta calculations during local search operations.

use std::fmt::Debug;

use crate::problem::SetCoverProblem;
use crate::solution::ImageSet;

/// Trait for a single objective tracker that maintains incremental state.
pub trait ObjectiveTracker<const D: usize>: Clone + Debug + Send + Sync {
    /// Calculate the change in objective value if we remove an image.
    /// Does NOT modify internal state.
    fn peek_removal_delta(
        &self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
        solution: &impl ImageSet<D>,
    ) -> i64;

    /// Calculate the change in objective value if we add an image.
    /// Does NOT modify internal state.
    fn peek_addition_delta(
        &self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
        solution: &impl ImageSet<D>,
    ) -> i64;

    /// Update internal state after removing an image.
    /// Returns the delta in objective value.
    fn track_image_removal(&mut self, image_index: usize, problem: &impl SetCoverProblem<D>)
    -> i64;

    /// Update internal state after adding an image.
    /// Returns the delta in objective value.
    fn track_image_addition(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> i64;

    /// Get the current absolute value of the objective (if tracked).
    fn value(&self) -> u64;
}

/// A collection of D trackers corresponding to D objectives.
pub trait TrackerCollection<const D: usize>: Clone + Debug + Send + Sync {
    type Tracker: ObjectiveTracker<D>;

    /// Get a reference to the tracker for a specific objective index.
    fn get(&self, index: usize) -> &Self::Tracker;

    /// Get a mutable reference to the tracker.
    fn get_mut(&mut self, index: usize) -> &mut Self::Tracker;

    /// Initialize the collection based on the problem definition.
    fn new(problem: &impl SetCoverProblem<D>) -> Self;

    /// Get the initial objective values from all trackers.
    /// Returns an array of D objective values corresponding to the empty solution state.
    ///
    /// # Panics
    /// Panics if any tracker fails to provide an initial value.
    #[must_use]
    fn initial_objectives(&self) -> [u64; D];

    /// Calculate the delta for removing an image across all objectives.
    /// Does NOT modify internal state.
    #[must_use]
    fn peek_removal_delta(
        &self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
        solution: &impl ImageSet<D>,
    ) -> [i64; D];

    /// Calculate the delta for adding an image across all objectives.
    /// Does NOT modify internal state.
    #[must_use]
    fn peek_addition_delta(
        &self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
        solution: &impl ImageSet<D>,
    ) -> [i64; D];

    /// Track the removal of an image across all objectives.
    /// Returns the delta array and updates internal state.
    fn track_image_removal(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> [i64; D];

    /// Track the addition of an image across all objectives.
    /// Returns the delta array and updates internal state.
    fn track_image_addition(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> [i64; D];

    /// Get the current values from all trackers.
    /// Returns an array of D values.
    #[must_use]
    fn values(&self) -> [u64; D];

    /// Reinitialize tracker state from a solution's selected images.
    /// Reuses existing allocations for efficiency.
    fn initialize_from(&mut self, solution: &impl ImageSet<D>, problem: &impl SetCoverProblem<D>);
}

// Re-export common tracker implementations so call sites can depend on a stable path.
pub use crate::objective_tracker_impl::standard_trackers::{StandardTracker, StandardTrackerArray};
pub use crate::objective_tracker_impl::simd_trackers::SimdTrackerArray;
pub use crate::objective_tracker_impl::proven_safe_trackers::ProvenSafeTrackerArray;

// Re-export additional tracker implementations when feature is enabled.
#[cfg(feature = "additional_trackers")]
pub use crate::objective_tracker_impl::alternative_trackers::AltTrackerArray;
#[cfg(feature = "additional_trackers")]
pub use crate::objective_tracker_impl::explicit_simd_trackers::ExplicitSimdTrackerArray;
#[cfg(feature = "additional_trackers")]
pub use crate::objective_tracker_impl::safe_simd_trackers::SafeTrackerArray;
#[cfg(feature = "additional_trackers")]
pub use crate::objective_tracker_impl::saturating_trackers::SaturatingTrackerArray;
#[cfg(feature = "additional_trackers")]
pub use crate::objective_tracker_impl::simplified_trackers::SimpleTrackerArray;


