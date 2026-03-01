use std::sync::Arc;

use arc_swap::ArcSwap;
use pareto::{HasObjectives, MoSolution, Objectives, ParetoFront};

use crate::solution::ImageSet;
use crate::solution_set_impl::NdTreeSolutionSet;

/// Compact snapshot of a single solution, used for cross-region sharing.
/// Includes the full solution representation so adopting regions can insert it.
#[derive(Clone)]
pub struct ObjectiveSnapshot<T: Clone, const D: usize> {
    /// Cached objective values (u64 per objective, same as `HasObjectives<D>`).
    pub objectives: [u64; D],
    /// Fingerprint hash of the solution representation (for dedup / tombstoning).
    pub fingerprint: u64,
    /// The full solution (cheap clone for `BitsetEncodedSolution`).
    pub solution: T,
}

// ---------------------------------------------------------------------------
// Trait impls enabling ObjectiveSnapshot to be stored in NdTreeSolutionSet
// ---------------------------------------------------------------------------

impl<T: Clone, const D: usize> HasObjectives<D> for ObjectiveSnapshot<T, D> {
    fn objectives(&self) -> &Objectives<D> {
        &self.objectives
    }
}

impl<T: Clone, const D: usize> MoSolution<D> for ObjectiveSnapshot<T, D> {}

impl<T: Clone + ImageSet<D>, const D: usize> ImageSet<D> for ObjectiveSnapshot<T, D> {
    fn selected_images(&self) -> impl Iterator<Item = usize> {
        self.solution.selected_images()
    }
    fn unselected_images(&self) -> impl Iterator<Item = usize> {
        self.solution.unselected_images()
    }
    fn is_image_selected(&self, image_index: usize) -> bool {
        self.solution.is_image_selected(image_index)
    }
    fn num_selected_images(&self) -> usize {
        self.solution.num_selected_images()
    }
    fn set_image(&mut self, image_index: usize, selected: bool) {
        self.solution.set_image(image_index, selected)
    }
}

impl<T: Clone, const D: usize> PartialEq for ObjectiveSnapshot<T, D> {
    fn eq(&self, other: &Self) -> bool {
        self.fingerprint == other.fingerprint && self.objectives == other.objectives
    }
}

impl<T: Clone, const D: usize> Eq for ObjectiveSnapshot<T, D> {}

impl<T: Clone, const D: usize> std::fmt::Debug for ObjectiveSnapshot<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectiveSnapshot")
            .field("objectives", &self.objectives)
            .field("fingerprint", &self.fingerprint)
            .finish()
    }
}

impl<T: Clone, const D: usize> std::hash::Hash for ObjectiveSnapshot<T, D> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.fingerprint.hash(state);
        self.objectives.hash(state);
    }
}

/// Published snapshot of the global (merged) Pareto front.
/// Workers read this to tombstone locally dominated solutions and adopt new ones.
/// Uses `NdTreeSolutionSet` for O(n log n) merge and efficient non-dominated storage.
pub struct GlobalFrontSnapshot<T, const D: usize>
where
    T: Clone + ImageSet<D>,
{
    /// Non-dominated global front stored in an ND-Tree.
    pub front: NdTreeSolutionSet<ObjectiveSnapshot<T, D>, D>,
    /// Best (minimum) known value per objective across all regions.
    pub ideal_point: Objectives<D>,
    /// Worst (maximum) known value per objective across the current front.
    /// Together with `ideal_point`, defines the normalization range for
    /// Tchebycheff scoring. Updated by the merger every merge cycle.
    pub nadir_point: Objectives<D>,
}

impl<T, const D: usize> Default for GlobalFrontSnapshot<T, D>
where
    T: Clone + ImageSet<D>,
{
    fn default() -> Self {
        Self {
            // Build from empty iterator to avoid requiring Debug bound on T
            front: std::iter::empty().collect(),
            ideal_point: [u64::MAX; D],
            nadir_point: [0; D],
        }
    }
}

/// Worker-owned handle for publishing a snapshot.
pub type SnapshotSlot<T, const D: usize> = Arc<ArcSwap<Vec<ObjectiveSnapshot<T, D>>>>;
/// Shared slot through which the merger publishes the global front.
pub type GlobalFrontSlot<T, const D: usize> = Arc<ArcSwap<GlobalFrontSnapshot<T, D>>>;

/// Compute fingerprint hash of a solution using `DefaultHasher`.
#[inline]
pub fn fingerprint<T: std::hash::Hash>(solution: &T) -> u64 {
    use std::hash::Hasher;
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    solution.hash(&mut h);
    h.finish()
}

/// Check if `candidate_objectives` is dominated by ANY solution in the global front ND-Tree.
///
/// Uses `MoSolution::dominates` from the `pareto` crate, which compares objectives
/// component-wise with short-circuit evaluation. `.any()` returns on the first dominator.
#[inline]
pub fn is_dominated_by_front<T, const D: usize>(
    candidate_objectives: &[u64; D],
    front: &NdTreeSolutionSet<ObjectiveSnapshot<T, D>, D>,
) -> bool
where
    T: Clone + ImageSet<D>,
{
    front.iter().any(|g| g.dominates(candidate_objectives))
}
