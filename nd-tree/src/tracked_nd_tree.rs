//! Tracked ND-Tree: ND-Tree with solution index tracking and domination reporting.
//!
//! This module provides `TrackedNdTree`, which extends the functionality of `NDTree`
//! by tracking solution indices and reporting domination relationships on insertion.
//!
//! # Features
//!
//! - **Index Tracking**: Each solution gets a unique sequential index
//! - **Domination Reporting**: Returns list of dominated solution indices on insertion
//! - **Configurable Filtering**: Choose whether to include dominated-at-discovery solutions
//!
//! # Filtering Modes
//!
//! ## No-Filtering Mode (`filter_dominated = false`)
//! - All solutions receive an index, even if dominated at discovery
//! - Dominated solutions are tracked but may be removed from the tree
//! - Useful for complete trace generation
//!
//! ## Filtering Mode (`filter_dominated = true`)
//! - Only non-dominated-at-discovery solutions receive an index
//! - Dominated solutions are rejected and don't get an index
//! - Useful for maintaining a pure Pareto front
//!
//! # Example
//!
//! ```ignore
//! use nd_tree::tracked_nd_tree::{TrackedNdTree, InsertionResult};
//! use nd_tree::nd_tree::Solution;
//!
//! let mut tree: TrackedNdTree<Solution<2>, 8, 2, 4> =
//!     TrackedNdTree::new(false); // no-filtering mode
//!
//! let s1 = Solution::new([10, 20]);
//! let result = tree.insert(s1);
//! assert!(result.inserted);
//! assert_eq!(result.assigned_index, Some(0));
//! assert_eq!(result.dominated_indices.len(), 0);
//!
//! let s2 = Solution::new([5, 15]); // Dominates s1
//! let result = tree.insert(s2);
//! assert!(result.inserted);
//! assert_eq!(result.assigned_index, Some(1));
//! assert_eq!(result.dominated_indices, vec![0]); // Dominated s1
//! ```

use crate::nd_tree::{NDTree, Solution};
use pareto::{HasObjectives, MoSolution};
use std::collections::HashMap;

/// Result of a solution insertion into `TrackedNdTree`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsertionResult {
    /// Whether the solution was inserted into the tree.
    /// - In filtering mode: `true` only if non-dominated at discovery
    /// - In no-filtering mode: always `true` (but may be marked as dominated)
    pub inserted: bool,

    /// The index assigned to this solution.
    /// - `Some(index)` if the solution was accepted
    /// - `None` if rejected (only in filtering mode when dominated at discovery)
    pub assigned_index: Option<u32>,

    /// Indices of solutions that this solution dominated.
    /// These solutions may have been removed from the tree or marked as dominated.
    pub dominated_indices: Vec<u32>,

    /// Whether the inserted solution was immediately dominated by existing solutions.
    /// Only relevant in no-filtering mode where dominated solutions still get indices.
    pub was_dominated_at_discovery: bool,
}

impl InsertionResult {
    /// Create a result for a rejected solution (filtering mode only).
    fn rejected() -> Self {
        Self {
            inserted: false,
            assigned_index: None,
            dominated_indices: Vec::new(),
            was_dominated_at_discovery: true,
        }
    }

    /// Create a result for an accepted solution.
    fn accepted(index: u32, dominated: Vec<u32>, was_dominated: bool) -> Self {
        Self {
            inserted: true,
            assigned_index: Some(index),
            dominated_indices: dominated,
            was_dominated_at_discovery: was_dominated,
        }
    }
}

/// Configuration for `TrackedNdTree`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackedNdTreeConfig {
    /// If `true`, only non-dominated-at-discovery solutions are assigned indices.
    /// If `false`, all solutions get indices (no-filtering mode).
    pub filter_dominated: bool,
}

impl Default for TrackedNdTreeConfig {
    fn default() -> Self {
        Self {
            filter_dominated: true,
        }
    }
}

/// ND-Tree with solution index tracking and domination reporting.
///
/// Generic parameters:
/// - `T`: Solution type (must implement `MoSolution` and `HasObjectives`)
/// - `N`: Maximum solutions per leaf node
/// - `D`: Number of objectives (dimensions)
/// - `C`: Maximum children per internal node
pub struct TrackedNdTree<T, const N: usize, const D: usize, const C: usize>
where
    T: MoSolution<D> + HasObjectives<D> + Clone + PartialEq,
{
    /// The underlying ND-tree structure.
    tree: NDTree<T, N, D, C>,

    /// Configuration.
    config: TrackedNdTreeConfig,

    /// Next index to assign to a solution.
    next_index: u32,

    /// Map from solution index to solution data.
    /// In filtering mode: only contains non-dominated solutions
    /// In no-filtering mode: contains all solutions (even dominated ones)
    index_to_solution: HashMap<u32, T>,

    /// Map from solution (via objectives hash) to its index.
    /// Used to quickly find the index of a solution in the tree.
    solution_to_index: HashMap<Vec<u64>, u32>,
}

impl<T, const N: usize, const D: usize, const C: usize> TrackedNdTree<T, N, D, C>
where
    T: MoSolution<D> + HasObjectives<D> + Clone + PartialEq,
{
    /// Create a new `TrackedNdTree` with the given configuration.
    pub fn new(config: TrackedNdTreeConfig) -> Self {
        Self {
            tree: NDTree::new(),
            config,
            next_index: 0,
            index_to_solution: HashMap::new(),
            solution_to_index: HashMap::new(),
        }
    }

    /// Create a new `TrackedNdTree` with filtering enabled.
    pub fn new_with_filtering() -> Self {
        Self::new(TrackedNdTreeConfig {
            filter_dominated: true,
        })
    }

    /// Create a new `TrackedNdTree` with filtering disabled (no-filtering mode).
    pub fn new_without_filtering() -> Self {
        Self::new(TrackedNdTreeConfig {
            filter_dominated: false,
        })
    }

    /// Insert a solution into the tracked tree.
    ///
    /// Returns an `InsertionResult` containing:
    /// - Whether the solution was inserted
    /// - The assigned index (if inserted)
    /// - Indices of solutions dominated by this solution
    pub fn insert(&mut self, solution: T) -> InsertionResult {
        let objectives = solution.objectives();

        // Check if solution is dominated by any existing solution
        let is_dominated = self.is_dominated_by_any(&solution);

        // In filtering mode, reject dominated solutions
        if self.config.filter_dominated && is_dominated {
            return InsertionResult::rejected();
        }

        // Find solutions dominated by the new solution (before insertion)
        let dominated_indices = self.find_dominated_by(&solution);

        // Assign index to the solution
        let assigned_index = self.next_index;
        self.next_index += 1;

        // Store the solution with its index
        self.index_to_solution
            .insert(assigned_index, solution.clone());
        self.solution_to_index
            .insert(objectives.to_vec(), assigned_index);

        // Insert into the underlying tree (this handles domination removal)
        self.tree.update(solution);

        // In no-filtering mode, keep dominated solutions in index maps
        // In filtering mode, remove dominated solutions from index maps
        if self.config.filter_dominated {
            for &idx in &dominated_indices {
                if let Some(dominated_sol) = self.index_to_solution.get(&idx) {
                    let dominated_objs = dominated_sol.objectives().to_vec();
                    self.solution_to_index.remove(&dominated_objs);
                }
                self.index_to_solution.remove(&idx);
            }
        }
        // In no-filtering mode, we keep the dominated solutions in our maps
        // even though they're removed from the tree

        InsertionResult::accepted(assigned_index, dominated_indices, is_dominated)
    }

    /// Check if a solution is dominated by any solution in the tree.
    fn is_dominated_by_any(&self, solution: &T) -> bool {
        for existing in self.tree.iter() {
            if solution.is_dominated_by(existing.objectives()) {
                return true;
            }
        }
        false
    }

    /// Find all solutions in the tree that are dominated by the given solution.
    fn find_dominated_by(&self, solution: &T) -> Vec<u32> {
        let mut dominated = Vec::new();

        for existing in self.tree.iter() {
            if existing.is_dominated_by(solution.objectives()) {
                // Find the index of this existing solution
                let obj_vec = existing.objectives().to_vec();
                if let Some(&idx) = self.solution_to_index.get(&obj_vec) {
                    dominated.push(idx);
                }
            }
        }

        dominated
    }

    /// Get the solution associated with an index.
    ///
    /// Returns `None` if:
    /// - The index was never assigned
    /// - In filtering mode: the solution was dominated and removed
    pub fn get_solution(&self, index: u32) -> Option<&T> {
        self.index_to_solution.get(&index)
    }

    /// Get the index of a solution by its objectives.
    pub fn get_index(&self, objectives: &[u64; D]) -> Option<u32> {
        self.solution_to_index.get(&objectives.to_vec()).copied()
    }

    /// Get the number of solutions currently tracked.
    ///
    /// - In filtering mode: equals the number of non-dominated solutions
    /// - In no-filtering mode: equals the total number of solutions inserted
    pub fn num_tracked_solutions(&self) -> usize {
        self.index_to_solution.len()
    }

    /// Get the number of solutions in the underlying ND-tree.
    ///
    /// This is always the number of non-dominated solutions.
    pub fn num_tree_solutions(&self) -> usize {
        self.tree.len()
    }

    /// Get the next index that will be assigned.
    pub fn next_index(&self) -> u32 {
        self.next_index
    }

    /// Check if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    /// Get an iterator over all solutions in the ND-tree.
    ///
    /// Note: This only iterates over non-dominated solutions in the tree.
    /// In no-filtering mode, dominated solutions are tracked but not in the tree.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.tree.iter()
    }

    /// Get all tracked solutions with their indices.
    ///
    /// Returns a vector of (index, solution) pairs sorted by index.
    pub fn get_all_tracked(&self) -> Vec<(u32, &T)> {
        let mut items: Vec<_> = self
            .index_to_solution
            .iter()
            .map(|(&idx, sol)| (idx, sol))
            .collect();
        items.sort_by_key(|(idx, _)| *idx);
        items
    }

    /// Get the configuration.
    pub fn config(&self) -> TrackedNdTreeConfig {
        self.config
    }

    /// Clear the tree and reset all tracking.
    pub fn clear(&mut self) {
        self.tree = NDTree::new();
        self.next_index = 0;
        self.index_to_solution.clear();
        self.solution_to_index.clear();
    }
}

impl<T, const N: usize, const D: usize, const C: usize> Default for TrackedNdTree<T, N, D, C>
where
    T: MoSolution<D> + HasObjectives<D> + Clone + PartialEq,
{
    fn default() -> Self {
        Self::new(TrackedNdTreeConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestTree = TrackedNdTree<Solution<2>, 8, 2, 4>;

    fn sol(objectives: [u64; 2]) -> Solution<2> {
        Solution::new(objectives)
    }

    #[test]
    fn test_insertion_no_filtering_all_get_indices() {
        let mut tree = TestTree::new_without_filtering();

        // Insert first solution
        let result = tree.insert(sol([10, 20]));
        assert!(result.inserted);
        assert_eq!(result.assigned_index, Some(0));
        assert_eq!(result.dominated_indices.len(), 0);
        assert!(!result.was_dominated_at_discovery);

        // Insert dominating solution
        let result = tree.insert(sol([5, 15]));
        assert!(result.inserted);
        assert_eq!(result.assigned_index, Some(1));
        assert_eq!(result.dominated_indices, vec![0]);
        assert!(!result.was_dominated_at_discovery);

        // Insert dominated solution (should still get index in no-filtering mode)
        let result = tree.insert(sol([12, 25]));
        assert!(result.inserted);
        assert_eq!(result.assigned_index, Some(2));
        assert_eq!(result.dominated_indices.len(), 0);
        assert!(result.was_dominated_at_discovery);

        // All 3 solutions should be tracked
        assert_eq!(tree.num_tracked_solutions(), 3);
        // But only non-dominated solutions in tree
        assert_eq!(tree.num_tree_solutions(), 1); // Only [5, 15]
    }

    #[test]
    fn test_insertion_with_filtering_dominated_rejected() {
        let mut tree = TestTree::new_with_filtering();

        // Insert first solution
        let result = tree.insert(sol([10, 20]));
        assert!(result.inserted);
        assert_eq!(result.assigned_index, Some(0));

        // Insert dominating solution
        let result = tree.insert(sol([5, 15]));
        assert!(result.inserted);
        assert_eq!(result.assigned_index, Some(1));
        assert_eq!(result.dominated_indices, vec![0]);

        // Insert dominated solution (should be rejected in filtering mode)
        let result = tree.insert(sol([12, 25]));
        assert!(!result.inserted);
        assert_eq!(result.assigned_index, None);
        assert_eq!(result.dominated_indices.len(), 0);

        // Only 2 solutions tracked (dominated one was rejected)
        assert_eq!(tree.num_tracked_solutions(), 1); // Only [5, 15] remains
        assert_eq!(tree.num_tree_solutions(), 1);
        assert_eq!(tree.next_index(), 2); // Next index would be 2
    }

    #[test]
    fn test_sequential_indices() {
        let mut tree = TestTree::new_without_filtering();

        let result = tree.insert(sol([10, 20]));
        assert_eq!(result.assigned_index, Some(0));

        let result = tree.insert(sol([15, 15]));
        assert_eq!(result.assigned_index, Some(1));

        let result = tree.insert(sol([20, 10]));
        assert_eq!(result.assigned_index, Some(2));

        assert_eq!(tree.next_index(), 3);
    }

    #[test]
    fn test_get_solution_by_index() {
        let mut tree = TestTree::new_without_filtering();

        tree.insert(sol([10, 20]));
        tree.insert(sol([5, 15]));

        let sol0 = tree.get_solution(0);
        assert!(sol0.is_some());
        assert_eq!(sol0.unwrap().objectives(), &[10, 20]);

        let sol1 = tree.get_solution(1);
        assert!(sol1.is_some());
        assert_eq!(sol1.unwrap().objectives(), &[5, 15]);

        let sol_invalid = tree.get_solution(999);
        assert!(sol_invalid.is_none());
    }

    #[test]
    fn test_get_index_by_objectives() {
        let mut tree = TestTree::new_without_filtering();

        tree.insert(sol([10, 20]));
        tree.insert(sol([5, 15]));

        let idx0 = tree.get_index(&[10, 20]);
        assert_eq!(idx0, Some(0));

        let idx1 = tree.get_index(&[5, 15]);
        assert_eq!(idx1, Some(1));

        let idx_invalid = tree.get_index(&[99, 99]);
        assert_eq!(idx_invalid, None);
    }

    #[test]
    fn test_multiple_dominated_solutions() {
        let mut tree = TestTree::new_without_filtering();

        tree.insert(sol([10, 20]));
        tree.insert(sol([15, 15]));
        tree.insert(sol([20, 10]));

        // Insert solution that dominates all three
        let result = tree.insert(sol([5, 5]));
        assert!(result.inserted);
        assert_eq!(result.assigned_index, Some(3));
        assert_eq!(result.dominated_indices.len(), 3);
        assert!(result.dominated_indices.contains(&0));
        assert!(result.dominated_indices.contains(&1));
        assert!(result.dominated_indices.contains(&2));
    }

    #[test]
    fn test_get_all_tracked_sorted() {
        let mut tree = TestTree::new_without_filtering();

        tree.insert(sol([10, 20]));
        tree.insert(sol([15, 15]));
        tree.insert(sol([20, 10]));

        let all = tree.get_all_tracked();
        assert_eq!(all.len(), 3);

        // Should be sorted by index
        assert_eq!(all[0].0, 0);
        assert_eq!(all[1].0, 1);
        assert_eq!(all[2].0, 2);
    }

    #[test]
    fn test_clear() {
        let mut tree = TestTree::new_without_filtering();

        tree.insert(sol([10, 20]));
        tree.insert(sol([15, 15]));

        assert_eq!(tree.num_tracked_solutions(), 2);
        assert_eq!(tree.next_index(), 2);

        tree.clear();

        assert_eq!(tree.num_tracked_solutions(), 0);
        assert_eq!(tree.next_index(), 0);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_dominated_in_filtering_mode_removes_from_tracking() {
        let mut tree = TestTree::new_with_filtering();

        tree.insert(sol([10, 20]));
        assert_eq!(tree.num_tracked_solutions(), 1);

        // This dominates the first solution
        tree.insert(sol([5, 15]));

        // First solution should be removed from tracking
        assert_eq!(tree.num_tracked_solutions(), 1);
        assert!(tree.get_solution(0).is_none()); // Index 0 removed
        assert!(tree.get_solution(1).is_some()); // Index 1 exists
    }

    #[test]
    fn test_dominated_in_no_filtering_mode_keeps_in_tracking() {
        let mut tree = TestTree::new_without_filtering();

        tree.insert(sol([10, 20]));
        assert_eq!(tree.num_tracked_solutions(), 1);

        // This dominates the first solution
        tree.insert(sol([5, 15]));

        // First solution should still be tracked in no-filtering mode
        assert_eq!(tree.num_tracked_solutions(), 2);
        assert!(tree.get_solution(0).is_some()); // Index 0 still tracked
        assert!(tree.get_solution(1).is_some()); // Index 1 exists

        // But tree only contains non-dominated
        assert_eq!(tree.num_tree_solutions(), 1);
    }
}
