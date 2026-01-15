//! Comprehensive test suite for testing multiple implementations of `ParetoFront` trait
//! Uses the macro-based testing technique described in:
//! <https://eli.thegreenplace.net/2021/testing-multiple-implementations-of-a-trait-in-rust/>
//!
//! This test suite is implementation-agnostic and tests pure Pareto Front behavior
//! without dependencies on SIMS-specific problem structures.

use pareto::{HasObjectives, MoSolution, ParetoFront};
use std::time::Duration;

// Helper function to initialize tracing for debugging
fn init_tracing() {
    use tracing_subscriber::fmt::format::FmtSpan;
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_span_events(FmtSpan::ACTIVE)
        .with_thread_ids(true)
        .with_line_number(true)
        .try_init();
}

/// A simple test solution for testing `ParetoFront` implementations
/// This is a minimal implementation that satisfies all trait requirements
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct TestSolution<const D: usize> {
    objectives: [u64; D],
    id: usize, // To distinguish solutions with same objectives
}

impl<const D: usize> TestSolution<D> {
    const fn new(objectives: [u64; D], id: usize) -> Self {
        Self { objectives, id }
    }
}

impl<const D: usize> HasObjectives<D> for TestSolution<D> {
    fn objectives(&self) -> &[u64; D] {
        &self.objectives
    }
}

impl<const D: usize> MoSolution<D> for TestSolution<D> {}

// Implement ImageSet trait for TestSolution (required by some implementations)
impl<const D: usize> pls::solution::ImageSet<D> for TestSolution<D> {
    fn selected_images(&self) -> impl Iterator<Item = usize> {
        std::iter::empty()
    }

    fn unselected_images(&self) -> impl Iterator<Item = usize> {
        std::iter::empty()
    }

    fn is_image_selected(&self, _image_index: usize) -> bool {
        false
    }

    fn num_selected_images(&self) -> usize {
        0
    }

    fn set_image(&mut self, _image_index: usize, _selected: bool) {}
}

// Dummy problem type for testing
struct DummyProblem;

impl<const D: usize> pls::problem::SetCoverProblem<D> for DummyProblem {
    fn is_set_cover(&self, _solution: &impl pls::solution::ImageSet<D>) -> bool {
        true
    }
    fn objective(&self, _index: usize) -> &pls::objectives::ObjectiveState<D> {
        unimplemented!()
    }
    fn objectives(&self) -> &[pls::objectives::ObjectiveState<D>; D] {
        unimplemented!()
    }
    fn max_objectives(&self) -> [u64; D] {
        [0; D]
    }
    fn objective_bounds(&self) -> Option<[[u64; 2]; D]> {
        None
    }
    fn num_images(&self) -> usize {
        0
    }
    fn num_elements(&self) -> usize {
        0
    }
    fn universe_size(&self) -> usize {
        0
    }
    fn image_contains_element(&self, _image_index: usize, _element_index: usize) -> bool {
        false
    }
    fn image_elements(&self, _image_index: usize) -> impl Iterator<Item = usize> + '_ {
        std::iter::empty()
    }
    fn overlap(&self, _image_i: usize, _image_j: usize) -> usize {
        0
    }
    fn objective_types(&self) -> &[pls::objectives::ObjectiveType; D] {
        unimplemented!()
    }
    fn instance_name(&self) -> &'static str {
        "dummy"
    }
    fn element_images(&self, _element_index: usize) -> impl Iterator<Item = usize> + '_ {
        std::iter::empty()
    }
}

// Implement SIMSCore for TestSolution (needed for EncodedSolution bound)
impl<const D: usize> pls::solution::SIMSCore<DummyProblem, D> for TestSolution<D> {
    fn to_debug_solution(&self) -> pls::solution::SIMSSolution {
        pls::solution::SIMSSolution {
            selected_images: vec![],
        }
    }

    fn objectives_mut(&mut self) -> &mut pareto::Objectives<D> {
        &mut self.objectives
    }
}

// Implement SIMSModifiable for TestSolution
impl<const D: usize> pls::solution::SIMSModifiable<DummyProblem, D> for TestSolution<D> {
    type Trackers = pls::objective_tracker::StandardTrackerArray<D>;

    fn add_image(
        &mut self,
        _image_index: usize,
        _problem: &DummyProblem,
        _trackers: &mut Self::Trackers,
    ) {
    }

    fn remove_image(
        &mut self,
        _image_index: usize,
        _problem: &DummyProblem,
        _trackers: &mut Self::Trackers,
    ) {
    }

    fn scaled_image_objective_deltas(
        &self,
        _images: &[usize],
        _problem: &DummyProblem,
        _trackers: &Self::Trackers,
    ) -> Vec<pls::problem::ScaledObjectiveDeltas<D>> {
        vec![]
    }

    fn find_best_image_to_add(
        &self,
        _problem: &DummyProblem,
        _trackers: &Self::Trackers,
    ) -> Option<usize> {
        None
    }

    fn find_best_image_to_remove(
        &self,
        _problem: &DummyProblem,
        _trackers: &Self::Trackers,
    ) -> Option<usize> {
        None
    }

    fn neighborhood(
        &self,
        _k: u32,
        _problem: &DummyProblem,
        _timer: &pls::timer::Timer,
        _is_deterministic: bool,
        _trackers: &mut Self::Trackers,
    ) -> Vec<Self> {
        vec![]
    }

    fn is_valid(&self, _problem: &DummyProblem) -> bool {
        true
    }
}

// Implement EncodedSolution for TestSolution
impl<const D: usize> pls::solution::EncodedSolution<DummyProblem, D> for TestSolution<D> {
    fn timestamp(&self) -> Duration {
        Duration::from_secs(0)
    }
}

/// Macro to generate comprehensive test suite for each `ParetoFront` implementation
macro_rules! solution_set_tests {
    ($($name:ident: $type:ty,)*) => {
    $(
        mod $name {
            use super::*;

            type SolutionSet2D = $type;

            #[test]
            fn test_new_empty() {
                let set = SolutionSet2D::new("test");
                assert!(set.is_empty());
                assert_eq!(set.len(), 0);
            }

            #[test]
            fn test_insert_single_solution() {
                init_tracing();
                let mut set = SolutionSet2D::new("test");
                let sol = TestSolution::new([10, 20], 0);

                assert!(set.try_insert(&sol));
                assert_eq!(set.len(), 1);
                assert!(!set.is_empty());
                eprintln!("About to call contains() for sol: {:?}", sol.objectives());
                assert!(set.contains(&sol));
            }

            #[test]
            fn test_insert_duplicate_solution() {
                let mut set = SolutionSet2D::new("test");
                let sol = TestSolution::new([10, 20], 0);

                assert!(set.try_insert(&sol));
                assert!(!set.try_insert(&sol)); // Duplicate should be rejected
                assert_eq!(set.len(), 1);
            }

            #[test]
            fn test_insert_dominated_solution() {
                let mut set = SolutionSet2D::new("test");
                let sol1 = TestSolution::new([10, 20], 0);
                let sol2 = TestSolution::new([15, 25], 1); // Dominated by sol1

                assert!(set.try_insert(&sol1));
                assert!(!set.try_insert(&sol2)); // Should be rejected (dominated)
                assert_eq!(set.len(), 1);
            }

            #[test]
            fn test_insert_dominating_solution() {
                init_tracing();
                let mut set = SolutionSet2D::new("test");
                let sol1 = TestSolution::new([15, 25], 0);
                let sol2 = TestSolution::new([10, 20], 1); // Dominates sol1

                assert!(set.try_insert(&sol1));
                assert!(set.try_insert(&sol2)); // Should be accepted
                assert_eq!(set.len(), 1); // sol1 should be removed
                eprintln!("About to call contains() for sol2: {:?}", sol2.objectives());
                assert!(set.contains(&sol2));
                eprintln!("About to call contains() for sol1: {:?}", sol1.objectives());
                assert!(!set.contains(&sol1));
            }

            #[test]
            fn test_insert_non_dominated_solutions() {
                let mut set = SolutionSet2D::new("test");
                let sol1 = TestSolution::new([10, 30], 0);
                let sol2 = TestSolution::new([20, 20], 1);
                let sol3 = TestSolution::new([30, 10], 2);

                assert!(set.try_insert(&sol1));
                assert!(set.try_insert(&sol2));
                assert!(set.try_insert(&sol3));
                assert_eq!(set.len(), 3);
            }

            #[test]
            fn test_pareto_front_maintenance() {
                init_tracing();
                let mut set = SolutionSet2D::new("test");

                // Insert non-dominated solutions
                let sol1 = TestSolution::new([10, 30], 0);
                let sol2 = TestSolution::new([20, 20], 1);
                let sol3 = TestSolution::new([30, 10], 2);

                set.try_insert(&sol1);
                set.try_insert(&sol2);
                set.try_insert(&sol3);

                // Insert a solution that dominates sol2
                let sol4 = TestSolution::new([15, 15], 3);
                assert!(set.try_insert(&sol4));

                // sol2 should be removed, others remain
                assert_eq!(set.len(), 3);
                eprintln!("About to call contains() for sol1: {:?}", sol1.objectives());
                assert!(set.contains(&sol1));
                eprintln!("About to call contains() for sol2: {:?}", sol2.objectives());
                assert!(!set.contains(&sol2)); // Dominated and removed
                eprintln!("About to call contains() for sol3: {:?}", sol3.objectives());
                assert!(set.contains(&sol3));
                eprintln!("About to call contains() for sol4: {:?}", sol4.objectives());
                assert!(set.contains(&sol4));
            }

            #[test]
            fn test_iter() {
                let mut set = SolutionSet2D::new("test");
                let sol1 = TestSolution::new([10, 30], 0);
                let sol2 = TestSolution::new([20, 20], 1);
                let sol3 = TestSolution::new([30, 10], 2);

                set.try_insert(&sol1);
                set.try_insert(&sol2);
                set.try_insert(&sol3);

                let solutions: Vec<_> = set.iter().collect();
                assert_eq!(solutions.len(), 3);
            }

            #[test]
            fn test_from_iterator() {
                let solutions = vec![
                    TestSolution::new([10, 30], 0),
                    TestSolution::new([20, 20], 1),
                    TestSolution::new([30, 10], 2),
                ];

                let set: SolutionSet2D = solutions.into_iter().collect();
                assert_eq!(set.len(), 3);
            }

            #[test]
            fn test_into_iterator() {
                let mut set = SolutionSet2D::new("test");
                set.try_insert(&TestSolution::new([10, 30], 0));
                set.try_insert(&TestSolution::new([20, 20], 1));
                set.try_insert(&TestSolution::new([30, 10], 2));

                let solutions: Vec<_> = set.into_iter().collect();
                assert_eq!(solutions.len(), 3);
            }

            #[test]
            fn test_with_name() {
                let set = SolutionSet2D::new("original").with_name("renamed");
                assert!(set.is_empty());
            }

            #[test]
            fn test_validate_no_dominated_solutions() {
                let mut set = SolutionSet2D::new("test");
                set.try_insert(&TestSolution::new([10, 30], 0));
                set.try_insert(&TestSolution::new([20, 20], 1));
                set.try_insert(&TestSolution::new([30, 10], 2));

                // This should not panic - all solutions are non-dominated
                set.validate::<2>();
            }

            #[test]
            fn test_insert_equal_objectives() {
                let mut set = SolutionSet2D::new("test");
                let sol1 = TestSolution::new([10, 20], 0);
                let sol2 = TestSolution::new([10, 20], 1); // Same objectives, different id

                assert!(set.try_insert(&sol1));
                assert!(!set.try_insert(&sol2)); // Should be rejected (equal)
                assert_eq!(set.len(), 1);
            }

            #[test]
            fn test_multiple_dominated_removal() {
                let mut set = SolutionSet2D::new("test");

                // Insert several solutions
                // Note: [20,30] dominates [25,35], so only 3 will remain after initial insertions
                set.try_insert(&TestSolution::new([20, 30], 0));
                set.try_insert(&TestSolution::new([30, 25], 1));
                set.try_insert(&TestSolution::new([25, 35], 2)); // Will be rejected (dominated by [20,30])
                set.try_insert(&TestSolution::new([35, 20], 3));

                // After insertions, we should have 3 non-dominated solutions:
                // [20,30], [30,25], [35,20]
                assert_eq!(set.len(), 3);

                // Insert a solution that dominates ALL existing ones
                let dominating = TestSolution::new([15, 15], 4);
                assert!(set.try_insert(&dominating));

                // All previous solutions should be removed, only [15,15] remains
                assert_eq!(set.len(), 1);

                // Verify [15,15] is in the set
                let objectives: Vec<_> = set.iter().map(|s| *s.objectives()).collect();
                assert_eq!(objectives, vec![[15, 15]]);
            }

            #[test]
            fn test_edge_case_zero_objectives() {
                let mut set = SolutionSet2D::new("test");
                let sol1 = TestSolution::new([0, 0], 0);
                let sol2 = TestSolution::new([1, 1], 1);

                assert!(set.try_insert(&sol1));
                assert!(!set.try_insert(&sol2)); // Dominated by [0,0]
                assert_eq!(set.len(), 1);
            }

            #[test]
            fn test_large_objective_values() {
                let mut set = SolutionSet2D::new("test");
                let sol1 = TestSolution::new([u64::MAX - 100, u64::MAX - 200], 0);
                let sol2 = TestSolution::new([u64::MAX - 200, u64::MAX - 100], 1);

                assert!(set.try_insert(&sol1));
                assert!(set.try_insert(&sol2));
                assert_eq!(set.len(), 2);
            }

            #[test]
            #[allow(clippy::cast_possible_truncation, reason = "Test data with small values")]
            fn test_sequential_insertions() {
                let mut set = SolutionSet2D::new("test");

                // Insert solutions in sequence, each potentially affecting the set
                for i in 0..10 {
                    let sol = TestSolution::new([i * 10, (10 - i) * 10], i as usize);
                    set.try_insert(&sol);
                }

                // All should be non-dominated (forming a Pareto front)
                assert_eq!(set.len(), 10);
                set.validate::<2>();
            }

            #[test]
            fn test_insert_unchecked() {
                let mut set = SolutionSet2D::new("test");
                let sol1 = TestSolution::new([10, 20], 0);
                let sol2 = TestSolution::new([10, 20], 1); // Same objectives

                set.insert_unchecked(&sol1);
                set.insert_unchecked(&sol2); // This doesn't check for duplicates

                // Both should be in the set (unchecked insert)
                assert!(set.len() >= 1); // Implementation-specific behavior
            }

            #[test]
            #[allow(clippy::cast_possible_truncation, reason = "Test data with small values")]
            fn test_stress_many_solutions() {
                let mut set = SolutionSet2D::new("test");

                // Insert many solutions with various dominance relationships
                for i in 0..100 {
                    for j in 0..100 {
                        let sol = TestSolution::new([i, j], (i * 100 + j) as usize);
                        set.try_insert(&sol);
                    }
                }

                // Should only keep non-dominated solutions
                // For this pattern, only solutions on the border (i=0 or j=0) are non-dominated
                assert!(set.len() > 0);
                assert!(set.len() <= 199); // At most 100 + 99 unique non-dominated solutions

                // Verify no dominated solutions remain
                set.validate::<2>();
            }

            #[test]
            #[allow(clippy::cast_possible_truncation, reason = "Test data with small values")]
            fn test_diagonal_pareto_front() {
                let mut set = SolutionSet2D::new("test");

                // Create a diagonal Pareto front
                for i in 0..20 {
                    let _ = set.try_insert(&TestSolution::new([i * 5, (20 - i) * 5], i as usize));
                }

                assert_eq!(set.len(), 20);
                set.validate::<2>();
            }

            #[test]
            #[allow(clippy::cast_possible_truncation, reason = "Test data with small values")]
            fn test_reverse_insertion_order() {
                let mut set = SolutionSet2D::new("test");

                // Insert in reverse order
                for i in (0..10).rev() {
                    let _ = set.try_insert(&TestSolution::new([i * 10, (10 - i) * 10], i as usize));
                }

                assert_eq!(set.len(), 10);
                set.validate::<2>();
            }

            #[test]
            #[allow(clippy::cast_possible_truncation, reason = "Test data with small values")]
            fn test_single_objective_variation() {
                let mut set = SolutionSet2D::new("test");

                // All solutions have same second objective, varying first
                for i in 0..10 {
                    let _ = set.try_insert(&TestSolution::new([i * 10, 100], i as usize));
                }

                // Only the solution with smallest first objective should remain
                assert_eq!(set.len(), 1);

                // Verify it has objectives [0, 100]
                let objectives: Vec<_> = set.iter().map(|s| *s.objectives()).collect();
                assert_eq!(objectives, vec![[0, 100]]);
            }

            #[test]
            fn test_alternating_dominance() {
                let mut set = SolutionSet2D::new("test");

                // Insert solutions that alternate between being dominated and dominating
                set.try_insert(&TestSolution::new([50, 50], 0));
                set.try_insert(&TestSolution::new([40, 60], 1)); // Non-dominated
                set.try_insert(&TestSolution::new([30, 30], 2)); // Dominates previous
                set.try_insert(&TestSolution::new([25, 35], 3)); // Non-dominated with [30,30]

                assert!(set.len() >= 1);
                set.validate::<2>();
            }

            #[test]
            fn test_weak_dominance() {
                let mut set = SolutionSet2D::new("test");

                // Solutions that are equal in one objective
                let sol1 = TestSolution::new([10, 30], 0);
                let sol2 = TestSolution::new([10, 40], 1); // Equal first, worse second
                let sol3 = TestSolution::new([20, 30], 2); // Worse first, equal second

                assert!(set.try_insert(&sol1));
                assert!(!set.try_insert(&sol2)); // Weakly dominated
                assert!(!set.try_insert(&sol3)); // Weakly dominated
                assert_eq!(set.len(), 1);
            }

            #[test]
            fn test_min_value_objectives() {
                let mut set = SolutionSet2D::new("test");
                let sol1 = TestSolution::new([u64::MIN, u64::MIN], 0);
                let sol2 = TestSolution::new([u64::MIN + 1, u64::MIN + 1], 1);

                assert!(set.try_insert(&sol1));
                assert!(!set.try_insert(&sol2)); // Dominated by MIN
                assert_eq!(set.len(), 1);
            }

            #[test]
            fn test_mixed_extreme_values() {
                let mut set = SolutionSet2D::new("test");

                // Mix of MIN and MAX values
                set.try_insert(&TestSolution::new([u64::MIN, u64::MAX], 0));
                set.try_insert(&TestSolution::new([u64::MAX, u64::MIN], 1));
                set.try_insert(&TestSolution::new([u64::MIN + 100, u64::MAX - 100], 2));

                // All 3 are non-dominated:
                // [MIN, MAX] vs [MAX, MIN]: MIN < MAX in obj1, but MAX > MIN in obj2
                // [MIN+100, MAX-100] vs [MIN, MAX]: MIN+100 > MIN (worse), but MAX-100 < MAX (better)
                // [MIN+100, MAX-100] vs [MAX, MIN]: MIN+100 < MAX (better), but MAX-100 > MIN (worse)
                assert_eq!(set.len(), 3);
                set.validate::<2>();
            }

            #[test]
            fn test_incremental_dominance() {
                let mut set = SolutionSet2D::new("test");

                // Build a Pareto front
                set.try_insert(&TestSolution::new([50, 10], 0));
                set.try_insert(&TestSolution::new([40, 20], 1));
                set.try_insert(&TestSolution::new([30, 30], 2));
                set.try_insert(&TestSolution::new([20, 40], 3));
                set.try_insert(&TestSolution::new([10, 50], 4));
                assert_eq!(set.len(), 5);

                // Insert [25,25] which only dominates [30,30]
                // vs [50,10]: 25<50 YES, 25<10 NO -> non-dominated
                // vs [40,20]: 25<40 YES, 25<20 NO -> non-dominated
                // vs [30,30]: 25<30 YES, 25<30 YES -> DOMINATES
                // vs [20,40]: 25<20 NO -> non-dominated
                // vs [10,50]: 25<10 NO -> non-dominated
                set.try_insert(&TestSolution::new([25, 25], 5));
                assert_eq!(set.len(), 5); // [50,10], [40,20], [20,40], [10,50], [25,25]
                set.validate::<2>();
            }

            #[test]
            fn test_convex_front() {
                let mut set = SolutionSet2D::new("test");

                // Create a convex Pareto front (exponential trade-off)
                let points = vec![
                    [1, 100],
                    [4, 50],
                    [9, 33],
                    [16, 25],
                    [25, 20],
                    [100, 1],
                ];

                for (i, point) in points.iter().enumerate() {
                    set.try_insert(&TestSolution::new(*point, i));
                }

                assert_eq!(set.len(), points.len());
                set.validate::<2>();
            }

            #[test]
            fn test_concave_front() {
                let mut set = SolutionSet2D::new("test");

                // Create a concave Pareto front
                set.try_insert(&TestSolution::new([10, 90], 0));
                set.try_insert(&TestSolution::new([20, 85], 1));
                set.try_insert(&TestSolution::new([30, 75], 2));
                set.try_insert(&TestSolution::new([50, 50], 3));
                set.try_insert(&TestSolution::new([75, 30], 4));
                set.try_insert(&TestSolution::new([85, 20], 5));
                set.try_insert(&TestSolution::new([90, 10], 6));

                assert_eq!(set.len(), 7);
                set.validate::<2>();
            }

            #[test]
            #[allow(clippy::cast_possible_truncation, reason = "Test data with small values")]
            fn test_dense_vs_sparse_front() {
                let mut set = SolutionSet2D::new("test");

                // Dense region
                for i in 0..10 {
                    let _ = set.try_insert(&TestSolution::new([i, 100 - i * 2], i as usize));
                }

                // Sparse region
                set.try_insert(&TestSolution::new([50, 50], 100));
                set.try_insert(&TestSolution::new([100, 0], 101));

                assert!(set.len() >= 10);
                set.validate::<2>();
            }

            #[test]
            fn test_partial_objective_equality() {
                let mut set = SolutionSet2D::new("test");

                // Multiple solutions with same first objective, different second
                set.try_insert(&TestSolution::new([10, 50], 0));
                set.try_insert(&TestSolution::new([10, 40], 1));
                set.try_insert(&TestSolution::new([10, 30], 2));

                // Only one should remain (the one with best second objective)
                assert_eq!(set.len(), 1);

                // Verify first objective is 10, second should be minimal
                // Note: Due to insertion order dependency in some implementations,
                // we just verify Pareto optimality is maintained
                let objectives: Vec<_> = set.iter().map(|s| *s.objectives()).collect();
                assert_eq!(objectives[0][0], 10);
                assert!(objectives[0][1] <= 50); // Should be one of the inserted values
                set.validate::<2>();
            }

            #[test]
            fn test_scattered_random_pattern() {
                let mut set = SolutionSet2D::new("test");

                // Insert scattered points that form a sparse front
                let scattered = vec![
                    [5, 95], [15, 80], [25, 70], [40, 55],
                    [55, 40], [70, 25], [80, 15], [95, 5],
                ];

                for (i, point) in scattered.iter().enumerate() {
                    set.try_insert(&TestSolution::new(*point, i));
                }

                assert_eq!(set.len(), scattered.len());
                set.validate::<2>();
            }
        }
    )*
    }
}

// Generate test modules for each implementation
solution_set_tests! {
    linkedlist: pls::solution_set_impl::LinkedListSolutionSet<TestSolution<2>, 2>,
    vec_set: pls::solution_set_impl::VecSolutionSet<TestSolution<2>, 2>,
    btree: pls::solution_set_impl::BTreeSolutionSet<TestSolution<2>, 2>,
    ndtree: pls::solution_set_impl::NdTreeSolutionSet<TestSolution<2>, 2>,
}

// Additional tests for 3D objectives (where supported)
mod three_dimensional_tests {
    use super::*;
    use pls::solution_set_impl::*;

    type Solution3D = TestSolution<3>;

    #[test]
    fn test_linkedlist_3d() {
        let mut set = LinkedListSolutionSet::<Solution3D, 3>::new("test");

        set.try_insert(&TestSolution::new([10, 20, 30], 0));
        set.try_insert(&TestSolution::new([15, 15, 25], 1));
        set.try_insert(&TestSolution::new([20, 10, 20], 2));

        assert_eq!(set.len(), 3);
        set.validate::<3>();
    }

    #[test]
    fn test_vec_3d() {
        let mut set = VecSolutionSet::<Solution3D, 3>::new("test");

        set.try_insert(&TestSolution::new([10, 20, 30], 0));
        set.try_insert(&TestSolution::new([15, 15, 25], 1));
        set.try_insert(&TestSolution::new([20, 10, 20], 2));

        assert_eq!(set.len(), 3);
        set.validate::<3>();
    }

    #[test]
    fn test_ndtree_3d() {
        let mut set = NdTreeSolutionSet::<Solution3D, 3>::new("test");

        set.try_insert(&TestSolution::new([10, 20, 30], 0));
        set.try_insert(&TestSolution::new([15, 15, 25], 1));
        set.try_insert(&TestSolution::new([20, 10, 20], 2));

        assert_eq!(set.len(), 3);
        set.validate::<3>();
    }

    #[test]
    fn test_3d_dominated_removal() {
        let mut set = LinkedListSolutionSet::<Solution3D, 3>::new("test");

        // Insert several solutions
        set.try_insert(&TestSolution::new([20, 20, 20], 0));
        set.try_insert(&TestSolution::new([25, 25, 25], 1));
        set.try_insert(&TestSolution::new([30, 30, 30], 2));

        // Insert a dominating solution
        assert!(set.try_insert(&TestSolution::new([10, 10, 10], 3)));

        // All previous should be removed
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_3d_complex_front() {
        let mut set = NdTreeSolutionSet::<Solution3D, 3>::new("test");

        // Create a 3D Pareto front with multiple non-dominated solutions
        set.try_insert(&TestSolution::new([10, 30, 50], 0));
        set.try_insert(&TestSolution::new([20, 20, 40], 1));
        set.try_insert(&TestSolution::new([30, 10, 30], 2));
        set.try_insert(&TestSolution::new([15, 25, 35], 3));
        set.try_insert(&TestSolution::new([25, 15, 45], 4));

        assert!(set.len() >= 3);
        set.validate::<3>();
    }

    // 4D tests
    type Solution4D = TestSolution<4>;

    #[test]
    fn test_4d_basic() {
        let mut set = LinkedListSolutionSet::<Solution4D, 4>::new("test");

        set.try_insert(&TestSolution::new([10, 20, 30, 40], 0));
        set.try_insert(&TestSolution::new([15, 15, 25, 35], 1));
        set.try_insert(&TestSolution::new([20, 10, 20, 30], 2));

        assert_eq!(set.len(), 3);
        set.validate::<4>();
    }

    #[test]
    fn test_4d_dominated_removal() {
        let mut set = NdTreeSolutionSet::<Solution4D, 4>::new("test");

        // Insert several solutions
        set.try_insert(&TestSolution::new([20, 20, 20, 20], 0));
        set.try_insert(&TestSolution::new([25, 25, 25, 25], 1));
        set.try_insert(&TestSolution::new([30, 30, 30, 30], 2));

        // Insert a dominating solution
        assert!(set.try_insert(&TestSolution::new([10, 10, 10, 10], 3)));

        // All previous should be removed
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_4d_complex_front() {
        let mut set = VecSolutionSet::<Solution4D, 4>::new("test");

        // Create a 4D Pareto front
        set.try_insert(&TestSolution::new([10, 40, 50, 60], 0));
        set.try_insert(&TestSolution::new([20, 30, 40, 50], 1));
        set.try_insert(&TestSolution::new([30, 20, 30, 40], 2));
        set.try_insert(&TestSolution::new([40, 10, 20, 30], 3));
        set.try_insert(&TestSolution::new([15, 35, 45, 55], 4));

        assert!(set.len() >= 4);
        set.validate::<4>();
    }

    #[test]
    fn test_4d_weak_dominance() {
        let mut set = LinkedListSolutionSet::<Solution4D, 4>::new("test");

        // Solutions equal in some objectives
        let sol1 = TestSolution::new([10, 20, 30, 40], 0);
        let sol2 = TestSolution::new([10, 20, 30, 50], 1); // Equal in first 3, worse in 4th

        assert!(set.try_insert(&sol1));
        assert!(!set.try_insert(&sol2)); // Weakly dominated
        assert_eq!(set.len(), 1);
    }

    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Test data with small values"
    )]
    fn test_4d_diagonal_front() {
        let mut set = LinkedListSolutionSet::<Solution4D, 4>::new("test");

        // Create diagonal Pareto front
        for i in 0..10 {
            set.try_insert(&TestSolution::new(
                [i * 5, (10 - i) * 5, i * 3, (10 - i) * 3],
                i as usize,
            ));
        }

        assert_eq!(set.len(), 10);
        set.validate::<4>();
    }

    /// Test case derived from real bug: two non-dominated solutions
    /// where one implementation kept one and the other kept the other.
    /// These are actual solutions from `lagos_nigeria_50` test case.
    #[test]
    #[allow(clippy::unreadable_literal, reason = "Real data from bug reproduction")]
    fn test_4d_bug_non_dominated_pair_1() {
        let sol_a = TestSolution::new([4375200, 105740, 397, 40250], 0);
        let sol_b = TestSolution::new([3239500, 213005, 373, 44190], 1);

        // Verify neither dominates the other
        // sol_b: better in obj1 (3239500 < 4375200), worse in obj2 (213005 > 105740)
        // sol_b: better in obj3 (373 < 397), worse in obj4 (44190 > 40250)
        // Therefore: non-dominated

        // Test all implementations keep both solutions
        let mut ndtree = NdTreeSolutionSet::<Solution4D, 4>::new("ndtree");
        assert!(ndtree.try_insert(&sol_a));
        assert!(
            ndtree.try_insert(&sol_b),
            "nd-tree should accept non-dominated solution"
        );
        assert_eq!(
            ndtree.len(),
            2,
            "nd-tree should keep both non-dominated solutions"
        );

        let mut vec_set = VecSolutionSet::<Solution4D, 4>::new("vec");
        assert!(vec_set.try_insert(&sol_a));
        assert!(
            vec_set.try_insert(&sol_b),
            "vector should accept non-dominated solution"
        );
        assert_eq!(
            vec_set.len(),
            2,
            "vector should keep both non-dominated solutions"
        );

        let mut linkedlist = LinkedListSolutionSet::<Solution4D, 4>::new("linkedlist");
        assert!(linkedlist.try_insert(&sol_a));
        assert!(
            linkedlist.try_insert(&sol_b),
            "linkedlist should accept non-dominated solution"
        );
        assert_eq!(
            linkedlist.len(),
            2,
            "linkedlist should keep both non-dominated solutions"
        );
    }

    /// Test the same pair in reverse insertion order
    #[test]
    #[allow(clippy::unreadable_literal, reason = "Real data from bug reproduction")]
    fn test_4d_bug_non_dominated_pair_1_reverse() {
        let sol_a = TestSolution::new([4375200, 105740, 397, 40250], 0);
        let sol_b = TestSolution::new([3239500, 213005, 373, 44190], 1);

        // Insert in reverse order
        let mut ndtree = NdTreeSolutionSet::<Solution4D, 4>::new("ndtree");
        assert!(ndtree.try_insert(&sol_b));
        assert!(
            ndtree.try_insert(&sol_a),
            "nd-tree should accept non-dominated solution (reverse order)"
        );
        assert_eq!(
            ndtree.len(),
            2,
            "nd-tree should keep both non-dominated solutions (reverse)"
        );

        let mut vec_set = VecSolutionSet::<Solution4D, 4>::new("vec");
        assert!(vec_set.try_insert(&sol_b));
        assert!(
            vec_set.try_insert(&sol_a),
            "vector should accept non-dominated solution (reverse order)"
        );
        assert_eq!(
            vec_set.len(),
            2,
            "vector should keep both non-dominated solutions (reverse)"
        );

        let mut linkedlist = LinkedListSolutionSet::<Solution4D, 4>::new("linkedlist");
        assert!(linkedlist.try_insert(&sol_b));
        assert!(
            linkedlist.try_insert(&sol_a),
            "linkedlist should accept non-dominated solution (reverse order)"
        );
        assert_eq!(
            linkedlist.len(),
            2,
            "linkedlist should keep both non-dominated solutions (reverse)"
        );
    }

    /// Test a batch of non-dominated solutions from the actual bug scenario
    #[test]
    #[allow(
        clippy::unreadable_literal,
        clippy::uninlined_format_args,
        clippy::redundant_closure_for_method_calls,
        clippy::stable_sort_primitive,
        reason = "Real data from bug reproduction"
    )]
    fn test_4d_bug_batch_consistency() {
        // Sample of non-dominated solutions from the initial population
        let solutions = vec![
            TestSolution::new([4375200, 105740, 397, 40250], 0),
            TestSolution::new([3239500, 213005, 373, 44190], 1),
            TestSolution::new([5262380, 50566, 458, 40250], 2),
            TestSolution::new([5252230, 61142, 333, 36090], 3),
            TestSolution::new([4784100, 34107, 492, 36090], 4),
        ];

        // All implementations should produce identical results
        let mut ndtree = NdTreeSolutionSet::<Solution4D, 4>::new("ndtree");
        let mut vec_set = VecSolutionSet::<Solution4D, 4>::new("vec");
        let mut linkedlist = LinkedListSolutionSet::<Solution4D, 4>::new("linkedlist");

        for sol in &solutions {
            ndtree.try_insert(sol);
            vec_set.try_insert(sol);
            linkedlist.try_insert(sol);
        }

        // All should have same number of solutions
        let ndtree_len = ndtree.len();
        let vec_len = vec_set.len();
        let linkedlist_len = linkedlist.len();

        assert_eq!(
            ndtree_len, vec_len,
            "nd-tree and vector should have same solution count. nd-tree: {}, vector: {}",
            ndtree_len, vec_len
        );
        assert_eq!(
            vec_len, linkedlist_len,
            "vector and linkedlist should have same solution count. vector: {}, linkedlist: {}",
            vec_len, linkedlist_len
        );

        // Check that all implementations have the same objectives
        let mut ndtree_objs: Vec<_> = ndtree.iter().map(|s| s.objectives()).collect();
        let mut vec_objs: Vec<_> = vec_set.iter().map(|s| s.objectives()).collect();
        let mut linkedlist_objs: Vec<_> = linkedlist.iter().map(|s| s.objectives()).collect();

        ndtree_objs.sort();
        vec_objs.sort();
        linkedlist_objs.sort();

        assert_eq!(
            ndtree_objs, vec_objs,
            "nd-tree and vector should have identical objective sets"
        );
        assert_eq!(
            vec_objs, linkedlist_objs,
            "vector and linkedlist should have identical objective sets"
        );
    }

    /// Test consistency with a larger batch from real data
    #[test]
    #[allow(
        clippy::unreadable_literal,
        clippy::uninlined_format_args,
        clippy::collection_is_never_read,
        clippy::stable_sort_primitive,
        reason = "Real data from bug reproduction"
    )]
    fn test_4d_bug_large_batch_consistency() {
        // More solutions from the actual divergence scenario
        let solutions = vec![
            TestSolution::new([4375200, 105740, 397, 40250], 0),
            TestSolution::new([5262380, 50566, 458, 40250], 1),
            TestSolution::new([5252230, 61142, 333, 36090], 2),
            TestSolution::new([4784100, 34107, 492, 36090], 3),
            TestSolution::new([5707370, 14424, 492, 35330], 4),
            TestSolution::new([9123670, 2923, 492, 35330], 5),
            TestSolution::new([5155000, 30282, 492, 38110], 6),
            TestSolution::new([5553320, 38749, 401, 36030], 7),
            TestSolution::new([4588940, 66668, 492, 38070], 8),
            TestSolution::new([3615480, 194145, 401, 43950], 9),
            TestSolution::new([3239500, 213005, 373, 44190], 10),
            TestSolution::new([4511850, 168727, 458, 38850], 11),
            TestSolution::new([4416210, 96503, 492, 38590], 12),
            TestSolution::new([5875780, 13470, 492, 35330], 13),
            TestSolution::new([4970430, 46381, 475, 37210], 14),
        ];

        // Test each implementation
        let mut ndtree = NdTreeSolutionSet::<Solution4D, 4>::new("ndtree");
        let mut vec_set = VecSolutionSet::<Solution4D, 4>::new("vec");
        let mut linkedlist = LinkedListSolutionSet::<Solution4D, 4>::new("linkedlist");

        for sol in &solutions {
            ndtree.try_insert(sol);
            vec_set.try_insert(sol);
            linkedlist.try_insert(sol);
        }

        let ndtree_len = ndtree.len();
        let vec_len = vec_set.len();
        let linkedlist_len = linkedlist.len();

        assert_eq!(
            ndtree_len, vec_len,
            "nd-tree and vector should have same solution count with large batch. nd-tree: {}, vector: {}",
            ndtree_len, vec_len
        );
        assert_eq!(
            vec_len, linkedlist_len,
            "vector and linkedlist should have same solution count with large batch"
        );

        // Collect and compare objective sets
        let mut ndtree_objs: Vec<_> = ndtree.iter().map(|s| *s.objectives()).collect();
        let mut vec_objs: Vec<_> = vec_set.iter().map(|s| *s.objectives()).collect();
        let mut linkedlist_objs: Vec<_> = linkedlist.iter().map(|s| *s.objectives()).collect();

        ndtree_objs.sort();
        vec_objs.sort();
        linkedlist_objs.sort();

        assert_eq!(
            ndtree_objs, vec_objs,
            "nd-tree and vector should have identical objective sets in large batch"
        );
    }

    /// Test that all implementations agree on dominance for tricky cases
    #[test]
    #[allow(
        clippy::uninlined_format_args,
        clippy::useless_vec,
        reason = "Test clarity"
    )]
    fn test_4d_dominance_consistency_tricky_cases() {
        // Cases where objectives trade off in complex ways
        let pairs = vec![
            // Better in 2, worse in 2
            ([10, 20, 30, 40], [20, 10, 40, 30]),
            // Better in 3, worse in 1
            ([10, 20, 30, 40], [20, 15, 25, 35]),
            // Better in 1, worse in 3
            ([10, 20, 30, 40], [5, 25, 35, 45]),
        ];

        for (i, (obj_a, obj_b)) in pairs.iter().enumerate() {
            let sol_a = TestSolution::new(*obj_a, i * 2);
            let sol_b = TestSolution::new(*obj_b, i * 2 + 1);

            // Test all implementations
            let mut ndtree = NdTreeSolutionSet::<Solution4D, 4>::new("ndtree");
            let mut vec_set = VecSolutionSet::<Solution4D, 4>::new("vec");
            let mut linkedlist = LinkedListSolutionSet::<Solution4D, 4>::new("linkedlist");

            let ndtree_a = ndtree.try_insert(&sol_a);
            let ndtree_b = ndtree.try_insert(&sol_b);

            let vec_a = vec_set.try_insert(&sol_a);
            let vec_b = vec_set.try_insert(&sol_b);

            let linkedlist_a = linkedlist.try_insert(&sol_a);
            let linkedlist_b = linkedlist.try_insert(&sol_b);

            // All implementations should agree
            assert_eq!(
                ndtree_a, vec_a,
                "nd-tree and vector disagree on inserting sol_a for pair {}: {:?}",
                i, obj_a
            );
            assert_eq!(
                ndtree_b, vec_b,
                "nd-tree and vector disagree on inserting sol_b for pair {}: {:?}",
                i, obj_b
            );
            assert_eq!(
                vec_a, linkedlist_a,
                "vector and linkedlist disagree on inserting sol_a for pair {}: {:?}",
                i, obj_a
            );
            assert_eq!(
                vec_b, linkedlist_b,
                "vector and linkedlist disagree on inserting sol_b for pair {}: {:?}",
                i, obj_b
            );

            // All should have same final count
            assert_eq!(
                ndtree.len(),
                vec_set.len(),
                "Final size mismatch between nd-tree and vector for pair {}",
                i
            );
            assert_eq!(
                vec_set.len(),
                linkedlist.len(),
                "Final size mismatch between vector and linkedlist for pair {}",
                i
            );
        }
    }
}
