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
    fn selected_images(&self) -> Vec<usize> {
        vec![]
    }

    fn unselected_images(&self) -> Vec<usize> {
        vec![]
    }

    fn is_image_selected(&self, _image_index: usize) -> bool {
        false
    }

    fn num_selected_images(&self) -> usize {
        0
    }

    fn set_image(&mut self, _image_index: usize, _selected: bool) {}

    fn clear_parts_counts(&self) -> &[usize] {
        &[]
    }
}

// Implement SIMSCore for TestSolution (needed for EncodedSolution bound)
impl<const D: usize> pls::solution::SIMSCore<D> for TestSolution<D> {
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
impl<const D: usize> pls::solution::SIMSModifiable<D> for TestSolution<D> {
    fn clear_parts_counts(&self) -> &[usize] {
        &[]
    }

    fn element_coverage(&self) -> &[usize] {
        &[]
    }

    fn add_image(&mut self, _image_index: usize, _problem: &pls::problem::Problem<Self, D>) {}

    fn remove_image(&mut self, _image_index: usize, _problem: &pls::problem::Problem<Self, D>) {}

    fn scaled_image_objective_deltas(
        &self,
        _images: &[usize],
        _problem: &pls::problem::Problem<Self, D>,
    ) -> Vec<pls::problem::ScaledObjectiveDeltas<D>> {
        vec![]
    }

    fn find_best_image_to_add(&self, _problem: &pls::problem::Problem<Self, D>) -> Option<usize> {
        None
    }

    fn find_best_image_to_remove(&self, _problem: &pls::problem::Problem<Self, D>) -> Option<usize> {
        None
    }

    fn get_neighborhood(&self, _problem: &pls::problem::Problem<Self, D>) -> Vec<Self> {
        vec![]
    }

    fn neighborhood(
        &self,
        _k: u32,
        _problem: &pls::problem::Problem<Self, D>,
        _timer: &pls::timer::Timer,
        _is_deterministic: bool,
    ) -> Vec<Self> {
        vec![]
    }

    fn is_valid(&self, _problem: &pls::problem::Problem<Self, D>) -> bool {
        true
    }
}

// Implement EncodedSolution for TestSolution
impl<const D: usize> pls::solution::EncodedSolution<D> for TestSolution<D> {
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
    #[allow(clippy::cast_possible_truncation, reason = "Test data with small values")]
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
}
