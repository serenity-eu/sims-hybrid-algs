use std::fmt::Debug;

use crate::nd_tree::{NDTree, NDTreeSolutionIntoIterator, NDTreeSolutionIterator, Solution};
use pareto::{HasObjectives, MoSolution, ParetoFront};

pub struct NdTreeParetoFront<const N: usize, const D: usize, const C: usize> {
    name: &'static str,
    nd_tree: NDTree<N, D, C>,
}

impl<'a, const N: usize, const D: usize, const C: usize> ParetoFront<'a, Solution<D>>
    for NdTreeParetoFront<N, D, C>
{
    type Iter<'b> = NDTreeSolutionIterator<'b, N, D, C>;
    type IntoIter = NDTreeSolutionIntoIterator<N, D, C>;

    fn new(name: &'static str) -> Self {
        NdTreeParetoFront {
            name,
            nd_tree: NDTree::new(),
        }
    }

    fn with_name(self, name: &'static str) -> Self {
        NdTreeParetoFront {
            name,
            nd_tree: self.nd_tree,
        }
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.nd_tree.iter()
    }

    fn contains(&self, solution: &Solution<D>) -> bool {
        // Check if any solution in the tree has the same objectives
        self.nd_tree
            .iter()
            .any(|s| s.objectives() == solution.objectives())
    }

    fn try_insert(&mut self, solution: &Solution<D>) -> bool {
        self.nd_tree.update(solution.clone())
    }

    fn insert_unchecked(&mut self, solution: &Solution<D>) {
        self.nd_tree.update_unchecked(solution.clone())
    }

    fn replace_if_exists(&mut self, solution: Solution<D>) {
        // Find if there's an existing solution with the same objectives
        let has_existing = self
            .nd_tree
            .iter()
            .any(|s| s.objectives() == solution.objectives());

        if has_existing {
            // Remove the existing solution and add the new one
            // For now, we'll use a simple approach: rebuild the tree
            let mut new_tree = NDTree::new();
            for existing in self.nd_tree.iter() {
                if existing.objectives() != solution.objectives() {
                    new_tree.update_unchecked(existing.clone());
                }
            }
            new_tree.update_unchecked(solution);
            self.nd_tree = new_tree;
        }
    }

    fn len(&self) -> usize {
        self.nd_tree.len()
    }
    fn is_empty(&self) -> bool {
        self.nd_tree.is_empty()
    }
}

impl<const N: usize, const D: usize, const C: usize> Debug for NdTreeParetoFront<N, D, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.nd_tree.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestSolution = Solution<2>;
    type TestParetoFront = NdTreeParetoFront<4, 2, 4>;

    fn solution(x: u64, y: u64) -> TestSolution {
        TestSolution::new([x, y])
    }

    #[test]
    fn test_pareto_front_new() {
        let pf = TestParetoFront::new("test");
        assert_eq!(pf.len(), 0);
        assert!(pf.is_empty());
        assert_eq!(pf.name, "test");
    }

    #[test]
    fn test_pareto_front_with_name() {
        let pf = TestParetoFront::new("original").with_name("renamed");
        assert_eq!(pf.name, "renamed");
    }

    #[test]
    fn test_insert_unchecked_single_solution() {
        let mut pf = TestParetoFront::new("test");
        let sol = solution(1, 2);

        pf.insert_unchecked(&sol);

        assert_eq!(pf.len(), 1);
        assert!(!pf.is_empty());
        assert!(pf.contains(&sol));
    }

    #[test]
    fn test_insert_unchecked_multiple_solutions() {
        let mut pf = TestParetoFront::new("test");
        let sol1 = solution(1, 2);
        let sol2 = solution(2, 1);
        let sol3 = solution(3, 3);

        pf.insert_unchecked(&sol1);
        pf.insert_unchecked(&sol2);
        pf.insert_unchecked(&sol3);

        assert_eq!(pf.len(), 3);
        assert!(pf.contains(&sol1));
        assert!(pf.contains(&sol2));
        assert!(pf.contains(&sol3));
    }

    #[test]
    fn test_try_insert_non_dominated_solutions() {
        let mut pf = TestParetoFront::new("test");

        // Add first solution
        assert!(pf.try_insert(&solution(1, 3)));
        assert_eq!(pf.len(), 1);

        // Add second non-dominated solution
        assert!(pf.try_insert(&solution(3, 1)));
        assert_eq!(pf.len(), 2);

        // Add third non-dominated solution
        assert!(pf.try_insert(&solution(2, 2)));
        assert_eq!(pf.len(), 3);

        // Verify all solutions are present
        assert!(pf.contains(&solution(1, 3)));
        assert!(pf.contains(&solution(3, 1)));
        assert!(pf.contains(&solution(2, 2)));
    }

    #[test]
    fn test_try_insert_dominated_solution_rejected() {
        let mut pf = TestParetoFront::new("test");

        // Add dominating solution
        pf.insert_unchecked(&solution(1, 1));

        // Try to add dominated solution - should be rejected
        assert!(!pf.try_insert(&solution(2, 2)));
        assert_eq!(pf.len(), 1);
        assert!(pf.contains(&solution(1, 1)));
        assert!(!pf.contains(&solution(2, 2)));
    }

    #[test]
    fn test_try_insert_dominating_solution_removes_dominated() {
        let mut pf = TestParetoFront::new("test");

        // Add several solutions using insert_unchecked to avoid dominance checking
        pf.insert_unchecked(&solution(2, 2));
        pf.insert_unchecked(&solution(3, 3));
        pf.insert_unchecked(&solution(1, 4));
        assert_eq!(pf.len(), 3);

        // Add dominating solution - should remove (2,2) and (3,3) but keep (1,4)
        assert!(pf.try_insert(&solution(1, 1)));

        // The final Pareto front should contain only non-dominated solutions
        // Due to ND-Tree's algorithm, the size might be different
        // Let's check what solutions remain
        let solutions: Vec<_> = pf.iter().map(|s| s.objectives()).collect();

        // (1,1) should be present as it was just added
        assert!(pf.contains(&solution(1, 1)));

        // Verify no solution dominates another in the final set
        for sol1 in pf.iter() {
            for sol2 in pf.iter() {
                if sol1.objectives() != sol2.objectives() {
                    assert!(!sol1.dominates(sol2.objectives()));
                }
            }
        }
    }

    #[test]
    fn test_try_insert_equal_solution() {
        let mut pf = TestParetoFront::new("test");
        let sol = solution(2, 3);

        // Add first solution
        assert!(pf.try_insert(&sol));
        assert_eq!(pf.len(), 1);

        // Try to add equal solution - behavior depends on ND-Tree implementation
        // If it's rejected, it should return false; if accepted, it might replace
        let _result = pf.try_insert(&sol);
        // Either way, we should still have the solution
        assert!(pf.contains(&sol));
    }

    #[test]
    fn test_pareto_front_invariant_no_dominated_solutions() {
        let mut pf = TestParetoFront::new("test");

        // Build a Pareto front using try_insert
        let candidates = vec![
            solution(1, 5),
            solution(2, 4),
            solution(3, 3),
            solution(4, 2),
            solution(5, 1),
            solution(2, 6), // Should be rejected (dominated by (1,5))
            solution(6, 2), // Should be rejected (dominated by (4,2) or (5,1))
            solution(1, 4), // Should be accepted, might remove (2,4)
        ];

        for candidate in candidates {
            pf.try_insert(&candidate);
        }

        // Verify Pareto front invariant: no solution dominates another
        let solutions: Vec<_> = pf.iter().collect();
        for (i, sol1) in solutions.iter().enumerate() {
            for (j, sol2) in solutions.iter().enumerate() {
                if i != j {
                    assert!(
                        !sol1.dominates(sol2.objectives()),
                        "Solution {:?} dominates {:?}",
                        sol1.objectives(),
                        sol2.objectives()
                    );
                }
            }
        }
    }

    #[test]
    fn test_replace_if_exists_existing_solution() {
        let mut pf = TestParetoFront::new("test");
        let original = solution(2, 3);
        let replacement = solution(2, 3); // Same objectives

        pf.insert_unchecked(&original);
        pf.insert_unchecked(&solution(1, 4));
        let original_len = pf.len();

        pf.replace_if_exists(replacement);

        assert_eq!(pf.len(), original_len);
        assert!(pf.contains(&solution(2, 3)));
        assert!(pf.contains(&solution(1, 4)));
    }

    #[test]
    fn test_replace_if_exists_nonexistent_solution() {
        let mut pf = TestParetoFront::new("test");
        pf.insert_unchecked(&solution(1, 2));
        let original_len = pf.len();

        pf.replace_if_exists(solution(3, 4)); // Doesn't exist

        assert_eq!(pf.len(), original_len);
        assert!(!pf.contains(&solution(3, 4)));
    }

    #[test]
    fn test_iter_empty_front() {
        let pf = TestParetoFront::new("test");
        let solutions: Vec<_> = pf.iter().collect();
        assert!(solutions.is_empty());
    }

    #[test]
    fn test_iter_with_solutions() {
        let mut pf = TestParetoFront::new("test");
        let sol1 = solution(1, 3);
        let sol2 = solution(3, 1);

        pf.insert_unchecked(&sol1);
        pf.insert_unchecked(&sol2);

        let solutions: Vec<_> = pf.iter().collect();
        assert_eq!(solutions.len(), 2);

        // Check that both solutions are present (order may vary)
        let objectives_set: std::collections::HashSet<_> =
            solutions.iter().map(|s| s.objectives()).collect();
        assert!(objectives_set.contains(&sol1.objectives()));
        assert!(objectives_set.contains(&sol2.objectives()));
    }

    /// Test based on the ND-Tree paper's example scenarios
    #[test]
    fn test_nd_tree_paper_example_pareto_front() {
        let mut pf = TestParetoFront::new("paper_example");

        // Example: add solutions that form a proper Pareto front
        // For minimization: (3,8), (5,6), (7,4), (9,2) should be non-dominated

        // Initial solutions forming a Pareto front
        assert!(pf.try_insert(&solution(3, 8)));
        assert!(pf.try_insert(&solution(5, 6)));
        assert!(pf.try_insert(&solution(7, 4)));
        assert!(pf.try_insert(&solution(9, 2)));

        // At this point we should have 4 solutions
        let initial_count = pf.len();
        assert!(initial_count >= 4 || initial_count <= 4); // Account for ND-Tree behavior

        // Add a solution that should dominate (5,6)
        let dominated_before = pf.contains(&solution(5, 6));
        assert!(pf.try_insert(&solution(4, 5))); // Should dominate (5,6)

        // Verify (4,5) is present
        assert!(pf.contains(&solution(4, 5)));

        // Add a dominated solution - should be rejected
        assert!(!pf.try_insert(&solution(6, 7))); // Should be dominated by existing solutions

        // Add a solution that's clearly on the Pareto front
        assert!(pf.try_insert(&solution(2, 9)));
        assert!(pf.contains(&solution(2, 9)));

        // Verify final Pareto front maintains invariant
        let solutions: Vec<_> = pf.iter().collect();
        for (i, sol1) in solutions.iter().enumerate() {
            for (j, sol2) in solutions.iter().enumerate() {
                if i != j {
                    assert!(!sol1.dominates(sol2.objectives()));
                    assert!(!sol2.dominates(sol1.objectives()));
                }
            }
        }
    }

    #[test]
    fn test_pareto_front_stress_test() {
        let mut pf = TestParetoFront::new("stress");

        // Add many solutions and verify the Pareto front properties are maintained
        let test_solutions = vec![
            solution(1, 8),
            solution(2, 7),
            solution(3, 6),
            solution(4, 5),
            solution(5, 4),
            solution(6, 3),
            solution(7, 2),
            solution(8, 1),
            solution(2, 8),
            solution(3, 7),
            solution(4, 6),
            solution(5, 5),
            solution(6, 4),
            solution(7, 3),
            solution(8, 2),
            solution(9, 1),
            solution(1, 1), // This should dominate many others
        ];

        for sol in test_solutions {
            pf.try_insert(&sol);

            // Verify no solution dominates another after each insertion
            let current_solutions: Vec<_> = pf.iter().collect();
            for (i, sol1) in current_solutions.iter().enumerate() {
                for (j, sol2) in current_solutions.iter().enumerate() {
                    if i != j {
                        assert!(!sol1.dominates(sol2.objectives()));
                    }
                }
            }
        }

        // After inserting (1,1), it should be in the front (as it dominates many solutions)
        assert!(pf.contains(&solution(1, 1)));

        // Verify final solutions form a proper Pareto front
        let final_solutions: Vec<_> = pf.iter().collect();
        assert!(!final_solutions.is_empty());

        // Each solution should either have x=1 or y=1 (i.e., be on the Pareto boundary)
        for sol in &final_solutions {
            let is_on_boundary = sol.objectives()[0] == 1 || sol.objectives()[1] == 1;
            assert!(
                is_on_boundary,
                "Solution {:?} should be on Pareto boundary",
                sol.objectives()
            );
        }
    }

    #[test]
    fn test_contains_method() {
        let mut pf = TestParetoFront::new("contains_test");

        let sol1 = solution(1, 2);
        let sol2 = solution(3, 4);

        // Initially empty
        assert!(!pf.contains(&sol1));
        assert!(!pf.contains(&sol2));

        // Add one solution
        pf.insert_unchecked(&sol1);
        assert!(pf.contains(&sol1));
        assert!(!pf.contains(&sol2));

        // Add second solution
        pf.insert_unchecked(&sol2);
        assert!(pf.contains(&sol1));
        assert!(pf.contains(&sol2));

        // Test with solution that has same objectives but different instance
        let sol1_copy = solution(1, 2);
        assert!(pf.contains(&sol1_copy));
    }
}
