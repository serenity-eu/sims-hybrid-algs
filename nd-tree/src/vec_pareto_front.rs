use pareto::{HasObjectives, MoSolution, ParetoFront};
use std::fmt::Debug;

/// Vec-based Pareto front implementation for reference and testing
#[derive(Debug, Clone)]
pub struct VecParetoFront<T, const D: usize> {
    name: &'static str,
    solutions: Vec<T>,
}

impl<T, const D: usize> Default for VecParetoFront<T, D> {
    fn default() -> Self {
        Self::new("default")
    }
}

impl<T, const D: usize> VecParetoFront<T, D> {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            solutions: Vec::new(),
        }
    }

    pub fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }

    /// Create from existing solutions (useful for testing)
    pub fn from_solutions(name: &'static str, solutions: Vec<T>) -> Self
    where
        T: HasObjectives<D> + MoSolution<D> + Clone + Debug,
    {
        let mut pf = Self::new(name);
        for solution in solutions {
            pf.insert_unchecked(&solution);
        }
        pf
    }
}

impl<'a, T, const D: usize> ParetoFront<'a, T> for VecParetoFront<T, D>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + Debug,
{
    type Iter<'b>
        = std::slice::Iter<'b, T>
    where
        Self: 'b;
    type IntoIter = std::vec::IntoIter<T>;

    fn new(name: &'static str) -> Self {
        Self::new(name)
    }

    fn with_name(self, name: &'static str) -> Self {
        self.with_name(name)
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.solutions.iter()
    }

    fn contains(&self, solution: &T) -> bool {
        let target_objectives = solution.objectives();
        self.solutions
            .iter()
            .any(|s| s.objectives() == target_objectives)
    }

    fn try_insert(&mut self, new_solution: &T) -> bool {
        let new_objectives = new_solution.objectives();

        // Check if new solution is dominated by any existing solution or is equal
        for existing in &self.solutions {
            if existing.dominates(new_objectives) || existing.objectives() == new_objectives {
                return false; // New solution is dominated or equal, reject it
            }
        }

        // Remove all solutions dominated by the new solution
        self.solutions
            .retain(|existing| !new_solution.dominates(existing.objectives()));

        // Add the new solution
        self.solutions.push(new_solution.clone());
        true
    }

    fn insert_unchecked(&mut self, solution: &T) {
        self.solutions.push(solution.clone());
    }

    fn replace_if_exists(&mut self, solution: T) {
        let target_objectives = solution.objectives();

        // Find existing solution with same objectives
        if let Some(pos) = self
            .solutions
            .iter()
            .position(|s| s.objectives() == target_objectives)
        {
            self.solutions[pos] = solution;
        }
    }

    fn len(&self) -> usize {
        self.solutions.len()
    }

    fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }
}

impl<T, const D: usize> VecParetoFront<T, D>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + Debug,
{
    /// Check if the Pareto front satisfies all invariants
    pub fn validate_pareto_invariants(&self) -> Result<(), String> {
        for (i, sol1) in self.solutions.iter().enumerate() {
            for (j, sol2) in self.solutions.iter().enumerate() {
                if i != j && sol1.dominates(sol2.objectives()) {
                    return Err(format!(
                        "Solution at index {} dominates solution at index {}: {:?} dominates {:?}",
                        i,
                        j,
                        sol1.objectives(),
                        sol2.objectives()
                    ));
                }
            }
        }
        Ok(())
    }

    /// Get all solutions as a Vec (for testing purposes)
    pub fn solutions(&self) -> &Vec<T> {
        &self.solutions
    }

    /// Clear all solutions
    pub fn clear(&mut self) {
        self.solutions.clear();
    }

    /// Get a mutable reference to solutions (for advanced operations)
    pub fn solutions_mut(&mut self) -> &mut Vec<T> {
        &mut self.solutions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct TestSolution {
        objectives: [u64; 2],
        id: u32, // Additional field to test solution identity
    }

    impl HasObjectives<2> for TestSolution {
        fn objectives(&self) -> &[u64; 2] {
            &self.objectives
        }
    }

    impl MoSolution<2> for TestSolution {}

    fn solution(x: u64, y: u64) -> TestSolution {
        TestSolution {
            objectives: [x, y],
            id: (x
                .saturating_mul(1000)
                .saturating_add(y)
                .min(u32::MAX as u64)) as u32, // Unique ID based on objectives
        }
    }

    fn solution_with_id(x: u64, y: u64, id: u32) -> TestSolution {
        TestSolution {
            objectives: [x, y],
            id,
        }
    }

    // Helper function to create 3D test solutions
    #[derive(Debug, Clone, PartialEq)]
    struct TestSolution3D {
        objectives: [u64; 3],
        id: u32,
    }

    impl HasObjectives<3> for TestSolution3D {
        fn objectives(&self) -> &[u64; 3] {
            &self.objectives
        }
    }

    impl MoSolution<3> for TestSolution3D {}

    fn solution_3d(x: u64, y: u64, z: u64) -> TestSolution3D {
        TestSolution3D {
            objectives: [x, y, z],
            id: (x
                .saturating_mul(1000000)
                .saturating_add(y.saturating_mul(1000))
                .saturating_add(z)
                .min(u32::MAX as u64)) as u32,
        }
    }

    mod creation_and_basic_operations {
        use super::*;

        #[test]
        fn test_new_empty_front() {
            let pf = VecParetoFront::<TestSolution, 2>::new("test");
            assert_eq!(pf.len(), 0);
            assert!(pf.is_empty());
            assert_eq!(pf.iter().count(), 0);
        }

        #[test]
        fn test_default_construction() {
            let pf = VecParetoFront::<TestSolution, 2>::default();
            assert_eq!(pf.len(), 0);
            assert!(pf.is_empty());
        }

        #[test]
        fn test_with_name() {
            let pf = VecParetoFront::<TestSolution, 2>::new("original").with_name("renamed");
            assert_eq!(pf.len(), 0);
        }

        #[test]
        fn test_from_solutions() {
            let solutions = vec![solution(1, 5), solution(2, 4), solution(3, 3)];
            let pf = VecParetoFront::from_solutions("test", solutions);
            assert_eq!(pf.len(), 3);
        }

        #[test]
        fn test_clear() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("test");
            pf.try_insert(&solution(1, 2));
            pf.try_insert(&solution(2, 1));
            assert_eq!(pf.len(), 2);

            pf.clear();
            assert_eq!(pf.len(), 0);
            assert!(pf.is_empty());
        }
    }

    mod insertion_and_dominance {
        use super::*;

        #[test]
        fn test_single_solution_insertion() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("test");
            let sol = solution(2, 3);

            assert!(pf.try_insert(&sol));
            assert_eq!(pf.len(), 1);
            assert!(pf.contains(&sol));
        }

        #[test]
        fn test_dominated_solution_rejection() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("test");

            // Insert first solution
            assert!(pf.try_insert(&solution(2, 3)));
            assert_eq!(pf.len(), 1);

            // Try to insert dominated solutions
            assert!(!pf.try_insert(&solution(3, 4)));
            assert!(!pf.try_insert(&solution(2, 4)));
            assert!(!pf.try_insert(&solution(3, 3)));
            assert_eq!(pf.len(), 1);
        }

        #[test]
        fn test_dominating_solution_removes_existing() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("test");

            // Insert initial solution
            assert!(pf.try_insert(&solution(3, 3)));
            assert_eq!(pf.len(), 1);

            // Insert dominating solution
            assert!(pf.try_insert(&solution(2, 2)));
            assert_eq!(pf.len(), 1);

            // Verify only the dominating solution remains
            let remaining: Vec<_> = pf.iter().cloned().collect();
            assert_eq!(remaining, vec![solution(2, 2)]);
        }

        #[test]
        fn test_non_dominated_solutions() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("test");

            // Insert multiple non-dominated solutions
            assert!(pf.try_insert(&solution(1, 5)));
            assert!(pf.try_insert(&solution(2, 4)));
            assert!(pf.try_insert(&solution(3, 3)));
            assert!(pf.try_insert(&solution(4, 2)));
            assert!(pf.try_insert(&solution(5, 1)));

            assert_eq!(pf.len(), 5);

            // Verify all solutions are present
            assert!(pf.contains(&solution(1, 5)));
            assert!(pf.contains(&solution(2, 4)));
            assert!(pf.contains(&solution(3, 3)));
            assert!(pf.contains(&solution(4, 2)));
            assert!(pf.contains(&solution(5, 1)));
        }

        #[test]
        fn test_equal_solutions() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("test");

            let sol1 = solution_with_id(2, 3, 1);
            let sol2 = solution_with_id(2, 3, 2); // Same objectives, different ID

            assert!(pf.try_insert(&sol1));
            assert_eq!(pf.len(), 1);

            // Equal solutions should be rejected (they have same objectives)
            assert!(!pf.try_insert(&sol2));
            assert_eq!(pf.len(), 1);
        }

        #[test]
        fn test_multiple_dominated_removal() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("test");

            // Insert multiple solutions
            pf.try_insert(&solution(4, 4));
            pf.try_insert(&solution(3, 5));
            pf.try_insert(&solution(5, 3));
            // Note: (6,6) would be dominated by (4,4), so we use a non-dominated solution
            pf.try_insert(&solution(6, 2)); // This is not dominated by others
            assert_eq!(pf.len(), 4);

            // Insert solution that dominates multiple existing ones
            assert!(pf.try_insert(&solution(2, 2)));
            assert_eq!(pf.len(), 1);

            let remaining: Vec<_> = pf.iter().cloned().collect();
            assert_eq!(remaining, vec![solution(2, 2)]);
        }
    }

    mod edge_cases {
        use super::*;

        #[test]
        fn test_boundary_dominance() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("test");

            // Test boundary cases where one objective is equal
            pf.try_insert(&solution(2, 3));

            // These should be rejected (dominated)
            assert!(!pf.try_insert(&solution(2, 4))); // Equal x, worse y
            assert!(!pf.try_insert(&solution(3, 3))); // Worse x, equal y
            assert!(!pf.try_insert(&solution(3, 4))); // Worse x, worse y

            // These should be accepted (non-dominated)
            assert!(pf.try_insert(&solution(1, 4))); // Better x, worse y
            assert!(pf.try_insert(&solution(3, 2))); // Worse x, better y
            assert!(pf.try_insert(&solution(1, 2))); // Better x, better y

            assert_eq!(pf.len(), 1); // Only (1,2) should remain as it dominates all others
        }

        #[test]
        fn test_zero_objectives() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("test");

            pf.try_insert(&solution(0, 5));
            pf.try_insert(&solution(5, 0));
            pf.try_insert(&solution(0, 0));

            assert_eq!(pf.len(), 1); // (0,0) should dominate others
            let remaining: Vec<_> = pf.iter().cloned().collect();
            assert_eq!(remaining, vec![solution(0, 0)]);
        }

        #[test]
        fn test_large_values() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("test");

            let max_val = u64::MAX;
            pf.try_insert(&solution(max_val, 1));
            pf.try_insert(&solution(1, max_val));
            pf.try_insert(&solution(max_val - 1, max_val - 1));

            assert_eq!(pf.len(), 3); // All should be non-dominated
        }

        #[test]
        fn test_insert_unchecked_bypasses_dominance() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("test");

            pf.try_insert(&solution(1, 1));
            assert_eq!(pf.len(), 1);

            // This would normally be rejected, but insert_unchecked allows it
            pf.insert_unchecked(&solution(2, 2));
            assert_eq!(pf.len(), 2);

            // Verify invariants are broken
            assert!(pf.validate_pareto_invariants().is_err());
        }
    }

    mod three_dimensional_tests {
        use super::*;

        #[test]
        fn test_3d_pareto_front() {
            let mut pf = VecParetoFront::<TestSolution3D, 3>::new("test_3d");

            // Insert 3D solutions
            assert!(pf.try_insert(&solution_3d(1, 5, 3)));
            assert!(pf.try_insert(&solution_3d(2, 4, 2)));
            assert!(pf.try_insert(&solution_3d(3, 3, 1)));
            assert!(pf.try_insert(&solution_3d(5, 1, 5)));

            assert_eq!(pf.len(), 4);

            // Insert dominated 3D solution
            assert!(!pf.try_insert(&solution_3d(2, 5, 3)));
            assert_eq!(pf.len(), 4);

            // Insert dominating 3D solution
            assert!(pf.try_insert(&solution_3d(1, 1, 1)));
            assert_eq!(pf.len(), 1);
        }

        #[test]
        fn test_3d_partial_dominance() {
            let mut pf = VecParetoFront::<TestSolution3D, 3>::new("test_3d");

            pf.try_insert(&solution_3d(2, 2, 2));

            // Partially dominated solutions (better in some objectives, worse in others)
            assert!(pf.try_insert(&solution_3d(1, 3, 3))); // Better x, worse y,z
            assert!(pf.try_insert(&solution_3d(3, 1, 3))); // Worse x, better y, worse z
            assert!(pf.try_insert(&solution_3d(3, 3, 1))); // Worse x,y, better z

            assert_eq!(pf.len(), 4); // All should coexist
        }
    }

    mod iteration_and_access {
        use super::*;

        #[test]
        fn test_iterator() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("test");

            let solutions = vec![solution(1, 3), solution(2, 2), solution(3, 1)];
            for sol in &solutions {
                pf.try_insert(sol);
            }

            let collected: Vec<_> = pf.iter().cloned().collect();
            assert_eq!(collected.len(), 3);

            // Verify all original solutions are present
            for sol in solutions {
                assert!(collected.contains(&sol));
            }
        }

        #[test]
        fn test_contains_method() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("test");

            let sol1 = solution(2, 3);
            let sol2 = solution(1, 4);
            let sol3 = solution_with_id(2, 3, 999); // Same objectives as sol1

            pf.try_insert(&sol1);

            assert!(pf.contains(&sol1));
            assert!(!pf.contains(&sol2));
            assert!(pf.contains(&sol3)); // Should match based on objectives
        }

        #[test]
        fn test_solutions_access() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("test");

            pf.try_insert(&solution(1, 3));
            pf.try_insert(&solution(2, 2));

            let solutions = pf.solutions();
            assert_eq!(solutions.len(), 2);

            // Test mutable access
            let solutions_mut = pf.solutions_mut();
            assert_eq!(solutions_mut.len(), 2);
        }

        #[test]
        fn test_replace_if_exists() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("test");

            let original = solution_with_id(2, 3, 1);
            let replacement = solution_with_id(2, 3, 2); // Same objectives, different ID
            let unrelated = solution_with_id(1, 4, 3);

            pf.try_insert(&original);
            pf.try_insert(&unrelated);
            assert_eq!(pf.len(), 2);

            // Replace existing solution
            pf.replace_if_exists(replacement.clone());
            assert_eq!(pf.len(), 2);
            assert!(pf.contains(&replacement));
            assert!(!pf.solutions().contains(&original));

            // Try to replace non-existing solution
            let non_existing = solution_with_id(5, 5, 4);
            pf.replace_if_exists(non_existing.clone());
            assert_eq!(pf.len(), 2); // Should remain unchanged
            assert!(!pf.contains(&non_existing));
        }
    }

    mod stress_and_performance {
        use super::*;

        #[test]
        fn test_large_pareto_front() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("stress_test");

            // Insert many non-dominated solutions
            for i in 1..=100 {
                let sol = solution(i, 101 - i);
                assert!(pf.try_insert(&sol));
            }

            assert_eq!(pf.len(), 100);
            assert!(pf.validate_pareto_invariants().is_ok());
        }

        #[test]
        fn test_dominated_solution_cascade() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("cascade_test");

            // Insert many solutions that will be dominated
            for i in 10..=20 {
                for j in 10..=20 {
                    pf.try_insert(&solution(i, j));
                }
            }

            let initial_count = pf.len();
            assert!(initial_count > 0);

            // Insert a solution that dominates all previous ones
            assert!(pf.try_insert(&solution(5, 5)));
            assert_eq!(pf.len(), 1);

            let remaining: Vec<_> = pf.iter().cloned().collect();
            assert_eq!(remaining, vec![solution(5, 5)]);
        }

        #[test]
        fn test_mixed_insertions() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("mixed_test");

            // Mix of dominated, dominating, and non-dominated insertions
            let test_cases = vec![
                (solution(5, 5), true),  // First insertion
                (solution(6, 6), false), // Dominated
                (solution(4, 6), true),  // Non-dominated
                (solution(6, 4), true),  // Non-dominated
                (solution(3, 3), true),  // Dominates some, creates new front
                (solution(7, 7), false), // Dominated by (3,3)
                (solution(2, 8), true),  // Non-dominated
                (solution(8, 2), true),  // Non-dominated
                (solution(1, 1), true),  // Dominates all
            ];

            for (sol, should_insert) in test_cases {
                let result = pf.try_insert(&sol);
                assert_eq!(result, should_insert, "Failed for solution {:?}", sol);
            }

            // Final front should contain only (1,1)
            assert_eq!(pf.len(), 1);
            assert!(pf.validate_pareto_invariants().is_ok());
        }

        #[test]
        fn test_vec_specific_operations() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("vec_test");

            // Test Vec-specific characteristics
            pf.try_insert(&solution(1, 10));
            pf.try_insert(&solution(5, 5));
            pf.try_insert(&solution(10, 1));

            // Vec maintains insertion order for non-dominated solutions
            let solutions = pf.solutions();
            assert_eq!(solutions.len(), 3);

            // Access underlying Vec directly
            let vec_ref = pf.solutions();
            assert_eq!(vec_ref.len(), 3);

            // Test mutable access
            let vec_mut = pf.solutions_mut();
            let original_len = vec_mut.len();

            // Direct manipulation should be possible but not recommended
            // (this is just to test the API exists)
            assert_eq!(vec_mut.len(), original_len);
        }
    }

    mod invariant_validation {
        use super::*;

        #[test]
        fn test_pareto_invariants_maintained() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("invariant_test");

            // Build a complex Pareto front
            let solutions = vec![
                solution(1, 10),
                solution(2, 8),
                solution(3, 6),
                solution(4, 5),
                solution(5, 4),
                solution(6, 3),
                solution(8, 2),
                solution(10, 1),
            ];

            for sol in solutions {
                pf.try_insert(&sol);
            }

            // All insertions should maintain Pareto invariants
            assert!(pf.validate_pareto_invariants().is_ok());

            // Verify no solution dominates another
            let solutions = pf.solutions();
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
        fn test_invariant_violation_detection() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("violation_test");

            // Start with a valid front
            pf.try_insert(&solution(1, 5));
            pf.try_insert(&solution(3, 3));
            pf.try_insert(&solution(5, 1));
            assert!(pf.validate_pareto_invariants().is_ok());

            // Force an invariant violation using insert_unchecked
            pf.insert_unchecked(&solution(2, 2)); // This dominates (3,3)
            assert!(pf.validate_pareto_invariants().is_err());
        }

        #[test]
        fn test_comprehensive_dominance_patterns() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("dominance_test");

            // Create a scenario with complex dominance relationships
            let initial_solutions = vec![
                solution(2, 8),
                solution(3, 6),
                solution(4, 5),
                solution(5, 4),
                solution(6, 3),
                solution(8, 2),
            ];

            for sol in initial_solutions {
                pf.try_insert(&sol);
            }

            assert_eq!(pf.len(), 6);
            assert!(pf.validate_pareto_invariants().is_ok());

            // Insert a solution that should remove some but not all
            assert!(pf.try_insert(&solution(3, 4))); // Should remove (3,6), (4,5), (5,4)

            // Verify expected solutions remain
            assert!(pf.contains(&solution(2, 8)));
            assert!(pf.contains(&solution(3, 4)));
            assert!(pf.contains(&solution(6, 3)));
            assert!(pf.contains(&solution(8, 2)));
            assert_eq!(pf.len(), 4);
            assert!(pf.validate_pareto_invariants().is_ok());
        }
    }

    mod advanced_edge_cases {
        use super::*;

        #[test]
        fn test_single_objective_optimization() {
            // Test with dimension 1 - essentially a simple min/max problem
            #[derive(Debug, Clone, PartialEq)]
            struct SimpleSolution {
                objectives: [u64; 1],
                id: u32,
            }

            impl HasObjectives<1> for SimpleSolution {
                fn objectives(&self) -> &[u64; 1] {
                    &self.objectives
                }
            }

            impl MoSolution<1> for SimpleSolution {}

            let mut pf = VecParetoFront::<SimpleSolution, 1>::new("1d_test");

            // In 1D, only the minimum value should remain
            assert!(pf.try_insert(&SimpleSolution {
                objectives: [5],
                id: 1
            }));
            assert!(pf.try_insert(&SimpleSolution {
                objectives: [3],
                id: 2
            }));
            assert!(!pf.try_insert(&SimpleSolution {
                objectives: [7],
                id: 3
            })); // Should be dominated by [3]
            assert!(pf.try_insert(&SimpleSolution {
                objectives: [1],
                id: 4
            }));

            assert_eq!(pf.len(), 1);
            assert_eq!(pf.solutions()[0].objectives[0], 1);
        }

        #[test]
        fn test_pathological_dominance_patterns() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("pathological_test");

            // Create a staircase pattern where each solution improves one objective
            let staircase = vec![
                solution(10, 1),
                solution(9, 2),
                solution(8, 3),
                solution(7, 4),
                solution(6, 5),
                solution(5, 6),
                solution(4, 7),
                solution(3, 8),
                solution(2, 9),
                solution(1, 10),
            ];

            for sol in &staircase {
                assert!(pf.try_insert(sol));
            }

            assert_eq!(pf.len(), 10);
            assert!(pf.validate_pareto_invariants().is_ok());

            // Now insert a solution that creates a "corner" and removes several solutions
            assert!(pf.try_insert(&solution(4, 4)));

            // Should remove (4,7), (5,6), (6,5), (7,4) but keep others
            assert_eq!(pf.len(), 7);
            assert!(pf.validate_pareto_invariants().is_ok());
        }

        #[test]
        fn test_mass_deletion_scenario() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("mass_deletion_test");

            // Create a large dominated region
            for i in 10..20 {
                for j in 10..20 {
                    pf.try_insert(&solution(i, j));
                }
            }

            let initial_size = pf.len();
            assert!(initial_size > 0);

            // Insert a solution that dominates everything
            assert!(pf.try_insert(&solution(5, 5)));
            assert_eq!(pf.len(), 1);

            // Verify the dominating solution is the only one left
            assert!(pf.contains(&solution(5, 5)));
        }

        #[test]
        fn test_precision_edge_cases() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("precision_test");

            // Test with very small differences
            pf.try_insert(&solution(1000, 1000));
            pf.try_insert(&solution(1001, 999)); // Should be accepted
            pf.try_insert(&solution(999, 1001)); // Should be accepted
            pf.try_insert(&solution(1000, 999)); // Should dominate (1000, 1000)

            assert_eq!(pf.len(), 2); // (1000,999) and (999,1001) should remain
            assert!(pf.contains(&solution(1000, 999)));
            assert!(pf.contains(&solution(999, 1001)));
            assert!(!pf.contains(&solution(1000, 1000)));
        }

        #[test]
        fn test_repeated_insertion_patterns() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("repeated_test");

            // Insert the same solution multiple times
            let sol = solution(5, 5);
            assert!(pf.try_insert(&sol));
            assert!(!pf.try_insert(&sol)); // Should be rejected
            assert!(!pf.try_insert(&sol)); // Should be rejected again
            assert_eq!(pf.len(), 1);

            // Insert equivalent solutions (same objectives, different IDs)
            let equiv1 = solution_with_id(5, 5, 999);
            let equiv2 = solution_with_id(5, 5, 888);
            assert!(!pf.try_insert(&equiv1));
            assert!(!pf.try_insert(&equiv2));
            assert_eq!(pf.len(), 1);
        }

        #[test]
        fn test_alternating_dominance_insertions() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("alternating_test");

            // Alternately insert dominating and dominated solutions
            assert!(pf.try_insert(&solution(10, 10))); // First solution
            assert!(!pf.try_insert(&solution(15, 15))); // Dominated
            assert!(pf.try_insert(&solution(5, 15))); // Non-dominated
            assert!(!pf.try_insert(&solution(12, 12))); // Dominated by (10,10)
            assert!(pf.try_insert(&solution(15, 5))); // Non-dominated
            assert!(pf.try_insert(&solution(3, 20))); // Non-dominated
            assert!(pf.try_insert(&solution(20, 3))); // Non-dominated
            assert!(pf.try_insert(&solution(1, 1))); // Dominates all others

            // Only (1,1) should remain
            assert_eq!(pf.len(), 1);
            assert!(pf.contains(&solution(1, 1)));
        }
    }

    mod property_based_tests {
        use super::*;

        #[test]
        fn test_monotonic_front_property() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("monotonic_test");

            // Build a proper Pareto front and verify it has the monotonic property
            let solutions = vec![
                solution(1, 20),
                solution(2, 18),
                solution(3, 15),
                solution(5, 12),
                solution(8, 10),
                solution(12, 8),
                solution(15, 5),
                solution(18, 3),
                solution(20, 1),
            ];

            for sol in solutions {
                pf.try_insert(&sol);
            }

            // Verify the monotonic property: if we sort by first objective,
            // the second objective should be decreasing
            let mut front: Vec<_> = pf.solutions().clone();
            front.sort_by_key(|s| s.objectives[0]);

            for i in 1..front.len() {
                assert!(front[i - 1].objectives[0] <= front[i].objectives[0]);
                assert!(front[i - 1].objectives[1] >= front[i].objectives[1]);
            }
        }

        #[test]
        fn test_pareto_front_completeness() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("completeness_test");

            // Insert a set of solutions and verify that the resulting front
            // is complete (no non-dominated solution is missing)
            let candidates = vec![
                solution(1, 10),
                solution(2, 9),
                solution(3, 8),
                solution(4, 7),
                solution(2, 8),
                solution(3, 6),
                solution(5, 5),
                solution(6, 4),
                solution(7, 3),
                solution(8, 2),
                solution(9, 1),
                solution(10, 1),
            ];

            for sol in &candidates {
                pf.try_insert(sol);
            }

            // Verify that no candidate dominates any solution in the front
            for candidate in &candidates {
                let mut candidate_dominates_front_member = false;
                for front_member in pf.solutions() {
                    if candidate.dominates(front_member.objectives()) {
                        candidate_dominates_front_member = true;
                        break;
                    }
                }

                // If the candidate dominates a front member, it should be in the front
                if candidate_dominates_front_member {
                    assert!(
                        pf.contains(candidate),
                        "Solution {:?} dominates front members but is not in front",
                        candidate
                    );
                }
            }
        }

        #[test]
        fn test_front_stability_under_reinsertions() {
            let mut pf = VecParetoFront::<TestSolution, 2>::new("stability_test");

            // Build initial front
            let initial_solutions = vec![
                solution(1, 8),
                solution(2, 6),
                solution(3, 4),
                solution(4, 3),
                solution(6, 2),
                solution(8, 1),
            ];

            for sol in &initial_solutions {
                pf.try_insert(sol);
            }

            let initial_front: Vec<_> = pf.solutions().clone();
            assert_eq!(initial_front.len(), 6);

            // Re-insert all solutions (should have no effect)
            for sol in &initial_solutions {
                assert!(!pf.try_insert(sol)); // Should be rejected
            }

            // Front should be unchanged
            assert_eq!(pf.solutions().len(), initial_front.len());
            for sol in &initial_front {
                assert!(pf.contains(sol));
            }
        }
    }

    mod cross_validation_tests {
        use super::*;
        use crate::linkedlist_pareto_front::LinkedListParetoFront;

        #[test]
        fn test_vec_vs_linkedlist_consistency() {
            let mut vec_pf = VecParetoFront::<TestSolution, 2>::new("vec_test");
            let mut list_pf = LinkedListParetoFront::<TestSolution, 2>::new("list_test");

            // Test with a complex set of solutions
            let test_solutions = vec![
                solution(1, 20),
                solution(2, 18),
                solution(5, 15),
                solution(8, 12),
                solution(12, 8),
                solution(15, 5),
                solution(18, 2),
                solution(20, 1),
                solution(3, 3),
                solution(10, 10),
                solution(7, 7),
                solution(4, 16),
                solution(16, 4),
                solution(6, 14),
                solution(14, 6),
                solution(9, 11),
                solution(11, 9),
                solution(13, 7),
                solution(7, 13),
            ];

            // Insert all solutions into both implementations
            let mut vec_results = Vec::new();
            let mut list_results = Vec::new();

            for sol in &test_solutions {
                vec_results.push(vec_pf.try_insert(sol));
                list_results.push(list_pf.try_insert(sol));
            }

            // Both should have the same insertion results
            assert_eq!(
                vec_results, list_results,
                "Insertion results differ between Vec and LinkedList"
            );

            // Both should have the same final size
            assert_eq!(
                vec_pf.len(),
                list_pf.len(),
                "Final sizes differ between Vec and LinkedList"
            );

            // Both should contain the same set of solutions (regardless of order)
            let mut vec_solutions: Vec<_> = vec_pf.iter().cloned().collect();
            let mut list_solutions: Vec<_> = list_pf.iter().cloned().collect();

            vec_solutions.sort_by_key(|s| (s.objectives[0], s.objectives[1]));
            list_solutions.sort_by_key(|s| (s.objectives[0], s.objectives[1]));

            assert_eq!(
                vec_solutions, list_solutions,
                "Final solution sets differ between Vec and LinkedList"
            );

            // Both should satisfy Pareto invariants
            assert!(vec_pf.validate_pareto_invariants().is_ok());
            assert!(list_pf.validate_pareto_invariants().is_ok());
        }

        #[test]
        fn test_implementation_consistency_with_random_operations() {
            let mut vec_pf = VecParetoFront::<TestSolution, 2>::new("vec_random");
            let mut list_pf = LinkedListParetoFront::<TestSolution, 2>::new("list_random");

            // Test with a series of operations that should behave identically
            let operations = vec![
                solution(10, 10),
                solution(5, 15),
                solution(15, 5),
                solution(8, 12),
                solution(12, 8),
                solution(6, 14),
                solution(14, 6),
                solution(9, 11),
                solution(11, 9),
                solution(3, 3),
                solution(1, 20),
                solution(20, 1),
                solution(2, 18),
                solution(18, 2),
                solution(4, 16),
                solution(16, 4),
                solution(7, 13),
                solution(13, 7),
                solution(0, 25),
                solution(25, 0),
            ];

            for (i, sol) in operations.iter().enumerate() {
                let vec_result = vec_pf.try_insert(sol);
                let list_result = list_pf.try_insert(sol);

                assert_eq!(
                    vec_result, list_result,
                    "Insertion result differs at step {}: Vec={}, List={}, Solution={:?}",
                    i, vec_result, list_result, sol
                );

                assert_eq!(
                    vec_pf.len(),
                    list_pf.len(),
                    "Sizes differ at step {}: Vec={}, List={}, after inserting {:?}",
                    i,
                    vec_pf.len(),
                    list_pf.len(),
                    sol
                );
            }

            // Final validation
            assert_eq!(vec_pf.len(), list_pf.len());
            assert!(vec_pf.validate_pareto_invariants().is_ok());
            assert!(list_pf.validate_pareto_invariants().is_ok());
        }

        #[test]
        fn test_implementation_consistency_with_3d_objectives() {
            let mut vec_pf = VecParetoFront::<TestSolution3D, 3>::new("vec_3d");
            let mut list_pf = LinkedListParetoFront::<TestSolution3D, 3>::new("list_3d");

            let test_solutions_3d = vec![
                solution_3d(1, 8, 5),
                solution_3d(2, 6, 7),
                solution_3d(3, 4, 9),
                solution_3d(4, 2, 11),
                solution_3d(5, 1, 12),
                solution_3d(6, 3, 8),
                solution_3d(7, 5, 6),
                solution_3d(8, 7, 4),
                solution_3d(9, 9, 2),
                solution_3d(10, 10, 1),
                solution_3d(0, 0, 15),
                solution_3d(0, 15, 0),
                solution_3d(15, 0, 0),
                solution_3d(2, 2, 2),
                solution_3d(5, 5, 5),
            ];

            for sol in &test_solutions_3d {
                let vec_result = vec_pf.try_insert(sol);
                let list_result = list_pf.try_insert(sol);
                assert_eq!(vec_result, list_result);
            }

            assert_eq!(vec_pf.len(), list_pf.len());

            // Verify both contain the same solutions
            for sol in vec_pf.iter() {
                assert!(
                    list_pf.contains(sol),
                    "Vec solution not found in LinkedList: {:?}",
                    sol
                );
            }
            for sol in list_pf.iter() {
                assert!(
                    vec_pf.contains(sol),
                    "LinkedList solution not found in Vec: {:?}",
                    sol
                );
            }
        }
    }
}
