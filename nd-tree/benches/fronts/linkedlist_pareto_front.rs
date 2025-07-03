#![expect(
    clippy::linkedlist,
    reason = "Using LinkedList for Pareto front as a reference implementation"
)]
#![allow(dead_code)]

use pareto::{HasObjectives, MoSolution, ParetoFront};
use std::collections::LinkedList;
use std::fmt::Debug;

/// LinkedList-based Pareto front implementation for reference and testing
#[derive(Debug, Clone)]
pub struct LinkedListParetoFront<T, const D: usize> {
    name: &'static str,
    solutions: LinkedList<T>,
}

impl<T, const D: usize> Default for LinkedListParetoFront<T, D> {
    fn default() -> Self {
        Self::new("default")
    }
}

impl<T, const D: usize> LinkedListParetoFront<T, D> {
    /// Creates a new, empty `LinkedListParetoFront`
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            solutions: LinkedList::new(),
        }
    }

    pub const fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }
}

impl<T, const D: usize> ParetoFront<'_, T> for LinkedListParetoFront<T, D>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + Debug,
{
    type Iter<'b>
        = std::collections::linked_list::Iter<'b, T>
    where
        Self: 'b,
        T: 'b;
    type IntoIter = std::collections::linked_list::IntoIter<T>;

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

    fn try_insert(&mut self, solution: &T) -> bool {
        let new_objectives = solution.objectives();

        // Check if new solution is dominated by any existing solution or is equal
        for existing in &self.solutions {
            if existing.covers(new_objectives) {
                return false; // New solution is dominated or equal, reject it
            }
        }

        // Remove all solutions dominated by the new solution using retain
        self.solutions
            .retain(|existing| !solution.dominates(existing.objectives()));

        // Add the new solution
        self.solutions.push_back(solution.clone());
        true
    }

    fn insert_unchecked(&mut self, solution: &T) {
        self.solutions.push_back(solution.clone());
    }

    fn replace_if_exists(&mut self, solution: T) {
        let target_objectives = solution.objectives();

        // Find existing solution with same objectives and replace it
        let mut cursor = self.solutions.cursor_front_mut();
        while let Some(current) = cursor.current() {
            if current.objectives() == target_objectives {
                *current = solution;
                return;
            }
            cursor.move_next();
        }
    }

    fn len(&self) -> usize {
        self.solutions.len()
    }

    fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }
}

impl<T, const D: usize> IntoIterator for LinkedListParetoFront<T, D>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + Debug,
{
    type Item = T;
    type IntoIter = std::collections::linked_list::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.solutions.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pareto::{HasObjectives, MoSolution};

    #[derive(Clone, Debug, PartialEq)]
    struct TestSolution<const D: usize> {
        objectives: [u64; D],
        id: u64,
    }

    impl<const D: usize> TestSolution<D> {
        #[must_use]
        fn new(objectives: [u64; D]) -> Self {
            Self { objectives, id: 0 }
        }
    }

    impl<const D: usize> HasObjectives<D> for TestSolution<D> {
        fn objectives(&self) -> &[u64; D] {
            &self.objectives
        }
    }

    impl<const D: usize> MoSolution<D> for TestSolution<D> {}

    fn solution(a: u64, b: u64) -> TestSolution<2> {
        TestSolution::new([a, b])
    }

    #[test]
    fn test_new() {
        let pf = LinkedListParetoFront::<TestSolution, 2>::new("test");
        assert_eq!(pf.len(), 0);
        assert!(pf.is_empty());
        assert_eq!(pf.iter().count(), 0);
    }

    #[test]
    fn test_default_construction() {
        let pf = LinkedListParetoFront::<TestSolution, 2>::default();
        assert_eq!(pf.len(), 0);
        assert!(pf.is_empty());
    }

    #[test]
    fn test_with_name() {
        let pf = LinkedListParetoFront::<TestSolution, 2>::new("original").with_name("renamed");
        assert_eq!(pf.len(), 0);
    }

    #[test]
    fn test_from_solutions() {
        let solutions = vec![solution(1, 5), solution(2, 4), solution(3, 3)];
        let pf = LinkedListParetoFront::from_solutions("test", solutions);
        assert_eq!(pf.len(), 3);
    }

    #[test]
    fn test_clear() {
        let mut pf = LinkedListParetoFront::<TestSolution, 2>::new("test");
        pf.try_insert(&solution(1, 2));
        pf.try_insert(&solution(2, 1));
        assert_eq!(pf.len(), 2);

        pf.clear();
        assert_eq!(pf.len(), 0);
        assert!(pf.is_empty());
    }

    #[test]
    fn test_try_insert_and_len() {
        let mut pf = LinkedListParetoFront::<TestSolution<2>, 2>::new("test");
        let sol1 = TestSolution {
            objectives: [1, 5],
            id: 0,
        };
        assert!(pf.try_insert(&sol1));
        assert_eq!(pf.len(), 1);
    }

    #[test]
    fn test_non_dominated_insertion() {
        let mut pf = LinkedListParetoFront::<TestSolution<2>, 2>::new("test");
        let sol1 = TestSolution {
            objectives: [1, 5],
            id: 0,
        };
        let sol2 = TestSolution {
            objectives: [2, 4],
            id: 0,
        };
        pf.try_insert(&sol1);
        pf.try_insert(&sol2);
        assert_eq!(pf.len(), 2);
    }

    #[test]
    fn test_dominated_insertion() {
        let mut pf = LinkedListParetoFront::<TestSolution<2>, 2>::new("test");
        let sol1 = TestSolution {
            objectives: [1, 5],
            id: 0,
        };
        let sol2 = TestSolution {
            objectives: [0, 4],
            id: 0,
        }; // Dominates sol1
        pf.try_insert(&sol1);
        assert!(!pf.try_insert(&sol2));
        assert_eq!(pf.len(), 1);
    }

    #[test]
    fn test_dominating_insertion() {
        let mut pf = LinkedListParetoFront::<TestSolution<2>, 2>::new("test");
        let sol1 = TestSolution {
            objectives: [1, 5],
            id: 0,
        };
        let sol2 = TestSolution {
            objectives: [2, 6],
            id: 0,
        }; // Is dominated by sol1
        pf.try_insert(&sol1);
        assert!(pf.try_insert(&sol2));
        assert_eq!(pf.len(), 1);
    }

    #[test]
    fn test_iter() {
        let mut pf = LinkedListParetoFront::<TestSolution<2>, 2>::new("test");
        let sol1 = TestSolution {
            objectives: [1, 5],
            id: 0,
        };
        let sol2 = TestSolution {
            objectives: [2, 4],
            id: 0,
        };
        pf.try_insert(&sol1);
        pf.try_insert(&sol2);
        let solutions: Vec<_> = pf.iter().cloned().collect();
        assert_eq!(solutions.len(), 2);
        assert!(solutions.contains(&sol1));
        assert!(solutions.contains(&sol2));
    }

    #[test]
    fn test_clear_again() {
        let mut pf = LinkedListParetoFront::<TestSolution<2>, 2>::new("test");
        let sol1 = TestSolution {
            objectives: [1, 5],
            id: 0,
        };
        pf.try_insert(&sol1);
        pf.clear();
        assert!(pf.is_empty());
    }

    #[test]
    fn test_is_empty() {
        let mut pf = LinkedListParetoFront::<TestSolution<2>, 2>::new("test");
        assert!(pf.is_empty());
        let sol1 = TestSolution {
            objectives: [1, 5],
            id: 0,
        };
        pf.try_insert(&sol1);
        assert!(!pf.is_empty());
    }

    #[test]
    fn test_complex_dominance_scenario() {
        let mut pf = LinkedListParetoFront::<TestSolution<2>, 2>::new("test");
        let solutions = vec![
            TestSolution {
                objectives: [5, 5],
                id: 0,
            },
            TestSolution {
                objectives: [4, 6],
                id: 0,
            },
            TestSolution {
                objectives: [6, 4],
                id: 0,
            },
            TestSolution {
                objectives: [3, 3],
                id: 0,
            }, // Dominates all previous
        ];

        for sol in &solutions[0..3] {
            pf.try_insert(sol);
        }
        assert_eq!(pf.len(), 3);

        assert!(pf.try_insert(&solutions[3]));
        assert_eq!(pf.len(), 1);
        assert_eq!(pf.iter().next().unwrap().objectives, [3, 3]);
    }

    #[test]
    fn test_duplicate_solutions() {
        let mut pf = LinkedListParetoFront::<TestSolution<2>, 2>::new("test");
        let sol1 = TestSolution {
            objectives: [1, 5],
            id: 0,
        };
        pf.try_insert(&sol1);
        assert!(!pf.try_insert(&sol1)); // Should not insert duplicates
        assert_eq!(pf.len(), 1);
    }

    #[test]
    fn test_many_insertions() {
        let mut pf = LinkedListParetoFront::<TestSolution<2>, 2>::new("test");
        for i in 0..100 {
            let sol = TestSolution {
                objectives: [100 - i, i],
                id: 0,
            };
            pf.try_insert(&sol);
        }
        assert_eq!(pf.len(), 100);
    }

    #[test]
    fn test_retain_logic() {
        let mut pf = LinkedListParetoFront::<TestSolution<2>, 2>::new("test");
        let sol1 = TestSolution {
            objectives: [10, 1],
            id: 0,
        };
        let sol2 = TestSolution {
            objectives: [9, 2],
            id: 0,
        };
        let sol3 = TestSolution {
            objectives: [8, 3],
            id: 0,
        };
        let sol4 = TestSolution {
            objectives: [7, 2],
            id: 0,
        }; // Dominates sol2

        pf.try_insert(&sol1);
        pf.try_insert(&sol2);
        pf.try_insert(&sol3);

        assert_eq!(pf.len(), 3);
        assert!(pf.try_insert(&sol4));
        assert_eq!(pf.len(), 3); // Should be [10,1], [8,3], [7,2]
    }

    #[cfg(test)]
    mod two_dimensional_tests {
        use super::*;

        fn solution(a: u64, b: u64) -> TestSolution<2> {
            TestSolution::new([a, b])
        }

        #[test]
        fn test_try_insert() {
            let mut pf = LinkedListParetoFront::<TestSolution<2>, 2>::new("2d_test");
            let sol1 = TestSolution {
                objectives: [10, 20],
                id: 0,
            };
            let sol2 = TestSolution {
                objectives: [20, 10],
                id: 0,
            };
            let sol3 = TestSolution {
                objectives: [5, 25],
                id: 0,
            }; // Dominates sol1

            assert!(pf.try_insert(&sol1));
            assert!(pf.try_insert(&sol2));
            assert_eq!(pf.len(), 2);

            assert!(pf.try_insert(&sol3));
            assert_eq!(pf.len(), 2); // [20, 10], [5, 25]
        }
    }

    #[cfg(test)]
    mod three_dimensional_tests {

        fn solution(a: u64, b: u64, c: u64) -> TestSolution<3> {
            TestSolution::new([a, b, c])
        }
        use super::*;

        #[test]
        fn test_try_insert() {
            let mut pf = LinkedListParetoFront::<TestSolution<3>, 3>::new("3d_test");
            let sol1 = TestSolution {
                objectives: [10, 20, 30],
                id: 0,
            };
            let sol2 = TestSolution {
                objectives: [20, 10, 30],
                id: 0,
            };
            let sol3 = TestSolution {
                objectives: [5, 25, 35],
                id: 0,
            }; // Dominates sol1

            assert!(pf.try_insert(&sol1));
            assert!(pf.try_insert(&sol2));
            assert_eq!(pf.len(), 2);

            assert!(pf.try_insert(&sol3));
            assert_eq!(pf.len(), 2);
        }
    }

    #[cfg(test)]
    mod property_based_tests {
        use super::*;
        use proptest::prelude::*;

        fn solution(a: u64, b: u64) -> TestSolution<2> {
            TestSolution::new([a, b])
        }

        proptest! {
            #[test]
            fn prop_try_insert_does_not_panic(
                solutions in prop::collection::vec(any::<([u64; 2], u64)>(), 1..100)
            ) {
                let mut pf = LinkedListParetoFront::<TestSolution<2>, 2>::new("prop_test");
                for (obj, id) in solutions {
                    let sol = TestSolution { objectives: obj, id: id as usize };
                    pf.try_insert(&sol);
                }
            }

            #[test]
            fn prop_len_is_consistent(
                solutions in prop::collection::vec(any::<([u64; 2], u64)>(), 1..100)
            ) {
                let mut pf = LinkedListParetoFront::<TestSolution<2>, 2>::new("prop_test");
                for (obj, id) in solutions {
                    let sol = TestSolution { objectives: obj, id: id as usize };
                    pf.try_insert(&sol);
                }
                let count = pf.iter().count();
                prop_assert_eq!(pf.len(), count);
            }
        }
    }
}
