use nd_tree::nd_tree::{NDTree, NDTreeSolutionIterator};
use pareto::{MoSolution, ParetoFront, Random, RandomCollection};
use std::fmt::Debug;

use crate::solution::ImageSet;
#[cfg(debug_assertions)]
use itertools::Itertools;

#[derive(Clone)]
pub struct NdTreeSolutionSet<T, const D: usize>
where
    T: ImageSet<D> + MoSolution<D> + PartialEq + Sized + Clone,
{
    name: &'static str,
    nd_tree: NDTree<T, 32, D, 4>,
}

impl<T, const D: usize> ParetoFront<'_, T> for NdTreeSolutionSet<T, D>
where
    T: ImageSet<D> + MoSolution<D> + PartialEq + Sized + Clone + Debug,
{
    type Iter<'b>
        = NDTreeSolutionIterator<'b, T, 32, D, 4>
    where
        T: 'b;

    fn new(name: &'static str) -> Self {
        Self {
            name,
            nd_tree: NDTree::new(),
        }
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.nd_tree.iter()
    }

    fn contains(&self, solution: &T) -> bool {
        self.nd_tree.contains(solution)
    }

    fn try_insert(&mut self, solution: &T) -> bool {
        let was_inserted = self.nd_tree.update(solution.clone());

        #[cfg(debug_assertions)]
        {
            let all_combinations_valid =
                self.nd_tree.iter().enumerate().combinations(2).all(|pair| {
                    let ((first_idx, first), (second_idx, second)) = (pair[0], pair[1]);
                    let first_objectives = first.objectives();
                    let second_objectives = second.objectives();
                    let dominates_first_second = first.dominates(second_objectives);
                    let dominates_second_first = second.dominates(first_objectives);
                    if dominates_first_second || dominates_second_first {
                        println!(
                            "Dominated solution index: {}, solutions: {:?} vs {:?}",
                            if dominates_first_second {
                                second_idx
                            } else {
                                first_idx
                            },
                            first,
                            second
                        );
                    }
                    !dominates_first_second && !dominates_second_first
                });
            debug_assert!(all_combinations_valid, "ND-tree invariant violated");
        }

        was_inserted
    }

    fn insert_unchecked(&mut self, solution: &T) {
        // Even in unchecked mode, prevent exact duplicates to maintain archive invariants
        if !self.nd_tree.contains(solution) {
            self.nd_tree.update(solution.clone());
        }
    }

    fn len(&self) -> usize {
        self.nd_tree.len()
    }

    fn is_empty(&self) -> bool {
        self.nd_tree.is_empty()
    }

    fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }
}

impl<T, const D: usize> FromIterator<T> for NdTreeSolutionSet<T, D>
where
    T: ImageSet<D> + MoSolution<D> + PartialEq + Sized + Clone,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut nd_tree = NDTree::new();

        for solution in iter {
            nd_tree.update(solution);
        }

        Self {
            name: "unnamed",
            nd_tree,
        }
    }
}

impl<T, const D: usize> IntoIterator for NdTreeSolutionSet<T, D>
where
    T: ImageSet<D> + MoSolution<D> + PartialEq + Sized + Clone,
{
    type Item = T;
    type IntoIter = nd_tree::nd_tree::NDTreeSolutionIntoIterator<T, 32, D, 4>;

    fn into_iter(self) -> Self::IntoIter {
        self.nd_tree.into_iter()
    }
}

impl<T, const D: usize> Default for NdTreeSolutionSet<T, D>
where
    T: ImageSet<D> + MoSolution<D> + PartialEq + Sized + Clone + Debug,
{
    fn default() -> Self {
        Self::new("default")
    }
}

impl<T, const D: usize> RandomCollection<T> for NdTreeSolutionSet<T, D> where
    T: ImageSet<D> + MoSolution<D> + PartialEq + Sized + Clone + Random
{
}
