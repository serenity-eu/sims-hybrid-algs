use std::collections::{btree_set, BTreeSet};

use log::{trace, warn};

use crate::{problem::Problem, solution::SIMSSolutionTrait, solution_set::SolutionSet};

#[derive(Clone)]
pub struct BTreeSolutionSet<T: SIMSSolutionTrait<D> + Sized, const D: usize> {
    btree_set: BTreeSet<T>,
    name: String,
}

impl<T, const D: usize> SolutionSet<'_, T, D> for BTreeSolutionSet<T, D>
where
    T: SIMSSolutionTrait<D> + Sized + Ord + Clone,
{
    type Iter<'b>
        = btree_set::Iter<'b, T>
    where
        T: 'b;
    type IntoIter = btree_set::IntoIter<T>;

    fn new(name: String) -> Self {
        Self {
            name,
            btree_set: BTreeSet::new(),
        }
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.btree_set.iter()
    }

    fn contains(&self, solution: &T) -> bool {
        self.btree_set.contains(solution)
    }

    /// Note: This can be done efficiently given that the solutions are sorted, first we find the first solution which is dominated by first objective, then we remove all solutions that are dominated by the second objective
    fn remove_dominated(&mut self, solution: &T) {
        let size_before = self.btree_set.len();

        // TODO: Can we use range() to iterate over only subset of the solution set?
        self.btree_set.retain(|s| !s.is_dominated(solution));

        let size_after = self.btree_set.len();
        if size_before != size_after {
            trace!(
                "Removed {} dominated solutions from the {} set. Size after: {}",
                size_before - size_after,
                self.name,
                size_after
            );
        }
        // let dominated_solutions = self.range(solution..).filter(|s| solution.dominates(s));
        // for dominated_solution in dominated_solutions {
        //     self.remove(dominated_solution);
        // }
    }

    fn insert_if_not_dominated(&mut self, solution: &T) -> bool {
        // Insert if set is empty
        if self.btree_set.is_empty() {
            self.btree_set.insert(solution.clone());
            return true;
        }

        if self.btree_set.contains(solution) {
            warn!("Solution set contains given solution!");
            return false;
        }

        if self
            .btree_set
            // .range(..solution)
            .iter()
            .all(|s| !solution.is_weakly_dominated(s))
        {
            self.btree_set.insert(solution.clone());
            return true;
        }
        return false;
    }

    fn try_add(&mut self, solution: &T) -> bool {
        let was_inserted = self.insert_if_not_dominated(solution);
        if was_inserted {
            self.remove_dominated(solution);
        }
        was_inserted
    }

    fn random(size: usize, problem: &Problem<D>) -> Self {
        let random_iter = (0..size).map(|_| T::random(problem));
        return Self::from_iter(random_iter);
    }

    fn random_with_seed(size: usize, problem: &Problem<D>, seed: u64) -> Self {
        let random_iter = (0..size).map(|_| T::random_with_seed(problem, seed));
        return Self::from_iter(random_iter);
    }

    fn len(&self) -> usize {
        self.iter().count()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    fn force_add(&mut self, solution: &T) {
        self.btree_set.insert(solution.clone());
    }

    fn replace_if_exists(&mut self, solution: T) {
        if self.btree_set.contains(&solution) {
            self.btree_set.replace(solution);
        }
    }

    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            name: "unnamed".to_string(),
            btree_set: iter.into_iter().collect(),
        }
    }

    fn into_iter(self) -> Self::IntoIter {
        self.btree_set.into_iter()
    }
}
