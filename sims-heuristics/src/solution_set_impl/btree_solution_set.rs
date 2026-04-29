use std::collections::{BTreeSet, btree_set};

use pareto::{
    MoSolution, Objectives, ParetoFront, Random, RandomCollection, ScalarizedArchiveQuery,
};
use tracing::{trace, warn};

#[derive(Clone)]
pub struct BTreeSolutionSet<T, const D: usize>
where
    T: MoSolution<2> + MoSolution<D> + PartialEq + Sized,
{
    name: &'static str,
    btree_set: BTreeSet<T>,
}

impl<T> BTreeSolutionSet<T, 2>
where
    T: MoSolution<2> + PartialEq + Sized + Ord + Clone,
{
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            btree_set: BTreeSet::new(),
        }
    }
}

impl<T> ParetoFront<'_, T> for BTreeSolutionSet<T, 2>
where
    T: MoSolution<2> + PartialEq + Sized + Ord + Clone,
{
    type Iter<'b>
        = btree_set::Iter<'b, T>
    where
        T: 'b;

    fn new(name: &'static str) -> Self {
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

    fn try_insert(&mut self, solution: &T) -> bool {
        // Insert if set is empty
        if self.btree_set.is_empty() {
            self.btree_set.insert(solution.clone());
            return true;
        }

        if self.btree_set.contains(solution) {
            warn!("Solution set contains given solution!");
            return false;
        }

        let was_inserted = if self
            .btree_set
            // .range(..solution)
            .iter()
            .all(|s| !solution.is_covered_by(s.objectives()))
        {
            self.btree_set.insert(solution.clone());
            true
        } else {
            false
        };

        if was_inserted {
            let size_before = self.btree_set.len();

            // TODO: Can we use range() to iterate over only subset of the solution set?
            self.btree_set
                .retain(|s| !s.is_dominated_by(solution.objectives()));

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
        was_inserted
    }

    fn len(&self) -> usize {
        self.iter().count()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }

    fn insert_unchecked(&mut self, solution: &T) {
        self.btree_set.insert(solution.clone());
    }
}

impl<T> FromIterator<T> for BTreeSolutionSet<T, 2>
where
    T: MoSolution<2> + PartialEq + Sized + Ord + Clone,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let btree_set = iter.into_iter().collect();
        Self {
            btree_set,
            name: "unnamed",
        }
    }
}

impl<T> IntoIterator for BTreeSolutionSet<T, 2>
where
    T: MoSolution<2> + PartialEq + Sized + Clone + Ord,
{
    type Item = T;
    type IntoIter = btree_set::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.btree_set.into_iter()
    }
}

impl<T> ScalarizedArchiveQuery<T, 2> for BTreeSolutionSet<T, 2>
where
    T: MoSolution<2> + PartialEq + Sized + Clone + Ord,
{
    fn find_best_with_pruning<Accept, NodeLowerBound, SolutionScore>(
        &self,
        mut accept: Accept,
        _node_lower_bound: NodeLowerBound,
        solution_score: SolutionScore,
    ) -> Option<(&T, f64)>
    where
        Accept: FnMut(&T) -> bool,
        NodeLowerBound: Fn(&Objectives<2>) -> f64,
        SolutionScore: Fn(&T) -> f64,
    {
        let mut best_solution: Option<&T> = None;
        let mut best_score = f64::INFINITY;

        for solution in &self.btree_set {
            if !accept(solution) {
                continue;
            }

            let score = solution_score(solution);
            if score < best_score {
                best_score = score;
                best_solution = Some(solution);
            }
        }

        best_solution.map(|solution| (solution, best_score))
    }
}

impl<T> Default for BTreeSolutionSet<T, 2>
where
    T: MoSolution<2> + PartialEq + Sized + Ord + Clone,
{
    fn default() -> Self {
        Self::new("default")
    }
}

impl<T> RandomCollection<T> for BTreeSolutionSet<T, 2> where
    T: MoSolution<2> + PartialEq + Sized + Ord + Clone + Random
{
}
