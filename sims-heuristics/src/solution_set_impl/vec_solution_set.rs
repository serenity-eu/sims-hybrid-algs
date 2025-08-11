use log::trace;
use pareto::{ParetoFront, Random, RandomCollection};

use crate::solution::EncodedSolution;

#[derive(Clone)]
pub struct VecSolutionSet<T: EncodedSolution<D> + Sized, const D: usize> {
    name: &'static str,
    last_added_position: usize,
    vec_set: Vec<T>,
}

impl<T, const D: usize> VecSolutionSet<T, D>
where
    T: EncodedSolution<D> + Sized + Clone,
{
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            last_added_position: 0,
            vec_set: Vec::new(),
        }
    }

    fn binary_search(&self, solution: &T) -> Result<usize, usize> {
        self.vec_set
            .binary_search_by_key(solution.objectives().first().unwrap(), |s| {
                *s.objectives().first().unwrap()
            })
    }
}

impl<T, const D: usize> ParetoFront<'_, T> for VecSolutionSet<T, D>
where
    T: EncodedSolution<D> + Sized + Clone,
{
    type Iter<'b>
        = std::slice::Iter<'b, T>
    where
        T: 'b;

    fn new(name: &'static str) -> Self {
        Self {
            name,
            last_added_position: 0,
            vec_set: Vec::new(),
        }
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.vec_set.iter()
    }

    fn contains(&self, solution: &T) -> bool {
        self.binary_search(solution).is_ok()
    }

    fn try_insert(&mut self, solution: &T) -> bool {
        // Insert if set is empty
        if self.vec_set.is_empty() {
            self.vec_set.push(solution.clone());
            return true;
        }

        let was_inserted;

        match self.binary_search(solution) {
            Ok(_) => {
                was_inserted = false;
            }
            Err(position) => {
                let mut first_equal_pos = position;
                // Use first objective for sorting (this maintains compatibility with existing sort behavior)
                let current_objective = if position < self.vec_set.len() {
                    self.vec_set[position].objectives()[0]
                } else {
                    // If position is at the end, use the last element's objective
                    self.vec_set[self.vec_set.len() - 1].objectives()[0]
                };
                while first_equal_pos > 0
                    && self.vec_set[first_equal_pos - 1].objectives()[0] == current_objective
                {
                    first_equal_pos -= 1;
                }
                if self.vec_set[..first_equal_pos]
                    .iter()
                    .any(|s| solution.is_covered_by(s.objectives()))
                {
                    was_inserted = false;
                } else {
                    self.vec_set.insert(position, solution.clone());
                    self.last_added_position = position;
                    was_inserted = true;
                }
            }
        }
        if was_inserted {
            let size_before = self.vec_set.len();
            let mut to_remove = Vec::new();

            let start_index = self.last_added_position;
            let potentially_dominated_solutions = &self.vec_set[start_index..];
            for (index, potentially_dominated) in potentially_dominated_solutions.iter().enumerate()
            {
                let original_index = start_index + index;
                if potentially_dominated.is_covered_by(solution.objectives()) {
                    to_remove.push(original_index);
                }
            }

            for index in to_remove.iter().rev() {
                self.vec_set.remove(*index);
            }

            let size_after = self.vec_set.len();
            if size_before != size_after {
                trace!(
                    "Removed {} dominated solutions from the {} set. Size after: {}",
                    size_before - size_after,
                    self.name,
                    size_after
                );
            }
        }
        was_inserted
    }

    fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }

    fn insert_unchecked(&mut self, solution: &T) {
        match self.binary_search(solution) {
            Ok(position) | Err(position) => {
                // Solution already exists, insert anyway (unchecked)
                self.vec_set.insert(position, solution.clone());
            }
        }
    }
}

impl<T, const D: usize> FromIterator<T> for VecSolutionSet<T, D>
where
    T: EncodedSolution<D> + Sized + Clone,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let vec_set: Vec<T> = iter.into_iter().collect();
        Self {
            name: "unnamed",
            vec_set,
            last_added_position: 0,
        }
    }
}

impl<T, const D: usize> IntoIterator for VecSolutionSet<T, D>
where
    T: EncodedSolution<D> + Sized + Clone,
{
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec_set.into_iter()
    }
}

impl<T, const D: usize> RandomCollection<T> for VecSolutionSet<T, D> where
    T: EncodedSolution<D> + Sized + Clone + Random
{
}
