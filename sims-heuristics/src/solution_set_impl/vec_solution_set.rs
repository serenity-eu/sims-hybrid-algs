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
        // Binary search only checks first objective, so we need to verify exact match
        match self.binary_search(solution) {
            Ok(pos) => {
                // Found a solution with same first objective, check if it's exactly the same
                self.vec_set[pos] == *solution
            }
            Err(pos) => {
                // Check solutions around the position that might have same first objective
                let first_obj = solution.objectives()[0];
                
                // Check backwards from position
                let mut idx = pos;
                while idx > 0 && self.vec_set[idx - 1].objectives()[0] == first_obj {
                    idx -= 1;
                    if self.vec_set[idx] == *solution {
                        return true;
                    }
                }
                
                // Check forward from position
                idx = pos;
                while idx < self.vec_set.len() && self.vec_set[idx].objectives()[0] == first_obj {
                    if self.vec_set[idx] == *solution {
                        return true;
                    }
                    idx += 1;
                }
                
                false
            }
        }
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
                // Check if new solution is dominated by ANY existing solution
                if self.vec_set
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

            // Check ALL solutions for dominance by the newly inserted solution
            for (index, existing) in self.vec_set.iter().enumerate() {
                if index != self.last_added_position && existing.is_dominated_by(solution.objectives()) {
                    to_remove.push(index);
                }
            }

            // Remove dominated solutions in reverse order to maintain indices
            for &index in to_remove.iter().rev() {
                self.vec_set.remove(index);
                // Adjust last_added_position if we removed something before it
                if index < self.last_added_position {
                    self.last_added_position -= 1;
                }
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
