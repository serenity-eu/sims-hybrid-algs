use std::slice;

use log::trace;

use crate::{solution::SIMSSolutionTrait, solution_set::SolutionSet};

#[derive(Clone)]
pub struct VecSolutionSet<T: SIMSSolutionTrait<D> + Sized, const D: usize> {
    name: String,
    last_added_position: usize,
    vec_set: Vec<T>,
}

impl<T, const D: usize> SolutionSet<'_, T, D> for VecSolutionSet<T, D>
where
    T: SIMSSolutionTrait<D> + Sized + Ord + Clone,
{
    type Iter<'b>
        = slice::Iter<'b, T>
    where
        T: 'b;
    type IntoIter = std::vec::IntoIter<T>;

    fn new(name: String) -> Self {
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
        self.vec_set.binary_search(solution).is_ok()
    }

    /// Note: This can be done efficiently given that the solutions are sorted, first we find the first solution which is dominated by first objective, then we remove all solutions that are dominated by the second objective
    fn remove_dominated(&mut self, solution: &T) {
        let size_before = self.vec_set.len();
        let mut to_remove = Vec::new();

        let start_index = self.last_added_position;
        let potentially_dominated_solutions = &self.vec_set[start_index..];
        for (index, potentially_dominated) in potentially_dominated_solutions.iter().enumerate() {
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

    fn insert_if_not_dominated(&mut self, solution: &T) -> bool {
        // Insert if set is empty
        if self.vec_set.is_empty() {
            self.vec_set.push(solution.clone());
            return true;
        }

        match self.vec_set.binary_search(solution) {
            Ok(_) => return false,
            Err(position) => {
                let mut first_equal_pos = position;
                let current_objective = self.vec_set[position].objectives()[0];
                while first_equal_pos > 0
                    && self.vec_set[first_equal_pos - 1].objectives()[0] == current_objective
                {
                    first_equal_pos -= 1;
                }
                if self.vec_set[..first_equal_pos]
                    .iter()
                    .any(|s| solution.is_covered_by(s.objectives()))
                {
                    return false;
                }
                self.vec_set.insert(position, solution.clone());
                self.last_added_position = position;
                return true;
            }
        }
    }

    fn try_add(&mut self, solution: &T) -> bool {
        let was_inserted = self.insert_if_not_dominated(solution);
        if was_inserted {
            self.remove_dominated(solution);
        }
        was_inserted
    }

    fn random(size: usize, problem: &crate::problem::Problem<D>) -> Self {
        let random_iter = (0..size).map(|_| T::random_with_problem(problem));
        return Self::from_iter(random_iter);
    }

    fn random_with_seed(size: usize, problem: &crate::problem::Problem<D>, seed: u64) -> Self {
        let random_iter = (0..size).map(|_| T::random_with_problem_and_seed(problem, seed));
        return Self::from_iter(random_iter);
    }

    fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    fn force_add(&mut self, solution: &T) {
        let position = self.vec_set.binary_search(solution).unwrap_err();
        self.vec_set.insert(position, solution.clone());
    }

    fn replace_if_exists(&mut self, solution: T) {
        if let Ok(position) = self.vec_set.binary_search(&solution) {
            self.vec_set[position] = solution;
        }
    }

    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            name: "unnamed".to_string(),
            vec_set: iter.into_iter().collect(),
            last_added_position: 0,
        }
    }

    fn into_iter(self) -> Self::IntoIter {
        return self.vec_set.into_iter();
    }
}
