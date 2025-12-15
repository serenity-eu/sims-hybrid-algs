#![expect(
    clippy::linkedlist,
    reason = "Using LinkedList for Pareto front as a reference implementation"
)]

use std::collections::LinkedList;

use pareto::{ParetoFront, Random, RandomCollection};
use tracing::{trace, warn};

use crate::solution::EncodedSolution;

#[derive(Clone)]
pub struct LinkedListSolutionSet<T: EncodedSolution<D> + Sized, const D: usize> {
    name: &'static str,
    solutions: LinkedList<T>,
}

impl<T, const D: usize> LinkedListSolutionSet<T, D>
where
    T: EncodedSolution<D> + Sized + Clone,
{
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            solutions: LinkedList::new(),
        }
    }
}

impl<T, const D: usize> ParetoFront<'_, T> for LinkedListSolutionSet<T, D>
where
    T: EncodedSolution<D> + Sized + Clone,
{
    type Iter<'b>
        = std::collections::linked_list::Iter<'b, T>
    where
        T: 'b;

    fn new(name: &'static str) -> Self {
        Self {
            name,
            solutions: LinkedList::new(),
        }
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.solutions.iter()
    }

    fn contains(&self, solution: &T) -> bool {
        self.solutions.iter().any(|s| s == solution)
    }

    fn try_insert(&mut self, solution: &T) -> bool {
        // Insert if set is empty
        if self.solutions.is_empty() {
            self.solutions.push_back(solution.clone());
            return true;
        }

        if self.contains(solution) {
            warn!("Solution set contains given solution!");
            return false;
        }

        // Check if new solution is dominated by any existing solution
        for existing in &self.solutions {
            if solution.is_covered_by(existing.objectives()) {
                return false; // New solution is dominated or equal, reject it
            }
        }

        // Remove all solutions dominated by the new solution
        let size_before = self.solutions.len();
        
        // Manually filter out dominated solutions since retain is unstable for LinkedList
        let mut new_solutions = LinkedList::new();
        for existing in &self.solutions {
            if !existing.is_dominated_by(solution.objectives()) {
                new_solutions.push_back(existing.clone());
            }
        }
        self.solutions = new_solutions;
        
        let size_after = self.solutions.len();
        if size_before != size_after {
            trace!(
                "Removed {} dominated solutions from the {} set. Size after: {}",
                size_before - size_after,
                self.name,
                size_after
            );
        }

        // Add the new solution
        self.solutions.push_back(solution.clone());
        true
    }

    fn len(&self) -> usize {
        self.solutions.len()
    }

    fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }

    fn insert_unchecked(&mut self, solution: &T) {
        self.solutions.push_back(solution.clone());
    }
}

impl<T, const D: usize> FromIterator<T> for LinkedListSolutionSet<T, D>
where
    T: EncodedSolution<D> + Sized + Clone,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let solutions = iter.into_iter().collect();
        Self {
            solutions,
            name: "unnamed",
        }
    }
}

impl<T, const D: usize> IntoIterator for LinkedListSolutionSet<T, D>
where
    T: EncodedSolution<D> + Sized + Clone,
{
    type Item = T;
    type IntoIter = std::collections::linked_list::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.solutions.into_iter()
    }
}

impl<T, const D: usize> RandomCollection<T> for LinkedListSolutionSet<T, D> where
    T: EncodedSolution<D> + Sized + Clone + Random
{
}