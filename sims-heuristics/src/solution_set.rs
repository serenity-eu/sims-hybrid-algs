use crate::{problem::Problem, solution::SIMSSolutionTrait};

pub trait SolutionSet<'a, T: SIMSSolutionTrait<D>, const D: usize> {
    type Iter<'b>: Iterator<Item = &'b T>
    where
        Self: 'b,
        T: 'b;

    type IntoIter: Iterator<Item = T>;

    /// Create an empty solution set
    fn new(name: String) -> Self;

    /// Set name
    #[must_use]
    fn with_name(self, name: String) -> Self;

    /// Create a set of given size of random solutions
    fn random(size: usize, problem: &Problem<D>) -> Self;

    /// Create a set of given size of random solutions with given seed
    fn random_with_seed(size: usize, problem: &Problem<D>, seed: u64) -> Self;

    /// Iterate over the solutions in the set
    fn iter(&self) -> Self::Iter<'_>;

    /// Check if solution is in the set
    fn contains(&self, solution: &T) -> bool;

    /// Remove all solutions that are dominated by the given solution
    fn remove_dominated(&mut self, solution: &T);

    /// Insert the given solution if it is not dominated by any of the existing solutions
    fn insert_if_not_dominated(&mut self, solution: &T) -> bool;

    /// Try add new solution to the set, return true if it there was no solution in the set that dominated it and it was added
    fn try_add(&mut self, solution: &T) -> bool;

    /// Forcefuly add solution to the set, for use only with collect from another `SolutionSet`
    fn force_add(&mut self, solution: &T);

    /// Replace given solution with updated solution
    fn replace_if_exists(&mut self, solution: T);

    /// Create a solution set from an iterator
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self;

    /// Convert the solution set into an iterator
    fn into_iter(self) -> Self::IntoIter;

    /// Return length of the solution set
    fn len(&self) -> usize {
        self.iter().count()
    }

    /// Return true if the solution set is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
