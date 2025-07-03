use log::{trace, warn};
use nd_tree::nd_tree::{NDTree, Solution};
use pareto::{HasObjectives, MoSolution};

use crate::{problem::Problem, solution::SIMSSolutionTrait, solution_set::SolutionSet};

#[derive(Clone)]
pub struct NdTreeSolutionSet<T>
where
    T: SIMSSolutionTrait<2> + Sized + Ord + Clone,
{
    nd_tree: NDTree<Solution<2>, 32, 2, 4>,
    solutions: Vec<T>, // Keep original solutions alongside nd-tree
    name: String,
}

impl<T> SolutionSet<'_, T, 2> for NdTreeSolutionSet<T>
where
    T: SIMSSolutionTrait<2> + Sized + Ord + Clone,
{
    type Iter<'b>
        = std::slice::Iter<'b, T>
    where
        T: 'b;
    type IntoIter = std::vec::IntoIter<T>;

    fn new(name: String) -> Self {
        NdTreeSolutionSet {
            name,
            nd_tree: NDTree::new(),
            solutions: Vec::new(),
        }
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.solutions.iter()
    }

    fn contains(&self, solution: &T) -> bool {
        self.solutions.contains(solution)
    }

    fn remove_dominated(&mut self, solution: &T) {
        let size_before = self.solutions.len();

        // Convert solution to nd-tree format for efficient dominance checking
        let nd_solution = Solution::new(*solution.objectives());

        // Remove dominated solutions
        self.solutions.retain(|s| {
            let s_nd = Solution::new(*s.objectives());
            !nd_solution.is_dominated_by(s_nd.objectives())
        });

        let size_after = self.solutions.len();
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
        if self.solutions.is_empty() {
            self.solutions.push(solution.clone());
            let nd_solution = Solution::new(*solution.objectives());
            self.nd_tree.update(nd_solution);
            return true;
        }

        if self.solutions.contains(solution) {
            warn!("Solution set contains given solution!");
            return false;
        }

        // Check if dominated using nd-tree
        let nd_solution = Solution::new(*solution.objectives());

        // Check if any existing solution dominates the new one
        let is_dominated = self.solutions.iter().any(|s| {
            let s_nd = Solution::new(*s.objectives());
            nd_solution.is_dominated_by(s_nd.objectives())
        });

        if !is_dominated {
            self.solutions.push(solution.clone());
            self.nd_tree.update(nd_solution);
            return true;
        }

        false
    }

    fn try_add(&mut self, solution: &T) -> bool {
        let was_inserted = self.insert_if_not_dominated(solution);
        if was_inserted {
            self.remove_dominated(solution);
        }
        was_inserted
    }

    fn random(size: usize, problem: &Problem<2>) -> Self {
        let random_iter = (0..size).map(|_| T::random(problem));
        return Self::from_iter(random_iter);
    }

    fn random_with_seed(size: usize, problem: &Problem<2>, seed: u64) -> Self {
        let random_iter = (0..size).map(|_| T::random_with_seed(problem, seed));
        return Self::from_iter(random_iter);
    }

    fn len(&self) -> usize {
        self.solutions.len()
    }

    fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    fn force_add(&mut self, solution: &T) {
        self.solutions.push(solution.clone());
        let nd_solution = Solution::new(*solution.objectives());
        self.nd_tree.update(nd_solution);
    }

    fn replace_if_exists(&mut self, solution: T) {
        if let Some(pos) = self.solutions.iter().position(|s| s == &solution) {
            self.solutions[pos] = solution;
        }
    }

    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let solutions: Vec<T> = iter.into_iter().collect();
        let mut nd_tree = NDTree::new();

        // Add all solutions to nd-tree
        for solution in &solutions {
            let nd_solution = Solution::new(*solution.objectives());
            nd_tree.update(nd_solution);
        }

        NdTreeSolutionSet {
            name: "unnamed".to_string(),
            nd_tree,
            solutions,
        }
    }

    fn into_iter(self) -> Self::IntoIter {
        self.solutions.into_iter()
    }
}
