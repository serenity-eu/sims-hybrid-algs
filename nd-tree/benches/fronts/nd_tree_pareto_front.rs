use nd_tree::nd_tree::{NDTree, NDTreeSolutionIntoIterator, NDTreeSolutionIterator};
use pareto::{HasObjectives, MoSolution, ParetoFront};
use std::fmt::Debug;

/// A Pareto front implementation that uses an ND-Tree for efficient dominance checking.
pub struct NdTreeParetoFront<T, const N: usize, const D: usize, const C: usize>
where
    T: HasObjectives<D> + MoSolution<D> + Clone,
{
    nd_tree: NDTree<T, N, D, C>,
}

impl<T, const N: usize, const D: usize, const C: usize> ParetoFront<'static, T>
    for NdTreeParetoFront<T, N, D, C>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + Debug + 'static,
{
    type Iter<'b>
        = NDTreeSolutionIterator<'b, T, N, D, C>
    where
        T: 'b;
    type IntoIter = NDTreeSolutionIntoIterator<T, N, D, C>;

    fn new(_name: &'static str) -> Self {
        Self {
            nd_tree: NDTree::new(),
        }
    }

    fn with_name(self, _name: &'static str) -> Self {
        self
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.nd_tree.iter()
    }

    fn contains(&self, solution: &T) -> bool {
        // Check if any solution in the tree has the same objectives
        self.nd_tree
            .iter()
            .any(|s| s.objectives() == solution.objectives())
    }

    fn try_insert(&mut self, solution: &T) -> bool {
        self.nd_tree.update(solution.clone())
    }

    fn insert_unchecked(&mut self, solution: &T) {
        self.nd_tree.update_unchecked(solution.clone());
    }

    fn replace_if_exists(&mut self, solution: T) {
        // Find if there's an existing solution with the same objectives
        let has_existing = self
            .nd_tree
            .iter()
            .any(|s| s.objectives() == solution.objectives());

        if has_existing {
            // Remove the existing solution and add the new one
            // For now, we'll use a simple approach: rebuild the tree
            let mut new_tree = NDTree::new();
            for existing in &self.nd_tree {
                if existing.objectives() != solution.objectives() {
                    new_tree.update_unchecked(existing.clone());
                }
            }
            new_tree.update_unchecked(solution);
            self.nd_tree = new_tree;
        }
    }

    fn len(&self) -> usize {
        self.nd_tree.len()
    }
    fn is_empty(&self) -> bool {
        self.nd_tree.is_empty()
    }
}

impl<T, const N: usize, const D: usize, const C: usize> Debug for NdTreeParetoFront<T, N, D, C>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("NdTreeParetoFront")
            .field("nd_tree", &self.nd_tree)
            .finish()
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_pareto_front_new() {
        let pf = NdTreeParetoFront::<Solution<2>, 4, 2, 4>::new("test");
        assert_eq!(pf.len(), 0);
        assert!(pf.is_empty());
    }

    #[test]
    fn test_pareto_front_try_insert() {
        let mut pf = NdTreeParetoFront::<Solution<2>, 4, 2, 4>::new("test");
        assert!(pf.try_insert(&Solution::new([1, 1])));
        assert_eq!(pf.len(), 1);
        assert!(!pf.is_empty());
    }

    #[test]
    fn test_pareto_front_dominance() {
        let mut pf = NdTreeParetoFront::<Solution<2>, 4, 2, 4>::new("test");
        pf.try_insert(&Solution::new([2, 2]));
        assert!(!pf.try_insert(&Solution::new([3, 3]))); // dominated
        assert!(pf.try_insert(&Solution::new([1, 1]))); // dominates
        assert_eq!(pf.len(), 1);
        assert!(pf.contains(&Solution::new([1, 1])));
        assert!(!pf.contains(&Solution::new([2, 2])));
    }

    #[test]
    fn test_pareto_front_insert_many() {
        let mut pf = NdTreeParetoFront::<Solution<2>, 4, 2, 4>::new("test");
        pf.try_insert(&Solution::new([5, 1]));
        pf.try_insert(&Solution::new([4, 2]));
        pf.try_insert(&Solution::new([3, 3]));
        pf.try_insert(&Solution::new([2, 4]));
        pf.try_insert(&Solution::new([1, 5]));
        assert_eq!(pf.len(), 5);
    }

    #[test]
    fn test_pareto_front_iter() {
        let mut pf = NdTreeParetoFront::<Solution<2>, 4, 2, 4>::new("test");
        pf.try_insert(&Solution::new([2, 2]));
        pf.try_insert(&Solution::new([1, 1]));
        let solutions: Vec<_> = pf.iter().collect();
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].objectives(), &[1, 1]);
    }
}
