use arrayvec::ArrayVec;
use slotmap::{DefaultKey, SlotMap};
use std::{array, vec};

use pareto::{Dominance, HasObjectives, MoSolution, Objectives};

/// A solution in D-dimensional objective space.
#[derive(Clone, Debug)]
pub struct Solution<const D: usize> {
    pub objectives: Objectives<D>,
}

impl<const D: usize> Solution<D> {
    pub fn new(objectives: Objectives<D>) -> Self {
        Self { objectives }
    }
}

impl<const D: usize> HasObjectives<D> for Solution<D> {
    fn objectives(&self) -> &Objectives<D> {
        &self.objectives
    }
}

impl<const D: usize> MoSolution<D> for Solution<D> {}

/// An ND‑Tree node: either a Leaf (up to N solutions) or an Internal node
/// (exactly C children).  Each stores its ideal/nadir bounds.
#[derive(Debug, Clone)]
pub enum Node<const N: usize, const D: usize, const C: usize> {
    Leaf {
        solutions: ArrayVec<Solution<D>, N>,
        ideal: Objectives<D>,
        middle: Objectives<D>,
        nadir: Objectives<D>,
    },
    Internal {
        children: ArrayVec<DefaultKey, C>, // fixed array of C arena indices
        ideal: Objectives<D>,
        middle: Objectives<D>,
        nadir: Objectives<D>,
    },
}

impl<const N: usize, const D: usize, const C: usize> Node<N, D, C> {
    /// Start new leaf with single solution
    pub fn new_leaf_with_solution(solution: Solution<D>) -> Self {
        let objectives = solution.objectives;

        let mut solutions = ArrayVec::new();
        solutions.push(solution);

        Node::Leaf {
            solutions,
            ideal: objectives,
            middle: objectives,
            nadir: objectives,
        }
    }

    /// Component‑wise update of ideal/nadir from one more solution.
    pub fn update_bounds(&mut self, sol: &Solution<D>) {
        match self {
            Node::Leaf { ideal, nadir, .. } | Node::Internal { ideal, nadir, .. } => {
                *ideal = array::from_fn(|i| ideal[i].min(sol.objectives[i]));
                *nadir = array::from_fn(|i| nadir[i].max(sol.objectives[i]));
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Node::Leaf { solutions, .. } => solutions.is_empty(),
            Node::Internal { children, .. } => children.is_empty(),
        }
    }
}

/// The whole tree, storing nodes in an arena Vec and pointing to root by index.
pub struct NDTree<const N: usize, const D: usize, const C: usize> {
    arena: SlotMap<DefaultKey, Node<N, D, C>>,
    root: Option<DefaultKey>,
}

impl<const N: usize, const D: usize, const C: usize> NDTree<N, D, C> {
    /// Create an empty tree
    pub fn new() -> Self {
        NDTree {
            arena: SlotMap::new(),
            root: None,
        }
    }

    pub fn update(&mut self, new_solution: Solution<D>) -> bool {
        if let Some(root_key) = self.root {
            if self.update_node(root_key, &new_solution) != Dominance::IsDominatedBy {
                if self.arena[root_key].is_empty() {
                    // If the root node is empty after update, deallocate it
                    self.deallocate_root();
                }
                // If the solution is not dominated, we can insert it
                self.insert(new_solution);
                return true;
            }
            return false;
        } else {
            self.insert(new_solution);
            return true;
        }
    }

    pub fn update_unchecked(&mut self, new_solution: Solution<D>) {
        // This method is unsafe because it does not check the Pareto dominance relation
        self.insert(new_solution);
    }

    fn insert(&mut self, sol: Solution<D>) {
        if let Some(root_key) = self.root {
            // If the root exists, insert into it
            self.insert_into(root_key, sol);
        } else {
            // If the root does not exist, create a new root with the solution
            let leaf_node = Node::new_leaf_with_solution(sol);
            self.root = Some(self.arena.insert(leaf_node));
        }
    }

    fn closest_child(&self, node_key: DefaultKey, solution: &Solution<D>) -> DefaultKey {
        if let Node::Internal { children, .. } = &self.arena[node_key] {
            // Find the child with the closest middle point to the solution
            *children
                .iter()
                .min_by_key(|&&child_key| {
                    let child_middle_point = match &self.arena[child_key] {
                        Node::Leaf { middle, .. } | Node::Internal { middle, .. } => middle,
                    };
                    solution.squared_distance_to(child_middle_point)
                })
                .expect("For internal node, at least one child should exist")
        } else {
            panic!("Closes child called on a non-internal node");
        }
    }

    fn deallocate_subtree(&mut self, node_key: DefaultKey) {
        // Stack for DFS
        let mut stack = vec![node_key];
        while let Some(node_key) = stack.pop() {
            if let Node::Internal { children, .. } = &self.arena[node_key] {
                for &child_key in children.iter() {
                    stack.push(child_key);
                }
            }
            // Remove the node from the arena
            self.arena.remove(node_key);
        }

        if let Some(root_key) = self.root {
            // If the node being deallocated is the root, we need to update the root
            if node_key == root_key {
                self.root = None;
            }
        }
    }

    fn is_root_node(&self, node_key: DefaultKey) -> bool {
        self.root == Some(node_key)
    }

    fn deallocate_root(&mut self) {
        if let Some(root_key) = self.root {
            self.arena.remove(root_key);
            self.root = None;
        }
    }

    fn update_node(&mut self, node_key: DefaultKey, new_solution: &Solution<D>) -> Dominance {
        let mut dominance_relation = Dominance::NonDominated;

        match &mut self.arena[node_key] {
            Node::Leaf {
                nadir,
                ideal,
                solutions,
                ..
            } => {
                if new_solution.is_covered_by(nadir) {
                    // Property 1
                    // Exit early, new solution is rejected
                    return Dominance::IsDominatedBy;
                } else if new_solution.covers(ideal) {
                    // Property 2
                    // Clear solutions in the leaf
                    solutions.clear();
                    return Dominance::Dominates;
                } else if new_solution.is_covered_by(ideal) || new_solution.covers(nadir) {
                    // Property 4}
                    // Use inplace filter with early return
                    let mut i = 0;

                    while i < solutions.len() {
                        if solutions[i].covers(new_solution.objectives()) {
                            // Return, new solution is rejected
                            return Dominance::IsDominatedBy;
                        } else if new_solution.dominates(solutions[i].objectives()) {
                            // Remove dominated solution
                            solutions.swap_remove(i);
                            dominance_relation = Dominance::Dominates;
                        } else {
                            i += 1;
                        }
                    }
                } else {
                    dominance_relation = Dominance::NonDominated;
                }
            }
            Node::Internal {
                ideal,
                nadir,
                children,
                ..
            } => {
                if new_solution.is_covered_by(nadir) {
                    // Property 1
                    // Exit early, new solution is rejected
                    return Dominance::IsDominatedBy;
                } else if new_solution.covers(ideal) {
                    // Property 2
                    // Extract children keys first to avoid borrow conflicts
                    let children_to_deallocate: ArrayVec<_, C> = children.clone();
                    children.clear();
                    // Now we can call deallocate_subtree without borrow conflicts
                    for child_key in children_to_deallocate {
                        self.deallocate_subtree(child_key);
                    }
                    return Dominance::Dominates;
                } else if new_solution.is_covered_by(ideal) || new_solution.covers(nadir) {
                    // Property 4
                    // Extract children to avoid borrow conflicts
                    let children_keys: ArrayVec<_, C> = children.clone();

                    let mut dominance_relation = Dominance::NonDominated;
                    for child_key in children_keys {
                        dominance_relation = self.update_node(child_key, new_solution);
                        if dominance_relation == Dominance::IsDominatedBy {
                            // If any child rejects the solution, we can stop early
                            return dominance_relation;
                        }
                    }

                    let mut empty_indices = ArrayVec::<usize, C>::new();
                    if let Node::Internal { children, .. } = &self.arena[node_key] {
                        for (i, &child_key) in children.iter().enumerate() {
                            if self.arena[child_key].is_empty() {
                                empty_indices.push(i);
                            }
                        }
                    }
                    if let Node::Internal { children, .. } = &mut self.arena[node_key] {
                        for &i in empty_indices.iter().rev() {
                            children.swap_remove(i);
                        }
                    }
                } else {
                    dominance_relation = Dominance::NonDominated;
                }
            }
        }
        return dominance_relation;
    }

    fn insert_into(&mut self, node_key: DefaultKey, new_solution: Solution<D>) {
        self.arena[node_key].update_bounds(&new_solution);

        match &mut self.arena[node_key] {
            Node::Leaf { solutions, .. } => {
                // 2a) if the leaf is full, split it *first*, then re‑insert here:
                if solutions.is_full() {
                    self.split(node_key);
                    self.insert_into(node_key, new_solution)
                } else {
                    // 2b) otherwise safe to push
                    solutions.push(new_solution);
                }
            }
            Node::Internal { .. } => {
                let closest_child_key = self.closest_child(node_key, &new_solution);
                self.insert_into(closest_child_key, new_solution)
            }
        }
    }

    /// Turn a full leaf at `key` into an internal node with up to C children.
    fn split(&mut self, key: DefaultKey) {
        let Node::Leaf {
            solutions,
            ideal,
            middle,
            nadir,
            ..
        } = self.arena[key].clone()
        else {
            panic!("Expected a leaf node to split");
        };

        if solutions.len() <= C {
            // If the number of solutions is less than or equal to C, we can just distribute them
            let children: ArrayVec<DefaultKey, C> = solutions
                .into_iter()
                .map(|sol| {
                    let child = Node::new_leaf_with_solution(sol);
                    self.arena.insert(child)
                })
                .collect();
            self.arena[key] = Node::Internal {
                children,
                ideal,
                middle,
                nadir,
            };
            return;
        }

        let squared_distance_half_matrix: [[u64; N]; N] = array::from_fn(|i| {
            array::from_fn(|j| {
                if i >= j {
                    return 0;
                }
                solutions[i].squared_distance_to(&solutions[j].objectives)
            })
        });

        let average_distances: [u64; N] = array::from_fn(|i| {
            let total_distance: u64 = (0..solutions.len())
                .filter_map(|j| {
                    if i == j {
                        None
                    } else {
                        Some(squared_distance_half_matrix[i.min(j)][i.max(j)])
                    }
                })
                .sum();
            if solutions.len() <= 1 {
                0
            } else {
                total_distance / (solutions.len() as u64 - 1)
            }
        });

        let mut indices_sorted_by_distance_desc: [usize; N] = array::from_fn(|i| i);

        let (distant_solution_indices, first_remaining_solution_index, remaining_solution_indices) =
            indices_sorted_by_distance_desc
                .select_nth_unstable_by_key(C, |&i| std::cmp::Reverse(average_distances[i]));

        let mut distant_children_keys: ArrayVec<DefaultKey, C> = ArrayVec::new();

        // Take first C solutions with largest average distance and place them in separate children
        for distant_solution_idx in distant_solution_indices.iter() {
            let child = Node::new_leaf_with_solution(solutions[*distant_solution_idx].clone());
            let child_key = self.arena.insert(child);
            distant_children_keys.push(child_key);
        }

        // Reinsert the remaining solutions into the children
        for &solution_idx in remaining_solution_indices
            .iter()
            .chain(std::iter::once(&*first_remaining_solution_index))
        {
            let (_, closest_child_key) = distant_solution_indices
                .iter()
                .zip(distant_children_keys.iter())
                .min_by_key(|(&child_solution_idx, _)| {
                    squared_distance_half_matrix[solution_idx.min(child_solution_idx)]
                        [solution_idx.max(child_solution_idx)]
                })
                .expect("At least one child should exist");

            let solution = &solutions[solution_idx];
            self.insert_into(*closest_child_key, solution.clone());
        }

        // We should be ok to replace the original leaf with an internal node now
        self.arena[key] = Node::Internal {
            children: distant_children_keys,
            ideal,
            middle,
            nadir,
        };
    }

    pub fn iter(&self) -> NDTreeSolutionIterator<N, D, C> {
        NDTreeSolutionIterator::new(self)
    }

    pub fn len(&self) -> usize {
        self.iter().count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get an iterator over all nodes in the tree (depth-first traversal)
    pub fn node_iter(&self) -> NDTreeNodeIterator<N, D, C> {
        NDTreeNodeIterator::new(self)
    }
}

impl<const N: usize, const D: usize, const C: usize> Default for NDTree<N, D, C> {
    fn default() -> Self {
        Self::new()
    }
}

/// Consuming iterator for NDTree, yielding owned Solutions
pub struct NDTreeSolutionIntoIterator<const N: usize, const D: usize, const C: usize> {
    tree: NDTree<N, D, C>,
    stack: Vec<DefaultKey>,
    next_solution_loc: Option<(DefaultKey, usize)>,
}

impl<const N: usize, const D: usize, const C: usize> NDTreeSolutionIntoIterator<N, D, C> {
    pub fn new(tree: NDTree<N, D, C>) -> Self {
        let stack = if let Some(root_key) = tree.root {
            vec![root_key]
        } else {
            vec![]
        };

        Self {
            stack,
            tree,
            next_solution_loc: None,
        }
    }
}

impl<const N: usize, const D: usize, const C: usize> Iterator
    for NDTreeSolutionIntoIterator<N, D, C>
{
    type Item = Solution<D>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((idx, sol_idx)) = self.next_solution_loc {
                if let Node::Leaf { solutions, .. } = &mut self.tree.arena[idx] {
                    if sol_idx < solutions.len() {
                        self.next_solution_loc = Some((idx, sol_idx + 1));
                        // Remove and return the solution by value
                        return Some(solutions[sol_idx].clone());
                    }
                }
                self.next_solution_loc = None;
            }
            let idx = self.stack.pop()?;
            match &self.tree.arena[idx] {
                Node::Leaf { solutions, .. } => {
                    if !solutions.is_empty() {
                        self.next_solution_loc = Some((idx, 0));
                        continue;
                    }
                }
                Node::Internal { children, .. } => {
                    for &child_idx in children.iter().rev() {
                        self.stack.push(child_idx);
                    }
                }
            }
        }
    }
}

impl<const N: usize, const D: usize, const C: usize> IntoIterator for NDTree<N, D, C> {
    type Item = Solution<D>;
    type IntoIter = NDTreeSolutionIntoIterator<N, D, C>;

    fn into_iter(self) -> Self::IntoIter {
        NDTreeSolutionIntoIterator::new(self)
    }
}

/// Depth-First Search-like iterator for NDTree
pub struct NDTreeSolutionIterator<'a, const N: usize, const D: usize, const C: usize> {
    leaf_iterator: NDTreeNodeIterator<'a, N, D, C>,
    current_leaf_solutions: Option<&'a ArrayVec<Solution<D>, N>>,
    current_solution_index: usize,
}

impl<'a, const N: usize, const D: usize, const C: usize> NDTreeSolutionIterator<'a, N, D, C> {
    pub fn new(tree: &'a NDTree<N, D, C>) -> Self {
        Self {
            leaf_iterator: NDTreeNodeIterator::new(tree),
            current_leaf_solutions: None,
            current_solution_index: 0,
        }
    }
}

impl<'a, const N: usize, const D: usize, const C: usize> Iterator
    for NDTreeSolutionIterator<'a, N, D, C>
{
    type Item = &'a Solution<D>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have current leaf solutions, try to get the next solution from them
            if let Some(solutions) = self.current_leaf_solutions {
                if self.current_solution_index < solutions.len() {
                    let solution = &solutions[self.current_solution_index];
                    self.current_solution_index += 1;
                    return Some(solution);
                }
                // Current leaf is exhausted, move to next
                self.current_leaf_solutions = None;
                self.current_solution_index = 0;
            }

            // Get the next node from the leaf iterator
            let node = self.leaf_iterator.next()?;

            // Check if this node is a leaf with solutions
            match node {
                Node::Leaf { solutions, .. } if !solutions.is_empty() => {
                    self.current_leaf_solutions = Some(solutions);
                    self.current_solution_index = 0;
                    // Continue to the next iteration to get the first solution
                }
                _ => {
                    // Not a leaf with solutions, continue to next node
                }
            }
        }
    }
}

/// Generator for pre-order traversal of NDTree nodes (yields node references)
pub struct NDTreeNodeIterator<'a, const N: usize, const D: usize, const C: usize> {
    tree: &'a NDTree<N, D, C>,
    stack: Vec<DefaultKey>,
}

impl<'a, const N: usize, const D: usize, const C: usize> NDTreeNodeIterator<'a, N, D, C> {
    pub fn new(tree: &'a NDTree<N, D, C>) -> Self {
        let stack = if let Some(root_key) = tree.root {
            vec![root_key]
        } else {
            vec![]
        };

        Self { stack, tree }
    }
}

impl<'a, const N: usize, const D: usize, const C: usize> Iterator
    for NDTreeNodeIterator<'a, N, D, C>
{
    type Item = &'a Node<N, D, C>;

    fn next(&mut self) -> Option<Self::Item> {
        let key = self.stack.pop()?;
        let node = &self.tree.arena[key];
        if let Node::Internal { children, .. } = node {
            // Push children in reverse order so leftmost is visited first
            for &child in children.iter().rev() {
                self.stack.push(child);
            }
        }
        Some(node)
    }
}

impl<const N: usize, const D: usize, const C: usize> std::fmt::Debug for NDTree<N, D, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn write_node<const N: usize, const D: usize, const C: usize>(
            f: &mut std::fmt::Formatter<'_>,
            tree: &NDTree<N, D, C>,
            key: DefaultKey,
            prefix: &str,
            last: bool,
        ) -> std::fmt::Result {
            let node = &tree.arena[key];
            let connector = if last { "└── " } else { "├── " };
            let new_prefix = if last { "    " } else { "│   " };

            match node {
                Node::Leaf {
                    solutions,
                    ideal,
                    nadir,
                    ..
                } => {
                    writeln!(
                        f,
                        "{}{}Leaf ({} solutions, ideal: {:?}, nadir: {:?})",
                        prefix,
                        connector,
                        solutions.len(),
                        ideal,
                        nadir
                    )?;
                    for (i, solution) in solutions.iter().enumerate() {
                        writeln!(
                            f,
                            "{}{}    Solution {}: {:?}",
                            prefix, new_prefix, i, solution
                        )?;
                    }
                }
                Node::Internal {
                    children,
                    ideal,
                    nadir,
                    ..
                } => {
                    writeln!(
                        f,
                        "{}{}Internal (ideal: {:?}, nadir: {:?})",
                        prefix, connector, ideal, nadir
                    )?;
                    for (i, &child_key) in children.iter().enumerate() {
                        write_node(
                            f,
                            tree,
                            child_key,
                            &format!("{}{}", prefix, new_prefix),
                            i == children.len() - 1,
                        )?;
                    }
                }
            }
            Ok(())
        }

        writeln!(f, "ND-Tree")?;
        if let Some(root_key) = self.root {
            write_node(f, self, root_key, "", true)
        } else {
            writeln!(f, "Empty tree")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Type aliases for common test configurations
    type TestTree2D = NDTree<4, 2, 2>; // 4 solutions per leaf, 2D, 2 children
    type TestTree3D = NDTree<3, 3, 3>; // 3 solutions per leaf, 3D, 3 children
    type TestSolution2D = Solution<2>;
    type TestSolution3D = Solution<3>;

    // Helper functions for creating test solutions
    fn sol2d(x: u64, y: u64) -> TestSolution2D {
        Solution::new([x, y])
    }

    fn sol3d(x: u64, y: u64, z: u64) -> TestSolution3D {
        Solution::new([x, y, z])
    }

    #[test]
    fn test_solution_creation() {
        let sol = sol2d(10, 20);
        assert_eq!(sol.objectives, [10, 20]);
        assert_eq!(sol.objectives(), &[10, 20]);
    }

    #[test]
    fn test_solution_clone_and_debug() {
        let sol1 = sol2d(1, 2);
        let sol2 = sol1.clone();
        assert_eq!(sol1.objectives, sol2.objectives);

        // Test Debug formatting
        let debug_str = format!("{:?}", sol1);
        assert!(debug_str.contains("Solution"));
        assert!(debug_str.contains("1"));
        assert!(debug_str.contains("2"));
    }

    #[test]
    fn test_node_new_leaf_with_solution() {
        let sol = sol2d(5, 10);
        let leaf: Node<4, 2, 2> = Node::new_leaf_with_solution(sol.clone());

        match leaf {
            Node::Leaf {
                solutions,
                ideal,
                middle,
                nadir,
            } => {
                assert_eq!(solutions.len(), 1);
                assert_eq!(solutions[0].objectives, [5, 10]);
                assert_eq!(ideal, [5, 10]);
                assert_eq!(middle, [5, 10]);
                assert_eq!(nadir, [5, 10]);
            }
            _ => panic!("Expected leaf node"),
        }
    }

    #[test]
    fn test_node_update_bounds_leaf() {
        let sol1 = sol2d(5, 10);
        let mut leaf: Node<4, 2, 2> = Node::new_leaf_with_solution(sol1);

        let sol2 = sol2d(3, 15);
        leaf.update_bounds(&sol2);

        match leaf {
            Node::Leaf { ideal, nadir, .. } => {
                assert_eq!(ideal, [3, 10]); // min of each dimension
                assert_eq!(nadir, [5, 15]); // max of each dimension
            }
            _ => panic!("Expected leaf node"),
        }
    }

    #[test]
    fn test_ndtree_insert_single_solution() {
        let mut tree: TestTree2D = NDTree::new();
        let sol = sol2d(10, 20);

        tree.insert(sol.clone());

        match &tree.arena[tree.root.expect("Root should exist")] {
            Node::Leaf {
                solutions,
                ideal,
                nadir,
                ..
            } => {
                assert_eq!(solutions.len(), 1);
                assert_eq!(solutions[0].objectives, [10, 20]);
                assert_eq!(*ideal, [10, 20]);
                assert_eq!(*nadir, [10, 20]);
            }
            _ => panic!("Root should be a leaf after inserting one solution"),
        }
    }

    #[test]
    fn test_ndtree_insert_multiple_solutions_no_split() {
        let mut tree: TestTree2D = NDTree::new();

        // Insert 3 solutions (less than capacity of 4)
        tree.insert(sol2d(1, 1));
        tree.insert(sol2d(2, 3));
        tree.insert(sol2d(4, 2));

        match &tree.arena[tree.root.expect("Root should exist")] {
            Node::Leaf {
                solutions,
                ideal,
                nadir,
                ..
            } => {
                assert_eq!(solutions.len(), 3);
                assert_eq!(*ideal, [1, 1]); // min of each dimension
                assert_eq!(*nadir, [4, 3]); // max of each dimension
            }
            _ => panic!("Root should still be a leaf"),
        }
    }

    #[test]
    fn test_ndtree_insert_causes_split() {
        let mut tree: TestTree2D = NDTree::new();

        // Insert 4 solutions to fill the leaf
        tree.insert(sol2d(1, 10));
        tree.insert(sol2d(2, 8));
        tree.insert(sol2d(3, 6));
        tree.insert(sol2d(4, 4));

        // This should cause a split
        tree.insert(sol2d(5, 2));

        // Root should now be internal
        match &tree.arena[tree.root.expect("Root should exist")] {
            Node::Internal { children, .. } => {
                assert!(!children.is_empty()); // Should have children
            }
            _ => panic!("Root should be internal after split"),
        }

        // Tree should have more than 1 node now
        assert!(tree.len() > 1);
    }

    #[test]
    fn test_ndtree_iterator_empty() {
        let tree: TestTree2D = NDTree::new();
        let solutions: Vec<_> = tree.iter().collect();
        assert_eq!(solutions.len(), 0);
    }

    #[test]
    fn test_ndtree_iterator_single_solution() {
        let mut tree: TestTree2D = NDTree::new();
        tree.insert(sol2d(5, 10));

        let solutions: Vec<_> = tree.iter().collect();
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].objectives, [5, 10]);
    }

    #[test]
    fn test_ndtree_iterator_multiple_solutions() {
        let mut tree: TestTree2D = NDTree::new();
        let test_solutions = vec![sol2d(1, 1), sol2d(2, 2), sol2d(3, 3)];

        for sol in &test_solutions {
            tree.insert(sol.clone());
        }

        let collected: Vec<_> = tree.iter().collect();
        assert_eq!(collected.len(), 3);

        // Check that all solutions are present (order may vary)
        for test_sol in &test_solutions {
            assert!(collected
                .iter()
                .any(|&sol| sol.objectives == test_sol.objectives));
        }
    }

    // #[test]
    // fn test_ndtree_from_iter() {
    //     let solutions = vec![
    //         sol2d(1, 5),
    //         sol2d(2, 4),
    //         sol2d(3, 3),
    //         sol2d(4, 2),
    //         sol2d(5, 1),
    //     ];

    //     let tree: TestTree2D = NDTree::from_iter(solutions.clone());
    //     let collected: Vec<_> = tree.iter().collect();

    //     assert_eq!(collected.len(), solutions.len());
    //     for sol in &solutions {
    //         assert!(collected.iter().any(|&s| s.objectives == sol.objectives));
    //     }
    // }

    #[test]
    fn test_ndtree_into_iter() {
        let solutions = vec![sol2d(1, 1), sol2d(2, 2)];

        let mut tree: TestTree2D = NDTree::new();
        for sol in &solutions {
            tree.insert(sol.clone());
        }

        let collected: Vec<_> = tree.into_iter().collect();
        assert_eq!(collected.len(), 2);

        for sol in &solutions {
            assert!(collected.iter().any(|s| s.objectives == sol.objectives));
        }
    }

    #[test]
    fn test_ndtree_debug_formatting() {
        let mut tree: TestTree2D = NDTree::new();
        tree.insert(sol2d(1, 2));

        let debug_str = format!("{:?}", tree);
        assert!(debug_str.contains("ND-Tree"));
        assert!(debug_str.contains("Leaf"));
    }

    #[test]
    fn test_ndtree_3d() {
        let mut tree: TestTree3D = NDTree::new();

        tree.insert(sol3d(1, 2, 3));
        tree.insert(sol3d(4, 5, 6));

        let solutions: Vec<_> = tree.iter().collect();
        assert_eq!(solutions.len(), 2);

        assert!(solutions.iter().any(|&s| s.objectives == [1, 2, 3]));
        assert!(solutions.iter().any(|&s| s.objectives == [4, 5, 6]));
    }

    #[test]
    fn test_edge_case_identical_solutions() {
        let mut tree: TestTree2D = NDTree::new();
        let sol = sol2d(5, 5);

        // Insert the same solution multiple times
        for _ in 0..3 {
            tree.insert(sol.clone());
        }

        let solutions: Vec<_> = tree.iter().collect();
        assert_eq!(solutions.len(), 3); // All should be stored
        for collected_sol in solutions {
            assert_eq!(collected_sol.objectives, [5, 5]);
        }
    }

    #[test]
    fn test_edge_case_extreme_values() {
        let mut tree: TestTree2D = NDTree::new();

        // Test with u64::MAX and 0
        tree.insert(sol2d(0, 0));
        tree.insert(sol2d(u64::MAX, u64::MAX));
        tree.insert(sol2d(0, u64::MAX));
        tree.insert(sol2d(u64::MAX, 0));

        let solutions: Vec<_> = tree.iter().collect();
        assert_eq!(solutions.len(), 4);

        // Verify bounds are updated correctly
        match &tree.arena[tree.root.expect("Root should exist")] {
            Node::Leaf { ideal, nadir, .. } => {
                assert_eq!(*ideal, [0, 0]);
                assert_eq!(*nadir, [u64::MAX, u64::MAX]);
            }
            Node::Internal { ideal, nadir, .. } => {
                assert_eq!(*ideal, [0, 0]);
                assert_eq!(*nadir, [u64::MAX, u64::MAX]);
            }
        }
    }

    #[test]
    fn test_stress_test_many_insertions() {
        let mut tree: TestTree2D = NDTree::new();
        let n = 100;

        // Insert many solutions
        for i in 0..n {
            tree.insert(sol2d(i, i * 2));
        }

        let solutions: Vec<_> = tree.iter().collect();
        assert_eq!(solutions.len(), n as usize);

        // Verify all solutions are present
        for i in 0..n {
            assert!(solutions.iter().any(|&s| s.objectives == [i, i * 2]));
        }
    }

    #[test]
    fn test_bounds_consistency_after_multiple_operations() {
        let mut tree: TestTree2D = NDTree::new();

        // Insert solutions with known bounds
        let solutions = vec![sol2d(10, 50), sol2d(20, 30), sol2d(5, 60), sol2d(25, 10)];

        for sol in solutions {
            tree.insert(sol);
        }

        // Check that bounds are consistent throughout the tree
        fn check_bounds_recursive<const N: usize, const D: usize, const C: usize>(
            tree: &NDTree<N, D, C>,
            key: DefaultKey,
        ) {
            match &tree.arena[key] {
                Node::Leaf {
                    solutions,
                    ideal,
                    nadir,
                    ..
                } => {
                    if !solutions.is_empty() {
                        for d in 0..D {
                            let min_val = solutions.iter().map(|s| s.objectives[d]).min().unwrap();
                            let max_val = solutions.iter().map(|s| s.objectives[d]).max().unwrap();
                            assert_eq!(
                                ideal[d], min_val,
                                "Ideal bound incorrect at dimension {}",
                                d
                            );
                            assert_eq!(
                                nadir[d], max_val,
                                "Nadir bound incorrect at dimension {}",
                                d
                            );
                        }
                    }
                }
                Node::Internal { children, .. } => {
                    for &child_key in children.iter() {
                        check_bounds_recursive(tree, child_key);
                    }
                }
            }
        }

        check_bounds_recursive(&tree, tree.root.expect("Root should exist"));
    }

    #[test]
    fn test_tree_structure_after_splits() {
        let mut tree: TestTree2D = NDTree::new();

        // Force multiple splits by inserting many solutions
        for i in 0..20 {
            tree.insert(sol2d(i, i));
        }

        // Verify tree has internal nodes
        let has_internal = tree
            .arena
            .values()
            .any(|node| matches!(node, Node::Internal { .. }));
        assert!(
            has_internal,
            "Tree should have internal nodes after multiple splits"
        );

        // Verify all solutions are still accessible
        let solutions: Vec<_> = tree.iter().collect();
        assert_eq!(solutions.len(), 20);
    }

    #[test]
    fn test_iterator_consistency() {
        let mut tree: TestTree2D = NDTree::new();
        let test_solutions = vec![
            sol2d(1, 10),
            sol2d(2, 20),
            sol2d(3, 30),
            sol2d(4, 40),
            sol2d(5, 50),
        ];

        for sol in &test_solutions {
            tree.insert(sol.clone());
        }

        // Multiple iterations should return the same solutions
        let iter1: Vec<_> = tree.iter().map(|s| s.objectives).collect();
        let iter2: Vec<_> = tree.iter().map(|s| s.objectives).collect();

        assert_eq!(iter1.len(), iter2.len());
        for obj in iter1 {
            assert!(iter2.contains(&obj));
        }
    }

    #[test]
    fn test_different_tree_configurations() {
        // Test with different N, D, C values
        type SmallTree = NDTree<2, 2, 2>; // Very small capacity
        type LargeTree = NDTree<10, 2, 4>; // Larger capacity

        let mut small_tree = SmallTree::new();
        let mut large_tree = LargeTree::new();

        let solutions = vec![sol2d(1, 3), sol2d(2, 2), sol2d(3, 1)];

        for sol in &solutions {
            small_tree.update(sol.clone());
            large_tree.update(sol.clone());
        }

        // Both should contain all solutions
        assert_eq!(small_tree.iter().count(), 3);
        assert_eq!(large_tree.iter().count(), 3);

        // Small tree should have split (capacity 2), large tree should not
        let small_has_internal = small_tree
            .arena
            .values()
            .any(|n| matches!(n, Node::Internal { .. }));
        let large_has_internal = large_tree
            .arena
            .values()
            .any(|n| matches!(n, Node::Internal { .. }));

        assert!(small_has_internal, "Small tree should have split");
        assert!(!large_has_internal, "Large tree should not have split yet");
    }

    #[test]
    fn test_empty_tree_operations() {
        let tree: TestTree2D = NDTree::new();

        // Test all operations on empty tree
        assert_eq!(tree.iter().count(), 0);
        assert_eq!(tree.len(), 0);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_zero_objective_values() {
        let mut tree: TestTree2D = NDTree::new();
        tree.insert(sol2d(0, 0));
        tree.insert(sol2d(0, 1));
        tree.insert(sol2d(1, 0));

        let solutions: Vec<_> = tree.iter().collect();
        assert_eq!(solutions.len(), 3);

        // Check bounds are correct
        match &tree.arena[tree.root.expect("Root should exist")] {
            Node::Leaf { ideal, nadir, .. } => {
                assert_eq!(*ideal, [0, 0]);
                assert_eq!(*nadir, [1, 1]);
            }
            _ => panic!("Should be leaf node"),
        }
    }

    #[test]
    fn test_single_dimension_differences() {
        let mut tree: TestTree2D = NDTree::new();

        // Solutions that differ in only one dimension
        tree.insert(sol2d(5, 10));
        tree.insert(sol2d(6, 10));
        tree.insert(sol2d(7, 10));
        tree.insert(sol2d(8, 10));

        let solutions: Vec<_> = tree.iter().collect();
        assert_eq!(solutions.len(), 4);

        // All should have the same y-coordinate
        for sol in solutions {
            assert_eq!(sol.objectives[1], 10);
        }
    }

    #[test]
    fn test_very_large_tree() {
        let mut tree: TestTree2D = NDTree::new();
        let n = 1000;

        // Insert a lot of solutions
        for i in 0..n {
            tree.insert(sol2d(i % 100, (i * 2) % 100));
        }

        let solutions: Vec<_> = tree.iter().collect();
        assert_eq!(solutions.len(), n as usize);

        // Verify tree has internal nodes (splits occurred)
        let has_internal = tree
            .arena
            .values()
            .any(|node| matches!(node, Node::Internal { .. }));
        assert!(has_internal, "Large tree should have internal nodes");
    }

    #[test]
    fn test_from_iter_empty() {
        let empty_vec: Vec<TestSolution2D> = vec![];
        let mut tree: TestTree2D = NDTree::new();
        for sol in empty_vec {
            tree.insert(sol);
        }

        assert_eq!(tree.iter().count(), 0);
        matches!(tree.root, None);
    }

    #[test]
    fn test_from_iter_single() {
        let single_vec = vec![sol2d(42, 24)];
        let mut tree: TestTree2D = NDTree::new();
        for sol in single_vec {
            tree.insert(sol);
        }

        let solutions: Vec<_> = tree.iter().collect();
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].objectives, [42, 24]);
    }

    #[test]
    fn test_bounds_propagation_after_splits() {
        let mut tree: TestTree2D = NDTree::new();

        // Insert solutions that will create a specific bounds pattern
        tree.insert(sol2d(0, 100)); // extreme high y
        tree.insert(sol2d(100, 0)); // extreme high x
        tree.insert(sol2d(50, 50)); // middle
        tree.insert(sol2d(25, 75)); //
        tree.insert(sol2d(75, 25)); // force split

        // Check that root bounds encompass all solutions
        match &tree.arena[tree.root.expect("Root should exist")] {
            Node::Leaf { ideal, nadir, .. } => {
                assert_eq!(*ideal, [0, 0]);
                assert_eq!(*nadir, [100, 100]);
            }
            Node::Internal { ideal, nadir, .. } => {
                assert_eq!(*ideal, [0, 0]);
                assert_eq!(*nadir, [100, 100]);
            }
        }
    }

    #[test]
    fn test_iterator_order_consistency() {
        let mut tree: TestTree2D = NDTree::new();

        // Insert in one order
        let original_order = vec![sol2d(3, 1), sol2d(1, 3), sol2d(2, 2), sol2d(4, 4)];

        for sol in &original_order {
            tree.insert(sol.clone());
        }

        // Collect multiple times - should be consistent
        let iter1: Vec<_> = tree.iter().map(|s| s.objectives).collect();
        let iter2: Vec<_> = tree.iter().map(|s| s.objectives).collect();
        let iter3: Vec<_> = tree.iter().map(|s| s.objectives).collect();

        assert_eq!(iter1, iter2);
        assert_eq!(iter2, iter3);

        // All original solutions should be present
        for sol in &original_order {
            assert!(iter1.contains(&sol.objectives));
        }
    }

    #[test]
    fn test_into_iter_consumes_tree() {
        let mut tree: TestTree2D = NDTree::new();
        tree.insert(sol2d(1, 2));
        tree.insert(sol2d(3, 4));

        let solutions: Vec<_> = tree.into_iter().collect();
        assert_eq!(solutions.len(), 2);

        // Tree is consumed - can't use it anymore
        // This test just verifies the into_iter works without compile errors
    }

    #[test]
    fn test_mixed_value_ranges() {
        let mut tree: TestTree2D = NDTree::new();

        // Mix small and large values, but avoid overflow in squared distance
        tree.insert(sol2d(1, 1_000_000));
        tree.insert(sol2d(1_000_000, 1));
        tree.insert(sol2d(500_000, 500_000));
        tree.insert(sol2d(0, 0));
        tree.insert(sol2d(2_000_000, 2_000_000)); // Use smaller values to avoid overflow

        let solutions: Vec<_> = tree.iter().collect();
        assert_eq!(solutions.len(), 5);

        // Check that bounds are correct
        let all_x: Vec<_> = solutions.iter().map(|s| s.objectives[0]).collect();
        let all_y: Vec<_> = solutions.iter().map(|s| s.objectives[1]).collect();

        assert_eq!(*all_x.iter().min().unwrap(), 0);
        assert_eq!(*all_x.iter().max().unwrap(), 2_000_000);
        assert_eq!(*all_y.iter().min().unwrap(), 0);
        assert_eq!(*all_y.iter().max().unwrap(), 2_000_000);
    }

    #[test]
    fn test_high_dimensional_tree() {
        type HighDimTree = NDTree<4, 5, 3>; // 5 dimensions

        let mut tree: HighDimTree = NDTree::new();

        let sol1 = Solution::new([1, 2, 3, 4, 5]);
        let sol2 = Solution::new([6, 7, 8, 9, 10]);
        let sol3 = Solution::new([11, 12, 13, 14, 15]);

        tree.insert(sol1);
        tree.insert(sol2);
        tree.insert(sol3);

        let solutions: Vec<_> = tree.iter().collect();
        assert_eq!(solutions.len(), 3);

        // Check that all dimensions are handled correctly
        assert!(solutions.iter().any(|&s| s.objectives == [1, 2, 3, 4, 5]));
        assert!(solutions.iter().any(|&s| s.objectives == [6, 7, 8, 9, 10]));
        assert!(solutions
            .iter()
            .any(|&s| s.objectives == [11, 12, 13, 14, 15]));
    }

    #[test]
    fn test_tree_with_small_capacity() {
        type SmallTree = NDTree<2, 2, 2>; // 2 solutions per leaf, reasonable capacity

        let mut tree: SmallTree = NDTree::new();

        // Insert enough solutions to cause splits
        tree.insert(sol2d(1, 1));
        tree.insert(sol2d(2, 2));
        tree.insert(sol2d(3, 3)); // This should cause split
        tree.insert(sol2d(4, 4)); // This might cause another split

        let solutions: Vec<_> = tree.iter().collect();
        assert_eq!(solutions.len(), 4);

        // Tree should have internal nodes due to small capacity
        let has_internal = tree
            .arena
            .values()
            .any(|node| matches!(node, Node::Internal { .. }));
        assert!(
            has_internal,
            "Tree with small capacity should have internal nodes"
        );
    }

    #[test]
    fn test_node_clone() {
        let sol = sol2d(10, 20);
        let node: Node<4, 2, 2> = Node::new_leaf_with_solution(sol);
        let cloned_node = node.clone();

        // Both nodes should be equivalent
        match (&node, &cloned_node) {
            (
                Node::Leaf {
                    solutions: s1,
                    ideal: i1,
                    nadir: n1,
                    ..
                },
                Node::Leaf {
                    solutions: s2,
                    ideal: i2,
                    nadir: n2,
                    ..
                },
            ) => {
                assert_eq!(s1.len(), s2.len());
                assert_eq!(s1[0].objectives, s2[0].objectives);
                assert_eq!(i1, i2);
                assert_eq!(n1, n2);
            }
            _ => panic!("Both should be leaf nodes"),
        }
    }

    #[test]
    fn test_update_bounds_multiple_times() {
        let mut node: Node<4, 2, 2> = Node::new_leaf_with_solution(sol2d(50, 50));

        // Update bounds with different solutions
        node.update_bounds(&sol2d(10, 90)); // New min x, new max y
        node.update_bounds(&sol2d(90, 10)); // New max x, new min y
        node.update_bounds(&sol2d(20, 80)); // No change to bounds

        match node {
            Node::Leaf { ideal, nadir, .. } => {
                assert_eq!(ideal, [10, 10]);
                assert_eq!(nadir, [90, 90]);
            }
            _ => panic!("Should be leaf node"),
        }
    }

    #[test]
    fn test_node_iterator() {
        let mut tree: TestTree2D = NDTree::new();

        // Add enough solutions to cause splits
        for i in 0..10 {
            tree.insert(sol2d(i, i));
        }

        let leaf_count = tree
            .node_iter()
            .filter(|node| matches!(node, Node::Leaf { .. }))
            .count();

        let total_nodes = tree.node_iter().count();

        // Should have at least one leaf
        assert!(leaf_count > 0);
        // Should have some internal nodes after splits
        assert!(total_nodes > leaf_count);
    }
}
