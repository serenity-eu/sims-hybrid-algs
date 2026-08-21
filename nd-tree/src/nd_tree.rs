use arrayvec::ArrayVec;
use slotmap::{DefaultKey, SlotMap};
use std::{array, cmp::Ordering, marker::PhantomData, vec};

use pareto::{Dominance, HasObjectives, MoSolution, Objectives};
use tracing::trace;

/// A solution in D-dimensional objective space.
#[derive(Clone, std::fmt::Debug, PartialEq, Eq)]
pub struct Solution<const D: usize> {
    pub objectives: Objectives<D>,
}

impl<const D: usize> Solution<D> {
    #[must_use]
    pub const fn new(objectives: Objectives<D>) -> Self {
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
#[derive(Clone)]
pub enum Node<T, const N: usize, const D: usize, const C: usize>
where
    T: HasObjectives<D> + MoSolution<D> + Clone,
{
    Leaf {
        solutions: ArrayVec<T, N>,
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

impl<T, const N: usize, const D: usize, const C: usize> Node<T, N, D, C>
where
    T: HasObjectives<D> + MoSolution<D> + Clone,
{
    /// Start new leaf with single solution
    pub fn new_leaf_with_solution(solution: T) -> Self {
        let objectives = *solution.objectives();

        let mut solutions = ArrayVec::new();
        solutions.push(solution);

        Self::Leaf {
            solutions,
            ideal: objectives,
            middle: objectives,
            nadir: objectives,
        }
    }

    /// Component‑wise update of ideal/nadir from one more solution.
    pub fn update_bounds(&mut self, sol: &T) {
        match self {
            Self::Leaf { ideal, nadir, .. } | Self::Internal { ideal, nadir, .. } => {
                *ideal = array::from_fn(|i| ideal[i].min(sol.objectives()[i]));
                *nadir = array::from_fn(|i| nadir[i].max(sol.objectives()[i]));
            }
        }
    }

    pub const fn is_empty(&self) -> bool {
        match self {
            Self::Leaf { solutions, .. } => solutions.is_empty(),
            Self::Internal { children, .. } => children.is_empty(),
        }
    }
}

/// Result of a branch-and-bound scalarized query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarizedQueryStats {
    pub visited_nodes: usize,
    pub pruned_nodes: usize,
    pub evaluated_solutions: usize,
}

/// Borrowed result of a scalarized query.
#[derive(Debug, Clone, Copy)]
pub struct ScalarizedQueryResult<'a, T> {
    pub solution: &'a T,
    pub score: f64,
    pub stats: ScalarizedQueryStats,
}

/// Immutable closures driving a scalarized branch-and-bound query, threaded
/// through the recursion so [`NDTree::scalarized_query_node`] stays low-arity.
struct ScalarizedQueryCtx<'f, Accept, NodeLowerBound, SolutionScore> {
    accept: &'f mut Accept,
    node_lower_bound: &'f NodeLowerBound,
    solution_score: &'f SolutionScore,
}

/// Mutable accumulators for a scalarized branch-and-bound query.
struct ScalarizedQueryState<'a, T> {
    best_solution: Option<&'a T>,
    best_score: f64,
    stats: ScalarizedQueryStats,
}

/// The whole tree, storing nodes in an arena Vec and pointing to root by index.
#[derive(Clone)]
pub struct NDTree<T, const N: usize, const D: usize, const C: usize>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq,
{
    arena: SlotMap<DefaultKey, Node<T, N, D, C>>,
    root: Option<DefaultKey>,
    _phantom: PhantomData<T>,
}

impl<T, const N: usize, const D: usize, const C: usize> NDTree<T, N, D, C>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq,
{
    /// Create an empty tree
    #[must_use]
    pub fn new() -> Self {
        Self {
            arena: SlotMap::new(),
            root: None,
            _phantom: PhantomData,
        }
    }

    /// Update the tree with a new solution.
    /// Returns true if the solution was added, false otherwise.
    pub fn update(&mut self, new_solution: T) -> bool {
        if let Some(root_key) = self.root {
            if self.update_node(root_key, &new_solution) == Dominance::IsDominated {
                false
            } else {
                if self.arena[root_key].is_empty() {
                    // If the root node is empty after update, deallocate it
                    self.deallocate_root();
                }
                // If the solution is not dominated, we can insert it
                self.insert(new_solution);
                true
            }
        } else {
            self.insert(new_solution);
            true
        }
    }

    pub fn update_unchecked(&mut self, new_solution: T) {
        // This method is unsafe because it does not check the Pareto dominance relation
        self.insert(new_solution);
    }

    /// Remove all solutions in the tree that are dominated by `dominator`, without inserting it.
    /// This is the pruning-only half of `update`: identical domination scan, no insertion step.
    pub fn remove_dominated(&mut self, dominator: &T) {
        let Some(root_key) = self.root else { return };
        self.update_node(root_key, dominator);
        if let Some(root_key) = self.root {
            if self.arena[root_key].is_empty() {
                self.deallocate_root();
            }
        }
    }

    fn insert(&mut self, sol: T) {
        if let Some(root_key) = self.root {
            // If the root exists, insert into it
            self.insert_into(root_key, sol);
        } else {
            // If the root does not exist, create a new root with the solution
            let leaf_node = Node::new_leaf_with_solution(sol);
            self.root = Some(self.arena.insert(leaf_node));
        }
    }

    fn closest_child(&self, node_key: DefaultKey, solution: &T) -> DefaultKey {
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
                for &child_key in children {
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

    fn deallocate_root(&mut self) {
        if let Some(root_key) = self.root {
            self.arena.remove(root_key);
            self.root = None;
        }
    }

    fn update_node(&mut self, node_key: DefaultKey, new_solution: &T) -> Dominance {
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
                    return Dominance::IsDominated;
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
                            return Dominance::IsDominated;
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
                    return Dominance::IsDominated;
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

                    for child_key in children_keys {
                        dominance_relation = self.update_node(child_key, new_solution);
                        if dominance_relation == Dominance::IsDominated {
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
        dominance_relation
    }

    fn scalarized_query_node<'a, Accept, NodeLowerBound, SolutionScore>(
        &'a self,
        node_key: DefaultKey,
        ctx: &mut ScalarizedQueryCtx<'_, Accept, NodeLowerBound, SolutionScore>,
        state: &mut ScalarizedQueryState<'a, T>,
    ) where
        Accept: FnMut(&T) -> bool,
        NodeLowerBound: Fn(&Objectives<D>) -> f64,
        SolutionScore: Fn(&T) -> f64,
    {
        state.stats.visited_nodes += 1;

        match &self.arena[node_key] {
            Node::Leaf {
                solutions, ideal, ..
            } => {
                let lower_bound = (ctx.node_lower_bound)(ideal);
                if lower_bound >= state.best_score {
                    state.stats.pruned_nodes += 1;
                    return;
                }

                for solution in solutions {
                    if !(ctx.accept)(solution) {
                        continue;
                    }

                    state.stats.evaluated_solutions += 1;
                    let score = (ctx.solution_score)(solution);
                    if score < state.best_score {
                        state.best_score = score;
                        state.best_solution = Some(solution);
                    }
                }
            }
            Node::Internal { children, .. } => {
                let mut child_bounds: ArrayVec<(DefaultKey, f64), C> = ArrayVec::new();

                for &child_key in children {
                    let child_ideal = match &self.arena[child_key] {
                        Node::Leaf { ideal, .. } | Node::Internal { ideal, .. } => ideal,
                    };
                    let lower_bound = (ctx.node_lower_bound)(child_ideal);
                    child_bounds.push((child_key, lower_bound));
                }

                child_bounds
                    .sort_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal));

                for (child_key, lower_bound) in child_bounds {
                    if lower_bound >= state.best_score {
                        state.stats.pruned_nodes += 1;
                        continue;
                    }

                    self.scalarized_query_node(child_key, ctx, state);
                }
            }
        }
    }

    /// Find the best acceptable solution under a scalarized objective using
    /// branch-and-bound pruning over subtree ideal points.
    ///
    /// `node_lower_bound` must be a valid lower bound when evaluated on a
    /// subtree's component-wise ideal point. `solution_score` must return the
    /// exact scalarized score for a concrete solution. Lower scores are better.
    #[must_use]
    pub fn find_best_with_pruning<Accept, NodeLowerBound, SolutionScore>(
        &self,
        mut accept: Accept,
        node_lower_bound: NodeLowerBound,
        solution_score: SolutionScore,
    ) -> Option<ScalarizedQueryResult<'_, T>>
    where
        Accept: FnMut(&T) -> bool,
        NodeLowerBound: Fn(&Objectives<D>) -> f64,
        SolutionScore: Fn(&T) -> f64,
    {
        let root_key = self.root?;
        let mut ctx = ScalarizedQueryCtx {
            accept: &mut accept,
            node_lower_bound: &node_lower_bound,
            solution_score: &solution_score,
        };
        let mut state = ScalarizedQueryState {
            best_solution: None,
            best_score: f64::INFINITY,
            stats: ScalarizedQueryStats {
                visited_nodes: 0,
                pruned_nodes: 0,
                evaluated_solutions: 0,
            },
        };

        self.scalarized_query_node(root_key, &mut ctx, &mut state);

        state.best_solution.map(|solution| ScalarizedQueryResult {
            solution,
            score: state.best_score,
            stats: state.stats,
        })
    }

    fn insert_into(&mut self, node_key: DefaultKey, new_solution: T) {
        self.arena[node_key].update_bounds(&new_solution);

        match &mut self.arena[node_key] {
            Node::Leaf { solutions, .. } => {
                // 2a) if the leaf is full, split it *first*, then re‑insert here:
                if solutions.is_full() {
                    self.split(node_key);
                    self.insert_into(node_key, new_solution);
                } else {
                    // 2b) otherwise safe to push
                    solutions.push(new_solution);
                }
            }
            Node::Internal { .. } => {
                let closest_child_key = self.closest_child(node_key, &new_solution);
                self.insert_into(closest_child_key, new_solution);
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
                    let child: Node<T, N, D, C> = Node::new_leaf_with_solution(sol);
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
                solutions[i].squared_distance_to(solutions[j].objectives())
            })
        });

        let average_distances: [u64; N] = array::from_fn(|i| {
            // Saturating, not `.sum()`: this is a node-splitting balance
            // heuristic, not correctness-critical, and summing up to N-1
            // already-large squared distances (see squared_distance_to) can
            // genuinely exceed u64::MAX on instances with large objective
            // magnitudes -- saturating still means "very far apart".
            let total_distance: u64 = (0..solutions.len())
                .filter_map(|j| {
                    if i == j {
                        None
                    } else {
                        Some(squared_distance_half_matrix[i.min(j)][i.max(j)])
                    }
                })
                .fold(0u64, u64::saturating_add);
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

    #[must_use]
    /// Is `sol` dominated by any solution currently in the tree?
    ///
    /// Read-only twin of the domination test `update_node` performs while
    /// inserting, using the same node bounds to prune whole subtrees instead
    /// of scanning every solution (which is what `iter()` would do).
    ///
    /// Pruning rule: every solution `s` in a subtree satisfies `ideal <= s`,
    /// so a solution with `s <= sol` can exist only if `ideal <= sol`. When
    /// that fails, nothing in the subtree can dominate `sol` and the whole
    /// branch is skipped.
    #[must_use]
    pub fn is_dominated_by_any(&self, sol: &T) -> bool {
        let Some(root) = self.root else {
            return false;
        };
        let mut stack = vec![root];
        while let Some(key) = stack.pop() {
            match &self.arena[key] {
                Node::Leaf {
                    solutions, ideal, ..
                } => {
                    if !sol.is_covered_by(ideal) {
                        continue;
                    }
                    if solutions.iter().any(|s| s.covers(sol.objectives())) {
                        return true;
                    }
                }
                Node::Internal {
                    children, ideal, ..
                } => {
                    if !sol.is_covered_by(ideal) {
                        continue;
                    }
                    stack.extend(children.iter().copied());
                }
            }
        }
        false
    }

    /// Is `sol` *strictly* dominated by any solution in the tree?
    ///
    /// Same pruning as [`Self::is_dominated_by_any`], but equality does not
    /// count as domination. Callers that treat equal-objective solutions as
    /// distinct (the trace format does: two different image selections can
    /// share objectives, and both are recorded) need this variant.
    #[must_use]
    pub fn is_strictly_dominated_by_any(&self, sol: &T) -> bool {
        let Some(root) = self.root else {
            return false;
        };
        let mut stack = vec![root];
        while let Some(key) = stack.pop() {
            match &self.arena[key] {
                Node::Leaf {
                    solutions, ideal, ..
                } => {
                    if !sol.is_covered_by(ideal) {
                        continue;
                    }
                    if solutions.iter().any(|s| s.dominates(sol.objectives())) {
                        return true;
                    }
                }
                Node::Internal {
                    children, ideal, ..
                } => {
                    if !sol.is_covered_by(ideal) {
                        continue;
                    }
                    stack.extend(children.iter().copied());
                }
            }
        }
        false
    }

    /// Call `f` for every solution in the tree that `sol` strictly dominates.
    ///
    /// Pruning rule: every solution `s` in a subtree satisfies `s <= nadir`,
    /// so a solution with `sol <= s` can exist only if `sol <= nadir`. When
    /// that fails, `sol` dominates nothing in the subtree and the branch is
    /// skipped. Domination is still tested per solution rather than assumed
    /// for a fully covered node, because `covers` admits equality while
    /// `dominates` does not.
    pub fn for_each_dominated_by<F>(&self, sol: &T, mut f: F)
    where
        F: FnMut(&T),
    {
        let Some(root) = self.root else {
            return;
        };
        let mut stack = vec![root];
        while let Some(key) = stack.pop() {
            match &self.arena[key] {
                Node::Leaf {
                    solutions, nadir, ..
                } => {
                    if !sol.covers(nadir) {
                        continue;
                    }
                    for s in solutions {
                        if sol.dominates(s.objectives()) {
                            f(s);
                        }
                    }
                }
                Node::Internal {
                    children, nadir, ..
                } => {
                    if !sol.covers(nadir) {
                        continue;
                    }
                    stack.extend(children.iter().copied());
                }
            }
        }
    }

    pub fn iter(&self) -> NDTreeSolutionIterator<'_, T, N, D, C> {
        NDTreeSolutionIterator::new(self)
    }

    #[must_use]
    pub fn contains(&self, solution: &T) -> bool {
        trace!(
            "contains() called for solution: {:?}",
            solution.objectives()
        );

        let Some(root_key) = self.root else {
            trace!("  No root, returning false");
            return false;
        };

        let mut stack = vec![root_key];
        let mut nodes_visited = 0;
        let mut nodes_skipped = 0;

        while let Some(node_key) = stack.pop() {
            nodes_visited += 1;
            let node = &self.arena[node_key];

            // Get node bounds
            let (ideal, nadir) = match node {
                Node::Leaf { ideal, nadir, .. } | Node::Internal { ideal, nadir, .. } => {
                    (ideal, nadir)
                }
            };

            trace!(
                "  Visiting node #{}, ideal: {:?}, nadir: {:?}",
                nodes_visited,
                ideal,
                nadir
            );

            // Bounds checking: skip this subtree if solution is outside [ideal, nadir]
            // For minimization: ideal has the best (smallest) values, nadir has worst (largest) values
            // A solution is within bounds if all objectives are in range [ideal[i], nadir[i]]
            // Note: We cannot use covers/is_covered_by here because they check dominance relations,
            // not coordinate-wise bounds. A solution at [10,30] with ideal=[10,10] and nadir=[30,30]
            // is within bounds coordinate-wise, but doesn't dominate ideal.
            let within_bounds = solution
                .objectives()
                .iter()
                .zip(ideal.iter().zip(nadir.iter()))
                .all(|(sol_obj, (ideal_obj, nadir_obj))| {
                    // For minimization: ideal <= solution <= nadir
                    sol_obj >= ideal_obj && sol_obj <= nadir_obj
                });

            trace!(
                "    within_bounds={} (checking if {:?} is in [{:?}, {:?}])",
                within_bounds,
                solution.objectives(),
                ideal,
                nadir
            );

            if !within_bounds {
                nodes_skipped += 1;
                trace!("    SKIPPING this subtree (solution outside bounds)");
                continue; // Solution cannot be in this subtree, skip
            }

            match node {
                Node::Leaf { solutions, .. } => {
                    trace!("    Leaf node with {} solutions", solutions.len());
                    for (i, sol) in solutions.iter().enumerate() {
                        trace!("      Solution[{}]: {:?}", i, sol.objectives());
                    }

                    // Check if the solution exists in this leaf
                    if solutions.contains(solution) {
                        trace!(
                            "  FOUND! Visited {} nodes, skipped {}",
                            nodes_visited,
                            nodes_skipped
                        );
                        return true;
                    }
                    trace!("    Not found in this leaf");
                }
                Node::Internal { children, .. } => {
                    trace!("    Internal node with {} children", children.len());
                    // Add children to stack for further exploration
                    for &child_key in children {
                        stack.push(child_key);
                    }
                }
            }
        }

        trace!(
            "  NOT FOUND after visiting {} nodes, skipped {}",
            nodes_visited,
            nodes_skipped
        );
        false
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.iter().count()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T, const N: usize, const D: usize, const C: usize> IntoIterator for &NDTree<T, N, D, C>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq,
{
    type Item = T;
    type IntoIter = NDTreeSolutionIntoIterator<T, N, D, C>;

    fn into_iter(self) -> Self::IntoIter {
        NDTreeSolutionIntoIterator::new(self.clone())
    }
}

impl<T, const N: usize, const D: usize, const C: usize> Default for NDTree<T, N, D, C>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Consuming iterator for `NDTree`, yielding owned Solutions
pub struct NDTreeSolutionIntoIterator<T, const N: usize, const D: usize, const C: usize>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq,
{
    leaf_iterator: NDTreeNodeIntoIterator<T, N, D, C>,
    current_leaf_solutions: Option<ArrayVec<T, N>>,
    current_solution_index: usize,
}

impl<T, const N: usize, const D: usize, const C: usize> NDTreeSolutionIntoIterator<T, N, D, C>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq,
{
    #[must_use]
    pub fn new(tree: NDTree<T, N, D, C>) -> Self {
        Self {
            leaf_iterator: NDTreeNodeIntoIterator::new(tree),
            current_leaf_solutions: None,
            current_solution_index: 0,
        }
    }
}

impl<T, const N: usize, const D: usize, const C: usize> Iterator
    for NDTreeSolutionIntoIterator<T, N, D, C>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have current leaf solutions, try to get the next solution from them
            if let Some(ref mut solutions) = self.current_leaf_solutions {
                if self.current_solution_index < solutions.len() {
                    let solution = solutions[self.current_solution_index].clone();
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

impl<T, const N: usize, const D: usize, const C: usize> IntoIterator for NDTree<T, N, D, C>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq,
{
    type Item = T;
    type IntoIter = NDTreeSolutionIntoIterator<T, N, D, C>;

    fn into_iter(self) -> Self::IntoIter {
        NDTreeSolutionIntoIterator::new(self)
    }
}

/// Iterator for `NDTree`
pub struct NDTreeSolutionIterator<'a, T, const N: usize, const D: usize, const C: usize>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq,
{
    leaf_iterator: NDTreeNodeIterator<'a, T, N, D, C>,
    current_leaf_solutions: Option<&'a ArrayVec<T, N>>,
    current_solution_index: usize,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T, const N: usize, const D: usize, const C: usize> NDTreeSolutionIterator<'a, T, N, D, C>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq,
{
    #[must_use]
    pub fn new(tree: &'a NDTree<T, N, D, C>) -> Self {
        Self {
            leaf_iterator: NDTreeNodeIterator::new(tree),
            current_leaf_solutions: None,
            current_solution_index: 0,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T, const N: usize, const D: usize, const C: usize> Iterator
    for NDTreeSolutionIterator<'a, T, N, D, C>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq,
{
    type Item = &'a T;

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

/// Generator for pre-order traversal of `NDTree` nodes (yields node references)
pub struct NDTreeNodeIterator<'a, T, const N: usize, const D: usize, const C: usize>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq,
{
    tree: &'a NDTree<T, N, D, C>,
    stack: Vec<DefaultKey>,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T, const N: usize, const D: usize, const C: usize> NDTreeNodeIterator<'a, T, N, D, C>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq,
{
    #[must_use]
    pub fn new(tree: &'a NDTree<T, N, D, C>) -> Self {
        let stack = tree.root.map_or_else(Vec::new, |root_key| vec![root_key]);

        Self {
            stack,
            tree,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T, const N: usize, const D: usize, const C: usize> Iterator
    for NDTreeNodeIterator<'a, T, N, D, C>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq,
{
    type Item = &'a Node<T, N, D, C>;

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

/// Generator for pre-order traversal of `NDTree` nodes (yields owned nodes)
pub struct NDTreeNodeIntoIterator<T, const N: usize, const D: usize, const C: usize>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq,
{
    tree: NDTree<T, N, D, C>,
    stack: Vec<DefaultKey>,
}

impl<T, const N: usize, const D: usize, const C: usize> NDTreeNodeIntoIterator<T, N, D, C>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq,
{
    #[must_use]
    pub fn new(tree: NDTree<T, N, D, C>) -> Self {
        let stack = tree.root.map_or_else(Vec::new, |root_key| vec![root_key]);

        Self { tree, stack }
    }
}

impl<T, const N: usize, const D: usize, const C: usize> Iterator
    for NDTreeNodeIntoIterator<T, N, D, C>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq,
{
    type Item = Node<T, N, D, C>;

    fn next(&mut self) -> Option<Self::Item> {
        let key = self.stack.pop()?;
        let node = self.tree.arena.remove(key)?;
        if let Node::Internal { children, .. } = &node {
            // Push children in reverse order so leftmost is visited first
            for &child in children.iter().rev() {
                self.stack.push(child);
            }
        }
        Some(node)
    }
}

impl<T, const N: usize, const D: usize, const C: usize> std::fmt::Debug for NDTree<T, N, D, C>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn write_node<T, const N: usize, const D: usize, const C: usize>(
            f: &mut std::fmt::Formatter<'_>,
            tree: &NDTree<T, N, D, C>,
            key: DefaultKey,
            prefix: &str,
            last: bool,
        ) -> std::fmt::Result
        where
            T: HasObjectives<D> + MoSolution<D> + Clone + PartialEq + std::fmt::Debug,
        {
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
                        writeln!(f, "{prefix}{new_prefix}    Solution {i}: {solution:?}")?;
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
                        "{prefix}{connector}Internal (ideal: {ideal:?}, nadir: {nadir:?})"
                    )?;
                    for (i, &child_key) in children.iter().enumerate() {
                        write_node(
                            f,
                            tree,
                            child_key,
                            &format!("{prefix}{new_prefix}"),
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

impl<T, const N: usize, const D: usize, const C: usize> std::fmt::Debug for Node<T, N, D, C>
where
    T: HasObjectives<D> + MoSolution<D> + Clone + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Leaf {
                solutions,
                ideal,
                middle,
                nadir,
            } => f
                .debug_struct("Leaf")
                .field("solutions", &solutions)
                .field("ideal", &ideal)
                .field("middle", &middle)
                .field("nadir", &nadir)
                .finish(),
            Self::Internal {
                children,
                ideal,
                middle,
                nadir,
            } => f
                .debug_struct("Internal")
                .field("children", &children)
                .field("ideal", &ideal)
                .field("middle", &middle)
                .field("nadir", &nadir)
                .finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Type aliases for common test configurations
    type TestTree2D = NDTree<TestSolution2D, 4, 2, 2>; // 4 solutions per leaf, 2D, 2 children
    type TestTree3D = NDTree<TestSolution3D, 3, 3, 3>; // 3 solutions per leaf, 3D, 3 children
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
        let debug_str = format!("{sol1:?}");
        assert!(debug_str.contains("Solution"));
        assert!(debug_str.contains('1'));
        assert!(debug_str.contains('2'));
    }

    #[test]
    fn test_node_new_leaf_with_solution() {
        let sol = sol2d(5, 10);
        let leaf: Node<TestSolution2D, 4, 2, 2> = Node::new_leaf_with_solution(sol);

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
            Node::Internal { .. } => panic!("Expected leaf node"),
        }
    }

    #[test]
    fn test_node_update_bounds_leaf() {
        let sol1 = sol2d(5, 10);
        let mut leaf: Node<TestSolution2D, 4, 2, 2> = Node::new_leaf_with_solution(sol1);

        let sol2 = sol2d(3, 15);
        leaf.update_bounds(&sol2);

        match leaf {
            Node::Leaf { ideal, nadir, .. } => {
                assert_eq!(ideal, [3, 10]); // min of each dimension
                assert_eq!(nadir, [5, 15]); // max of each dimension
            }
            Node::Internal { .. } => panic!("Expected leaf node"),
        }
    }

    #[test]
    fn test_ndtree_insert_single_solution() {
        let mut tree: TestTree2D = NDTree::new();
        let sol = sol2d(10, 20);

        tree.insert(sol);

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
            Node::Internal { .. } => panic!("Root should be a leaf after inserting one solution"),
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
            Node::Internal { .. } => panic!("Root should still be a leaf"),
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
            Node::Leaf { .. } => panic!("Root should be internal after split"),
        }

        // Tree should have more than 1 node now
        assert!(tree.len() > 1);
    }

    #[test]
    fn test_ndtree_iterator_empty() {
        let tree: TestTree2D = NDTree::new();

        assert_eq!(tree.iter().count(), 0);
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

        let collected: Vec<TestSolution2D> = tree.iter().cloned().collect();
        assert_eq!(collected.len(), 3);

        // Check that all solutions are present (order may vary)
        for test_sol in &test_solutions {
            assert!(collected
                .iter()
                .any(|s| s.objectives() == test_sol.objectives()));
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

        let debug_str = format!("{tree:?}");
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

        assert!(solutions.iter().any(|s| s.objectives == [1, 2, 3]));
        assert!(solutions.iter().any(|s| s.objectives == [4, 5, 6]));
    }

    #[test]
    fn test_edge_case_identical_solutions() {
        let mut tree: TestTree2D = NDTree::new();
        let sol = sol2d(5, 5);

        // Insert the same solution multiple times
        for _ in 0..3 {
            tree.insert(sol.clone());
        }

        assert_eq!(tree.iter().count(), 3); // All should be stored
    }

    #[test]
    fn test_edge_case_extreme_values() {
        let mut tree: TestTree2D = NDTree::new();
        tree.insert(sol2d(u64::MIN, u64::MAX));
        tree.insert(sol2d(u64::MAX, u64::MIN));

        assert_eq!(tree.iter().count(), 2);
    }

    #[test]
    fn test_stress_test_many_insertions() {
        let mut tree: TestTree2D = NDTree::new();
        for i in 0..100 {
            tree.insert(sol2d(i, 100 - i));
        }

        assert_eq!(tree.len(), 100);
    }

    /// The pruned queries must agree with brute force on every input.
    ///
    /// `is_dominated_by_any` and `for_each_dominated_by` skip whole subtrees
    /// using node bounds, so a wrong prune silently returns fewer results
    /// rather than failing loudly. This compares them against a linear scan
    /// over the same tree for many random trees and probes.
    #[test]
    fn pruned_queries_match_brute_force() {
        // Deterministic xorshift: no dev-dependency needed for a property test.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % 60
        };

        for trial in 0..40 {
            let mut tree: NDTree<Solution<2>, 8, 2, 4> = NDTree::new();
            let mut inserted: Vec<Solution<2>> = Vec::new();
            for _ in 0..60 {
                let sol = Solution::new([next(), next()]);
                if tree.update(sol.clone()) {
                    // Mirror the tree's own pruning of newly dominated members.
                    inserted.retain(|k| !sol.dominates(k.objectives()));
                    inserted.push(sol);
                }
            }

            for _ in 0..60 {
                let probe = Solution::new([next(), next()]);

                let expected_dominated_by_any =
                    inserted.iter().any(|s| s.covers(probe.objectives()));
                assert_eq!(
                    tree.is_dominated_by_any(&probe),
                    expected_dominated_by_any,
                    "trial {trial}: is_dominated_by_any disagreed for {:?}",
                    probe.objectives()
                );

                let expected_strict = inserted.iter().any(|s| s.dominates(probe.objectives()));
                assert_eq!(
                    tree.is_strictly_dominated_by_any(&probe),
                    expected_strict,
                    "trial {trial}: is_strictly_dominated_by_any disagreed for {:?}",
                    probe.objectives()
                );

                let mut got: Vec<[u64; 2]> = Vec::new();
                tree.for_each_dominated_by(&probe, |s| got.push(*s.objectives()));
                got.sort_unstable();
                let mut want: Vec<[u64; 2]> = inserted
                    .iter()
                    .filter(|s| probe.dominates(s.objectives()))
                    .map(|s| *s.objectives())
                    .collect();
                want.sort_unstable();
                assert_eq!(
                    got, want,
                    "trial {trial}: for_each_dominated_by disagreed for {:?}",
                    probe.objectives()
                );
            }
        }
    }
}
