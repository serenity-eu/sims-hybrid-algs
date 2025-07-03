use std::{
    cmp::Reverse,
    collections::{hash_map::DefaultHasher, hash_map::Entry, HashMap},
    hash::{Hash, Hasher},
    time::Duration,
};

use log::{trace, warn};
use pareto::Objectives;

use crate::solution::EncodedSolution;

#[derive(Debug)]
pub struct SolutionFingerprint<const D: usize> {
    pub explored_neighborhood_size: u8,
    pub objectives: pareto::Objectives<D>,
    pub iteration: u16,
    pub time: Duration,
}

#[derive(Debug, Eq, Clone, Copy)]
pub struct SolutionPoint<const D: usize> {
    pub iteration: usize,
    pub objectives: Objectives<D>,
}

impl<const D: usize> PartialEq for SolutionPoint<D> {
    fn eq(&self, other: &Self) -> bool {
        self.objectives == other.objectives
    }
}

impl<const D: usize> Ord for SolutionPoint<D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.objectives.cmp(&other.objectives)
    }
}

#[expect(
    clippy::non_canonical_partial_ord_impl,
    reason = "Use array comparison for objectives"
)]
impl<const D: usize> PartialOrd for SolutionPoint<D> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.objectives.cmp(&other.objectives))
    }
}

impl<const D: usize> From<&SolutionFingerprint<D>> for SolutionPoint<D> {
    fn from(solution: &SolutionFingerprint<D>) -> Self {
        Self {
            iteration: solution.iteration as usize,
            objectives: solution.objectives,
        }
    }
}

pub struct ExploredSolutionsData<const D: usize> {
    pub solutions: HashMap<u64, SolutionFingerprint<D>>,
    pub num_iterations: usize,
    pub max_cost: u64,
    pub max_cloudy_area: u64,
    pub pareto_front_snapshots: Vec<ParetoFrontSnapshot>,
}

pub struct ParetoFrontSnapshot {
    pub iteration: usize,
    pub timestamp: Duration,
    pub solutions: Vec<Vec<usize>>,
}

impl ParetoFrontSnapshot {
    #[must_use]
    pub const fn new(iteration: usize, timestamp: Duration, solutions: Vec<Vec<usize>>) -> Self {
        Self {
            iteration,
            timestamp,
            solutions,
        }
    }
}

impl<const D: usize> ExploredSolutionsData<D> {
    #[must_use]
    pub fn new(max_cost: u64, max_cloudy_area: u64) -> Self {
        Self {
            solutions: HashMap::new(),
            num_iterations: 0,
            max_cost,
            max_cloudy_area,
            pareto_front_snapshots: Vec::new(),
        }
    }

    #[must_use]
    pub fn get_solution_fingerprint(
        &self,
        solution: &EncodedSolution<D>,
    ) -> Option<&SolutionFingerprint<D>> {
        let hash = Self::hash(solution);
        self.solutions.get(&hash)
    }

    fn hash(solution: &EncodedSolution<D>) -> u64 {
        let mut hasher = DefaultHasher::new();
        solution.hash(&mut hasher);
        hasher.finish()
    }

    pub fn register(&mut self, iteration: usize, solution: &EncodedSolution<D>, time: Duration) {
        let hash = Self::hash(solution);

        if let Entry::Vacant(e) = self.solutions.entry(hash) {
            let new_entry = SolutionFingerprint {
                iteration: iteration as u16,
                objectives: solution.objectives,
                explored_neighborhood_size: 0,
                time,
            };
            e.insert(new_entry);
            trace!("Registered new solution: {:?}", solution);
        } else {
            warn!("Solution was already registered in the explored solutions set");
        }
    }

    /// Updates the explored neighborhood size for a registered solution.
    ///
    /// # Panics
    ///
    /// Panics if the solution is not registered in the explored solutions set.
    pub fn update_explored_neighborhood_size(
        &mut self,
        solution: &EncodedSolution<D>,
        explored_neighborhood_size: u32,
    ) {
        let hash = Self::hash(solution);
        let entry = self
            .solutions
            .get_mut(&hash)
            .expect("solution is not registered");
        entry.explored_neighborhood_size = explored_neighborhood_size as u8;
    }

    /// Returns the explored neighborhood size for a registered solution.
    ///
    /// # Panics
    ///
    /// Panics if the solution is not registered in the explored solutions set.
    pub fn explored_neighborhood_size(&mut self, solution: &EncodedSolution<D>) -> u32 {
        let hash = Self::hash(solution);
        let entry = self
            .solutions
            .get(&hash)
            .expect("solution is not registered");
        u32::from(entry.explored_neighborhood_size)
    }

    #[must_use]
    pub fn is_registered(&self, solution: &EncodedSolution<D>) -> bool {
        let hash = Self::hash(solution);
        self.solutions.contains_key(&hash)
    }

    #[must_use]
    pub fn initial_solutions(&self) -> Vec<SolutionPoint<D>> {
        let initial_solutions: Vec<SolutionPoint<D>> = self
            .solutions
            .values()
            .filter_map(|solution| {
                if solution.iteration == 0 {
                    Some(SolutionPoint::from(solution))
                } else {
                    None
                }
            })
            .collect();
        initial_solutions
    }

    pub fn solutions(&self) -> Vec<SolutionPoint<D>> {
        let solutions: Vec<SolutionPoint<D>> =
            self.solutions.values().map(SolutionPoint::from).collect();
        solutions
    }

    #[must_use]
    pub fn non_dominated(&self) -> Vec<SolutionPoint<D>> {
        let mut non_dominated_solutions: Vec<SolutionPoint<D>> = Vec::new();
        let mut solutions = self.solutions();

        // Sort solutions in reversed lexicographical order by objectives
        solutions.sort_by_key(|solution| Reverse((solution.objectives[0], solution.objectives[1])));

        let mut smallest_obj2;

        // Smallest point is vacuously non-dominated
        if let Some(first_point) = solutions.pop() {
            smallest_obj2 = first_point.objectives[1];
            non_dominated_solutions.push(first_point);
        } else {
            // Solution set is empty, return empty vector
            return non_dominated_solutions;
        }

        while let Some(point) = solutions.pop() {
            if point.objectives[1] < smallest_obj2 {
                smallest_obj2 = point.objectives[1];
                non_dominated_solutions.push(point);
            }
        }
        non_dominated_solutions
    }

    pub fn register_pareto_snapshot<'a, I>(
        &mut self,
        iteration: usize,
        elapsed: Duration,
        solutions: I,
    ) where
        I: Iterator<Item = &'a EncodedSolution<D>>,
    {
        let solutions: Vec<Vec<usize>> = solutions
            .map(|solution| solution.selected_images().collect())
            .collect();
        let pareto_front_snapshot = ParetoFrontSnapshot::new(iteration, elapsed, solutions);
        self.pareto_front_snapshots.push(pareto_front_snapshot);
    }
}
