use std::{
    cmp::Reverse,
    collections::{hash_map::DefaultHasher, hash_map::Entry, HashMap},
    hash::{Hash, Hasher},
    time::Duration,
};

use log::{trace, warn};

use crate::{objectives::Objectives, solution::EncodedSolution};

#[derive(Debug)]
pub struct SolutionFingerprint {
    pub explored_neighborhood_size: u8,
    pub objectives: Objectives,
    pub iteration: u16,
    pub time: Duration,
}

#[derive(Debug, Eq, Clone, Copy)]
pub struct SolutionPoint(pub usize, pub i32, pub i32);

impl PartialEq for SolutionPoint {
    fn eq(&self, other: &Self) -> bool {
        (self.1, self.2) == (other.1, other.2)
    }
}

impl Ord for SolutionPoint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.1, self.2).cmp(&(other.1, other.2))
    }
}

impl PartialOrd for SolutionPoint {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some((self.1, self.2).cmp(&(other.1, other.2)))
    }
}

impl From<&SolutionFingerprint> for SolutionPoint {
    fn from(solution: &SolutionFingerprint) -> Self {
        SolutionPoint(
            solution.iteration as usize,
            solution.objectives.0 as i32,
            solution.objectives.1 as i32,
        )
    }
}

pub struct ExploredSolutionsData {
    pub solutions: HashMap<u64, SolutionFingerprint>,
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
    pub fn new(iteration: usize, timestamp: Duration, solutions: Vec<Vec<usize>>) -> Self {
        Self {
            iteration,
            timestamp,
            solutions,
        }
    }
}

impl ExploredSolutionsData {
    pub fn new(max_cost: u64, max_cloudy_area: u64) -> Self {
        Self {
            solutions: HashMap::new(),
            num_iterations: 0,
            max_cost,
            max_cloudy_area,
            pareto_front_snapshots: Vec::new(),
        }
    }

    pub fn get_solution_fingerprint(
        &self,
        solution: &EncodedSolution,
    ) -> Option<&SolutionFingerprint> {
        let hash = ExploredSolutionsData::hash(solution);
        self.solutions.get(&hash)
    }

    fn hash(solution: &EncodedSolution) -> u64 {
        let mut hasher = DefaultHasher::new();
        solution.hash(&mut hasher);
        hasher.finish()
    }

    pub fn register(&mut self, iteration: usize, solution: &EncodedSolution, time: Duration) {
        let hash = ExploredSolutionsData::hash(solution);

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

    pub fn update_explored_neighborhood_size(
        &mut self,
        solution: &EncodedSolution,
        explored_neighborhood_size: u32,
    ) {
        let hash = ExploredSolutionsData::hash(solution);
        let entry = self
            .solutions
            .get_mut(&hash)
            .expect("solution is not registered");
        entry.explored_neighborhood_size = explored_neighborhood_size as u8;
    }

    pub fn explored_neighborhood_size(&mut self, solution: &EncodedSolution) -> u32 {
        let hash = ExploredSolutionsData::hash(solution);
        let entry = self
            .solutions
            .get(&hash)
            .expect("solution is not registered");
        entry.explored_neighborhood_size as u32
    }

    pub fn is_registered(&self, solution: &EncodedSolution) -> bool {
        let hash = ExploredSolutionsData::hash(solution);
        self.solutions.contains_key(&hash)
    }

    pub fn initial_solutions(&self) -> Vec<SolutionPoint> {
        let initial_solutions: Vec<SolutionPoint> = self
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

    pub fn solutions(&self) -> Vec<SolutionPoint> {
        let solutions: Vec<SolutionPoint> =
            self.solutions.values().map(SolutionPoint::from).collect();
        solutions
    }

    pub fn non_dominated(&self) -> Vec<SolutionPoint> {
        let mut non_dominated_solutions: Vec<SolutionPoint> = Vec::new();
        let mut solutions = self.solutions();

        // Sort solutions in reversed lexicographical order by objectives
        solutions.sort_by_key(|solution| Reverse((solution.1, solution.2)));

        let mut smallest_obj2;

        // Smallest point is vacuously non-dominated
        if let Some(first_point) = solutions.pop() {
            smallest_obj2 = first_point.2;
            non_dominated_solutions.push(first_point);
        } else {
            // Solution set is empty, return empty vector
            return non_dominated_solutions;
        }

        while let Some(point) = solutions.pop() {
            if point.2 < smallest_obj2 {
                smallest_obj2 = point.2;
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
        I: Iterator<Item = &'a EncodedSolution>,
    {
        let solutions: Vec<Vec<usize>> = solutions
            .map(|solution| solution.selected_images().collect())
            .collect();
        let pareto_front_snapshot = ParetoFrontSnapshot::new(iteration, elapsed, solutions);
        self.pareto_front_snapshots.push(pareto_front_snapshot);
    }
}
