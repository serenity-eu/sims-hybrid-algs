use std::{
    cmp::Reverse,
    collections::{HashMap, hash_map::DefaultHasher, hash_map::Entry},
    hash::{Hash, Hasher},
    time::Duration,
};

use pareto::{HasObjectives, MoSolution, Objectives};
use tracing::{instrument, warn};

#[derive(Debug, Clone)]
pub struct SolutionFingerprint<const D: usize> {
    pub explored_neighborhood_size: u8,
    pub objectives: pareto::Objectives<D>,
    pub iteration: u16,
    pub timestamp: Duration,
    pub selected_images: Vec<usize>,
}

impl<const D: usize> HasObjectives<D> for SolutionFingerprint<D> {
    fn objectives(&self) -> &Objectives<D> {
        &self.objectives
    }
}

impl<const D: usize> MoSolution<D> for SolutionFingerprint<D> { }

#[derive(Debug, Eq, Clone, Copy)]
pub struct SolutionPoint<const D: usize> {
    pub iteration: usize,
    pub timestamp: Duration,
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
            timestamp: solution.timestamp,
            objectives: solution.objectives,
        }
    }
}

/// Data structure for tracking explored solutions during optimization.
///
/// This structure stores solution fingerprints with their objectives, iteration counts,
/// and timing information. It supports generic dimensionality for objective spaces.
pub struct ExploredSolutionsData<const D: usize> {
    pub solutions: HashMap<u64, SolutionFingerprint<D>>,
    pub num_iterations: usize,
    /// Maximum values for each objective dimension, used for plotting bounds and normalization
    pub max_objectives: [u64; D],
}

impl<const D: usize> ExploredSolutionsData<D> {
    #[must_use]
    pub fn new(max_objectives: [u64; D]) -> Self {
        Self {
            solutions: HashMap::new(),
            num_iterations: 0,
            max_objectives,
        }
    }

    #[must_use]
    pub fn get_solution_fingerprint<T: Hash>(
        &self,
        solution: &T,
    ) -> Option<&SolutionFingerprint<D>> {
        let hash = Self::hash(solution);
        self.solutions.get(&hash)
    }

    fn hash<T: Hash>(solution: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        solution.hash(&mut hasher);
        hasher.finish()
    }

    pub fn register<T: HasObjectives<D> + Hash>(
        &mut self,
        iteration: usize,
        solution: &T,
        time: Duration,
        selected_images: Vec<usize>,
    ) {
        let hash = Self::hash(solution);

        if let Entry::Vacant(e) = self.solutions.entry(hash) {
            let new_entry = SolutionFingerprint {
                iteration: iteration as u16,
                objectives: *solution.objectives(),
                explored_neighborhood_size: 0,
                timestamp: time,
                selected_images,
            };
            e.insert(new_entry);
        } else {
            warn!("Solution was already registered in the explored solutions set");
        }
    }

    /// Updates the explored neighborhood size for a registered solution.
    ///
    /// # Panics
    ///
    /// Panics if the solution is not registered in the explored solutions set.
    #[instrument(level = "debug", skip(self, solution), fields(explored_neighborhood_size))]
    pub fn update_explored_neighborhood_size<T: Hash>(
        &mut self,
        solution: &T,
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
    pub fn explored_neighborhood_size<T: Hash>(&mut self, solution: &T) -> u32 {
        let hash = Self::hash(solution);
        let entry = self
            .solutions
            .get(&hash)
            .expect("solution is not registered");
        u32::from(entry.explored_neighborhood_size)
    }

    #[must_use]
    pub fn is_registered<T: Hash>(&self, solution: &T) -> bool {
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

    /// Get the maximum value for a specific objective index
    #[must_use]
    pub const fn max_objective(&self, index: usize) -> u64 {
        self.max_objectives[index]
    }

    /// Get all maximum objective values
    #[must_use]
    pub const fn max_objectives(&self) -> &[u64; D] {
        &self.max_objectives
    }

    /// Export explored solutions to JSON string
    #[must_use]
    pub fn to_json(&self) -> String {
        use std::fmt::Write;
        
        let mut json = String::from("{\n  \"solutions\": [\n");
        
        let mut first = true;
        for (hash, solution) in &self.solutions {
            if !first {
                json.push_str(",\n");
            }
            first = false;
            
            let _ = writeln!(json, "    {{");
            let _ = writeln!(json, "      \"hash\": {hash},");
            let _ = writeln!(json, "      \"iteration\": {},", solution.iteration);
            let _ = writeln!(json, "      \"timestamp_us\": {},", solution.timestamp.as_micros());
            let _ = writeln!(json, "      \"explored_neighborhood_size\": {},", solution.explored_neighborhood_size);
            let _ = write!(json, "      \"objectives\": [");
            for (i, obj) in solution.objectives.iter().enumerate() {
                if i > 0 {
                    json.push_str(", ");
                }
                let _ = write!(json, "{obj}");
            }
            json.push_str("],\n");
            let _ = write!(json, "      \"selected_images\": [");
            for (i, img) in solution.selected_images.iter().enumerate() {
                if i > 0 {
                    json.push_str(", ");
                }
                let _ = write!(json, "{img}");
            }
            json.push_str("]\n");
            json.push_str("    }");
        }
        
        json.push_str("\n  ],\n");
        let _ = writeln!(json, "  \"num_iterations\": {},", self.num_iterations);
        json.push_str("  \"max_objectives\": [");
        for (i, obj) in self.max_objectives.iter().enumerate() {
            if i > 0 {
                json.push_str(", ");
            }
            let _ = write!(json, "{obj}");
        }
        json.push_str("]\n}\n");
        
        json
    }
}
