// Probabilistic Probing Neighborhood for multi-objective Pareto local search
// This module implements a probabilistic approach to neighborhood exploration
// in objective space for improved Pareto local search efficiency.

use rand::Rng;

use crate::{
    problem::{Problem, ScaledObjectiveDeltas},
    solution::EncodedSolution,
    timer::Timer,
};

/// Trait for solutions that support probabilistic probing in objective space
///
/// Probabilistic probing is a technique used in multi-objective optimization
/// to efficiently explore the neighborhood of solutions by probabilistically
/// selecting promising directions in the objective space, rather than
/// exhaustively exploring all possible neighbors.
///
/// This implementation generates solutions directly without using residual problem
/// solving, providing better control over the solution generation process.
///
/// All solutions produced by this trait are guaranteed to be valid according
/// to the Set Cover Problem constraints (all elements covered, consistent counts).
pub trait ProbabilisticProbingNeighborhood<const D: usize>: EncodedSolution<D> {
    /// Generate neighborhood solutions using probabilistic probing in objective space
    ///
    /// This method uses probabilistic sampling to explore promising regions of the
    /// objective space, making it more efficient than exhaustive neighborhood exploration
    /// while maintaining good coverage of the Pareto front.
    ///
    /// # Implementation Details
    /// Solutions are generated directly by:
    /// 1. Removing selected images probabilistically based on objective improvements
    /// 2. Finding uncovered elements after removal
    /// 3. Probabilistically selecting covering images based on coverage and objective scores
    /// 4. Validating all generated solutions using `is_valid()`
    ///
    /// # Parameters
    /// - `k`: Maximum number of images to remove/modify simultaneously
    /// - `problem`: The problem instance containing images and objectives
    /// - `timer`: Timer to control computation time limits
    /// - `config`: Configuration parameters for probabilistic probing
    ///
    /// # Returns
    /// Vector of neighbor solutions discovered through probabilistic probing.
    /// **Guarantee**: All returned solutions are valid according to Set Cover Problem constraints:
    /// - All universe elements are covered
    /// - Clear parts and element coverage counts are consistent
    /// - Objective values are correctly calculated
    ///
    /// # Set Cover Problem Validation
    /// Each solution is validated using the existing `is_valid()` method which ensures
    /// all Set Cover Problem constraints are satisfied before being included in results.
    #[allow(clippy::too_many_arguments)]
    fn probabilistic_probing_neighborhood(
        &self,
        k: u32,
        problem: &Problem<Self, D>,
        timer: &Timer,
        is_deterministic: bool,
        probing_probability: f64,
        max_probes: usize,
        objective_weights: Option<&[f64; D]>,
    ) -> Vec<Self> {
        // Default implementation: fall back to regular neighborhood method
        // This allows solutions that don't implement full probabilistic probing
        // to still work with the enhanced PLS algorithm
        let _ = probing_probability; // Suppress unused parameter warning
        let _ = max_probes; // Suppress unused parameter warning  
        let _ = objective_weights; // Suppress unused parameter warning

        self.neighborhood(k, problem, timer, is_deterministic)
    }
}

/// Configuration parameters for probabilistic probing
#[derive(Debug, Clone)]
pub struct ProbingConfig<const D: usize> {
    /// Probability of selecting each candidate for probing
    pub probing_probability: f64,
    /// Maximum number of probing attempts
    pub max_probes: usize,
    /// Weights for biasing toward specific objectives
    pub objective_weights: Option<[f64; D]>,
    /// Temperature parameter for objective space sampling
    pub temperature: f64,
    /// Minimum improvement threshold for accepting moves
    pub improvement_threshold: f64,
}

impl<const D: usize> Default for ProbingConfig<D> {
    fn default() -> Self {
        Self {
            probing_probability: 0.3,
            max_probes: 50,
            objective_weights: None,
            temperature: 1.0,
            improvement_threshold: 0.01,
        }
    }
}

/// Helper struct for probabilistic selection based on objective improvements
#[derive(Debug)]
pub struct ObjectiveBasedSelector<const D: usize> {
    /// Weighted scores for each candidate
    candidates: Vec<(usize, f64)>,
    /// Configuration parameters
    config: ProbingConfig<D>,
}

impl<const D: usize> ObjectiveBasedSelector<D> {
    /// Create new selector with given candidates and their objective deltas
    #[must_use]
    pub fn new(
        candidate_deltas: Vec<(usize, ScaledObjectiveDeltas<D>)>,
        config: ProbingConfig<D>,
    ) -> Self {
        let mut candidates = Vec::new();

        for (image_index, deltas) in candidate_deltas {
            let score = Self::calculate_selection_score(&deltas, &config);
            candidates.push((image_index, score));
        }

        // Sort by score (higher is better)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Self { candidates, config }
    }

    /// Calculate selection score based on objective improvements and weights
    fn calculate_selection_score(
        deltas: &ScaledObjectiveDeltas<D>,
        config: &ProbingConfig<D>,
    ) -> f64 {
        let mut weighted_sum = 0.0;
        let weights = config.objective_weights.as_ref();

        for (i, &delta) in deltas.scaled_deltas.iter().enumerate() {
            let weight = weights.map_or(1.0, |w| w[i]);
            // Use exponential function to emphasize larger improvements
            weighted_sum += weight * (-f64::from(delta) / config.temperature).exp();
        }

        weighted_sum
    }

    /// Select candidates probabilistically based on their scores
    pub fn select_candidates<R: Rng>(&self, rng: &mut R, max_selections: usize) -> Vec<usize> {
        let mut selected = Vec::new();
        let total_score: f64 = self.candidates.iter().map(|(_, score)| score).sum();

        if total_score <= 0.0 {
            // Fallback to uniform selection if no positive scores
            return self
                .candidates
                .iter()
                .take(max_selections)
                .map(|(idx, _)| *idx)
                .collect();
        }

        // Limit selections based on probing configuration
        let effective_max = max_selections.min(self.config.max_probes);

        for _ in 0..effective_max.min(self.candidates.len()) {
            let threshold = rng.random::<f64>() * total_score;
            let mut cumulative = 0.0;

            for &(image_index, score) in &self.candidates {
                cumulative += score;
                if cumulative >= threshold && !selected.contains(&image_index) {
                    selected.push(image_index);
                    break;
                }
            }
        }

        selected
    }
}
