//! GPBA algorithm implementations for advanced Pareto front representation
//!
//! This module provides implementations of the three Grid Point Based Algorithms (GPBA)
//! from Mesquita-Cunha et al. (2023) for generating high-quality representations of
//! Pareto fronts in multi-objective integer linear programming.

use crate::{
    bounds::BoundsCalculator,
    epsilon_constraint::EpsilonConstraintBuilder,
    error::{AugmeconError, Result},
    model::MultiObjectiveProblem,
    options::Options,
    solution::{HasObjectives, ParetoFront, Solution},
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for GPBA representation algorithms
#[derive(Debug, Clone)]
pub struct GpbaConfig {
    /// Primary objective index to optimize directly
    pub primary_objective: usize,
    /// Target number of points per objective for grid construction
    pub target_points_per_objective: HashMap<usize, usize>,
    /// Optional manual bounds (ideal, nadir) - if None, will be computed
    pub manual_bounds: Option<(Vec<f64>, Vec<f64>)>,
}

/// GPBA-A: Coverage-focused representation algorithm
///
/// Minimizes the maximum distance between consecutive points to ensure
/// good coverage of the entire Pareto front.
pub struct GpbaA {
    config: GpbaConfig,
    acceptable_coverage_error: HashMap<usize, f64>,
    discarded_points: HashMap<usize, Vec<f64>>,
    /// Start time for timeout tracking
    start_time: Option<Instant>,
    /// Total timeout duration
    timeout_duration: Option<Duration>,
}

impl GpbaA {
    /// Create new GPBA-A instance with coverage focus
    #[must_use]
    pub fn new(config: GpbaConfig) -> Self {
        Self {
            acceptable_coverage_error: HashMap::new(),
            discarded_points: HashMap::new(),
            config,
            start_time: None,
            timeout_duration: None,
        }
    }

    /// Set timeout for the solver
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout_duration = Some(timeout);
        self
    }

    /// Initialize timeout tracking
    fn start_timeout_tracking(&mut self) {
        if self.timeout_duration.is_some() {
            self.start_time = Some(Instant::now());
        }
    }

    /// Check if the timeout has been reached
    fn is_timeout_reached(&self) -> bool {
        if let (Some(start), Some(duration)) = (self.start_time, self.timeout_duration) {
            start.elapsed() >= duration
        } else {
            false
        }
    }

    /// Get remaining timeout duration
    fn get_remaining_timeout(&self) -> Option<Duration> {
        if let (Some(start), Some(duration)) = (self.start_time, self.timeout_duration) {
            let elapsed = start.elapsed();
            if elapsed >= duration {
                Some(Duration::ZERO)
            } else {
                Some(duration - elapsed)
            }
        } else {
            None
        }
    }

    /// Initialize coverage parameters based on objective ranges and target points
    fn initialize_coverage_parameters(&mut self, ideal: &[f64], nadir: &[f64]) {
        for (&k, &pi_k) in &self.config.target_points_per_objective {
            if k != self.config.primary_objective {
                let range_k = ideal[k] - nadir[k];
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "Converting grid size usize to f64 for mathematical calculation - precision loss acceptable for algorithm"
                )]
                let gamma_k = range_k / (pi_k as f64);
                self.acceptable_coverage_error.insert(k, gamma_k);
            }
        }
    }

    /// Adjust epsilon parameter for objective k (Algorithm 2 from paper)
    fn adjust_epsilon_k(
        &self,
        k: usize,
        current_epsilon_k: f64,
        current_z_k: f64,
        ideal_z_k: f64,
        _nadir_z_k: f64,
    ) -> f64 {
        let gamma_k = self.acceptable_coverage_error[&k];

        // Find next grid point within acceptable coverage error
        let mut next_epsilon = current_epsilon_k + gamma_k;

        // Check discarded points for better coverage
        if let Some(discarded) = self.discarded_points.get(&k) {
            for &discarded_point in discarded {
                if discarded_point > current_z_k
                    && discarded_point < next_epsilon
                    && (discarded_point - current_z_k) <= gamma_k
                {
                    next_epsilon = discarded_point;
                    break;
                }
            }
        }

        next_epsilon.min(ideal_z_k)
    }

    /// Generate Pareto front representation with coverage focus
    ///
    /// # Errors
    /// Returns an error if the optimization solver fails or problem validation fails
    pub fn generate_representation(
        &mut self,
        problem: &MultiObjectiveProblem,
        options: &Options,
    ) -> Result<ParetoFront> {
        const MAX_ITERATIONS: usize = 10000; // Prevent infinite loops

        // Initialize timeout tracking
        self.start_timeout_tracking();

        // Step 1: Compute or use provided bounds using shared calculator
        let (ideal, nadir) = if let Some((ideal, nadir)) = &self.config.manual_bounds {
            (ideal.clone(), nadir.clone())
        } else {
            BoundsCalculator::new(problem, options)
                .calculate_bounds(self.get_remaining_timeout())?
        };

        // Step 2: Initialize coverage parameters
        self.initialize_coverage_parameters(&ideal, &nadir);

        // Step 3: Main epsilon-constraint iteration using shared builder
        let mut epsilons = self.initialize_epsilons(&nadir);
        let ranges = Self::calculate_objective_ranges(&ideal, &nadir);
        let mut pareto_front = ParetoFront::new(vec![
            crate::model::ObjectiveDirection::Minimize;
            problem.num_objectives()
        ]);
        let mut iteration = 0;

        while iteration < MAX_ITERATIONS {
            // Check timeout before each iteration
            if self.is_timeout_reached() {
                break;
            }

            match self.solve_epsilon_constraint_problem_shared(problem, &epsilons, &ranges)? {
                Some(solution) => {
                    // Convert Solution to Solution and add to front
                    let pareto_solution = Solution::new(
                        solution.objective_values.clone(),
                        solution.decision_variables.clone(),
                    );
                    pareto_front.add_solution(pareto_solution);

                    // Update epsilon values using GPBA-A logic
                    let mut converged = true;
                    for (&k, epsilon_k) in &mut epsilons {
                        let current_z_k = solution.objectives()[k];
                        let new_epsilon =
                            self.adjust_epsilon_k(k, *epsilon_k, current_z_k, ideal[k], nadir[k]);

                        if (new_epsilon - *epsilon_k).abs() > 1e-6 && new_epsilon < ideal[k] {
                            *epsilon_k = new_epsilon;
                            converged = false;
                        }
                    }

                    if converged {
                        break;
                    }
                }
                None => {
                    // No more feasible solutions found
                    break;
                }
            }
            iteration += 1;
        }

        if iteration >= MAX_ITERATIONS {
            return Err(AugmeconError::OptimizationError(
                "GPBA-A reached maximum iterations without convergence".to_string(),
            ));
        }

        Ok(pareto_front)
    }

    fn initialize_epsilons(&self, nadir: &[f64]) -> HashMap<usize, f64> {
        let mut epsilons = HashMap::new();
        for (k, &_nadir_val) in nadir.iter().enumerate() {
            if k != self.config.primary_objective {
                epsilons.insert(k, nadir[k] - 0.001); // Small perturbation for relaxation
            }
        }
        epsilons
    }

    /// Solve epsilon-constraint problem with proper ranges
    fn solve_epsilon_constraint_problem_shared(
        &self,
        problem: &MultiObjectiveProblem,
        epsilons: &HashMap<usize, f64>,
        ranges: &HashMap<usize, f64>,
    ) -> Result<Option<Solution>> {
        let default_options = Options::default();
        let mut builder =
            EpsilonConstraintBuilder::new(problem, &default_options, self.config.primary_objective);

        for (&k, &epsilon) in epsilons {
            let range = ranges.get(&k).copied().unwrap_or(1000.0);
            builder = builder.add_constraint_with_range(k, epsilon, range);
        }

        match builder.solve_with_slack(self.get_remaining_timeout())? {
            Some(solution_with_slack) => Ok(Some(solution_with_slack.solution)),
            None => Ok(None),
        }
    }

    /// Calculate objective ranges for proper augmentation coefficient scaling
    fn calculate_objective_ranges(ideal: &[f64], nadir: &[f64]) -> HashMap<usize, f64> {
        let mut ranges = HashMap::new();
        for i in 0..ideal.len() {
            let range = (ideal[i] - nadir[i]).abs();
            // Ensure minimum range to avoid division by zero
            ranges.insert(i, range.max(1e-6));
        }
        ranges
    }
}

/// GPBA-B: Uniformity-focused representation algorithm
///
/// Maximizes the minimum distance between points to achieve uniform distribution
/// across the Pareto front.
pub struct GpbaB {
    config: GpbaConfig,
    acceptable_uniformity_level: HashMap<usize, f64>,
    /// Start time for timeout tracking
    start_time: Option<Instant>,
    /// Total timeout duration
    timeout_duration: Option<Duration>,
}

impl GpbaB {
    /// Create a new GPBA-B instance with uniformity focus
    #[must_use]
    pub fn new(config: GpbaConfig) -> Self {
        Self {
            acceptable_uniformity_level: HashMap::new(),
            config,
            start_time: None,
            timeout_duration: None,
        }
    }

    /// Set timeout for the solver
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout_duration = Some(timeout);
        self
    }

    /// Initialize timeout tracking
    fn start_timeout_tracking(&mut self) {
        if self.timeout_duration.is_some() {
            self.start_time = Some(Instant::now());
        }
    }

    /// Check if the timeout has been reached
    fn is_timeout_reached(&self) -> bool {
        if let (Some(start), Some(duration)) = (self.start_time, self.timeout_duration) {
            start.elapsed() >= duration
        } else {
            false
        }
    }

    /// Get remaining timeout duration
    fn get_remaining_timeout(&self) -> Option<Duration> {
        if let (Some(start), Some(duration)) = (self.start_time, self.timeout_duration) {
            let elapsed = start.elapsed();
            if elapsed >= duration {
                Some(Duration::ZERO)
            } else {
                Some(duration - elapsed)
            }
        } else {
            None
        }
    }

    fn initialize_uniformity_parameters(&mut self, ideal: &[f64], nadir: &[f64]) {
        for (&k, &pi_k) in &self.config.target_points_per_objective {
            if k != self.config.primary_objective {
                let range_k = ideal[k] - nadir[k];
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "Converting grid size usize to f64 for mathematical calculation - precision loss acceptable for algorithm"
                )]
                let delta_k = range_k / (pi_k as f64);
                self.acceptable_uniformity_level.insert(k, delta_k);
            }
        }
    }

    /// Simple uniform step adjustment (Algorithm 3 from paper)
    fn adjust_epsilon_k(&self, k: usize, current_z_k: f64, ideal_z_k: f64) -> f64 {
        let delta_k = self.acceptable_uniformity_level[&k];
        (current_z_k + delta_k).min(ideal_z_k)
    }

    /// Generate Pareto front representation with uniformity focus
    ///
    /// # Errors
    /// Returns an error if the optimization solver fails or problem validation fails
    pub fn generate_representation(
        &mut self,
        problem: &MultiObjectiveProblem,
        options: &Options,
    ) -> Result<ParetoFront> {
        const MAX_ITERATIONS: usize = 10000;

        // Initialize timeout tracking
        self.start_timeout_tracking();

        let (ideal, nadir) = if let Some((ideal, nadir)) = &self.config.manual_bounds {
            (ideal.clone(), nadir.clone())
        } else {
            BoundsCalculator::new(problem, options)
                .calculate_bounds(self.get_remaining_timeout())?
        };

        self.initialize_uniformity_parameters(&ideal, &nadir);

        let mut pareto_front = ParetoFront::new(vec![
            crate::model::ObjectiveDirection::Minimize;
            problem.num_objectives()
        ]);
        let mut epsilons = self.initialize_epsilons(&nadir);
        let ranges = Self::calculate_objective_ranges(&ideal, &nadir);
        let mut iteration = 0;

        while iteration < MAX_ITERATIONS {
            // Check timeout before each iteration
            if self.is_timeout_reached() {
                break;
            }

            // Create epsilon constraint problem with proper ranges
            let default_options = Options::default();
            let mut builder = EpsilonConstraintBuilder::new(
                problem,
                &default_options,
                self.config.primary_objective,
            );
            for (&k, &epsilon) in &epsilons {
                let range = ranges.get(&k).copied().unwrap_or(1000.0);
                builder = builder.add_constraint_with_range(k, epsilon, range);
            }

            match builder.solve_with_slack(self.get_remaining_timeout())? {
                Some(solution_with_slack) => {
                    // Convert and add solution to front
                    let pareto_solution = Solution::new(
                        solution_with_slack.solution.objective_values.clone(),
                        solution_with_slack.solution.decision_variables.clone(),
                    );
                    pareto_front.add_solution(pareto_solution);

                    // Update with uniform steps
                    let mut continue_iteration = false;
                    for (&k, epsilon_k) in &mut epsilons {
                        let current_z_k = solution_with_slack.solution.objective_values[k];
                        let new_epsilon = self.adjust_epsilon_k(k, current_z_k, ideal[k]);

                        if (new_epsilon - *epsilon_k).abs() > 1e-6 && new_epsilon < ideal[k] {
                            *epsilon_k = new_epsilon;
                            continue_iteration = true;
                        }
                    }

                    if !continue_iteration {
                        break;
                    }
                }
                None => {
                    break;
                }
            }
            iteration += 1;
        }

        if iteration >= MAX_ITERATIONS {
            return Err(AugmeconError::OptimizationError(
                "GPBA-B reached maximum iterations without convergence".to_string(),
            ));
        }

        Ok(pareto_front)
    }

    /// Calculate objective ranges for proper augmentation coefficient scaling
    fn calculate_objective_ranges(ideal: &[f64], nadir: &[f64]) -> HashMap<usize, f64> {
        let mut ranges = HashMap::new();
        for i in 0..ideal.len() {
            let range = (ideal[i] - nadir[i]).abs();
            // Ensure minimum range to avoid division by zero
            ranges.insert(i, range.max(1e-6));
        }
        ranges
    }

    fn initialize_epsilons(&self, nadir: &[f64]) -> HashMap<usize, f64> {
        let mut epsilons = HashMap::new();
        for (k, &_nadir_val) in nadir.iter().enumerate() {
            if k != self.config.primary_objective {
                epsilons.insert(k, nadir[k]);
            }
        }
        epsilons
    }
}

/// GPBA-C: Cardinality-focused representation algorithm
///
/// Maintains target cardinality through adaptive grid refinement, balancing
/// coverage and uniformity based on actual Pareto front structure.
pub struct GpbaC {
    config: GpbaConfig,
    grid_state: HashMap<usize, GridState>,
    /// Start time for timeout tracking
    start_time: Option<Instant>,
    /// Total timeout duration
    timeout_duration: Option<Duration>,
}

#[derive(Debug, Clone)]
struct GridState {
    start_point: f64,
    current_position: usize,
    remaining_points: usize,
}

impl GpbaC {
    /// Create a new GPBA-C instance with cardinality focus
    #[must_use]
    pub fn new(config: GpbaConfig) -> Self {
        Self {
            grid_state: HashMap::new(),
            config,
            start_time: None,
            timeout_duration: None,
        }
    }

    /// Set timeout for the solver
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout_duration = Some(timeout);
        self
    }

    /// Initialize timeout tracking
    fn start_timeout_tracking(&mut self) {
        if self.timeout_duration.is_some() {
            self.start_time = Some(Instant::now());
        }
    }

    /// Check if the timeout has been reached
    fn is_timeout_reached(&self) -> bool {
        if let (Some(start), Some(duration)) = (self.start_time, self.timeout_duration) {
            start.elapsed() >= duration
        } else {
            false
        }
    }

    /// Get remaining timeout duration
    fn get_remaining_timeout(&self) -> Option<Duration> {
        if let (Some(start), Some(duration)) = (self.start_time, self.timeout_duration) {
            let elapsed = start.elapsed();
            if elapsed >= duration {
                Some(Duration::ZERO)
            } else {
                Some(duration - elapsed)
            }
        } else {
            None
        }
    }

    fn initialize_grid_state(&mut self, nadir: &[f64]) {
        for (&k, &pi_k) in &self.config.target_points_per_objective {
            if k != self.config.primary_objective {
                self.grid_state.insert(
                    k,
                    GridState {
                        start_point: nadir[k],
                        current_position: 0,
                        remaining_points: pi_k.saturating_sub(1),
                    },
                );
            }
        }
    }

    /// Adaptive grid refinement (Algorithm 4 from paper)
    fn adjust_epsilon_k(
        &mut self,
        k: usize,
        ideal_z_k: f64,
        current_z_k: f64,
        slack_k: f64,
    ) -> f64 {
        let grid_state = self.grid_state.get_mut(&k).unwrap();

        if grid_state.remaining_points == 0 {
            return ideal_z_k;
        }

        // Calculate current step size
        let current_range = ideal_z_k - grid_state.start_point;
        #[expect(
            clippy::cast_precision_loss,
            reason = "Converting remaining_points usize to f64 for step size calculation - precision loss acceptable for GPBA algorithm"
        )]
        let step = (current_range / (grid_state.remaining_points as f64)).max(1.0);

        // Determine how many grid points can be skipped
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "Converting f64 floor result to usize for grid point counting - truncation and sign loss are expected behavior in GPBA grid refinement"
        )]
        let skippable_points = (slack_k / step).floor() as usize;

        if skippable_points > 0 {
            // Refine grid: start from current position
            grid_state.start_point = current_z_k;
            grid_state.remaining_points =
                grid_state.remaining_points.saturating_sub(skippable_points);
            grid_state.current_position = 0;

            // Calculate next epsilon with refined grid
            if grid_state.remaining_points > 0 {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "Converting remaining_points usize to f64 for new step calculation - precision loss acceptable for GPBA adaptive grid refinement"
                )]
                let new_step = (ideal_z_k - current_z_k) / (grid_state.remaining_points as f64);
                current_z_k + new_step
            } else {
                ideal_z_k
            }
        } else {
            // Normal grid progression
            grid_state.current_position += 1;
            grid_state.remaining_points = grid_state.remaining_points.saturating_sub(1);
            #[expect(
                clippy::cast_precision_loss,
                reason = "Converting current_position usize to f64 for grid position calculation - precision loss acceptable for GPBA grid traversal"
            )]
            let current_position_f64 = grid_state.current_position as f64;
            step.mul_add(current_position_f64, grid_state.start_point)
        }
    }

    /// Generate Pareto front representation with cardinality focus
    ///
    /// # Errors
    /// Returns an error if the optimization solver fails or problem validation fails
    pub fn generate_representation(
        &mut self,
        problem: &MultiObjectiveProblem,
        options: &Options,
    ) -> Result<ParetoFront> {
        const MAX_ITERATIONS: usize = 10000;

        // Initialize timeout tracking
        self.start_timeout_tracking();

        let (ideal, nadir) = if let Some((ideal, nadir)) = &self.config.manual_bounds {
            (ideal.clone(), nadir.clone())
        } else {
            BoundsCalculator::new(problem, options)
                .calculate_bounds(self.get_remaining_timeout())?
        };

        self.initialize_grid_state(&nadir);

        let mut pareto_front = ParetoFront::new(vec![
            crate::model::ObjectiveDirection::Minimize;
            problem.num_objectives()
        ]);
        let mut epsilons = self.initialize_epsilons(&nadir);
        let ranges = Self::calculate_objective_ranges(&ideal, &nadir);
        let mut iteration = 0;

        while iteration < MAX_ITERATIONS {
            // Check timeout before each iteration
            if self.is_timeout_reached() {
                break;
            }

            // For GPBA-C, we need slack values for adaptive grid refinement
            let default_options = Options::default();
            let mut builder = EpsilonConstraintBuilder::new(
                problem,
                &default_options,
                self.config.primary_objective,
            );
            for (&k, &epsilon) in &epsilons {
                let range = ranges.get(&k).copied().unwrap_or(1000.0);
                builder = builder.add_constraint_with_range(k, epsilon, range);
            }

            match builder.solve_with_slack(self.get_remaining_timeout())? {
                Some(solution_with_slack) => {
                    // Convert and add solution to front
                    let pareto_solution = Solution::new(
                        solution_with_slack.solution.objective_values.clone(),
                        solution_with_slack.solution.decision_variables.clone(),
                    );
                    pareto_front.add_solution(pareto_solution);

                    // Update using adaptive grid refinement
                    let mut continue_iteration = false;
                    for (&k, epsilon_k) in &mut epsilons {
                        let current_z_k = solution_with_slack.solution.objective_values[k];
                        let slack_k = solution_with_slack
                            .slack_values
                            .get(&k)
                            .copied()
                            .unwrap_or(0.0);

                        let new_epsilon = self.adjust_epsilon_k(k, ideal[k], current_z_k, slack_k);

                        if (new_epsilon - *epsilon_k).abs() > 1e-6
                            && new_epsilon < ideal[k]
                            && self.grid_state[&k].remaining_points > 0
                        {
                            *epsilon_k = new_epsilon;
                            continue_iteration = true;
                        }
                    }

                    if !continue_iteration {
                        break;
                    }
                }
                None => {
                    break;
                }
            }
            iteration += 1;
        }

        if iteration >= MAX_ITERATIONS {
            return Err(AugmeconError::OptimizationError(
                "GPBA-C reached maximum iterations without convergence".to_string(),
            ));
        }

        Ok(pareto_front)
    }

    /// Calculate objective ranges for proper augmentation coefficient scaling
    fn calculate_objective_ranges(ideal: &[f64], nadir: &[f64]) -> HashMap<usize, f64> {
        let mut ranges = HashMap::new();
        for i in 0..ideal.len() {
            let range = (ideal[i] - nadir[i]).abs();
            // Ensure minimum range to avoid division by zero
            ranges.insert(i, range.max(1e-6));
        }
        ranges
    }

    fn initialize_epsilons(&self, nadir: &[f64]) -> HashMap<usize, f64> {
        let mut epsilons = HashMap::new();
        for (k, &_nadir_val) in nadir.iter().enumerate() {
            if k != self.config.primary_objective {
                epsilons.insert(k, nadir[k]);
            }
        }
        epsilons
    }
}

/// Helper function to create GPBA configurations for common use cases
pub mod presets {
    use super::{GpbaConfig, HashMap};

    /// Configuration for high-coverage representation (GPBA-A)
    #[must_use]
    pub fn high_coverage_config(primary_obj: usize, points_per_obj: usize) -> GpbaConfig {
        let mut target_points = HashMap::new();
        target_points.insert(0, points_per_obj);
        target_points.insert(1, points_per_obj);

        GpbaConfig {
            primary_objective: primary_obj,
            target_points_per_objective: target_points,
            manual_bounds: None,
        }
    }

    /// Configuration for uniform distribution (GPBA-B)
    #[must_use]
    pub fn uniform_distribution_config(primary_obj: usize, points_per_obj: usize) -> GpbaConfig {
        let mut target_points = HashMap::new();
        target_points.insert(0, points_per_obj);
        target_points.insert(1, points_per_obj);

        GpbaConfig {
            primary_objective: primary_obj,
            target_points_per_objective: target_points,
            manual_bounds: None,
        }
    }

    /// Configuration for balanced representation with target size (GPBA-C)
    #[must_use]
    pub fn balanced_cardinality_config(primary_obj: usize, total_target: usize) -> GpbaConfig {
        let points_per_obj = total_target / 2; // For bi-objective problems
        let mut target_points = HashMap::new();
        target_points.insert(0, points_per_obj);
        target_points.insert(1, points_per_obj);

        GpbaConfig {
            primary_objective: primary_obj,
            target_points_per_objective: target_points,
            manual_bounds: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpba_config_creation() {
        let config = presets::high_coverage_config(0, 25);
        assert_eq!(config.primary_objective, 0);
        assert_eq!(config.target_points_per_objective[&0], 25);
        assert_eq!(config.target_points_per_objective[&1], 25);
    }

    #[test]
    fn test_gpba_a_initialization() {
        let config = presets::high_coverage_config(0, 20);
        let gpba_a = GpbaA::new(config);
        assert_eq!(gpba_a.config.primary_objective, 0);
    }

    #[test]
    fn test_grid_state_initialization() {
        let config = presets::balanced_cardinality_config(1, 50);
        let mut gpba_c = GpbaC::new(config);
        let nadir = vec![0.0, 0.0];
        gpba_c.initialize_grid_state(&nadir);

        assert!(gpba_c.grid_state.contains_key(&0));
        assert_eq!(gpba_c.grid_state[&0].remaining_points, 24); // 25 - 1
    }
}
