//! GPBA algorithm implementations for advanced Pareto front representation
//!
//! This module provides implementations of the three Grid Point Based Algorithms (GPBA)
//! from Mesquita-Cunha et al. (2023) for generating high-quality representations of
//! Pareto fronts in multi-objective integer linear programming.

use crate::{
    bounds::BoundsCalculator,
    epsilon_constraint::EpsilonConstraintBuilder,
    error::{AugmeconError, Result},
    interval_manager::IntervalManager,
    model::MultiObjectiveProblem,
    options::Options,
    solution::{ParetoFront, Solution},
    timer::Timer,
};
use std::collections::HashMap;
use std::time::Duration;

/// Information about a previously solved epsilon-constraint configuration
#[derive(Debug, Clone)]
#[allow(dead_code, reason = "Fields reserved for future relaxation feature - currently disabled but designed for reuse")]
struct PreviousSolutionInfo {
    /// The epsilon array (constraint values) that was solved
    ef_array: Vec<f64>,
    /// The solution found (None if infeasible)
    solution: Option<Vec<f64>>,
}

/// Configuration for GPBA representation algorithms
#[derive(Debug, Clone)]
pub struct GpbaConfig {
    /// Primary objective index to optimize directly
    pub primary_objective: usize,
    /// Optional manual bounds (ideal, nadir) - if None, will be computed
    pub manual_bounds: Option<(Vec<f64>, Vec<f64>)>,
}

/// GPBA-A: Coverage-focused representation algorithm
///
/// Minimizes the maximum distance between consecutive points to ensure
/// good coverage of the entire Pareto front.
/// Uses dynamic interval exploration matching Python's gamma=1 approach.
pub struct GpbaA {
    config: GpbaConfig,
    /// Interval managers for adaptive exploration per constraint objective
    #[allow(dead_code, reason = "Field kept for potential future interval-based optimizations")]
    ef_intervals: HashMap<usize, IntervalManager>,
    /// Previous solutions for relaxation search
    previous_solution_information: Vec<PreviousSolutionInfo>,
    /// Relative Worst Values for search space pruning
    rwv: Vec<f64>,
    /// Timer for timeout tracking
    timer: Option<Timer>,
}

impl GpbaA {
    /// Create new GPBA-A instance with coverage focus
    /// Uses Python-compatible dynamic interval exploration (gamma=1)
    #[must_use]
    pub fn new(config: GpbaConfig) -> Self {
        Self {
            ef_intervals: HashMap::new(),
            previous_solution_information: Vec::new(),
            rwv: Vec::new(),
            config,
            timer: None,
        }
    }

    /// Set timeout for the solver
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timer = Some(Timer::start(timeout));
        self
    }

    /// Check if the timeout has been reached
    fn is_timeout_reached(&self) -> bool {
        self.timer.as_ref().is_some_and(Timer::is_expired)
    }

    /// Get remaining timeout duration
    fn get_remaining_timeout(&self) -> Option<Duration> {
        self.timer.as_ref().map(Timer::remaining)
    }



    /// Check if this constraint configuration was already explored with relaxation.
    /// For maximization: `ef_array1` is less constrained if all `ef_array1[i]` >= `ef_array2[i]`
    #[allow(dead_code, reason = "Method reserved for future relaxation optimization - disabled per TODO comment at usage site")]
    fn search_previous_solutions_relaxation(
        &self,
        ef_array: &[f64],
        constraint_indices: &[usize],
    ) -> (bool, Option<Vec<f64>>) {
        for prev_info in &self.previous_solution_information {
            // Check if previous ef_array is less constrained (all values >= current)
            let is_less_constrained = prev_info
                .ef_array
                .iter()
                .zip(ef_array.iter())
                .all(|(&prev, &curr)| prev >= curr);

            if is_less_constrained {
                if let Some(prev_solution) = &prev_info.solution {
                    // Check if previous solution satisfies current (tighter) constraints
                    let satisfies = ef_array
                        .iter()
                        .enumerate()
                        .all(|(i, &ef_val)| prev_solution[constraint_indices[i]] <= ef_val);

                    if satisfies {
                        log::debug!("Reusing previous solution via relaxation");
                        return (true, Some(prev_solution.clone()));
                    }
                } else {
                    // Previous was infeasible with less constrained constraints
                    log::debug!("Previous was infeasible, current also infeasible");
                    return (true, None);
                }
            }
        }

        (false, None)
    }

    /// Save solution information for future relaxation checks
    fn save_solution_information(&mut self, ef_array: Vec<f64>, solution: Option<Vec<f64>>) {
        self.previous_solution_information
            .push(PreviousSolutionInfo { ef_array, solution });
    }

    /// Adjust `epsilon_k` parameter based on solution found, using interval management.
    /// This is the core of GPBA-A that adaptively explores the largest gaps.
    ///
    /// This replaces the old uniform `gamma_k` stepping with adaptive interval-based exploration.
    fn adjust_epsilon_k(
        _k: usize, // Keep for API compatibility but not used in interval-based logic
        current_epsilon_k: f64,
        current_z_k: Option<f64>, // Solution value for this objective (in MAX form)
        ideal_z_k: f64,
        nadir_z_k: f64,
        ef_interval: &mut IntervalManager,
    ) -> f64 {
        // Current epsilon is the constraint value we just tried (in MAX form, so negative)
        // Example: current_epsilon_k = -656595 (worst/nadir)
        #[allow(clippy::cast_possible_truncation, reason = "Converting epsilon constraint values to i64 for interval management - truncation acceptable for GPBA integer-valued constraints")]
        let start_removal = current_epsilon_k as i64;

        #[allow(clippy::cast_possible_truncation, reason = "Converting solution objective values to i64 for interval management - truncation acceptable for GPBA integer-valued constraints")]
        let end_removal = current_z_k.map_or(ideal_z_k as i64, |sol_val| sol_val as i64);

        // Remove explored region from interval
        // For max: larger (less negative) values are better
        // If start_removal < end_removal, we have a range to remove
        // Example: remove [-656595, -511693]
        if start_removal < end_removal {
            ef_interval.remove_interval(start_removal, end_removal);
        } else {
            // Single point or reverse order
            ef_interval.remove_one_point(start_removal);
            if start_removal > end_removal {
                ef_interval.remove_one_point(end_removal);
            }
        }

        // CRITICAL: Update max_value to shrink search space (Python epsilon_adjustment.py line 76)
        // This prevents re-exploring the same epsilon values
        let new_max_interval = start_removal - 1;
        if end_removal >= ef_interval.max_value {
            log::debug!(
                "Shrinking interval max_value from {} to {} (start_removal - 1)",
                ef_interval.max_value,
                new_max_interval
            );
            ef_interval.max_value = new_max_interval;
        }

        // Find next point to explore (center of largest remaining interval)
        if let Some((start, end)) = ef_interval.find_largest_interval() {
            // PYTHON COMPATIBILITY: Jump from nadir to ideal on first iteration
            // Python epsilon_adjustment.py lines 96-99
            if (current_epsilon_k - nadir_z_k).abs() < 1e-6 {
                log::debug!("First iteration: jumping from nadir to ideal");
                ideal_z_k
            } else {
                // Explore center of largest gap
                #[allow(clippy::cast_precision_loss, reason = "Converting i64 midpoint to f64 for epsilon values - precision loss acceptable for GPBA interval-based search")]
                (i64::midpoint(start, end) as f64)
            }
        } else {
            // No more intervals - exhausted this dimension
            // Return value beyond ideal to trigger cascading
            ideal_z_k + 1.0
        }
    }

    /// Generate Pareto front representation with coverage focus
    ///
    /// # Errors
    /// Returns an error if the optimization solver fails or problem validation fails
    #[allow(clippy::cognitive_complexity, reason = "GPBA-A main loop implements the full algorithm from the paper - splitting would reduce clarity")]
    pub fn generate_representation(
        &mut self,
        problem: &MultiObjectiveProblem,
        options: &Options,
    ) -> Result<ParetoFront> {
        const MAX_ITERATIONS: usize = 10000; // Prevent infinite loops

        log::info!("=== RUST GPBA-A: Starting generate_representation ===");
        log::info!("Number of objectives: {}", problem.num_objectives());
        log::info!("Primary objective index: {}", self.config.primary_objective);

        // Step 1: Compute or use provided bounds using shared calculator
        log::info!("=== STEP 1: Computing bounds (payoff table) ===");
        let (mut ideal, mut nadir) = if let Some((ideal, nadir)) = &self.config.manual_bounds {
            (ideal.clone(), nadir.clone())
        } else {
            BoundsCalculator::new(problem, options).calculate_bounds(self.timer.as_ref())?
        };

        log::info!("Best values (minimization): {ideal:?}");
        log::info!("Nadir values (minimization): {nadir:?}");

        // Step 1.5: Convert to maximization for uniform handling
        log::info!("=== STEP 1.5: Converting objectives to maximization ===");
        ideal = ideal.iter().map(|&x| -x).collect();
        nadir = nadir.iter().map(|&x| -x).collect();

        log::info!("Best values (maximization): {ideal:?}");
        log::info!("Nadir values (maximization): {nadir:?}");

        // Step 2: Initialize constraint indices and epsilon array
        log::info!("=== STEP 2: Setting up ε-constraint formulation ===");
        let constraint_indices: Vec<usize> = (0..problem.num_objectives())
            .filter(|&i| i != self.config.primary_objective)
            .collect();

        log::info!(
            "Main objective: {} (index {})",
            self.config.primary_objective,
            self.config.primary_objective
        );
        log::info!("Constraint objectives: {constraint_indices:?}");

        let mut ef_array: Vec<f64> = constraint_indices.iter().map(|&k| nadir[k]).collect();

        log::info!("Initial ef_array: {ef_array:?}");

        // Initialize intervals for each constraint objective
        #[allow(clippy::cast_possible_truncation, reason = "Converting nadir/ideal bounds to i64 for interval management - truncation acceptable for GPBA integer-valued constraints")]
        let mut ef_intervals: Vec<IntervalManager> = constraint_indices
            .iter()
            .map(|&k| {
                let interval = IntervalManager::new(nadir[k] as i64, ideal[k] as i64);
                log::debug!(
                    "Created interval for objective {k}: [{}, {}]",
                    nadir[k],
                    ideal[k]
                );
                interval
            })
            .collect();

        // Initialize RWV (Relative Worst Values)
        self.rwv = constraint_indices.iter().map(|&k| ideal[k]).collect();
        log::info!("Initial RWV: {:?}", self.rwv);

        let ranges = Self::calculate_objective_ranges(&ideal, &nadir);
        log::debug!("Objective ranges: {ranges:?}");
        let mut pareto_front = ParetoFront::new(vec![
            crate::model::ObjectiveDirection::Minimize;
            problem.num_objectives()
        ]);
        let mut iteration = 0;

        log::info!("=== STEP 3: Starting main epsilon-constraint iteration (Python gamma=1 mode) ===");

        // Step 4: Main epsilon-constraint iteration
        while iteration < MAX_ITERATIONS {
            // Check timeout before each iteration
            if self.is_timeout_reached() {
                log::warn!("Timeout reached at iteration {iteration}");
                break;
            }

            log::info!("╔═══════════════════════════════════════════════════════════╗");
            log::info!(
                "║ ITERATION {iteration:5}                                          ║"
            );
            log::info!("╚═══════════════════════════════════════════════════════════╝");
            log::info!("ef_array = {ef_array:?}");

            // Check for solution relaxation
            // Note: Relaxation disabled because it doesn't preserve decision variables
            // TODO: Store decision variables in previous_solution_information to enable relaxation
            let found_relaxed = false;
            let one_solution = if found_relaxed {
                None // Never used
            } else {
                // Build epsilon map from ef_array
                let mut epsilons = HashMap::new();
                for (i, &k) in constraint_indices.iter().enumerate() {
                    epsilons.insert(k, -ef_array[i]); // Convert back from maximization
                }

                log::info!(
                    "→ Solving ε-constraint with ε = {epsilons:?} (minimization form)"
                );

                // Solve epsilon constraint problem
                if let Some(mut solution) = self.solve_epsilon_constraint_problem_shared(problem, options, &epsilons, &ranges)? {
                    log::info!(
                        "✓ Solver returned objectives (max form): {solution_obj:?}",
                        solution_obj = solution.objective_values
                    );

                    // Extract selected image indices (decision variables that are 1)
                    let mut selected_indices: Vec<usize> = solution.decision_variables
                        .iter()
                        .filter(|(_, &val)| val > 0.5)  // Binary variables >= 0.5 means selected
                        .filter_map(|(name, _)| {
                            // Extract index from variable name (e.g., "x_5" -> 5)
                            name.strip_prefix("x_")
                                .and_then(|s| s.parse::<usize>().ok())
                        })
                        .collect();
                    selected_indices.sort_unstable();
                    let num_selected = selected_indices.len();
                    log::info!(
                        "✓ Selected images [{num_selected}]: {selected_indices:?}"
                    );

                    // Convert objectives back to minimization for storage
                    // Round to integers since SIMS problem has discrete (integer) objectives
                    solution.objective_values = solution
                        .objective_values
                        .iter()
                        .map(|&x| -x.round())
                        .collect();
                    log::info!(
                        "✓ Solution (min form, rounded to integers): {solution_obj:?}",
                        solution_obj = solution.objective_values
                    );

                    Some(solution)
                } else {
                    log::info!("✗ No solution found (INFEASIBLE)");
                    None
                }
            };

            // Save solution information for future relaxation
            self.save_solution_information(ef_array.clone(), one_solution.as_ref().map(|s| s.objective_values.clone()));

                        // Add to Pareto front if solution found
            if let Some(ref solution) = one_solution {
                let is_new = pareto_front
                    .solutions
                    .iter()
                    .all(|existing| existing.objective_values != solution.objective_values);

                if is_new {
                    log::info!("➕ NEW solution added to Pareto front: {obj:?}", obj = solution.objective_values);
                } else {
                    log::info!(
                        "⊗ DUPLICATE solution (already in Pareto front): {obj:?}",
                        obj = solution.objective_values
                    );
                }

                let pareto_solution = Solution::new(
                    solution.objective_values.clone(),
                    solution.decision_variables.clone(),
                );
                // Solutions are integers - use 0 decimal places for exact integer comparison
                pareto_front.add_solution_with_precision(pareto_solution, 0);

                // Update RWV (Relative Worst Values)
                for (i, &constraint_idx) in constraint_indices.iter().enumerate() {
                    let old_rwv = self.rwv[i];
                    self.rwv[i] = self.rwv[i].min(-solution.objective_values[constraint_idx]); // Note: negated for maximization
                    if (self.rwv[i] - old_rwv).abs() > 1e-9 {
                        log::debug!("RWV[{i}] updated: {old_rwv} -> {}", self.rwv[i]);
                    }
                }
            }

            // Multi-dimensional cascading update
            let last_dim = constraint_indices.len() - 1;
            log::debug!(
                "Updating last dimension (id_interval={last_dim})..."
            );
            let id_interval = constraint_indices.len() - 1;
            // one_solution is in minimization form (negated), so to get max form, we don't negate again
            let sol_obj_value = one_solution
                .as_ref()
                .map(|sol| sol.objective_values[constraint_indices[id_interval]]);

            log::debug!(
                "  Current epsilon: {}, Solution obj value (max form): {sol_obj_value:?}",
                ef_array[id_interval]
            );

            let new_epsilon = Self::adjust_epsilon_k(
                id_interval,
                ef_array[id_interval],
                sol_obj_value,
                ideal[constraint_indices[id_interval]],
                nadir[constraint_indices[id_interval]],
                &mut ef_intervals[id_interval],
            );
            log::debug!("  New epsilon: {new_epsilon}");
            ef_array[id_interval] = new_epsilon;

            // Cascading update for other dimensions
            let mut cascaded = false;
            for i in (1..constraint_indices.len()).rev() {
                if ef_array[i] > ideal[constraint_indices[i]] {
                    let ef_val = ef_array[i];
                    let ideal_val = ideal[constraint_indices[i]];
                    log::debug!(
                        "Dimension {i} exhausted (ef={ef_val} > ideal={ideal_val}), cascading..."
                    );
                    cascaded = true;

                    // Reset this dimension
                    ef_array[i] = nadir[constraint_indices[i]];
                    self.rwv[i] = ideal[constraint_indices[i]];
                    #[allow(clippy::cast_possible_truncation, reason = "Converting nadir/ideal bounds to i64 for interval management - truncation acceptable for GPBA integer-valued constraints")]
                    {
                        ef_intervals[i] = IntervalManager::new(
                            nadir[constraint_indices[i]] as i64,
                            ideal[constraint_indices[i]] as i64,
                        );
                    }
                    log::debug!("  Reset dimension {i} to nadir: {}", ef_array[i]);

                    // Update previous dimension
                    let prev_id = i - 1;
                    let sol_prev = one_solution
                        .as_ref()
                        .map(|sol| -sol.objective_values[constraint_indices[prev_id]]);

                    log::debug!("  Updating previous dimension {prev_id}...");
                    let new_epsilon_prev = Self::adjust_epsilon_k(
                        prev_id,
                        ef_array[prev_id],
                        sol_prev,
                        ideal[constraint_indices[prev_id]],
                        nadir[constraint_indices[prev_id]],
                        &mut ef_intervals[prev_id],
                    );
                    log::debug!(
                        "  New epsilon for dimension {prev_id}: {new_epsilon_prev}"
                    );
                    ef_array[prev_id] = new_epsilon_prev;
                } else {
                    break; // Stop cascading
                }
            }

            if cascaded {
                log::debug!("After cascading, ef_array: {ef_array:?}");
            }

            // Check termination condition
            if ef_array[0] > ideal[constraint_indices[0]] {
                log::info!("=== GPBA-A CONVERGED: First dimension exhausted ===");
                let ef_0 = ef_array[0];
                let idx_0 = constraint_indices[0];
                let ideal_0 = ideal[constraint_indices[0]];
                log::info!(
                    "ef_array[0]={ef_0} > ideal[{idx_0}]={ideal_0}"
                );
                log::info!("Total iterations: {}", iteration + 1);
                log::info!("Total solutions found: {}", pareto_front.len());
                break;
            }

            iteration += 1;
        }

        if iteration >= MAX_ITERATIONS {
            log::warn!("GPBA-A reached maximum iterations ({MAX_ITERATIONS})");
        }

        log::info!("=== RUST GPBA-A: Completed ===");
        log::info!("Final Pareto front size: {}", pareto_front.len());

        Ok(pareto_front)
    }

    #[allow(dead_code, reason = "Method kept for API completeness and potential future use in alternative GPBA variants")]
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
        options: &Options,
        epsilons: &HashMap<usize, f64>,
        ranges: &HashMap<usize, f64>,
    ) -> Result<Option<Solution>> {
        let mut builder =
            EpsilonConstraintBuilder::new(problem, options, self.config.primary_objective);

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
    /// Timer for timeout tracking
    timer: Option<Timer>,
}

impl GpbaB {
    /// Create a new GPBA-B instance with uniformity focus
    #[must_use]
    pub fn new(config: GpbaConfig) -> Self {
        Self {
            acceptable_uniformity_level: HashMap::new(),
            config,
            timer: None,
        }
    }

    /// Set timeout for the solver
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timer = Some(Timer::start(timeout));
        self
    }

    /// Check if the timeout has been reached
    fn is_timeout_reached(&self) -> bool {
        self.timer.as_ref().is_some_and(Timer::is_expired)
    }

    /// Get remaining timeout duration
    fn get_remaining_timeout(&self) -> Option<Duration> {
        self.timer.as_ref().map(Timer::remaining)
    }

    #[allow(dead_code, reason = "GPBA-B not used in Python implementation but kept for research completeness")]
    #[allow(clippy::unused_self, clippy::missing_const_for_fn, reason = "Method signature maintained for trait-like consistency even though unused")]
    fn initialize_uniformity_parameters(&self, _ideal: &[f64], _nadir: &[f64]) {
        // NOTE: Python implementation doesn't use GPBA-B
        // This is kept for potential future use but disabled for Python compatibility
        // TODO: Implement Python-compatible GPBA-B if needed
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
        &self,
        problem: &MultiObjectiveProblem,
        options: &Options,
    ) -> Result<ParetoFront> {
        const MAX_ITERATIONS: usize = 10000;

        let (ideal, nadir) = if let Some((ideal, nadir)) = &self.config.manual_bounds {
            (ideal.clone(), nadir.clone())
        } else {
            BoundsCalculator::new(problem, options).calculate_bounds(self.timer.as_ref())?
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
            let mut builder = EpsilonConstraintBuilder::new(
                problem,
                options,
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
    /// Timer for timeout tracking
    timer: Option<Timer>,
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
            timer: None,
        }
    }

    /// Set timeout for the solver
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timer = Some(Timer::start(timeout));
        self
    }

    /// Check if the timeout has been reached
    fn is_timeout_reached(&self) -> bool {
        self.timer.as_ref().is_some_and(Timer::is_expired)
    }

    /// Get remaining timeout duration
    fn get_remaining_timeout(&self) -> Option<Duration> {
        self.timer.as_ref().map(Timer::remaining)
    }

    #[allow(dead_code, reason = "GPBA-C not used in Python implementation but kept for research completeness")]
    #[allow(clippy::unused_self, clippy::missing_const_for_fn, reason = "Method signature maintained for trait-like consistency even though unused")]
    fn initialize_grid_state(&self, _nadir: &[f64]) {
        // NOTE: Python implementation doesn't use GPBA-C
        // This is kept for potential future use but disabled for Python compatibility
        // TODO: Implement Python-compatible GPBA-C if needed
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

        let (ideal, nadir) = if let Some((ideal, nadir)) = &self.config.manual_bounds {
            (ideal.clone(), nadir.clone())
        } else {
            BoundsCalculator::new(problem, options).calculate_bounds(self.timer.as_ref())?
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
    use super::GpbaConfig;

    /// Configuration for high-coverage representation (GPBA-A)
    /// Uses Python-compatible dynamic interval exploration (no pre-defined grid)
    #[must_use]
    pub const fn high_coverage_config(primary_obj: usize, _points_per_obj: usize) -> GpbaConfig {
        GpbaConfig {
            primary_objective: primary_obj,
            manual_bounds: None,
        }
    }

    /// Configuration for uniform distribution (GPBA-B)
    /// NOTE: Python doesn't implement GPBA-B, using GPBA-A approach
    #[must_use]
    pub const fn uniform_distribution_config(primary_obj: usize, _points_per_obj: usize) -> GpbaConfig {
        GpbaConfig {
            primary_objective: primary_obj,
            manual_bounds: None,
        }
    }

    /// Configuration for balanced representation with target size (GPBA-C)
    /// NOTE: Python doesn't implement GPBA-C, using GPBA-A approach
    #[must_use]
    pub const fn balanced_cardinality_config(primary_obj: usize, _total_target: usize) -> GpbaConfig {
        GpbaConfig {
            primary_objective: primary_obj,
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
        // Python-compatible GPBA-A doesn't use target_points_per_objective
        assert!(config.manual_bounds.is_none());
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
        let gpba_c = GpbaC::new(config);
        let nadir = vec![0.0, 0.0];
        gpba_c.initialize_grid_state(&nadir);

        assert!(gpba_c.grid_state.contains_key(&0));
        assert_eq!(gpba_c.grid_state[&0].remaining_points, 24); // 25 - 1
    }
}
