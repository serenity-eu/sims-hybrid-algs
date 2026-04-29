//! GPBA algorithm implementations for advanced Pareto front representation
//!
//! This module provides implementations of the three Grid Point Based Algorithms (GPBA)
//! from Mesquita-Cunha et al. (2023) for generating high-quality representations of
//! Pareto fronts in multi-objective integer linear programming.
//!
//! ## Sign Convention
//!
//! Internally, GPBA-A works in **maximization form** (all objectives negated):
//! - `ideal[k]` = best value for objective k in MAX form (least negative = smallest cost)
//! - `nadir[k]` = worst value for objective k in MAX form (most negative = largest cost)
//! - `ef_array[i]` = current epsilon constraint value in MAX form
//! - Interval managers track ranges in MAX form [nadir, ideal]
//!
//! Solutions returned by the epsilon-constraint solver are in **minimization form**
//! (the original SIMS objective values). They are converted to MAX form (negated)
//! for interval tracking, and stored in minimization form in the Pareto front.

use crate::{
    bounds::BoundsCalculator,
    epsilon_constraint::EpsilonConstraintBuilder,
    error::Result,
    interval_manager::IntervalManager,
    model::MultiObjectiveProblem,
    options::Options,
    solution::{ParetoFront, Solution},
    timer::Timer,
};
use std::collections::HashMap;
use std::time::Duration;

/// Information about a previously solved epsilon-constraint configuration.
/// Used by relaxation search to avoid redundant MILP solves.
#[derive(Debug, Clone)]
struct PreviousSolutionInfo {
    /// The epsilon array (constraint values) that was solved (MAX form)
    ef_array: Vec<f64>,
    /// The solution objective values found (minimization form), None if infeasible
    solution: Option<Vec<f64>>,
}

/// Configuration for GPBA representation algorithms
#[derive(Debug, Clone)]
pub struct GpbaConfig {
    /// Primary objective index to optimize directly
    pub primary_objective: usize,
    /// Optional manual bounds (ideal, nadir) in minimization form - if None, will be computed
    pub manual_bounds: Option<(Vec<f64>, Vec<f64>)>,
    /// Target number of Pareto-optimal solutions.  When the front reaches this
    /// size the algorithm terminates early.  `None` means no target (explore
    /// until the search space is exhausted or the global timeout fires).
    pub target_solutions: Option<usize>,
    /// Maximum wall-clock time budget for a single ε-constraint MILP solve.
    /// When `None`, each solve is only bounded by the remaining global timeout.
    /// Setting this prevents a single hard sub-problem from consuming the
    /// entire time budget.
    pub per_solve_timeout: Option<Duration>,
}

/// GPBA-A: Coverage-focused representation algorithm
///
/// Minimizes the maximum distance between consecutive points to ensure
/// good coverage of the entire Pareto front.
/// Uses dynamic interval exploration matching Python's gamma=1 approach.
pub struct GpbaA {
    config: GpbaConfig,
    /// Previous solutions for relaxation search
    previous_solution_information: Vec<PreviousSolutionInfo>,
    /// Relative Worst Values for search space pruning (MAX form)
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
    ///
    /// For maximization: a previous `ef_array` is "less constrained" (more relaxed) if
    /// all `prev_ef[i] >= current_ef[i]`.  When such a previous configuration exists:
    /// - If its solution is feasible **and** satisfies the current (tighter) constraints,
    ///   we can reuse it without solving a new MILP.
    /// - If it was infeasible, the current (tighter) configuration is also infeasible.
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
                    // prev_solution is in MIN form, constraint_indices index into it
                    // ef_array is in MAX form, so compare negated solution values
                    let satisfies = ef_array.iter().enumerate().all(|(i, &ef_val)| {
                        let sol_val_max = -prev_solution[constraint_indices[i]];
                        sol_val_max >= ef_val
                    });

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
    /// All values (`current_epsilon_k`, `current_z_k`, `ideal_z_k`, `nadir_z_k`) must be
    /// in **MAX form** (negated minimization values). The interval manager also works in
    /// MAX form where the range is [nadir (most negative), ideal (least negative)].
    ///
    /// Returns the next epsilon value to try (in MAX form).
    fn adjust_epsilon_k(
        _k: usize, // Keep for API compatibility
        current_epsilon_k: f64,
        current_z_k: Option<f64>, // Solution value for this objective (MAX form)
        ideal_z_k: f64,
        nadir_z_k: f64,
        ef_interval: &mut IntervalManager,
    ) -> f64 {
        // current_epsilon_k = the constraint value we just tried (MAX form)
        // current_z_k = the solution's actual value for this objective (MAX form)
        //
        // In MAX form: ideal > nadir (ideal is least negative, nadir is most negative)
        // The solver finds the best primary objective subject to constraint_obj >= epsilon
        // So if epsilon = nadir, the constraint is very loose; if epsilon = ideal, very tight.
        //
        // After solving with epsilon = current_epsilon_k:
        // - If feasible, the solution has current_z_k >= current_epsilon_k
        // - The range [current_epsilon_k, current_z_k] has been "explored" because
        //   any epsilon in this range would yield the same or dominated solution
        //   (the solver already found the best it could with the looser constraint)

        #[allow(
            clippy::cast_possible_truncation,
            reason = "Converting epsilon constraint values to i64 for interval management - truncation acceptable for GPBA integer-valued constraints"
        )]
        let epsilon_i64 = current_epsilon_k as i64;

        #[allow(
            clippy::cast_possible_truncation,
            reason = "Converting solution objective values to i64 for interval management - truncation acceptable for GPBA integer-valued constraints"
        )]
        let solution_i64 = current_z_k.map_or(ideal_z_k as i64, |sol_val| sol_val as i64);

        // Remove explored region from interval
        // In MAX form: epsilon <= solution_value (epsilon is the lower bound constraint)
        // We explored everything from epsilon up to the solution value
        match epsilon_i64.cmp(&solution_i64) {
            std::cmp::Ordering::Less => {
                log::debug!(
                    "  Removing interval [{epsilon_i64}, {solution_i64}] from search space"
                );
                ef_interval.remove_interval(epsilon_i64, solution_i64);
            }
            std::cmp::Ordering::Equal => {
                // Exact match: just remove the single point
                ef_interval.remove_one_point(epsilon_i64);
            }
            std::cmp::Ordering::Greater => {
                // epsilon > solution: infeasible or solver returned worse than constraint
                // Remove both points
                ef_interval.remove_one_point(epsilon_i64);
                ef_interval.remove_one_point(solution_i64);
            }
        }

        // Shrink the search space upper bound
        // If the solution value is at or above the current max, we know everything
        // above the epsilon has been explored
        let new_max = epsilon_i64 - 1;
        if solution_i64 >= ef_interval.max_value {
            log::debug!(
                "  Shrinking interval max_value from {} to {}",
                ef_interval.max_value,
                new_max
            );
            ef_interval.max_value = new_max;
        }

        // Find next point to explore (center of largest remaining interval)
        if let Some((start, end)) = ef_interval.find_largest_interval() {
            let interval_size = end - start;
            if interval_size <= 0 {
                // Single point interval
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "Converting i64 to f64 for epsilon values - precision loss acceptable for GPBA"
                )]
                return start as f64;
            }

            // On first iteration (epsilon was at nadir), jump to ideal to establish bounds
            if (current_epsilon_k - nadir_z_k).abs() < 1e-6 {
                log::debug!("  First iteration: jumping from nadir to ideal");
                return ideal_z_k;
            }

            // Explore center of largest gap
            #[allow(
                clippy::cast_precision_loss,
                reason = "Converting i64 midpoint to f64 for epsilon values - precision loss acceptable for GPBA interval-based search"
            )]
            let midpoint = (i64::midpoint(start, end)) as f64;
            log::debug!(
                "  Next epsilon: {midpoint} (midpoint of largest interval [{start}, {end}], size={interval_size})"
            );
            midpoint
        } else {
            // No more intervals - exhausted this dimension
            // Return value beyond ideal to trigger cascading
            log::debug!("  No intervals remaining, returning beyond ideal to trigger cascade");
            ideal_z_k + 1.0
        }
    }

    /// Generate Pareto front representation with coverage focus
    ///
    /// # Errors
    /// Returns an error if the optimization solver fails or problem validation fails
    #[allow(
        clippy::cognitive_complexity,
        reason = "GPBA-A main loop implements the full algorithm from the paper - splitting would reduce clarity"
    )]
    pub fn generate_representation(
        &mut self,
        problem: &MultiObjectiveProblem,
        options: &Options,
    ) -> Result<ParetoFront> {
        const MAX_ITERATIONS: usize = 10000; // Prevent infinite loops
        const MAX_CONSECUTIVE_DUPLICATES: usize = 10; // Safety valve (lowered from 20)

        log::info!("=== GPBA-A: Starting generate_representation ===");
        log::info!("Number of objectives: {}", problem.num_objectives());
        log::info!("Primary objective index: {}", self.config.primary_objective);

        // Step 1: Compute or use provided bounds using shared calculator
        log::info!("=== STEP 1: Computing bounds (payoff table) ===");
        let (ideal_min, nadir_min) = if let Some((ideal, nadir)) = &self.config.manual_bounds {
            (ideal.clone(), nadir.clone())
        } else {
            BoundsCalculator::new(problem, options).calculate_bounds(self.timer.as_ref())?
        };

        log::info!("Ideal point (minimization): {ideal_min:?}");
        log::info!("Nadir point (minimization): {nadir_min:?}");

        // Step 1.5: Convert to maximization form for uniform handling
        // In MAX form: negate minimization values
        // ideal_max[k] = -ideal_min[k] (best = least negative)
        // nadir_max[k] = -nadir_min[k] (worst = most negative)
        log::info!("=== STEP 1.5: Converting to maximization form ===");
        let ideal_max: Vec<f64> = ideal_min.iter().map(|&x| -x).collect();
        let nadir_max: Vec<f64> = nadir_min.iter().map(|&x| -x).collect();

        log::info!("Ideal (MAX form): {ideal_max:?}");
        log::info!("Nadir (MAX form): {nadir_max:?}");

        // Validate: in MAX form, ideal[k] >= nadir[k] for all k
        for k in 0..ideal_max.len() {
            if ideal_max[k] < nadir_max[k] {
                log::warn!(
                    "Objective {k}: ideal_max ({}) < nadir_max ({}) — swapping",
                    ideal_max[k],
                    nadir_max[k]
                );
            }
        }

        // Step 2: Initialize constraint indices and epsilon array
        log::info!("=== STEP 2: Setting up ε-constraint formulation ===");
        let constraint_indices: Vec<usize> = (0..problem.num_objectives())
            .filter(|&i| i != self.config.primary_objective)
            .collect();

        log::info!(
            "Primary objective: {} (index {})",
            self.config.primary_objective,
            self.config.primary_objective
        );
        log::info!("Constraint objectives: {constraint_indices:?}");

        // ef_array: current epsilon values in MAX form, starting at nadir (loosest constraint)
        let mut ef_array: Vec<f64> = constraint_indices.iter().map(|&k| nadir_max[k]).collect();

        log::info!("Initial ef_array (MAX form): {ef_array:?}");

        // Initialize intervals for each constraint objective (in MAX form)
        // Range: [nadir_max[k], ideal_max[k]] where nadir < ideal
        #[allow(
            clippy::cast_possible_truncation,
            reason = "Converting nadir/ideal bounds to i64 for interval management - truncation acceptable for GPBA integer-valued constraints"
        )]
        let mut ef_intervals: Vec<IntervalManager> = constraint_indices
            .iter()
            .map(|&k| {
                let lo = nadir_max[k] as i64;
                let hi = ideal_max[k] as i64;
                // Ensure lo <= hi
                let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
                let interval = IntervalManager::new(lo, hi);
                log::debug!("Created interval for objective {k}: [{lo}, {hi}]");
                interval
            })
            .collect();

        // Initialize RWV (Relative Worst Values) in MAX form
        // Start at ideal (best) — will be updated to the worst solution found
        self.rwv = constraint_indices.iter().map(|&k| ideal_max[k]).collect();
        log::info!("Initial RWV (MAX form): {:?}", self.rwv);

        let ranges = Self::calculate_objective_ranges(&ideal_max, &nadir_max);
        log::debug!("Objective ranges: {ranges:?}");

        let mut pareto_front = ParetoFront::new(vec![
            crate::model::ObjectiveDirection::Minimize;
            problem.num_objectives()
        ]);

        let mut iteration = 0;
        let mut consecutive_duplicates = 0;
        let mut relaxation_reuses: usize = 0;
        // Track explored epsilon configurations to avoid exact re-solves
        let mut explored_epsilons: std::collections::HashSet<Vec<i64>> =
            std::collections::HashSet::new();

        log::info!("=== STEP 3: Starting main epsilon-constraint iteration ===");

        while iteration < MAX_ITERATIONS {
            // Check timeout before each iteration
            if self.is_timeout_reached() {
                log::warn!("Timeout reached at iteration {iteration}");
                break;
            }

            // Early termination: target number of solutions reached
            if let Some(target) = self.config.target_solutions {
                if pareto_front.len() >= target {
                    log::info!(
                        "🎯 Target of {target} solutions reached (front has {}), stopping early",
                        pareto_front.len()
                    );
                    break;
                }
            }

            // Safety: too many consecutive duplicates means we're stuck
            if consecutive_duplicates >= MAX_CONSECUTIVE_DUPLICATES {
                log::warn!(
                    "Stopping after {consecutive_duplicates} consecutive duplicate solutions — search space likely exhausted"
                );
                break;
            }

            log::info!("╔═══════════════════════════════════════════════════════════╗");
            log::info!(
                "║ ITERATION {iteration:5}  (front size: {:4})                     ║",
                pareto_front.len()
            );
            log::info!("╚═══════════════════════════════════════════════════════════╝");
            log::info!("ef_array (MAX form) = {ef_array:?}");

            // Check if we've already tried this exact epsilon configuration
            #[allow(
                clippy::cast_possible_truncation,
                reason = "Converting epsilon to i64 for dedup tracking"
            )]
            let ef_key: Vec<i64> = ef_array.iter().map(|&v| v as i64).collect();
            if !explored_epsilons.insert(ef_key) {
                log::info!("⊘ Epsilon configuration already explored, skipping solve");
                consecutive_duplicates += 1;

                // Force advancement: try to move to next interval in last dimension
                let last_dim = constraint_indices.len() - 1;
                ef_array[last_dim] = Self::force_advance_epsilon(
                    ef_array[last_dim],
                    ideal_max[constraint_indices[last_dim]],
                    &mut ef_intervals[last_dim],
                );

                // Check if we need to cascade
                if ef_array[last_dim] > ideal_max[constraint_indices[last_dim]] {
                    // Cascade through dimensions
                    let did_cascade = Self::cascade_dimensions(
                        &mut ef_array,
                        &mut ef_intervals,
                        &constraint_indices,
                        &ideal_max,
                        &nadir_max,
                        &mut self.rwv,
                    );
                    if !did_cascade {
                        log::info!("All dimensions exhausted after forced advance");
                        break;
                    }
                }

                iteration += 1;
                continue;
            }

            // ─── Relaxation search: try to reuse a previous solution ────
            let (relaxation_hit, relaxation_solution) =
                self.search_previous_solutions_relaxation(&ef_array, &constraint_indices);

            #[allow(
                clippy::option_if_let_else,
                reason = "if-let reads more clearly than map_or_else for this branching logic"
            )]
            let one_solution = if relaxation_hit {
                relaxation_reuses += 1;
                // Relaxation search found a match — skip the expensive MILP solve
                if let Some(prev_obj_vals) = relaxation_solution {
                    log::info!(
                        "♻ Relaxation reuse: previous solution satisfies current constraints: {prev_obj_vals:?}"
                    );
                    // Construct a lightweight Solution from the cached objective values.
                    // Decision variables are not available from the cache, so we store
                    // an empty map.  This is acceptable because the Pareto front only
                    // needs objective values for dominance checking.
                    let mut sol = Solution::new(prev_obj_vals.clone(), HashMap::new());
                    sol.objective_values = prev_obj_vals;
                    Some(sol)
                } else {
                    log::info!(
                        "♻ Relaxation propagation: previous config was infeasible → current is too"
                    );
                    None
                }
            } else {
                // No relaxation match — solve the MILP

                // Build epsilon map: convert from MAX form to MIN form for the solver
                let mut epsilons = HashMap::new();
                for (i, &k) in constraint_indices.iter().enumerate() {
                    epsilons.insert(k, -ef_array[i]); // MAX -> MIN: negate
                }

                log::info!("→ Solving ε-constraint with ε = {epsilons:?} (minimization form)");

                // Compute per-solve timeout: use the configured per-solve cap if set,
                // but never exceed the remaining global timeout.
                let per_solve_timeout = {
                    let remaining = self.get_remaining_timeout();
                    match (self.config.per_solve_timeout, remaining) {
                        (Some(cap), Some(rem)) => Some(cap.min(rem)),
                        (Some(cap), None) => Some(cap),
                        (None, Some(rem)) => Some(rem),
                        (None, None) => None,
                    }
                };

                // Solve epsilon constraint problem
                if let Some(mut solution) = self.solve_epsilon_constraint_problem_shared(
                    problem,
                    options,
                    &epsilons,
                    &ranges,
                    per_solve_timeout,
                )? {
                    log::info!(
                        "✓ Solver returned objectives (MIN form): {:?}",
                        solution.objective_values
                    );

                    // Extract selected image indices for logging
                    let mut selected_indices: Vec<usize> = solution
                        .decision_variables
                        .iter()
                        .filter(|(_, &val)| val > 0.5)
                        .filter_map(|(name, _)| {
                            name.strip_prefix("x_")
                                .and_then(|s| s.parse::<usize>().ok())
                        })
                        .collect();
                    selected_indices.sort_unstable();
                    log::info!(
                        "✓ Selected images [{}]: {:?}",
                        selected_indices.len(),
                        selected_indices
                    );

                    // Round to integers since SIMS has discrete objectives
                    // Keep in MINIMIZATION form for storage and Pareto front
                    solution.objective_values = solution
                        .objective_values
                        .iter()
                        .map(|&x| x.round())
                        .collect();
                    log::info!(
                        "✓ Solution (MIN form, rounded): {:?}",
                        solution.objective_values
                    );

                    Some(solution)
                } else {
                    log::info!("✗ No solution found (INFEASIBLE)");
                    None
                }
            };

            // Save solution information for future relaxation checks
            self.save_solution_information(
                ef_array.clone(),
                one_solution.as_ref().map(|s| s.objective_values.clone()),
            );

            // Add to Pareto front if solution found
            if let Some(ref solution) = one_solution {
                let is_new = pareto_front
                    .solutions
                    .iter()
                    .all(|existing| existing.objective_values != solution.objective_values);

                if is_new {
                    log::info!(
                        "➕ NEW solution added to Pareto front: {:?}",
                        solution.objective_values
                    );
                    consecutive_duplicates = 0;
                } else {
                    log::info!(
                        "⊗ DUPLICATE solution (already in Pareto front): {:?}",
                        solution.objective_values
                    );
                    consecutive_duplicates += 1;
                }

                let pareto_solution = Solution::new(
                    solution.objective_values.clone(),
                    solution.decision_variables.clone(),
                );
                // Solutions are integers - use 0 decimal places for exact comparison
                pareto_front.add_solution_with_precision(pareto_solution, 0);

                // Update RWV (Relative Worst Values) in MAX form
                // RWV tracks the worst (most constrained) solution value seen
                // In MAX form: worse = more negative = smaller value
                for (i, &constraint_idx) in constraint_indices.iter().enumerate() {
                    let sol_val_max = -solution.objective_values[constraint_idx]; // MIN -> MAX
                    let old_rwv = self.rwv[i];
                    self.rwv[i] = self.rwv[i].min(sol_val_max);
                    if (self.rwv[i] - old_rwv).abs() > 1e-9 {
                        log::debug!("RWV[{i}] updated: {old_rwv} -> {}", self.rwv[i]);
                    }
                }
            }

            // ─── Interval update and cascading ──────────────────────────

            // Get solution value in MAX form for interval update
            // solution.objective_values are in MIN form, so negate to get MAX form
            let last_dim = constraint_indices.len() - 1;

            let sol_max_last = one_solution.as_ref().map(|sol| {
                -sol.objective_values[constraint_indices[last_dim]] // MIN -> MAX
            });

            log::debug!(
                "Updating last dimension {last_dim}: current_epsilon={}, sol_value_max={sol_max_last:?}",
                ef_array[last_dim]
            );

            let new_epsilon = Self::adjust_epsilon_k(
                last_dim,
                ef_array[last_dim],
                sol_max_last,
                ideal_max[constraint_indices[last_dim]],
                nadir_max[constraint_indices[last_dim]],
                &mut ef_intervals[last_dim],
            );
            log::debug!("  New epsilon for dim {last_dim}: {new_epsilon}");
            ef_array[last_dim] = new_epsilon;

            // Cascading update for other dimensions (from last-1 down to 0)
            if ef_array[last_dim] > ideal_max[constraint_indices[last_dim]] {
                for i in (1..constraint_indices.len()).rev() {
                    if ef_array[i] > ideal_max[constraint_indices[i]] {
                        log::debug!(
                            "Dimension {i} exhausted (ef={} > ideal={}), cascading...",
                            ef_array[i],
                            ideal_max[constraint_indices[i]]
                        );

                        // Reset this dimension
                        ef_array[i] = nadir_max[constraint_indices[i]];
                        self.rwv[i] = ideal_max[constraint_indices[i]];
                        #[allow(
                            clippy::cast_possible_truncation,
                            reason = "Converting bounds to i64 for interval management"
                        )]
                        {
                            let lo = nadir_max[constraint_indices[i]] as i64;
                            let hi = ideal_max[constraint_indices[i]] as i64;
                            let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
                            ef_intervals[i] = IntervalManager::new(lo, hi);
                        }
                        log::debug!("  Reset dimension {i} to nadir: {}", ef_array[i]);

                        // Update previous dimension using the solution value in MAX form
                        // CRITICAL FIX: solution values are in MIN form, negate to get MAX form
                        // Previously this had a double-negation bug
                        let prev_id = i - 1;
                        let sol_max_prev = one_solution.as_ref().map(|sol| {
                            -sol.objective_values[constraint_indices[prev_id]] // MIN -> MAX
                        });

                        log::debug!(
                            "  Updating dimension {prev_id}: current_epsilon={}, sol_value_max={sol_max_prev:?}",
                            ef_array[prev_id]
                        );

                        let new_epsilon_prev = Self::adjust_epsilon_k(
                            prev_id,
                            ef_array[prev_id],
                            sol_max_prev,
                            ideal_max[constraint_indices[prev_id]],
                            nadir_max[constraint_indices[prev_id]],
                            &mut ef_intervals[prev_id],
                        );
                        log::debug!("  New epsilon for dim {prev_id}: {new_epsilon_prev}");
                        ef_array[prev_id] = new_epsilon_prev;
                    } else {
                        break; // Stop cascading
                    }
                }

                log::debug!("After cascading, ef_array: {ef_array:?}");
            }

            // Check termination condition: first dimension exhausted
            if ef_array[0] > ideal_max[constraint_indices[0]] {
                log::info!("=== GPBA-A CONVERGED: First dimension exhausted ===");
                log::info!(
                    "ef_array[0]={} > ideal[{}]={}",
                    ef_array[0],
                    constraint_indices[0],
                    ideal_max[constraint_indices[0]]
                );
                log::info!("Total iterations: {}", iteration + 1);
                log::info!("Total unique solutions found: {}", pareto_front.len());
                break;
            }

            iteration += 1;
        }

        if iteration >= MAX_ITERATIONS {
            log::warn!("GPBA-A reached maximum iterations ({MAX_ITERATIONS})");
        }

        log::info!("=== GPBA-A: Completed ===");
        log::info!("Final Pareto front size: {}", pareto_front.len());
        log::info!("Total iterations: {iteration}");
        log::info!(
            "Total epsilon configurations explored: {}",
            explored_epsilons.len()
        );
        log::info!("MILP solves avoided via relaxation reuse: {relaxation_reuses}");

        Ok(pareto_front)
    }

    /// Force-advance epsilon past the current position when we detect a duplicate
    /// epsilon configuration. Removes the current point and returns the next
    /// available point in the interval, or beyond-ideal if exhausted.
    fn force_advance_epsilon(current: f64, ideal: f64, interval: &mut IntervalManager) -> f64 {
        #[allow(clippy::cast_possible_truncation)]
        let current_i64 = current as i64;
        interval.remove_one_point(current_i64);

        if let Some((start, end)) = interval.find_largest_interval() {
            #[allow(clippy::cast_precision_loss)]
            let midpoint = (i64::midpoint(start, end)) as f64;
            midpoint
        } else {
            ideal + 1.0
        }
    }

    /// Cascade dimension resets from dimension `start_dim` down.
    /// Returns true if cascading succeeded (search continues), false if all dimensions exhausted.
    fn cascade_dimensions(
        ef_array: &mut [f64],
        ef_intervals: &mut [IntervalManager],
        constraint_indices: &[usize],
        ideal_max: &[f64],
        nadir_max: &[f64],
        rwv: &mut [f64],
    ) -> bool {
        for i in (1..constraint_indices.len()).rev() {
            if ef_array[i] > ideal_max[constraint_indices[i]] {
                // Reset this dimension
                ef_array[i] = nadir_max[constraint_indices[i]];
                rwv[i] = ideal_max[constraint_indices[i]];
                #[allow(clippy::cast_possible_truncation)]
                {
                    let lo = nadir_max[constraint_indices[i]] as i64;
                    let hi = ideal_max[constraint_indices[i]] as i64;
                    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
                    ef_intervals[i] = IntervalManager::new(lo, hi);
                }

                // Advance previous dimension
                let prev = i - 1;
                let new_eps = Self::force_advance_epsilon(
                    ef_array[prev],
                    ideal_max[constraint_indices[prev]],
                    &mut ef_intervals[prev],
                );
                ef_array[prev] = new_eps;
            } else {
                return true; // Cascading stopped, search continues
            }
        }

        // Check if dimension 0 is also exhausted
        ef_array[0] <= ideal_max[constraint_indices[0]]
    }

    #[allow(dead_code, reason = "Method kept for API completeness")]
    fn initialize_epsilons(&self, nadir: &[f64]) -> HashMap<usize, f64> {
        let mut epsilons = HashMap::new();
        for (k, &_nadir_val) in nadir.iter().enumerate() {
            if k != self.config.primary_objective {
                epsilons.insert(k, _nadir_val);
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
        timeout: Option<Duration>,
    ) -> Result<Option<Solution>> {
        let mut builder =
            EpsilonConstraintBuilder::new(problem, options, self.config.primary_objective);

        for (&k, &epsilon) in epsilons {
            let range = ranges.get(&k).copied().unwrap_or(1000.0);
            builder = builder.add_constraint_with_range(k, epsilon, range);
        }

        match builder.solve_with_slack(timeout)? {
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

// ═══════════════════════════════════════════════════════════════════════════
//  GPBA-B: Uniformity-focused representation
// ═══════════════════════════════════════════════════════════════════════════

/// GPBA-B: Uniformity-focused representation algorithm
///
/// Maximizes the minimum distance between consecutive points to ensure
/// good uniformity in the Pareto front representation.
pub struct GpbaB {
    config: GpbaConfig,
    acceptable_uniformity_level: f64,
    /// Timer for timeout tracking
    timer: Option<Timer>,
}

impl GpbaB {
    /// Create new GPBA-B instance with uniformity focus
    #[must_use]
    pub const fn new(config: GpbaConfig) -> Self {
        Self {
            acceptable_uniformity_level: 0.0,
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

    /// Initialize uniformity parameters
    fn initialize_uniformity_parameters(&mut self, ideal: &[f64], nadir: &[f64]) {
        let total_range: f64 = ideal
            .iter()
            .zip(nadir.iter())
            .map(|(i, n)| (i - n).abs())
            .sum();
        self.acceptable_uniformity_level = total_range * 0.01;
    }

    /// Adjust `epsilon_k` for GPBA-B (simple midpoint bisection)
    fn adjust_epsilon_k(current: f64, ideal: f64) -> f64 {
        f64::midpoint(current, ideal)
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
        const MAX_ITERATIONS: usize = 5000;

        log::info!("=== GPBA-B: Starting generate_representation ===");

        let (ideal_min, nadir_min) = if let Some((ideal, nadir)) = &self.config.manual_bounds {
            (ideal.clone(), nadir.clone())
        } else {
            BoundsCalculator::new(problem, options).calculate_bounds(self.timer.as_ref())?
        };

        let ideal_max: Vec<f64> = ideal_min.iter().map(|&x| -x).collect();
        let nadir_max: Vec<f64> = nadir_min.iter().map(|&x| -x).collect();

        self.initialize_uniformity_parameters(&ideal_max, &nadir_max);

        let constraint_indices: Vec<usize> = (0..problem.num_objectives())
            .filter(|&i| i != self.config.primary_objective)
            .collect();

        let mut ef_array: Vec<f64> = constraint_indices.iter().map(|&k| nadir_max[k]).collect();

        let ranges = Self::calculate_objective_ranges(&ideal_max, &nadir_max);
        let mut pareto_front = ParetoFront::new(vec![
            crate::model::ObjectiveDirection::Minimize;
            problem.num_objectives()
        ]);

        let mut iteration = 0;

        while iteration < MAX_ITERATIONS {
            if self.is_timeout_reached() {
                log::warn!("Timeout reached at iteration {iteration}");
                break;
            }

            let mut epsilons = HashMap::new();
            for (i, &k) in constraint_indices.iter().enumerate() {
                epsilons.insert(k, -ef_array[i]);
            }

            let mut builder =
                EpsilonConstraintBuilder::new(problem, options, self.config.primary_objective);
            for (&k, &epsilon) in &epsilons {
                let range = ranges.get(&k).copied().unwrap_or(1000.0);
                builder = builder.add_constraint_with_range(k, epsilon, range);
            }

            match builder.solve_with_slack(self.get_remaining_timeout())? {
                Some(solution_with_slack) => {
                    let mut solution = solution_with_slack.solution;
                    solution.objective_values = solution
                        .objective_values
                        .iter()
                        .map(|&x| x.round())
                        .collect();
                    pareto_front.add_solution_with_precision(
                        Solution::new(
                            solution.objective_values.clone(),
                            solution.decision_variables.clone(),
                        ),
                        0,
                    );
                }
                None => {
                    log::debug!("Infeasible at iteration {iteration}");
                }
            }

            // Simple advancement
            let last = constraint_indices.len() - 1;
            ef_array[last] =
                Self::adjust_epsilon_k(ef_array[last], ideal_max[constraint_indices[last]]);

            if ef_array[last] > ideal_max[constraint_indices[last]] {
                break;
            }

            iteration += 1;
        }

        log::info!(
            "=== GPBA-B: Completed with {} solutions ===",
            pareto_front.len()
        );
        Ok(pareto_front)
    }

    /// Calculate objective ranges
    fn calculate_objective_ranges(ideal: &[f64], nadir: &[f64]) -> HashMap<usize, f64> {
        let mut ranges = HashMap::new();
        for i in 0..ideal.len() {
            let range = (ideal[i] - nadir[i]).abs();
            ranges.insert(i, range.max(1e-6));
        }
        ranges
    }

    #[allow(dead_code)]
    fn initialize_epsilons(&self, nadir: &[f64]) -> HashMap<usize, f64> {
        let mut epsilons = HashMap::new();
        for (k, &nadir_val) in nadir.iter().enumerate() {
            if k != self.config.primary_objective {
                epsilons.insert(k, nadir_val);
            }
        }
        epsilons
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  GPBA-C: Cardinality-focused representation
// ═══════════════════════════════════════════════════════════════════════════

/// GPBA-C: Cardinality-focused representation algorithm
///
/// Balances coverage and uniformity to achieve a target number of
/// Pareto-optimal solutions.
pub struct GpbaC {
    config: GpbaConfig,
    grid_state: GridState,
    /// Timer for timeout tracking
    timer: Option<Timer>,
}

/// Internal grid state for GPBA-C
#[derive(Debug, Clone)]
struct GridState {
    #[allow(dead_code)]
    start_point: f64,
    current_position: f64,
    remaining_points: usize,
}

impl GpbaC {
    /// Create new GPBA-C instance
    #[must_use]
    pub const fn new(config: GpbaConfig) -> Self {
        Self {
            grid_state: GridState {
                start_point: 0.0,
                current_position: 0.0,
                remaining_points: 0,
            },
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

    /// Initialize grid state
    const fn initialize_grid_state(&mut self, nadir_val: f64, ideal_val: f64, num_points: usize) {
        self.grid_state = GridState {
            start_point: nadir_val,
            current_position: nadir_val,
            remaining_points: num_points,
        };
        let _ = ideal_val; // Used for range computation in actual grid stepping
    }

    /// Adjust `epsilon_k` for GPBA-C (uniform grid stepping)
    fn adjust_epsilon_k(
        &mut self,
        current_epsilon: f64,
        ideal_val: f64,
        nadir_val: f64,
        found_solution: bool,
    ) -> f64 {
        if self.grid_state.remaining_points == 0 {
            return ideal_val + 1.0; // Trigger termination
        }

        let range = (ideal_val - nadir_val).abs();
        if range < 1e-10 {
            return ideal_val + 1.0;
        }

        let step = range / self.grid_state.remaining_points as f64;
        let next = current_epsilon + step;

        if found_solution {
            self.grid_state.remaining_points = self.grid_state.remaining_points.saturating_sub(1);
        }

        self.grid_state.current_position = next;
        next
    }

    /// Generate Pareto front representation
    ///
    /// # Errors
    /// Returns an error if the optimization solver fails or problem validation fails
    pub fn generate_representation(
        &mut self,
        problem: &MultiObjectiveProblem,
        options: &Options,
    ) -> Result<ParetoFront> {
        const MAX_ITERATIONS: usize = 5000;
        let target_points = 50; // Default target cardinality

        log::info!("=== GPBA-C: Starting generate_representation ===");

        let (ideal_min, nadir_min) = if let Some((ideal, nadir)) = &self.config.manual_bounds {
            (ideal.clone(), nadir.clone())
        } else {
            BoundsCalculator::new(problem, options).calculate_bounds(self.timer.as_ref())?
        };

        let ideal_max: Vec<f64> = ideal_min.iter().map(|&x| -x).collect();
        let nadir_max: Vec<f64> = nadir_min.iter().map(|&x| -x).collect();

        let constraint_indices: Vec<usize> = (0..problem.num_objectives())
            .filter(|&i| i != self.config.primary_objective)
            .collect();

        let mut ef_array: Vec<f64> = constraint_indices.iter().map(|&k| nadir_max[k]).collect();

        // Initialize grid state for last dimension
        if let Some(&last_constraint) = constraint_indices.last() {
            self.initialize_grid_state(
                nadir_max[last_constraint],
                ideal_max[last_constraint],
                target_points,
            );
        }

        let ranges = Self::calculate_objective_ranges(&ideal_max, &nadir_max);
        let mut pareto_front = ParetoFront::new(vec![
            crate::model::ObjectiveDirection::Minimize;
            problem.num_objectives()
        ]);

        let mut iteration = 0;

        while iteration < MAX_ITERATIONS {
            if self.is_timeout_reached() {
                log::warn!("Timeout reached at iteration {iteration}");
                break;
            }

            let mut epsilons = HashMap::new();
            for (i, &k) in constraint_indices.iter().enumerate() {
                epsilons.insert(k, -ef_array[i]);
            }

            let mut builder =
                EpsilonConstraintBuilder::new(problem, options, self.config.primary_objective);
            for (&k, &epsilon) in &epsilons {
                let range = ranges.get(&k).copied().unwrap_or(1000.0);
                builder = builder.add_constraint_with_range(k, epsilon, range);
            }

            let found_solution = match builder.solve_with_slack(self.get_remaining_timeout())? {
                Some(solution_with_slack) => {
                    let mut solution = solution_with_slack.solution;
                    solution.objective_values = solution
                        .objective_values
                        .iter()
                        .map(|&x| x.round())
                        .collect();
                    pareto_front.add_solution_with_precision(
                        Solution::new(
                            solution.objective_values.clone(),
                            solution.decision_variables.clone(),
                        ),
                        0,
                    );
                    true
                }
                None => false,
            };

            // Advance
            let last = constraint_indices.len() - 1;
            let last_idx = constraint_indices[last];
            ef_array[last] = self.adjust_epsilon_k(
                ef_array[last],
                ideal_max[last_idx],
                nadir_max[last_idx],
                found_solution,
            );

            if ef_array[last] > ideal_max[last_idx] {
                break;
            }

            iteration += 1;
        }

        log::info!(
            "=== GPBA-C: Completed with {} solutions ===",
            pareto_front.len()
        );
        Ok(pareto_front)
    }

    /// Calculate objective ranges
    fn calculate_objective_ranges(ideal: &[f64], nadir: &[f64]) -> HashMap<usize, f64> {
        let mut ranges = HashMap::new();
        for i in 0..ideal.len() {
            let range = (ideal[i] - nadir[i]).abs();
            ranges.insert(i, range.max(1e-6));
        }
        ranges
    }

    #[allow(dead_code)]
    fn initialize_epsilons(&self, nadir: &[f64]) -> HashMap<usize, f64> {
        let mut epsilons = HashMap::new();
        for (k, &nadir_val) in nadir.iter().enumerate() {
            if k != self.config.primary_objective {
                epsilons.insert(k, nadir_val);
            }
        }
        epsilons
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Preset configurations
// ═══════════════════════════════════════════════════════════════════════════

/// Preset configurations for common use cases
pub mod presets {
    use super::GpbaConfig;

    /// Configuration optimized for maximum coverage of the Pareto front
    #[must_use]
    pub const fn high_coverage_config() -> GpbaConfig {
        GpbaConfig {
            primary_objective: 0,
            manual_bounds: None,
            target_solutions: None,
            per_solve_timeout: None,
        }
    }

    /// Configuration optimized for uniform distribution of solutions
    #[must_use]
    pub const fn uniform_distribution_config() -> GpbaConfig {
        GpbaConfig {
            primary_objective: 0,
            manual_bounds: None,
            target_solutions: None,
            per_solve_timeout: None,
        }
    }

    /// Configuration for achieving a target number of well-distributed solutions
    #[must_use]
    pub const fn balanced_cardinality_config() -> GpbaConfig {
        GpbaConfig {
            primary_objective: 0,
            manual_bounds: None,
            target_solutions: None,
            per_solve_timeout: None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpba_config_creation() {
        let config = GpbaConfig {
            primary_objective: 0,
            manual_bounds: None,
            target_solutions: None,
            per_solve_timeout: None,
        };
        assert_eq!(config.primary_objective, 0);
        assert!(config.manual_bounds.is_none());
        assert!(config.target_solutions.is_none());
        assert!(config.per_solve_timeout.is_none());
    }

    #[test]
    fn test_gpba_a_initialization() {
        let config = GpbaConfig {
            primary_objective: 0,
            manual_bounds: None,
            target_solutions: None,
            per_solve_timeout: None,
        };
        let gpba = GpbaA::new(config);
        assert!(gpba.previous_solution_information.is_empty());
    }

    #[test]
    fn test_grid_state_initialization() {
        let config = GpbaConfig {
            primary_objective: 0,
            manual_bounds: None,
            target_solutions: None,
            per_solve_timeout: None,
        };
        let mut gpba_c = GpbaC::new(config);
        gpba_c.initialize_grid_state(-100.0, -10.0, 50);
        assert_eq!(gpba_c.grid_state.start_point, -100.0);
        assert_eq!(gpba_c.grid_state.current_position, -100.0);
        assert_eq!(gpba_c.grid_state.remaining_points, 50);
    }

    #[test]
    fn test_adjust_epsilon_k_first_iteration() {
        // Test that first iteration jumps from nadir to ideal
        let mut interval = IntervalManager::new(-1000, -100);
        let result = GpbaA::adjust_epsilon_k(
            0,
            -1000.0,      // current epsilon at nadir
            Some(-500.0), // solution found at -500
            -100.0,       // ideal
            -1000.0,      // nadir
            &mut interval,
        );
        // First iteration should jump to ideal
        assert!(
            (result - (-100.0)).abs() < 1e-6,
            "Expected ideal=-100, got {result}"
        );
    }

    #[test]
    fn test_adjust_epsilon_k_midpoint() {
        // Test that subsequent iterations find midpoints
        let mut interval = IntervalManager::new(-1000, -100);

        // First iteration: nadir to ideal
        let _ = GpbaA::adjust_epsilon_k(0, -1000.0, Some(-500.0), -100.0, -1000.0, &mut interval);

        // Second iteration: should explore midpoint of remaining interval
        let result = GpbaA::adjust_epsilon_k(
            0,
            -100.0,       // now at ideal
            Some(-100.0), // solution at ideal
            -100.0,
            -1000.0,
            &mut interval,
        );

        // Should be in the remaining interval
        assert!(
            result < -100.0 || result > -100.0,
            "Result should differ from ideal after exploration"
        );
    }

    #[test]
    fn test_adjust_epsilon_k_exhausted() {
        // Test that exhausted interval returns beyond ideal
        let mut interval = IntervalManager::new(-100, -100); // Single point
        interval.remove_one_point(-100);

        let result =
            GpbaA::adjust_epsilon_k(0, -100.0, Some(-100.0), -100.0, -1000.0, &mut interval);

        assert!(result > -100.0, "Should return beyond ideal when exhausted");
    }

    #[test]
    fn test_force_advance_epsilon() {
        let mut interval = IntervalManager::new(-1000, -100);

        // Remove a point and check advancement
        let result = GpbaA::force_advance_epsilon(-500.0, -100.0, &mut interval);

        // Should return some value in the remaining intervals
        assert!(
            result >= -1000.0 && result <= -100.0,
            "Should return value within interval bounds, got {result}"
        );
    }

    #[test]
    fn test_force_advance_exhausted() {
        let mut interval = IntervalManager::new(-100, -100);

        let result = GpbaA::force_advance_epsilon(-100.0, -100.0, &mut interval);
        assert!(result > -100.0, "Should return beyond ideal when exhausted");
    }

    // ────────────────────────────────────────────────────────────────
    //  Relaxation search tests (GpbaA integration)
    // ────────────────────────────────────────────────────────────────

    fn make_gpba_a() -> GpbaA {
        let config = GpbaConfig {
            primary_objective: 0,
            manual_bounds: None,
            target_solutions: None,
            per_solve_timeout: None,
        };
        GpbaA::new(config)
    }

    #[test]
    fn test_relaxation_search_no_previous_solutions() {
        let gpba = make_gpba_a();
        let ef_array = vec![-500.0, -200.0];
        let constraint_indices = vec![1, 2];

        let (found, _solution) =
            gpba.search_previous_solutions_relaxation(&ef_array, &constraint_indices);
        assert!(!found, "Should not find anything with empty history");
    }

    #[test]
    fn test_relaxation_search_reuses_feasible_solution() {
        let mut gpba = make_gpba_a();

        // Constraint indices: objectives 1 and 2 are constrained
        let constraint_indices = vec![1, 2];

        // Save a previous solution:
        //   ef_array (MAX form): [-400, -150]  (less constrained / more relaxed)
        //   solution (MIN form): [100, 300, 120]  (obj0=100, obj1=300, obj2=120)
        //
        // In MAX form the solution values for constrained objectives are:
        //   obj1_max = -300, obj2_max = -120
        gpba.save_solution_information(vec![-400.0, -150.0], Some(vec![100.0, 300.0, 120.0]));

        // Current (tighter) epsilon: [-500, -200]
        // Previous ef [-400, -150] >= current [-500, -200]?  -400 >= -500 ✓, -150 >= -200 ✓
        // Does solution satisfy current constraints?
        //   sol_val_max for obj1 = -300, ef_val = -500  →  -300 >= -500  ✓
        //   sol_val_max for obj2 = -120, ef_val = -200  →  -120 >= -200  ✓
        let ef_array = vec![-500.0, -200.0];
        let (found, solution) =
            gpba.search_previous_solutions_relaxation(&ef_array, &constraint_indices);

        assert!(found, "Should find a reusable solution");
        let sol = solution.expect("Should return feasible solution");
        assert_eq!(sol, vec![100.0, 300.0, 120.0]);
    }

    #[test]
    fn test_relaxation_search_rejects_when_solution_violates() {
        let mut gpba = make_gpba_a();
        let constraint_indices = vec![1, 2];

        // Previous solution: ef=[-400, -150], solution=[100, 300, 120]
        // sol_max for obj1 = -300, sol_max for obj2 = -120
        gpba.save_solution_information(vec![-400.0, -150.0], Some(vec![100.0, 300.0, 120.0]));

        // Current epsilon: [-350, -200]
        // Previous ef [-400, -150] >= [-350, -200]?  -400 >= -350 ✗  →  NOT less constrained
        let ef_array = vec![-350.0, -200.0];
        let (found, _) = gpba.search_previous_solutions_relaxation(&ef_array, &constraint_indices);

        assert!(
            !found,
            "Previous ef is NOT less constrained, should not match"
        );
    }

    #[test]
    fn test_relaxation_search_propagates_infeasibility() {
        let mut gpba = make_gpba_a();
        let constraint_indices = vec![1];

        // Previous config was infeasible with a less constrained ef
        gpba.save_solution_information(vec![-200.0], None);

        // Current is tighter: [-300] <= [-200], so previous ef >= current ef
        let ef_array = vec![-300.0];
        let (found, solution) =
            gpba.search_previous_solutions_relaxation(&ef_array, &constraint_indices);

        assert!(found, "Should propagate infeasibility");
        assert!(
            solution.is_none(),
            "Infeasible propagation should return None"
        );
    }

    #[test]
    fn test_relaxation_search_does_not_propagate_to_looser() {
        let mut gpba = make_gpba_a();
        let constraint_indices = vec![1];

        // Previous config was infeasible with ef=[-300]
        gpba.save_solution_information(vec![-300.0], None);

        // Current is LOOSER: [-200], previous ef [-300] >= [-200]? -300 >= -200 ✗
        let ef_array = vec![-200.0];
        let (found, _) = gpba.search_previous_solutions_relaxation(&ef_array, &constraint_indices);

        assert!(
            !found,
            "Should NOT propagate infeasibility to a looser configuration"
        );
    }

    #[test]
    fn test_save_solution_information_accumulates() {
        let mut gpba = make_gpba_a();

        assert_eq!(gpba.previous_solution_information.len(), 0);

        gpba.save_solution_information(vec![-100.0], Some(vec![50.0, 80.0]));
        assert_eq!(gpba.previous_solution_information.len(), 1);

        gpba.save_solution_information(vec![-200.0], None);
        assert_eq!(gpba.previous_solution_information.len(), 2);

        gpba.save_solution_information(vec![-300.0], Some(vec![60.0, 90.0]));
        assert_eq!(gpba.previous_solution_information.len(), 3);
    }

    #[test]
    fn test_relaxation_search_picks_first_match() {
        let mut gpba = make_gpba_a();
        let constraint_indices = vec![1];

        // Two previous solutions, both with less-constrained ef
        // Solution A: ef=[-100], solution=[50, 80]  → obj1_max = -80
        gpba.save_solution_information(vec![-100.0], Some(vec![50.0, 80.0]));
        // Solution B: ef=[-50], solution=[40, 70]   → obj1_max = -70
        gpba.save_solution_information(vec![-50.0], Some(vec![40.0, 70.0]));

        // Current ef: [-150]
        // Both previous ef values are >= -150 ✓
        // Solution A: obj1_max=-80 >= -150 ✓  (first match wins)
        let ef_array = vec![-150.0];
        let (found, solution) =
            gpba.search_previous_solutions_relaxation(&ef_array, &constraint_indices);

        assert!(found);
        // Should return first matching solution (A)
        assert_eq!(solution.unwrap(), vec![50.0, 80.0]);
    }
}
