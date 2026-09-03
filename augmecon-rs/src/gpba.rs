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
    epsilon_constraint::{EpsilonConstraintBuilder, EpsilonSolveOutcome},
    error::Result,
    interval_manager::IntervalManager,
    model::MultiObjectiveProblem,
    options::Options,
    solution::{ParetoFront, Solution},
    timer::Timer,
};
use good_lp::constraint;
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

/// The session handed to [`refine_lexicographic`], or a unit placeholder when
/// no backend supports in-place editing.
#[cfg(feature = "gurobi")]
type RefineSession<'a, 'p> = Option<&'a mut crate::single_objective::ScalarisationSession<'p>>;
#[cfg(not(feature = "gurobi"))]
type RefineSession<'a, 'p> = ();

/// Borrow the refinement session, where the backend has one.
///
/// A function rather than a macro so the call sites read the same in both
/// feature configurations without reaching into their caller's locals.
#[cfg(feature = "gurobi")]
fn refine_session<'a, 'p>(
    held: &'a mut Option<crate::single_objective::ScalarisationSession<'p>>,
) -> RefineSession<'a, 'p> {
    held.as_mut()
}

#[cfg(not(feature = "gurobi"))]
const fn refine_session<'a, 'p>(held: &'a mut LexSessionSlot) -> RefineSession<'a, 'p> {
    *held
}

/// What a build without an editable backend holds in place of a session.
#[cfg(not(feature = "gurobi"))]
type LexSessionSlot = ();

/// Re-solve one emitted point lexicographically: pin the primary objective at
/// the value just found and minimise the other objective.
///
/// The epsilon-constraint solve only guarantees a *weakly* efficient point.
/// The augmentation term is supposed to upgrade that to efficient, but its
/// coefficient (`rho * 10^-(k+1) / range`) lands near 1e-10 here against a
/// primary objective of ~1e6 -- a 1e16 coefficient range that no
/// double-precision simplex resolves, and which `mip_gap = 0` does not rescue
/// (measured: the dominated extreme came back unchanged). Pinning and
/// re-minimising sidesteps the scaling entirely; it is the same two-stage
/// construction `BoundsCalculator::calculate_payoff_table` already uses for
/// the off-diagonal payoff entries.
///
/// Returns the refined objective values and decision variables, or `None` when
/// refinement does not apply or does not improve anything.
fn refine_lexicographic(
    problem: &MultiObjectiveProblem,
    options: &Options,
    objective_values: &[f64],
    primary: usize,
    timer: Option<&Timer>,
    session: RefineSession<'_, '_>,
) -> Option<(Vec<f64>, HashMap<String, f64>)> {
    if !options.lexicographic_refine || problem.num_objectives() != 2 {
        return None;
    }
    let secondary = 1 - primary;
    let deadline = timer.map(Timer::remaining);

    // Through the session when there is one: pinning is then a right-hand-side
    // edit on a model already built, which is what makes one extra solve per
    // emitted point affordable. Rebuilding for it was measured at six emitted
    // points falling to two, and is why this pass is off by default.
    #[cfg(feature = "gurobi")]
    let solved = match session {
        Some(session) => {
            session.solve_pinned(secondary, primary, objective_values[primary], deadline)
        }
        None => refine_by_rebuild(
            problem,
            options,
            objective_values,
            primary,
            secondary,
            deadline,
        ),
    };
    #[cfg(not(feature = "gurobi"))]
    let solved = {
        let () = session;
        refine_by_rebuild(
            problem,
            options,
            objective_values,
            primary,
            secondary,
            deadline,
        )
    };

    match solved {
        Ok(sol) if sol.feasible => {
            // Only accept a strict improvement on the secondary objective; an
            // equal value means the original point was already efficient.
            if sol.objective_values[secondary] < objective_values[secondary] {
                log::debug!(
                    "Lexicographic refinement: obj{secondary} {} -> {}",
                    objective_values[secondary],
                    sol.objective_values[secondary]
                );
                crate::verify::bump(&crate::verify::LEX_REFINED);
                Some((sol.objective_values, sol.decision_variables))
            } else {
                None
            }
        }
        Ok(_) => None,
        Err(e) => {
            log::warn!("Lexicographic refinement failed, keeping original point: {e}");
            None
        }
    }
}

impl GpbaA {
    /// Create new GPBA-A instance with coverage focus
    /// Uses Python-compatible dynamic interval exploration (gamma=1)
    #[must_use]
    pub const fn new(config: GpbaConfig) -> Self {
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
    /// `ef_array` is in **MAX form**, where the constraint is `z_k >= ef_k`, so a *higher*
    /// `ef_k` is a *tighter* constraint. A previous `ef_array` is therefore "less constrained"
    /// (more relaxed, larger feasible region) when all `prev_ef[i] <= current_ef[i]`.
    /// (The phased reference impl in `gpba_phases::relaxation_search` uses the opposite
    /// `>=` because it works in the un-negated MIN/upper-bound convention; negating ε flips
    /// the relation. Using `>=` here selected *tighter* previous problems and reused their
    /// solutions for looser current ones, collapsing the front to the 2 payoff extremes.)
    ///
    /// When such a less-constrained previous configuration exists:
    /// - If its solution is feasible **and** satisfies the current (tighter) constraints,
    ///   we can reuse it without solving a new MILP.
    /// - If it was infeasible, the current (tighter) configuration is also infeasible.
    fn search_previous_solutions_relaxation(
        &self,
        ef_array: &[f64],
        constraint_indices: &[usize],
    ) -> (bool, Option<Vec<f64>>) {
        for prev_info in &self.previous_solution_information {
            // Previous is less constrained (larger feasible region) when its MAX-form
            // epsilon lower bounds are all <= the current ones.
            let is_less_constrained = prev_info
                .ef_array
                .iter()
                .zip(ef_array.iter())
                .all(|(&prev, &curr)| prev <= curr);

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
        crate::verify::bump(&crate::verify::INTERVAL_REMOVALS);
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
    #[allow(
        clippy::cast_possible_truncation,
        reason = "elapsed microseconds for one solve stay far below u64::MAX"
    )]
    pub fn generate_representation(
        &mut self,
        problem: &MultiObjectiveProblem,
        options: &Options,
    ) -> Result<ParetoFront> {
        const MAX_ITERATIONS: usize = 10000; // Prevent infinite loops

        log::info!("=== GPBA-A: Starting generate_representation ===");
        log::info!("Number of objectives: {}", problem.num_objectives());
        log::info!("Primary objective index: {}", self.config.primary_objective);

        crate::verify::reset();
        let t_total = std::time::Instant::now();

        // Step 1: Compute or use provided bounds using shared calculator
        log::info!("=== STEP 1: Computing bounds (payoff table) ===");
        let (ideal_min, nadir_min, lex_extremes) =
            if let Some((ideal, nadir)) = &self.config.manual_bounds {
                (ideal.clone(), nadir.clone(), Vec::new())
            } else {
                BoundsCalculator::new(problem, options)
                    .calculate_bounds_with_solutions(self.timer.as_ref())?
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

        // Seed the front with the payoff table's lexicographic extremes.
        //
        // These solves have already happened while computing the bounds, so
        // this is free. It matters because the sweep's own extreme points come
        // from *unaugmented* single-objective solves and are therefore only
        // weakly efficient: on tokyo_bay_225 the min-cost point came back with
        // 3.6% more cloud than achievable at the same cost, and the zero-cloud
        // point with 0.8% more cost. The augmentation of (P5) cannot fix those
        // because it does not apply to them.
        for extreme in lex_extremes.into_iter().flatten() {
            let mut sol = extreme;
            let elapsed_us = self
                .timer
                .as_ref()
                .map_or(0, |t| t.elapsed().as_micros() as u64);
            sol.metadata
                .insert("timestamp_us".to_string(), elapsed_us.to_string());
            log::debug!(
                "Seeding front with payoff extreme {:?}",
                sol.objective_values
            );
            pareto_front.add_solution_with_precision(sol, 0);
        }

        // The lexicographic post-pass pins one objective and minimises the
        // other. Through a session that is a bound-row edit on a model already
        // built, so it costs a solve rather than a solve plus a rebuild.
        #[cfg(feature = "gurobi")]
        let mut lex_session = (options.lexicographic_refine
            && matches!(options.solver, crate::solver_enum::Solver::Gurobi))
        .then(|| crate::single_objective::ScalarisationSession::new(problem, options));
        #[cfg(not(feature = "gurobi"))]
        let mut lex_session: LexSessionSlot = ();

        let mut iteration = 0;
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
                    crate::verify::bump(&crate::verify::RELAXATION_REUSE); // SAUGMECON Lemma 1
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
                    crate::verify::bump(&crate::verify::INFEASIBLE_PROP); // SAUGMECON Lemma 2
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
                match self.solve_epsilon_constraint_problem_shared(
                    problem,
                    options,
                    &epsilons,
                    &ranges,
                    per_solve_timeout,
                )? {
                    EpsilonSolveOutcome::Solved(mut solution) => {
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
                    }
                    EpsilonSolveOutcome::Infeasible => {
                        log::info!("✗ No solution found (proven INFEASIBLE)");
                        None
                    }
                    EpsilonSolveOutcome::Inconclusive(reason) => {
                        // NOT a proven infeasibility (most commonly a solver
                        // timeout with no incumbent found yet) — the interval-
                        // pruning/cascade logic below assumes `None` means
                        // "this epsilon value is impossible", which would be
                        // wrong here and could silently discard a feasible
                        // region. Stop the sweep instead of risking a false
                        // "converged" result; whatever's already in
                        // pareto_front is still returned.
                        log::warn!(
                            "⚠ ε-constraint solve inconclusive at iteration {iteration} \
                             (NOT proven infeasible): {reason}. Stopping GPBA-A here — the \
                             remaining search space has not been ruled out, it just wasn't \
                             explored in the time available."
                        );
                        break;
                    }
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
                } else {
                    log::info!(
                        "⊗ DUPLICATE solution (already in Pareto front): {:?}",
                        solution.objective_values
                    );
                }

                // Upgrade weak efficiency to efficiency before the point is
                // recorded (see `refine_lexicographic`).
                let (refined_objs, refined_vars) = refine_lexicographic(
                    problem,
                    options,
                    &solution.objective_values,
                    self.config.primary_objective,
                    self.timer.as_ref(),
                    refine_session(&mut lex_session),
                )
                .unwrap_or_else(|| {
                    (
                        solution.objective_values.clone(),
                        solution.decision_variables.clone(),
                    )
                });
                let mut pareto_solution = Solution::new(refined_objs, refined_vars);
                // Record the wall-clock discovery time (µs since start) so callers can
                // reconstruct the GPBA-A front's timeline (used for hybrid pseudo-seeding).
                let elapsed_us = self
                    .timer
                    .as_ref()
                    .map_or(0, |t| t.elapsed().as_micros() as u64);
                pareto_solution
                    .metadata
                    .insert("timestamp_us".to_string(), elapsed_us.to_string());
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
        crate::verify::add_ns(&crate::verify::TOTAL_NS, t_total.elapsed());
        crate::verify::report();

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
        crate::verify::bump(&crate::verify::CASCADE_EXITS);
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
        for (k, &nadir_val) in nadir.iter().enumerate() {
            if k != self.config.primary_objective {
                epsilons.insert(k, nadir_val);
            }
        }
        epsilons
    }

    /// Solve epsilon-constraint problem with proper ranges.
    ///
    /// Returns the full [`EpsilonSolveOutcome`] rather than collapsing it to
    /// `Option<Solution>` — callers must distinguish a proven infeasibility
    /// (safe to prune) from an inconclusive result like a solver timeout
    /// (must NOT be treated as proof nothing exists there).
    fn solve_epsilon_constraint_problem_shared(
        &self,
        problem: &MultiObjectiveProblem,
        options: &Options,
        epsilons: &HashMap<usize, f64>,
        ranges: &HashMap<usize, f64>,
        timeout: Option<Duration>,
    ) -> Result<EpsilonSolveOutcome<Solution>> {
        let mut builder =
            EpsilonConstraintBuilder::new(problem, options, self.config.primary_objective);

        for (&k, &epsilon) in epsilons {
            let range = ranges.get(&k).copied().unwrap_or(1000.0);
            builder = builder.add_constraint_with_range(k, epsilon, range);
        }

        Ok(match builder.solve_with_slack(timeout)? {
            EpsilonSolveOutcome::Solved(solution_with_slack) => {
                EpsilonSolveOutcome::Solved(solution_with_slack.solution)
            }
            EpsilonSolveOutcome::Infeasible => EpsilonSolveOutcome::Infeasible,
            EpsilonSolveOutcome::Inconclusive(reason) => EpsilonSolveOutcome::Inconclusive(reason),
        })
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
    const fn adjust_epsilon_k(current: f64, ideal: f64) -> f64 {
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

        let (ideal_min, nadir_min, lex_extremes) =
            if let Some((ideal, nadir)) = &self.config.manual_bounds {
                (ideal.clone(), nadir.clone(), Vec::new())
            } else {
                BoundsCalculator::new(problem, options)
                    .calculate_bounds_with_solutions(self.timer.as_ref())?
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
        // Seed with the payoff table's lexicographic extremes (see the note in
        // GpbaA::generate_representation): these solves have already happened
        // while computing the bounds, and the sweep's own extremes come from
        // unaugmented single-objective solves that are only weakly efficient.
        for extreme in lex_extremes.into_iter().flatten() {
            let mut sol = extreme;
            let elapsed_us = self
                .timer
                .as_ref()
                .map_or(0, |t| t.elapsed().as_micros() as u64);
            sol.metadata
                .insert("timestamp_us".to_string(), elapsed_us.to_string());
            pareto_front.add_solution_with_precision(sol, 0);
        }

        // Every subproblem in the sweep below shares its variables, its
        // constraints and its slack structure; only the epsilon right-hand
        // sides move. Build the model once and edit it, on the native Gurobi
        // backend where in-place editing exists. Other backends keep
        // rebuilding, which is correct, only slower.
        #[cfg(feature = "gurobi")]
        let mut session = matches!(options.solver, crate::solver_enum::Solver::Gurobi).then(|| {
            crate::epsilon_constraint::EpsilonSession::new(
                problem,
                options,
                self.config.primary_objective,
                &constraint_indices,
                &ranges,
                options.epsilon_augmentation,
            )
        });

        // The lexicographic post-pass pins one objective and minimises the
        // other. Through a session that is a bound-row edit on a model already
        // built, so it costs a solve rather than a solve plus a rebuild.
        #[cfg(feature = "gurobi")]
        let mut lex_session = (options.lexicographic_refine
            && matches!(options.solver, crate::solver_enum::Solver::Gurobi))
        .then(|| crate::single_objective::ScalarisationSession::new(problem, options));
        #[cfg(not(feature = "gurobi"))]
        let mut lex_session: LexSessionSlot = ();

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

            let outcome = {
                #[cfg(feature = "gurobi")]
                {
                    match session.as_mut() {
                        Some(session) => {
                            session.solve(&epsilons, &ranges, options, self.get_remaining_timeout())
                        }
                        None => solve_epsilon_by_rebuild(
                            problem,
                            options,
                            self.config.primary_objective,
                            &epsilons,
                            &ranges,
                            self.get_remaining_timeout(),
                        )?,
                    }
                }
                #[cfg(not(feature = "gurobi"))]
                {
                    solve_epsilon_by_rebuild(
                        problem,
                        options,
                        self.config.primary_objective,
                        &epsilons,
                        &ranges,
                        self.get_remaining_timeout(),
                    )?
                }
            };

            match outcome {
                EpsilonSolveOutcome::Solved(mut solution_with_slack) => {
                    let pooled = std::mem::take(&mut solution_with_slack.pool);
                    let mut solution = solution_with_slack.solution;
                    solution.objective_values = solution
                        .objective_values
                        .iter()
                        .map(|&x| x.round())
                        .collect();
                    // Pooled candidates from the same solve, dominance-filtered
                    // by the front like any other point. They cost no extra
                    // solver time -- Gurobi found them on the way to this
                    // solution and would otherwise discard them.
                    for cand in pooled {
                        pareto_front.add_solution_with_precision(cand, 0);
                    }

                    let (r_objs, r_vars) = refine_lexicographic(
                        problem,
                        options,
                        &solution.objective_values,
                        self.config.primary_objective,
                        self.timer.as_ref(),
                        refine_session(&mut lex_session),
                    )
                    .unwrap_or_else(|| {
                        (
                            solution.objective_values.clone(),
                            solution.decision_variables.clone(),
                        )
                    });
                    pareto_front.add_solution_with_precision(Solution::new(r_objs, r_vars), 0);
                }
                EpsilonSolveOutcome::Infeasible => {
                    log::debug!("Infeasible at iteration {iteration}");
                }
                EpsilonSolveOutcome::Inconclusive(reason) => {
                    // GPBA-B's advancement is unconditional (doesn't prune
                    // based on infeasibility), so a timeout here just means
                    // this grid point is skipped — safe, unlike GPBA-A's
                    // interval-based cascade.
                    log::warn!(
                        "Inconclusive (not proven infeasible) at iteration {iteration}: {reason}"
                    );
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
    #[allow(
        clippy::cast_precision_loss,
        reason = "remaining grid-point count is small; f64 conversion for the step size is exact in range"
    )]
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

        let (ideal_min, nadir_min, lex_extremes) =
            if let Some((ideal, nadir)) = &self.config.manual_bounds {
                (ideal.clone(), nadir.clone(), Vec::new())
            } else {
                BoundsCalculator::new(problem, options)
                    .calculate_bounds_with_solutions(self.timer.as_ref())?
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
        // Seed with the payoff table's lexicographic extremes (see the note in
        // GpbaA::generate_representation): these solves have already happened
        // while computing the bounds, and the sweep's own extremes come from
        // unaugmented single-objective solves that are only weakly efficient.
        for extreme in lex_extremes.into_iter().flatten() {
            let mut sol = extreme;
            let elapsed_us = self
                .timer
                .as_ref()
                .map_or(0, |t| t.elapsed().as_micros() as u64);
            sol.metadata
                .insert("timestamp_us".to_string(), elapsed_us.to_string());
            pareto_front.add_solution_with_precision(sol, 0);
        }

        // The lexicographic post-pass pins one objective and minimises the
        // other. Through a session that is a bound-row edit on a model already
        // built, so it costs a solve rather than a solve plus a rebuild.
        #[cfg(feature = "gurobi")]
        let mut lex_session = (options.lexicographic_refine
            && matches!(options.solver, crate::solver_enum::Solver::Gurobi))
        .then(|| crate::single_objective::ScalarisationSession::new(problem, options));
        #[cfg(not(feature = "gurobi"))]
        let mut lex_session: LexSessionSlot = ();

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
                EpsilonSolveOutcome::Solved(mut solution_with_slack) => {
                    let pooled = std::mem::take(&mut solution_with_slack.pool);
                    let mut solution = solution_with_slack.solution;
                    solution.objective_values = solution
                        .objective_values
                        .iter()
                        .map(|&x| x.round())
                        .collect();
                    // Pooled candidates from the same solve, dominance-filtered
                    // by the front like any other point. They cost no extra
                    // solver time -- Gurobi found them on the way to this
                    // solution and would otherwise discard them.
                    for cand in pooled {
                        pareto_front.add_solution_with_precision(cand, 0);
                    }

                    let (r_objs, r_vars) = refine_lexicographic(
                        problem,
                        options,
                        &solution.objective_values,
                        self.config.primary_objective,
                        self.timer.as_ref(),
                        refine_session(&mut lex_session),
                    )
                    .unwrap_or_else(|| {
                        (
                            solution.objective_values.clone(),
                            solution.decision_variables.clone(),
                        )
                    });
                    pareto_front.add_solution_with_precision(Solution::new(r_objs, r_vars), 0);
                    true
                }
                EpsilonSolveOutcome::Infeasible => false,
                EpsilonSolveOutcome::Inconclusive(reason) => {
                    // `found_solution=false` feeds into adjust_epsilon_k's search
                    // direction below, same false-pruning risk as GPBA-A — stop
                    // rather than let a timeout masquerade as infeasibility.
                    log::warn!(
                        "⚠ ε-constraint solve inconclusive at iteration {iteration} \
                         (NOT proven infeasible): {reason}. Stopping GPBA-C here."
                    );
                    break;
                }
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
    #![allow(
        clippy::float_cmp,
        clippy::nonminimal_bool,
        clippy::manual_range_contains,
        clippy::double_comparisons,
        reason = "test assertions favour explicit exact comparisons and bounds for readability"
    )]
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

        // MAX-form constraint is z_k >= ef_k, so a *lower* ef is *less* constrained (larger
        // feasible region). Save a previous solution from a LESS-constrained (looser) config:
        //   ef_array (MAX form): [-500, -200]  (lower ef → looser)
        //   solution (MIN form): [100, 300, 120]  → obj1_max = -300, obj2_max = -120
        gpba.save_solution_information(vec![-500.0, -200.0], Some(vec![100.0, 300.0, 120.0]));

        // Current (tighter) epsilon: [-400, -150]  (higher ef → tighter)
        // Previous ef [-500, -200] <= current [-400, -150]?  -500 <= -400 ✓, -200 <= -150 ✓
        // Does the looser solution satisfy the current (tighter) constraint z_k >= ef?
        //   obj1_max = -300 >= -400 ✓, obj2_max = -120 >= -150 ✓  → still optimal, reuse.
        let ef_array = vec![-400.0, -150.0];
        let (found, solution) =
            gpba.search_previous_solutions_relaxation(&ef_array, &constraint_indices);

        assert!(found, "Should reuse the looser config's solution");
        let sol = solution.expect("Should return feasible solution");
        assert_eq!(sol, vec![100.0, 300.0, 120.0]);
    }

    #[test]
    fn test_relaxation_search_rejects_when_solution_violates() {
        let mut gpba = make_gpba_a();
        let constraint_indices = vec![1, 2];

        // Previous solution from a looser config: ef=[-500, -200], solution=[100, 300, 120]
        // sol_max for obj1 = -300, sol_max for obj2 = -120
        gpba.save_solution_information(vec![-500.0, -200.0], Some(vec![100.0, 300.0, 120.0]));

        // Current epsilon: [-400, -100].  Previous ef [-500, -200] <= [-400, -100] ✓ (looser),
        // but does the solution satisfy the tighter constraint z_k >= ef?
        //   obj2_max = -120 >= -100 ✗  →  violates, must re-solve.
        let ef_array = vec![-400.0, -100.0];
        let (found, _) = gpba.search_previous_solutions_relaxation(&ef_array, &constraint_indices);

        assert!(
            !found,
            "Looser solution violates the tighter constraint, should not match"
        );
    }

    #[test]
    fn test_relaxation_search_propagates_infeasibility() {
        let mut gpba = make_gpba_a();
        let constraint_indices = vec![1];

        // A LESS-constrained (looser, lower ef) config was infeasible.
        gpba.save_solution_information(vec![-300.0], None);

        // Current is TIGHTER (higher ef): [-200].  prev [-300] <= curr [-200] ✓ (prev looser).
        // If the looser problem is infeasible, the tighter (subset) one is too.
        let ef_array = vec![-200.0];
        let (found, solution) =
            gpba.search_previous_solutions_relaxation(&ef_array, &constraint_indices);

        assert!(found, "Looser-infeasible should propagate to tighter");
        assert!(
            solution.is_none(),
            "Infeasible propagation should return None"
        );
    }

    #[test]
    fn test_relaxation_search_does_not_propagate_to_looser() {
        let mut gpba = make_gpba_a();
        let constraint_indices = vec![1];

        // A MORE-constrained (tighter, higher ef) config was infeasible.
        gpba.save_solution_information(vec![-200.0], None);

        // Current is LOOSER (lower ef): [-300].  prev [-200] <= curr [-300]? -200 <= -300 ✗.
        // A tighter problem being infeasible says nothing about the looser one → no match.
        let ef_array = vec![-300.0];
        let (found, _) = gpba.search_previous_solutions_relaxation(&ef_array, &constraint_indices);

        assert!(
            !found,
            "Tighter-infeasible must NOT propagate to a looser configuration"
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

        // Two previous solutions, both from LESS-constrained (looser, lower ef) configs.
        // Solution A: ef=[-200], solution=[50, 80]  → obj1_max = -80
        gpba.save_solution_information(vec![-200.0], Some(vec![50.0, 80.0]));
        // Solution B: ef=[-150], solution=[40, 70]  → obj1_max = -70
        gpba.save_solution_information(vec![-150.0], Some(vec![40.0, 70.0]));

        // Current ef: [-100]  (tighter).  Both prev ef <= -100 ✓ (looser).
        // Solution A: obj1_max = -80 >= -100 ✓  (first match wins)
        let ef_array = vec![-100.0];
        let (found, solution) =
            gpba.search_previous_solutions_relaxation(&ef_array, &constraint_indices);

        assert!(found);
        // Should return first matching solution (A)
        assert_eq!(solution.unwrap(), vec![50.0, 80.0]);
    }
}

/// Solve one epsilon subproblem by building a fresh model.
///
/// The path every backend without in-place editing takes, and the reference the
/// reused path is checked against.
fn solve_epsilon_by_rebuild(
    problem: &MultiObjectiveProblem,
    options: &Options,
    primary_objective: usize,
    epsilons: &HashMap<usize, f64>,
    ranges: &HashMap<usize, f64>,
    timeout: Option<Duration>,
) -> Result<EpsilonSolveOutcome<crate::epsilon_constraint::SolutionWithSlack>> {
    let mut builder = EpsilonConstraintBuilder::new(problem, options, primary_objective);
    for (&k, &epsilon) in epsilons {
        let range = ranges.get(&k).copied().unwrap_or(1000.0);
        builder = builder.add_constraint_with_range(k, epsilon, range);
    }
    builder.solve_with_slack(timeout)
}

/// The pinned re-solve, building a fresh model.
///
/// What every backend without in-place editing uses, and the reference the
/// session path is checked against.
fn refine_by_rebuild(
    problem: &MultiObjectiveProblem,
    options: &Options,
    objective_values: &[f64],
    primary: usize,
    secondary: usize,
    deadline: Option<Duration>,
) -> Result<crate::solution::Solution> {
    let (primary_expr, _) = &problem.objectives[primary];
    let pinned = constraint!(primary_expr.clone() == objective_values[primary]);
    crate::single_objective::SingleObjectiveSolver::new(problem, options)
        .solve_objective_with_constraints(secondary, &[pinned], deadline)
}
