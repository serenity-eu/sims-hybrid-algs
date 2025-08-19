//! # AUGMECON Solver Implementation
//!
//! This module contains the core implementation of the AUGMECON (Augmented ε-constraint) method
//! for solving multi-objective optimization problems. It provides the main solver struct and
//! algorithm implementations for all AUGMECON variants.
//!
//! ## Overview
//!
//! The solver implements several algorithmic enhancements:
//! - **Classic AUGMECON**: Basic augmented ε-constraint method
//! - **AUGMECON2**: Enhanced with bypass coefficient optimization  
//! - **AUGMECON-R**: Advanced with flag array optimization
//! - **Early Exit**: Smart termination strategies for efficiency
//!
//! ## Core Algorithm
//!
//! The AUGMECON method works in three main phases:
//!
//! 1. **Payoff Table Calculation**: Solve single-objective problems to determine objective ranges
//! 2. **Grid Generation**: Create systematic ε-constraint problems across the objective space
//! 3. **Solution Filtering**: Identify truly Pareto-optimal solutions from all computed solutions
//!
//! ## Main Interface
//!
//! The [`Augmecon`] struct provides the primary interface for solving multi-objective problems:
//!
//! ```rust
//! use augmecon::{Augmecon, MultiObjectiveProblem, Options};
//!
//! # fn create_problem() -> MultiObjectiveProblem { MultiObjectiveProblem::new() }
//! let problem = create_problem();
//! let options = Options::new().with_grid_points(50);
//!
//! let mut solver = Augmecon::new(problem, options)?;
//! solver.solve()?;
//!
//! let pareto_solutions = solver.get_pareto_solutions();
//! let payoff_table = solver.get_payoff_table();
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Algorithm Variants
//!
//! ### Classic AUGMECON
//!
//! The basic augmented ε-constraint method:
//! - Systematic exploration of the constraint space
//! - Augmentation parameter to ensure proper solutions
//! - Guaranteed Pareto optimality
//!
//! ### AUGMECON2 (Bypass Coefficient)
//!
//! Enhanced version with performance improvements:
//! - Bypass coefficient to skip dominated regions
//! - Reduced computational overhead
//! - Particularly effective for problems with many dominated solutions
//!
//! ### AUGMECON-R (Flag Array)
//!
//! Advanced version with memory optimization:
//! - Flag array to track explored regions  
//! - Elimination of redundant computations
//! - Significant speedup for large grid sizes
//!
//! ## Performance Characteristics
//!
//! ### Complexity Analysis
//! - **Grid Points**: O(p^(k-1)) where p = grid points, k = objectives
//! - **Memory Usage**: O(p^(k-1) + n) where n = variables
//! - **Solve Time**: Depends on individual LP solve times and problem structure
//!
//! ### Scalability Guidelines
//! - **2 Objectives**: Can handle 100-500 grid points efficiently
//! - **3 Objectives**: Recommend 10-50 grid points  
//! - **4+ Objectives**: Requires careful grid point selection (5-20 points)
//!
//! ## Error Handling
//!
//! The solver provides comprehensive error handling for:
//! - Invalid problem configurations
//! - Infeasible optimization problems
//! - Solver backend failures
//! - Memory and timeout constraints
//!
//! ```rust
//! # use augmecon::*;
//! # let problem = MultiObjectiveProblem::new();
//! # let options = Options::new().with_grid_points(50);
//! match Augmecon::new(problem, options) {
//!     Ok(mut solver) => {
//!         match solver.solve() {
//!             Ok(()) => println!("Optimization successful"),
//!             Err(e) => eprintln!("Solve failed: {}", e),
//!         }
//!     },
//!     Err(e) => eprintln!("Solver creation failed: {}", e),
//! }
//! ```

use log::{debug, info, warn};
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::{
    bounds::BoundsCalculator,
    epsilon_constraint::{EpsilonConstraintBuilder, SolutionWithSlack},
    error::{AugmeconError, Result},
    flag::FlagArray,
    grid::GridGenerator,
    model::MultiObjectiveProblem,
    options::Options,
    solution::{ParetoFront, Solution},
};

/// Main AUGMECON solver struct
pub struct Augmecon {
    /// The multi-objective problem to solve
    problem: MultiObjectiveProblem,
    /// Solver options
    options: Options,
    /// The computed Pareto front
    pareto_front: ParetoFront,
    /// Progress tracking
    models_solved: usize,
    /// Total number of models to solve
    total_models: usize,
    /// Flag array for tracking grid point status and skip logic
    flag_array: Option<FlagArray>,
    /// Start time for timeout tracking
    start_time: Option<Instant>,
    /// Total timeout duration
    timeout_duration: Option<Duration>,
}

impl Augmecon {
    /// Create a new AUGMECON solver
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The problem fails validation (invalid constraints, objectives, etc.)
    /// - The options are invalid for the given problem
    ///
    /// # Panics
    ///
    /// Panics if `options.grid_points` is `None`
    pub fn try_new(problem: MultiObjectiveProblem, options: Options) -> Result<Self> {
        // Validate inputs
        problem.validate()?;
        options.validate(problem.num_objectives())?;

        let grid_points = options.grid_points.ok_or_else(|| {
            AugmeconError::InvalidOptions("Grid points must be specified in options".to_string())
        })?;
        let num_objectives = problem.num_objectives();
        let total_models = grid_points
            .pow(u32::try_from(num_objectives - 1).expect("Too many objectives for grid points"))
            + num_objectives;

        let objective_directions: Vec<_> = problem
            .objectives
            .iter()
            .map(|(_, direction)| *direction)
            .collect();

        Ok(Self {
            problem,
            flag_array: if options.flag_array {
                Some(FlagArray::new())
            } else {
                None
            },
            timeout_duration: options.process_timeout.map(Duration::from_secs),
            options,
            pareto_front: ParetoFront::new(objective_directions),
            models_solved: 0,
            total_models,
            start_time: None,
        })
    }

    /// Initialize timeout tracking
    fn start_timeout_tracking(&mut self) {
        if self.timeout_duration.is_some() {
            self.start_time = Some(Instant::now());
            info!(
                "Timeout tracking started with duration: {:?}",
                self.timeout_duration
            );
        }
    }

    /// Check if the timeout has been reached
    fn is_timeout_reached(&self) -> bool {
        if let (Some(start), Some(duration)) = (self.start_time, self.timeout_duration) {
            let elapsed = start.elapsed();
            let is_expired = elapsed >= duration;
            if is_expired {
                warn!("Timeout reached: elapsed = {elapsed:?}, limit = {duration:?}");
            }
            is_expired
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

    /// Solve the multi-objective optimization problem
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Single objective optimization fails
    /// - ε-constraint method fails during solving
    /// - Solver encounters an unrecoverable error
    pub fn solve(&mut self) -> Result<Vec<Solution>> {
        println!(
            "DEBUG: Starting AUGMECON solver for problem: {}",
            self.options.name
        );
        println!(
            "DEBUG: Number of objectives: {}",
            self.problem.num_objectives()
        );
        println!("DEBUG: Grid points: {:?}", self.options.grid_points);
        println!("DEBUG: Timeout: {:?}", self.timeout_duration);

        info!(
            "Starting AUGMECON solver for problem: {}",
            self.options.name
        );
        info!("Number of objectives: {}", self.problem.num_objectives());
        info!("Grid points: {:?}", self.options.grid_points);
        info!("Timeout: {:?}", self.timeout_duration);

        // Initialize timeout tracking
        self.start_timeout_tracking();

        // Step 1: Calculate payoff table
        println!("DEBUG: Step 1 - Calculating payoff table");
        if self.is_timeout_reached() {
            warn!("Timeout reached during payoff table calculation setup");
            return Ok(self.pareto_front.solutions.clone());
        }

        let payoff_table = self.calculate_payoff_table()?;
        println!("DEBUG: Payoff table calculated: {payoff_table:?}");
        self.pareto_front.set_payoff_table(payoff_table);

        // Check timeout after payoff table calculation
        if self.is_timeout_reached() {
            warn!("Timeout reached after payoff table calculation");
            return Ok(self.pareto_front.solutions.clone());
        }

        // Step 2: Generate ε-constraint problems and solve them
        println!("DEBUG: Step 2 - Solving epsilon constraint problems");
        self.solve_epsilon_constraint_problems()?;

        // Step 3: Filter for unique Pareto-optimal solutions
        println!("DEBUG: Step 3 - Filtering Pareto solutions");
        self.filter_pareto_solutions();

        println!(
            "DEBUG: AUGMECON solver completed. Found {} Pareto-optimal solutions",
            self.pareto_front.len()
        );
        info!(
            "AUGMECON solver completed. Found {} Pareto-optimal solutions",
            self.pareto_front.len()
        );

        Ok(self.pareto_front.solutions.clone())
    }

    /// Get the Pareto front (all Pareto-optimal solutions)
    #[must_use]
    pub const fn get_pareto_front(&self) -> &ParetoFront {
        &self.pareto_front
    }

    /// Get just the Pareto-optimal solutions
    #[must_use]
    pub fn get_pareto_solutions(&self) -> &[Solution] {
        &self.pareto_front.solutions
    }

    /// Get the payoff table
    #[must_use]
    pub fn get_payoff_table(&self) -> &[Vec<f64>] {
        &self.pareto_front.payoff_table
    }

    /// Calculate the payoff table by solving single-objective problems
    fn calculate_payoff_table(&mut self) -> Result<Vec<Vec<f64>>> {
        info!("Calculating payoff table using shared bounds calculator...");

        let bounds_calculator = BoundsCalculator::new(&self.problem, &self.options);
        let payoff_table =
            bounds_calculator.calculate_payoff_table(self.get_remaining_timeout())?;

        // Update models solved count
        self.models_solved += payoff_table.len();

        info!("Payoff table calculation completed");
        Ok(payoff_table)
    }

    /// Generate and solve ε-constraint problems using proper AUGMECON nested loop structure
    fn solve_epsilon_constraint_problems(&mut self) -> Result<()> {
        info!("Generating and solving ε-constraint problems...");

        let grid_points = self.options.grid_points.unwrap();
        let num_objectives = self.problem.num_objectives();

        if num_objectives < 2 {
            return Err(AugmeconError::InvalidObjectiveCount(num_objectives));
        }

        // Initialize counters
        let mut processed_count = 0;
        let mut skipped_count = 0;

        match num_objectives {
            2 => {
                self.solve_two_objective_problems(
                    grid_points,
                    &mut processed_count,
                    &mut skipped_count,
                )?;
            }
            _ => {
                self.solve_multi_objective_problems(
                    grid_points,
                    num_objectives,
                    &mut processed_count,
                    &mut skipped_count,
                );
            }
        }

        info!(
            "Completed solving ε-constraint problems: {processed_count} processed, {skipped_count} skipped"
        );

        log::info!(
            "Processed {} grid points, found {} unique solutions",
            processed_count,
            self.pareto_front.len()
        );
        log::debug!(
            "Final pareto front contains {} solutions",
            self.pareto_front.solutions.len()
        );

        Ok(())
    }

    /// Solve two-objective problems using an optimized single loop structure
    fn solve_two_objective_problems(
        &mut self,
        grid_points: usize,
        processed_count: &mut usize,
        skipped_count: &mut usize,
    ) -> Result<()> {
        let mut jump_remaining: usize = 0;
        let mut index = 0;

        while index < grid_points {
            // Check timeout before processing each grid point
            if self.is_timeout_reached() {
                warn!("Timeout reached during two-objective solving at grid point {index}");
                break;
            }

            // Handle jump logic for the single dimension
            if jump_remaining > 0 {
                let skip_count = jump_remaining.min(grid_points - index);
                debug!("Jumping {skip_count} points due to pending jump");
                *skipped_count += skip_count;
                index += skip_count;
                jump_remaining = jump_remaining.saturating_sub(skip_count);
                continue;
            }

            // For a 2-objective problem, the grid point is 1D.
            let grid_point = vec![index];

            // Check flag array. For 2-obj, it's 1D.
            if let Some(jump) = self.check_flag_array(&grid_point, index, grid_points) {
                if jump > 0 {
                    jump_remaining = jump.saturating_sub(1);
                    *skipped_count += 1;
                    index += 1; // Progress loop
                    continue;
                }
            }

            // Log progress
            Self::log_progress(*processed_count, grid_points, &grid_point);

            // Process the grid point
            match self.process_grid_point(&grid_point, grid_points) {
                Ok(Some(new_jump)) => {
                    if new_jump > 0 {
                        jump_remaining = new_jump;
                    }
                }
                Ok(None) => {
                    // No jump, continue normally
                }
                Err(e) => {
                    warn!("Error processing grid point {grid_point:?}: {e}");
                    // Optionally, decide whether to stop or continue
                    return Err(e);
                }
            }

            *processed_count += 1;
            self.models_solved += 1;
            index += 1;
        }

        info!("Completed solving for 2-objective problem");
        Ok(())
    }

    /// Solve multi-objective problems for 3+ objectives using grid-based approach
    fn solve_multi_objective_problems(
        &mut self,
        grid_points: usize,
        num_objectives: usize,
        processed_count: &mut usize,
        _skipped_count: &mut usize,
    ) {
        info!("Processing {num_objectives}-objective problem with {grid_points} grid points");

        let all_grid_indices = GridGenerator::generate_uniform_grid(num_objectives, grid_points);

        if self.options.parallel_execution {
            self.solve_grid_points_parallel(&all_grid_indices, processed_count);
        } else {
            self.solve_grid_points_sequential(&all_grid_indices, processed_count);
        }
    }

    /// Solve grid points in parallel using Rayon
    fn solve_grid_points_parallel(
        &mut self,
        all_grid_indices: &[Vec<usize>],
        processed_count: &mut usize,
    ) {
        info!(
            "Using parallel execution with {} threads",
            rayon::current_num_threads()
        );

        // Extract immutable data needed for parallel computation
        let problem = &self.problem;
        let options = &self.options;
        let payoff_table = self.get_payoff_table().to_vec();

        // Process grid points in parallel
        let solutions: Vec<_> = all_grid_indices
            .par_iter()
            .filter_map(|grid_point| {
                match Self::solve_single_epsilon_constraint_with_slack(
                    problem,
                    options,
                    &payoff_table,
                    grid_point,
                    None, // TODO: In parallel execution, we can't access remaining time easily
                ) {
                    Ok(solution_with_slack) => {
                        if solution_with_slack.solution.feasible {
                            Some(solution_with_slack.solution)
                        } else {
                            None
                        }
                    }
                    Err(e) => {
                        warn!("Failed to solve ε-constraint problem at {grid_point:?}: {e}");
                        None
                    }
                }
            })
            .collect();

        // Add all solutions to Pareto front (without filtering yet)
        for solution in solutions {
            self.pareto_front.add_solution(solution);
        }

        *processed_count += all_grid_indices.len();
        self.models_solved += all_grid_indices.len();

        info!(
            "Processed {} grid points in parallel",
            all_grid_indices.len()
        );
    }

    /// Static method to solve a single epsilon constraint problem with slack
    /// This can be called in parallel since it doesn't need mutable access to self
    fn solve_single_epsilon_constraint_with_slack(
        problem: &MultiObjectiveProblem,
        options: &Options,
        payoff_table: &[Vec<f64>],
        grid_point: &[usize],
        timeout: Option<Duration>,
    ) -> Result<SolutionWithSlack> {
        debug!("Solving ε-constraint problem with slack for grid point {grid_point:?}");

        let num_objectives = problem.num_objectives();

        if payoff_table.is_empty() {
            return Err(AugmeconError::OptimizationError(
                "Cannot solve ε-constraint problem without payoff table".to_string(),
            ));
        }

        // Debug: Print payoff table for verification
        debug!("Payoff table: {payoff_table:?}");

        // Calculate ε values for objectives 2..n based on grid_point and payoff table
        let mut epsilon_values = HashMap::new();
        let mut objective_ranges = HashMap::new();
        #[expect(
            clippy::cast_precision_loss,
            reason = "Converting grid_points usize to f64 for epsilon-constraint grid size calculation - precision loss acceptable for algorithm"
        )]
        let grid_size = options
            .grid_points
            .expect("Grid points must be set in options") as f64;

        debug!("Grid size: {grid_size}, Number of objectives: {num_objectives}");

        for obj_idx in 1..num_objectives {
            // Find the range for this objective from the payoff table
            // According to the AUGMECON algorithm:
            // - Use global minimum and maximum across ALL payoff table entries for this objective

            let mut min_value = f64::INFINITY;
            let mut max_value = f64::NEG_INFINITY;

            // Find min and max across all rows of the payoff table for this objective
            for row in payoff_table {
                if obj_idx < row.len() {
                    min_value = min_value.min(row[obj_idx]);
                    max_value = max_value.max(row[obj_idx]);
                }
            }

            // Fallback if no valid values found
            if min_value == f64::INFINITY || max_value == f64::NEG_INFINITY {
                min_value = 0.0;
                max_value = 0.0;
            }

            // Store the range for this objective
            let range = (max_value - min_value).abs();
            objective_ranges.insert(obj_idx, range);

            debug!("Objective {obj_idx}: min={min_value}, max={max_value}, range={range} (global range)");

            // Grid point position for this objective (obj_idx - 1 because we skip the first objective)
            #[expect(
                clippy::cast_precision_loss,
                reason = "Converting grid point usize to f64 for epsilon value calculation - precision loss acceptable for algorithm"
            )]
            let grid_position = grid_point[obj_idx - 1] as f64;

            // Calculate ε value using uniform grid spacing
            // Formula: min + (max - min) * (grid_position / (grid_size - 1))
            let epsilon = if grid_size > 1.0 {
                let ratio = grid_position / (grid_size - 1.0);
                ratio.mul_add(max_value - min_value, min_value)
            } else {
                min_value
            };

            epsilon_values.insert(obj_idx, epsilon);
            debug!("Objective {obj_idx}: ε = {epsilon} (grid position {grid_position})");
        }

        debug!("Epsilon values for constraints: {epsilon_values:?}");

        // Build and solve epsilon-constraint problem with slack values
        let mut builder = EpsilonConstraintBuilder::new(problem, options, 0); // Primary objective is always 0

        for (&obj_idx, &epsilon) in &epsilon_values {
            let range = objective_ranges.get(&obj_idx).copied().unwrap_or(1000.0);
            builder = builder.add_constraint_with_range(obj_idx, epsilon, range);
        }

        // Solve with slack values
        builder.solve_with_slack(timeout)?.map_or_else(
            || {
                // Return infeasible solution with empty slack values
                let infeasible_solution = Solution::infeasible(num_objectives);
                Ok(SolutionWithSlack::new(infeasible_solution, HashMap::new()))
            },
            Ok,
        )
    }

    /// Solve grid points sequentially (original implementation)
    fn solve_grid_points_sequential(
        &mut self,
        all_grid_indices: &[Vec<usize>],
        processed_count: &mut usize,
    ) {
        for grid_point in all_grid_indices {
            // Check timeout before processing each grid point
            if self.is_timeout_reached() {
                warn!("Timeout reached during sequential solving at grid point {grid_point:?}");
                break;
            }

            match self.solve_epsilon_constraint_problem_with_slack(grid_point) {
                Ok(solution_with_slack) => {
                    if solution_with_slack.solution.feasible {
                        // Add solution without filtering yet
                        self.pareto_front.add_solution(solution_with_slack.solution);
                    }
                }
                Err(e) => {
                    warn!("Failed to solve ε-constraint problem at {grid_point:?}: {e}");
                }
            }

            *processed_count += 1;
            self.models_solved += 1;

            // Log progress periodically
            if (*processed_count).is_multiple_of(50) || *processed_count < 10 {
                info!(
                    "Processed {processed_count} of {} grid points",
                    all_grid_indices.len()
                );
            }
        }
    }

    /// Check flag array for skip instructions
    fn check_flag_array(
        &self,
        grid_point: &[usize],
        inner_index: usize,
        grid_points: usize,
    ) -> Option<usize> {
        let flag_value = self
            .flag_array
            .as_ref()
            .map_or(0, |flag_array| flag_array.get(grid_point));

        if flag_value != 0 {
            let jump = FlagArray::calculate_jump(inner_index, flag_value, grid_points);
            if jump > 0 {
                debug!("Skipping grid point {grid_point:?} due to flag value {flag_value}");
                return Some(jump);
            }
        }
        None
    }

    /// Log progress periodically
    fn log_progress(processed_count: usize, total_points: usize, grid_point: &[usize]) {
        if processed_count.is_multiple_of(50) || processed_count < 10 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "Converting problem counts to f64 for progress percentage calculation"
            )]
            let progress = ((processed_count as f64) / (total_points as f64)) * 100.0;
            info!(
                "Progress: {progress:.1}% ({processed_count} problems completed) - Grid point: {grid_point:?}"
            );
        }
    }

    /// Process a single grid point and return jump value if applicable
    fn process_grid_point(
        &mut self,
        grid_point: &[usize],
        grid_points: usize,
    ) -> Result<Option<usize>> {
        debug!("Processing grid point {grid_point:?}");

        match self.solve_epsilon_constraint_problem_with_slack(grid_point) {
            Ok(solution_with_slack) => {
                if solution_with_slack.solution.feasible {
                    debug!(
                        "Found feasible solution at grid point {grid_point:?}: {:?}",
                        solution_with_slack.solution.objective_values
                    );

                    self.pareto_front
                        .add_solution(solution_with_slack.solution.clone());

                    // Apply bypass coefficient logic
                    if self.options.bypass_coefficient {
                        return Ok(self.apply_bypass_logic(
                            &solution_with_slack,
                            grid_point,
                            grid_points,
                        ));
                    }
                } else {
                    // Handle infeasible solution
                    self.handle_infeasible_solution(
                        grid_point,
                        self.problem.num_objectives(),
                        grid_points,
                    );

                    // Set jump for early exit
                    if self.options.early_exit {
                        let remaining = grid_points
                            .saturating_sub(grid_point.first().copied().unwrap_or(0) + 1);
                        debug!("Early exit: jumping {remaining} remaining points");
                        return Ok(Some(remaining));
                    }
                }
            }
            Err(e) => {
                warn!("Failed to solve ε-constraint problem at {grid_point:?}: {e}");
                return Err(e);
            }
        }

        Ok(None)
    }

    /// Apply bypass logic and return jump value
    fn apply_bypass_logic(
        &mut self,
        solution_with_slack: &SolutionWithSlack,
        grid_point: &[usize],
        grid_points: usize,
    ) -> Option<usize> {
        if let Some(bypass_values) = self.calculate_bypass_values(solution_with_slack, grid_points)
        {
            debug!("Bypass values calculated: {bypass_values:?}");

            // Set flag for bypass range
            if let Some(ref mut flag_array) = self.flag_array {
                // For a problem with N objectives, the secondary objectives are indexed from 0 to N-2.
                let num_secondary_objectives = self.problem.num_objectives() - 1;
                let secondary_objectives: Vec<usize> = (0..num_secondary_objectives).collect();
                flag_array.set_bypass_range(grid_point, &bypass_values, &secondary_objectives);
            }

            // Return jump based on first bypass value
            if !bypass_values.is_empty() && bypass_values[0] > 0 {
                let jump = bypass_values[0].try_into().unwrap_or(0);
                debug!("Setting jump to {jump} from bypass values");
                return Some(jump);
            }
        }
        None
    }

    /// Solve a single ε-constraint problem and return slack values
    ///
    /// This version is used when bypass coefficient optimization is enabled
    fn solve_epsilon_constraint_problem_with_slack(
        &self,
        grid_point: &[usize],
    ) -> Result<SolutionWithSlack> {
        debug!("Solving ε-constraint problem with slack for grid point {grid_point:?}");

        let num_objectives = self.problem.num_objectives();
        let payoff_table = self.get_payoff_table();

        if payoff_table.is_empty() {
            return Err(AugmeconError::OptimizationError(
                "Cannot solve ε-constraint problem without payoff table".to_string(),
            ));
        }

        // Debug: Print payoff table for verification
        debug!("Payoff table: {payoff_table:?}");

        // Calculate ε values for objectives 2..n based on grid_point and payoff table
        let mut epsilon_values = HashMap::new();
        let mut objective_ranges = HashMap::new();
        #[expect(
            clippy::cast_precision_loss,
            reason = "Converting grid_points usize to f64 for epsilon-constraint grid size calculation - precision loss acceptable for algorithm"
        )]
        let grid_size = self
            .options
            .grid_points
            .expect("Grid points must be set in options") as f64;

        debug!("Grid size: {grid_size}, Number of objectives: {num_objectives}");

        for obj_idx in 1..num_objectives {
            // Find the range for this objective from the payoff table
            // According to the AUGMECON algorithm:
            // - Use global minimum and maximum across ALL payoff table entries for this objective

            let mut min_value = f64::INFINITY;
            let mut max_value = f64::NEG_INFINITY;

            // Find min and max across all rows of the payoff table for this objective
            for row in payoff_table {
                if obj_idx < row.len() {
                    min_value = min_value.min(row[obj_idx]);
                    max_value = max_value.max(row[obj_idx]);
                }
            }

            // Fallback if no valid values found
            if min_value == f64::INFINITY || max_value == f64::NEG_INFINITY {
                min_value = 0.0;
                max_value = 0.0;
            }

            // Store the range for this objective
            let range = (max_value - min_value).abs();
            objective_ranges.insert(obj_idx, range);

            debug!("Objective {obj_idx}: min={min_value}, max={max_value}, range={range} (global range)");

            // Grid point position for this objective (obj_idx - 1 because we skip the first objective)
            #[expect(
                clippy::cast_precision_loss,
                reason = "Converting grid_position usize to f64 for epsilon value interpolation - precision loss acceptable for constraint calculation"
            )]
            let grid_position = (if obj_idx - 1 < grid_point.len() {
                grid_point[obj_idx - 1]
            } else {
                0
            }) as f64;

            debug!("Objective {obj_idx}: grid_position={grid_position}");

            // Calculate ε value using linear interpolation
            let trade_off_factor = if grid_size > 1.0 {
                grid_position / (grid_size - 1.0)
            } else {
                0.0
            };

            debug!("Objective {obj_idx}: trade_off_factor={trade_off_factor}");

            // Calculate ε value using linear interpolation (like Python's np.linspace)
            // epsilon = min_value + trade_off_factor * (max_value - min_value)
            let epsilon = trade_off_factor.mul_add(max_value - min_value, min_value);
            debug!("Objective {obj_idx}: epsilon={epsilon} (interpolated from {min_value} to {max_value})");
            println!("DEBUG: Objective {obj_idx}: epsilon={epsilon} (interpolated from {min_value} to {max_value})");
            epsilon_values.insert(obj_idx, epsilon);
        }

        debug!("Epsilon values for constraints: {epsilon_values:?}");

        // Build and solve epsilon-constraint problem with slack values
        let mut builder = EpsilonConstraintBuilder::new(&self.problem, &self.options, 0); // Primary objective is always 0

        for (&obj_idx, &epsilon) in &epsilon_values {
            let range = objective_ranges.get(&obj_idx).copied().unwrap_or(1000.0);
            builder = builder.add_constraint_with_range(obj_idx, epsilon, range);
        }

        // Solve with slack values
        builder
            .solve_with_slack(self.get_remaining_timeout())?
            .map_or_else(
                || {
                    // Return infeasible solution with empty slack values
                    let infeasible_solution = Solution::infeasible(num_objectives);
                    Ok(SolutionWithSlack::new(infeasible_solution, HashMap::new()))
                },
                Ok,
            )
    }

    /// Filter solutions to keep only Pareto-optimal ones
    fn filter_pareto_solutions(&mut self) {
        info!("Filtering for Pareto-optimal solutions...");

        let initial_count = self.pareto_front.solutions.len();

        // Only keep feasible solutions first
        self.pareto_front.solutions.retain(|sol| sol.feasible);

        if self.pareto_front.solutions.is_empty() {
            warn!("No feasible solutions found");
            return;
        }

        // Use the efficient filtering method from ParetoFront
        self.pareto_front.filter_dominated_solutions();

        info!(
            "Filtered {initial_count} solutions to {} Pareto-optimal solutions",
            self.pareto_front.solutions.len()
        );
    }

    /// Get solving progress as a percentage
    #[must_use]
    pub fn get_progress(&self) -> f64 {
        if self.total_models == 0 {
            0.0
        } else {
            #[expect(
                clippy::cast_precision_loss,
                reason = "Converting model counts to f64 for progress percentage calculation - precision loss acceptable for progress reporting"
            )]
            let models_solved_f64 = self.models_solved as f64;
            #[expect(
                clippy::cast_precision_loss,
                reason = "Converting model counts to f64 for progress percentage calculation - precision loss acceptable for progress reporting"
            )]
            let total_models_f64 = self.total_models as f64;
            (models_solved_f64 / total_models_f64) * 100.0
        }
    }

    /// Get detailed progress information
    #[must_use]
    pub fn get_progress_info(&self) -> (usize, usize, f64) {
        let progress = self.get_progress();
        (self.models_solved, self.total_models, progress)
    }

    /// Get the number of solutions found so far
    #[must_use]
    pub const fn get_solutions_found(&self) -> usize {
        self.pareto_front.len()
    }

    /// Calculate bypass values based on slack values from the solution
    fn calculate_bypass_values(
        &self,
        solution_with_slack: &SolutionWithSlack,
        grid_points: usize,
    ) -> Option<Vec<i32>> {
        if !self.options.bypass_coefficient {
            return None;
        }

        // Get objective ranges from payoff table
        let payoff_table = &self.pareto_front.payoff_table;
        if payoff_table.is_empty() {
            return None;
        }

        let mut bypass_values = Vec::new();

        // Calculate bypass values for each secondary objective (skip first one)
        for (&obj_idx, &slack_value) in &solution_with_slack.slack_values {
            if obj_idx > 0 && obj_idx < payoff_table.len() && obj_idx < payoff_table[0].len() {
                // Calculate objective range
                let obj_min = payoff_table
                    .iter()
                    .map(|row| row[obj_idx])
                    .fold(f64::INFINITY, f64::min);
                let obj_max = payoff_table
                    .iter()
                    .map(|row| row[obj_idx])
                    .fold(f64::NEG_INFINITY, f64::max);
                let obj_range = obj_max - obj_min;

                if obj_range > 0.0 {
                    // Calculate step size and bypass value
                    #[allow(clippy::cast_precision_loss)]
                    let step = obj_range / (grid_points as f64 - 1.0);
                    let bypass_value = if step > 0.0 {
                        #[allow(clippy::cast_possible_truncation)]
                        ((slack_value.round() / step).round() as i32)
                    } else {
                        0
                    };
                    bypass_values.push(bypass_value.max(0));
                } else {
                    bypass_values.push(0);
                }
            }
        }

        if bypass_values.iter().any(|&v| v > 0) {
            Some(bypass_values)
        } else {
            None
        }
    }

    /// Handle infeasible solutions and early exit logic
    fn handle_infeasible_solution(
        &mut self,
        grid_point: &[usize],
        num_objectives: usize,
        grid_points: usize,
    ) {
        debug!("Solution infeasible for grid point {grid_point:?}");

        // Set flag for early exit if flag array is enabled and early exit is enabled
        if let Some(flag_array) = &mut self.flag_array {
            if self.options.early_exit {
                let objective_indices: Vec<usize> = (0..num_objectives - 1).collect();
                flag_array.set_early_exit_range(grid_point, 1, &objective_indices, grid_points);
            }
        }
    }
}
