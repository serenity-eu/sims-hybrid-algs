//! # Solver Configuration Module
//!
//! This module provides comprehensive configuration options for the AUGMECON solver,
//! allowing fine-tuned control over algorithm behavior, performance characteristics,
//! and output formatting.
//!
//! ## Overview
//!
//! The [`Options`] struct serves as the central configuration hub, offering:
//! - **Algorithm Control**: Enable/disable AUGMECON variants and optimizations
//! - **Performance Tuning**: Grid size, precision, and parallelization settings
//! - **Output Configuration**: Formatting, logging, and export options
//! - **Solver Backend**: Custom solver parameters and timeouts
//!
//! ## Quick Configuration
//!
//! ```rust
//! use augmecon::Options;
//!
//! // Basic configuration
//! let options = Options::new()
//!     .with_name("my_problem")
//!     .with_grid_points(50);
//!
//! // Performance-optimized configuration
//! let fast_options = Options::new()
//!     .with_grid_points(30)
//!     .with_early_exit(true)
//!     .with_bypass_coefficient(true)
//!     .with_flag_array(true);
//!
//! // High-precision configuration
//! let precise_options = Options::new()
//!     .with_grid_points(100)
//!     .with_penalty_weight(1e-6)
//!     .with_round_decimals(10);
//! ```
//!
//! ## Algorithm Variants
//!
//! ### Classic AUGMECON
//! ```rust
//! # use augmecon::Options;
//! let classic = Options::new()
//!     .with_grid_points(50)
//!     .with_early_exit(false)
//!     .with_bypass_coefficient(false)
//!     .with_flag_array(false);
//! ```
//!
//! ### AUGMECON2 (Bypass Coefficient)
//! ```rust
//! # use augmecon::Options;
//! let augmecon2 = Options::new()
//!     .with_grid_points(50)
//!     .with_bypass_coefficient(true)
//!     .with_early_exit(true);
//! ```
//!
//! ### AUGMECON-R (Flag Array)
//! ```rust
//! # use augmecon::Options;
//! let augmecon_r = Options::new()
//!     .with_grid_points(50)
//!     .with_flag_array(true)
//!     .with_bypass_coefficient(true);
//! ```
//!
//! ## Performance Tuning
//!
//! ### Grid Points Configuration
//! The number of grid points determines the resolution of the Pareto front exploration:
//!
//! ```rust
//! # use augmecon::Options;
//! // Conservative (good coverage, slower)
//! let detailed = Options::new().with_grid_points(100);
//!
//! // Balanced (good trade-off)
//! let balanced = Options::new().with_grid_points(50);
//!
//! // Fast (quick exploration)
//! let quick = Options::new().with_grid_points(20);
//! ```
//!
//! **Note**: Complexity grows as `grid_points^(objectives-1)`, so use larger values carefully for 3+ objectives.
//!
//! ### Precision Control
//! ```rust
//! # use augmecon::Options;
//! // High precision (financial applications)
//! let financial = Options::new()
//!     .with_penalty_weight(1e-6)
//!     .with_round_decimals(8);
//!
//! // Standard precision (most applications)
//! let standard = Options::new()
//!     .with_penalty_weight(1e-3)
//!     .with_round_decimals(6);
//!
//! // Fast approximation (preliminary studies)
//! let approx = Options::new()
//!     .with_penalty_weight(1e-1)
//!     .with_round_decimals(3);
//! ```
//!
//! ## Adaptive Configuration
//!
//! ```rust
//! # use augmecon::Options;
//! fn adaptive_options(num_objectives: usize, problem_size: usize) -> Options {
//!     let grid_points = match (num_objectives, problem_size) {
//!         (2, 0..=50) => 100,
//!         (2, 51..=200) => 50,
//!         (3, 0..=50) => 30,
//!         (3, 51..=200) => 15,
//!         _ => 20,
//!     };
//!
//!     Options::new()
//!         .with_grid_points(grid_points)
//!         .with_early_exit(problem_size > 100)
//!         .with_bypass_coefficient(true)
//!         .with_flag_array(grid_points > 30)
//! }
//! ```
//!
//! ## Validation
//!
//! Options are automatically validated against the problem structure:
//!
//! ```rust
//! # use augmecon::Options;
//! let options = Options::new().with_grid_points(50);
//!
//! // Validation occurs during solver creation
//! // Invalid configurations will return an error
//! ```

use crate::error::{AugmeconError, Result};
use crate::solver_enum::Solver;
use std::collections::HashMap;

/// Configuration options for the AUGMECON solver
#[derive(Debug, Clone)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Configuration struct requires multiple boolean flags for different features - each boolean represents a distinct optimization or output option"
)]
pub struct Options {
    /// Name of the problem (used for logging and output)
    pub name: String,
    /// Number of grid points for the ε-constraint method
    pub grid_points: Option<usize>,
    /// Nadir points for each objective (except the first one)
    pub nadir_points: Option<Vec<f64>>,
    /// Penalty weight (epsilon value)
    pub penalty_weight: f64,
    /// Number of decimal places to round results to
    pub round_decimals: usize,
    /// Nadir ratio for automatic nadir point calculation
    pub nadir_ratio: f64,
    /// Enable early exit optimization
    pub early_exit: bool,
    /// Enable bypass coefficient optimization
    pub bypass_coefficient: bool,
    /// Enable flag array optimization
    pub flag_array: bool,
    /// Enable parallel processing for grid point evaluation
    pub parallel_execution: bool,
    /// Number of CPU cores to use for parallel processing
    pub cpu_count: usize,
    /// Enable work redistribution in parallel processing
    pub redivide_work: bool,
    /// Enable shared flag array in parallel processing
    pub shared_flag: bool,
    /// Output results to Excel format
    pub output_excel: bool,
    /// Enable process logging
    pub process_logging: bool,
    /// Timeout for solver processes (in seconds)
    pub process_timeout: Option<u64>,
    /// Linear programming solver to use
    pub solver: Solver,
    /// Solver-specific configuration parameters (only used if solver supports parameters)
    pub solver_parameters: HashMap<String, String>,
    /// Relative MIP optimality gap for backends that expose it (currently the
    /// native Gurobi backend).
    ///
    /// The default of 0 is deliberate. The AUGMECON augmentation term that
    /// makes an epsilon-constraint solution *efficient* rather than merely
    /// *weakly* efficient is scaled by `rho * 10^-(k+1) / range`; with a
    /// second objective whose range is ~1e9 that coefficient lands around
    /// 1e-10, so the reward for choosing the strictly better of two
    /// primary-optimal solutions is a tiny fraction of the solver's default
    /// relative gap (1e-4, i.e. hundreds of absolute units on a 1e6-scale
    /// objective). The solver then stops at whichever incumbent it reached
    /// first and returns a dominated point. Solving to a zero gap restores the
    /// tie-break the formulation intends.
    pub mip_gap: Option<f64>,
    /// Re-solve each emitted solution lexicographically before adding it to the
    /// front: fix the primary objective at the value just found and minimise
    /// the remaining objective.
    ///
    /// Without this the epsilon-constraint solve only guarantees *weak*
    /// efficiency. Efficiency is supposed to come from the augmentation term,
    /// but on SIMS that term is numerically inert -- the coefficient works out
    /// around 1e-10 against a primary objective of ~1e6, a range no
    /// double-precision simplex can resolve, and tightening `mip_gap` does not
    /// help. The Python reference implementation reaches the same conclusion
    /// from the other direction and disables augmentation outright for this
    /// model (`is_numerically_possible_augment_objective` -> false), noting
    /// that its single-objective solutions "are not necessarily on the Pareto
    /// front".
    ///
    /// This costs one extra MILP solve per emitted point, which in a
    /// wall-clock-bounded phase means fewer points found in the same budget.
    /// Only implemented for the bi-objective case; ignored otherwise.
    pub lexicographic_refine: bool,
    /// Ask the solver for an efficient point directly, by declaring the
    /// epsilon subproblem's objectives as a two-level hierarchy.
    ///
    /// The alternative to both of this crate's other answers to weak
    /// efficiency, and cheaper than either: the augmentation term costs
    /// nothing but only works while it stays above the solver's sensitivity,
    /// and [`Self::lexicographic_refine`] always works but costs a second
    /// solve per emitted point. A priority costs neither -- Gurobi runs both
    /// passes inside one `optimize()` call -- and, being structural rather
    /// than numerical, cannot be too small to take effect.
    ///
    /// Native Gurobi only; other backends ignore it and keep the augmentation.
    pub hierarchical_efficiency: bool,
    /// `rho` in Problem (P5) of the GPBA-A paper. Theorem 3 requires it to be
    /// "sufficiently small", usually between 1e-3 and 1e-6, for the optimum of
    /// (P5) to be an *efficient* (not merely weakly efficient) solution.
    ///
    /// The default follows Section 4.2 of that paper, which reports rho = 1e-2
    /// for the GPBA algorithms (1e-3 was used only for AUGMECON2) and warns
    /// that when the slack is divided by a wide objective range the product
    /// "may become smaller than the implementation software's sensitivity",
    /// whose consequence is "the failure to compute some non-dominated
    /// criterion vectors" -- exactly the defect measured here.
    ///
    /// The augmentation term saturates at `rho * 10^(k-1)`, so with the paper's
    /// weights and rho = 1e-3 it tops out near 1e-2: small enough not to
    /// reorder an integer primary objective (whose smallest gap is 1), large
    /// enough to break ties in it -- provided the solver's optimality tolerance
    /// can see it (see `mip_gap` and the Gurobi `OptimalityTol` set alongside).
    pub epsilon_augmentation: f64,
    /// How many solutions to retain from each MIP solve's solution pool.
    ///
    /// A scalarised solve visits many feasible integer solutions before it
    /// proves optimality and normally reports only the best. Those discarded
    /// solutions improve the scalarised objective while ranging freely over
    /// the others, so a good share of them are non-dominated in the original
    /// objective space -- free Pareto candidates from a solve already paid
    /// for. `PoolSearchMode = 1` merely *keeps* what the search already found,
    /// as opposed to mode 2 which searches for more and does cost time.
    ///
    /// 0 disables the pool.
    pub solution_pool_size: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            name: "Undefined".to_string(),
            grid_points: None,
            nadir_points: None,
            penalty_weight: 1e-3,
            round_decimals: 9,
            nadir_ratio: 1.0,
            early_exit: true,
            bypass_coefficient: true,
            flag_array: true,
            parallel_execution: true,
            cpu_count: num_cpus::get(),
            redivide_work: true,
            shared_flag: true,
            output_excel: true,
            process_logging: false,
            process_timeout: None,
            solver: Solver::default(),
            solver_parameters: HashMap::new(),
            mip_gap: Some(0.0),
            lexicographic_refine: false,
            hierarchical_efficiency: false,
            epsilon_augmentation: 1e-2,
            solution_pool_size: 64,
        }
    }
}

impl Options {
    /// Create new options with default values
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the problem name
    #[must_use]
    pub fn with_name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = name.into();
        self
    }

    /// Set the solver to use
    #[must_use]
    pub const fn with_solver(mut self, solver: Solver) -> Self {
        self.solver = solver;
        self
    }

    /// Set the number of grid points
    #[must_use]
    pub const fn with_grid_points(mut self, grid_points: usize) -> Self {
        self.grid_points = Some(grid_points);
        self
    }

    /// Set nadir points
    #[must_use]
    pub fn with_nadir_points(mut self, nadir_points: Vec<f64>) -> Self {
        self.nadir_points = Some(nadir_points);
        self
    }

    /// Set penalty weight
    #[must_use]
    pub const fn with_penalty_weight(mut self, penalty_weight: f64) -> Self {
        self.penalty_weight = penalty_weight;
        self
    }

    /// Set number of decimal places for rounding
    #[must_use]
    pub const fn with_round_decimals(mut self, round_decimals: usize) -> Self {
        self.round_decimals = round_decimals;
        self
    }

    /// Enable or disable early exit
    #[must_use]
    pub const fn with_early_exit(mut self, early_exit: bool) -> Self {
        self.early_exit = early_exit;
        self
    }

    /// Set number of CPU cores to use
    #[must_use]
    pub const fn with_cpu_count(mut self, cpu_count: usize) -> Self {
        self.cpu_count = cpu_count;
        self
    }

    /// Set bypass coefficient optimization
    #[must_use]
    pub const fn with_bypass_coefficient(mut self, bypass_coefficient: bool) -> Self {
        self.bypass_coefficient = bypass_coefficient;
        self
    }

    /// Set flag array optimization
    #[must_use]
    pub const fn with_flag_array(mut self, flag_array: bool) -> Self {
        self.flag_array = flag_array;
        self
    }

    /// Set parallel execution for grid point processing
    #[must_use]
    pub const fn with_parallel_execution(mut self, parallel_execution: bool) -> Self {
        self.parallel_execution = parallel_execution;
        self
    }

    /// Set timeout for solver processes
    #[must_use]
    pub const fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.process_timeout = Some(timeout.as_secs());
        self
    }

    /// Add solver-specific option
    #[must_use]
    pub fn with_solver_option<S: Into<String>, V: Into<String>>(
        mut self,
        key: S,
        value: V,
    ) -> Self {
        self.solver_parameters.insert(key.into(), value.into());
        self
    }

    /// Validate the options
    ///
    /// # Errors
    /// Returns an error if the options are inconsistent or invalid
    pub const fn validate(&self, num_objectives: usize) -> Result<()> {
        if self.grid_points.is_none() {
            return Err(AugmeconError::NoGridPoints);
        }

        if let Some(ref nadir_points) = self.nadir_points {
            let expected = num_objectives - 1;
            if nadir_points.len() != expected {
                return Err(AugmeconError::InvalidNadirPoints {
                    expected,
                    actual: nadir_points.len(),
                });
            }
        }

        Ok(())
    }
}

// Helper function to get number of CPUs (fallback implementation)
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
    }
}

/// Apply solver parameters that the native Gurobi backend can accept.
///
/// `good_lp`'s `GurobiProblem` exposes the underlying `grb::Model` through
/// `as_inner_mut`, so parameters *are* reachable even though
/// `Solver::supports_parameters` reports false for the generic string-keyed
/// path. Only settings that materially affect correctness go through here;
/// see `Options::mip_gap` for why the gap is one of them.
#[cfg(feature = "gurobi")]
pub(crate) fn apply_gurobi_options(
    model: &mut good_lp::solvers::gurobi::GurobiProblem,
    options: &Options,
) {
    if let Some(gap) = options.mip_gap {
        if let Err(e) = model.as_inner_mut().set_param(grb::param::MIPGap, gap) {
            log::warn!("Could not set Gurobi MIPGap to {gap}: {e}");
        }
    }
    // The augmentation term of (P5) is ~1e-2 against a primary objective of
    // ~1e6, i.e. 1e-8 relative -- below the default OptimalityTol of 1e-6, so
    // the solver would not act on it. 1e-9 is Gurobi's minimum and puts the
    // term comfortably above the resolution floor.
    if let Err(e) = model
        .as_inner_mut()
        .set_param(grb::param::OptimalityTol, 1e-9)
    {
        log::warn!("Could not tighten Gurobi OptimalityTol: {e}");
    }
    if options.solution_pool_size > 0 {
        let m = model.as_inner_mut();
        // Mode 1 keeps the solutions the search finds anyway; mode 2 would go
        // looking for more, which costs time we do not have.
        if let Err(e) = m.set_param(grb::param::PoolSearchMode, 1) {
            log::warn!("Could not set Gurobi PoolSearchMode: {e}");
        }
        if let Err(e) = m.set_param(
            grb::param::PoolSolutions,
            i32::try_from(options.solution_pool_size).unwrap_or(i32::MAX),
        ) {
            log::warn!("Could not set Gurobi PoolSolutions: {e}");
        }
    }
}
