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
