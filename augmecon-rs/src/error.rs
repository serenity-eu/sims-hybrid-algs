//! # Error Handling Module
//!
//! This module provides comprehensive error handling for the AUGMECON library,
//! including specific error types for different failure modes and helpful error messages.
//!
//! ## Overview
//!
//! The error system is designed around the [`AugmeconError`] enum, which covers:
//! - **Problem Validation**: Invalid problem configurations and constraints
//! - **Solver Issues**: Optimization failures and backend problems  
//! - **Configuration Errors**: Invalid options and parameter settings
//! - **I/O Failures**: File reading/writing and external data issues
//!
//! ## Error Categories
//!
//! ### Problem Definition Errors
//! ```rust
//! use augmecon::{AugmeconError, MultiObjectiveProblem};
//!
//! let problem = MultiObjectiveProblem::new();
//! // Problem with only one objective will fail validation
//! match problem.validate() {
//!     Err(AugmeconError::InvalidObjectiveCount(count)) => {
//!         println!("Need at least 2 objectives, found: {}", count);
//!     },
//!     Ok(()) => println!("Problem is valid"),
//!     Err(e) => println!("Other validation error: {}", e),
//! }
//! ```
//!
//! ### Configuration Errors
//! ```rust
//! use augmecon::{AugmeconError, Options};
//!
//! // Missing required configuration
//! let options = Options::new(); // No grid points set
//!
//! // This will fail during solver creation
//! match options.validate(2) {
//!     Err(AugmeconError::NoGridPoints) => {
//!         println!("Grid points must be specified");
//!     },
//!     Err(AugmeconError::InvalidOptions(msg)) => {
//!         println!("Invalid options: {}", msg);
//!     },
//!     _ => {},
//! }
//! ```
//!
//! ### Solver Runtime Errors
//! ```rust
//! use augmecon::AugmeconError;
//!
//! // Handle optimization failures gracefully
//! fn handle_solver_error(error: AugmeconError) {
//!     match error {
//!         AugmeconError::OptimizationError(msg) => {
//!             eprintln!("Solver failed: {}", msg);
//!             // Try with different settings or problem formulation
//!         },
//!         AugmeconError::ModelError(msg) => {
//!             eprintln!("Model issue: {}", msg);
//!             // Check problem constraints and variable bounds
//!         },
//!         _ => eprintln!("Unexpected error: {}", error),
//!     }
//! }
//! ```
//!
//! ## Error Recovery Strategies
//!
//! ### Automatic Retry with Adjusted Settings
//! ```rust
//! # use augmecon::*;
//! fn solve_with_retry(
//!     problem: MultiObjectiveProblem,
//!     mut options: Options
//! ) -> Result<Augmecon> {
//!     // Try with original settings
//!     match Augmecon::new(problem, options.clone()) {
//!         Ok(mut solver) => {
//!             match solver.solve() {
//!                 Ok(()) => return Ok(solver),
//!                 Err(AugmeconError::OptimizationError(_)) => {
//!                     // Try with more conservative settings
//!                     options = options
//!                         .with_grid_points(options.grid_points.unwrap_or(50) / 2)
//!                         .with_penalty_weight(options.penalty_weight * 10.0);
//!                 },
//!                 Err(e) => return Err(e),
//!             }
//!         },
//!         Err(e) => return Err(e),
//!     }
//!     
//!     // Second attempt with adjusted settings
//!     let mut solver = Augmecon::new(problem, options)?;
//!     solver.solve()?;
//!     Ok(solver)
//! }
//! ```
//!
//! ## Best Practices
//!
//! 1. **Always Handle Errors**: Don't `unwrap()` in production code
//! 2. **Provide Context**: Use error messages to guide problem fixing
//! 3. **Validate Early**: Check problem structure before solving
//! 4. **Graceful Degradation**: Fall back to simpler settings on failure
//! 5. **Log Appropriately**: Use different log levels for different error types

use thiserror::Error;

/// Error types for the AUGMECON solver
#[derive(Error, Debug)]
pub enum AugmeconError {
    /// Invalid number of objectives provided (must be at least 2)
    #[error("Invalid number of objectives: {0}. Must be at least 2")]
    InvalidObjectiveCount(usize),

    /// No grid points specified for epsilon-constraint method
    #[error("No grid points specified")]
    NoGridPoints,

    /// Invalid options configuration
    #[error("Invalid options: {0}")]
    InvalidOptions(String),

    /// Mismatch between expected and actual number of nadir points
    #[error("Invalid nadir points: expected {expected}, got {actual}")]
    InvalidNadirPoints {
        /// Expected number of nadir points
        expected: usize,
        /// Actual number of nadir points provided
        actual: usize,
    },

    /// Error during optimization process
    #[error("Optimization error: {0}")]
    OptimizationError(String),

    /// Error in problem model definition
    #[error("Model error: {0}")]
    ModelError(String),

    /// Input/output error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Error reading Excel files
    #[error("Excel reading error: {0}")]
    ExcelError(#[from] calamine::Error),

    /// Solver not supported (disabled at compile time)
    #[error("Unsupported solver: {0}")]
    UnsupportedSolver(String),
}

/// Result type used throughout the AUGMECON library
pub type Result<T> = std::result::Result<T, AugmeconError>;
