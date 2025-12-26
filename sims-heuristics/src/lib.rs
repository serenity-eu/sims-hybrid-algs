//! # SIMS Heuristics Library
//!
//! This library provides heuristic algorithms for solving the Satellite Image Selection Problem (SIMS)
//! using Multi-Objective Optimization techniques, particularly Pareto Local Search.
//!
//! ## Generic Objectives System
//!
//! The library supports generic, extensible objectives using a trait-based
//! design pattern. This allows for flexible definition of optimization objectives with
//! arbitrary dimensionality.
//!
//! ### Key Components
//!
//! - **`ObjectiveDefinition<D>`**: A trait that defines how to calculate objective values and deltas
//! - **`SolutionEvaluator<D>`**: A trait that solutions implement to evaluate themselves against objectives
//! - **`ProblemBuilder<D>`**: A builder pattern for constructing problems with custom objectives
//! - **Generic Solutions**: All solution types support arbitrary dimensionality `D`
//!
//! ### Example Usage
//!
//! ```rust,ignore
//! use sims_heuristics::{
//!     objectives::{TotalCostObjective, CloudyAreaObjective},
//!     problem::ProblemBuilder,
//! };
//!
//! // Create a problem with custom objectives
//! let problem = ProblemBuilder::<3>::new()
//!     .add_objective(Box::new(TotalCostObjective { index: 0 }))
//!     .add_objective(Box::new(CloudyAreaObjective { index: 1 }))
//!     .add_objective(Box::new(CustomObjective { index: 2 }))
//!     .build_from_legacy(raw_problem)?;
//! ```
//!
//! ### Generic Weight System
//!
//! All weight generation and calculations use generic arrays `[f32; D]` instead of tuples,
//! providing flexibility for any number of objectives. Use `generate_weights::<D>()` to
//! create random weights that sum to 1.0 for D-dimensional objective spaces.

#![expect(
    clippy::cast_precision_loss,
    reason = "Legacy code style, extensive refactor needed"
)]
#![expect(
    clippy::cast_sign_loss,
    reason = "Legacy code style, extensive refactor needed"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "Legacy code style, extensive refactor needed"
)]
#![expect(
    clippy::cast_possible_wrap,
    reason = "Legacy code style, extensive refactor needed"
)]
pub mod examples;
pub mod explored_solutions_data;
pub mod objectives;
pub mod pareto_local_search;
#[cfg(feature = "plotting")]
pub mod plotting;
pub mod probabilistic_probing_neighborhood;
pub mod problem;
pub mod residual_problem;
pub mod residual_solution;
pub mod solution;
pub mod solution_impl;
pub mod solution_set_impl;
pub mod timer;
pub mod trackers;
pub mod util;
pub mod problem_bitset;
pub use problem_bitset::ProblemBitset;

// Re-export key traits
pub use problem::SetCoverProblem;
