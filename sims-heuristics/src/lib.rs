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

#![feature(portable_simd)]
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

/// Unchecked read: uses bounds-checked indexing when `bounds_check` feature is enabled.
#[cfg(feature = "bounds_check")]
macro_rules! unchecked_get {
    ($slice:expr, $idx:expr) => {
        $slice[$idx]
    };
}

/// Unchecked read: uses unsafe get_unchecked when `bounds_check` feature is disabled.
#[cfg(not(feature = "bounds_check"))]
macro_rules! unchecked_get {
    ($slice:expr, $idx:expr) => {
        unsafe { *$slice.get_unchecked($idx) }
    };
}

/// Unchecked mutable reference: uses bounds-checked indexing when `bounds_check` feature is enabled.
#[cfg(feature = "bounds_check")]
macro_rules! unchecked_get_mut {
    ($slice:expr, $idx:expr) => {
        &mut $slice[$idx]
    };
}

/// Unchecked mutable reference: uses unsafe get_unchecked_mut when `bounds_check` feature is disabled.
#[cfg(not(feature = "bounds_check"))]
macro_rules! unchecked_get_mut {
    ($slice:expr, $idx:expr) => {
        unsafe { $slice.get_unchecked_mut($idx) }
    };
}

/// Unchecked slice: uses bounds-checked slicing when `bounds_check` feature is enabled.
#[cfg(feature = "bounds_check")]
macro_rules! unchecked_slice {
    ($slice:expr, $range:expr) => {
        &$slice[$range]
    };
}

/// Unchecked slice: uses unsafe get_unchecked when `bounds_check` feature is disabled.
#[cfg(not(feature = "bounds_check"))]
macro_rules! unchecked_slice {
    ($slice:expr, $range:expr) => {
        unsafe { $slice.get_unchecked($range) }
    };
}

/// Unchecked pointer read: uses bounds-checked indexing when `bounds_check` feature is enabled.
/// For raw pointer patterns like `*ptr.add(idx)`, pass the original slice and index instead.
#[cfg(feature = "bounds_check")]
macro_rules! unchecked_ptr_read {
    ($slice:expr, $idx:expr) => {
        $slice[$idx]
    };
}

/// Unchecked pointer read: uses raw pointer arithmetic when `bounds_check` feature is disabled.
#[cfg(not(feature = "bounds_check"))]
macro_rules! unchecked_ptr_read {
    ($slice:expr, $idx:expr) => {
        unsafe { *$slice.as_ptr().add($idx) }
    };
}

/// Unchecked pointer write: uses bounds-checked indexing when `bounds_check` feature is enabled.
#[cfg(feature = "bounds_check")]
macro_rules! unchecked_ptr_write {
    ($slice:expr, $idx:expr, $val:expr) => {
        $slice[$idx] = $val
    };
}

/// Unchecked pointer write: uses raw pointer arithmetic when `bounds_check` feature is disabled.
#[cfg(not(feature = "bounds_check"))]
macro_rules! unchecked_ptr_write {
    ($slice:expr, $idx:expr, $val:expr) => {
        unsafe { *$slice.as_mut_ptr().add($idx) = $val }
    };
}

pub(crate) use unchecked_get;
pub(crate) use unchecked_get_mut;
pub(crate) use unchecked_slice;
pub(crate) use unchecked_ptr_read;
pub(crate) use unchecked_ptr_write;
pub mod explored_solutions_data;
pub mod objective_tracker;
pub mod objective_tracker_impl;
pub mod objectives;
pub mod pareto_local_search;
#[cfg(feature = "plotting")]
pub mod plotting;
pub mod problem;
pub mod problem_bitset;
pub mod residual_problem;
pub mod residual_solution;
pub mod solution;
pub mod solution_impl;
pub mod solution_set_impl;
pub mod timer;
pub mod util;
pub use problem_bitset::ProblemBitset;

// Re-export key traits
pub use problem::SetCoverProblem;
