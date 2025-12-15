//! Phased GPBA-A API for testing and validation
//!
//! This module provides a phased API for GPBA-A that breaks down the algorithm
//! into discrete testable phases. Each phase can be tested independently with
//! hardcoded inputs from Python reference implementation.
//!
//! This exists alongside the existing `gpba.rs` implementation and is used for:
//! - Testing individual algorithm phases
//! - Validating Rust implementation against Python reference
//! - Understanding algorithm flow
//!
//! The existing `gpba.rs::GpbaA::generate_representation()` remains the primary
//! production implementation.

/// Cascading dimension reset phase
pub mod cascading;
/// Epsilon parameter adjustment phase
pub mod epsilon_adjustment;
/// Epsilon constraint setup phase
pub mod epsilon_setup;
/// Epsilon constraint solving phase
pub mod epsilon_solve;
/// Interval management for adaptive exploration
pub mod interval_manager;
/// Payoff table calculation phase
pub mod payoff_table;
/// Solution relaxation search phase
pub mod relaxation_search;

// TODO: Add remaining phase modules as they are implemented
// pub mod epsilon_setup;
// pub mod epsilon_solve;
// pub mod cascading;
// pub mod main_loop;
