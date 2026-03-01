//! Concurrent Pareto Local Search via objective-space decomposition.
//!
//! Parallelizes PLS by decomposing objective space into `N` regions using
//! Das-Dennis weight vectors. Each thread runs an independent PLS instance on its
//! assigned region, periodically publishing snapshots for cross-region adoption and pruning.
//!
//! # Feature Gate
//!
//! This module requires the `parallel` cargo feature:
//!
//! ```toml
//! # Cargo.toml
//! sims-heuristics = { path = "..", features = ["parallel"] }
//! ```
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use pls::concurrent_pls::{ConcurrentPLS, ConcurrentPLSConfig};
//! use std::time::Duration;
//!
//! let config = ConcurrentPLSConfig::default_with_threads(
//!     num_cpus::get(),
//!     Duration::from_secs(120),
//! );
//! let result = ConcurrentPLS::new(&problem, config).solve(&initial_population);
//! println!("Found {} Pareto-optimal solutions", result.len());
//! ```

pub mod config;
pub mod decomposition;
pub mod orchestrator;
pub mod snapshot;
pub mod worker;

pub use config::{ConcurrentPLSConfig, RegionSearchMode};
pub use orchestrator::{ConcurrentPLS, ConcurrentPLSResult};
pub use worker::{RegionResult, RegionStats};
