//! Evolutionary Multi-Objective Optimization Algorithms for SIMS
//!
//! This module provides population-based evolutionary algorithms adapted for the
//! Satellite Image Mosaic Selection (SIMS) problem, a constrained set-cover problem
//! with multiple objectives (all minimization).
//!
//! ## Algorithms
//!
//! ### Custom Implementations
//!
//! - **NSGA-II**: Non-dominated Sorting Genetic Algorithm II with coverage-aware
//!   crossover and repair operators.
//! - **MOEA/D**: Multi-Objective Evolutionary Algorithm based on Decomposition using
//!   Tchebycheff scalarization with adaptive weight vectors.
//!
//! ### Literal paper baselines (`nsga2_baseline`, `nsga3_baseline`, `moead_baseline`)
//!
//! Unmodified reference implementations of Deb et al. (2002), Deb & Jain
//! (2014), and Zhang & Li (2007), used to measure whether the SIMS-specific
//! extensions in the custom implementations above actually help, rather than
//! assuming they do because the techniques are individually established
//! elsewhere. See each module's doc comment for the exact deviations (all
//! either unavoidable for the bitset set-cover representation, or later
//! additions from other papers, not the original three).
//!
//! ### External Crate Adapters (feature `external_solvers`)
//!
//! - **moors adapter**: NSGA-II, SPEA-2, and AGE-MOEA from the [`moors`](https://crates.io/crates/moors)
//!   crate, using native binary operators (uniform/single-point/two-point crossover,
//!   bit-flip mutation) with greedy repair in the fitness function.
//! - **optirustic adapter**: NSGA-II and NSGA-III from the [`optirustic`](https://crates.io/crates/optirustic)
//!   crate, using continuous relaxation ([0,1] variables thresholded at 0.5) with
//!   SBX crossover + polynomial mutation and repair in the evaluator.
//!
//! ## Design
//!
//! All algorithms share common genetic operators (`operators` module) that are
//! specifically designed for set-cover feasibility:
//!
//! - **Crossover**: Uniform crossover on image-selection bitsets followed by greedy repair
//! - **Mutation**: Swap, add/remove, and shift mutations that maintain or restore feasibility
//! - **Repair**: Greedy coverage repair + redundancy removal for lean feasible solutions
//!
//! All algorithms use the same `BitsetEncodedSolution` and `ProblemBitset` types as PLS,
//! ensuring full compatibility with the rest of the framework (tracking, explored solutions
//! storage, and output).

pub mod memetic;
pub mod moead;
pub mod moead_baseline;
pub mod nsga2;
pub mod nsga2_baseline;
pub mod nsga3;
pub mod nsga3_baseline;
pub mod operators;

#[cfg(feature = "external_solvers")]
pub mod moors_adapter;
#[cfg(feature = "external_solvers")]
pub mod optirustic_adapter;

// Re-export main algorithm entry points
pub use memetic::{
    EaBackend, MemeticAlgorithm, MemeticConfig, MemeticResult, run_memetic_moead,
    run_memetic_nsga2, run_memetic_nsga3,
};
pub use moead::Moead;
pub use moead_baseline::{MoeadBaseline, MoeadBaselineConfig, run_moead_baseline};
pub use nsga2::Nsga2;
pub use nsga2_baseline::{Nsga2Baseline, Nsga2BaselineConfig, run_nsga2_baseline};
pub use nsga3::{Nsga3, Nsga3Config};
pub use nsga3_baseline::{Nsga3Baseline, Nsga3BaselineConfig, run_nsga3_baseline};

// Re-export external adapter entry points when the feature is enabled
#[cfg(feature = "external_solvers")]
pub use moors_adapter::{
    MoorsConfig, MoorsCrossoverType, run_moors_age_moea, run_moors_nsga2, run_moors_spea2,
};
#[cfg(feature = "external_solvers")]
pub use optirustic_adapter::{OptirusticConfig, run_optirustic_nsga2, run_optirustic_nsga3};
