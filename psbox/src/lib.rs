//! Exact multiobjective integer programming by criterion-space rectangle search.
//!
//! Implements the algorithm of Kirlik & Sayin, *A new algorithm for generating
//! all nondominated solutions of multiobjective discrete optimization problems*
//! (EJOR 232(3), 2014), following the structure of the reference implementation
//! in [`MultiObjectiveAlgorithms.jl`](https://github.com/jump-dev/MultiObjectiveAlgorithms.jl).
//!
//! It is used here for one property that scalarisation-with-augmentation
//! methods do not offer: **every reported point is nondominated**, established
//! by an explicit second-stage solve rather than by an augmentation term whose
//! weight has to stay above the solver's tolerance. That distinction is not
//! academic. In the AUGMECON-family implementation this crate exists to
//! complement, the augmentation coefficient worked out around seven orders of
//! magnitude too small against the primary objective, and roughly a fifth of
//! the "exact" points it returned were dominated.
//!
//! Two further properties follow from the structure:
//!
//! * **A failed solve costs one rectangle, not the run.** A subproblem that
//!   hits its time limit or trips a numerical error is skipped; the search
//!   continues and reports itself non-exhaustive.
//! * **Useful truncation.** The rectangle reaching furthest from the ideal
//!   point is explored first, so a run stopped by its deadline returns points
//!   spread over the frontier, and never an unproven incumbent.
//!
//! Requires a solver that reports why it stopped, since the guarantee rests on
//! distinguishing a proven optimum from an incumbent; only the `gurobi` backend
//! does, and without it every solve reports as unproven.
//!
//! Deliberately standalone: it shares no solver layer with `augmecon-rs`, so the
//! two can be compared head to head without a common defect flattering both.
//!
//! # Name
//!
//! The crate was first written around the Pascoletti-Serafini box search of
//! Dogan, Karsu & Ulus, hence `psbox`. That formulation needed a free
//! continuous variable per subproblem and an equality-pinned second stage which
//! could not be solved within budget on the target instances; the algorithm
//! here replaces it. The name is kept for continuity with existing result
//! directories.

pub mod front;
pub mod model;
pub mod quadtree;
pub mod rectangle;
pub mod search;
pub mod solve;

pub use front::{Front, Solution};
pub use model::{Bounds, Config, Feasibility, Problem, Split};
pub use rectangle::Rectangle;
pub use search::{Outcome, compute_bounds, solve};
