//! Anytime Aneja & Nair (dichotomic search) for the bi-objective SIMS problem.
//!
//! Aneja & Nair (1979) generate the **supported** Pareto-optimal solutions of a
//! bi-objective problem by minimising weighted sums `w₁f₁ + w₂f₂`. We use the
//! *anytime* variant of Dubois-Lacoste et al. (2011): start from the two extreme
//! points, then repeatedly fill the **largest** remaining gap so the front is
//! well distributed at any interruption.
//!
//! This mirrors [`crate::gpba::GpbaA`] (`new` → `with_timeout` →
//! `generate_representation → ParetoFront`) and reuses
//! [`SingleObjectiveSolver`] for every MILP solve, so it inherits the same
//! solver backends, per-solve timeout, incumbent acceptance, and (for `HiGHS`)
//! `presolve=off`. Unlike GPBA-A it solves a plain weighted sum — no slack, no
//! augmentation — and finds supported points only.
//!
//! ## Numerical note
//! The SIMS objectives are large integers (cost ~10⁶, cloudy area ~10⁹), so the
//! raw equalizing weights (objective *differences*) reach ~10⁹. We therefore
//! (a) pass the weight vector **normalised to sum 1** to the solver — scaling the
//! objective does not change its argmin but keeps coefficients well inside the
//! `f64` exact range — and (b) perform the "is this a new supported point" test
//! with **exact `i128` integer arithmetic** on the un-normalised weights.

use crate::{
    error::{AugmeconError, Result},
    model::MultiObjectiveProblem,
    options::Options,
    single_objective::SingleObjectiveSolver,
    solution::{ParetoFront, Solution},
    timer::Timer,
};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Duration;

/// Configuration for the Anytime Aneja & Nair method.
#[derive(Debug, Clone, Default)]
pub struct AnejaNairConfig {
    /// Wall-clock cap for a single weighted-sum solve. `None` = bounded only by
    /// the remaining global timeout. Prevents one hard subproblem from consuming
    /// the whole budget (mirrors `GpbaConfig::per_solve_timeout`).
    pub per_solve_timeout: Option<Duration>,
    /// Stop once the front reaches this many points. `None` = fill until the
    /// front is exhausted or the timeout fires.
    pub target_solutions: Option<usize>,
}

/// Anytime Aneja & Nair dichotomic-search driver.
pub struct AnejaNair {
    config: AnejaNairConfig,
    timer: Option<Timer>,
}

/// A front point carrying its full solution and (f₁, f₂) image.
#[derive(Clone)]
struct Point {
    f1: f64,
    f2: f64,
    sol: Solution,
}

/// A gap between two adjacent supported points, prioritised by the objective-space
/// rectangle area between them (largest gap filled first → anytime behaviour).
struct Gap {
    /// smaller f₁, larger f₂
    lo: Point,
    /// larger f₁, smaller f₂
    hi: Point,
    area: i64,
}

impl Gap {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "objective differences are integers stored as f64; .round() casts are exact"
    )]
    fn new(lo: Point, hi: Point) -> Self {
        let df1 = (hi.f1 - lo.f1).round() as i64;
        let df2 = (lo.f2 - hi.f2).round() as i64;
        Self {
            area: df1.saturating_mul(df2),
            lo,
            hi,
        }
    }
}

impl PartialEq for Gap {
    fn eq(&self, other: &Self) -> bool {
        self.area == other.area
    }
}
impl Eq for Gap {}
impl PartialOrd for Gap {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Gap {
    // Max-heap on area: the largest gap is filled first.
    fn cmp(&self, other: &Self) -> Ordering {
        self.area.cmp(&other.area)
    }
}

impl AnejaNair {
    /// Create a new Anytime Aneja & Nair driver.
    #[must_use]
    pub const fn new(config: AnejaNairConfig) -> Self {
        Self {
            config,
            timer: None,
        }
    }

    /// Set the global wall-clock budget.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timer = Some(Timer::start(timeout));
        self
    }

    fn is_timeout_reached(&self) -> bool {
        self.timer.as_ref().is_some_and(Timer::is_expired)
    }

    /// Record the wall-clock discovery time (µs since start) in the solution's
    /// metadata, so callers (and the hybrid pseudo-seeding) can reconstruct the
    /// anytime front timeline — matching the GPBA-A convention.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "elapsed microseconds for one solve stay far below u64::MAX"
    )]
    fn stamp(&self, mut sol: Solution) -> Solution {
        let elapsed_us = self
            .timer
            .as_ref()
            .map_or(0, |t| t.elapsed().as_micros() as u64);
        sol.metadata
            .insert("timestamp_us".to_string(), elapsed_us.to_string());
        sol
    }

    /// Budget for one gap search: the configured cap normally, the whole of
    /// the remaining time on the retry pass, where no other gap is waiting.
    fn gap_budget(&self, uncapped: bool) -> Option<Duration> {
        if uncapped {
            self.timer.as_ref().map(Timer::remaining)
        } else {
            self.per_solve()
        }
    }

    /// Per-solve budget = min(configured cap, remaining global time).
    fn per_solve(&self) -> Option<Duration> {
        let remaining = self.timer.as_ref().map(Timer::remaining);
        match (self.config.per_solve_timeout, remaining) {
            (Some(cap), Some(rem)) => Some(cap.min(rem)),
            (Some(cap), None) => Some(cap),
            (None, rem) => rem,
        }
    }

    /// Generate a representative subset of the Pareto front via anytime dichotomic
    /// search. Returns supported Pareto-optimal points, well distributed across
    /// the front, within the configured timeout.
    ///
    /// # Errors
    /// Returns an error if any MILP solve fails.
    ///
    /// # Panics
    /// Panics if the problem is not bi-objective (Aneja & Nair is defined for two
    /// objectives).
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::needless_pass_by_ref_mut,
        reason = "objectives are integers stored as f64 (.round() casts are exact); &mut self mirrors GpbaA::generate_representation for API symmetry"
    )]
    pub fn generate_representation(
        &mut self,
        problem: &MultiObjectiveProblem,
        options: &Options,
    ) -> Result<ParetoFront> {
        assert_eq!(
            problem.num_objectives(),
            2,
            "Aneja & Nair is a bi-objective method"
        );
        log::info!("=== Anytime Aneja & Nair: starting ===");
        let solver = SingleObjectiveSolver::new(problem, options);
        // The gap-filling loop below solves dozens of weighted sums that differ
        // only in their weights, so it builds the model once and replaces just
        // the objective. Available on the native Gurobi backend alone; every
        // other backend keeps rebuilding, which is correct, only slower.
        #[cfg(feature = "gurobi")]
        let mut session = matches!(options.solver, crate::solver_enum::Solver::Gurobi)
            .then(|| crate::single_objective::ScalarisationSession::new(problem, options));
        let directions = problem.objectives.iter().map(|(_, dir)| *dir).collect();
        let mut front = ParetoFront::new(directions);
        // Gaps whose subproblem could not be proven optimal within the
        // per-solve cap, retried once the queue drains and the remaining budget
        // can be given to a single solve.
        let mut deferred: Vec<Gap> = Vec::new();
        let mut unproven_gaps = 0usize;

        // --- Two lexicographic extreme points ---
        // Without both extremes there is no initial gap to bisect, so an
        // unprovable extreme leaves nothing to search. Return the empty front
        // rather than an error: "no point could be proven in the budget" is a
        // result the caller can record, where an exception discards the run and
        // reads like a crash.
        let (a, b) = match (
            self.lexicographic_extreme(&solver, problem, 0, 1), // min f1, tie-break min f2
            self.lexicographic_extreme(&solver, problem, 1, 0), // min f2, tie-break min f1
        ) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(AugmeconError::NotProvenOptimal), _) | (_, Err(AugmeconError::NotProvenOptimal)) => {
                log::warn!(
                    "Aneja & Nair: a lexicographic extreme could not be proven optimal within \
                     the budget; returning an empty front"
                );
                return Ok(front);
            }
            (Err(e), _) | (_, Err(e)) => return Err(e),
        };
        front.add_solution(self.stamp(a.sol.clone()));
        if (a.f1, a.f2) == (b.f1, b.f2) {
            log::info!("Aneja & Nair: single Pareto point (extremes coincide)");
            return Ok(front);
        }
        front.add_solution(self.stamp(b.sol.clone()));

        // Order so lo has the smaller f1 (larger f2).
        let (lo, hi) = if a.f1 <= b.f1 { (a, b) } else { (b, a) };

        // --- Anytime gap-filling: always split the largest remaining gap ---
        let mut gaps = BinaryHeap::new();
        gaps.push(Gap::new(lo, hi));

        // Once the queue drains, hand the deferred gaps the rest of the budget
        // and search them again. The per-solve cap is right while many gaps
        // compete for time, but when none are left the remainder has nowhere
        // else to go: a 150-image instance was finishing in 60s of 200s with
        // its only gap abandoned, and the solve it gave up on is one measured
        // to close in ~236s.
        let mut uncapped = false;
        loop {
            let Some(gap) = gaps.pop() else {
                if uncapped || deferred.is_empty() {
                    break;
                }
                let left = self.timer.as_ref().map_or(Duration::MAX, Timer::remaining);
                if left.is_zero() {
                    break;
                }
                log::debug!(
                    "Aneja & Nair: retrying {} deferred gap(s) with {left:?} left",
                    deferred.len()
                );
                gaps.extend(deferred.drain(..));
                uncapped = true;
                continue;
            };
            if self.is_timeout_reached() {
                log::info!(
                    "Aneja & Nair: timeout reached, front size {}",
                    front.solutions.len()
                );
                break;
            }
            if let Some(target) = self.config.target_solutions {
                if front.solutions.len() >= target {
                    break;
                }
            }
            if gap.area <= 0 {
                continue; // adjacent integer points; nothing can lie strictly between
            }

            // Equalizing weight (positive integer differences), normalised to sum 1.
            let wi1 = (gap.lo.f2 - gap.hi.f2).round() as i64;
            let wi2 = (gap.hi.f1 - gap.lo.f1).round() as i64;
            let sum = (wi1 + wi2) as f64;
            let weights = [wi1 as f64 / sum, wi2 as f64 / sum];

            // The gap test below only asks whether a point lies *strictly* below
            // the equalizing line, so hand the solver that threshold as a
            // cutoff: proving the half-space empty prunes, where proving the
            // optimum over the whole space searches.
            //
            // The threshold must be strictly below the line. Equalizing weights
            // put both endpoints exactly on it (`w.lo == w.hi == wline`), so a
            // cutoff at `wline` never prunes -- an optimum at the cutoff always
            // exists. Objectives and weights are integers, so `wz` is an
            // integer and "strictly below" means `wz <= wline - 1`; the
            // half-unit offset separates the two cases without depending on
            // whether Gurobi compares with `>` or `>=`.
            let wline_i = i128::from(wi1) * i128::from(gap.lo.f1.round() as i64)
                + i128::from(wi2) * i128::from(gap.lo.f2.round() as i64);
            let wline_f = (wline_i as f64 - 0.5) / sum;
            let solved = {
                #[cfg(feature = "gurobi")]
                {
                    match session.as_mut() {
                        Some(session) => session.solve_with_cutoff(
                            &weights,
                            self.gap_budget(uncapped),
                            Some(wline_f),
                        ),
                        None => solver.solve_weighted_sum(&weights, self.gap_budget(uncapped)),
                    }
                }
                #[cfg(not(feature = "gurobi"))]
                {
                    solver.solve_weighted_sum(&weights, self.gap_budget(uncapped))
                }
            };
            let sol = match solved {
                Ok(sol) => sol,
                // Proof that this gap holds no supported point: close it and
                // move on, exactly as the `wz >= wline` branch below would.
                Err(AugmeconError::ProvedNoneBelowCutoff) => {
                    log::warn!("Aneja & Nair: gap proven empty by cutoff");
                    continue;
                }
                // The incumbent is feasible but unproven, so it cannot be added
                // and this gap cannot be subdivided. Abandon just this gap and
                // carry on with the rest of the queue: whether a given solve
                // closes inside its cap is timing-dependent, so stopping the
                // whole sweep here made the result a coin flip -- the same
                // instance yielded 13 points in one run and none in the next.
                // Other gaps are independent and may well be provable.
                Err(AugmeconError::NotProvenOptimal) => {
                    // Defer rather than discard. The per-solve cap exists to
                    // spread the budget over many gaps, but when the queue
                    // drains early the remaining time is simply wasted: a
                    // 150-image instance was finishing in 60s of 200s with its
                    // only gap abandoned. Retried below with whatever budget is
                    // left, which is often far more than the cap allowed.
                    if uncapped {
                        unproven_gaps += 1;
                    } else {
                        deferred.push(gap);
                    }
                    continue;
                }
                Err(e) => return Err(e),
            };
            let z = Point {
                f1: sol.objective_values[0],
                f2: sol.objective_values[1],
                sol,
            };

            // Exact i128 test: is z strictly below the equalizing line through lo,hi?
            let z1 = z.f1.round() as i64;
            let z2 = z.f2.round() as i64;
            let wz = i128::from(wi1) * i128::from(z1) + i128::from(wi2) * i128::from(z2);
            let wline = i128::from(wi1) * i128::from(gap.lo.f1.round() as i64)
                + i128::from(wi2) * i128::from(gap.lo.f2.round() as i64);

            if wz < wline {
                log::debug!("Aneja & Nair: new supported point ({z1}, {z2})");
                front.add_solution(self.stamp(z.sol.clone()));
                gaps.push(Gap::new(gap.lo.clone(), z.clone()));
                gaps.push(Gap::new(z, gap.hi));
            }
            // else: no supported solution in this gap (exact solver) → gap closed.
        }

        if unproven_gaps > 0 {
            log::warn!(
                "Aneja & Nair: {unproven_gaps} gap(s) abandoned without a proven optimum; \
                 the returned front is proven but may be incomplete"
            );
        }
        log::info!(
            "=== Aneja & Nair: done, {} points ({unproven_gaps} gaps abandoned) ===",
            front.solutions.len()
        );
        Ok(front)
    }

    /// Lexicographic extreme: minimise objective `primary`, then minimise
    /// `secondary` subject to `f_primary <= its optimum`, yielding a true Pareto
    /// extreme (not a merely weakly-dominated optimum of `primary`).
    fn lexicographic_extreme(
        &self,
        solver: &SingleObjectiveSolver,
        problem: &MultiObjectiveProblem,
        primary: usize,
        secondary: usize,
    ) -> Result<Point> {
        let s1 = solver.solve_objective(primary, self.per_solve())?;
        let primary_star = s1.objective_values[primary];
        let f_primary = problem.objectives[primary].0.clone();
        let f_secondary = problem.objectives[secondary].0.clone();
        let s2 = solver.solve_minimize_with_constraints(
            f_secondary,
            &[f_primary.leq(primary_star)],
            self.per_solve(),
        )?;
        Ok(Point {
            f1: s2.objective_values[0],
            f2: s2.objective_values[1],
            sol: s2,
        })
    }
}
