//! A HiGHS model that outlives a single large-neighbourhood move.
//!
//! Rebuilding the MILP per move dominates the cost of an LNS pass: the columns
//! and the coverage rows are identical every time, and only the variable bounds
//! and the epsilon right-hand side change. This wrapper builds the model once
//! and mutates it in place, which is what makes a move cost a solve rather than
//! a solve plus a rebuild.
//!
//! # Formulation
//!
//! For images `i` and universe elements `e`:
//!
//! * `x[i] ∈ {0,1}` — image `i` is selected. Objective coefficient `cost[i]`.
//! * `y[e] ∈ [0,1]` — element `e` is seen *clearly* by some selected image.
//!
//! subject to
//!
//! * coverage:  `Σ_{i ∋ e} x[i] ≥ 1`                    (one row per element)
//! * linking:   `Σ_{i: e clear in i} x[i] − y[e] ≥ 0`   (one row per element)
//! * epsilon:   `Σ_e area[e]·y[e] ≥ ε`                  (one row)
//!
//! `y` is continuous rather than integral on purpose. Because `y[e]` is pushed
//! up only by the epsilon row and bounded above by an integral quantity, an
//! optimal `y` is integral whenever `x` is, so relaxing it removes variables
//! from the branching tree without weakening the formulation.
use fixedbitset::FixedBitSet;
use highs::{HighsModelStatus, Model, RowProblem, Sense};

use crate::{objectives::ObjectiveState, problem_bitset::ProblemBitset};

/// Which objective the next solve minimises. Balanced Box needs both
/// directions: lexicographic optimisation alternates between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// Minimise total cost.
    Cost,
    /// Minimise cloudy area, i.e. maximise the clear area covered.
    Cloud,
}

/// Outcome of one solve. `Feasible` carries the best incumbent found within the
/// time limit; a proven-optimal solve reports `Optimal` so callers can tell the
/// difference when deciding whether to keep searching a neighbourhood.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveOutcome {
    Optimal,
    Feasible,
    NoSolution,
}

pub struct PersistentModel {
    model: Option<Model>,
    /// Clear-coverage sets, kept so a warm start can supply consistent `y`
    /// values alongside the `x` it sets.
    clear: Vec<FixedBitSet>,
    warm: Vec<f64>,
    num_images: usize,
    num_elements: usize,
    /// Row index of the clear-area floor, mutated per solve.
    epsilon_row: usize,
    /// Row index of the cost ceiling, used for the lexicographic second stage.
    cost_row: usize,
    /// Objective currently loaded into the column costs.
    objective: Objective,
    costs: Vec<f64>,
    areas: Vec<f64>,
}

impl PersistentModel {
    /// Build the model once for `problem`.
    ///
    /// # Panics
    /// Panics if the instance has an element that no image covers, which makes
    /// the set-cover infeasible regardless of the neighbourhood.
    #[must_use]
    pub fn new<const D: usize>(problem: &ProblemBitset<D>, clear: &[FixedBitSet], areas: &[u64]) -> Self {
        let n = problem.images.len();
        let m = problem.universe_size;
        let mut pb = RowProblem::default();

        let x: Vec<_> = (0..n)
            .map(|i| pb.add_integer_column(problem.image_cost(i) as f64, 0..=1))
            .collect();
        let y: Vec<_> = (0..m).map(|_| pb.add_column(0.0, 0.0..=1.0)).collect();

        for e in 0..m {
            let cols: Vec<_> = problem
                .element_covering_images(e)
                .iter()
                .map(|&i| (x[i], 1.0))
                .collect();
            assert!(!cols.is_empty(), "element {e} is covered by no image");
            pb.add_row(1.0.., &cols);
        }

        // Transpose the clear-coverage bitsets once: which images see e clearly.
        let mut clear_cols: Vec<Vec<usize>> = vec![Vec::new(); m];
        for (i, bs) in clear.iter().enumerate().take(n) {
            for e in bs.ones() {
                if e < m {
                    clear_cols[e].push(i);
                }
            }
        }
        for (e, imgs) in clear_cols.iter().enumerate() {
            if imgs.is_empty() {
                // No image sees e clearly, so y[e] can never be 1.
                pb.add_row(..=0.0, &[(y[e], 1.0)]);
                continue;
            }
            let mut cols: Vec<_> = imgs.iter().map(|&i| (x[i], 1.0)).collect();
            cols.push((y[e], -1.0));
            pb.add_row(0.0.., &cols);
        }

        // Clear-area floor, RHS rewritten per solve.
        let area_cols: Vec<_> = (0..m).map(|e| (y[e], areas[e] as f64)).collect();
        pb.add_row(0.0.., &area_cols);
        let epsilon_row = 2 * m;
        // Cost ceiling, needed to pin stage one of a lexicographic solve while
        // stage two optimises the other objective. Open by default.
        let cost_cols: Vec<_> = (0..n).map(|i| (x[i], problem.image_cost(i) as f64)).collect();
        pb.add_row(..=f64::INFINITY, &cost_cols);
        let cost_row = 2 * m + 1;

        let mut model = pb.optimise(Sense::Minimise);
        model.make_quiet();
        Self {
            model: Some(model),
            clear: clear.to_vec(),
            warm: vec![0.0; n + m],
            costs: (0..n).map(|i| problem.image_cost(i) as f64).collect(),
            areas: areas.iter().map(|&a| a as f64).collect(),
            objective: Objective::Cost,
            cost_row,
            num_images: n,
            num_elements: m,
            epsilon_row,
        }
    }

    fn ptr(&mut self) -> *mut std::ffi::c_void {
        self.model.as_mut().expect("model present").as_mut_ptr()
    }

    /// Point every column cost at `objective`. Only the columns that actually
    /// carry the objective change, so switching direction is O(n + m) FFI calls
    /// rather than a rebuild.
    fn set_objective(&mut self, objective: Objective) {
        if self.objective == objective {
            return;
        }
        let ptr = self.ptr();
        for i in 0..self.num_images {
            let c = match objective {
                Objective::Cost => self.costs[i],
                Objective::Cloud => 0.0,
            };
            // SAFETY: `i` indexes a column created in `new`.
            unsafe { highs_sys::Highs_changeColCost(ptr, i as highs_sys::HighsInt, c) };
        }
        for e in 0..self.num_elements {
            // Minimising cloudy area maximises the clear area seen, so `y`
            // carries a negative coefficient in that direction.
            let c = match objective {
                Objective::Cost => 0.0,
                Objective::Cloud => -self.areas[e],
            };
            let col = (self.num_images + e) as highs_sys::HighsInt;
            // SAFETY: `col` indexes a column created in `new`.
            unsafe { highs_sys::Highs_changeColCost(ptr, col, c) };
        }
        self.objective = objective;
    }

    /// Fix every image to its incumbent value except those in `free`, apply the
    /// clear-area floor and cost ceiling, and solve for `objective`.
    pub fn solve(
        &mut self,
        incumbent: &FixedBitSet,
        free: &FixedBitSet,
        min_clear_area: f64,
        max_cost: f64,
        objective: Objective,
        seconds: f64,
    ) -> (SolveOutcome, FixedBitSet) {
        self.set_objective(objective);
        // SAFETY: `cost_row` is the row index returned when it was added.
        unsafe {
            highs_sys::Highs_changeRowBounds(
                self.ptr(),
                self.cost_row as highs_sys::HighsInt,
                -f64::INFINITY,
                max_cost,
            );
        }
        self.solve_inner(incumbent, free, min_clear_area, seconds)
    }

    /// Cost-minimising solve with no cost ceiling. Every mutable bound is set
    /// on entry, so a previous lexicographic solve cannot leak its ceiling in.
    pub fn solve_move(
        &mut self,
        incumbent: &FixedBitSet,
        free: &FixedBitSet,
        min_clear_area: f64,
        seconds: f64,
    ) -> (SolveOutcome, FixedBitSet) {
        self.solve(incumbent, free, min_clear_area, f64::INFINITY, Objective::Cost, seconds)
    }

    fn solve_inner(
        &mut self,
        incumbent: &FixedBitSet,
        free: &FixedBitSet,
        min_clear_area: f64,
        seconds: f64,
    ) -> (SolveOutcome, FixedBitSet) {
        for i in 0..self.num_images {
            let (lo, hi) = if free.contains(i) {
                (0.0, 1.0)
            } else {
                let v = f64::from(u8::from(incumbent.contains(i)));
                (v, v)
            };
            // SAFETY: `i < num_images` indexes a column created in `new`, and
            // the model pointer is valid for the lifetime of `self`.
            unsafe {
                highs_sys::Highs_changeColBounds(self.ptr(), i as highs_sys::HighsInt, lo, hi);
            }
        }
        // SAFETY: `epsilon_row` is the row index returned when it was added.
        unsafe {
            highs_sys::Highs_changeRowBounds(
                self.ptr(),
                self.epsilon_row as highs_sys::HighsInt,
                min_clear_area,
                f64::INFINITY,
            );
        }

        // Warm start: hand the solver the incumbent as a primal point. Without
        // it a time-limited solve can spend its whole budget finding any
        // feasible cover, and returns a needlessly expensive one.
        if incumbent.count_ones(..) > 0 {
            self.warm.iter_mut().for_each(|v| *v = 0.0);
            let mut seen = FixedBitSet::with_capacity(self.num_elements);
            for i in incumbent.ones() {
                self.warm[i] = 1.0;
                if let Some(bs) = self.clear.get(i) {
                    seen.union_with(bs);
                }
            }
            for e in seen.ones() {
                if e < self.num_elements {
                    self.warm[self.num_images + e] = 1.0;
                }
            }
            let ptr = self.ptr();
            let warm_ptr = self.warm.as_ptr();
            // SAFETY: `warm` has one entry per column and outlives the call.
            unsafe {
                highs_sys::Highs_setSolution(
                    ptr,
                    warm_ptr,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                );
            }
        }

        let mut model = self.model.take().expect("model present");
        model.set_option("time_limit", seconds);
        model.set_option("mip_rel_gap", 0.0);
        // Areas run to ~1e9, so the default (relative) feasibility tolerance
        // accepts a slack of thousands of area units. An epsilon-constraint
        // recursion that advances the floor by one unit is then invisible to
        // the solver and the same point is returned forever.
        model.set_option("primal_feasibility_tolerance", 1e-9);
        model.set_option("mip_feasibility_tolerance", 1e-9);

        let solved = model.solve();
        let status = solved.status();
        let outcome = match status {
            HighsModelStatus::Optimal => SolveOutcome::Optimal,
            HighsModelStatus::ReachedTimeLimit
            | HighsModelStatus::ReachedIterationLimit
            | HighsModelStatus::ObjectiveBound
            | HighsModelStatus::ObjectiveTarget => SolveOutcome::Feasible,
            _ => SolveOutcome::NoSolution,
        };

        let mut out = FixedBitSet::with_capacity(self.num_images);
        if outcome != SolveOutcome::NoSolution {
            let sol = solved.get_solution();
            let cols = sol.columns();
            // A time-limited run can report a status without having stored a
            // primal solution; treat an all-zero vector as "no cover found".
            for i in 0..self.num_images {
                if cols.get(i).copied().unwrap_or(0.0) > 0.5 {
                    out.insert(i);
                }
            }
        }
        self.model = Some(Model::from(solved));
        if out.count_ones(..) == 0 {
            return (SolveOutcome::NoSolution, out);
        }
        (outcome, out)
    }

    /// Total cost of a selection, using the same coefficients the model does.
    #[must_use]
    pub fn cost_of_selection(&self, s: &FixedBitSet) -> u64 {
        s.ones().map(|i| self.costs[i] as u64).sum()
    }

    /// Area seen clearly by at least one selected image, using the same clear
    /// sets the model links `y` against.
    #[must_use]
    pub fn clear_area_of(&self, s: &FixedBitSet) -> u64 {
        let mut seen = FixedBitSet::with_capacity(self.num_elements);
        for i in s.ones() {
            if let Some(bs) = self.clear.get(i) {
                seen.union_with(bs);
            }
        }
        seen.ones().map(|e| self.areas[e] as u64).sum()
    }

    #[must_use]
    pub const fn num_images(&self) -> usize {
        self.num_images
    }

    #[must_use]
    pub const fn num_elements(&self) -> usize {
        self.num_elements
    }
}

/// Clear-coverage bitsets and per-element areas, pulled out of the objective
/// state once so the model builder and the evaluator agree on them.
#[must_use]
pub fn clear_coverage<const D: usize>(problem: &ProblemBitset<D>) -> (Vec<FixedBitSet>, Vec<u64>) {
    for obj in &problem.objectives {
        if let ObjectiveState::CloudyArea { clear_images, areas, .. } = obj {
            return (clear_images.clone(), areas.clone());
        }
    }
    (
        vec![FixedBitSet::with_capacity(problem.universe_size); problem.images.len()],
        vec![0; problem.universe_size],
    )
}
