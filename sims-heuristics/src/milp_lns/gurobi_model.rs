//! Gurobi-backed subproblem solver.
//!
//! The HiGHS backend in [`super::model`] leaves most subproblems unsolved on
//! the larger instances: on `mexico_city_250` from the hard set, 29 of 32
//! Balanced Box solves hit a 20 s limit and only 3 were proved optimal, which
//! caps the front it can produce. Gurobi closes these far faster, and a proved
//! optimum per box is what turns the method from an approximation into an
//! enumeration.
//!
//! The formulation matches [`super::model`] exactly — see its docs — so the two
//! backends are interchangeable behind [`SubproblemSolver`] and their results
//! are directly comparable.
use fixedbitset::FixedBitSet;
use grb::prelude::*;

use super::model::{Objective, SolveOutcome, SubproblemSolver};
use crate::problem_bitset::ProblemBitset;

pub struct GurobiModel {
    model: Model,
    x: Vec<Var>,
    y: Vec<Var>,
    epsilon: Constr,
    cost_cap: Constr,
    costs: Vec<f64>,
    areas: Vec<f64>,
    clear: Vec<FixedBitSet>,
    num_images: usize,
    num_elements: usize,
    objective: Option<Objective>,
}

impl GurobiModel {
    /// Build the model once for `problem`.
    ///
    /// # Errors
    /// Returns the Gurobi error if the environment cannot be created (most
    /// often a missing or rejected licence) or the model cannot be built.
    pub fn new<const D: usize>(
        problem: &ProblemBitset<D>,
        clear: &[FixedBitSet],
        areas: &[u64],
        threads: Option<i32>,
    ) -> grb::Result<Self> {
        let mut env = Env::new("")?;
        env.set(param::OutputFlag, 0)?;
        let mut model = Model::with_env("milp_lns", env)?;
        if let Some(t) = threads {
            model.set_param(param::Threads, t)?;
        }
        // Match the HiGHS backend: areas reach ~1e9, where a relative
        // feasibility tolerance would swallow a unit-sized epsilon step.
        model.set_param(param::FeasibilityTol, 1e-9)?;
        model.set_param(param::IntFeasTol, 1e-9)?;
        model.set_param(param::MIPGap, 0.0)?;

        let n = problem.images.len();
        let m = problem.universe_size;

        let x: Vec<Var> = (0..n)
            .map(|i| add_binvar!(model, name: &format!("x{i}")))
            .collect::<grb::Result<_>>()?;
        // Continuous, as in the HiGHS backend: `y` is bounded above by an
        // integral quantity and pushed up only by the epsilon row, so an
        // optimal `y` is integral whenever `x` is.
        let y: Vec<Var> = (0..m)
            .map(|e| add_ctsvar!(model, name: &format!("y{e}"), bounds: 0.0..1.0))
            .collect::<grb::Result<_>>()?;

        for e in 0..m {
            let cover: Expr = problem
                .element_covering_images(e)
                .iter()
                .map(|&i| Expr::from(x[i]))
                .fold(Expr::from(0.0), |a, b| a + b);
            model.add_constr(&format!("cover{e}"), c!(cover >= 1.0))?;
        }

        let mut clear_cols: Vec<Vec<usize>> = vec![Vec::new(); m];
        for (i, bs) in clear.iter().enumerate().take(n) {
            for e in bs.ones() {
                if e < m {
                    clear_cols[e].push(i);
                }
            }
        }
        for (e, imgs) in clear_cols.iter().enumerate() {
            let seen: Expr = imgs
                .iter()
                .map(|&i| Expr::from(x[i]))
                .fold(Expr::from(0.0), |a, b| a + b);
            model.add_constr(&format!("link{e}"), c!(seen - y[e] >= 0.0))?;
        }

        let area_expr: Expr = (0..m)
            .map(|e| y[e] * areas[e] as f64)
            .fold(Expr::from(0.0), |a, b| a + b);
        let epsilon = model.add_constr("clear_floor", c!(area_expr >= 0.0))?;

        let cost_expr: Expr = (0..n)
            .map(|i| x[i] * problem.image_cost(i) as f64)
            .fold(Expr::from(0.0), |a, b| a + b);
        let cost_cap = model.add_constr("cost_cap", c!(cost_expr <= f64::INFINITY))?;

        model.update()?;
        Ok(Self {
            model,
            x,
            y,
            epsilon,
            cost_cap,
            costs: (0..n).map(|i| problem.image_cost(i) as f64).collect(),
            areas: areas.iter().map(|&a| a as f64).collect(),
            clear: clear.to_vec(),
            num_images: n,
            num_elements: m,
            objective: None,
        })
    }

    fn set_objective(&mut self, objective: Objective) -> grb::Result<()> {
        if self.objective == Some(objective) {
            return Ok(());
        }
        let expr: Expr = match objective {
            Objective::Cost => (0..self.num_images)
                .map(|i| self.x[i] * self.costs[i])
                .fold(Expr::from(0.0), |a, b| a + b),
            // Minimising cloudy area maximises the clear area seen.
            Objective::Cloud => (0..self.num_elements)
                .map(|e| self.y[e] * -self.areas[e])
                .fold(Expr::from(0.0), |a, b| a + b),
        };
        self.model.set_objective(expr, ModelSense::Minimize)?;
        self.objective = Some(objective);
        Ok(())
    }

    fn solve_impl(
        &mut self,
        incumbent: &FixedBitSet,
        free: &FixedBitSet,
        min_clear_area: f64,
        max_cost: f64,
        objective: Objective,
        seconds: f64,
    ) -> grb::Result<(SolveOutcome, FixedBitSet)> {
        self.set_objective(objective)?;
        for i in 0..self.num_images {
            let (lo, hi) = if free.contains(i) {
                (0.0, 1.0)
            } else {
                let v = f64::from(u8::from(incumbent.contains(i)));
                (v, v)
            };
            self.model.set_obj_attr(attr::LB, &self.x[i], lo)?;
            self.model.set_obj_attr(attr::UB, &self.x[i], hi)?;
        }
        self.model.set_obj_attr(attr::RHS, &self.epsilon, min_clear_area)?;
        self.model.set_obj_attr(attr::RHS, &self.cost_cap, max_cost)?;

        // Warm start, so a time-limited solve does not spend its budget looking
        // for any feasible cover.
        if incumbent.count_ones(..) > 0 {
            for i in 0..self.num_images {
                let v = f64::from(u8::from(incumbent.contains(i)));
                self.model.set_obj_attr(attr::Start, &self.x[i], v)?;
            }
        }
        self.model.set_param(param::TimeLimit, seconds.max(0.01))?;
        self.model.update()?;
        self.model.optimize()?;

        let status = self.model.status()?;
        let outcome = match status {
            Status::Optimal => SolveOutcome::Optimal,
            Status::SubOptimal | Status::TimeLimit | Status::IterationLimit
            | Status::NodeLimit | Status::SolutionLimit | Status::Interrupted => {
                // These carry a solution only if one was found.
                if self.model.get_attr(attr::SolCount)? > 0 {
                    SolveOutcome::Feasible
                } else {
                    SolveOutcome::NoSolution
                }
            }
            _ => SolveOutcome::NoSolution,
        };

        let mut out = FixedBitSet::with_capacity(self.num_images);
        if outcome != SolveOutcome::NoSolution {
            let vals = self.model.get_obj_attr_batch(attr::X, self.x.clone())?;
            for (i, v) in vals.iter().enumerate() {
                if *v > 0.5 {
                    out.insert(i);
                }
            }
        }
        if out.count_ones(..) == 0 {
            return Ok((SolveOutcome::NoSolution, out));
        }
        Ok((outcome, out))
    }
}

impl SubproblemSolver for GurobiModel {
    fn solve(
        &mut self,
        incumbent: &FixedBitSet,
        free: &FixedBitSet,
        min_clear_area: f64,
        max_cost: f64,
        objective: Objective,
        seconds: f64,
    ) -> (SolveOutcome, FixedBitSet) {
        match self.solve_impl(incumbent, free, min_clear_area, max_cost, objective, seconds) {
            Ok(r) => r,
            Err(e) => {
                // A solver error is not a proof that the box is empty, but the
                // search has no better move than to treat it as unsolved.
                tracing::warn!("gurobi solve failed: {e}");
                (SolveOutcome::NoSolution, FixedBitSet::with_capacity(self.num_images))
            }
        }
    }

    fn num_images(&self) -> usize {
        self.num_images
    }

    fn cost_of_selection(&self, s: &FixedBitSet) -> u64 {
        s.ones().map(|i| self.costs[i] as u64).sum()
    }

    fn clear_area_of(&self, s: &FixedBitSet) -> u64 {
        let mut seen = FixedBitSet::with_capacity(self.num_elements);
        for i in s.ones() {
            if let Some(bs) = self.clear.get(i) {
                seen.union_with(bs);
            }
        }
        seen.ones().map(|e| self.areas[e] as u64).sum()
    }
}
