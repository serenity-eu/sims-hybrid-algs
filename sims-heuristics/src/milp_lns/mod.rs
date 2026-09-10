//! MILP-based large-neighbourhood search over a Pareto archive.
//!
//! # Why this exists
//!
//! Pareto local search explores a `k`-bounded neighbourhood: remove up to `k`
//! images, then repair by adding any subset of size `0..=5`. On instances where
//! the optimum sits far away in decision space that neighbourhood cannot reach
//! it at all. Measured on `mexico_city_250`, every solution on the exact front
//! is 12–18 images away (7–11 removals paired with 5–9 additions) from every
//! solution PLS produces, while the move set spans a symmetric difference of at
//! most 11 — so no sequence of accepted moves crosses the gap, and PLS floors at
//! 16 images where the optimum uses 12.
//!
//! This module replaces the bounded enumeration with destroy-and-solve: free a
//! large set of images and let the MILP choose the replacement exactly. A single
//! move spans an arbitrary symmetric difference within the freed pool, which is
//! what escapes the basin.
//!
//! # Scope
//!
//! Two objectives — total cost and cloudy area. Cost is linear in the selection
//! and cloudy area linearises through the per-element clear-coverage variables
//! (see [`model`]). `MaxIncidenceAngle` would add one bounding variable, but
//! `MinResolution` is a sum of per-element minima and needs a different
//! formulation, so neither is wired up here.
pub mod balanced_box;
#[cfg(feature = "gurobi_lns")]
pub mod gurobi_model;
pub mod model;
pub mod operators;

use std::time::{Duration, Instant};

use fixedbitset::FixedBitSet;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::problem_bitset::ProblemBitset;
use model::{PersistentModel, SolveOutcome};
use operators::{AdaptiveWeights, DestroyOperator};

#[derive(Debug, Clone, Copy)]
pub struct LnsConfig {
    /// Incumbent images released per move.
    pub destroy_size: usize,
    /// Extra candidate images opened per move. Uncapped, the destroy step frees
    /// nearly every image and the subproblem degenerates into the full MILP,
    /// which cannot finish inside the per-move limit.
    pub candidate_pool: usize,
    /// Per-move solver time limit.
    pub mip_time: Duration,
    /// Budget for the whole pass.
    pub total_time: Duration,
    pub seed: u64,
    pub weight_decay: f64,
    pub weight_reward: f64,
}

impl Default for LnsConfig {
    fn default() -> Self {
        Self {
            destroy_size: 10,
            candidate_pool: 40,
            mip_time: Duration::from_secs(3),
            total_time: Duration::from_secs(60),
            seed: 42,
            weight_decay: 0.9,
            weight_reward: 1.0,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct LnsStats {
    pub moves: usize,
    pub improving: usize,
    pub optimal_solves: usize,
    pub timed_out_solves: usize,
    pub failed_solves: usize,
}

pub struct MilpLns<'a, const D: usize> {
    problem: &'a ProblemBitset<D>,
    model: PersistentModel,
    clear: Vec<FixedBitSet>,
    areas: Vec<u64>,
    total_area: u64,
    config: LnsConfig,
    weights: AdaptiveWeights,
    rng: StdRng,
    stats: LnsStats,
    scratch: Vec<usize>,
    pool: Vec<usize>,
    seen: FixedBitSet,
}

impl<'a, const D: usize> MilpLns<'a, D> {
    #[must_use]
    pub fn new(problem: &'a ProblemBitset<D>, config: LnsConfig) -> Self {
        let (clear, areas) = model::clear_coverage(problem);
        let model = PersistentModel::new(problem, &clear, &areas);
        let total_area = areas.iter().sum();
        let seen = FixedBitSet::with_capacity(problem.universe_size);
        Self {
            problem,
            model,
            clear,
            areas,
            total_area,
            weights: AdaptiveWeights::new(config.weight_decay, config.weight_reward),
            rng: StdRng::seed_from_u64(config.seed),
            config,
            stats: LnsStats::default(),
            scratch: Vec::new(),
            pool: Vec::new(),
            seen,
        }
    }

    #[must_use]
    pub fn cost_of(&self, s: &FixedBitSet) -> u64 {
        s.ones().map(|i| self.problem.image_cost(i)).sum()
    }

    /// Area seen clearly by at least one selected image.
    #[must_use]
    pub fn clear_area_of(&mut self, s: &FixedBitSet) -> u64 {
        self.seen.clear();
        for i in s.ones() {
            if let Some(bs) = self.clear.get(i) {
                self.seen.union_with(bs);
            }
        }
        self.seen.ones().map(|e| self.areas[e]).sum()
    }

    #[must_use]
    pub fn cloudy_area_of(&mut self, s: &FixedBitSet) -> u64 {
        self.total_area - self.clear_area_of(s)
    }

    /// Improve one solution in place until the neighbourhood stops paying or
    /// `deadline` passes. Cloud coverage is held at or above the starting
    /// value, so any accepted move is a genuine Pareto improvement.
    pub fn improve(&mut self, start: &FixedBitSet, deadline: Instant) -> FixedBitSet {
        let target = self.clear_area_of(start) as f64;
        self.improve_with_target(start, target, deadline)
    }

    /// As [`Self::improve`], but with the clear-area floor supplied by the
    /// caller. Driving this floor across a grid is what turns a set of
    /// independent cost descents into a spread front: inheriting the floor from
    /// each start lets many starts collapse onto the same few points.
    pub fn improve_with_target(
        &mut self,
        start: &FixedBitSet,
        target: f64,
        deadline: Instant,
    ) -> FixedBitSet {
        let mut cur = start.clone();
        let mut cur_cost = self.cost_of(&cur);

        while Instant::now() < deadline {
            let (idx, op) = self.weights.pick(&mut self.rng);
            let mut free = op.select(
                self.problem,
                &cur,
                self.config.destroy_size,
                &mut self.rng,
                &mut self.scratch,
            );
            self.open_candidates(&free);
            for &i in &self.pool {
                free.insert(i);
            }

            let secs = self
                .config
                .mip_time
                .as_secs_f64()
                .min((deadline - Instant::now()).as_secs_f64().max(0.05));
            let (outcome, next) = self.model.solve_move(&cur, &free, target, secs);
            self.stats.moves += 1;
            match outcome {
                SolveOutcome::Optimal => self.stats.optimal_solves += 1,
                SolveOutcome::Feasible => self.stats.timed_out_solves += 1,
                SolveOutcome::NoSolution => {
                    self.stats.failed_solves += 1;
                    self.weights.update(idx, false);
                    continue;
                }
            }

            let cost = self.cost_of(&next);
            let improved = cost < cur_cost;
            self.weights.update(idx, improved);
            if improved {
                cur = next;
                cur_cost = cost;
                self.stats.improving += 1;
            } else if outcome == SolveOutcome::Optimal {
                // The neighbourhood was searched exhaustively and held nothing
                // better; another draw of the same size is unlikely to differ.
                break;
            }
        }
        cur
    }

    /// Open a capped, randomly chosen set of images that can cover what the
    /// destroyed ones were covering. Without these the solver may only delete
    /// images, never substitute, which is not a neighbourhood at all.
    fn open_candidates(&mut self, destroyed: &FixedBitSet) {
        self.pool.clear();
        self.seen.clear();
        for i in destroyed.ones() {
            self.seen.union_with(&self.problem.images[i]);
        }
        for e in self.seen.ones() {
            for &i in self.problem.element_covering_images(e) {
                if !destroyed.contains(i) {
                    self.pool.push(i);
                }
            }
        }
        self.pool.sort_unstable();
        self.pool.dedup();
        let cap = self.config.candidate_pool.min(self.pool.len());
        // Partial Fisher-Yates: only the prefix we keep needs to be shuffled.
        for k in 0..cap {
            let j = self.rng.random_range(k..self.pool.len());
            self.pool.swap(k, j);
        }
        self.pool.truncate(cap);
    }

    #[must_use]
    pub const fn stats(&self) -> LnsStats {
        self.stats
    }

    #[must_use]
    pub const fn operator_weights(&self) -> &[f64; 3] {
        self.weights.weights()
    }

    #[must_use]
    pub const fn config(&self) -> LnsConfig {
        self.config
    }
}

impl<'a, const D: usize> MilpLns<'a, D> {
    /// Heuristic epsilon-constraint sweep: drive the clear-area floor across a
    /// grid and run an LNS descent at each level, seeded from the best known
    /// solution that satisfies it.
    ///
    /// Each level yields a point in a different region of the front, so the
    /// result is a spread approximation rather than a cluster. Levels get an
    /// equal share of the budget; unreachable levels simply return their seed.
    pub fn sweep(
        &mut self,
        seeds: &[FixedBitSet],
        levels: usize,
        deadline: Instant,
    ) -> Vec<FixedBitSet> {
        assert!(levels > 0, "a sweep needs at least one level");
        let mut scored: Vec<(u64, u64, FixedBitSet)> = seeds
            .iter()
            .map(|s| (self.clear_area_of(s), self.cost_of(s), s.clone()))
            .collect();
        if scored.is_empty() {
            return Vec::new();
        }
        scored.sort_unstable_by_key(|(area, cost, _)| (*area, *cost));
        let lo = scored.first().map_or(0, |(a, _, _)| *a);
        let hi = scored.last().map_or(0, |(a, _, _)| *a);

        let mut out = Vec::with_capacity(levels);
        for level in 0..levels {
            if Instant::now() >= deadline {
                break;
            }
            let frac = if levels == 1 { 0.0 } else { level as f64 / (levels - 1) as f64 };
            let target = (lo as f64).mul_add(1.0 - frac, hi as f64 * frac);
            // Cheapest seed that already meets this floor, else the one with the
            // most clear area, which the descent can then trade down from.
            let seed = scored
                .iter()
                .filter(|(area, _, _)| *area as f64 >= target)
                .min_by_key(|(_, cost, _)| *cost)
                .or_else(|| scored.last())
                .map(|(_, _, s)| s.clone())
                .expect("seeds is non-empty");
            let remaining = levels - level;
            let slice = (deadline - Instant::now()) / u32::try_from(remaining).unwrap_or(1);
            out.push(self.improve_with_target(&seed, target, Instant::now() + slice));
        }
        out
    }
}

impl<'a, const D: usize> MilpLns<'a, D> {
    /// Adaptive epsilon-constraint enumeration.
    ///
    /// A fixed grid of epsilon levels wastes solves: cost as a function of the
    /// clear-area floor is a step function, so evenly spaced levels land on the
    /// same breakpoints again and again. Instead, after finding a point with
    /// clear area `c`, raise the floor to `c + 1`; the next solve is forced to a
    /// strictly different point. With every solve proved optimal this visits
    /// each non-dominated point exactly once and terminates when the floor
    /// becomes infeasible, which is the classical Aneja & Nair recursion. Under
    /// a per-solve time limit it degrades gracefully into a heuristic that still
    /// walks the whole front.
    ///
    /// `warm` seeds the first solve; passing a heuristic front gives the solver
    /// an incumbent instead of making it find one.
    pub fn enumerate_front(
        &mut self,
        warm: &[FixedBitSet],
        deadline: Instant,
    ) -> Vec<FixedBitSet> {
        let all_free = {
            let mut f = FixedBitSet::with_capacity(self.problem.images.len());
            f.insert_range(..);
            f
        };
        let empty = FixedBitSet::with_capacity(self.problem.images.len());
        // The floor only ever rises, so the incumbent from the previous step is
        // no longer feasible; the solver restarts from its own presolve each
        // time, and `warm` only matters for the first call.
        // Index the warm front by clear area so each step can be started from a
        // point that actually satisfies its floor. The previous step's solution
        // never does: the floor is raised past it by construction.
        let mut archive: Vec<(u64, u64, FixedBitSet)> = warm
            .iter()
            .map(|s| (self.clear_area_of(s), self.cost_of(s), s.clone()))
            .collect();
        archive.sort_unstable_by_key(|(area, cost, _)| (*area, *cost));
        let mut incumbent = empty.clone();

        let mut out: Vec<FixedBitSet> = Vec::new();
        let mut floor = 0.0_f64;
        while Instant::now() < deadline {
            let secs = self
                .config
                .mip_time
                .as_secs_f64()
                .min((deadline - Instant::now()).as_secs_f64().max(0.05));
            incumbent = archive
                .iter()
                .filter(|(area, _, _)| (*area as f64) >= floor)
                .min_by_key(|(_, cost, _)| *cost)
                .map_or_else(|| empty.clone(), |(_, _, s)| s.clone());
            let (outcome, next) = self.model.solve_move(&incumbent, &all_free, floor, secs);
            self.stats.moves += 1;
            match outcome {
                SolveOutcome::Optimal => self.stats.optimal_solves += 1,
                SolveOutcome::Feasible => self.stats.timed_out_solves += 1,
                // Infeasible means the floor has passed the best achievable
                // coverage: the front is exhausted.
                SolveOutcome::NoSolution => {
                    self.stats.failed_solves += 1;
                    break;
                }
            }
            let area = self.clear_area_of(&next);
            // A time-limited solve can hand back an incumbent that predates the
            // bound change and so violates the floor. Recording it would stall
            // the recursion on a point it has already emitted, so verify the
            // constraint here rather than trusting the status alone.
            if (area as f64) < floor {
                self.stats.failed_solves += 1;
                break;
            }
            if std::env::var("LNS_DEBUG").is_ok() {
                eprintln!(
                    "floor {floor:.0} -> clear {area} cost {} |S| {} {outcome:?}",
                    self.cost_of(&next),
                    next.count_ones(..)
                );
            }
            let cost = self.cost_of(&next);
            archive.push((area, cost, next.clone()));
            archive.sort_unstable_by_key(|(a, c, _)| (*a, *c));
            out.push(next);
            floor = area as f64 + 1.0;
        }
        out
    }
}

/// Operators are listed for callers that want to report which mix was used.
#[must_use]
pub const fn operator_names() -> [&'static str; 3] {
    ["random", "worst-cost", "related"]
}

#[must_use]
pub const fn operator_kinds() -> [DestroyOperator; 3] {
    DestroyOperator::ALL
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objectives::ObjectiveType;

    const OBJ: [ObjectiveType; 2] = [ObjectiveType::TotalCost, ObjectiveType::CloudyArea];

    fn instance() -> ProblemBitset<2> {
        let path = std::path::Path::new("tests/data/lagos_nigeria_30.dzn");
        ProblemBitset::from_minizinc_datafile(path, OBJ).expect("instance loads")
    }

    fn greedy_cover(p: &ProblemBitset<2>) -> FixedBitSet {
        let mut covered = FixedBitSet::with_capacity(p.universe_size);
        let mut sel = FixedBitSet::with_capacity(p.images.len());
        while covered.count_ones(..) < p.universe_size {
            let best = (0..p.images.len())
                .filter(|i| !sel.contains(*i))
                .max_by_key(|&i| {
                    let mut g = p.images[i].clone();
                    g.difference_with(&covered);
                    g.count_ones(..)
                })
                .expect("an uncovered element must have a covering image");
            covered.union_with(&p.images[best]);
            sel.insert(best);
        }
        sel
    }

    #[test]
    fn model_reproduces_incumbent_when_nothing_is_freed() {
        let p = instance();
        let mut lns = MilpLns::new(&p, LnsConfig::default());
        let start = greedy_cover(&p);
        let target = lns.clear_area_of(&start) as f64;
        let free = FixedBitSet::with_capacity(p.images.len());
        let (outcome, out) = lns.model.solve_move(&start, &free, target, 10.0);
        assert_ne!(outcome, SolveOutcome::NoSolution);
        assert_eq!(out, start, "with no freed columns the model must return the incumbent");
    }

    #[test]
    fn improve_never_loses_cloud_coverage_and_never_raises_cost() {
        let p = instance();
        let mut lns = MilpLns::new(&p, LnsConfig { total_time: Duration::from_secs(5), ..Default::default() });
        let start = greedy_cover(&p);
        let c0 = lns.cost_of(&start);
        let a0 = lns.clear_area_of(&start);
        let out = lns.improve(&start, Instant::now() + Duration::from_secs(5));
        let mut covered = FixedBitSet::with_capacity(p.universe_size);
        for i in out.ones() {
            covered.union_with(&p.images[i]);
        }
        assert_eq!(covered.count_ones(..), p.universe_size, "result must remain a set cover");
        assert!(lns.cost_of(&out) <= c0, "cost must not increase");
        assert!(lns.clear_area_of(&out) >= a0, "clear area must not decrease");
    }

    #[test]
    fn adaptive_weights_reinforce_success_and_decay_failure() {
        let mut w = AdaptiveWeights::new(0.9, 1.0);
        let before = w.weights()[0];
        w.update(0, true);
        assert!(w.weights()[0] > before);
        let peak = w.weights()[0];
        for _ in 0..20 {
            w.update(0, false);
        }
        assert!(w.weights()[0] < peak);
        assert!(w.weights()[0] >= 0.05, "weights are floored so operators stay reachable");
    }

    #[test]
    fn destroy_operators_release_the_requested_count() {
        let p = instance();
        let start = greedy_cover(&p);
        let n = start.count_ones(..).min(4);
        let mut rng = StdRng::seed_from_u64(7);
        let mut scratch = Vec::new();
        for op in DestroyOperator::ALL {
            let d = op.select(&p, &start, n, &mut rng, &mut scratch);
            assert_eq!(d.count_ones(..), n, "{op:?} must free exactly {n} images");
            assert!(d.is_subset(&start), "{op:?} may only free selected images");
        }
    }
}

#[cfg(test)]
mod epsilon_tests {
    use super::*;
    use crate::objectives::ObjectiveType;

    #[test]
    fn raising_the_floor_forces_a_different_point() {
        let p = ProblemBitset::<2>::from_minizinc_datafile(
            std::path::Path::new("tests/data/lagos_nigeria_30.dzn"),
            [ObjectiveType::TotalCost, ObjectiveType::CloudyArea],
        )
        .expect("instance loads");
        let mut lns = MilpLns::new(&p, LnsConfig::default());
        let mut all = FixedBitSet::with_capacity(p.images.len());
        all.insert_range(..);
        let empty = FixedBitSet::with_capacity(p.images.len());

        let (o1, s1) = lns.model.solve_move(&empty, &all, 0.0, 20.0);
        assert_ne!(o1, SolveOutcome::NoSolution);
        let a1 = lns.clear_area_of(&s1);
        let c1 = lns.cost_of(&s1);

        let (o2, s2) = lns.model.solve_move(&s1, &all, a1 as f64 + 1.0, 20.0);
        assert_ne!(o2, SolveOutcome::NoSolution, "a higher floor should stay feasible here");
        let a2 = lns.clear_area_of(&s2);
        assert!(
            a2 > a1,
            "the epsilon row must bind: floor {} gave clear area {a2}, previous was {a1} (cost {c1})",
            a1 + 1
        );
    }
}
