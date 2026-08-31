//! The rectangle search (Kirlik & Sayin, Algorithm 1).
//!
//! The criterion space, projected onto objectives `1..p`, is covered by a list
//! of rectangles. Each iteration takes the rectangle reaching furthest from the
//! ideal point, solves a two-stage model inside it, and refines the list around
//! whatever that returns. The run ends when the list is empty, at which point
//! the front is complete.
//!
//! Three properties matter here, and each comes from the structure rather than
//! from numerical tuning:
//!
//! * only nondominated points are reported -- the second stage establishes
//!   that directly, with no augmentation term whose weight has to stay above
//!   the solver's tolerance;
//! * a solve that fails costs one rectangle, not the run.
//!
//! Where the epsilon bound goes inside the chosen rectangle is a separate
//! choice, [`crate::model::Split`], and it decides what a run stopped by its
//! deadline has to show for itself. Selecting the largest rectangle first only
//! spreads the points if the list ever holds more than one rectangle, and under
//! the reference's bound it does not: at two objectives the projected space is
//! a line, each solve shaves one step off the live interval, and the search
//! walks the frontier from one end. Bisecting instead splits the interval, so
//! selection has something to choose between.

use crate::front::{Front, Solution};
use crate::model::{Bounds, Config, Problem, SolveStatus, Split};
use crate::rectangle::{Rectangle, remove_subsets, update_list};
use crate::solve::{Budget, Session, maximise, minimise, stage_one, stage_two};
use log::{debug, info, warn};

/// Outcome of a run.
pub struct Outcome {
    /// The nondominated points found. Complete iff `exhaustive`.
    pub front: Front,
    /// Solver calls issued by this crate.
    ///
    /// Excludes any work the caller did to supply [`Config::bounds`] or
    /// [`Config::seeds`].
    pub solves: usize,
    /// True when the search covered the whole criterion space rather than
    /// stopping on the deadline.
    pub exhaustive: bool,
}

/// Run the search.
///
/// Returns whatever was proven within the budget. Points from solves that hit
/// the time limit are discarded rather than reported: an unproven incumbent is
/// feasible but may be dominated, and admitting it would forfeit the guarantee
/// that is the whole point of this algorithm.
///
/// # Panics
/// If the problem has fewer than two objectives.
#[must_use]
pub fn solve(problem: &Problem, config: &Config) -> Outcome {
    assert!(
        problem.num_objectives() >= 2,
        "the search needs at least two objectives"
    );
    let budget = Budget::new(config);
    let mut front = Front::new();
    let mut solves = 0usize;
    // Built once and edited between solves; see `Session`.
    let mut session = Session::new(problem);

    let Some(bounds) = resolve_bounds(&mut session, config, &budget, &mut solves) else {
        return Outcome {
            front,
            solves,
            exhaustive: false,
        };
    };

    // The first objective is minimised in stage one, so only the others need
    // bounding: the search happens in the projection that drops objective 0.
    let reference = project(&bounds.ideal);
    // One step per objective; the projected ones are what bound the rectangles.
    let deltas: Vec<i64> = bounds
        .ideal
        .iter()
        .zip(&bounds.anti_ideal)
        .map(|(lo, hi)| {
            config
                .delta
                .unwrap_or_else(|| crate::model::scaled_delta(*lo, *hi))
        })
        .collect();
    let steps = project(&deltas);
    let ceiling: Vec<i64> = project(&bounds.anti_ideal)
        .iter()
        .zip(&steps)
        .map(|(v, d)| v + d)
        .collect();
    let weights = weights(&bounds);
    debug!("deltas {deltas:?}");

    let mut rectangles = vec![Rectangle::new(reference.clone(), ceiling.clone())];
    debug!(
        "bounds ideal={:?} anti_ideal={:?}",
        bounds.ideal, bounds.anti_ideal
    );
    debug!("initial rectangle {reference:?}..{ceiling:?}");

    // Seeds must refine the list exactly as a discovered point would. The
    // search relies on the invariant that every point in the front has already
    // been split on: a rectangle whose solve returns a known point only makes
    // progress because that point lies on its boundary, which is true only if
    // the list was refined when the point was first admitted. Adding seeds to
    // the front without refining the list leaves them strictly inside the
    // initial rectangle, and the search then re-derives one of them forever --
    // measured as 18 solves producing nothing on `lagos_nigeria_150`.
    for seed in &config.seeds {
        let projected = project(&seed.objectives);
        if front.insert(seed.clone()) {
            rectangles = update_list(rectangles, &projected);
        }
    }

    let mut exhaustive = true;

    while !rectangles.is_empty() {
        if budget.exhausted() {
            exhaustive = false;
            break;
        }

        let index = argmax_volume(&rectangles, &reference);
        let searched = rectangles[index].clone();
        let upper = searched.upper().to_vec();
        // The whole region below this rectangle's upper corner. Dropped only
        // when a solve *proves* it holds nothing -- see `give_up`.
        let spent = Rectangle::new(reference.clone(), upper.clone());
        let bound = epsilon(&searched, &steps, config.split);
        let epsilon = &bound.at;

        debug!(
            "rectangle {:?}..{:?} of {}: epsilon={epsilon:?}",
            searched.lower(),
            searched.upper(),
            rectangles.len()
        );

        solves += 1;
        let first = stage_one(&mut session, epsilon, budget.next_solve());
        let Some(values) = admit(&first.status, first.values.clone(), &mut exhaustive) else {
            give_up(
                &mut rectangles,
                &searched,
                &spent,
                &bound,
                first.status,
                "stage 1",
            );
            continue;
        };
        // The solver's own figure, not one re-derived from variable values.
        let first_optimum = first
            .objective_value
            .unwrap_or_else(|| problem.objective_values_exact(&values)[0]);
        debug!(
            "  stage 1 point {:?} optimum {first_optimum} (epsilon {epsilon:?})",
            problem.objective_values(&values)
        );

        solves += 1;
        let second = stage_two(
            &mut session,
            &weights,
            first_optimum,
            epsilon,
            budget.next_solve(),
        );
        // The second stage searches a subset of the first stage's feasible set
        // that still contains its optimum, so it cannot genuinely be
        // infeasible. If the solver says so, the model is wrong or its
        // tolerances were exceeded -- either way the region has not been
        // cleared, and the run must stop claiming to have covered it.
        if second.status == SolveStatus::Infeasible {
            warn!(
                "stage 2 infeasible on {:?}..{:?} though stage 1 found an optimum of \
                 {first_optimum}; the region is not exhausted",
                searched.lower(),
                searched.upper()
            );
            exhaustive = false;
            remove_subsets(&mut rectangles, &spent);
            continue;
        }
        let Some(values) = admit(&second.status, second.values, &mut exhaustive) else {
            give_up(
                &mut rectangles,
                &searched,
                &spent,
                &bound,
                second.status,
                "stage 2",
            );
            continue;
        };

        let objectives = problem.objective_values(&values);
        let projected = project(&objectives);
        let added = front.insert(Solution {
            objectives: objectives.clone(),
            variables: problem.named_values(&values),
            found_at: budget.elapsed_secs(),
        });
        debug!(
            "  point {objectives:?} {}",
            if added { "added" } else { "already known" }
        );
        if added {
            rectangles = update_list(rectangles, &projected);
        }
        let covered = bound.covers(projected, &steps);
        // Splitting at the region's upper corner makes it a union of list
        // rectangles, which is what lets `remove_subsets` drop it. Under
        // `Split::Sweep` that corner *is* this rectangle's upper corner, so the
        // split is a no-op and this reduces to the reference rule.
        rectangles = update_list(rectangles, covered.upper());
        remove_subsets(&mut rectangles, &covered);

        // Termination guard. By the invariant above, the rectangle just
        // searched was either split or removed, so it cannot still be present.
        // If it is, the search would select it again and re-solve it forever;
        // dropping it costs at most one region and cannot hang.
        if let Some(position) = rectangles.iter().position(|r| *r == searched) {
            debug_assert!(false, "rectangle {searched:?} made no progress");
            rectangles.remove(position);
        }
    }

    info!(
        "psbox: {} points, {solves} solves, {}",
        front.len(),
        if exhaustive {
            "exhaustive"
        } else {
            "stopped early"
        }
    );
    Outcome {
        front,
        solves,
        exhaustive,
    }
}

/// Retire a rectangle whose solve returned nothing usable.
///
/// The two reasons that happens are not equivalent, and treating them alike
/// costs the run everything it had left:
///
/// * **Infeasible** is an answer. Nothing satisfies the epsilon bound, so
///   nothing lies below this rectangle's upper corner either, and the whole
///   region can go.
/// * **Deadline** is not an answer. The solver was asked a question it did not
///   finish; the region may be full of points. Discarding it on the strength of
///   a timeout throws away unexplored space -- and since the first rectangle
///   spans the entire criterion space, the very first slow solve ended the
///   search. Measured on `tokyo_bay_225`: three solves, no points, the whole
///   budget spent. The region is instead cut in two at the bound that timed
///   out, which keeps every part of it live and makes each half a smaller,
///   easier question than the one that failed.
fn give_up(
    rectangles: &mut Vec<Rectangle>,
    searched: &Rectangle,
    spent: &Rectangle,
    bound: &Bound,
    status: SolveStatus,
    stage: &str,
) {
    if status == SolveStatus::Infeasible {
        debug!(
            "  {stage} infeasible; the region below {:?} is empty",
            spent.upper()
        );
        remove_subsets(rectangles, spent);
        return;
    }
    debug!("  {stage} {status:?}; subdividing rather than discarding the region");
    rectangles.retain(|r| r != searched);
    // A rectangle with no axis wide enough to cut is already at the search's
    // resolution; there is no smaller question left to ask of it.
    if let Some(axis) = bound.bisected {
        rectangles.extend(searched.halves(axis, bound.at[axis]));
    }
}

/// Accept a solve's values only when it proved optimality.
///
/// A rectangle whose solve was cut short is skipped rather than trusted, and
/// the run stops claiming to be exhaustive. Infeasible is different: it is a
/// definite answer, meaning the rectangle holds nothing, and costs the run
/// nothing.
///
/// That reading is only sound for the *first* stage, where infeasibility means
/// the region is genuinely empty. The caller handles second-stage infeasibility
/// separately, because there it cannot be genuine.
fn admit<T>(status: &SolveStatus, values: Option<T>, exhaustive: &mut bool) -> Option<T> {
    match status {
        SolveStatus::Optimal => values,
        SolveStatus::Infeasible => None,
        SolveStatus::Deadline => {
            *exhaustive = false;
            None
        }
    }
}

/// Drop the first coordinate: the search runs in the projected space.
fn project(point: &[i64]) -> Vec<i64> {
    point[1..].to_vec()
}

/// Place the first stage's epsilon bound inside `rectangle`.
///
/// [`Split::Sweep`] puts it one step below the upper corner on every axis, so
/// the solve returns the point nearest that corner. [`Split::Bisect`] does the
/// same on every axis but the widest, which it cuts in half instead -- widest
/// measured in steps, so that objectives on different scales compare.
///
/// Bisection needs room: an axis spanning two steps or fewer has no interior
/// midpoint to cut at, and asking for one would place the bound at or below the
/// rectangle's lower corner and search a region that is not there. Such an axis
/// falls back to the sweep bound, which is also what makes the search finish --
/// every rectangle eventually becomes too thin to bisect and is then walked out
/// in steps.
fn epsilon(rectangle: &Rectangle, steps: &[i64], split: Split) -> Bound {
    let mut at: Vec<i64> = rectangle
        .upper()
        .iter()
        .zip(steps)
        .map(|(u, d)| u - d)
        .collect();
    let mut bisected = None;
    if split == Split::Bisect
        && let Some(axis) = widest_axis(rectangle, steps)
    {
        let (lo, hi) = (rectangle.lower()[axis], rectangle.upper()[axis]);
        at[axis] = lo + (hi - lo) / 2;
        bisected = Some(axis);
    }
    Bound { at, bisected }
}

/// Where the first stage's epsilon bound was placed, and on which axis it was a
/// bisection rather than a step in from the corner.
struct Bound {
    at: Vec<i64>,
    bisected: Option<usize>,
}

impl Bound {
    /// Upper corner of the region a solve at this bound proves empty of new
    /// points, given the point it returned.
    ///
    /// The solve minimised objective 0 over everything at or below `at`, so
    /// every feasible point there costs at least as much as the one returned,
    /// and those at or above it are dominated by it. That covers `[point, at]`
    /// and no more.
    ///
    /// On a *swept* axis the region may be widened by one step, to `at + delta`
    /// -- which is the rectangle's own upper corner. That strip is narrower
    /// than the step the search resolves and is Kirlik & Sayin's stated
    /// discard. A *bisected* axis gets no such widening: there the strip would
    /// fall in the middle of the rectangle rather than off its end, and
    /// discarding it could drop genuine points from the interior of the
    /// frontier while the run still reported itself exhaustive.
    fn covers(&self, point: Vec<i64>, steps: &[i64]) -> Rectangle {
        let upper = self
            .at
            .iter()
            .zip(steps)
            .enumerate()
            .map(|(axis, (at, step))| {
                if self.bisected == Some(axis) {
                    *at
                } else {
                    at + step
                }
            })
            .collect();
        Rectangle::new(point, upper)
    }
}

/// The axis with the most steps across it, or `None` if none has room to cut.
fn widest_axis(rectangle: &Rectangle, steps: &[i64]) -> Option<usize> {
    rectangle
        .lower()
        .iter()
        .zip(rectangle.upper())
        .zip(steps)
        .map(|((lo, hi), step)| (hi - lo) / (*step).max(1))
        .enumerate()
        .filter(|(_, span)| *span > 2)
        .max_by_key(|(_, span)| *span)
        .map(|(axis, _)| axis)
}

/// Index of the rectangle reaching furthest from the ideal point.
fn argmax_volume(rectangles: &[Rectangle], reference: &[i64]) -> usize {
    rectangles
        .iter()
        .enumerate()
        .max_by_key(|(_, r)| r.volume(reference))
        .map_or(0, |(i, _)| i)
}

/// Per-objective weights for the second stage, normalising by each objective's
/// range so that no objective dominates the tie-break through scale alone.
fn weights(bounds: &Bounds) -> Vec<f64> {
    bounds
        .ideal
        .iter()
        .zip(&bounds.anti_ideal)
        .map(|(lo, hi)| 1.0 / crate::model::objective_to_f64((hi - lo).max(1)))
        .collect()
}

/// Use the caller's bounds, or compute them.
fn resolve_bounds(
    session: &mut Session,
    config: &Config,
    budget: &Budget,
    solves: &mut usize,
) -> Option<Bounds> {
    if let Some(bounds) = config.bounds.clone() {
        return Some(bounds);
    }
    compute_bounds(session, budget, solves)
}

/// Establish bounds on the criterion space by solving for each extreme.
///
/// Costs `2p` solves, charged to the deadline like any other work. `None` means
/// they could not be established, which leaves nothing for a search to do.
///
/// The upper bounds are anti-ideal values -- the maximum of each objective over
/// the whole feasible set -- which bound the nondominated set safely but not
/// tightly. A caller holding the lexicographic extremes has tighter ones for
/// free and should pass those instead.
#[must_use]
pub fn compute_bounds(
    session: &mut Session,
    budget: &Budget,
    solves: &mut usize,
) -> Option<Bounds> {
    let objectives = session.problem().num_objectives();
    let mut ideal = Vec::with_capacity(objectives);
    let mut anti_ideal = Vec::with_capacity(objectives);
    for index in 0..objectives {
        *solves += 1;
        let lo = minimise(session, index, budget.next_solve());
        let lo = (lo.status == SolveStatus::Optimal).then_some(lo.values?)?;

        *solves += 1;
        let hi = maximise(session, index, budget.next_solve());
        let hi = (hi.status == SolveStatus::Optimal).then_some(hi.values?)?;

        ideal.push(session.problem().objective_values(&lo)[index]);
        anti_ideal.push(session.problem().objective_values(&hi)[index]);
    }
    Some(Bounds { ideal, anti_ideal })
}

#[cfg(test)]
mod tests {
    use super::{argmax_volume, epsilon, project, weights, widest_axis};
    use crate::model::{Bounds, Split};
    use crate::rectangle::Rectangle;

    #[test]
    fn projection_drops_the_first_objective() {
        assert_eq!(project(&[1, 2, 3]), vec![2, 3]);
        assert_eq!(project(&[1, 2]), vec![2]);
    }

    #[test]
    fn selection_prefers_the_rectangle_reaching_furthest() {
        let rectangles = vec![
            Rectangle::new(vec![0], vec![10]),
            Rectangle::new(vec![10], vec![100]),
            Rectangle::new(vec![100], vec![110]),
        ];
        assert_eq!(argmax_volume(&rectangles, &[0]), 2, "largest upper corner");
    }

    /// Weights must not let a large-magnitude objective decide the second-stage
    /// tie-break on its own -- cost and area here differ by two orders.
    #[test]
    fn weights_normalise_by_range() {
        let bounds = Bounds {
            ideal: vec![0, 0],
            anti_ideal: vec![100, 1_000_000],
        };
        let w = weights(&bounds);
        assert!((w[0] - 0.01).abs() < 1e-12);
        assert!((w[1] - 1e-6).abs() < 1e-18);
        assert!(
            w.iter().all(|&x| x > 0.0),
            "positivity is what makes it exact"
        );
    }

    /// A degenerate range must not divide by zero.
    #[test]
    fn weights_survive_a_collapsed_range() {
        let bounds = Bounds {
            ideal: vec![7, 7],
            anti_ideal: vec![7, 7],
        };
        assert!(weights(&bounds).iter().all(|w| w.is_finite() && *w > 0.0));
    }

    /// The reference bound sits one step below the upper corner on every axis.
    #[test]
    fn the_sweep_bound_shaves_one_step_off_the_corner() {
        let r = Rectangle::new(vec![0, 0], vec![1000, 500]);
        assert_eq!(epsilon(&r, &[10, 10], Split::Sweep).at, vec![990, 490]);
    }

    /// Bisection halves the widest axis and leaves the others swept, so one
    /// solve splits the region instead of shortening it.
    #[test]
    fn bisection_halves_the_widest_axis() {
        let r = Rectangle::new(vec![0, 0], vec![1000, 500]);
        let bound = epsilon(&r, &[10, 10], Split::Bisect).at;
        assert_eq!(
            bound,
            vec![500, 490],
            "axis 0 is wider, so it is the one cut"
        );
        assert!(
            r.lower()[0] < bound[0] && bound[0] < r.upper()[0],
            "the bound must fall strictly inside, or it splits nothing"
        );
    }

    /// Width is counted in steps, so an objective with a large raw range does
    /// not win the axis choice on scale alone -- the cost and area objectives
    /// here differ by three orders in both range and step.
    #[test]
    fn the_widest_axis_is_measured_in_steps() {
        // Axis 1 spans a thousand times more ground, and loses whenever its
        // step is more than a thousand times coarser.
        let r = Rectangle::new(vec![0, 0], vec![1_000, 1_000_000]);
        assert_eq!(
            widest_axis(&r, &[1, 10_000]),
            Some(0),
            "1000 steps against 100"
        );
        assert_eq!(
            widest_axis(&r, &[10, 1_000]),
            Some(1),
            "100 steps against 1000"
        );
        // Equal spans tie, and the tie goes to the last axis. Which one wins
        // does not matter -- both cuts are valid -- but it must be decided.
        assert_eq!(widest_axis(&r, &[1, 1_000]), Some(1), "1000 steps each");
    }

    /// An axis with no room to cut has no interior midpoint, and bisecting it
    /// would put the bound on or below the lower corner. Such a rectangle falls
    /// back to the sweep -- which is also why the search terminates.
    #[test]
    fn a_rectangle_too_thin_to_bisect_falls_back_to_the_sweep() {
        let steps = [10];
        for span in 0..=20 {
            let r = Rectangle::new(vec![100], vec![100 + span]);
            let bound = epsilon(&r, &steps, Split::Bisect).at;
            if let Some(axis) = widest_axis(&r, &steps) {
                assert!(
                    r.lower()[axis] < bound[axis],
                    "span {span} bisected below its own lower corner"
                );
            } else {
                assert_eq!(bound, epsilon(&r, &steps, Split::Sweep).at, "span {span}");
            }
        }
    }

    /// Regression for the clustering that motivated bisection.
    ///
    /// Replaying the sweep's own rule on the `lagos_nigeria_150` cloud interval
    /// shows the shape of the failure: each solve returns a point one step
    /// below the live upper corner, so after eight solves the search has moved
    /// eight steps into a range of a billion. Bisection reaches the middle of
    /// that range on its first solve.
    #[test]
    fn bisection_covers_ground_the_sweep_cannot() {
        let (lo, hi) = (620_841i64, 1_428_692_042i64);
        let step = crate::model::scaled_delta(lo, hi);
        let live = Rectangle::new(vec![lo], vec![hi]);

        let swept = epsilon(&live, &[step], Split::Sweep).at[0];
        assert_eq!(hi - swept, step, "one step in from the corner");

        let bisected = epsilon(&live, &[step], Split::Bisect).at[0];
        let reach = |bound: i64| {
            crate::model::objective_to_f64(hi - bound) / crate::model::objective_to_f64(hi - lo)
        };
        assert!(
            reach(swept) < 1e-5,
            "the sweep barely moves: {}",
            reach(swept)
        );
        assert!(
            (reach(bisected) - 0.5).abs() < 1e-6,
            "bisection reaches the middle: {}",
            reach(bisected)
        );
    }

    /// Regression: a solve that ran out of time must not delete the region it
    /// failed to answer for.
    ///
    /// `spent` reaches from the ideal point to the rectangle's upper corner, so
    /// dropping it on a timeout discards everything the search had left --
    /// which, on the first iteration, is the entire criterion space. Measured
    /// on `tokyo_bay_225`: three solves, no points, the full budget spent.
    #[test]
    fn a_timeout_subdivides_the_region_instead_of_discarding_it() {
        let searched = Rectangle::new(vec![0], vec![1000]);
        let spent = Rectangle::new(vec![0], vec![1000]);
        let bound = super::Bound {
            at: vec![500],
            bisected: Some(0),
        };
        let mut list = vec![searched.clone()];

        super::give_up(
            &mut list,
            &searched,
            &spent,
            &bound,
            crate::model::SolveStatus::Deadline,
            "test",
        );

        assert_eq!(
            list,
            vec![
                Rectangle::new(vec![0], vec![500]),
                Rectangle::new(vec![500], vec![1000]),
            ],
            "both halves stay live, and each is a smaller question than the one that failed"
        );
    }

    /// Infeasible is an answer, so the region genuinely goes.
    #[test]
    fn an_infeasible_region_is_discarded() {
        let searched = Rectangle::new(vec![400], vec![1000]);
        let spent = Rectangle::new(vec![0], vec![1000]);
        let bound = super::Bound {
            at: vec![700],
            bisected: Some(0),
        };
        let mut list = vec![searched.clone(), Rectangle::new(vec![0], vec![400])];

        super::give_up(
            &mut list,
            &searched,
            &spent,
            &bound,
            crate::model::SolveStatus::Infeasible,
            "test",
        );

        assert!(list.is_empty(), "nothing lies below an infeasible bound");
    }

    /// A rectangle too thin to bisect has no smaller question left, so it is
    /// retired rather than split forever.
    #[test]
    fn a_timeout_on_an_uncuttable_region_retires_it() {
        let searched = Rectangle::new(vec![0], vec![1]);
        let bound = super::Bound {
            at: vec![0],
            bisected: None,
        };
        let mut list = vec![searched.clone()];
        super::give_up(
            &mut list,
            &searched,
            &Rectangle::new(vec![0], vec![1]),
            &bound,
            crate::model::SolveStatus::Deadline,
            "test",
        );
        assert!(list.is_empty());
    }

    /// Regression: bisection must not discard the `delta` band above its bound.
    ///
    /// The solve proves only that `[point, epsilon]` holds nothing new. Under
    /// the sweep the band above `epsilon` is the top of the rectangle and is
    /// the reference's own resolution discard; under bisection it lies in the
    /// interior, and dropping it would lose genuine points from the middle of
    /// the frontier while the run still called itself exhaustive.
    #[test]
    fn bisection_does_not_discard_the_band_above_its_bound() {
        let steps = [10];
        let point = vec![100];

        let swept = super::Bound {
            at: vec![990],
            bisected: None,
        };
        assert_eq!(
            swept.covers(point.clone(), &steps).upper(),
            &[1000],
            "the sweep may widen to the rectangle's own corner"
        );

        let bisected = super::Bound {
            at: vec![500],
            bisected: Some(0),
        };
        assert_eq!(
            bisected.covers(point, &steps).upper(),
            &[500],
            "bisection covers exactly what the solve proved"
        );
    }

    /// Regression: a seeded point must refine the rectangle list, or the search
    /// re-derives it forever.
    ///
    /// The values are the `lagos_nigeria_150` extremes that exposed this. The
    /// minimum-cost extreme projects strictly inside the initial rectangle. If
    /// it is added to the front without splitting there, the first solve
    /// returns it, the front rejects it as a duplicate, and the removal region
    /// `[point, upper]` does not contain the rectangle -- so the same rectangle
    /// is selected again. Measured as 18 solves and no new points.
    #[test]
    fn a_seed_inside_the_initial_rectangle_must_split_it() {
        let ideal = [6_400_456, 620_841];
        let anti_ideal = [31_286_476, 1_428_692_042];
        let initial = Rectangle::new(
            project(&ideal),
            project(&anti_ideal).iter().map(|v| v + 1).collect(),
        );

        let min_cost_extreme = project(&[6_400_456, 1_428_692_042]);
        assert!(
            initial.lower()[0] < min_cost_extreme[0] && min_cost_extreme[0] < initial.upper()[0],
            "the seed lies strictly inside, so it splits rather than bounds"
        );

        let refined = crate::rectangle::update_list(vec![initial], &min_cost_extreme);
        assert_eq!(
            refined.len(),
            2,
            "the seed must split the initial rectangle"
        );

        // With the list refined, the region dominated by the seed is removable,
        // which is what lets the search make progress.
        let mut refined = refined;
        crate::rectangle::remove_subsets(
            &mut refined,
            &Rectangle::new(min_cost_extreme, vec![anti_ideal[1] + 1]),
        );
        assert_eq!(refined.len(), 1, "the dominated part is exhausted");
    }

    /// The other extreme lands on the boundary and correctly splits nothing.
    #[test]
    fn a_seed_on_the_boundary_leaves_the_list_alone() {
        let initial = Rectangle::new(vec![620_841], vec![1_428_692_043]);
        let min_cloud_extreme = project(&[31_286_476, 620_841]);
        let refined = crate::rectangle::update_list(vec![initial.clone()], &min_cloud_extreme);
        assert_eq!(refined, vec![initial]);
    }
}
