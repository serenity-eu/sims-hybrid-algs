//! The Quadtree Search Method (Basu, Bhattarai, Das, Keshanian & Charkhgard,
//! 2026) for biobjective integer programs.
//!
//! A branch-and-bound over the criterion space that asks each subproblem only
//! whether a feasible point *exists* in a region, never whether a particular
//! one is optimal. That single change is what makes it robust where
//! optimality-based criterion-space methods stall: proving existence is far
//! cheaper than proving optimality, and a check stopped early still answers the
//! question whenever it found an incumbent.
//!
//! The region is a rectangle. Each iteration takes the rectangle nearest the
//! ideal corner, and either prunes it, or checks it and splits it into four.
//! A check that runs out of time leaves the region [`Feasibility::Undecided`]
//! and the region is split anyway, so no part of the criterion space is ever
//! discarded on the strength of a question the solver could not answer.
//!
//! # What a truncated run guarantees
//!
//! Two different things, and the difference matters:
//!
//! * [`Outcome::gap`] is the paper's aggregate measure: the share of the
//!   criterion space still unexplored. It bounds where the missing points can
//!   be, not whether the points in hand are nondominated.
//! * [`Outcome::certified`] is stronger and is not in the paper. A point is
//!   certified when no unexplored region can hold anything dominating it, which
//!   makes it nondominated in the full problem rather than merely nondominated
//!   among the points found so far. Callers that need proven-nondominated
//!   output -- seeding an exact method, say -- should use the certified subset;
//!   callers that want coverage should use the whole front.

use crate::front::{Front, Solution};
use crate::model::{Bounds, Feasibility, Problem, SolveStatus};
use crate::solve::{Budget, Session, feasibility_check, weighted_sum};
use log::{debug, info};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::time::Duration;

/// A rectangle in the criterion space, inclusive of both corners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    /// Corner nearest the ideal point.
    pub lower: [i64; 2],
    /// Corner furthest from it.
    pub upper: [i64; 2],
}

impl Region {
    /// Whether the rectangle contains anything at all.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.upper[0] < self.lower[0] || self.upper[1] < self.lower[1]
    }

    /// Number of integer points, as a measure of how much space is unexplored.
    #[must_use]
    pub const fn area(self) -> i128 {
        if self.is_empty() {
            return 0;
        }
        let width = (self.upper[0] - self.lower[0]) as i128 + 1;
        let height = (self.upper[1] - self.lower[1]) as i128 + 1;
        width * height
    }

    /// Does the rectangle contain this point?
    #[must_use]
    pub const fn contains(self, point: [i64; 2]) -> bool {
        self.lower[0] <= point[0]
            && point[0] <= self.upper[0]
            && self.lower[1] <= point[1]
            && point[1] <= self.upper[1]
    }

    /// Split each side at its midpoint, giving up to four disjoint rectangles
    /// whose union is exactly this one.
    ///
    /// A side that is already a single value cannot be halved, so a rectangle
    /// one unit wide yields two children and a single point yields one -- which
    /// is how a leaf is recognised.
    #[must_use]
    pub fn split(self) -> Vec<Self> {
        let halves = |lo: i64, hi: i64| -> Vec<(i64, i64)> {
            if lo >= hi {
                vec![(lo, hi)]
            } else {
                // Midpoint computed as an offset so it cannot overflow.
                let mid = lo + (hi - lo) / 2;
                vec![(lo, mid), (mid + 1, hi)]
            }
        };
        let mut children = Vec::with_capacity(4);
        for (lo0, hi0) in halves(self.lower[0], self.upper[0]) {
            for (lo1, hi1) in halves(self.lower[1], self.upper[1]) {
                children.push(Self {
                    lower: [lo0, lo1],
                    upper: [hi0, hi1],
                });
            }
        }
        children
    }

    /// A leaf holds a single criterion vector and cannot be subdivided.
    #[must_use]
    pub const fn is_leaf(self) -> bool {
        self.lower[0] == self.upper[0] && self.lower[1] == self.upper[1]
    }
}

/// Search order: nearest the ideal corner first, oldest first among equals,
/// and anything deferred after everything else.
///
/// Exploring the south-west of the criterion space first is what makes a
/// truncated run useful -- those are the regions holding good trade-offs -- and
/// it also makes dominance pruning bite sooner, since every point found there
/// prunes regions to its north-east.
///
/// "Nearest" is measured after dividing each coordinate by that objective's
/// range. Summing the raw coordinates instead makes the objective with the
/// larger numbers the only one that counts: on `lagos_nigeria_150` the cloud
/// values reach `1.4e9` against costs of `3.1e7`, so cost contributed about 2%
/// of the ordering and the search advanced along a single axis, which is the
/// behaviour this ordering exists to avoid.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Priority {
    deferred: bool,
    corner_sum: i128,
    created: u64,
}

/// Fixed-point unit for normalised coordinates.
///
/// Integer arithmetic keeps the ordering exact and total, which floating point
/// would not; `i128` leaves ample headroom above this scale.
const NORMALISED_UNIT: i128 = 1 << 30;

/// Distance from the ideal corner, with each objective on a common scale.
fn corner_distance(lower: [i64; 2], bounds: &Bounds) -> i128 {
    (0..2)
        .map(|axis| {
            let span = i128::from((bounds.anti_ideal[axis] - bounds.ideal[axis]).max(1));
            let offset = i128::from(lower[axis] - bounds.ideal[axis]);
            offset * NORMALISED_UNIT / span
        })
        .sum()
}

struct Node {
    region: Region,
    priority: Priority,
}

/// A supporting half-plane of the criterion space, learned from one optimal
/// weighted-sum solve.
///
/// Minimising `w . f` to a proven optimum of `z` establishes `w . f >= z` for
/// *every* feasible point, not just the one returned. With `w > 0` the maximum
/// of `w . f` over a rectangle is attained at its upper corner, so a region
/// whose upper corner already falls short of `z` cannot contain a feasible
/// point -- and that is settled by arithmetic, with no call to the solver.
///
/// This is the cheapest information in the whole search: each cut costs nothing
/// beyond a solve already being paid for, and every region it eliminates is a
/// solve not issued. It is the biobjective form of the idea Tamby &
/// Vanderpooten (2021) use to discard zones without solving the integer
/// programs associated with them.
#[derive(Clone, Copy, Debug)]
pub struct Cut {
    /// Strictly positive weights the sum was taken with.
    weights: [f64; 2],
    /// Proven optimal value of `w . f`.
    bound: f64,
}

impl Cut {
    /// Whether this cut proves `region` holds no feasible point.
    ///
    /// The comparison is slackened relative to the bound's own magnitude.
    /// Criterion values here reach `1e9`, so an exact `<` would let ordinary
    /// floating-point error eliminate a region that is merely on the boundary
    /// -- and a region wrongly eliminated is a region never searched.
    #[must_use]
    fn excludes(&self, region: Region) -> bool {
        if region.is_empty() {
            return true;
        }
        let best = self
            .weights
            .iter()
            .zip(region.upper)
            .map(|(w, u)| w * crate::model::objective_to_f64(u))
            .sum::<f64>();
        best < self.bound - self.bound.abs().mul_add(1e-9, 1e-6)
    }
}

/// Weights normal to the segment `a -- b`, scaled so neither objective is
/// favoured by magnitude alone.
///
/// Minimising this sum searches strictly below the segment: the two endpoints
/// score equally, and any point scoring less lies on the far side of the line
/// through them. Both components are strictly positive whenever `a` and `b` are
/// mutually nondominated, which is what makes the resulting optimum
/// nondominated rather than merely weakly so.
fn separating_weights(a: [i64; 2], b: [i64; 2], bounds: &Bounds) -> Option<[f64; 2]> {
    let span = |axis: usize| {
        crate::model::objective_to_f64((bounds.anti_ideal[axis] - bounds.ideal[axis]).max(1))
    };
    // Normalised, so the normal is not dominated by whichever objective carries
    // the larger numbers, then divided back out to apply to raw values.
    let rise = crate::model::objective_to_f64(a[1] - b[1]) / span(1);
    let run = crate::model::objective_to_f64(b[0] - a[0]) / span(0);
    if rise <= 0.0 || run <= 0.0 {
        return None;
    }
    Some([rise / span(0), run / span(1)])
}

/// Configuration.
pub struct Config {
    /// Wall-clock budget for the whole run.
    pub deadline: Option<Duration>,
    /// Time limit for a single feasibility check.
    ///
    /// The paper's `tau`. A check that exceeds it leaves its region undecided,
    /// which costs a subdivision rather than the region, so this can be set
    /// aggressively low without risking correctness.
    pub node_timeout: Option<Duration>,
    /// Bounds on the criterion space. Computed by the caller, since it usually
    /// already has them.
    pub bounds: Bounds,
    /// Feasible points known in advance, used to prune from the first
    /// iteration.
    ///
    /// Lexicographic extremes or a heuristic front. Any feasible points serve;
    /// they need not be optimal.
    pub seeds: Vec<Solution>,
    /// How many weighted-sum scalarisations to solve before searching.
    ///
    /// The paper's warm start, and it is not optional in practice. Dominance
    /// pruning is what makes this search tractable, and it needs points to
    /// prune with: from the two extremes alone the criterion space is
    /// subdivided blindly, which on these instances leaves essentially all of
    /// it unexplored. Zero disables it.
    pub warm_start: usize,
    /// Time limit for one warm-start solve.
    ///
    /// Deliberately tight on a first attempt. A segment whose solve does not
    /// finish is retried once with whatever budget is left, since by then the
    /// alternative is abandoning that stretch of frontier.
    pub warm_start_timeout: Option<Duration>,
    /// Share of the deadline the warm start may consume before handing over.
    ///
    /// Every dichotomic solve is productive, so nothing else stops it: without
    /// a share of its own it spends the whole budget and the region search --
    /// which is what applies the bounds those solves prove -- never runs.
    pub warm_start_share: f64,
}

/// Outcome of a run.
pub struct Outcome {
    /// Mutually nondominated feasible points found.
    pub front: Front,
    /// Whether each point of `front` is proven nondominated in the full
    /// problem. Aligned with `front.solutions()`.
    ///
    /// True when *either* the point came from a solve that proved it -- an
    /// optimal, strictly positively weighted sum cannot return a dominated
    /// point -- or no unexplored region can hold anything dominating it. The
    /// first is much the commoner route, and unlike the second it does not
    /// depend on the search having closed off the criterion space.
    pub certified: Vec<bool>,
    /// Regions retired by a proven bound instead of a solve.
    pub eliminated: usize,
    /// Feasibility checks issued.
    pub checks: usize,
    /// True when the whole criterion space was explored.
    pub exhaustive: bool,
    /// Percentage of the criterion space left unexplored, the paper's `Delta`.
    pub gap: f64,
}

/// Run the search.
///
/// # Panics
/// If the problem does not have exactly two objectives.
#[must_use]
pub fn solve(problem: &Problem, config: &Config) -> Outcome {
    assert!(
        problem.num_objectives() == 2,
        "the quadtree search is biobjective"
    );
    let budget = Budget::new_with(config.deadline, config.node_timeout);
    // Built once and edited between solves; see `Session`.
    let mut session = Session::new(problem);
    let mut front = Front::new();
    let mut checks = 0usize;

    let root = Region {
        lower: [config.bounds.ideal[0], config.bounds.ideal[1]],
        upper: [config.bounds.anti_ideal[0], config.bounds.anti_ideal[1]],
    };
    let total_area = root.area();

    for seed in &config.seeds {
        front.insert(seed.clone());
    }
    // Seeds are the caller's lexicographic extremes: optimal for their own
    // objective, hence nondominated, hence proven.
    let mut proven: HashSet<[i64; 2]> = front
        .solutions()
        .iter()
        .map(|s| [s.objectives[0], s.objectives[1]])
        .collect();
    let mut cuts: Vec<Cut> = Vec::new();
    checks += warm_start(
        &mut session,
        config,
        &budget,
        &mut front,
        &mut cuts,
        &mut proven,
    );

    let mut created = 0u64;
    let mut queue = BinaryHeap::new();
    let mut push = |queue: &mut BinaryHeap<Reverse<Node>>, region: Region, deferred: bool| {
        if region.is_empty() {
            return;
        }
        created += 1;
        queue.push(Reverse(Node {
            region,
            priority: Priority {
                deferred,
                corner_sum: corner_distance(region.lower, &config.bounds),
                created,
            },
        }));
    };
    push(&mut queue, root, false);

    let mut exhaustive = true;
    let mut eliminated = 0usize;
    while let Some(Reverse(node)) = queue.pop() {
        if budget.exhausted() {
            exhaustive = false;
            queue.push(Reverse(node));
            break;
        }
        let Some(region) = tighten(node.region, &front) else {
            continue;
        };
        // Settled by arithmetic against bounds already paid for.
        if cuts.iter().any(|cut| cut.excludes(region)) {
            eliminated += 1;
            continue;
        }

        // A point already known inside the region answers the question without
        // a solve.
        if front
            .solutions()
            .iter()
            .any(|s| region.contains([s.objectives[0], s.objectives[1]]))
        {
            for child in region.split() {
                push(&mut queue, child, false);
            }
            continue;
        }

        let timeout = if node.priority.deferred {
            budget.remaining()
        } else {
            budget.next_solve()
        };
        checks += 1;
        let (status, values) =
            feasibility_check(&mut session, &region.lower, &region.upper, timeout);
        debug!(
            "region {:?}..{:?} -> {status:?}",
            region.lower, region.upper
        );

        match status {
            // Nothing here; the whole region goes.
            Feasibility::Infeasible => {}
            Feasibility::Feasible => {
                if let Some(values) = values {
                    let objectives = problem.objective_values(&values);
                    front.insert(Solution {
                        objectives,
                        variables: problem.named_values(&values),
                        found_at: budget.elapsed_secs(),
                    });
                }
                for child in region.split() {
                    push(&mut queue, child, false);
                }
            }
            // Unanswered. Subdivide rather than discard: the region may hold
            // points, and the smaller pieces are easier questions. A leaf
            // cannot be subdivided, so it is deferred to the end of the queue
            // and retried there without a time limit.
            Feasibility::Undecided => {
                exhaustive = false;
                if region.is_leaf() {
                    if !node.priority.deferred {
                        push(&mut queue, region, true);
                    }
                } else {
                    for child in region.split() {
                        push(&mut queue, child, false);
                    }
                }
            }
        }
    }

    let open: Vec<Region> = queue.iter().map(|Reverse(n)| n.region).collect();
    let certified: Vec<bool> = certify(&front, &open, &config.bounds)
        .into_iter()
        .zip(front.solutions())
        .map(|(closed, s)| closed || proven.contains(&[s.objectives[0], s.objectives[1]]))
        .collect();
    let unexplored: i128 = open.iter().map(|r| r.area()).sum();
    let gap = if total_area > 0 {
        100.0 * (unexplored as f64) / (total_area as f64)
    } else {
        0.0
    };

    info!(
        "quadtree: {} points ({} proven), {checks} checks, {eliminated} regions retired \
         by bound, {}, gap {gap:.2}%",
        front.len(),
        certified.iter().filter(|c| **c).count(),
        if exhaustive {
            "exhaustive"
        } else {
            "stopped early"
        }
    );
    Outcome {
        front,
        certified,
        eliminated,
        checks,
        exhaustive,
        gap,
    }
}

/// Seed the front by dichotomic weighted-sum search, and collect the
/// half-planes those solves prove along the way.
///
/// Replaces a uniform sweep of weights from one objective to the other. A
/// uniform sweep has no idea where the front is: it spends solves on weights
/// that return points it already holds, and leaves whole stretches untouched
/// because equal steps in *weight* are not equal steps along the frontier. The
/// dichotomic construction (Aneja & Nair) instead takes the weights normal to
/// the segment between two adjacent known points, which searches exactly the
/// region between them -- so every solve either finds a genuinely new point or
/// proves that segment is an edge of the frontier and can be retired. Neither
/// outcome is wasted.
///
/// Segments are taken largest-first, measured by the area they could still be
/// hiding, so a run cut short has spread its points across the frontier rather
/// than refined one end of it.
///
/// Only proven optima are kept. That is not caution for its own sake: with
/// strictly positive weights an optimum is nondominated, so these points need
/// no later certification, while an unproven incumbent is merely feasible and
/// may well be dominated -- reporting those measured as four dominated points
/// out of seven on `tokyo_bay_225`.
///
/// Returns the solves issued; found points go to `front` and proven bounds to
/// `cuts`.
fn warm_start(
    session: &mut Session,
    config: &Config,
    budget: &Budget,
    front: &mut Front,
    cuts: &mut Vec<Cut>,
    proven: &mut HashSet<[i64; 2]>,
) -> usize {
    if config.warm_start == 0 {
        return 0;
    }
    let deadline = warm_start_deadline(config, budget);
    let span = |axis: usize| {
        crate::model::objective_to_f64(
            (config.bounds.anti_ideal[axis] - config.bounds.ideal[axis]).max(1),
        )
    };
    // How much frontier a segment could still be hiding, normalised: the
    // triangle between its endpoints and the corner they bracket.
    let potential = |s: &Segment| {
        (crate::model::objective_to_f64(s.b[0] - s.a[0]) / span(0))
            * (crate::model::objective_to_f64(s.a[1] - s.b[1]) / span(1))
    };

    let mut solves = 0;
    let mut segments: Vec<Segment> = extremes(front)
        .into_iter()
        .map(|(a, b)| Segment {
            a,
            b,
            retried: false,
        })
        .collect();

    while solves < config.warm_start && !segments.is_empty() {
        if budget.exhausted() || deadline.is_some_and(|d| budget.elapsed_secs() >= d) {
            break;
        }
        // Largest potential first, so a run cut short is spread rather than
        // locally refined. Deferred segments rank below fresh ones of equal
        // promise only by having already consumed a solve.
        let index = segments
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| {
                potential(x)
                    .partial_cmp(&potential(y))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(i, _)| i);
        let segment = segments.swap_remove(index);
        let Some(weights) = separating_weights(segment.a, segment.b, &config.bounds) else {
            continue;
        };

        // A first attempt is held to the per-solve cap; a retry gets whatever
        // is left, because by then the alternative is abandoning the segment.
        let timeout = if segment.retried {
            budget.remaining()
        } else {
            match (config.warm_start_timeout, budget.remaining()) {
                (Some(cap), Some(left)) => Some(cap.min(left)),
                (Some(cap), None) => Some(cap),
                (None, left) => left,
            }
        };
        solves += 1;
        let result = weighted_sum(session, &weights, timeout);
        let point = result.values.as_ref().map(|values| {
            let objectives = session.problem().objective_values(values);
            [objectives[0], objectives[1]]
        });

        // A solve that proved nothing has not shown the segment is an edge. It
        // holds whatever it held before, so retiring it on that basis discards
        // frontier -- and with a single starting segment, discards the entire
        // warm start. Measured on `tokyo_bay_225`: one solve, then nothing.
        if result.status != SolveStatus::Optimal {
            debug!(
                "warm start {:?}--{:?}: {:?}",
                segment.a, segment.b, result.status
            );
            match (point, segment.retried) {
                // An incumbent still splits the segment into two easier
                // questions, even though it is not proven nondominated.
                (Some(p), _) if strictly_between(p, segment.a, segment.b) => {
                    if let Some(values) = result.values {
                        let objectives = session.problem().objective_values(&values);
                        front.insert(Solution {
                            objectives,
                            variables: session.problem().named_values(&values),
                            found_at: budget.elapsed_secs(),
                        });
                    }
                    segments.push(Segment::new(segment.a, p));
                    segments.push(Segment::new(p, segment.b));
                }
                (_, false) => segments.push(segment.retry()),
                (_, true) => (),
            }
            continue;
        }
        let Some(values) = result.values else {
            continue;
        };

        // The solve proves this bound for the whole feasible set, not just for
        // the point it returned.
        if let Some(bound) = result.objective_value {
            cuts.push(Cut { weights, bound });
        }

        let Some(point) = point else { continue };
        // On the segment itself: it is an edge of the frontier, and there is
        // nothing strictly between its endpoints to find.
        if !strictly_between(point, segment.a, segment.b) {
            debug!(
                "warm start {:?}--{:?}: edge of the frontier",
                segment.a, segment.b
            );
            continue;
        }
        let objectives = session.problem().objective_values(&values);
        front.insert(Solution {
            objectives,
            variables: session.problem().named_values(&values),
            found_at: budget.elapsed_secs(),
        });
        // Optimal under strictly positive weights, hence nondominated.
        proven.insert(point);
        debug!(
            "warm start {:?}--{:?}: found {point:?}",
            segment.a, segment.b
        );
        segments.push(Segment::new(segment.a, point));
        segments.push(Segment::new(point, segment.b));
    }
    info!(
        "warm start: {} points and {} bounds from {solves} solves",
        front.len(),
        cuts.len()
    );
    solves
}

/// When the warm start must hand the budget over to the region search.
///
/// Without a share of its own the warm start can spend the entire deadline:
/// every solve it makes is productive, so nothing stops it, and the search that
/// is supposed to use the bounds it proves never runs a single iteration.
/// Measured on `lagos_nigeria_150`: twenty-two solves, twenty-two checks, no
/// region ever examined and none of the nineteen proven bounds ever applied.
fn warm_start_deadline(config: &Config, budget: &Budget) -> Option<f64> {
    let total = config.deadline?.as_secs_f64();
    Some(budget.elapsed_secs() + total * config.warm_start_share)
}

/// A stretch of frontier between two known points, still to be searched.
#[derive(Clone, Copy, Debug)]
struct Segment {
    a: [i64; 2],
    b: [i64; 2],
    /// Whether a solve on it has already come back without an answer.
    retried: bool,
}

impl Segment {
    const fn new(a: [i64; 2], b: [i64; 2]) -> Self {
        Self {
            a,
            b,
            retried: false,
        }
    }

    const fn retry(self) -> Self {
        Self {
            retried: true,
            ..self
        }
    }
}

/// Whether `p` lies strictly inside the box the segment brackets.
///
/// A point on either endpoint means the segment is an edge of the frontier with
/// nothing between its ends; one outside means the solve answered a different
/// question than the one asked.
const fn strictly_between(p: [i64; 2], a: [i64; 2], b: [i64; 2]) -> bool {
    a[0] < p[0] && p[0] < b[0] && b[1] < p[1] && p[1] < a[1]
}

/// The two ends of the known front, as one segment to start the search from.
///
/// `None` when fewer than two distinct points are known, which leaves the
/// dichotomic construction nothing to bisect.
fn extremes(front: &Front) -> Option<([i64; 2], [i64; 2])> {
    let points: Vec<[i64; 2]> = front
        .solutions()
        .iter()
        .map(|s| [s.objectives[0], s.objectives[1]])
        .collect();
    let best = points.iter().min_by_key(|p| (p[0], p[1])).copied();
    let other = points.iter().max_by_key(|p| (p[0], p[1])).copied();
    match (best, other) {
        (Some(a), Some(b)) if a != b => Some((a, b)),
        _ => None,
    }
}

/// Prune and shrink a region against the points already found.
///
/// Returns `None` when nothing new can live there. Three reductions, all from
/// the paper:
///
/// * a region whose ideal corner is already dominated holds only dominated
///   points;
/// * a point to the region's west caps how large the second objective may be
///   inside it, since anything larger is dominated by that point;
/// * symmetrically for a point to its south and the first objective.
fn tighten(mut region: Region, front: &Front) -> Option<Region> {
    if region.is_empty() {
        return None;
    }
    for point in front.solutions() {
        let y = [point.objectives[0], point.objectives[1]];
        // Dominance pruning: y dominates every point of the region.
        if y[0] <= region.lower[0] && y[1] <= region.lower[1] {
            return None;
        }
        // A neighbour to the west: inside the region only points strictly
        // better than it in the second objective can be nondominated.
        if y[0] <= region.lower[0] && region.lower[1] < y[1] && y[1] <= region.upper[1] {
            region.upper[1] = y[1] - 1;
        }
        // A neighbour to the south, restricting the first objective.
        if y[1] <= region.lower[1] && region.lower[0] < y[0] && y[0] <= region.upper[0] {
            region.upper[0] = y[0] - 1;
        }
    }
    (!region.is_empty()).then_some(region)
}

/// Which points are proven nondominated in the full problem.
///
/// A point can only be dominated by something in the box between the ideal
/// corner and itself. If no unexplored region meets that box, every point in it
/// has been accounted for, and the point is nondominated -- not merely
/// nondominated among what has been found.
///
/// This is what lets a run stopped by its deadline still hand back proven
/// output. It is stronger than the paper's aggregate gap, which says where the
/// missing points are but nothing about the ones in hand.
fn certify(front: &Front, open: &[Region], bounds: &Bounds) -> Vec<bool> {
    front
        .solutions()
        .iter()
        .map(|point| {
            let dominating = Region {
                lower: [bounds.ideal[0], bounds.ideal[1]],
                upper: [point.objectives[0], point.objectives[1]],
            };
            !open.iter().any(|region| overlaps(*region, dominating))
        })
        .collect()
}

/// Do two rectangles share any point?
fn overlaps(a: Region, b: Region) -> bool {
    !a.is_empty()
        && !b.is_empty()
        && a.lower[0] <= b.upper[0]
        && b.lower[0] <= a.upper[0]
        && a.lower[1] <= b.upper[1]
        && b.lower[1] <= a.upper[1]
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl Eq for Node {}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority)
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, Priority, Region, certify, corner_distance, overlaps, tighten};
    use crate::front::{Front, Solution};
    use crate::model::Bounds;
    use crate::solve::Budget;
    use std::collections::HashMap;

    fn region(lower: [i64; 2], upper: [i64; 2]) -> Region {
        Region { lower, upper }
    }

    fn point(objectives: [i64; 2]) -> Solution {
        Solution {
            objectives: objectives.to_vec(),
            variables: HashMap::new(),
            found_at: 0.0,
        }
    }

    fn front_of(points: &[[i64; 2]]) -> Front {
        let mut front = Front::new();
        for p in points {
            front.insert(point(*p));
        }
        front
    }

    #[test]
    fn splitting_partitions_the_region_exactly() {
        let parent = region([0, 0], [7, 7]);
        let children = parent.split();
        assert_eq!(children.len(), 4);
        // Areas sum to the parent's, which can only happen if the children are
        // disjoint and cover it.
        assert_eq!(
            children.iter().map(|c| c.area()).sum::<i128>(),
            parent.area()
        );
        for child in &children {
            assert!(overlaps(*child, parent));
        }
        // And no two children share a point.
        for (i, a) in children.iter().enumerate() {
            for b in &children[i + 1..] {
                assert!(!overlaps(*a, *b), "{a:?} and {b:?} overlap");
            }
        }
    }

    #[test]
    fn a_thin_region_splits_into_two_and_a_point_into_one() {
        assert_eq!(region([0, 5], [7, 5]).split().len(), 2);
        assert_eq!(region([3, 5], [3, 5]).split().len(), 1);
        assert!(region([3, 5], [3, 5]).is_leaf());
        assert!(!region([3, 5], [4, 5]).is_leaf());
    }

    /// Splitting must terminate: every child of a non-leaf is strictly smaller.
    #[test]
    fn splitting_always_shrinks() {
        let parent = region([0, 0], [1_000_000_000, 1_000_000_000]);
        for child in parent.split() {
            assert!(child.area() < parent.area());
        }
    }

    #[test]
    fn a_region_whose_ideal_corner_is_dominated_is_pruned() {
        let front = front_of(&[[10, 10]]);
        assert!(tighten(region([20, 20], [30, 30]), &front).is_none());
        assert!(tighten(region([10, 10], [30, 30]), &front).is_none());
        // Not dominated: better in the first objective.
        assert!(tighten(region([5, 20], [30, 30]), &front).is_some());
    }

    /// A point to the west caps the second objective inside the region.
    #[test]
    fn a_western_neighbour_tightens_the_second_objective() {
        let front = front_of(&[[5, 50]]);
        let tightened = tighten(region([10, 20], [100, 100]), &front).unwrap();
        assert_eq!(tightened.upper[1], 49, "must strictly improve on 50");
        assert_eq!(tightened.upper[0], 100, "the first objective is untouched");
    }

    /// And a point to the south caps the first.
    #[test]
    fn a_southern_neighbour_tightens_the_first_objective() {
        let front = front_of(&[[50, 5]]);
        let tightened = tighten(region([20, 10], [100, 100]), &front).unwrap();
        assert_eq!(tightened.upper[0], 49);
        assert_eq!(tightened.upper[1], 100);
    }

    /// Tightening can flatten a region to a line but never empty it: the
    /// western and southern rules only fire when the neighbour sits strictly
    /// above the region's own lower corner, so the cut always lands at or above
    /// it. Emptiness comes from dominance pruning alone.
    #[test]
    fn tightening_flattens_but_only_dominance_empties() {
        let front = front_of(&[[5, 21]]);
        let flattened = tighten(region([10, 20], [100, 100]), &front).unwrap();
        assert_eq!(flattened.lower[1], 20);
        assert_eq!(flattened.upper[1], 20, "collapsed to a line, still live");
        assert_eq!(flattened.area(), 91);

        // Move the neighbour one unit down and it dominates the corner instead.
        let front = front_of(&[[5, 20]]);
        assert!(tighten(region([10, 20], [100, 100]), &front).is_none());
    }

    /// A proven optimum bounds the whole criterion space, so regions the search
    /// has never looked at can be retired by arithmetic alone.
    #[test]
    fn a_cut_retires_regions_below_its_bound() {
        // min 1*f0 + 1*f1 was proven to be 100.
        let cut = super::Cut {
            weights: [1.0, 1.0],
            bound: 100.0,
        };
        assert!(
            cut.excludes(region([0, 0], [40, 40])),
            "upper corner scores 80 < 100, so nothing feasible lives here"
        );
        assert!(
            !cut.excludes(region([0, 0], [60, 60])),
            "upper corner scores 120, so the region may hold points"
        );
        assert!(
            !cut.excludes(region([90, 90], [200, 200])),
            "wholly above the bound"
        );
    }

    /// A region touching the bound must survive: eliminating it would discard
    /// the very point the solve returned.
    #[test]
    fn a_cut_keeps_the_region_it_was_derived_from() {
        let cut = super::Cut {
            weights: [1.0, 1.0],
            bound: 100.0,
        };
        assert!(
            !cut.excludes(region([50, 50], [50, 50])),
            "scores exactly 100"
        );
        // And the slack must absorb ordinary floating-point error at the
        // magnitudes these criterion values reach.
        let big = super::Cut {
            weights: [1.0, 1.0],
            bound: 2.000_000_000_1e9,
        };
        assert!(!big.excludes(region([0, 0], [1_000_000_000, 1_000_000_000])));
    }

    /// The separating weights must be strictly positive -- that is what makes
    /// the optimum they produce nondominated rather than weakly nondominated --
    /// and must not be decided by whichever objective carries larger numbers.
    #[test]
    fn separating_weights_are_positive_and_scaled() {
        let bounds = Bounds {
            ideal: vec![6_400_456, 620_841],
            anti_ideal: vec![31_286_476, 1_428_692_042],
        };
        let a = [bounds.ideal[0], bounds.anti_ideal[1]];
        let b = [bounds.anti_ideal[0], bounds.ideal[1]];
        let w = super::separating_weights(a, b, &bounds).expect("a mutually nondominated pair");
        assert!(w[0] > 0.0 && w[1] > 0.0);
        // Both endpoints score equally: the sum searches strictly between them.
        let score = |p: [i64; 2]| w[0] * (p[0] as f64) + w[1] * (p[1] as f64);
        assert!((score(a) - score(b)).abs() < 1e-9 * score(a).abs());
    }

    /// A dominated or duplicate pair has no region between it to search.
    #[test]
    fn separating_weights_reject_a_degenerate_pair() {
        let bounds = Bounds {
            ideal: vec![0, 0],
            anti_ideal: vec![100, 100],
        };
        assert!(super::separating_weights([10, 10], [10, 10], &bounds).is_none());
        assert!(
            super::separating_weights([10, 10], [20, 20], &bounds).is_none(),
            "the second is dominated, so nothing lies between them"
        );
    }

    /// The dichotomic construction needs two distinct extremes to bisect.
    #[test]
    fn extremes_need_two_distinct_points() {
        assert!(super::extremes(&front_of(&[])).is_none(), "empty");
        assert!(
            super::extremes(&front_of(&[[10, 90]])).is_none(),
            "one point"
        );
        let front = front_of(&[[10, 90], [90, 10]]);
        let (a, b) = super::extremes(&front).expect("two extremes");
        assert_eq!((a, b), ([10, 90], [90, 10]), "cheapest first");
    }

    /// Regression: a solve that proved nothing must not retire the segment.
    ///
    /// It has not shown the segment is an edge -- the stretch still holds
    /// whatever it held before. With a single starting segment, retiring it
    /// ends the entire warm start after one solve, which is what happened on
    /// `tokyo_bay_225`: one solve, two points, nothing further attempted.
    #[test]
    fn a_segment_is_retried_before_it_is_retired() {
        let s = super::Segment::new([0, 100], [100, 0]);
        assert!(!s.retried, "a fresh segment gets the tight per-solve cap");
        assert!(s.retry().retried, "a retry gets whatever budget is left");
        assert_eq!(
            (s.retry().a, s.retry().b),
            (s.a, s.b),
            "a retry searches the same stretch"
        );
    }

    /// A point on either endpoint means the segment is an edge with nothing
    /// between its ends; one outside means a different question was answered.
    #[test]
    fn strictly_between_rejects_endpoints_and_outsiders() {
        let (a, b) = ([0, 100], [100, 0]);
        assert!(super::strictly_between([50, 50], a, b));
        assert!(!super::strictly_between(a, a, b), "on an endpoint");
        assert!(!super::strictly_between(b, a, b), "on the other endpoint");
        assert!(!super::strictly_between([150, 50], a, b), "outside");
        assert!(
            !super::strictly_between([50, 150], a, b),
            "outside on the other axis"
        );
    }

    /// The warm start must hand the budget over, or the region search that
    /// applies its bounds never runs at all.
    #[test]
    fn the_warm_start_gives_up_a_share_of_the_deadline() {
        let mut config = config_with_share(0.6);
        let budget = Budget::new_with(config.deadline, None);
        let deadline = super::warm_start_deadline(&config, &budget).expect("a bounded run");
        assert!(
            (deadline - 120.0).abs() < 1.0,
            "60% of 200 seconds, from the start: {deadline}"
        );
        config.warm_start_share = 1.0;
        assert!(
            super::warm_start_deadline(&config, &budget).expect("bounded") > 199.0,
            "a full share is allowed, it just must be chosen"
        );
        config.deadline = None;
        assert!(
            super::warm_start_deadline(&config, &budget).is_none(),
            "an unbounded run has no share to compute"
        );
    }

    fn config_with_share(share: f64) -> Config {
        Config {
            deadline: Some(std::time::Duration::from_secs(200)),
            node_timeout: None,
            bounds: Bounds {
                ideal: vec![0, 0],
                anti_ideal: vec![100, 100],
            },
            seeds: Vec::new(),
            warm_start: 8,
            warm_start_timeout: None,
            warm_start_share: share,
        }
    }

    /// Regression: the ordering must not be decided by whichever objective
    /// happens to carry the larger numbers.
    ///
    /// These are the `lagos_nigeria_150` ranges. Region `cheap` sits at the
    /// worst cost and the best cloud; `clean` is its mirror. They are equally
    /// far from the ideal corner in normalised terms, and a raw coordinate sum
    /// would rank `cheap` -- the one that gives up everything on cost -- as
    /// almost 45 times the better prospect.
    #[test]
    fn the_search_order_weighs_both_objectives_equally() {
        let bounds = Bounds {
            ideal: vec![6_400_456, 620_841],
            anti_ideal: vec![31_286_476, 1_428_692_042],
        };
        let cheap = [bounds.anti_ideal[0], bounds.ideal[1]];
        let clean = [bounds.ideal[0], bounds.anti_ideal[1]];

        let raw = |c: [i64; 2]| i128::from(c[0]) + i128::from(c[1]);
        assert!(
            raw(cheap) * 40 < raw(clean),
            "a raw sum is decided by the cloud objective alone"
        );

        assert_eq!(
            corner_distance(cheap, &bounds),
            corner_distance(clean, &bounds),
            "normalised, giving up all of one objective costs the same either way"
        );
        assert_eq!(
            corner_distance([bounds.ideal[0], bounds.ideal[1]], &bounds),
            0
        );
    }

    /// A degenerate range must not divide by zero.
    #[test]
    fn corner_distance_survives_a_collapsed_range() {
        let bounds = Bounds {
            ideal: vec![7, 7],
            anti_ideal: vec![7, 7],
        };
        assert_eq!(corner_distance([7, 7], &bounds), 0);
    }

    /// Nearest the ideal corner first; deferred nodes last regardless.
    #[test]
    fn search_order_prefers_the_ideal_corner_and_defers_last() {
        let near = Priority {
            deferred: false,
            corner_sum: 10,
            created: 5,
        };
        let far = Priority {
            deferred: false,
            corner_sum: 99,
            created: 1,
        };
        let older = Priority {
            deferred: false,
            corner_sum: 10,
            created: 1,
        };
        let deferred = Priority {
            deferred: true,
            corner_sum: 0,
            created: 0,
        };

        assert!(near < far, "smaller corner sum wins");
        assert!(older < near, "ties break by creation order");
        assert!(near < deferred, "a deferred node sorts after everything");
        assert!(far < deferred);
    }

    /// A point is certified only when nothing unexplored could dominate it.
    #[test]
    fn certification_requires_the_dominating_box_to_be_explored() {
        let bounds = Bounds {
            ideal: vec![0, 0],
            anti_ideal: vec![100, 100],
        };
        let front = front_of(&[[40, 40]]);

        // An open region to the north-east cannot dominate it.
        let certified = certify(&front, &[region([60, 60], [100, 100])], &bounds);
        assert_eq!(certified, vec![true]);

        // One to the south-west can, so the point is not proven.
        let certified = certify(&front, &[region([10, 10], [20, 20])], &bounds);
        assert_eq!(certified, vec![false]);

        // Nothing open at all: everything is proven.
        assert_eq!(certify(&front, &[], &bounds), vec![true]);
    }

    /// Overlapping the dominating box only partly is still enough to withhold
    /// certification -- the unexplored sliver may hold a dominating point.
    #[test]
    fn partial_overlap_withholds_certification() {
        let bounds = Bounds {
            ideal: vec![0, 0],
            anti_ideal: vec![100, 100],
        };
        let front = front_of(&[[40, 40]]);
        let straddling = region([30, 30], [80, 80]);
        assert_eq!(certify(&front, &[straddling], &bounds), vec![false]);
    }

    /// The warm-start sweep must reach both extremes and stay inside `[0, 1]`.
    ///
    /// The endpoints are what give the front its spread; a sweep that never
    /// reaches them returns interior points only and prunes far less.
    #[test]
    fn the_warm_start_sweep_spans_both_objectives() {
        let shares = |n: usize| -> Vec<f64> {
            (0..n)
                .map(|i| {
                    if n == 1 {
                        0.5
                    } else {
                        i as f64 / (n - 1) as f64
                    }
                })
                .collect()
        };
        let sweep = shares(5);
        assert_eq!(sweep, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(shares(1), vec![0.5], "a single solve splits the difference");
        assert_eq!(shares(2), vec![0.0, 1.0], "two solves take the extremes");
        assert!(shares(20).iter().all(|s| (0.0..=1.0).contains(s)));
    }

    /// Scaling by range is what makes the sweep produce different points.
    ///
    /// Without it the objective with the larger magnitude decides the weighted
    /// sum at every setting but the most extreme, and the sweep returns the
    /// same corner over and over.
    #[test]
    fn warm_start_weights_are_scaled_by_range() {
        let bounds = Bounds {
            ideal: vec![0, 0],
            anti_ideal: vec![100, 100_000_000],
        };
        let scale: Vec<f64> = bounds
            .ideal
            .iter()
            .zip(&bounds.anti_ideal)
            .map(|(lo, hi)| 1.0 / ((hi - lo).max(1) as f64))
            .collect();

        // At the midpoint the two scaled contributions are comparable.
        let share = 0.5;
        let weights = [scale[0] * (1.0 - share), scale[1] * share];
        let first = weights[0] * 100.0;
        let second = weights[1] * 100_000_000.0;
        assert!(
            (first - second).abs() < 1e-9,
            "a full-range move in either objective must weigh the same"
        );
    }

    #[test]
    fn area_counts_inclusive_corners_and_empties_are_zero() {
        assert_eq!(region([0, 0], [0, 0]).area(), 1);
        assert_eq!(region([0, 0], [1, 1]).area(), 4);
        assert_eq!(region([5, 0], [4, 9]).area(), 0, "inverted is empty");
        assert!(region([5, 0], [4, 9]).is_empty());
    }
}
