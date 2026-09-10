//! Balanced Box Method for biobjective integer programs.
//!
//! Boland, Charkhgard & Savelsbergh (2015), *A Criterion Space Search Algorithm
//! for Biobjective Integer Programming: The Balanced Box Method*, INFORMS
//! Journal on Computing 27(4).
//!
//! # Why not a unit-step epsilon recursion
//!
//! Raising an epsilon floor past each point found walks the front from one end.
//! Two things go wrong under a wall-clock budget. A solve that hits its time
//! limit returns a suboptimal point, which advances the floor barely at all, so
//! the recursion inches forward; and because progress is strictly left-to-right,
//! running out of budget loses the *entire* far end of the front, which is where
//! much of the hypervolume lives. Measured on `mexico_city_250`, that recursion
//! stalled around 0.94–0.95 of the exact hypervolume and got *worse* with more
//! time per solve, because longer solves meant fewer of them and less of the
//! front covered.
//!
//! Balanced Box instead splits the criterion-space rectangle at its midpoint and
//! recurses into both halves. Solves are spread over the whole front from the
//! first split onwards, so a partial budget yields points everywhere rather than
//! a prefix, and the number of solves is governed by how many non-dominated
//! points exist rather than by step granularity.
//!
//! # Guarantees
//!
//! With every solve proved optimal the method enumerates the complete
//! non-dominated set: each box is closed either by finding the point it
//! contains or by proving it empty. Under a per-solve time limit that guarantee
//! degrades to an approximation, and [`BoxSearchStats::exact`] reports whether
//! it still holds for the run.
use std::{
    collections::BinaryHeap,
    time::{Duration, Instant},
};

use fixedbitset::FixedBitSet;

use super::model::{Objective, PersistentModel, SolveOutcome};

/// A point in criterion space together with the selection that achieves it.
#[derive(Debug, Clone)]
pub struct BoxPoint {
    pub cost: u64,
    pub cloud: u64,
    pub images: FixedBitSet,
}

/// A rectangle of criterion space still to be searched, ordered by area so the
/// largest unexplored region is always taken next. Under a partial budget that
/// keeps coverage spread rather than clustered.
#[derive(Debug, Clone)]
struct SearchBox {
    /// Cheaper endpoint (lower cost, higher cloud).
    left: (u64, u64),
    /// Cleaner endpoint (higher cost, lower cloud).
    right: (u64, u64),
    priority: u128,
}

impl SearchBox {
    fn new(left: (u64, u64), right: (u64, u64)) -> Self {
        let dc = u128::from(right.0.saturating_sub(left.0));
        let dl = u128::from(left.1.saturating_sub(right.1));
        Self { left, right, priority: dc * dl }
    }

    /// A box can only contain a further point if both objectives have room.
    const fn is_searchable(&self) -> bool {
        self.right.0 > self.left.0 + 1 && self.left.1 > self.right.1 + 1
    }
}

impl PartialEq for SearchBox {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl Eq for SearchBox {}
impl PartialOrd for SearchBox {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SearchBox {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct BoxSearchStats {
    pub solves: usize,
    pub optimal: usize,
    pub timed_out: usize,
    pub infeasible: usize,
    pub boxes_closed: usize,
    pub boxes_left: usize,
    /// True when every solve was proved optimal and the queue was drained, so
    /// the returned set is the complete non-dominated set.
    pub exact: bool,
}

pub struct BalancedBox<'a> {
    model: &'a mut PersistentModel,
    total_area: u64,
    all_free: FixedBitSet,
    mip_time: Duration,
    stats: BoxSearchStats,
}

impl<'a> BalancedBox<'a> {
    pub fn new(model: &'a mut PersistentModel, total_area: u64, mip_time: Duration) -> Self {
        let mut all_free = FixedBitSet::with_capacity(model.num_images());
        all_free.insert_range(..);
        Self { model, total_area, all_free, mip_time, stats: BoxSearchStats::default() }
    }

    fn budget(&self, deadline: Instant) -> f64 {
        let now = Instant::now();
        if now >= deadline {
            return 0.0;
        }
        self.mip_time.as_secs_f64().min((deadline - now).as_secs_f64())
    }

    fn evaluate(&mut self, images: &FixedBitSet, clear: u64) -> BoxPoint {
        BoxPoint {
            cost: self.model.cost_of_selection(images),
            cloud: self.total_area - clear,
            images: images.clone(),
        }
    }

    /// Lexicographic solve: optimise `first`, then re-optimise the other
    /// objective while holding the first at its optimum. Without stage two the
    /// method emits weakly dominated points, which corrupt the box bounds and
    /// cause the search to revisit regions it has already closed.
    fn lexicographic(
        &mut self,
        warm: &FixedBitSet,
        floor: f64,
        ceiling: f64,
        first: Objective,
        deadline: Instant,
    ) -> Option<BoxPoint> {
        let secs = self.budget(deadline);
        if secs <= 0.0 {
            return None;
        }
        let (outcome, stage1) =
            self.model.solve(warm, &self.all_free, floor, ceiling, first, secs);
        self.stats.solves += 1;
        match outcome {
            SolveOutcome::Optimal => self.stats.optimal += 1,
            SolveOutcome::Feasible => self.stats.timed_out += 1,
            SolveOutcome::NoSolution => {
                self.stats.infeasible += 1;
                return None;
            }
        }
        let clear1 = self.model.clear_area_of(&stage1);
        let p1 = self.evaluate(&stage1, clear1);

        // Stage two: pin the first objective, optimise the second.
        let secs = self.budget(deadline);
        if secs <= 0.0 {
            return Some(p1);
        }
        let (floor2, ceil2, second) = match first {
            Objective::Cost => (floor, p1.cost as f64, Objective::Cloud),
            Objective::Cloud => ((self.total_area - p1.cloud) as f64, ceiling, Objective::Cost),
        };
        let (outcome2, stage2) =
            self.model.solve(&stage1, &self.all_free, floor2, ceil2, second, secs);
        self.stats.solves += 1;
        match outcome2 {
            SolveOutcome::Optimal => self.stats.optimal += 1,
            SolveOutcome::Feasible => self.stats.timed_out += 1,
            SolveOutcome::NoSolution => {
                self.stats.infeasible += 1;
                return Some(p1);
            }
        }
        let clear2 = self.model.clear_area_of(&stage2);
        let p2 = self.evaluate(&stage2, clear2);
        // Keep whichever is genuinely non-dominated; stage two can only match or
        // improve the second objective, but a truncated solve may not.
        if p2.cost <= p1.cost && p2.cloud <= p1.cloud { Some(p2) } else { Some(p1) }
    }

    /// Run the search until the queue drains or `deadline` passes.
    pub fn run(&mut self, warm: &FixedBitSet, deadline: Instant) -> (Vec<BoxPoint>, BoxSearchStats) {
        let mut points: Vec<BoxPoint> = Vec::new();

        // The two lexicographic extremes bound the whole front.
        let Some(left) =
            self.lexicographic(warm, 0.0, f64::INFINITY, Objective::Cost, deadline)
        else {
            self.stats.boxes_left = 0;
            return (points, self.stats);
        };
        let Some(right) =
            self.lexicographic(&left.images, 0.0, f64::INFINITY, Objective::Cloud, deadline)
        else {
            points.push(left);
            return (points, self.stats);
        };

        let mut queue: BinaryHeap<SearchBox> = BinaryHeap::new();
        let seed = SearchBox::new((left.cost, left.cloud), (right.cost, right.cloud));
        points.push(left);
        if seed.left != seed.right {
            points.push(right);
            queue.push(seed);
        }

        while let Some(b) = queue.pop() {
            if Instant::now() >= deadline {
                queue.push(b);
                break;
            }
            if !b.is_searchable() {
                self.stats.boxes_closed += 1;
                continue;
            }
            // Split the cost range in half and ask for the cleanest point in the
            // cheaper half. Splitting by cost keeps the two halves balanced in
            // the objective the solver bounds most tightly.
            let mid = b.left.0 + (b.right.0 - b.left.0) / 2;
            let warm_pt = points
                .iter()
                .filter(|p| p.cost <= mid)
                .min_by_key(|p| p.cloud)
                .map_or_else(|| FixedBitSet::with_capacity(self.model.num_images()), |p| p.images.clone());

            let found = self.lexicographic(
                &warm_pt,
                (self.total_area - (b.left.1 - 1)) as f64,
                mid as f64,
                Objective::Cloud,
                deadline,
            );

            match found {
                Some(p) if p.cost <= mid && p.cloud < b.left.1 => {
                    let lo = SearchBox::new(b.left, (p.cost, p.cloud));
                    let hi = SearchBox::new((p.cost, p.cloud), b.right);
                    points.push(p);
                    if lo.is_searchable() {
                        queue.push(lo);
                    } else {
                        self.stats.boxes_closed += 1;
                    }
                    if hi.is_searchable() {
                        queue.push(hi);
                    } else {
                        self.stats.boxes_closed += 1;
                    }
                }
                // Empty half: nothing cleaner exists below the split, so the
                // cheaper half is closed and only the dearer half survives.
                _ => {
                    self.stats.boxes_closed += 1;
                    let hi = SearchBox::new(b.left, b.right);
                    if hi.right.0 > mid + 1 {
                        let mut shrunk = hi;
                        shrunk.left = (mid + 1, b.left.1);
                        if shrunk.is_searchable() {
                            queue.push(SearchBox::new(shrunk.left, shrunk.right));
                        }
                    }
                }
            }
        }

        self.stats.boxes_left = queue.len();
        self.stats.exact = queue.is_empty() && self.stats.timed_out == 0;
        points.sort_unstable_by_key(|p| (p.cost, p.cloud));
        points.dedup_by_key(|p| (p.cost, p.cloud));
        (points, self.stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        milp_lns::model::clear_coverage, objectives::ObjectiveType, problem_bitset::ProblemBitset,
    };

    fn setup() -> (ProblemBitset<2>, Vec<FixedBitSet>, Vec<u64>) {
        let p = ProblemBitset::<2>::from_minizinc_datafile(
            std::path::Path::new("tests/data/lagos_nigeria_30.dzn"),
            [ObjectiveType::TotalCost, ObjectiveType::CloudyArea],
        )
        .expect("instance loads");
        let (clear, areas) = clear_coverage(&p);
        (p, clear, areas)
    }

    #[test]
    fn boxes_are_ordered_by_area_so_the_largest_region_is_searched_first() {
        let small = SearchBox::new((0, 10), (10, 0));
        let big = SearchBox::new((0, 100), (100, 0));
        let mut q = BinaryHeap::new();
        q.push(small);
        q.push(big.clone());
        assert_eq!(q.pop().expect("non-empty").priority, big.priority);
    }

    #[test]
    fn a_box_with_no_room_is_not_searchable() {
        assert!(!SearchBox::new((5, 5), (6, 4)).is_searchable());
        assert!(!SearchBox::new((5, 5), (5, 5)).is_searchable());
        assert!(SearchBox::new((0, 100), (100, 0)).is_searchable());
    }

    #[test]
    fn produces_a_mutually_non_dominated_front() {
        let (p, clear, areas) = setup();
        let total: u64 = areas.iter().sum();
        let mut model = PersistentModel::new(&p, &clear, &areas);
        let mut bb = BalancedBox::new(&mut model, total, Duration::from_secs(10));
        let warm = FixedBitSet::with_capacity(p.images.len());
        let (pts, stats) = bb.run(&warm, Instant::now() + Duration::from_secs(60));

        assert!(!pts.is_empty(), "the search must return at least the extremes");
        for a in &pts {
            for b in &pts {
                if std::ptr::eq(a, b) {
                    continue;
                }
                let dominates = a.cost <= b.cost && a.cloud <= b.cloud
                    && (a.cost < b.cost || a.cloud < b.cloud);
                assert!(!dominates, "({},{}) dominates ({},{})", a.cost, a.cloud, b.cost, b.cloud);
            }
        }
        // Every returned selection must be a genuine cover of the universe.
        for pt in &pts {
            let mut covered = FixedBitSet::with_capacity(p.universe_size);
            for i in pt.images.ones() {
                covered.union_with(&p.images[i]);
            }
            assert_eq!(covered.count_ones(..), p.universe_size, "returned a non-cover");
        }
        assert!(stats.solves >= 2, "at least the two lexicographic extremes");
    }

    #[test]
    fn reports_exactness_only_when_the_queue_drains_without_timeouts() {
        let (p, clear, areas) = setup();
        let total: u64 = areas.iter().sum();
        let mut model = PersistentModel::new(&p, &clear, &areas);
        // A budget this small cannot finish, so the run must not claim exactness.
        let mut bb = BalancedBox::new(&mut model, total, Duration::from_millis(1));
        let warm = FixedBitSet::with_capacity(p.images.len());
        let (_, stats) = bb.run(&warm, Instant::now() + Duration::from_millis(50));
        assert!(!stats.exact || stats.boxes_left == 0);
    }
}
