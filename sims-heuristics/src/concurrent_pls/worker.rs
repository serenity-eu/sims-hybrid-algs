use std::time::{Duration, Instant};

use log::info;
use pareto::{Objectives, ParetoFront};
use tracing::{debug_span, info_span};

const WORKER_LOG_INTERVAL: Duration = Duration::from_secs(30);

use crate::{
    concurrent_pls::{
        config::{ConcurrentPLSConfig, RegionSearchMode},
        decomposition::{belongs_to_region, Region},
        snapshot::{fingerprint, GlobalFrontSlot, ObjectiveSnapshot, SnapshotSlot},
    },
    pareto_local_search::{ParetoLocalSearch, SAPlsStatus, StepStatus},
    pls_config::PlsOptimizations,
    problem::SetCoverProblem,
    solution::{EncodedSolution, ImageSet},
    solution_set_impl::NdTreeSolutionSet,
    timer::Timer,
};

/// Per-region statistics collected during a run.
///
/// Includes diagnostic metrics from design doc section 11.
#[derive(Debug, Default, Clone)]
pub struct RegionStats {
    pub iterations_completed: usize,
    pub snapshots_published: usize,
    pub solutions_adopted: usize,
    pub wall_time: Duration,
    /// Number of solutions in the initial population assigned to this region.
    pub initial_pop_size: usize,
    /// Number of solutions in the archive at end of run.
    pub final_archive_size: usize,
    /// Number of final archive solutions that do NOT belong to this region
    /// (i.e. another region has a lower Tchebycheff score for them).
    /// Computed by the orchestrator after the run finishes.
    pub out_of_region_count: usize,
    // -- Diagnostic metrics (section 11) --
    /// Total neighbors evaluated across all PLS steps in this region.
    pub neighbors_explored: usize,
    /// Total neighbors rejected by dedup across all PLS steps.
    pub duplicates_skipped: usize,
    // -- SA-PLS metrics (Phase 5) --
    /// Iteration at which scalarized search exhausted all neighborhoods.
    /// `None` if scalarized search was not used or did not exhaust.
    pub scalarized_exhausted_at_iteration: Option<usize>,
    // -- Cross-worker sync metrics --
    /// Total solutions removed from this worker's archive by global-front dominance.
    pub solutions_pruned: usize,
    /// Number of sync events where at least one solution was pruned from this archive.
    pub prune_events: usize,
}

/// Result returned by a region worker after it finishes.
pub struct RegionResult<T, const D: usize>
where
    T: Clone + ImageSet<D> + pareto::MoSolution<D> + PartialEq + std::fmt::Debug,
{
    pub region_index: usize,
    /// Weight vector that defined this region's search focus.
    pub weight_vector: [f64; D],
    pub archive: NdTreeSolutionSet<T, D>,
    pub stats: RegionStats,
}

/// A region worker that runs a single PLS instance on an assigned sub-region of objective space,
/// and periodically synchronizes with the global front via lock-free `Arc` pointer swaps.
///
/// Supports three modes via `RegionSearchMode`:
/// - `Unconstrained`: standard Phase 3 PLS with double Pareto gate.
/// - `ScalarizedAuxiliary`: SA-PLS with decoupled archive/auxiliary gates.
/// - `ScalarizedAuxiliaryWithFallback`: SA-PLS, then standard PLS after exhaustion.
pub struct RegionWorker<'prob, T, P, const D: usize>
where
    T: ImageSet<D> + EncodedSolution<P, D> + std::hash::Hash + Send + Sync + 'prob,
    P: SetCoverProblem<D> + Send + Sync + 'prob,
{
    region: Region<D>,
    all_regions: Vec<Region<D>>,
    pls: ParetoLocalSearch<'prob, T, NdTreeSolutionSet<T, D>, P, D>,
    snapshot_slot: SnapshotSlot<T, D>,
    global_front_slot: GlobalFrontSlot<T, D>,
    /// Dropping this Sender signals the merger that this worker is done.
    _done_guard: crossbeam_channel::Sender<()>,
    config: ConcurrentPLSConfig,
    stats: RegionStats,
    /// Local ideal point for Tchebycheff scalarization (SA-PLS).
    /// Initialized from the initial population, updated per-neighbor (Gate 3)
    /// and synchronized with global ideal on each sync event.
    local_ideal: Objectives<D>,
}

impl<'prob, T, P, const D: usize> RegionWorker<'prob, T, P, D>
where
    T: ImageSet<D> + EncodedSolution<P, D> + std::hash::Hash + Send + Sync + 'prob,
    P: SetCoverProblem<D> + Send + Sync + 'prob,
    NdTreeSolutionSet<T, D>: ParetoFront<'prob, T>
        + Clone
        + FromIterator<T>
        + IntoIterator<Item = T>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        region: Region<D>,
        all_regions: Vec<Region<D>>,
        problem: &'prob P,
        initial_pop: NdTreeSolutionSet<T, D>,
        initial_pop_size: usize,
        snapshot_slot: SnapshotSlot<T, D>,
        global_front_slot: GlobalFrontSlot<T, D>,
        done_guard: crossbeam_channel::Sender<()>,
        config: ConcurrentPLSConfig,
    ) -> Self {
        // Compute local ideal from initial population (section 16.8)
        let mut local_ideal = [u64::MAX; D];
        for s in initial_pop.iter() {
            for (j, &obj) in s.objectives().iter().enumerate() {
                local_ideal[j] = local_ideal[j].min(obj);
            }
        }

        // Sync local ideal with global ideal snapshot
        {
            let global = global_front_slot.load();
            for j in 0..D {
                local_ideal[j] = local_ideal[j].min(global.ideal_point[j]);
            }
        }

        let pls = ParetoLocalSearch::new(
            problem,
            &initial_pop,
            config.neighborhood_size_range.clone(),
            config.is_deterministic,
            PlsOptimizations::default(),
        );
        Self {
            region,
            all_regions,
            pls,
            snapshot_slot,
            global_front_slot,
            _done_guard: done_guard,
            config,
            stats: RegionStats {
                initial_pop_size,
                ..RegionStats::default()
            },
            local_ideal,
        }
    }

    pub fn run(mut self) -> RegionResult<T, D> {
        let start = Instant::now();
        let timer = Timer::start(self.config.max_duration);
        let _run_span = info_span!(
            "region.run",
            region = self.region.index,
            mode = ?self.config.region_search_mode,
            initial_archive = self.pls.archive().len(),
        )
        .entered();

        info!(
            "Region {} worker starting: {} solutions, timeout={:.0}s",
            self.region.index,
            self.pls.archive().len(),
            self.config.max_duration.as_secs_f64(),
        );

        match self.config.region_search_mode {
            RegionSearchMode::Unconstrained => self.run_unconstrained(&timer),
            RegionSearchMode::ScalarizedAuxiliary => self.run_scalarized(&timer, false),
            RegionSearchMode::ScalarizedAuxiliaryWithFallback => {
                self.run_scalarized(&timer, true);
            }
        }

        // Final publish so merger gets latest state
        self.publish_snapshot();

        self.stats.wall_time = start.elapsed();

        let archive = self.pls.archive().clone();
        self.stats.final_archive_size = archive.len();

        let weight_vector = self.region.weight_vector;
        RegionResult {
            region_index: self.region.index,
            weight_vector,
            archive,
            stats: self.stats,
        }
    }

    // -------------------------------------------------------------------------
    // Run modes
    // -------------------------------------------------------------------------

    /// Phase 3 unconstrained loop: standard PLS with periodic sync.
    fn run_unconstrained(&mut self, timer: &Timer) {
        let _phase_span =
            info_span!("region.phase", region = self.region.index, phase = "unconstrained")
                .entered();
        let mut last_log = Instant::now();
        for iteration in 1..=self.config.max_iterations {
            let step_result = self.pls.step(iteration, timer);
            self.stats.iterations_completed = iteration;

            // Aggregate diagnostic metrics
            self.stats.neighbors_explored += step_result.stats.explored_neighbor_count;
            self.stats.duplicates_skipped += step_result.stats.duplicated_neighbor_count;

            if iteration % self.config.sync_interval_steps == 0 {
                {
                    let _s = debug_span!("sync.publish", region = self.region.index).entered();
                    self.publish_snapshot();
                }
                {
                    let _s = debug_span!("sync.prune", region = self.region.index).entered();
                    self.prune_from_global_front();
                }
                {
                    let _s = debug_span!("sync.adopt", region = self.region.index).entered();
                    self.adopt_from_global_front();
                }
                tracing::debug!(
                    region = self.region.index,
                    iteration,
                    archive_size = self.pls.archive().len(),
                    adopted_total = self.stats.solutions_adopted,
                    pruned_total = self.stats.solutions_pruned,
                    prune_events = self.stats.prune_events,
                    neighbors_total = self.stats.neighbors_explored,
                    dupes_total = self.stats.duplicates_skipped,
                    "sync"
                );

                if last_log.elapsed() >= WORKER_LOG_INTERVAL {
                    info!(
                        "Region {} iter={} arch={} explored={} elapsed={:.1}s",
                        self.region.index,
                        iteration,
                        self.pls.archive().len(),
                        self.stats.neighbors_explored,
                        timer.elapsed().as_secs_f64(),
                    );
                    last_log = Instant::now();
                }
            }

            if step_result.status == StepStatus::AllNeighborhoodStructuresExplored
                || timer.is_expired()
            {
                break;
            }
        }
    }

    /// SA-PLS loop (sections 16.4-16.10): scalarized auxiliary with optional fallback.
    fn run_scalarized(&mut self, timer: &Timer, with_fallback: bool) {
        let _phase_span =
            info_span!("region.phase", region = self.region.index, phase = "scalarized")
                .entered();
        let weight = self.region.weight_vector;
        let rho = self.config.scalarized_rho;
        let mut last_log = Instant::now();

        for iteration in 1..=self.config.max_iterations {
            // Derive normalization bounds from global front (updated by merger)
            let bounds = self.current_bounds();
            let (sa_status, step_stats) = self.pls.step_scalarized(
                iteration,
                timer,
                &weight,
                &mut self.local_ideal,
                &bounds,
                rho,
            );
            self.stats.iterations_completed = iteration;
            self.stats.neighbors_explored += step_stats.explored_neighbor_count;
            self.stats.duplicates_skipped += step_stats.duplicated_neighbor_count;

            if iteration % self.config.sync_interval_steps == 0 {
                {
                    let _s = debug_span!("sync.publish", region = self.region.index).entered();
                    self.publish_snapshot();
                }
                {
                    let _s = debug_span!("sync.prune", region = self.region.index).entered();
                    self.prune_from_global_front();
                }
                let sync_bounds = self.current_bounds();
                {
                    let _s = debug_span!("sync.adopt", region = self.region.index).entered();
                    self.adopt_from_global_front_scalarized(&weight, &sync_bounds, rho);
                }
                self.sync_ideal_from_global();
                tracing::debug!(
                    region = self.region.index,
                    iteration,
                    archive_size = self.pls.archive().len(),
                    adopted_total = self.stats.solutions_adopted,
                    pruned_total = self.stats.solutions_pruned,
                    prune_events = self.stats.prune_events,
                    neighbors_total = self.stats.neighbors_explored,
                    dupes_total = self.stats.duplicates_skipped,
                    "sync"
                );

                if last_log.elapsed() >= WORKER_LOG_INTERVAL {
                    info!(
                        "Region {} (sa-pls) iter={} arch={} explored={} elapsed={:.1}s",
                        self.region.index,
                        iteration,
                        self.pls.archive().len(),
                        self.stats.neighbors_explored,
                        timer.elapsed().as_secs_f64(),
                    );
                    last_log = Instant::now();
                }
            }

            match sa_status {
                SAPlsStatus::NewPopulation | SAPlsStatus::IncreasedNeighborhoodStructure => {}
                SAPlsStatus::ScalarizedExhausted => {
                    self.stats.scalarized_exhausted_at_iteration = Some(iteration);
                    if with_fallback {
                        tracing::info!(
                            region = self.region.index,
                            iteration,
                            "SA-PLS exhausted, switching to standard PLS fallback"
                        );
                        self.pls.reseed_population_from_archive();
                        self.run_unconstrained_remainder(timer, iteration + 1);
                    }
                    break;
                }
            }

            if timer.is_expired() {
                break;
            }
        }
    }

    /// Continue with standard PLS for remaining time after SA-PLS exhaustion (fallback).
    fn run_unconstrained_remainder(&mut self, timer: &Timer, start_iteration: usize) {
        let _phase_span =
            info_span!("region.phase", region = self.region.index, phase = "fallback")
                .entered();
        let mut last_log = Instant::now();
        for iteration in start_iteration..=self.config.max_iterations {
            let step_result = self.pls.step(iteration, timer);
            self.stats.iterations_completed = iteration;
            self.stats.neighbors_explored += step_result.stats.explored_neighbor_count;
            self.stats.duplicates_skipped += step_result.stats.duplicated_neighbor_count;

            if iteration % self.config.sync_interval_steps == 0 {
                {
                    let _s = debug_span!("sync.publish", region = self.region.index).entered();
                    self.publish_snapshot();
                }
                {
                    let _s = debug_span!("sync.prune", region = self.region.index).entered();
                    self.prune_from_global_front();
                }
                {
                    let _s = debug_span!("sync.adopt", region = self.region.index).entered();
                    self.adopt_from_global_front();
                }
                tracing::debug!(
                    region = self.region.index,
                    iteration,
                    archive_size = self.pls.archive().len(),
                    adopted_total = self.stats.solutions_adopted,
                    pruned_total = self.stats.solutions_pruned,
                    prune_events = self.stats.prune_events,
                    neighbors_total = self.stats.neighbors_explored,
                    dupes_total = self.stats.duplicates_skipped,
                    "sync"
                );

                if last_log.elapsed() >= WORKER_LOG_INTERVAL {
                    info!(
                        "Region {} (fallback) iter={} arch={} explored={} elapsed={:.1}s",
                        self.region.index,
                        iteration,
                        self.pls.archive().len(),
                        self.stats.neighbors_explored,
                        timer.elapsed().as_secs_f64(),
                    );
                    last_log = Instant::now();
                }
            }

            if step_result.status == StepStatus::AllNeighborhoodStructuresExplored
                || timer.is_expired()
            {
                break;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Snapshot publication (non-blocking Arc swap)
    // -------------------------------------------------------------------------

    fn publish_snapshot(&mut self) {
        let snapshot: Vec<ObjectiveSnapshot<T, D>> = self
            .pls
            .archive()
            .iter()
            .map(|s| ObjectiveSnapshot {
                objectives: *s.objectives(),
                fingerprint: fingerprint(s),
                solution: s.clone(),
            })
            .collect();

        self.snapshot_slot
            .store(std::sync::Arc::new(snapshot));
        self.stats.snapshots_published += 1;
    }

    // -------------------------------------------------------------------------
    // Prune from global front: evict locally dominated solutions
    // -------------------------------------------------------------------------

    fn prune_from_global_front(&mut self) {
        let global = self.global_front_slot.load();

        if global.front.is_empty() {
            return;
        }

        let before = self.pls.archive().len();
        // For each solution in the global front, remove all archive entries it dominates.
        // NDTree::remove_dominated reuses the same scan as insertion without inserting.
        let dominators: Vec<T> = global.front.iter().map(|g| g.solution.clone()).collect();
        for dom in &dominators {
            self.pls.archive_mut().remove_dominated(dom);
        }
        let pruned = before.saturating_sub(self.pls.archive().len());
        if pruned > 0 {
            self.stats.solutions_pruned += pruned;
            self.stats.prune_events += 1;
        }
        tracing::debug!(
            region = self.region.index,
            pruned,
            archive_size = self.pls.archive().len(),
            "prune"
        );
    }

    // -------------------------------------------------------------------------
    // Adopt globally non-dominated solutions belonging to this region
    // -------------------------------------------------------------------------

    /// Unconstrained adoption: accept any globally non-dominated solution belonging
    /// to this region that hasn't been explored yet.
    fn adopt_from_global_front(&mut self) {
        let global = self.global_front_slot.load();
        let ideal = global.ideal_point;
        let bounds = bounds_from_ideal_nadir(&global.ideal_point, &global.nadir_point);

        let candidates: Vec<T> = global
            .front
            .iter()
            .filter(|g| {
                belongs_to_region(&g.objectives, &self.region, &self.all_regions, &ideal, &bounds)
                    && !self
                        .pls
                        .explored_solutions_data()
                        .is_registered(&g.solution)
            })
            .map(|g| g.solution.clone())
            .collect();

        let candidates_count = candidates.len();
        let mut newly_adopted = 0usize;
        for solution in candidates {
            if self.pls.try_adopt_solution(&solution) {
                self.stats.solutions_adopted += 1;
                newly_adopted += 1;
            }
        }
        tracing::debug!(
            region = self.region.index,
            candidates = candidates_count,
            adopted = newly_adopted,
            "adopt"
        );
    }

    /// SA-PLS adoption gate (section 16.11): accept only if the candidate also improves
    /// g_atch relative to the worker's current best scalarized solution.
    /// This prevents regional pollution from cross-worker solutions.
    fn adopt_from_global_front_scalarized(
        &mut self,
        weight: &[f64; D],
        bounds: &[(f64, f64); D],
        rho: f64,
    ) {
        use crate::concurrent_pls::decomposition::TchebycheffCoeffs;

        let global = self.global_front_slot.load();
        let ideal = global.ideal_point;
        let global_bounds = bounds_from_ideal_nadir(&global.ideal_point, &global.nadir_point);
        let coeffs = TchebycheffCoeffs::new(weight, bounds, rho);

        // Compute best (lowest) scalarized score in current archive
        let best_local_score = self
            .pls
            .archive()
            .iter()
            .map(|s| coeffs.score(s.objectives(), &self.local_ideal))
            .fold(f64::INFINITY, f64::min);

        let candidates: Vec<T> = global
            .front
            .iter()
            .filter(|g| {
                belongs_to_region(&g.objectives, &self.region, &self.all_regions, &ideal, &global_bounds)
                    && !self
                        .pls
                        .explored_solutions_data()
                        .is_registered(&g.solution)
                    && coeffs.score(&g.objectives, &self.local_ideal) < best_local_score
            })
            .map(|g| g.solution.clone())
            .collect();

        let candidates_count = candidates.len();
        let mut newly_adopted = 0usize;
        for solution in candidates {
            // Update local ideal for adopted solution (Gate 3)
            for j in 0..D {
                let fj = solution.objectives()[j];
                if fj < self.local_ideal[j] {
                    self.local_ideal[j] = fj;
                }
            }
            if self.pls.try_adopt_solution(&solution) {
                self.stats.solutions_adopted += 1;
                newly_adopted += 1;
            }
        }
        tracing::debug!(
            region = self.region.index,
            candidates = candidates_count,
            adopted = newly_adopted,
            "adopt_scalarized"
        );
    }

    /// Take component-wise minimum of local ideal and global ideal (section 16.8).
    fn sync_ideal_from_global(&mut self) {
        let global = self.global_front_slot.load();
        for j in 0..D {
            self.local_ideal[j] = self.local_ideal[j].min(global.ideal_point[j]);
        }
    }

    /// Derive normalization bounds from the current global front snapshot.
    /// Returns `(ideal[j], nadir[j])` per objective, giving a dynamically
    /// updating normalization range that tracks the PF envelope.
    fn current_bounds(&self) -> [(f64, f64); D] {
        let global = self.global_front_slot.load();
        bounds_from_ideal_nadir(&global.ideal_point, &global.nadir_point)
    }
}

/// Compute the ideal point from a set of objective snapshots.
pub fn compute_ideal<T: Clone, const D: usize>(
    solutions: &[ObjectiveSnapshot<T, D>],
) -> Objectives<D> {
    let mut ideal = [u64::MAX; D];
    for s in solutions {
        for (j, &obj) in s.objectives.iter().enumerate() {
            ideal[j] = ideal[j].min(obj);
        }
    }
    ideal
}

/// Compute the nadir point (component-wise maximum) from a set of objective snapshots.
pub fn compute_nadir<T: Clone, const D: usize>(
    solutions: &[ObjectiveSnapshot<T, D>],
) -> Objectives<D> {
    let mut nadir = [0u64; D];
    for s in solutions {
        for (j, &obj) in s.objectives.iter().enumerate() {
            nadir[j] = nadir[j].max(obj);
        }
    }
    nadir
}

/// Merge multiple region snapshots into a single non-dominated front using ND-Tree.
/// Deduplicates by fingerprint first (O(n) HashMap), then uses `NdTreeSolutionSet::from_iter`
/// which internally enforces non-dominance via the ND-Tree update algorithm.
///
/// Accepts `&[&[ObjectiveSnapshot]]` to avoid requiring owned `Vec` copies --
/// callers can pass slices derived from `Arc<Vec<...>>` references.
pub fn merge_snapshots_to_front<T, const D: usize>(
    snapshots: &[&[ObjectiveSnapshot<T, D>]],
) -> NdTreeSolutionSet<ObjectiveSnapshot<T, D>, D>
where
    T: Clone + ImageSet<D>,
{
    use std::collections::HashMap;
    let total_len: usize = snapshots.iter().map(|s| s.len()).sum();
    let mut by_fingerprint: HashMap<u64, ObjectiveSnapshot<T, D>> =
        HashMap::with_capacity(total_len);
    for snapshot in snapshots {
        for item in *snapshot {
            by_fingerprint
                .entry(item.fingerprint)
                .or_insert_with(|| item.clone());
        }
    }

    by_fingerprint.into_values().collect()
}

/// Compute the ideal point from a non-dominated front stored in an ND-Tree.
pub fn compute_ideal_from_front<T, const D: usize>(
    front: &NdTreeSolutionSet<ObjectiveSnapshot<T, D>, D>,
) -> Objectives<D>
where
    T: Clone + ImageSet<D>,
{
    let mut ideal = [u64::MAX; D];
    for s in front.iter() {
        for (j, &obj) in s.objectives.iter().enumerate() {
            ideal[j] = ideal[j].min(obj);
        }
    }
    ideal
}

/// Compute the nadir point (component-wise maximum) from a non-dominated front.
pub fn compute_nadir_from_front<T, const D: usize>(
    front: &NdTreeSolutionSet<ObjectiveSnapshot<T, D>, D>,
) -> Objectives<D>
where
    T: Clone + ImageSet<D>,
{
    let mut nadir = [0u64; D];
    for s in front.iter() {
        for (j, &obj) in s.objectives.iter().enumerate() {
            nadir[j] = nadir[j].max(obj);
        }
    }
    nadir
}

/// Derive `(min, max)` normalization bounds per objective from ideal and nadir points.
///
/// The resulting bounds have a guaranteed minimum range of 1.0 to avoid division by zero.
/// This is called by workers on every sync to get up-to-date normalization from the
/// global front's ideal/nadir, which the merger updates every merge cycle.
#[inline]
#[must_use]
pub fn bounds_from_ideal_nadir<const D: usize>(
    ideal: &Objectives<D>,
    nadir: &Objectives<D>,
) -> [(f64, f64); D] {
    std::array::from_fn(|j| (ideal[j] as f64, nadir[j] as f64))
}
