use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use pareto::ParetoFront;
use tracing::info;

use crate::{
    concurrent_pls::{
        config::ConcurrentPLSConfig,
        decomposition::{assign_to_regions, belongs_to_region, build_regions},
        snapshot::{
            fingerprint, GlobalFrontSnapshot, GlobalFrontSlot, ObjectiveSnapshot, SnapshotSlot,
        },
        worker::{
            bounds_from_ideal_nadir, compute_ideal, compute_ideal_from_front, compute_nadir,
            compute_nadir_from_front, merge_snapshots_to_front, RegionResult, RegionStats,
            RegionWorker,
        },
    },
    pareto_local_search::ParetoLocalSearch,
    pls_config::PlsOptimizations,
    problem::SetCoverProblem,
    solution::{EncodedSolution, ImageSet},
    solution_set_impl::NdTreeSolutionSet,
};

/// Result returned by `ConcurrentPLS::solve`, containing the merged Pareto front
/// and per-region diagnostics.
pub struct ConcurrentPLSResult<T, const D: usize>
where
    T: Clone + ImageSet<D> + pareto::MoSolution<D> + PartialEq + std::fmt::Debug,
{
    /// Final merged non-dominated Pareto front across all regions.
    pub archive: NdTreeSolutionSet<T, D>,
    /// Per-region results, including per-region archives and statistics.
    pub region_results: Vec<RegionResult<T, D>>,
    /// Total PLS iterations summed across all regions.
    pub total_iterations: usize,
    /// Number of Das-Dennis regions (threads) that were used.
    pub num_regions: usize,
}

/// Top-level concurrent PLS orchestrator.
///
/// # Usage
///
/// ```rust,ignore
/// let result = ConcurrentPLS::new(&problem, config)
///     .solve(initial_population);
/// ```
pub struct ConcurrentPLS<'prob, T, P, const D: usize>
where
    T: ImageSet<D> + EncodedSolution<P, D> + std::hash::Hash + Send + Sync + Clone + 'prob,
    P: SetCoverProblem<D> + Send + Sync + 'prob,
{
    problem: &'prob P,
    config: ConcurrentPLSConfig,
    _phantom: std::marker::PhantomData<T>,
}

impl<'prob, T, P, const D: usize> ConcurrentPLS<'prob, T, P, D>
where
    T: ImageSet<D> + EncodedSolution<P, D> + std::hash::Hash + Send + Sync + Clone + 'prob,
    P: SetCoverProblem<D> + Send + Sync + 'prob,
    NdTreeSolutionSet<T, D>: ParetoFront<'prob, T>
        + Clone
        + FromIterator<T>
        + IntoIterator<Item = T>,
{
    pub fn new(problem: &'prob P, config: ConcurrentPLSConfig) -> Self {
        Self {
            problem,
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Run the concurrent PLS and return detailed results including per-region metrics.
    ///
    /// # Arguments
    /// * `initial_population` – Non-dominated starting solutions (e.g. from an exact solver).
    ///
    /// # Returns
    /// `ConcurrentPLSResult` containing the merged Pareto front and per-region diagnostics.
    pub fn solve(
        &self,
        initial_population: &NdTreeSolutionSet<T, D>,
    ) -> ConcurrentPLSResult<T, D>
    where
        T: std::fmt::Debug,
    {
        let total_start = Instant::now();
        let num_threads = self.config.num_threads;
        let _solve_span = tracing::info_span!(
            "concurrent_pls::solve",
            dims = D,
            num_threads,
            initial_pop = initial_population.len(),
        )
        .entered();

        // -------------------------------------------------------------------
        // 0. Warm-up: run a short sequential PLS to expand the objective
        //    range, then derive data-driven normalization bounds.
        //    Uses 5% of the total timeout (capped at 5s) to build a
        //    substantially broader objective envelope than the random
        //    initial population provides.
        // -------------------------------------------------------------------
        let warmup_frac = 0.05_f64;
        let warmup_cap = Duration::from_secs(5);
        let warmup_duration = Duration::from_secs_f64(
            (self.config.max_duration.as_secs_f64() * warmup_frac).min(warmup_cap.as_secs_f64()),
        );

        let warmed_pop: NdTreeSolutionSet<T, D>;
        if !initial_population.is_empty() && warmup_duration.as_millis() > 0 {
            let _warmup_span = tracing::info_span!(
                "concurrent_pls::warmup",
                duration_s = warmup_duration.as_secs_f64(),
            )
            .entered();
            let mut pls = ParetoLocalSearch::new(
                self.problem,
                initial_population,
                self.config.neighborhood_size_range.clone(),
                self.config.is_deterministic,
                PlsOptimizations::default(),
            );
            warmed_pop = pls.run(usize::MAX, warmup_duration);
            info!(
                "Warm-up PLS: expanded {} -> {} solutions in {:.2}s",
                initial_population.len(),
                warmed_pop.len(),
                total_start.elapsed().as_secs_f64(),
            );
        } else {
            warmed_pop = initial_population.clone();
        }

        // -------------------------------------------------------------------
        // 1. Build regions (weight vectors only; normalization bounds are
        //    derived dynamically from the global front's ideal/nadir).
        // -------------------------------------------------------------------
        let regions = build_regions::<D>(
            num_threads,
            self.config.das_dennis_h,
        );

        let actual_num_threads = regions.len();
        info!(
            "ConcurrentPLS: {} threads, {} Das-Dennis regions",
            actual_num_threads,
            actual_num_threads
        );

        // -------------------------------------------------------------------
        // 2. Compute ideal point from warmed-up population
        // -------------------------------------------------------------------
        let initial_snapshots: Vec<ObjectiveSnapshot<T, D>> = warmed_pop
            .iter()
            .map(|s| ObjectiveSnapshot {
                objectives: *s.objectives(),
                fingerprint: fingerprint(s),
                solution: s.clone(),
            })
            .collect();
        let ideal = compute_ideal(&initial_snapshots);
        let nadir = compute_nadir(&initial_snapshots);
        let initial_bounds = bounds_from_ideal_nadir(&ideal, &nadir);

        // -------------------------------------------------------------------
        // 3. Distribute the warmed-up population across regions
        // -------------------------------------------------------------------
        let region_pops = self.distribute_initial_population(
            &warmed_pop,
            &regions,
            &ideal,
            &initial_bounds,
        );

        // -------------------------------------------------------------------
        // 4. Create shared snapshot slots
        // -------------------------------------------------------------------
        let snapshot_slots: Vec<SnapshotSlot<T, D>> = (0..actual_num_threads)
            .map(|_| Arc::new(ArcSwap::new(Arc::new(Vec::<ObjectiveSnapshot<T, D>>::new()))))
            .collect();

        let global_front_slot: GlobalFrontSlot<T, D> = Arc::new(ArcSwap::new(Arc::new(
            GlobalFrontSnapshot {
                front: initial_snapshots.into_iter().collect(),
                ideal_point: ideal,
                nadir_point: nadir,
            },
        )));

        // -------------------------------------------------------------------
        // 5. Spawn all threads (workers + 1 background merger)
        // -------------------------------------------------------------------
        let (done_tx, done_rx) = crossbeam_channel::bounded::<()>(0);

        // Clone Arcs for the merger thread
        let merger_slots = snapshot_slots.clone();
        let merger_global_slot = global_front_slot.clone();
        let merge_interval = self.config.merge_interval;

        let mut region_results: Vec<RegionResult<T, D>> =
            {
                let _threads_span = tracing::info_span!(
                    "concurrent_pls::run_threads",
                    num_threads = actual_num_threads,
                )
                .entered();
                std::thread::scope(|scope| {
                // Spawn merger thread
                let merger_handle = scope.spawn(move || {
                    merger_loop(merger_slots, merger_global_slot, done_rx, merge_interval);
                });

                // Spawn worker threads
                let mut worker_handles = Vec::new();
                // Adjust worker timeout to account for warm-up time spent
                let warmup_elapsed = total_start.elapsed();
                let remaining = self
                    .config
                    .max_duration
                    .saturating_sub(warmup_elapsed);
                for i in 0..actual_num_threads {
                    let region = regions[i].clone();
                    let all_regions = regions.clone();
                    let snapshot_slot = snapshot_slots[i].clone();
                    let global_slot = global_front_slot.clone();
                    let done_guard = done_tx.clone();
                    let mut config = self.config.clone();
                    config.max_duration = remaining;
                    let worker_pop = region_pops[i].clone();
                    let problem = self.problem;

                    let handle = scope.spawn(move || {
                        let worker_pop_size = worker_pop.len();
                        let worker = RegionWorker::new(
                            region,
                            all_regions,
                            problem,
                            worker_pop,
                            worker_pop_size,
                            snapshot_slot,
                            global_slot,
                            done_guard,
                            config,
                        );
                        worker.run()
                    });
                    worker_handles.push(handle);
                }

                // Drop coordinator's done_tx so merger exits when all workers drop their guards
                drop(done_tx);

                // Collect worker results and wait for merger
                let raw: Vec<RegionResult<T, D>> = worker_handles
                    .into_iter()
                    .map(|h| h.join().expect("Region worker panicked"))
                    .collect();

                merger_handle.join().expect("Merger thread panicked");

                raw
            })
            };

        // -------------------------------------------------------------------
        // 6. Compute out_of_region_count for each region + log stats
        // -------------------------------------------------------------------
        // We need ideal + nadir for Tchebycheff scoring.
        let (global_ideal, global_nadir): (pareto::Objectives<D>, pareto::Objectives<D>) = {
            let global_snap = global_front_slot.load_full();
            (global_snap.ideal_point, global_snap.nadir_point)
        };
        let global_bounds = bounds_from_ideal_nadir(&global_ideal, &global_nadir);

        let all_regions_for_check = regions.clone();

        let mut total_stats = RegionStats::default();
        for result in &mut region_results {
            // Count solutions whose best-matching region is NOT this region
            let out_of_region = result
                .archive
                .iter()
                .filter(|s| {
                    !belongs_to_region(
                        s.objectives(),
                        &regions[result.region_index],
                        &all_regions_for_check,
                        &global_ideal,
                        &global_bounds,
                    )
                })
                .count();
            result.stats.out_of_region_count = out_of_region;

            let s = &result.stats;
            tracing::info!(
                region = result.region_index,
                iterations = s.iterations_completed,
                snapshots = s.snapshots_published,
                adopted = s.solutions_adopted,
                pruned = s.solutions_pruned,
                prune_events = s.prune_events,
                neighbors = s.neighbors_explored,
                dupes = s.duplicates_skipped,
                archive_size = s.final_archive_size,
                out_of_region = s.out_of_region_count,
                wall_time_s = s.wall_time.as_secs_f64(),
                "region_complete"
            );
            info!(
                "Region {}: {} iter, {} snap, {} adopted, \
                 neighbors={}, dupes={}, arch={}, out_of_region={}{}, {:.2}s",
                result.region_index,
                s.iterations_completed,
                s.snapshots_published,
                s.solutions_adopted,
                s.neighbors_explored,
                s.duplicates_skipped,
                s.final_archive_size,
                s.out_of_region_count,
                s.scalarized_exhausted_at_iteration
                    .map(|it| format!(" sa_exhaust@{it}"))
                    .unwrap_or_default(),
                s.wall_time.as_secs_f64(),
            );
            total_stats.iterations_completed += s.iterations_completed;
            total_stats.snapshots_published += s.snapshots_published;
            total_stats.solutions_adopted += s.solutions_adopted;
            total_stats.neighbors_explored += s.neighbors_explored;
            total_stats.duplicates_skipped += s.duplicates_skipped;
        }
        tracing::info!(
            total_iterations = total_stats.iterations_completed,
            total_snapshots = total_stats.snapshots_published,
            total_adopted = total_stats.solutions_adopted,
            total_neighbors = total_stats.neighbors_explored,
            total_dupes = total_stats.duplicates_skipped,
            "all_regions_complete"
        );
        info!(
            "ConcurrentPLS total: {} iterations, {} snapshots, \
             {} adopted, {} neighbors, {} dupes",
            total_stats.iterations_completed,
            total_stats.snapshots_published,
            total_stats.solutions_adopted,
            total_stats.neighbors_explored,
            total_stats.duplicates_skipped,
        );

        // -------------------------------------------------------------------
        // 7. Final global merge: combine all region archives, dedup & prune
        // -------------------------------------------------------------------
        let _final_merge_span =
            tracing::debug_span!("concurrent_pls::final_merge").entered();
        let mut final_archive: NdTreeSolutionSet<T, D> =
            NdTreeSolutionSet::new("final_concurrent");
        let total_iterations = total_stats.iterations_completed;
        let num_regions = actual_num_threads;

        for result in &region_results {
            for solution in result.archive.iter() {
                final_archive.try_insert(solution);
            }
        }

        info!(
            "ConcurrentPLS final Pareto front: {} solutions",
            final_archive.len()
        );

        ConcurrentPLSResult {
            archive: final_archive,
            region_results,
            total_iterations,
            num_regions,
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Distribute the initial population across regions using soft Tchebycheff assignment.
    /// Regions that receive no solutions are seeded by round-robin sampling from initial_pop.
    fn distribute_initial_population(
        &self,
        initial_pop: &NdTreeSolutionSet<T, D>,
        regions: &[crate::concurrent_pls::decomposition::Region<D>],
        ideal: &pareto::Objectives<D>,
        bounds: &[(f64, f64); D],
    ) -> Vec<NdTreeSolutionSet<T, D>> {
        let n = regions.len();
        let mut region_pops: Vec<Vec<T>> = vec![Vec::new(); n];

        for solution in initial_pop.iter() {
            let assigned = assign_to_regions(
                solution.objectives(),
                regions,
                ideal,
                bounds,
                self.config.boundary_threshold,
            );
            for idx in assigned {
                region_pops[idx].push(solution.clone());
            }
        }

        // Seed empty regions by cycling through the initial population
        let all_solutions: Vec<T> = initial_pop.iter().cloned().collect();
        if !all_solutions.is_empty() {
            for (i, pop) in region_pops.iter_mut().enumerate() {
                if pop.is_empty() {
                    pop.push(all_solutions[i % all_solutions.len()].clone());
                }
            }
        }

        region_pops
            .into_iter()
            .map(|pop| NdTreeSolutionSet::from_iter(pop))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Background merger loop
// ---------------------------------------------------------------------------

fn merger_loop<T, const D: usize>(
    snapshot_slots: Vec<SnapshotSlot<T, D>>,
    global_front_slot: GlobalFrontSlot<T, D>,
    done_rx: crossbeam_channel::Receiver<()>,
    merge_interval: std::time::Duration,
)
where
    T: Clone + ImageSet<D>,
{
    let mut merge_count = 0u64;
    loop {
        // Block until either the merge interval elapses (Timeout) or all workers
        // have dropped their done_tx guard (Disconnected). This is more responsive
        // than sleep() + try_recv() -- the merger exits immediately when workers finish
        // rather than waiting up to one full merge_interval.
        match done_rx.recv_timeout(merge_interval) {
            Ok(()) => {
                // Channel is bounded(0); no sender ever calls send(), so Ok is unreachable.
                // Treat as spurious wake and continue merging.
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // Normal periodic wake: merge all region snapshots.
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                // All workers finished (Senders dropped). Do a final merge and exit.
                break;
            }
        }

        merge_count += 1;
        tracing::debug!(merge_count, "merger_cycle");
        do_merge(&snapshot_slots, &global_front_slot);
    }

    // Final merge to capture any snapshots published after the last periodic merge.
    merge_count += 1;
    tracing::debug!(merge_count, is_final = true, "merger_cycle");
    do_merge(&snapshot_slots, &global_front_slot);
}

/// Read all region snapshots, merge into a single non-dominated front, and
/// publish via the global front slot. The `load_full()` call on each `ArcSwap`
/// returns an owned `Arc` (an atomic ref-count bump, not a deep clone), so
/// the only per-solution clones are inside `merge_snapshots_to_front` for
/// dedup-by-fingerprint. This keeps the merger's allocation footprint low.
fn do_merge<T, const D: usize>(
    snapshot_slots: &[SnapshotSlot<T, D>],
    global_front_slot: &GlobalFrontSlot<T, D>,
)
where
    T: Clone + ImageSet<D>,
{
    let _merge_span = tracing::debug_span!("merger::do_merge").entered();
    let t0 = std::time::Instant::now();
    let snapshot_arcs: Vec<Arc<Vec<ObjectiveSnapshot<T, D>>>> = snapshot_slots
        .iter()
        .map(|slot| slot.load_full())
        .collect();

    // Build slice-of-slices for merge_snapshots_to_front
    let snapshot_slices: Vec<&[ObjectiveSnapshot<T, D>]> =
        snapshot_arcs.iter().map(|arc| arc.as_slice()).collect();

    let merged = merge_snapshots_to_front(&snapshot_slices);
    let ideal = compute_ideal_from_front(&merged);
    let nadir = compute_nadir_from_front(&merged);
    let merged_size = merged.len();

    global_front_slot.store(Arc::new(GlobalFrontSnapshot {
        front: merged,
        ideal_point: ideal,
        nadir_point: nadir,
    }));

    tracing::debug!(
        merged_size,
        duration_us = t0.elapsed().as_micros(),
        "merge_complete"
    );
}
