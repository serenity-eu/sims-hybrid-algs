use std::{ops::RangeInclusive, time::Duration};

/// Controls how region workers steer their auxiliary population.
///
/// See design doc section 16 for the full SA-PLS specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RegionSearchMode {
    /// Phase 3 behaviour: unconstrained Pareto search, parallel restarts.
    /// The auxiliary gate is coupled to the archive gate (double Pareto check).
    #[default]
    Unconstrained,
    /// SA-PLS: auxiliary criterion = scalarized improvement over parent.
    /// Archive remains global Pareto non-dominated.
    /// When scalarized search exhausts all neighborhood structures, the worker stops.
    ScalarizedAuxiliary,
    /// SA-PLS with fallback: when scalarized auxiliary is exhausted, finish
    /// remaining time with standard PLS from the current regional seed.
    ScalarizedAuxiliaryWithFallback,
}

/// Configuration for the concurrent Pareto Local Search algorithm.
#[derive(Clone, Debug)]
pub struct ConcurrentPLSConfig {
    /// Number of region worker threads to spawn.
    pub num_threads: usize,
    /// How many PLS steps between snapshot publish + global-front prune.
    pub sync_interval_steps: usize,
    /// How often the background merger wakes up to rebuild the global front.
    pub merge_interval: Duration,
    /// Boundary threshold for soft region assignment (0.0 = hard, 0.05 = 5% tolerance).
    pub boundary_threshold: f64,
    /// Das-Dennis `H` parameter; `None` = auto-select based on `num_threads`.
    pub das_dennis_h: Option<usize>,
    /// Maximum iterations per region worker.
    pub max_iterations: usize,
    /// Wall-clock timeout for the entire concurrent run.
    pub max_duration: Duration,
    /// Neighborhood size range passed to each region's inner PLS.
    pub neighborhood_size_range: RangeInclusive<u32>,
    /// Whether search should be deterministic (for reproducibility testing).
    pub is_deterministic: bool,
    /// How region workers steer their auxiliary population.
    pub region_search_mode: RegionSearchMode,
    /// Augmentation coefficient for Tchebycheff scalarization in SA-PLS.
    /// Controls tie-breaking strength. Default: 1e-3.
    pub scalarized_rho: f64,
}

impl ConcurrentPLSConfig {
    /// Sensible defaults for a production run.
    #[must_use]
    pub fn default_with_threads(num_threads: usize, max_duration: Duration) -> Self {
        Self {
            num_threads,
            sync_interval_steps: 5,
            merge_interval: Duration::from_millis(100),
            boundary_threshold: 0.05,
            das_dennis_h: None,
            max_iterations: 100_000,
            max_duration,
            neighborhood_size_range: 1..=6,
            is_deterministic: false,
            region_search_mode: RegionSearchMode::default(),
            scalarized_rho: 1e-3,
        }
    }
}
