//! Runtime configuration for PLS algorithmic optimizations.
//!
//! Each flag toggles one optimization independently, enabling ablation
//! studies that measure the contribution of each technique.
//!
//! Scalarized parent selection can optionally use ND-tree accelerated
//! archive queries when both the build-time feature and this runtime
//! toggle are enabled.

use bitflags::bitflags;

bitflags! {
    /// Independent boolean PLS optimization toggles, packed into one field so
    /// the config struct stays flat. Each flag gates a single technique.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PlsFlags: u8 {
        /// Bulk checkpoint/restore of tracker state instead of per-image undo.
        const USE_CHECKPOINT = 1 << 0;
        /// Rank k=1 removal candidates "worst first" and cap at `max_k1_candidates`.
        const USE_RANKED_CANDIDATES = 1 << 1;
        /// Seed the initial population with greedy set-cover solutions.
        const USE_GREEDY_INITIAL_POPULATION = 1 << 2;
        /// Inject perturbed archive solutions when the auxiliary is empty.
        const USE_PERTURBATION_RESTART = 1 << 3;
        /// Explore a diverse population subset via farthest-point sampling.
        const USE_DIVERSE_PROBING = 1 << 4;
        /// Allow ND-tree accelerated queries for scalarized archive selection.
        const USE_ND_TREE_SCALARIZED_QUERY = 1 << 5;
    }
}

impl PlsFlags {
    /// Build a flag set from `(flag, enabled)` pairs, keeping only the enabled
    /// ones. Lets callers map individual boolean toggles (e.g. the public Python
    /// `solve_*` keyword arguments) onto flags without a wide boolean signature.
    #[must_use]
    pub fn from_pairs<I: IntoIterator<Item = (Self, bool)>>(pairs: I) -> Self {
        pairs
            .into_iter()
            .filter_map(|(flag, enabled)| enabled.then_some(flag))
            .fold(Self::empty(), |acc, flag| acc | flag)
    }
}

/// How parent solutions are selected for neighborhood exploration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolutionSelectionMode {
    /// Explore the full working population in random order.
    #[default]
    RandomShuffle,
    /// Explore a well-spread subset selected by farthest-point sampling.
    DiverseProbe,
    /// Explore parents selected by weighted Chebycheff scalarization.
    #[cfg(feature = "scalarized_selection")]
    ScalarizedChebycheff,
    /// First prefilter by diversity, then rank by weighted Chebycheff scalarization.
    #[cfg(feature = "scalarized_selection")]
    DiverseThenScalarizedChebycheff,
}

/// Which solution pool scalarized parent selection should draw from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScalarizedSelectionSource {
    /// Rank the current working population.
    #[default]
    Population,
    /// Rank the current approximated Pareto archive.
    Archive,
}

/// Runtime-toggleable PLS optimization switches.
#[derive(Debug, Clone)]
pub struct PlsOptimizations {
    /// Independent boolean optimization toggles (see [`PlsFlags`]):
    /// `USE_CHECKPOINT`, `USE_RANKED_CANDIDATES`, `USE_GREEDY_INITIAL_POPULATION`,
    /// `USE_PERTURBATION_RESTART`, `USE_DIVERSE_PROBING`,
    /// `USE_ND_TREE_SCALARIZED_QUERY`.
    pub flags: PlsFlags,

    /// Maximum removal candidates to evaluate for k=1 when
    /// `USE_RANKED_CANDIDATES` is set. Ignored when unset.
    pub max_k1_candidates: usize,

    /// Use probabilistic GRASP-based residual probing instead of
    /// exhaustive subset enumeration. Controlled via the existing
    /// `set_runtime_probing_budget` mechanism in `residual_problem`.
    /// `None` = exhaustive (default), `Some(n)` = budget of n samples.
    pub probing_budget: Option<usize>,

    /// Optional cap on total neighbors yielded per solution.
    /// `None` = unlimited (explore full neighborhood).
    pub neighborhood_budget: Option<usize>,

    /// Number of solutions to probe per step when `USE_DIVERSE_PROBING` is set.
    /// `None` = auto-select `2 * D * sqrt(N)` where N is the population size.
    pub diverse_probe_budget: Option<usize>,

    /// Policy used to choose parent solutions for neighborhood exploration.
    pub solution_selection_mode: SolutionSelectionMode,

    /// Source pool used by scalarized parent selection.
    #[cfg(feature = "scalarized_selection")]
    pub scalarized_selection_source: ScalarizedSelectionSource,

    /// Maximum number of parent solutions selected per step by scalarized
    /// selection. `None` means all ranked candidates may be explored.
    #[cfg(feature = "scalarized_selection")]
    pub scalarized_parent_budget: Option<usize>,

    /// Number of random weight vectors sampled per step for scalarized parent
    /// selection. Each sampled direction may contribute one selected parent.
    #[cfg(feature = "scalarized_selection")]
    pub scalarized_weight_samples: usize,

    /// Augmentation coefficient for weighted Chebycheff scalarization.
    #[cfg(feature = "scalarized_selection")]
    pub scalarized_rho: f64,
}

impl Default for PlsOptimizations {
    fn default() -> Self {
        Self {
            // All optimizations on except diverse probing (deprecated default-off).
            flags: PlsFlags::USE_CHECKPOINT
                | PlsFlags::USE_RANKED_CANDIDATES
                | PlsFlags::USE_GREEDY_INITIAL_POPULATION
                | PlsFlags::USE_PERTURBATION_RESTART
                | PlsFlags::USE_ND_TREE_SCALARIZED_QUERY,
            max_k1_candidates: 15,
            probing_budget: None,
            neighborhood_budget: None,
            diverse_probe_budget: None,
            solution_selection_mode: SolutionSelectionMode::RandomShuffle,
            #[cfg(feature = "scalarized_selection")]
            scalarized_selection_source: ScalarizedSelectionSource::Population,
            #[cfg(feature = "scalarized_selection")]
            scalarized_parent_budget: Some(1),
            #[cfg(feature = "scalarized_selection")]
            scalarized_weight_samples: 1,
            #[cfg(feature = "scalarized_selection")]
            scalarized_rho: 1e-3,
        }
    }
}

impl PlsOptimizations {
    /// Baseline configuration: all optimizations disabled.
    /// Produces behaviour equivalent to the original PLS before any changes.
    #[must_use]
    pub const fn baseline() -> Self {
        Self {
            flags: PlsFlags::empty(),
            max_k1_candidates: usize::MAX,
            probing_budget: None,
            neighborhood_budget: None,
            diverse_probe_budget: None,
            solution_selection_mode: SolutionSelectionMode::RandomShuffle,
            #[cfg(feature = "scalarized_selection")]
            scalarized_selection_source: ScalarizedSelectionSource::Population,
            #[cfg(feature = "scalarized_selection")]
            scalarized_parent_budget: None,
            #[cfg(feature = "scalarized_selection")]
            scalarized_weight_samples: 1,
            #[cfg(feature = "scalarized_selection")]
            scalarized_rho: 1e-3,
        }
    }
}
