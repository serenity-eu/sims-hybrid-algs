//! Runtime configuration for PLS algorithmic optimizations.
//!
//! Each flag toggles one optimization independently, enabling ablation
//! studies that measure the contribution of each technique.
//!
//! Scalarized parent selection can optionally use ND-tree accelerated
//! archive queries when both the build-time feature and this runtime
//! toggle are enabled.

/// How parent solutions are selected for neighborhood exploration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolutionSelectionMode {
    /// Explore the full working population in random order.
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

impl Default for SolutionSelectionMode {
    fn default() -> Self {
        Self::RandomShuffle
    }
}

/// Which solution pool scalarized parent selection should draw from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarizedSelectionSource {
    /// Rank the current working population.
    Population,
    /// Rank the current approximated Pareto archive.
    Archive,
}

impl Default for ScalarizedSelectionSource {
    fn default() -> Self {
        Self::Population
    }
}

/// Runtime-toggleable PLS optimization switches.
#[derive(Debug, Clone)]
pub struct PlsOptimizations {
    /// Use bulk checkpoint/restore for tracker state instead of per-image
    /// undo operations after each merge. Pure performance win — same
    /// algorithmic behaviour, fewer tracker operations.
    pub use_checkpoint: bool,

    /// For k=1 neighborhoods, rank removal candidates by "worst first"
    /// heuristic and limit to the top `max_k1_candidates` instead of
    /// exhaustively trying every replaceable selected image.
    pub use_ranked_candidates: bool,

    /// Maximum removal candidates to evaluate for k=1 when
    /// `use_ranked_candidates` is true. Ignored when false.
    pub max_k1_candidates: usize,

    /// Use probabilistic GRASP-based residual probing instead of
    /// exhaustive subset enumeration. Controlled via the existing
    /// `set_runtime_probing_budget` mechanism in `residual_problem`.
    /// `None` = exhaustive (default), `Some(n)` = budget of n samples.
    pub probing_budget: Option<usize>,

    /// Optional cap on total neighbors yielded per solution.
    /// `None` = unlimited (explore full neighborhood).
    pub neighborhood_budget: Option<usize>,

    /// When true, includes greedy-constructed solutions in the initial population.
    /// These are built by greedy set cover heuristics targeting each objective
    /// individually, providing better initial coverage of the objective space.
    pub use_greedy_initial_population: bool,

    /// When true, inject perturbed copies of archive solutions into the
    /// population when the auxiliary is empty before increasing k.
    /// This avoids expensive higher-k neighborhoods by restarting search
    /// from slightly modified Pareto-optimal solutions.
    pub use_perturbation_restart: bool,

    /// Select a diverse subset of the population to explore via farthest-point
    /// sampling in normalised objective space, instead of exploring all members.
    /// Each selected solution receives proportionally more neighbourhood
    /// evaluation time within the same budget.
    ///
    /// Deprecated in favour of `solution_selection_mode = DiverseProbe`, but
    /// retained for backward compatibility with existing callers.
    pub use_diverse_probing: bool,

    /// Number of solutions to probe per step when `use_diverse_probing` is true.
    /// `None` = auto-select `2 * D * sqrt(N)` where N is the population size.
    pub diverse_probe_budget: Option<usize>,

    /// Policy used to choose parent solutions for neighborhood exploration.
    pub solution_selection_mode: SolutionSelectionMode,

    /// When true, scalarized archive selection may use ND-tree accelerated
    /// branch-and-bound queries instead of linear archive scans.
    ///
    /// This toggle only has an effect when:
    /// - scalarized selection is enabled at build time, and
    /// - the selected source is `ScalarizedSelectionSource::Archive`.
    ///
    /// When false, scalarized archive selection falls back to a linear scan,
    /// which is useful for ablation and correctness comparisons.
    pub use_nd_tree_scalarized_query: bool,

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
            use_checkpoint: true,
            use_ranked_candidates: true,
            max_k1_candidates: 15,
            probing_budget: None,
            neighborhood_budget: None,
            use_greedy_initial_population: true,
            use_perturbation_restart: true,
            use_diverse_probing: false,
            diverse_probe_budget: None,
            solution_selection_mode: SolutionSelectionMode::RandomShuffle,
            use_nd_tree_scalarized_query: true,
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
    pub fn baseline() -> Self {
        Self {
            use_checkpoint: false,
            use_ranked_candidates: false,
            max_k1_candidates: usize::MAX,
            probing_budget: None,
            neighborhood_budget: None,
            use_greedy_initial_population: false,
            use_perturbation_restart: false,
            use_diverse_probing: false,
            diverse_probe_budget: None,
            solution_selection_mode: SolutionSelectionMode::RandomShuffle,
            use_nd_tree_scalarized_query: false,
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
