use std::collections::HashMap;
#[cfg(feature = "probabilistic_probing")]
use std::sync::atomic::{AtomicUsize, Ordering};

use fixedbitset::FixedBitSet;
use itertools::Itertools;
use log::{debug, trace};
use pareto::{HasObjectives, ParetoFront};
#[cfg(feature = "probabilistic_probing")]
use rand::rngs::SmallRng;
#[cfg(feature = "probabilistic_probing")]
use rand::{Rng, SeedableRng};

/// Zero-allocation combination iterator over all subsets of `{0..n}` with sizes `0..=max_k`,
/// produced in lexicographic order.
///
/// Unlike `itertools::Combinations`, `next_into` fills a caller-supplied `[usize; 6]` buffer
/// in-place and returns the length of the current combination — no heap allocation per step.
/// The caller allocates a `Vec` with `buf[..k].to_vec()` only on the rare paths where a
/// combination passes all validity checks and must be stored.
struct CombSliceIter {
    n: usize,
    max_k: usize,
    k: usize,            // current subset size being enumerated
    indices: [usize; 6], // current combination indices (at most 6 elements for k=0..5)
    first_in_k: bool,    // whether we still need to emit the first combination of size k
}

impl CombSliceIter {
    #[inline]
    fn new(n: usize, max_k: usize) -> Self {
        Self {
            n,
            max_k,
            k: 0,
            indices: [0usize; 6],
            first_in_k: true,
        }
    }

    /// Fill `buf[0..k]` with the next combination and return `Some(k)`, or `None` when all
    /// subsets of `{0..n}` with sizes `0..=max_k` have been produced.
    ///
    /// `indices[i]` is guaranteed to be in `0..n` and `indices[] is strictly increasing.
    #[inline]
    fn next_into(&mut self, buf: &mut [usize; 6]) -> Option<usize> {
        loop {
            if self.k > self.max_k {
                return None;
            }
            let k = self.k;

            if self.first_in_k {
                // Skip sizes that exceed the number of available items.
                if k > self.n {
                    self.k += 1;
                    continue;
                }
                // Initialise to the lexicographically first k-combination: [0, 1, …, k-1].
                for i in 0..k {
                    self.indices[i] = i;
                }
                self.first_in_k = false;
                // The empty combination (k=0) has no "advance" state; bump k now so the next
                // call starts at k=1 instead of trying to advance a zero-length combination.
                if k == 0 {
                    self.k = 1;
                    self.first_in_k = true;
                }
                buf[..k].copy_from_slice(&self.indices[..k]);
                return Some(k);
            }

            // Advance to the next k-combination in lexicographic order.
            // Position i may hold values in 0..=(n-k+i), i.e. the condition to increment is
            // `indices[i] < n - k + i`.
            let mut incremented = false;
            for i in (0..k).rev() {
                if self.indices[i] < self.n - k + i {
                    self.indices[i] += 1;
                    for j in (i + 1)..k {
                        self.indices[j] = self.indices[j - 1] + 1;
                    }
                    buf[..k].copy_from_slice(&self.indices[..k]);
                    incremented = true;
                    return Some(k);
                }
            }
            if !incremented {
                // Exhausted all k-combinations; move to size k+1.
                self.k += 1;
                self.first_in_k = true;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Probabilistic GRASP-biased combination sampler (feature-gated)
// ---------------------------------------------------------------------------

/// Compute the binomial coefficient C(n, k) without overflow for typical SIMS sizes
/// (n < 200, k ≤ 5).
#[cfg(feature = "probabilistic_probing")]
fn comb(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result: usize = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Total number of combinations C(n,0) + C(n,1) + … + C(n,max_k).
#[cfg(feature = "probabilistic_probing")]
fn total_combinations(n: usize, max_k: usize) -> usize {
    (0..=max_k).map(|k| comb(n, k)).sum()
}

/// Default total budget (number of combinations to probe) when the exhaustive count exceeds
/// this value. Tune via the `SIMS_PROBING_BUDGET` environment variable at **compile time**
/// (parsed in `ProbabilisticCombIter::new`).
#[cfg(feature = "probabilistic_probing")]
const DEFAULT_PROBING_BUDGET: usize = 1_000;

/// Runtime-configurable probing budget for benchmarking.
///
/// When set to `usize::MAX`, the probabilistic iterator is never used (exhaustive mode).
/// When set to any other value, that value is used as the probing budget instead of
/// `DEFAULT_PROBING_BUDGET`.
///
/// This allows a single binary compiled with `probabilistic_probing` to compare
/// exhaustive and probabilistic residual enumeration at runtime.
///
/// # Usage
/// ```ignore
/// use pls::residual_problem::{set_runtime_probing_budget, PROBING_BUDGET_EXHAUSTIVE};
/// // Force exhaustive enumeration:
/// set_runtime_probing_budget(PROBING_BUDGET_EXHAUSTIVE);
/// // Restore default probabilistic probing:
/// set_runtime_probing_budget(DEFAULT_PROBING_BUDGET);
/// // Custom budget:
/// set_runtime_probing_budget(500);
/// ```
#[cfg(feature = "probabilistic_probing")]
static RUNTIME_PROBING_BUDGET: AtomicUsize = AtomicUsize::new(0);

/// Sentinel value: `0` means "use DEFAULT_PROBING_BUDGET" (no override).
#[cfg(feature = "probabilistic_probing")]
const NO_OVERRIDE: usize = 0;

/// Sentinel value to force exhaustive enumeration regardless of combination count.
#[cfg(feature = "probabilistic_probing")]
pub const PROBING_BUDGET_EXHAUSTIVE: usize = usize::MAX;

/// Set the runtime probing budget.
///
/// - `0` → use the compile-time default (`DEFAULT_PROBING_BUDGET = 1000`).
/// - `usize::MAX` (`PROBING_BUDGET_EXHAUSTIVE`) → always use exhaustive enumeration.
/// - Any other value → use that as the probing budget threshold.
///
/// This is safe to call from any thread; it uses `Relaxed` ordering because
/// exact synchronisation is not required — each `ResidualProblem` reads the
/// budget once at construction time.
#[cfg(feature = "probabilistic_probing")]
pub fn set_runtime_probing_budget(budget: usize) {
    RUNTIME_PROBING_BUDGET.store(budget, Ordering::Relaxed);
}

/// Read the effective probing budget (internal helper).
#[cfg(feature = "probabilistic_probing")]
fn effective_probing_budget() -> usize {
    let runtime = RUNTIME_PROBING_BUDGET.load(Ordering::Relaxed);
    if runtime == NO_OVERRIDE {
        DEFAULT_PROBING_BUDGET
    } else {
        runtime
    }
}

/// GRASP restricted-candidate-list parameter α ∈ [0,1].
/// 0 = pure greedy, 1 = pure random.  0.3 is a well-studied default.
#[cfg(feature = "probabilistic_probing")]
const GRASP_ALPHA: f64 = 0.3;

/// Phase of the `ProbabilisticCombIter` state machine.
#[cfg(feature = "probabilistic_probing")]
#[derive(Debug, Clone)]
enum ProbePhase {
    /// Emit the greedy (top-k by score) combination for the current `k`.
    Greedy { k: usize },
    /// Emit random GRASP-constructed k-subsets.
    Sampling { k: usize, remaining: usize },
    /// All budgets exhausted.
    Done,
}

/// GRASP-biased probabilistic combination sampler.
///
/// Instead of exhaustively enumerating all C(m, ≤`max_k`) subsets of the condensed image
/// set, this iterator:
///
/// 1. **Always emits the greedy combination** (top-k images by coverage score) for each k.
/// 2. **Samples `budget_per_k[k]` additional random k-subsets** using GRASP-style weighted
///    random sampling without replacement, biased toward high-coverage images.
/// 3. **Falls back to exhaustive enumeration** when C(m, ≤max_k) ≤ `total_budget`.
///
/// Coverage scores are the number of condensed elements each image covers (popcount of the
/// corresponding row in the `FlatBitMatrix`), plus a floor of 1.0 so every image has a
/// nonzero selection probability.
#[cfg(feature = "probabilistic_probing")]
struct ProbabilisticCombIter {
    n: usize,
    max_k: usize,
    /// Per-image GRASP score (coverage cardinality + 1 floor).
    scores: Vec<f64>,
    /// Image indices sorted by score descending (precomputed for greedy emission).
    sorted_by_score: Vec<usize>,
    /// Budget of samples allocated per k-value (index 0..=5).
    budget_per_k: [usize; 6],
    /// Seeded PRNG for reproducible sampling.
    rng: SmallRng,
    /// Current state-machine phase.
    phase: ProbePhase,
}

#[cfg(feature = "probabilistic_probing")]
impl ProbabilisticCombIter {
    /// Create a new probabilistic iterator.
    ///
    /// * `n`        — number of condensed images (universe size for subsets)
    /// * `max_k`    — maximum subset size (typically 5)
    /// * `scores`   — per-image coverage scores, length `n`
    /// * `seed`     — deterministic seed for the PRNG
    /// * `budget`   — total number of combinations to emit (across all k)
    fn new(n: usize, max_k: usize, scores: Vec<f64>, seed: u64, budget: usize) -> Self {
        // Pre-sort images by score descending for greedy emission.
        let mut sorted_by_score: Vec<usize> = (0..n).collect();
        sorted_by_score.sort_unstable_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let budget_per_k = Self::distribute_budget(n, max_k, budget);

        Self {
            n,
            max_k,
            scores,
            sorted_by_score,
            budget_per_k,
            rng: SmallRng::seed_from_u64(seed),
            phase: ProbePhase::Greedy { k: 0 },
        }
    }

    /// Distribute `total_budget` across k-values proportional to C(n,k), capped at C(n,k).
    fn distribute_budget(n: usize, max_k: usize, total_budget: usize) -> [usize; 6] {
        let mut budget = [0usize; 6];

        // k=0 always gets exactly 1 (the empty set).
        budget[0] = 1;
        let mut remaining = total_budget.saturating_sub(1);

        // Compute raw C(n,k) for k=1..=max_k to use as proportional weights.
        let mut cnk = [0usize; 6];
        let mut weight_sum: usize = 0;
        for k in 1..=max_k.min(5) {
            cnk[k] = comb(n, k);
            weight_sum += cnk[k];
        }

        if weight_sum == 0 {
            return budget;
        }

        // Proportional allocation, capped at actual C(n,k).
        for k in 1..=max_k.min(5) {
            if remaining == 0 {
                break;
            }
            let share =
                ((remaining as f64) * (cnk[k] as f64) / (weight_sum as f64)).ceil() as usize;
            budget[k] = share.min(cnk[k]).min(remaining);
            remaining = remaining.saturating_sub(budget[k]);
        }

        // If there's leftover budget (due to rounding), distribute to highest-weight k.
        if remaining > 0 {
            for k in (1..=max_k.min(5)).rev() {
                let can_add = cnk[k].saturating_sub(budget[k]);
                let add = can_add.min(remaining);
                budget[k] += add;
                remaining -= add;
                if remaining == 0 {
                    break;
                }
            }
        }

        budget
    }

    /// Fill `buf[0..k]` with the greedy combination: the top-k images by score,
    /// sorted by index ascending for consistency with `CombSliceIter` output.
    fn emit_greedy(&self, k: usize, buf: &mut [usize; 6]) {
        buf[..k].copy_from_slice(&self.sorted_by_score[..k]);
        buf[..k].sort_unstable();
    }

    /// Fill `buf[0..k]` with a GRASP-constructed random k-subset.
    ///
    /// At each of the k selection steps:
    /// 1. Compute the restricted candidate list (RCL) from eligible images whose score ≥
    ///    `max_score − α·(max_score − min_score)`.
    /// 2. Pick uniformly at random from the RCL.
    ///
    /// The resulting subset is sorted ascending by index.
    fn sample_grasp(&mut self, k: usize, buf: &mut [usize; 6]) {
        // Working copies: (original_index, score) pairs, shrinks as items are selected.
        let mut available: Vec<(usize, f64)> = (0..self.n).map(|i| (i, self.scores[i])).collect();

        for pos in 0..k {
            debug_assert!(!available.is_empty(), "available pool empty at pos {pos}");
            // Find score bounds among eligible items.
            let max_s = available
                .iter()
                .map(|&(_, s)| s)
                .fold(f64::NEG_INFINITY, f64::max);
            let min_s = available
                .iter()
                .map(|&(_, s)| s)
                .fold(f64::INFINITY, f64::min);
            let threshold = max_s - GRASP_ALPHA * (max_s - min_s);

            // Build the RCL (indices into `available`).
            let rcl: Vec<usize> = available
                .iter()
                .enumerate()
                .filter(|&(_, &(_, s))| s >= threshold)
                .map(|(idx, _)| idx)
                .collect();

            // Pick uniformly from RCL.
            let chosen_rcl_idx = if rcl.len() == 1 {
                0
            } else {
                self.rng.random_range(0..rcl.len())
            };
            let avail_idx = rcl[chosen_rcl_idx];
            let (image_idx, _) = available[avail_idx];
            buf[pos] = image_idx;
            available.swap_remove(avail_idx);
        }

        // Sort so output matches the ascending-index convention of CombSliceIter.
        buf[..k].sort_unstable();
    }

    /// Produce the next combination into `buf`, returning `Some(k)` or `None` when exhausted.
    /// API-compatible with `CombSliceIter::next_into`.
    #[inline]
    fn next_into(&mut self, buf: &mut [usize; 6]) -> Option<usize> {
        loop {
            match self.phase.clone() {
                ProbePhase::Greedy { k } => {
                    if k > self.max_k || k > self.n {
                        self.phase = ProbePhase::Done;
                        return None;
                    }
                    if k == 0 {
                        // The empty combination — nothing to fill.
                        // Transition: skip sampling for k=0 (only 1 possible combo).
                        self.phase = ProbePhase::Greedy { k: 1 };
                        return Some(0);
                    }
                    self.emit_greedy(k, buf);
                    let extra_samples = self.budget_per_k[k].saturating_sub(1);
                    self.phase = ProbePhase::Sampling {
                        k,
                        remaining: extra_samples,
                    };
                    return Some(k);
                }
                ProbePhase::Sampling { k, remaining } => {
                    if remaining == 0 {
                        self.phase = ProbePhase::Greedy { k: k + 1 };
                        continue;
                    }
                    self.sample_grasp(k, buf);
                    self.phase = ProbePhase::Sampling {
                        k,
                        remaining: remaining - 1,
                    };
                    return Some(k);
                }
                ProbePhase::Done => return None,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unified combination iterator (dispatches exhaustive vs probabilistic)
// ---------------------------------------------------------------------------

/// Combination iterator used by `ResidualProblem`.
///
/// When the `probabilistic_probing` feature is **disabled**, this is always the exhaustive
/// `CombSliceIter`.  When the feature is **enabled**, the constructor decides at runtime
/// whether exhaustive enumeration is cheap enough (total combos ≤ budget) or whether to
/// switch to `ProbabilisticCombIter`.
enum CombIter {
    Exhaustive(CombSliceIter),
    #[cfg(feature = "probabilistic_probing")]
    Probabilistic(ProbabilisticCombIter),
}

impl CombIter {
    /// Construct an exhaustive iterator (used when `probabilistic_probing` is disabled).
    #[cfg(not(feature = "probabilistic_probing"))]
    #[inline]
    fn exhaustive(n: usize, max_k: usize) -> Self {
        Self::Exhaustive(CombSliceIter::new(n, max_k))
    }

    /// Construct the best iterator for the given parameters.
    ///
    /// When `probabilistic_probing` is enabled and the exhaustive count exceeds the budget,
    /// a `ProbabilisticCombIter` is returned.  Otherwise falls back to exhaustive.
    ///
    /// The budget is read from the runtime-configurable `effective_probing_budget()`.
    /// If the runtime budget is `PROBING_BUDGET_EXHAUSTIVE` (`usize::MAX`), the
    /// exhaustive iterator is always returned regardless of combination count.
    #[cfg(feature = "probabilistic_probing")]
    fn for_residual(n: usize, max_k: usize, scores: Vec<f64>, seed: u64) -> Self {
        let budget = effective_probing_budget();
        if budget == PROBING_BUDGET_EXHAUSTIVE || total_combinations(n, max_k) <= budget {
            Self::Exhaustive(CombSliceIter::new(n, max_k))
        } else {
            Self::Probabilistic(ProbabilisticCombIter::new(n, max_k, scores, seed, budget))
        }
    }

    /// API-compatible `next_into`: delegates to the active variant.
    #[inline]
    fn next_into(&mut self, buf: &mut [usize; 6]) -> Option<usize> {
        match self {
            Self::Exhaustive(inner) => inner.next_into(buf),
            #[cfg(feature = "probabilistic_probing")]
            Self::Probabilistic(inner) => inner.next_into(buf),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod comb_iter_tests {
    use super::CombSliceIter;

    fn collect_all(n: usize, max_k: usize) -> Vec<Vec<usize>> {
        let mut iter = CombSliceIter::new(n, max_k);
        let mut buf = [0usize; 6];
        let mut out = Vec::new();
        while let Some(k) = iter.next_into(&mut buf) {
            out.push(buf[..k].to_vec());
        }
        out
    }

    #[test]
    fn test_empty_n() {
        // n=0: only the empty combination
        assert_eq!(collect_all(0, 5), vec![Vec::<usize>::new()]);
    }

    #[test]
    fn test_k0() {
        // max_k=0: only the empty combination
        assert_eq!(collect_all(3, 0), vec![Vec::<usize>::new()]);
    }

    #[test]
    fn test_small() {
        // n=3, max_k=2: 1 + 3 + 3 = 7 combinations
        let got = collect_all(3, 2);
        let expected: Vec<Vec<usize>> = vec![
            vec![],
            vec![0],
            vec![1],
            vec![2],
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn test_matches_itertools() {
        use itertools::Itertools;
        let n = 7;
        let expected: Vec<Vec<usize>> = (0..=5).flat_map(|k| (0..n).combinations(k)).collect();
        assert_eq!(collect_all(n, 5), expected);
    }
}

#[cfg(test)]
#[cfg(feature = "probabilistic_probing")]
mod probabilistic_iter_tests {
    use super::{ProbabilisticCombIter, comb, total_combinations};

    #[test]
    fn test_comb_basic() {
        assert_eq!(comb(5, 0), 1);
        assert_eq!(comb(5, 1), 5);
        assert_eq!(comb(5, 2), 10);
        assert_eq!(comb(5, 3), 10);
        assert_eq!(comb(5, 5), 1);
        assert_eq!(comb(5, 6), 0);
        assert_eq!(comb(10, 3), 120);
        assert_eq!(comb(14, 5), 2002);
    }

    #[test]
    fn test_total_combinations() {
        // C(4,0)+C(4,1)+C(4,2)+C(4,3)+C(4,4) = 1+4+6+4+1 = 16
        assert_eq!(total_combinations(4, 4), 16);
        assert_eq!(total_combinations(10, 5), 1 + 10 + 45 + 120 + 210 + 252);
    }

    #[test]
    fn test_budget_distribution_small() {
        // When C(n,≤max_k) ≤ budget, the distribution should cover all combos.
        let budget = ProbabilisticCombIter::distribute_budget(4, 3, 1000);
        // k=0: 1, k=1: C(4,1)=4, k=2: C(4,2)=6, k=3: C(4,3)=4 → total 15 ≤ 1000
        // All should be capped at C(n,k).
        assert_eq!(budget[0], 1);
        assert!(budget[1] <= 4);
        assert!(budget[2] <= 6);
        assert!(budget[3] <= 4);
    }

    #[test]
    fn test_budget_distribution_large() {
        // n=30, max_k=5 → C(30,≤5) = 174437, budget=1000
        let budget = ProbabilisticCombIter::distribute_budget(30, 5, 1000);
        let total: usize = budget.iter().sum();
        assert!(total <= 1000, "total budget {total} exceeds 1000");
        assert_eq!(budget[0], 1);
        // Higher k should get proportionally more budget since C(30,k) grows.
        assert!(budget[1] > 0);
        assert!(budget[2] > 0);
        assert!(budget[3] > 0);
    }

    #[test]
    fn test_probabilistic_yields_valid_subsets() {
        let n = 10;
        let max_k = 3;
        let scores: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut iter = ProbabilisticCombIter::new(n, max_k, scores, 42, 100);
        let mut buf = [0usize; 6];
        let mut count = 0;
        while let Some(k) = iter.next_into(&mut buf) {
            count += 1;
            // Check sorted ascending and in range.
            for i in 0..k {
                assert!(buf[i] < n, "index {} out of range", buf[i]);
                if i > 0 {
                    assert!(buf[i] > buf[i - 1], "not strictly ascending at pos {i}");
                }
            }
        }
        assert!(count > 0, "iterator should yield at least some combos");
        assert!(
            count <= 100 + 4,
            "count {count} should be ≤ budget + greedy combos"
        );
    }

    #[test]
    fn test_probabilistic_deterministic_with_same_seed() {
        let n = 15;
        let max_k = 4;
        let scores: Vec<f64> = (0..n).map(|i| (i as f64).sqrt() + 1.0).collect();

        let collect = |seed: u64| {
            let mut iter = ProbabilisticCombIter::new(n, max_k, scores.clone(), seed, 200);
            let mut buf = [0usize; 6];
            let mut out = Vec::new();
            while let Some(k) = iter.next_into(&mut buf) {
                out.push(buf[..k].to_vec());
            }
            out
        };

        let run1 = collect(99);
        let run2 = collect(99);
        assert_eq!(run1, run2, "same seed must produce identical sequences");
    }

    #[test]
    fn test_greedy_combo_always_present() {
        // The top-2 by score should be images 8,9 (highest scores).
        let n = 10;
        let scores: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut iter = ProbabilisticCombIter::new(n, 2, scores, 0, 50);
        let mut buf = [0usize; 6];
        let mut combos = Vec::new();
        while let Some(k) = iter.next_into(&mut buf) {
            combos.push(buf[..k].to_vec());
        }
        // k=0 greedy = empty
        assert!(combos.contains(&vec![]), "must contain empty combo");
        // k=1 greedy = [9] (highest score)
        assert!(combos.contains(&vec![9]), "must contain greedy k=1");
        // k=2 greedy = [8, 9] (top-2 by score, sorted by index)
        assert!(combos.contains(&vec![8, 9]), "must contain greedy k=2");
    }
}

use crate::{
    objectives::ObjectiveState,
    problem::SetCoverProblem,
    residual_solution::ResidualSolution,
    solution::{ImageSet, MergeableWithResidual},
    util::UnionIterator,
};

/// A flat 2D bit matrix: `nrows` rows of `ncols` bits stored in a single `Vec<usize>`.
/// Row `i` occupies words `i * wpr .. (i+1) * wpr`.  A single allocation replaces
/// `nrows` separate `FixedBitSet` heap allocations.
pub struct FlatBitMatrix {
    pub data: Vec<usize>,
    /// Words per row = `ncols.div_ceil(usize::BITS as usize)` (may be 0 if ncols=0)
    pub wpr: usize,
    pub nrows: usize,
    pub ncols: usize,
}

impl FlatBitMatrix {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        let wpr = ncols.div_ceil(usize::BITS as usize);
        Self {
            data: vec![0usize; nrows * wpr],
            wpr,
            nrows,
            ncols,
        }
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[usize] {
        let s = i * self.wpr;
        &self.data[s..s + self.wpr]
    }

    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [usize] {
        let s = i * self.wpr;
        let e = s + self.wpr;
        &mut self.data[s..e]
    }

    #[inline]
    pub fn set(&mut self, row: usize, bit: usize) {
        let w = bit / usize::BITS as usize;
        let b = bit % usize::BITS as usize;
        self.row_mut(row)[w] |= 1 << b;
    }

    #[inline]
    pub fn contains(&self, row: usize, bit: usize) -> bool {
        let w = bit / usize::BITS as usize;
        let b = bit % usize::BITS as usize;
        self.row(row)[w] & (1 << b) != 0
    }

    /// OR row `i` into `dst` (which must have length `self.wpr`).
    #[inline]
    pub fn or_row_into(&self, i: usize, dst: &mut [usize]) {
        let src = self.row(i);
        for (d, &s) in dst.iter_mut().zip(src) {
            *d |= s;
        }
    }

    /// Iterate set-bit indices in row `i`.
    pub fn row_ones(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        let row = self.row(i);
        row.iter().enumerate().flat_map(|(wi, &w)| {
            let base = wi * usize::BITS as usize;
            BitIter(w).map(move |b| base + b)
        })
    }

    /// Check whether the first `ncols` bits (taken from `data`) are all set.
    pub fn is_full_slice(data: &[usize], ncols: usize) -> bool {
        if ncols == 0 {
            return true;
        }
        let full_words = ncols / usize::BITS as usize;
        let remaining = ncols % usize::BITS as usize;
        data[..full_words].iter().all(|&w| w == usize::MAX)
            && (remaining == 0 || data[full_words] == (1usize << remaining) - 1)
    }
}

/// Iterator over set bit positions in a single `usize` word.
struct BitIter(usize);
impl Iterator for BitIter {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }
        let bit = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1;
        Some(bit)
    }
}

/// Precomputed data for efficiently computing merged CloudyArea objectives
/// on residual solutions, without materializing the full merged solution.
pub struct CloudyAreaData {
    /// Index of the CloudyArea objective in the objectives array
    pub objective_index: usize,
    /// Total cloudy area of the base solution (S \ R)
    pub base_cloudy_area: u64,
    /// Per candidate image (indexed by condensed image index):
    /// bitset of which *cloudy* elements this image covers clearly.
    /// Only elements that are cloudy in the base are tracked.
    /// Stored as a flat bit matrix (single allocation instead of m separate FixedBitSets).
    pub condensed_clear_parts: FlatBitMatrix,
    /// Areas of the cloudy elements, indexed by condensed cloudy-element index
    pub condensed_areas: Vec<u64>,
}

pub struct ResidualProblem<R, P, const D: usize>
where
    R: MergeableWithResidual<P, D> + Clone,
    P: SetCoverProblem<D>,
{
    /// Owned copy of original solution with unmodified images only
    pub unmodified_solution: R,
    /// Condensed indices corresponding to `removed_images` in `image_index_map`.
    /// Used to skip the "no-op" residual combination that would reconstruct the original solution.
    pub condensed_original_removed_images: FixedBitSet,
    /// Same set as `condensed_original_removed_images`, stored as a sorted `Vec<usize>`.
    /// Used for an allocation-free equality check inside the hot combination loop in `solve()` /
    /// `solve_next()` — avoids building a per-combination `FixedBitSet` just for the skip test.
    condensed_original_removed_images_vec: Vec<usize>,
    /// Map from condensed image index to original image index
    pub image_map_condensed_to_original: Vec<usize>,
    /// Map from condensed element index to original element index
    pub element_map_condensed_to_original: Vec<usize>,
    /// Condensed images as bitsets (each bitset represents which condensed elements the image covers).
    /// Stored as a flat bit matrix (single allocation instead of m separate FixedBitSets).
    condensed_images: FlatBitMatrix,
    /// Precomputed data for merged CloudyArea computation (None if no CloudyArea objective)
    pub cloudy_area_data: Option<CloudyAreaData>,
    /// Combination iterator over subsets of the condensed image indices.
    ///
    /// When `probabilistic_probing` is enabled and the condensed image count is large enough,
    /// this is a `CombIter::Probabilistic` that uses GRASP-biased sampling.  Otherwise it is
    /// the exhaustive `CombIter::Exhaustive` (zero-allocation `CombSliceIter`).
    combination_iter: CombIter,
    /// Precomputed per-image coverage scores for probabilistic probing.
    /// Stored so that `solve()` can create a fresh `CombIter` from the same data.
    #[cfg(feature = "probabilistic_probing")]
    image_scores: Vec<f64>,
    /// Deterministic seed derived from the removed-images set.
    #[cfg(feature = "probabilistic_probing")]
    probing_seed: u64,
    /// Phantom data to use P type parameter
    _phantom: std::marker::PhantomData<P>,
    /// Scratch FixedBitSet for is_set_cover_mut — avoids per-call heap allocation (capacity =
    /// num condensed elements, cleared before each use).
    scratch_covered: FixedBitSet,
    /// Scratch FixedBitSet for CloudyArea fast objective computation (capacity = num cloudy
    /// elements).
    scratch_patch_clear: FixedBitSet,
    /// Scratch buffer for MinResolution fast objective computation, indexed by condensed element
    /// index.  Pre-filled with `u64::MAX`; dirty entries are reset to `u64::MAX` after each use
    /// so the buffer is always in a clean state between calls.
    element_mins_scratch: Vec<u64>,
}

/// Compute the CloudyArea objective value for a patch (set of condensed image indices) using
/// precomputed `CloudyAreaData`.  Takes a mutable scratch `FixedBitSet` to avoid allocating
/// a new one on each call — caller is responsible for ensuring the capacity covers
/// `data.condensed_areas.len()`.
fn compute_cloudy_area_for_patch(
    data: &CloudyAreaData,
    condensed_indices: &[usize],
    scratch: &mut FixedBitSet,
) -> u64 {
    if data.condensed_areas.is_empty() {
        return 0;
    }
    scratch.clear();
    for &ci in condensed_indices {
        // OR flat-matrix row into scratch's backing words (avoids FixedBitSet per-row allocation)
        let scratch_words = scratch.as_mut_slice();
        data.condensed_clear_parts.or_row_into(ci, scratch_words);
    }
    let newly_cleared: u64 = scratch.ones().map(|e| data.condensed_areas[e]).sum();
    data.base_cloudy_area.saturating_sub(newly_cleared)
}

impl<R, P, const D: usize> ResidualProblem<R, P, D>
where
    R: MergeableWithResidual<P, D> + Clone,
    P: SetCoverProblem<D>,
{
    /// Creates a new residual problem from a solution with removed images.
    ///
    /// # Panics
    ///
    /// Panics if a removed image index is not found in the constructed image index map.
    /// This should never happen if the inputs are valid.
    #[must_use]
    pub fn new(
        unmodified_solution: R,
        removed_images: &[usize],
        addition_candidates: &[usize],
        uncovered_elements_indices: Vec<usize>,
        problem: &P,
    ) -> Self {
        debug!("######################################################");
        debug!("######## RESIDUAL PROBLEM removed images: {removed_images:?} ######");
        debug!("######## RESIDUAL PROBLEM addition candidates: {addition_candidates:?} ######");
        debug!("######## base: {unmodified_solution:?} ########");
        debug!("######################################################");

        // Build element index map (condensed -> original)
        let element_map_condensed_to_original = uncovered_elements_indices;

        // Create reverse map (original -> condensed) for fast lookup.
        // NOTE: element_map_condensed_to_original is sorted (ascending) because
        // uncovered_elements() iterates a FixedBitSet from lowest to highest bit.
        // We therefore use binary search instead of a HashMap for O(log n) lookups
        // without any heap allocation or hashing overhead.
        #[cfg(debug_assertions)]
        debug_assert!(
            element_map_condensed_to_original
                .windows(2)
                .all(|w| w[0] < w[1]),
            "element_map_condensed_to_original must be sorted ascending"
        );

        // Build image index map as a UNION of both ordered lists.
        // NOTE: `util::UnionIterator::union()` is a merge-based union, so it assumes:
        // - both inputs are sorted ascending
        // - both inputs are unique (set semantics)
        // Keep ordering stable for deterministic behavior.
        #[cfg(debug_assertions)]
        {
            debug_assert!(
                removed_images.windows(2).all(|w| w[0] < w[1]),
                "removed_images must be a sorted unique set"
            );
            debug_assert!(
                addition_candidates.windows(2).all(|w| w[0] < w[1]),
                "addition_candidates must be a sorted unique set"
            );
        }
        let image_map_condensed_to_original: Vec<usize> = removed_images
            .iter()
            .copied()
            .union(addition_candidates.iter().copied())
            .collect();

        // Build reverse map (original image index -> condensed index) for removed-images bitset.
        let image_map_original_to_condensed: HashMap<usize, usize> =
            image_map_condensed_to_original
                .iter()
                .enumerate()
                .map(|(condensed_idx, &original_idx)| (original_idx, condensed_idx))
                .collect();

        // Bitset of condensed indices corresponding to the original `removed_images`.
        let condensed_original_removed_images_vec: Vec<usize> = removed_images
            .iter()
            .map(|&img| {
                image_map_original_to_condensed
                    .get(&img)
                    .copied()
                    .expect("removed image must be in image_map_original_to_condensed")
            })
            .collect();
        // `condensed_original_removed_images_vec` is already sorted because
        // `removed_images` is sorted and the condensed index mapping preserves order.
        let condensed_original_removed_images = condensed_original_removed_images_vec
            .iter()
            .copied()
            .collect::<FixedBitSet>();

        // Build condensed images as a flat bit matrix (single allocation instead of m FixedBitSets)
        let num_condensed_elements = element_map_condensed_to_original.len();
        let mut condensed_images = FlatBitMatrix::new(
            image_map_condensed_to_original.len(),
            num_condensed_elements,
        );
        // Build a flat O(1) reverse map: original_element_idx -> condensed_idx.
        // universe_size is ~4421 for typical SIMS instances, so this is ~17 KB —
        // fits in L1 cache and is far cheaper than O(log n) binary search per element.
        let universe_size = problem.universe_size();
        let mut element_reverse = vec![u32::MAX; universe_size];
        for (ci, &orig) in element_map_condensed_to_original.iter().enumerate() {
            element_reverse[orig] = ci as u32;
        }
        for (img_idx, &original_img_idx) in image_map_condensed_to_original.iter().enumerate() {
            for original_elem_idx in problem.image_elements(original_img_idx) {
                let ce = element_reverse[original_elem_idx];
                if ce != u32::MAX {
                    condensed_images.set(img_idx, ce as usize);
                }
            }
        }

        // Build CloudyAreaData if the problem has a CloudyArea objective
        let cloudy_area_data = Self::build_cloudy_area_data(
            &unmodified_solution,
            &image_map_condensed_to_original,
            problem,
        );

        let cloudy_scratch_size = cloudy_area_data
            .as_ref()
            .map_or(0, |d| d.condensed_areas.len());
        let m = condensed_images.nrows;

        // --- Build the combination iterator ----------------------------------
        #[cfg(feature = "probabilistic_probing")]
        let image_scores: Vec<f64> = (0..m)
            .map(|img| condensed_images.row_ones(img).count() as f64 + 1.0)
            .collect();

        #[cfg(feature = "probabilistic_probing")]
        let probing_seed: u64 =
            removed_images
                .iter()
                .fold(0x517c_c1b7_2722_0a95_u64, |acc, &idx| {
                    acc.wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(idx as u64)
                });

        #[cfg(feature = "probabilistic_probing")]
        let combination_iter = CombIter::for_residual(m, 5, image_scores.clone(), probing_seed);

        #[cfg(not(feature = "probabilistic_probing"))]
        let combination_iter = CombIter::exhaustive(m, 5);

        Self {
            unmodified_solution,
            condensed_original_removed_images,
            condensed_original_removed_images_vec,
            image_map_condensed_to_original,
            element_map_condensed_to_original,
            condensed_images,
            cloudy_area_data,
            combination_iter,
            _phantom: std::marker::PhantomData,
            scratch_covered: FixedBitSet::with_capacity(num_condensed_elements),
            scratch_patch_clear: FixedBitSet::with_capacity(cloudy_scratch_size),
            element_mins_scratch: vec![u64::MAX; num_condensed_elements],
            #[cfg(feature = "probabilistic_probing")]
            image_scores,
            #[cfg(feature = "probabilistic_probing")]
            probing_seed,
        }
    }

    /// Build CloudyAreaData by computing base clear coverage and condensing
    /// the clear parts of candidate images down to only the cloudy elements.
    fn build_cloudy_area_data(
        unmodified_solution: &R,
        image_map_condensed_to_original: &[usize],
        problem: &P,
    ) -> Option<CloudyAreaData> {
        // Find the CloudyArea objective index and extract its data
        let (objective_index, clear_images, areas) = problem
            .objectives()
            .iter()
            .enumerate()
            .find_map(|(i, obj)| match obj {
                ObjectiveState::CloudyArea {
                    clear_images,
                    areas,
                    ..
                } => Some((i, clear_images, areas)),
                _ => None,
            })?;

        let universe_size = problem.universe_size();

        // Compute base clear coverage: union of clear_images for all selected images in base
        let mut base_clear = FixedBitSet::with_capacity(universe_size);
        for img_idx in unmodified_solution.selected_images() {
            if let Some(img_clear) = clear_images.get(img_idx) {
                base_clear.union_with(img_clear);
            }
        }

        // Identify cloudy elements and their areas
        // cloudy_element_map[condensed_cloudy_idx] = original_element_idx
        let cloudy_elements: Vec<usize> = (0..universe_size)
            .filter(|&e| !base_clear.contains(e))
            .collect();

        if cloudy_elements.is_empty() {
            // Everything is clear in the base -- no improvement possible
            return Some(CloudyAreaData {
                objective_index,
                base_cloudy_area: 0,
                condensed_clear_parts: FlatBitMatrix::new(image_map_condensed_to_original.len(), 0),
                condensed_areas: Vec::new(),
            });
        }

        // Reverse map: original element -> condensed cloudy index
        let mut original_to_cloudy = vec![usize::MAX; universe_size];
        for (condensed_idx, &original_idx) in cloudy_elements.iter().enumerate() {
            original_to_cloudy[original_idx] = condensed_idx;
        }

        let base_cloudy_area: u64 = cloudy_elements.iter().map(|&e| areas[e]).sum();
        let condensed_areas: Vec<u64> = cloudy_elements.iter().map(|&e| areas[e]).collect();

        // For each candidate image, build a condensed clear bitset over cloudy elements only
        let num_cloudy = cloudy_elements.len();
        let mut condensed_clear_parts =
            FlatBitMatrix::new(image_map_condensed_to_original.len(), num_cloudy);
        for (img_idx, &original_img_idx) in image_map_condensed_to_original.iter().enumerate() {
            if let Some(img_clear) = clear_images.get(original_img_idx) {
                for elem in img_clear.ones() {
                    let condensed = original_to_cloudy[elem];
                    if condensed != usize::MAX {
                        condensed_clear_parts.set(img_idx, condensed);
                    }
                }
            }
        }

        Some(CloudyAreaData {
            objective_index,
            base_cloudy_area,
            condensed_clear_parts,
            condensed_areas,
        })
    }

    /// Compute the merged CloudyArea objective for a residual solution.
    /// Returns the cloudy area of (base union patch), using precomputed condensed data.
    /// The residual solution stores condensed image indices.
    pub fn compute_merged_cloudy_area(
        &self,
        residual_solution: &ResidualSolution<D>,
    ) -> Option<u64> {
        let data = self.cloudy_area_data.as_ref()?;

        if data.condensed_areas.is_empty() {
            return Some(0);
        }

        // OR together the condensed clear parts for all selected images in the patch
        let wpr = data.condensed_clear_parts.wpr;
        let mut patch_clear_words = vec![0usize; wpr];
        for &condensed_img_idx in &residual_solution.selected_images {
            data.condensed_clear_parts
                .or_row_into(condensed_img_idx, &mut patch_clear_words);
        }

        // Sum areas of elements newly cleared by the patch
        let newly_cleared_area: u64 = patch_clear_words
            .iter()
            .enumerate()
            .flat_map(|(wi, &w)| BitIter(w).map(move |b| wi * usize::BITS as usize + b))
            .map(|e| data.condensed_areas[e])
            .sum();

        Some(data.base_cloudy_area - newly_cleared_area)
    }

    /// Check if the given selection of condensed images forms a set cover
    /// Uses efficient flat bit matrix operations
    #[must_use]
    pub fn is_set_cover(&self, selected_images: &FixedBitSet) -> bool {
        let wpr = self.condensed_images.wpr;
        let ncols = self.condensed_images.ncols;
        let mut covered = vec![0usize; wpr];
        for img_idx in selected_images.ones() {
            self.condensed_images.or_row_into(img_idx, &mut covered);
            if FlatBitMatrix::is_full_slice(&covered, ncols) {
                return true;
            }
        }
        FlatBitMatrix::is_full_slice(&covered, ncols)
    }

    /// Get images that cover a specific element
    #[must_use]
    pub fn images_covering_element(&self, element_idx: usize) -> Vec<usize> {
        (0..self.condensed_images.nrows)
            .filter(|&img_idx| self.condensed_images.contains(img_idx, element_idx))
            .collect()
    }

    /// Like `is_set_cover` but reuses `self.scratch_covered` to avoid a heap allocation on
    /// every call.  The scratch buffer is cleared before use so state never leaks between calls.
    #[must_use]
    fn is_set_cover_mut(&mut self, selected_images: &FixedBitSet) -> bool {
        self.scratch_covered.clear();
        for img_idx in selected_images.ones() {
            {
                let scratch = self.scratch_covered.as_mut_slice();
                self.condensed_images.or_row_into(img_idx, scratch);
            }
            if self.scratch_covered.is_full() {
                return true;
            }
        }
        self.scratch_covered.is_full()
    }

    /// Variant of `is_set_cover_mut` that accepts a sorted **slice** of condensed image indices
    /// instead of a `FixedBitSet`.  Avoids heap-allocating a `FixedBitSet` on every call inside
    /// the hot combination enumeration loop.  Includes an early-exit once all elements are
    /// covered so that longer combinations (k=4,5) short-circuit quickly.
    #[must_use]
    #[inline]
    fn is_set_cover_slice(&mut self, selected_images: &[usize]) -> bool {
        self.scratch_covered.clear();
        for &img_idx in selected_images {
            {
                let scratch = self.scratch_covered.as_mut_slice();
                self.condensed_images.or_row_into(img_idx, scratch);
            }
            if self.scratch_covered.is_full() {
                return true;
            }
        }
        self.scratch_covered.is_full()
    }

    /// Compute objectives for a patch described by `condensed_indices` without allocating a
    /// temporary solution or universe-sized vectors.
    ///
    /// * `TotalCost` / `MaxIncidenceAngle` — direct index lookups, zero allocations.
    /// * `CloudyArea` — uses precomputed `CloudyAreaData` and `scratch_patch_clear`.
    /// * `MinResolution` — operates in condensed-element space (much smaller than universe)
    ///   using `element_mins_scratch`; dirty entries are reset after each call.
    fn compute_residual_objectives_fast(
        &mut self,
        condensed_indices: &[usize],
        problem: &P,
    ) -> pareto::Objectives<D> {
        let mut objectives = [0u64; D];
        for (i, obj) in problem.objectives().iter().enumerate() {
            objectives[i] = match obj {
                ObjectiveState::TotalCost { costs, .. } => condensed_indices
                    .iter()
                    .map(|&ci| costs[self.image_map_condensed_to_original[ci]])
                    .sum(),

                ObjectiveState::CloudyArea { .. } => {
                    // Borrow the two fields separately so the compiler sees disjoint borrows.
                    let data_opt = self.cloudy_area_data.as_ref();
                    match data_opt {
                        None => 0,
                        Some(data) if data.condensed_areas.is_empty() => 0,
                        Some(data) => compute_cloudy_area_for_patch(
                            data,
                            condensed_indices,
                            &mut self.scratch_patch_clear,
                        ),
                    }
                }

                ObjectiveState::MinResolution { resolutions, .. } => {
                    // Borrow fields individually so the compiler allows &mut element_mins_scratch
                    // alongside the immutable borrow of condensed_images and image_map.
                    let img_map = &self.image_map_condensed_to_original;
                    let scratch = &mut self.element_mins_scratch;

                    // Fill phase: write minimum resolution per condensed element.
                    for &ci in condensed_indices {
                        let res = resolutions[img_map[ci]];
                        for elem_ci in self.condensed_images.row_ones(ci) {
                            if res < scratch[elem_ci] {
                                scratch[elem_ci] = res;
                            }
                        }
                    }
                    // Collect-and-reset phase: sum filled entries, resetting each on first visit
                    // so duplicate coverage across images doesn't double-count.
                    let mut sum = 0u64;
                    for &ci in condensed_indices {
                        for elem_ci in self.condensed_images.row_ones(ci) {
                            let v = scratch[elem_ci];
                            if v != u64::MAX {
                                sum += v;
                                scratch[elem_ci] = u64::MAX;
                            }
                        }
                    }
                    sum
                }

                ObjectiveState::MaxIncidenceAngle {
                    incidence_angles, ..
                } => condensed_indices
                    .iter()
                    .map(|&ci| incidence_angles[self.image_map_condensed_to_original[ci]])
                    .max()
                    .unwrap_or(0),
            };
        }
        objectives
    }

    pub fn solve_with_backtracing<'a, S: ParetoFront<'a, ResidualSolution<D>> + Default>(
        &mut self,
        problem: &P,
        timer: &crate::timer::Timer,
    ) -> Vec<ResidualSolution<D>> {
        let mut non_dominated_residual_set: S = S::default();

        // Build list of images covering each element for cartesian product
        let element_images: Vec<Vec<usize>> = (0..self.element_map_condensed_to_original.len())
            .map(|elem_idx| self.images_covering_element(elem_idx))
            .collect();

        let element_images_refs: Vec<_> = element_images.iter().map(|v| v.iter()).collect();

        for cover in element_images_refs.into_iter().multi_cartesian_product() {
            let mut unique_cover: Vec<usize> = cover.into_iter().copied().collect();
            unique_cover.sort_unstable();
            unique_cover.dedup();

            // Keep condensed indices for ResidualSolution
            let residual_solution = ResidualSolution::from_selected_images_condensed(
                &unique_cover,
                &self.image_map_condensed_to_original,
                problem,
                timer,
            );

            let was_added = non_dominated_residual_set.try_insert(&residual_solution);

            trace!("#####################################################");
            trace!(
                "######### RESIDUAL: OBJECTIVES {:?} | IMAGES {:?} {} #########################",
                residual_solution.objectives(),
                residual_solution.selected_images().collect::<Vec<_>>(),
                if was_added { "ADDED" } else { "NOT ADDED" }
            );
        }

        let solutions_iter: Vec<ResidualSolution<D>> =
            non_dominated_residual_set.into_iter().collect();

        trace!("*****************************************************");
        for solution in &solutions_iter {
            trace!(
                "****** NONDOMINANT RESIDUAL: OBJECTIVES {:?} | IMAGES {:?} ******",
                solution.objectives(),
                solution.selected_images().collect::<Vec<_>>()
            );
        }
        trace!("*****************************************************");

        solutions_iter
    }

    // // Check whether selected images cover all elements
    // fn do_selected_images_cover(
    //     &self,
    //     selected_images: &[usize],
    //     coverage_bitmaps: &[ElementSubset],
    //     all_elements_mask: ElementSubset,
    // ) -> bool {
    //     selected_images.iter().fold(ElementSubset::default(), |acc, &image_index| {
    //         acc | coverage_bitmaps[image_index]
    //     }) == all_elements_mask
    // }

    /*
    pub fn solve_with_bitmaps(&self) -> MergedSolutionIter {
        const K: usize = 3;
        const MAX_IMAGES: usize = 20;
        const MAX_ELEMENTS: usize = 128;
        type ImageSubset = BitArr!(for MAX_IMAGES);
        type ElementSubset = BitArr!(for MAX_ELEMENTS);

        let images_count = self.all_images.len();
        let elements_count = self.uncovered_elements.len();

        let all_elements_mask: ElementSubset = ElementSubset::default();
        all_elements_mask[0..elements_count].fill(true);
        let all_images_mask: ImageSubset = ImageSubset::default();
        all_images_mask[0..images_count].fill(true);

        let coverage_bitmaps: Vec<ElementSubset> = self.uncovered_elements.iter().map(|element| {
            let mut bitmap = ElementSubset::default();
            element.images.iter().for_each(|&image_index| {
                bitmap.set(image_index, true);
            });
            bitmap
        }).collect();

        let clear_part_bitmaps: Vec<ElementSubset> = vec![ElementSubset::default(); self.uncovered_elements.len()];
        self.all_images.iter().enumerate().for_each(|(image_index, image)| {
            image.clear_parts.iter().for_each(|&clear_part| {
                clear_part_bitmaps[clear_part].set(image_index, true);
            })
        });

        let mut non_dominated_residual_set: BTreeSet<ResidualSolution<D>> = BTreeSet::new();

        let mut subset_storage = [bitarr![0; MAX_IMAGES]; K];
        let mut current_indices = [0; K];
        let mut current_recursion_level: usize = 0;
        let mut recursive_subsets: [&mut BitSlice; K] = subset_storage.iter_mut().map(|subset| subset.get_mut(0..images_count).unwrap()).collect();

        loop {
            if current_indices[current_recursion_level] == 0 {
                // Select first element empty subset
                recursive_subsets[current_recursion_level].set(current_indices[current_recursion_level], true);
            } else {
                // Select next element in subset
                recursive_subsets[current_recursion_level].set(current_indices[current_recursion_level]-1, false);
                recursive_subsets[current_recursion_level].set(current_indices[current_recursion_level], true);
            }
            current_indices[current_recursion_level] += 1;

            let selected_images: Vec<usize> = recursive_subsets[current_recursion_level].iter().enumerate().filter_map(|(i, &is_selected)| {
                if is_selected {
                    Some(i)
                } else {
                    None
                }
            }).collect();

            if self.do_selected_images_cover(&selected_images, &coverage_bitmaps, &all_elements_mask) {
                let residual_solution = ResidualSolution::<D>::from_selected_images(selected_images.clone(), self, timer);

                if !non_dominated_residual_set.contains(&residual_solution) {
                // Simple dominance check - only add if not dominated by existing solutions
                let is_dominated = non_dominated_residual_set
                    .iter()
                    .any(|existing| residual_solution.is_dominated_by(existing.objectives()));

                let was_added = if !is_dominated {
                    // Remove solutions dominated by the new one
                    non_dominated_residual_set.retain(|existing| !existing.is_dominated_by(residual_solution.objectives()));
                    non_dominated_residual_set.insert(residual_solution.clone());
                    true
                } else {
                    false
                };
                    trace!("#####################################################");
                    trace!(
                        "######### RESIDUAL: OBJECTIVES {:?} | IMAGES {:?} {} #########################",
                        residual_solution.objectives,
                        residual_solution.selected_images,
                        if was_added { "ADDED" } else { "NOT ADDED" }
                    );
                }
            }
            break;
        }

        MergedSolutionIter {
            unmodified_solution: &self.unmodified_solution,
            solutions_iter: Vec::new(),
            residual_problem: self,
            problem: self.problem,
        }
    }
    */

    /// Get the next valid residual solution from the combination iterator.
    /// Returns `None` when all combinations have been exhausted.
    pub fn solve_next(
        &mut self,
        problem: &P,
        timer: &crate::timer::Timer,
    ) -> Option<ResidualSolution<D>> {
        // Use a local stack buffer so that the `&[usize]` slice does NOT borrow `self`.
        // This lets is_set_cover_slice / compute_residual_objectives_fast borrow `self`
        // mutably at the same time without a borrow-checker conflict.
        let mut buf = [0usize; 6];
        while let Some(k) = self.combination_iter.next_into(&mut buf) {
            let combination: &[usize] = &buf[..k];

            // Skip-check: compare against the pre-stored sorted vec — no FixedBitSet needed.
            if combination == self.condensed_original_removed_images_vec.as_slice() {
                continue;
            }

            if !self.is_set_cover_slice(combination) {
                continue;
            }

            let objectives = self.compute_residual_objectives_fast(combination, problem);
            let residual_solution = ResidualSolution {
                // Allocate a Vec only for the small fraction of combinations that are valid
                // set covers — the hot rejection path never reaches this `.to_vec()` call.
                selected_images: combination.to_vec(),
                objectives,
                timestamp: timer.elapsed(),
            };

            trace!(
                "RESIDUAL: OBJ {:?} | IMG {:?}",
                residual_solution.objectives(),
                residual_solution.selected_images().collect::<Vec<_>>(),
            );

            return Some(residual_solution);
        }

        // All combinations exhausted
        None
    }

    #[tracing::instrument(skip(self, problem, timer), level = "debug")]
    pub fn solve<'a, S: ParetoFront<'a, ResidualSolution<D>> + Default>(
        &'a mut self,
        problem: &'a P,
        timer: &crate::timer::Timer,
        partial_trackers: R::Trackers,
    ) -> MergedSolutionIter<'a, R, P, D> {
        use tracing::debug_span;

        let init_span = debug_span!("initialize_residual_solve");
        let _init_guard = init_span.enter();

        let m = self.image_map_condensed_to_original.len();

        // Build the combination iterator — probabilistic when the feature is enabled and
        // the search space is large enough, otherwise exhaustive (zero-allocation).
        #[cfg(feature = "probabilistic_probing")]
        let mut comb_iter =
            CombIter::for_residual(m, 5, self.image_scores.clone(), self.probing_seed);

        #[cfg(not(feature = "probabilistic_probing"))]
        let mut comb_iter = CombIter::exhaustive(m, 5);

        let mut comb_buf = [0usize; 6];

        let mut non_dominated_residual_set: S = S::default();

        let enumerate_span = debug_span!("enumerate_combinations");
        let _enum_guard = enumerate_span.enter();

        while let Some(k) = comb_iter.next_into(&mut comb_buf) {
            let image_combination = &comb_buf[..k];

            // Skip-check: compare against the pre-stored sorted vec — no FixedBitSet allocation.
            if image_combination == self.condensed_original_removed_images_vec.as_slice() {
                trace!("Skipping image combination as it is equal to original one");
                continue;
            }

            // Set-cover check: use the slice variant to avoid per-combination FixedBitSet
            // construction.  Includes early-exit once full coverage is reached.
            if !self.is_set_cover_slice(image_combination) {
                continue;
            }

            // Build objectives using fast condensed-space paths.  Allocate the Vec for
            // selected_images only here — this path is taken for the small fraction of
            // combinations that form valid set covers.
            let objectives = self.compute_residual_objectives_fast(image_combination, problem);
            let residual_solution = ResidualSolution {
                selected_images: image_combination.to_vec(),
                objectives,
                timestamp: timer.elapsed(),
            };

            // Add to Pareto front
            let was_added = non_dominated_residual_set.try_insert(&residual_solution);

            trace!(
                "RESIDUAL: OBJ {:?} | IMG {:?} {}",
                residual_solution.objectives(),
                residual_solution.selected_images().collect::<Vec<_>>(),
                if was_added { "ADDED" } else { "SKIP" }
            );
        }

        let solutions_iter: Vec<ResidualSolution<D>> =
            non_dominated_residual_set.into_iter().collect();

        MergedSolutionIter {
            unmodified_solution: &self.unmodified_solution,
            solutions_iter,
            residual_problem: self,
            problem,
            partial_trackers,
        }
    }
}

pub struct MergedSolutionIter<'a, R, P, const D: usize>
where
    R: MergeableWithResidual<P, D> + Clone,
    P: SetCoverProblem<D>,
{
    pub unmodified_solution: &'a R,
    pub solutions_iter: Vec<ResidualSolution<D>>,
    pub residual_problem: &'a ResidualProblem<R, P, D>,
    pub problem: &'a P,
    pub partial_trackers: R::Trackers,
}
/*
pub struct ResidualSolutionIter<'b, 'a, R, P, const D: usize, S>
where
    R: MergeableWithResidual<P, D> + Clone,
    P: SetCoverProblem<D>,
    S: ParetoFront<'a, ResidualSolution<D>> + Default,
{
    residual_problem: &'b ResidualProblem<'a, R, P, D>,
    timer: &'b crate::timer::Timer,
    partial_trackers: R::Trackers,
    images_indices: Vec<usize>,
    combination_iter: Option<Box<dyn Iterator<Item = Vec<&'b usize>> + 'b>>,
    non_dominated_set: S,
    non_dominated_iter: Option<Box<dyn Iterator<Item = ResidualSolution<D>> + 'a>>,
    unmodified_solution: &'a R,
    problem: &'a P,
}

impl<'b, 'a, R, P, const D: usize, S> Iterator for ResidualSolutionIter<'b, 'a, R, P, D, S>
where
    R: MergeableWithResidual<P, D> + Clone,
    P: SetCoverProblem<D>,
    S: ParetoFront<'a, ResidualSolution<D>> + Default,
{
    type Item = R;

    fn next(&mut self) -> Option<Self::Item> {
        // If we already have non-dominated solutions collected, yield from them
        if let Some(iter) = &mut self.non_dominated_iter {
            if let Some(residual_solution) = iter.next() {
                let mut new_solution = self.unmodified_solution.clone();
                new_solution.merge_residual_solution(
                    &residual_solution,
                    self.residual_problem,
                    self.problem,
                    self.partial_trackers.clone(),
                );
                return Some(new_solution);
            }
        }

        // Initialize combination iterator on first call
        if self.combination_iter.is_none() {
            let combs_0_to_5 = (0..=5).flat_map(|i| self.images_indices.iter().combinations(i));
            self.combination_iter = Some(Box::new(combs_0_to_5));
        }

        // Enumerate combinations and build non-dominated set
        if let Some(iter) = &mut self.combination_iter {
            for image_combination in iter.by_ref() {
                // Check if this combination matches the initially selected images (removed candidates)
                let test_bitset: FixedBitSet = image_combination.iter().copied().copied().collect();
                if test_bitset == self.residual_problem.condensed_original_removed_images {
                    trace!("Skipping image combination as it is equal to original one");
                    continue;
                }

                let selected: FixedBitSet = image_combination.iter().copied().copied().collect();
                if !self.residual_problem.is_set_cover(&selected) {
                    continue;
                }

                // Keep condensed indices for ResidualSolution
                let condensed_images: Vec<usize> = image_combination
                    .iter()
                    .map(|&&condensed_idx| condensed_idx)
                    .collect();
                let residual_solution = ResidualSolution::from_selected_images_condensed(
                    &condensed_images,
                    &self.residual_problem.image_index_map,
                    self.problem,
                    self.timer,
                );

                let was_added = self.non_dominated_set.try_insert(&residual_solution);

                trace!("#####################################################");
                trace!(
                    "######### RESIDUAL: OBJECTIVES {:?} | IMAGES {:?} {} #########################",
                    residual_solution.objectives(),
                    residual_solution.selected_images().collect::<Vec<_>>(),
                    if was_added { "ADDED" } else { "NOT ADDED" }
                );
            }
        }

        // All combinations processed, now yield from non-dominated set
        trace!("*****************************************************");
        trace!("****** Processing non-dominated solutions ******");

        let solutions: Vec<ResidualSolution<D>> = std::mem::take(&mut self.non_dominated_set).into_iter().collect();
        for solution in &solutions {
            trace!(
                "****** NONDOMINANT RESIDUAL: OBJECTIVES {:?} | IMAGES {:?} ******",
                solution.objectives(),
                solution.selected_images().collect::<Vec<_>>()
            );
        }
        trace!("*****************************************************");

        self.non_dominated_iter = Some(Box::new(solutions.into_iter()));

        // Yield first solution from the iterator
        if let Some(iter) = &mut self.non_dominated_iter {
            if let Some(residual_solution) = iter.next() {
                let mut new_solution = self.unmodified_solution.clone();
                new_solution.merge_residual_solution(
                    &residual_solution,
                    self.residual_problem,
                    self.problem,
                    self.partial_trackers.clone(),
                );
                return Some(new_solution);
            }
        }

        None
    }
}
*/

impl<R, P, const D: usize> Iterator for MergedSolutionIter<'_, R, P, D>
where
    R: MergeableWithResidual<P, D> + Clone,
    P: SetCoverProblem<D>,
{
    type Item = R;

    fn next(&mut self) -> Option<Self::Item> {
        let residual_solution = self.solutions_iter.pop()?;
        let mut new_solution = self.unmodified_solution.clone();
        new_solution.merge_residual_solution(
            &residual_solution,
            self.residual_problem,
            self.problem,
            &mut self.partial_trackers,
        );
        return Some(new_solution);
    }
}
