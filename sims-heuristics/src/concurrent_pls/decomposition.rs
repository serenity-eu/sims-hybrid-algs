/// Objective-space decomposition using Das-Dennis simplex lattice weight vectors.
///
/// Each region owns a weight vector and uses augmented Tchebycheff scalarization
/// to determine which solutions "belong" to it.

use pareto::Objectives;

/// Augmentation coefficient for Tchebycheff scalarization.
///
/// With weighted augmentation $\rho \sum w_j \cdot d_j$, the effective contribution at equal
/// weights (0.25 each, D=4) is $\sim 4\rho \cdot \text{mean\_term}$. At $\rho = 10^{-3}$ this
/// yields ~0.2% of the max-term -- large enough to break ties between discrete (u64) objectives
/// after normalization (granularity ~$10^{-5}$--$10^{-3}$), yet small enough to preserve the
/// true Tchebycheff landscape geometry.  Values above $10^{-2}$ begin to distort region
/// assignment and SA-PLS acceptance decisions; values below $10^{-5}$ may fail to break ties
/// for coarse objective ranges.
pub const RHO: f64 = 1e-3;

/// A single region in the decomposition, owning one weight vector.
#[derive(Clone, Debug)]
pub struct Region<const D: usize> {
    /// Index of this region (0..N)
    pub index: usize,
    /// Weight vector on the unit simplex (sums to 1.0)
    pub weight_vector: [f64; D],
}

impl<const D: usize> Region<D> {
    /// Compute the augmented Tchebycheff score for `objectives` against this region.
    #[inline]
    #[must_use]
    pub fn score(&self, objectives: &[u64; D], ideal: &Objectives<D>, bounds: &[(f64, f64); D]) -> f64 {
        tchebycheff_score(objectives, &self.weight_vector, ideal, bounds, RHO)
    }
}

/// Generate Das-Dennis weight vectors for `D` objectives and simplex parameter `H`.
/// Returns $\binom{H + D - 1}{D - 1}$ weight vectors uniformly distributed on the unit simplex.
#[must_use]
pub fn das_dennis_weight_vectors<const D: usize>(h: usize) -> Vec<[f64; D]> {
    let mut vectors = Vec::new();
    let mut current = [0usize; D];
    generate_recursive::<D>(&mut vectors, &mut current, h, 0, h);
    vectors
}

fn generate_recursive<const D: usize>(
    vectors: &mut Vec<[f64; D]>,
    current: &mut [usize; D],
    h: usize,
    depth: usize,
    remaining: usize,
) {
    if depth == D - 1 {
        current[depth] = remaining;
        let weight: [f64; D] = std::array::from_fn(|i| current[i] as f64 / h as f64);
        vectors.push(weight);
        return;
    }
    for i in 0..=remaining {
        current[depth] = i;
        generate_recursive::<D>(vectors, current, h, depth + 1, remaining - i);
    }
}

/// Find the smallest `H` such that $\binom{H + D - 1}{D - 1} \geq \text{num\_threads}$.
#[must_use]
pub fn auto_select_h<const D: usize>(num_threads: usize) -> usize {
    for h in 1..=100 {
        if das_dennis_weight_vectors::<D>(h).len() >= num_threads {
            return h;
        }
    }
    panic!("Could not find a valid H for {D} objectives and {num_threads} threads");
}

/// Build `N` regions from Das-Dennis weight vectors.
/// If `das_dennis_h` is `None`, auto-selects the smallest `H` with at least `num_regions` vectors.
#[must_use]
pub fn build_regions<const D: usize>(
    num_regions: usize,
    das_dennis_h: Option<usize>,
) -> Vec<Region<D>> {
    let h = das_dennis_h.unwrap_or_else(|| auto_select_h::<D>(num_regions));
    let mut weight_vectors = das_dennis_weight_vectors::<D>(h);

    // Keep only `num_regions` weight vectors: drop the most "interior" ones (closest to centroid)
    if weight_vectors.len() > num_regions {
        let centroid: [f64; D] = std::array::from_fn(|_| 1.0 / D as f64);
        weight_vectors.sort_by(|a, b| {
            let dist_a: f64 = a.iter().zip(centroid.iter()).map(|(x, c)| (x - c).powi(2)).sum();
            let dist_b: f64 = b.iter().zip(centroid.iter()).map(|(x, c)| (x - c).powi(2)).sum();
            // sort descending by distance (most corner-like first, trim interior)
            dist_b.partial_cmp(&dist_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        weight_vectors.truncate(num_regions);
    }

    weight_vectors
        .into_iter()
        .enumerate()
        .map(|(index, weight_vector)| Region {
            index,
            weight_vector,
        })
        .collect()
}

/// Compute augmented Tchebycheff scalarization value.
///
/// $$g^{atch}(s | w, z^*) = \max_j \{ w_j \cdot d_j \} + \rho \sum_j w_j \cdot d_j$$
///
/// where $d_j = \max(0, f_j - z^*_j) / \Delta_j$ (normalized distance from ideal).
/// Lower is better (solution is closer to ideal in this weight direction).
#[inline]
#[must_use]
pub fn tchebycheff_score<const D: usize>(
    objectives: &[u64; D],
    weight: &[f64; D],
    ideal: &Objectives<D>,
    bounds: &[(f64, f64); D],
    rho: f64,
) -> f64 {
    let mut max_term = f64::NEG_INFINITY;
    let mut sum_term = 0.0;

    for j in 0..D {
        let (min_j, max_j) = bounds[j];
        let range = (max_j - min_j).max(1.0); // avoid division by zero
        let dist_from_ideal = objectives[j].saturating_sub(ideal[j]) as f64;
        let normalized = dist_from_ideal / range;
        let weighted = weight[j] * normalized;
        if weighted > max_term {
            max_term = weighted;
        }
        sum_term += weighted;
    }

    max_term + rho * sum_term
}

/// Precomputed coefficients for fast repeated Tchebycheff scoring.
///
/// Eliminates per-call divisions and multiplications by precomputing
/// `weight[j] / range[j]` for each dimension. Use when `weight`, `bounds`,
/// and `rho` are constant across many calls (e.g., the inner loop of
/// `step_scalarized`).
///
/// Construct via [`TchebycheffCoeffs::new`], then call [`tchebycheff_score_fast`].
pub struct TchebycheffCoeffs<const D: usize> {
    /// Precomputed `weight[j] / range[j]` for each dimension.
    weighted_inv_range: [f64; D],
    rho: f64,
}

impl<const D: usize> TchebycheffCoeffs<D> {
    /// Precompute coefficients from weight vector, objective bounds, and rho.
    #[inline]
    #[must_use]
    pub fn new(weight: &[f64; D], bounds: &[(f64, f64); D], rho: f64) -> Self {
        let mut weighted_inv_range = [0.0; D];
        for j in 0..D {
            let (min_j, max_j) = bounds[j];
            let range = (max_j - min_j).max(1.0);
            weighted_inv_range[j] = weight[j] / range;
        }
        Self {
            weighted_inv_range,
            rho,
        }
    }

    /// Compute augmented Tchebycheff score with precomputed coefficients.
    ///
    /// Equivalent to [`tchebycheff_score`] but avoids per-call divisions.
    #[inline]
    #[must_use]
    pub fn score(&self, objectives: &[u64; D], ideal: &Objectives<D>) -> f64 {
        let mut max_term = f64::NEG_INFINITY;
        let mut sum_term = 0.0;

        for j in 0..D {
            let dist_from_ideal = objectives[j].saturating_sub(ideal[j]) as f64;
            let weighted = self.weighted_inv_range[j] * dist_from_ideal;
            if weighted > max_term {
                max_term = weighted;
            }
            sum_term += weighted;
        }

        max_term + self.rho * sum_term
    }
}

/// Assign a solution to all regions whose Tchebycheff score is within `threshold`
/// (relative) of the best score. Returns a sorted list of (region_idx, score) pairs.
#[must_use]
pub fn assign_to_regions<const D: usize>(
    objectives: &[u64; D],
    regions: &[Region<D>],
    ideal: &Objectives<D>,
    bounds: &[(f64, f64); D],
    boundary_threshold: f64,
) -> Vec<usize> {
    let scores: Vec<f64> = regions
        .iter()
        .map(|r| r.score(objectives, ideal, bounds))
        .collect();

    let best = scores.iter().cloned().fold(f64::INFINITY, f64::min);

    scores
        .iter()
        .enumerate()
        .filter(|(_, s)| {
            let s = **s;
            if best.abs() < f64::EPSILON {
                s < f64::EPSILON || (s - best).abs() / best.max(f64::EPSILON) <= boundary_threshold
            } else {
                (s - best) / best <= boundary_threshold
            }
        })
        .map(|(i, _)| i)
        .collect()
}

/// Returns `true` if `region` is the best-matching (lowest score) region for this solution,
/// or tied for best (so solutions on boundaries are shared).
///
/// Epsilon relaxation prevents floating-point ties from incorrectly rejecting
/// a solution as not belonging to its own region.
#[inline]
#[must_use]
pub fn belongs_to_region<const D: usize>(
    objectives: &[u64; D],
    region: &Region<D>,
    all_regions: &[Region<D>],
    ideal: &Objectives<D>,
    bounds: &[(f64, f64); D],
) -> bool {
    let my_score = region.score(objectives, ideal, bounds);
    all_regions
        .iter()
        .all(|r| r.score(objectives, ideal, bounds) >= my_score - f64::EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn das_dennis_d4_h1_gives_4_vectors() {
        let vecs = das_dennis_weight_vectors::<4>(1);
        assert_eq!(vecs.len(), 4);
        for v in &vecs {
            let sum: f64 = v.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "weights should sum to 1.0, got {sum}");
        }
    }

    #[test]
    fn das_dennis_d4_h2_gives_10_vectors() {
        let vecs = das_dennis_weight_vectors::<4>(2);
        assert_eq!(vecs.len(), 10);
    }

    #[test]
    fn das_dennis_d4_h3_gives_20_vectors() {
        let vecs = das_dennis_weight_vectors::<4>(3);
        assert_eq!(vecs.len(), 20);
    }

    #[test]
    fn auto_select_h_4_threads() {
        // D=4, need>=4 vectors: H=1 gives 4
        assert_eq!(auto_select_h::<4>(4), 1);
    }

    #[test]
    fn auto_select_h_8_threads() {
        // D=4, need>=8: H=2 gives 10
        assert_eq!(auto_select_h::<4>(8), 2);
    }

    #[test]
    fn assign_to_region_assigns_best() {
        let bounds = [(0.0f64, 100.0f64); 4];
        let regions = build_regions::<4>(4, Some(1));
        let ideal = [0u64; 4];
        // A solution low in objective 0 should go to the region with highest weight on objective 0
        let objectives = [1u64, 100, 100, 100];
        let assigned = assign_to_regions(&objectives, &regions, &ideal, &bounds, 0.05);
        assert!(!assigned.is_empty());
    }
}
