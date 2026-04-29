//! Shared weighted Chebycheff scalarization utilities.
//!
//! These helpers are used to steer Pareto Local Search toward promising
//! solutions under sampled trade-off directions.
//!
//! The implementation assumes a minimization setting:
//! lower objective values are better, and the ideal point is the
//! component-wise minimum.

use pareto::Objectives;
use rand::{Rng, SeedableRng};

/// Default augmentation coefficient for augmented weighted Chebycheff.
///
/// The score is:
///
/// `max_j (w_j * d_j) + rho * sum_j (w_j * d_j)`
///
/// where `d_j` is the normalized distance from the ideal point.
pub const DEFAULT_RHO: f64 = 1e-3;

/// Compute augmented weighted Chebycheff scalarization value.
///
/// Lower is better.
///
/// # Parameters
///
/// - `objectives`: objective vector of the candidate solution
/// - `weight`: weight vector on the simplex
/// - `ideal`: component-wise ideal point
/// - `bounds`: per-objective `(min, max)` normalization bounds
/// - `rho`: augmentation coefficient
#[inline]
#[must_use]
pub fn weighted_chebycheff_score<const D: usize>(
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
        let range = (max_j - min_j).max(1.0);
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

/// Precomputed coefficients for repeated weighted Chebycheff scoring.
///
/// This avoids repeated divisions in tight loops when `weight`, `bounds`,
/// and `rho` stay constant across many score evaluations.
#[derive(Debug, Clone)]
pub struct WeightedChebycheffCoeffs<const D: usize> {
    weighted_inv_range: [f64; D],
    rho: f64,
}

impl<const D: usize> WeightedChebycheffCoeffs<D> {
    /// Build precomputed coefficients from a weight vector and bounds.
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

    /// Compute the augmented weighted Chebycheff score using precomputed coefficients.
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

/// Sample a random weight vector on the unit simplex.
///
/// The returned weights are non-negative and sum to 1.0.
///
/// This uses the standard exponential-normalization trick:
/// sample i.i.d. exponential variates and normalize them.
#[must_use]
pub fn sample_weight_vector<const D: usize, R: Rng + ?Sized>(rng: &mut R) -> [f64; D] {
    let mut raw = [0.0; D];
    let mut sum = 0.0;

    for value in &mut raw {
        // If u is in (0,1], then -ln(u) is exponential(1).
        // Clamp away from zero to avoid ln(0).
        let u = rng.random::<f64>().clamp(f64::MIN_POSITIVE, 1.0);
        let x = -u.ln();
        *value = x;
        sum += x;
    }

    if sum <= f64::EPSILON {
        return std::array::from_fn(|_| 1.0 / D as f64);
    }

    std::array::from_fn(|j| raw[j] / sum)
}

/// Sample multiple random weight vectors on the unit simplex.
///
/// Uses a deterministic RNG seeded from `seed`.
#[must_use]
pub fn sample_weight_vectors<const D: usize>(count: usize, seed: u64) -> Vec<[f64; D]> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    (0..count)
        .map(|_| sample_weight_vector::<D, _>(&mut rng))
        .collect()
}

/// Compute the component-wise ideal point from a collection of objective vectors.
///
/// Returns `[u64::MAX; D]` for an empty iterator.
#[must_use]
pub fn compute_ideal_from_objectives<I, const D: usize>(objectives: I) -> Objectives<D>
where
    I: IntoIterator<Item = [u64; D]>,
{
    let mut ideal = [u64::MAX; D];
    for obj in objectives {
        for j in 0..D {
            ideal[j] = ideal[j].min(obj[j]);
        }
    }
    ideal
}

/// Compute the component-wise nadir point from a collection of objective vectors.
///
/// Returns `[0; D]` for an empty iterator.
#[must_use]
pub fn compute_nadir_from_objectives<I, const D: usize>(objectives: I) -> Objectives<D>
where
    I: IntoIterator<Item = [u64; D]>,
{
    let mut nadir = [0u64; D];
    for obj in objectives {
        for j in 0..D {
            nadir[j] = nadir[j].max(obj[j]);
        }
    }
    nadir
}

/// Convert ideal/nadir points into normalization bounds.
#[must_use]
pub fn bounds_from_ideal_nadir<const D: usize>(
    ideal: &Objectives<D>,
    nadir: &Objectives<D>,
) -> [(f64, f64); D] {
    std::array::from_fn(|j| (ideal[j] as f64, nadir[j] as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampled_weights_sum_to_one() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let w = sample_weight_vector::<4, _>(&mut rng);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(w.iter().all(|x| *x >= 0.0));
    }

    #[test]
    fn multiple_weight_vectors_have_expected_count() {
        let weights = sample_weight_vectors::<4>(7, 123);
        assert_eq!(weights.len(), 7);
    }

    #[test]
    fn weighted_chebycheff_matches_precomputed_coeffs() {
        let objectives = [10u64, 20, 30, 40];
        let ideal = [0u64, 0, 0, 0];
        let bounds = [(0.0, 100.0); 4];
        let weight = [0.1, 0.2, 0.3, 0.4];
        let rho = DEFAULT_RHO;

        let direct = weighted_chebycheff_score(&objectives, &weight, &ideal, &bounds, rho);
        let coeffs = WeightedChebycheffCoeffs::new(&weight, &bounds, rho);
        let fast = coeffs.score(&objectives, &ideal);

        assert!((direct - fast).abs() < 1e-12);
    }

    #[test]
    fn ideal_and_nadir_are_computed_componentwise() {
        let points = vec![[5u64, 9, 3], [2, 7, 8], [4, 1, 6]];
        let ideal = compute_ideal_from_objectives(points.clone());
        let nadir = compute_nadir_from_objectives(points);

        assert_eq!(ideal, [2, 1, 3]);
        assert_eq!(nadir, [5, 9, 8]);
    }

    #[test]
    fn bounds_follow_ideal_and_nadir() {
        let ideal = [2u64, 1, 3];
        let nadir = [5u64, 9, 8];
        let bounds = bounds_from_ideal_nadir(&ideal, &nadir);

        assert_eq!(bounds, [(2.0, 5.0), (1.0, 9.0), (3.0, 8.0)]);
    }
}
