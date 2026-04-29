//! Diverse probing: select a uniform, well-spread subset of the population
//! to explore, using farthest-point sampling in normalised objective space.
//!
//! Instead of exploring every member of the working population (or a random
//! shuffle), [`DiverseProbeIter`] picks *K ≪ N* members whose objective
//! vectors are maximally spread.  Each selected member therefore receives
//! proportionally more neighbourhood evaluation time within the same budget.

use std::collections::VecDeque;

use pareto::HasObjectives;
use rand::SeedableRng;

/// Iterator that yields a diverse subset of a population, selected by
/// greedy farthest-point sampling in normalised objective space.
///
/// # Algorithm
///
/// 1. Normalise all objective vectors to \[0, 1\] using the population
///    min/max per objective.
/// 2. Seed with a deterministic first pick (from `seed`).
/// 3. Iteratively pick the candidate with the **largest minimum Chebyshev
///    distance** to the already-selected set, until `budget` solutions are
///    selected.
///
/// The resulting subset uniformly covers the Pareto front approximation.
/// Solutions are yielded in selection order (most structurally important
/// first), so early timer termination still explores the most spread-out
/// members.
pub struct DiverseProbeIter<T> {
    /// The selected subset, in farthest-point order.
    selected: VecDeque<T>,
}

impl<T> DiverseProbeIter<T> {
    /// Number of solutions that will be yielded.
    #[must_use]
    pub fn budget(&self) -> usize {
        self.selected.len()
    }

    /// Number of solutions remaining.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.selected.len()
    }
}

/// Build a [`DiverseProbeIter`] from a population, selecting `budget`
/// solutions by farthest-point sampling in normalised objective space.
///
/// - `population`: the full working population (consumed).
/// - `budget`: how many solutions to select (*K*).
///   If `None`, defaults to `2 · D · √|population|`, clamped to
///   `[1, |population|]`.
/// - `seed`: RNG seed for the deterministic first-pick tiebreaker.
#[must_use]
pub fn diverse_probe_iter<T, const D: usize>(
    population: Vec<T>,
    budget: Option<usize>,
    seed: u64,
) -> DiverseProbeIter<T>
where
    T: HasObjectives<D>,
{
    let n = population.len();
    if n == 0 {
        return DiverseProbeIter {
            selected: VecDeque::new(),
        };
    }

    let k = budget
        .unwrap_or_else(|| 2 * D * ((n as f64).sqrt().ceil() as usize).max(1))
        .clamp(1, n);

    if k >= n {
        return DiverseProbeIter {
            selected: VecDeque::from(population),
        };
    }

    // ── 1. Normalise objectives to [0, 1] ──────────────────────────
    let mut obj_min = [f64::INFINITY; D];
    let mut obj_max = [f64::NEG_INFINITY; D];
    for sol in &population {
        for d in 0..D {
            let v = sol.objectives()[d] as f64;
            if v < obj_min[d] {
                obj_min[d] = v;
            }
            if v > obj_max[d] {
                obj_max[d] = v;
            }
        }
    }
    let ranges: [f64; D] = std::array::from_fn(|d| {
        let r = obj_max[d] - obj_min[d];
        if r < f64::EPSILON {
            1.0
        } else {
            r
        }
    });

    let normalised: Vec<[f64; D]> = population
        .iter()
        .map(|sol| {
            std::array::from_fn(|d| (sol.objectives()[d] as f64 - obj_min[d]) / ranges[d])
        })
        .collect();

    // ── 2. Greedy farthest-point selection ─────────────────────────
    let mut indices: Vec<usize> = Vec::with_capacity(k);
    let mut available = vec![true; n];
    let mut min_dist = vec![f64::INFINITY; n];

    // First pick: seeded deterministic choice
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let first: usize = rand::Rng::random_range(&mut rng, 0..n);
    indices.push(first);
    available[first] = false;

    // Update distances from first pick
    for i in 0..n {
        if available[i] {
            let d = chebyshev_distance(&normalised[i], &normalised[first]);
            if d < min_dist[i] {
                min_dist[i] = d;
            }
        }
    }

    while indices.len() < k {
        // Pick the available candidate with the largest min-distance
        let mut best_idx: Option<usize> = None;
        let mut best_dist = f64::NEG_INFINITY;
        for i in 0..n {
            if available[i] && min_dist[i] > best_dist {
                best_dist = min_dist[i];
                best_idx = Some(i);
            }
        }

        let Some(picked) = best_idx else { break };
        indices.push(picked);
        available[picked] = false;

        // Update min-distances from newly picked point
        for i in 0..n {
            if available[i] {
                let d = chebyshev_distance(&normalised[i], &normalised[picked]);
                if d < min_dist[i] {
                    min_dist[i] = d;
                }
            }
        }
    }

    // ── 3. Extract selected solutions in selection order ───────────
    let mut taken: Vec<Option<T>> = population.into_iter().map(Some).collect();
    let selected: Vec<T> = indices
        .into_iter()
        .filter_map(|i| taken[i].take())
        .collect();

    DiverseProbeIter {
        selected: VecDeque::from(selected),
    }
}

impl<T> Iterator for DiverseProbeIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.selected.pop_front()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.selected.len();
        (n, Some(n))
    }
}

impl<T> ExactSizeIterator for DiverseProbeIter<T> {}

/// Chebyshev (L∞) distance in D-dimensional normalised objective space.
///
/// Preferred over Euclidean for uniform coverage: it treats all objectives
/// equally at the extremes, avoiding interior-concentration artifacts.
#[inline]
fn chebyshev_distance<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut max = 0.0f64;
    for d in 0..D {
        let diff = (a[d] - b[d]).abs();
        if diff > max {
            max = diff;
        }
    }
    max
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial HasObjectives implementation for testing.
    #[derive(Clone, Debug, PartialEq)]
    struct FakeSolution<const D: usize> {
        objectives: [u64; D],
    }

    impl<const D: usize> HasObjectives<D> for FakeSolution<D> {
        fn objectives(&self) -> &[u64; D] {
            &self.objectives
        }
    }

    #[test]
    fn empty_population_yields_nothing() {
        let iter = diverse_probe_iter::<FakeSolution<2>, 2>(vec![], None, 42);
        assert_eq!(iter.budget(), 0);
        assert_eq!(iter.collect::<Vec<_>>().len(), 0);
    }

    #[test]
    fn budget_exceeds_population_returns_all() {
        let pop = vec![
            FakeSolution { objectives: [0, 0] },
            FakeSolution { objectives: [100, 100] },
        ];
        let results: Vec<_> = diverse_probe_iter(pop.clone(), Some(10), 0).collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn selects_diverse_subset() {
        // 5 solutions: corners + center in 2D
        let pop = vec![
            FakeSolution { objectives: [0, 0] },       // corner
            FakeSolution { objectives: [100, 0] },      // corner
            FakeSolution { objectives: [0, 100] },      // corner
            FakeSolution { objectives: [100, 100] },    // corner
            FakeSolution { objectives: [50, 50] },      // center
        ];
        // Select 4 — should pick all 4 corners, skip center
        let results: Vec<_> = diverse_probe_iter(pop, Some(4), 0).collect();
        assert_eq!(results.len(), 4);

        // The center (50,50) should NOT be selected
        let has_center = results
            .iter()
            .any(|s| s.objectives == [50, 50]);
        assert!(!has_center, "Center should not be among the 4 most diverse");
    }

    #[test]
    fn chebyshev_distance_correct() {
        assert!((chebyshev_distance(&[0.0, 0.0], &[1.0, 0.5]) - 1.0).abs() < 1e-10);
        assert!((chebyshev_distance(&[0.3, 0.7], &[0.3, 0.7]) - 0.0).abs() < 1e-10);
    }
}
