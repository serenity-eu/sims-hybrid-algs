//! Spacing indicator — a reference-free density/uniformity metric
//! complementing [`crate::hypervolume`] and [`crate::igd`].
//!
//! Hypervolume rewards *area/volume coverage*, not how evenly a front is
//! resolved: adding many interior points to an already-covered envelope
//! barely moves HV, even though it makes the front far more useful in
//! practice (more actual tradeoff choices, no large unrepresented gaps).
//! Spacing is exactly the indicator for that story — it measures only how
//! uniformly a front's own points are distributed, independent of any
//! reference/target front and independent of the front's extent.
//!
//! Given a front `F = {f_1, …, f_n}` (minimisation form):
//!
//! * for each point `f_i`, `d_i` = the (Manhattan / L1) distance to its
//!   nearest **other** neighbour in `F`;
//! * `d̄` = mean of the `d_i`;
//! * **Spacing** `S = sqrt( (1/n) · Σ_i (d_i − d̄)² )` — the population
//!   standard deviation of nearest-neighbour distances.
//!
//! `S = 0` means every point is equally spaced from its nearest neighbour
//! (perfectly uniform); larger `S` means the front has both crowded regions
//! and sparse gaps. This mirrors the reference `SpacingIndicator` in
//! `pymoo.indicators.spacing` (Manhattan distance, population — not sample —
//! standard deviation), including its `zero_to_one` normalisation, so values
//! are directly comparable across the two implementations (see the
//! `tests` module for numeric cross-checks against pymoo's own output).
//!
//! `S` is undefined for fronts with fewer than 2 points (there is no
//! "nearest neighbour"); this implementation returns `0.0` for `n ≤ 1`
//! rather than erroring, since "trivially uniform" is a more useful default
//! for a library API than a panic or an exception — note pymoo itself
//! actually *errors* for `n == 1` (`kth(=1) out of bounds`), so this is a
//! deliberate, documented deviation for the single-point edge case only;
//! `n == 0` matches pymoo's own `0.0` default.
//!
//! Complexity is `O(n² · D)` (all-pairs Manhattan distance), the same class
//! as the [`crate::igd`] module; for the front sizes here (hundreds of
//! points, `D ≤ 4`) this is microseconds.

use crate::hypervolume::HVNumeric;
use crate::igd::{extract_points, flatten, normalize_in_place};
use crate::solution::Solution;
use pyo3::prelude::*;

/// Spacing indicator of `front` (minimisation form, flattened into a
/// contiguous `d`-strided buffer of `f64`).
///
/// Returns `0.0` for `n ≤ 1` (see module docs for why this differs from
/// pymoo's `n == 1` error).
fn spacing_flat(front: &[f64], d: usize) -> f64 {
    let n = front.len() / d;
    if n <= 1 {
        return 0.0;
    }

    let mut nearest = vec![f64::INFINITY; n];
    for i in 0..n {
        let pi = &front[i * d..i * d + d];
        for j in (i + 1)..n {
            let pj = &front[j * d..j * d + d];
            let dist: f64 = pi.iter().zip(pj).map(|(&a, &b)| (a - b).abs()).sum();
            if dist < nearest[i] {
                nearest[i] = dist;
            }
            if dist < nearest[j] {
                nearest[j] = dist;
            }
        }
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "Front sizes are always far below 2^53, so f64 conversion is safe"
    )]
    let n_f64 = n as f64;
    let mean: f64 = nearest.iter().sum::<f64>() / n_f64;
    let variance: f64 = nearest.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / n_f64;
    variance.sqrt()
}

/// Spacing indicator of `front` for a generic numeric point type (mirrors
/// [`crate::igd::igd_generic`]'s API shape).
///
/// # Panics
/// Panics if any row has fewer than `d` coordinates.
#[must_use]
pub fn spacing_generic<T: HVNumeric>(front: &[Vec<T>], d: usize) -> f64 {
    spacing_flat(&flatten(front, d), d)
}

/// Compute the Spacing indicator of a front (density/uniformity of the
/// front's own points — no reference front needed).
///
/// # Arguments
/// * `front` — solver front: `list[list[float]]` or `list[Solution]`.
/// * `objective_bounds` — optional `[[min,max], …]` per objective; required
///   when `normalized=True`. Matches [`crate::igd::compute_igd`]'s
///   normalisation exactly, so IGD/IGD+ and Spacing stay comparable when
///   reported together.
/// * `normalized` — normalise the front with `objective_bounds` (to `[0,1]`
///   per objective) before measuring distances — strongly recommended when
///   objectives have very different scales (cost ~10⁶ vs cloud ~10⁹ here),
///   since otherwise the larger-magnitude objective dominates the distance.
///
/// # Returns
/// The indicator value (`f64`, lower = more uniform; `0.0` for fronts with
/// fewer than 2 points).
///
/// # Errors
/// Returns an error if `front`/`objective_bounds` can't be parsed, if the
/// objective dimension can't be inferred (empty front and no
/// `objective_bounds`), if it's outside the supported `2..=4` range, if any
/// row has too few coordinates, or if `normalized=True` without
/// `objective_bounds`.
#[pyfunction]
#[pyo3(signature = (front, objective_bounds=None, normalized=false))]
pub fn compute_spacing(
    front: &Bound<'_, PyAny>,
    objective_bounds: Option<Vec<Vec<f64>>>,
    normalized: bool,
) -> PyResult<f64> {
    let dim = if let Some(b) = &objective_bounds {
        b.len()
    } else if let Ok(rows) = front.extract::<Vec<Vec<f64>>>() {
        rows.first().map_or(0, Vec::len)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot infer objective dimension: pass objective_bounds or a non-empty numeric front",
        ));
    };
    if dim == 0 {
        return Ok(0.0);
    }
    if !(2..=4).contains(&dim) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported dimension {dim}: only 2, 3 and 4 objectives are supported"
        )));
    }

    let points = extract_points(front, dim)?;
    for (i, row) in points.iter().enumerate() {
        if row.len() < dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "front point {i} has {} coordinates, expected {dim}",
                row.len()
            )));
        }
    }

    let mut buf = flatten(&points, dim);
    if normalized {
        let bounds = objective_bounds.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("normalized=True requires objective_bounds")
        })?;
        if bounds.len() != dim || bounds.iter().any(|b| b.len() != 2) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "objective_bounds must be [[min,max], …] with one [min,max] per objective",
            ));
        }
        normalize_in_place(&mut buf, bounds, dim);
    }

    let value = spacing_flat(&buf, dim);
    debug_assert!(value >= 0.0, "Spacing must be non-negative, got {value}");
    Ok(value)
}

/// Front cardinality — the simplest, most legible density indicator of all:
/// how many non-dominated solutions were actually found. Trivial by design
/// (no algorithm to get wrong), exposed here purely so callers building a
/// "density/spread" report have every metric behind one consistent API
/// (`compute_hypervolume`, `compute_igd`, `compute_spacing`,
/// `front_cardinality`) instead of reaching for Python's own `len()` for
/// just this one number.
///
/// # Errors
/// Returns an error if `front` is neither `list[list[float]]` nor
/// `list[Solution]`.
#[pyfunction]
pub fn front_cardinality(front: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(rows) = front.extract::<Vec<Vec<f64>>>() {
        return Ok(rows.len());
    }
    if let Ok(sols) = front.extract::<Vec<Solution>>() {
        return Ok(sols.len());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected a list of numeric points (list[list[float]]) or a list of Solution objects",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    // Reference values below were cross-checked against pymoo 0.6.1.6's
    // `pymoo.indicators.spacing.SpacingIndicator` directly (Manhattan
    // distance, population std-dev — matching this implementation exactly):
    //
    //   from pymoo.indicators.spacing import SpacingIndicator
    //   SpacingIndicator()(np.array(F))

    #[test]
    fn empty_front_is_zero() {
        let f: Vec<Vec<f64>> = vec![];
        assert!(close(spacing_generic(&f, 2), 0.0));
    }

    #[test]
    fn single_point_is_zero() {
        // pymoo errors here ("kth(=1) out of bounds"); we deliberately
        // return 0.0 instead — see module docs.
        let f = vec![vec![1.0, 2.0]];
        assert!(close(spacing_generic(&f, 2), 0.0));
    }

    #[test]
    fn two_points_reference() {
        // pymoo: 0.0
        let f = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        assert!(close(spacing_generic(&f, 2), 0.0));
    }

    #[test]
    fn three_uniform_reference() {
        // pymoo: 0.0 (all gaps equal -> perfectly uniform)
        let f = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        assert!(close(spacing_generic(&f, 2), 0.0));
    }

    #[test]
    fn three_nonuniform_reference() {
        // pymoo: 7.542472332656507
        let f = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![10.0, 10.0]];
        assert!(close(spacing_generic(&f, 2), 7.542_472_332_656_507));
    }

    #[test]
    fn four_random_reference() {
        // pymoo: 1.299038105676658
        let f = vec![
            vec![0.0, 0.0],
            vec![1.0, 3.0],
            vec![2.0, 1.0],
            vec![5.0, 5.0],
        ];
        assert!(close(spacing_generic(&f, 2), 1.299_038_105_676_658));
    }

    #[test]
    fn three_dims_reference() {
        // pymoo: 1.4142135623730951
        let f = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![3.0, 3.0, 3.0],
        ];
        assert!(close(spacing_generic(&f, 3), 2.0f64.sqrt()));
    }

    #[test]
    fn denser_front_is_more_uniform_than_sparse_subset() {
        // The core "density improves spacing" claim this indicator exists
        // to support: adding evenly-spaced interior points to a sparse
        // front should *decrease* (improve) spacing, even though it barely
        // changes hypervolume.
        let sparse = vec![vec![0.0, 10.0], vec![10.0, 0.0]];
        let dense = vec![
            vec![0.0, 10.0],
            vec![2.0, 8.0],
            vec![4.0, 6.0],
            vec![6.0, 4.0],
            vec![8.0, 2.0],
            vec![10.0, 0.0],
        ];
        let s_sparse = spacing_generic(&sparse, 2);
        let s_dense = spacing_generic(&dense, 2);
        assert!(
            close(s_dense, 0.0),
            "evenly-spaced dense front should be ~0, got {s_dense}"
        );
        assert!(
            s_dense <= s_sparse + 1e-12,
            "denser, uniform front should not be *less* uniform than the sparse one"
        );
    }

    #[test]
    fn normalization_changes_result_when_scales_differ() {
        // pymoo (raw): 82915.77428347371 ; pymoo (normalised to [0,1] per
        // objective): 0.30445192103552615. Without normalisation the
        // huge-scale second objective dominates the Manhattan distance, so
        // the two implementations must agree on both the raw value AND on
        // normalisation producing a materially different (and much smaller)
        // number — this is exactly why compute_spacing's `normalized`
        // option matters whenever objectives differ in scale, as they do
        // here (cost ~10⁶ vs cloud ~10⁹).
        let f = vec![
            vec![0.0, 0.0],
            vec![0.1, 1_000_000.0],
            vec![0.15, 1_900_000.0],
            vec![0.6, 3_000_000.0],
        ];
        let raw = spacing_generic(&f, 2);
        assert!(close(raw, 82_915.774_283_473_71));

        let bounds = vec![vec![0.0, 0.6], vec![0.0, 3_000_000.0]];
        let mut buf = flatten(&f, 2);
        normalize_in_place(&mut buf, &bounds, 2);
        let normalized = spacing_flat(&buf, 2);
        assert!(close(normalized, 0.304_451_921_035_526_15));
    }
}
