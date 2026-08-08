//! Inverted Generational Distance (IGD / IGD+) and Generational Distance
//! (GD / GD+) — convergence+diversity indicators complementing the
//! [`crate::hypervolume`] module.
//!
//! For a **reference set** `Z` (the target / true Pareto front) and an
//! **approximation set** `A` (the front produced by a solver):
//!
//! * **IGD**  = mean over `z ∈ Z` of `min over a ∈ A` of `dist(z, a)`
//!   — how well `A` *covers* `Z` (lower = better; rewards convergence AND spread).
//! * **GD**   = mean over `a ∈ A` of `min over z ∈ Z` of `dist(a, z)`
//!   — how close `A` sits to `Z` (lower = better; convergence only).
//!
//! The `+` variants (Ishibuchi et al., 2015) replace the Euclidean distance with
//! the **weakly-Pareto-compliant** modified distance: for minimisation, only the
//! objectives where the approximation point `a` is *worse* than the reference
//! point `z` count, i.e. `d⁺ = sqrt(Σ_i max(aᵢ − zᵢ, 0)²)`. IGD⁺ never ranks a
//! dominating front worse than a dominated one — unlike plain IGD.
//!
//! All objectives are treated in **minimisation** form (matching the HV module).
//! Complexity is `O(|A| · |Z| · D)`; the point sets are flattened into contiguous
//! `f64` buffers and the inner loop early-abandons once the running squared
//! distance exceeds the best-so-far, so in practice it is a few microseconds for
//! the front sizes here (hundreds of points, `D ≤ 4`).

use crate::hypervolume::HVNumeric;
use crate::solution::Solution;
use pyo3::prelude::*;

/// Squared distance from reference point `p` to the nearest point in `set`,
/// where both are laid out as flat `d`-strided `f64` buffers.
///
/// The `+`-distance is defined on `(approx − ref)`, so `flip` selects which of
/// `p`/`q` is the approximation point: `flip == false` ⇒ `p` is the reference and
/// `q` the approximation (IGD); `flip == true` ⇒ `p` is the approximation and `q`
/// the reference (GD). For the plain Euclidean distance the sign is irrelevant.
#[inline]
fn nearest_sq(p: &[f64], set: &[f64], d: usize, plus: bool, flip: bool) -> f64 {
    let n = set.len() / d;
    let mut best = f64::INFINITY;
    for j in 0..n {
        let q = &set[j * d..j * d + d];
        let mut s = 0.0f64;
        for (&pi, &qi) in p.iter().zip(q) {
            // delta = approx_i − ref_i
            let mut delta = if flip { pi - qi } else { qi - pi };
            if plus && delta < 0.0 {
                delta = 0.0;
            }
            s += delta * delta;
            if s >= best {
                break; // early abandon: this candidate cannot beat the current best
            }
        }
        if s < best {
            best = s;
        }
    }
    best
}

/// Flatten `points` (row-major `Vec<Vec<T>>`) into a contiguous `Vec<f64>` of
/// length `points.len() * d`, taking the first `d` coordinates of each row.
fn flatten<T: HVNumeric>(points: &[Vec<T>], d: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(points.len() * d);
    for p in points {
        for v in &p[..d] {
            out.push(v.to_f64());
        }
    }
    out
}

/// Mean, over every point in `from`, of the (Euclidean or `+`) distance to the
/// nearest point in `to`. Shared core of IGD/GD.
///
/// * `from_is_reference == true`  ⇒ IGD-style (`from` = Z, `to` = A).
/// * `from_is_reference == false` ⇒ GD-style  (`from` = A, `to` = Z).
fn mean_nearest_distance(
    from: &[f64],
    to: &[f64],
    d: usize,
    plus: bool,
    from_is_reference: bool,
) -> f64 {
    let n_from = from.len() / d;
    if n_from == 0 {
        return 0.0; // nothing to average over
    }
    if to.is_empty() {
        return f64::INFINITY; // no target points: every distance is unbounded
    }
    // `+`-distance uses (approx − ref). When we iterate the reference set
    // (`from_is_reference`), the *other* set is the approximation ⇒ flip == false.
    let flip = !from_is_reference;
    let mut sum = 0.0f64;
    for k in 0..n_from {
        let p = &from[k * d..k * d + d];
        sum += nearest_sq(p, to, d, plus, flip).sqrt();
    }
    sum / n_from as f64
}

/// Inverted Generational Distance of `approximation` w.r.t. `reference`.
///
/// `plus == true` selects **IGD⁺** (Pareto-compliant). Objectives are
/// minimisation. Returns `0.0` when `reference` is empty and `+∞` when
/// `approximation` is empty (but `reference` is not).
///
/// # Panics
/// Panics if any row has fewer than `d` coordinates.
#[must_use]
pub fn igd_generic<T: HVNumeric>(
    approximation: &[Vec<T>],
    reference: &[Vec<T>],
    d: usize,
    plus: bool,
) -> f64 {
    if reference.is_empty() {
        return 0.0;
    }
    let a = flatten(approximation, d);
    let z = flatten(reference, d);
    mean_nearest_distance(&z, &a, d, plus, /*from_is_reference=*/ true)
}

/// Generational Distance of `approximation` w.r.t. `reference`.
///
/// `plus == true` selects **GD⁺**. Objectives are minimisation. Returns `0.0`
/// when `approximation` is empty and `+∞` when `reference` is empty (but
/// `approximation` is not).
#[must_use]
pub fn gd_generic<T: HVNumeric>(
    approximation: &[Vec<T>],
    reference: &[Vec<T>],
    d: usize,
    plus: bool,
) -> f64 {
    if approximation.is_empty() {
        return 0.0;
    }
    let a = flatten(approximation, d);
    let z = flatten(reference, d);
    mean_nearest_distance(&a, &z, d, plus, /*from_is_reference=*/ false)
}

/// Normalise a flat `d`-strided buffer in place to `[0,1]` using per-objective
/// `[min,max]` bounds (range 0 ⇒ divide by 1). Values outside the bounds are
/// allowed (kept proportional), unlike the HV module's strict version.
fn normalize_in_place(buf: &mut [f64], bounds: &[Vec<f64>], d: usize) {
    let ranges: Vec<f64> = bounds
        .iter()
        .map(|b| {
            let r = b[1] - b[0];
            if r > 0.0 {
                r
            } else {
                1.0
            }
        })
        .collect();
    let n = buf.len() / d;
    for k in 0..n {
        for i in 0..d {
            buf[k * d + i] = (buf[k * d + i] - bounds[i][0]) / ranges[i];
        }
    }
}

/// Extract `Vec<Vec<f64>>` point rows from a Python object that is either a list
/// of numeric rows or a list of [`Solution`] objects.
fn extract_points(data: &Bound<'_, PyAny>, d: usize) -> PyResult<Vec<Vec<f64>>> {
    if let Ok(rows) = data.extract::<Vec<Vec<f64>>>() {
        return Ok(rows);
    }
    if let Ok(sols) = data.extract::<Vec<Solution>>() {
        return Ok(sols
            .iter()
            .map(|s| match d {
                2 => s.objectives_2d().iter().map(|&v| v as f64).collect(),
                3 => s.objectives_3d().iter().map(|&v| v as f64).collect(),
                _ => s.objectives_4d().iter().map(|&v| v as f64).collect(),
            })
            .collect());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected a list of numeric points (list[list[float]]) or a list of Solution objects",
    ))
}

/// Compute an IGD-family indicator between an approximation set and a reference
/// set (both in minimisation form).
///
/// # Arguments
/// * `approximation` — solver front: `list[list[float]]` or `list[Solution]`.
/// * `reference_set` — target/true front: same accepted formats.
/// * `objective_bounds` — optional `[[min,max], …]` per objective; required when
///   `normalized=True`, in which case both sets are mapped to `[0,1]` so every
///   objective contributes comparably to the distance (strongly recommended when
///   objectives have very different scales, as here: cost ~10⁶ vs cloud ~10⁹).
/// * `normalized` — normalise both sets with `objective_bounds` before measuring.
/// * `plus` — use the Pareto-compliant `+` distance (IGD⁺ / GD⁺).
/// * `gd` — compute **GD**(⁺) instead of **IGD**(⁺).
///
/// # Returns
/// The indicator value (`f64`, lower is better).
#[pyfunction]
#[pyo3(signature = (approximation, reference_set, objective_bounds=None, normalized=false, plus=false, gd=false))]
pub fn compute_igd(
    approximation: &Bound<'_, PyAny>,
    reference_set: &Bound<'_, PyAny>,
    objective_bounds: Option<Vec<Vec<f64>>>,
    normalized: bool,
    plus: bool,
    gd: bool,
) -> PyResult<f64> {
    // Dimension: from bounds if given, else the first reference row, else approx.
    let dim = if let Some(b) = &objective_bounds {
        b.len()
    } else if let Ok(r) = reference_set.extract::<Vec<Vec<f64>>>() {
        r.first().map_or(0, Vec::len)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot infer objective dimension: pass objective_bounds or non-empty numeric points",
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

    let approx = extract_points(approximation, dim)?;
    let reference = extract_points(reference_set, dim)?;
    for (name, set) in [("approximation", &approx), ("reference_set", &reference)] {
        for (i, row) in set.iter().enumerate() {
            if row.len() < dim {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{name} point {i} has {} coordinates, expected {dim}",
                    row.len()
                )));
            }
        }
    }

    let (mut af, mut zf) = (flatten(&approx, dim), flatten(&reference, dim));
    if normalized {
        let bounds = objective_bounds.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("normalized=True requires objective_bounds")
        })?;
        if bounds.len() != dim || bounds.iter().any(|b| b.len() != 2) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "objective_bounds must be [[min,max], …] with one [min,max] per objective",
            ));
        }
        normalize_in_place(&mut af, bounds, dim);
        normalize_in_place(&mut zf, bounds, dim);
    }

    // Operate directly on the (possibly normalised) flat buffers.
    let value = if gd {
        if af.is_empty() {
            0.0
        } else {
            mean_nearest_distance(&af, &zf, dim, plus, false)
        }
    } else if zf.is_empty() {
        0.0
    } else {
        mean_nearest_distance(&zf, &af, dim, plus, true)
    };

    debug_assert!(value >= 0.0, "IGD/GD must be non-negative, got {value}");
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn igd_zero_when_sets_match() {
        let f = vec![vec![1u64, 2], vec![2, 1]];
        assert!(close(igd_generic(&f, &f, 2, false), 0.0));
        assert!(close(igd_generic(&f, &f, 2, true), 0.0));
    }

    #[test]
    fn igd_single_point_euclidean() {
        // ref (0,0), approx (3,4) -> distance 5
        let z = vec![vec![0.0, 0.0]];
        let a = vec![vec![3.0, 4.0]];
        assert!(close(igd_generic(&a, &z, 2, false), 5.0));
        // gd is symmetric here (one point each)
        assert!(close(gd_generic(&a, &z, 2, false), 5.0));
    }

    #[test]
    fn igd_plus_ignores_dominating_direction() {
        // approx (0,0) dominates ref (1,1): plain IGD = sqrt(2); IGD+ = 0.
        let z = vec![vec![1.0, 1.0]];
        let a = vec![vec![0.0, 0.0]];
        assert!(close(igd_generic(&a, &z, 2, false), 2.0f64.sqrt()));
        assert!(close(igd_generic(&a, &z, 2, true), 0.0));
        // ...but GD+ (approx worse than ref? no, approx dominates) is also 0,
        // while GD euclidean is sqrt(2).
        assert!(close(gd_generic(&a, &z, 2, false), 2.0f64.sqrt()));
        assert!(close(gd_generic(&a, &z, 2, true), 0.0));
    }

    #[test]
    fn igd_plus_never_exceeds_igd() {
        let z = vec![vec![0.0, 0.0], vec![2.0, 2.0], vec![1.0, 3.0]];
        let a = vec![vec![0.5, 2.5], vec![3.0, 0.5], vec![1.5, 1.5]];
        let igd = igd_generic(&a, &z, 2, false);
        let igdp = igd_generic(&a, &z, 2, true);
        assert!(igdp <= igd + 1e-12, "IGD+ {igdp} must be <= IGD {igd}");
    }

    #[test]
    fn igd_averages_over_reference() {
        // ref {(0,0),(0,10)}, approx {(0,0)}: distances 0 and 10 -> mean 5.
        let z = vec![vec![0.0, 0.0], vec![0.0, 10.0]];
        let a = vec![vec![0.0, 0.0]];
        assert!(close(igd_generic(&a, &z, 2, false), 5.0));
    }

    #[test]
    fn empty_approx_is_infinite_igd() {
        let z = vec![vec![1.0, 1.0]];
        let a: Vec<Vec<f64>> = vec![];
        assert!(igd_generic(&a, &z, 2, false).is_infinite());
    }

    #[test]
    fn empty_reference_is_zero_igd() {
        let z: Vec<Vec<f64>> = vec![];
        let a = vec![vec![1.0, 1.0]];
        assert!(close(igd_generic(&a, &z, 2, false), 0.0));
    }

    #[test]
    fn nearest_picks_minimum() {
        // ref (0,0); approx has a far and a near point -> near one wins.
        let z = vec![vec![0.0, 0.0]];
        let a = vec![vec![10.0, 10.0], vec![1.0, 0.0]];
        assert!(close(igd_generic(&a, &z, 2, false), 1.0));
    }

    #[test]
    fn works_in_three_and_four_dims() {
        let z3 = vec![vec![0.0, 0.0, 0.0]];
        let a3 = vec![vec![1.0, 2.0, 2.0]]; // dist 3
        assert!(close(igd_generic(&a3, &z3, 3, false), 3.0));
        let z4 = vec![vec![0.0, 0.0, 0.0, 0.0]];
        let a4 = vec![vec![1.0, 1.0, 1.0, 1.0]]; // dist 2
        assert!(close(igd_generic(&a4, &z4, 4, false), 2.0));
    }
}
