use pareto::{HasObjectives, Objectives};
use pyo3::prelude::*;
use crate::solution::Solution;
use std::ops::{Sub, Mul, Add};
use std::cmp::PartialOrd;

/// Trait for numeric types that can be used in hypervolume computation
pub trait HVNumeric: Copy + PartialOrd + Sub<Output = Self> + Mul<Output = Self> + Add<Output = Self> + From<u8> + std::fmt::Debug {
    const ZERO: Self;
    const ONE: Self;
    fn to_f64(self) -> f64;
    fn from_f64(val: f64) -> Self;
}

impl HVNumeric for u64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    fn to_f64(self) -> f64 { self as f64 }
    fn from_f64(val: f64) -> Self { val as u64 }
}

impl HVNumeric for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn to_f64(self) -> f64 { self }
    fn from_f64(val: f64) -> Self { val }
}

/// Compute hypervolume for 4D minimization front (generic version).
/// Input: non-dominated points (Pareto front), reference point >= all coords.
/// Returns exact HV.
pub fn hypervolume_4d_min_generic<T>(points: &mut [Vec<T>], reference: &[T]) -> T
where
    T: HVNumeric + std::iter::Sum,
{
    if points.is_empty() {
        return T::ZERO;
    }

    // Sort by 4th dim
    points.sort_by(|a, b| a[3].partial_cmp(&b[3]).unwrap());

    let mut total = T::ZERO;
    let mut prev = reference[3];
    let n = points.len();
    let mut i = n;

    while i > 0 {
        i -= 1;
        let bound = points[i][3];
        
        if bound < prev {
            let width = prev - bound;
            
            // Extract 3D points from slice [0..=i]
            let mut points_3d: Vec<Vec<T>> = points[..=i].iter()
                .map(|p| vec![p[0], p[1], p[2]])
                .collect();
                
            let vol3d = hypervolume_3d_min_generic(&mut points_3d, &[reference[0], reference[1], reference[2]]);
            total = total + width * vol3d;
            prev = bound;
        }
        
        // Skip all points with the same coordinate to handle ties correctly
        while i > 0 && points[i-1][3] == bound {
            i -= 1;
        }
    }

    total
}

/// 3D HV with slice view (generic version)
fn hypervolume_3d_min_generic<T>(points: &mut [Vec<T>], reference: &[T]) -> T
where
    T: HVNumeric + std::iter::Sum,
{
    if points.is_empty() {
        return T::ZERO;
    }

    points.sort_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());

    let mut total = T::ZERO;
    let mut prev = reference[2];
    let n = points.len();
    let mut i = n;

    while i > 0 {
        i -= 1;
        let bound = points[i][2];
        
        if bound < prev {
            let width = prev - bound;
            
            // Extract 2D points from slice [0..=i]
            let mut points_2d: Vec<Vec<T>> = points[..=i].iter()
                .map(|p| vec![p[0], p[1]])
                .collect();
            
            let area2d = hypervolume_2d_min_generic(&mut points_2d, &[reference[0], reference[1]]);
            total = total + width * area2d;
            prev = bound;
        }
        
        // Skip all points with the same z-coordinate to handle ties correctly
        while i > 0 && points[i-1][2] == bound {
            i -= 1;
        }
    }

    total
}

/// 2D HV: union area = sum over strips (generic version)
fn hypervolume_2d_min_generic<T>(points: &mut [Vec<T>], reference: &[T]) -> T
where
    T: HVNumeric + std::iter::Sum,
{
    if points.is_empty() {
        return T::ZERO;
    }

    // Sort by y-coordinate (second dimension)
    points.sort_by(|a, b| a[1].partial_cmp(&b[1]).unwrap());

    let mut total = T::ZERO;
    let mut prev_y = reference[1];

    for i in (0..points.len()).rev() {
        let curr_y = points[i][1];
        
        if curr_y < prev_y {
            // Find minimum x-coordinate among points[0..=i]
            let min_x = points[..=i].iter().map(|p| p[0]).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            
            if min_x < reference[0] {
                let width_y = prev_y - curr_y;
                let width_x = reference[0] - min_x;
                total = total + width_y * width_x;
            }
            
            prev_y = curr_y;
        }
    }

    total
}

/// Compute hypervolume for generic structure implementing HasObjectives
pub fn compute<const D: usize, T: HasObjectives<D>>(pareto_front: Vec<T>, reference_point: Objectives<D>) -> u128 {
    match D {
        2 => {
            let mut points_vec: Vec<Vec<u64>> = pareto_front.iter().map(|s| {
                let objectives = *s.objectives();
                vec![objectives[0], objectives[1]]
            }).collect();
            let reference_vec = vec![reference_point[0], reference_point[1]];
            hypervolume_2d_min_generic(&mut points_vec, &reference_vec) as u128
        }
        3 => {
            let mut points_vec: Vec<Vec<u64>> = pareto_front.iter().map(|s| {
                let objectives = *s.objectives();
                vec![objectives[0], objectives[1], objectives[2]]
            }).collect();
            let reference_vec = vec![reference_point[0], reference_point[1], reference_point[2]];
            hypervolume_3d_min_generic(&mut points_vec, &reference_vec) as u128
        }
        4 => {
            let mut points_vec: Vec<Vec<u64>> = pareto_front.iter().map(|s| {
                let objectives = *s.objectives();
                vec![objectives[0], objectives[1], objectives[2], objectives[3]]
            }).collect();
            let reference_vec = vec![reference_point[0], reference_point[1], reference_point[2], reference_point[3]];
            hypervolume_4d_min_generic(&mut points_vec, &reference_vec) as u128
        }
        _ => 0
    }
}

/// Unified hypervolume computation function.
/// 
/// # Arguments
/// * `data` - Either a list of points (Vec<Vec<u64>>) or a list of Solution objects
/// * `objective_bounds` - Bounds for each dimension. Format: [[min1, max1], [min2, max2], ...] for each dimension
/// * `reference_point` - Optional reference point. If not provided, computed as the maximum bounds
/// * `normalized` - Optional normalization flag. If true, normalizes objectives to [0, 1] range using objective_bounds
/// 
/// # Returns
/// The hypervolume as an integer
/// 
/// # Examples
/// ```python
/// # Basic usage with bounds only (reference computed as max bounds)
/// points = [[1, 2], [2, 1]]
/// bounds = [[0, 10], [0, 10]]  # min/max for each dimension
/// hv = compute_hypervolume(points, bounds)
/// 
/// # With explicit reference point
/// reference = [8, 8]
/// hv = compute_hypervolume(points, bounds, reference_point=reference)
/// 
/// # With normalization
/// hv_normalized = compute_hypervolume(points, bounds, normalized=True)
/// 
/// # Solutions with bounds
/// hv = compute_hypervolume(solutions, bounds)
/// 
/// # Solutions with normalization and custom reference
/// hv_normalized = compute_hypervolume(solutions, bounds, reference_point=reference, normalized=True)
/// ```
#[pyfunction]
#[pyo3(signature = (data, objective_bounds, reference_point=None, normalized=false))]
pub fn compute_hypervolume(
    data: &Bound<'_, pyo3::PyAny>, 
    objective_bounds: Vec<Vec<u64>>,
    reference_point: Option<Vec<u64>>,
    normalized: bool
) -> PyResult<f64> {
    // Validate objective bounds first
    let dimension = objective_bounds.len();
    validate_objective_bounds(&objective_bounds, dimension)?;
    
    // Compute reference point if not provided (use max bounds)
    let reference_point = if let Some(ref_point) = reference_point {
        // Validate provided reference point matches dimensions
        if ref_point.len() != dimension {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Reference point must have {} dimensions to match objective bounds, but got {}",
                dimension, ref_point.len()
            )));
        }
        ref_point
    } else {
        // Use max bounds as reference point
        objective_bounds.iter().map(|bound| bound[1]).collect()
    };
    
    // Convert input data to points format
    let points = if let Ok(solutions) = data.extract::<Vec<Solution>>() {
        // Input is solutions
        if solutions.is_empty() {
            return Ok(0.0);
        }
        solutions_to_points(&solutions, dimension)
    } else if let Ok(points_vec) = data.extract::<Vec<Vec<u64>>>() {
        // Input is points
        if points_vec.is_empty() {
            return Ok(0.0);
        }
        points_vec
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Input data must be either a list of points (Vec<Vec<u64>>) or a list of Solution objects"
        ));
    };

    // Validate that all points have the same dimension
    for point in &points {
        if point.len() != dimension {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "All points must have the same dimension as reference point ({}), but found point with dimension {}",
                dimension, point.len()
            )));
        }
    }

    // Compute hypervolume based on normalization mode
    let result = if normalized {
        // Normalize to [0,1] range like pymoo
        let (normalized_points, normalized_reference) = normalize_points_to_unit_range(&points, &reference_point, &objective_bounds);
        
        match dimension {
            2 => {
                let mut points_vec: Vec<Vec<f64>> = normalized_points.to_vec();
                hypervolume_2d_min_generic(&mut points_vec, &normalized_reference)
            }
            3 => {
                let mut points_vec: Vec<Vec<f64>> = normalized_points.to_vec();
                hypervolume_3d_min_generic(&mut points_vec, &normalized_reference)
            }
            4 => {
                let mut points_vec: Vec<Vec<f64>> = normalized_points.to_vec();
                hypervolume_4d_min_generic(&mut points_vec, &normalized_reference)
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unsupported dimension: {}. Only 2D, 3D, and 4D are supported.",
                    dimension
                )));
            }
        }
    } else {
        // Use generic u64 implementation
        let result = match dimension {
            2 => {
                let mut points_vec: Vec<Vec<u64>> = points.iter().map(|p| vec![p[0], p[1]]).collect();
                let reference_vec = vec![reference_point[0], reference_point[1]];
                hypervolume_2d_min_generic(&mut points_vec, &reference_vec)
            }
            3 => {
                let mut points_vec: Vec<Vec<u64>> = points.iter().map(|p| vec![p[0], p[1], p[2]]).collect();
                let reference_vec = vec![reference_point[0], reference_point[1], reference_point[2]];
                hypervolume_3d_min_generic(&mut points_vec, &reference_vec)
            }
            4 => {
                let mut points_vec: Vec<Vec<u64>> = points.iter().map(|p| vec![p[0], p[1], p[2], p[3]]).collect();
                let reference_vec = vec![reference_point[0], reference_point[1], reference_point[2], reference_point[3]];
                hypervolume_4d_min_generic(&mut points_vec, &reference_vec)
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unsupported dimension: {}. Only 2D, 3D, and 4D are supported.",
                    dimension
                )));
            }
        };
        result as f64
    };

    Ok(result)
}

/// Validate that objective bounds have the correct format and dimensions
fn validate_objective_bounds(bounds: &[Vec<u64>], dimension: usize) -> PyResult<()> {
    if bounds.len() != dimension {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Objective bounds dimension mismatch: expected {}, but got {}",
            dimension, bounds.len()
        )));
    }
    
    for (i, bound) in bounds.iter().enumerate() {
        if bound.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Each bound must have exactly 2 values [min, max], but dimension {} has {} values",
                i, bound.len()
            )));
        }
        
        if bound[0] > bound[1] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Bound minimum must be <= maximum for dimension {}, but got min={}, max={}",
                i, bound[0], bound[1]
            )));
        }
    }
    
    Ok(())
}


/// Normalize points to [0,1] unit range for pymoo compatibility
fn normalize_points_to_unit_range(
    points: &[Vec<u64>], 
    reference_point: &[u64], 
    bounds: &[Vec<u64>]
) -> (Vec<Vec<f64>>, Vec<f64>) {
    // Calculate ranges for each dimension
    let ranges: Vec<f64> = bounds.iter().map(|bound| {
        let range = bound[1] - bound[0];
        if range > 0 { range as f64 } else { 1.0 }
    }).collect();
    
    // Normalize points to [0, 1] range
    let normalized_points: Vec<Vec<f64>> = points.iter().map(|point| {
        point.iter().enumerate().map(|(i, &val)| {
            let min_bound = bounds[i][0] as f64;
            let clamped_val = (val.max(bounds[i][0]).min(bounds[i][1])) as f64;
            (clamped_val - min_bound) / ranges[i]
        }).collect()
    }).collect();
    
    // Normalize reference point
    let normalized_reference: Vec<f64> = reference_point.iter().enumerate().map(|(i, &val)| {
        let min_bound = bounds[i][0] as f64;
        let clamped_val = (val.max(bounds[i][0]).min(bounds[i][1])) as f64;
        (clamped_val - min_bound) / ranges[i]
    }).collect();
    
    (normalized_points, normalized_reference)
}

/// Convert solutions to points Vec<Vec<u64>> format
/// Uses the specified dimension from reference point length
fn solutions_to_points(solutions: &[Solution], dimension: usize) -> Vec<Vec<u64>> {
    if solutions.is_empty() {
        return Vec::new();
    }
    
    solutions.iter().map(|s| {
        match dimension {
            2 => s.objectives_2d().to_vec(),
            3 => s.objectives_3d().to_vec(),
            4 => s.objectives_4d().to_vec(),
            _ => unreachable!(), // We only handle 2, 3, 4 dimensions
        }
    }).collect()
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hv_single_point() {
        let mut pts = vec![vec![1, 2, 3, 4]];
        let r = vec![5, 6, 7, 8];
        assert_eq!(hypervolume_4d_min_generic::<u64>(&mut pts, &r) as u128, 256);
    }

    #[test]
    fn hv_two_points_cross() {
        let mut pts = vec![vec![1, 4, 2, 2], vec![4, 1, 2, 2]];
        let r = vec![5, 5, 3, 3];
        assert_eq!(hypervolume_4d_min_generic::<u64>(&mut pts, &r) as u128, 7);
    }
    #[test]
    fn hv_empty_front() {
        let mut pts: Vec<Vec<u64>> = vec![];
        let r = vec![5, 6, 7, 8];
        assert_eq!(hypervolume_4d_min_generic::<u64>(&mut pts, &r) as u128, 0);
    }

    #[test]
    fn hv_identical_points() {
        let mut pts = vec![vec![2, 2, 2, 2], vec![2, 2, 2, 2]];
        let r = vec![5, 5, 5, 5];
        assert_eq!(hypervolume_4d_min_generic::<u64>(&mut pts, &r) as u128, 81);
    }

    #[test]
    fn hv_points_on_reference() {
        let mut pts = vec![vec![5, 6, 7, 8]];
        let r = vec![5, 6, 7, 8];
        assert_eq!(hypervolume_4d_min_generic::<u64>(&mut pts, &r) as u128, 0);
    }

    #[test]
    fn hv_points_outside_reference() {
        let mut pts = vec![vec![6, 7, 8, 9]];
        let r = vec![5, 6, 7, 8];
        assert_eq!(hypervolume_4d_min_generic::<u64>(&mut pts, &r) as u128, 0);
    }

    #[test]
    fn hv_multiple_non_dominated_points() {
        let mut pts = vec![
            vec![1, 2, 3, 4],
            vec![2, 1, 3, 4],
            vec![1, 1, 2, 3],
            vec![3, 3, 3, 3],
        ];
        let r = vec![5, 5, 5, 5];
        let hv = hypervolume_4d_min_generic::<u64>(&mut pts, &r) as u128;
        assert!(hv > 0);
    }

    #[test]
    fn hv_2d_min_u64_basic() {
        let mut pts = vec![vec![1, 2], vec![2, 1]];
        let r = vec![3, 3];
        assert_eq!(hypervolume_2d_min_generic::<u64>(&mut pts, &r) as u128, 3);
    }

    #[test]
    fn hv_3d_min_u64_basic() {
        let mut pts = vec![vec![1, 2, 3], vec![2, 1, 3]];
        let r = vec![4, 4, 4];
        assert_eq!(hypervolume_3d_min_generic::<u64>(&mut pts, &r) as u128, 8);
    }

    #[test]
    fn hv_2d_min_u64_empty() {
        let mut pts: Vec<Vec<u64>> = vec![];
        let r = vec![3, 3];
        assert_eq!(hypervolume_2d_min_generic::<u64>(&mut pts, &r) as u128, 0);
    }

    #[test]
    fn hv_3d_min_u64_empty() {
        let mut pts: Vec<Vec<u64>> = vec![];
        let r = vec![4, 4, 4];
        assert_eq!(hypervolume_3d_min_generic::<u64>(&mut pts, &r) as u128, 0);
    }

    #[test]
    fn hv_2d_min_u64_identical_points() {
        let mut pts = vec![vec![2, 2], vec![2, 2]];
        let r = vec![3, 3];
        assert_eq!(hypervolume_2d_min_generic::<u64>(&mut pts, &r) as u128, 1);
    }

    #[test]
    fn hv_3d_min_u64_identical_points() {
        let mut pts = vec![vec![2, 2, 2], vec![2, 2, 2]];
        let r = vec![4, 4, 4];
        assert_eq!(hypervolume_3d_min_generic::<u64>(&mut pts, &r) as u128, 8);
    }

    #[test]
    fn hv_2d_min_u64_points_on_reference() {
        let mut pts = vec![vec![3, 3]];
        let r = vec![3, 3];
        assert_eq!(hypervolume_2d_min_generic::<u64>(&mut pts, &r) as u128, 0);
    }

    #[test]
    fn hv_3d_min_u64_points_on_reference() {
        let mut pts = vec![vec![4, 4, 4]];
        let r = vec![4, 4, 4];
        assert_eq!(hypervolume_3d_min_generic::<u64>(&mut pts, &r) as u128, 0);
    }

    #[test]
    fn hv_2d_min_u64_points_outside_reference() {
        let mut pts = vec![vec![4, 4]];
        let r = vec![3, 3];
        assert_eq!(hypervolume_2d_min_generic::<u64>(&mut pts, &r) as u128, 0);
    }

    #[test]
    fn hv_3d_min_u64_points_outside_reference() {
        let mut pts = vec![vec![5, 5, 5]];
        let r = vec![4, 4, 4];
        assert_eq!(hypervolume_3d_min_generic::<u64>(&mut pts, &r) as u128, 0);
    }
}
