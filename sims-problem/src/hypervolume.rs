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

/// Brute force 2D hypervolume for debugging
#[allow(dead_code)]
fn hypervolume_2d_brute_force(points: &[Objectives<2>], reference: Objectives<2>) -> u128 {
    if points.is_empty() {
        return 0;
    }
    
    let mut total = 0u128;
    
    // Check every integer point in the grid from (0,0) to reference
    // Try both exclusive and inclusive bounds
    for x in 0..reference[0] {
        for y in 0..reference[1] {
            // Check if this point (x,y) is dominated by any point in the set
            let dominated = points.iter().any(|p| p[0] <= x && p[1] <= y);
            if dominated {
                total += 1;
            }
        }
    }
    
    println!("Grid points dominated (exclusive bounds [0, ref)):");
    for x in 0..reference[0] {
        for y in 0..reference[1] {
            let dominated = points.iter().any(|p| p[0] <= x && p[1] <= y);
            if dominated {
                println!("  ({}, {})", x, y);
            }
        }
    }
    
    // Also try inclusive bounds
    let mut total_incl = 0u128;
    println!("Grid points dominated (inclusive bounds [0, ref]):");
    for x in 0..=reference[0] {
        for y in 0..=reference[1] {
            let dominated = points.iter().any(|p| p[0] <= x && p[1] <= y);
            if dominated {
                println!("  ({}, {})", x, y);
                total_incl += 1;
            }
        }
    }
    println!("Inclusive result: {}", total_incl);
    
    total
}

/// 2D HV: union area = sum over strips
fn hypervolume_2d_min_u64(points: &mut [Objectives<2>], reference: Objectives<2>) -> u128 {
    if points.is_empty() {
        return 0;
    }

    // Sort by y-coordinate (second dimension)
    points.sort_by_key(|p| p[1]);

    let mut total: u128 = 0;
    let mut prev_y = reference[1];

    for i in (0..points.len()).rev() {
        let curr_y = points[i][1];
        
        if curr_y < prev_y {
            // Find minimum x-coordinate among points[0..=i]
            let min_x = points[..=i].iter().map(|p| p[0]).min().unwrap();
            
            if min_x < reference[0] {
                let width_y = diff_to_u128(prev_y, curr_y);
                let width_x = diff_to_u128(reference[0], min_x);
                total += width_y * width_x;
            }
            
            prev_y = curr_y;
        }
    }

    total
}

/// 3D HV: union volume = sum over "slabs" (u64 version)
fn hypervolume_3d_min_u64(points: &mut [Objectives<3>], reference: Objectives<3>) -> u128 {
    if points.is_empty() {
        return 0;
    }

    // Sort by z-coordinate (third dimension)
    points.sort_by_key(|p| p[2]);

    let mut total: u128 = 0;
    let mut prev = reference[2];
    let mut i = points.len();

    while i > 0 {
        i -= 1;
        let bound = points[i][2];
        
        if bound < prev {
            let width = diff_to_u128(prev, bound);
            
            // Extract 2D points from slice [0..=i]
            let mut points_2d: Vec<Objectives<2>> = points[..=i].iter()
                .map(|p| [p[0], p[1]])
                .collect();
            
            let area2d = hypervolume_2d_min_u64(&mut points_2d, [reference[0], reference[1]]);
            total += width * area2d;
            prev = bound;
        }
        
        // Skip all points with the same z-coordinate to handle ties correctly
        while i > 0 && points[i-1][2] == bound {
            i -= 1;
        }
    }

    total
}

/// 4D HV: union hypervolume = sum over "hyperslabs" (u64 version)
fn hypervolume_4d_min_u64(points: &mut [Objectives<4>], reference: Objectives<4>) -> u128 {
    if points.is_empty() {
        return 0;
    }

    // Sort by w-coordinate (fourth dimension)
    points.sort_by_key(|p| p[3]);

    let mut total: u128 = 0;
    let mut prev = reference[3];
    let mut i = points.len();

    while i > 0 {
        i -= 1;
        let bound = points[i][3];
        
        if bound < prev {
            let width = diff_to_u128(prev, bound);
            
            // Extract 3D points from slice [0..=i]
            let mut points_3d: Vec<Objectives<3>> = points[..=i].iter()
                .map(|p| [p[0], p[1], p[2]])
                .collect();
            
            let volume3d = hypervolume_3d_min_u64(&mut points_3d, [reference[0], reference[1], reference[2]]);
            total += width * volume3d;
            prev = bound;
        }
        
        // Skip all points with the same w-coordinate to handle ties correctly
        while i > 0 && points[i-1][3] == bound {
            i -= 1;
        }
    }

    total
}

pub fn compute<const D: usize, T: HasObjectives<D>>(pareto_front: Vec<T>, reference_point: Objectives<D>) -> u128 {
    match D {
        2 => {
            let mut points_2d: Vec<Objectives<2>> = pareto_front.iter().map(|s| {
            let objectives = *s.objectives();
            [objectives[0], objectives[1]]
            }).collect();
            hypervolume_2d_min_u64(&mut points_2d, [reference_point[0], reference_point[1]])
        }
        3 => {
            let mut points_3d: Vec<Objectives<3>> = pareto_front.iter().map(|s| {
            let objectives = *s.objectives();
            [objectives[0], objectives[1], objectives[2]]
            }).collect();
            hypervolume_3d_min_u64(&mut points_3d, [reference_point[0], reference_point[1], reference_point[2]])
        }
        4 => {
            let mut points_4d: Vec<Objectives<4>> = pareto_front.iter().map(|s| {
            let objectives = *s.objectives();
            [objectives[0], objectives[1], objectives[2], objectives[3]]
            }).collect();
            hypervolume_4d_min_u64(&mut points_4d, [reference_point[0], reference_point[1], reference_point[2], reference_point[3]])
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
        // Use original u64 implementation
        let result_u128 = match dimension {
            2 => {
                let reference: Objectives<2> = [reference_point[0], reference_point[1]];
                let mut objectives: Vec<Objectives<2>> = points.iter().map(|p| [p[0], p[1]]).collect();
                hypervolume_2d_min_u64(&mut objectives, reference)
            }
            3 => {
                let reference: Objectives<3> = [reference_point[0], reference_point[1], reference_point[2]];
                let mut objectives: Vec<Objectives<3>> = points.iter().map(|p| [p[0], p[1], p[2]]).collect();
                hypervolume_3d_min_u64(&mut objectives, reference)
            }
            4 => {
                let reference: Objectives<4> = [reference_point[0], reference_point[1], reference_point[2], reference_point[3]];
                let mut objectives: Vec<Objectives<4>> = points.iter().map(|p| [p[0], p[1], p[2], p[3]]).collect();
                hypervolume_4d_min_u64(&mut objectives, reference)
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unsupported dimension: {}. Only 2D, 3D, and 4D are supported.",
                    dimension
                )));
            }
        };
        result_u128 as f64
    };

    Ok(result)
}

#[inline(always)]
fn diff_to_u128(a: u64, b: u64) -> u128 {
    if a > b {
        (a - b) as u128
    } else {
        0
    }
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
        let mut pts = [[1, 2, 3, 4]];
        let r = [5, 6, 7, 8];
        assert_eq!(hypervolume_4d_min_u64(&mut pts, r), 256);
    }

    #[test]
    fn hv_two_points_cross() {
        let mut pts = [[1, 4, 2, 2], [4, 1, 2, 2]];
        let r = [5, 5, 3, 3];
        assert_eq!(hypervolume_4d_min_u64(&mut pts, r), 7);
    }
    #[test]
    fn hv_empty_front() {
        let mut pts: [[u64; 4]; 0] = [];
        let r = [5, 6, 7, 8];
        assert_eq!(hypervolume_4d_min_u64(&mut pts, r), 0);
    }

    #[test]
    fn hv_identical_points() {
        let mut pts = [[2, 2, 2, 2], [2, 2, 2, 2]];
        let r = [5, 5, 5, 5];
        assert_eq!(hypervolume_4d_min_u64(&mut pts, r), 81);
    }

    #[test]
    fn hv_points_on_reference() {
        let mut pts = [[5, 6, 7, 8]];
        let r = [5, 6, 7, 8];
        assert_eq!(hypervolume_4d_min_u64(&mut pts, r), 0);
    }

    #[test]
    fn hv_points_outside_reference() {
        let mut pts = [[6, 7, 8, 9]];
        let r = [5, 6, 7, 8];
        assert_eq!(hypervolume_4d_min_u64(&mut pts, r), 0);
    }

    #[test]
    fn hv_multiple_non_dominated_points() {
        let mut pts = [
            [1, 2, 3, 4],
            [2, 1, 3, 4],
            [1, 1, 2, 3],
            [3, 3, 3, 3],
        ];
        let r = [5, 5, 5, 5];
        let hv = hypervolume_4d_min_u64(&mut pts, r);
        assert!(hv > 0);
    }

    #[test]
    fn hv_2d_min_u64_basic() {
        let mut pts = [[1, 2], [2, 1]];
        let r = [3, 3];
        assert_eq!(hypervolume_2d_min_u64(&mut pts, r), 3);
    }

    #[test]
    fn hv_3d_min_u64_basic() {
        let mut pts = [[1, 2, 3], [2, 1, 3]];
        let r = [4, 4, 4];
        assert_eq!(hypervolume_3d_min_u64(&mut pts, r), 8);
    }

    #[test]
    fn hv_2d_min_u64_empty() {
        let mut pts: [[u64; 2]; 0] = [];
        let r = [3, 3];
        assert_eq!(hypervolume_2d_min_u64(&mut pts, r), 0);
    }

    #[test]
    fn hv_3d_min_u64_empty() {
        let mut pts: [[u64; 3]; 0] = [];
        let r = [4, 4, 4];
        assert_eq!(hypervolume_3d_min_u64(&mut pts, r), 0);
    }

    #[test]
    fn hv_2d_min_u64_identical_points() {
        let mut pts = [[2, 2], [2, 2]];
        let r = [3, 3];
        assert_eq!(hypervolume_2d_min_u64(&mut pts, r), 1);
    }

    #[test]
    fn hv_3d_min_u64_identical_points() {
        let mut pts = [[2, 2, 2], [2, 2, 2]];
        let r = [4, 4, 4];
        assert_eq!(hypervolume_3d_min_u64(&mut pts, r), 8);
    }

    #[test]
    fn hv_2d_min_u64_points_on_reference() {
        let mut pts = [[3, 3]];
        let r = [3, 3];
        assert_eq!(hypervolume_2d_min_u64(&mut pts, r), 0);
    }

    #[test]
    fn hv_3d_min_u64_points_on_reference() {
        let mut pts = [[4, 4, 4]];
        let r = [4, 4, 4];
        assert_eq!(hypervolume_3d_min_u64(&mut pts, r), 0);
    }

    #[test]
    fn hv_2d_min_u64_points_outside_reference() {
        let mut pts = [[4, 4]];
        let r = [3, 3];
        assert_eq!(hypervolume_2d_min_u64(&mut pts, r), 0);
    }

    #[test]
    fn hv_3d_min_u64_points_outside_reference() {
        let mut pts = [[5, 5, 5]];
        let r = [4, 4, 4];
        assert_eq!(hypervolume_3d_min_u64(&mut pts, r), 0);
    }
}
