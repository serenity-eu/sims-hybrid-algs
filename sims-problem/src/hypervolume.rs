use pareto::{HasObjectives, Objectives};
use pyo3::prelude::*;
use crate::solution::Solution;

/// Compute hypervolume for 4D minimization front.
/// Input: non-dominated points (Pareto front), reference point >= all coords.
/// Returns exact HV as u128.
pub fn hypervolume_4d_min_u64(points: &mut [Objectives<4>], reference: Objectives<4>) -> u128 {
    // Sort by 4th dim
    points.sort_by_key(|p| p[3]);

    let mut total: u128 = 0;
    let mut prev = reference[3];
    let n = points.len();
    let mut i = n;

    while i > 0 {
        i -= 1;
        let bound = points[i][3];
        
        if bound < prev {
            let width = diff_to_u128(prev, bound);
            
            // Extract 3D points from slice [0..=i]
            let mut points_3d: Vec<[u64; 3]> = points[..=i].iter()
                .map(|p| [p[0], p[1], p[2]])
                .collect();
                
            let vol3d = hypervolume_3d_min_u64(&mut points_3d, [reference[0], reference[1], reference[2]]);
            total += width * vol3d;
            prev = bound;
        }
        
        // Skip all points with the same coordinate to handle ties correctly
        while i > 0 && points[i-1][3] == bound {
            i -= 1;
        }
    }

    total
}

/// 3D HV with slice view
fn hypervolume_3d_min_u64(points: &mut [Objectives<3>], reference: Objectives<3>) -> u128 {
    points.sort_by_key(|p| p[2]);

    let mut total: u128 = 0;
    let mut prev = reference[2];
    let n = points.len();
    let mut i = n;

    while i > 0 {
        i -= 1;
        let bound = points[i][2];
        
        if bound < prev {
            let width = diff_to_u128(prev, bound);
            
            // Extract 2D points from slice [0..=i]
            let mut points_2d: Vec<[u64; 2]> = points[..=i].iter()
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

/// Compute hypervolume for a Pareto front of solutions.
/// 
/// # Arguments
/// * `solutions` - List of Solution objects
/// * `reference_point` - Reference point as a list of objective values
/// 
/// # Returns
/// The hypervolume as an integer
/// 
/// # Example
/// ```python
/// # 2D case with solutions
/// solutions = [solution1, solution2]  # Solution objects
/// reference = [1000, 500]  # reference for (cost, cloudy_area)
/// hv = compute_hypervolume_solutions(solutions, reference)
/// ```
#[pyfunction]
pub fn compute_hypervolume_solutions(solutions: Vec<Solution>, reference_point: Vec<u64>) -> PyResult<u128> {
    if solutions.is_empty() {
        return Ok(0);
    }

    let dimension = reference_point.len();
    
    let result = match dimension {
        2 => {
            let reference: Objectives<2> = [reference_point[0], reference_point[1]];
            let mut points: Vec<Objectives<2>> = solutions.iter().map(|s| s.objectives_2d()).collect();
            hypervolume_2d_min_u64(&mut points, reference)
        }
        3 => {
            let reference: Objectives<3> = [reference_point[0], reference_point[1], reference_point[2]];
            let mut points: Vec<Objectives<3>> = solutions.iter().map(|s| s.objectives_3d()).collect();
            hypervolume_3d_min_u64(&mut points, reference)
        }
        4 => {
            let reference: Objectives<4> = [reference_point[0], reference_point[1], reference_point[2], reference_point[3]];
            let mut points: Vec<Objectives<4>> = solutions.iter().map(|s| s.objectives_4d()).collect();
            hypervolume_4d_min_u64(&mut points, reference)
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported dimension: {}. Only 2D, 3D, and 4D are supported.",
                dimension
            )));
        }
    };

    Ok(result)
}

/// Compute hypervolume for a Pareto front.
/// 
/// # Arguments
/// * `points` - List of points, where each point is a list of objective values
/// * `reference_point` - Reference point as a list of objective values
/// 
/// # Returns
/// The hypervolume as an integer
/// 
/// # Example
/// ```python
/// # 2D case
/// points = [[1, 2], [2, 1]]
/// reference = [3, 3]
/// hv = compute_hypervolume(points, reference)
/// assert hv == 3
/// ```
#[pyfunction]
pub fn compute_hypervolume(points: Vec<Vec<u64>>, reference_point: Vec<u64>) -> PyResult<u128> {
    if points.is_empty() {
        return Ok(0);
    }

    let dimension = reference_point.len();
    
    // Validate that all points have the same dimension
    for point in &points {
        if point.len() != dimension {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "All points must have the same dimension as reference point ({}), but found point with dimension {}",
                dimension, point.len()
            )));
        }
    }

    let result = match dimension {
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
