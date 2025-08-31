use pareto::{HasObjectives, Objectives};

/// Compute hypervolume for 4D minimization front.
/// Input: non-dominated points (Pareto front), reference point >= all coords.
/// Returns exact HV as u128.
pub fn hypervolume_4d_min_u64(points: &mut [[u64; 4]], reference: [u64; 4]) -> u128 {
    // Sort by 4th dim
    points.sort_by_key(|p| p[3]);

    let mut total: u128 = 0;
    let mut prev = reference[3];
    let n = points.len();

    for i in (0..n).rev() {
        let bound = points[i][3];
        if bound < prev {
            let width = diff_to_u128(prev, bound);
            // Compute 3D volume for slice [0..=i]
            let slice3d: &mut [[u64; 3]] = unsafe {
                // Reinterpret as 3D slice view
                &mut *(&mut points[..=i] as *mut [[u64; 4]] as *mut [[u64; 3]])
            };
            let vol3d = hypervolume_3d_min_u64(slice3d, [reference[0], reference[1], reference[2]]);
            total += width * vol3d;
            prev = bound;
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

    for i in (0..n).rev() {
        let bound = points[i][2];
        if bound < prev {
            let width = diff_to_u128(prev, bound);
            // Compute 2D area for slice [0..=i]
            let slice2d: &mut [[u64; 2]] =
                unsafe { &mut *(&mut points[..=i] as *mut [[u64; 3]] as *mut [[u64; 2]]) };
            let area2d = hypervolume_2d_min_u64(slice2d, [reference[0], reference[1]]);
            total += width * area2d;
            prev = bound;
        }
    }

    total
}

/// 2D HV: union area = sum over strips
fn hypervolume_2d_min_u64(points: &mut [[u64; 2]], reference: [u64; 2]) -> u128 {
    points.sort_by_key(|p| p[1]);

    let mut total: u128 = 0;
    let mut prev = reference[1];
    let n = points.len();

    let mut min_x = reference[0]; // running min for x

    for i in (0..n).rev() {
        let bound = points[i][1];
        if bound < prev {
            // update running min of x among points[0..=i]
            if points[i][0] < min_x {
                min_x = points[i][0];
            }
            let width = diff_to_u128(prev, bound);
            let len_x = diff_to_u128(reference[0], min_x);
            total += width * len_x;
            prev = bound;
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
}
