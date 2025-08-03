use log::debug;
use pls::solution::EncodedSolution;

use crate::solution::Solution;

/// Wrapper for PLS solutions that includes timestamp information for conversion
pub struct PlsSolutionWithTimestamp<'a, T> {
    pub solution: &'a T,
    pub timestamp_us: u64,
}

impl<'a, T> PlsSolutionWithTimestamp<'a, T> {
    pub fn new(solution: &'a T, timestamp_us: u64) -> Self {
        Self {
            solution,
            timestamp_us,
        }
    }
}

/// Convert 2D PLS EncodedSolution to Python Solution
impl<'a> From<PlsSolutionWithTimestamp<'a, EncodedSolution<2>>> for Solution {
    fn from(val: PlsSolutionWithTimestamp<'a, EncodedSolution<2>>) -> Self {
        let pls_solution = val.solution;
        let timestamp_us = val.timestamp_us;

        // Debug logging: Show raw PLS solution data
        let selected_images: Vec<usize> = pls_solution.selected_images().collect();
        debug!(
            "Converting PLS solution: {} selected images, objectives: {:?}",
            selected_images.len(),
            pls_solution.objectives
        );

        debug!("Selected images (0-based): {selected_images:?}");

        let cost = pls_solution.objectives[0] as i32;
        let cloudy_area = pls_solution.objectives[1] as i32;

        debug!("Created Python solution: cost={cost}, cloudy_area={cloudy_area}");

        Solution::create(
            selected_images,
            cost,
            cloudy_area,
            timestamp_us,
            None, // max_incidence_angle will be computed later if needed
            None, // min_resolutions_sum will be computed later if needed
        )
    }
}

/// Convert 4D PLS BitsetEncodedSolution to Python Solution (includes resolution and incidence angle objectives)
impl<'a> From<PlsSolutionWithTimestamp<'a, EncodedSolution<4>>> for Solution {
    fn from(val: PlsSolutionWithTimestamp<'a, EncodedSolution<4>>) -> Self {
        let pls_solution = val.solution;
        let timestamp_us = val.timestamp_us;

        // Debug logging: Show raw PLS solution data
        let selected_images: Vec<usize> = pls_solution.selected_images().collect();
        debug!(
            "Converting 4D PLS solution: {} selected images, objectives: {:?}",
            selected_images.len(),
            pls_solution.objectives
        );

        let cost = pls_solution.objectives[0] as i32;
        let cloudy_area = pls_solution.objectives[1] as i32;
        let min_resolution_sum = pls_solution.objectives[2] as i32;
        let max_incidence_angle = pls_solution.objectives[3] as i32;

        debug!("Created 4D Python solution: cost={cost}, cloudy_area={cloudy_area}, min_resolution_sum={min_resolution_sum}, max_incidence_angle={max_incidence_angle}");

        Solution::create(
            selected_images,
            cost,
            cloudy_area,
            timestamp_us,
            Some(max_incidence_angle),
            Some(min_resolution_sum),
        )
    }
}
