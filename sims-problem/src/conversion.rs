use log::debug;
use pls::objectives::ObjectiveType;
use pls::problem::Problem;
use pls::solution::ImageSet;
use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;

use crate::solution::Solution;

/// Wrapper for PLS solutions that includes timestamp information for conversion
pub struct PlsSolutionWithTimestamp<'a, T: ImageSet<D>, const D: usize> {
    pub solution: &'a T,
    pub timestamp_us: u64,
    pub problem: &'a Problem<T, D>,
}

impl<'a, T: ImageSet<D>, const D: usize> PlsSolutionWithTimestamp<'a, T, D> {
    pub fn new(solution: &'a T, timestamp_us: u64, problem: &'a Problem<T, D>) -> Self {
        Self {
            solution,
            timestamp_us,
            problem,
        }
    }
}

/// Helper function to extract objective values by type from solution
fn extract_objective_values<T: ImageSet<D> + std::fmt::Debug, const D: usize>(
    objectives: &[u64],
    problem: &Problem<T, D>,
) -> (i32, i32, Option<i32>, Option<i32>) {
    let mut cost = 0i32;
    let mut cloudy_area = 0i32;
    let mut min_resolutions_sum: Option<i32> = None;
    let mut max_incidence_angle: Option<i32> = None;

    for (i, objective_type) in problem.objectives.iter().enumerate() {
        match objective_type {
            ObjectiveType::TotalCost => {
                cost = objectives[i] as i32;
            }
            ObjectiveType::CloudyArea => {
                cloudy_area = objectives[i] as i32;
            }
            ObjectiveType::MinResolution => {
                min_resolutions_sum = Some(objectives[i] as i32);
            }
            ObjectiveType::MaxIncidenceAngle => {
                max_incidence_angle = Some(objectives[i] as i32);
            }
            _ => panic!(
                "Unsupported objective type: {:?} at index {}",
                objective_type, i
            ),
        }
    }

    (cost, cloudy_area, min_resolutions_sum, max_incidence_angle)
}

/// Convert 2D PLS BitsetEncodedSolution to Python Solution
impl<'a> From<PlsSolutionWithTimestamp<'a, BitsetEncodedSolution<2>, 2>> for Solution {
    fn from(val: PlsSolutionWithTimestamp<'a, BitsetEncodedSolution<2>, 2>) -> Self {
        let pls_solution = val.solution;
        let timestamp_us = val.timestamp_us;
        let problem = val.problem;

        // Debug logging: Show raw PLS solution data
        let selected_images: Vec<usize> = pls_solution.selected_images().collect();
        debug!(
            "Converting 2D PLS solution: {} selected images, objectives: {:?}",
            selected_images.len(),
            pls_solution.objectives
        );

        debug!("Selected images (0-based): {selected_images:?}");

        // Extract objective values using objective definitions
        let (cost, cloudy_area, min_resolutions_sum, max_incidence_angle) =
            extract_objective_values(&pls_solution.objectives, problem);

        debug!("Created 2D Python solution: cost={cost}, cloudy_area={cloudy_area}");

        Solution::create(
            selected_images,
            cost,
            cloudy_area,
            timestamp_us,
            max_incidence_angle,
            min_resolutions_sum,
        )
    }
}

/// Convert 3D PLS BitsetEncodedSolution to Python Solution
impl<'a> From<PlsSolutionWithTimestamp<'a, BitsetEncodedSolution<3>, 3>> for Solution {
    fn from(val: PlsSolutionWithTimestamp<'a, BitsetEncodedSolution<3>, 3>) -> Self {
        let pls_solution = val.solution;
        let timestamp_us = val.timestamp_us;
        let problem = val.problem;

        // Debug logging: Show raw PLS solution data
        let selected_images: Vec<usize> = pls_solution.selected_images().collect();
        debug!(
            "Converting 3D PLS solution: {} selected images, objectives: {:?}",
            selected_images.len(),
            pls_solution.objectives
        );

        // Extract objective values using objective definitions
        let (cost, cloudy_area, min_resolutions_sum, max_incidence_angle) =
            extract_objective_values(&pls_solution.objectives, problem);

        debug!("Created 3D Python solution: cost={cost}, cloudy_area={cloudy_area}, min_resolutions_sum={min_resolutions_sum:?}, max_incidence_angle={max_incidence_angle:?}");

        Solution::create(
            selected_images,
            cost,
            cloudy_area,
            timestamp_us,
            max_incidence_angle,
            min_resolutions_sum,
        )
    }
}

/// Convert 4D PLS BitsetEncodedSolution to Python Solution
impl<'a> From<PlsSolutionWithTimestamp<'a, BitsetEncodedSolution<4>, 4>> for Solution {
    fn from(val: PlsSolutionWithTimestamp<'a, BitsetEncodedSolution<4>, 4>) -> Self {
        let pls_solution = val.solution;
        let timestamp_us = val.timestamp_us;
        let problem = val.problem;

        // Debug logging: Show raw PLS solution data
        let selected_images: Vec<usize> = pls_solution.selected_images().collect();
        debug!(
            "Converting 4D PLS solution: {} selected images, objectives: {:?}",
            selected_images.len(),
            pls_solution.objectives
        );

        // Extract objective values using objective definitions
        let (cost, cloudy_area, min_resolutions_sum, max_incidence_angle) =
            extract_objective_values(&pls_solution.objectives, problem);

        debug!("Created 4D Python solution: cost={cost}, cloudy_area={cloudy_area}, min_resolutions_sum={min_resolutions_sum:?}, max_incidence_angle={max_incidence_angle:?}");

        Solution::create(
            selected_images,
            cost,
            cloudy_area,
            timestamp_us,
            max_incidence_angle,
            min_resolutions_sum,
        )
    }
}
