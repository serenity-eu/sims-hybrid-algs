use log::debug;
use pls::objectives::ObjectiveType;
use pls::problem::Problem;
use pls::solution::{bitset_encoded_solution::BitsetEncodedSolution, EncodedSolution};

use crate::solution::Solution;

/// Helper function to extract objective values by type from solution
fn extract_objective_values<const D: usize>(
    objectives: &[u64],
    problem: &Problem<BitsetEncodedSolution<D>, D>,
) -> (Option<u64>, Option<u64>, Option<u64>, Option<u64>) {
    let mut cost: Option<u64> = None;
    let mut cloudy_area: Option<u64> = None;
    let mut min_resolutions_sum: Option<u64> = None;
    let mut max_incidence_angle: Option<u64> = None;

    for (i, objective_type) in problem.objectives.iter().enumerate() {
        match objective_type {
            ObjectiveType::TotalCost => {
                cost = Some(objectives[i]);
            }
            ObjectiveType::CloudyArea => {
                cloudy_area = Some(objectives[i]);
            }
            ObjectiveType::MinResolution => {
                min_resolutions_sum = Some(objectives[i]);
            }
            ObjectiveType::MaxIncidenceAngle => {
                max_incidence_angle = Some(objectives[i]);
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
impl
    From<(
        &BitsetEncodedSolution<2>,
        &Problem<BitsetEncodedSolution<2>, 2>,
    )> for Solution
{
    fn from(
        val: (
            &BitsetEncodedSolution<2>,
            &Problem<BitsetEncodedSolution<2>, 2>,
        ),
    ) -> Self {
        let (pls_solution, problem) = val;
        let timestamp_us = pls_solution.timestamp().as_micros() as u64;

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

        debug!("Created 2D Python solution: cost={cost:?}, cloudy_area={cloudy_area:?}");

        Solution::create(
            selected_images,
            cost,
            cloudy_area,
            timestamp_us,
            max_incidence_angle,
            min_resolutions_sum,
        )
        .expect("PLS solution should always have at least 2 objectives set")
    }
}

/// Convert 3D PLS BitsetEncodedSolution to Python Solution
impl
    From<(
        &BitsetEncodedSolution<3>,
        &Problem<BitsetEncodedSolution<3>, 3>,
    )> for Solution
{
    fn from(
        val: (
            &BitsetEncodedSolution<3>,
            &Problem<BitsetEncodedSolution<3>, 3>,
        ),
    ) -> Self {
        let (pls_solution, problem) = val;
        let timestamp_us = pls_solution.timestamp().as_micros() as u64;

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

        debug!("Created 3D Python solution: cost={cost:?}, cloudy_area={cloudy_area:?}, min_resolutions_sum={min_resolutions_sum:?}, max_incidence_angle={max_incidence_angle:?}");

        Solution::create(
            selected_images,
            cost,
            cloudy_area,
            timestamp_us,
            max_incidence_angle,
            min_resolutions_sum,
        )
        .expect("PLS solution should always have at least 2 objectives set")
    }
}

/// Convert 4D PLS BitsetEncodedSolution to Python Solution  
impl
    From<(
        &BitsetEncodedSolution<4>,
        &Problem<BitsetEncodedSolution<4>, 4>,
    )> for Solution
{
    fn from(
        val: (
            &BitsetEncodedSolution<4>,
            &Problem<BitsetEncodedSolution<4>, 4>,
        ),
    ) -> Self {
        let (pls_solution, problem) = val;
        let timestamp_us = pls_solution.timestamp().as_micros() as u64;
        
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

        debug!("Created 4D Python solution: cost={cost:?}, cloudy_area={cloudy_area:?}, min_resolutions_sum={min_resolutions_sum:?}, max_incidence_angle={max_incidence_angle:?}");

        Solution::create(
            selected_images,
            cost,
            cloudy_area,
            timestamp_us,
            max_incidence_angle,
            min_resolutions_sum,
        )
        .expect("PLS solution should always have at least 2 objectives set")
    }
}
