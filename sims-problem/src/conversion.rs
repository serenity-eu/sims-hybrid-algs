use log::debug;
use pls::objectives::ObjectiveState;
use pls::problem_bitset::ProblemBitset;
use pls::solution::{bitset_encoded_solution::BitsetEncodedSolution, EncodedSolution};

use crate::solution::Solution;

/// Helper function to extract objective values by type from solution
fn extract_objective_values<const D: usize>(
    objectives: &[u64],
    problem: &ProblemBitset<D>,
) -> (Option<u64>, Option<u64>, Option<u64>, Option<u64>) {
    let mut cost: Option<u64> = None;
    let mut cloudy_area: Option<u64> = None;
    let mut min_resolutions_sum: Option<u64> = None;
    let mut max_incidence_angle: Option<u64> = None;

    for (i, objective_state) in problem.objectives.iter().enumerate() {
        match objective_state {
            ObjectiveState::TotalCost { .. } => {
                cost = Some(objectives[i]);
            }
            ObjectiveState::CloudyArea { .. } => {
                cloudy_area = Some(objectives[i]);
            }
            ObjectiveState::MinResolution { .. } => {
                min_resolutions_sum = Some(objectives[i]);
            }
            ObjectiveState::MaxIncidenceAngle { .. } => {
                max_incidence_angle = Some(objectives[i]);
            }
        }
    }

    (cost, cloudy_area, min_resolutions_sum, max_incidence_angle)
}

/// Convert a D-dimensional PLS `BitsetEncodedSolution` to a Python `Solution`.
impl<const D: usize>
    From<(
        &BitsetEncodedSolution<ProblemBitset<D>, D>,
        &ProblemBitset<D>,
    )> for Solution
{
    fn from(
        val: (
            &BitsetEncodedSolution<ProblemBitset<D>, D>,
            &ProblemBitset<D>,
        ),
    ) -> Self {
        let (pls_solution, problem) = val;
        let timestamp_us = pls_solution.timestamp().as_micros() as u64;

        let selected_images: Vec<usize> = pls_solution.selected_images().collect();
        debug!(
            "Converting {D}D PLS solution: {} selected images, objectives: {:?}",
            selected_images.len(),
            pls_solution.objectives
        );

        let (cost, cloudy_area, min_resolutions_sum, max_incidence_angle) =
            extract_objective_values(&pls_solution.objectives, problem);

        debug!(
            "Created {D}D Python solution: cost={cost:?}, cloudy_area={cloudy_area:?}, \
             min_resolutions_sum={min_resolutions_sum:?}, max_incidence_angle={max_incidence_angle:?}"
        );

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
