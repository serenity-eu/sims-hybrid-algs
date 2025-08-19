//! Bypass coefficient optimization module
//!
//! This module implements the bypass coefficient logic from AUGMECON2 algorithm,
//! which allows skipping grid points based on slack variable values from previous solutions.
//! This optimization can significantly reduce the number of epsilon-constraint problems
//! that need to be solved.

use std::collections::HashMap;

/// Calculator for bypass coefficient optimization
pub struct BypassCalculator {
    /// Ranges for each objective (ideal - nadir)
    objective_ranges: Vec<f64>,
    /// Grid size for each objective
    grid_points: usize,
}

impl BypassCalculator {
    /// Create a new bypass calculator
    #[must_use]
    pub const fn new(objective_ranges: Vec<f64>, grid_points: usize) -> Self {
        Self {
            objective_ranges,
            grid_points,
        }
    }

    /// Calculate how many grid points to skip based on slack values
    ///
    /// The bypass coefficient logic works as follows:
    /// 1. If `slack_i` > 0 for objective i, then we can skip some grid points
    /// 2. The number of points to skip is proportional to `slack_i` / `range_i`
    /// 3. We skip forward in the grid iteration by this amount
    #[must_use]
    pub fn calculate_skip_count(
        &self,
        slack_values: &HashMap<usize, f64>,
        current_grid_point: &[usize],
    ) -> Vec<usize> {
        let mut skip_counts = vec![0; current_grid_point.len()];

        for (obj_idx, &slack_value) in slack_values {
            // Only skip if we have positive slack and the objective index is valid
            if slack_value > 0.0
                && *obj_idx > 0  // Skip primary objective (index 0)
                && (*obj_idx - 1) < self.objective_ranges.len()
                && (*obj_idx - 1) < current_grid_point.len()
            {
                let range = self.objective_ranges[*obj_idx - 1];

                if range > 0.0 && self.grid_points > 1 {
                    // Calculate step size for this objective
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "Converting grid_points to f64 for step size calculation - precision loss acceptable for bypass optimization"
                    )]
                    let step_size = range / (self.grid_points as f64 - 1.0);

                    // Calculate how many steps we can skip
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "floor() ensures positive value and usize range is appropriate for grid coordinates"
                    )]
                    let skip_steps = (slack_value / step_size).floor() as usize;

                    // Ensure we don't skip beyond the grid bounds
                    let current_pos = current_grid_point[*obj_idx - 1];
                    let max_skip = if current_pos < self.grid_points {
                        self.grid_points - current_pos - 1
                    } else {
                        0
                    };

                    skip_counts[*obj_idx - 1] = skip_steps.min(max_skip);
                }
            }
        }

        skip_counts
    }

    /// Calculate the next grid point after applying bypass logic
    ///
    /// This advances the current grid point by the calculated skip amounts
    #[must_use]
    pub fn advance_grid_point(
        &self,
        current_grid_point: &[usize],
        skip_counts: &[usize],
    ) -> Option<Vec<usize>> {
        let mut next_point = current_grid_point.to_vec();
        let mut advanced = false;

        // Apply skip counts to each dimension
        for (i, &skip_count) in skip_counts.iter().enumerate() {
            if i < next_point.len() && skip_count > 0 {
                next_point[i] = (next_point[i] + skip_count).min(self.grid_points - 1);
                advanced = true;
            }
        }

        if advanced {
            // Check if the advanced point is still within bounds
            if next_point.iter().all(|&p| p < self.grid_points) {
                Some(next_point)
            } else {
                None // We've reached the end of the grid
            }
        } else {
            None // No advancement needed
        }
    }

    /// Check if bypass is beneficial for the given slack values
    ///
    /// Returns true if any slack value is significant enough to warrant skipping
    #[must_use]
    pub fn should_apply_bypass(&self, slack_values: &HashMap<usize, f64>) -> bool {
        for (obj_idx, &slack_value) in slack_values {
            if *obj_idx > 0 && (*obj_idx - 1) < self.objective_ranges.len() {
                let range = self.objective_ranges[*obj_idx - 1];
                if range > 0.0 {
                    // Consider bypass if slack is more than 1% of the range
                    let threshold = range * 0.01;
                    if slack_value > threshold {
                        return true;
                    }
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bypass_calculator_creation() {
        let ranges = vec![100.0, 200.0, 150.0];
        let calculator = BypassCalculator::new(ranges, 10);
        assert_eq!(calculator.grid_points, 10);
        assert_eq!(calculator.objective_ranges.len(), 3);
    }

    #[test]
    fn test_skip_count_calculation() {
        let ranges = vec![100.0, 200.0];
        let calculator = BypassCalculator::new(ranges, 11); // 10 intervals, 11 points

        let mut slack_values = HashMap::new();
        slack_values.insert(1, 20.0); // 20% of range for objective 1
        slack_values.insert(2, 40.0); // 20% of range for objective 2

        let current_point = vec![0, 0];
        let skip_counts = calculator.calculate_skip_count(&slack_values, &current_point);

        // step_size for obj 1: 100/10 = 10, skip = 20/10 = 2
        // step_size for obj 2: 200/10 = 20, skip = 40/20 = 2
        assert_eq!(skip_counts, vec![2, 2]);
    }

    #[test]
    fn test_advance_grid_point() {
        let ranges = vec![100.0, 200.0];
        let calculator = BypassCalculator::new(ranges, 10);

        let current_point = vec![1, 2];
        let skip_counts = vec![2, 1];

        let next_point = calculator.advance_grid_point(&current_point, &skip_counts);
        assert_eq!(next_point, Some(vec![3, 3]));
    }

    #[test]
    fn test_should_apply_bypass() {
        let ranges = vec![100.0, 200.0];
        let calculator = BypassCalculator::new(ranges, 10);

        let mut slack_values = HashMap::new();
        slack_values.insert(1, 5.0); // 5% of range, should apply

        assert!(calculator.should_apply_bypass(&slack_values));

        slack_values.insert(1, 0.5); // 0.5% of range, should not apply
        assert!(!calculator.should_apply_bypass(&slack_values));
    }
}
