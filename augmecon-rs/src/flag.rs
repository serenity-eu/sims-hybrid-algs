//! Flag Array Implementation
//!
//! This module provides the flag array functionality for AUGMECON algorithms,
//! which allows tracking grid point completion status and implementing skip logic
//! for performance optimization.
//!
//! The flag array is a key optimization technique that:
//! 1. Tracks which grid points have been processed or should be skipped
//! 2. Enables jump/skip logic based on infeasibility or bypass coefficients
//! 3. Prevents redundant computation of grid points that are known to be infeasible
//!
//! This implementation follows the standard flag array algorithm used in
//! augmented ε-constraint methods for efficient multi-objective optimization.

use std::collections::HashMap;

/// Flag array for tracking grid point status and implementing skip logic
///
/// The flag array maps grid point coordinates to status values:
/// - 0: Not processed / no flag
/// - Positive integer: Skip count (number of grid points to skip)
/// - Can be extended for other status indicators
#[derive(Debug, Clone)]
pub struct FlagArray {
    /// The internal flag storage mapping grid points to status values
    flags: HashMap<Vec<usize>, i32>,
}

impl Default for FlagArray {
    fn default() -> Self {
        Self::new()
    }
}

impl FlagArray {
    /// Create a new flag array
    #[must_use]
    pub fn new() -> Self {
        Self {
            flags: HashMap::new(),
        }
    }

    /// Set flag values for a range of grid points
    ///
    /// This is used to mark grid points that should be skipped based on:
    /// - Early exit when infeasibility is detected
    /// - Bypass coefficient calculations
    ///
    /// # Arguments
    /// * `value` - The flag value to set (typically skip count)
    /// * `objective_indices` - The objective indices to iterate over
    /// * `range_fn` - Function that returns the range of grid points for each objective
    pub fn set_range<F>(&mut self, value: i32, objective_indices: &[usize], range_fn: F)
    where
        F: Fn(usize) -> std::ops::Range<usize>,
    {
        // Generate all combinations of grid points in the specified ranges
        let ranges: Vec<_> = objective_indices.iter().map(|&i| range_fn(i)).collect();
        let combinations = cartesian_product(&ranges);

        // Set the flag value for all combinations
        for combination in combinations {
            self.flags.insert(combination, value);
        }
    }

    /// Set flag for early exit range
    ///
    /// Used when a grid point is infeasible and we want to skip related points
    pub fn set_early_exit_range(
        &mut self,
        grid_point: &[usize],
        skip_count: i32,
        objective_indices: &[usize],
        grid_size: usize,
    ) {
        self.set_range(skip_count, objective_indices, |obj_idx| {
            if obj_idx == 0 {
                // For the first objective, only mark the current point
                grid_point[obj_idx]..grid_point[obj_idx] + 1
            } else {
                // For other objectives, mark from current to end
                grid_point[obj_idx]..grid_size
            }
        });
    }

    /// Set flag for bypass range
    ///
    /// Used when bypass coefficients indicate we can skip ahead
    pub fn set_bypass_range(
        &mut self,
        grid_point: &[usize],
        bypass_values: &[i32],
        objective_indices: &[usize],
    ) {
        self.set_range(bypass_values[0] + 1, objective_indices, |obj_idx| {
            if obj_idx == 0 {
                // For the first objective, only mark the current point
                grid_point[obj_idx]..grid_point[obj_idx] + 1
            } else if obj_idx - 1 < bypass_values.len() {
                // For other objectives, extend by bypass value
                #[expect(
                    clippy::cast_sign_loss,
                    reason = "bypass_values are always non-negative in this context"
                )]
                let bypass_offset = bypass_values[obj_idx - 1] as usize;
                grid_point[obj_idx]..grid_point[obj_idx] + bypass_offset + 1
            } else {
                // Fallback to current point only
                grid_point[obj_idx]..grid_point[obj_idx] + 1
            }
        });
    }

    /// Get flag value for a grid point
    ///
    /// Returns 0 if no flag is set, otherwise returns the flag value
    #[must_use]
    pub fn get(&self, grid_point: &[usize]) -> i32 {
        self.flags.get(grid_point).copied().unwrap_or(0)
    }

    /// Calculate jump value based on current position and flag value
    ///
    /// This implements the jump optimization logic for the flag array algorithm
    #[must_use]
    pub fn calculate_jump(current_pos: usize, flag_value: i32, end_pos: usize) -> usize {
        if flag_value <= 0 {
            return 0;
        }

        // Jump is minimum of flag value and distance to end
        #[expect(
            clippy::cast_sign_loss,
            reason = "flag_value is checked to be positive before this call"
        )]
        (flag_value as usize).min(end_pos.saturating_sub(current_pos))
    }

    /// Clear all flags
    pub fn clear(&mut self) {
        self.flags.clear();
    }

    /// Get number of flags set
    #[must_use]
    pub fn len(&self) -> usize {
        self.flags.len()
    }

    /// Check if flag array is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.flags.is_empty()
    }
}

/// Generate cartesian product of ranges
///
/// Helper function to generate all combinations of grid points
fn cartesian_product(ranges: &[std::ops::Range<usize>]) -> Vec<Vec<usize>> {
    if ranges.is_empty() {
        return vec![vec![]];
    }

    let mut result = vec![];
    cartesian_product_recursive(ranges, 0, &mut vec![], &mut result);
    result
}

/// Recursive helper for cartesian product
fn cartesian_product_recursive(
    ranges: &[std::ops::Range<usize>],
    index: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if index >= ranges.len() {
        result.push(current.clone());
        return;
    }

    for i in ranges[index].clone() {
        current.push(i);
        cartesian_product_recursive(ranges, index + 1, current, result);
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flag_array_creation() {
        let flag_array = FlagArray::new();
        assert!(flag_array.is_empty());
        assert_eq!(flag_array.len(), 0);
    }

    #[test]
    fn test_set_and_get_flag() {
        let mut flag_array = FlagArray::new();

        // Set flag for specific ranges
        flag_array.set_range(3, &[0, 1], |i| match i {
            0 => 1..2,
            1 => 2..4,
            _ => 0..1,
        });

        // Check that flags are set correctly
        assert_eq!(flag_array.get(&[1, 2]), 3);
        assert_eq!(flag_array.get(&[1, 3]), 3);
        assert_eq!(flag_array.get(&[0, 0]), 0); // Not in range
    }

    #[test]
    fn test_early_exit_range() {
        let mut flag_array = FlagArray::new();
        let grid_point = vec![2, 3];

        flag_array.set_early_exit_range(&grid_point, 5, &[0, 1], 10);

        // First objective: only current point
        assert_eq!(flag_array.get(&[2, 3]), 5);
        assert_eq!(flag_array.get(&[1, 3]), 0); // Different first objective

        // Second objective: current to end
        assert_eq!(flag_array.get(&[2, 4]), 5);
        assert_eq!(flag_array.get(&[2, 9]), 5);
    }

    #[test]
    fn test_bypass_range() {
        let mut flag_array = FlagArray::new();
        let grid_point = vec![1, 2];
        let bypass_values = vec![2, 1];

        flag_array.set_bypass_range(&grid_point, &bypass_values, &[0, 1]);

        // Should set flag value of bypass_values[0] + 1 = 3
        assert_eq!(flag_array.get(&[1, 2]), 3);
        assert_eq!(flag_array.get(&[1, 3]), 3); // Extended by bypass value
    }

    #[test]
    fn test_calculate_jump() {
        // Normal jump
        assert_eq!(FlagArray::calculate_jump(5, 10, 20), 10);

        // Jump limited by distance to end
        assert_eq!(FlagArray::calculate_jump(15, 10, 20), 5);

        // No jump for zero or negative flag
        assert_eq!(FlagArray::calculate_jump(5, 0, 20), 0);
        assert_eq!(FlagArray::calculate_jump(5, -1, 20), 0);
    }

    #[test]
    fn test_cartesian_product() {
        let ranges = vec![0..2, 1..3];
        let result = cartesian_product(&ranges);

        let expected = vec![vec![0, 1], vec![0, 2], vec![1, 1], vec![1, 2]];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_cartesian_product_empty() {
        let ranges = vec![];
        let result = cartesian_product(&ranges);
        assert_eq!(result, vec![vec![]]);
    }
}
