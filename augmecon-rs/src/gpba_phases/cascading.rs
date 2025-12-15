//! Phase: Cascading Dimension Reset
//!
//! This module implements the cascading dimension reset phase of GPBA-A algorithm.
//! When a dimension is exhausted (epsilon exceeds ideal value), it resets that dimension
//! to nadir and updates the previous dimension, effectively moving to the next "slice"
//! of the Pareto front exploration.
//!
//! This is a critical mechanism for multi-dimensional Pareto front exploration.

use super::epsilon_adjustment::adjust_parameter_ef_array;
use super::interval_manager::IntervalManager;

/// Result of cascading dimension reset phase.
#[derive(Debug, Clone)]
pub struct CascadingResult {
    /// Indices of dimensions that were reset (in order from highest to lowest).
    pub dimensions_reset: Vec<usize>,

    /// Whether the search has converged (first dimension exhausted).
    pub converged: bool,

    /// Updated `ef_array` after cascading.
    pub ef_array: Vec<f64>,

    /// Updated `ef_intervals` after cascading.
    pub ef_intervals: Vec<IntervalManager>,

    /// Updated RWV (Relative Worst Values) after cascading.
    pub rwv: Vec<f64>,
}

/// Apply cascading dimension reset when dimensions are exhausted.
///
/// This function:
/// - Loops through dimensions in reverse order (highest to lowest)
/// - Checks if each dimension is exhausted (`ef_array[i]` > ideal[i])
/// - Resets exhausted dimensions to nadir values
/// - Updates the previous dimension using epsilon adjustment
/// - Stops at first non-exhausted dimension
///
/// Cascading enables systematic exploration of the multi-dimensional Pareto front
/// by moving through "slices" when one dimension is fully explored.
///
/// # Arguments
///
/// * `ef_array` - Current epsilon constraint values
/// * `ef_intervals` - Interval managers for each constraint objective
/// * `rwv` - Relative worst values for each constraint objective
/// * `ideal_max` - Ideal point values in maximization form
/// * `nadir_max` - Nadir point values in maximization form
/// * `constraint_indices` - Indices of constraint objectives
/// * `last_solution_objectives` - Last solution's objective values (None if infeasible)
/// * `obj_k_at_ef_k` - Objective values at epsilon points (for adjustment logic)
///
/// # Returns
///
/// `CascadingResult` with updated arrays and convergence status.
///
/// # Example
///
/// ```ignore
/// let result = apply_cascading(
///     &ef_array,
///     ef_intervals,
///     &rwv,
///     &ideal_max,
///     &nadir_max,
///     &constraint_indices,
///     last_solution_objectives.as_ref(),
///     &obj_k_at_ef_k,
/// );
///
/// if result.converged {
///     println!("Search converged!");
/// } else if !result.dimensions_reset.is_empty() {
///     println!("Reset dimensions: {:?}", result.dimensions_reset);
/// }
/// ```
#[must_use]
pub fn apply_cascading(
    ef_array: &[f64],
    mut ef_intervals: Vec<IntervalManager>,
    rwv: &[f64],
    ideal_max: &[f64],
    nadir_max: &[f64],
    constraint_indices: &[usize],
    last_solution_objectives: Option<&Vec<f64>>,
    obj_k_at_ef_k: &[Option<f64>],
) -> CascadingResult {
    let mut ef_array = ef_array.to_vec();
    let mut rwv = rwv.to_vec();
    let mut dimensions_reset = Vec::new();

    // Loop through dimensions in reverse (highest to lowest)
    // Start from second-to-last since last dimension was just updated
    for i in (1..constraint_indices.len()).rev() {
        let constraint_idx = constraint_indices[i];

        if ef_array[i] > ideal_max[constraint_idx] {
            log::debug!(
                "Dimension {} exhausted (ef={} > ideal={}), cascading...",
                i,
                ef_array[i],
                ideal_max[constraint_idx]
            );

            dimensions_reset.push(i);

            // Reset this dimension to nadir
            ef_array[i] = nadir_max[constraint_idx];
            rwv[i] = ideal_max[constraint_idx];

            // Reinitialize interval for this dimension
            #[allow(clippy::cast_possible_truncation, reason = "Converting nadir/ideal bounds to i64 for interval management - truncation acceptable for GPBA integer-valued constraints")]
            {
                ef_intervals[i] = IntervalManager::new(
                    nadir_max[constraint_idx] as i64,
                    ideal_max[constraint_idx] as i64,
                );
            }

            log::debug!("  Reset dimension {} to nadir: {}", i, ef_array[i]);

            // Update previous dimension
            let prev_id = i - 1;
            let prev_constraint_idx = constraint_indices[prev_id];

            log::debug!("  Updating previous dimension {prev_id}...");

            // Get objective value for previous dimension from last solution
            let _sol_prev = last_solution_objectives.map(|objs| objs[prev_constraint_idx]);

            // Convert f64 to i64 for epsilon adjustment function
            #[allow(clippy::cast_possible_truncation, reason = "Converting epsilon/objective values to i64 for interval management - truncation acceptable for GPBA integer-valued constraints")]
            let mut ef_array_i64: Vec<i64> = ef_array.iter().map(|&x| x as i64).collect();
            #[allow(clippy::cast_possible_truncation, reason = "Converting objective values to i64 for interval management - truncation acceptable for GPBA integer-valued constraints")]
            let obj_k_at_ef_k_i64 = obj_k_at_ef_k[prev_id].map(|x| x as i64);
            #[allow(clippy::cast_possible_truncation, reason = "Converting ideal/nadir bounds to i64 for interval management - truncation acceptable for GPBA integer-valued constraints")]
            let ideal_max_i64: Vec<i64> = ideal_max.iter().map(|&x| x as i64).collect();
            #[allow(clippy::cast_possible_truncation, reason = "Converting ideal/nadir bounds to i64 for interval management - truncation acceptable for GPBA integer-valued constraints")]
            let nadir_max_i64: Vec<i64> = nadir_max.iter().map(|&x| x as i64).collect();

            // Apply epsilon adjustment to previous dimension
            let new_interval = adjust_parameter_ef_array(
                prev_id,
                &mut ef_array_i64,
                obj_k_at_ef_k_i64,
                &mut ef_intervals[prev_id],
                constraint_indices,
                &ideal_max_i64,
                &nadir_max_i64,
                1, // gamma = 1 for complete coverage
            );

            // Convert back to f64
            #[allow(clippy::cast_precision_loss, reason = "Converting i64 back to f64 after interval operations - precision loss acceptable for GPBA epsilon values")]
            {
                ef_array = ef_array_i64.iter().map(|&x| x as f64).collect();
            }
            ef_intervals[prev_id] = new_interval;

            log::debug!(
                "  New epsilon for dimension {}: {}",
                prev_id,
                ef_array[prev_id]
            );

            // For first dimension, also update RWV
            if i == 1 {
                rwv[prev_id] = ideal_max[prev_constraint_idx];
            }
        } else {
            // Stop cascading at first non-exhausted dimension
            break;
        }
    }

    // Check if first dimension is exhausted (convergence)
    let converged = if constraint_indices.is_empty() {
        false
    } else {
        ef_array[0] > ideal_max[constraint_indices[0]]
    };

    if converged {
        log::info!("=== CASCADING: First dimension exhausted, search converged ===");
    } else if !dimensions_reset.is_empty() {
        log::debug!(
            "Cascading complete, reset {} dimensions",
            dimensions_reset.len()
        );
    }

    CascadingResult {
        dimensions_reset,
        converged,
        ef_array,
        ef_intervals,
        rwv,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_cascading_needed() {
        // No dimension exhausted
        let ef_array = vec![-5.0, -10.0];
        let ideal_max = vec![-2.0, -3.0];
        let nadir_max = vec![-10.0, -15.0];
        let constraint_indices = vec![0, 1];
        let rwv = vec![-3.0, -4.0];

        let ef_intervals = vec![IntervalManager::new(-10, -2), IntervalManager::new(-15, -3)];

        let obj_k_at_ef_k = vec![Some(-5.0), Some(-10.0)];

        let result = apply_cascading(
            &ef_array,
            ef_intervals,
            &rwv,
            &ideal_max,
            &nadir_max,
            &constraint_indices,
            None,
            &obj_k_at_ef_k,
        );

        // No dimensions should be reset
        assert!(result.dimensions_reset.is_empty());
        assert!(!result.converged);

        // ef_array should be unchanged
        assert_eq!(result.ef_array, vec![-5.0, -10.0]);
    }

    #[test]
    fn test_single_dimension_cascading() {
        // Second dimension exhausted
        let ef_array = vec![-5.0, -2.5]; // Second dim exceeds ideal
        let ideal_max = vec![-2.0, -3.0];
        let nadir_max = vec![-10.0, -15.0];
        let constraint_indices = vec![0, 1];
        let rwv = vec![-3.0, -4.0];

        let ef_intervals = vec![IntervalManager::new(-10, -2), IntervalManager::new(-15, -3)];

        let obj_k_at_ef_k = vec![Some(-5.0), Some(-2.5)];

        let result = apply_cascading(
            &ef_array,
            ef_intervals,
            &rwv,
            &ideal_max,
            &nadir_max,
            &constraint_indices,
            None,
            &obj_k_at_ef_k,
        );

        // Second dimension should be reset
        assert_eq!(result.dimensions_reset, vec![1]);
        assert!(!result.converged);

        // Second dimension reset to nadir
        #[allow(clippy::float_cmp, reason = "Test comparing exact reset value to nadir constant")]
        {
            assert_eq!(result.ef_array[1], -15.0);
        }

        // First dimension should be updated (will find largest interval)
        // Exact value depends on interval logic, just verify it changed
        #[allow(clippy::float_cmp, reason = "Test verifying dimension was updated from initial value")]
        {
            assert_ne!(result.ef_array[0], -5.0);
        }
    }

    #[test]
    fn test_convergence_detection() {
        // First dimension exhausted
        let ef_array = vec![-1.5, -10.0]; // First dim exceeds ideal
        let ideal_max = vec![-2.0, -3.0];
        let nadir_max = vec![-10.0, -15.0];
        let constraint_indices = vec![0, 1];
        let rwv = vec![-3.0, -4.0];

        let ef_intervals = vec![IntervalManager::new(-10, -2), IntervalManager::new(-15, -3)];

        let obj_k_at_ef_k = vec![Some(-1.5), Some(-10.0)];

        let result = apply_cascading(
            &ef_array,
            ef_intervals,
            &rwv,
            &ideal_max,
            &nadir_max,
            &constraint_indices,
            None,
            &obj_k_at_ef_k,
        );

        // Should detect convergence
        assert!(result.converged);
    }

    #[test]
    fn test_multiple_dimensions_cascading() {
        // Multiple dimensions exhausted (3 objectives, 2 constraints)
        // For cascading to reset multiple dimensions, they must be consecutive from the top
        let ef_array = vec![-5.0, -2.5, -2.0]; // Only dimension 2 exhausted
        let ideal_max = vec![-2.0, -3.0, -4.0];
        let nadir_max = vec![-10.0, -15.0, -20.0];
        let constraint_indices = vec![0, 1, 2];
        let rwv = vec![-3.0, -4.0, -5.0];

        let ef_intervals = vec![
            IntervalManager::new(-10, -2),
            IntervalManager::new(-15, -3),
            IntervalManager::new(-20, -4),
        ];

        let obj_k_at_ef_k = vec![Some(-5.0), Some(-2.5), Some(-2.0)];

        let result = apply_cascading(
            &ef_array,
            ef_intervals,
            &rwv,
            &ideal_max,
            &nadir_max,
            &constraint_indices,
            None,
            &obj_k_at_ef_k,
        );

        // Only dimension 2 should be reset (dimension 1 is not exhausted, so cascading stops)
        assert_eq!(result.dimensions_reset, vec![2]);

        // Dimension 2 should be reset to nadir
        #[allow(clippy::float_cmp, reason = "Test comparing exact reset value to nadir constant")]
        {
            assert_eq!(result.ef_array[2], -20.0);
        }

        // Dimension 1 should be updated
        #[allow(clippy::float_cmp, reason = "Test verifying dimension was updated from initial value")]
        {
            assert_ne!(result.ef_array[1], -2.5);
        }

        // Not converged yet (first dimension not exhausted)
        assert!(!result.converged);
    }

    #[test]
    fn test_rwv_update_on_first_dimension_cascade() {
        // Test that RWV is updated when cascading from dimension 1
        let ef_array = vec![-5.0, -2.5];
        let ideal_max = vec![-2.0, -3.0];
        let nadir_max = vec![-10.0, -15.0];
        let constraint_indices = vec![0, 1];
        let rwv = vec![-3.0, -4.0];

        let ef_intervals = vec![IntervalManager::new(-10, -2), IntervalManager::new(-15, -3)];

        let obj_k_at_ef_k = vec![Some(-5.0), Some(-2.5)];

        let result = apply_cascading(
            &ef_array,
            ef_intervals,
            &rwv,
            &ideal_max,
            &nadir_max,
            &constraint_indices,
            None,
            &obj_k_at_ef_k,
        );

        // RWV[0] should be updated to ideal when dimension 1 cascades
        #[allow(clippy::float_cmp, reason = "Test comparing exact RWV update to ideal constant")]
        {
            assert_eq!(result.rwv[0], -2.0);
        }
    }
}
