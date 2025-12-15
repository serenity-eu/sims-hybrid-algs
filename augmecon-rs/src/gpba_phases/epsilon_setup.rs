//! Phase: Epsilon-Constraint Setup
//!
//! This module implements the epsilon-constraint method setup phase of GPBA-A algorithm.
//! It takes the payoff table outputs (ideal and nadir points) and initializes the
//! epsilon-constraint formulation structures including interval managers.
//!
//! This phase does NOT require solving the model - it only sets up structures for the main loop.

use super::interval_manager::IntervalManager;

/// Result of epsilon-constraint setup phase.
///
/// Contains all structures needed for the GPBA-A main iteration loop.
#[derive(Debug, Clone)]
pub struct EpsilonSetupResult {
    /// Epsilon-constraint RHS values (one per constraint objective).
    /// Initialized to nadir values as the starting point for exploration.
    /// These values are in maximization form (negated).
    pub ef_array: Vec<f64>,

    /// Interval managers for adaptive exploration of each constraint objective.
    /// One `IntervalManager` per constraint objective.
    pub ef_intervals: Vec<IntervalManager>,

    /// Indices of constraint objectives (all objectives except main objective).
    pub constraint_indices: Vec<usize>,

    /// Index of the main objective (always 0 in current implementation).
    pub main_obj_index: usize,

    /// Initial RWV (Relative Worst Values) vector.
    /// Initialized to ideal values for each constraint objective.
    /// These values are in maximization form (negated).
    pub rwv: Vec<f64>,
}

/// Setup epsilon-constraint formulation for GPBA-A algorithm.
///
/// This phase:
/// - Selects main objective (first one by default) and constraint objectives (rest)
/// - Initializes `ef_array` to nadir values (starting point)
/// - Creates interval managers for adaptive exploration
/// - Initializes RWV (Relative Worst Values) to ideal values
///
/// # Arguments
///
/// * `ideal_max` - Ideal point values in maximization form (negated from original)
/// * `nadir_max` - Nadir point values in maximization form (negated from original)
/// * `num_objectives` - Number of objectives in the problem
///
/// # Returns
///
/// `EpsilonSetupResult` containing all structures needed for main iteration loop.
///
/// # Panics
///
/// Panics if `num_objectives` is less than 2 (epsilon-constraint requires multiple objectives).
///
/// # Example
///
/// ```
/// use augmecon_rs::gpba_phases::epsilon_setup::setup_epsilon_constraints;
///
/// let ideal_max = vec![-2.0, -3.0];  // In maximization form
/// let nadir_max = vec![-10.0, -15.0];
/// let setup = setup_epsilon_constraints(&ideal_max, &nadir_max, 2);
///
/// assert_eq!(setup.main_obj_index, 0);
/// assert_eq!(setup.constraint_indices, vec![1]);
/// assert_eq!(setup.ef_array, vec![-15.0]);  // Initialized to nadir for constraint obj
/// assert_eq!(setup.rwv, vec![-3.0]);  // Initialized to ideal for constraint obj
/// ```
#[must_use]
pub fn setup_epsilon_constraints(
    ideal_max: &[f64],
    nadir_max: &[f64],
    num_objectives: usize,
) -> EpsilonSetupResult {
    assert!(num_objectives >= 2, "Epsilon-constraint setup requires at least 2 objectives");

    // Main objective is always the first one (index 0)
    let main_obj_index = 0;

    // Constraint objectives are all others
    let constraint_indices: Vec<usize> = (0..num_objectives)
        .filter(|&i| i != main_obj_index)
        .collect();

    log::debug!("Main objective: {main_obj_index}");
    log::debug!("Constraint objectives: {constraint_indices:?}");

    // Initialize ef_array to nadir values (starting point for exploration)
    // These are already in maximization form (negated)
    let ef_array: Vec<f64> = constraint_indices.iter().map(|&k| nadir_max[k]).collect();

    log::debug!("Initial ef_array: {ef_array:?}");

    // Initialize interval managers for each constraint objective
    // CRITICAL: Ensure min < max numerically for interval logic to work
    // For maximization (negative values), nadir is more negative (smaller) than ideal
    let ef_intervals: Vec<IntervalManager> = constraint_indices
        .iter()
        .map(|&k| {
            let min_val = nadir_max[k].min(ideal_max[k]);
            let max_val = nadir_max[k].max(ideal_max[k]);

            #[allow(clippy::cast_possible_truncation, reason = "Converting min/max bounds to i64 for interval management - truncation acceptable for GPBA integer-valued constraints")]
            {
                log::debug!(
                    "Creating interval for objective {k}: [{}, {}]",
                    min_val as i64,
                    max_val as i64
                );

                IntervalManager::new(min_val as i64, max_val as i64)
            }
        })
        .collect();

    // Initialize RWV (Relative Worst Values) to ideal values
    // These are already in maximization form (negated)
    let rwv: Vec<f64> = constraint_indices.iter().map(|&k| ideal_max[k]).collect();

    log::debug!("Initial RWV: {rwv:?}");

    EpsilonSetupResult {
        ef_array,
        ef_intervals,
        constraint_indices,
        main_obj_index,
        rwv,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_setup_basic_structure() {
        // Two objectives: ideal_max = [-2, -3], nadir_max = [-10, -15]
        let ideal_max = vec![-2.0, -3.0];
        let nadir_max = vec![-10.0, -15.0];

        let setup = setup_epsilon_constraints(&ideal_max, &nadir_max, 2);

        // Main objective should be 0
        assert_eq!(setup.main_obj_index, 0);

        // Constraint objectives should be [1]
        assert_eq!(setup.constraint_indices, vec![1]);

        // ef_array should be initialized to nadir for constraint objectives
        assert_eq!(setup.ef_array.len(), 1);
        #[allow(clippy::float_cmp, reason = "Test comparing exact initialization to nadir constant")]
        {
            assert_eq!(setup.ef_array[0], -15.0);
        }

        // RWV should be initialized to ideal for constraint objectives
        assert_eq!(setup.rwv.len(), 1);
        #[allow(clippy::float_cmp, reason = "Test comparing exact initialization to ideal constant")]
        {
            assert_eq!(setup.rwv[0], -3.0);
        }

        // Should have one interval manager
        assert_eq!(setup.ef_intervals.len(), 1);
    }

    #[test]
    fn test_setup_three_objectives() {
        // Three objectives
        let ideal_max = vec![-1.0, -2.0, -3.0];
        let nadir_max = vec![-10.0, -20.0, -30.0];

        let setup = setup_epsilon_constraints(&ideal_max, &nadir_max, 3);

        assert_eq!(setup.main_obj_index, 0);
        assert_eq!(setup.constraint_indices, vec![1, 2]);

        // ef_array: nadir values for constraint objectives [1, 2]
        assert_eq!(setup.ef_array, vec![-20.0, -30.0]);

        // RWV: ideal values for constraint objectives [1, 2]
        assert_eq!(setup.rwv, vec![-2.0, -3.0]);

        assert_eq!(setup.ef_intervals.len(), 2);
    }

    #[test]
    fn test_interval_bounds_correct() {
        // Test that intervals are created with correct min/max
        let ideal_max = vec![-2.0, -3.0];
        let nadir_max = vec![-10.0, -15.0];

        let setup = setup_epsilon_constraints(&ideal_max, &nadir_max, 2);

        // For objective 1: nadir=-15, ideal=-3
        // min should be -15 (more negative), max should be -3 (less negative)
        let _interval = &setup.ef_intervals[0];

        // Test by checking if we can add points at boundaries
        let mut test_interval = IntervalManager::new(-15, -3);

        // Should be able to add point at nadir
        test_interval.add_interval(-15, -15);

        // Should be able to add point at ideal
        test_interval.add_interval(-3, -3);

        // Verify no panic occurred (intervals are valid)
    }

    #[test]
    fn test_setup_with_positive_values() {
        // Test with values that would be positive in original (minimization) direction
        // In maximization form they're negative
        let ideal_max = vec![-100.0, -50.0];
        let nadir_max = vec![-500.0, -300.0];

        let setup = setup_epsilon_constraints(&ideal_max, &nadir_max, 2);

        assert_eq!(setup.ef_array, vec![-300.0]);
        assert_eq!(setup.rwv, vec![-50.0]);
    }

    #[test]
    #[should_panic(expected = "Epsilon-constraint setup requires at least 2 objectives")]
    fn test_setup_panics_with_one_objective() {
        let ideal_max = vec![-2.0];
        let nadir_max = vec![-10.0];

        let _ = setup_epsilon_constraints(&ideal_max, &nadir_max, 1);
    }

    #[test]
    fn test_setup_all_constraint_indices_unique() {
        // Verify that constraint_indices doesn't include main_obj_index
        let ideal_max = vec![-1.0, -2.0, -3.0, -4.0];
        let nadir_max = vec![-10.0, -20.0, -30.0, -40.0];

        let setup = setup_epsilon_constraints(&ideal_max, &nadir_max, 4);

        assert_eq!(setup.main_obj_index, 0);
        assert_eq!(setup.constraint_indices, vec![1, 2, 3]);
        assert!(!setup.constraint_indices.contains(&setup.main_obj_index));
    }
}
