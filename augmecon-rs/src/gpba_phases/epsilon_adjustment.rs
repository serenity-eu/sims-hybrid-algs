//! Phase: Epsilon Adjustment
//!
//! This module implements the epsilon adjustment logic for GPBA-A algorithm.
//! This is the core of GPBA-A that adaptively explores the largest gaps in the Pareto front.
//!
//! After solving an epsilon-constraint problem, this phase updates the epsilon parameters
//! to focus on the largest unexplored regions of the objective space.
use super::interval_manager::IntervalManager;

/// Adjust `ef_array` parameter based on solution found, using interval management.
///
/// This is the core of GPBA-A that adaptively explores the largest gaps in the Pareto front.
///
/// # Arguments
///
/// * `id_constraint_objective` - Index of the objective being constrained (0 to n_objectives-2)
/// * `ef_array` - Current epsilon values for each constrained objective (modified in-place)
/// * `sol_obj_k` - Objective value found for the constrained objective, or None if infeasible
/// * `ef_interval` - `IntervalManager` tracking unexplored regions for this objective
/// * `constraint_indices` - Mapping from constraint index to actual objective index
/// * `best_objective_values` - Ideal point (best value for each objective)
/// * `nadir_objectives_values` - Nadir point (worst value for each objective)
/// * `_gamma` - Unused parameter (kept for API compatibility)
///
/// # Returns
///
/// Updated `IntervalManager` (or newly created one if exhausted)
#[allow(clippy::too_many_arguments)]
pub fn adjust_parameter_ef_array(
    id_constraint_objective: usize,
    ef_array: &mut [i64],
    sol_obj_k: Option<i64>,
    ef_interval: &mut IntervalManager,
    constraint_indices: &[usize],
    best_objective_values: &[i64],
    nadir_objectives_values: &[i64],
    _gamma: i64,
) -> IntervalManager {
    let actual_obj_index = constraint_indices[id_constraint_objective];

    log::debug!(
        "EPS ADJUST: ENTRY - ef_array[{}]={}, sol_obj_k={:?}, interval.max={}, best[{}]={}, nadir[{}]={}",
        id_constraint_objective,
        ef_array[id_constraint_objective],
        sol_obj_k,
        ef_interval.max_value,
        actual_obj_index,
        best_objective_values[actual_obj_index],
        actual_obj_index,
        nadir_objectives_values[actual_obj_index]
    );

    let start_removal = ef_array[id_constraint_objective];
    let new_max_interval = start_removal - 1;

    let end_removal = if let Some(sol_val) = sol_obj_k {
        // Feasible - remove up to solution value
        sol_val.min(ef_interval.max_value)
    } else {
        // Infeasible - remove entire range up to max
        ef_interval.max_value
    };

    log::debug!(
        "EPS ADJUST: REMOVAL - start={start_removal}, end={end_removal}, new_max={new_max_interval}"
    );

    // Remove explored region from interval
    if start_removal < end_removal {
        ef_interval.remove_interval(start_removal, end_removal);
    } else {
        ef_interval.remove_one_point(start_removal);
        if start_removal > end_removal {
            ef_interval.remove_one_point(end_removal);
        }
    }

    // Update max_value if needed
    if end_removal >= ef_interval.max_value {
        ef_interval.max_value = new_max_interval;
    }

    // Find next point to explore (center of largest remaining interval)
    let max_interval = ef_interval.find_largest_interval();

    log::debug!(
        "EPS ADJUST: FIND LARGEST - max_interval={:?}, intervals={:?}",
        max_interval,
        ef_interval.intervals
    );

    if let Some((start, end)) = max_interval {
        if ef_array[id_constraint_objective] == nadir_objectives_values[actual_obj_index] {
            // First iteration in this dimension - jump to best
            ef_array[id_constraint_objective] = best_objective_values[actual_obj_index];
            log::debug!(
                "EPS ADJUST: At nadir, jump to best: ef_array[{}]={}",
                id_constraint_objective,
                ef_array[id_constraint_objective]
            );
        } else {
            // Explore center of largest gap
            ef_array[id_constraint_objective] = i64::midpoint(start, end);
            log::debug!(
                "EPS ADJUST: Explore center of gap: ef_array[{}]={}",
                id_constraint_objective,
                ef_array[id_constraint_objective]
            );
        }

        // Return current interval manager
        ef_interval.clone()
    } else {
        // Interval exhausted - reinitialize for next exploration
        log::debug!("EPS ADJUST: Interval exhausted! Reinitializing...");

        ef_array[id_constraint_objective] = best_objective_values[actual_obj_index] + 1;
        log::debug!(
            "EPS ADJUST: Set ef_array[{}]={}",
            id_constraint_objective,
            ef_array[id_constraint_objective]
        );

        // Recreate interval - CRITICAL: Ensure min < max numerically
        let min_interval =
            nadir_objectives_values[actual_obj_index].min(best_objective_values[actual_obj_index]);
        let max_interval_val =
            nadir_objectives_values[actual_obj_index].max(best_objective_values[actual_obj_index]);

        let new_interval = IntervalManager::new(min_interval, max_interval_val);
        log::debug!(
            "EPS ADJUST: Recreated interval: min={min_interval}, max={max_interval_val}"
        );

        new_interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjust_parameter_feasible_solution() {
        // Test adjusting parameter when a feasible solution is found
        let mut ef_array = vec![-656_595, -1000]; // Maximization form
        let constraint_indices = vec![1];
        let best = vec![-100, -500];
        let nadir = vec![-1000, -2000];

        let mut ef_interval = IntervalManager::new(-2000, -500);

        // Simulate finding a solution at -750
        let new_interval = adjust_parameter_ef_array(
            0,
            &mut ef_array,
            Some(-750),
            &mut ef_interval,
            &constraint_indices,
            &best,
            &nadir,
            1,
        );

        // ef_array should be updated to explore center of remaining gap
        // Interval manager should have removed explored region
        assert!(ef_array[0] != -656_595, "ef_array should be updated");
        assert!(
            !new_interval.intervals.is_empty(),
            "Interval should still have regions"
        );
    }

    #[test]
    fn test_adjust_parameter_infeasible_solution() {
        // Test adjusting parameter when solution is infeasible
        let mut ef_array = vec![-100];
        let constraint_indices = vec![0];
        let best = vec![-100];
        let nadir = vec![-1000];

        let mut ef_interval = IntervalManager::new(-1000, -100);

        // Simulate infeasible solution (None)
        let _new_interval = adjust_parameter_ef_array(
            0,
            &mut ef_array,
            None,
            &mut ef_interval,
            &constraint_indices,
            &best,
            &nadir,
            1,
        );

        // Should remove more of the interval since infeasible
        assert!(ef_array[0] != -100, "ef_array should be updated");
    }

    #[test]
    fn test_adjust_parameter_interval_exhaustion() {
        // Test that interval gets recreated when exhausted
        let mut ef_array = vec![-500];
        let constraint_indices = vec![0];
        let best = vec![-100];
        let nadir = vec![-1000];

        // Create small interval that will be exhausted
        let mut ef_interval = IntervalManager::new(-550, -500);

        // Remove the interval completely
        ef_interval.remove_interval(-550, -500);

        // Should recreate interval when exhausted
        let new_interval = adjust_parameter_ef_array(
            0,
            &mut ef_array,
            Some(-400),
            &mut ef_interval,
            &constraint_indices,
            &best,
            &nadir,
            1,
        );

        // New interval should be created
        assert!(
            !new_interval.intervals.is_empty(),
            "New interval should be created"
        );
        assert_eq!(
            ef_array[0],
            best[0] + 1,
            "ef_array should be set to best + 1"
        );
    }
}
