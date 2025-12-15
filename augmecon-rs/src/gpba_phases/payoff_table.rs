//! Phase: Payoff Table Computation
//!
//! This module implements the payoff table computation phase of GPBA-A algorithm.
//!
//! The payoff table consists of:
//! - Ideal points: Best value for each objective when optimized individually
//! - Nadir points: Worst value for each objective across all extreme solutions
//!
//! This phase solves 2*n single-objective problems (for n objectives) using the
//! existing BoundsCalculator infrastructure.

use crate::{
    bounds::BoundsCalculator, error::Result, model::MultiObjectiveProblem, options::Options,
    timer::Timer,
};

/// Results from payoff table computation
#[derive(Debug, Clone)]
pub struct PayoffTableResult {
    /// Ideal points in original optimization direction (max objectives get max values, min get min values)
    pub ideal_min: Vec<f64>,
    /// Nadir points in original optimization direction
    pub nadir_min: Vec<f64>,
    /// Ideal points converted to maximization form (all values negated for GPBA-A algorithm)
    pub ideal_max: Vec<f64>,
    /// Nadir points converted to maximization form (all values negated for GPBA-A algorithm)
    pub nadir_max: Vec<f64>,
}

/// Compute ideal and nadir points by solving single-objective optimizations.
///
/// This is Phase 1 (Payoff Table) of GPBA-A. It's the most expensive phase as it
/// solves 2*n MILP problems (for n objectives via the payoff table algorithm).
///
/// # Arguments
///
/// * `problem` - Multi-objective problem instance
/// * `options` - Solver options and configuration
/// * `timer` - Optional timer for timeout handling
///
/// # Returns
///
/// `PayoffTableResult` containing ideal and nadir points. Note: `ideal_min` and `nadir_min`
/// contain values in the problem's ORIGINAL optimization direction (not converted to minimization).
/// The `ideal_max` and `nadir_max` fields contain these values negated for use in GPBA-A's
/// maximization-based algorithm.
///
/// # Errors
///
/// Returns error if bounds calculation fails (e.g., infeasible problem, timeout)
pub fn compute_payoff_table(
    problem: &MultiObjectiveProblem,
    options: &Options,
    timer: Option<&Timer>,
) -> Result<PayoffTableResult> {
    log::info!("=== PHASE 1: Computing payoff table (ideal/nadir bounds) ===");

    // Use existing BoundsCalculator to compute ideal and nadir
    // Note: These values are in the problem's original optimization direction
    let (ideal_original, nadir_original) =
        BoundsCalculator::new(problem, options).calculate_bounds(timer)?;

    log::info!("Best values (original direction): {ideal_original:?}");
    log::info!("Nadir values (original direction): {nadir_original:?}");

    // Convert ALL objectives to maximization for uniform handling in GPBA-A
    // (This matches the behavior in gpba.rs which negates all values)
    log::info!("Converting all objectives to maximization");
    let ideal_max: Vec<f64> = ideal_original.iter().map(|&x| -x).collect();
    let nadir_max: Vec<f64> = nadir_original.iter().map(|&x| -x).collect();

    log::info!("Best values (maximization): {ideal_max:?}");
    log::info!("Nadir values (maximization): {nadir_max:?}");

    Ok(PayoffTableResult {
        ideal_min: ideal_original,
        nadir_min: nadir_original,
        ideal_max,
        nadir_max,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::MultiObjectiveProblem, options::Options, ObjectiveDirection};
    use good_lp::*;
    use std::collections::HashMap;

    fn create_simple_2obj_problem() -> MultiObjectiveProblem {
        // Simple 2-objective problem:
        // max z1 = x + y
        // max z2 = x
        // s.t. x + y <= 2
        //      x, y >= 0

        variables!(
            problem:
               0 <= x;
               0 <= y;
        );

        let var_map = HashMap::from([("x".to_string(), x), ("y".to_string(), y)]);

        let constraints = vec![constraint!(x + y <= 2.0)];

        let objectives = vec![
            (x + y, ObjectiveDirection::Maximize),    // z1 = x + y
            (x.into(), ObjectiveDirection::Maximize), // z2 = x
        ];

        MultiObjectiveProblem {
            variables: problem,
            constraints,
            objectives,
            var_map,
            variable_types: HashMap::new(),
        }
    }

    #[test]
    fn test_payoff_table_structure() {
        // Test that payoff table computation returns correct structure
        let problem = create_simple_2obj_problem();
        let options = Options::default();

        let result = compute_payoff_table(&problem, &options, None);

        // Should succeed for this simple feasible problem
        assert!(result.is_ok(), "Payoff table computation should succeed");

        let payoff = result.unwrap();

        // Check that all vectors have correct length
        assert_eq!(payoff.ideal_min.len(), 2);
        assert_eq!(payoff.nadir_min.len(), 2);
        assert_eq!(payoff.ideal_max.len(), 2);
        assert_eq!(payoff.nadir_max.len(), 2);

        // Check maximization conversion: max values should be negative of min values
        for i in 0..2 {
            assert!(
                (payoff.ideal_max[i] + payoff.ideal_min[i]).abs() < 1e-6,
                "ideal_max should be -ideal_min"
            );
            assert!(
                (payoff.nadir_max[i] + payoff.nadir_min[i]).abs() < 1e-6,
                "nadir_max should be -nadir_min"
            );
        }

        // For minimization problems: ideal <= nadir
        for i in 0..2 {
            assert!(
                payoff.ideal_min[i] <= payoff.nadir_min[i] + 1e-6,
                "Ideal should be better (<=) than nadir for minimization"
            );
        }

        // For maximization: ideal >= nadir (since values are negated)
        for i in 0..2 {
            assert!(
                payoff.ideal_max[i] >= payoff.nadir_max[i] - 1e-6,
                "Ideal should be better (>=) than nadir for maximization"
            );
        }
    }

    #[test]
    fn test_payoff_table_values_simple_problem() {
        const TOL: f64 = 1e-3;
        
        // Test with known optimal values
        let problem = create_simple_2obj_problem();
        let options = Options::default();

        let result = compute_payoff_table(&problem, &options, None);
        assert!(result.is_ok());

        let payoff = result.unwrap();

        // For the problem:
        // max z1 = x + y, max z2 = x
        // s.t. x + y <= 2, x,y >= 0
        //
        // Optimal for z1: x=y=1 → z1=2, z2=1
        // Optimal for z2: x=2, y=0 → z1=2, z2=2
        //
        // BoundsCalculator returns values in original optimization direction
        // ideal_min contains the best values (despite the name)
        // nadir_min contains the worst values across payoff table

        // Allow some tolerance for solver precision
        // Ideal values should be the best possible for each objective
        assert!(
            (payoff.ideal_min[0] - 2.0).abs() < TOL,
            "z1 ideal should be ≈2.0, got {}",
            payoff.ideal_min[0]
        );
        assert!(
            (payoff.ideal_min[1] - 2.0).abs() < TOL,
            "z2 ideal should be ≈2.0, got {}",
            payoff.ideal_min[1]
        );

        // Nadir values are computed from payoff table
        // The actual nadir computation depends on bounds.rs algorithm
        // Just verify they are reasonable (between 0 and ideal)
        assert!(
            payoff.nadir_min[0] >= -TOL && payoff.nadir_min[0] <= payoff.ideal_min[0] + TOL,
            "z1 nadir should be between 0 and ideal"
        );
        assert!(
            payoff.nadir_min[1] >= -TOL && payoff.nadir_min[1] <= payoff.ideal_min[1] + TOL,
            "z2 nadir should be between 0 and ideal"
        );

        // Verify conversion: ideal_max = -ideal_min, nadir_max = -nadir_min
        for i in 0..2 {
            assert!(
                (payoff.ideal_max[i] + payoff.ideal_min[i]).abs() < TOL,
                "ideal_max[{i}] should be -ideal_min[{i}]"
            );
            assert!(
                (payoff.nadir_max[i] + payoff.nadir_min[i]).abs() < TOL,
                "nadir_max[{i}] should be -nadir_min[{i}]"
            );
        }
    }
}
