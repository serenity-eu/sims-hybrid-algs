//! Grid generation utilities for multi-objective optimization
//!
//! This module provides functionality for generating systematic grid points
//! and converting them to epsilon values for use in constraint-based algorithms.

use std::collections::HashMap;

/// Generator for systematic grid points used in optimization algorithms
pub struct GridGenerator;

impl GridGenerator {
    /// Generate systematic grid points for traditional AUGMECON
    ///
    /// Creates uniform grid with specified number of points per objective.
    /// Returns a vector of grid points, where each point represents coordinates
    /// for the non-primary objectives.
    ///
    /// Note: Order is reversed to match Python's `PyAugmecon` implementation
    /// which applies `[i[::-1] for i in cartesian_product]`
    #[must_use]
    pub fn generate_uniform_grid(
        num_objectives: usize,
        points_per_objective: usize,
    ) -> Vec<Vec<usize>> {
        use itertools::Itertools;

        if num_objectives < 2 {
            return vec![];
        }

        // Generate cartesian product for (num_objectives - 1) dimensions
        // The primary objective is not constrained
        (0..num_objectives - 1)
            .map(|_| (0..points_per_objective).collect::<Vec<usize>>())
            .multi_cartesian_product()
            .map(|mut point| {
                // Reverse each grid point to match Python's implementation:
                // Python does `[i[::-1] for i in cartesian_product]`
                point.reverse();
                point
            })
            .collect()
    }

    /// Convert grid point indices to actual epsilon values using bounds
    ///
    /// Takes grid point coordinates and linearly interpolates between nadir and ideal
    /// points to generate epsilon constraint values.
    #[must_use]
    pub fn grid_to_epsilon_values(
        grid_point: &[usize],
        ideal: &[f64],
        nadir: &[f64],
        grid_size: usize,
    ) -> HashMap<usize, f64> {
        let mut epsilon_values = HashMap::new();

        for (dim_index, &grid_position) in grid_point.iter().enumerate() {
            // Map grid dimensions to actual objective indices
            // Skip the primary objective (typically 0), so dim_index maps to dim_index + 1
            let objective_index = dim_index + 1;

            if objective_index < ideal.len() {
                // Linear interpolation between nadir and ideal
                let range = ideal[objective_index] - nadir[objective_index];
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "Converting grid_size usize to f64 for epsilon-constraint interpolation - precision loss acceptable for grid calculation"
                )]
                let step = range / ((grid_size - 1) as f64);
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "Converting grid_position usize to f64 for epsilon value calculation - precision loss acceptable for constraint generation"
                )]
                let epsilon = (grid_position as f64).mul_add(step, nadir[objective_index]);

                epsilon_values.insert(objective_index, epsilon);
            }
        }

        epsilon_values
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn test_grid_generation() {
        let grid = GridGenerator::generate_uniform_grid(3, 5);
        // For 3 objectives, should generate 5^2 = 25 grid points
        assert_eq!(grid.len(), 25);

        // Each grid point should have 2 dimensions (3 objectives - 1)
        for point in &grid {
            assert_eq!(point.len(), 2);
            // All values should be in range [0, 4]
            for &val in point {
                assert!(val < 5);
            }
        }
    }

    #[test]
    fn test_epsilon_value_conversion() {
        let grid_point = vec![0, 2]; // Example grid point
        let ideal = vec![100.0, 200.0, 300.0];
        let nadir = vec![0.0, 50.0, 100.0];
        let grid_size = 5;

        let epsilon_values =
            GridGenerator::grid_to_epsilon_values(&grid_point, &ideal, &nadir, grid_size);

        // Should have epsilon values for objectives 1 and 2 (not 0, the primary)
        assert_eq!(epsilon_values.len(), 2);

        // Check the calculated values
        // For objective 1: nadir[1] + (0 / 4) * (ideal[1] - nadir[1]) = 50.0 + 0 * 150.0 = 50.0
        assert_eq!(epsilon_values[&1], 50.0);

        // For objective 2: nadir[2] + (2 / 4) * (ideal[2] - nadir[2]) = 100.0 + 0.5 * 200.0 = 200.0
        assert_eq!(epsilon_values[&2], 200.0);
    }

    #[test]
    fn test_detailed_grid_generation() {
        let grid = GridGenerator::generate_uniform_grid(3, 10);
        println!("Total grid points: {}", grid.len());
        println!("First 20 grid points:");
        for (i, point) in grid.iter().take(20).enumerate() {
            println!("{i}: {point:?}");
        }
    }
}
