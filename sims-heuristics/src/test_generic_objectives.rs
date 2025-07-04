// Test file for verifying the generic objectives refactoring
#[cfg(test)]
mod tests {
    use crate::objectives::{CloudyAreaObjective, TotalCostObjective};
    use crate::problem::{Problem, SIMSProblemInstanceRaw};
    use crate::solution::VecEncodedSolution;

    #[test]
    fn test_generic_objectives_2d() {
        // Create a simple test problem
        let raw_data = SIMSProblemInstanceRaw {
            name: "test_problem".to_string(),
            num_images: 3,
            universe_size: 4,
            images: vec![
                vec![1, 2], // Image 0 covers elements 0, 1 (1-based, will be converted to 0-based)
                vec![2, 3], // Image 1 covers elements 1, 2
                vec![3, 4], // Image 2 covers elements 2, 3
            ],
            costs: vec![10, 20, 30],
            clouds: vec![
                vec![],     // Image 0 has no clouds (all clear)
                vec![2],    // Image 1 has clouds on element 1 (0-based)
                vec![3, 4], // Image 2 has clouds on elements 2, 3 (0-based)
            ],
            areas: vec![100, 200, 150, 250], // Areas for elements 0, 1, 2, 3
            max_cloud_area: 700,
            resolution: vec![10, 20, 30],
            incidence_angle: vec![45, 60, 30],
        };

        // Test with default 2D objectives (backward compatibility)
        let problem_legacy: Problem<2> = Problem::from_raw(raw_data.clone());
        assert_eq!(problem_legacy.num_objectives(), 2);
        assert_eq!(
            problem_legacy.objective_names(),
            vec!["Total Cost", "Cloudy Area"]
        );

        // Test with explicit objective definitions using builder pattern
        let problem_builder_result: Result<Problem<2>, String> = Problem::builder(raw_data)
            .with_objective_definitions(vec![
                Box::new(TotalCostObjective { index: 0 }),
                Box::new(CloudyAreaObjective { index: 1 }),
            ])
            .build();

        assert!(problem_builder_result.is_ok());
        let problem_generic = problem_builder_result.unwrap();

        assert_eq!(problem_generic.num_objectives(), 2);
        assert!(problem_generic.has_objective_definitions());
        assert_eq!(
            problem_generic.get_objective_definition(0).unwrap().id(),
            "total_cost"
        );
        assert_eq!(
            problem_generic.get_objective_definition(1).unwrap().id(),
            "cloudy_area"
        );

        // Test that both legacy and generic systems produce the same results
        let solution_legacy = VecEncodedSolution::from_selected_images(&[0, 1], &problem_legacy);
        let solution_generic = VecEncodedSolution::from_selected_images(&[0, 1], &problem_generic);

        assert_eq!(solution_legacy.objectives, solution_generic.objectives);
        println!("Legacy objectives: {:?}", solution_legacy.objectives);
        println!("Generic objectives: {:?}", solution_generic.objectives);
    }

    #[test]
    fn test_generic_objectives_validation() {
        let raw_data = SIMSProblemInstanceRaw {
            name: "test_validation".to_string(),
            num_images: 2,
            universe_size: 2,
            images: vec![vec![1], vec![2]],
            costs: vec![10, 20],
            clouds: vec![vec![], vec![]],
            areas: vec![100, 200],
            max_cloud_area: 300,
            resolution: vec![10, 20],
            incidence_angle: vec![45, 60],
        };

        // Test validation - wrong number of objectives
        let result: Result<Problem<3>, String> = Problem::builder(raw_data.clone())
            .with_objective_definitions(vec![
                Box::new(TotalCostObjective { index: 0 }),
                Box::new(CloudyAreaObjective { index: 1 }),
                // Missing third objective
            ])
            .build();

        assert!(result.is_err());
        let error_msg = result.err().unwrap();
        assert!(error_msg.contains("Expected 3 objectives, got 2"));

        // Test validation - wrong objective index
        let result: Result<Problem<2>, String> = Problem::builder(raw_data)
            .with_objective_definitions(vec![
                Box::new(TotalCostObjective { index: 0 }),
                Box::new(CloudyAreaObjective { index: 2 }), // Should be index 1
            ])
            .build();

        assert!(result.is_err());
        let error_msg = result.err().unwrap();
        assert!(error_msg.contains("incorrect index"));
    }

    #[test]
    fn test_generic_weight_generation() {
        use crate::objectives::generate_weights;
        use crate::objectives::weighted_sum_f32;

        // Test 2D weights
        let weights_2d: [f32; 2] = generate_weights::<2>();
        assert_eq!(weights_2d.len(), 2);
        let sum_2d: f32 = weights_2d.iter().sum();
        assert!(
            (sum_2d - 1.0).abs() < 1e-6,
            "Weights should sum to 1.0, got {sum_2d}"
        );

        // Test 3D weights
        let weights_3d: [f32; 3] = generate_weights::<3>();
        assert_eq!(weights_3d.len(), 3);
        let sum_3d: f32 = weights_3d.iter().sum();
        assert!(
            (sum_3d - 1.0).abs() < 1e-6,
            "Weights should sum to 1.0, got {sum_3d}"
        );

        // Test weighted sum function
        let values = [2.0, 4.0, 6.0];
        let weights = [0.3, 0.4, 0.3];
        #[expect(clippy::suboptimal_flops, reason = "Keep multiplication for clarity")]
        let expected = 2.0 * 0.3 + 4.0 * 0.4 + 6.0 * 0.3; // = 0.6 + 1.6 + 1.8 = 4.0
        let result = weighted_sum_f32(&values, &weights);
        assert!(
            (result - expected).abs() < 1e-6,
            "Expected {expected}, got {result}"
        );
    }

    #[test]
    fn test_explored_solutions_data_generic() {
        use crate::explored_solutions_data::ExploredSolutionsData;

        // Test 2D
        let explored_2d = ExploredSolutionsData::<2>::new([1000, 2000]);
        assert_eq!(explored_2d.max_objective(0), 1000);
        assert_eq!(explored_2d.max_objective(1), 2000);
        assert_eq!(explored_2d.max_objectives(), &[1000, 2000]);

        // Test 3D
        let explored_3d = ExploredSolutionsData::<3>::new([100, 200, 300]);
        assert_eq!(explored_3d.max_objective(0), 100);
        assert_eq!(explored_3d.max_objective(1), 200);
        assert_eq!(explored_3d.max_objective(2), 300);
        assert_eq!(explored_3d.max_objectives(), &[100, 200, 300]);

        // Test higher dimensions
        let explored_5d = ExploredSolutionsData::<5>::new([10, 20, 30, 40, 50]);
        assert_eq!(explored_5d.max_objectives().len(), 5);
        assert_eq!(explored_5d.max_objective(4), 50);
    }
}
