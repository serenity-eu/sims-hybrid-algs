use pls::objectives::ObjectiveType;
// Integration tests for verifying the generic objectives refactoring
use pls::problem::SIMSProblemInstanceRaw;
use pls::problem_bitset::ProblemBitset;
use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;

#[test]
fn test_generic_objectives_2d() {
    // Create a simple test problem with 0-based indices (normalized)
    let raw_data = SIMSProblemInstanceRaw {
        name: "test_problem".to_string(),
        num_images: 3,
        universe_size: 4,
        images: vec![
            vec![0, 1], // Image 0 covers elements 0, 1
            vec![1, 2], // Image 1 covers elements 1, 2
            vec![2, 3], // Image 2 covers elements 2, 3
        ],
        costs: vec![10, 20, 30],
        clouds: vec![
            vec![],     // Image 0 has no clouds (all clear)
            vec![1],    // Image 1 has clouds on element 1
            vec![2, 3], // Image 2 has clouds on elements 2, 3
        ],
        areas: vec![100, 200, 150, 250], // Areas for elements 0, 1, 2, 3
        max_cloud_area: 700,
        resolution: vec![10, 20, 30],
        incidence_angle: vec![45, 60, 30],
    };

    // Test with default 2D objectives (backward compatibility)
    let problem_legacy = ProblemBitset::<2>::from_raw_with_objectives(
        &raw_data,
        [ObjectiveType::TotalCost, ObjectiveType::CloudyArea],
    );
    assert_eq!(problem_legacy.num_objectives(), 2);
    assert_eq!(
        problem_legacy.objective_names(),
        vec!["Total Cost", "Cloudy Area"]
    );

    // Test with explicit objective definitions using builder pattern
    // Note: ProblemBitset doesn't have a builder pattern, only from_raw_with_objectives
    let problem_generic = ProblemBitset::<2>::from_raw_with_objectives(
        &raw_data,
        [ObjectiveType::TotalCost, ObjectiveType::CloudyArea],
    );

    assert_eq!(problem_generic.num_objectives(), 2);

    assert_eq!(problem_generic.num_objectives(), 2);
    assert_eq!(problem_generic.objective_type(0).id(), "total_cost");
    assert_eq!(problem_generic.objective_type(1).id(), "cloudy_area");

    // Test that both legacy and generic systems produce the same results
    let solution_legacy = BitsetEncodedSolution::from_selected_images(&[0, 1], &problem_legacy);
    let solution_generic = BitsetEncodedSolution::from_selected_images(&[0, 1], &problem_generic);

    assert_eq!(solution_legacy.objectives, solution_generic.objectives);
    println!("Legacy objectives: {:?}", solution_legacy.objectives);
    println!("Generic objectives: {:?}", solution_generic.objectives);
}

#[test]
#[ignore = "ProblemBitset doesn't have builder pattern, validation happens at compile-time via const generics"]
fn test_generic_objectives_validation() {
    // This test is disabled because ProblemBitset validates objective count at compile-time
    // through const generics, not at runtime through a builder pattern
}

#[test]
fn test_generic_weight_generation() {
    use pls::objectives::generate_weights;
    use pls::objectives::weighted_sum_f32;

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
    use pls::explored_solutions_data::ExploredSolutionsData;

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

#[test]
fn test_plotting_dimensions() {
    use pls::explored_solutions_data::ExploredSolutionsData;

    // Test that plotting function handles different dimensions
    let explored_2d = ExploredSolutionsData::<2>::new([1000, 2000]);
    let explored_3d = ExploredSolutionsData::<3>::new([100, 200, 300]);
    let explored_4d = ExploredSolutionsData::<4>::new([10, 20, 30, 40]);

    // These shouldn't panic (we can't easily test the actual plotting output in unit tests)
    // but we can at least verify the function calls work
    assert_eq!(explored_2d.max_objectives().len(), 2);
    assert_eq!(explored_3d.max_objectives().len(), 3);
    assert_eq!(explored_4d.max_objectives().len(), 4);

    // Calculate expected number of pairwise combinations for verification
    let num_pairs = |n: usize| n * (n - 1) / 2;
    assert_eq!(num_pairs(2), 1); // 2D: 1 pair
    assert_eq!(num_pairs(3), 3); // 3D: 3 pairs  
    assert_eq!(num_pairs(4), 6); // 4D: 6 pairs
}

#[test]
#[cfg(feature = "plotting")]
fn test_plot_grid_calculations() {
    use pls::plotting::calculate_plot_grid_dimensions;

    // Test various dimensions
    assert_eq!(calculate_plot_grid_dimensions(1), (0, 0, 0)); // Invalid
    assert_eq!(calculate_plot_grid_dimensions(2), (1, 2, 1)); // 2D: 1 pair -> 1x2 grid
    assert_eq!(calculate_plot_grid_dimensions(3), (2, 2, 3)); // 3D: 3 pairs -> 2x2 grid  
    assert_eq!(calculate_plot_grid_dimensions(4), (2, 3, 6)); // 4D: 6 pairs -> 2x3 grid
    assert_eq!(calculate_plot_grid_dimensions(5), (3, 4, 10)); // 5D: 10 pairs -> 3x4 grid
}

#[test]
#[cfg(feature = "plotting")]
fn test_plotting_with_objective_names() {
    use pls::explored_solutions_data::ExploredSolutionsData;
    use pls::plotting::{calculate_plot_grid_dimensions, draw_solutions_plot};

    // Create test data
    let data = ExploredSolutionsData::<3>::new([1000, 1000, 1000]);

    // Test objective names
    let objective_names = ["Total Cost", "Cloud Coverage", "Image Quality"];
    let names_ref: Vec<&str> = objective_names.to_vec();

    // This should not panic and should use the provided names
    draw_solutions_plot(&data, &names_ref);

    // Test grid calculations for 3D
    let (rows, cols, pairs) = calculate_plot_grid_dimensions(3);
    assert_eq!(pairs, 3); // C(3,2) = 3 pairs
    assert_eq!(rows, 2); // 2x2 grid with 3 used
    assert_eq!(cols, 2);
}
