//! Example: Satellite Image Selection Problem (SIMS)

use augmecon::{
    sims_problem::{create_sample_sims_problem, create_sims_problem, SimsInstance},
    Augmecon, HasObjectives, MultiObjectiveProblem, Options,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛰️  Satellite Image Selection Problem (SIMS) Example");
    println!("====================================================");

    // Create a sample SIMS problem
    let problem = create_sample_sims_problem();

    println!("📊 Problem Statistics:");
    println!("  - Variables: {}", problem.var_map.len());
    println!("  - Constraints: {}", problem.constraints.len());
    println!("  - Objectives: {}", problem.objectives.len());

    // Configure AUGMECON options
    let options = Options::new()
        .with_name("SIMS Example".to_string())
        .with_grid_points(10) // Use fewer grid points for demo
        .with_solver_option("log", "0"); // Disable solver output

    println!("\n🔧 AUGMECON Configuration:");
    println!("  - Grid points: {:?}", options.grid_points);
    println!("  - Algorithm: AUGMECON");

    // Solve the problem
    println!("\n🚀 Solving multi-objective optimization...");
    let mut augmecon = Augmecon::try_new(problem, options)?;
    let solutions = augmecon.solve()?;

    println!("\n✅ Solution Results:");
    println!("  - Total solutions found: {}", solutions.len());

    if !solutions.is_empty() {
        println!("  - Pareto front objectives:");
        for (i, solution) in solutions.iter().take(5).enumerate() {
            let objectives = solution.objectives();
            println!(
                "    Solution {}: [{:.2}, {:.2}, {:.2}, {:.2}]",
                i + 1,
                objectives[0],
                objectives[1],
                objectives[2],
                objectives[3]
            );
        }

        if solutions.len() > 5 {
            println!("    ... and {} more solutions", solutions.len() - 5);
        }
    }

    // Get payoff table
    let payoff_table = augmecon.get_payoff_table();
    if !payoff_table.is_empty() {
        println!("\n📈 Payoff Table:");
        for (i, row) in payoff_table.iter().enumerate() {
            println!(
                "    Objective {}: [{:.2}, {:.2}, {:.2}, {:.2}]",
                i + 1,
                row[0],
                row[1],
                row[2],
                row[3]
            );
        }
    }

    println!("\n🎯 Problem Interpretation:");
    println!("  Objective 1: Total cost (minimize)");
    println!("  Objective 2: Uncovered cloud area (minimize)");
    println!("  Objective 3: Sum of minimum resolutions (minimize for better quality)");
    println!("  Objective 4: Maximum incidence angle (minimize)");

    println!("\n✨ Example completed successfully!");

    Ok(())
}

/// Create a more realistic SIMS example
#[expect(
    dead_code,
    reason = "Example function provided for reference and potential future use in extended examples"
)]
fn create_realistic_sims_example() -> MultiObjectiveProblem {
    let mut config = SimsInstance::new(8, 6, 6, 200); // 8 images, 6 universe points, 6 clouds

    // Set up image coverage patterns (more realistic scenario)
    config.set_image_coverage(0, [0, 1, 2].iter().copied().collect());
    config.set_image_coverage(1, [1, 2, 3].iter().copied().collect());
    config.set_image_coverage(2, [2, 3, 4].iter().copied().collect());
    config.set_image_coverage(3, [3, 4, 5].iter().copied().collect());
    config.set_image_coverage(4, [0, 4, 5].iter().copied().collect());
    config.set_image_coverage(5, [0, 1, 5].iter().copied().collect());
    config.set_image_coverage(6, [0, 2, 4].iter().copied().collect());
    config.set_image_coverage(7, [1, 3, 5].iter().copied().collect());

    // Set up cloud coverage (which cloud entities each image can cover)
    config.set_cloud_coverage(0, std::iter::once(1).collect()); // Image 0 can cover cloud 1
    config.set_cloud_coverage(1, [2, 3].iter().copied().collect()); // Image 1 can cover clouds 2, 3
    config.set_cloud_coverage(2, std::iter::empty().collect()); // Image 2 covers no clouds
    config.set_cloud_coverage(3, std::iter::once(4).collect()); // Image 3 can cover cloud 4
    config.set_cloud_coverage(4, [0, 5].iter().copied().collect()); // Image 4 can cover clouds 0, 5
    config.set_cloud_coverage(5, std::iter::once(1).collect()); // Image 5 can cover cloud 1
    config.set_cloud_coverage(6, std::iter::once(2).collect()); // Image 6 can cover cloud 2
    config.set_cloud_coverage(7, std::iter::empty().collect()); // Image 7 covers no clouds

    // Set realistic costs (higher for better images)
    config.set_cost(0, 100.0);
    config.set_cost(1, 150.0);
    config.set_cost(2, 200.0); // Expensive but clear
    config.set_cost(3, 120.0);
    config.set_cost(4, 180.0);
    config.set_cost(5, 110.0);
    config.set_cost(6, 140.0);
    config.set_cost(7, 220.0); // Most expensive but clear

    // Set universe point areas
    config.set_area(0, 10.0);
    config.set_area(1, 8.0);
    config.set_area(2, 12.0);
    config.set_area(3, 6.0);
    config.set_area(4, 15.0);
    config.set_area(5, 9.0);

    // Set cloud areas
    config.set_cloud_area(0, 3.0);
    config.set_cloud_area(1, 2.5);
    config.set_cloud_area(2, 4.0);
    config.set_cloud_area(3, 1.8);
    config.set_cloud_area(4, 5.2);
    config.set_cloud_area(5, 2.1);

    // Set resolutions (higher is better)
    config.set_resolution(0, 1.0);
    config.set_resolution(1, 2.5);
    config.set_resolution(2, 4.0); // High resolution
    config.set_resolution(3, 1.8);
    config.set_resolution(4, 3.2);
    config.set_resolution(5, 1.5);
    config.set_resolution(6, 2.8);
    config.set_resolution(7, 4.5); // Highest resolution

    // Set incidence angles (lower is better)
    config.set_incidence_angle(0, 25.0);
    config.set_incidence_angle(1, 15.0);
    config.set_incidence_angle(2, 10.0); // Good angle
    config.set_incidence_angle(3, 30.0);
    config.set_incidence_angle(4, 20.0);
    config.set_incidence_angle(5, 35.0);
    config.set_incidence_angle(6, 18.0);
    config.set_incidence_angle(7, 8.0); // Best angle

    create_sims_problem(&config)
}
