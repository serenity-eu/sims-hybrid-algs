// Example usage of the ProbabilisticProbingNeighborhood trait
// This demonstrates how to use the new probabilistic probing functionality
//
// The ProbabilisticProbingNeighborhood trait provides an alternative to traditional
// exhaustive neighborhood exploration by using probabilistic sampling in objective
// space to find promising neighbors more efficiently.

use crate::{
    probabilistic_probing_neighborhood::ProbabilisticProbingNeighborhood, problem::Problem,
    solution::SIMSModifiable, solution_impl::bitset_encoded_solution::BitsetEncodedSolution,
    timer::Timer,
};

/// Example function showing how to use probabilistic probing neighborhood
pub fn example_probabilistic_probing<const D: usize>(
    solution: &BitsetEncodedSolution<D>,
    problem: &Problem<BitsetEncodedSolution<D>, D>,
) {
    // Create a timer for the neighborhood exploration
    let timer = Timer::start(std::time::Duration::from_secs(30));

    // Basic probabilistic probing with default parameters
    let neighbors_basic = solution.probabilistic_probing_neighborhood(
        1,       // k: single image removal
        problem, // problem instance
        &timer,  // timer for time control
        false,   // non-deterministic
        0.3,     // 30% probing probability
        50,      // maximum 50 probes
        None,    // no objective weights (uniform)
    );

    println!(
        "Found {} neighbors with basic probabilistic probing",
        neighbors_basic.len()
    );

    // All returned solutions are guaranteed to be valid (satisfy Set Cover constraints)
    for (i, neighbor) in neighbors_basic.iter().take(3).enumerate() {
        println!(
            "Neighbor {}: {} selected images, valid: {}",
            i,
            neighbor.selected_images().count(),
            neighbor.is_valid(problem)
        );
    }

    // Advanced probabilistic probing with custom objective weights
    // We'll create weights dynamically based on D
    let mut objective_weights = vec![1.0 / D as f64; D];

    // If D is small, we can customize the weights
    if D >= 2 {
        objective_weights[0] = 0.6; // Higher weight for first objective
        objective_weights[1] = 0.4; // Lower weight for second objective
        if D > 2 {
            // Redistribute remaining weight
            let weight_sum = 0.6 + 0.4;
            let remaining = 1.0 - weight_sum;
            let remaining_per_obj = remaining / (D - 2) as f64;
            for weight in objective_weights.iter_mut().skip(2) {
                *weight = remaining_per_obj;
            }
        }
    }

    // Convert Vec to array for the API
    let weights_array: [f64; D] = objective_weights.try_into().unwrap_or_else(|_| {
        // Fallback to uniform weights if conversion fails
        [1.0 / D as f64; D]
    });

    let neighbors_weighted = solution.probabilistic_probing_neighborhood(
        2,                    // k: double image removal
        problem,              // problem instance
        &timer,               // timer for time control
        true,                 // deterministic for reproducibility
        0.4,                  // 40% probing probability
        100,                  // maximum 100 probes
        Some(&weights_array), // custom objective weights
    );

    println!(
        "Found {} neighbors with weighted probabilistic probing",
        neighbors_weighted.len()
    );

    // High-intensity probabilistic exploration
    let neighbors_intensive = solution.probabilistic_probing_neighborhood(
        1,       // k: single image removal
        problem, // problem instance
        &timer,  // timer for time control
        false,   // non-deterministic
        0.8,     // 80% probing probability (more exploration)
        200,     // maximum 200 probes
        None,    // no objective weights
    );

    println!(
        "Found {} neighbors with intensive probabilistic probing",
        neighbors_intensive.len()
    );
}

/// Comparison between traditional neighborhood and probabilistic probing
pub fn compare_neighborhood_methods<const D: usize>(
    solution: &BitsetEncodedSolution<D>,
    problem: &Problem<BitsetEncodedSolution<D>, D>,
) {
    let timer = Timer::start(std::time::Duration::from_secs(60));

    // Traditional neighborhood exploration
    let start_time = std::time::Instant::now();
    let traditional_neighbors = solution.neighborhood(1, problem, &timer, false);
    let traditional_time = start_time.elapsed();

    // Reset timer for fair comparison
    let timer = Timer::start(std::time::Duration::from_secs(60));

    // Probabilistic probing neighborhood
    let start_time = std::time::Instant::now();
    let probabilistic_neighbors =
        solution.probabilistic_probing_neighborhood(1, problem, &timer, false, 0.3, 50, None);
    let probabilistic_time = start_time.elapsed();

    println!("Comparison Results:");
    println!(
        "Traditional neighborhood: {} solutions in {:?}",
        traditional_neighbors.len(),
        traditional_time
    );
    println!(
        "Probabilistic probing: {} solutions in {:?}",
        probabilistic_neighbors.len(),
        probabilistic_time
    );

    if probabilistic_time < traditional_time {
        println!("Probabilistic probing was faster!");
    } else {
        println!("Traditional method was faster!");
    }
}
