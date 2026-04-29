//! Integration tests for NSGA-II and MOEA/D on real SIMS problem instances.
//!
//! These tests mirror `test_real_instances.rs` but exercise the evolutionary
//! algorithms instead of PLS.  Every test validates that:
//!
//! 1. The archive is non-empty.
//! 2. Every solution in the archive is a feasible set cover.
//! 3. No solution in the archive dominates another (Pareto-optimality).
//! 4. Objective values are correctly computed (recalculated from scratch).

use std::path::Path;
use std::time::Duration;

use pareto::{HasObjectives, MoSolution};
use pls::{
    evolutionary::{
        moead::{Moead, MoeadConfig},
        nsga2::{Nsga2, Nsga2Config},
    },
    objectives::ObjectiveType,
    problem::SetCoverProblem,
    problem_bitset::ProblemBitset,
    solution::bitset_encoded_solution::BitsetEncodedSolution,
};

// ── Constants ────────────────────────────────────────────────────────────────

const INSTANCES_PATH: &str = "tests/data";

const OBJECTIVE_TYPES_2D: [ObjectiveType; 2] =
    [ObjectiveType::TotalCost, ObjectiveType::CloudyArea];

const OBJECTIVE_TYPES_4D: [ObjectiveType; 4] = [
    ObjectiveType::TotalCost,
    ObjectiveType::CloudyArea,
    ObjectiveType::MinResolution,
    ObjectiveType::MaxIncidenceAngle,
];

const SMALL_INSTANCES: [&str; 10] = [
    "lagos_nigeria_30.dzn",
    "rio_de_janeiro_30.dzn",
    "paris_30.dzn",
    "tokyo_bay_30.dzn",
    "mexico_city_30.dzn",
    "lagos_nigeria_50.dzn",
    "rio_de_janeiro_50.dzn",
    "paris_50.dzn",
    "tokyo_bay_50.dzn",
    "mexico_city_50.dzn",
];

const MEDIUM_INSTANCES: [&str; 5] = [
    "lagos_nigeria_100.dzn",
    "rio_de_janeiro_100.dzn",
    "paris_100.dzn",
    "tokyo_bay_100.dzn",
    "mexico_city_100.dzn",
];

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Validate that the archive is non-empty, all solutions are feasible set
/// covers with correct objectives, and no solution dominates another.
fn validate_archive<P, const D: usize>(
    archive: &[BitsetEncodedSolution<P, D>],
    problem: &P,
    label: &str,
) where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    assert!(!archive.is_empty(), "[{label}] archive must not be empty");

    for (idx, sol) in archive.iter().enumerate() {
        // Feasibility
        assert!(
            problem.is_set_cover(sol),
            "[{label}] solution {idx} is not a valid set cover"
        );

        // Objective correctness: recalculate from scratch and compare
        for obj_idx in 0..D {
            let expected = problem.objective(obj_idx).calculate_value(sol, problem);
            let actual = sol.objectives()[obj_idx];
            assert_eq!(
                actual, expected,
                "[{label}] solution {idx} objective {obj_idx} mismatch: stored={actual} expected={expected}"
            );
        }
    }

    // Non-dominance within the archive
    for i in 0..archive.len() {
        for j in 0..archive.len() {
            if i != j {
                assert!(
                    !archive[i].dominates(archive[j].objectives()),
                    "[{label}] archive solution {i} dominates solution {j}: {:?} vs {:?}",
                    archive[i].objectives(),
                    archive[j].objectives(),
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  NSGA-II tests
// ═══════════════════════════════════════════════════════════════════════════

// ── 2-objective, small instances ─────────────────────────────────────────

#[test]
fn nsga2_2d_small_instances() {
    let _ = env_logger::try_init();

    let config = Nsga2Config {
        population_size: 50,
        crossover_rate: 0.9,
        swap_mutation_rate: 0.3,
        add_prune_mutation_rate: 0.2,
        multi_swap_rate: 0.15,
        multi_swap_max_removals: 3,
        coverage_biased_crossover_fraction: 0.5,
        ..Default::default()
    };

    for instance_file in SMALL_INSTANCES {
        let instance_path = Path::new(INSTANCES_PATH).join(instance_file);
        let problem =
            ProblemBitset::<2>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES_2D)
                .unwrap_or_else(|e| panic!("failed to load {instance_file}: {e}"));

        let mut nsga2 = Nsga2::new(&problem, config.clone(), None, 42);
        let archive = nsga2.run(200, Duration::from_secs(30));

        validate_archive(&archive, &problem, &format!("nsga2-2d-{instance_file}"));
    }
}

// ── 4-objective, small instances ─────────────────────────────────────────

#[test]
fn nsga2_4d_small_instances() {
    let _ = env_logger::try_init();

    let config = Nsga2Config {
        population_size: 50,
        crossover_rate: 0.9,
        swap_mutation_rate: 0.3,
        add_prune_mutation_rate: 0.2,
        multi_swap_rate: 0.15,
        multi_swap_max_removals: 3,
        coverage_biased_crossover_fraction: 0.5,
        ..Default::default()
    };

    for instance_file in SMALL_INSTANCES {
        let instance_path = Path::new(INSTANCES_PATH).join(instance_file);
        let problem =
            ProblemBitset::<4>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES_4D)
                .unwrap_or_else(|e| panic!("failed to load {instance_file}: {e}"));

        let mut nsga2 = Nsga2::new(&problem, config.clone(), None, 42);
        let archive = nsga2.run(200, Duration::from_secs(30));

        validate_archive(&archive, &problem, &format!("nsga2-4d-{instance_file}"));
    }
}

// ── 2-objective, medium instances ────────────────────────────────────────

#[test]
fn nsga2_2d_medium_instances() {
    let _ = env_logger::try_init();

    let config = Nsga2Config {
        population_size: 80,
        crossover_rate: 0.9,
        swap_mutation_rate: 0.3,
        add_prune_mutation_rate: 0.2,
        multi_swap_rate: 0.15,
        multi_swap_max_removals: 3,
        coverage_biased_crossover_fraction: 0.5,
        ..Default::default()
    };

    for instance_file in MEDIUM_INSTANCES {
        let instance_path = Path::new(INSTANCES_PATH).join(instance_file);
        let problem =
            ProblemBitset::<2>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES_2D)
                .unwrap_or_else(|e| panic!("failed to load {instance_file}: {e}"));

        let mut nsga2 = Nsga2::new(&problem, config.clone(), None, 42);
        let archive = nsga2.run(200, Duration::from_secs(60));

        validate_archive(&archive, &problem, &format!("nsga2-2d-{instance_file}"));
    }
}

// ── 4-objective, medium instances ────────────────────────────────────────

#[test]
fn nsga2_4d_medium_instances() {
    let _ = env_logger::try_init();

    let config = Nsga2Config {
        population_size: 80,
        ..Default::default()
    };

    for instance_file in MEDIUM_INSTANCES {
        let instance_path = Path::new(INSTANCES_PATH).join(instance_file);
        let problem =
            ProblemBitset::<4>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES_4D)
                .unwrap_or_else(|e| panic!("failed to load {instance_file}: {e}"));

        let mut nsga2 = Nsga2::new(&problem, config.clone(), None, 42);
        let archive = nsga2.run(200, Duration::from_secs(60));

        validate_archive(&archive, &problem, &format!("nsga2-4d-{instance_file}"));
    }
}

// ── NSGA-II with initial population ──────────────────────────────────────

#[test]
fn nsga2_2d_with_initial_population() {
    let _ = env_logger::try_init();

    let instance_path = Path::new(INSTANCES_PATH).join("lagos_nigeria_30.dzn");
    let problem = ProblemBitset::<2>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES_2D)
        .expect("failed to load lagos_nigeria_30.dzn");

    // Seed with a few random solutions
    let initial: Vec<BitsetEncodedSolution<ProblemBitset<2>, 2>> = (0..5)
        .map(|seed| BitsetEncodedSolution::random_with_seed(&problem, seed))
        .collect();

    let config = Nsga2Config {
        population_size: 30,
        ..Default::default()
    };

    let mut nsga2 = Nsga2::new(&problem, config, Some(initial), 42);
    let archive = nsga2.run(100, Duration::from_secs(10));

    validate_archive(&archive, &problem, "nsga2-2d-initial-pop-lagos30");
}

// ── NSGA-II determinism ──────────────────────────────────────────────────

#[test]
fn nsga2_determinism_with_same_seed() {
    let _ = env_logger::try_init();

    let instance_path = Path::new(INSTANCES_PATH).join("lagos_nigeria_30.dzn");
    let problem = ProblemBitset::<2>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES_2D)
        .expect("failed to load lagos_nigeria_30.dzn");

    let config = Nsga2Config {
        population_size: 20,
        ..Default::default()
    };

    let mut nsga2_a = Nsga2::new(&problem, config.clone(), None, 123);
    let archive_a = nsga2_a.run(50, Duration::from_secs(60));

    let mut nsga2_b = Nsga2::new(&problem, config, None, 123);
    let archive_b = nsga2_b.run(50, Duration::from_secs(60));

    assert_eq!(
        archive_a.len(),
        archive_b.len(),
        "deterministic runs should yield same archive size"
    );

    let mut objs_a: Vec<_> = archive_a.iter().map(|s| *s.objectives()).collect();
    let mut objs_b: Vec<_> = archive_b.iter().map(|s| *s.objectives()).collect();
    objs_a.sort();
    objs_b.sort();
    assert_eq!(
        objs_a, objs_b,
        "deterministic runs should yield same objectives"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
//  MOEA/D tests
// ═══════════════════════════════════════════════════════════════════════════

// ── 2-objective, small instances ─────────────────────────────────────────

#[test]
fn moead_2d_small_instances() {
    let _ = env_logger::try_init();

    let config = MoeadConfig {
        num_divisions: 49, // 50 weight vectors
        neighbourhood_size: 15,
        delta: 0.9,
        max_replacements: 2,
        crossover_rate: 1.0,
        swap_mutation_rate: 0.3,
        add_prune_mutation_rate: 0.2,
        multi_swap_rate: 0.15,
        multi_swap_max_removals: 3,
        coverage_biased_crossover_fraction: 0.5,
        ..Default::default()
    };

    for instance_file in SMALL_INSTANCES {
        let instance_path = Path::new(INSTANCES_PATH).join(instance_file);
        let problem =
            ProblemBitset::<2>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES_2D)
                .unwrap_or_else(|e| panic!("failed to load {instance_file}: {e}"));

        let mut moead = Moead::new(&problem, config.clone(), None, 42);
        let archive = moead.run(200, Duration::from_secs(30));

        validate_archive(&archive, &problem, &format!("moead-2d-{instance_file}"));
    }
}

// ── 4-objective, small instances ─────────────────────────────────────────

#[test]
fn moead_4d_small_instances() {
    let _ = env_logger::try_init();

    let config = MoeadConfig {
        num_divisions: 7, // C(7+3, 3) = 120 weight vectors for 4 objectives
        neighbourhood_size: 20,
        delta: 0.9,
        max_replacements: 2,
        crossover_rate: 1.0,
        swap_mutation_rate: 0.3,
        add_prune_mutation_rate: 0.2,
        multi_swap_rate: 0.15,
        multi_swap_max_removals: 3,
        coverage_biased_crossover_fraction: 0.5,
        ..Default::default()
    };

    for instance_file in SMALL_INSTANCES {
        let instance_path = Path::new(INSTANCES_PATH).join(instance_file);
        let problem =
            ProblemBitset::<4>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES_4D)
                .unwrap_or_else(|e| panic!("failed to load {instance_file}: {e}"));

        let mut moead = Moead::new(&problem, config.clone(), None, 42);
        let archive = moead.run(200, Duration::from_secs(30));

        validate_archive(&archive, &problem, &format!("moead-4d-{instance_file}"));
    }
}

// ── 2-objective, medium instances ────────────────────────────────────────

#[test]
fn moead_2d_medium_instances() {
    let _ = env_logger::try_init();

    let config = MoeadConfig {
        num_divisions: 79, // 80 weight vectors
        neighbourhood_size: 20,
        delta: 0.9,
        max_replacements: 2,
        ..Default::default()
    };

    for instance_file in MEDIUM_INSTANCES {
        let instance_path = Path::new(INSTANCES_PATH).join(instance_file);
        let problem =
            ProblemBitset::<2>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES_2D)
                .unwrap_or_else(|e| panic!("failed to load {instance_file}: {e}"));

        let mut moead = Moead::new(&problem, config.clone(), None, 42);
        let archive = moead.run(200, Duration::from_secs(60));

        validate_archive(&archive, &problem, &format!("moead-2d-{instance_file}"));
    }
}

// ── 4-objective, medium instances ────────────────────────────────────────

#[test]
fn moead_4d_medium_instances() {
    let _ = env_logger::try_init();

    let config = MoeadConfig {
        num_divisions: 7, // 120 weight vectors for 4D
        neighbourhood_size: 20,
        ..Default::default()
    };

    for instance_file in MEDIUM_INSTANCES {
        let instance_path = Path::new(INSTANCES_PATH).join(instance_file);
        let problem =
            ProblemBitset::<4>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES_4D)
                .unwrap_or_else(|e| panic!("failed to load {instance_file}: {e}"));

        let mut moead = Moead::new(&problem, config.clone(), None, 42);
        let archive = moead.run(200, Duration::from_secs(60));

        validate_archive(&archive, &problem, &format!("moead-4d-{instance_file}"));
    }
}

// ── MOEA/D with initial population ───────────────────────────────────────

#[test]
fn moead_2d_with_initial_population() {
    let _ = env_logger::try_init();

    let instance_path = Path::new(INSTANCES_PATH).join("lagos_nigeria_30.dzn");
    let problem = ProblemBitset::<2>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES_2D)
        .expect("failed to load lagos_nigeria_30.dzn");

    let initial: Vec<BitsetEncodedSolution<ProblemBitset<2>, 2>> = (0..5)
        .map(|seed| BitsetEncodedSolution::random_with_seed(&problem, seed))
        .collect();

    let config = MoeadConfig {
        num_divisions: 19, // 20 weight vectors
        neighbourhood_size: 8,
        ..Default::default()
    };

    let mut moead = Moead::new(&problem, config, Some(initial), 42);
    let archive = moead.run(100, Duration::from_secs(10));

    validate_archive(&archive, &problem, "moead-2d-initial-pop-lagos30");
}

// ── MOEA/D PBI mode ──────────────────────────────────────────────────────

#[test]
fn moead_2d_pbi_small_instances() {
    let _ = env_logger::try_init();

    let config = MoeadConfig {
        num_divisions: 49,
        neighbourhood_size: 15,
        use_pbi: true,
        pbi_theta: 5.0,
        ..Default::default()
    };

    for instance_file in &SMALL_INSTANCES[..5] {
        let instance_path = Path::new(INSTANCES_PATH).join(instance_file);
        let problem =
            ProblemBitset::<2>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES_2D)
                .unwrap_or_else(|e| panic!("failed to load {instance_file}: {e}"));

        let mut moead = Moead::new(&problem, config.clone(), None, 42);
        let archive = moead.run(200, Duration::from_secs(30));

        validate_archive(&archive, &problem, &format!("moead-pbi-{instance_file}"));
    }
}

// ── MOEA/D determinism ───────────────────────────────────────────────────

#[test]
fn moead_determinism_with_same_seed() {
    let _ = env_logger::try_init();

    let instance_path = Path::new(INSTANCES_PATH).join("lagos_nigeria_30.dzn");
    let problem = ProblemBitset::<2>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES_2D)
        .expect("failed to load lagos_nigeria_30.dzn");

    let config = MoeadConfig {
        num_divisions: 19,
        neighbourhood_size: 8,
        ..Default::default()
    };

    let mut moead_a = Moead::new(&problem, config.clone(), None, 456);
    let archive_a = moead_a.run(50, Duration::from_secs(60));

    let mut moead_b = Moead::new(&problem, config, None, 456);
    let archive_b = moead_b.run(50, Duration::from_secs(60));

    assert_eq!(
        archive_a.len(),
        archive_b.len(),
        "deterministic runs should yield same archive size"
    );

    let mut objs_a: Vec<_> = archive_a.iter().map(|s| *s.objectives()).collect();
    let mut objs_b: Vec<_> = archive_b.iter().map(|s| *s.objectives()).collect();
    objs_a.sort();
    objs_b.sort();
    assert_eq!(
        objs_a, objs_b,
        "deterministic runs should yield same objectives"
    );
}

// ── MOEA/D ideal point convergence ───────────────────────────────────────

#[test]
fn moead_ideal_point_improves() {
    let _ = env_logger::try_init();

    let instance_path = Path::new(INSTANCES_PATH).join("lagos_nigeria_30.dzn");
    let problem = ProblemBitset::<2>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES_2D)
        .expect("failed to load lagos_nigeria_30.dzn");

    let config = MoeadConfig {
        num_divisions: 49,
        neighbourhood_size: 15,
        ..Default::default()
    };

    let mut moead = Moead::new(&problem, config, None, 42);
    let initial_ideal = *moead.ideal_point();
    let _archive = moead.run(100, Duration::from_secs(30));
    let final_ideal = *moead.ideal_point();

    for i in 0..2 {
        assert!(
            final_ideal[i] <= initial_ideal[i],
            "ideal point objective {i} should not worsen: was {:.0} now {:.0}",
            initial_ideal[i],
            final_ideal[i],
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Cross-algorithm: same instance, both algorithms produce valid Pareto sets
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn both_algorithms_produce_valid_fronts_2d() {
    let _ = env_logger::try_init();

    let instance_path = Path::new(INSTANCES_PATH).join("paris_30.dzn");
    let problem = ProblemBitset::<2>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES_2D)
        .expect("failed to load paris_30.dzn");

    // NSGA-II
    let nsga2_config = Nsga2Config {
        population_size: 50,
        ..Default::default()
    };
    let mut nsga2 = Nsga2::new(&problem, nsga2_config, None, 42);
    let nsga2_archive = nsga2.run(200, Duration::from_secs(20));
    validate_archive(&nsga2_archive, &problem, "cross-nsga2-paris30");

    // MOEA/D
    let moead_config = MoeadConfig {
        num_divisions: 49,
        neighbourhood_size: 15,
        ..Default::default()
    };
    let mut moead = Moead::new(&problem, moead_config, None, 42);
    let moead_archive = moead.run(200, Duration::from_secs(20));
    validate_archive(&moead_archive, &problem, "cross-moead-paris30");

    // Both should find *something*
    println!(
        "paris_30 NSGA-II archive size: {}, MOEA/D archive size: {}",
        nsga2_archive.len(),
        moead_archive.len()
    );
}

#[test]
fn both_algorithms_produce_valid_fronts_4d() {
    let _ = env_logger::try_init();

    let instance_path = Path::new(INSTANCES_PATH).join("tokyo_bay_50.dzn");
    let problem = ProblemBitset::<4>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES_4D)
        .expect("failed to load tokyo_bay_50.dzn");

    // NSGA-II
    let nsga2_config = Nsga2Config {
        population_size: 60,
        ..Default::default()
    };
    let mut nsga2 = Nsga2::new(&problem, nsga2_config, None, 77);
    let nsga2_archive = nsga2.run(200, Duration::from_secs(30));
    validate_archive(&nsga2_archive, &problem, "cross-nsga2-tokyo50-4d");

    // MOEA/D
    let moead_config = MoeadConfig {
        num_divisions: 7,
        neighbourhood_size: 20,
        ..Default::default()
    };
    let mut moead = Moead::new(&problem, moead_config, None, 77);
    let moead_archive = moead.run(200, Duration::from_secs(30));
    validate_archive(&moead_archive, &problem, "cross-moead-tokyo50-4d");

    println!(
        "tokyo_bay_50 (4D) NSGA-II archive: {}, MOEA/D archive: {}",
        nsga2_archive.len(),
        moead_archive.len()
    );
}
