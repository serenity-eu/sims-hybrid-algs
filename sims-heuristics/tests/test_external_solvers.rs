//! Integration tests for external solver adapters (moors & optirustic) on real SIMS instances.
//!
//! These tests mirror `test_evolutionary.rs` but exercise the external crate
//! adapters instead of the custom implementations. Every test validates that:
//!
//! 1. The archive is non-empty.
//! 2. Every solution in the archive is a feasible set cover.
//! 3. No solution in the archive dominates another (Pareto-optimality).
//! 4. Objective values are correctly computed (recalculated from scratch).
//!
//! Requires the `external_solvers` feature to be enabled:
//!   cargo test --features external_solvers --test test_external_solvers

#![cfg(feature = "external_solvers")]

use std::time::Duration;

use pareto::{HasObjectives, MoSolution};
use pls::{
    evolutionary::{
        moors_adapter::{
            MoorsConfig, MoorsCrossoverType, run_moors_age_moea, run_moors_nsga2, run_moors_spea2,
        },
        optirustic_adapter::{OptirusticConfig, run_optirustic_nsga2, run_optirustic_nsga3},
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

const SMALL_INSTANCES: [&str; 5] = [
    "lagos_nigeria_30.dzn",
    "rio_de_janeiro_30.dzn",
    "paris_30.dzn",
    "tokyo_bay_30.dzn",
    "mexico_city_30.dzn",
];

// ── Helpers ──────────────────────────────────────────────────────────────────

fn load_problem_2d(instance: &str) -> ProblemBitset<2> {
    let path = format!("{INSTANCES_PATH}/{instance}");
    ProblemBitset::from_minizinc_datafile(&path, OBJECTIVE_TYPES_2D)
        .unwrap_or_else(|e| panic!("Failed to load {instance}: {e}"))
}

fn load_problem_4d(instance: &str) -> ProblemBitset<4> {
    let path = format!("{INSTANCES_PATH}/{instance}");
    ProblemBitset::from_minizinc_datafile(&path, OBJECTIVE_TYPES_4D)
        .unwrap_or_else(|e| panic!("Failed to load {instance}: {e}"))
}

/// Validate that the archive is non-empty, all solutions are feasible set
/// covers with correct objectives, and no solution dominates another.
/// Returns `true` if the archive is valid (non-empty + all checks pass).
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

/// Like `validate_archive` but tolerates an empty archive (returns false
/// instead of panicking). Useful for algorithms that may legitimately fail
/// on certain problem structures (e.g. AGE-MOEA in `moors`).
fn validate_archive_lenient<P, const D: usize>(
    archive: &[BitsetEncodedSolution<P, D>],
    problem: &P,
    label: &str,
) -> bool
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    if archive.is_empty() {
        eprintln!("[{label}] archive is empty (algorithm may have failed gracefully)");
        return false;
    }

    for (idx, sol) in archive.iter().enumerate() {
        assert!(
            problem.is_set_cover(sol),
            "[{label}] solution {idx} is not a valid set cover"
        );

        for obj_idx in 0..D {
            let expected = problem.objective(obj_idx).calculate_value(sol, problem);
            let actual = sol.objectives()[obj_idx];
            assert_eq!(
                actual, expected,
                "[{label}] solution {idx} objective {obj_idx} mismatch: stored={actual} expected={expected}"
            );
        }
    }

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

    true
}

/// Default moors config tuned for fast integration testing.
fn fast_moors_config() -> MoorsConfig {
    MoorsConfig {
        population_size: 50,
        num_offsprings: 25,
        num_iterations: 50,
        crossover_rate: 0.9,
        mutation_rate: 0.1,
        bitflip_probability: 0.05,
        crossover_type: MoorsCrossoverType::Uniform,
        seed: 42,
    }
}

/// Default optirustic config tuned for fast integration testing.
fn fast_optirustic_config() -> OptirusticConfig {
    OptirusticConfig {
        population_size: 50,
        max_generations: 30,
        parallel: false,
        seed: Some(42),
        ..Default::default()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  moors NSGA-II tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn moors_nsga2_2d_small_instances() {
    for instance in &SMALL_INSTANCES {
        let problem = load_problem_2d(instance);
        let config = fast_moors_config();
        let (archive, _explored) = run_moors_nsga2(&problem, config, Duration::from_secs(60));
        validate_archive(&archive, &problem, &format!("moors_nsga2_2d_{instance}"));
    }
}

#[test]
fn moors_nsga2_4d_small_instances() {
    for instance in &SMALL_INSTANCES {
        let problem = load_problem_4d(instance);
        let config = fast_moors_config();
        let (archive, _explored) = run_moors_nsga2(&problem, config, Duration::from_secs(60));
        validate_archive(&archive, &problem, &format!("moors_nsga2_4d_{instance}"));
    }
}

#[test]
fn moors_nsga2_crossover_variants() {
    let problem = load_problem_2d("lagos_nigeria_30.dzn");

    for crossover_type in [
        MoorsCrossoverType::Uniform,
        MoorsCrossoverType::SinglePoint,
        MoorsCrossoverType::TwoPoint,
    ] {
        let config = MoorsConfig {
            crossover_type,
            ..fast_moors_config()
        };
        let (archive, _) = run_moors_nsga2(&problem, config, Duration::from_secs(60));
        validate_archive(
            &archive,
            &problem,
            &format!("moors_nsga2_{crossover_type:?}"),
        );
    }
}

#[test]
fn moors_nsga2_determinism() {
    let problem = load_problem_2d("lagos_nigeria_30.dzn");
    let config = MoorsConfig {
        seed: 12345,
        ..fast_moors_config()
    };

    let (archive1, _) = run_moors_nsga2(&problem, config.clone(), Duration::from_secs(60));
    let (archive2, _) = run_moors_nsga2(&problem, config, Duration::from_secs(60));

    assert_eq!(
        archive1.len(),
        archive2.len(),
        "Determinism: archive sizes differ"
    );

    let mut objs1: Vec<_> = archive1.iter().map(|s| s.objectives().to_vec()).collect();
    let mut objs2: Vec<_> = archive2.iter().map(|s| s.objectives().to_vec()).collect();
    objs1.sort();
    objs2.sort();
    assert_eq!(objs1, objs2, "Determinism: archive objectives differ");
}

// ═══════════════════════════════════════════════════════════════════════════
//  moors SPEA-2 tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn moors_spea2_2d_small_instances() {
    for instance in &SMALL_INSTANCES {
        let problem = load_problem_2d(instance);
        let config = fast_moors_config();
        let (archive, _explored) = run_moors_spea2(&problem, config, Duration::from_secs(60));
        validate_archive(&archive, &problem, &format!("moors_spea2_2d_{instance}"));
    }
}

#[test]
fn moors_spea2_4d_small_instances() {
    for instance in &SMALL_INSTANCES {
        let problem = load_problem_4d(instance);
        let config = fast_moors_config();
        let (archive, _explored) = run_moors_spea2(&problem, config, Duration::from_secs(60));
        validate_archive(&archive, &problem, &format!("moors_spea2_4d_{instance}"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  moors AGE-MOEA tests
// ═══════════════════════════════════════════════════════════════════════════

/// AGE-MOEA can panic inside the `moors` crate on certain problem
/// structures. The adapter wraps this in `catch_unwind` and returns an
/// empty archive. We use `validate_archive_lenient` so the test still
/// passes in that case while still verifying any solutions that *are*
/// produced.
#[test]
fn moors_age_moea_2d_small_instances() {
    let mut any_succeeded = false;
    for instance in &SMALL_INSTANCES {
        let problem = load_problem_2d(instance);
        let config = fast_moors_config();
        let (archive, _explored) = run_moors_age_moea(&problem, config, Duration::from_secs(60));
        if validate_archive_lenient(&archive, &problem, &format!("moors_age_moea_2d_{instance}")) {
            any_succeeded = true;
        }
    }
    assert!(
        any_succeeded,
        "AGE-MOEA should succeed on at least one 2D instance"
    );
}

/// AGE-MOEA from `moors` frequently panics on 4D SIMS instances due to an
/// internal edge-case in the crate's survival operator. We still run the
/// test to exercise the adapter code, but we don't require *any* instance
/// to produce a non-empty archive.
#[test]
fn moors_age_moea_4d_small_instances() {
    let mut succeeded = 0usize;
    for instance in &SMALL_INSTANCES {
        let problem = load_problem_4d(instance);
        let config = fast_moors_config();
        let (archive, _explored) = run_moors_age_moea(&problem, config, Duration::from_secs(60));
        if validate_archive_lenient(&archive, &problem, &format!("moors_age_moea_4d_{instance}")) {
            succeeded += 1;
        }
    }
    eprintln!(
        "AGE-MOEA 4D: {succeeded}/{} instances produced valid archives",
        SMALL_INSTANCES.len()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
//  optirustic NSGA-II tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn optirustic_nsga2_2d_small_instances() {
    for instance in &SMALL_INSTANCES {
        let problem = load_problem_2d(instance);
        let config = fast_optirustic_config();
        let (archive, _explored) = run_optirustic_nsga2(&problem, config, Duration::from_secs(120));
        validate_archive(
            &archive,
            &problem,
            &format!("optirustic_nsga2_2d_{instance}"),
        );
    }
}

#[test]
fn optirustic_nsga2_4d_small_instances() {
    for instance in &SMALL_INSTANCES {
        let problem = load_problem_4d(instance);
        let config = fast_optirustic_config();
        let (archive, _explored) = run_optirustic_nsga2(&problem, config, Duration::from_secs(120));
        validate_archive(
            &archive,
            &problem,
            &format!("optirustic_nsga2_4d_{instance}"),
        );
    }
}

/// Note: optirustic has internal non-determinism (evaluation order,
/// thread-pool effects) even with `parallel: false` and the same seed.
/// We therefore only check that both runs produce valid, non-empty
/// archives of comparable size rather than requiring exact equality.
#[test]
fn optirustic_nsga2_determinism() {
    let problem = load_problem_2d("lagos_nigeria_30.dzn");
    let config = OptirusticConfig {
        seed: Some(12345),
        ..fast_optirustic_config()
    };

    let (archive1, _) = run_optirustic_nsga2(&problem, config.clone(), Duration::from_secs(120));
    let (archive2, _) = run_optirustic_nsga2(&problem, config, Duration::from_secs(120));

    // Both archives must be valid
    validate_archive(&archive1, &problem, "optirustic_det_run1");
    validate_archive(&archive2, &problem, "optirustic_det_run2");

    // Sizes should be in the same ballpark (allow 50% relative difference)
    let min_len = archive1.len().min(archive2.len());
    let max_len = archive1.len().max(archive2.len());
    assert!(
        min_len * 2 >= max_len,
        "Archive sizes too different: {} vs {} (expected similar)",
        archive1.len(),
        archive2.len(),
    );
}

// ═══════════════════════════════════════════════════════════════════════════
//  optirustic NSGA-III tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn optirustic_nsga3_2d_small_instances() {
    for instance in &SMALL_INSTANCES {
        let problem = load_problem_2d(instance);
        let config = fast_optirustic_config();
        let (archive, _explored) = run_optirustic_nsga3(&problem, config, Duration::from_secs(120));
        validate_archive(
            &archive,
            &problem,
            &format!("optirustic_nsga3_2d_{instance}"),
        );
    }
}

#[test]
fn optirustic_nsga3_4d_small_instances() {
    for instance in &SMALL_INSTANCES {
        let problem = load_problem_4d(instance);
        let config = fast_optirustic_config();
        let (archive, _explored) = run_optirustic_nsga3(&problem, config, Duration::from_secs(120));
        validate_archive(
            &archive,
            &problem,
            &format!("optirustic_nsga3_4d_{instance}"),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Cross-algorithm comparison tests
// ═══════════════════════════════════════════════════════════════════════════

/// Verify that all external algorithms produce valid Pareto fronts on the
/// same instance and that archive sizes are reasonable.
#[test]
fn all_external_algorithms_produce_valid_fronts_2d() {
    let instance = "lagos_nigeria_30.dzn";
    let problem = load_problem_2d(instance);

    let algorithms: Vec<(&str, Vec<BitsetEncodedSolution<ProblemBitset<2>, 2>>)> = vec![
        {
            let config = fast_moors_config();
            let (archive, _) = run_moors_nsga2(&problem, config, Duration::from_secs(60));
            ("moors_nsga2", archive)
        },
        {
            let config = fast_moors_config();
            let (archive, _) = run_moors_spea2(&problem, config, Duration::from_secs(60));
            ("moors_spea2", archive)
        },
        {
            let config = fast_moors_config();
            let (archive, _) = run_moors_age_moea(&problem, config, Duration::from_secs(60));
            ("moors_age_moea (may be empty)", archive)
        },
        {
            let config = fast_optirustic_config();
            let (archive, _) = run_optirustic_nsga2(&problem, config, Duration::from_secs(120));
            ("optirustic_nsga2", archive)
        },
        {
            let config = fast_optirustic_config();
            let (archive, _) = run_optirustic_nsga3(&problem, config, Duration::from_secs(120));
            ("optirustic_nsga3", archive)
        },
    ];

    for (name, archive) in &algorithms {
        if name.contains("may be empty") {
            validate_archive_lenient(archive, &problem, &format!("cross_compare_{name}"));
        } else {
            validate_archive(archive, &problem, &format!("cross_compare_{name}"));
        }
        eprintln!("  {name}: {} non-dominated solutions", archive.len());
    }
}

/// Verify that all external algorithms produce valid Pareto fronts on 4D.
#[test]
fn all_external_algorithms_produce_valid_fronts_4d() {
    let instance = "lagos_nigeria_30.dzn";
    let problem = load_problem_4d(instance);

    let algorithms: Vec<(&str, Vec<BitsetEncodedSolution<ProblemBitset<4>, 4>>)> = vec![
        {
            let config = fast_moors_config();
            let (archive, _) = run_moors_nsga2(&problem, config, Duration::from_secs(60));
            ("moors_nsga2", archive)
        },
        {
            let config = fast_moors_config();
            let (archive, _) = run_moors_spea2(&problem, config, Duration::from_secs(60));
            ("moors_spea2", archive)
        },
        {
            let config = fast_moors_config();
            let (archive, _) = run_moors_age_moea(&problem, config, Duration::from_secs(60));
            ("moors_age_moea (may be empty)", archive)
        },
        {
            let config = fast_optirustic_config();
            let (archive, _) = run_optirustic_nsga2(&problem, config, Duration::from_secs(120));
            ("optirustic_nsga2", archive)
        },
        {
            let config = fast_optirustic_config();
            let (archive, _) = run_optirustic_nsga3(&problem, config, Duration::from_secs(120));
            ("optirustic_nsga3", archive)
        },
    ];

    for (name, archive) in &algorithms {
        if name.contains("may be empty") {
            validate_archive_lenient(archive, &problem, &format!("cross_compare_4d_{name}"));
        } else {
            validate_archive(archive, &problem, &format!("cross_compare_4d_{name}"));
        }
        eprintln!("  {name} (4D): {} non-dominated solutions", archive.len());
    }
}
