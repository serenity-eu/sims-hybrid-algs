//! Benchmark GPBA-A on a single SIMS instance across available solvers.
//!
//! Usage:
//!   cargo run --release --bin solver-benchmark --features coin_cbc \
//!     -- --dzn tests/input/sims/lagos_nigeria_30.dzn --solver coin_cbc --timeout 120
//!
//!   cargo run --release --bin solver-benchmark --features scip,scip_bundled \
//!     -- --dzn tests/input/sims/lagos_nigeria_30.dzn --solver scip --timeout 120

use std::{
    collections::HashSet,
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

use augmecon::{
    sims_problem::{create_sims_problem_with_objectives, SimsInstance, SimsObjective},
    GpbaA, GpbaConfig, Options, Solver,
};

fn usage() {
    eprintln!(
        "Usage: solver-benchmark \
         --dzn <path> \
         [--solver coin_cbc|highs|scip] \
         [--timeout <secs>] \
         [--per-solve-timeout <secs>] \
         [--grid-points <n>]"
    );
}

struct Args {
    dzn: PathBuf,
    solver: Solver,
    timeout: u64,
    per_solve_timeout: u64,
    grid_points: usize,
}

fn parse_args() -> Option<Args> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut dzn = None;
    let mut solver = Solver::CoinCbc;
    let mut timeout = 120u64;
    let mut per_solve_timeout = 0u64;
    let mut grid_points = 50usize;

    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--dzn" => {
                dzn = Some(PathBuf::from(&raw[i + 1]));
                i += 2;
            }
            "--solver" => {
                solver = match raw[i + 1].as_str() {
                    "coin_cbc" | "cbc" => Solver::CoinCbc,
                    "highs" => Solver::HiGHS,
                    "scip" => Solver::SCIP,
                    _ => Solver::Default,
                };
                i += 2;
            }
            "--timeout" => {
                timeout = raw[i + 1].parse().ok()?;
                i += 2;
            }
            "--per-solve-timeout" => {
                per_solve_timeout = raw[i + 1].parse().ok()?;
                i += 2;
            }
            "--grid-points" => {
                grid_points = raw[i + 1].parse().ok()?;
                i += 2;
            }
            other => {
                eprintln!("Unknown arg: {other}");
                return None;
            }
        }
    }

    Some(Args {
        dzn: dzn?,
        solver,
        timeout,
        per_solve_timeout,
        grid_points,
    })
}

// ── DZN parser (mirrors test_sims.rs) ────────────────────────────────────────

fn parse_array_of_sets(content: &str, key: &str) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut inside_array = false;
    let mut current_set = String::new();
    let mut brace_count: i32 = 0;

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with(&format!("{key} = [")) {
            inside_array = true;
            current_set = line.split('=').nth(1).unwrap_or("").trim().to_string();
        } else if inside_array {
            current_set.push(' ');
            current_set.push_str(line);
        }
        if inside_array && line.contains("];") {
            break;
        }
    }

    let array_content = current_set
        .trim_start_matches('[')
        .trim_end_matches("];")
        .trim();
    let mut in_set = false;
    let mut token = String::new();

    for ch in array_content.chars() {
        match ch {
            '{' => {
                in_set = true;
                brace_count += 1;
            }
            '}' => {
                brace_count -= 1;
                if brace_count == 0 {
                    in_set = false;
                    result.push(if token.is_empty() {
                        Vec::new()
                    } else {
                        token
                            .split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect()
                    });
                    token.clear();
                }
            }
            _ if in_set => {
                token.push(ch);
            }
            _ => {}
        }
    }
    result
}

fn parse_numeric_array(content: &str, key: &str) -> Vec<f64> {
    let mut inside = false;
    let mut buf = String::new();
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with(&format!("{key} = [")) {
            inside = true;
            buf = line.split('=').nth(1).unwrap_or("").trim().to_string();
        } else if inside {
            buf.push(' ');
            buf.push_str(line);
        }
        if inside && line.contains("];") {
            break;
        }
    }
    buf.trim_start_matches('[')
        .trim_end_matches("];")
        .trim()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

fn parse_dzn(path: &PathBuf) -> Result<SimsInstance, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;

    let num_images: usize = content
        .lines()
        .find(|l| l.trim().starts_with("num_images = "))
        .and_then(|l| {
            l.split('=')
                .nth(1)?
                .trim()
                .trim_end_matches(';')
                .parse()
                .ok()
        })
        .unwrap_or(0);
    let universe: usize = content
        .lines()
        .find(|l| l.trim().starts_with("universe = "))
        .and_then(|l| {
            l.split('=')
                .nth(1)?
                .trim()
                .trim_end_matches(';')
                .parse()
                .ok()
        })
        .unwrap_or(0);
    let max_cloud_area: i32 = content
        .lines()
        .find(|l| l.trim().starts_with("max_cloud_area = "))
        .and_then(|l| {
            l.split('=')
                .nth(1)?
                .trim()
                .trim_end_matches(';')
                .parse()
                .ok()
        })
        .unwrap_or(0);

    let images_raw = parse_array_of_sets(&content, "images");
    let clouds_raw = parse_array_of_sets(&content, "clouds");
    let costs = parse_numeric_array(&content, "costs");
    let areas = parse_numeric_array(&content, "areas");
    let resolution = parse_numeric_array(&content, "resolution");
    let incidence_angle = parse_numeric_array(&content, "incidence_angle");

    let max_cloud_id = clouds_raw
        .iter()
        .flat_map(|s| s.iter())
        .copied()
        .max()
        .unwrap_or(0);
    let num_clouds = max_cloud_id.max(1);

    let mut config = SimsInstance::new(num_images, universe, num_clouds, max_cloud_area);

    for (i, set) in images_raw.iter().enumerate() {
        config.set_image_coverage(i, set.iter().map(|&x| x - 1).collect::<HashSet<_>>());
    }
    for (i, set) in clouds_raw.iter().enumerate() {
        config.set_cloud_coverage(i, set.iter().map(|&x| x - 1).collect::<HashSet<_>>());
    }
    for (i, &v) in costs.iter().enumerate() {
        config.set_cost(i, v);
    }
    for (i, &v) in areas.iter().enumerate() {
        config.set_area(i, v);
    }
    for cloud_id in 0..num_clouds {
        config.set_cloud_area(cloud_id, 1.0);
    }
    for (i, &v) in resolution.iter().enumerate() {
        config.set_resolution(i, v);
    }
    for (i, &v) in incidence_angle.iter().enumerate() {
        config.set_incidence_angle(i, v);
    }

    Ok(config)
}

// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let args = match parse_args() {
        Some(a) => a,
        None => {
            usage();
            std::process::exit(1);
        }
    };

    println!("Instance        : {}", args.dzn.display());
    println!("Solver          : {}", args.solver.name());
    println!("Timeout         : {}s", args.timeout);
    println!("Grid points     : {}", args.grid_points);
    if args.per_solve_timeout > 0 {
        println!("Per-solve limit : {}s", args.per_solve_timeout);
    }
    println!();

    let instance = parse_dzn(&args.dzn)?;
    println!(
        "Loaded: {} images, {} universe pts",
        instance.num_images, instance.universe_size,
    );

    let objectives: HashSet<SimsObjective> = [
        SimsObjective::MinCost,
        SimsObjective::CloudCoverage,
        SimsObjective::MinResolution,
        SimsObjective::MaxIncidenceAngle,
    ]
    .into_iter()
    .collect();
    let problem = create_sims_problem_with_objectives(&instance, Some(&objectives));

    let options = Options::new()
        .with_name("benchmark".to_string())
        .with_grid_points(args.grid_points)
        .with_bypass_coefficient(true)
        .with_flag_array(true)
        .with_early_exit(true)
        .with_solver(args.solver);

    let per_solve =
        (args.per_solve_timeout > 0).then(|| Duration::from_secs(args.per_solve_timeout));

    let config = GpbaConfig {
        primary_objective: 0,
        manual_bounds: None,
        target_solutions: None,
        per_solve_timeout: per_solve,
    };

    println!("\nRunning GPBA-A …");
    let t0 = Instant::now();

    let mut gpba = GpbaA::new(config).with_timeout(Duration::from_secs(args.timeout));
    let result = gpba.generate_representation(&problem, &options);

    let elapsed = t0.elapsed();

    match result {
        Ok(front) => {
            println!("\n=== Results ===");
            println!("Wall time  : {:.2}s", elapsed.as_secs_f64());
            println!("Front size : {} solutions", front.len());
        }
        Err(e) => {
            eprintln!("\nERROR after {:.2}s: {e}", elapsed.as_secs_f64());
            std::process::exit(1);
        }
    }

    Ok(())
}
