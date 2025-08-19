# Solver Configuration Guide

This guide covers all aspects of configuring the AUGMECON solver for optimal performance and customized behavior.

## Table of Contents

1. [Options Overview](#options-overview)
2. [Grid Configuration](#grid-configuration)
3. [Algorithm Variants](#algorithm-variants)
4. [Performance Tuning](#performance-tuning)
5. [Advanced Settings](#advanced-settings)
6. [Solver Backend Configuration](#solver-backend-configuration)
7. [Logging and Monitoring](#logging-and-monitoring)

## Options Overview

The `Options` struct provides comprehensive control over the AUGMECON solver behavior:

```rust
use augmecon::Options;

let options = Options::new()
    .with_name("my_problem")
    .with_grid_points(100)
    .with_penalty_weight(1e-3)
    .with_round_decimals(6);
```

### Default Configuration

```rust
let default_options = Options::default();
// Equivalent to:
let options = Options {
    name: "Undefined".to_string(),
    grid_points: None,                    // Must be set manually
    nadir_points: None,                   // Auto-calculated if not provided
    penalty_weight: 1e-3,                 // Standard augmentation parameter
    round_decimals: 9,                    // High precision for results
    nadir_ratio: 1.0,                     // No nadir adjustment
    early_exit: true,                     // Enable AUGMECON optimization
    bypass_coefficient: true,             // Enable AUGMECON2 optimization
    flag_array: true,                     // Enable AUGMECON-R optimization
    cpu_count: num_cpus::get(),           // Use all available cores
    redivide_work: true,                  // Dynamic work redistribution
    shared_flag: true,                    // Shared flag array optimization
    output_excel: true,                   // Generate Excel output
    process_logging: false,               // Minimal logging
    process_timeout: None,                // No timeout
    solver_options: HashMap::new(),       // No custom solver settings
};
```

## Grid Configuration

The grid configuration determines how the ε-constraint method explores the objective space.

### Basic Grid Points

```rust
// Conservative: Good coverage, longer solve time
let options = Options::new().with_grid_points(100);

// Balanced: Good trade-off between quality and speed
let options = Options::new().with_grid_points(50);

// Fast: Quick exploration, lower resolution
let options = Options::new().with_grid_points(20);
```

### Grid Points Guidelines

| Problem Size | Variables | Objectives | Recommended Grid Points | Expected Solutions |
|-------------|-----------|------------|------------------------|-------------------|
| Small | < 50 | 2 | 50-100 | 50-100 |
| Small | < 50 | 3 | 20-30 | 400-27,000 |
| Medium | 50-200 | 2 | 100-200 | 100-200 |
| Medium | 50-200 | 3 | 10-20 | 1,000-8,000 |
| Large | > 200 | 2 | 200-500 | 200-500 |
| Large | > 200 | 3 | 5-15 | 125-3,375 |

**Important**: Grid points complexity grows exponentially: `grid_points^(num_objectives-1)`

### Custom Nadir Points

Control the range of exploration by setting custom nadir points:

```rust
// Auto-calculate nadir points (recommended)
let options = Options::new()
    .with_grid_points(50);

// Custom nadir points for 3-objective problem
// (Only for objectives 2 and 3; first objective is always optimized)
let options = Options::new()
    .with_grid_points(20)
    .with_nadir_points(vec![1000.0, 500.0]);

// Adjust nadir ratio for conservative/aggressive exploration
let options = Options::new()
    .with_grid_points(50)
    .with_nadir_ratio(1.1);  // 10% beyond worst case
```

### Nadir Point Strategy

```rust
fn configure_nadir_strategy(
    problem_type: &str,
    num_objectives: usize
) -> Options {
    match problem_type {
        "financial" => {
            // Conservative: stay within known ranges
            Options::new()
                .with_grid_points(100)
                .with_nadir_ratio(1.0)
        },
        "engineering" => {
            // Exploratory: push beyond typical ranges
            Options::new()
                .with_grid_points(50)
                .with_nadir_ratio(1.2)
        },
        "research" => {
            // Comprehensive: maximum coverage
            Options::new()
                .with_grid_points(if num_objectives == 2 { 200 } else { 30 })
                .with_nadir_ratio(1.1)
        },
        _ => Options::new().with_grid_points(50)
    }
}
```

## Algorithm Variants

AUGMECON-RS implements several algorithmic enhancements that can be enabled or disabled.

### Classic AUGMECON

Basic augmented ε-constraint method:

```rust
let options = Options::new()
    .with_grid_points(50)
    .with_early_exit(false)      // Disable early termination
    .with_bypass_coefficient(false)  // Disable bypass optimization
    .with_flag_array(false);     // Disable flag array optimization
```

### AUGMECON2 (Bypass Coefficient)

Improved version with bypass coefficient optimization:

```rust
let options = Options::new()
    .with_grid_points(50)
    .with_bypass_coefficient(true)   // Enable bypass optimization
    .with_early_exit(true);          // Keep early exit enabled
```

**Benefits:**
- Reduces redundant constraint evaluations
- Improves performance on problems with many dominated solutions
- Particularly effective for 3+ objective problems

### AUGMECON-R (Flag Array)

Advanced version with flag array optimization:

```rust
let options = Options::new()
    .with_grid_points(50)
    .with_flag_array(true)          // Enable flag array
    .with_bypass_coefficient(true)   // Combine with bypass coefficient
    .with_early_exit(true);
```

**Benefits:**
- Eliminates redundant computations
- Significant speedup for large grid sizes
- Memory efficient implementation

### Recommended Configurations

#### For Speed (Quick Results)
```rust
let fast_options = Options::new()
    .with_grid_points(20)
    .with_early_exit(true)
    .with_bypass_coefficient(true)
    .with_flag_array(true)
    .with_penalty_weight(1e-2);  // Larger penalty for faster convergence
```

#### For Quality (Comprehensive Results)
```rust
let quality_options = Options::new()
    .with_grid_points(100)
    .with_early_exit(false)      // Explore fully
    .with_bypass_coefficient(true)
    .with_flag_array(true)
    .with_penalty_weight(1e-4)   // Smaller penalty for precision
    .with_round_decimals(12);    // Higher precision
```

#### For Research (Maximum Coverage)
```rust
let research_options = Options::new()
    .with_grid_points(200)
    .with_early_exit(false)
    .with_bypass_coefficient(false)  // Methodical exploration
    .with_flag_array(false)
    .with_penalty_weight(1e-6)
    .with_round_decimals(15);
```

## Performance Tuning

### Memory vs Speed Trade-offs

```rust
// Memory optimized (slower but uses less memory)
let memory_efficient = Options::new()
    .with_grid_points(50)
    .with_flag_array(false)      // Reduced memory usage
    .with_shared_flag(false);    // No shared memory structures

// Speed optimized (faster but uses more memory)
let speed_optimized = Options::new()
    .with_grid_points(50)
    .with_flag_array(true)       // Cache intermediate results
    .with_shared_flag(true)      // Shared memory optimizations
    .with_early_exit(true);      // Early termination
```

### Parallel Processing

```rust
// Maximum parallelization
let parallel_options = Options::new()
    .with_grid_points(100)
    .with_cpu_count(num_cpus::get())  // Use all cores
    .with_redivide_work(true)         // Dynamic load balancing
    .with_shared_flag(true);          // Shared memory optimizations

// Conservative parallelization (for constrained environments)
let conservative_parallel = Options::new()
    .with_grid_points(100)
    .with_cpu_count(4)                // Limit core usage
    .with_redivide_work(false)        // Static work division
    .with_shared_flag(false);         // No shared memory
```

### Adaptive Configuration

```rust
fn adaptive_configuration(
    num_objectives: usize,
    num_variables: usize,
    num_constraints: usize
) -> Options {
    let complexity_score = num_objectives * num_variables + num_constraints;
    
    let (grid_points, penalty_weight) = match complexity_score {
        0..=100 => (100, 1e-3),
        101..=500 => (50, 1e-3),
        501..=1000 => (30, 1e-2),
        _ => (20, 1e-2),
    };
    
    Options::new()
        .with_grid_points(grid_points)
        .with_penalty_weight(penalty_weight)
        .with_early_exit(complexity_score > 200)
        .with_bypass_coefficient(complexity_score > 100)
        .with_flag_array(grid_points > 30)
}
```

## Advanced Settings

### Penalty Weight Configuration

The penalty weight (ε) is crucial for solution quality:

```rust
// High precision (slower, more accurate)
let precision_options = Options::new()
    .with_penalty_weight(1e-6)
    .with_round_decimals(12);

// Balanced performance
let balanced_options = Options::new()
    .with_penalty_weight(1e-3)      // Default
    .with_round_decimals(9);

// Fast approximation
let fast_options = Options::new()
    .with_penalty_weight(1e-1)
    .with_round_decimals(3);
```

### Precision and Rounding

```rust
// Financial applications (high precision)
let financial_options = Options::new()
    .with_round_decimals(8)
    .with_penalty_weight(1e-4);

// Engineering applications (moderate precision)
let engineering_options = Options::new()
    .with_round_decimals(6)
    .with_penalty_weight(1e-3);

// Approximation studies (low precision)
let approx_options = Options::new()
    .with_round_decimals(3)
    .with_penalty_weight(1e-2);
```

### Timeout Configuration

```rust
// Set timeout for long-running problems
let timeout_options = Options::new()
    .with_grid_points(100)
    .with_process_timeout(Some(3600));  // 1 hour timeout

// No timeout (let it run to completion)
let no_timeout_options = Options::new()
    .with_grid_points(100)
    .with_process_timeout(None);
```

## Solver Backend Configuration

### Custom Solver Options

```rust
use std::collections::HashMap;

let mut solver_opts = HashMap::new();

// CBC-specific options
solver_opts.insert("ratioGap".to_string(), "0.01".to_string());        // 1% optimality gap
solver_opts.insert("seconds".to_string(), "300".to_string());          // 5 minute time limit
solver_opts.insert("threads".to_string(), "4".to_string());            // 4 threads
solver_opts.insert("presolve".to_string(), "1".to_string());           // Enable presolve

let options = Options::new()
    .with_grid_points(50)
    .with_solver_option("log", "0");  // Correct method name and string values
```

### Solver Selection Strategy

```rust
fn configure_for_problem_type(problem_type: &str) -> HashMap<String, String> {
    let mut solver_options = HashMap::new();
    
    match problem_type {
        "integer_heavy" => {
            // Optimize for integer programming
            solver_options.insert("cutoff".to_string(), "1e-5".to_string());
            solver_options.insert("ratioGap".to_string(), "0.001".to_string());
            solver_options.insert("allowableGap".to_string(), "0.0".to_string());
        },
        "large_scale" => {
            // Optimize for large problems
            solver_options.insert("presolve".to_string(), "2".to_string());  // Aggressive presolve
            solver_options.insert("threads".to_string(), "8".to_string());
            solver_options.insert("seconds".to_string(), "600".to_string());
        },
        "high_precision" => {
            // Optimize for accuracy
            solver_options.insert("primalTolerance".to_string(), "1e-9".to_string());
            solver_options.insert("dualTolerance".to_string(), "1e-9".to_string());
            solver_options.insert("ratioGap".to_string(), "1e-6".to_string());
        },
        _ => {
            // Default configuration
            solver_options.insert("ratioGap".to_string(), "0.01".to_string());
            solver_options.insert("seconds".to_string(), "300".to_string());
        }
    }
    
    solver_options
}
```

## Logging and Monitoring

### Logging Configuration

```rust
use env_logger;

// Basic logging setup
fn setup_logging() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
}

// Advanced logging setup
fn setup_detailed_logging() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .format_timestamp_secs()
        .init();
}

// Usage with options
let options = Options::new()
    .with_grid_points(50)
    .with_process_logging(true);  // Enable detailed process logging
```

### Monitoring Progress

```rust
use log::{info, debug};

fn solve_with_monitoring(
    mut solver: Augmecon,
    options: &Options
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting AUGMECON solver with {} grid points", 
          options.grid_points.unwrap_or(0));
    
    let start_time = std::time::Instant::now();
    
    // Solve the problem
    solver.solve()?;
    
    let elapsed = start_time.elapsed();
    let solutions = solver.get_pareto_solutions();
    
    info!("Optimization completed in {:.2?}", elapsed);
    info!("Found {} Pareto-optimal solutions", solutions.len());
    info!("Solutions per second: {:.2}", 
          solutions.len() as f64 / elapsed.as_secs_f64());
    
    Ok(())
}
```

## Configuration Examples

### Production Environment

```rust
fn production_config() -> Options {
    Options::new()
        .with_name("production_optimization")
        .with_grid_points(50)                // Balanced performance
        .with_penalty_weight(1e-3)           // Standard precision
        .with_round_decimals(6)              // Business precision
        .with_early_exit(true)               // Efficiency optimization
        .with_bypass_coefficient(true)
        .with_flag_array(true)
        .with_process_timeout(Some(1800))    // 30 minute timeout
        .with_process_logging(false)         // Minimal logging
}
```

### Development Environment

```rust
fn development_config() -> Options {
    Options::new()
        .with_name("development_test")
        .with_grid_points(20)                // Fast iteration
        .with_penalty_weight(1e-2)           // Lower precision for speed
        .with_round_decimals(4)
        .with_early_exit(true)
        .with_bypass_coefficient(true)
        .with_flag_array(true)
        .with_process_timeout(Some(300))     // 5 minute timeout
        .with_process_logging(true)          // Detailed logging
}
```

### Research Environment

```rust
fn research_config(num_objectives: usize) -> Options {
    let grid_points = if num_objectives == 2 { 200 } else { 50 };
    
    Options::new()
        .with_name("research_study")
        .with_grid_points(grid_points)
        .with_penalty_weight(1e-4)           // High precision
        .with_round_decimals(10)
        .with_early_exit(false)              // Complete exploration
        .with_bypass_coefficient(false)      // Methodical approach
        .with_flag_array(false)
        .with_process_timeout(None)          // No timeout
        .with_process_logging(true)
}
```

## Troubleshooting Common Issues

### Slow Performance

```rust
// Diagnose and fix slow performance
fn optimize_for_speed(current_options: Options) -> Options {
    current_options
        .with_grid_points(
            current_options.grid_points.unwrap_or(50).min(50)  // Reduce grid size
        )
        .with_early_exit(true)               // Enable early termination
        .with_bypass_coefficient(true)       // Enable optimizations
        .with_flag_array(true)
        .with_penalty_weight(1e-2)           // Reduce precision for speed
}
```

### Memory Issues

```rust
// Reduce memory usage
fn reduce_memory_usage(current_options: Options) -> Options {
    current_options
        .with_flag_array(false)              // Disable memory-intensive features
        .with_shared_flag(false)
        .with_grid_points(
            current_options.grid_points.unwrap_or(50).min(30)
        )
}
```

### Poor Solution Quality

```rust
// Improve solution quality
fn improve_quality(current_options: Options) -> Options {
    current_options
        .with_grid_points(
            current_options.grid_points.unwrap_or(50).max(100)  // Increase resolution
        )
        .with_penalty_weight(1e-4)           // Increase precision
        .with_round_decimals(8)
        .with_early_exit(false)              // Complete exploration
}
```

## Best Practices

1. **Start Conservative**: Begin with default settings and adjust based on results
2. **Profile Performance**: Measure solve times and adjust grid points accordingly
3. **Match Problem Scale**: Larger problems need more conservative settings
4. **Consider Time Constraints**: Set appropriate timeouts for production systems
5. **Monitor Memory Usage**: Watch memory consumption with large grid sizes
6. **Test Incrementally**: Validate configuration changes on small problems first
7. **Document Settings**: Keep track of successful configurations for problem types

### Configuration Checklist

- [ ] Grid points appropriate for problem size and time constraints
- [ ] Penalty weight suitable for required precision
- [ ] Optimization flags enabled for performance
- [ ] Timeout set for production environments
- [ ] Logging configured appropriately
- [ ] Solver backend options optimized for problem type
- [ ] Memory usage within acceptable limits
- [ ] Configuration tested on representative problems

This comprehensive configuration guide should help you optimize AUGMECON-RS for your specific use case and performance requirements!
