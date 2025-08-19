# Getting Started with AUGMECON-RS

Welcome to AUGMECON-RS! This guide will help you get up and running with multi-objective optimization using the AUGMECON method in Rust.

## Table of Contents

1. [What is Multi-Objective Optimization?](#what-is-multi-objective-optimization)
2. [The AUGMECON Method](#the-augmecon-method)
3. [Installation](#installation)
4. [Your First Problem](#your-first-problem)
5. [Understanding the Results](#understanding-the-results)
6. [Next Steps](#next-steps)

## What is Multi-Objective Optimization?

Multi-objective optimization involves finding solutions that optimize multiple conflicting objectives simultaneously. Unlike single-objective optimization where we seek a single optimal solution, multi-objective problems typically have multiple "Pareto-optimal" solutions that represent different trade-offs between objectives.

### Real-World Examples

- **Portfolio Management**: Maximize returns while minimizing risk
- **Engineering Design**: Minimize weight while maximizing strength
- **Resource Allocation**: Minimize cost while maximizing quality and coverage
- **Supply Chain**: Minimize delivery time and cost while maximizing reliability

### Key Concepts

- **Pareto Front**: The set of all non-dominated solutions
- **Dominance**: Solution A dominates solution B if A is at least as good as B in all objectives and strictly better in at least one
- **Trade-offs**: The inherent conflicts between objectives that create multiple optimal solutions

## The AUGMECON Method

The Augmented ε-constraint (AUGMECON) method is a proven approach for finding Pareto-optimal solutions. It works by:

1. **Payoff Table Calculation**: Solving single-objective problems to determine the range of each objective
2. **Grid Generation**: Creating a systematic grid of constraint values
3. **ε-Constraint Problems**: Solving a series of constrained single-objective problems
4. **Solution Filtering**: Identifying truly Pareto-optimal solutions

### Advantages of AUGMECON

- **Guaranteed Pareto Optimality**: All solutions are guaranteed to be Pareto-optimal
- **Systematic Coverage**: Provides good coverage of the Pareto front
- **Flexibility**: Works with any linear or mixed-integer programming solver
- **Proven Performance**: Extensively validated in academic and industrial applications

## Installation

### Prerequisites

Ensure you have Rust installed (version 1.70 or later):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### Adding AUGMECON-RS to Your Project

Add the following to your `Cargo.toml`:

```toml
[dependencies]
augmecon = "0.1.0"
env_logger = "0.11"  # For logging (optional but recommended)
```

### Development Dependencies (Optional)

For running examples and tests:

```toml
[dev-dependencies]
approx = "0.5"
```

## Your First Problem

Let's solve a classic two-objective optimization problem: maximizing profit while minimizing environmental impact.

### Problem Description

We have two products to manufacture:
- Product A: Profit = $3 per unit, Environmental impact = 2 units per unit
- Product B: Profit = $2 per unit, Environmental impact = 1 unit per unit

Constraints:
- Maximum 100 units of Product A
- Maximum 150 units of Product B  
- Total production capacity: 200 units

### Step 1: Create the Project

```bash
cargo new my_optimization_project
cd my_optimization_project
```

Add dependencies to `Cargo.toml` as shown above.

### Step 2: Define the Problem

```rust
// src/main.rs
use augmecon::{
    Augmecon, MultiObjectiveProblem, ObjectiveDirection, Options,
    VariableType, LinearExpression, LinearConstraint, ConstraintType
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    env_logger::init();

    // Create a new multi-objective problem
    let mut problem = MultiObjectiveProblem::new();
    
    // Add decision variables
    let product_a = problem.add_variable(
        "product_a".to_string(),
        VariableType::Continuous { min: Some(0.0), max: Some(100.0) }
    );
    
    let product_b = problem.add_variable(
        "product_b".to_string(), 
        VariableType::Continuous { min: Some(0.0), max: Some(150.0) }
    );

    // Add capacity constraint: product_a + product_b <= 200
    let mut capacity_constraint = LinearExpression::new();
    capacity_constraint.add_term(1.0, "product_a".to_string());
    capacity_constraint.add_term(1.0, "product_b".to_string());
    
    problem.add_linear_constraint(LinearConstraint {
        expression: capacity_constraint,
        bound: 200.0,
        constraint_type: ConstraintType::LessEqual,
    });

    // Objective 1: Maximize profit (3*A + 2*B)
    let mut profit_objective = LinearExpression::new();
    profit_objective.add_term(3.0, "product_a".to_string());
    profit_objective.add_term(2.0, "product_b".to_string());
    problem.add_linear_objective(profit_objective, ObjectiveDirection::Maximize);

    // Objective 2: Minimize environmental impact (2*A + 1*B)
    let mut impact_objective = LinearExpression::new();
    impact_objective.add_term(2.0, "product_a".to_string());
    impact_objective.add_term(1.0, "product_b".to_string());
    problem.add_linear_objective(impact_objective, ObjectiveDirection::Minimize);

    // Configure solver options
    let options = Options::new()
        .with_name("profit_vs_environment")
        .with_grid_points(50)
        .with_penalty_weight(1e-3);

    // Create and solve
    let mut solver = Augmecon::new(problem, options)?;
    solver.solve()?;

    // Display results
    display_results(&solver);

    Ok(())
}

fn display_results(solver: &Augmecon) {
    let pareto_front = solver.get_pareto_front();
    
    println!("🎯 AUGMECON Optimization Results");
    println!("================================");
    println!("Found {} Pareto-optimal solutions\\n", pareto_front.len());
    
    println!("📊 Payoff Table:");
    for (i, row) in solver.get_payoff_table().iter().enumerate() {
        println!("  Objective {}: {:?}", i + 1, row);
    }
    println!();
    
    println!("🏆 Pareto-Optimal Solutions:");
    println!("  {:>8} {:>12} {:>18}", "Solution", "Profit ($)", "Env. Impact");
    println!("  {}", "-".repeat(40));
    
    for (i, solution) in solver.get_pareto_solutions().iter().enumerate() {
        let objectives = solution.objectives();
        println!("  {:>8} {:>12.2} {:>18.2}", 
                i + 1, objectives[0], objectives[1]);
    }
    
    if let Some(best_profit) = solver.get_pareto_solutions().first() {
        println!("\\n💡 Best profit solution: ${:.2}", best_profit.objectives()[0]);
    }
    
    if let Some(best_environment) = solver.get_pareto_solutions().last() {
        println!("🌱 Best environmental solution: {:.2} impact units", 
                solver.get_pareto_solutions().last().unwrap().objectives()[1]);
    }
}
```

### Step 3: Run the Problem

```bash
cargo run
```

You should see output similar to:

```
🎯 AUGMECON Optimization Results
================================
Found 51 Pareto-optimal solutions

📊 Payoff Table:
  Objective 1: [600.0, 50.0]
  Objective 2: [200.0, 150.0]

🏆 Pareto-Optimal Solutions:
  Solution      Profit ($)     Env. Impact
  ----------------------------------------
         1         600.00           200.00
         2         594.00           194.00
         3         588.00           188.00
         ...
        51          50.00           150.00

💡 Best profit solution: $600.00
🌱 Best environmental solution: 150.00 impact units
```

## Understanding the Results

### Payoff Table
The payoff table shows the best and worst values for each objective:
- Row 1: When maximizing profit, we get $600 profit but 200 environmental impact
- Row 2: When minimizing environmental impact, we get 150 impact but only $50 profit

### Pareto Front
Each solution in the Pareto front represents a different trade-off:
- **High Profit Solutions**: Focus on profitability at the cost of environment
- **Balanced Solutions**: Moderate values for both objectives  
- **Environmental Solutions**: Minimize impact at the cost of some profit

### Choosing a Solution
The "best" solution depends on your preferences:
- If profit is most important: Choose solutions near the top
- If environment is critical: Choose solutions near the bottom
- For balanced approach: Choose solutions in the middle

## Key Components Explained

### MultiObjectiveProblem
The main container for your optimization problem:
```rust
let mut problem = MultiObjectiveProblem::new();
```

### Variables
Decision variables represent what you're optimizing:
```rust
// Continuous variable with bounds
problem.add_variable("x".to_string(), 
    VariableType::Continuous { min: Some(0.0), max: Some(100.0) });

// Integer variable
problem.add_variable("y".to_string(),
    VariableType::Integer { min: Some(0), max: Some(50) });

// Binary variable (0 or 1)
problem.add_variable("z".to_string(), VariableType::Binary);
```

### Constraints
Constraints limit the feasible region:
```rust
let mut constraint = LinearExpression::new();
constraint.add_term(2.0, "x".to_string());
constraint.add_term(1.0, "y".to_string());

problem.add_linear_constraint(LinearConstraint {
    expression: constraint,
    bound: 100.0,
    constraint_type: ConstraintType::LessEqual,  // <=, >=, or ==
});
```

### Objectives
Define what you want to optimize:
```rust
let mut objective = LinearExpression::new();
objective.add_term(3.0, "x".to_string());
objective.add_term(-1.0, "y".to_string());

problem.add_linear_objective(objective, ObjectiveDirection::Maximize);
```

### Options
Configure solver behavior:
```rust
let options = Options::new()
    .with_name("my_problem")           // Problem name for logging
    .with_grid_points(100)             // Grid resolution
    .with_penalty_weight(1e-3)         // Augmentation parameter
    .with_round_decimals(6);           // Precision for results
```

## Common Patterns

### Builder Pattern for Complex Problems
```rust
fn build_complex_problem() -> MultiObjectiveProblem {
    let mut problem = MultiObjectiveProblem::new();
    
    // Add multiple variables at once
    for i in 0..10 {
        problem.add_variable(
            format!("x_{}", i),
            VariableType::Continuous { min: Some(0.0), max: None }
        );
    }
    
    // Add constraints in a loop
    for i in 0..5 {
        let mut constraint = LinearExpression::new();
        constraint.add_term(1.0, format!("x_{}", i));
        constraint.add_term(1.0, format!("x_{}", i + 5));
        
        problem.add_linear_constraint(LinearConstraint {
            expression: constraint,
            bound: 10.0,
            constraint_type: ConstraintType::LessEqual,
        });
    }
    
    problem
}
```

### Error Handling
```rust
fn solve_with_error_handling() -> Result<(), Box<dyn Error>> {
    let problem = build_problem();
    let options = Options::new().with_grid_points(50);
    
    // Handle solver creation errors
    let mut solver = Augmecon::new(problem, options)
        .map_err(|e| format!("Failed to create solver: {}", e))?;
    
    // Handle solving errors
    solver.solve()
        .map_err(|e| format!("Failed to solve: {}", e))?;
    
    // Check if we found solutions
    if solver.get_pareto_solutions().is_empty() {
        return Err("No Pareto-optimal solutions found".into());
    }
    
    Ok(())
}
```

## Next Steps

Now that you've mastered the basics, explore these advanced topics:

1. **[Problem Modeling](problem-modeling.md)**: Learn about complex constraint types and problem structures
2. **[Solver Configuration](solver-configuration.md)**: Optimize performance and customize behavior
3. **[Results Analysis](results-analysis.md)**: Advanced techniques for interpreting and using results
4. **[Examples](../examples/)**: Real-world applications and use cases

### Quick Tips for Success

1. **Start Simple**: Begin with 2-3 objectives and simple constraints
2. **Validate Results**: Compare payoff table values with manual calculations
3. **Tune Grid Points**: Start with 50-100 points, adjust based on solution quality
4. **Use Logging**: Enable debug logging to understand solver behavior
5. **Check Feasibility**: Ensure your problem has feasible solutions before adding complexity

### Common Gotchas

- **Infeasible Problems**: Check constraint compatibility
- **Unbounded Objectives**: Ensure variables have appropriate bounds
- **Too Many Grid Points**: Can lead to very long solve times
- **Conflicting Objectives**: Make sure objectives actually conflict (otherwise you get a single point)

Happy optimizing! 🚀
