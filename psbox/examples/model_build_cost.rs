//! How much of a solve is spent building the model rather than solving it?
//!
//! The search issues one model per scalarisation and throws it away. The
//! reference implementation keeps a single model for the whole run and only
//! adds and deletes the epsilon constraints around each solve. This measures
//! what that difference is worth at the shape of the instances in use:
//! `lagos_nigeria_150` is a set-cover with 150 binaries over a universe of
//! 7530, so roughly 7530 covering constraints.
//!
//! Run with: `cargo run --release --features gurobi --example model_build_cost`

use good_lp::{Expression, ProblemVariables, Solution, SolverModel, constraint, variable};
use std::time::Instant;

const IMAGES: usize = 150;
const UNIVERSE: usize = 7530;
/// Images covering each universe element, as the real instances roughly do.
const COVER: usize = 12;

fn main() {
    let mut vars = ProblemVariables::new();
    let x: Vec<_> = (0..IMAGES).map(|_| vars.add(variable().binary())).collect();

    // Each element must be covered by one of the images that see it.
    let built = Instant::now();
    let constraints: Vec<_> = (0..UNIVERSE)
        .map(|e| {
            let lhs: Expression = (0..COVER)
                .map(|k| x[(e * 7 + k * 13) % IMAGES])
                .fold(Expression::from(0.0), |acc, v| acc + v);
            constraint!(lhs >= 1.0)
        })
        .collect();
    println!(
        "built {UNIVERSE} constraints in rust: {:?}",
        built.elapsed()
    );

    let objective: Expression = x
        .iter()
        .enumerate()
        .fold(Expression::from(0.0), |acc, (i, v)| {
            acc + f64::from(u32::try_from(i % 977).unwrap() + 1) * *v
        });

    for round in 0..3 {
        let handed = Instant::now();
        let mut model = vars
            .clone()
            .minimise(objective.clone())
            .using(good_lp::solvers::gurobi::gurobi);
        for c in constraints.iter().cloned() {
            model = model.with(c);
        }
        let build = handed.elapsed();

        let started = Instant::now();
        let solution = model.solve().expect("feasible");
        let solve = started.elapsed();

        let total = build + solve;
        println!(
            "round {round}: build {build:?} ({:.1}%), solve {solve:?}, objective {:.0}",
            100.0 * build.as_secs_f64() / total.as_secs_f64(),
            solution.eval(&objective),
        );
    }
}
