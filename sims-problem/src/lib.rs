// Module declarations
pub mod conversion;
pub mod problem;
pub mod solution;
pub mod solver;
pub mod trace;
pub mod hypervolume;

// Re-export the main types
pub use problem::SimsDiscreteProblem;
pub use solution::{Solution, SolvingResult};
pub use solver::{MilpConfig, PlsConfig};

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn sims_problem(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize logging bridge from Rust to Python
    pyo3_log::init();

    // Add solver functions
    m.add_function(wrap_pyfunction!(solver::solve_with_pls, m)?)?;
    m.add_function(wrap_pyfunction!(solver::solve_with_milp, m)?)?;
    m.add_function(wrap_pyfunction!(solver::solve_with_hybrid, m)?)?;

    // Add classes
    m.add_class::<SimsDiscreteProblem>()?;
    m.add_class::<Solution>()?;
    m.add_class::<SolvingResult>()?;
    m.add_class::<MilpConfig>()?;
    m.add_class::<PlsConfig>()?;

    Ok(())
}
