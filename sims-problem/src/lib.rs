// Module declarations
pub mod conversion;
pub mod problem;
pub mod solution;
pub mod solver;

// Re-export the main types
pub use problem::SimsDiscreteProblem;
pub use solution::Solution;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn sims_problem(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize logging bridge from Rust to Python
    pyo3_log::init();

    // Add solver function
    m.add_function(wrap_pyfunction!(solver::solve_with_pls, m)?)?;

    // Add classes
    m.add_class::<SimsDiscreteProblem>()?;
    m.add_class::<Solution>()?;

    Ok(())
}
