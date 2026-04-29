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
#[cfg(feature = "milp")]
pub use solver::MilpConfig;
pub use solver::PlsConfig;

use pyo3::prelude::*;
use std::sync::Once;

static INIT_LOGGER: Once = Once::new();

/// A Python module implemented in Rust.
#[pymodule]
fn sims_problem(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize logging bridge from Rust to Python (only once)
    INIT_LOGGER.call_once(|| {
        pyo3_log::init();
    });

    // Add solver functions
    m.add_function(wrap_pyfunction!(solver::solve_with_pls, m)?)?;
    #[cfg(feature = "milp")]
    m.add_function(wrap_pyfunction!(solver::solve_with_milp, m)?)?;
    #[cfg(feature = "milp")]
    m.add_function(wrap_pyfunction!(solver::solve_with_hybrid, m)?)?;
    m.add_function(wrap_pyfunction!(solver::solve_with_nsga2, m)?)?;
    m.add_function(wrap_pyfunction!(solver::solve_with_moead, m)?)?;

    // Add hypervolume function
    m.add_function(wrap_pyfunction!(hypervolume::compute_hypervolume, m)?)?;

    // Add trace generation function
    m.add_function(wrap_pyfunction!(trace::generate_trace, m)?)?;

    // Add trace merging function
    m.add_function(wrap_pyfunction!(trace::merge_traces, m)?)?;

    // Add HV-over-time curve computation from trace
    m.add_function(wrap_pyfunction!(trace::compute_hv_curve_from_trace, m)?)?;

    // Add classes
    m.add_class::<SimsDiscreteProblem>()?;
    m.add_class::<Solution>()?;
    m.add_class::<SolvingResult>()?;
    #[cfg(feature = "milp")]
    m.add_class::<MilpConfig>()?;
    m.add_class::<PlsConfig>()?;

    Ok(())
}
