#![expect(
    clippy::cast_precision_loss,
    reason = "Legacy code style, extensive refactor needed"
)]
#![expect(
    clippy::cast_sign_loss,
    reason = "Legacy code style, extensive refactor needed"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "Legacy code style, extensive refactor needed"
)]
#![expect(
    clippy::cast_possible_wrap,
    reason = "Legacy code style, extensive refactor needed"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "Legacy code style, extensive refactor needed"
)]
pub mod explored_solutions_data;
pub mod objectives;
pub mod pareto_local_search;
pub mod problem;
pub mod residual_problem;
pub mod residual_solution;
pub mod solution;
pub mod solution_impl;
pub mod solution_set;
pub mod solution_set_impl;
pub mod timer;
pub mod util;
