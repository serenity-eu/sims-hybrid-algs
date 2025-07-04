//! # Plotting Module for Multi-Objective Visualization
//!
//! This module provides adaptive plotting capabilities for multi-objective optimization results:
//!
//! ## 2D Objectives (D = 2)
//! Creates a single scatter plot showing:
//! - Green circles: Non-dominated (Pareto optimal) solutions
//! - Blue triangles: Initial solutions
//! - Output: `pareto_solutions_2d.svg`
//!
//! ## Multi-Objective (D > 2)
//! Creates a grid of subplots showing all pairwise combinations of objectives:
//! - For D objectives, creates D×(D-1)/2 pairwise plots
//! - Automatically calculates grid layout (rows × columns)
//! - Each subplot shows the same solution sets projected onto 2D
//! - Output: `pareto_solutions_grid.svg`
//!
//! ## Examples
//! - 3D: 3 pairs → 2×2 grid with 3 subplots used
//! - 4D: 6 pairs → 2×3 grid with all 6 subplots used  
//! - 5D: 10 pairs → 3×4 grid with 10 subplots used

use plotters::prelude::*;
#[cfg(feature = "plotting")]
use pls::explored_solutions_data::ExploredSolutionsData;

/// Draw solutions plot for visualization of pareto fronts.
///
/// For 2D objectives: Creates a single scatter plot
/// For >2D objectives: Creates a grid of subplots showing all pairwise combinations
#[cfg(feature = "plotting")]
pub fn draw_solutions_plot<const D: usize>(solutions_data: &ExploredSolutionsData<D>) {
    if D == 2 {
        draw_2d_plot(solutions_data);
    } else if D > 2 {
        draw_multi_objective_grid(solutions_data);
    } else {
        eprintln!("Cannot plot with D={}, need at least 2 objectives", D);
    }
}

/// Draw a single 2D scatter plot for 2-objective problems
#[cfg(feature = "plotting")]
fn draw_2d_plot<const D: usize>(solutions_data: &ExploredSolutionsData<D>) {
    let root_drawing_area =
        SVGBackend::new("pareto_solutions_2d.svg", (1024, 768)).into_drawing_area();
    root_drawing_area.fill(&WHITE).unwrap();

    let mut chart_ctx = ChartBuilder::on(&root_drawing_area)
        .caption("Pareto Local Search - 2D Objectives", ("Arial", 30))
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 50)
        .margin(20)
        .build_cartesian_2d(
            0u64..solutions_data.max_objective(0),
            0u64..solutions_data.max_objective(1),
        )
        .unwrap();

    chart_ctx
        .configure_mesh()
        .x_desc("Objective 1")
        .y_desc("Objective 2")
        .draw()
        .unwrap();

    // Draw non-dominated solutions
    chart_ctx
        .draw_series(
            solutions_data
                .non_dominated()
                .into_iter()
                .map(|solution_point| {
                    Circle::new(
                        (solution_point.objectives[0], solution_point.objectives[1]),
                        6,
                        GREEN.filled(),
                    )
                }),
        )
        .unwrap()
        .label("Non-dominated solutions")
        .legend(|(x, y)| Circle::new((x + 10, y), 4, GREEN.filled()));

    // Draw initial solutions
    chart_ctx
        .draw_series(
            solutions_data
                .initial_solutions()
                .into_iter()
                .map(|solution_point| {
                    TriangleMarker::new(
                        (solution_point.objectives[0], solution_point.objectives[1]),
                        6,
                        BLUE.filled(),
                    )
                }),
        )
        .unwrap()
        .label("Initial solutions")
        .legend(|(x, y)| TriangleMarker::new((x + 10, y), 4, BLUE.filled()));

    chart_ctx.configure_series_labels().draw().unwrap();
}

/// Draw a grid of subplots for multi-objective problems (D > 2)
#[cfg(feature = "plotting")]
fn draw_multi_objective_grid<const D: usize>(solutions_data: &ExploredSolutionsData<D>) {
    // Calculate grid dimensions for all pairwise combinations
    let (grid_rows, grid_cols, num_pairs) = calculate_plot_grid_dimensions(D);

    // Calculate image dimensions
    let subplot_width = 300;
    let subplot_height = 250;
    let total_width = grid_cols * subplot_width + 100; // Extra space for margins
    let total_height = grid_rows * subplot_height + 150; // Extra space for title and margins

    let root_drawing_area = SVGBackend::new(
        "pareto_solutions_grid.svg",
        (total_width as u32, total_height as u32),
    )
    .into_drawing_area();
    root_drawing_area.fill(&WHITE).unwrap();

    // Add main title
    let (upper, lower) = root_drawing_area.split_vertically(80);
    upper
        .titled(
            &format!("Pareto Local Search - {D}D Objectives (Pairwise Plots)"),
            ("Arial", 24),
        )
        .unwrap();

    // Create grid of subplots
    let subplots = lower.split_evenly((grid_rows, grid_cols));

    let mut subplot_index = 0;
    for i in 0..D {
        for j in (i + 1)..D {
            if subplot_index < subplots.len() {
                draw_objective_pair_subplot(
                    &subplots[subplot_index],
                    solutions_data,
                    i,
                    j,
                    subplot_width,
                    subplot_height,
                );
                subplot_index += 1;
            }
        }
    }
}

/// Draw a single subplot for a pair of objectives
#[cfg(feature = "plotting")]
fn draw_objective_pair_subplot<const D: usize>(
    drawing_area: &DrawingArea<SVGBackend, plotters::coord::Shift>,
    solutions_data: &ExploredSolutionsData<D>,
    obj_x: usize,
    obj_y: usize,
    width: usize,
    height: usize,
) {
    let max_x = solutions_data.max_objective(obj_x);
    let max_y = solutions_data.max_objective(obj_y);

    let mut chart_ctx = ChartBuilder::on(drawing_area)
        .caption(
            &format!("Obj {} vs Obj {}", obj_x + 1, obj_y + 1),
            ("Arial", 14),
        )
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 30)
        .margin(10)
        .build_cartesian_2d(0u64..max_x, 0u64..max_y)
        .unwrap();

    chart_ctx
        .configure_mesh()
        .x_desc(&format!("Objective {}", obj_x + 1))
        .y_desc(&format!("Objective {}", obj_y + 1))
        .x_label_formatter(&|x| format!("{:.0}", x))
        .y_label_formatter(&|y| format!("{:.0}", y))
        .draw()
        .unwrap();

    // Draw non-dominated solutions
    chart_ctx
        .draw_series(
            solutions_data
                .non_dominated()
                .into_iter()
                .map(|solution_point| {
                    Circle::new(
                        (
                            solution_point.objectives[obj_x],
                            solution_point.objectives[obj_y],
                        ),
                        3,
                        GREEN.filled(),
                    )
                }),
        )
        .unwrap();

    // Draw initial solutions
    chart_ctx
        .draw_series(
            solutions_data
                .initial_solutions()
                .into_iter()
                .map(|solution_point| {
                    TriangleMarker::new(
                        (
                            solution_point.objectives[obj_x],
                            solution_point.objectives[obj_y],
                        ),
                        3,
                        BLUE.filled(),
                    )
                }),
        )
        .unwrap();
}

/// Calculate the grid dimensions for plotting pairwise objective combinations
///
/// Returns (rows, cols, num_pairs) for a given number of objectives D
#[cfg(feature = "plotting")]
pub fn calculate_plot_grid_dimensions(num_objectives: usize) -> (usize, usize, usize) {
    if num_objectives < 2 {
        return (0, 0, 0);
    }

    let num_pairs = num_objectives * (num_objectives - 1) / 2;
    let grid_cols = ((num_pairs as f64).sqrt().ceil() as usize).max(2);
    let grid_rows = (num_pairs + grid_cols - 1) / grid_cols; // Ceiling division

    (grid_rows, grid_cols, num_pairs)
}
