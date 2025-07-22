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

#[cfg(feature = "plotting")]
use crate::explored_solutions_data::ExploredSolutionsData;
use plotters::prelude::*;
use std::cmp::Ordering;

/// Generate a rainbow color based on iteration number and total iterations
#[cfg(feature = "plotting")]
fn rainbow_color(iteration: usize, max_iteration: usize) -> RGBColor {
    if max_iteration == 0 {
        return BLUE;
    }

    // Normalize iteration to 0.0-1.0 range
    let normalized_iteration = iteration as f64 / max_iteration as f64;

    // Convert to HSV color space: H varies from 0 (red) to 300 (magenta), S=1, V=1
    // This gives us a rainbow progression: red -> orange -> yellow -> green -> blue -> magenta
    let hue = normalized_iteration * 300.0; // 0-300 degrees

    // Convert HSV to RGB
    let chroma = 1.0; // chroma (since saturation = 1)
    let h_prime = hue / 60.0;
    let secondary = chroma * (1.0 - ((h_prime % 2.0) - 1.0).abs());

    let (r, g, b) = if h_prime < 1.0 {
        (chroma, secondary, 0.0)
    } else if h_prime < 2.0 {
        (secondary, chroma, 0.0)
    } else if h_prime < 3.0 {
        (0.0, chroma, secondary)
    } else if h_prime < 4.0 {
        (0.0, secondary, chroma)
    } else if h_prime < 5.0 {
        (secondary, 0.0, chroma)
    } else {
        (chroma, 0.0, secondary)
    };

    // Convert to 0-255 range and create RGBColor
    RGBColor((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Draw solutions plot for visualization of pareto fronts.
///
/// For 2D objectives: Creates a single scatter plot
/// For >2D objectives: Creates a grid of subplots showing all pairwise combinations
///
/// Uses objective names from the problem instance for proper labeling of axes and captions.
#[cfg(feature = "plotting")]
pub fn draw_solutions_plot<const D: usize>(
    solutions_data: &ExploredSolutionsData<D>,
    objective_names: &[&str],
) {
    if objective_names.len() != D {
        eprintln!(
            "Warning: Expected {} objective names, got {}",
            D,
            objective_names.len()
        );
    }

    match D.cmp(&2) {
        Ordering::Equal => draw_2d_plot(solutions_data, objective_names),
        Ordering::Greater => draw_multi_objective_grid(solutions_data, objective_names),
        Ordering::Less => eprintln!("Cannot plot with D={D}, need at least 2 objectives"),
    }
}

/// Convenience function to draw solutions plot using a Problem instance for objective names
///
/// This function extracts objective names from the Problem and calls `draw_solutions_plot`.
#[cfg(feature = "plotting")]
pub fn draw_solutions_plot_with_problem<const D: usize>(
    solutions_data: &ExploredSolutionsData<D>,
    problem: &crate::problem::Problem<D>,
) {
    let objective_names = problem.objective_names();
    draw_solutions_plot(solutions_data, &objective_names);
}

/// Calculate dynamic axis ranges from actual solution data
#[cfg(feature = "plotting")]
fn calculate_axis_ranges<const D: usize>(
    solutions_data: &ExploredSolutionsData<D>,
    obj_x: usize,
    obj_y: usize,
) -> ((u64, u64), (u64, u64)) {
    let all_solutions = solutions_data.solutions();

    if all_solutions.is_empty() {
        return ((0, 100), (0, 100));
    }

    let x_values: Vec<u64> = all_solutions.iter().map(|s| s.objectives[obj_x]).collect();
    let y_values: Vec<u64> = all_solutions.iter().map(|s| s.objectives[obj_y]).collect();

    let min_x = *x_values.iter().min().unwrap_or(&0);
    let max_x = *x_values.iter().max().unwrap_or(&100);
    let min_y = *y_values.iter().min().unwrap_or(&0);
    let max_y = *y_values.iter().max().unwrap_or(&100);

    // Add 10% padding to ranges for better visualization
    let x_range = max_x.saturating_sub(min_x);
    let y_range = max_y.saturating_sub(min_y);
    let x_padding = (x_range / 10).max(1);
    let y_padding = (y_range / 10).max(1);

    let x_min = min_x.saturating_sub(x_padding);
    let x_max = max_x.saturating_add(x_padding);
    let y_min = min_y.saturating_sub(y_padding);
    let y_max = max_y.saturating_add(y_padding);

    ((x_min, x_max), (y_min, y_max))
}

/// Draw a single 2D scatter plot for 2-objective problems
#[cfg(feature = "plotting")]
fn draw_2d_plot<const D: usize>(
    solutions_data: &ExploredSolutionsData<D>,
    objective_names: &[&str],
) {
    let root_drawing_area =
        SVGBackend::new("pareto_solutions_2d.svg", (1024, 768)).into_drawing_area();
    root_drawing_area.fill(&WHITE).unwrap();

    // Get objective names with fallbacks
    let obj1_name = objective_names.first().copied().unwrap_or("Objective 1");
    let obj2_name = objective_names.get(1).copied().unwrap_or("Objective 2");

    // Calculate dynamic axis ranges from actual solution data
    let ((x_min, x_max), (y_min, y_max)) = calculate_axis_ranges(solutions_data, 0, 1);

    let mut chart_ctx = ChartBuilder::on(&root_drawing_area)
        .caption(
            format!("Pareto Local Search - {obj1_name} vs {obj2_name} (Rainbow Timeline)"),
            ("Arial", 30),
        )
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 50)
        .margin(20)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .unwrap();

    chart_ctx
        .configure_mesh()
        .x_desc(obj1_name)
        .y_desc(obj2_name)
        .draw()
        .unwrap();

    // Get all solutions and sort them by iteration to draw in discovery order
    let mut all_solutions = solutions_data.solutions();
    all_solutions.sort_by_key(|s| s.iteration);

    // Find max iteration for rainbow color scaling
    let max_iteration = all_solutions.iter().map(|s| s.iteration).max().unwrap_or(0);

    // Get non-dominated solutions for highlighting
    let non_dominated = solutions_data.non_dominated();
    let non_dominated_set: std::collections::HashSet<_> =
        non_dominated.iter().map(|s| s.objectives).collect();

    // Draw all solutions with rainbow colors based on discovery time
    for solution in &all_solutions {
        let color = rainbow_color(solution.iteration, max_iteration);
        let is_non_dominated = non_dominated_set.contains(&solution.objectives);

        if is_non_dominated {
            // Non-dominated solutions: larger circles with black border
            chart_ctx
                .draw_series(std::iter::once(Circle::new(
                    (solution.objectives[0], solution.objectives[1]),
                    8,
                    color.filled(),
                )))
                .unwrap();
            chart_ctx
                .draw_series(std::iter::once(Circle::new(
                    (solution.objectives[0], solution.objectives[1]),
                    8,
                    BLACK.stroke_width(2),
                )))
                .unwrap();
        } else {
            // Other solutions: smaller circles
            chart_ctx
                .draw_series(std::iter::once(Circle::new(
                    (solution.objectives[0], solution.objectives[1]),
                    4,
                    color.filled(),
                )))
                .unwrap();
        }
    }

    // Create a simple text legend in the chart area
    chart_ctx
        .draw_series(std::iter::once(Text::new(
            "Rainbow colors show discovery order (early=red, late=magenta)".to_string(),
            (x_min + (x_max - x_min) / 20, y_max - (y_max - y_min) / 20),
            ("Arial", 12).into_font().color(&BLACK),
        )))
        .unwrap();

    chart_ctx
        .draw_series(std::iter::once(Text::new(
            "Large circles with black border = Pareto optimal solutions".to_string(),
            (x_min + (x_max - x_min) / 20, y_max - (y_max - y_min) / 10),
            ("Arial", 12).into_font().color(&BLACK),
        )))
        .unwrap();
}

/// Draw a grid of subplots for multi-objective problems (D > 2)
#[cfg(feature = "plotting")]
fn draw_multi_objective_grid<const D: usize>(
    solutions_data: &ExploredSolutionsData<D>,
    objective_names: &[&str],
) {
    // Calculate grid dimensions for all pairwise combinations
    let (grid_rows, grid_cols, _num_pairs) = calculate_plot_grid_dimensions(D);

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
            &format!("Pareto Local Search - {D}D Objectives (Rainbow Timeline)"),
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
                    objective_names,
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
    objective_names: &[&str],
    _width: usize,
    _height: usize,
) {
    // Calculate dynamic axis ranges from actual solution data
    let ((x_min, x_max), (y_min, y_max)) = calculate_axis_ranges(solutions_data, obj_x, obj_y);

    // Get objective names with fallbacks
    let x_axis_name = objective_names
        .get(obj_x)
        .copied()
        .unwrap_or("Unknown Objective");
    let y_axis_name = objective_names
        .get(obj_y)
        .copied()
        .unwrap_or("Unknown Objective");

    let mut chart_ctx = ChartBuilder::on(drawing_area)
        .caption(format!("{x_axis_name} vs {y_axis_name}"), ("Arial", 14))
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 30)
        .margin(10)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .unwrap();

    chart_ctx
        .configure_mesh()
        .x_desc(x_axis_name)
        .y_desc(y_axis_name)
        .x_label_formatter(&|x| format!("{x:.0}"))
        .y_label_formatter(&|y| format!("{y:.0}"))
        .draw()
        .unwrap();

    // Get all solutions and sort them by iteration to draw in discovery order
    let mut all_solutions = solutions_data.solutions();
    all_solutions.sort_by_key(|s| s.iteration);

    // Find max iteration for rainbow color scaling
    let max_iteration = all_solutions.iter().map(|s| s.iteration).max().unwrap_or(0);

    // Get non-dominated solutions for highlighting
    let non_dominated = solutions_data.non_dominated();
    let non_dominated_set: std::collections::HashSet<_> =
        non_dominated.iter().map(|s| s.objectives).collect();

    // Draw all solutions with rainbow colors based on discovery time
    for solution in &all_solutions {
        let color = rainbow_color(solution.iteration, max_iteration);
        let is_non_dominated = non_dominated_set.contains(&solution.objectives);

        if is_non_dominated {
            // Non-dominated solutions: larger circles with black border
            chart_ctx
                .draw_series(std::iter::once(Circle::new(
                    (solution.objectives[obj_x], solution.objectives[obj_y]),
                    4,
                    color.filled(),
                )))
                .unwrap();
            chart_ctx
                .draw_series(std::iter::once(Circle::new(
                    (solution.objectives[obj_x], solution.objectives[obj_y]),
                    4,
                    BLACK.stroke_width(1),
                )))
                .unwrap();
        } else {
            // Other solutions: smaller circles
            chart_ctx
                .draw_series(std::iter::once(Circle::new(
                    (solution.objectives[obj_x], solution.objectives[obj_y]),
                    2,
                    color.filled(),
                )))
                .unwrap();
        }
    }
}

/// Calculate the grid dimensions for plotting pairwise objective combinations
///
/// Returns (rows, cols, `num_pairs`) for a given number of objectives D
#[cfg(feature = "plotting")]
#[must_use]
pub fn calculate_plot_grid_dimensions(num_objectives: usize) -> (usize, usize, usize) {
    if num_objectives < 2 {
        return (0, 0, 0);
    }

    let num_pairs = num_objectives * (num_objectives - 1) / 2;
    let grid_cols = ((num_pairs as f64).sqrt().ceil() as usize).max(2);
    let grid_rows = num_pairs.div_ceil(grid_cols);

    (grid_rows, grid_cols, num_pairs)
}
