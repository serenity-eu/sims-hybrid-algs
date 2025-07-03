#[cfg(feature = "plotting")]
pub fn draw_solutions_plot<const D: usize>(solutions_data: &ExploredSolutionsData<D>) {
    let rainbow_colormap = DerivedColorMap::new(&[RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]);
    let num_iterations = solutions_data.num_iterations + 1;
    let root_drawing_area = SVGBackend::new("test.svg", (1024, 768)).into_drawing_area();
    // let root_drawing_area = BitMapBackend::new("test.png", (1024, 768)).into_drawing_area();

    root_drawing_area.fill(&WHITE).unwrap();

    // Clean up data

    let mut chart_ctx = ChartBuilder::on(&root_drawing_area)
        .caption("Pareto Local Search", ("Arial", 30))
        .set_label_area_size(LabelAreaPosition::Left, 50)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        // .build_cartesian_2d(0..(solutions_data.max_cost as i32), 0..(solutions_data.max_cloudy_area as i32))
        .build_cartesian_2d(0..12_000_000, 0..4_000_000)
        .unwrap();

    chart_ctx
        .configure_mesh()
        .x_desc("Cost")
        .y_desc("Cloudy area")
        .draw()
        .unwrap();

    // chart_ctx
    //     .draw_series(solutions_data.solutions().into_iter().map(
    //         |solution_point| {
    //             TriangleMarker::new(
    //                 (solution_point.objectives[0], solution_point.objectives[1]),
    //                 3,
    //                 // &GREY)
    //                 rainbow_colormap.get_color(solution_point.iteration as f32 / num_iterations as f32),
    //             )
    //             // Text::new(format!("{}", solution_point.iteration), (solution_point.objectives[0], solution_point.objectives[1]), ("sans-serif", 10))
    //         },
    //     ))
    //     .unwrap();

    chart_ctx
        .draw_series(solutions_data.non_dominated().into_iter().map(
            |solution_point| {
                Circle::new(
                    (solution_point.objectives[0], solution_point.objectives[1]),
                    6,
                    &GREEN, // rainbow_colormap.get_color(iteration as f32 / num_iterations as f32),
                )
            },
        ))
        .unwrap();

    chart_ctx
        .draw_series(
            solutions_data
                .initial_solutions()
                .into_iter()
                .map(|solution_point| TriangleMarker::new((solution_point.objectives[0], solution_point.objectives[1]), 6, &BLUE)),
        )
        .unwrap();
}
