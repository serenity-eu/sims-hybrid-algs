use std::{
    fs::{self, OpenOptions},
    path::{Path, PathBuf},
};

use chrono::Local;
use csv::{ReaderBuilder, WriterBuilder};
use itertools::Itertools;
use pareto::HasObjectives;
use pls::{
    problem::{SetCoverProblem, parse_set_of_vecs},
    solution::ImageSet,
};

pub fn solution_list_from_csv<const D: usize>(path: &Path) -> Vec<Vec<usize>> {
    let mut reader = ReaderBuilder::new()
        .delimiter(b';')
        .from_path(path)
        .unwrap_or_else(|_| panic!("Failed to open CSV file {}", path.display()));
    let solutions_pareto_front_record_index = reader
        .headers()
        .unwrap()
        .iter()
        .position(|header| header == "solutions_pareto_front")
        .unwrap();
    let last_record = reader.records().last().unwrap().unwrap();
    let solutions_pareto_front = last_record
        .get(solutions_pareto_front_record_index)
        .unwrap();
    let selected_images_vecs: Vec<Vec<usize>> = parse_set_of_vecs(solutions_pareto_front);
    return selected_images_vecs;
}

pub fn append_solutions_to_csv<
    T: ImageSet<D> + HasObjectives<D>,
    P: SetCoverProblem<D>,
    const D: usize,
>(
    path: &PathBuf,
    solutions: &[T],
    probem_instance: &P,
    timeout_s: u64,
    solution_time_s: &[f32],
    elapsed_time_s: u64,
) {
    let instance = probem_instance.instance_name().to_string();
    let problem = "sims".to_string();
    let solver_name = "pls".to_string();
    let solver_timeout_sec = timeout_s.to_string();
    let datetime = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let hypervolume = 0.0.to_string();
    let number_of_solutions = solutions.len().to_string();
    let time_solver_sec = elapsed_time_s.to_string();
    let threads = 1.to_string();
    let cores = 1.to_string();
    let exhaustive = "False".to_string();
    let incomplete_timeout_solution_added_to_front = "False".to_string();
    let solutions_time_list = 0.0.to_string();

    let pareto_front_str = solutions
        .iter()
        .map(|solution| {
            let objectives_str = solution.objectives().iter().join(", ");
            format!("[{objectives_str}]")
        })
        .join(", ");
    let pareto_front = format!("{{{pareto_front_str}}}");
    let pareto_solutions_time_list = format!("[{}]", solution_time_s.iter().join(", "));

    let solutions_pareto_front_str = solutions
        .iter()
        .map(|solution| format!("[{}]", solution.selected_images().join(", ")))
        .join(", ");
    let solutions_pareto_front = format!("{{{solutions_pareto_front_str}}}");

    // Dummy values
    let front_strategy = "None".to_string();
    let solver_search_strategy = "None".to_string();
    let fzn_optimization_level = "None".to_string();
    let minizinc_model = "None".to_string();
    let total_nodes = 0.to_string();
    let minizinc_time_fzn_sec = 0.0.to_string();
    let hypervolume_current_solutions = "None".to_string();

    let record = vec![
        instance,
        problem,
        solver_name,
        front_strategy,
        solver_search_strategy,
        fzn_optimization_level,
        threads,
        cores,
        solver_timeout_sec,
        minizinc_model,
        exhaustive,
        hypervolume,
        datetime,
        number_of_solutions,
        total_nodes,
        time_solver_sec,
        minizinc_time_fzn_sec,
        hypervolume_current_solutions,
        solutions_time_list,
        pareto_solutions_time_list,
        pareto_front,
        solutions_pareto_front,
        incomplete_timeout_solution_added_to_front,
    ];

    let path = Path::new(&path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|_| panic!("Failed to create directories for {}", path.display()));
    }

    let file = OpenOptions::new()
        .append(true)
        .create(true) // Create the file if it does not exist
        .open(path)
        .unwrap_or_else(|_| panic!("Failed to open CSV file {}", path.display()));

    let mut writer = WriterBuilder::new().delimiter(b';').from_writer(file);
    writer
        .write_record(record)
        .expect("Failed to write CSV record");
    writer.flush().expect("Failed to flush CSV writer");
}
