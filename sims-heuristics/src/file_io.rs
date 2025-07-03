use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::Duration,
};

use chrono::Local;
use csv::{ReaderBuilder, WriterBuilder};
use itertools::Itertools;
use pls::{
    explored_solutions_data::ParetoFrontSnapshot,
    problem::{parse_set_of_vecs, Problem},
    solution::EncodedSolution,
};

pub fn solution_list_from_csv<const D: usize>(path: &PathBuf, probem_instance: &Problem<D>) -> Vec<EncodedSolution<D>> {
    let mut reader = ReaderBuilder::new()
        .delimiter(b';')
        .from_path(path.clone())
        .unwrap_or_else(|_| panic!("Failed to open CSV file {:?}", path));
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
    return selected_images_vecs
        .into_iter()
        .map(|selected_images| {
            EncodedSolution::from_selected_images(selected_images, probem_instance)
        })
        .collect();
}

pub fn append_solutions_to_csv<const D: usize>(
    path: &PathBuf,
    solutions: &[EncodedSolution<D>],
    probem_instance: &Problem<D>,
    timeout_s: u64,
    solution_time_s: &[f32],
    elapsed_time_s: u64,
) {
    let instance = probem_instance.instance_name.clone();
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
        .map(|solution| format!("[{}, {}]", solution.objectives[0], solution.objectives[1]))
        .join(", ");
    let pareto_front = format!("{{{}}}", pareto_front_str);
    let pareto_solutions_time_list = format!("[{}]", solution_time_s.iter().join(", "));

    let solutions_pareto_front_str = solutions
        .iter()
        .map(|solution| format!("[{}]", solution.selected_images().join(", ")))
        .join(", ");
    let solutions_pareto_front = format!("{{{}}}", solutions_pareto_front_str);

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
            .unwrap_or_else(|_| panic!("Failed to create directories for {:?}", path));
    }

    let file = OpenOptions::new()
        .append(true)
        .create(true) // Create the file if it does not exist
        .open(path)
        .unwrap_or_else(|_| panic!("Failed to open CSV file {:?}", path));

    let mut writer = WriterBuilder::new().delimiter(b';').from_writer(file);
    writer
        .write_record(record)
        .expect("Failed to write CSV record");
    writer.flush().expect("Failed to flush CSV writer");
}

pub fn dump_invalid_initial_solutions<const D: usize>(
    invalid_solutions: Vec<EncodedSolution<D>>,
    instance_path: &Path,
    output_path: &Path,
    timeout: Duration,
) {
    let invalid_solutions_report = invalid_solutions
        .into_iter()
        .map(|solution| format!("Invalid solution: {:?}", solution))
        .join("\n");

    let instance_name = instance_path.file_stem().unwrap().to_str().unwrap();
    let output_dir = output_path.parent().unwrap();

    fs::write(
        output_dir.join(format!(
            "invalid_solutions_{}_timeout_{}.txt",
            instance_name,
            timeout.as_secs()
        )),
        invalid_solutions_report,
    )
    .expect("Failed to write invalid solutions report");
}

pub fn dump_pareto_front_snapshots(
    pareto_front_snapshots: Vec<ParetoFrontSnapshot>,
    output_path: &PathBuf,
) {
    let mut file =
        fs::File::create(output_path).expect("Failed to create pareto front snapshots file");

    for snapshot in pareto_front_snapshots {
        writeln!(
            file,
            "{} {} {}",
            snapshot.iteration,
            snapshot.timestamp.as_secs(),
            snapshot.solutions.len()
        )
        .unwrap();
        for solution in snapshot.solutions {
            writeln!(file, "{}", solution.into_iter().join(" ")).unwrap();
        }
    }
}
