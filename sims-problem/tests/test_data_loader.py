from pathlib import Path
import sims_problem


def load_test_instance_as_problem(filename: str) -> 'sims_problem.SimsDiscreteProblem':
    """Load a test instance directly as a SimsDiscreteProblem using the Rust implementation."""
    test_dir = Path(__file__).parent / "data"
    file_path = test_dir / filename
    return sims_problem.SimsDiscreteProblem.from_dzn(str(file_path))


def get_all_test_instances() -> list[str]:
    """Get list of all test instance files in the data directory."""
    test_dir = Path(__file__).parent / "data"
    return [f.name for f in test_dir.glob("*.dzn")]