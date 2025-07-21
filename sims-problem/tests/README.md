# SIMS Problem Integration Tests

This directory contains comprehensive integration tests for the SIMS problem solver using the PLS (Pareto Local Search) algorithm.

## Test Structure

### Files

- `test_data_loader.py` - Utilities for loading test instances from `.dzn` files
- `test_data_loader_unit.py` - Unit tests for the data loading functionality
- `test_pls_real_data_integration.py` - Integration tests using real data instances
- `test_pls_integration.py` - Basic PLS integration tests (existing)

### Test Data

The `data/` directory contains real SIMS problem instances in MiniZinc data format (`.dzn` files):

- Small instances (30-50 images): Lagos Nigeria, Mexico City, Paris, Rio de Janeiro, Tokyo Bay
- Medium instances (100 images): All cities
- Large instances (145-200 images): All cities (marked as slow tests)

## Usage

### Loading Test Instances

You can now load SIMS problem instances directly using the new `from_dzn()` static method:

```python
import sims_problem

# Load directly from .dzn file (Rust implementation - faster)
problem = sims_problem.SimsDiscreteProblem.from_dzn("path/to/instance.dzn")

# Or use the helper function
from test_data_loader import load_test_instance_as_problem
problem = load_test_instance_as_problem("lagos_nigeria_30.dzn")
```

## Usage

### Running Tests

```bash
# Navigate to the sims-problem directory
cd /home/hlvlad/code/serenity/sims-hybrid-algs/sims-problem

# Run a specific instance
pytest tests/test_pls_real_data_integration.py::TestPLSIntegrationWithRealData::test_pls_on_small_instances[lagos_nigeria_30.dzn] -v

# Run all small instances
pytest tests/test_pls_real_data_integration.py::TestPLSIntegrationWithRealData::test_pls_on_small_instances -v

# Run all medium instances  
pytest tests/test_pls_real_data_integration.py::TestPLSIntegrationWithRealData::test_pls_on_medium_instances -v

# Run tests for specific city (all sizes)
pytest tests/test_pls_real_data_integration.py -k "paris" -v

# Run tests for specific size (all cities)
pytest tests/test_pls_real_data_integration.py -k "100.dzn" -v

# Run all tests except slow ones
pytest tests/test_pls_real_data_integration.py -m "not slow" -v

# Run all tests including large instances
pytest tests/test_pls_real_data_integration.py -v
```

### Test Categories

Tests are organized by instance size:

1. **Small instances** (30-50 images): 1 minute timeout, moderate parameters
2. **Medium instances** (100 images): 3 minutes timeout, generous parameters  
3. **Large instances** (150+ images): 5 minutes timeout, extended parameters (marked as `@pytest.mark.slow`)

## Features Tested

- **Solution validity**: All solutions are validated using `solution.validate(problem)`
- **Coverage**: Solutions must cover all universe elements
- **Constraints**: Solutions must satisfy cloud area constraints
- **Deterministic behavior**: Deterministic runs produce consistent results
- **Solution diversity**: PLS produces diverse solutions with different objectives
- **Performance**: Execution times are measured and reported

## Configuration

Test configuration is in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
```

Run `pytest -m "not slow"` to skip time-consuming large instance tests during development.

## Example Output

```
Testing lagos_nigeria_30.dzn: 30 images, 443 universe
Execution time: 5.23 seconds
Found 15 solutions
Cost range: 1234567 - 2345678
Cloudy area range: 123456 - 234567
```

## Notes

- The `from_dzn()` method automatically converts from 1-based indexing (MiniZinc format) to 0-based indexing (Python format)
- Large instance tests are marked as slow and can be skipped during development
- Solution validation uses the built-in `solution.validate(problem)` method which checks both coverage and constraints
- All test instances are loaded once per test class for better performance
