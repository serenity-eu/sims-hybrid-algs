import pytest
import sys
import os
from pathlib import Path

# Add the sims-solvers directory to the Python path
sims_solvers_path = Path(__file__).parent.parent
sys.path.insert(0, str(sims_solvers_path))

# Test configuration
TIMEOUT_SECONDS = 60
TEST_INSTANCES_DIR = "/home/hlvlad/code/serenity/sims-hybrid-algs/sims-problem/tests/data"
MZN_MODEL_PATH = "/home/hlvlad/code/serenity/sims-hybrid-algs/sims-solvers/sims_solvers/mzn_models/mosaic_cloud2.mzn"

# Test instances to use
TEST_INSTANCES_30 = [
    "lagos_nigeria_30.dzn",
    "mexico_city_30.dzn", 
    "paris_30.dzn",
    "rio_de_janeiro_30.dzn",
    "tokyo_bay_30.dzn"
]

TEST_INSTANCES_50 = [
    "lagos_nigeria_50.dzn",
    "mexico_city_50.dzn",
    "paris_50.dzn",
    "rio_de_janeiro_50.dzn",
    "tokyo_bay_50.dzn"
]

@pytest.fixture
def test_data_dir():
    """Fixture providing the test data directory path."""
    return TEST_INSTANCES_DIR

@pytest.fixture
def mzn_model_path():
    """Fixture providing the MiniZinc model path."""
    return MZN_MODEL_PATH

@pytest.fixture
def timeout():
    """Fixture providing the timeout for tests."""
    return TIMEOUT_SECONDS
