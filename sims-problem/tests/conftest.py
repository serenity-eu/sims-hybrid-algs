import pytest
# Import fixtures from real_data_utils to make them available to all tests
from .real_data_utils import (
    all_test_instances,
    plots_directory,
    logger,
    small_pls_config,
    medium_pls_config,
    large_pls_config,
    small_milp_config,
    medium_milp_config,
    large_milp_config
)


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names or content."""
    for item in items:
        # Mark tests that contain "large_instances" as slow
        if "large_instances" in item.name:
            item.add_marker(pytest.mark.slow)
