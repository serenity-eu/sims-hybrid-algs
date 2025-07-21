import pytest


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
