import pytest
from src.core.registry import method_registry
from tests.utils.mock_methods import register_mock_methods


@pytest.fixture(scope="session", autouse=True)
def register_all_methods():
    """Register all mock methods for the test session."""
    register_mock_methods()


@pytest.fixture(autouse=True)
def clear_and_register_method_registry():
    """
    Clears the global method_registry before each test and re-registers mock methods.
    This ensures that tests always have the required methods registered.
    """
    method_registry.clear()
    register_mock_methods()
