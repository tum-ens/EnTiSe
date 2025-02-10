import pytest
from src.core.registry import method_registry
from tests.utils.mock_methods import register_mock_methods
from src.utils.validation import Validator


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

@pytest.fixture(autouse=True)
def reset_validator_cache():
    """Ensure the validator cache is cleared before each test."""
    Validator.disable_cache()  # Disables and clears cache
    Validator.enable_cache()  # Re-enable it for test consistency

