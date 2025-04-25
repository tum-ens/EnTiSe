import pytest
from src.constants.ts_types import Types


def test_register_method():
    from src.core.registry import register_method, get_method, method_registry

    class MockMethod:
        dependencies = []
        supported_types = [Types.HVAC]

    method_name = "mockmethod"

    # Ensure the method is not registered
    if method_name in method_registry:
        del method_registry[method_name]

    register_method(method_name, MockMethod)

    assert get_method(method_name) == MockMethod
    with pytest.raises(ValueError):
        register_method(method_name, MockMethod)  # Duplicate registration


def test_get_method():
    from src.core.registry import get_method

    with pytest.raises(ValueError):
        get_method("nonexistentmethod")  # Should raise error


def test_mock_methods_registration():
    from src.core.registry import get_method
    method1 = get_method("method1")
    method2 = get_method("method2")
    assert method1 is not None
    assert method2 is not None
