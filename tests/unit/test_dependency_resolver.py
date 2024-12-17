from src.core.dependency_resolver import DependencyResolver
from src.constants.ts_types import Types as T
from unittest.mock import patch

# Mock equivalent for WEATHER
WEATHER = "weather"


class MockMethod:
    def __init__(self, dependencies=None):
        self.dependencies = dependencies or []


def test_resolve_valid_dependencies():
    methods = {
        T.HVAC: MockMethod(dependencies=[T.ELECTRICITY]),
        T.ELECTRICITY: MockMethod(dependencies=[WEATHER]),
        WEATHER: MockMethod(),
    }
    resolver = DependencyResolver()
    resolved = resolver.resolve(methods)
    assert resolved == [WEATHER, T.ELECTRICITY, T.HVAC]


def test_resolve_no_dependencies():
    methods = {
        WEATHER: MockMethod(),
        T.HVAC: MockMethod(),
    }
    resolver = DependencyResolver()
    resolved = resolver.resolve(methods)
    assert set(resolved) == {WEATHER, T.HVAC}


def test_resolve_circular_dependencies():
    methods = {
        T.HVAC: MockMethod(dependencies=[T.ELECTRICITY]),
        T.ELECTRICITY: MockMethod(dependencies=[WEATHER]),
        WEATHER: MockMethod(dependencies=[T.HVAC]),
    }
    resolver = DependencyResolver()
    try:
        resolver.resolve(methods)
    except ValueError as e:
        assert "Circular dependency detected" in str(e)


def test_resolve_caching():
    methods = {
        T.HVAC: MockMethod(dependencies=[T.ELECTRICITY]),
        T.ELECTRICITY: MockMethod(),
    }
    resolver = DependencyResolver()

    # First resolution builds the order
    first_resolve = resolver.resolve(methods)
    assert first_resolve == [T.ELECTRICITY, T.HVAC]

    # Add a new method without dependencies
    methods[WEATHER] = MockMethod()

    # Cache should prevent recomputation for the same input
    cached_resolve = resolver.resolve({k: methods[k] for k in [T.HVAC, T.ELECTRICITY]})
    assert cached_resolve == first_resolve


@patch("matplotlib.pyplot.show")
@patch("networkx.draw")
def test_visualize_dependencies(mock_draw, mock_show):
    methods = {
        T.HVAC: MockMethod(dependencies=[T.ELECTRICITY]),
        T.ELECTRICITY: MockMethod(dependencies=[WEATHER]),
        WEATHER: MockMethod(),
    }

    DependencyResolver.visualize_dependencies(methods)

    # Ensure the drawing and plotting functions are called
    mock_draw.assert_called_once()
    mock_show.assert_called_once()
