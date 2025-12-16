import pytest

from entise.core.base import Method
from entise.core.registry import get_strategy, list_strategies


class DummyMethod(Method):
    types = ["hvac"]
    name = "example"
    required_keys = []
    produces = []

    def generate(self, obj, data, ts_type):
        return {"summary": {"test": 1}, "timeseries": {}}


def test_autoregister_explicit_name():
    class ExplicitName(Method):
        types = ["hvac"]
        name = "myStrategy"
        required_keys = []
        produces = []

        def generate(self, obj, data, ts_type):
            return {"summary": {}, "timeseries": {}}

    assert get_strategy("mystrategy") is ExplicitName


def test_get_strategy_error():
    with pytest.raises(ValueError):
        get_strategy("nonexistent")


def test_list_strategies_keys():
    strategies = list_strategies()
    assert isinstance(strategies, list)
    assert "example" in strategies


if __name__ == "__main__":
    pytest.main()
