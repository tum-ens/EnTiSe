"""Time-series generation methods and their discovery helpers."""

from entise.constants import VALID_TYPES
from entise.core.registry import get_methods_by_type, get_strategy, list_strategies

__all__ = ["method_registry", "methods_by_type"]


def __getattr__(name):
    if name == "method_registry":
        from entise.core.base import method_registry

        for strategy_name in list_strategies():
            get_strategy(strategy_name)
        value = method_registry
    elif name == "methods_by_type":
        value = {ts_type: get_methods_by_type(ts_type) for ts_type in VALID_TYPES}
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return globals().setdefault(name, value)


def __dir__():
    return sorted(set(globals()).union(__all__))
