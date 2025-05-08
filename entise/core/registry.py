
from typing import Dict, Type
from entise.core.base import Method, method_registry
import importlib
import pkgutil
import entise.methods


def register(cls: Type[Method], strategy_name: str = None):
    name = (strategy_name or getattr(cls, "name", cls.__name__)).lower()
    if name in method_registry:
        raise ValueError(f"Duplicate strategy name: '{name}' is already registered.")
    method_registry[name] = cls


def get_strategy(strategy_name: str) -> Type[Method]:
    name = strategy_name.lower()
    if name not in method_registry:
        raise ValueError(f"Strategy '{strategy_name}' not found.")
    return method_registry[name]


def list_strategies() -> list:
    return list(method_registry.keys())


def get_methods_by_type(ts_type: str):
    return [
        method for method in method_registry.values()
        if ts_type in getattr(method, "types", [])
    ]


def import_all_methods():
    for _, modname, _ in pkgutil.walk_packages(entise.methods.__path__, entise.methods.__name__ + "."):
        importlib.import_module(modname)

# Automatically import all methods on startup
import_all_methods()
