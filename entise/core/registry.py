import importlib
import pkgutil
from typing import List, Optional, Type

from entise.core.base import Method, method_registry


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
    return [method for method in method_registry.values() if ts_type in getattr(method, "types", [])]


def import_all_methods(path: Optional[List[str]] = None, package_name: Optional[str] = None):
    """Import all method modules from the specified path.

    Args:
        path: The path to search for modules. Must be provided.
        package_name: The package name to use as a prefix. Must be provided.
    """
    if path is None or package_name is None:
        raise ValueError("Both path and package_name must be provided to avoid circular imports.")

    for _, modname, _ in pkgutil.walk_packages(path, package_name + "."):
        importlib.import_module(modname)


# The import_all_methods() function is called explicitly in entise/methods/__init__.py
# to avoid circular imports
