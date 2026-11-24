"""Resolve public class exports without importing sibling strategies."""

from importlib import import_module


def load_export(package, exports, name):
    if name not in exports:
        raise AttributeError(f"module {package!r} has no attribute {name!r}")
    return getattr(import_module(f".{exports[name]}", package), name)


def list_exports(namespace, exports):
    return sorted(set(namespace).union(exports))
