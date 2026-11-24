"""Install hints for optional EnTiSe features."""

from contextlib import contextmanager
from importlib import import_module


@contextmanager
def requires_extra(extra, *modules, feature=None):
    """Translate a missing feature dependency while preserving other import errors."""
    try:
        yield
    except ModuleNotFoundError as exc:
        if exc.name not in modules:
            raise
        raise ImportError(
            f'{feature or extra} requires {exc.name}. Install it with pip install "entise[{extra}]".'
        ) from exc


def require_module(extra, module):
    """Load a declared feature dependency on first use."""
    with requires_extra(extra, module):
        return import_module(module)
