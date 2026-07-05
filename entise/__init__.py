"""EnTiSe - Energy Time Series Generator."""

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("entise")
    except PackageNotFoundError:
        __version__ = "unknown"
except ImportError:
    __version__ = "unknown"

# Import and expose the methods module
from entise.core.generator import Generator
from entise.perf import get_accelerator, set_accelerator

__all__ = [
    "core",
    "Generator",
    "methods",
    "get_accelerator",
    "set_accelerator",
]
